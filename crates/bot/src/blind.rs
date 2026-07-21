//! M10b blind agent: the in-battle flagship (`skuct`, state-keyed UCB +
//! argmax visits) restricted to legitimate information — public observations
//! (M10a `Observer`) plus the meta-pool prior (`Belief`) — with the hidden
//! opponent state imputed by per-iteration determinization.
//!
//! # Search shape
//!
//! One search per decision, `cfg.iterations` iterations, each drawn from a
//! fresh determinization: `Belief::determinize` samples a consistent pool
//! candidate (uniform), overwrites every hidden opponent field, resamples
//! unseen pick identities, and reseeds — the M10a contract. The tree is the
//! same state-keyed transposition-table machinery as `SkuctSearch`
//! (`run_iteration` shared verbatim), with one structural difference at the
//! root:
//!
//! - **Own root action: one global UCB over the public choice list** —
//!   probabilities of an information set, aggregated across ALL
//!   determinizations (our legal choices are public and identical in every
//!   determinization; a `debug_assert` guards this). The chosen action is
//!   *forced* into the iteration (`force_root[side]`).
//! - **Opponent root action (and everything below): per-determinization.**
//!   The determinized root's state key differs per candidate / pick
//!   assignment, so each determinization gets its own root node whose
//!   cached legal-action set matches its imputed moveset (naively sharing
//!   one root node panics the moment two candidates disagree on the active
//!   mon's moves — the collision pair). This is decoupled UCB where the
//!   opponent is modeled knowing their own team, which they do.
//!
//! Play = argmax visits over the global root stats — the `skuct` play rule.
//!
//! M10c: the loop body lives in `BlindSearch` — the persistent, steppable
//! form (mirroring `SkuctSearch`) that the wasm bridge's ponder loop pumps —
//! and `BlindAgent` drives that same struct internally, so the stepped form
//! can never drift from the gate-measured agent.
//!
//! # Team preview
//!
//! The opponent's pool team is publicly identifiable at preview by
//! species+levels (the belief's preview filter) except for the known
//! collision pair. Policy, simplest-correct first:
//!
//! - exactly one candidate alive AND the own-side matchup resolves to a
//!   baked pair table → play the M8 mixed equilibrium sample (same rule as
//!   `baked:<inner>`), resolved through the belief — never by reading the
//!   opponent's hidden set signature;
//! - otherwise (collision pair, unbaked matchup, fallback opponent, no
//!   tables) → the determinized preview search above (UCB + argmax over the
//!   120-action root, the existing `skuct` preview approach, on determinized
//!   states).
//!
//! # Per-game lifecycle
//!
//! The arena/duel harness constructs agents fresh per game, so `choose`
//! lazily creates the observer+belief on first call (team preview — where
//! `Observer::new` reads the preview-public facts). Defensively, the state
//! is also rebuilt whenever a new game is detectable (a team-preview
//! request, a side change, or a turn counter that went backwards).
//! The observer wants the outer battle log-ON (`DuelSpec::log_on`, set by
//! the arena for blind specs): the trace-free reveal channel (Leftovers /
//! Focus Band / Sleep Talk) degrades silently when the log is off.

use std::collections::HashMap;
use std::sync::Arc;

use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::belief::Belief;
use crate::observe::Observer;
use crate::preview::{MetaPool, TableSet};
use crate::rng::SplitMix64;
use crate::smmcts::{key_of, run_iteration, Node, RmConfig};

struct GameState {
    side: usize,
    last_turn: u16,
    observer: Observer,
    belief: Belief,
}

pub struct BlindAgent {
    cfg: RmConfig,
    rng: SplitMix64,
    pool: Arc<MetaPool>,
    tables: Option<Arc<TableSet>>,
    game: Option<GameState>,
}

impl BlindAgent {
    /// `cfg.iterations` / `c` / `hp_buckets` / `horizon` / `playout` are
    /// honored; the RM root layer fields are ignored (blind mirrors the
    /// argmax `skuct` rule).
    pub fn new(
        cfg: RmConfig,
        pool: Arc<MetaPool>,
        tables: Option<Arc<TableSet>>,
        seed: u64,
    ) -> Self {
        BlindAgent { cfg, rng: SplitMix64::new(seed), pool, tables, game: None }
    }

    /// The live belief (None before the first decision) — test surface.
    pub fn belief(&self) -> Option<&Belief> {
        self.game.as_ref().map(|g| &g.belief)
    }

    pub fn observer(&self) -> Option<&Observer> {
        self.game.as_ref().map(|g| &g.observer)
    }

    // ------------------------------------------------------ baked preview

    /// Belief-mediated M8 table lookup (see `baked_preview_pick`). `None` ⇔
    /// fall through to the determinized preview search.
    fn baked_pick(&mut self, battle: &Battle, side: usize) -> Option<SearchChoice> {
        let tables = self.tables.as_ref()?;
        let g = self.game.as_ref().expect("baked_pick after lifecycle setup");
        baked_preview_pick(tables, &g.belief, battle, side, &mut self.rng)
    }

    // -------------------------------------------------------------- search

    fn search(&mut self, battle: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> SearchChoice {
        let g = self.game.as_ref().expect("search after lifecycle setup");
        search_choose(&self.cfg, &mut self.rng, g, battle, dex, side, choices)
    }
}

/// One full `cfg.iterations` blind search at a decision point — the agent
/// loop over `BlindSearch`, shared by `BlindAgent` and `OpenAgent` (same
/// operation order as the original `BlindAgent::search`, bit-identical).
fn search_choose(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    g: &GameState,
    battle: &Battle,
    dex: &Dex,
    side: usize,
    choices: &[SearchChoice],
) -> SearchChoice {
    let mut bs = BlindSearch::with_rng(battle, dex, cfg.clone(), side, rng.clone());
    debug_assert_eq!(bs.actions(), choices, "root action set drifted from caller's choices");
    for _ in 0..cfg.iterations {
        bs.step_one(dex, &g.belief, &g.observer);
    }
    *rng = bs.rng.clone();
    bs.best().expect("search called with a non-empty choice list")
}

/// Belief-mediated M8 table lookup at team preview: own side by signature
/// (public to us), opponent by the single consistent pool candidate —
/// never by reading the opponent's hidden set signature. Samples the mixed
/// equilibrium (same rule as `BakedPreviewAgent`). `None` ⇔ collision pair
/// / fallback / unbaked matchup: play the determinized preview search.
pub fn baked_preview_pick(
    tables: &TableSet,
    belief: &Belief,
    battle: &Battle,
    side: usize,
    rng: &mut SplitMix64,
) -> Option<SearchChoice> {
    if belief.candidate_count() != 1 {
        return None; // collision pair (or fallback): identity unresolved
    }
    let opp = belief.alive()[0];
    debug_assert_eq!(
        tables.ids[opp],
        belief.candidate_id(opp),
        "TableSet/MetaPool pool-order drift"
    );
    let me = tables.side_index(battle, side)?;
    let (tab, i_am_a) = tables.pair_by_idx(me, opp)?;
    // sample the mixed equilibrium (same rule as BakedPreviewAgent)
    let p = if i_am_a { &tab.sol.p_a } else { &tab.sol.p_b };
    Some(SearchChoice::Team(tables.actions()[sample_mixed(p, rng)]))
}

/// Open-team-sheet M8 table lookup at team preview (the M12 product
/// policy): BOTH sides resolved by full-set signature — legitimate because
/// both sheets are public — so the collision pair resolves exactly and no
/// identification condition applies. Samples the mixed equilibrium. `None`
/// ⇔ either team off-pool or the pair not baked: play the (pinned-belief)
/// preview search instead.
pub fn open_preview_pick(
    tables: &TableSet,
    battle: &Battle,
    side: usize,
    rng: &mut SplitMix64,
) -> Option<SearchChoice> {
    let (tab, i_am_a) = tables.lookup(battle, side)?;
    let p = if i_am_a { &tab.sol.p_a } else { &tab.sol.p_b };
    Some(SearchChoice::Team(tables.actions()[sample_mixed(p, rng)]))
}

/// One draw from a mixed policy (one `next_f64`; degenerate rows fall back
/// to argmax — same rule and rng pattern as `BakedPreviewAgent`).
fn sample_mixed(p: &[f64], rng: &mut SplitMix64) -> usize {
    let u = rng.next_f64();
    let mut acc = 0.0;
    let mut pick = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
    for (i, &pr) in p.iter().enumerate() {
        acc += pr;
        if u < acc {
            pick = i;
            break;
        }
    }
    pick
}

// --------------------------------------------------- stepped search (M10c)

/// Persistent, incrementally steppable blind search over ONE decision point
/// — `BlindAgent`'s search loop in the form the wasm bridge's ponder loop
/// needs, mirroring `SkuctSearch`: create it at the current (true) battle
/// state, pump `step(n)` in slices, read `best()` / visit stats when the
/// move is wanted. The belief/observer pair is passed per call (it lives
/// with the per-game agent state, not the per-decision search).
/// `cfg.iterations` is ignored — the caller owns the budget.
///
/// `BlindAgent` drives this same struct internally, so the stepped form can
/// never drift from the gate-measured agent (Gate B + arena identity are
/// the net).
pub struct BlindSearch {
    cfg: RmConfig,
    rng: SplitMix64,
    /// Log-off base clone: the outer battle may run log-ON for the
    /// observer, and determinize clones its input — don't pay for cloning
    /// the whole protocol log every iteration.
    base: Battle,
    turn_cap: u16,
    side: usize,
    /// The public own-side choice list — the information-set root.
    my_acts: Vec<SearchChoice>,
    my_n: Vec<u32>,
    my_w: Vec<f64>,
    /// M15: optional root-action legality mask (masked actions are never
    /// selected or reported best). `None` = all allowed, bit-identical to
    /// the pre-M15 behavior. Historical purpose — Max Total Level at
    /// preview — is enforced by the engine's own enumeration since the
    /// 2026-07-17 preview-space fix; the API stays (harmless, generic).
    my_mask: Option<Vec<bool>>,
    /// Dominated root actions — certain immediate self-loss
    /// (`smmcts::certain_self_loss`) or provable no-op
    /// (`smmcts::certain_noop`): `best()` never argmaxes them while an
    /// alternative exists.
    my_dominated: Vec<bool>,
    /// Per-determinization roots + everything below (state-keyed).
    nodes: Vec<Node>,
    table: HashMap<u64, usize>,
    done: u32,
}

impl BlindSearch {
    pub fn new(battle: &Battle, dex: &Dex, cfg: RmConfig, side: usize, seed: u64) -> BlindSearch {
        Self::with_rng(battle, dex, cfg, side, SplitMix64::new(seed))
    }

    fn with_rng(
        battle: &Battle,
        dex: &Dex,
        cfg: RmConfig,
        side: usize,
        rng: SplitMix64,
    ) -> BlindSearch {
        let mut base = battle.clone();
        base.set_log_enabled(false);
        let turn_cap = base.turn.saturating_add(cfg.horizon);
        let my_acts = base.legal_choices(dex, side);
        let my_dominated = my_acts
            .iter()
            .map(|&c| {
                crate::smmcts::certain_self_loss(&base, dex, side, c)
                    || crate::smmcts::certain_noop(&base, dex, side, c)
            })
            .collect();
        BlindSearch {
            cfg,
            rng,
            base,
            turn_cap,
            side,
            my_n: vec![0; my_acts.len()],
            my_w: vec![0.0; my_acts.len()],
            my_mask: None,
            my_dominated,
            my_acts,
            nodes: Vec::new(),
            table: HashMap::new(),
            done: 0,
        }
    }

    /// One iteration: fresh determinization, global-UCB own root pick
    /// forced into the shared `run_iteration`. Returns the side-0 reward.
    pub fn step_one(&mut self, dex: &Dex, belief: &Belief, obs: &Observer) -> f64 {
        let mut sim = belief.determinize(dex, &self.base, obs, &mut self.rng);
        let key = key_of(&self.cfg, &sim);
        let root = match self.table.get(&key) {
            Some(&i) => i,
            None => {
                let node = Node::at(&mut sim, dex);
                debug_assert_eq!(
                    node.acts[self.side], self.my_acts,
                    "determinization changed the observer's own root actions"
                );
                self.nodes.push(node);
                self.table.insert(key, self.nodes.len() - 1);
                self.nodes.len() - 1
            }
        };
        let my_pick = select_global(
            &self.cfg,
            &mut self.rng,
            &mut self.my_n,
            &self.my_w,
            self.my_mask.as_deref(),
        );
        let mut force = [None, None];
        force[self.side] = Some(my_pick);
        let mut joint = [0usize; 2];
        let r = run_iteration(
            &self.cfg,
            &mut self.rng,
            &mut self.nodes,
            &mut self.table,
            &mut sim,
            dex,
            self.turn_cap,
            root,
            force,
            &mut joint,
        );
        self.my_w[my_pick] += if self.side == 0 { r } else { 1.0 - r };
        self.done += 1;
        r
    }

    /// Pump `n` iterations, return the total run so far.
    pub fn step(&mut self, dex: &Dex, belief: &Belief, obs: &Observer, n: u32) -> u32 {
        for _ in 0..n {
            self.step_one(dex, belief, obs);
        }
        self.done
    }

    pub fn iterations(&self) -> u32 {
        self.done
    }

    /// The global root's actions — the side's public legal choice list.
    pub fn actions(&self) -> &[SearchChoice] {
        &self.my_acts
    }

    /// Per-action visit counts on the global (information-set) root stats.
    pub fn visits(&self) -> &[u32] {
        &self.my_n
    }

    /// Per-action mean rewards (own perspective — `my_w` is accumulated
    /// from this side's view), 0.5 when unvisited.
    pub fn means(&self) -> Vec<f64> {
        (0..self.my_acts.len())
            .map(|a| {
                if self.my_n[a] == 0 {
                    0.5
                } else {
                    self.my_w[a] / self.my_n[a] as f64
                }
            })
            .collect()
    }

    /// M15: restrict the root to `allowed` actions (aligned with
    /// `actions()`). Masked actions keep their index (the per-iteration
    /// root forcing is index-aligned) but are never selected or returned by
    /// `best`. Panics if nothing stays allowed.
    pub fn mask_actions(&mut self, allowed: &[bool]) {
        assert_eq!(allowed.len(), self.my_acts.len(), "mask length mismatch");
        assert!(allowed.iter().any(|&a| a), "mask leaves no legal action");
        self.my_mask = Some(allowed.to_vec());
    }

    /// Current best choice: argmax visits over the global root stats (the
    /// blind play rule), restricted to the mask when one is set. `None`
    /// when the side owes nothing.
    pub fn best(&self) -> Option<SearchChoice> {
        // Deep-loss roots tie every action at mean 0 with exactly uniform
        // visits; excluding certain-immediate-self-loss actions keeps the
        // tie-break from picking a guaranteed instant loss (2026-07-21
        // last-mon-Explosion report). Falls back to mask-only when nothing
        // else is allowed.
        let allowed = |a: usize| self.my_mask.as_ref().map_or(true, |m| m[a]);
        (0..self.my_acts.len())
            .filter(|&a| allowed(a) && !self.my_dominated[a])
            .max_by_key(|&a| self.my_n[a])
            .or_else(|| (0..self.my_acts.len()).filter(|&a| allowed(a)).max_by_key(|&a| self.my_n[a]))
            .map(|a| self.my_acts[a])
    }

    /// Whether the root decision is a team preview.
    pub fn is_preview(&self) -> bool {
        matches!(self.my_acts.first(), Some(SearchChoice::Team(_)))
    }
}

/// UCB1 over the global (information-set) root stats — same rule and rng
/// draw pattern as `smmcts::select_ucb`, on plain arrays. `mask` (M15)
/// restricts the pick to allowed indices; `None` is bit-identical to the
/// unmasked original.
fn select_global(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    n: &mut [u32],
    w: &[f64],
    mask: Option<&[bool]>,
) -> usize {
    let k = n.len();
    let ok = |a: usize| mask.map_or(true, |m| m[a]);
    let untried: Vec<usize> = (0..k).filter(|&a| n[a] == 0 && ok(a)).collect();
    let pick = if !untried.is_empty() {
        untried[rng.below(untried.len())]
    } else {
        let total: u32 = n.iter().sum();
        let ln_total = (total as f64).ln();
        let mut best = (0..k).find(|&a| ok(a)).unwrap_or(0);
        let mut best_v = f64::NEG_INFINITY;
        for a in (0..k).filter(|&a| ok(a)) {
            let (na, wa) = (n[a] as f64, w[a]);
            let v = wa / na + cfg.c * (ln_total / na).sqrt();
            if v > best_v {
                best_v = v;
                best = a;
            }
        }
        best
    };
    n[pick] += 1;
    pick
}

// ------------------------------------------------- open-sheet agent (M14)

/// M14 `open` agent: the M12 open-team-sheet product policy in arena form —
/// the blind machinery with the opponent's TRUE sets pinned as a singleton
/// belief (`Belief::pinned_from_battle`; legitimate because both sheets are
/// public under the policy), so determinizations equal the truth except
/// what stays hidden by policy: unseen pick identities (which 3 of 6 +
/// lead) and the mid-turn pending-move scrub. Team preview mirrors the
/// wasm worker's pinned path: `open_preview_pick` (both sides resolved by
/// public signature — baked pair tables answer when the matchup is baked),
/// else the pinned determinized preview search. This is exactly what the
/// shipped web bot plays; `skuct` (perfect info incl. picks) is its
/// upper-bound opponent.
pub struct OpenAgent {
    cfg: RmConfig,
    rng: SplitMix64,
    tables: Option<Arc<TableSet>>,
    game: Option<GameState>,
}

impl OpenAgent {
    /// Same config surface as `BlindAgent` (the RM root layer fields are
    /// ignored — the blind/open play rule is argmax over the global root).
    pub fn new(cfg: RmConfig, tables: Option<Arc<TableSet>>, seed: u64) -> Self {
        OpenAgent { cfg, rng: SplitMix64::new(seed), tables, game: None }
    }

    /// The live belief (None before the first decision) — test surface.
    pub fn belief(&self) -> Option<&Belief> {
        self.game.as_ref().map(|g| &g.belief)
    }
}

impl Agent for OpenAgent {
    fn name(&self) -> String {
        format!("open:{}:{}:{}", self.cfg.iterations, self.cfg.c, self.cfg.hp_buckets)
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let is_preview = matches!(choices[0], SearchChoice::Team(_));

        // ---- per-game lifecycle (mirrors BlindAgent): (re)build at
        // preview / on a new game. The pinned belief snapshots the
        // opponent's true roster, so it must be built at team preview
        // (fresh mons); the defensive mid-game rebuild degrades gracefully
        // (refs then carry live PP marks — still a superset of public).
        let stale = match &self.game {
            None => true,
            Some(g) => g.side != side || is_preview || battle.turn < g.last_turn,
        };
        if stale {
            let observer = Observer::new(battle, side);
            let belief = Belief::pinned_from_battle(battle, &observer);
            self.game = Some(GameState { side, last_turn: battle.turn, observer, belief });
        }
        {
            let g = self.game.as_mut().unwrap();
            g.last_turn = battle.turn;
            g.observer.observe(battle, dex);
            g.belief.sync(dex, &g.observer);
        }

        if choices.len() == 1 {
            return choices[0];
        }
        if is_preview {
            // Open sheet: both rosters are public — resolve the pair by
            // signature (no identification condition; the wasm worker's
            // pinned-mode rule). None ⇔ off-pool team or unbaked pair.
            if let Some(tables) = self.tables.as_ref() {
                if let Some(c) = open_preview_pick(tables, battle, side, &mut self.rng) {
                    debug_assert!(choices.contains(&c), "open preview outside legal set");
                    if choices.contains(&c) {
                        return c;
                    }
                }
            }
        }
        let g = self.game.as_ref().expect("choose after lifecycle setup");
        search_choose(&self.cfg, &mut self.rng, g, battle, dex, side, choices)
    }
}

impl Agent for BlindAgent {
    fn name(&self) -> String {
        format!("blind:{}:{}:{}", self.cfg.iterations, self.cfg.c, self.cfg.hp_buckets)
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let is_preview = matches!(choices[0], SearchChoice::Team(_));

        // ---- per-game lifecycle: (re)build at preview / on a new game
        let stale = match &self.game {
            None => true,
            Some(g) => g.side != side || is_preview || battle.turn < g.last_turn,
        };
        if stale {
            let observer = Observer::new(battle, side);
            let belief = Belief::new(dex, &self.pool, &observer);
            self.game = Some(GameState { side, last_turn: battle.turn, observer, belief });
        }
        {
            let g = self.game.as_mut().unwrap();
            g.last_turn = battle.turn;
            g.observer.observe(battle, dex);
            g.belief.sync(dex, &g.observer);
        }

        if choices.len() == 1 {
            return choices[0];
        }
        if is_preview {
            if let Some(c) = self.baked_pick(battle, side) {
                debug_assert!(choices.contains(&c), "baked preview outside legal set");
                if choices.contains(&c) {
                    return c;
                }
            }
        }
        self.search(battle, dex, side, choices)
    }
}
