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

use std::sync::Arc;

use nc2000_engine::battle::enumerate::enumerate_step_with_damage_mode;
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::{Dex, SpeciesId};
use nc2000_engine::fxhash::FxHashMap;
use nc2000_engine::prng::DamageRollMode;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::belief::{Belief, FallbackPolicy};
use crate::observe::Observer;
use crate::preview::{MetaPool, TableSet};
use crate::prior::BeliefPrior;
use crate::rng::SplitMix64;
use crate::smmcts::{key_of, run_iteration, Node, RmConfig, SkuctSearch};

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
    fallback_policy: FallbackPolicy,
    /// M18 community belief prior. `None` (the default) leaves the fallback
    /// imputation exactly as shipped.
    prior: Option<Arc<BeliefPrior>>,
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
        Self::new_with_fallback_policy(
            cfg,
            pool,
            tables,
            seed,
            FallbackPolicy::Layered,
        )
    }

    /// Evaluation control for fallback-policy A/B tests. The ordinary
    /// constructor remains permanently bound to the shipped layered policy.
    pub fn new_with_fallback_policy(
        cfg: RmConfig,
        pool: Arc<MetaPool>,
        tables: Option<Arc<TableSet>>,
        seed: u64,
        fallback_policy: FallbackPolicy,
    ) -> Self {
        BlindAgent {
            cfg,
            rng: SplitMix64::new(seed),
            pool,
            tables,
            fallback_policy,
            prior: None,
            game: None,
        }
    }

    /// M18: consume a community belief prior for the hidden-team fallback
    /// imputation. Takes effect from the next game/preview onward, and is
    /// inert on the open-sheet path (`OpenAgent` pins its belief).
    pub fn set_belief_prior(&mut self, prior: Arc<BeliefPrior>) {
        self.prior = Some(prior);
        if let Some(g) = self.game.as_mut() {
            g.belief.set_prior(self.prior.clone().unwrap());
        }
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
    table: FxHashMap<u64, usize>,
    done: u32,
    /// Root joint statistics: `(own root action index, the opponent action
    /// it met, samples, summed reward from THIS side)`. Free — every entry
    /// comes from an iteration the search was running anyway — and it is the
    /// only place the simultaneous structure is visible at all: the per-side
    /// visit counts marginalize it away, so "this move is good, but only
    /// because they rarely stay in" cannot be read off `visits()`.
    ///
    /// Keyed by the opponent's `SearchChoice`, never by its index: the
    /// opponent's legal list is determinization-dependent (an imputed set
    /// decides which moves exist), so an index means nothing across
    /// iterations while a `Move(id)` means the same move every time.
    joint: Vec<(usize, SearchChoice, u32, f64)>,
    /// How often each opponent action was even LEGAL, across iterations.
    /// A blind root's opponent action list is determinization-dependent — a
    /// move exists only in the candidates that carry it — so a column's
    /// sample count conflates "rarely chosen" with "rarely available", and
    /// only this tells them apart.
    avail: Vec<(SearchChoice, u32)>,
}

impl BlindSearch {
    pub fn new(battle: &Battle, dex: &Dex, cfg: RmConfig, side: usize, seed: u64) -> BlindSearch {
        Self::with_rng(battle, dex, cfg, side, SplitMix64::new(seed))
    }

    fn with_rng(
        battle: &Battle,
        dex: &Dex,
        mut cfg: RmConfig,
        side: usize,
        rng: SplitMix64,
    ) -> BlindSearch {
        if cfg.key_no_damage && battle.damage_bookkeeping_observable(dex) {
            cfg.key_no_damage = false;
        }
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
            table: FxHashMap::default(),
            done: 0,
            joint: Vec::new(),
            avail: Vec::new(),
        }
    }

    /// One iteration: fresh determinization, global-UCB own root pick
    /// forced into the shared `run_iteration`. Returns the side-0 reward.
    pub fn step_one(&mut self, dex: &Dex, belief: &Belief, obs: &Observer) -> f64 {
        let mut sim = belief.determinize(dex, &self.base, obs, &mut self.rng);
        let key = key_of(&self.cfg, dex, &mut sim);
        let root = match self.table.get(&key) {
            Some(&i) => i,
            None => {
                let node = Node::at(&mut sim, dex, &self.cfg);
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
            &mut 0,
        );
        self.record_joint(root, my_pick, joint, r);
        self.my_w[my_pick] += if self.side == 0 { r } else { 1.0 - r };
        self.done += 1;
        r
    }

    /// One iteration with a caller-selected own root action, while keeping
    /// the same determinization table and opponent-root statistics as every
    /// other forced action. This is the equal-allocation measurement path
    /// used by M17a confirmation; unlike separate one-hot trees, it does not
    /// let the simultaneous opponent condition its root policy on our move.
    pub fn step_forced(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        my_pick: usize,
    ) -> f64 {
        assert!(my_pick < self.my_acts.len(), "forced root action out of range");
        assert!(
            self.my_mask.as_ref().map_or(true, |mask| mask[my_pick]),
            "forced root action is masked"
        );
        let mut sim = belief.determinize(dex, &self.base, obs, &mut self.rng);
        let key = key_of(&self.cfg, dex, &mut sim);
        let root = match self.table.get(&key) {
            Some(&i) => i,
            None => {
                let node = Node::at(&mut sim, dex, &self.cfg);
                debug_assert_eq!(
                    node.acts[self.side], self.my_acts,
                    "determinization changed the observer's own root actions"
                );
                self.nodes.push(node);
                self.table.insert(key, self.nodes.len() - 1);
                self.nodes.len() - 1
            }
        };
        self.my_n[my_pick] += 1;
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
            &mut 0,
        );
        self.record_joint(root, my_pick, joint, r);
        self.my_w[my_pick] += if self.side == 0 { r } else { 1.0 - r };
        self.done += 1;
        r
    }

    /// Fold one iteration's root joint into the matrix. Silently drops the
    /// iterations where the opponent owed nothing (a forced switch on our
    /// side alone) — there is no cell for "they did not act".
    fn record_joint(&mut self, root: usize, my_pick: usize, joint: [usize; 2], r: f64) {
        let opp = 1 - self.side;
        let acts = &self.nodes[root].acts[opp];
        for &c in acts.iter() {
            match self.avail.iter_mut().find(|(x, _)| *x == c) {
                Some((_, n)) => *n += 1,
                None => self.avail.push((c, 1)),
            }
        }
        let acts = &self.nodes[root].acts[opp];
        let Some(&opp_act) = acts.get(joint[opp]) else { return };
        let mine = if self.side == 0 { r } else { 1.0 - r };
        match self
            .joint
            .iter_mut()
            .find(|(a, c, _, _)| *a == my_pick && *c == opp_act)
        {
            Some((_, _, n, w)) => {
                *n += 1;
                *w += mine;
            }
            None => self.joint.push((my_pick, opp_act, 1, mine)),
        }
    }

    /// The root joint cells: `(own action index, opponent action, samples,
    /// mean reward from this side)`. Unvisited cells are absent rather than
    /// zero — "never tried" and "tried and scored 0" are different answers,
    /// and a reader that cannot tell them apart will draw the wrong one.
    pub fn root_matrix(&self) -> Vec<(usize, SearchChoice, u32, f64)> {
        self.joint
            .iter()
            .map(|&(a, c, n, w)| (a, c, n, if n > 0 { w / n as f64 } else { 0.5 }))
            .collect()
    }

    /// The line both sides would play from here — the "読み筋" a study screen
    /// shows under the score.
    ///
    /// Three things make this a line rather than a branch, and each one is a
    /// correction of the obvious implementation:
    ///
    /// - **Each ply is its own search.** Reading a continuation off the
    ///   blind tree's visit counts looks right and is not: below the root
    ///   those nodes are state-keyed across determinizations, HP-bucketed,
    ///   and often entered a handful of times, so their argmax is noise
    ///   wearing the search's authority. A fresh `SkuctSearch` per ply costs
    ///   a fraction of the root search and answers the question actually
    ///   being asked — what would a good player do HERE.
    /// - **Chance is not sampled.** Advancing with the battle's own PRNG
    ///   shows one roll of the dice as though it were the plan; a paralysis
    ///   that lands 30% of the time reads exactly like one that always does.
    ///   Every step is enumerated exactly instead (`enumerate_step`, damage
    ///   collapsed to its probability-weighted mean) and the line follows the
    ///   single most likely outcome, carrying its probability so the reader
    ///   can see how typical it is.
    /// - **It stops rather than guess.** No leaves under the cap, nothing
    ///   left to choose, or the game over — the line ends there.
    ///
    /// One assumption remains, and it is stated rather than hidden: the whole
    /// continuation runs inside ONE determinization, so `assumed` reports the
    /// opponent set it was played against, and both sides play as if that set
    /// were common knowledge.
    pub fn principal_line(
        &self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        seed: u64,
        plies: usize,
        iters: u32,
        from: Option<SearchChoice>,
    ) -> PrincipalLine {
        let mut rng = SplitMix64::new(seed);
        let mut sim = belief.determinize(dex, &self.base, obs, &mut rng);
        sim.set_log_enabled(false);
        let assumed = sim.sides[1 - self.side]
            .roster
            .iter()
            .map(|p| {
                (
                    p.species,
                    p.base_move_slots.iter().map(|m| m.id).collect::<Vec<_>>(),
                )
            })
            .collect();
        let mut steps = Vec::new();
        for ply in 0..plies {
            if sim.outcome().is_some() {
                break;
            }
            let mut search =
                SkuctSearch::new(&sim, dex, self.cfg.clone(), seed ^ (0x9E37_79B9 * (ply as u64 + 1)));
            search.step(dex, iters);
            let mut joint = [search.best(0), search.best(1)];
            // The line exists to explain the move the analysis recommends, so
            // that move opens it. Without this the first ply comes from a
            // different search than the score above it — full information
            // instead of the blind root, a tenth of the playouts — and the
            // two can disagree, leaving the screen recommending one move and
            // illustrating another.
            if ply == 0 {
                if let Some(c) = from {
                    if search.actions(self.side).contains(&c) {
                        joint[self.side] = Some(c);
                    }
                }
            }
            if joint == [None, None] {
                break;
            }
            let Some(step) =
                enumerate_step_with_damage_mode(dex, &sim, joint, LINE_ENUM_CAP, DamageRollMode::Mean)
            else {
                break;
            };
            let Some(leaf) = step
                .leaves
                .into_iter()
                .max_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap_or(std::cmp::Ordering::Equal))
            else {
                break;
            };
            // Resolve switch targets before the step, while the party that
            // `switch N` indexes into is still the one the choice was made
            // against. A bare "switch 3" is not a reading of the line.
            let target = |side: usize, c: Option<SearchChoice>| -> Option<SpeciesId> {
                match c {
                    Some(SearchChoice::Switch(pos)) => sim.sides[side]
                        .party
                        .get(pos as usize - 1)
                        .map(|&slot| sim.sides[side].roster[slot as usize].species),
                    _ => None,
                }
            };
            let mine_target = target(self.side, joint[self.side]);
            let theirs_target = target(1 - self.side, joint[1 - self.side]);
            let effects = diff_actives(&sim, &leaf.battle);
            let mut next = leaf.battle;
            // The enumerator hands back a spent Oracle; the next ply's search
            // plays seeded rollouts off this state.
            next.reseed(rng.next());
            steps.push(LineStep {
                mine: joint[self.side],
                theirs: joint[1 - self.side],
                mine_target,
                theirs_target,
                iterations: iters,
                prob: leaf.prob,
                effects,
                outcome: next.outcome(),
            });
            sim = next;
        }
        PrincipalLine { assumed, steps }
    }

    /// Per opponent action, the number of iterations in which it was legal.
    /// Read next to `root_matrix`: a reply available in a third of the
    /// determinizations is a statement about a third of the candidate teams.
    pub fn root_replies(&self) -> &[(SearchChoice, u32)] {
        &self.avail
    }

    /// Pump `n` iterations, return the total run so far.
    pub fn step(&mut self, dex: &Dex, belief: &Belief, obs: &Observer, n: u32) -> u32 {
        for _ in 0..n {
            self.step_one(dex, belief, obs);
        }
        self.done
    }

    /// Distinct states in the determinized tree (see `SkuctSearch::node_count`).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
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

    /// Per-action dominated flags — the guaranteed-fail / certain-self-loss
    /// mask [`Self::best`] applies. Exposed because a harness that ranks by
    /// raw visits does NOT reproduce the shipped choice: it can report a move
    /// the product would never play (2026-07-27, `human_agreement` was doing
    /// exactly that, which sent a corpus review chasing a Swagger the bot
    /// could not have chosen).
    pub fn dominated(&self) -> &[bool] {
        &self.my_dominated
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

    /// Experimental fixed-budget best-arm rule: argmax empirical mean over
    /// the same eligible root actions as [`Self::best`]. Allocation remains
    /// caller-controlled (`step` for UCB, `step_forced` for a designed
    /// schedule); M17a uses this to audit simple-regret alternatives without
    /// changing the shipped argmax-visits policy.
    pub fn best_mean(&self) -> Option<SearchChoice> {
        let allowed = |a: usize| self.my_mask.as_ref().map_or(true, |m| m[a]);
        let means = self.means();
        (0..self.my_acts.len())
            .filter(|&a| allowed(a) && !self.my_dominated[a])
            .max_by(|&a, &b| means[a].total_cmp(&means[b]))
            .or_else(|| {
                (0..self.my_acts.len())
                    .filter(|&a| allowed(a))
                    .max_by(|&a, &b| means[a].total_cmp(&means[b]))
            })
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
            let mut belief = Belief::with_fallback_policy(
                dex,
                &self.pool,
                &observer,
                self.fallback_policy,
            );
            if let Some(prior) = self.prior.clone() {
                belief.set_prior(prior);
            }
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

/// One turn of a [`PrincipalLine`].
#[derive(Clone, Debug)]
pub struct LineStep {
    /// The analyzing side's action (`None` = it owed nothing).
    pub mine: Option<SearchChoice>,
    pub theirs: Option<SearchChoice>,
    /// Species a `Switch` action brings in, resolved against the party as it
    /// stood when the choice was made.
    pub mine_target: Option<SpeciesId>,
    pub theirs_target: Option<SpeciesId>,
    /// Playouts behind this step's two choices.
    pub iterations: u32,
    /// Probability of the outcome shown, over this step's chance events
    /// (1.0 when the step had none). The line follows the likeliest branch;
    /// this is how likely that was.
    pub prob: f64,
    /// What changed on the board — the line's content, in place of a
    /// protocol log the enumerator never produces.
    pub effects: Vec<LineEffect>,
    pub outcome: Option<nc2000_engine::battle::Outcome>,
}

/// One Pokémon the step moved. Only the mons that actually changed appear,
/// so a step reads as its consequences rather than as a board dump.
#[derive(Clone, Debug)]
pub struct LineEffect {
    pub side: usize,
    pub slot: u8,
    pub species: nc2000_engine::dex::SpeciesId,
    pub hp_before: i32,
    pub hp_after: i32,
    pub maxhp: i32,
    pub status_before: nc2000_engine::state::Status,
    pub status_after: nc2000_engine::state::Status,
    /// It is the one standing on the field after the step.
    pub active: bool,
    /// It came in during the step (a switch, or a replacement after a faint).
    pub switched_in: bool,
}

/// Engine runs one line step may spend on exact chance enumeration. Damage
/// is already collapsed to its mean, so a normal step resolves in a handful;
/// the cap only bounds the pathological ones (multi-hit into a Substitute
/// into a berry), which end the line instead of being approximated.
const LINE_ENUM_CAP: usize = 256;

/// Per-mon board diff across one step.
fn diff_actives(before: &Battle, after: &Battle) -> Vec<LineEffect> {
    let mut out = Vec::new();
    for side in 0..2 {
        for slot in 0..after.sides[side].roster.len() {
            let (b, a) = (&before.sides[side].roster[slot], &after.sides[side].roster[slot]);
            let active = after.sides[side].active == Some(slot as u8);
            let switched_in = active && before.sides[side].active != Some(slot as u8);
            if b.hp == a.hp && b.status == a.status && !switched_in {
                continue;
            }
            // A mon that was already down and stayed down did not do
            // anything this step; the faint bookkeeping around its status is
            // not a board event and reads as one.
            if b.hp == 0 && a.hp == 0 && !switched_in {
                continue;
            }
            out.push(LineEffect {
                side,
                slot: slot as u8,
                species: a.species,
                hp_before: b.hp,
                hp_after: a.hp,
                maxhp: a.maxhp,
                status_before: b.status,
                status_after: a.status,
                active,
                switched_in,
            });
        }
    }
    out
}

/// A searched continuation under one assumed opponent team.
#[derive(Clone, Debug)]
pub struct PrincipalLine {
    /// The opponent roster the line assumed: `(species, the four moves the
    /// determinizer gave it)` per roster slot. Meaningful for the mons whose
    /// set is still hidden; for a revealed one it is just the truth.
    pub assumed: Vec<(nc2000_engine::dex::SpeciesId, Vec<nc2000_engine::dex::MoveId>)>,
    pub steps: Vec<LineStep>,
}

/// Most-visited action index, skipping dominated ones while any alternative
/// survives — the same play rule as `BlindSearch::best`.
fn argmax_visits(visits: &[u32], dominated: Option<&[bool]>) -> usize {
    let ok = |i: usize| dominated.map(|d| !d.get(i).copied().unwrap_or(false)).unwrap_or(true);
    let mut best = None;
    for i in 0..visits.len() {
        if !ok(i) {
            continue;
        }
        if best.map(|b: usize| visits[i] > visits[b]).unwrap_or(true) {
            best = Some(i);
        }
    }
    best.unwrap_or_else(|| {
        (0..visits.len()).max_by_key(|&i| visits[i]).unwrap_or(0)
    })
}
