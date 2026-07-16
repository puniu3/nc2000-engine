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

    /// Belief-mediated M8 table lookup: own side by signature (public to
    /// us), opponent by the single consistent pool candidate. `None` ⇔ fall
    /// through to the determinized preview search.
    fn baked_pick(&mut self, battle: &Battle, side: usize) -> Option<SearchChoice> {
        let tables = self.tables.as_ref()?;
        let g = self.game.as_ref().expect("baked_pick after lifecycle setup");
        if g.belief.candidate_count() != 1 {
            return None; // collision pair (or fallback): identity unresolved
        }
        let opp = g.belief.alive()[0];
        debug_assert_eq!(
            tables.ids[opp],
            g.belief.candidate_id(opp),
            "TableSet/MetaPool pool-order drift"
        );
        let me = tables.side_index(battle, side)?;
        let (tab, i_am_a) = tables.pair_by_idx(me, opp)?;
        // sample the mixed equilibrium (same rule as BakedPreviewAgent)
        let p = if i_am_a { &tab.sol.p_a } else { &tab.sol.p_b };
        let u = self.rng.next_f64();
        let mut acc = 0.0;
        let mut pick = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
        for (i, &pr) in p.iter().enumerate() {
            acc += pr;
            if u < acc {
                pick = i;
                break;
            }
        }
        Some(SearchChoice::Team(tables.actions()[pick]))
    }

    // -------------------------------------------------------------- search

    fn search(&mut self, battle: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> SearchChoice {
        // Log-off base once per decision: the outer battle runs log-ON for
        // the observer, and determinize clones its input — don't pay for
        // cloning the whole protocol log every iteration.
        let mut base = battle.clone();
        base.set_log_enabled(false);
        let turn_cap = base.turn.saturating_add(self.cfg.horizon);

        let BlindAgent { cfg, rng, game, .. } = self;
        let g = game.as_ref().expect("search after lifecycle setup");

        let my_acts: Vec<SearchChoice> = choices.to_vec();
        let mut my_n = vec![0u32; my_acts.len()];
        let mut my_w = vec![0.0f64; my_acts.len()];
        let mut nodes: Vec<Node> = Vec::new();
        let mut table: HashMap<u64, usize> = HashMap::new();

        for _ in 0..cfg.iterations {
            let mut sim = g.belief.determinize(dex, &base, &g.observer, rng);
            let key = key_of(cfg, &sim);
            let root = match table.get(&key) {
                Some(&i) => i,
                None => {
                    let node = Node::at(&mut sim, dex);
                    debug_assert_eq!(
                        node.acts[side], my_acts,
                        "determinization changed the observer's own root actions"
                    );
                    nodes.push(node);
                    table.insert(key, nodes.len() - 1);
                    nodes.len() - 1
                }
            };
            let my_pick = select_global(cfg, rng, &mut my_n, &my_w);
            let mut force = [None, None];
            force[side] = Some(my_pick);
            let mut joint = [0usize; 2];
            let r = run_iteration(
                cfg, rng, &mut nodes, &mut table, &mut sim, dex, turn_cap, root, force,
                &mut joint,
            );
            my_w[my_pick] += if side == 0 { r } else { 1.0 - r };
        }

        let best = (0..my_acts.len()).max_by_key(|&a| my_n[a]).unwrap();
        my_acts[best]
    }
}

/// UCB1 over the global (information-set) root stats — same rule and rng
/// draw pattern as `smmcts::select_ucb`, on plain arrays.
fn select_global(cfg: &RmConfig, rng: &mut SplitMix64, n: &mut [u32], w: &[f64]) -> usize {
    let k = n.len();
    let untried: Vec<usize> = (0..k).filter(|&a| n[a] == 0).collect();
    let pick = if !untried.is_empty() {
        untried[rng.below(untried.len())]
    } else {
        let total: u32 = n.iter().sum();
        let ln_total = (total as f64).ln();
        let mut best = 0;
        let mut best_v = f64::NEG_INFINITY;
        for a in 0..k {
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
