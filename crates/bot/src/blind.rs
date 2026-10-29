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
//!   With `unseen_claw_root`, idle root statistics also share the hidden
//!   preempt coin; simulated transitions retain its sampled value.
//!
//! Final choice excludes joint-probe counts whenever eligible ordinary visits exist.
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

use crate::agent::{Agent, RootSearchStats};
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

struct AdaptiveMixture {
    probe_iterations: u32,
    rng: SplitMix64,
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
    adaptive_mixture: Option<AdaptiveMixture>,
    decision_policy: Option<Vec<f64>>,
    root_stats: RootSearchStats,
    root_pruning: bool,
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
            FallbackPolicy::default(),
        )
    }

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
            adaptive_mixture: None,
            decision_policy: None,
            root_stats: RootSearchStats::default(),
            root_pruning: false,
        }
    }

    pub fn with_adaptive_mixture(mut self, probe_iterations: u32) -> Self {
        assert!(probe_iterations < self.cfg.iterations);
        self.adaptive_mixture = Some(AdaptiveMixture {
            probe_iterations,
            rng: SplitMix64::new(self.rng.0 ^ 0xa24baed4963ee407),
        });
        self
    }

    pub fn with_root_pruning(mut self) -> Self {
        self.root_pruning = true;
        self
    }

    /// Set the next decision's iteration budget without resetting observations
    /// or the search RNG. This only changes allocation, not the search policy.
    pub fn set_iterations(&mut self, iterations: u32) {
        assert!(iterations > 0, "search budget must be positive");
        self.cfg.iterations = iterations;
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

    pub fn search_rng_state(&self) -> u64 {
        self.rng.0
    }

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
        if !matches!(choices[0], SearchChoice::Team(_)) {
            self.root_stats.searches += 1;
            if let Some(mixture) = self.adaptive_mixture.as_mut() {
                let mut search = BlindSearch::with_rng(battle, dex, self.cfg.clone(), side, self.rng.clone());
                if self.root_pruning { search.prune_dominated(); }
                debug_assert_eq!(search.actions(), choices);
                let probes = search.finish_with_typed_mixed_probes(dex, &g.belief, &g.observer, self.cfg.iterations, mixture.probe_iterations);
                self.root_stats.probed_searches += u64::from(probes > 0);
                self.root_stats.probe_iterations += u64::from(probes);
                self.rng = search.rng.clone();
                self.decision_policy = search.mixed_policy().map(|mixed| mixed.policy);
                let draw = mixture.rng.next_f64();
                if let Some(policy) = &self.decision_policy {
                    self.root_stats.mixed_searches += 1;
                    return draw_policy(choices, policy, draw).unwrap();
                }
                return search.best().expect("search called with a non-empty choice list");
            }
        }
        search_choose(&self.cfg, &mut self.rng, g, battle, dex, side, choices, self.root_pruning)
    }
}

fn search_choose(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    g: &GameState,
    battle: &Battle,
    dex: &Dex,
    side: usize,
    choices: &[SearchChoice],
    prune: bool,
) -> SearchChoice {
    let mut bs = BlindSearch::with_rng(battle, dex, cfg.clone(), side, rng.clone());
    bs.prune_preview_aliases(dex);
    if prune { bs.prune_dominated(); }
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

fn draw_policy(actions: &[SearchChoice], policy: &[f64], draw: f64) -> Option<SearchChoice> {
    let mut total = 0.;
    let mut last = None;
    for (&action, &mass) in actions.iter().zip(policy) {
        if mass > 0. { last = Some(action); }
        total += mass;
        if draw < total { return Some(action); }
    }
    last
}

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

enum ForcedRoot {
    Action { own: usize, reply: Option<SearchChoice> },
    TypeProbe,
}

struct RootTypeStats {
    samples: u32,
    replies: usize,
    cells: Vec<(u32, f64)>,
}

pub(crate) fn root_search_key(cfg: &RmConfig, dex: &Dex, battle: &mut Battle) -> u64 {
    if !cfg.unseen_claw_root || !battle.is_idle_move_request() {
        return key_of(cfg, dex, battle);
    }
    let coin = battle.quick_claw_roll;
    let speeds = [0,1].map(|side| battle.active_id(side).map(|id| (id,battle.poke(id).speed)));
    battle.quick_claw_roll = false;
    battle.update_all_speeds(dex);
    let key = key_of(cfg, dex, battle);
    battle.quick_claw_roll = coin;
    for (id,speed) in speeds.into_iter().flatten() { battle.poke_mut(id).speed=speed; }
    key
}

fn root_type_key(battle: &Battle) -> u64 {
    battle.type_key()
}

struct CachedDetRoot {
    sim: Battle,
    root: usize,
    type_index: usize,
}

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
    first_continuation: Option<crate::first_continuation::FirstContinuation>,
    history_continuation: Option<crate::history_continuation::HistoryContinuation>,
    rng: SplitMix64,
    /// Search resets omit the observer's protocol log.
    base: Battle,
    scratch: Option<Box<Battle>>,
    det_cache: FxHashMap<(Option<usize>, u64, [u8; 6], bool), CachedDetRoot>,
    det_cache_obs_rev: u64,
    turn_cap: u16,
    side: usize,
    /// The public own-side choice list — the information-set root.
    my_acts: Vec<SearchChoice>,
    my_n: Vec<u32>,
    joint_probe_n: Vec<u32>,
    my_w: Vec<f64>,
    /// M15: optional root-action legality mask (masked actions are never
    /// selected or reported best). `None` = all allowed, bit-identical to
    /// the pre-M15 behavior. Historical purpose — Max Total Level at
    /// preview — is enforced by the engine's own enumeration since the
    /// 2026-07-17 preview-space fix; the API stays (harmless, generic).
    my_mask: Option<Vec<bool>>,
    /// Final-choice exclusions: self-KO, sleep-forfeit risk, and proven no-ops.
    my_dominated: Vec<bool>,
    opponent_preview_prior: bool,
    /// Per-determinization roots + everything below (state-keyed).
    nodes: Vec<Node>,
    table: FxHashMap<u64, usize>,
    done: u32,
    /// Diagnostic columns use SearchChoice because legal reply lists vary by type.
    joint: Vec<(usize, SearchChoice, u32, f64)>,
    joint_index: FxHashMap<(usize, SearchChoice), usize>,
    root_avail: FxHashMap<usize, Vec<usize>>,
    root_node_type: FxHashMap<usize, usize>,
    root_types: Vec<RootTypeStats>,
    root_type_index: FxHashMap<u64, usize>,
    /// How often each opponent action was even LEGAL, across iterations.
    /// A blind root's opponent action list is determinization-dependent — a
    /// move exists only in the candidates that carry it — so a column's
    /// sample count conflates "rarely chosen" with "rarely available", and
    /// only this tells them apart.
    avail: Vec<(SearchChoice, u32)>,
}

impl BlindSearch {
    pub fn enable_history_continuation(&mut self, shared: bool) {
        assert_eq!(self.done, 0);
        assert!(self.first_continuation.is_none());
        assert!(!self.cfg.continuation_exp3_ix && !self.cfg.continuation_exp3_mean);
        self.history_continuation = Some(crate::history_continuation::HistoryContinuation::new(self.side, shared));
    }

    pub fn enable_compact_history_continuation(&mut self, shared: bool) {
        self.enable_history_continuation(shared);
        self.history_continuation.as_mut().unwrap().use_compact_index();
    }

    pub fn history_continuation(&self) -> Option<&crate::history_continuation::HistoryContinuation> {
        self.history_continuation.as_ref()
    }

    pub fn enable_first_continuation(&mut self, shared: bool) {
        assert!(self.history_continuation.is_none());
        assert_eq!(self.done, 0);
        assert!(!self.cfg.continuation_exp3_ix && !self.cfg.continuation_exp3_mean);
        self.first_continuation = Some(crate::first_continuation::FirstContinuation::new(self.side, shared));
    }

    pub fn first_continuation(&self) -> Option<&crate::first_continuation::FirstContinuation> {
        self.first_continuation.as_ref()
    }

    pub fn set_opponent_preview_prior(&mut self, enabled: bool) {
        self.opponent_preview_prior = enabled;
    }

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
        if cfg.unseen_claw_root && base.is_idle_move_request() {
            base.quick_claw_roll=false;
            base.update_all_speeds(dex);
        }
        let turn_cap = base.turn.saturating_add(cfg.horizon);
        let my_acts = base.legal_choices(dex, side);
        let my_dominated = my_acts
            .iter()
            .map(|&c| {
                crate::smmcts::dominated_reason(&base, dex, side, c, cfg.mask_rules).is_some()
            })
            .collect();
        BlindSearch {
            first_continuation: None,
            history_continuation: None,
            cfg,
            rng,
            base,
            scratch: None,
            det_cache: FxHashMap::default(),
            det_cache_obs_rev: u64::MAX,
            turn_cap,
            side,
            my_n: vec![0; my_acts.len()],
            joint_probe_n: vec![0; my_acts.len()],
            my_w: vec![0.0; my_acts.len()],
            my_mask: None,
            my_dominated,
            opponent_preview_prior: false,
            my_acts,
            nodes: Vec::new(),
            table: FxHashMap::default(),
            done: 0,
            joint: Vec::new(),
            joint_index: FxHashMap::default(),
            root_avail: FxHashMap::default(),
            root_node_type: FxHashMap::default(),
            root_types: Vec::new(),
            root_type_index: FxHashMap::default(),
            avail: Vec::new(),
        }
    }

    /// One iteration: fresh determinization, global-UCB own root pick
    /// forced into the shared `run_iteration`. Returns the side-0 reward.
    pub fn step_one(&mut self, dex: &Dex, belief: &Belief, obs: &Observer) -> f64 {
        self.step_one_impl(dex, belief, obs, None, None)
    }

    fn resolve_root(&mut self, dex: &Dex, sim: &mut Battle) -> (usize, usize) {
        let key = root_search_key(&self.cfg, dex, sim);
        let root = match self.table.get(&key) {
            Some(&i) => i,
            None => {
                let node = Node::at(sim, dex);
                debug_assert_eq!(
                    node.acts[self.side], self.my_acts,
                    "determinization changed the observer's own root actions"
                );
                self.nodes.push(node);
                let idx = self.nodes.len() - 1;
                self.table.insert(key, idx);
                idx
            }
        };
        if !self.cfg.unseen_claw_root {
            if let Some(&ti) = self.root_node_type.get(&root) {
                return (root, ti);
            }
        }
        let type_key = root_type_key(sim);
        let opp = 1 - self.side;
        let replies = self.nodes[root].acts[opp].len();
        let my_len = self.my_acts.len();
        let ti = *self.root_type_index.entry(type_key).or_insert_with(|| {
            self.root_types.push(RootTypeStats {
                samples: 0,
                replies,
                cells: vec![(0, 0.); my_len * replies],
            });
            self.root_types.len() - 1
        });
        if !self.cfg.unseen_claw_root {
            self.root_node_type.insert(root, ti);
        }
        (root, ti)
    }

    /// Predict the RNG draws and resulting opponent party of `Belief::determinize_in_place`
    /// when no pick distribution or extra per-mon RNG draws occur.
    #[inline]
    fn predict_determinization_draws(
        &self,
        obs: &Observer,
        rng_probe: &mut SplitMix64,
    ) -> (u64, bool, [u8; 6], usize) {
        let seed = rng_probe.next();
        let mut prng = nc2000_engine::prng::Prng::new(seed);
        let qc = prng.random_chance(60, 256);
        let post_qc_seed = prng.seed();

        let opp = 1 - self.side;
        let opp_side = &self.base.sides[opp];
        let roster_len = opp_side.roster.len();
        let party_len = opp_side.party.len();
        let mut party_key = [0xFFu8; 6];
        for (i, &s) in opp_side.party.iter().enumerate() {
            party_key[i] = s;
        }
        if party_len < roster_len {
            let mut appeared = [false; 6];
            for (i, p) in opp_side.roster.iter().enumerate() {
                appeared[i] = p.previously_switched_in > 0 || p.is_active;
            }
            let mut hidden_positions = [0usize; 6];
            let mut hidden_len = 0usize;
            for pos in 0..party_len {
                if !appeared[party_key[pos] as usize] {
                    hidden_positions[hidden_len] = pos;
                    hidden_len += 1;
                }
            }
            let mut pool = [0u8; 6];
            let mut pool_len = 0usize;
            for s in 0..roster_len {
                if !appeared[s] {
                    pool[pool_len] = s as u8;
                    pool_len += 1;
                }
            }
            for &pos in &hidden_positions[..hidden_len] {
                let idx = rng_probe.below(pool_len);
                let new_slot = pool[idx];
                pool_len -= 1;
                pool[idx] = pool[pool_len];
                party_key[pos] = new_slot;
            }
            if let Some(cap) = self.base.preview_level_caps[opp].filter(|_| hidden_len > 0) {
                let level = |mask: u8| {
                    obs.mons()
                        .iter()
                        .enumerate()
                        .filter(|(slot, _)| mask & (1 << slot) != 0)
                        .map(|(_, mon)| u32::from(mon.level))
                        .sum::<u32>()
                };
                let mask = party_key[..party_len]
                    .iter()
                    .fold(0u8, |m, &slot| m | (1 << slot));
                if level(mask) > cap {
                    let public = appeared[..roster_len]
                        .iter()
                        .enumerate()
                        .fold(0u8, |m, (slot, &seen)| m | (u8::from(seen) << slot));
                    let mut legal = [0u8; 64];
                    let mut legal_len = 0usize;
                    for m in 0..(1u8 << roster_len) {
                        if m.count_ones() as usize == party_len
                            && (m & public) == public
                            && level(m) <= cap
                        {
                            legal[legal_len] = m;
                            legal_len += 1;
                        }
                    }
                    if legal_len > 0 {
                        let chosen_mask = legal[rng_probe.below(legal_len)];
                        pool_len = 0;
                        for slot in 0..roster_len {
                            if (chosen_mask & (1 << slot)) != 0 && !appeared[slot] {
                                pool[pool_len] = slot as u8;
                                pool_len += 1;
                            }
                        }
                        for &pos in &hidden_positions[..hidden_len] {
                            let idx = rng_probe.below(pool_len);
                            let new_slot = pool[idx];
                            pool_len -= 1;
                            pool[idx] = pool[pool_len];
                            party_key[pos] = new_slot;
                        }
                    }
                }
            }
        }
        (post_qc_seed, qc, party_key, party_len)
    }

    fn determinize_and_resolve_root(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        pick: Option<usize>,
    ) -> (Box<Battle>, usize, usize) {
        let obs_rev = obs.revision();
        if self.det_cache_obs_rev != obs_rev {
            self.det_cache.clear();
            self.det_cache_obs_rev = obs_rev;
        }
        if self.base.queue.is_empty() && !belief.has_active_pick_draws(obs) {
            let mut rng_probe = self.rng.clone();
            if let Some(fb_key) =
                belief.predict_fallback_draws(pick, &self.base, 1 - self.side, &mut rng_probe)
            {
                let (post_qc_seed, qc, party_key, party_len) =
                    self.predict_determinization_draws(obs, &mut rng_probe);
                let cache_key = (pick, fb_key, party_key, qc);
                if let Some(cached) = self.det_cache.get(&cache_key) {
                    self.rng = rng_probe;
                    let mut sim = match self.scratch.take() {
                        Some(mut sim) => {
                            sim.as_mut().clone_from(&cached.sim);
                            sim
                        }
                        None => Box::new(cached.sim.clone()),
                    };
                    sim.reseed(post_qc_seed);
                    std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                    return (sim, cached.root, cached.type_index);
                }

                let mut sim = match self.scratch.take() {
                    Some(mut sim) => {
                        sim.as_mut().clone_from(&self.base);
                        sim
                    }
                    None => Box::new(self.base.clone()),
                };
                belief.determinize_in_place(dex, &mut sim, obs, pick, &mut self.rng);
                if self.cfg.unseen_claw_root && sim.is_idle_move_request() {
                    sim.update_all_speeds(dex);
                }
                let opp = 1 - self.side;
                let can_cache = self.rng.0 == rng_probe.0
                    && sim.sides[opp].party.as_slice() == &party_key[..party_len];
                std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                let (root, type_index) = self.resolve_root(dex, &mut sim);
                if can_cache {
                    std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                    let cached_sim = (*sim).clone();
                    std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                    self.det_cache.insert(
                        cache_key,
                        CachedDetRoot {
                            sim: cached_sim,
                            root,
                            type_index,
                        },
                    );
                }
                return (sim, root, type_index);
            }
        }

        let mut sim = match self.scratch.take() {
            Some(mut sim) => {
                sim.as_mut().clone_from(&self.base);
                sim
            }
            None => Box::new(self.base.clone()),
        };
        belief.determinize_in_place(dex, &mut sim, obs, pick, &mut self.rng);
        if self.cfg.unseen_claw_root && sim.is_idle_move_request() {
            sim.update_all_speeds(dex);
        }
        std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
        let (root, type_index) = self.resolve_root(dex, &mut sim);
        (sim, root, type_index)
    }

    fn step_one_impl(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        leaf: Option<&mut dyn FnMut(&mut Battle, &mut SplitMix64, bool) -> f64>,
        trace: Option<&mut dyn FnMut(crate::smmcts::SearchTrace<'_>)>,
    ) -> f64 {
        let pick = belief.sample(&mut self.rng);
        let (mut sim, root, type_index) =
            self.determinize_and_resolve_root(dex, belief, obs, pick);
        let my_pick = if self.cfg.root_revisit_interval > 0 && !self.is_preview()
            && self.done > 0 && self.done % self.cfg.root_revisit_interval == 0
        {
            let action = (0..self.my_acts.len())
                .filter(|&i| self.my_mask.as_ref().map_or(true, |mask| mask[i]))
                .min_by_key(|&i| self.my_n[i]).unwrap();
            self.my_n[action] += 1;
            action
        } else { select_global(
            &self.cfg,
            &mut self.rng,
            &mut self.my_n,
            &self.my_w,
            self.my_mask.as_deref(),
        ) };
        let mut force = [None, None];
        force[self.side] = Some(my_pick);
        if self.opponent_preview_prior && self.is_preview() {
            force[1 - self.side] = belief.sample_preview_prior(
                pick, obs, &self.nodes[root].acts[1 - self.side], &mut self.rng,
            );
        }
        let mut joint = [0usize; 2];
        let r = if let Some(policy) = self.history_continuation.as_mut() {
            assert!(leaf.is_none());
            crate::smmcts::run_iteration_with_root_child_value(
                &self.cfg, &mut self.rng, &mut self.nodes, &mut self.table,
                &mut sim, dex, self.turn_cap, root, force, &mut joint, &mut 0,
                &mut |sim, rng, rollout| crate::smmcts::default_leaf(&self.cfg, sim, dex, self.turn_cap, rng, rollout),
                trace, None, None, Some(policy),
            )
        } else if let Some(policy) = self.first_continuation.as_mut() {
            assert!(leaf.is_none());
            crate::smmcts::run_iteration_with_root_child_value(
                &self.cfg, &mut self.rng, &mut self.nodes, &mut self.table,
                &mut sim, dex, self.turn_cap, root, force, &mut joint, &mut 0,
                &mut |sim, rng, rollout| crate::smmcts::default_leaf(&self.cfg, sim, dex, self.turn_cap, rng, rollout),
                trace, None, Some(policy), None,
            )
        } else if let Some(leaf) = leaf {
            crate::smmcts::run_iteration_with_leaf(
                &self.cfg, &mut self.rng, &mut self.nodes, &mut self.table,
                &mut sim, dex, self.turn_cap, root, force, &mut joint, &mut 0, leaf, None,
            )
        } else if let Some(trace) = trace {
            crate::smmcts::run_iteration_with_leaf(
                &self.cfg, &mut self.rng, &mut self.nodes, &mut self.table,
                &mut sim, dex, self.turn_cap, root, force, &mut joint, &mut 0,
                &mut |sim, rng, rollout| {
                    if rollout {
                        crate::mcts::playout_value(sim, dex, &self.cfg.playout, self.turn_cap, rng, self.cfg.rollout_rules())
                    } else {
                        crate::smmcts::leaf_eval(&self.cfg, sim, dex)
                    }
                }, Some(trace),
            )
        } else {
            run_iteration(
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
            )
        };
        std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
        self.scratch = Some(sim);
        self.record_joint(root, type_index, my_pick, joint, r);
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
        self.step_forced_impl(dex, belief, obs, ForcedRoot::Action { own: my_pick, reply: None })
    }

    // Unavailable replies use the sampled opponent's tree policy.
    pub fn step_forced_joint(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        my_pick: usize,
        reply: SearchChoice,
    ) -> f64 {
        let value=self.step_forced_impl(dex, belief, obs, ForcedRoot::Action { own: my_pick, reply: Some(reply) });
        self.joint_probe_n[my_pick]+=1;
        value
    }

    fn step_forced_impl(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        forcing: ForcedRoot,
    ) -> f64 {
        self.step_forced_impl_observed(dex, belief, obs, forcing, None)
    }

    pub fn step_forced_joint_observed(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        my_pick: usize,
        reply: SearchChoice,
        trace: &mut impl FnMut(crate::smmcts::SearchTrace<'_>),
    ) -> f64 {
        let value = self.step_forced_impl_observed(dex, belief, obs,
            ForcedRoot::Action { own: my_pick, reply: Some(reply) }, Some(trace));
        self.joint_probe_n[my_pick] += 1;
        value
    }

    fn step_forced_impl_observed(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        forcing: ForcedRoot,
        trace: Option<&mut dyn FnMut(crate::smmcts::SearchTrace<'_>)>,
    ) -> f64 {
        self.step_forced_impl_with_root_child_value(dex,belief,obs,forcing,trace,None)
    }

    pub fn step_forced_joint_with_root_child_value(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        my_pick: usize,
        reply: SearchChoice,
        evaluate: &mut impl FnMut(&Battle) -> Option<f64>,
    ) -> f64 {
        let value = self.step_forced_impl_with_root_child_value(dex,belief,obs,
            ForcedRoot::Action {own:my_pick,reply:Some(reply)},None,Some(evaluate));
        self.joint_probe_n[my_pick] += 1;
        value
    }

    fn step_forced_impl_with_root_child_value(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        forcing: ForcedRoot,
        trace: Option<&mut dyn FnMut(crate::smmcts::SearchTrace<'_>)>,
        root_child_value: Option<&mut dyn FnMut(&Battle) -> Option<f64>>,
    ) -> f64 {
        if let ForcedRoot::Action { own, .. } = forcing {
            assert!(own < self.my_acts.len(), "forced root action out of range");
            assert!(self.my_mask.as_ref().map_or(true, |mask| mask[own]), "forced root action is masked");
        }
        let pick = belief.sample(&mut self.rng);
        let (mut sim, root, type_index) =
            self.determinize_and_resolve_root(dex, belief, obs, pick);
        let (my_pick, forced_reply, override_reply) = match forcing {
            ForcedRoot::Action { own, reply } => {
                self.my_n[own] += 1;
                (own, reply.and_then(|a| self.nodes[root].acts[1 - self.side].iter().position(|&c| c == a)), reply.is_some())
            }
            ForcedRoot::TypeProbe => {
                let replies = self.nodes[root].acts[1 - self.side].len();
                let stats = &self.root_types[type_index];
                let selected = (0..self.my_acts.len()).filter(|&own|
                    !self.my_dominated[own] && self.my_mask.as_ref().map_or(true, |mask| mask[own]))
                    .flat_map(|own| (0..replies).map(move |reply| (own, reply)))
                    .min_by_key(|&(own, reply)| stats.cells[own * replies + reply].0);
                match selected {
                    Some((own, reply)) => {
                        self.my_n[own] += 1;
                        self.joint_probe_n[own] += 1;
                        (own, Some(reply), true)
                    }
                    None => (select_global(&self.cfg, &mut self.rng, &mut self.my_n,
                        &self.my_w, self.my_mask.as_deref()), None, false),
                }
            }
        };
        let mut force = [None, None];
        force[self.side] = Some(my_pick);
        if self.opponent_preview_prior && self.is_preview() {
            force[1 - self.side] = belief.sample_preview_prior(
                pick, obs, &self.nodes[root].acts[1 - self.side], &mut self.rng,
            );
        }
        if override_reply {
            force[1 - self.side] = forced_reply;
        }
        let mut joint = [0usize; 2];
        let r = if let Some(policy) = self.history_continuation.as_mut() {
            assert!(root_child_value.is_none());
            crate::smmcts::run_iteration_with_root_child_value(
                &self.cfg, &mut self.rng, &mut self.nodes, &mut self.table,
                &mut sim, dex, self.turn_cap, root, force, &mut joint, &mut 0,
                &mut |sim, rng, rollout| crate::smmcts::default_leaf(&self.cfg, sim, dex, self.turn_cap, rng, rollout),
                trace, None, None, Some(policy),
            )
        } else if let Some(policy) = self.first_continuation.as_mut() {
            assert!(root_child_value.is_none());
            crate::smmcts::run_iteration_with_root_child_value(
                &self.cfg, &mut self.rng, &mut self.nodes, &mut self.table,
                &mut sim, dex, self.turn_cap, root, force, &mut joint, &mut 0,
                &mut |sim, rng, rollout| crate::smmcts::default_leaf(&self.cfg, sim, dex, self.turn_cap, rng, rollout),
                trace, None, Some(policy), None,
            )
        } else if let Some(evaluate) = root_child_value {
            crate::smmcts::run_iteration_with_root_child_value(
                &self.cfg,&mut self.rng,&mut self.nodes,&mut self.table,&mut sim,dex,
                self.turn_cap,root,force,&mut joint,&mut 0,
                &mut |sim,rng,rollout| crate::smmcts::default_leaf(&self.cfg,sim,dex,self.turn_cap,rng,rollout),
                trace,Some(evaluate),None,None,
            )
        } else if let Some(trace) = trace {
            crate::smmcts::run_iteration_with_leaf(
                &self.cfg, &mut self.rng, &mut self.nodes, &mut self.table,
                &mut sim, dex, self.turn_cap, root, force, &mut joint, &mut 0,
                &mut |sim, rng, rollout| crate::smmcts::default_leaf(
                    &self.cfg, sim, dex, self.turn_cap, rng, rollout), Some(trace),
            )
        } else { run_iteration(
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
        ) };
        std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
        self.scratch = Some(sim);
        self.record_joint(root, type_index, my_pick, joint, r);
        self.my_w[my_pick] += if self.side == 0 { r } else { 1.0 - r };
        self.done += 1;
        r
    }

    fn record_joint(&mut self, root: usize, type_index: usize, my_pick: usize, joint: [usize; 2], r: f64) {
        let opp = 1 - self.side;
        let acts = &self.nodes[root].acts[opp];
        let stats = &mut self.root_types[type_index];
        stats.samples += 1;
        if joint[opp] < stats.replies {
            let cell = &mut stats.cells[my_pick * stats.replies + joint[opp]];
            cell.0 += 1;
            cell.1 += if self.side == 0 { r } else { 1. - r };
        }
        let indices = self.root_avail.entry(root).or_insert_with(|| {
            acts.iter().map(|&choice| {
                match self.avail.iter().position(|(action, _)| *action == choice) {
                    Some(index) => index,
                    None => {
                        self.avail.push((choice, 0));
                        self.avail.len() - 1
                    }
                }
            }).collect()
        });
        for &index in indices.iter() {
            self.avail[index].1 += 1;
        }
        let Some(&opp_act) = acts.get(joint[opp]) else { return };
        let mine = if self.side == 0 { r } else { 1.0 - r };
        let index = match self.joint_index.entry((my_pick, opp_act)) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                self.joint.push((my_pick, opp_act, 0, 0.0));
                *entry.insert(self.joint.len() - 1)
            }
        };
        self.joint[index].2 += 1;
        self.joint[index].3 += mine;
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

    /// The callback returns a side-0 reward in [0, 1]; the flag marks a newly expanded leaf.
    pub fn step_with_leaf(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        n: u32,
        leaf: &mut impl FnMut(&mut Battle, &mut SplitMix64, bool) -> f64,
    ) -> u32 {
        for _ in 0..n {
            self.step_one_impl(dex, belief, obs, Some(leaf), None);
        }
        self.done
    }

    pub fn step_with_endgame(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        n: u32,
        session: &mut crate::endgame::EndgameSession,
        clock: &mut impl FnMut() -> f64,
    ) -> u32 {
        let cfg = self.cfg.clone();
        let turn_cap = self.turn_cap;
        for _ in 0..n {
            if session.expired(clock()) { break; }
            self.step_with_leaf(dex, belief, obs, 1, &mut |sim, rng, rollout| {
                session.leaf(dex, &cfg, turn_cap, sim, rng, rollout, clock)
            });
        }
        self.done
    }

    pub fn step_observed(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        n: u32,
        trace: &mut impl FnMut(crate::smmcts::SearchTrace<'_>),
    ) -> u32 {
        for _ in 0..n {
            self.step_one_impl(dex, belief, obs, None, Some(trace));
        }
        self.done
    }

    /// Distinct states in the determinized tree (see `SkuctSearch::node_count`).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Evaluates a fixed root action without updating the tree or its search RNG.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_frozen(
        &self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        action: SearchChoice,
        policy: [crate::frozen::FrozenPolicy; 2],
        seed: u64,
        trace: &mut impl FnMut(crate::frozen::FrozenChoice<'_>),
    ) -> crate::frozen::FrozenResult {
        assert!(self.my_acts.contains(&action));
        let mut forced = [None, None];
        forced[self.side] = Some(action);
        self.evaluate_frozen_joint(dex, belief, obs, forced, policy, seed, trace)
    }

    /// Forced opponent actions must be legal in every sampled determinization.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_frozen_joint(
        &self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        forced: [Option<SearchChoice>; 2],
        policy: [crate::frozen::FrozenPolicy; 2],
        seed: u64,
        trace: &mut impl FnMut(crate::frozen::FrozenChoice<'_>),
    ) -> crate::frozen::FrozenResult {
        let mut rng = SplitMix64::new(seed);
        let pick = belief.sample(&mut rng);
        let mut sim = belief.determinize_with(dex, &self.base, obs, pick, &mut rng);
        if self.cfg.unseen_claw_root && sim.is_idle_move_request() { sim.update_all_speeds(dex); }
        crate::frozen::evaluate_sim(
            dex, &self.cfg, &mut sim, &self.nodes, &self.table,
            forced, policy, &mut rng, trace,
        )
    }

    pub fn iterations(&self) -> u32 {
        self.done
    }

    pub fn search_rng_state(&self) -> u64 {
        self.rng.0
    }

    /// The global root's actions — the side's public legal choice list.
    pub fn actions(&self) -> &[SearchChoice] {
        &self.my_acts
    }

    /// Per-action visit counts on the global (information-set) root stats.
    pub fn visits(&self) -> &[u32] {
        &self.my_n
    }

    /// Final-choice exclusions applied by `best`; raw visits do not include this filter.
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

    pub fn prune_dominated(&mut self) {
        let allowed: Vec<bool> = self.my_dominated.iter().enumerate()
            .map(|(i, &dominated)| !dominated && self.my_mask.as_ref().map_or(true, |m| m[i]))
            .collect();
        if allowed.iter().any(|&x| x) {
            self.mask_actions(&allowed);
        }
    }

    pub fn prune_preview_aliases(&mut self, dex: &Dex) -> usize {
        if !self.is_preview() || self.base.sides[self.side].roster.iter().any(|p| {
            p.base_move_slots.iter().chain(p.move_slots.iter()).any(|m| {
                matches!(dex.moves.key(m.id),
                    "beatup" | "metronome" | "mimic" | "mirrormove" | "sketch" | "transform")
            })
        }) {
            return 0;
        }
        assert_eq!(self.done, 0, "preview aliases must be pruned before search");
        let mut allowed = self.my_mask.clone().unwrap_or_else(|| vec![true; self.my_acts.len()]);
        let mut seen = Vec::new();
        let mut removed = 0;
        for (i, action) in self.my_acts.iter().enumerate() {
            if !allowed[i] { continue; }
            let SearchChoice::Team(mut triple) = *action else { continue };
            if triple[2] == 0 { continue; }
            triple[1..].sort_unstable();
            if seen.contains(&triple) {
                allowed[i] = false;
                removed += 1;
            } else {
                seen.push(triple);
            }
        }
        if removed > 0 { self.mask_actions(&allowed); }
        removed
    }

    pub fn prune_rest_aliases(&mut self, dex: &Dex) -> usize {
        use nc2000_engine::state::{Status, DK};
        let Some(id) = self.base.active_id(self.side) else { return 0 };
        let own = self.base.poke(id);
        if own.status != Status::Slp || own.status_state.source != Some(id)
            || own.status_state.get_int(DK::StartTime) != 3
        {
            return 0;
        }
        let allowed = self.my_mask.clone().unwrap_or_else(|| vec![true; self.my_acts.len()]);
        let mut indices: Vec<_> = (0..self.my_acts.len()).filter(|&i| allowed[i]).collect();
        indices.sort_by_key(|&i| self.my_dominated[i]);
        let mut representatives: Vec<_> = indices.iter().map(|&i| self.my_acts[i]).collect();
        crate::smmcts::collapse_sleep_choices(&self.base, dex, self.side, &mut representatives);
        let removed = indices.len() - representatives.len();
        if removed > 0 {
            self.mask_actions(&self.my_acts.iter().enumerate()
                .map(|(i,a)| allowed[i] && representatives.contains(a)).collect::<Vec<_>>());
        }
        removed
    }

    pub fn joint_probe_visits(&self) -> &[u32] { &self.joint_probe_n }

    pub fn best(&self) -> Option<SearchChoice> {
        let allowed=|i:usize|self.my_mask.as_ref().map_or(true,|m|m[i]);
        let safe=(0..self.my_acts.len()).any(|i|allowed(i) && !self.my_dominated[i]);
        let eligible=|i:usize|allowed(i) && (!safe || !self.my_dominated[i]);
        let ordinary=(0..self.my_acts.len()).any(|i|eligible(i) && self.my_n[i]>self.joint_probe_n[i]);
        (0..self.my_acts.len()).filter(|&i|eligible(i))
            .max_by_key(|&i|if ordinary {self.my_n[i]-self.joint_probe_n[i]} else {self.my_n[i]})
            .map(|i|self.my_acts[i])
    }

    /// Actual execution probabilities, not `rootPolicy` visit diagnostics.
    /// Neither empirical mode certifies equilibrium or policy improvement.
    pub fn decision_policy(&self, mode: crate::decision::PolicyMode) -> Option<crate::decision::DecisionPolicy<SearchChoice>> {
        use crate::decision::{DecisionPolicy, PolicyMode};
        let best = self.best()?;
        if mode == PolicyMode::EstimatedMix {
            if let Some(mixed) = self.mixed_policy() {
                if let Ok(policy) = DecisionPolicy::new("estimated-mix", self.my_acts.iter().copied().zip(mixed.policy)) {
                    return Some(policy);
                }
            }
            return Some(DecisionPolicy::pure("estimated-mix-fallback", best));
        }
        if mode == PolicyMode::VisitSample {
            let allowed = |i: usize| self.my_mask.as_ref().map_or(true, |mask| mask[i]);
            let safe = (0..self.my_acts.len()).any(|i| allowed(i) && !self.my_dominated[i]);
            let entries = self.my_acts.iter().enumerate()
                .filter(|&(i, _)| allowed(i) && (!safe || !self.my_dominated[i]))
                .map(|(i, &action)| (action, f64::from(self.my_n[i] - self.joint_probe_n[i])));
            if let Ok(policy) = DecisionPolicy::new("visit-sample", entries) { return Some(policy); }
            return Some(DecisionPolicy::pure("visit-sample-fallback", best));
        }
        Some(DecisionPolicy::pure("legacy", best))
    }

    pub fn root_policy_evidence(&self) -> (Vec<bool>, Vec<crate::root_policy::RootType>) {
        let eligible: Vec<bool> = (0..self.my_acts.len()).map(|i|
            self.my_mask.as_ref().map_or(true, |m| m[i]) && !self.my_dominated[i]).collect();
        let types = self.root_types.iter().map(|t| crate::root_policy::RootType {
            samples: t.samples,
            replies: t.replies,
            cells: t.cells.iter().enumerate().filter(|(_, (n, _))| *n > 0)
                .map(|(i, &(samples, sum))| crate::root_policy::RootCell {
                    own: i / t.replies, reply: i % t.replies, samples, mean: sum / f64::from(samples),
                }).collect(),
        }).collect();
        (eligible, types)
    }

    pub fn mixed_policy(&self) -> Option<crate::root_policy::MixedRoot> {
        if self.my_acts.iter().any(|a|matches!(a,SearchChoice::Team(_))) {return None;}
        let (eligible, types) = self.root_policy_evidence();
        crate::root_policy::bayesian_mixed_root(&eligible, self.done, &types)
    }

    pub fn estimated_mix_gain(&self) -> Option<f64> {
        if self.my_acts.iter().any(|a|matches!(a,SearchChoice::Team(_))) {return None;}
        let (eligible, types) = self.root_policy_evidence();
        crate::root_policy::estimated_bayesian_mix_gain(&eligible, self.done, &types)
    }

    pub fn step_joint_probes(&mut self,dex:&Dex,belief:&Belief,obs:&Observer,n:u32) -> u32 {
        for _ in 0..n {
            let mut selected=None;
            for own in 0..self.my_acts.len() {
                if self.my_dominated[own] || !self.my_mask.as_ref().map_or(true,|m|m[own]) {continue;}
                for &(reply,available) in &self.avail {
                    if available!=self.done {continue;}
                    let count=self.joint.iter().find(|&&(i,a,_,_)|i==own && a==reply).map(|c|c.2).unwrap_or(0);
                    if selected.as_ref().map_or(true,|&(_,_,n)|count<n) {selected=Some((own,reply,count));}
                }
            }
            if let Some((own,reply,_))=selected {self.step_forced_joint(dex,belief,obs,own,reply);}
            else {self.step_one(dex,belief,obs);}
        }
        self.done
    }

    pub fn step_typed_joint_probes(&mut self, dex: &Dex, belief: &Belief, obs: &Observer, n: u32) -> u32 {
        if self.is_preview() { return self.step(dex, belief, obs, n); }
        for _ in 0..n {
            self.step_forced_impl(dex, belief, obs, ForcedRoot::TypeProbe);
        }
        self.done
    }

    pub fn finish_with_mixed_probes(&mut self,dex:&Dex,belief:&Belief,obs:&Observer,total:u32,reserved:u32) -> u32 {
        self.finish_mixed_probes(dex, belief, obs, total, reserved, false)
    }

    pub fn finish_with_typed_mixed_probes(&mut self,dex:&Dex,belief:&Belief,obs:&Observer,total:u32,reserved:u32) -> u32 {
        self.finish_mixed_probes(dex, belief, obs, total, reserved, true)
    }

    fn finish_mixed_probes(&mut self,dex:&Dex,belief:&Belief,obs:&Observer,total:u32,reserved:u32,typed:bool) -> u32 {
        assert!(reserved<total && self.done<=total-reserved);
        self.step(dex,belief,obs,total-reserved-self.done);
        let probe=reserved>0 && self.estimated_mix_gain().is_some_and(|gain|gain>0.02) && self.mixed_policy().is_none();
        if probe {
            if typed { self.step_typed_joint_probes(dex,belief,obs,reserved); }
            else { self.step_joint_probes(dex,belief,obs,reserved); }
            reserved
        } else {self.step(dex,belief,obs,reserved);0}
    }

    pub fn sample_mixed(&self, draw: f64) -> Option<SearchChoice> {
        assert!(draw.is_finite() && (0.0..1.0).contains(&draw));
        let Some(mixed) = self.mixed_policy() else { return self.best(); };
        draw_policy(&self.my_acts, &mixed.policy, draw)
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
    let mut n_untried = 0usize;
    for a in 0..k {
        if n[a] == 0 && ok(a) {
            n_untried += 1;
        }
    }
    let pick = if n_untried > 0 {
        let mut target = rng.below(n_untried);
        let mut chosen = 0;
        for a in 0..k {
            if n[a] == 0 && ok(a) {
                if target == 0 {
                    chosen = a;
                    break;
                }
                target -= 1;
            }
        }
        chosen
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
    opp_pick_policy: Option<(Vec<f64>, f64)>,
    move_marginals: [[f64; 4]; 6],
    move_turns: [u32; 6],
    game: Option<GameState>,
}

impl OpenAgent {
    /// Same config surface as `BlindAgent` (the RM root layer fields are
    /// ignored — the blind/open play rule is argmax over the global root).
    pub fn new(cfg: RmConfig, tables: Option<Arc<TableSet>>, seed: u64) -> Self {
        OpenAgent {
            cfg,
            rng: SplitMix64::new(seed),
            tables,
            opp_pick_policy: None,
            move_marginals: [[0.0; 4]; 6],
            move_turns: [0; 6],
            game: None,
        }
    }

    /// Condition hidden opponent bench sampling during battle on a 60-action
    /// canonical preview distribution `policy` (e.g., solved Nash equilibrium),
    /// mixed with `smoothing` over all 3-mon subsets containing already-revealed mons.
    pub fn with_opponent_pick_policy(mut self, policy: Vec<f64>, smoothing: f64) -> Self {
        self.opp_pick_policy = Some((policy, smoothing));
        self
    }

    /// Per-slot, per-move counterfactual root marginal values `\Delta Q_{-m_k}`
    /// accumulated over mid-battle move decisions, along with active move-decision
    /// turn counts per roster slot.
    pub fn move_telemetry(&self) -> ([[f64; 4]; 6], [u32; 6]) {
        (self.move_marginals, self.move_turns)
    }

    /// Explicitly initialize the per-game observer & pinned belief at Team Preview
    /// before an external forced Turn-0 preview choice is applied to `battle`.
    pub fn init_at_preview(&mut self, battle: &Battle, side: usize) {
        let observer = Observer::new(battle, side);
        let belief = Belief::pinned_from_battle(battle, &observer);
        self.game = Some(GameState {
            side,
            last_turn: battle.turn,
            observer,
            belief,
        });
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
            if let Some((ref pol, smoothing)) = self.opp_pick_policy {
                g.belief.condition_canonical_pick_policy(pol, smoothing, &g.observer);
            }
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
        let mut bs = BlindSearch::with_rng(battle, dex, self.cfg.clone(), side, self.rng.clone());
        bs.prune_preview_aliases(dex);
        debug_assert_eq!(bs.actions(), choices, "root action set drifted from caller's choices");
        for _ in 0..self.cfg.iterations {
            bs.step_one(dex, &g.belief, &g.observer);
        }
        self.rng = bs.rng.clone();
        let picked = bs.best().expect("search called with a non-empty choice list");

        // Record mid-battle move-level counterfactual regret \Delta Q_{-m_k}
        if !is_preview {
            if let Some(active_u8) = battle.sides[side].active {
                let slot = (active_u8 as usize).min(5);
                if slot < battle.sides[side].roster.len()
                    && choices.iter().any(|c| matches!(c, SearchChoice::Move(_)))
                {
                    self.move_turns[slot] += 1;
                    if let SearchChoice::Move(chosen_mid) = picked {
                        let means = bs.means();
                        let visits = bs.visits();
                        if let Some(chosen_idx) = choices.iter().position(|&c| c == picked) {
                            let q_best = means[chosen_idx];
                            let q_alt = (0..choices.len())
                                .filter(|&i| i != chosen_idx && visits[i] > 0)
                                .map(|i| means[i])
                                .fold(f64::NEG_INFINITY, f64::max);
                            if q_alt.is_finite() {
                                let dq = (q_best - q_alt).max(0.0);
                                let mon = &battle.sides[side].roster[slot];
                                for (m_idx, ms) in mon.base_move_slots.iter().enumerate().take(4) {
                                    if ms.id == chosen_mid {
                                        self.move_marginals[slot][m_idx] += dq;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        picked
    }
}

impl Agent for BlindAgent {
    fn root_search_stats(&self) -> Option<RootSearchStats> { Some(self.root_stats) }

    fn root_policy(&mut self, battle: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> Vec<f64> {
        let picked = self.choose(battle, dex, side, choices);
        if let Some(policy) = &self.decision_policy { return policy.clone(); }
        if matches!(choices[0], SearchChoice::Team(_)) {
            if let (Some(tables), Some(game)) = (&self.tables, &self.game) {
                if game.belief.candidate_count() == 1 {
                    if let Some(me) = tables.side_index(battle, side) {
                        if let Some((table, first)) = tables.pair_by_idx(me, game.belief.alive()[0]) {
                            let policy = if first { &table.sol.p_a } else { &table.sol.p_b };
                            let aligned: Vec<f64> = choices.iter().map(|action|
                                tables.actions().iter().position(|t| SearchChoice::Team(*t) == *action)
                                    .map_or(0., |i| policy[i])).collect();
                            if (aligned.iter().sum::<f64>() - 1.).abs() < 1e-8 { return aligned; }
                        }
                    }
                }
            }
        }
        choices.iter().map(|&c| if c == picked {1.} else {0.}).collect()
    }

    fn name(&self) -> String {
        let mut name = format!("blind:{}:{}:{}", self.cfg.iterations, self.cfg.c, self.cfg.hp_buckets);
        if let Some(mixture) = &self.adaptive_mixture {
            name.push_str(&format!(":mix-probes={}", mixture.probe_iterations));
        }
        if self.root_pruning { name.push_str(":root-prune"); }
        if self.cfg.unseen_claw_root { name.push_str(":unseen-claw-root"); }
        if self.cfg.canonical_claw_key { name.push_str(":canonical-claw"); }
        if self.cfg.collapse_sleep_actions { name.push_str(":sleep-collapse"); }
        if self.cfg.root_revisit_interval > 0 { name.push_str(&format!(":revisit-{}", self.cfg.root_revisit_interval)); }
        name
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let is_preview = matches!(choices[0], SearchChoice::Team(_));

        self.decision_policy = None;
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

#[cfg(test)]
mod mixed_selection_tests {
    use super::*;

    #[test]
    fn root_revisits_match_forced_steps_preserve_masks_and_skip_preview() {
        let dex = conformance::load_dex();
        let set = |species: &str, moves: &[&str]| -> nc2000_engine::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({"species":species,"name":species,"level":50,"moves":moves})).unwrap()
        };
        let mut b = Battle::from_fixture(&dex,"1,2,3,4",
            &[set("Snorlax", &["Return","Curse","Rest","Sleep Talk"]),set("Magikarp", &["Splash"])],
            &[set("Suicune", &["Surf","Rest","Sleep Talk"]),set("Pidgey", &["Tackle"])]).unwrap();
        let preview = b.clone();
        b.choose(&dex,0,"team 1,2").unwrap();
        b.choose(&dex,1,"team 1,2").unwrap();
        for base in [&preview,&b] {
            for side in 0..2 {
                let obs = Observer::new(base,side);
                let belief = Belief::pinned_from_battle(base,&obs);
                let cfg = RmConfig {c:0.4,..Default::default()};
                let mut reference = BlindSearch::new(base,&dex,cfg.clone(),side,901);
                let mut candidate = BlindSearch::new(base,&dex,RmConfig {root_revisit_interval:64,..cfg},side,901);
                let allowed: Vec<_> = (0..reference.actions().len()).map(|i|reference.is_preview() || i!=1).collect();
                reference.mask_actions(&allowed);
                candidate.mask_actions(&allowed);
                for i in 0..300 {
                    if i>0 && i%64==0 && !reference.is_preview() {
                        let action = (0..allowed.len()).filter(|&i|allowed[i]).min_by_key(|&i|reference.visits()[i]).unwrap();
                        reference.step_forced(&dex,&belief,&obs,action);
                    } else {reference.step(&dex,&belief,&obs,1);}
                }
                candidate.step(&dex,&belief,&obs,300);
                assert_eq!(candidate.visits(),reference.visits());
                assert_eq!(candidate.means(),reference.means());
                assert_eq!(candidate.rng.0,reference.rng.0);
                assert_eq!(candidate.iterations(),300);
                assert!(candidate.visits().iter().zip(&allowed).all(|(&n,&ok)|ok || n==0));
            }
        }
    }

    #[test]
    fn root_pruning_preserves_agent_rng_and_all_masked_fallback() {
        let dex = conformance::load_dex();
        let set = |species: &str, moves: &[&str]| -> nc2000_engine::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({"name":species,"species":species,
                "level":50,"moves":moves,"happiness":255})).unwrap()
        };
        for (species, moves, expected) in [
            ("Alakazam", vec!["Recover", "Psychic"], vec![true, false]),
            ("Gengar", vec!["Destiny Bond", "Perish Song"], vec![true, true]),
            ("Alakazam", vec!["Psychic", "Thunder Punch"], vec![false, false]),
        ] {
            let mut battle = Battle::from_fixture(&dex, "1,2,3,4",
                &[set(species, &moves)], &[set("Snorlax", &["Return"])]).unwrap();
            let preview = battle.clone();
            let mut observer = Observer::new(&battle, 0);
            battle.choose(&dex, 0, "team 1").unwrap();
            battle.choose(&dex, 1, "team 1").unwrap();
            observer.observe(&battle, &dex);
            let legal = battle.legal_choices(&dex, 0);
            let cfg = RmConfig { iterations: 300, c: 0.4, ..Default::default() };
            for prune in [false, true] {
                let mut belief = Belief::pinned_from_battle(&preview, &observer);
                belief.sync(&dex, &observer);
                let mut reference = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 17);
                assert_eq!(reference.dominated(), &expected);
                if prune { reference.prune_dominated(); }
                reference.step(&dex, &belief, &observer, cfg.iterations);
                let mut agent = BlindAgent::new(cfg.clone(), Arc::new(MetaPool { teams: vec![] }), None, 17);
                if prune { agent = agent.with_root_pruning(); }
                agent.game = Some(GameState { side: 0, last_turn: battle.turn,
                    observer: observer.clone(), belief });
                assert_eq!(Some(agent.choose(&battle, &dex, 0, &legal)), reference.best());
                assert_eq!(agent.search_rng_state(), reference.rng.0);
                if prune && expected.iter().any(|&x| !x) {
                    assert!(reference.visits().iter().zip(&expected).all(|(&n, &masked)| !masked || n == 0));
                } else {
                    assert!(reference.visits().iter().all(|&n| n > 0));
                }
            }
        }
    }

    #[test]
    fn adaptive_agent_uses_the_search_mixture_and_a_separate_draw_stream() {
        let dex = conformance::load_dex();
        let set = |species: &str, moves: &[&str]| -> nc2000_engine::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({"name":species,"species":species,"ability":"",
                "level":50,"moves":moves,"happiness":255,
                "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}})).unwrap()
        };
        for (foe, moves, iterations, expect_mix) in [
            ("Wobbuffet", vec!["Counter", "Mirror Coat", "Safeguard"], 27000, true),
            ("Magikarp", vec!["Splash"], 200, false),
        ] {
            let mut battle = Battle::from_fixture(&dex, "1,2,3,4",
                &[set("Starmie", &["Return", "Surf"])], &[set(foe, &moves)]).unwrap();
            let preview = battle.clone();
            let mut observer = Observer::new(&battle, 0);
            let mut belief = Belief::pinned_from_battle(&battle, &observer);
            battle.choose(&dex, 0, "team 1").unwrap();
            battle.choose(&dex, 1, "team 1").unwrap();
            for (side, hp) in [(0, 68), (1, 120)] {
                let id = battle.active_id(side).unwrap(); battle.poke_mut(id).hp = hp;
            }
            observer.observe(&battle, &dex); belief.sync(&dex, &observer);
            let cfg = RmConfig {iterations, c:0.4, ..Default::default()};
            let reserve = iterations * 2 / 5;
            let mut reference = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 7);
            let probes = reference.finish_with_typed_mixed_probes(&dex, &belief, &observer, iterations, reserve);
            assert_eq!(reference.mixed_policy().is_some(), expect_mix, "{:?}", reference.root_matrix());
            let mut draws = SplitMix64::new(7 ^ 0xa24baed4963ee407);
            let expected = reference.sample_mixed(draws.next_f64()).unwrap();
            let mut agent = BlindAgent::new(cfg.clone(), Arc::new(MetaPool {teams:vec![]}), None, 7)
                .with_adaptive_mixture(reserve);
            agent.game = Some(GameState {side:0, last_turn:battle.turn, observer, belief});
            let legal = battle.legal_choices(&dex, 0);
            assert_eq!(agent.choose(&battle, &dex, 0, &legal), expected);
            assert_eq!(agent.search_rng_state(), reference.rng.0);
            assert_eq!(agent.adaptive_mixture.as_ref().unwrap().rng.0, draws.0);
            let stats = RootSearchStats {searches:1, probed_searches:u64::from(probes>0),
                probe_iterations:u64::from(probes), mixed_searches:u64::from(expect_mix)};
            assert_eq!(agent.root_search_stats(), Some(stats));
            let obs = agent.game.as_ref().unwrap().observer.clone();
            let mut exporter = BlindAgent::new(cfg.clone(), Arc::new(MetaPool {teams:vec![]}), None, 7)
                .with_adaptive_mixture(reserve);
            exporter.game = Some(GameState {side:0, last_turn:battle.turn,
                belief:Belief::pinned_from_battle(&preview,&obs), observer:obs});
            let expected_policy = reference.mixed_policy().map(|m|m.policy)
                .unwrap_or_else(||legal.iter().map(|&a|if a==expected {1.} else {0.}).collect());
            assert_eq!(exporter.root_policy(&battle,&dex,0,&legal), expected_policy);
            assert_eq!(exporter.root_search_stats(), Some(stats));
            assert_eq!(exporter.search_rng_state(), agent.search_rng_state());
            let mut forced = battle.clone(); let id = forced.active_id(0).unwrap();
            forced.poke_mut(id).get_move_slot_mut(dex.moves.id("return").unwrap()).unwrap().pp = 0;
            let only = forced.legal_choices(&dex,0); assert_eq!(only.len(),1);
            assert_eq!(exporter.root_policy(&forced,&dex,0,&only),vec![1.]);
            assert_eq!(exporter.root_search_stats(), Some(stats));
            if !expect_mix {
                assert_eq!(probes, 0);
                let g = agent.game.as_ref().unwrap();
                let mut plain = BlindSearch::new(&battle, &dex, cfg, 0, 7);
                plain.step(&dex, &g.belief, &g.observer, iterations);
                assert_eq!(plain.best(), Some(expected));
                assert_eq!(plain.rng.0, agent.search_rng_state());
                assert_eq!(plain.root_matrix(), reference.root_matrix());
            }
        }
    }

    #[test]
    fn unseen_claw_root_shares_policy_but_preserves_transitions_and_phase_boundaries() {
        let dex = conformance::load_dex();
        let set: nc2000_engine::battle::PokemonSet = serde_json::from_value(serde_json::json!({
            "species":"Snorlax","name":"Snorlax","level":50,"item":"Quick Claw","moves":["Rest","Swagger"]
        })).unwrap();
        let mut b = Battle::from_fixture(&dex,"1,2,3,4",&[set.clone()],&[set]).unwrap();
        let cfg = RmConfig {unseen_claw_root:true,..Default::default()};
        assert!(!RmConfig::default().unseen_claw_root);
        for idle in [false,true] {
            if idle {
                b.choose(&dex,0,"team 1").unwrap();
                b.choose(&dex,1,"team 1").unwrap();
            }
            assert_eq!(b.is_idle_move_request(),idle);
            let mut roots=Vec::new();let mut continuation=Vec::new();
            for coin in [false,true] {
                b.quick_claw_roll=coin;
                let before=format!("{b:?}");
                roots.push(root_search_key(&cfg,&dex,&mut b));
                continuation.push(key_of(&cfg,&dex,&mut b));
                assert_eq!(before,format!("{b:?}"));
            }
            assert_eq!(roots[0]==roots[1],idle);
            assert_ne!(continuation[0],continuation[1]);
        }
        let obs=Observer::new(&b,0);
        let belief=Belief::pinned_from_battle(&b,&obs);
        let mut alternative=b.clone();
        alternative.quick_claw_roll=!b.quick_claw_roll;
        alternative.update_all_speeds(&dex);
        let mut states=[&b,&alternative].map(|state| BlindSearch::new(state,&dex,RmConfig {horizon:1,..cfg.clone()},0,771));
        assert_eq!(states[0].dominated(),states[1].dominated());
        for state in &mut states { state.step(&dex,&belief,&obs,256); }
        assert_eq!(states[0].visits(),states[1].visits());
        assert_eq!(states[0].means(),states[1].means());
        assert_eq!(states[0].node_count(),states[1].node_count());
        let mut search=BlindSearch::new(&b,&dex,RmConfig {horizon:1,..cfg.clone()},0,771);
        search.step(&dex,&belief,&obs,256);
        let before=(search.visits().to_vec(),search.means(),search.node_count());
        let mut roots=std::collections::BTreeSet::new();
        let mut coins=std::collections::BTreeSet::new();
        for seed in 0..64 {
            let mut first=true;
            search.evaluate_frozen_joint(&dex,&belief,&obs,[None,None],
                [crate::frozen::FrozenPolicy::MostVisited;2],seed,&mut |choice| {
                    if first {
                        first=false;
                        roots.insert(choice.node.expect("shared frozen root"));
                        coins.insert(choice.battle.quick_claw_roll);
                    }
                });
        }
        assert_eq!(roots.len(),1);
        assert_eq!(coins.len(),2);
        assert_eq!(before,(search.visits().to_vec(),search.means(),search.node_count()));
        b.mid_turn=true;
        assert!(!b.is_idle_move_request());
        b.quick_claw_roll=false;
        let key=root_search_key(&cfg,&dex,&mut b);
        b.quick_claw_roll=true;
        assert_ne!(key,root_search_key(&cfg,&dex,&mut b));
    }

    #[test]
    fn root_types_distinguish_exact_hp_but_do_not_reveal_the_turn_coin() {
        let dex = conformance::load_dex();
        let set: nc2000_engine::battle::PokemonSet = serde_json::from_value(serde_json::json!({
            "name":"starmie", "species":"starmie", "ability":"", "level":50, "moves":["Return", "Surf"]
        })).unwrap();
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &[set.clone()], &[set]).unwrap();
        let choices = [0, 1].map(|side| Some(battle.legal_choices(&dex, side)[0]));
        battle.apply_choices(&dex, choices).unwrap();
        battle.quick_claw_roll = false;
        let key = root_type_key(&mut battle);
        let rng = battle.prng.clone();
        battle.quick_claw_roll = true;
        assert_eq!(root_type_key(&mut battle), key);
        assert!(battle.quick_claw_roll);
        assert_eq!(format!("{rng:?}"), format!("{:?}", battle.prng));
        let cfg = RmConfig { hp_buckets: 4, threshold_key: false, ..Default::default() };
        let opponent = battle.active_id(1).unwrap();
        battle.poke_mut(opponent).hp = 100;
        let key = root_type_key(&mut battle);
        let abstract_key = key_of(&cfg, &dex, &mut battle);
        battle.poke_mut(opponent).hp = 101;
        assert_eq!(key_of(&cfg, &dex, &mut battle), abstract_key);
        assert_ne!(root_type_key(&mut battle), key);
    }

    #[test]
    fn mixed_sampling_respects_masks_without_consuming_search_randomness() {
        let dex=conformance::load_dex();
        let set: nc2000_engine::battle::PokemonSet=serde_json::from_value(serde_json::json!({
            "name":"starmie","species":"starmie","ability":"","level":50,"moves":["Return","Surf"]
        })).unwrap();
        let battle=Battle::from_fixture(&dex,"1,2,3,4",&[set.clone()],&[set]).unwrap();
        let mut search=BlindSearch::new(&battle,&dex,RmConfig::default(),0,123);
        let actions=["return","surf"].map(|m|SearchChoice::Move(dex.moves.id(m).unwrap()));
        search.my_acts=actions.to_vec();search.my_n=vec![20000,20000];search.joint_probe_n=vec![0,0];search.my_w=vec![10000.,10000.];
        search.my_dominated=vec![false,false];search.done=40000;
        search.avail=vec![(actions[1],40000),(actions[0],40000)];
        search.joint=vec![(0,actions[0],10000,0.),(0,actions[1],10000,10000.),
            (1,actions[0],10000,10000.),(1,actions[1],10000,0.)];
        search.root_types = vec![RootTypeStats { samples: 40000, replies: 2,
            cells: vec![(10000, 0.), (10000, 10000.), (10000, 10000.), (10000, 0.)] }];
        let rng=search.rng.0;
        use crate::decision::PolicyMode;
        let policy = search.decision_policy(PolicyMode::EstimatedMix).unwrap();
        assert_eq!(policy.source(), "estimated-mix");
        assert_eq!(policy.actions().len(), 2);
        assert_eq!(policy.sample(0.), actions[0]);
        assert_eq!(policy.sample(0.9), actions[1]);
        assert_eq!(search.sample_mixed(0.),Some(actions[0]));
        assert_eq!(search.sample_mixed(0.5),Some(actions[1]));
        assert_eq!(search.sample_mixed(1.-f64::EPSILON),Some(actions[1]));
        assert_eq!(search.rng.0,rng);
        assert_eq!(search.visits(),&[20000,20000]);
        search.my_n=vec![20000,10000];
        assert_eq!(search.best(),Some(actions[0]));
        search.my_n[1]+=30000;search.joint_probe_n[1]+=30000;
        assert_eq!(search.best(),Some(actions[0]));
        let visits = search.decision_policy(PolicyMode::VisitSample).unwrap();
        assert!((visits.actions()[0].probability - 2. / 3.).abs() < 1e-12);
        search.mask_actions(&[false,true]);
        assert!(search.mixed_policy().is_none());
        assert_eq!(search.sample_mixed(0.),Some(actions[1]));
        for mode in [PolicyMode::Legacy, PolicyMode::VisitSample, PolicyMode::EstimatedMix] {
            assert_eq!(search.decision_policy(mode).unwrap().actions().len(), 1);
            assert_eq!(search.decision_policy(mode).unwrap().sample(0.), actions[1]);
        }
        search.mask_actions(&[true,true]);search.my_dominated[0]=true;
        assert!(search.mixed_policy().is_none());
        assert_eq!(search.sample_mixed(0.),Some(actions[1]));
    }

    #[test]
    fn determinization_cache_is_bit_identical() {
        let dex = conformance::load_dex();
        let pool = crate::preview::load_meta_pool(std::path::Path::new(
            "../../data/meta-pool-v0/meta-pool.json",
        ));
        let t0 = &pool.teams[0].sets;
        let t1 = &pool.teams[1].sets;
        let mut battle = Battle::from_fixture(&dex, "10,20,30,40", t0, t1).unwrap();
        battle.set_log_enabled(true);
        let mut obs = Observer::new(&battle, 0);
        let mut belief = Belief::new(&dex, &pool, &obs);
        // Advance past team preview into turn 1 idle move request.
        let c0 = battle.legal_choices(&dex, 0)[0];
        let c1 = battle.legal_choices(&dex, 1)[0];
        battle.apply_choices(&dex, [Some(c0), Some(c1)]).unwrap();
        obs.observe(&battle, &dex);
        belief.sync(&dex, &obs);

        let cfg = RmConfig::default();
        let mut cached = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 0xCAFE_BABE);
        cached.step(&dex, &belief, &obs, 400);
        assert!(
            !cached.det_cache.is_empty(),
            "determinization cache should populate on standard battle root"
        );

        let mut uncached = BlindSearch::new(&battle, &dex, cfg, 0, 0xCAFE_BABE);
        for _ in 0..400 {
            uncached.det_cache.clear();
            uncached.step_one(&dex, &belief, &obs);
        }

        assert_eq!(cached.visits(), uncached.visits());
        assert_eq!(cached.means(), uncached.means());
        assert_eq!(cached.best(), uncached.best());
        assert_eq!(cached.search_rng_state(), uncached.search_rng_state());
        assert_eq!(cached.node_count(), uncached.node_count());
        assert_eq!(cached.root_matrix(), uncached.root_matrix());
        assert_eq!(cached.root_replies(), uncached.root_replies());
    }
}
