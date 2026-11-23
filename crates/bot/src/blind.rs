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
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::{Dex, SpeciesId};
use nc2000_engine::fxhash::FxHashMap;
use nc2000_engine::prng::DamageRollMode;
use nc2000_engine::state::{Battle, PokeId, Pokemon, Status, DK};

use crate::agent::{Agent, RootSearchStats};
use crate::belief::{Belief, FallbackPolicy, TeamPlanPick};
use crate::observe::Observer;
use crate::player_state::{clean_sets_from_roster, submitted_team, LocalPlayer, ObservedPosition};
use crate::preview::{MetaPool, TableSet};
use crate::prior::BeliefPrior;
use crate::rng::SplitMix64;
#[cfg(test)]
use crate::smmcts::key_of;
use crate::smmcts::{key_of_cached, run_iteration_cached, Node, RmConfig, SkuctSearch};

enum BeliefSource {
    Pool {
        pool: Arc<MetaPool>,
        fallback_policy: FallbackPolicy,
    },
    OpenSheet {
        explicit_sets: Option<Vec<PokemonSet>>,
    },
}

struct PendingPreview {
    side: usize,
    rosters: [Vec<Pokemon>; 2],
}

impl PendingPreview {
    fn matches(&self, battle: &Battle, side: usize) -> bool {
        if self.side != side {
            return false;
        }
        (0..2).all(|s| {
            let cur = &battle.sides[s].roster;
            let saved = &self.rosters[s];
            cur.len() == saved.len()
                && cur.iter().zip(saved.iter()).all(|(a, b)| {
                    a.base_species == b.base_species
                        && a.level == b.level
                        && a.name == b.name
                })
        })
    }
}

struct PlayerGameState {
    local: LocalPlayer,
    side: usize,
    last_turn: u16,
    observer: Observer,
    belief: Belief,
}

struct AdaptiveMixture {
    probe_iterations: u32,
    rng: SplitMix64,
}

struct ImperfectInfoCore {
    cfg: RmConfig,
    rng: SplitMix64,
    tables: Option<Arc<TableSet>>,
    source: BeliefSource,
    /// M18 community belief prior (inert when `source` is `OpenSheet`).
    prior: Option<Arc<BeliefPrior>>,
    pick_prior: Option<Arc<crate::pick_prior::PickPrior>>,
    own_preview_policy: Option<Vec<f64>>,
    opp_pick_policy: Option<(Vec<f64>, f64)>,
    preview_iterations: Option<u32>,
    game: Option<PlayerGameState>,
    pending_preview: Option<PendingPreview>,
    adaptive_mixture: Option<AdaptiveMixture>,
    adaptive_battle_budget: bool,
    decision_policy: Option<Vec<f64>>,
    root_stats: RootSearchStats,
    track_move_telemetry: bool,
    move_marginals: [[f64; 4]; 6],
    move_turns: [u32; 6],
}

impl ImperfectInfoCore {
    fn new(
        cfg: RmConfig,
        source: BeliefSource,
        tables: Option<Arc<TableSet>>,
        seed: u64,
        track_move_telemetry: bool,
    ) -> Self {
        Self {
            cfg,
            rng: SplitMix64::new(seed),
            tables,
            source,
            prior: None,
            pick_prior: None,
            own_preview_policy: None,
            opp_pick_policy: None,
            preview_iterations: None,
            game: None,
            pending_preview: None,
            adaptive_mixture: None,
            adaptive_battle_budget: false,
            decision_policy: None,
            root_stats: RootSearchStats::default(),
            track_move_telemetry,
            move_marginals: [[0.0; 4]; 6],
            move_turns: [0; 6],
        }
    }

    fn set_adaptive_mixture(&mut self, probe_iterations: u32) {
        assert!(!self.adaptive_battle_budget, "adaptive battle budget requires legacy root selection");
        assert!(probe_iterations < self.cfg.iterations);
        self.adaptive_mixture = Some(AdaptiveMixture {
            probe_iterations,
            rng: SplitMix64::new(self.rng.0 ^ 0xa24baed4963ee407),
        });
    }

    fn set_preview_iterations(&mut self, preview_iterations: u32) {
        assert!(
            preview_iterations > 0,
            "preview search budget must be positive"
        );
        self.preview_iterations = Some(preview_iterations);
    }

    fn set_iterations(&mut self, iterations: u32) {
        assert!(iterations > 0, "search budget must be positive");
        self.cfg.iterations = iterations;
    }

    fn format_name(&self, prefix: &str) -> String {
        let mut name = format!(
            "{prefix}:{}:{}:{}",
            self.cfg.iterations, self.cfg.c, self.cfg.hp_buckets
        );
        if self.adaptive_battle_budget {
            name.push_str(":adaptive-budget");
        }
        if let Some(mixture) = &self.adaptive_mixture {
            name.push_str(&format!(":mix-probes={}", mixture.probe_iterations));
        }
        name
    }

    fn record_move_telemetry(
        &mut self,
        model: &Battle,
        side: usize,
        choices: &[SearchChoice],
        picked: SearchChoice,
        search: &BlindSearch,
    ) {
        if !self.track_move_telemetry {
            return;
        }
        if let Some(active_u8) = model.sides[side].active {
            let slot = (active_u8 as usize).min(5);
            if slot < model.sides[side].roster.len()
                && choices.iter().any(|c| matches!(c, SearchChoice::Move(_)))
            {
                self.move_turns[slot] += 1;
                if let SearchChoice::Move(chosen_mid) = picked {
                    let means = search.means();
                    let visits = search.visits();
                    let actions = search.actions();
                    if let Some(chosen_idx) = actions.iter().position(|&c| c == picked) {
                        let q_best = means[chosen_idx];
                        let q_alt = (0..actions.len())
                            .filter(|&i| i != chosen_idx && visits[i] > 0)
                            .map(|i| means[i])
                            .fold(f64::NEG_INFINITY, f64::max);
                        if q_alt.is_finite() {
                            let dq = (q_best - q_alt).max(0.0);
                            let mon = &model.sides[side].roster[slot];
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

    fn preview_table_pick(&mut self, dex: &Dex) -> Option<SearchChoice> {
        let tables = self.tables.as_ref()?;
        let g = self
            .game
            .as_ref()
            .expect("preview_table_pick after lifecycle setup");
        match &self.source {
            BeliefSource::Pool { .. } => {
                baked_preview_pick_sets(tables, &g.belief, dex, g.local.own_sets(), &mut self.rng)
            }
            BeliefSource::OpenSheet { .. } => g.belief.pinned_sets().and_then(|opp_sets| {
                open_preview_pick_sets(tables, dex, g.local.own_sets(), opp_sets, &mut self.rng)
            }),
        }
    }

    fn search(
        &mut self,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let is_preview = matches!(choices[0], SearchChoice::Team(_));
        let (position, has_belief_preview_prior) = {
            let g = self.game.as_mut().expect("search after lifecycle setup");
            g.belief.sync(dex, &g.observer);
            let pos = g
                .local
                .position(dex, &mut g.belief, &g.observer, &mut self.rng)
                .expect("player reconstruction failed");
            let has_prior = g.belief.has_preview_prior(&g.observer);
            (pos, has_prior)
        };
        let opp_prior = self.opp_pick_policy.is_some() || has_belief_preview_prior;
        if !is_preview {
            self.root_stats.searches += 1;
            if let Some(mixture) = self.adaptive_mixture.as_mut() {
                let mut search = BlindSearch::from_observation_with_rng(
                    &position,
                    dex,
                    self.cfg.clone(),
                    side,
                    self.rng.clone(),
                );
                search.prune_preview_aliases(dex);
                if opp_prior {
                    search.set_opponent_preview_prior(true);
                }
                debug_assert_eq!(search.actions(), choices);
                let g = self.game.as_ref().expect("search after lifecycle setup");
                let probes = search.finish_with_typed_mixed_probes(
                    dex,
                    &g.belief,
                    &g.observer,
                    self.cfg.iterations,
                    mixture.probe_iterations,
                );
                self.root_stats.probed_searches += u64::from(probes > 0);
                self.root_stats.probe_iterations += u64::from(probes);
                self.rng = search.rng.clone();
                self.decision_policy = search.mixed_policy().map(|mixed| mixed.policy);
                let draw = mixture.rng.next_f64();
                let picked = if let Some(policy) = &self.decision_policy {
                    self.root_stats.mixed_searches += 1;
                    draw_policy(choices, policy, draw).unwrap()
                } else {
                    search
                        .best()
                        .expect("search called with a non-empty choice list")
                };
                self.record_move_telemetry(position.model(), side, choices, picked, &search);
                return picked;
            }
        }
        let mut cfg = self.cfg.clone();
        if is_preview {
            if let Some(p_iters) = self.preview_iterations {
                cfg.iterations = p_iters;
            }
        }
        let mut bs = BlindSearch::from_observation_with_rng(
            &position,
            dex,
            cfg.clone(),
            side,
            self.rng.clone(),
        );
        bs.prune_preview_aliases(dex);
        if opp_prior {
            bs.set_opponent_preview_prior(true);
        }
        debug_assert_eq!(
            bs.actions(),
            choices,
            "root action set drifted from caller's choices"
        );
        let picked = {
            let g = self.game.as_ref().expect("search after lifecycle setup");
            if self.adaptive_battle_budget && !is_preview {
                crate::adaptive_budget::AdaptiveBattleBudget::default()
                    .step(&mut bs, dex, &g.belief, &g.observer, cfg.iterations);
            } else {
                bs.step(dex, &g.belief, &g.observer, cfg.iterations);
            }
            bs.best()
                .expect("search called with a non-empty choice list")
        };
        self.rng = bs.rng.clone();
        if !is_preview {
            self.record_move_telemetry(position.model(), side, choices, picked, &bs);
        }
        picked
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
            Some(g) => {
                g.side != side
                    || is_preview
                    || battle.turn < g.last_turn
                    || !g.local.is_compatible(battle)
            }
        };
        if stale {
            let pending = self
                .pending_preview
                .take()
                .filter(|p| p.matches(battle, side));
            let (own_sets, preview_opp_sets) = match pending {
                Some(p) => (
                    clean_sets_from_roster(&p.rosters[side], dex),
                    Some(clean_sets_from_roster(&p.rosters[1 - side], dex)),
                ),
                None => (submitted_team(battle, dex, side), None),
            };
            let (local, observer) =
                LocalPlayer::new_with_own_sets(battle, dex, side, own_sets)
                    .expect("player observation failed");
            let belief = match &self.source {
                BeliefSource::Pool {
                    pool,
                    fallback_policy,
                } => {
                    let mut b = Belief::with_fallback_policy(
                        dex,
                        pool,
                        &observer,
                        *fallback_policy,
                    );
                    if let Some(prior) = self.prior.clone() {
                        b.set_prior(prior);
                    }
                    b
                }
                BeliefSource::OpenSheet { explicit_sets } => {
                    let opp_sets = explicit_sets
                        .clone()
                        .or(preview_opp_sets)
                        .unwrap_or_else(|| submitted_team(battle, dex, 1 - side));
                    Belief::pinned_from_sets(dex, "opponent", &opp_sets, &observer)
                        .expect("pinned opponent team must match public preview")
                }
            };
            self.game = Some(PlayerGameState {
                local,
                side,
                last_turn: battle.turn,
                observer,
                belief,
            });
        }
        {
            let g = self.game.as_mut().unwrap();
            g.last_turn = battle.turn;
            g.local
                .observe(battle, dex, &mut g.observer)
                .expect("player observation failed");
            g.belief.sync(dex, &g.observer);
            if let Some(ref pick_prior) = self.pick_prior {
                let own_preview = g.local.own_preview_mons(dex);
                g.belief
                    .condition_pick_prior(dex, pick_prior, &own_preview, &g.observer);
            }
            if let Some((ref pol, smoothing)) = self.opp_pick_policy {
                g.belief
                    .condition_canonical_pick_policy(pol, smoothing, &g.observer);
            }
        }

        if choices.len() == 1 {
            return choices[0];
        }
        if is_preview {
            if let Some(ref own_pol) = self.own_preview_policy {
                let canon_acts = crate::preview::preview_actions();
                if own_pol.len() == canon_acts.len() {
                    let mut legal_probs = Vec::with_capacity(canon_acts.len());
                    let mut legal_choices = Vec::with_capacity(canon_acts.len());
                    let mut p_sum = 0.0f64;
                    for (&triple, &prob) in canon_acts.iter().zip(own_pol.iter()) {
                        let sc = SearchChoice::Team(triple);
                        if prob > 0.0 && choices.contains(&sc) {
                            legal_choices.push(sc);
                            legal_probs.push(prob);
                            p_sum += prob;
                        }
                    }
                    if p_sum > 0.0 && !legal_choices.is_empty() {
                        for p in &mut legal_probs {
                            *p /= p_sum;
                        }
                        return legal_choices[sample_mixed(&legal_probs, &mut self.rng)];
                    }
                }
            }
            if !self
                .game
                .as_ref()
                .is_some_and(|g| g.belief.has_active_pick_draws(&g.observer))
            {
                if let Some(c) = self.preview_table_pick(dex) {
                    debug_assert!(choices.contains(&c), "preview pick outside legal set");
                    if choices.contains(&c) {
                        return c;
                    }
                }
            }
        }
        self.search(dex, side, choices)
    }

    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> Vec<f64> {
        let picked = self.choose(battle, dex, side, choices);
        if let Some(policy) = &self.decision_policy {
            return policy.clone();
        }
        if matches!(choices[0], SearchChoice::Team(_))
            && !self
                .game
                .as_ref()
                .is_some_and(|g| g.belief.has_active_pick_draws(&g.observer))
        {
            if let (Some(tables), Some(game)) = (&self.tables, &self.game) {
                if let Some(me) = tables.set_index(dex, game.local.own_sets()) {
                    let opp_idx = match &self.source {
                        BeliefSource::Pool { .. } => (game.belief.candidate_count() == 1)
                            .then(|| {
                                let cid = game.belief.candidate_id(game.belief.alive()[0]);
                                tables.ids.iter().position(|id| id == cid)
                            })
                            .flatten(),
                        BeliefSource::OpenSheet { .. } => game
                            .belief
                            .pinned_sets()
                            .and_then(|opp_sets| tables.set_index(dex, opp_sets)),
                    };
                    if let Some(opp) = opp_idx {
                        if let Some((table, first)) = tables.pair_by_idx(me, opp) {
                            let policy = if first {
                                &table.sol.p_a
                            } else {
                                &table.sol.p_b
                            };
                            let aligned: Vec<f64> = choices
                                .iter()
                                .map(|action| {
                                    tables
                                        .actions()
                                        .iter()
                                        .position(|t| SearchChoice::Team(*t) == *action)
                                        .map_or(0., |i| policy[i])
                                })
                                .collect();
                            if (aligned.iter().sum::<f64>() - 1.).abs() < 1e-8 {
                                return aligned;
                            }
                        }
                    }
                }
            }
        }
        choices
            .iter()
            .map(|&c| if c == picked { 1. } else { 0. })
            .collect()
    }
}

pub struct BlindAgent {
    core: ImperfectInfoCore,
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
        Self::new_with_fallback_policy(cfg, pool, tables, seed, FallbackPolicy::default())
    }

    pub fn new_with_fallback_policy(
        cfg: RmConfig,
        pool: Arc<MetaPool>,
        tables: Option<Arc<TableSet>>,
        seed: u64,
        fallback_policy: FallbackPolicy,
    ) -> Self {
        BlindAgent {
            core: ImperfectInfoCore::new(
                cfg,
                BeliefSource::Pool {
                    pool,
                    fallback_policy,
                },
                tables,
                seed,
                false,
            ),
        }
    }

    pub fn with_own_preview_policy(mut self, policy: Vec<f64>) -> Self {
        self.core.own_preview_policy = Some(policy);
        self
    }

    pub fn with_opponent_pick_policy(mut self, policy: Vec<f64>, smoothing: f64) -> Self {
        self.core.opp_pick_policy = Some((policy, smoothing));
        self
    }

    pub fn with_preview_iterations(mut self, preview_iterations: u32) -> Self {
        self.core.set_preview_iterations(preview_iterations);
        self
    }

    /// Stop clear battle decisions early, up to the smaller of the configured
    /// iterations and 243k. Intended for the legacy root decision policy.
    pub fn with_adaptive_battle_budget(mut self) -> Self {
        assert!(self.core.adaptive_mixture.is_none(), "adaptive battle budget requires legacy root selection");
        self.core.adaptive_battle_budget = true;
        self
    }

    pub fn with_adaptive_mixture(mut self, probe_iterations: u32) -> Self {
        self.core.set_adaptive_mixture(probe_iterations);
        self
    }

    /// Set the next decision's iteration budget without resetting observations
    /// or the search RNG. This only changes allocation, not the search policy.
    pub fn set_iterations(&mut self, iterations: u32) {
        self.core.set_iterations(iterations);
    }

    /// M18: consume a community belief prior for the hidden-team fallback
    /// imputation. Takes effect from the next game/preview onward, and is
    /// inert on the open-sheet path (`OpenAgent` pins its belief).
    pub fn set_belief_prior(&mut self, prior: Arc<BeliefPrior>) {
        self.core.prior = Some(prior.clone());
        if let Some(g) = self.core.game.as_mut() {
            g.belief.set_prior(prior);
        }
    }

    /// Condition opponent Team Preview & hidden-bench sampling on an empirical
    /// pick prior (`nc2000-pick-prior-v1`). Inert on `OpenAgent`.
    pub fn with_pick_prior(mut self, prior: Arc<crate::pick_prior::PickPrior>) -> Self {
        self.set_pick_prior(prior);
        self
    }

    pub fn set_pick_prior(&mut self, prior: Arc<crate::pick_prior::PickPrior>) {
        self.core.pick_prior = Some(prior);
        if let Some(g) = self.core.game.as_mut() {
            g.belief.clear_pick_draws();
        }
    }

    pub fn clear_pick_prior(&mut self) {
        self.core.pick_prior = None;
        if let Some(g) = self.core.game.as_mut() {
            g.belief.clear_pick_draws();
        }
    }

    pub fn search_rng_state(&self) -> u64 {
        self.core.rng.0
    }

    pub fn belief(&self) -> Option<&Belief> {
        self.core.game.as_ref().map(|g| &g.belief)
    }

    pub fn observer(&self) -> Option<&Observer> {
        self.core.game.as_ref().map(|g| &g.observer)
    }
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

pub(crate) fn baked_preview_pick_sets(
    tables: &TableSet,
    belief: &Belief,
    dex: &Dex,
    own_sets: &[PokemonSet],
    rng: &mut SplitMix64,
) -> Option<SearchChoice> {
    if belief.candidate_count() != 1 {
        return None;
    }
    let cand_id = belief.candidate_id(belief.alive()[0]);
    let opp = tables.ids.iter().position(|id| id == cand_id)?;
    let me = tables.set_index(dex, own_sets)?;
    let (tab, i_am_a) = tables.pair_by_idx(me, opp)?;
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

pub(crate) fn open_preview_pick_sets(
    tables: &TableSet,
    dex: &Dex,
    own_sets: &[PokemonSet],
    opp_sets: &[PokemonSet],
    rng: &mut SplitMix64,
) -> Option<SearchChoice> {
    let me = tables.set_index(dex, own_sets)?;
    let opp = tables.set_index(dex, opp_sets)?;
    let (tab, i_am_a) = tables.pair_by_idx(me, opp)?;
    let p = if i_am_a { &tab.sol.p_a } else { &tab.sol.p_b };
    Some(SearchChoice::Team(tables.actions()[sample_mixed(p, rng)]))
}

pub(crate) fn draw_policy(actions: &[SearchChoice], policy: &[f64], draw: f64) -> Option<SearchChoice> {
    let mut total = 0.;
    let mut last = None;
    for (&action, &mass) in actions.iter().zip(policy) {
        if mass > 0. {
            last = Some(action);
        }
        total += mass;
        if draw < total {
            return Some(action);
        }
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
    Action {
        own: usize,
        reply: Option<SearchChoice>,
    },
    TypeProbe,
}

const MAX_SUBSAMPLED_WORLDS: usize = 24;

struct RootTypeStats {
    samples: u32,
    late_samples: u32,
    replies: usize,
    reply_acts: Vec<SearchChoice>,
    cells: Vec<(u32, f64)>,
    late_cells: Vec<(u32, f64)>,
}

#[cfg(test)]
pub(crate) fn root_search_key(cfg: &RmConfig, dex: &Dex, battle: &mut Battle) -> u64 {
    root_search_key_cached(cfg, dex, battle, None)
}

pub(crate) fn root_search_key_cached(
    cfg: &RmConfig,
    dex: &Dex,
    battle: &mut Battle,
    cache: Option<&nc2000_engine::battle::BattleSearchCache>,
) -> u64 {
    if battle.mid_turn
        || battle.request_state != nc2000_engine::state::RequestState::Move
        || !battle.is_idle_move_request()
    {
        return key_of_cached(cfg, dex, battle, cache);
    }
    let s0 = battle.sides[0].active.is_some_and(|r| {
        battle.sides[0].roster[r as usize].speed == nc2000_engine::state::QUICK_CLAW_ACTION_SPEED
    });
    let s1 = battle.sides[1].active.is_some_and(|r| {
        battle.sides[1].roster[r as usize].speed == nc2000_engine::state::QUICK_CLAW_ACTION_SPEED
    });
    if !battle.any_quick_claw_roll() && !s0 && !s1 {
        return crate::smmcts::key_of_with_claw_omit(cfg, dex, battle, true, cache);
    }
    let coin = battle.quick_claw_rolls;
    let speeds = [0, 1].map(|side| battle.active_id(side).map(|id| (id, battle.poke(id).speed)));
    battle.quick_claw_rolls = [false, false];
    if s0 || s1 {
        battle.update_all_speeds(dex);
    }
    let key = crate::smmcts::key_of_with_claw_omit(cfg, dex, battle, true, cache);
    battle.quick_claw_rolls = coin;
    if s0 || s1 {
        for (id, speed) in speeds.into_iter().flatten() {
            battle.poke_mut(id).speed = speed;
        }
    }
    key
}

fn party_order_matters(dex: &Dex, roster: &[Pokemon]) -> bool {
    roster.iter().any(|mon| {
        mon.base_move_slots.iter().chain(&mon.move_slots).any(|slot| {
            matches!(dex.moves.key(slot.id),
                "beatup" | "metronome" | "mimic" | "mirrormove" | "sketch" | "transform")
        })
    })
}

pub(crate) fn canonicalize_unseen_bench(dex: &Dex, sim: &mut Battle, opponent: usize) {
    if sim.request_state == nc2000_engine::state::RequestState::TeamPreview
        || sim.mid_turn || !sim.queue.is_empty()
    {
        return;
    }
    let side = &sim.sides[opponent];
    let &[_, second, third] = side.party.as_slice() else { return };
    if second <= third || [second, third].iter().any(|&slot| {
        let mon = &side.roster[usize::from(slot)];
        mon.previously_switched_in != 0 || mon.is_active || mon.fainted
    }) || sim.sides.iter().any(|side| party_order_matters(dex, &side.roster)) {
        return;
    }
    let side = &mut sim.sides[opponent];
    side.party.swap(1, 2);
    side.roster[usize::from(third)].position = 1;
    side.roster[usize::from(second)].position = 2;
}

fn root_type_key(battle: &Battle) -> u64 {
    battle.type_key()
}

struct CompactWorld {
    table: nc2000_engine::compact::MatchTable,
    cb: nc2000_engine::compact::CompactBattle,
    root: usize,
}

struct CompactPreviewWorld {
    roster: nc2000_engine::compact::CompactPreviewRoster,
    sim_template: Arc<Battle>,
}

struct CachedDetRoot {
    sim: Arc<Battle>,
    root: usize,
    type_index: usize,
    cache: Arc<nc2000_engine::battle::BattleSearchCache>,
    world_id: u64,
    compact: Option<Arc<CompactWorld>>,
    preview_compact: Option<Arc<CompactPreviewWorld>>,
}

struct SampledWorld {
    sim: Battle,
    root: usize,
    type_index: usize,
    pick: Option<usize>,
    cache: Arc<nc2000_engine::battle::BattleSearchCache>,
    world_id: u64,
    compact: Option<Arc<CompactWorld>>,
}

enum NextWorld {
    Compact {
        cw: Arc<CompactWorld>,
        cb: nc2000_engine::compact::CompactBattle,
        world_idx: usize,
    },
    CompactPreview {
        cpw: Arc<CompactPreviewWorld>,
        turn1_seed: u64,
    },
    Battle {
        sim: Box<Battle>,
        cache: Arc<nc2000_engine::battle::BattleSearchCache>,
    },
}

enum DetRootOutcome {
    CompactPreview {
        cpw: Arc<CompactPreviewWorld>,
        turn1_seed: u64,
        root: usize,
        type_index: usize,
    },
    Battle {
        sim: Box<Battle>,
        root: usize,
        type_index: usize,
        cache: Arc<nc2000_engine::battle::BattleSearchCache>,
        world_id: u64,
        compact: Option<Arc<CompactWorld>>,
    },
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
    rng: SplitMix64,
    timer_rng: SplitMix64,
    decision_seed: u64,
    hidden_timers: Vec<crate::player_state::HiddenTimerSpec>,
    /// Search resets omit the observer's protocol log.
    base: Battle,
    base_idle: bool,
    base_cache: nc2000_engine::battle::BattleSearchCache,
    scratch: Option<Box<Battle>>,
    scratch_world_id: u64,
    next_world_id: u64,
    det_cache: FxHashMap<(TeamPlanPick, u64, [u8; 6], [bool; 2]), CachedDetRoot>,
    det_cache_obs_rev: u64,
    world_pool: Vec<SampledWorld>,
    planned_team_picks: Vec<TeamPlanPick>,
    planned_obs_rev: Option<u64>,
    max_worlds_override: Option<usize>,
    bayesian_row_q: Option<Vec<f64>>,
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
    my_dominated_reasons: Vec<Option<&'static str>>,
    /// When enabled (always true on `from_observation`), UCB root exploration
    /// skips `my_dominated` actions whenever an undominated allowed action exists,
    /// without overwriting explicit `mask_actions` / alias masks.
    prune_dominated: bool,
    opponent_preview_prior: bool,
    /// Per-determinization roots + full-state interior nodes.
    nodes: Vec<Node>,
    table: FxHashMap<u64, usize>,
    /// Native 256-byte `CompactBattle` tree nodes and transposition table (in-battle fast path).
    compact_nodes: Vec<crate::smmcts::CompactNode>,
    compact_table: FxHashMap<u64, usize>,
    /// Maps each battle root index in `nodes` that also backs a compact root in `compact_nodes`
    /// so `node_count()` never double-counts shared roots.
    compact_root_battle_nodes: FxHashMap<usize, usize>,
    done: u32,
    depth_sum: u64,
    max_depth: u32,
    /// Diagnostic columns use SearchChoice because legal reply lists vary by type.
    joint: Vec<(usize, SearchChoice, u32, f64)>,
    joint_index: FxHashMap<(usize, SearchChoice), usize>,
    root_avail: FxHashMap<usize, Vec<usize>>,
    root_types: Vec<RootTypeStats>,
    root_type_index: FxHashMap<u64, usize>,
    /// How often each opponent action was even LEGAL, across iterations.
    /// A blind root's opponent action list is determinization-dependent — a
    /// move exists only in the candidates that carry it — so a column's
    /// sample count conflates "rarely chosen" with "rarely available", and
    /// only this tells them apart.
    avail: Vec<(SearchChoice, u32)>,
    endgame: crate::smmcts::EndgameRootStats,
}

impl BlindSearch {
    pub fn set_opponent_preview_prior(&mut self, enabled: bool) {
        self.opponent_preview_prior = enabled;
    }

    pub fn with_max_worlds(mut self, k: usize) -> Self {
        self.max_worlds_override = Some(k.max(1));
        self
    }

    #[inline]
    fn target_worlds(&self) -> usize {
        self.max_worlds_override
            .unwrap_or(MAX_SUBSAMPLED_WORLDS)
            .max(1)
    }

    /// Product search: only a player-protocol reconstruction can enter here.
    pub fn from_observation(
        position: &ObservedPosition,
        dex: &Dex,
        cfg: RmConfig,
        side: usize,
        seed: u64,
    ) -> BlindSearch {
        Self::from_observation_with_rng(position, dex, cfg, side, SplitMix64::new(seed))
    }

    /// Product search with caller-threaded PRNG: populates `my_dominated` only
    /// from strictly public / own-certain no-op rules (`public_noop_reason`),
    /// enables root dominance pruning before UCB allocation, and installs the
    /// reconstructed hidden condition timers.
    pub(crate) fn from_observation_with_rng(
        position: &ObservedPosition,
        dex: &Dex,
        cfg: RmConfig,
        side: usize,
        rng: SplitMix64,
    ) -> BlindSearch {
        let mut search = Self::with_rng_inner(position.model(), dex, cfg, side, rng, false);
        search.use_public_exclusions(dex);
        search.prune_dominated();
        search.set_hidden_timers(position.hidden_timers().to_vec());
        search
    }

    pub(crate) fn set_hidden_timers(&mut self, timers: Vec<crate::player_state::HiddenTimerSpec>) {
        self.hidden_timers = timers;
    }

    fn set_dominated_reasons(&mut self, reasons: Vec<Option<&'static str>>) {
        self.my_dominated = reasons.iter().map(Option::is_some).collect();
        self.my_dominated_reasons = reasons;
    }

    #[inline]
    fn is_allowed(&self, i: usize) -> bool {
        self.my_mask.as_ref().map_or(true, |mask| mask[i])
    }

    #[inline]
    fn has_undominated_allowed(&self) -> bool {
        (0..self.my_acts.len()).any(|i| self.is_allowed(i) && !self.my_dominated[i])
    }

    #[inline]
    fn is_eligible(&self, i: usize, has_safe: bool) -> bool {
        self.is_allowed(i) && (!has_safe || !self.my_dominated[i])
    }

    #[inline]
    fn is_ucb_allowed(&self, i: usize, has_safe: bool) -> bool {
        self.is_allowed(i) && (!self.prune_dominated || !has_safe || !self.my_dominated[i])
    }

    pub(crate) fn eligible_flags(&self) -> Vec<bool> {
        let has_safe = self.has_undominated_allowed();
        (0..self.my_acts.len())
            .map(|i| self.is_eligible(i, has_safe))
            .collect()
    }

    fn effective_visits(&self, is_elig: impl Fn(usize) -> bool) -> (bool, Vec<u32>) {
        let ordinary =
            (0..self.my_acts.len()).any(|i| is_elig(i) && self.my_n[i] > self.joint_probe_n[i]);
        let eff_n = (0..self.my_acts.len())
            .map(|i| {
                if ordinary {
                    self.my_n[i] - self.joint_probe_n[i]
                } else {
                    self.my_n[i]
                }
            })
            .collect();
        (ordinary, eff_n)
    }

    fn use_public_exclusions(&mut self, dex: &Dex) {
        // Guessed hidden opponent attributes (Speed, Quick Claw, bench sets,
        // HP rolls) cannot prove action dominance. Exclude only pure-public /
        // own-certain no-ops that agree across every legal world.
        let reasons = self
            .my_acts
            .iter()
            .map(|&c| crate::smmcts::public_noop_reason(&self.base, dex, self.side, c))
            .collect();
        self.set_dominated_reasons(reasons);
    }

    /// Full-world research API. Product callers use `from_observation`.
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
        Self::with_rng_inner(battle, dex, cfg, side, rng, true)
    }

    fn with_rng_inner(
        battle: &Battle,
        dex: &Dex,
        mut cfg: RmConfig,
        side: usize,
        rng: SplitMix64,
        compute_dominated: bool,
    ) -> BlindSearch {
        if matches!(
            battle.request_state,
            nc2000_engine::state::RequestState::TeamPreview
        ) && matches!(
            cfg.settlement,
            crate::accounting::LeafSettlementMode::BlindSpotLeaf
                | crate::accounting::LeafSettlementMode::BlindSpotLeafNoRelay
        ) {
            cfg.settlement = crate::accounting::LeafSettlementMode::LegacyRollout;
        }
        let mut base = battle.clone();
        base.set_log_enabled(false);
        let base_idle = base.is_idle_move_request();
        if base_idle {
            base.quick_claw_rolls = [false, false];
            base.update_all_speeds(dex);
        }
        let mut turn_cap = base.turn.saturating_add(cfg.horizon);
        if let Some(limit) = base.ruleset.turn_limit() {
            turn_cap = turn_cap.min(limit);
        }
        let my_acts = base.legal_choices(dex, side);
        let my_dominated_reasons: Vec<Option<&'static str>> = if compute_dominated {
            my_acts
                .iter()
                .map(|&c| crate::smmcts::dominated_reason(&base, dex, side, c))
                .collect()
        } else {
            vec![None; my_acts.len()]
        };
        let my_dominated = my_dominated_reasons.iter().map(Option::is_some).collect();
        let endgame = crate::smmcts::EndgameRootStats::new(&base, dex, side, &my_acts);
        let base_cache = crate::smmcts::build_search_cache(&cfg, dex, &base);
        let decision_seed = rng.0 ^ 0xa24baed4963ee407;
        let timer_rng = SplitMix64(decision_seed ^ 0x714e_525f_726e_6701);
        BlindSearch {
            cfg,
            rng,
            timer_rng,
            decision_seed,
            hidden_timers: Vec::new(),
            base,
            base_idle,
            base_cache,
            scratch: None,
            scratch_world_id: 0,
            next_world_id: 1,
            det_cache: FxHashMap::default(),
            det_cache_obs_rev: u64::MAX,
            world_pool: Vec::new(),
            planned_team_picks: Vec::new(),
            planned_obs_rev: None,
            max_worlds_override: None,
            bayesian_row_q: None,
            turn_cap,
            side,
            my_n: vec![0; my_acts.len()],
            joint_probe_n: vec![0; my_acts.len()],
            my_w: vec![0.0; my_acts.len()],
            my_mask: None,
            my_dominated,
            my_dominated_reasons,
            prune_dominated: false,
            opponent_preview_prior: false,
            my_acts,
            nodes: Vec::new(),
            table: FxHashMap::default(),
            compact_nodes: Vec::new(),
            compact_table: FxHashMap::default(),
            compact_root_battle_nodes: FxHashMap::default(),
            done: 0,
            depth_sum: 0,
            max_depth: 0,
            joint: Vec::new(),
            joint_index: FxHashMap::default(),
            root_avail: FxHashMap::default(),
            root_types: Vec::new(),
            root_type_index: FxHashMap::default(),
            avail: Vec::new(),
            endgame,
        }
    }

    /// One iteration: fresh determinization, global-UCB own root pick
    /// forced into the shared `run_iteration`. Returns the side-0 reward.
    pub fn step_one(&mut self, dex: &Dex, belief: &Belief, obs: &Observer) -> f64 {
        self.step_one_impl(dex, belief, obs, None, None)
    }

    fn resolve_root(
        &mut self,
        dex: &Dex,
        sim: &mut Battle,
        cache: Option<&nc2000_engine::battle::BattleSearchCache>,
    ) -> (usize, usize) {
        let key = root_search_key_cached(&self.cfg, dex, sim, cache);
        let root = match self.table.get(&key) {
            Some(&i) => i,
            None => {
                let node = Node::at(sim, dex);
                debug_assert_eq!(
                    node.acts[self.side].as_slice(),
                    self.my_acts.as_slice(),
                    "determinization changed the observer's own root actions"
                );
                self.nodes.push(node);
                let idx = self.nodes.len() - 1;
                self.table.insert(key, idx);
                idx
            }
        };
        let type_key = match cache {
            Some(c) => sim.type_key_with_cache(c),
            None => root_type_key(sim),
        };
        let opp = 1 - self.side;
        let reply_acts = self.nodes[root].acts[opp].to_vec();
        let replies = reply_acts.len();
        let my_len = self.my_acts.len();
        let ti = *self.root_type_index.entry(type_key).or_insert_with(|| {
            self.root_types.push(RootTypeStats {
                samples: 0,
                late_samples: 0,
                replies,
                reply_acts,
                cells: vec![(0, 0.); my_len * replies],
                late_cells: vec![(0, 0.); my_len * replies],
            });
            self.root_types.len() - 1
        });
        (root, ti)
    }

    /// Predict the RNG draws and resulting opponent party of `Belief::determinize_in_place`
    /// by delegating directly to `Belief::sample_hidden_party`.
    #[inline]
    fn predict_determinization_draws(
        &self,
        belief: &Belief,
        obs: &Observer,
        plan_pick: TeamPlanPick,
        rng_probe: &mut SplitMix64,
    ) -> (u64, [bool; 2], [u8; 6], usize) {
        let seed = rng_probe.next();
        let mut prng = nc2000_engine::prng::Prng::new(seed);
        let qc = Battle::roll_quick_claw_with(self.base.ruleset, &mut prng);
        let post_qc_seed = prng.seed();

        let opp = 1 - self.side;
        let opp_side = &self.base.sides[opp];
        let (raw_party, party_len, _) = belief.sample_hidden_party(
            obs,
            plan_pick.pool_index(),
            opp_side,
            self.base.preview_level_caps[opp],
            rng_probe,
        );
        let mut party_key = [0xFFu8; 6];
        party_key[..party_len].copy_from_slice(&raw_party[..party_len]);
        (post_qc_seed, qc, party_key, party_len)
    }

    #[inline]
    fn has_bide_hidden_timer(&self) -> bool {
        self.hidden_timers
            .iter()
            .any(|s| matches!(s.kind, crate::player_state::HiddenTimerKind::Bide { .. }))
    }

    fn try_build_compact_world(
        &mut self,
        dex: &Dex,
        sim: &Battle,
        battle_root: usize,
        cache: &nc2000_engine::battle::BattleSearchCache,
    ) -> Option<Arc<CompactWorld>> {
        if self.is_preview()
            || self.has_bide_hidden_timer()
            || !nc2000_engine::compact::is_compact_tree_supported(sim, dex)
        {
            return None;
        }
        let (table, mut cb) =
            nc2000_engine::compact::CompactBattle::from_battle_with_roster_cache(
                sim,
                dex,
                Some(&cache.roster_compact),
            );
        if self.base_idle {
            cb.quick_claw_roll = 0;
        }
        let c_node = crate::smmcts::CompactNode::at(&cb, &table);
        if !c_node.matches_battle_acts(
            &cb,
            &table,
            dex,
            &self.nodes[battle_root].acts,
            [&sim.sides[0].party, &sim.sides[1].party],
        ) {
            return None;
        }
        let omit_qc = self.base_idle || !cb.quick_claw_observable(&table);
        let c_key = cb.search_key(&table, self.cfg.hp_buckets, omit_qc);
        let c_root = match self.compact_table.get(&c_key) {
            Some(&i) => i,
            None => {
                let idx = self.compact_nodes.len();
                self.compact_nodes.push(c_node);
                self.compact_table.insert(c_key, idx);
                idx
            }
        };
        self.compact_root_battle_nodes
            .entry(battle_root)
            .or_insert(c_root);
        Some(Arc::new(CompactWorld {
            table,
            cb,
            root: c_root,
        }))
    }

    fn try_build_compact_preview_world(
        &self,
        dex: &Dex,
        sim: &Battle,
        battle_root: usize,
        cache: &nc2000_engine::battle::BattleSearchCache,
        sim_template: Arc<Battle>,
    ) -> Option<Arc<CompactPreviewWorld>> {
        if !self.is_preview()
            || !self.hidden_timers.is_empty()
            || !nc2000_engine::compact::is_compact_preview_supported(sim, dex)
        {
            return None;
        }
        let root_acts = &self.nodes[battle_root].acts;
        if root_acts[0].is_empty()
            || root_acts[1].is_empty()
            || !root_acts[0]
                .iter()
                .all(|c| matches!(c, SearchChoice::Team(_)))
            || !root_acts[1]
                .iter()
                .all(|c| matches!(c, SearchChoice::Team(_)))
        {
            return None;
        }
        let roster = nc2000_engine::compact::CompactPreviewRoster::from_battle(
            sim,
            dex,
            Some(&cache.roster_compact),
        );
        Some(Arc::new(CompactPreviewWorld {
            roster,
            sim_template,
        }))
    }

    fn determinize_and_resolve_root(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        plan_pick: TeamPlanPick,
    ) -> DetRootOutcome {
        let obs_rev = obs.revision();
        if self.det_cache_obs_rev != obs_rev {
            self.det_cache.clear();
            self.world_pool.clear();
            self.det_cache_obs_rev = obs_rev;
        }
        let unseen_idle = self.base_idle;
        let zero_preview_qc = self.is_preview();
        let share_qc_cache = unseen_idle || zero_preview_qc;
        let preview_queue_ok = zero_preview_qc
            && self.base.queue.len() == 1
            && matches!(
                self.base.queue[0].choice,
                nc2000_engine::state::ActionKind::Start
            );
        if (self.base.queue.is_empty() && !belief.has_active_pick_draws(obs)) || preview_queue_ok {
            let mut rng_probe = self.rng.clone();
            if let Some(fb_key) =
                belief.predict_plan_draws(plan_pick, &self.base, 1 - self.side, &mut rng_probe)
            {
                let (post_qc_seed, qc, party_key, party_len) =
                    self.predict_determinization_draws(belief, obs, plan_pick, &mut rng_probe);
                let cache_key = (
                    plan_pick,
                    fb_key,
                    party_key,
                    if share_qc_cache { [false, false] } else { qc },
                );
                if let Some(cached) = self.det_cache.get(&cache_key) {
                    self.rng = rng_probe;
                    let root = cached.root;
                    let type_index = cached.type_index;
                    if let Some(cpw) = cached.preview_compact.clone() {
                        return DetRootOutcome::CompactPreview {
                            cpw,
                            turn1_seed: post_qc_seed,
                            root,
                            type_index,
                        };
                    }
                    let cache = cached.cache.clone();
                    let world_id = cached.world_id;
                    let compact = cached.compact.clone();
                    let mut same_sets = [false, false];
                    same_sets[self.side] = self.scratch_world_id != 0;
                    same_sets[1 - self.side] = self.scratch_world_id == world_id;
                    let mut sim = match self.scratch.take() {
                        Some(mut sim) => {
                            sim.reset_search_sim(&cached.sim, same_sets);
                            sim
                        }
                        None => Box::new((*cached.sim).clone()),
                    };
                    self.scratch_world_id = world_id;
                    sim.reseed(post_qc_seed);
                    sim.quick_claw_rolls = if zero_preview_qc { [false, false] } else { qc };
                    std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                    return DetRootOutcome::Battle {
                        sim,
                        root,
                        type_index,
                        cache,
                        world_id,
                        compact,
                    };
                }

                let mut sim = match self.scratch.take() {
                    Some(mut sim) => {
                        sim.as_mut().clone_from(&self.base);
                        sim
                    }
                    None => Box::new(self.base.clone()),
                };
                belief.determinize_in_place_planned(dex, &mut sim, obs, plan_pick, &mut self.rng);
                let sampled_qc = sim.quick_claw_rolls;
                let opp = 1 - self.side;
                if zero_preview_qc {
                    sim.quick_claw_rolls = [false, false];
                } else if unseen_idle && (sampled_qc[0] || sampled_qc[1]) {
                    sim.quick_claw_rolls = [false, false];
                    if let Some(a) = sim.active_id(opp) {
                        if !sim.poke(a).fainted {
                            sim.update_speed(dex, a);
                        }
                    }
                }
                let can_cache = self.rng.0 == rng_probe.0
                    && sim.sides[opp].party.as_slice() == &party_key[..party_len];
                // Cache by the raw draw. An imputed move can make two bench orders distinct.
                canonicalize_unseen_bench(dex, &mut sim, opp);
                let cache = Arc::new(crate::smmcts::build_search_cache_with_template(
                    &self.cfg,
                    dex,
                    &sim,
                    self.side,
                    &self.base_cache,
                ));
                let world_id = self.next_world_id;
                self.next_world_id += 1;
                self.scratch_world_id = world_id;
                std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                let (root, type_index) = self.resolve_root(dex, &mut sim, Some(&cache));
                let compact = self.try_build_compact_world(dex, &sim, root, &cache);
                let mut cached_sim_arc: Option<Arc<Battle>> = None;
                let preview_compact = if zero_preview_qc
                    && self.hidden_timers.is_empty()
                    && nc2000_engine::compact::is_compact_preview_supported(&sim, dex)
                {
                    std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                    let arc = Arc::new((*sim).clone());
                    std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                    cached_sim_arc = Some(arc.clone());
                    self.try_build_compact_preview_world(dex, &sim, root, &cache, arc)
                } else {
                    None
                };
                if can_cache {
                    let sim_arc = match cached_sim_arc {
                        Some(arc) => arc,
                        None => {
                            std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                            let arc = Arc::new((*sim).clone());
                            std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                            arc
                        }
                    };
                    self.det_cache.insert(
                        cache_key,
                        CachedDetRoot {
                            sim: sim_arc,
                            root,
                            type_index,
                            cache: cache.clone(),
                            world_id,
                            compact: compact.clone(),
                            preview_compact: preview_compact.clone(),
                        },
                    );
                }
                if let Some(cpw) = preview_compact {
                    let turn1_seed = sim.prng.lcg.seed();
                    std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                    self.scratch = Some(sim);
                    return DetRootOutcome::CompactPreview {
                        cpw,
                        turn1_seed,
                        root,
                        type_index,
                    };
                }
                if unseen_idle {
                    sim.quick_claw_rolls = sampled_qc;
                }
                return DetRootOutcome::Battle {
                    sim,
                    root,
                    type_index,
                    cache,
                    world_id,
                    compact,
                };
            }
        }

        let mut sim = match self.scratch.take() {
            Some(mut sim) => {
                sim.as_mut().clone_from(&self.base);
                sim
            }
            None => Box::new(self.base.clone()),
        };
        belief.determinize_in_place_planned(dex, &mut sim, obs, plan_pick, &mut self.rng);
        canonicalize_unseen_bench(dex, &mut sim, 1 - self.side);
        let sampled_qc = sim.quick_claw_rolls;
        let opp = 1 - self.side;
        if zero_preview_qc {
            sim.quick_claw_rolls = [false, false];
        } else if unseen_idle && (sampled_qc[0] || sampled_qc[1]) {
            sim.quick_claw_rolls = [false, false];
            if let Some(a) = sim.active_id(opp) {
                if !sim.poke(a).fainted {
                    sim.update_speed(dex, a);
                }
            }
        }
        let cache = Arc::new(crate::smmcts::build_search_cache_with_template(
            &self.cfg,
            dex,
            &sim,
            self.side,
            &self.base_cache,
        ));
        let world_id = self.next_world_id;
        self.next_world_id += 1;
        self.scratch_world_id = world_id;
        std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
        let (root, type_index) = self.resolve_root(dex, &mut sim, Some(&cache));
        if zero_preview_qc
            && self.hidden_timers.is_empty()
            && nc2000_engine::compact::is_compact_preview_supported(&sim, dex)
        {
            std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
            let sim_arc = Arc::new((*sim).clone());
            if let Some(cpw) =
                self.try_build_compact_preview_world(dex, &sim, root, &cache, sim_arc)
            {
                let turn1_seed = sim.prng.lcg.seed();
                self.scratch = Some(sim);
                return DetRootOutcome::CompactPreview {
                    cpw,
                    turn1_seed,
                    root,
                    type_index,
                };
            }
            std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
        }
        let compact = self.try_build_compact_world(dex, &sim, root, &cache);
        if unseen_idle {
            sim.quick_claw_rolls = sampled_qc;
        }
        DetRootOutcome::Battle {
            sim,
            root,
            type_index,
            cache,
            world_id,
            compact,
        }
    }

    fn next_determinized_world(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
    ) -> (NextWorld, usize, usize, Option<usize>) {
        let obs_rev = obs.revision();
        if self.det_cache_obs_rev != obs_rev {
            self.det_cache.clear();
            self.world_pool.clear();
            self.det_cache_obs_rev = obs_rev;
        }
        let target_k = self.target_worlds();
        if self.is_preview() || self.world_pool.len() < target_k {
            let plan_pick = if self.is_preview() {
                belief
                    .sample(&mut self.rng)
                    .map(TeamPlanPick::Pool)
                    .unwrap_or(TeamPlanPick::FallbackDefault)
            } else {
                if self.planned_obs_rev != Some(obs_rev) || self.planned_team_picks.len() != target_k
                {
                    let mut plan_rng = SplitMix64::new(
                        self.decision_seed
                            ^ obs_rev.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            ^ (target_k as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9),
                    );
                    self.planned_team_picks = belief.root_sampling_plan(target_k, &mut plan_rng);
                    self.planned_obs_rev = Some(obs_rev);
                }
                self.planned_team_picks[self.world_pool.len()]
            };
            let pick = plan_pick.pool_index();
            let (mut sim, root, type_index, cache, world_id, compact) =
                match self.determinize_and_resolve_root(dex, belief, obs, plan_pick) {
                    DetRootOutcome::CompactPreview {
                        cpw,
                        turn1_seed,
                        root,
                        type_index,
                    } => {
                        return (
                            NextWorld::CompactPreview { cpw, turn1_seed },
                            root,
                            type_index,
                            pick,
                        );
                    }
                    DetRootOutcome::Battle {
                        sim,
                        root,
                        type_index,
                        cache,
                        world_id,
                        compact,
                    } => (sim, root, type_index, cache, world_id, compact),
                };
            if !self.is_preview() {
                std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                let cached_sim = (*sim).clone();
                let w_idx = self.world_pool.len();
                self.world_pool.push(SampledWorld {
                    sim: cached_sim,
                    root,
                    type_index,
                    pick,
                    cache: cache.clone(),
                    world_id,
                    compact: compact.clone(),
                });
                if let Some(cw) = compact {
                    if !self.has_bide_hidden_timer() {
                        let mut cb = cw.cb;
                        cb.prng_seed = sim.prng.lcg.seed();
                        cb.quick_claw_roll = (sim.quick_claw_rolls[0] as u8)
                            | ((sim.quick_claw_rolls[1] as u8) << 1);
                        if !self.hidden_timers.is_empty() {
                            resample_compact_hidden_timers(
                                &mut cb,
                                &cw.table,
                                &self.hidden_timers,
                                &mut self.timer_rng,
                            );
                        }
                        self.scratch = Some(sim);
                        return (
                            NextWorld::Compact {
                                cw,
                                cb,
                                world_idx: w_idx,
                            },
                            root,
                            type_index,
                            pick,
                        );
                    }
                }
                std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                if !self.hidden_timers.is_empty() {
                    resample_hidden_timers(
                        dex,
                        &mut sim,
                        &self.hidden_timers,
                        &mut self.timer_rng,
                    );
                }
            }
            return (NextWorld::Battle { sim, cache }, root, type_index, pick);
        }
        let w_idx = (self.done as usize) % self.world_pool.len();
        if !self.has_bide_hidden_timer() {
            if let Some(cw) = self.world_pool[w_idx].compact.clone() {
                let w = &self.world_pool[w_idx];
                let (root, type_index, pick) = (w.root, w.type_index, w.pick);
                let mut cb = cw.cb;
                let seed = self.rng.next();
                let mut prng = nc2000_engine::prng::Prng::new(seed);
                let qc = Battle::roll_quick_claw_with(cw.table.ruleset, &mut prng);
                cb.prng_seed = prng.seed();
                cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
                if !self.hidden_timers.is_empty() {
                    resample_compact_hidden_timers(
                        &mut cb,
                        &cw.table,
                        &self.hidden_timers,
                        &mut self.timer_rng,
                    );
                }
                return (
                    NextWorld::Compact {
                        cw,
                        cb,
                        world_idx: w_idx,
                    },
                    root,
                    type_index,
                    pick,
                );
            }
        }
        let (root, type_index, pick, cache, world_id) = {
            let w = &self.world_pool[w_idx];
            (w.root, w.type_index, w.pick, w.cache.clone(), w.world_id)
        };
        let mut same_sets = [false, false];
        same_sets[self.side] = self.scratch_world_id != 0;
        same_sets[1 - self.side] = self.scratch_world_id == world_id;
        let mut sim = match self.scratch.take() {
            Some(mut sim) => {
                sim.reset_search_sim(&self.world_pool[w_idx].sim, same_sets);
                sim
            }
            None => Box::new(self.world_pool[w_idx].sim.clone()),
        };
        self.scratch_world_id = world_id;
        let seed = self.rng.next();
        let mut prng = nc2000_engine::prng::Prng::new(seed);
        let qc = Battle::roll_quick_claw_with(sim.ruleset, &mut prng);
        sim.reseed(prng.seed());
        sim.quick_claw_rolls = qc;
        std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
        if !self.hidden_timers.is_empty() {
            resample_hidden_timers(dex, &mut sim, &self.hidden_timers, &mut self.timer_rng);
        }
        (NextWorld::Battle { sim, cache }, root, type_index, pick)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_world_iteration(
        &mut self,
        dex: &Dex,
        world: NextWorld,
        root: usize,
        force: [Option<usize>; 2],
        joint: &mut [usize; 2],
        leaf: Option<&mut dyn FnMut(&mut Battle, &mut SplitMix64, bool) -> f64>,
        trace: Option<&mut dyn FnMut(crate::smmcts::SearchTrace<'_>)>,
        root_child_value: Option<&mut dyn FnMut(&Battle) -> Option<f64>>,
    ) -> f64 {
        let mut iter_depth = 0u32;
        let r = match world {
            NextWorld::Compact {
                cw,
                mut cb,
                world_idx,
            } => {
                if leaf.is_none() && trace.is_none() && root_child_value.is_none() {
                    crate::smmcts::run_compact_iteration(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.compact_nodes,
                        &mut self.compact_table,
                        &mut cb,
                        &cw.table,
                        self.turn_cap,
                        cw.root,
                        force,
                        joint,
                        &mut iter_depth,
                    )
                } else {
                    let w_id = self.world_pool[world_idx].world_id;
                    let mut same_sets = [false, false];
                    same_sets[self.side] = self.scratch_world_id != 0;
                    same_sets[1 - self.side] = self.scratch_world_id == w_id;
                    let mut battle_buf = match self.scratch.take() {
                        Some(mut sim) => {
                            sim.reset_search_sim(&self.world_pool[world_idx].sim, same_sets);
                            sim
                        }
                        None => Box::new(self.world_pool[world_idx].sim.clone()),
                    };
                    self.scratch_world_id = w_id;
                    std::mem::swap(&mut battle_buf.listener_pool, &mut self.base.listener_pool);
                    let v = crate::smmcts::run_compact_iteration_with_hooks(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.compact_nodes,
                        &mut self.compact_table,
                        &mut cb,
                        &cw.table,
                        dex,
                        &mut battle_buf,
                        self.turn_cap,
                        cw.root,
                        force,
                        joint,
                        &mut iter_depth,
                        leaf,
                        trace,
                        root_child_value,
                    );
                    std::mem::swap(&mut battle_buf.listener_pool, &mut self.base.listener_pool);
                    // `sync_to_battle` mutates picked bench mons without advancing `previously_switched_in`;
                    // invalidate `scratch_world_id` so subsequent `reset_search_sim` resets all mons cleanly.
                    self.scratch_world_id = 0;
                    self.scratch = Some(battle_buf);
                    v
                }
            }
            NextWorld::CompactPreview { cpw, turn1_seed } => {
                if leaf.is_none() && trace.is_none() && root_child_value.is_none() {
                    crate::smmcts::run_compact_preview_iteration(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.nodes[root],
                        &mut self.compact_nodes,
                        &mut self.compact_table,
                        &cpw.roster,
                        turn1_seed,
                        self.turn_cap,
                        force,
                        joint,
                        &mut iter_depth,
                    )
                } else {
                    let mut battle_buf = match self.scratch.take() {
                        Some(mut sim) => {
                            sim.reset_search_sim(&cpw.sim_template, [false, false]);
                            sim
                        }
                        None => Box::new((*cpw.sim_template).clone()),
                    };
                    battle_buf.reseed(turn1_seed);
                    battle_buf.quick_claw_rolls = [false, false];
                    std::mem::swap(&mut battle_buf.listener_pool, &mut self.base.listener_pool);
                    let v = crate::smmcts::run_compact_preview_iteration_with_hooks(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.nodes[root],
                        root,
                        &mut self.compact_nodes,
                        &mut self.compact_table,
                        &cpw.roster,
                        turn1_seed,
                        dex,
                        &mut battle_buf,
                        self.turn_cap,
                        force,
                        joint,
                        &mut iter_depth,
                        leaf,
                        trace,
                        root_child_value,
                    );
                    std::mem::swap(&mut battle_buf.listener_pool, &mut self.base.listener_pool);
                    self.scratch_world_id = 0;
                    self.scratch = Some(battle_buf);
                    v
                }
            }
            NextWorld::Battle { mut sim, cache } => {
                if trace.is_some() && self.base_idle && sim.any_quick_claw_roll() {
                    sim.update_all_speeds(dex);
                }
                let v = if let Some(evaluate) = root_child_value {
                    crate::smmcts::run_iteration_with_root_child_value_cached(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.nodes,
                        &mut self.table,
                        &mut sim,
                        dex,
                        self.turn_cap,
                        root,
                        force,
                        joint,
                        &mut iter_depth,
                        &mut |sim, rng, rollout| {
                            crate::smmcts::default_leaf_cached(
                                &self.cfg,
                                sim,
                                dex,
                                self.turn_cap,
                                rng,
                                rollout,
                                Some(&cache),
                            )
                        },
                        trace,
                        Some(evaluate),
                        Some(&cache),
                    )
                } else if let Some(leaf) = leaf {
                    crate::smmcts::run_iteration_with_leaf_cached(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.nodes,
                        &mut self.table,
                        &mut sim,
                        dex,
                        self.turn_cap,
                        root,
                        force,
                        joint,
                        &mut iter_depth,
                        leaf,
                        trace,
                        Some(&cache),
                    )
                } else if let Some(trace) = trace {
                    crate::smmcts::run_iteration_with_leaf_cached(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.nodes,
                        &mut self.table,
                        &mut sim,
                        dex,
                        self.turn_cap,
                        root,
                        force,
                        joint,
                        &mut iter_depth,
                        &mut |sim, rng, rollout| {
                            crate::smmcts::default_leaf_cached(
                                &self.cfg,
                                sim,
                                dex,
                                self.turn_cap,
                                rng,
                                rollout,
                                Some(&cache),
                            )
                        },
                        Some(trace),
                        Some(&cache),
                    )
                } else {
                    run_iteration_cached(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.nodes,
                        &mut self.table,
                        &mut sim,
                        dex,
                        self.turn_cap,
                        root,
                        force,
                        joint,
                        &mut iter_depth,
                        Some(&cache),
                    )
                };
                std::mem::swap(&mut sim.listener_pool, &mut self.base.listener_pool);
                self.scratch = Some(sim);
                v
            }
        };
        self.depth_sum += u64::from(iter_depth);
        self.max_depth = self.max_depth.max(iter_depth);
        r
    }

    fn step_one_impl(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        leaf: Option<&mut dyn FnMut(&mut Battle, &mut SplitMix64, bool) -> f64>,
        trace: Option<&mut dyn FnMut(crate::smmcts::SearchTrace<'_>)>,
    ) -> f64 {
        if !self.is_preview() && self.done >= 512 && self.done % 256 == 0 {
            if let Some(eq) = self.bayesian_equilibrium() {
                self.bayesian_row_q = Some(eq.action_values);
            }
        }
        let (world, root, type_index, pick) = self.next_determinized_world(dex, belief, obs);
        // Forced switches have `replies == 0` and an empty `cells` matrix, so
        // world-floor coverage does not apply and UCB1 must drive allocation.
        let min_floor_pick = if !self.is_preview()
            && self.done >= 512
            && self.root_types[type_index].replies > 0
        {
            let has_safe = self.has_undominated_allowed();
            let elig: nc2000_engine::smallvec::SmallVec<[usize; 8]> = (0..self.my_acts.len())
                .filter(|&i| self.is_eligible(i, has_safe))
                .collect();
            let n_elig = (elig.len().max(1)) as u32;
            let stats = &self.root_types[type_index];
            let replies = stats.replies;
            let world_floor = (stats.samples / (3 * n_elig)).min(48);
            elig.into_iter()
                .filter_map(|i| {
                    let row_start = i * replies;
                    let w_n: u32 = stats
                        .cells
                        .get(row_start..row_start + replies)
                        .map_or(0, |row| row.iter().map(|&(n, _)| n).sum());
                    (w_n < world_floor).then_some((w_n, self.my_n[i], i))
                })
                .min_by_key(|&(w_n, g_n, _)| (w_n, g_n))
                .map(|(_, _, i)| i)
        } else {
            None
        };
        let my_pick = if let Some(action) = min_floor_pick {
            self.my_n[action] += 1;
            action
        } else {
            let has_safe = self.has_undominated_allowed();
            let my_mask = &self.my_mask;
            let my_dominated = &self.my_dominated;
            let prune_dominated = self.prune_dominated;
            select_global_with_prior(
                &self.cfg,
                &mut self.rng,
                &mut self.my_n,
                &self.my_w,
                |a| {
                    my_mask.as_ref().map_or(true, |m| m[a])
                        && (!prune_dominated || !has_safe || !my_dominated[a])
                },
                self.bayesian_row_q.as_deref(),
            )
        };
        let mut force = [None, None];
        force[self.side] = Some(my_pick);
        if self.opponent_preview_prior && self.is_preview() {
            force[1 - self.side] = belief.sample_preview_prior(
                pick,
                obs,
                &self.nodes[root].acts[1 - self.side],
                &mut self.rng,
            );
        }
        let mut joint = [0usize; 2];
        let r = self.run_world_iteration(dex, world, root, force, &mut joint, leaf, trace, None);
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
        self.step_forced_impl(
            dex,
            belief,
            obs,
            ForcedRoot::Action {
                own: my_pick,
                reply: None,
            },
        )
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
        let value = self.step_forced_impl(
            dex,
            belief,
            obs,
            ForcedRoot::Action {
                own: my_pick,
                reply: Some(reply),
            },
        );
        self.joint_probe_n[my_pick] += 1;
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
        let value = self.step_forced_impl_observed(
            dex,
            belief,
            obs,
            ForcedRoot::Action {
                own: my_pick,
                reply: Some(reply),
            },
            Some(trace),
        );
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
        self.step_forced_impl_with_root_child_value(dex, belief, obs, forcing, trace, None)
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
        let value = self.step_forced_impl_with_root_child_value(
            dex,
            belief,
            obs,
            ForcedRoot::Action {
                own: my_pick,
                reply: Some(reply),
            },
            None,
            Some(evaluate),
        );
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
            assert!(
                self.is_ucb_allowed(own, self.has_undominated_allowed()),
                "forced root action is masked"
            );
        }
        if !self.is_preview() && self.done >= 4096 && self.done % 512 == 0 {
            if let Some(eq) = self.bayesian_equilibrium() {
                self.bayesian_row_q = Some(eq.action_values);
            }
        }
        let (world, root, type_index, pick) = self.next_determinized_world(dex, belief, obs);
        let (my_pick, forced_reply, override_reply) = match forcing {
            ForcedRoot::Action { own, reply } => {
                self.my_n[own] += 1;
                (
                    own,
                    reply.and_then(|a| {
                        self.nodes[root].acts[1 - self.side]
                            .iter()
                            .position(|&c| c == a)
                    }),
                    reply.is_some(),
                )
            }
            ForcedRoot::TypeProbe => {
                let replies = self.nodes[root].acts[1 - self.side].len();
                let has_safe = self.has_undominated_allowed();
                let stats = &self.root_types[type_index];
                let selected = (0..self.my_acts.len())
                    .filter(|&own| self.is_eligible(own, has_safe))
                    .flat_map(|own| (0..replies).map(move |reply| (own, reply)))
                    .min_by_key(|&(own, reply)| stats.cells[own * replies + reply].0);
                match selected {
                    Some((own, reply)) => {
                        self.my_n[own] += 1;
                        self.joint_probe_n[own] += 1;
                        (own, Some(reply), true)
                    }
                    None => {
                        let my_mask = &self.my_mask;
                        let my_dominated = &self.my_dominated;
                        let prune_dominated = self.prune_dominated;
                        (
                            select_global(
                                &self.cfg,
                                &mut self.rng,
                                &mut self.my_n,
                                &self.my_w,
                                |a| {
                                    my_mask.as_ref().map_or(true, |m| m[a])
                                        && (!prune_dominated || !has_safe || !my_dominated[a])
                                },
                            ),
                            None,
                            false,
                        )
                    }
                }
            }
        };
        let mut force = [None, None];
        force[self.side] = Some(my_pick);
        if self.opponent_preview_prior && self.is_preview() {
            force[1 - self.side] = belief.sample_preview_prior(
                pick,
                obs,
                &self.nodes[root].acts[1 - self.side],
                &mut self.rng,
            );
        }
        if override_reply {
            force[1 - self.side] = forced_reply;
        }
        let mut joint = [0usize; 2];
        let r = self.run_world_iteration(
            dex,
            world,
            root,
            force,
            &mut joint,
            None,
            trace,
            root_child_value,
        );
        self.record_joint(root, type_index, my_pick, joint, r);
        self.my_w[my_pick] += if self.side == 0 { r } else { 1.0 - r };
        self.done += 1;
        r
    }

    fn record_joint(
        &mut self,
        root: usize,
        type_index: usize,
        my_pick: usize,
        joint: [usize; 2],
        r: f64,
    ) {
        let mine = if self.side == 0 { r } else { 1.0 - r };
        self.endgame.record(
            self.base.turn,
            self.side,
            my_pick,
            mine,
            crate::mcts::last_sim_end(),
        );
        let opp = 1 - self.side;
        let acts = &self.nodes[root].acts[opp];
        let world_warmup = (192 / self.root_types.len().max(2)).clamp(48, 96) as u32;
        let stats = &mut self.root_types[type_index];
        stats.samples += 1;
        if joint[opp] < stats.replies {
            let cell_idx = my_pick * stats.replies + joint[opp];
            let cell = &mut stats.cells[cell_idx];
            cell.0 += 1;
            cell.1 += mine;
            if stats.samples > world_warmup {
                stats.late_samples += 1;
                let late_cell = &mut stats.late_cells[cell_idx];
                late_cell.0 += 1;
                late_cell.1 += mine;
            }
        }
        let indices = self.root_avail.entry(root).or_insert_with(|| {
            acts.iter()
                .map(
                    |&choice| match self.avail.iter().position(|(action, _)| *action == choice) {
                        Some(index) => index,
                        None => {
                            self.avail.push((choice, 0));
                            self.avail.len() - 1
                        }
                    },
                )
                .collect()
        });
        for &index in indices.iter() {
            self.avail[index].1 += 1;
        }
        let Some(&opp_act) = acts.get(joint[opp]) else {
            return;
        };
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
            let mut search = SkuctSearch::new(
                &sim,
                dex,
                self.cfg.clone(),
                seed ^ (0x9E37_79B9 * (ply as u64 + 1)),
            );
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
            let Some(step) = enumerate_step_with_damage_mode(
                dex,
                &sim,
                joint,
                LINE_ENUM_CAP,
                DamageRollMode::Mean,
            ) else {
                break;
            };
            let Some(leaf) = step.leaves.into_iter().max_by(|a, b| {
                a.prob
                    .partial_cmp(&b.prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) else {
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
            if session.expired(clock()) {
                break;
            }
            if !session.can_probe() {
                self.step_one_impl(dex, belief, obs, None, None);
                continue;
            }
            if !self.is_preview() && self.done >= 512 && self.done % 256 == 0 {
                if let Some(eq) = self.bayesian_equilibrium() {
                    self.bayesian_row_q = Some(eq.action_values);
                }
            }
            let (world, root, type_index, pick) = self.next_determinized_world(dex, belief, obs);
            let min_floor_pick = if !self.is_preview()
                && self.done >= 512
                && self.root_types[type_index].replies > 0
            {
                let has_safe = self.has_undominated_allowed();
                let elig: nc2000_engine::smallvec::SmallVec<[usize; 8]> = (0..self.my_acts.len())
                    .filter(|&i| self.is_eligible(i, has_safe))
                    .collect();
                let n_elig = (elig.len().max(1)) as u32;
                let stats = &self.root_types[type_index];
                let replies = stats.replies;
                let world_floor = (stats.samples / (3 * n_elig)).min(48);
                elig.into_iter()
                    .filter_map(|i| {
                        let row_start = i * replies;
                        let w_n: u32 = stats
                            .cells
                            .get(row_start..row_start + replies)
                            .map_or(0, |row| row.iter().map(|&(n, _)| n).sum());
                        (w_n < world_floor).then_some((w_n, self.my_n[i], i))
                    })
                    .min_by_key(|&(w_n, g_n, _)| (w_n, g_n))
                    .map(|(_, _, i)| i)
            } else {
                None
            };
            let my_pick = if let Some(action) = min_floor_pick {
                self.my_n[action] += 1;
                action
            } else {
                let has_safe = self.has_undominated_allowed();
                let my_mask = &self.my_mask;
                let my_dominated = &self.my_dominated;
                let prune_dominated = self.prune_dominated;
                select_global_with_prior(
                    &self.cfg,
                    &mut self.rng,
                    &mut self.my_n,
                    &self.my_w,
                    |a| {
                        my_mask.as_ref().map_or(true, |m| m[a])
                            && (!prune_dominated || !has_safe || !my_dominated[a])
                    },
                    self.bayesian_row_q.as_deref(),
                )
            };
            let mut force = [None, None];
            force[self.side] = Some(my_pick);
            if self.opponent_preview_prior && self.is_preview() {
                force[1 - self.side] = belief.sample_preview_prior(
                    pick,
                    obs,
                    &self.nodes[root].acts[1 - self.side],
                    &mut self.rng,
                );
            }
            let mut joint = [0usize; 2];
            let r = match world {
                NextWorld::Compact {
                    cw,
                    mut cb,
                    world_idx,
                } => {
                    let mut iter_depth = 0u32;
                    let w_sim = &self.world_pool[world_idx].sim;
                    let scratch = &mut self.scratch;
                    let scratch_world_id = &mut self.scratch_world_id;
                    let base_lp = &mut self.base.listener_pool;
                    let v = crate::smmcts::run_compact_iteration_with_endgame(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.compact_nodes,
                        &mut self.compact_table,
                        &mut cb,
                        &cw.table,
                        self.turn_cap,
                        cw.root,
                        force,
                        &mut joint,
                        &mut iter_depth,
                        session,
                        clock,
                        &mut |cb_leaf, mt_leaf, rng_leaf, rollout, sess, clk| {
                            let mut battle_buf = match scratch.take() {
                                Some(mut sim) => {
                                    sim.reset_search_sim(w_sim, [false, false]);
                                    sim
                                }
                                None => Box::new(w_sim.clone()),
                            };
                            std::mem::swap(&mut battle_buf.listener_pool, base_lp);
                            cb_leaf.sync_to_battle(mt_leaf, dex, &mut battle_buf);
                            let mut clk_ref = || clk();
                            let lv = sess.leaf(
                                dex,
                                &cfg,
                                turn_cap,
                                &mut battle_buf,
                                rng_leaf,
                                rollout,
                                &mut clk_ref,
                            );
                            std::mem::swap(&mut battle_buf.listener_pool, base_lp);
                            *scratch_world_id = 0;
                            *scratch = Some(battle_buf);
                            lv
                        },
                    );
                    self.depth_sum += u64::from(iter_depth);
                    self.max_depth = self.max_depth.max(iter_depth);
                    v
                }
                NextWorld::CompactPreview { cpw, turn1_seed } => {
                    let mut iter_depth = 0u32;
                    let scratch = &mut self.scratch;
                    let scratch_world_id = &mut self.scratch_world_id;
                    let base_lp = &mut self.base.listener_pool;
                    let sim_template = &cpw.sim_template;
                    let v = crate::smmcts::run_compact_preview_iteration_with_endgame(
                        &self.cfg,
                        &mut self.rng,
                        &mut self.nodes[root],
                        &mut self.compact_nodes,
                        &mut self.compact_table,
                        &cpw.roster,
                        turn1_seed,
                        self.turn_cap,
                        force,
                        &mut joint,
                        &mut iter_depth,
                        session,
                        clock,
                        &mut |cb_leaf, mt_leaf, rng_leaf, rollout, sess, clk| {
                            let mut battle_buf = match scratch.take() {
                                Some(mut sim) => {
                                    sim.reset_search_sim(sim_template, [false, false]);
                                    sim
                                }
                                None => Box::new((**sim_template).clone()),
                            };
                            std::mem::swap(&mut battle_buf.listener_pool, base_lp);
                            cb_leaf.sync_to_battle(mt_leaf, dex, &mut battle_buf);
                            let mut clk_ref = || clk();
                            let lv = sess.leaf(
                                dex,
                                &cfg,
                                turn_cap,
                                &mut battle_buf,
                                rng_leaf,
                                rollout,
                                &mut clk_ref,
                            );
                            std::mem::swap(&mut battle_buf.listener_pool, base_lp);
                            *scratch_world_id = 0;
                            *scratch = Some(battle_buf);
                            lv
                        },
                    );
                    self.depth_sum += u64::from(iter_depth);
                    self.max_depth = self.max_depth.max(iter_depth);
                    v
                }
                battle_world => self.run_world_iteration(
                    dex,
                    battle_world,
                    root,
                    force,
                    &mut joint,
                    Some(&mut |sim, rng, rollout| {
                        session.leaf(dex, &cfg, turn_cap, sim, rng, rollout, clock)
                    }),
                    None,
                    None,
                ),
            };
            self.record_joint(root, type_index, my_pick, joint, r);
            self.my_w[my_pick] += if self.side == 0 { r } else { 1.0 - r };
            self.done += 1;
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
        self.nodes.len() + self.compact_nodes.len() - self.compact_root_battle_nodes.len()
    }

    pub fn mean_depth(&self) -> f64 {
        if self.done == 0 {
            0.0
        } else {
            self.depth_sum as f64 / f64::from(self.done)
        }
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
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

    /// Exclusion reasons aligned 1:1 with `actions()` and `dominated()`.
    pub fn dominated_reasons(&self) -> &[Option<&'static str>] {
        &self.my_dominated_reasons
    }

    /// Optional root-action legality/pruning mask aligned with `actions()`.
    pub fn mask(&self) -> Option<&[bool]> {
        self.my_mask.as_deref()
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
        self.prune_dominated = true;
    }

    pub fn prune_preview_aliases(&mut self, dex: &Dex) -> usize {
        if !self.is_preview()
            || party_order_matters(dex, &self.base.sides[self.side].roster)
        {
            return 0;
        }
        assert_eq!(self.done, 0, "preview aliases must be pruned before search");
        let mut allowed = self
            .my_mask
            .clone()
            .unwrap_or_else(|| vec![true; self.my_acts.len()]);
        let mut seen = Vec::new();
        let mut removed = 0;
        for (i, action) in self.my_acts.iter().enumerate() {
            if !allowed[i] {
                continue;
            }
            let SearchChoice::Team(mut triple) = *action else {
                continue;
            };
            if triple[2] == 0 {
                continue;
            }
            triple[1..].sort_unstable();
            if seen.contains(&triple) {
                allowed[i] = false;
                removed += 1;
            } else {
                seen.push(triple);
            }
        }
        if removed > 0 {
            self.mask_actions(&allowed);
        }
        removed
    }

    pub fn prune_rest_aliases(&mut self, dex: &Dex) -> usize {
        let Some(id) = self.base.active_id(self.side) else {
            return 0;
        };
        if crate::mask::public_rest_timer(&self.base, id).is_none() {
            return 0;
        }
        let allowed = self
            .my_mask
            .clone()
            .unwrap_or_else(|| vec![true; self.my_acts.len()]);
        let mut indices: Vec<_> = (0..self.my_acts.len()).filter(|&i| allowed[i]).collect();
        indices.sort_by_key(|&i| self.my_dominated[i]);
        let mut representatives: Vec<_> = indices.iter().map(|&i| self.my_acts[i]).collect();
        crate::smmcts::collapse_sleep_choices(&self.base, dex, self.side, &mut representatives);
        let removed = indices.len() - representatives.len();
        if removed > 0 {
            self.mask_actions(
                &self
                    .my_acts
                    .iter()
                    .enumerate()
                    .map(|(i, a)| allowed[i] && representatives.contains(a))
                    .collect::<Vec<_>>(),
            );
        }
        removed
    }

    pub fn joint_probe_visits(&self) -> &[u32] {
        &self.joint_probe_n
    }

    fn reply_metrics(&self, own: usize, n_i: u32, mean_i: f64) -> (f64, f64) {
        // Compute within-type reply risk and opponent-mistake EV per `RootTypeStats`
        // (where the opponent's legal reply set is fixed), then aggregate across
        // determinization types weighted by empirical visits `type_n`. Never pool
        // `self.joint` across distinct opponent movesets, which would mistake a hard
        // opponent candidate set for an opponent reply choice within the same game.
        let mut trap_risk = 0.0_f64;
        let mut mistake_weighted_sum = 0.0_f64;
        let mut mistake_weight_total = 0.0_f64;
        for t in &self.root_types {
            if t.replies == 0 {
                continue;
            }
            let row_start = own * t.replies;
            let row_end = row_start + t.replies;
            let Some(row) = t.cells.get(row_start..row_end) else {
                continue;
            };
            let type_n: u32 = row.iter().map(|&(n, _)| n).sum();
            if type_n == 0 {
                continue;
            }
            let type_w: f64 = row.iter().map(|&(_, w)| w).sum();
            let type_mean = type_w / type_n as f64;
            let (t_risk, t_mistake) =
                crate::smmcts::reply_risk_and_mistake_ev(type_n, type_mean, row.iter().copied());
            let w_t = type_n as f64;
            mistake_weighted_sum += w_t * t_mistake;
            mistake_weight_total += w_t;
            if type_n >= 4 {
                let type_shrink = (w_t / (w_t + 6.0)).clamp(0.0, 1.0);
                let type_drop = ((mean_i - type_mean).max(0.0) * type_shrink).clamp(0.0, 1.0);
                trap_risk = trap_risk.max(type_drop.max(t_risk * type_shrink));
            }
        }
        let mistake_ev = if mistake_weight_total > 0.0 {
            let unattributed = (n_i as f64 - mistake_weight_total).max(0.0);
            (mistake_weighted_sum + unattributed * mean_i) / (mistake_weight_total + unattributed)
        } else {
            mean_i
        };
        (trap_risk, mistake_ev)
    }

    pub fn endgame_analysis(&self) -> Option<crate::smmcts::EndgameDecisionInfo> {
        let has_safe = self.has_undominated_allowed();
        let eligible = |i: usize| self.is_eligible(i, has_safe);
        let (_, eff_n) = self.effective_visits(eligible);
        self.endgame.analyze(
            &self.my_acts,
            &self.my_n,
            &eff_n,
            &self.my_w,
            eligible,
            |own, n_i, mean_i| self.reply_metrics(own, n_i, mean_i),
        )
    }

    pub fn best(&self) -> Option<SearchChoice> {
        let has_safe = self.has_undominated_allowed();
        let eligible = |i: usize| self.is_eligible(i, has_safe);
        let (_, eff_n) = self.effective_visits(eligible);
        if !self.is_preview() {
            return crate::endgame_root::select_blind_root_choice(
                &self.endgame,
                &self.my_acts,
                &self.my_n,
                &eff_n,
                &self.my_w,
                eligible,
                |own, n_i, mean_i| self.reply_metrics(own, n_i, mean_i),
                |base_idx| {
                    if !self.nodes.is_empty() && self.done >= 64 {
                        if let Some(eq) = self.bayesian_equilibrium() {
                            if eq.override_value
                                > eq.pure_guarantees[base_idx] + crate::root_policy::MIN_MIX_MARGIN
                            {
                                let draw =
                                    SplitMix64::new(self.decision_seed ^ u64::from(self.done))
                                        .next_f64();
                                return draw_policy(&self.my_acts, &eq.policy, draw);
                            }
                        }
                    }
                    None
                },
            );
        }
        if self.is_preview() {
            return (0..self.my_acts.len())
                .filter(|&i| eligible(i))
                .max_by(|&a, &b| {
                    eff_n[a]
                        .cmp(&eff_n[b])
                        .then_with(|| self.my_w[a].total_cmp(&self.my_w[b]))
                        .then_with(|| b.cmp(&a))
                })
                .map(|i| self.my_acts[i]);
        }
        unreachable!("non-preview roots return above")
    }

    pub fn bayesian_equilibrium(&self) -> Option<crate::root_policy::BayesianEquilibrium> {
        if self.is_preview() || self.root_types.is_empty() {
            return None;
        }
        let eligible = self.eligible_flags();
        let (_, eff_n) = self.effective_visits(|i| eligible[i]);
        let total_late: u64 = self
            .root_types
            .iter()
            .map(|t| u64::from(t.late_samples))
            .sum();
        let use_late = total_late >= 512;
        let mut pooled_late: FxHashMap<(usize, SearchChoice), (u32, f64)> = FxHashMap::default();
        let mut late_row_n = vec![0u32; self.my_acts.len()];
        let mut late_row_w = vec![0.0f64; self.my_acts.len()];
        if use_late {
            for t in &self.root_types {
                if t.late_samples == 0 || t.replies == 0 {
                    continue;
                }
                for (i, &(n, sum)) in t.late_cells.iter().enumerate() {
                    if n == 0 {
                        continue;
                    }
                    let own = i / t.replies;
                    let reply = i % t.replies;
                    if let Some(&opp_act) = t.reply_acts.get(reply) {
                        let entry = pooled_late.entry((own, opp_act)).or_insert((0, 0.0));
                        entry.0 += n;
                        entry.1 += sum;
                    }
                    if own < late_row_n.len() {
                        late_row_n[own] += n;
                        late_row_w[own] += sum;
                    }
                }
            }
        }
        let mut types: Vec<crate::root_policy::RootType> = Vec::with_capacity(self.root_types.len());
        let mut pooled_priors: Vec<Vec<crate::root_policy::RootCell>> =
            Vec::with_capacity(self.root_types.len());
        for t in &self.root_types {
            let (samples, src_cells) = if use_late && t.late_samples >= 16 {
                (t.late_samples, &t.late_cells)
            } else {
                (t.samples, &t.cells)
            };
            if samples == 0 || t.replies == 0 {
                continue;
            }
            let cells: Vec<crate::root_policy::RootCell> = src_cells
                .iter()
                .enumerate()
                .filter(|(_, (n, _))| *n > 0)
                .map(|(i, &(n, sum))| crate::root_policy::RootCell {
                    own: i / t.replies,
                    reply: i % t.replies,
                    samples: n,
                    mean: sum / f64::from(n),
                })
                .collect();
            let mut priors = Vec::new();
            for own in 0..self.my_acts.len() {
                if !eligible[own] {
                    continue;
                }
                for (reply, &opp_act) in t.reply_acts.iter().enumerate() {
                    if let Some(&(ln, lw)) = pooled_late.get(&(own, opp_act)).filter(|&&(n, _)| n >= 8) {
                        priors.push(crate::root_policy::RootCell {
                            own,
                            reply,
                            samples: ln,
                            mean: lw / f64::from(ln),
                        });
                    } else if let Some(&idx) = self.joint_index.get(&(own, opp_act)) {
                        let (_, _, jn, jw) = self.joint[idx];
                        if jn > 0 {
                            priors.push(crate::root_policy::RootCell {
                                own,
                                reply,
                                samples: jn,
                                mean: jw / f64::from(jn),
                            });
                        }
                    }
                }
            }
            types.push(crate::root_policy::RootType {
                samples,
                replies: t.replies,
                cells,
            });
            pooled_priors.push(priors);
        }
        let mut means = self.means();
        if use_late {
            for own in 0..means.len() {
                if late_row_n[own] >= 16 {
                    means[own] = late_row_w[own] / f64::from(late_row_n[own]);
                }
            }
        }
        crate::root_policy::solve_bayesian_equilibrium_with_priors(
            &eligible,
            &eff_n,
            &means,
            &types,
            &pooled_priors,
        )
    }

    /// Actual execution probabilities, not `rootPolicy` visit diagnostics.
    /// Neither empirical mode certifies equilibrium or policy improvement.
    pub fn decision_policy(
        &self,
        mode: crate::decision::PolicyMode,
    ) -> Option<crate::decision::DecisionPolicy<SearchChoice>> {
        use crate::decision::{DecisionPolicy, PolicyMode};
        let best = self.best()?;
        if mode == PolicyMode::EstimatedMix {
            if let Some(mixed) = self.mixed_policy() {
                if let Ok(policy) = DecisionPolicy::new(
                    "estimated-mix",
                    self.my_acts.iter().copied().zip(mixed.policy),
                ) {
                    return Some(policy);
                }
            }
            return Some(DecisionPolicy::pure("estimated-mix-fallback", best));
        }
        if mode == PolicyMode::VisitSample {
            let has_safe = self.has_undominated_allowed();
            let eligible = |i: usize| self.is_eligible(i, has_safe);
            let (_, eff_n) = self.effective_visits(eligible);
            let entries = self
                .my_acts
                .iter()
                .enumerate()
                .filter(|&(i, _)| eligible(i))
                .map(|(i, &action)| (action, f64::from(eff_n[i])));
            if let Ok(policy) = DecisionPolicy::new("visit-sample", entries) {
                return Some(policy);
            }
            return Some(DecisionPolicy::pure("visit-sample-fallback", best));
        }
        Some(DecisionPolicy::pure("legacy", best))
    }

    pub fn root_policy_evidence(&self) -> (Vec<bool>, Vec<crate::root_policy::RootType>) {
        let eligible = self.eligible_flags();
        let types = self
            .root_types
            .iter()
            .map(|t| crate::root_policy::RootType {
                samples: t.samples,
                replies: t.replies,
                cells: t
                    .cells
                    .iter()
                    .enumerate()
                    .filter(|(_, (n, _))| *n > 0)
                    .map(|(i, &(samples, sum))| crate::root_policy::RootCell {
                        own: i / t.replies,
                        reply: i % t.replies,
                        samples,
                        mean: sum / f64::from(samples),
                    })
                    .collect(),
            })
            .collect();
        (eligible, types)
    }

    pub fn mixed_policy(&self) -> Option<crate::root_policy::MixedRoot> {
        if self
            .my_acts
            .iter()
            .any(|a| matches!(a, SearchChoice::Team(_)))
        {
            return None;
        }
        let (eligible, types) = self.root_policy_evidence();
        crate::root_policy::bayesian_mixed_root(&eligible, self.done, &types)
    }

    pub fn estimated_mix_gain(&self) -> Option<f64> {
        if self
            .my_acts
            .iter()
            .any(|a| matches!(a, SearchChoice::Team(_)))
        {
            return None;
        }
        let (eligible, types) = self.root_policy_evidence();
        crate::root_policy::estimated_bayesian_mix_gain(&eligible, self.done, &types)
    }

    pub fn step_joint_probes(&mut self, dex: &Dex, belief: &Belief, obs: &Observer, n: u32) -> u32 {
        for _ in 0..n {
            let has_safe = self.has_undominated_allowed();
            let mut selected = None;
            for own in 0..self.my_acts.len() {
                if !self.is_eligible(own, has_safe) {
                    continue;
                }
                for &(reply, available) in &self.avail {
                    if available != self.done {
                        continue;
                    }
                    let count = self
                        .joint
                        .iter()
                        .find(|&&(i, a, _, _)| i == own && a == reply)
                        .map(|c| c.2)
                        .unwrap_or(0);
                    if selected.as_ref().map_or(true, |&(_, _, n)| count < n) {
                        selected = Some((own, reply, count));
                    }
                }
            }
            if let Some((own, reply, _)) = selected {
                self.step_forced_joint(dex, belief, obs, own, reply);
            } else {
                self.step_one(dex, belief, obs);
            }
        }
        self.done
    }

    pub fn step_typed_joint_probes(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        n: u32,
    ) -> u32 {
        if self.is_preview() {
            return self.step(dex, belief, obs, n);
        }
        for _ in 0..n {
            self.step_forced_impl(dex, belief, obs, ForcedRoot::TypeProbe);
        }
        self.done
    }

    pub fn finish_with_mixed_probes(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        total: u32,
        reserved: u32,
    ) -> u32 {
        self.finish_mixed_probes(dex, belief, obs, total, reserved, false)
    }

    pub fn finish_with_typed_mixed_probes(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        total: u32,
        reserved: u32,
    ) -> u32 {
        self.finish_mixed_probes(dex, belief, obs, total, reserved, true)
    }

    fn finish_mixed_probes(
        &mut self,
        dex: &Dex,
        belief: &Belief,
        obs: &Observer,
        total: u32,
        reserved: u32,
        typed: bool,
    ) -> u32 {
        assert!(reserved < total && self.done <= total - reserved);
        self.step(dex, belief, obs, total - reserved - self.done);
        let probe = reserved > 0
            && self.estimated_mix_gain().is_some_and(|gain| gain > 0.02)
            && self.mixed_policy().is_none();
        if probe {
            if typed {
                self.step_typed_joint_probes(dex, belief, obs, reserved);
            } else {
                self.step_joint_probes(dex, belief, obs, reserved);
            }
            reserved
        } else {
            self.step(dex, belief, obs, reserved);
            0
        }
    }

    pub fn sample_mixed(&self, draw: f64) -> Option<SearchChoice> {
        assert!(draw.is_finite() && (0.0..1.0).contains(&draw));
        let Some(mixed) = self.mixed_policy() else {
            return self.best();
        };
        draw_policy(&self.my_acts, &mixed.policy, draw)
    }

    /// Whether the root decision is a team preview.
    pub fn is_preview(&self) -> bool {
        matches!(self.my_acts.first(), Some(SearchChoice::Team(_)))
    }
}

/// UCB1 over the global (information-set) root stats — same rule and rng
/// draw pattern as `smmcts::select_ucb`, on plain arrays. `ok` restricts the
/// pick to eligible indices while preserving the exact RNG draw stream.
fn select_global(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    n: &mut [u32],
    w: &[f64],
    ok: impl Fn(usize) -> bool,
) -> usize {
    select_global_with_prior(cfg, rng, n, w, ok, None)
}

fn select_global_with_prior(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    n: &mut [u32],
    w: &[f64],
    ok: impl Fn(usize) -> bool,
    bayesian_q: Option<&[f64]>,
) -> usize {
    let k = n.len();
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
            let raw_mean = wa / na;
            let mean_a = match bayesian_q.and_then(|q| q.get(a).copied()) {
                Some(qa) => qa,
                None => raw_mean,
            };
            let v = mean_a + cfg.c * (ln_total / na).sqrt();
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
    core: ImperfectInfoCore,
}

impl OpenAgent {
    /// Same config surface as `BlindAgent` (the RM root layer fields are
    /// ignored — the blind/open play rule is argmax over the global root).
    pub fn new(cfg: RmConfig, tables: Option<Arc<TableSet>>, seed: u64) -> Self {
        OpenAgent {
            core: ImperfectInfoCore::new(
                cfg,
                BeliefSource::OpenSheet { explicit_sets: None },
                tables,
                seed,
                true,
            ),
        }
    }

    pub fn with_opponent_sets(mut self, sets: Vec<PokemonSet>) -> Self {
        self.core.source = BeliefSource::OpenSheet {
            explicit_sets: Some(sets),
        };
        self
    }

    pub fn with_own_preview_policy(mut self, policy: Vec<f64>) -> Self {
        self.core.own_preview_policy = Some(policy);
        self
    }

    /// Condition hidden opponent bench sampling during battle on a 60-action
    /// canonical preview distribution `policy` (e.g., solved Nash equilibrium),
    /// mixed with `smoothing` over all 3-mon subsets containing already-revealed mons.
    pub fn with_opponent_pick_policy(mut self, policy: Vec<f64>, smoothing: f64) -> Self {
        self.core.opp_pick_policy = Some((policy, smoothing));
        self
    }

    pub fn with_preview_iterations(mut self, preview_iterations: u32) -> Self {
        self.core.set_preview_iterations(preview_iterations);
        self
    }

    pub fn with_adaptive_mixture(mut self, probe_iterations: u32) -> Self {
        self.core.set_adaptive_mixture(probe_iterations);
        self
    }

    pub fn set_iterations(&mut self, iterations: u32) {
        self.core.set_iterations(iterations);
    }

    pub fn search_rng_state(&self) -> u64 {
        self.core.rng.0
    }

    /// Per-slot, per-move counterfactual root marginal values `\Delta Q_{-m_k}`
    /// accumulated over mid-battle move decisions, along with active move-decision
    /// turn counts per roster slot.
    pub fn move_telemetry(&self) -> ([[f64; 4]; 6], [u32; 6]) {
        (self.core.move_marginals, self.core.move_turns)
    }

    /// Explicitly reset per-game state and capture clean Turn-0 rosters at
    /// Team Preview before an external forced preview choice advances `battle`
    /// to Turn 1.
    pub fn init_at_preview(&mut self, battle: &Battle, side: usize) {
        self.core.game = None;
        self.core.pending_preview = Some(PendingPreview {
            side,
            rosters: [battle.sides[0].roster.clone(), battle.sides[1].roster.clone()],
        });
    }

    /// The live belief (None before the first decision) — test surface.
    pub fn belief(&self) -> Option<&Belief> {
        self.core.game.as_ref().map(|g| &g.belief)
    }

    pub fn observer(&self) -> Option<&Observer> {
        self.core.game.as_ref().map(|g| &g.observer)
    }
}

impl Agent for OpenAgent {
    fn root_search_stats(&self) -> Option<RootSearchStats> {
        Some(self.core.root_stats)
    }

    fn requires_player_log(&self) -> bool {
        true
    }

    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> Vec<f64> {
        self.core.root_policy(battle, dex, side, choices)
    }

    fn name(&self) -> String {
        self.core.format_name("open")
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        self.core.choose(battle, dex, side, choices)
    }
}

impl Agent for BlindAgent {
    fn root_search_stats(&self) -> Option<RootSearchStats> {
        Some(self.core.root_stats)
    }

    fn requires_player_log(&self) -> bool {
        true
    }

    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> Vec<f64> {
        self.core.root_policy(battle, dex, side, choices)
    }

    fn name(&self) -> String {
        self.core.format_name("blind")
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        self.core.choose(battle, dex, side, choices)
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
            let (b, a) = (
                &before.sides[side].roster[slot],
                &after.sides[side].roster[slot],
            );
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
    pub assumed: Vec<(
        nc2000_engine::dex::SpeciesId,
        Vec<nc2000_engine::dex::MoveId>,
    )>,
    pub steps: Vec<LineStep>,
}

#[inline]
fn draw_remaining_uniform(rng: &mut SplitMix64, min_d: u8, max_d: u8, elapsed: u8) -> i32 {
    let lo = min_d.max(elapsed.saturating_add(1));
    if lo >= max_d {
        return 1;
    }
    let span = (max_d - lo + 1) as usize;
    let d = lo + rng.below(span) as u8;
    (d - elapsed).max(1) as i32
}

pub fn resample_hidden_timers(
    dex: &Dex,
    sim: &mut Battle,
    specs: &[crate::player_state::HiddenTimerSpec],
    rng: &mut SplitMix64,
) {
    use crate::player_state::HiddenTimerKind;
    let is_castle = sim.is_castle();
    for &spec in specs {
        let side = spec.side as usize;
        let slot = spec.slot as usize;
        if side >= 2 || slot >= sim.sides[side].roster.len() {
            continue;
        }
        let id = PokeId {
            side: spec.side,
            slot: spec.slot,
        };
        if sim.poke(id).hp <= 0 || sim.poke(id).fainted {
            continue;
        }
        match spec.kind {
            HiddenTimerKind::Sleep { slept } => {
                if sim.poke(id).status == Status::Slp {
                    // SlpBeforeMove decrements Time then wakes at <= 0:
                    // PS non-Rest sleep 1..=3 turns => Time in 2..=4;
                    // Castle non-Rest sleep 1..=4 turns => Time in 2..=5.
                    let max_d = if is_castle { 5 } else { 4 };
                    let rem = draw_remaining_uniform(rng, 2, max_d, slept) as i64;
                    sim.poke_mut(id).status_state.set_int(DK::Time, rem);
                }
            }
            HiddenTimerKind::Confusion { ticks } => {
                if let Some(cid) = dex.conds_id("confusion") {
                    if let Some(vs) = sim.poke_mut(id).volatile_mut(cid) {
                        // ConfusionBeforeMove decrements Time then snaps out at 0:
                        // 1..=4 confused turns => Time in 2..=5 (both PS and Castle).
                        let rem = draw_remaining_uniform(rng, 2, 5, ticks) as i64;
                        vs.set_int(DK::Time, rem);
                    }
                }
            }
            HiddenTimerKind::Encore { upkeeps } => {
                if let Some(cid) = dex.conds_id("encore") {
                    if let Some(vs) = sim.poke_mut(id).volatile_mut(cid) {
                        vs.duration = Some(draw_remaining_uniform(rng, 3, 6, upkeeps));
                    }
                }
            }
            HiddenTimerKind::Disable {
                upkeeps,
                acted_on_start,
            } => {
                if let Some(cid) = dex.conds_id("disable") {
                    if let Some(vs) = sim.poke_mut(id).volatile_mut(cid) {
                        let bonus = u8::from(acted_on_start);
                        let (min_b, max_b) = if is_castle { (1, 7) } else { (2, 5) };
                        vs.duration = Some(draw_remaining_uniform(
                            rng,
                            min_b + bonus,
                            max_b + bonus,
                            upkeeps,
                        ));
                    }
                }
            }
            HiddenTimerKind::PartiallyTrapped { upkeeps } => {
                if let Some(cid) = dex.conds_id("partiallytrapped") {
                    if let Some(vs) = sim.poke_mut(id).volatile_mut(cid) {
                        vs.duration = Some(draw_remaining_uniform(rng, 3, 5, upkeeps));
                    }
                }
            }
            HiddenTimerKind::Bide { upkeeps } => {
                if let Some(cid) = dex.conds_id("bide") {
                    if let Some(vs) = sim.poke_mut(id).volatile_mut(cid) {
                        // Bide unleashes when duration == 1:
                        // PS stores 2..=3 turns => duration 3..=4;
                        // Castle stores 1..=2 turns => duration 2..=3.
                        let (min_d, max_d) = if is_castle { (2, 3) } else { (3, 4) };
                        vs.duration = Some(draw_remaining_uniform(rng, min_d, max_d, upkeeps));
                    }
                }
            }
            HiddenTimerKind::LockedMove { upkeeps } => {
                if let Some(cid) = dex.conds_id("lockedmove") {
                    if let Some(vs) = sim.poke_mut(id).volatile_mut(cid) {
                        vs.duration = Some(draw_remaining_uniform(rng, 2, 3, upkeeps));
                    }
                }
            }
        }
    }
}

fn resample_compact_hidden_timers(
    cb: &mut nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    specs: &[crate::player_state::HiddenTimerSpec],
    rng: &mut SplitMix64,
) {
    use crate::player_state::HiddenTimerKind;
    use nc2000_engine::compact::{ext_timer_idx, ext_vol_bit, timer_idx, vol_bit};
    let is_castle = table.ruleset.is_castle();
    for &spec in specs {
        let side = spec.side as usize;
        if side >= 2 {
            continue;
        }
        let Some(p_idx) = (0..3usize).find(|&i| cb.sides[side].party_roster[i] == spec.slot) else {
            continue;
        };
        if (cb.sides[side].alive_mask & (1u8 << p_idx)) == 0 {
            continue;
        }
        let mon = &mut cb.mons[side * 3 + p_idx];
        if mon.hp() <= 0 {
            continue;
        }
        match spec.kind {
            HiddenTimerKind::Sleep { slept } => {
                if mon.status_raw() == Status::Slp as u8 {
                    let max_d = if is_castle { 5 } else { 4 };
                    let rem = draw_remaining_uniform(rng, 2, max_d, slept) as u8;
                    mon.set_status(Status::Slp as u8, rem);
                }
            }
            HiddenTimerKind::Confusion { ticks } => {
                if (mon.vol_flags & vol_bit::CONFUSION) != 0 {
                    mon.timers[timer_idx::CONFUSION] = draw_remaining_uniform(rng, 2, 5, ticks) as u8;
                }
            }
            HiddenTimerKind::Encore { upkeeps } => {
                if (mon.vol_flags & vol_bit::ENCORE) != 0 {
                    mon.timers[timer_idx::ENCORE] = draw_remaining_uniform(rng, 3, 6, upkeeps) as u8;
                }
            }
            HiddenTimerKind::Disable {
                upkeeps,
                acted_on_start,
            } => {
                if (mon.vol_flags & vol_bit::DISABLE) != 0 {
                    let bonus = u8::from(acted_on_start);
                    let (min_b, max_b) = if is_castle { (1, 7) } else { (2, 5) };
                    mon.timers[timer_idx::DISABLE] =
                        draw_remaining_uniform(rng, min_b + bonus, max_b + bonus, upkeeps) as u8;
                }
            }
            HiddenTimerKind::PartiallyTrapped { upkeeps } => {
                if mon.partially_trapped() {
                    mon.ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] =
                        draw_remaining_uniform(rng, 3, 5, upkeeps) as u8;
                }
            }
            HiddenTimerKind::Bide { .. } => {}
            HiddenTimerKind::LockedMove { upkeeps } => {
                if (mon.ext_flags & ext_vol_bit::LOCKED_OUTRAGE) != 0 {
                    mon.ext_timers[ext_timer_idx::LOCKED_TURNS] =
                        draw_remaining_uniform(rng, 2, 3, upkeeps) as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod mixed_selection_tests {
    use super::*;

    #[test]
    fn canonicalizes_only_unseen_interchangeable_bench_slots() {
        let dex = conformance::load_dex();
        let team: Vec<nc2000_engine::battle::PokemonSet> = ["Snorlax", "Zapdos", "Raikou"]
            .map(|species| serde_json::from_value(serde_json::json!({
                "name": species, "species": species, "level": 50, "moves": ["Rest"]
            })).unwrap()).to_vec();
        let mut base = Battle::from_fixture(&dex, "1,2,3,4", &team, &team).unwrap();
        for side in 0..2 {
            base.choose(&dex, side, "team 1,3,2").unwrap();
        }
        for opponent in 0..2 {
            let mut sim = base.clone();
            let saved = sim.sides[1-opponent].party.clone();
            canonicalize_unseen_bench(&dex, &mut sim, opponent);
            assert_eq!(sim.sides[opponent].party.as_slice(), &[0, 1, 2]);
            assert_eq!(sim.sides[1-opponent].party, saved);
            for (pos, &slot) in sim.sides[opponent].party.iter().enumerate() {
                assert_eq!(sim.sides[opponent].roster[usize::from(slot)].position as usize, pos);
            }
            let mut seen = base.clone();
            seen.sides[opponent].roster[2].previously_switched_in = 1;
            canonicalize_unseen_bench(&dex, &mut seen, opponent);
            assert_eq!(seen.sides[opponent].party, base.sides[opponent].party);
            for move_id in ["beatup", "metronome", "mimic", "mirrormove", "sketch", "transform"] {
                let mut ordered = base.clone();
                ordered.sides[opponent].roster[2].move_slots[0].id = dex.moves.id(move_id).unwrap();
                canonicalize_unseen_bench(&dex, &mut ordered, opponent);
                assert_eq!(ordered.sides[opponent].party, base.sides[opponent].party);
            }
        }
    }

    #[test]
    fn forced_steps_preserve_masks_and_skip_preview() {
        let dex = conformance::load_dex();
        let set = |species: &str, moves: &[&str]| -> nc2000_engine::battle::PokemonSet {
            serde_json::from_value(
                serde_json::json!({"species":species,"name":species,"level":50,"moves":moves}),
            )
            .unwrap()
        };
        let mut b = Battle::from_fixture(
            &dex,
            "1,2,3,4",
            &[
                set("Snorlax", &["Return", "Curse", "Rest", "Sleep Talk"]),
                set("Magikarp", &["Splash"]),
            ],
            &[
                set("Suicune", &["Surf", "Rest", "Sleep Talk"]),
                set("Pidgey", &["Tackle"]),
            ],
        )
        .unwrap();
        let preview = b.clone();
        b.choose(&dex, 0, "team 1,2").unwrap();
        b.choose(&dex, 1, "team 1,2").unwrap();
        for base in [&preview, &b] {
            for side in 0..2 {
                let obs = Observer::new(base, side);
                let belief = Belief::pinned_from_battle(base, &obs);
                let cfg = RmConfig {
                    c: 0.4,
                    ..Default::default()
                };
                let mut reference = BlindSearch::new(base, &dex, cfg, side, 901);
                let allowed: Vec<_> = (0..reference.actions().len())
                    .map(|i| reference.is_preview() || i != 1)
                    .collect();
                reference.mask_actions(&allowed);
                for i in 0..300 {
                    if i > 0 && i % 64 == 0 && !reference.is_preview() {
                        let action = (0..allowed.len())
                            .filter(|&i| allowed[i])
                            .min_by_key(|&i| reference.visits()[i])
                            .unwrap();
                        reference.step_forced(&dex, &belief, &obs, action);
                    } else {
                        reference.step(&dex, &belief, &obs, 1);
                    }
                }
                assert_eq!(reference.iterations(), 300);
                assert!(reference
                    .visits()
                    .iter()
                    .zip(&allowed)
                    .all(|(&n, &ok)| ok || n == 0));
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
        for (species,moves) in [("Alakazam",vec!["Recover","Psychic"]),
            ("Gengar",vec!["Destiny Bond","Perish Song"])] {
            let mut battle=Battle::from_fixture(&dex,"1,2,3,4",&[set(species,&moves)],&[set("Snorlax",&["Return"])]).unwrap();
            battle.choose(&dex,0,"team 1").unwrap();battle.choose(&dex,1,"team 1").unwrap();
            let cfg=RmConfig{iterations:300,c:0.4,..Default::default()};
            let (local, obs) = LocalPlayer::new(&battle, &dex, 0).unwrap();
            let mut bel_a = Belief::new(&dex, &MetaPool { teams: vec![] }, &obs);
            let mut bel_b = Belief::new(&dex, &MetaPool { teams: vec![] }, &obs);
            let pos_a = local.position(&dex, &mut bel_a, &obs, &mut SplitMix64::new(17)).unwrap();
            let pos_b = local.position(&dex, &mut bel_b, &obs, &mut SplitMix64::new(17)).unwrap();
            let mut pruned = BlindSearch::from_observation(&pos_a, &dex, cfg.clone(), 0, 99);
            let mut unpruned = BlindSearch::from_observation(&pos_b, &dex, cfg, 0, 99);
            unpruned.prune_dominated = false;
            // Non-public guesses (Alakazam Recover) and all-dominated fallbacks
            // (last-mon Gengar with ONLY Destiny Bond + Perish Song) must preserve
            // the exact unpruned UCB visit distribution and RNG stream.
            pruned.step(&dex, &bel_a, &obs, 300);
            unpruned.step(&dex, &bel_b, &obs, 300);
            assert_eq!(pruned.visits(), unpruned.visits());
            assert_eq!(pruned.best(), unpruned.best());
            assert_eq!(pruned.rng.0, unpruned.rng.0);
        }
    }

    #[test]
    fn adaptive_agent_uses_the_search_mixture_and_a_separate_draw_stream() {
        let dex=conformance::load_dex();
        let set=|species:&str,moves:&[&str]|->nc2000_engine::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({"name":species,"species":species,"level":50,"moves":moves})).unwrap()
        };
        let mut battle=Battle::from_fixture(&dex,"1,2,3,4",&[set("Starmie",&["Return","Surf"])],
            &[set("Wobbuffet",&["Counter","Mirror Coat","Safeguard"])]).unwrap();
        battle.choose(&dex,0,"team 1").unwrap();battle.choose(&dex,1,"team 1").unwrap();
        let cfg=RmConfig{iterations:1000,c:0.4,..Default::default()};
        let mut chooser=BlindAgent::new(cfg.clone(),Arc::new(MetaPool{teams:vec![]}),None,7).with_adaptive_mixture(400);
        let mut exporter=BlindAgent::new(cfg,Arc::new(MetaPool{teams:vec![]}),None,7).with_adaptive_mixture(400);
        let before=chooser.core.adaptive_mixture.as_ref().unwrap().rng.0;
        let mut draw=SplitMix64::new(before);draw.next_f64();
        let choices=battle.legal_choices(&dex,0);
        let chosen=chooser.choose(&battle,&dex,0,&choices);
        let policy=exporter.root_policy(&battle,&dex,0,&choices);
        let expected=chooser.core.decision_policy.clone().unwrap_or_else(||choices.iter().map(|&c|if c==chosen {1.} else {0.}).collect());
        assert_eq!(policy,expected);
        assert_eq!(chooser.core.adaptive_mixture.as_ref().unwrap().rng.0,draw.0);
        assert_eq!(chooser.search_rng_state(),exporter.search_rng_state());
        assert_eq!(chooser.root_search_stats(),exporter.root_search_stats());
        assert_eq!(chooser.core.root_stats.searches,1);
        assert!(chooser.core.root_stats.probe_iterations<=400);
        assert!((policy.iter().sum::<f64>()-1.).abs()<1e-10);
    }

    #[test]
    fn unseen_claw_root_shares_policy_but_preserves_transitions_and_phase_boundaries() {
        let dex = conformance::load_dex();
        let set: nc2000_engine::battle::PokemonSet = serde_json::from_value(serde_json::json!({
            "species":"Snorlax","name":"Snorlax","level":50,"item":"Quick Claw","moves":["Rest","Swagger"]
        })).unwrap();
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &[set.clone()], &[set]).unwrap();
        let cfg = RmConfig::default();
        for idle in [false, true] {
            if idle {
                b.choose(&dex, 0, "team 1").unwrap();
                b.choose(&dex, 1, "team 1").unwrap();
            }
            assert_eq!(b.is_idle_move_request(), idle);
            let mut roots = Vec::new();
            let mut continuation = Vec::new();
            for coin in [false, true] {
                b.set_quick_claw_roll(coin);
                let before = format!("{b:?}");
                roots.push(root_search_key(&cfg, &dex, &mut b));
                continuation.push(key_of(&cfg, &dex, &mut b));
                assert_eq!(before, format!("{b:?}"));
            }
            assert_eq!(roots[0] == roots[1], idle);
            assert_ne!(continuation[0], continuation[1]);
        }
        let obs = Observer::new(&b, 0);
        let belief = Belief::pinned_from_battle(&b, &obs);
        let mut alternative = b.clone();
        alternative.set_quick_claw_roll(!b.quick_claw_rolls[0]);
        alternative.update_all_speeds(&dex);
        let mut states = [&b, &alternative].map(|state| {
            BlindSearch::new(
                state,
                &dex,
                RmConfig {
                    horizon: 1,
                    ..RmConfig::default()
                },
                0,
                771,
            )
        });
        assert_eq!(states[0].dominated(), states[1].dominated());
        for state in &mut states {
            state.step(&dex, &belief, &obs, 256);
        }
        assert_eq!(states[0].visits(), states[1].visits());
        assert_eq!(states[0].means(), states[1].means());
        assert_eq!(states[0].node_count(), states[1].node_count());
        assert_eq!(
            states[0].root_types.len(),
            1,
            "opponent active Quick Claw must not split RootTypeStats by sampled coin"
        );
        b.mid_turn = true;
        assert!(!b.is_idle_move_request());
        b.set_quick_claw_roll(false);
        let key = root_search_key(&cfg, &dex, &mut b);
        b.set_quick_claw_roll(true);
        assert_ne!(key, root_search_key(&cfg, &dex, &mut b));
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
        battle.set_quick_claw_roll(false);
        let key = root_type_key(&mut battle);
        let rng = battle.prng.clone();
        battle.set_quick_claw_roll(true);
        assert_eq!(root_type_key(&mut battle), key);
        assert!(battle.any_quick_claw_roll());
        assert_eq!(format!("{rng:?}"), format!("{:?}", battle.prng));
        let cfg = RmConfig {
            hp_buckets: 4,
            ..Default::default()
        };
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
        let dex = conformance::load_dex();
        let set: nc2000_engine::battle::PokemonSet = serde_json::from_value(serde_json::json!({
            "name":"starmie","species":"starmie","ability":"","level":50,"moves":["Return","Surf"]
        }))
        .unwrap();
        let battle = Battle::from_fixture(&dex, "1,2,3,4", &[set.clone()], &[set]).unwrap();
        let mut search = BlindSearch::new(&battle, &dex, RmConfig::default(), 0, 123);
        let actions = ["return", "surf"].map(|m| SearchChoice::Move(dex.moves.id(m).unwrap()));
        search.my_acts = actions.to_vec();
        search.my_n = vec![20000, 20000];
        search.joint_probe_n = vec![0, 0];
        search.my_w = vec![10000., 10000.];
        search.my_dominated = vec![false, false];
        search.done = 40000;
        search.avail = vec![(actions[1], 40000), (actions[0], 40000)];
        search.joint = vec![
            (0, actions[0], 10000, 0.),
            (0, actions[1], 10000, 10000.),
            (1, actions[0], 10000, 10000.),
            (1, actions[1], 10000, 0.),
        ];
        search.root_types = vec![RootTypeStats {
            samples: 40000,
            late_samples: 0,
            replies: 2,
            reply_acts: actions.to_vec(),
            cells: vec![(10000, 0.), (10000, 10000.), (10000, 10000.), (10000, 0.)],
            late_cells: vec![(0, 0.); 4],
        }];
        let rng = search.rng.0;
        use crate::decision::PolicyMode;
        let policy = search.decision_policy(PolicyMode::EstimatedMix).unwrap();
        assert_eq!(policy.source(), "estimated-mix");
        assert_eq!(policy.actions().len(), 2);
        assert_eq!(policy.sample(0.), actions[0]);
        assert_eq!(policy.sample(0.9), actions[1]);
        assert_eq!(search.sample_mixed(0.), Some(actions[0]));
        assert_eq!(search.sample_mixed(0.5), Some(actions[1]));
        assert_eq!(search.sample_mixed(1. - f64::EPSILON), Some(actions[1]));
        assert_eq!(search.rng.0, rng);
        assert_eq!(search.visits(), &[20000, 20000]);
        search.my_n = vec![20000, 10000];
        assert_eq!(search.best(), Some(actions[0]));
        search.my_n[1] += 30000;
        search.joint_probe_n[1] += 30000;
        assert_eq!(search.best(), Some(actions[0]));
        let visits = search.decision_policy(PolicyMode::VisitSample).unwrap();
        assert!((visits.actions()[0].probability - 2. / 3.).abs() < 1e-12);
        search.mask_actions(&[false, true]);
        assert!(search.mixed_policy().is_none());
        assert_eq!(search.sample_mixed(0.), Some(actions[1]));
        for mode in [
            PolicyMode::Legacy,
            PolicyMode::VisitSample,
            PolicyMode::EstimatedMix,
        ] {
            assert_eq!(search.decision_policy(mode).unwrap().actions().len(), 1);
            assert_eq!(search.decision_policy(mode).unwrap().sample(0.), actions[1]);
        }
        search.mask_actions(&[true, true]);
        search.my_dominated[0] = true;
        assert!(search.mixed_policy().is_none());
        assert_eq!(search.sample_mixed(0.), Some(actions[1]));
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

    #[test]
    fn hidden_timer_distributions_and_resampling() {
        use crate::player_state::{HiddenTimerKind, HiddenTimerSpec};
        let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);
        let wake_prob = |slept: u8, n: usize, r: &mut SplitMix64| -> f64 {
            let mut wakes = 0usize;
            for _ in 0..n {
                if draw_remaining_uniform(r, 2, 4, slept) == 1 {
                    wakes += 1;
                }
            }
            wakes as f64 / n as f64
        };
        assert_eq!(wake_prob(0, 1000, &mut rng), 0.0);
        let p1 = wake_prob(1, 6000, &mut rng);
        assert!((p1 - 1.0 / 3.0).abs() < 0.025, "p1={p1}");
        let p2 = wake_prob(2, 6000, &mut rng);
        assert!((p2 - 0.5).abs() < 0.025, "p2={p2}");
        assert_eq!(wake_prob(3, 1000, &mut rng), 1.0);

        // Disable acted_on_start = true shifts D in {2..=5} -> {3..=6};
        // at upkeeps=2, rem is uniform in {1,2,3,4}.
        let mut disable_rem = [0usize; 5];
        for _ in 0..4000 {
            let rem = draw_remaining_uniform(&mut rng, 3, 6, 2) as usize;
            disable_rem[rem] += 1;
        }
        for c in &disable_rem[1..=4] {
            assert!(*c > 850 && *c < 1150, "disable histogram={disable_rem:?}");
        }

        // Confusion ticks=1 => rem in {1,2,3,4}; Encore upkeeps=2 => rem in {1,2,3,4};
        // PartiallyTrapped upkeeps=1 => rem in {2,3,4}; Bide upkeeps=2 => {1,2};
        // LockedMove upkeeps=1 => {1,2}.
        let dex = conformance::load_dex();
        let set = |species: &str, moves: &[&str]| -> nc2000_engine::battle::PokemonSet {
            serde_json::from_value(
                serde_json::json!({"species":species,"name":species,"level":50,"moves":moves}),
            )
            .unwrap()
        };
        let mut sim = Battle::from_fixture(
            &dex,
            "1,2,3,4",
            &[set("Snorlax", &["Body Slam", "Rest"])],
            &[set("Zapdos", &["Thunderbolt", "Rest"])],
        )
        .unwrap();
        sim.choose(&dex, 0, "team 1").unwrap();
        sim.choose(&dex, 1, "team 1").unwrap();
        let opp = PokeId { side: 1, slot: 0 };
        let tbolt = dex.moves.id("thunderbolt").unwrap();
        sim.poke_mut(opp).last_move = Some(tbolt);
        sim.poke_mut(opp).last_move_encore = Some(tbolt);
        sim.restore_status(&dex, opp, Status::Slp, None);
        sim.add_volatile(&dex, opp, "confusion", None, nc2000_engine::battle::EffectHandle::None);
        sim.add_volatile(&dex, opp, "encore", None, nc2000_engine::battle::EffectHandle::None);
        sim.add_volatile(&dex, opp, "disable", None, nc2000_engine::battle::EffectHandle::None);
        sim.add_volatile(&dex, opp, "partiallytrapped", None, nc2000_engine::battle::EffectHandle::MoveEff(tbolt));
        sim.add_volatile(&dex, opp, "bide", None, nc2000_engine::battle::EffectHandle::None);
        sim.add_volatile(&dex, opp, "lockedmove", None, nc2000_engine::battle::EffectHandle::MoveEff(tbolt));
        let specs = vec![
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::Sleep { slept: 1 } },
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::Confusion { ticks: 1 } },
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::Encore { upkeeps: 2 } },
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::Disable { upkeeps: 2, acted_on_start: true } },
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::PartiallyTrapped { upkeeps: 1 } },
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::Bide { upkeeps: 2 } },
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::LockedMove { upkeeps: 1 } },
        ];
        let mut seen_sleep = [false; 4];
        let mut seen_pt = [false; 5];
        let mut seen_bide = [false; 3];
        let mut seen_locked = [false; 3];
        for _ in 0..200 {
            resample_hidden_timers(&dex, &mut sim, &specs, &mut rng);
            let st = sim.poke(opp).status_state.get_int(DK::Time) as usize;
            assert!((1..=3).contains(&st), "sleep time={st}");
            seen_sleep[st] = true;
            let cf = sim.poke(opp).volatile(dex.conds_id("confusion").unwrap()).unwrap().get_int(DK::Time);
            assert!((1..=4).contains(&cf), "confusion time={cf}");
            let enc = sim.poke(opp).volatile(dex.conds_id("encore").unwrap()).unwrap().duration.unwrap();
            assert!((1..=4).contains(&enc), "encore dur={enc}");
            let dis = sim.poke(opp).volatile(dex.conds_id("disable").unwrap()).unwrap().duration.unwrap();
            assert!((1..=4).contains(&dis), "disable dur={dis}");
            let pt = sim.poke(opp).volatile(dex.conds_id("partiallytrapped").unwrap()).unwrap().duration.unwrap() as usize;
            assert!((2..=4).contains(&pt), "pt dur={pt}");
            seen_pt[pt] = true;
            let bd = sim.poke(opp).volatile(dex.conds_id("bide").unwrap()).unwrap().duration.unwrap() as usize;
            assert!((1..=2).contains(&bd), "bide dur={bd}");
            seen_bide[bd] = true;
            let lm = sim.poke(opp).volatile(dex.conds_id("lockedmove").unwrap()).unwrap().duration.unwrap() as usize;
            assert!((1..=2).contains(&lm), "locked dur={lm}");
            seen_locked[lm] = true;
        }
        assert!(seen_sleep[1] && seen_sleep[2] && seen_sleep[3]);
        assert!(seen_pt[2] && seen_pt[3] && seen_pt[4]);
        assert!(seen_bide[1] && seen_bide[2]);
        assert!(seen_locked[1] && seen_locked[2]);
    }

    #[test]
    fn systematic_resampling_respects_strata_and_shuffle_bounds() {
        let dex = conformance::load_dex();
        let mut pool = crate::preview::load_meta_pool(std::path::Path::new(
            "../../data/meta-pool-v0/meta-pool.json",
        ));
        let base_team = pool.teams[0].clone();
        pool.teams = (0..4)
            .map(|i| {
                let mut t = base_team.clone();
                t.id = format!("variant-{i}");
                t
            })
            .collect();
        let t0 = &pool.teams[0].sets;
        let t1 = &pool.teams[0].sets;
        let battle = Battle::from_fixture(&dex, "1,2,3,4", t0, t1).unwrap();
        let obs = Observer::new(&battle, 0);
        let belief = Belief::new(&dex, &pool, &obs);
        let alive = belief.alive();
        let n = alive.len();
        assert_eq!(n, 4, "expected 4 surviving variants for shared preview");

        let k = n * 2 + 3;
        let mut counts = std::collections::HashMap::new();
        let mut r1 = SplitMix64::new(0x1111_2222_3333_4444);
        let plan1 = belief.root_sampling_plan(k, &mut r1);
        assert_eq!(plan1.len(), k);
        for pick in &plan1 {
            match pick {
                TeamPlanPick::Pool(idx) => *counts.entry(*idx).or_insert(0usize) += 1,
                other => panic!("expected Pool pick while alive non-empty, got {other:?}"),
            }
        }
        let expected = k as f64 / n as f64;
        let lo = expected.floor() as usize;
        let hi = expected.ceil() as usize;
        for &idx in alive {
            let c = counts.get(&idx).copied().unwrap_or(0);
            assert!(
                c >= lo && c <= hi,
                "pool cand {idx} count {c} outside strata bounds [{lo}, {hi}]"
            );
        }
        let mut r2 = SplitMix64::new(0x5555_6666_7777_8888);
        let plan2 = belief.root_sampling_plan(k, &mut r2);
        assert_ne!(plan1, plan2, "Fisher-Yates shuffle must vary order across seeds");
    }

    #[test]
    fn with_max_worlds_and_det_cache_parity_on_fallback_cands() {
        let dex = conformance::load_dex();
        let pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();
        let p1 = pool.teams[0].sets.clone();
        let mut p2 = pool.teams[1].sets.clone();
        p2[0] = serde_json::from_value(serde_json::json!({
            "species": "Pinsir",
            "name": "Pinsir",
            "item": "BrightPowder",
            "ability": "No Ability",
            "moves": ["Swords Dance", "Submission", "Body Slam", "Rock Throw"],
            "level": 50
        }))
        .unwrap();

        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        battle.set_log_enabled(true);
        let mut obs = Observer::new(&battle, 0);
        let mut belief = Belief::new(&dex, &pool, &obs);
        let c0 = battle.legal_choices(&dex, 0)[0];
        let c1 = battle.legal_choices(&dex, 1)[0];
        battle.apply_choices(&dex, [Some(c0), Some(c1)]).unwrap();
        obs.observe(&battle, &dex);
        belief.sync(&dex, &obs);
        assert!(belief.is_fallback());
        assert!(belief.fallback_cand_count() > 1);

        let cfg = RmConfig::default();
        let mut cached = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 0xFACE_B00C)
            .with_max_worlds(12);
        cached.step(&dex, &belief, &obs, 200);
        assert_eq!(cached.world_pool.len(), 12);
        assert!(
            !cached.det_cache.is_empty(),
            "det_cache should populate on FallbackCandPool worlds"
        );
        assert!(
            cached
                .det_cache
                .keys()
                .all(|(pick, _, _, _)| matches!(pick, TeamPlanPick::FallbackCand(_))),
            "det_cache keys should record TeamPlanPick::FallbackCand"
        );

        let mut uncached = BlindSearch::new(&battle, &dex, cfg, 0, 0xFACE_B00C)
            .with_max_worlds(12);
        for _ in 0..200 {
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

    #[test]
    fn forced_switch_uses_ucb1_past_512_iterations_and_breaks_visit_ties_by_mean() {
        let dex = conformance::load_dex();
        let make_mon = |species: &str, moves: &[&str]| -> nc2000_engine::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({
                "name": species,
                "species": species,
                "level": 50,
                "moves": moves,
                "happiness": 255
            }))
            .unwrap()
        };
        let own = [
            make_mon("Caterpie", &["Tackle"]),
            make_mon("Zapdos", &["Thunderbolt", "Thunder", "Hidden Power Ice", "Rest"]),
            make_mon("Magikarp", &["Splash"]),
        ];
        let foe = [make_mon("Slowbro", &["Surf", "Psychic", "Rest", "Thunder Wave"])];
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &own, &foe).unwrap();
        let p0 = battle.legal_choices(&dex, 0)[0];
        let p1 = battle.legal_choices(&dex, 1)[0];
        battle.apply_choices(&dex, [Some(p0), Some(p1)]).unwrap();
        // Let Slowbro genuinely KO a 1-HP Caterpie so the engine produces a
        // real RequestState::Switch forced switch (`replies == 0`).
        let lead = battle.active_id(0).unwrap();
        battle.poke_mut(lead).hp = 1;
        let surf = SearchChoice::Move(dex.moves.id("surf").unwrap());
        let tackle = SearchChoice::Move(dex.moves.id("tackle").unwrap());
        battle.apply_choices(&dex, [Some(tackle), Some(surf)]).unwrap();
        assert_eq!(
            battle.request_state,
            nc2000_engine::state::RequestState::Switch
        );

        let mut obs = Observer::new(&battle, 0);
        obs.observe(&battle, &dex);
        let pool = MetaPool {
            teams: vec![crate::preview::MetaTeam {
                id: "foe".into(),
                sets: foe.to_vec(),
                weight: Some(1.0),
            }],
        };
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.sync(&dex, &obs);
        let cfg = RmConfig {
            iterations: 1200,
            c: 0.4,
            ..Default::default()
        };
        let mut search = BlindSearch::new(&battle, &dex, cfg, 0, 12345);
        search.step(&dex, &belief, &obs, 1200);
        assert!(search.root_replies().is_empty());
        let zap_idx = search
            .actions()
            .iter()
            .position(|&c| c == SearchChoice::Switch(2))
            .unwrap();
        let karp_idx = search
            .actions()
            .iter()
            .position(|&c| c == SearchChoice::Switch(3))
            .unwrap();
        let visits = search.visits();
        assert!(
            visits[zap_idx] > visits[karp_idx] + 200,
            "forced switch should let UCB1 allocate beyond 512 iters (was 50/50 split), got {:?}",
            visits
        );
        assert_eq!(search.best(), Some(SearchChoice::Switch(2)));

        // Visit ties must break by mean reward, never last-index preference.
        let mut tied = BlindSearch::new(&battle, &dex, RmConfig::default(), 0, 99);
        tied.my_acts = vec![SearchChoice::Switch(1), SearchChoice::Switch(2)];
        tied.my_n = vec![4500, 4500];
        tied.joint_probe_n = vec![0, 0];
        tied.my_w = vec![3600.0, 900.0];
        tied.my_dominated = vec![false, false];
        tied.done = 9000;
        assert_eq!(
            tied.best(),
            Some(SearchChoice::Switch(1)),
            "tied visits should pick earlier higher-mean action"
        );
    }

    #[test]
    fn safewin_root_with_chosen_eq_legacy_is_never_overridden_by_bayesian_equilibrium() {
        let dex = conformance::load_dex();
        let make_mon = |species: &str, moves: &[&str]| -> nc2000_engine::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({
                "name": species,
                "species": species,
                "level": 50,
                "moves": moves,
                "happiness": 255
            }))
            .unwrap()
        };
        let mut battle = Battle::from_fixture(
            &dex,
            "1,2,3,4",
            &[make_mon("Snorlax", &["Return", "Curse"])],
            &[make_mon("Suicune", &["Surf"])],
        )
        .unwrap();
        battle.choose(&dex, 0, "team 1").unwrap();
        battle.choose(&dex, 1, "team 1").unwrap();
        let obs = Observer::new(&battle, 0);
        let belief = Belief::pinned_from_battle(&battle, &obs);
        let mut search = BlindSearch::new(
            &battle,
            &dex,
            RmConfig {
                iterations: 128,
                ..Default::default()
            },
            0,
            77,
        );
        search.step(&dex, &belief, &obs, 128);
        assert!(!search.nodes.is_empty());
        assert_eq!(search.my_acts.len(), 2);

        // Configure a SafeWin root where chosen_idx == legacy_idx == 0 (80% winrate, 5,600 visits),
        // while root_types artificially favors action 1 so bayesian_equilibrium() would pick action 1.
        search.my_n = vec![5600, 2900];
        search.joint_probe_n = vec![0, 0];
        search.my_w = vec![5600.0 * 0.80, 2900.0 * 0.66];
        search.my_dominated = vec![false, false];
        search.done = 8500;
        let opp_act = search.root_types[0].reply_acts[0];
        search.root_types = vec![RootTypeStats {
            samples: 8500,
            late_samples: 0,
            replies: 1,
            reply_acts: vec![opp_act],
            cells: vec![(5600, 5600.0 * 0.55), (2900, 2900.0 * 0.90)],
            late_cells: vec![(0, 0.0), (0, 0.0)],
        }];
        search.joint.clear();
        search.joint_index.clear();

        let eq = search.bayesian_equilibrium().unwrap();
        assert_eq!(eq.best_pure_action, 1);

        let info = search
            .endgame
            .analyze(
                &search.my_acts,
                &search.my_n,
                &search.my_n,
                &search.my_w,
                |_| true,
                |own, n_i, mean_i| search.reply_metrics(own, n_i, mean_i),
            )
            .unwrap();
        assert_eq!(info.regime, crate::smmcts::EndgameRegime::SafeWin);
        assert_eq!(info.chosen_idx, 0);
        assert_eq!(info.legacy_idx, 0);
        assert_eq!(search.best(), Some(search.my_acts[0]));
    }

    #[test]
    fn compact_preview_det_cache_hooks_and_castle_parity() {
        let dex = conformance::load_dex();
        let pool = crate::preview::load_meta_pool(std::path::Path::new(
            "../../data/meta-pool-v0/meta-pool.json",
        ));
        let t0 = &pool.teams[0].sets;
        let t1 = &pool.teams[1].sets;
        let battle = Battle::from_fixture(&dex, "10,20,30,40", t0, t1).unwrap();
        assert_eq!(
            battle.request_state,
            nc2000_engine::state::RequestState::TeamPreview
        );
        let obs = Observer::new(&battle, 0);
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.sync(&dex, &obs);

        // 1. `det_cache` populates at TeamPreview with `preview_compact` and is bit-identical to uncached.
        let cfg = RmConfig::default();
        let mut cached = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 0xBEEF_1234);
        cached.step(&dex, &belief, &obs, 400);
        assert!(
            !cached.det_cache.is_empty(),
            "det_cache must populate at TeamPreview"
        );
        assert!(
            cached
                .det_cache
                .values()
                .all(|e| e.preview_compact.is_some()),
            "supported TeamPreview worlds must populate preview_compact"
        );
        assert!(
            !cached.compact_nodes.is_empty(),
            "ply 1+ TeamPreview nodes must live in compact_nodes"
        );
        assert_eq!(cached.node_count(), cached.nodes.len() + cached.compact_nodes.len());

        let mut uncached = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 0xBEEF_1234);
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

        // 2. `step_observed` matches plain `step` bit-for-bit on CompactPreview.
        let mut ref_s = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 0x5151_5151);
        let mut obs_s = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 0x5151_5151);
        let mut results = 0usize;
        let mut root_choices = 0usize;
        let mut ply0_nodes = std::collections::HashSet::new();
        let mut ply1_nodes = std::collections::HashSet::new();
        ref_s.step(&dex, &belief, &obs, 120);
        obs_s.step_observed(&dex, &belief, &obs, 120, &mut |ev| match ev {
            crate::smmcts::SearchTrace::Choice { battle, node, .. } => {
                if battle.turn == 0 {
                    root_choices += 1;
                    ply0_nodes.insert(node);
                } else {
                    ply1_nodes.insert(node);
                }
            }
            crate::smmcts::SearchTrace::Result { battle, .. } => {
                assert!(battle.turn >= 1);
                results += 1;
            }
            _ => {}
        });
        assert_eq!(results, 120);
        assert_eq!(root_choices, 120);
        assert!(ply0_nodes.is_disjoint(&ply1_nodes));
        assert_eq!(ref_s.visits(), obs_s.visits());
        assert_eq!(ref_s.means(), obs_s.means());
        assert_eq!(ref_s.node_count(), obs_s.node_count());

        // 3. `step_forced_joint_with_root_child_value` works on CompactPreview.
        let mut cut = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 0x7777);
        cut.step(&dex, &belief, &obs, 80);
        let before_nodes = cut.node_count();
        let opp_pick = SearchChoice::Team([1, 2, 3]);
        for _ in 0..10 {
            let v = cut.step_forced_joint_with_root_child_value(
                &dex,
                &belief,
                &obs,
                0,
                opp_pick,
                &mut |child| {
                    assert_eq!(child.turn, 1);
                    assert_eq!(child.sides[0].party.len(), 3);
                    assert_eq!(child.sides[1].party.len(), 3);
                    Some(0.625)
                },
            );
            assert_eq!(v, 0.625);
        }
        assert_eq!(cut.node_count(), before_nodes);

        // 4. `step_with_endgame` runs cleanly on CompactPreview.
        let mut eg_s = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 0x8888);
        let mut session = crate::endgame::EndgameSession::new(
            crate::endgame::EndgameConfig::default(),
            0.0,
            1000.0,
        )
        .unwrap();
        eg_s.step_with_endgame(&dex, &belief, &obs, 100, &mut session, &mut || 0.0);
        assert_eq!(eg_s.iterations(), 100);
        assert!(!eg_s.compact_nodes.is_empty());

        // 5. Castle PBS TeamPreview activates CompactPreview cleanly.
        let b_castle = Battle::from_fixture_with_ruleset(
            &dex,
            "10,20,30,40",
            t0,
            t1,
            nc2000_engine::rules::Ruleset::castle_tournament(),
        )
        .unwrap();
        let obs_c = Observer::new(&b_castle, 0);
        let mut belief_c = Belief::new(&dex, &pool, &obs_c);
        belief_c.sync(&dex, &obs_c);
        let mut sc = BlindSearch::new(&b_castle, &dex, cfg.clone(), 0, 0x9999);
        sc.step(&dex, &belief_c, &obs_c, 160);
        assert!(sc.det_cache.values().all(|e| e.preview_compact.is_some()));
        assert!(!sc.compact_nodes.is_empty());

        // 6. Berserk Gene activates CompactPreview natively; Solar Beam falls back cleanly to Battle tree.
        let mut t0_bg = t0.clone();
        t0_bg[0].item = "Berserk Gene".to_string();
        let b_bg = Battle::from_fixture(&dex, "10,20,30,40", &t0_bg, t1).unwrap();
        let obs_bg = Observer::new(&b_bg, 0);
        let mut belief_bg = Belief::new(&dex, &pool, &obs_bg);
        belief_bg.sync(&dex, &obs_bg);
        let mut s_bg = BlindSearch::new(&b_bg, &dex, cfg.clone(), 0, 0xAAAA);
        s_bg.step(&dex, &belief_bg, &obs_bg, 60);
        assert!(!s_bg.compact_nodes.is_empty());
        assert!(s_bg.det_cache.values().all(|e| e.preview_compact.is_some()));
        assert_eq!(s_bg.iterations(), 60);

        let mut t0_sb = t0.clone();
        t0_sb[0].moves[0] = "Solar Beam".to_string();
        let b_sb = Battle::from_fixture(&dex, "10,20,30,40", &t0_sb, t1).unwrap();
        let obs_sb = Observer::new(&b_sb, 0);
        let mut belief_sb = Belief::new(&dex, &pool, &obs_sb);
        belief_sb.sync(&dex, &obs_sb);
        let mut s_sb = BlindSearch::new(&b_sb, &dex, cfg.clone(), 0, 0xBBBB);
        s_sb.step(&dex, &belief_sb, &obs_sb, 60);
        assert!(s_sb.compact_nodes.is_empty());
        assert!(s_sb.det_cache.values().all(|e| e.preview_compact.is_none()));
        assert_eq!(s_sb.iterations(), 60);

        // 7. Active PickPrior (`has_active_pick_draws == true` + `opponent_preview_prior`) must still
        //    hit `det_cache` and stay bit-identical to uncached TeamPreview search.
        let pick_prior = crate::pick_prior::PickPrior::from_json(
            &serde_json::json!({
                "schema": "nc2000-pick-prior-v1",
                "smoothing": 0.2,
                "source": {},
                "rows": [{
                    "team": pool.teams[1].id,
                    "side": 1,
                    "enemy_preview": [],
                    "choices": [{
                        "species": [
                            pool.teams[1].sets[0].species.to_lowercase(),
                            pool.teams[1].sets[1].species.to_lowercase(),
                            pool.teams[1].sets[2].species.to_lowercase()
                        ],
                        "count": 100
                    }]
                }]
            })
            .to_string(),
        )
        .unwrap();
        let mut belief_pp = Belief::new(&dex, &pool, &obs);
        belief_pp.sync(&dex, &obs);
        let own_preview: Vec<crate::pick_prior::PreviewMon> = battle.sides[0]
            .roster
            .iter()
            .map(|p| crate::pick_prior::PreviewMon {
                species: dex.species.key(p.species).to_string(),
                level: p.level,
                gender: String::new(),
                item: p.item.is_some(),
            })
            .collect();
        belief_pp.condition_pick_prior(&dex, &pick_prior, &own_preview, &obs);
        assert!(belief_pp.has_active_pick_draws(&obs));

        let mut pp_cached = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 0xCCCC_1234);
        pp_cached.set_opponent_preview_prior(true);
        pp_cached.step(&dex, &belief_pp, &obs, 200);
        assert!(
            !pp_cached.det_cache.is_empty(),
            "det_cache must still populate at TeamPreview when PickPrior is active"
        );
        assert!(pp_cached.det_cache.values().all(|e| e.preview_compact.is_some()));

        let mut pp_uncached = BlindSearch::new(&battle, &dex, cfg, 0, 0xCCCC_1234);
        pp_uncached.set_opponent_preview_prior(true);
        for _ in 0..200 {
            pp_uncached.det_cache.clear();
            pp_uncached.step_one(&dex, &belief_pp, &obs);
        }
        assert_eq!(pp_cached.visits(), pp_uncached.visits());
        assert_eq!(pp_cached.means(), pp_uncached.means());
        assert_eq!(pp_cached.best(), pp_uncached.best());
        assert_eq!(pp_cached.search_rng_state(), pp_uncached.search_rng_state());
        assert_eq!(pp_cached.node_count(), pp_uncached.node_count());
    }
}
