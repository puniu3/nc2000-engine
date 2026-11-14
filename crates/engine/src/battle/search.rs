//! M3 search API: legal-choice enumeration + structured stepping for
//! DUCT/MCTS-style self-play.
//!
//! Contract with the rest of the engine: enumeration mirrors the validation
//! rules of `choices.rs` exactly (same helpers, same order of checks), and
//! Structured and text choices share validation and commit handling.
//!
//! Search usage: `Battle` is a plain deep-clonable value. Typical loop:
//! ```ignore
//! battle.set_log_enabled(false);          // search mode: skip protocol log
//! while battle.outcome().is_none() {
//!     let picks = [0, 1].map(|s| {
//!         let cs = battle.legal_choices(&dex, s);
//!         if cs.is_empty() { None } else { Some(cs[rng(cs.len())]) }
//!     });
//!     battle.apply_choices(&dex, picks).unwrap();
//! }
//! ```
//! Determinized playouts: clone the battle, then `reseed(...)` the clone so
//! each playout samples fresh chance outcomes.

use crate::dex::{Dex, MoveId};
use crate::prng::Prng;
use crate::state::*;

use super::EngineError;

use crate::fxhash::{FxHasher, PolyRecorder, PolyStep};

#[derive(Clone, Copy, Debug, Default)]
struct CachedMonPrefix {
    valid: bool,
    species: u16,
    hp_type: u8,
    hp_power: i32,
    stored_stats: [i32; 5],
    maxhp: i32,
    init_base_moves: MoveSlots,
    full_step: PolyStep,
    step_a: PolyStep,
    step_b: PolyStep,
}

#[derive(Clone, Copy, Debug, Default)]
struct CachedBenchMon {
    valid: bool,
    fainted: bool,
    position: u8,
    status: u8,
    species: u16,
    hp: i32,
    previously_switched_in: i32,
    status_effect_order: u32,
    item: Option<crate::dex::ItemId>,
    last_item: Option<crate::dex::ItemId>,
    step: PolyStep,
}

/// Precomputed per-world polynomial hash steps and compact roster metadata
/// accelerating `Battle::search_key_with_cache`, `Battle::reset_search_sim`,
/// and `CompactBattle::from_battle_with_roster_cache` with zero key/value drift.
#[derive(Clone, Debug)]
pub struct BattleSearchCache {
    pub hp_buckets: Option<i64>,
    pub omit_damage_bookkeeping: bool,
    pub selected: bool,
    picked_mask: [u8; 2],
    pub any_quick_claw: bool,
    prefix_steps: [[CachedMonPrefix; 6]; 2],
    bench_steps: [[CachedBenchMon; 6]; 2],
    pub roster_compact: crate::compact::RosterCompactCache,
}

impl BattleSearchCache {
    pub fn new(
        battle: &Battle,
        dex: &Dex,
        hp_buckets: Option<i64>,
        omit_damage_bookkeeping: bool,
    ) -> Self {
        let selected = battle.started
            && battle.turn > 0
            && battle.request_state != RequestState::TeamPreview
            && !battle
                .queue
                .iter()
                .any(|action| matches!(action.choice, ActionKind::Team { .. } | ActionKind::Start));
        let mut picked_mask = [0u8; 2];
        for s in 0..2 {
            if selected {
                for &slot in &battle.sides[s].party {
                    if slot < 8 {
                        picked_mask[s] |= 1u8 << slot;
                    }
                }
            } else {
                picked_mask[s] = 0xFF;
            }
        }
        let any_quick_claw = dex.known_items.quickclaw.is_some_and(|qc| {
            battle.sides.iter().enumerate().any(|(s, side)| {
                side.roster.iter().enumerate().any(|(slot, p)| {
                    p.item == Some(qc) && (!selected || (picked_mask[s] & (1u8 << slot.min(7))) != 0)
                })
            })
        });

        let mut prefix_steps = [[CachedMonPrefix::default(); 6]; 2];
        let mut bench_steps = [[CachedBenchMon::default(); 6]; 2];

        for s in 0..2 {
            for (slot, p) in battle.sides[s].roster.iter().take(6).enumerate() {
                if !p.transformed {
                    let mut rec_full = PolyRecorder::new();
                    p.hash_static_prefix_full(&mut rec_full);
                    let mut rec_a = PolyRecorder::new();
                    p.hash_static_prefix_a(&mut rec_a);
                    let mut rec_b = PolyRecorder::new();
                    p.hash_static_prefix_b(&mut rec_b);
                    prefix_steps[s][slot] = CachedMonPrefix {
                        valid: true,
                        species: p.species.0,
                        hp_type: p.hp_type.0,
                        hp_power: p.hp_power,
                        stored_stats: p.stored_stats,
                        maxhp: p.maxhp,
                        init_base_moves: p.base_move_slots,
                        full_step: rec_full.step(),
                        step_a: rec_a.step(),
                        step_b: rec_b.step(),
                    };
                }
                if selected
                    && !p.is_started
                    && !p.is_active
                    && !p.transformed
                    && p.volatiles.is_empty()
                {
                    let mut rec_u = PolyRecorder::new();
                    p.hash_state_with_options(&mut rec_u, hp_buckets, omit_damage_bookkeeping);
                    bench_steps[s][slot] = CachedBenchMon {
                        valid: true,
                        fainted: p.fainted,
                        position: p.position,
                        status: p.status as u8,
                        species: p.species.0,
                        hp: p.hp,
                        previously_switched_in: p.previously_switched_in,
                        status_effect_order: p.status_state.effect_order,
                        item: p.item,
                        last_item: p.last_item,
                        step: rec_u.step(),
                    };
                }
            }
        }

        let roster_compact = crate::compact::RosterCompactCache::from_battle(battle, dex);
        Self {
            hp_buckets,
            omit_damage_bookkeeping,
            selected,
            picked_mask,
            any_quick_claw,
            prefix_steps,
            bench_steps,
            roster_compact,
        }
    }
}

/// A single side's choice, compact and `Copy` (fits in search tree nodes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SearchChoice {
    /// Team-preview pick: 1-based display positions in lead order.
    /// Trailing zeros = unused (only when fewer than 3 are picked, which
    /// NC2000 never does).
    Team([u8; 3]),
    /// Use a move by id. Locked/recharge/struggle turns enumerate exactly one.
    Move(MoveId),
    /// Switch to the 1-based display position (bench = 2..=party size).
    Switch(u8),
    /// Forced pass (only legal when a forced switch has no legal target).
    Pass,
}

impl SearchChoice {
    /// The PS-canonical inputLog string for this choice (what fixtures carry
    /// and what `Battle::choose` parses).
    pub fn to_input(self, dex: &Dex) -> String {
        match self {
            SearchChoice::Team(slots) => {
                let parts: Vec<String> = slots
                    .iter()
                    .filter(|&&s| s != 0)
                    .map(|s| s.to_string())
                    .collect();
                format!("team {}", parts.join(", "))
            }
            SearchChoice::Move(id) => format!("move {}", dex.moves.key(id)),
            SearchChoice::Switch(pos) => format!("switch {pos}"),
            SearchChoice::Pass => "pass".to_string(),
        }
    }
}

/// Terminal battle result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    P1Win,
    P2Win,
    Tie,
}

impl Battle {
    /// Which sides must submit a choice before the battle advances.
    /// (`[false, false]` means the battle has ended.)
    pub fn needs_choice(&self) -> [bool; 2] {
        [0, 1].map(|n| !self.ended && self.sides[n].request_state().is_some())
    }

    /// Terminal result, `None` while the battle is still running.
    pub fn outcome(&self) -> Option<Outcome> {
        if !self.ended {
            return None;
        }
        match self.winner.as_deref() {
            Some(name) if !name.is_empty() => {
                if name == self.sides[0].name {
                    Some(Outcome::P1Win)
                } else {
                    Some(Outcome::P2Win)
                }
            }
            _ => Some(Outcome::Tie),
        }
    }

    /// Replace the PRNG (determinized playouts: clone the battle, reseed the
    /// clone). The seed is the raw 64-bit LCG state.
    pub fn reseed(&mut self, seed: u64) {
        self.prng = crate::prng::BattleRng::seeded(Prng::new(seed));
    }

    /// Hash of the decision-relevant battle state — everything the game's
    /// future depends on EXCEPT the PRNG (search resamples chance, so two
    /// sims that differ only in seed are the same decision point). Used by
    /// state-keyed search trees (M7): equal keys ⇒ same state ⇒ per-node
    /// statistics and cached legal-action sets stay valid; a 64-bit collision
    /// only mis-aggregates one node's statistics, never produces an illegal
    /// choice submission path (choices are still validated by `choose`).
    ///
    /// Skipped on purpose: `prng`, the protocol log and its bookkeeping
    /// (`log`, `log_enabled`, `sent_log_pos`, `last_move_line`), scratch
    /// buffers (`listener_pool`), and the derived `battle_mask` (a function
    /// of hashed state). Mid-event machinery is quiescent at request points;
    /// its occupancy is hashed as a cheap guard.
    pub fn state_key(&self) -> u64 {
        self.state_key_with(None, false)
    }

    /// `state_key` with search-tree abstraction: HP is hashed as one of
    /// `buckets` maxhp-relative buckets and pure roll-magnitude bookkeeping
    /// (`last_damage`, `attacked_by` damages, `hurt_this_turn`) in coarse
    /// steps, so chance outcomes that differ only in damage-roll detail map
    /// to the same node. Discrete chance (KOs, status procs, request kinds,
    /// volatile durations) still splits exactly — that is the open-loop
    /// aliasing M7 exists to fix. The bucketing trades a bounded value
    /// blur inside a node for sample density below it.
    pub fn state_key_bucketed(&self, buckets: i64) -> u64 {
        self.state_key_with(Some(buckets), false)
    }

    fn state_key_with(&self, hp_buckets: Option<i64>, omit_damage_bookkeeping: bool) -> u64 {
        use std::hash::Hasher;
        let mut h = FxHasher::default();
        self.state_hash_into(&mut h, hp_buckets, omit_damage_bookkeeping, false);
        h.finish()
    }

    /// Search-only key with the same field walk as the stable state keys.
    /// Damage bookkeeping may be omitted only when it cannot affect future play.
    #[inline]
    pub fn search_key(&self, hp_buckets: Option<i64>, omit_damage_bookkeeping: bool) -> u64 {
        use std::hash::Hasher;
        let mut h = crate::fxhash::SearchHasher::default();
        self.state_hash_into(&mut h, hp_buckets, omit_damage_bookkeeping, false);
        h.finish()
    }

    #[inline]
    pub fn search_key_canonical_claw(&self, dex: &Dex, hp_buckets: Option<i64>, omit_damage_bookkeeping: bool) -> u64 {
        use std::hash::Hasher;
        let mut h = crate::fxhash::SearchHasher::default();
        self.state_hash_into(&mut h, hp_buckets, omit_damage_bookkeeping, !self.quick_claw_roll_observable(dex));
        h.finish()
    }

    /// Exact search key accelerated by [`BattleSearchCache`] (collapsing unpicked
    /// roster Pokemon and static Pokemon prefixes into `O(1)` polynomial ring steps).
    #[inline]
    pub fn search_key_with_cache(
        &self,
        dex: &Dex,
        hp_buckets: Option<i64>,
        omit_damage_bookkeeping: bool,
        canonical_claw: bool,
        cache: &BattleSearchCache,
    ) -> u64 {
        if cache.hp_buckets != hp_buckets || cache.omit_damage_bookkeeping != omit_damage_bookkeeping
        {
            return if canonical_claw {
                self.search_key_canonical_claw(dex, hp_buckets, omit_damage_bookkeeping)
            } else {
                self.search_key(hp_buckets, omit_damage_bookkeeping)
            };
        }
        let omit_qc = if canonical_claw {
            if !cache.any_quick_claw {
                true
            } else {
                !self.quick_claw_roll_observable(dex)
            }
        } else {
            false
        };
        use std::hash::Hasher;
        let mut h = crate::fxhash::SearchHasher::default();
        self.state_hash_into_cached(&mut h, hp_buckets, omit_damage_bookkeeping, omit_qc, cache);
        let out = h.finish();
        if cfg!(debug_assertions) {
            let expected = if canonical_claw {
                self.search_key_canonical_claw(dex, hp_buckets, omit_damage_bookkeeping)
            } else {
                self.search_key(hp_buckets, omit_damage_bookkeeping)
            };
            debug_assert_eq!(out, expected, "BattleSearchCache key mismatch");
        }
        out
    }

    pub fn type_key(&self) -> u64 {
        use std::hash::Hasher;
        let mut h = crate::fxhash::SearchHasher::default();
        self.state_hash_into(&mut h, None, false, true);
        h.finish()
    }

    /// Two-independent-hash fingerprint (FxHash ‖ SipHash) for
    /// certificate-grade state identity (M17e): the 64-bit search key
    /// tolerates collisions (mis-aggregated statistics), a proof-carrying
    /// merge must not — two hashes over independent functions push the
    /// collision odds to ~2^-128.
    pub fn state_key128(&self) -> u128 {
        self.state_key128_with(false, false)
    }

    /// Certificate-grade solver key with damage-history fields removed.
    /// Callers MUST first establish `!damage_bookkeeping_observable(dex)`;
    /// the method is separate from `state_key128` so ordinary exact identity
    /// can never silently acquire this semantic quotient.
    pub fn state_key128_without_damage_bookkeeping(&self) -> u128 {
        self.state_key128_with(true, false)
    }

    fn state_key128_with(&self, omit_damage_bookkeeping: bool, omit_quick_claw_roll: bool) -> u128 {
        let mut h = crate::fxhash::FingerprintHasher::default();
        self.state_hash_into(&mut h, None, omit_damage_bookkeeping, omit_quick_claw_roll);
        h.finish128()
    }

    pub fn state_key128_without_quick_claw_roll(&self, dex: &Dex, omit_damage_bookkeeping: bool) -> u128 {
        assert!(!self.quick_claw_roll_observable(dex));
        assert!(!omit_damage_bookkeeping || !self.damage_bookkeeping_observable(dex));
        self.state_key128_with(omit_damage_bookkeeping, true)
    }

    /// After selection, unselected items cannot enter play. Fainted holders
    /// are ignored only at idle move requests after pending callbacks finish.
    pub fn quick_claw_roll_observable(&self, dex: &Dex) -> bool {
        let selected = self.started && self.turn > 0 && self.request_state != RequestState::TeamPreview
            && !self.queue.iter().any(|action| matches!(action.choice, ActionKind::Team { .. } | ActionKind::Start));
        let idle = self.is_idle_move_request();
        dex.known_items.quickclaw.is_some_and(|item| self.sides.iter().any(|side|
            side.roster.iter().enumerate().any(|(slot,pokemon)| pokemon.item == Some(item)
                && (!selected || side.party.contains(&(slot as u8)))
                && (!idle || !pokemon.fainted || pokemon.hp > 0))))
    }

    pub fn is_idle_move_request(&self) -> bool {
        self.started && !self.ended && self.request_state == RequestState::Move
            && self.needs_choice() == [true, true] && !self.mid_turn
            && self.queue.is_empty() && self.faint_queue.is_empty()
            && self.event_stack.is_empty() && self.effect_stack.is_empty()
            && self.active_move.is_none() && self.active_pokemon.is_none()
            && self.active_target.is_none() && self.pending_boosts.is_none()
            && self.sides.iter().all(|side| side.choice.actions.is_empty()
                && side.choice.switch_ins.is_empty() && side.choice.forced_switches_left == 0
                && side.choice.forced_passes_left == 0)
    }

    /// Whether dropping damage-history fields can change future behavior.
    /// Counter/Mirror Coat read `attacked_by`. Mimic/Mirror Move/Transform
    /// need no special case because their possible source move is present in
    /// some roster's base/current slots, both of which are scanned. Neither
    /// observer is in Metronome's flagged call pool.
    pub fn damage_bookkeeping_observable(&self, dex: &Dex) -> bool {
        self.sides.iter().flat_map(|s| s.roster.iter()).any(|p| {
            p.base_move_slots
                .iter()
                .chain(p.move_slots.iter())
                .any(|slot| matches!(dex.moves.key(slot.id), "counter" | "mirrorcoat"))
        })
    }

    #[inline]
    fn state_hash_into<H: crate::fxhash::StateHasher>(
        &self,
        h: &mut H,
        hp_buckets: Option<i64>,
        omit_damage_bookkeeping: bool,
        omit_quick_claw_roll: bool,
    ) {
        use std::hash::Hash;
        // Total destructuring on purpose: adding a `Battle` field breaks
        // this fn until the field is placed (hashed or explicitly skipped).
        let Battle {
            preview_level_caps,
            prng: _, // chance is resampled by search
            turn,
            request_state,
            mid_turn,
            started,
            ended,
            winner,
            field,
            sides,
            queue,
            faint_queue,
            log: _, // protocol log + bookkeeping: not state
            log_enabled: _,
            effect_order,
            event_depth,
            last_move_line: _, // log bookkeeping
            last_successful_move_this_turn,
            last_damage,
            quick_claw_roll,
            speed_order,
            format_data,
            sent_log_pos: _, // log bookkeeping
            event_stack,
            effect_stack,
            active_move,
            active_pokemon,
            active_target,
            last_move_id,
            pending_boosts,
            listener_pool: _, // scratch buffers
            battle_mask: _,   // derived from hashed state
        } = self;
        (turn, request_state, mid_turn, started, ended, winner).hash(h);
        if *preview_level_caps != [Some(super::MAX_TOTAL_LEVEL); 2] {
            preview_level_caps.hash(h);
        }
        field.weather.hash(h);
        field.weather_state.hash_state(h);
        field.pseudo_weather.len().hash(h);
        for (cid, st) in field.pseudo_weather.iter() {
            cid.hash(h);
            st.hash_state(h);
        }
        for side in sides.iter() {
            side.hash_with_options(h, hp_buckets, omit_damage_bookkeeping);
        }
        (queue, faint_queue, effect_order, event_depth).hash(h);
        last_successful_move_this_turn.hash(h);
        if !omit_damage_bookkeeping {
            match hp_buckets {
                None => last_damage.hash(h),
                Some(_) => (last_damage / 16).hash(h),
            }
        }
        (if omit_quick_claw_roll { false } else { *quick_claw_roll }, speed_order).hash(h);
        format_data.hash_state(h);
        // mid-event machinery: quiescent at request points, hashed as a guard
        (event_stack.len(), effect_stack.len(), active_move.is_some()).hash(h);
        (active_pokemon, active_target, last_move_id, pending_boosts).hash(h);
    }

    #[cold]
    #[inline(never)]
    fn hash_mon_uncached_fallback(
        p: &Pokemon,
        h: &mut crate::fxhash::SearchHasher,
        hp_buckets: Option<i64>,
        omit_damage_bookkeeping: bool,
    ) {
        p.hash_state_with_options(h, hp_buckets, omit_damage_bookkeeping);
    }

    #[inline]
    fn state_hash_into_cached(
        &self,
        h: &mut crate::fxhash::SearchHasher,
        hp_buckets: Option<i64>,
        omit_damage_bookkeeping: bool,
        omit_quick_claw_roll: bool,
        cache: &BattleSearchCache,
    ) {
        use std::hash::Hash;
        let Battle {
            preview_level_caps,
            prng: _,
            turn,
            request_state,
            mid_turn,
            started,
            ended,
            winner,
            field,
            sides,
            queue,
            faint_queue,
            log: _,
            log_enabled: _,
            effect_order,
            event_depth,
            last_move_line: _,
            last_successful_move_this_turn,
            last_damage,
            quick_claw_roll,
            speed_order,
            format_data,
            sent_log_pos: _,
            event_stack,
            effect_stack,
            active_move,
            active_pokemon,
            active_target,
            last_move_id,
            pending_boosts,
            listener_pool: _,
            battle_mask: _,
        } = self;
        (turn, request_state, mid_turn, started, ended, winner).hash(h);
        if *preview_level_caps != [Some(super::MAX_TOTAL_LEVEL); 2] {
            preview_level_caps.hash(h);
        }
        field.weather.hash(h);
        field.weather_state.hash_state(h);
        field.pseudo_weather.len().hash(h);
        for (cid, st) in field.pseudo_weather.iter() {
            cid.hash(h);
            st.hash_state(h);
        }
        for (side_i, side) in sides.iter().enumerate() {
            let Side {
                name,
                roster,
                party,
                active,
                pokemon_left,
                total_fainted,
                side_conditions,
                slot_conditions,
                handler_mask,
                last_move,
                fainted_this_turn,
                fainted_last_turn,
                request,
                choice,
            } = side;
            name.hash(h);
            roster.len().hash(h);
            for (slot, p) in roster.iter().enumerate() {
                if slot < 6 {
                    let u = &cache.bench_steps[side_i][slot];
                    if u.valid
                        && !p.is_active
                        && !p.is_started
                        && p.previously_switched_in == u.previously_switched_in
                        && p.position == u.position
                        && p.fainted == u.fainted
                        && !p.transformed
                        && p.status as u8 == u.status
                        && p.status_state.effect_order == u.status_effect_order
                        && p.species.0 == u.species
                        && p.hp == u.hp
                        && p.item == u.item
                        && p.last_item == u.last_item
                        && p.volatiles.is_empty()
                    {
                        h.apply_step(u.step);
                        continue;
                    }
                    let pr = &cache.prefix_steps[side_i][slot];
                    if pr.valid
                        && !p.transformed
                        && p.species.0 == pr.species
                        && p.stored_stats == pr.stored_stats
                        && p.hp_type.0 == pr.hp_type
                        && p.hp_power == pr.hp_power
                        && p.maxhp == pr.maxhp
                    {
                        if p.base_move_slots == pr.init_base_moves {
                            h.apply_step(pr.full_step);
                        } else {
                            h.apply_step(pr.step_a);
                            p.base_move_slots.hash_state(h);
                            h.apply_step(pr.step_b);
                        }
                        p.hash_dynamic_body(h, hp_buckets, omit_damage_bookkeeping);
                        continue;
                    }
                }
                Self::hash_mon_uncached_fallback(p, h, hp_buckets, omit_damage_bookkeeping);
            }
            (party, active, pokemon_left, total_fainted).hash(h);
            (side_conditions, slot_conditions, handler_mask).hash(h);
            (last_move, fainted_this_turn, fainted_last_turn).hash(h);
            (request, choice).hash(h);
        }
        (queue, faint_queue, effect_order, event_depth).hash(h);
        last_successful_move_this_turn.hash(h);
        if !omit_damage_bookkeeping {
            match hp_buckets {
                None => last_damage.hash(h),
                Some(_) => (last_damage / 16).hash(h),
            }
        }
        (
            if omit_quick_claw_roll {
                false
            } else {
                *quick_claw_roll
            },
            speed_order,
        )
            .hash(h);
        format_data.hash_state(h);
        (event_stack.len(), effect_stack.len(), active_move.is_some()).hash(h);
        (active_pokemon, active_target, last_move_id, pending_boosts).hash(h);
    }

    /// Populate `out` with all legal choices for `side_n`, avoiding heap allocation
    /// for all move/switch requests (at most 6 actions in 3v3 singles).
    pub fn legal_choices_into(
        &mut self,
        dex: &Dex,
        side_n: usize,
        out: &mut smallvec::SmallVec<[SearchChoice; 6]>,
    ) {
        out.clear();
        if self.ended || side_n >= 2 {
            return;
        }
        let Some(request) = self.sides[side_n].request_state() else {
            return;
        };
        match request {
            RequestKind::TeamPreview => {
                out.extend(self.legal_team_choices(side_n));
            }
            RequestKind::Move => self.legal_move_choices_into(dex, side_n, out),
            RequestKind::Switch => self.legal_forced_switch_choices_into(side_n, out),
            RequestKind::Wait => {}
        }
    }

    /// All legal choices for one side at the current request point. Empty ⇔
    /// the side has nothing to submit (waiting side, or battle ended).
    ///
    /// Needs `&mut self` because locked-move detection runs the (side-effect
    /// free, PRNG-free) `LockMove`/`SemiLockMove` priority events, exactly
    /// like PS request generation does.
    pub fn legal_choices(&mut self, dex: &Dex, side_n: usize) -> Vec<SearchChoice> {
        if self.ended || side_n >= 2 {
            return Vec::new();
        }
        let Some(request) = self.sides[side_n].request_state() else {
            return Vec::new();
        };
        match request {
            RequestKind::TeamPreview => self.legal_team_choices(side_n),
            RequestKind::Move => self.legal_move_choices(dex, side_n),
            RequestKind::Switch => self.legal_forced_switch_choices(side_n),
            RequestKind::Wait => Vec::new(),
        }
    }

    /// All ordered picks of `picked_team_size` distinct display positions
    /// whose level sum respects the side's preview cap. Never empty for a
    /// validator-legal team — the validator guarantees a legal triple.
    fn legal_team_choices(&self, side_n: usize) -> Vec<SearchChoice> {
        let n = self.sides[side_n].party.len() as u8;
        let k = self.picked_team_size(side_n).min(3) as u8;
        let mut out = Vec::new();
        let push = |out: &mut Vec<SearchChoice>, slots: [u8; 3]| {
            let mut positions = [0; 3];
            let mut len = 0;
            for slot in slots.into_iter().filter(|&s| s != 0) {
                positions[len] = slot as usize - 1;
                len += 1;
            }
            if self.preview_level_caps[side_n]
                .is_none_or(|cap| self.picked_total_level(side_n, &positions[..len]) <= cap)
            {
                out.push(SearchChoice::Team(slots));
            }
        };
        for a in 1..=n {
            if k <= 1 {
                push(&mut out, [a, 0, 0]);
                continue;
            }
            for b in 1..=n {
                if b == a {
                    continue;
                }
                if k == 2 {
                    push(&mut out, [a, b, 0]);
                    continue;
                }
                for c in 1..=n {
                    if c != a && c != b {
                        push(&mut out, [a, b, c]);
                    }
                }
            }
        }
        out
    }

    /// Mirrors `choose_move` + `choose_switch` legality on a move request.
    fn legal_move_choices(&mut self, dex: &Dex, side_n: usize) -> Vec<SearchChoice> {
        let mut buf = smallvec::SmallVec::<[SearchChoice; 6]>::new();
        self.legal_move_choices_into(dex, side_n, &mut buf);
        buf.into_vec()
    }

    fn legal_move_choices_into(
        &mut self,
        dex: &Dex,
        side_n: usize,
        out: &mut smallvec::SmallVec<[SearchChoice; 6]>,
    ) {
        let Some(active) = self.active_id(side_n) else {
            out.push(SearchChoice::Pass);
            return;
        };
        if self.poke(active).fainted {
            // only reachable defensively; PS would have made a switch request
            out.push(SearchChoice::Pass);
            return;
        }

        // locked moves preempt everything (thrash/rollout/recharge/...)
        let locked = self
            .get_locked_move(dex, active)
            .or_else(|| self.get_semi_locked_move(dex, active));
        if let Some(locked_id) = locked {
            let id = dex
                .moves
                .id(&locked_id)
                .or_else(|| dex.moves.id("recharge"))
                .expect("locked move id interned");
            out.push(SearchChoice::Move(id));
            return;
        }

        let moves = &self.poke(active).move_slots;
        for slot in moves {
            let choice = SearchChoice::Move(slot.id);
            if !slot.disabled && slot.pp > 0 && !out.contains(&choice) {
                out.push(choice);
            }
        }
        if out.is_empty() {
            out.push(SearchChoice::Move(dex.moves.id("struggle").unwrap()));
        }

        // voluntary switches
        if !self.poke(active).trapped {
            let side = &self.sides[side_n];
            for pos in 1..side.party.len() {
                let slot = side.party[pos];
                if !side.roster[slot as usize].fainted {
                    out.push(SearchChoice::Switch(pos as u8 + 1));
                }
            }
        }
    }

    /// Mirrors `choose_switch`/`choose_pass` legality on a forced switch.
    fn legal_forced_switch_choices(&self, side_n: usize) -> Vec<SearchChoice> {
        let mut buf = smallvec::SmallVec::<[SearchChoice; 6]>::new();
        self.legal_forced_switch_choices_into(side_n, &mut buf);
        buf.into_vec()
    }

    fn legal_forced_switch_choices_into(
        &self,
        side_n: usize,
        out: &mut smallvec::SmallVec<[SearchChoice; 6]>,
    ) {
        let side = &self.sides[side_n];
        if side.choice.forced_switches_left > 0 {
            for pos in 1..side.party.len() {
                let slot = side.party[pos];
                if !side.roster[slot as usize].fainted
                    && !side.choice.switch_ins.contains(&(pos as u8))
                {
                    out.push(SearchChoice::Switch(pos as u8 + 1));
                }
            }
        }
        if out.is_empty() && side.choice.forced_passes_left > 0 {
            out.push(SearchChoice::Pass);
        }
    }

    /// Submit one side's choice (advances the battle when it was the last
    /// side owing one).
    pub fn apply_choice(
        &mut self,
        dex: &Dex,
        side_n: usize,
        choice: SearchChoice,
    ) -> Result<(), EngineError> {
        if side_n >= 2 {
            return Err(EngineError::InvalidChoice(format!("invalid side {side_n}")));
        }
        self.choose_search(dex, side_n, choice)
    }

    /// Submit both sides' choices for this decision point (`None` for a side
    /// that owes none). Afterwards the battle has advanced to the next
    /// request point or ended.
    pub fn apply_choices(
        &mut self,
        dex: &Dex,
        choices: [Option<SearchChoice>; 2],
    ) -> Result<(), EngineError> {
        for (side_n, choice) in choices.into_iter().enumerate() {
            if let Some(c) = choice {
                self.apply_choice(dex, side_n, c)?;
            }
        }
        Ok(())
    }
}
