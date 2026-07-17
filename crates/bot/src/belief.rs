//! M10a belief + determinizer: a candidate set over the M8 meta-pool teams
//! consistent with an `Observer`'s accumulated public knowledge, and the
//! `(true battle, sampled candidate) → battle` imputation that overwrites
//! every hidden opponent field with the candidate's — the substrate M10b
//! plugs under `SkuctSearch` for per-iteration determinization.
//!
//! # Belief
//!
//! Preview filter: the opponent's 6 (species, level) pairs — public from
//! team preview — must match a pool team's exactly (bijectively; Species
//! Clause makes the species→set alignment unique), and per-mon item
//! *presence* must match the `|poke|` preview flag. The one known pool
//! collision (two teams sharing the species+level multiset) keeps both
//! candidates alive. In-battle filter: revealed moves ⊆ candidate set's
//! moves; a known original item must equal the candidate set's item.
//! Weights are uniform over the consistent candidates.
//!
//! **Fallback** (no pool team consistent — a human custom team): a per-mon
//! imputation roster is synthesized instead — nearest pool set by species
//! (first in pedigree order) merged with the revealed knowledge (revealed
//! moves first, pool filler after, observed level/gender, revealed item;
//! unknown item with a preview item flag keeps the pool set's item, or none
//! if the species is outside the pool). Marked by `is_fallback()`. The
//! construction is defensive at every step — a filter dead-end mid-game
//! degrades, never panics.
//!
//! # Determinizer — the hidden-field contract
//!
//! `determinize` clones the true battle and rewrites *everything the
//! observer cannot legitimately know*:
//!
//! - per-mon set fields: unrevealed moves, item (when not publicly known),
//!   DVs / stat exp / happiness → stats, hidden-power type/power, max HP
//!   recomputed from the candidate set (`impute_mon` — a total destructure
//!   of `Pokemon`, so a new state field fails the build here until it is
//!   triaged public/hidden, the `state_key` trick);
//! - the identity of never-appeared opponent picks: which of the 6 roster
//!   mons occupy the unseen party slots is resampled uniformly from the
//!   not-yet-appeared roster (the true picks stay in the support);
//! - a pending, not-yet-executed opponent `Move` action in the queue (only
//!   reachable at a mid-turn Baton Pass switch request — on faints gen 2
//!   cancels all pending moves): its move id is *chosen but unannounced*,
//!   so it is resampled from the imputed active's usable moves, priority /
//!   speed recomputed, and the Pursuit tell (the `pursuit` volatile that
//!   `beforeTurnCallback` plants at turn start) stripped or re-planted to
//!   match;
//! - the PRNG (reseeded from the caller's rng — harmlessly redundant under
//!   `SkuctSearch`, which reseeds per iteration anyway).
//!
//! Kept exactly (public or declared non-goals): HP amounts (never-appeared
//! mons are publicly full), status + status/volatile durations ("hidden
//! counter purity" is a README non-goal), boosts, volatiles, side
//! conditions, move history (`last_move*` — every writer is a public
//! `|move|`), `last_item` bookkeeping (every writer is public, including
//! the gen ≤ 4 switch-in migration), trapped flags, `quick_claw_roll` (a
//! shared per-turn coin — non-goal), and the whole observing side.

use std::collections::HashMap;

use nc2000_engine::battle::{tr, PokemonSet};
use nc2000_engine::dex::{toid, Dex, MoveId};
use nc2000_engine::state::{
    ActionKind, Battle, EffId, EffectState, MoveSlot, MoveSlots, PokeId, Pokemon,
};

use crate::observe::{MonObs, Observer};
use crate::preview::MetaPool;
use crate::rng::SplitMix64;

/// One pool team as an imputation source: reference mons aligned to the
/// opponent's roster slots (`None` = preview-inconsistent).
struct Candidate {
    id: String,
    sets: Vec<PokemonSet>,
    /// Constructed reference mons, index = opponent roster slot.
    refs: Option<Vec<Pokemon>>,
}

pub struct Belief {
    cands: Vec<Candidate>,
    /// Pool indices of the candidates consistent with all observations so
    /// far. Uniform weights. Monotonically non-increasing.
    alive: Vec<usize>,
    /// Synthesized imputation roster when `alive` is empty.
    fallback: Option<Vec<Pokemon>>,
    /// Open-team-sheet mode (M12): the single candidate IS the truth —
    /// filtering is skipped (provably a no-op, and skipping keeps a filter
    /// bug from silently dropping the truth to fallback).
    pinned: bool,
    synced: Option<u64>,
}

impl Belief {
    /// Build the candidate set and apply the preview filter. Call at team
    /// preview, right after `Observer::new`.
    pub fn new(dex: &Dex, pool: &MetaPool, obs: &Observer) -> Belief {
        let cands: Vec<Candidate> = pool
            .teams
            .iter()
            .map(|t| {
                let refs = build_refs(dex, &t.sets, obs.mons());
                Candidate { id: t.id.clone(), sets: t.sets.clone(), refs }
            })
            .collect();
        let mut b =
            Belief { cands, alive: Vec::new(), fallback: None, pinned: false, synced: None };
        b.refilter(dex, obs);
        b
    }

    /// Open-team-sheet belief (M12 product policy): the opponent's TRUE
    /// sets are public, so the belief is pinned to that single candidate —
    /// pool identification never runs. Determinizations then equal the
    /// truth except for what stays hidden by policy: unseen pick identities
    /// (which 3 of 6 + lead) and the mid-turn pending-move scrub. Works for
    /// pool and custom teams uniformly. Call at team preview, right after
    /// `Observer::new` (the refs alignment reads the preview-public facts);
    /// `sync` is then a no-op — the truth is consistent with every
    /// observation by construction.
    pub fn pinned(dex: &Dex, id: &str, sets: &[PokemonSet], obs: &Observer) -> Belief {
        let refs = build_refs(dex, sets, obs.mons());
        debug_assert!(refs.is_some(), "pinned belief: true team failed preview alignment");
        let alive = if refs.is_some() { vec![0] } else { Vec::new() };
        let mut b = Belief {
            cands: vec![Candidate { id: id.to_string(), sets: sets.to_vec(), refs }],
            alive,
            fallback: None,
            pinned: true,
            synced: Some(obs.revision()),
        };
        if b.alive.is_empty() {
            // defensive only: a malformed "true" team degrades like a
            // custom opponent under the identification path
            b.fallback = Some(b.build_fallback(dex, obs));
        }
        b
    }

    /// Open-team-sheet belief for a caller that holds the TRUE battle (the
    /// arena's `open` agent — the M12 product policy in native form):
    /// pinned to reference mons cloned straight from the opponent's true
    /// roster, which is legitimate because both sheets are public under the
    /// policy. Equivalent to `Belief::pinned` with the opponent's set list
    /// (the refs `build_refs` constructs from the sets ARE the roster mons
    /// at team preview); call at team preview, where the roster is fresh.
    pub fn pinned_from_battle(battle: &Battle, obs: &Observer) -> Belief {
        let refs = battle.sides[obs.opp()].roster.clone();
        Belief {
            cands: vec![Candidate {
                id: "opponent".to_string(),
                sets: Vec::new(),
                refs: Some(refs),
            }],
            alive: vec![0],
            fallback: None,
            pinned: true,
            synced: Some(obs.revision()),
        }
    }

    /// Re-filter after new observations. Cheap no-op at an unchanged
    /// observer revision, and always for a pinned belief holding its
    /// candidate (the pinned truth passes every filter by construction).
    pub fn sync(&mut self, dex: &Dex, obs: &Observer) {
        if self.synced == Some(obs.revision()) {
            return;
        }
        if self.pinned && !self.alive.is_empty() {
            debug_assert!(
                self.cands[0]
                    .refs
                    .as_deref()
                    .is_some_and(|refs| consistent(refs, obs.mons())),
                "pinned truth filtered out (observer drift)"
            );
            self.synced = Some(obs.revision());
            return;
        }
        self.refilter(dex, obs);
    }

    /// Pool indices of the consistent candidates (empty ⇔ fallback mode).
    pub fn alive(&self) -> &[usize] {
        &self.alive
    }

    pub fn candidate_count(&self) -> usize {
        self.alive.len()
    }

    pub fn is_fallback(&self) -> bool {
        self.alive.is_empty()
    }

    pub fn candidate_id(&self, pool_idx: usize) -> &str {
        &self.cands[pool_idx].id
    }

    /// Uniformly sample a consistent candidate (`None` = fallback roster).
    pub fn sample(&self, rng: &mut SplitMix64) -> Option<usize> {
        if self.alive.is_empty() {
            None
        } else {
            Some(self.alive[rng.below(self.alive.len())])
        }
    }

    /// Clone the true battle and overwrite all hidden opponent state with a
    /// uniformly sampled candidate's (see the module doc for the contract).
    /// The output is log-off and freshly reseeded.
    pub fn determinize(
        &self,
        dex: &Dex,
        battle: &Battle,
        obs: &Observer,
        rng: &mut SplitMix64,
    ) -> Battle {
        self.determinize_with(dex, battle, obs, self.sample(rng), rng)
    }

    /// `determinize` with an explicit candidate (`None` = fallback roster).
    /// Panics only on API misuse (a pick outside `alive()` / fallback not
    /// yet built) — `determinize` itself can never hit that.
    pub fn determinize_with(
        &self,
        dex: &Dex,
        battle: &Battle,
        obs: &Observer,
        pick: Option<usize>,
        rng: &mut SplitMix64,
    ) -> Battle {
        let refs: &[Pokemon] = match pick {
            Some(i) => self.cands[i]
                .refs
                .as_deref()
                .expect("determinize_with: candidate is preview-inconsistent"),
            None => self
                .fallback
                .as_deref()
                .expect("determinize_with: fallback roster not built (call sync first)"),
        };
        audit_battle_hidden(battle);

        let mut out = battle.clone();
        out.set_log_enabled(false);
        // chance is hidden: the search resamples it anyway, but the artifact
        // must not carry the true RNG stream
        out.reseed(rng.next());

        let opp = obs.opp();
        let roster_len = out.sides[opp].roster.len();

        // ---- hidden pick identities: never-appeared party slots hold one
        // of the not-yet-appeared roster mons — resample uniformly.
        // Position bookkeeping is rebuilt from scratch afterwards: pairwise
        // position swaps corrupt `party`/`position` coherence when the
        // sampled mon already sits in the party at another hidden slot
        // (party[i] duplicated, a party member left with an off-party
        // `position` — switch_in then indexes party[] out of bounds).
        if out.sides[opp].party.len() < roster_len {
            let appeared: Vec<bool> = out.sides[opp]
                .roster
                .iter()
                .map(|p| p.previously_switched_in > 0 || p.is_active)
                .collect();
            let party_len = out.sides[opp].party.len();
            let hidden_positions: Vec<usize> = (0..party_len)
                .filter(|&pos| !appeared[out.sides[opp].party[pos] as usize])
                .collect();
            let mut pool: Vec<u8> =
                (0..roster_len as u8).filter(|&s| !appeared[s as usize]).collect();
            for &pos in &hidden_positions {
                let new_slot = pool.swap_remove(rng.below(pool.len()));
                out.sides[opp].party[pos] = new_slot;
            }
            if !hidden_positions.is_empty() {
                // party members carry their display index; the rest are
                // parked canonically (same assignment ⇒ same state key)
                for pos in 0..party_len {
                    let slot = out.sides[opp].party[pos] as usize;
                    out.sides[opp].roster[slot].position = pos as u8;
                }
                let mut bench_pos = party_len as u8;
                for slot in 0..roster_len {
                    if !out.sides[opp].party.contains(&(slot as u8)) {
                        out.sides[opp].roster[slot].position = bench_pos;
                        bench_pos += 1;
                    }
                }
            }
        }

        // ---- per-mon set imputation (all 6 roster mons, picked or not)
        for slot in 0..roster_len {
            let mon = &mut out.sides[opp].roster[slot];
            impute_mon(mon, &refs[slot], &obs.mons()[slot], dex);
            out.refresh_poke_mask(dex, PokeId { side: opp as u8, slot: slot as u8 });
        }
        // active speed reflects stats (+ paralysis, quick claw) like the
        // engine's update_all_speeds
        if let Some(a) = out.active_id(opp) {
            if !out.poke(a).fainted {
                out.update_speed(dex, a);
            }
        }

        // ---- pending opponent Move in the queue: chosen but unannounced
        self.scrub_pending_move(dex, &mut out, opp, rng);

        out.battle_mask = out.recompute_battle_mask(dex);
        out
    }

    /// A not-yet-executed opponent `Move` action (mid-turn Baton Pass
    /// window) carries the opponent's hidden selection. Resample it from
    /// the imputed active's usable moves; locked/recharge turns are forced
    /// and public, so they keep the true id.
    fn scrub_pending_move(&self, dex: &Dex, out: &mut Battle, opp: usize, rng: &mut SplitMix64) {
        let Some(active) = out.active_id(opp) else { return };
        let pending: Vec<usize> = out
            .queue
            .iter()
            .enumerate()
            .filter(|(_, q)| {
                q.pokemon == Some(active) && matches!(q.choice, ActionKind::Move { .. })
            })
            .map(|(i, _)| i)
            .collect();
        if pending.is_empty() {
            return;
        }
        if out
            .get_locked_move(dex, active)
            .or_else(|| out.get_semi_locked_move(dex, active))
            .is_some()
        {
            return; // forced continuation: publicly known
        }
        let usable: Vec<MoveId> = out
            .poke(active)
            .move_slots
            .iter()
            .filter(|s| s.pp > 0 && !s.disabled)
            .map(|s| s.id)
            .collect();
        let new_id = if usable.is_empty() {
            dex.moves.id("struggle").expect("struggle interned")
        } else {
            usable[rng.below(usable.len())]
        };
        for i in pending {
            let old_id = match out.queue[i].choice {
                ActionKind::Move { move_id, .. } => move_id,
                _ => unreachable!(),
            };
            if old_id == new_id {
                continue;
            }
            // the pursuit volatile beforeTurnCallback planted at turn start
            // encodes the pending choice — strip it, re-plant if resampled
            if out.poke(active).has_volatile(
                dex.conds_id("pursuit").expect("pursuit interned"),
            ) {
                out.remove_volatile(dex, active, "pursuit");
            }
            if let ActionKind::Move { move_id, .. } = &mut out.queue[i].choice {
                *move_id = new_id;
            }
            out.queue[i].priority = dex.move_static(new_id).priority as f64;
            out.queue[i].fractional_priority = 0.0;
            out.queue[i].speed = out.get_pokemon_action_speed(dex, active) as f64;
            if dex.moves.key(new_id) == "pursuit" {
                if let Some(target) = out.active_id(1 - opp) {
                    out.before_turn_callback(dex, new_id, active, target);
                }
            }
        }
    }

    // ------------------------------------------------------------- filter

    fn refilter(&mut self, dex: &Dex, obs: &Observer) {
        self.alive = (0..self.cands.len())
            .filter(|&i| match &self.cands[i].refs {
                Some(refs) => consistent(refs, obs.mons()),
                None => false,
            })
            .collect();
        if self.alive.is_empty() {
            self.fallback = Some(self.build_fallback(dex, obs));
        }
        self.synced = Some(obs.revision());
    }

    // ----------------------------------------------------------- fallback

    /// Per-mon synthesized imputation roster for a non-pool opponent.
    fn build_fallback(&self, dex: &Dex, obs: &Observer) -> Vec<Pokemon> {
        let sets: Vec<PokemonSet> =
            obs.mons().iter().map(|mo| self.fallback_set(dex, mo)).collect();
        match Battle::from_fixture(dex, "1,2,3,4", &sets, &sets) {
            Ok(b) => b.sides[0].roster.clone(),
            Err(_) => {
                // defensive second stage: minimal sets cannot fail (species /
                // level / gender come from the live battle, no parsed strings)
                let minimal: Vec<PokemonSet> = obs
                    .mons()
                    .iter()
                    .map(|mo| base_set(dex, mo))
                    .collect();
                Battle::from_fixture(dex, "1,2,3,4", &minimal, &minimal)
                    .expect("minimal fallback set must construct")
                    .sides[0]
                    .roster
                    .clone()
            }
        }
    }

    fn fallback_set(&self, dex: &Dex, mo: &MonObs) -> PokemonSet {
        // nearest pool set by species (first in pedigree order)
        let nearest: Option<&PokemonSet> = self.cands.iter().find_map(|c| {
            c.sets
                .iter()
                .find(|s| dex.species.id(&toid(&s.species)) == Some(mo.species))
        });
        let mut set = nearest.cloned().unwrap_or_else(|| base_set(dex, mo));
        set.level = mo.level;
        set.gender = Some(match mo.gender.as_str() {
            "" => "N".to_string(),
            g => g.to_string(),
        });
        set.name = String::new();
        // moves: revealed first, pool filler after
        let mut moves: Vec<String> =
            mo.revealed_moves.iter().map(|&m| dex.moves.key(m).to_string()).collect();
        for name in nearest.map(|s| s.moves.clone()).unwrap_or_default() {
            if moves.len() >= 4 {
                break;
            }
            if dex.moves.id(&toid(&name)).map_or(true, |id| !mo.revealed_moves.contains(&id)) {
                moves.push(name);
            }
        }
        moves.truncate(4); // hard cap: >4 slots would assert in construction
        set.moves = moves;
        // item: revealed original > pool set's (when preview showed one) > none
        let pool_item = std::mem::take(&mut set.item);
        set.item = match mo.item.original {
            Some(Some(x)) => dex.items.key(x).to_string(),
            Some(None) => String::new(),
            None if mo.preview_has_item => pool_item, // pool set's, may be ""
            None => String::new(),
        };
        set
    }
}

/// Species/level/gender-only set (always constructible: every field comes
/// from the live battle, none is parsed). Max stat exp / DVs, the format
/// norm.
fn base_set(dex: &Dex, mo: &MonObs) -> PokemonSet {
    let evs = ["hp", "atk", "def", "spa", "spd", "spe"]
        .iter()
        .map(|k| (k.to_string(), 255u16))
        .collect();
    PokemonSet {
        name: String::new(),
        species: dex.species.key(mo.species).to_string(),
        item: String::new(),
        ability: "No Ability".to_string(),
        moves: Vec::new(),
        level: mo.level,
        evs: Some(evs),
        ivs: None,
        happiness: None,
        gender: Some(match mo.gender.as_str() {
            "" => "N".to_string(),
            g => g.to_string(),
        }),
    }
}

/// Construct a pool team's reference mons and align them to the opponent's
/// observed roster slots by (species, level). `None` = preview-inconsistent
/// (species/level mismatch, item-presence mismatch, or unbuildable team).
fn build_refs(dex: &Dex, sets: &[PokemonSet], mons: &[MonObs]) -> Option<Vec<Pokemon>> {
    if sets.len() != mons.len() {
        return None;
    }
    let built = Battle::from_fixture(dex, "1,2,3,4", sets, sets).ok()?;
    let roster = &built.sides[0].roster;
    // species → set indices (Species Clause ⇒ one each, but stay general)
    let mut by_species: HashMap<u16, Vec<usize>> = HashMap::new();
    for (i, p) in roster.iter().enumerate() {
        by_species.entry(p.species.0).or_default().push(i);
    }
    let mut out = Vec::with_capacity(mons.len());
    for mo in mons {
        let slots = by_species.get_mut(&mo.species.0)?;
        let k = slots
            .iter()
            .position(|&i| roster[i].level == mo.level && roster[i].item.is_some() == mo.preview_has_item)?;
        out.push(roster[slots.remove(k)].clone());
    }
    Some(out)
}

/// In-battle consistency: every observation must fit the candidate.
fn consistent(refs: &[Pokemon], mons: &[MonObs]) -> bool {
    refs.iter().zip(mons).all(|(r, mo)| {
        mo.revealed_moves
            .iter()
            .all(|m| r.base_move_slots.iter().any(|s| s.id == *m))
            && match mo.item.original {
                Some(orig) => r.item == orig,
                None => true,
            }
    })
}

// -------------------------------------------------------------- imputation

/// Overwrite one opponent mon's hidden set-level fields with the reference
/// mon's, keeping everything public exactly.
///
/// TOTAL destructure of `Pokemon` on purpose (the `state_key` trick): adding
/// a state field breaks this fn until the field is triaged public (keep) /
/// hidden (impute). Caller refreshes `handler_mask` afterwards.
fn impute_mon(dst: &mut Pokemon, refm: &Pokemon, mo: &MonObs, dex: &Dex) {
    debug_assert_eq!(refm.base_species, dst.base_species, "ref/roster species drift");
    debug_assert_eq!(refm.level, dst.level, "ref/roster level drift");
    let Pokemon {
        species,                  // public: preview details; Transform is announced
        base_species: _,          // public: preview details
        name: _,                  // public: shown on switch-in
        level,                    // public: preview details
        gender: _,                // public: preview details
        happiness,                // HIDDEN → candidate's (Return/Frustration power)
        set_ivs,                  // HIDDEN → candidate's (DV purity: only via the set)
        set_evs,                  // HIDDEN → candidate's (stat exp)
        base_move_slots,          // HIDDEN except revealed usage → merged below
        hp_type,                  // derived from DVs: HIDDEN unless transformed
        hp_power,                 //   (a Transform copy mirrors a public mon)
        base_hp_type,             // HIDDEN → candidate's
        base_hp_power,            // HIDDEN → candidate's
        base_stored_stats,        // HIDDEN → candidate's
        stored_stats,             // HIDDEN unless transformed (copies a public mon)
        base_maxhp,               // HIDDEN → candidate's
        maxhp,                    // HIDDEN → candidate's (Transform does not copy HP)
        hp,                       // PUBLIC amount (explicitly granted); clamped below
        status: _,                // public (announced)
        status_state: _,          // hidden sleep counter: declared non-goal, keep
        boosts: _,                // public (announced)
        move_slots,               // rebuilt from the merged base below
        item,                     // HIDDEN unless publicly known (ItemObs::current)
        last_item: _,             // public-equivalent: every writer is a public event
        item_state,               // follows `item`
        types: _,                 // public (species types; Conversion announced)
        volatiles: _,             // public (all announced); the pending-Pursuit tell
                                  //   is handled by the queue scrub
        handler_mask: _,          // derived — caller refreshes
        transformed,              // public (announced)
        fainted: _,               // public
        faint_queued: _,          // public
        is_active,                // public flow (read below)
        is_started: _,            // public flow
        position: _,              // public flow (display slots)
        active_turns: _,          // public flow
        active_move_actions: _,   // public flow
        newly_switched: _,        // public flow
        being_called_back: _,     // public flow
        dragged_in: _,            // public flow
        previously_switched_in,   // public flow (appearance count, read below)
        switch_flag: _,           // public (Baton Pass announced)
        force_switch_flag: _,     // public (phazing announced)
        skip_before_switch_out: _,// public flow
        trapped: _,               // public (Mean Look etc. announced)
        maybe_trapped: _,         // public
        last_move: _,             // public move history
        last_move_encore: _,      // public move history
        last_move_used: _,        // public move history (called moves are logged)
        last_move_target_loc: _,  // public
        move_this_turn: _,        // public
        move_this_turn_result: _, // public (success is visible)
        move_last_turn_result: _, // public
        hurt_this_turn: _,        // public damage events
        stats_raised_this_turn: _,// public
        stats_lowered_this_turn: _,// public
        used_item_this_turn: _,   // public (item events announced)
        last_damage: _,           // public damage events
        attacked_by: _,           // public damage events
        times_attacked: _,        // public
        speed,                    // derived from (hidden) stats → recomputed
    } = dst;

    let was_transformed = *transformed;

    *happiness = refm.happiness;
    *set_ivs = refm.set_ivs;
    *set_evs = refm.set_evs;

    *base_stored_stats = refm.base_stored_stats;
    *base_hp_type = refm.base_hp_type;
    *base_hp_power = refm.base_hp_power;
    if !was_transformed {
        *stored_stats = refm.base_stored_stats;
        *hp_type = refm.base_hp_type;
        *hp_power = refm.base_hp_power;
    }
    *base_maxhp = refm.base_maxhp;
    *maxhp = refm.base_maxhp;
    // HP amount is public; a mon that never switched in is publicly full
    let appeared = *previously_switched_in > 0 || *is_active || mo.appeared;
    if appeared {
        *hp = (*hp).min(*maxhp);
    } else {
        *hp = *maxhp;
    }

    // ---- moves: candidate's set; revealed slots keep their live usage
    let old_base = *base_move_slots;
    let mut nb = MoveSlots::default();
    for rs in refm.base_move_slots.iter() {
        if mo.revealed_moves.contains(&rs.id) {
            match old_base.iter().find(|s| s.id == rs.id) {
                Some(os) => nb.push(MoveSlot { shared: true, ..*os }),
                None => nb.push(*rs), // fallback-merge oddity: fresh slot
            }
        } else {
            nb.push(*rs); // unrevealed: candidate's move at full PP
        }
    }
    debug_assert!(
        mo.revealed_moves.iter().all(|m| nb.iter().any(|s| s.id == *m)),
        "revealed move missing from imputation source (filter drift)"
    );
    *base_move_slots = nb;
    if was_transformed {
        // move_slots are 5-PP copies of a public mon: keep
    } else if move_slots.iter().any(|s| !s.shared) {
        // Mimic overlay (public: announced): keep the overlay slot, rebuild
        // the rest from the merged base. Mirroring is by move id, so slot
        // order never has to line up with the base list.
        let overlay: Vec<MoveSlot> = move_slots.iter().filter(|s| !s.shared).copied().collect();
        let mut nm = nb;
        if let (Some(ov), Some(mimic)) = (overlay.first(), dex.moves.id("mimic")) {
            if let Some(pos) = (0..nm.len()).find(|&i| nm[i].id == mimic) {
                nm[pos] = *ov;
            }
        }
        *move_slots = nm;
    } else {
        *move_slots = nb;
    }

    // ---- item
    match mo.item.current {
        Some(known) => {
            // publicly known current item: the true field already equals it
            debug_assert_eq!(*item, known, "item tracking drift");
        }
        None => {
            *item = refm.item;
            *item_state = EffectState {
                id: refm.item.map(EffId::Item).unwrap_or_default(),
                effect_order: item_state.effect_order,
                ..Default::default()
            };
        }
    }

    // ---- speed cache
    if !was_transformed {
        *speed = refm.base_stored_stats[4];
    } else {
        // Transform caches the user's own spe (own DVs / stat exp / level)
        // on the copied species' base speed — mirror transform_into
        let base = dex.species.get(*species).base_stats.spe as f64;
        let iv = refm.set_ivs[5] as f64;
        let ev_term = tr(refm.set_evs[5] as f64 / 4.0);
        *speed = tr(tr(2.0 * base + iv + ev_term) * *level as f64 / 100.0 + 5.0) as i32;
    }
}

/// Battle- and Side-level hidden-field triage (documentation + drift guard,
/// mirroring `Battle::state_key`'s total destructure): a new `Battle` or
/// `Side` field fails the build here until it is placed on the public-keep
/// or hidden-overwrite side of the determinizer.
fn audit_battle_hidden(b: &Battle) {
    let Battle {
        prng: _,               // HIDDEN → reseeded by determinize
        turn: _,               // public
        request_state: _,      // public
        mid_turn: _,           // public flow
        started: _,            // public
        ended: _,              // public
        winner: _,             // public
        field: _,              // public (weather / pseudo-weathers announced)
        sides,                 // triaged below
        queue: _,              // pending opponent Move = HIDDEN → scrub_pending_move
        faint_queue: _,        // public (empty at request points)
        log: _,                // not state
        log_enabled: _,        // not state
        effect_order: _,       // bookkeeping (ordering of public events)
        event_depth: _,        // quiescent at request points
        last_move_line: _,     // log bookkeeping
        last_successful_move_this_turn: _, // public move history
        last_damage: _,        // public damage events
        quick_claw_roll: _,    // hidden shared per-turn coin: declared non-goal
        speed_order: _,        // public (resolved order was displayed)
        format_data: _,        // static
        sent_log_pos: _,       // log bookkeeping
        event_stack: _,        // quiescent at request points
        effect_stack: _,       // quiescent at request points
        active_move: _,        // quiescent at request points
        active_pokemon: _,     // quiescent at request points
        active_target: _,      // quiescent at request points
        last_move_id: _,       // public move history
        pending_boosts: _,     // quiescent at request points
        listener_pool: _,      // scratch
        battle_mask: _,        // derived — recomputed after imputation
    } = b;
    for side in sides.iter() {
        let nc2000_engine::state::Side {
            name: _,             // public
            roster: _,           // per-mon triage: impute_mon
            party: _,            // hidden pick identity → resampled in determinize
            active: _,           // public
            pokemon_left: _,     // public
            total_fainted: _,    // public
            side_conditions: _,  // public (announced)
            slot_conditions: _,  // public (announced)
            handler_mask: _,     // derived — refreshed after imputation
            last_move: _,        // public (self-KO clause bookkeeping of announced moves)
            fainted_this_turn: _,// public
            fainted_last_turn: _,// public
            request: _,          // public
            choice: _,           // empty between commits; forced-switch counters
                                 //   derive from public party state
        } = side;
    }
}
