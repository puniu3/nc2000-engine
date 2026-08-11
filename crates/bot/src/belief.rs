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
//! (first in pedigree order), then the community-rental prior when the
//! species is absent from the pool, merged with the revealed knowledge
//! (revealed moves first, prior filler after, observed level/gender,
//! revealed item). A species absent from both sources receives a
//! deterministic format-legal move, never an empty set/implicit Struggle.
//! Marked by `is_fallback()`. The construction is defensive at every step —
//! a filter dead-end mid-game degrades, never panics.
//!
//! **M18 — community marginal prior** (`set_prior`, opt-in). With a
//! `prior::BeliefPrior` installed, the fallback's fixed nearest-set filler is
//! replaced by a *weighted draw without replacement* over the species'
//! per-move carry-marginals, redrawn on `determinize`'s per-iteration rng —
//! so ISMCTS averages over the belief instead of over-committing to one MAP
//! set. Revealed moves still lead and are never resampled, legality is still
//! checked against the format learnsets, and the `≤ 4` clamp still holds:
//! the data can only choose among moves the code already permits. A species
//! the table does not mention, an empty table, and no table at all are all
//! the unchanged path, rng consumption included. The sampler is reachable
//! from exactly one branch of `determinize_with` (`pick == None`), so the
//! pinned / open-sheet product cannot be touched by it.
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
//! the gen ≤ 4 switch-in migration), trapped flags, and the whole observing
//! side. `quick_claw_roll` — this turn's preempt coin — is resampled rather
//! than kept: it is unobservable chance, and carrying the true value through
//! leaked it in the arena while pinning it to the importer's `false` on the
//! ladder.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use nc2000_engine::battle::{tr, PokemonSet};
use nc2000_engine::dex::{toid, Dex, MoveId};
use nc2000_engine::state::{
    ActionKind, Battle, EffId, EffectState, MoveSlot, MoveSlots, PokeId, Pokemon,
};
use nc2000_engine::validate::{validate_team, Learnsets};

use crate::observe::{move_matches, MonObs, Observer};
use crate::preview::MetaPool;
use crate::prior::BeliefPrior;
use crate::rng::SplitMix64;

/// Canonicalized no-OHKO community rentals, baked into the bot so the live
/// protocol/WASM paths get the prior without runtime filesystem plumbing.
/// Parsing is paid once per process; source order is the deterministic
/// pedigree order.
const COMMUNITY_RENTALS_JSON: &str =
    include_str!("../../../data/community-rentals-v0/teams.json");
const FORMAT_LEARNSETS_JSON: &str = include_str!("../../../data/learnsets-gen2.json");

#[derive(serde::Deserialize)]
struct CommunityRentalDb {
    teams: Vec<CommunityRentalTeam>,
}

#[derive(serde::Deserialize)]
struct CommunityRentalTeam {
    sets: Vec<PokemonSet>,
}

fn community_rental_sets() -> &'static [PokemonSet] {
    static SETS: OnceLock<Vec<PokemonSet>> = OnceLock::new();
    SETS.get_or_init(|| {
        serde_json::from_str::<CommunityRentalDb>(COMMUNITY_RENTALS_JSON)
            .expect("embedded community rentals must parse")
            .teams
            .into_iter()
            .flat_map(|t| t.sets)
            .collect()
    })
}

fn format_learnsets() -> &'static Learnsets {
    static LEARNSETS: OnceLock<Learnsets> = OnceLock::new();
    LEARNSETS.get_or_init(|| {
        Learnsets::from_json(FORMAT_LEARNSETS_JSON)
            .expect("embedded format learnsets must parse")
    })
}

/// One pool team as an imputation source: reference mons aligned to the
/// opponent's roster slots (`None` = preview-inconsistent).
struct Candidate {
    id: String,
    sets: Vec<PokemonSet>,
    /// Constructed reference mons, index = opponent roster slot.
    refs: Option<Vec<Pokemon>>,
}

/// Hidden-set synthesis policy used after every meta-pool candidate has
/// been rejected. `Layered` is the shipped policy. `LegacyMetaOnly` exists
/// only as an evaluation control: it reproduces the old revealed -> pool ->
/// empty-set/implicit-Struggle path without changing the production default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FallbackPolicy {
    LegacyMetaOnly,
    #[default]
    Layered,
}

impl FallbackPolicy {
    pub const fn id(self) -> &'static str {
        match self {
            Self::LegacyMetaOnly => "legacy-meta-only-v1",
            Self::Layered => "layered-meta-rentals-learnset-v1",
        }
    }
}

/// The per-mon source actually selected by fallback synthesis at the
/// current observation. Exposed so evaluation artifacts can prove they
/// exercised the layer they claim to measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackSource {
    Meta,
    CommunityRental,
    Learnset,
    LegacyEmpty,
}

impl FallbackSource {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Meta => "meta",
            Self::CommunityRental => "community-rental",
            Self::Learnset => "learnset",
            Self::LegacyEmpty => "legacy-empty",
        }
    }
}

/// M18 item 2: one fallback-roster slot's per-determinization move draw.
///
/// Rebuilt with the fallback roster at every re-filter, so `revealed` tracks
/// the reveals as they accumulate and `pool` shrinks to match. Splitting the
/// two is what makes reveal-dominance structural rather than checked: the
/// revealed slots are copied in first and unconditionally, and the pool it
/// draws the remainder from cannot contain them.
#[derive(Clone, Debug)]
struct MoveDraw {
    /// Slots the reveals pin. Always kept, always first, never resampled.
    revealed: Vec<MoveSlot>,
    /// `(fresh slot, marginal weight)` over the species' prior moves that are
    /// format-legal at the observed level and not already revealed. Ordered
    /// by the prior table's (sorted) move ids, so the draw never depends on
    /// JSON key order.
    pool: Vec<(MoveSlot, f64)>,
}

impl MoveDraw {
    /// Revealed slots, then a **weighted draw without replacement** for the
    /// `k` still-open ones, on the caller's (per-iteration) rng — the rule
    /// the design doc specifies. Independent Bernoulli draws are wrong here:
    /// they do not yield exactly the k slots that are actually open. Drawing
    /// without replacement distorts the marginals, which is accepted: the
    /// distortion lives entirely in the guess about unrevealed slots, i.e.
    /// inside class B.
    fn draw(&self, rng: &mut SplitMix64) -> MoveSlots {
        let mut out = MoveSlots::default();
        for slot in self.revealed.iter().take(4) {
            out.push(*slot);
        }
        let mut pool = self.pool.clone();
        while out.len() < 4 && !pool.is_empty() {
            let total: f64 = pool.iter().map(|(_, w)| *w).sum();
            let pick = if total > 0.0 {
                let mut u = rng.next_f64() * total;
                let mut pick = pool.len() - 1;
                for (i, (_, w)) in pool.iter().enumerate() {
                    u -= *w;
                    if u < 0.0 {
                        pick = i;
                        break;
                    }
                }
                pick
            } else {
                // Unreachable through the interpreter (it drops non-positive
                // weights); total-by-construction rather than by argument.
                rng.below(pool.len())
            };
            out.push(pool.swap_remove(pick).0);
        }
        out
    }
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
    fallback_policy: FallbackPolicy,
    /// M18: the community belief prior, when the owner loaded one.
    prior: Option<Arc<BeliefPrior>>,
    /// M18: per-fallback-slot draw plans, rebuilt alongside `fallback`.
    /// `None` whenever no prior governs any slot — which is the shipped
    /// default and keeps `determinize` bit-identical to pre-M18, rng draws
    /// included.
    draws: Option<Vec<Option<MoveDraw>>>,
    synced: Option<u64>,
}

impl Belief {
    /// Build the candidate set and apply the preview filter. Call at team
    /// preview, right after `Observer::new`.
    pub fn new(dex: &Dex, pool: &MetaPool, obs: &Observer) -> Belief {
        Self::with_fallback_policy(dex, pool, obs, FallbackPolicy::Layered)
    }

    /// Evaluation surface for comparing fallback synthesis policies. Live
    /// callers should use `new`, which is pinned to the shipped policy.
    pub fn with_fallback_policy(
        dex: &Dex,
        pool: &MetaPool,
        obs: &Observer,
        fallback_policy: FallbackPolicy,
    ) -> Belief {
        let cands: Vec<Candidate> = pool
            .teams
            .iter()
            .map(|t| {
                let refs = build_refs(dex, &t.sets, obs.mons());
                Candidate { id: t.id.clone(), sets: t.sets.clone(), refs }
            })
            .collect();
        let mut b = Belief {
            cands,
            alive: Vec::new(),
            fallback: None,
            pinned: false,
            fallback_policy,
            prior: None,
            draws: None,
            synced: None,
        };
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
        Self::pinned_checked(dex, id, sets, obs)
            .expect("pinned opponent team must be legal and match the public preview")
    }

    /// Checked open-sheet construction for untrusted external team JSON.
    /// Unlike the pool-identification path, an invalid public sheet must
    /// never degrade into a synthesized fallback belief: doing so would
    /// silently evaluate a different information set.
    pub fn pinned_checked(
        dex: &Dex,
        id: &str,
        sets: &[PokemonSet],
        obs: &Observer,
    ) -> Result<Belief, String> {
        let team_json =
            serde_json::to_string(sets).map_err(|e| format!("serialize pinned team: {e}"))?;
        let verdict = validate_team(dex, format_learnsets(), &team_json);
        // The engine is generation-locked and intentionally has no
        // nature/ability semantics or cosmetic shiny flag (the shiny DVs
        // themselves are retained). Every other validator finding can
        // change a public field, stats, DVs, moves, or damage.
        let unsupported_finding = verdict["findings"].as_array().is_none_or(|findings| {
            findings.iter().any(|finding| {
                !matches!(
                    finding["code"].as_str(),
                    Some("ability-canonical" | "nature-canonical" | "dv-shiny")
                )
            })
        });
        if unsupported_finding {
            return Err(format!(
                "pinned opponent team is illegal or noncanonical: {verdict}"
            ));
        }
        let refs = build_refs(dex, sets, obs.mons())
            .ok_or_else(|| "pinned opponent team does not match the public preview".to_string())?;
        Ok(Belief {
            cands: vec![Candidate {
                id: id.to_string(),
                sets: sets.to_vec(),
                refs: Some(refs),
            }],
            alive: vec![0],
            fallback: None,
            pinned: true,
            fallback_policy: FallbackPolicy::Layered,
            prior: None,
            draws: None,
            synced: Some(obs.revision()),
        })
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
            fallback_policy: FallbackPolicy::Layered,
            prior: None,
            draws: None,
            synced: Some(obs.revision()),
        }
    }

    /// Re-filter after new observations. Cheap no-op at an unchanged
    /// observer revision, and always for a pinned belief holding its
    /// candidate (the pinned truth passes every filter by construction).
    pub fn sync(&mut self, dex: &Dex, obs: &Observer) {
        self.sync_checked(dex, obs)
            .expect("pinned opponent team contradicted public observations");
    }

    /// Checked synchronization for protocol callers. Pool beliefs retain
    /// their defensive fallback behavior; pinned sheets instead fail
    /// closed if later public evidence contradicts the submitted sheet.
    pub fn sync_checked(&mut self, dex: &Dex, obs: &Observer) -> Result<(), String> {
        if self.synced == Some(obs.revision()) {
            return Ok(());
        }
        if self.pinned && !self.alive.is_empty() {
            if !self.cands[0]
                .refs
                .as_deref()
                .is_some_and(|refs| consistent(dex, refs, obs.mons()))
            {
                return Err(
                    "pinned opponent team contradicts public battle observations".to_string()
                );
            }
            self.synced = Some(obs.revision());
            return Ok(());
        }
        self.refilter(dex, obs);
        Ok(())
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

    pub fn fallback_policy(&self) -> FallbackPolicy {
        self.fallback_policy
    }

    pub fn mode_id(&self) -> &'static str {
        if self.pinned { "pinned" } else { "blind" }
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    pub fn fallback_reason(&self) -> Option<&'static str> {
        if !self.is_fallback() {
            None
        } else if self.pinned {
            Some("pinned-preview-alignment-failed")
        } else if self.cands.is_empty() {
            Some("empty-candidate-pool")
        } else {
            Some("no-consistent-candidates")
        }
    }

    /// M18: install the community belief prior. It governs **only** the
    /// fallback (hidden custom team) imputation:
    ///
    /// - the sampler is reachable from exactly one branch of
    ///   `determinize_with` — `pick == None`, i.e. no pool candidate survived
    ///   — and a `pinned` / open-sheet belief always holds its single
    ///   candidate, so the shipped open-sheet product cannot be reached from
    ///   here even if a caller installs a prior on a pinned belief;
    /// - a species the table does not mention keeps today's deterministic
    ///   filler, per the design doc's precedence.
    ///
    /// Takes effect at the next `sync` (the draw plans are built with the
    /// fallback roster, which is the only place the reveals are known).
    pub fn set_prior(&mut self, prior: Arc<BeliefPrior>) {
        self.prior = Some(prior);
        self.draws = None;
        self.synced = None;
    }

    pub fn has_prior(&self) -> bool {
        self.prior.as_deref().is_some_and(|p| !p.is_empty())
    }

    /// Which fallback roster slots the installed prior currently governs, in
    /// observer roster order (`false` = today's deterministic filler). Empty
    /// when nothing is governed. A coverage surface for the M18 gate.
    pub fn prior_governed(&self) -> Vec<bool> {
        self.draws
            .as_deref()
            .map(|d| d.iter().map(Option::is_some).collect())
            .unwrap_or_default()
    }

    /// Per-mon synthesis sources in observer roster order. Meaningful when
    /// `is_fallback()`; callers may inspect it at preview before any reveal.
    pub fn fallback_sources(&self, dex: &Dex, obs: &Observer) -> Vec<FallbackSource> {
        obs.mons().iter().map(|mo| self.fallback_source(dex, mo)).collect()
    }

    pub fn candidate_id(&self, pool_idx: usize) -> &str {
        &self.cands[pool_idx].id
    }

    /// M15 synthesis source: the imputation reference mons aligned to the
    /// opponent's observed roster slots — `pick` as in `determinize_with`
    /// (`None` = fallback roster; call `sync` first so it exists). The
    /// protocol importer builds its from-scratch battle on these; the
    /// per-iteration `determinize` then overwrites them with fair samples.
    pub(crate) fn refs(&self, pick: Option<usize>) -> &[Pokemon] {
        match pick {
            Some(i) => self.cands[i]
                .refs
                .as_deref()
                .expect("refs: candidate is preview-inconsistent"),
            None => self
                .fallback
                .as_deref()
                .expect("refs: fallback roster not built (call sync first)"),
        }
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
        // M18: with a prior installed, the fallback roster's *unrevealed*
        // slots are resampled per determinization instead of being the one
        // fixed nearest-set filler. `None` on every other path — including
        // every pinned/open-sheet call — and `None` consumes no rng, so the
        // no-prior default stays bit-identical.
        let sampled = self.sample_fallback_refs(pick, rng);
        let refs: &[Pokemon] = match (sampled.as_deref(), pick) {
            (Some(r), _) => r,
            (None, Some(i)) => self.cands[i]
                .refs
                .as_deref()
                .expect("determinize_with: candidate is preview-inconsistent"),
            (None, None) => self
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
        // This turn's Quick Claw coin is chance the player never sees
        // (`turn.rs` endTurn rolls it before the request). Copying it through
        // was a leak in the arena and a lie on the ladder, where the importer
        // leaves it false in every reconstructed state — the search then
        // never models a foe preempt on the one turn it is deciding. Resample
        // it at the engine's own rate, drawn from the battle PRNG that was
        // just reseeded so the belief's own stream stays bit-identical.
        out.quick_claw_roll = out.prng.random_chance(60, 256);

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
                Some(refs) => consistent(dex, refs, obs.mons()),
                None => false,
            })
            .collect();
        if self.alive.is_empty() {
            let roster = self.build_fallback(dex, obs);
            self.draws = self.build_draws(dex, obs, &roster);
            self.fallback = Some(roster);
        } else {
            self.draws = None;
        }
        self.synced = Some(obs.revision());
    }

    // ------------------------------------------------- M18 marginal sampling

    /// Build the per-slot draw plans against the freshly-rebuilt fallback
    /// roster. `None` (the shipped default) when no prior is installed, the
    /// table is empty, or it says nothing about any species on this roster —
    /// in which case `determinize_with` never enters the sampling branch at
    /// all.
    fn build_draws(
        &self,
        dex: &Dex,
        obs: &Observer,
        roster: &[Pokemon],
    ) -> Option<Vec<Option<MoveDraw>>> {
        let prior = self.prior.as_deref()?;
        if prior.is_empty() || self.pinned {
            return None;
        }
        let mut governed = false;
        let draws: Vec<Option<MoveDraw>> = obs
            .mons()
            .iter()
            .zip(roster)
            .map(|(mo, mon)| {
                let draw = self.move_draw(dex, prior, mo, mon)?;
                governed = true;
                Some(draw)
            })
            .collect();
        governed.then_some(draws)
    }

    /// One roster slot's plan, or `None` to keep today's filler for it.
    ///
    /// The revealed prefix is taken from `mo.revealed_moves` rather than from
    /// the built roster mon: the roster's *defensive* second stage (an
    /// unconstructible set) drops the reveals, and reading the observation
    /// directly means the sampler restores reveal-dominance there instead of
    /// inheriting the hole. Existing slots are reused when present so the
    /// live PP marks survive.
    fn move_draw(
        &self,
        dex: &Dex,
        prior: &BeliefPrior,
        mo: &MonObs,
        mon: &Pokemon,
    ) -> Option<MoveDraw> {
        let species = dex.species.key(mo.species);
        let sp = prior.species(species)?;
        let mut revealed: Vec<MoveSlot> = Vec::new();
        for &m in &mo.revealed_moves {
            if revealed.len() >= 4 {
                break; // the <=4 clamp, same truncation as `fallback_set`
            }
            if revealed.iter().any(|s| move_matches(dex, s.id, m)) {
                continue;
            }
            revealed.push(
                mon.base_move_slots
                    .iter()
                    .find(|s| move_matches(dex, s.id, m))
                    .copied()
                    .unwrap_or_else(|| fresh_move_slot(dex, m)),
            );
        }
        if revealed.len() >= 4 {
            return None; // fully revealed: nothing left to sample
        }
        let mut pool: Vec<(MoveSlot, f64)> = Vec::new();
        for (key, weight) in &sp.moves {
            let Some(id) = dex.moves.id(key) else { continue };
            // Legality is code and outranks the data: an off-format or
            // level-illegal entry in a community table is simply not drawn.
            if !format_move_legal_at_level(species, dex.moves.key(id), mo.level) {
                continue;
            }
            if revealed.iter().any(|s| move_matches(dex, s.id, id))
                || mo.revealed_moves.iter().any(|&m| move_matches(dex, id, m))
                || pool.iter().any(|(s, _)| s.id == id)
            {
                continue;
            }
            pool.push((fresh_move_slot(dex, id), *weight));
        }
        if pool.is_empty() {
            // Nothing legal to offer. Falling through to today's filler is
            // strictly better than emitting a short set, and is what the
            // "species in neither file" rule prescribes anyway.
            return None;
        }
        Some(MoveDraw { revealed, pool })
    }

    /// The per-determinization fallback roster, or `None` to use the stored
    /// one unchanged. Consumes rng **only** when it actually samples.
    fn sample_fallback_refs(
        &self,
        pick: Option<usize>,
        rng: &mut SplitMix64,
    ) -> Option<Vec<Pokemon>> {
        if pick.is_some() || self.pinned {
            return None;
        }
        let draws = self.draws.as_deref()?;
        let base = self.fallback.as_deref()?;
        let mut roster = base.to_vec();
        for (mon, draw) in roster.iter_mut().zip(draws) {
            let Some(draw) = draw else { continue };
            let slots = draw.draw(rng);
            if slots.is_empty() {
                continue; // never regress to the empty-set/implicit-Struggle
            }
            mon.base_move_slots = slots;
            mon.move_slots = slots;
        }
        Some(roster)
    }

    // ----------------------------------------------------------- fallback

    /// Per-mon synthesized imputation roster for a non-pool opponent.
    fn build_fallback(&self, dex: &Dex, obs: &Observer) -> Vec<Pokemon> {
        let sets: Vec<PokemonSet> =
            obs.mons().iter().map(|mo| self.fallback_set(dex, mo)).collect();
        self.build_fallback_from_sets(dex, obs, &sets)
    }

    fn build_fallback_from_sets(
        &self,
        dex: &Dex,
        obs: &Observer,
        sets: &[PokemonSet],
    ) -> Vec<Pokemon> {
        match Battle::from_fixture(dex, "1,2,3,4", &sets, &sets) {
            Ok(b) => b.sides[0].roster.clone(),
            Err(_) => {
                // defensive second stage: minimal sets cannot fail (species /
                // level / gender come from the live battle, no parsed strings)
                let minimal: Vec<PokemonSet> = obs
                    .mons()
                    .iter()
                    .map(|mo| match self.fallback_policy {
                        FallbackPolicy::LegacyMetaOnly => legacy_base_set(dex, mo),
                        FallbackPolicy::Layered => base_set(dex, mo),
                    })
                    .collect();
                Battle::from_fixture(dex, "1,2,3,4", &minimal, &minimal)
                    .expect("minimal fallback set must construct")
                    .sides[0]
                    .roster
                    .clone()
            }
        }
    }

    fn fallback_source(&self, dex: &Dex, mo: &MonObs) -> FallbackSource {
        if self.cands.iter().any(|c| {
            c.sets
                .iter()
                .any(|s| dex.species.id(&toid(&s.species)) == Some(mo.species))
        }) {
            return FallbackSource::Meta;
        }
        match self.fallback_policy {
            FallbackPolicy::LegacyMetaOnly => FallbackSource::LegacyEmpty,
            FallbackPolicy::Layered
                if community_rental_sets()
                    .iter()
                    .any(|s| dex.species.id(&toid(&s.species)) == Some(mo.species)) =>
            {
                FallbackSource::CommunityRental
            }
            FallbackPolicy::Layered => FallbackSource::Learnset,
        }
    }

    fn fallback_set(&self, dex: &Dex, mo: &MonObs) -> PokemonSet {
        // nearest pool set by species (first in pedigree order)
        let pool_set: Option<&PokemonSet> = self.cands.iter().find_map(|c| {
            c.sets
                .iter()
                .find(|s| dex.species.id(&toid(&s.species)) == Some(mo.species))
        });
        // A separate prior, not a pool candidate: it can fill hidden fields
        // without pretending an exact rental-team identity survived the
        // preview filter. Pool pedigree remains first, preserving existing
        // same-species behavior.
        let source = self.fallback_source(dex, mo);
        let prior = pool_set.or_else(|| match source {
            FallbackSource::CommunityRental => community_rental_sets()
                .iter()
                .find(|s| dex.species.id(&toid(&s.species)) == Some(mo.species)),
            _ => None,
        });
        let mut set = prior.cloned().unwrap_or_else(|| match self.fallback_policy {
            // Deliberately empty: the engine safely enumerates implicit
            // Struggle, matching the pre-M17d control without a fake move.
            FallbackPolicy::LegacyMetaOnly => legacy_base_set(dex, mo),
            FallbackPolicy::Layered => base_set(dex, mo),
        });
        set.level = mo.level;
        set.gender = Some(match mo.gender.as_str() {
            "" => "N".to_string(),
            g => g.to_string(),
        });
        set.name = String::new();
        // moves: revealed first, empirical-prior filler after
        let mut moves: Vec<String> =
            mo.revealed_moves.iter().map(|&m| dex.moves.key(m).to_string()).collect();
        for name in prior.map(|s| s.moves.clone()).unwrap_or_default() {
            if moves.len() >= 4 {
                break;
            }
            let Some(id) = dex.moves.id(&toid(&name)) else { continue };
            if format_move_legal_at_level(
                dex.species.key(mo.species),
                dex.moves.key(id),
                mo.level,
            )
                && !mo.revealed_moves.iter().any(|&m| move_matches(dex, id, m))
                && !moves.iter().any(|m| dex.moves.id(&toid(m)) == Some(id))
            {
                moves.push(name);
            }
        }
        if moves.is_empty() && self.fallback_policy == FallbackPolicy::Layered {
            moves.push(legal_fallback_move(dex, mo));
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

fn legacy_base_set(dex: &Dex, mo: &MonObs) -> PokemonSet {
    let mut set = base_set(dex, mo);
    set.moves.clear();
    set
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
        moves: vec![legal_fallback_move(dex, mo)],
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

/// A full-PP move slot, matching what `Battle::from_fixture` builds for a
/// set's move (PS `calculatePP`: `pp * (5 + ppUps) / 5`, and gen ≤ 2 subtracts
/// the ups again at base 40). M18 sampling swaps slots after construction, so
/// this has to agree with the constructor exactly or an imputed set would
/// carry different PP than the same set built from JSON.
fn fresh_move_slot(dex: &Dex, id: MoveId) -> MoveSlot {
    let ms = dex.move_static(id);
    let pp_ups = if ms.no_pp_boosts { 0 } else { 3 };
    let mut pp = ms.pp * (5 + pp_ups) / 5;
    if ms.pp == 40 {
        pp -= pp_ups;
    }
    MoveSlot { id, pp, maxpp: pp, disabled: false, used: false, shared: true }
}

/// Empirical priors predate the no-OHKO format lens and can contain a move
/// that is illegal at the observed level. Filter every filler through the
/// same exported acceptance table as the validator.
fn format_move_legal_at_level(species_id: &str, move_id: &str, level: u8) -> bool {
    let Some(learnset) = format_learnsets().species(species_id) else { return false };
    let base = if move_id.starts_with("hiddenpower") { "hiddenpower" } else { move_id };
    learnset.allows(base)
        && learnset.move_min_level.get(base).is_none_or(|&min| level as i64 >= min)
}

/// Deterministic nonempty set for a format-legal species absent from every
/// empirical prior: the lexicographically first move accepted at its
/// observed level by the format's full learnset lens.
fn legal_fallback_move(dex: &Dex, mo: &MonObs) -> String {
    let species = dex.species.key(mo.species);
    format_learnsets()
        .species(species)
        .and_then(|l| {
            l.moves.iter().find(|m| {
                dex.moves.id(m).is_some()
                    && format_move_legal_at_level(species, m, mo.level)
            })
        })
        .cloned()
        // Defensive malformed/off-format input: still never reintroduce the
        // empty-set/implicit-Struggle failure this function closes.
        .unwrap_or_else(|| "return".to_string())
}

/// Construct a pool team's reference mons and align them to the opponent's
/// observed roster slots by (species, level, gender). `None` =
/// preview-inconsistent (public detail mismatch or unbuildable team).
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
        let k = slots.iter().position(|&i| {
            roster[i].level == mo.level
                && roster[i].gender == mo.gender
                && roster[i].item.is_some() == mo.preview_has_item
        })?;
        out.push(roster[slots.remove(k)].clone());
    }
    Some(out)
}

/// In-battle consistency: every observation must fit the candidate.
/// (`move_matches`: a plain-hiddenpower reveal — M15 protocol mode —
/// matches any typed hidden power; M10 reveals are typed, bit-identical.)
fn consistent(dex: &Dex, refs: &[Pokemon], mons: &[MonObs]) -> bool {
    refs.iter().zip(mons).all(|(r, mo)| {
        mo.revealed_moves
            .iter()
            .all(|m| r.base_move_slots.iter().any(|s| move_matches(dex, s.id, *m)))
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
        if mo.revealed_moves.iter().any(|m| move_matches(dex, rs.id, *m)) {
            match old_base.iter().find(|s| s.id == rs.id) {
                Some(os) => nb.push(MoveSlot { shared: true, ..*os }),
                None => nb.push(*rs), // fallback-merge oddity: fresh slot
            }
        } else {
            nb.push(*rs); // unrevealed: candidate's move at full PP
        }
    }
    debug_assert!(
        mo.revealed_moves
            .iter()
            .all(|m| nb.iter().any(|s| move_matches(dex, s.id, *m))),
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
        quick_claw_roll: _,    // hidden per-turn coin → resampled by determinize
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

#[cfg(test)]
mod fallback_tests {
    use super::*;
    use crate::observe::ItemObs;
    use nc2000_engine::state::Gender;
    use serde_json::Value;

    const DEX_JSON: &str = include_str!("../../../data/gen2stadium2.json");
    const LEARNSETS_JSON: &str = include_str!("../../../data/learnsets-gen2.json");
    const META_POOL_JSON: &str = include_str!("../../../data/meta-pool-v0/meta-pool.json");

    fn test_dex() -> Dex {
        Dex::from_json(DEX_JSON).unwrap()
    }

    fn test_belief() -> Belief {
        let pool: MetaPool = serde_json::from_str(META_POOL_JSON).unwrap();
        Belief {
            cands: pool
                .teams
                .into_iter()
                .map(|t| Candidate { id: t.id, sets: t.sets, refs: None })
                .collect(),
            alive: Vec::new(),
            fallback: None,
            pinned: false,
            fallback_policy: FallbackPolicy::Layered,
            prior: None,
            draws: None,
            synced: None,
        }
    }

    fn mon(dex: &Dex, species: &str, level: u8, revealed: &[&str]) -> MonObs {
        MonObs {
            species: dex.species.id(&toid(species)).unwrap(),
            level,
            gender: Gender::N,
            name: species.to_string(),
            preview_has_item: false,
            revealed_moves: revealed.iter().map(|m| dex.moves.id(m).unwrap()).collect(),
            item: ItemObs::default(),
            appeared: false,
        }
    }

    fn base_move_id(name: &str) -> String {
        let id = toid(name);
        if id.starts_with("hiddenpower") {
            "hiddenpower".to_string()
        } else {
            id
        }
    }

    #[test]
    fn every_format_species_gets_one_to_four_legal_moves_at_zero_reveal() {
        let dex = test_dex();
        let belief = test_belief();
        let table: Value = serde_json::from_str(LEARNSETS_JSON).unwrap();
        let species = table["species"].as_object().unwrap();

        for (sid, entry) in species {
            let legal: Vec<&str> =
                entry["moves"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
            let min_level = entry["minLevel"].as_u64().unwrap_or(50).max(50) as u8;
            for level in min_level..=55 {
                let set = belief.fallback_set(&dex, &mon(&dex, sid, level, &[]));
                assert!(
                    (1..=4).contains(&set.moves.len()),
                    "{sid} L{level}: fallback move count {} ({:?})",
                    set.moves.len(),
                    set.moves
                );
                for mv in &set.moves {
                    let mid = base_move_id(mv);
                    assert!(
                        legal.contains(&mid.as_str()),
                        "{sid} L{level}: illegal fallback move {mv}"
                    );
                    let floor = entry["moveMinLevel"][&mid].as_u64().unwrap_or(50) as u8;
                    assert!(
                        level >= floor,
                        "{sid}: {mv} requires level {floor}, got {level}"
                    );
                }
            }
        }
    }

    #[test]
    fn fallback_priority_is_reveal_then_pool_then_rental_then_legal_default() {
        let dex = test_dex();
        let belief = test_belief();

        // Existing same-species pool behavior stays first for every pool
        // species, even when the rental DB carries another set.
        let mut seen = std::collections::HashSet::new();
        for source in belief.cands.iter().flat_map(|c| &c.sets) {
            let sid = toid(&source.species);
            if seen.insert(sid.clone()) {
                let fallback = belief.fallback_set(
                    &dex,
                    &mon(&dex, &sid, source.level, &[]),
                );
                assert_eq!(fallback.moves, source.moves, "pool source changed for {sid}");
            }
        }

        // Chansey is absent from the meta pool but present in the community
        // rentals, so its first rental set supplies the filler.
        assert!(belief
            .cands
            .iter()
            .flat_map(|c| &c.sets)
            .all(|s| toid(&s.species) != "chansey"));
        let rental_chansey = community_rental_sets()
            .iter()
            .find(|s| toid(&s.species) == "chansey")
            .unwrap();
        let chansey = belief.fallback_set(&dex, &mon(&dex, "chansey", 50, &[]));
        assert_eq!(chansey.moves, rental_chansey.moves);
        assert_eq!(
            belief.fallback_source(&dex, &mon(&dex, "chansey", 50, &[])),
            FallbackSource::CommunityRental
        );
        let revealed_chansey =
            belief.fallback_set(&dex, &mon(&dex, "chansey", 50, &["toxic"]));
        assert_eq!(toid(&revealed_chansey.moves[0]), "toxic");
        assert_eq!(revealed_chansey.moves.len(), 4);

        // Reveals lead and survive even when neither empirical source has
        // the species. Zero reveal gets the legal deterministic default.
        assert!(belief
            .cands
            .iter()
            .flat_map(|c| &c.sets)
            .chain(community_rental_sets())
            .all(|s| toid(&s.species) != "bulbasaur"));
        let revealed = belief.fallback_set(&dex, &mon(&dex, "bulbasaur", 50, &["tackle"]));
        assert_eq!(revealed.moves, ["tackle"]);
        let zero = belief.fallback_set(&dex, &mon(&dex, "bulbasaur", 50, &[]));
        assert_eq!(zero.moves, ["ancientpower"]);
        assert_eq!(
            belief.fallback_source(&dex, &mon(&dex, "bulbasaur", 50, &[])),
            FallbackSource::Learnset
        );
    }

    #[test]
    fn legacy_control_reproduces_meta_only_then_implicit_struggle() {
        let dex = test_dex();
        let mut belief = test_belief();
        belief.fallback_policy = FallbackPolicy::LegacyMetaOnly;

        let source = belief.cands.iter().flat_map(|c| &c.sets).next().unwrap();
        let from_pool =
            belief.fallback_set(&dex, &mon(&dex, &source.species, source.level, &[]));
        assert_eq!(from_pool.moves, source.moves);

        let zero = belief.fallback_set(&dex, &mon(&dex, "chansey", 50, &[]));
        assert!(zero.moves.is_empty());
        assert_eq!(
            belief.fallback_source(&dex, &mon(&dex, "chansey", 50, &[])),
            FallbackSource::LegacyEmpty
        );
        let revealed = belief.fallback_set(&dex, &mon(&dex, "chansey", 50, &["toxic"]));
        assert_eq!(revealed.moves, ["toxic"]);
        let z = vec![zero; 6];
        let r = vec![revealed; 6];
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &z, &r).unwrap();
        assert!(battle.sides[0].roster[0].base_move_slots.is_empty());
        let obs = Observer::new(&battle, 0);
        let mut invalid = z.clone();
        invalid[0].species = "not-a-species".to_string();
        let legacy_defensive = belief.build_fallback_from_sets(&dex, &obs, &invalid);
        assert!(legacy_defensive.iter().all(|p| p.base_move_slots.is_empty()));
        belief.fallback_policy = FallbackPolicy::Layered;
        let layered_defensive = belief.build_fallback_from_sets(&dex, &obs, &invalid);
        assert!(layered_defensive.iter().all(|p| !p.base_move_slots.is_empty()));
        let p0 = battle.legal_choices(&dex, 0)[0];
        let p1 = battle.legal_choices(&dex, 1)[0];
        battle.apply_choices(&dex, [Some(p0), Some(p1)]).unwrap();
        let moves: Vec<_> = battle
            .legal_choices(&dex, 0)
            .into_iter()
            .filter(|c| matches!(c, nc2000_engine::battle::SearchChoice::Move(_)))
            .collect();
        assert_eq!(
            moves,
            [nc2000_engine::battle::SearchChoice::Move(
                dex.moves.id("struggle").unwrap()
            )]
        );
    }

    #[test]
    fn pinned_sheet_rejects_illegal_and_preview_mismatched_teams() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(META_POOL_JSON).unwrap();
        let truth = &pool.teams[0].sets;
        let battle = Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[1].sets, truth).unwrap();
        let obs = Observer::new(&battle, 0);

        let mut wrong_level = truth.clone();
        wrong_level[0].level = if wrong_level[0].level == 50 { 51 } else { 50 };
        assert!(
            Belief::pinned_checked(&dex, "wrong-level", &wrong_level, &obs)
                .err()
                .unwrap()
                .contains("public preview")
        );

        let mut wrong_item = truth.clone();
        let item_slot = wrong_item
            .iter()
            .position(|set| !set.item.is_empty())
            .expect("fixture team should contain an item");
        wrong_item[item_slot].item.clear();
        assert!(
            Belief::pinned_checked(&dex, "wrong-item", &wrong_item, &obs)
                .err()
                .unwrap()
                .contains("public preview")
        );

        let mut illegal = truth.clone();
        illegal[0].moves = vec!["Fissure".to_string()];
        assert!(
            Belief::pinned_checked(&dex, "illegal", &illegal, &obs)
                .err()
                .unwrap()
                .contains("illegal")
        );

        let mut noncanonical = truth.clone();
        noncanonical[0].evs = None;
        assert!(
            Belief::pinned_checked(&dex, "noncanonical", &noncanonical, &obs)
                .err()
                .unwrap()
                .contains("noncanonical")
        );

        let mut wrong_gender = truth.clone();
        let gender = wrong_gender[0]
            .gender
            .as_deref()
            .expect("fixture set should have a canonical gender");
        wrong_gender[0].gender = Some(if gender == "M" { "F" } else { "M" }.to_string());
        assert!(
            build_refs(&dex, &wrong_gender, obs.mons()).is_none(),
            "preview-public gender mismatch must prevent pinned alignment"
        );

        let mut shiny = truth.clone();
        let shiny_slot = shiny
            .iter()
            .position(|set| {
                set.gender.as_deref() == Some("M")
                    && set.moves.iter().all(|mv| !toid(mv).starts_with("hiddenpower"))
            })
            .expect("fixture team should contain a male set without Hidden Power");
        shiny[shiny_slot].ivs = Some(
            [
                ("hp", 16),
                ("atk", 31),
                ("def", 21),
                ("spa", 21),
                ("spd", 21),
                ("spe", 21),
            ]
            .into_iter()
            .map(|(stat, value)| (stat.to_string(), value))
            .collect(),
        );
        let shiny_result = Belief::pinned_checked(&dex, "shiny-dvs", &shiny, &obs);
        assert!(
            shiny_result.is_ok(),
            "the engine retains shiny DVs even though PokemonSet omits the cosmetic shiny flag: {:?}",
            shiny_result.err()
        );
    }

    #[test]
    fn pinned_sheet_rejects_a_later_move_contradiction() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(META_POOL_JSON).unwrap();
        let signature = |sets: &[PokemonSet]| {
            let mut values: Vec<_> =
                sets.iter().map(|set| (toid(&set.species), set.level)).collect();
            values.sort();
            values
        };
        let (truth, sheet) = pool
            .teams
            .iter()
            .enumerate()
            .find_map(|(i, left)| {
                pool.teams[i + 1..]
                    .iter()
                    .find(|right| signature(&left.sets) == signature(&right.sets))
                    .map(|right| (&left.sets, &right.sets))
            })
            .expect("meta pool should retain the documented preview collision");
        let battle =
            Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[0].sets, truth).unwrap();
        let mut obs = Observer::new(&battle, 0);
        let mut belief = Belief::pinned_checked(&dex, "collision-sheet", sheet, &obs).unwrap();

        let (true_set, revealed) = truth
            .iter()
            .find_map(|true_set| {
                let sheet_set = sheet
                    .iter()
                    .find(|set| toid(&set.species) == toid(&true_set.species))?;
                true_set.moves.iter().find_map(|mv| {
                    (!sheet_set.moves.iter().any(|other| toid(other) == toid(mv)))
                        .then_some((true_set, mv))
                })
            })
            .expect("collision teams should differ by at least one move");
        obs.ingest_line(
            &format!("|move|p2a: {}|{}|p1a: Target", true_set.name, revealed),
            &dex,
        );
        assert!(
            belief
                .sync_checked(&dex, &obs)
                .unwrap_err()
                .contains("contradicts public battle observations")
        );
    }

    fn plain_set(species: &str) -> PokemonSet {
        PokemonSet {
            name: species.to_string(),
            species: species.to_string(),
            item: String::new(),
            ability: "No Ability".to_string(),
            moves: vec!["Return".to_string()],
            level: 50,
            evs: None,
            ivs: None,
            happiness: None,
            gender: Some("M".to_string()),
        }
    }

    fn determinized_fallback_moves(seed: u64) -> Vec<Vec<u16>> {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(META_POOL_JSON).unwrap();
        let custom: Vec<PokemonSet> = [
            "Bulbasaur",
            "Ivysaur",
            "Venusaur",
            "Charmander",
            "Charmeleon",
            "Charizard",
        ]
        .into_iter()
        .map(plain_set)
        .collect();
        let battle = Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[0].sets, &custom)
            .unwrap();
        let obs = Observer::new(&battle, 0);
        let belief = Belief::new(&dex, &pool, &obs);
        assert!(belief.is_fallback());
        let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(seed));
        det.sides[1]
            .roster
            .iter()
            .map(|p| p.base_move_slots.iter().map(|m| m.id.0).collect())
            .collect()
    }

    #[test]
    fn fallback_moves_are_seed_and_thread_deterministic() {
        let expected = determinized_fallback_moves(1);
        assert_eq!(expected, determinized_fallback_moves(u64::MAX));
        let jobs: Vec<_> = (0..4)
            .map(|i| std::thread::spawn(move || determinized_fallback_moves(10_000 + i)))
            .collect();
        for job in jobs {
            assert_eq!(expected, job.join().unwrap());
        }
    }

    // ------------------------------------------------- M18 marginal sampling

    /// A custom (off-pool) opponent, so the belief lands in fallback mode.
    /// Bulbasaur reveals `tackle`; every other slot is untouched.
    fn m18_fixture(reveal: bool) -> (Dex, MetaPool, Battle, Observer) {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(META_POOL_JSON).unwrap();
        let custom: Vec<PokemonSet> = [
            "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard",
        ]
        .into_iter()
        .map(plain_set)
        .collect();
        let battle =
            Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[0].sets, &custom).unwrap();
        let mut obs = Observer::new(&battle, 0);
        if reveal {
            obs.ingest_line("|move|p2a: Bulbasaur|Tackle|p1a: Target", &dex);
        }
        (dex, pool, battle, obs)
    }

    /// Bulbasaur is in neither empirical source, so the shipped filler gives
    /// it exactly one deterministic move — a clean baseline to sample against.
    const M18_PRIOR: &str = r#"{
        "format": "nc2000-belief-prior", "version": 1,
        "species": {"bulbasaur": {"moves": {
            "razorleaf": 0.9, "sleeppowder": 0.7, "swordsdance": 0.5,
            "bodyslam": 0.4, "synthesis": 0.3, "fissure": 1.0}}}}"#;

    fn m18_belief(dex: &Dex, pool: &MetaPool, obs: &Observer) -> Belief {
        let mut belief = Belief::new(dex, pool, obs);
        assert!(belief.is_fallback());
        belief.set_prior(Arc::new(BeliefPrior::from_json(M18_PRIOR)));
        belief.sync(dex, obs);
        belief
    }

    /// Bulbasaur's imputed move ids over `n` determinizations.
    fn m18_draws(
        dex: &Dex,
        belief: &Belief,
        battle: &Battle,
        obs: &Observer,
        n: u64,
    ) -> Vec<Vec<String>> {
        (0..n)
            .map(|seed| {
                let det =
                    belief.determinize(dex, battle, obs, &mut SplitMix64::new(seed * 7 + 1));
                det.sides[1].roster[0]
                    .base_move_slots
                    .iter()
                    .map(|s| dex.moves.key(s.id).to_string())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn the_prior_samples_only_legal_unrevealed_moves_and_reveals_always_survive() {
        let (dex, pool, battle, obs) = m18_fixture(true);
        let belief = m18_belief(&dex, &pool, &obs);
        assert_eq!(
            belief.prior_governed(),
            [true, false, false, false, false, false],
            "only the species named by the table is governed"
        );

        let mut distinct = std::collections::HashSet::new();
        for moves in m18_draws(&dex, &belief, &battle, &obs, 200) {
            assert_eq!(moves[0], "tackle", "the revealed move leads, always");
            assert_eq!(moves.len(), 4, "exactly the four open slots are filled");
            let unique: std::collections::HashSet<&String> = moves.iter().collect();
            assert_eq!(unique.len(), 4, "without replacement: {moves:?}");
            for mv in &moves[1..] {
                assert!(
                    ["razorleaf", "sleeppowder", "swordsdance", "bodyslam", "synthesis"]
                        .contains(&mv.as_str()),
                    "sampled {mv} outside the prior's legal support"
                );
            }
            distinct.insert(moves.clone());
        }
        // Fissure is in the table at p=1.0 and is banned by the no-OHKO
        // format: legality is code and outranks the data.
        assert!(distinct.len() > 1, "the draw is deterministic, not sampled");
    }

    #[test]
    fn heavier_marginals_are_drawn_more_often() {
        let (dex, pool, battle, obs) = m18_fixture(false);
        let belief = m18_belief(&dex, &pool, &obs);
        let draws = m18_draws(&dex, &belief, &battle, &obs, 2000);
        let rate = |key: &str| {
            draws.iter().filter(|m| m.iter().any(|x| x == key)).count() as f64
                / draws.len() as f64
        };
        // 4 of the 5 legal candidates are taken every time, so the question
        // is only which one is left out — and that must track the weights.
        assert!(
            rate("razorleaf") > rate("synthesis") + 0.05,
            "0.9 drawn {:.3}, 0.3 drawn {:.3}",
            rate("razorleaf"),
            rate("synthesis")
        );
    }

    #[test]
    fn without_a_prior_the_determinization_is_bit_identical() {
        let (dex, pool, battle, obs) = m18_fixture(true);
        let plain = Belief::new(&dex, &pool, &obs);
        // An empty / unparseable table must be indistinguishable from no file
        // at all: same imputed sets AND the same rng consumption.
        for text in ["", M18_PRIOR] {
            let mut belief = Belief::new(&dex, &pool, &obs);
            if text.is_empty() {
                belief.set_prior(Arc::new(BeliefPrior::from_json(text)));
                belief.sync(&dex, &obs);
            }
            let mut a = SplitMix64::new(9);
            let mut b = SplitMix64::new(9);
            let want = plain.determinize(&dex, &battle, &obs, &mut a);
            let got = belief.determinize(&dex, &battle, &obs, &mut b);
            let sets = |x: &Battle| -> Vec<Vec<u16>> {
                x.sides[1]
                    .roster
                    .iter()
                    .map(|p| p.base_move_slots.iter().map(|m| m.id.0).collect())
                    .collect()
            };
            if text.is_empty() {
                assert_eq!(sets(&want), sets(&got));
                assert_eq!(a.0, b.0, "an empty table must consume no rng");
            } else {
                // Sanity that the comparison above can fail at all.
                let mut c = SplitMix64::new(9);
                let live = m18_belief(&dex, &pool, &obs).determinize(&dex, &battle, &obs, &mut c);
                assert_ne!(sets(&want)[0], sets(&live)[0]);
            }
        }
    }

    #[test]
    fn a_pinned_open_sheet_belief_never_samples() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(META_POOL_JSON).unwrap();
        let truth = &pool.teams[0].sets;
        let battle =
            Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[1].sets, truth).unwrap();
        let obs = Observer::new(&battle, 0);
        let mut pinned = Belief::pinned_from_battle(&battle, &obs);
        let mut a = SplitMix64::new(5);
        let want = pinned.determinize(&dex, &battle, &obs, &mut a);

        // A table naming every species on the sheet, installed on the shipped
        // product's belief. It must be inert: `pick` is never `None` here.
        let sheet: String = truth
            .iter()
            .map(|s| format!("\"{}\":{{\"moves\":{{\"splash\":1.0}}}}", toid(&s.species)))
            .collect::<Vec<_>>()
            .join(",");
        pinned.set_prior(Arc::new(BeliefPrior::from_json(&format!(
            "{{\"species\":{{{sheet}}}}}"
        ))));
        pinned.sync(&dex, &obs);
        assert!(pinned.prior_governed().is_empty());
        let mut b = SplitMix64::new(5);
        let got = pinned.determinize(&dex, &battle, &obs, &mut b);
        for slot in 0..truth.len() {
            assert_eq!(
                want.sides[1].roster[slot].base_move_slots,
                got.sides[1].roster[slot].base_move_slots
            );
        }
        assert_eq!(a.0, b.0, "the pinned path must consume no extra rng");
    }

    #[test]
    fn a_fully_revealed_mon_is_left_alone() {
        let (dex, pool, battle, mut obs) = m18_fixture(true);
        for mv in ["Vine Whip", "Growl", "Leech Seed"] {
            obs.ingest_line(&format!("|move|p2a: Bulbasaur|{mv}|p1a: Target"), &dex);
        }
        let belief = m18_belief(&dex, &pool, &obs);
        assert!(
            belief.prior_governed().is_empty(),
            "four reveals leave no slot to sample, so the sampler is skipped whole"
        );
        for moves in m18_draws(&dex, &belief, &battle, &obs, 8) {
            assert_eq!(moves.len(), 4);
            for mv in ["tackle", "vinewhip", "growl", "leechseed"] {
                assert!(moves.iter().any(|m| m == mv), "{mv} lost from {moves:?}");
            }
        }
    }

    #[test]
    fn sampled_slots_carry_the_same_pp_as_a_set_built_from_json() {
        let (dex, pool, battle, obs) = m18_fixture(false);
        let belief = m18_belief(&dex, &pool, &obs);
        let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(3));
        let sampled = det.sides[1].roster[0].base_move_slots;
        let names: Vec<String> =
            sampled.iter().map(|s| dex.moves.key(s.id).to_string()).collect();
        let mut set = plain_set("Bulbasaur");
        set.moves = names;
        let team = vec![set; 6];
        let built = Battle::from_fixture(&dex, "1,2,3,4", &team, &team).unwrap();
        assert_eq!(built.sides[0].roster[0].base_move_slots, sampled);
    }
}
