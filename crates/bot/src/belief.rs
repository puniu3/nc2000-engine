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
//! Weights are uniform over the consistent candidates, unless the pool is a
//! **weighted prior** (any team carries `weight`): then a consistent
//! candidate is sampled in proportion to its weight (a team without one
//! counts 1).
//!
//! **Fallback** (no pool team consistent — a human custom team): a per-mon
//! imputation roster is synthesized instead — nearest pool set by species
//! (first in pedigree order; in a weighted prior, the loadout — item plus
//! move set — with the largest total weight among the pool teams carrying
//! the species, ties to the earliest), then the community-rental prior when the
//! species is absent from the pool, merged with the revealed knowledge
//! (revealed moves first, prior filler after, observed level/gender,
//! revealed item). A species absent from both sources receives a
//! deterministic format-legal move, never an empty set/implicit Struggle.
//! Marked by `is_fallback()`. The construction is defensive at every step —
//! a filter dead-end mid-game degrades, never panics.
//!
//! **M18 — community marginal prior** (`set_prior`, opt-in). With a
//! `prior::BeliefPrior` installed, slots without a loadout draw use a
//! *weighted draw without replacement* over the species'
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
//!   mons occupy the unseen party slots is resampled from the unappeared
//!   roster subject to the public total-level cap, uniformly unless a selection
//!   prior is installed and conditioned on the current observation revision;
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
use nc2000_engine::dex::{toid, Dex, ItemId, MoveId};
use nc2000_engine::rules::{LearnsetPolicy, Ruleset};
use nc2000_engine::state::{
    ActionKind, Battle, EffId, EffectState, MoveSlot, MoveSlots, PokeId, Pokemon,
};
use nc2000_engine::validate::{validate_team_with_ruleset, CastleRules, Learnsets};

use crate::observe::{move_matches, MonObs, Observer};
use crate::preview::MetaPool;
use crate::pick_prior::{PickDistribution, PickPrior, PreviewMon};
use crate::prior::{BeliefPrior, SpeciesPrior};
use crate::rng::SplitMix64;

/// Training-normalized no-OHKO community rentals, baked into the bot so the live
/// protocol/WASM paths get the prior without runtime filesystem plumbing.
/// Parsing is paid once per process; source order is the deterministic
/// pedigree order.
const COMMUNITY_RENTALS_JSON: &str =
    include_str!("../../../data/training-v3/rentals.json");
const SUPPLEMENTAL_PRIOR_JSON: &str = include_str!("../../../data/training-v3/supplemental.json");
const FORMAT_LEARNSETS_JSON: &str = include_str!("../../../data/learnsets-gen2.json");

#[derive(serde::Deserialize)]
struct CommunityRentalDb {
    teams: Vec<CommunityRentalTeam>,
}

#[derive(serde::Deserialize)]
struct CommunityRentalTeam {
    #[serde(default)]
    archetype: String,
    sets: Vec<PokemonSet>,
}

#[derive(serde::Deserialize)]
struct SupplementalPriorDb {
    teams: Vec<SupplementalPriorTeam>,
}

#[derive(Clone, serde::Deserialize)]
struct SupplementalPriorTeam {
    #[allow(dead_code)]
    id: String,
    weight: f64,
    sets: Vec<PokemonSet>,
}

/// Does the entry's own name say it belongs to a different rule set? The
/// source tags those with 「専用」("exclusive to"): a Little Cup team and a
/// 指振り (Metronome-only) team. Neither is illegal *as a team*, so no
/// validator can see them, and both are worthless as evidence about this
/// regulation — the Metronome entry alone would teach the bot that six
/// species carry exactly one move.
///
/// The author's third variant tag, 【一撃あり用】, is deliberately NOT
/// matched here: those teams are excluded by the OHKO Clause, which is the
/// right reason, and one that would stop applying by itself if the played
/// format ever allowed OHKO again.
pub(crate) fn names_another_ruleset(archetype: &str) -> bool {
    archetype.contains("専用")
}

/// The imputation pool, which is **not** the whole crawl. The file is an
/// archive of everything the source published, and eight of its 28 entries
/// are not legal teams in the played format
/// (`data/community-rentals-v0/ASSESSMENT.md` names them): a Little Cup team
/// at L5, a links page dressed as a one-mon entry, the author's three
/// explicitly-tagged OHKO variants, an Item Clause violation, and two
/// five-mon drafts. A team the format rejects is not evidence about the
/// format, so only teams `validate_team` accepts reach the pool.
///
/// A ninth entry is legal but belongs to another rule set entirely; see
/// [`names_another_ruleset`].
///
/// This matters more than the 8/28 ratio suggests: the rental layer only
/// ever fires for species the meta pool does not carry, and before this
/// filter eight of the ten species it served came from those rejected
/// entries — an opponent Chansey was modelled on a Little Cup set and an
/// opponent Dugtrio on a Fissure set. Dropping them costs those species
/// their rental prior; they fall through to the learnset layer, which is
/// what that layer is for.
///
/// The verdicts are cached with everything else here, so they are taken
/// under the first caller's `dex`. That is not a hidden dependency: the
/// format's dex is a build constant (`data/gen2stadium2.json`), the same
/// file every `Dex` in the process is built from.
fn community_rental_teams_for(dex: &Dex, ruleset: Ruleset) -> &'static [Vec<PokemonSet>] {
    if ruleset.learnset_policy() == LearnsetPolicy::CastleJpEvents {
        static CASTLE_TEAMS: OnceLock<Vec<Vec<PokemonSet>>> = OnceLock::new();
        return CASTLE_TEAMS.get_or_init(|| {
            serde_json::from_str::<CommunityRentalDb>(COMMUNITY_RENTALS_JSON)
                .expect("embedded community rentals must parse")
                .teams
                .into_iter()
                .filter(|t| !names_another_ruleset(&t.archetype))
                .filter(|t| team_is_format_legal(dex, &t.sets, Ruleset::castle_free()))
                .map(|t| t.sets)
                .collect()
        });
    }
    static PS_TEAMS: OnceLock<Vec<Vec<PokemonSet>>> = OnceLock::new();
    PS_TEAMS.get_or_init(|| {
        serde_json::from_str::<CommunityRentalDb>(COMMUNITY_RENTALS_JSON)
            .expect("embedded community rentals must parse")
            .teams
            .into_iter()
            .filter(|t| !names_another_ruleset(&t.archetype))
            .filter(|t| team_is_format_legal(dex, &t.sets, Ruleset::PsStadium2))
            .map(|t| t.sets)
            .collect()
    })
}

#[cfg(test)]
fn community_rental_teams(dex: &Dex) -> &'static [Vec<PokemonSet>] {
    community_rental_teams_for(dex, Ruleset::PsStadium2)
}

fn community_rental_sets_for(dex: &Dex, ruleset: Ruleset) -> &'static [PokemonSet] {
    if ruleset.learnset_policy() == LearnsetPolicy::CastleJpEvents {
        static CASTLE_SETS: OnceLock<Vec<PokemonSet>> = OnceLock::new();
        return CASTLE_SETS.get_or_init(|| {
            community_rental_teams_for(dex, Ruleset::castle_free())
                .iter()
                .flat_map(|sets| sets.iter().cloned())
                .collect()
        });
    }
    static PS_SETS: OnceLock<Vec<PokemonSet>> = OnceLock::new();
    PS_SETS.get_or_init(|| {
        community_rental_teams_for(dex, Ruleset::PsStadium2)
            .iter()
            .flat_map(|sets| sets.iter().cloned())
            .collect()
    })
}

#[cfg(test)]
fn community_rental_sets(dex: &Dex) -> &'static [PokemonSet] {
    community_rental_sets_for(dex, Ruleset::PsStadium2)
}

fn parse_raw_supplemental_teams() -> Vec<SupplementalPriorTeam> {
    let default_evs: std::collections::BTreeMap<String, u16> =
        ["hp", "atk", "def", "spa", "spd", "spe"]
            .into_iter()
            .map(|k| (k.to_string(), 255u16))
            .collect();
    let default_ivs: std::collections::BTreeMap<String, u8> =
        ["hp", "atk", "def", "spa", "spd", "spe"]
            .into_iter()
            .map(|k| (k.to_string(), 30u8))
            .collect();
    serde_json::from_str::<SupplementalPriorDb>(SUPPLEMENTAL_PRIOR_JSON)
        .expect("embedded supplemental prior must parse")
        .teams
        .into_iter()
        .map(|mut t| {
            for s in &mut t.sets {
                if s.name.is_empty() {
                    s.name = s.species.clone();
                }
                if s.ability.is_empty() {
                    s.ability = "No Ability".to_string();
                }
                if s.evs.is_none() {
                    s.evs = Some(default_evs.clone());
                }
                if s.ivs.is_none() {
                    s.ivs = Some(default_ivs.clone());
                }
                if s.happiness.is_none() {
                    s.happiness = Some(255);
                }
            }
            t
        })
        .collect()
}

fn supplemental_prior_teams_for(dex: &Dex, ruleset: Ruleset) -> &'static [SupplementalPriorTeam] {
    if ruleset.learnset_policy() == LearnsetPolicy::CastleJpEvents {
        static CASTLE_TEAMS: OnceLock<Vec<SupplementalPriorTeam>> = OnceLock::new();
        return CASTLE_TEAMS.get_or_init(|| {
            parse_raw_supplemental_teams()
                .into_iter()
                .filter(|t| {
                    t.weight.is_finite()
                        && t.weight > 0.0
                        && team_is_format_legal(dex, &t.sets, Ruleset::castle_free())
                })
                .collect()
        });
    }
    static PS_TEAMS: OnceLock<Vec<SupplementalPriorTeam>> = OnceLock::new();
    PS_TEAMS.get_or_init(|| {
        parse_raw_supplemental_teams()
            .into_iter()
            .filter(|t| {
                t.weight.is_finite()
                    && t.weight > 0.0
                    && team_is_format_legal(dex, &t.sets, Ruleset::PsStadium2)
            })
            .collect()
    })
}

/// Number of embedded `supplemental.json` teams legal under `ruleset`.
pub fn supplemental_prior_count(dex: &Dex, ruleset: Ruleset) -> usize {
    supplemental_prior_teams_for(dex, ruleset).len()
}

#[cfg(test)]
fn supplemental_prior_teams(dex: &Dex) -> &'static [SupplementalPriorTeam] {
    supplemental_prior_teams_for(dex, Ruleset::PsStadium2)
}

/// True iff `verdict` carries at least one `severity == "error"` finding.
/// Informational `"warning"` (e.g. `ev-effort-unverified`, `item-castle-inert`)
/// and `"fix"` findings never make a team illegal.
fn verdict_has_errors(verdict: &serde_json::Value, ignore_team_size: bool) -> bool {
    verdict["findings"].as_array().is_none_or(|findings| {
        findings.iter().any(|f| {
            f["severity"].as_str() == Some("error")
                && (!ignore_team_size || f["code"].as_str() != Some("team-size"))
        })
    })
}

/// Cached Castle validation error classification for an exact team definition.
#[derive(Clone, Copy, Debug)]
struct CastleTeamLegality {
    has_team_size_error: bool,
    has_other_error: bool,
}

struct CastleLegalityEntry {
    dex_fingerprint: [u8; 16],
    ruleset: Ruleset,
    sets: Vec<PokemonSet>,
    legality: CastleTeamLegality,
}

fn castle_legality_hash(
    dex_fingerprint: [u8; 16],
    ruleset: Ruleset,
    sets: &[PokemonSet],
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    dex_fingerprint.hash(&mut h);
    ruleset.hash(&mut h);
    sets.hash(&mut h);
    h.finish()
}

fn classify_verdict_errors(verdict: &serde_json::Value) -> CastleTeamLegality {
    let Some(findings) = verdict["findings"].as_array() else {
        return CastleTeamLegality {
            has_team_size_error: true,
            has_other_error: true,
        };
    };
    let mut has_team_size_error = false;
    let mut has_other_error = false;
    for f in findings {
        if f["severity"].as_str() == Some("error") {
            if f["code"].as_str() == Some("team-size") {
                has_team_size_error = true;
            } else {
                has_other_error = true;
            }
        }
    }
    CastleTeamLegality {
        has_team_size_error,
        has_other_error,
    }
}

fn cached_castle_team_legality(
    dex: &Dex,
    sets: &[PokemonSet],
    ruleset: Ruleset,
) -> CastleTeamLegality {
    const MAX_CACHE_BUCKETS: usize = 4096;
    static CACHE: OnceLock<std::sync::RwLock<HashMap<u64, Vec<CastleLegalityEntry>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::RwLock::new(HashMap::new()));
    let fp = dex.replay_fingerprint;
    let h = castle_legality_hash(fp, ruleset, sets);
    if let Ok(guard) = cache.read() {
        if let Some(bucket) = guard.get(&h) {
            if let Some(entry) = bucket.iter().find(|e| {
                e.dex_fingerprint == fp
                    && e.ruleset == ruleset
                    && e.sets.as_slice() == sets
            }) {
                return entry.legality;
            }
        }
    }
    let legality = match serde_json::to_string(sets) {
        Ok(json) => {
            let verdict = validate_team_with_ruleset(dex, format_learnsets(), &json, ruleset);
            classify_verdict_errors(&verdict)
        }
        Err(_) => CastleTeamLegality {
            has_team_size_error: true,
            has_other_error: true,
        },
    };
    if let Ok(mut guard) = cache.write() {
        if guard.len() >= MAX_CACHE_BUCKETS {
            guard.clear();
        }
        let bucket = guard.entry(h).or_default();
        if !bucket.iter().any(|e| {
            e.dex_fingerprint == fp && e.ruleset == ruleset && e.sets.as_slice() == sets
        }) {
            bucket.push(CastleLegalityEntry {
                dex_fingerprint: fp,
                ruleset,
                sets: sets.to_vec(),
                legality,
            });
        }
    }
    legality
}

/// Does the played regulation accept this team as entered? `fix`-severity
/// and `warning`-severity findings keep a team legal; `error`s do not.
fn team_is_format_legal(dex: &Dex, sets: &[PokemonSet], ruleset: Ruleset) -> bool {
    if ruleset.is_castle() {
        let leg = cached_castle_team_legality(dex, sets, ruleset);
        return !leg.has_other_error && !leg.has_team_size_error;
    }
    let Ok(json) = serde_json::to_string(sets) else { return false };
    let verdict = validate_team_with_ruleset(dex, format_learnsets(), &json, ruleset);
    !verdict_has_errors(&verdict, false)
}

fn pool_team_legal_for_obs(dex: &Dex, sets: &[PokemonSet], obs: &Observer) -> bool {
    let ruleset = obs.ruleset();
    if !ruleset.is_castle() {
        return true;
    }
    let leg = cached_castle_team_legality(dex, sets, ruleset);
    // Allow sub-6-mon synthetic unit-test fixtures when pool team size matches obs roster size,
    // while strictly enforcing full 6-mon validation (including team-size) on real 6-mon games.
    let ignore_team_size = obs.mons().len() < 6 && sets.len() == obs.mons().len();
    !leg.has_other_error && (ignore_team_size || !leg.has_team_size_error)
}

pub(crate) fn format_learnsets() -> &'static Learnsets {
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
    /// `Some` only for a positive finite weight from a weighted prior.
    weight: Option<f64>,
    sets: Vec<PokemonSet>,
    /// Constructed reference mons, index = opponent roster slot.
    refs: Option<Vec<Pokemon>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FallbackPolicy {
    LegacyMetaOnly,
    Layered,
    BayesianPointMass,
    #[default]
    Bayesian,
}

impl FallbackPolicy {
    pub const fn id(self) -> &'static str {
        match self {
            Self::LegacyMetaOnly => "legacy-meta-only-v1",
            Self::Layered => "layered-meta-rentals-learnset-v1",
            Self::BayesianPointMass => "bayesian-point-mass-v1",
            Self::Bayesian => "bayesian-soft-prior-v1",
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

const ITEM_FALLBACK_ORDER: &[&str] = &[
    "leftovers", "miracleberry", "mintberry", "goldberry", "przcureberry", "psncureberry",
    "iceberry", "burntberry", "bitterberry", "berry", "scopelens", "quickclaw",
    "charcoal", "magnet", "mysticwater", "nevermeltice", "softsand", "hardstone",
    "miracleseed", "sharpbeak", "pinkbow", "polkadotbow", "twistedspoon", "blackbelt",
    "blackglasses", "poisonbarb", "silverpowder", "spelltag", "dragonfang",
];

const CASTLE_ITEM_FALLBACK_ORDER: &[&str] = &[
    "leftovers", "miracleberry", "mintberry", "goldberry", "przcureberry", "psncureberry",
    "iceberry", "burntberry", "berry", "scopelens", "quickclaw",
    "charcoal", "magnet", "mysticwater", "nevermeltice", "softsand", "hardstone",
    "miracleseed", "sharpbeak", "pinkbow", "polkadotbow", "twistedspoon", "blackbelt",
    "blackglasses", "poisonbarb", "silverpowder", "spelltag", "dragonscale",
];

#[inline]
fn item_fallback_order(ruleset: Ruleset) -> &'static [&'static str] {
    if ruleset.is_castle() {
        CASTLE_ITEM_FALLBACK_ORDER
    } else {
        ITEM_FALLBACK_ORDER
    }
}

/// Returns `true` if `move_str` is an explicit typed Hidden Power (or untyped Hidden Power
/// when `poss_hp == 0` and `hp_type_mask != 0`) made impossible by GSC DV/gender rules or
/// observed effectiveness (`poss_hp`).
fn explicit_hp_contradicted(dex: &Dex, move_str: &str, poss_hp: u16, hp_type_mask: u16) -> bool {
    let k = toid(move_str);
    if k == "hiddenpower" {
        return poss_hp == 0 && hp_type_mask != 0;
    }
    if k.starts_with("hiddenpower") {
        if poss_hp == 0 {
            return true;
        }
        let Some(mid) = dex.moves.id(&k) else {
            return true;
        };
        let ty = dex.moves.get(mid).move_type.as_str();
        let Some(idx) = crate::hp_evidence::TYPES.iter().position(|&t| t == ty) else {
            return true;
        };
        return (poss_hp & (1u16 << idx)) == 0;
    }
    false
}

fn align_moves_with_reference(dex: &Dex, mut mon: Pokemon, fallback_mon: &Pokemon) -> Pokemon {
    let fb_slots = &fallback_mon.base_move_slots;
    let cand_slots = mon.base_move_slots;
    if cand_slots.len() != fb_slots.len() || cand_slots.is_empty() {
        return mon;
    }
    let n = cand_slots.len();
    let mut aligned: [Option<MoveSlot>; 4] = [None; 4];
    let mut used_cand = [false; 4];
    for (c_idx, cs) in cand_slots.iter().enumerate() {
        if let Some(f_idx) =
            (0..n).find(|&fi| aligned[fi].is_none() && fb_slots[fi].id == cs.id)
        {
            aligned[f_idx] = Some(*cs);
            used_cand[c_idx] = true;
        }
    }
    for (c_idx, cs) in cand_slots.iter().enumerate() {
        if used_cand[c_idx] {
            continue;
        }
        if dex.moves.key(cs.id).starts_with("hiddenpower") {
            if let Some(f_idx) = (0..n).find(|&fi| {
                aligned[fi].is_none() && dex.moves.key(fb_slots[fi].id).starts_with("hiddenpower")
            }) {
                aligned[f_idx] = Some(*cs);
                used_cand[c_idx] = true;
            }
        }
    }
    for (c_idx, cs) in cand_slots.iter().enumerate() {
        if used_cand[c_idx] {
            continue;
        }
        if let Some(f_idx) = (0..n).find(|&fi| aligned[fi].is_none()) {
            aligned[f_idx] = Some(*cs);
        }
    }
    let mut nb = MoveSlots::default();
    for fi in 0..n {
        if let Some(slot) = aligned[fi] {
            nb.push(slot);
        }
    }
    mon.base_move_slots = nb;
    mon.move_slots = nb;
    mon
}

fn clone_mon_with_item(base: &Pokemon, item_id: Option<ItemId>) -> Pokemon {
    let mut mon = base.clone();
    mon.item = item_id;
    mon.item_state = EffectState {
        id: item_id.map(EffId::Item).unwrap_or_default(),
        effect_order: base.item_state.effect_order,
        ..Default::default()
    };
    mon
}

fn choose_fallback_item_for_slot(
    dex: &Dex,
    obs_mons: &[MonObs],
    slot: usize,
    used_items: &[ItemId],
    ruleset: Ruleset,
) -> Option<ItemId> {
    item_fallback_order(ruleset)
        .iter()
        .filter_map(|k| dex.items.id(k))
        .find(|&id| {
            !used_items.contains(&id)
                && !Observer::is_item_excluded_for_slot(dex, obs_mons, slot, Some(id))
        })
}

fn distinct_revealed_moves(dex: &Dex, mo: &MonObs) -> Vec<MoveId> {
    let mut out = Vec::with_capacity(4);
    for &m in &mo.revealed_moves {
        if out.len() >= 4 {
            break;
        }
        if !out.iter().any(|&prev| move_matches(dex, prev, m)) {
            out.push(m);
        }
    }
    out
}

fn legal_prior_moves(
    dex: &Dex,
    sp: &SpeciesPrior,
    mo: &MonObs,
    ruleset: Ruleset,
) -> Vec<(MoveId, String, f64)> {
    let species_key = dex.species.key(mo.species);
    let poss = mo.hp_type_mask
        & crate::hp_evidence::possible_types_with_maxhp(
            dex,
            mo.species,
            mo.level,
            mo.gender,
            mo.observed_maxhp,
        );
    let mut out: Vec<(MoveId, String, f64)> = Vec::new();
    for (key, w) in &sp.moves {
        let w = *w;
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let Some(id) = dex.moves.id(&toid(key)) else {
            continue;
        };
        let move_key = dex.moves.key(id);
        if !format_move_legal_at_level_for(species_key, move_key, mo.level, ruleset) {
            continue;
        }
        if mo.revealed_moves.iter().any(|&r| move_matches(dex, id, r)) {
            continue;
        }
        if move_key == "hiddenpower" {
            if poss == 0 {
                continue;
            }
        } else if move_key.starts_with("hiddenpower") {
            let ty = dex.moves.get(id).move_type.as_str();
            let Some(idx) = crate::hp_evidence::TYPES.iter().position(|&t| t == ty) else {
                continue;
            };
            if (poss & (1u16 << idx)) == 0 {
                continue;
            }
        }
        if out.iter().any(|(prev_id, _, _)| *prev_id == id) {
            continue;
        }
        out.push((id, move_key.to_string(), w));
    }
    out.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    out
}

fn legal_prior_items(
    dex: &Dex,
    sp: &SpeciesPrior,
    obs_mons: &[MonObs],
    slot: usize,
) -> Vec<(ItemId, f64)> {
    let Some(mo) = obs_mons.get(slot) else {
        return Vec::new();
    };
    if !mo.preview_has_item || mo.item.original.is_some() {
        return Vec::new();
    }
    let mut out: Vec<(ItemId, f64)> = Vec::new();
    for (key, w) in &sp.items {
        let w = *w;
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let Some(id) = dex.items.id(&toid(key)) else {
            continue;
        };
        if Observer::is_item_excluded_for_slot(dex, obs_mons, slot, Some(id)) {
            continue;
        }
        if out.iter().any(|(prev_id, _)| *prev_id == id) {
            continue;
        }
        out.push((id, w));
    }
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| dex.items.key(a.0).cmp(dex.items.key(b.0)))
    });
    out
}

fn enforce_roster_item_clause(
    dex: &Dex,
    obs_mons: &[MonObs],
    prior: Option<&BeliefPrior>,
    roster: &mut [Pokemon],
    ruleset: Ruleset,
) {
    let mut used_items: Vec<ItemId> = Vec::with_capacity(roster.len());
    for mo in obs_mons {
        if let Some(Some(id)) = mo.item.original {
            if !used_items.contains(&id) {
                used_items.push(id);
            }
        }
    }
    let mut order: Vec<usize> = (0..roster.len().min(obs_mons.len())).collect();
    order.sort_by_key(|&s| (!obs_mons[s].appeared, s));
    for slot in order {
        let mo = &obs_mons[slot];
        if mo.item.original.is_some() || !mo.preview_has_item {
            continue;
        }
        let ok = roster[slot].item.is_some_and(|id| {
            !used_items.contains(&id)
                && !Observer::is_item_excluded_for_slot(dex, obs_mons, slot, Some(id))
        });
        if ok {
            if let Some(id) = roster[slot].item {
                used_items.push(id);
            }
        } else {
            let prior_rep = prior
                .and_then(|pr| pr.species(dex.species.key(mo.species)))
                .and_then(|sp| {
                    legal_prior_items(dex, sp, obs_mons, slot)
                        .into_iter()
                        .map(|(id, _)| id)
                        .find(|id| !used_items.contains(id))
                });
            if let Some(rep) = prior_rep
                .or_else(|| choose_fallback_item_for_slot(dex, obs_mons, slot, &used_items, ruleset))
            {
                roster[slot] = clone_mon_with_item(&roster[slot], Some(rep));
                used_items.push(rep);
            }
        }
    }
}

fn hp_compatible_with_obs(dex: &Dex, mon: &Pokemon, mo: &MonObs) -> bool {
    crate::hp_evidence::consistent(dex, mon, mo.hp_type_mask)
        || (mo.hp_type_mask != 0
            && (mo.hp_type_mask
                & crate::hp_evidence::possible_types_with_maxhp(
                    dex,
                    mon.species,
                    mon.level,
                    mo.gender,
                    mo.observed_maxhp,
                ))
                != 0
            && mon
                .base_move_slots
                .iter()
                .any(|m| dex.moves.key(m.id) == "hiddenpower"))
}

type TeamSig = Vec<(u16, Option<u16>, Vec<u16>, [i32; 5], u8)>;

fn team_signature(team: &[Pokemon]) -> TeamSig {
    team.iter()
        .map(|p| {
            let mut mvs: Vec<u16> = p.base_move_slots.iter().map(|s| s.id.0).collect();
            mvs.sort_unstable();
            (p.species.0, p.item.map(|i| i.0), mvs, p.base_stored_stats, p.base_hp_type.0)
        })
        .collect()
}

struct LoadoutDraw {
    entries: Vec<(Pokemon, f64)>,
    total: f64,
}

impl LoadoutDraw {
    #[inline]
    fn draw_index(&self, rng: &mut SplitMix64) -> usize {
        if self.entries.len() == 1 {
            return 0;
        }
        let mut u = rng.next_f64() * self.total;
        for (i, (_, weight)) in self.entries.iter().enumerate() {
            u -= weight;
            if u < 0.0 {
                return i;
            }
        }
        self.entries.len() - 1
    }

    #[inline]
    fn draw(&self, rng: &mut SplitMix64) -> &Pokemon {
        &self.entries[self.draw_index(rng)].0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TeamPlanPick {
    Pool(usize),
    FallbackCand(usize),
    FallbackDefault,
}

impl TeamPlanPick {
    #[inline]
    pub(crate) fn pool_index(self) -> Option<usize> {
        match self {
            TeamPlanPick::Pool(idx) => Some(idx),
            TeamPlanPick::FallbackCand(_) | TeamPlanPick::FallbackDefault => None,
        }
    }
}

fn systematic_resample_indices(weights: &[f64], k: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut picked = Vec::with_capacity(k);
    if k == 0 || weights.is_empty() {
        return picked;
    }
    let pos: Vec<(usize, f64)> = weights
        .iter()
        .enumerate()
        .filter_map(|(i, &w)| (w.is_finite() && w > 0.0).then_some((i, w)))
        .collect();
    let items: Vec<(usize, f64)> = if pos.is_empty() {
        (0..weights.len()).map(|i| (i, 1.0)).collect()
    } else {
        pos
    };
    let total: f64 = items.iter().map(|&(_, w)| w).sum();
    let step = total / (k as f64);
    let u0 = rng.next_f64() * step;
    let mut idx = 0usize;
    let mut cum = items[0].1;
    for j in 0..k {
        let target = u0 + (j as f64) * step;
        while idx + 1 < items.len() && cum <= target {
            idx += 1;
            cum += items[idx].1;
        }
        picked.push(items[idx].0);
    }
    for i in (1..picked.len()).rev() {
        let j = rng.below(i + 1);
        picked.swap(i, j);
    }
    picked
}

#[derive(Clone)]
struct FallbackCandPool {
    entries: Vec<(Vec<Pokemon>, f64)>,
    total: f64,
}

impl FallbackCandPool {
    #[inline]
    fn draw_index(&self, rng: &mut SplitMix64) -> usize {
        if self.entries.len() <= 1 || !(self.total > 0.0) {
            return 0;
        }
        let mut u = rng.next_f64() * self.total;
        for (i, (_, weight)) in self.entries.iter().enumerate() {
            u -= *weight;
            if u < 0.0 {
                return i;
            }
        }
        self.entries.len() - 1
    }
}

pub struct Belief {
    ruleset: Ruleset,
    hp_observations: Option<Vec<(i32, i32)>>,
    cands: Vec<Candidate>,
    /// Pool indices of the candidates consistent with all observations so
    /// far. Monotonically non-increasing.
    alive: Vec<usize>,
    /// Synthesized imputation roster when `alive` is empty.
    fallback: Option<Vec<Pokemon>>,
    fallback_visible: Option<Vec<Pokemon>>,
    /// Open-team-sheet mode (M12): the single candidate IS the truth —
    /// filtering is skipped (provably a no-op, and skipping keeps a filter
    /// bug from silently dropping the truth to fallback).
    pinned: bool,
    fallback_policy: FallbackPolicy,
    /// M18: the community belief prior, when the owner loaded one.
    prior: Option<Arc<BeliefPrior>>,
    pick_draws: Option<(u64, Vec<Option<PickDistribution>>)>,
    prior_governed_slots: Vec<bool>,
    fallback_lead_weights: Option<[f64; 6]>,
    loadout_draws: Option<Vec<Option<LoadoutDraw>>>,
    fallback_cands: Option<FallbackCandPool>,
    synced: Option<u64>,
}

impl Belief {
    pub(crate) fn set_hp_observations(&mut self, hp: Vec<(i32, i32)>) {
        self.hp_observations = Some(hp);
    }
    /// Build the candidate set and apply the preview filter. Call at team
    /// preview, right after `Observer::new`.
    pub fn new(dex: &Dex, pool: &MetaPool, obs: &Observer) -> Belief {
        Self::with_fallback_policy(dex, pool, obs, FallbackPolicy::default())
    }

    /// Evaluation surface for comparing fallback synthesis policies. Live
    /// callers should use `new`, which is pinned to the shipped policy.
    pub fn with_fallback_policy(
        dex: &Dex,
        pool: &MetaPool,
        obs: &Observer,
        fallback_policy: FallbackPolicy,
    ) -> Belief {
        let ruleset = obs.ruleset();
        let cands: Vec<Candidate> = pool
            .teams
            .iter()
            .filter(|t| pool_team_legal_for_obs(dex, &t.sets, obs))
            .map(|t| {
                let refs = build_refs_with_ruleset(dex, &t.sets, obs.mons(), ruleset);
                Candidate {
                    id: t.id.clone(),
                    weight: t.weight.filter(|w| w.is_finite() && *w > 0.0),
                    sets: t.sets.clone(),
                    refs,
                }
            })
            .collect();
        let mut b = Belief {
            ruleset,
            hp_observations: None,
            cands,
            alive: Vec::new(),
            fallback: None,
            fallback_visible: None,
            pinned: false,
            fallback_policy,
            prior: None,
            pick_draws: None,
            prior_governed_slots: Vec::new(),
            fallback_lead_weights: None,
            loadout_draws: None,
            fallback_cands: None,
            synced: None,
        };
        b.refilter(dex, obs);
        b
    }

    /// Number of ruleset-legal pool teams loaded into this belief's candidate set.
    pub fn pool_candidate_count(&self) -> usize {
        self.cands.len()
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
        let ruleset = obs.ruleset();
        let team_json =
            serde_json::to_string(sets).map_err(|e| format!("serialize pinned team: {e}"))?;
        let verdict = validate_team_with_ruleset(dex, format_learnsets(), &team_json, ruleset);
        // The engine is generation-locked and intentionally has no
        // nature/ability semantics or cosmetic shiny flag (the shiny DVs
        // themselves are retained). Every other error/fix validator finding can
        // change a public field, stats, DVs, moves, or damage; informational
        // Castle warnings (e.g. `ev-effort-unverified`, `item-castle-inert`)
        // keep a legal team accepted.
        let unsupported_finding = verdict["findings"].as_array().is_none_or(|findings| {
            findings.iter().any(|finding| {
                finding["severity"].as_str() != Some("warning")
                    && !matches!(
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
        let refs = build_refs_with_ruleset(dex, sets, obs.mons(), ruleset)
            .ok_or_else(|| "pinned opponent team does not match the public preview".to_string())?;
        Ok(Belief {
            ruleset,
            hp_observations: None,
            cands: vec![Candidate {
                id: id.to_string(),
                weight: None,
                sets: sets.to_vec(),
                refs: Some(refs),
            }],
            alive: vec![0],
            fallback: None,
            fallback_visible: None,
            pinned: true,
            fallback_policy: FallbackPolicy::Layered,
            prior: None,
            pick_draws: None,
            prior_governed_slots: Vec::new(),
            fallback_lead_weights: None,
            loadout_draws: None,
            fallback_cands: None,
            synced: Some(obs.revision()),
        })
    }

    /// Construct a singleton pinned belief from raw opponent `PokemonSet`s
    /// without running the full pool validator (which rejects non-pool test
    /// fixtures and 3-mon subteams). Uses `build_refs` to align slots to public
    /// preview order, falling back to slot-for-slot `Battle::from_fixture_with_ruleset`
    /// when `sets.len() == obs.mons().len()`.
    pub fn pinned_from_sets(
        dex: &Dex,
        id: &str,
        sets: &[PokemonSet],
        obs: &Observer,
    ) -> Result<Belief, String> {
        let ruleset = obs.ruleset();
        let refs = if let Some(refs) = build_refs_with_ruleset(dex, sets, obs.mons(), ruleset) {
            refs
        } else if sets.len() == obs.mons().len() {
            Battle::from_fixture_with_ruleset(dex, "1,2,3,4", sets, sets, ruleset)
                .map(|mut b| std::mem::take(&mut b.sides[0].roster))
                .map_err(|e| format!("construct pinned opponent refs: {e:?}"))?
        } else {
            return Err("pinned opponent team does not match the public preview".to_string());
        };
        Ok(Belief {
            ruleset,
            hp_observations: None,
            cands: vec![Candidate {
                id: id.to_string(),
                weight: None,
                sets: sets.to_vec(),
                refs: Some(refs),
            }],
            alive: vec![0],
            fallback: None,
            fallback_visible: None,
            pinned: true,
            fallback_policy: FallbackPolicy::Layered,
            prior: None,
            pick_draws: None,
            prior_governed_slots: Vec::new(),
            fallback_lead_weights: None,
            loadout_draws: None,
            fallback_cands: None,
            synced: Some(obs.revision()),
        })
    }

    /// The pinned opponent's Turn-0 `PokemonSet`s when constructed via
    /// `pinned`, `pinned_checked`, or `pinned_from_sets`.
    pub fn pinned_sets(&self) -> Option<&[PokemonSet]> {
        self.pinned
            .then(|| {
                self.cands
                    .first()
                    .map(|c| c.sets.as_slice())
                    .filter(|s| !s.is_empty())
            })
            .flatten()
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
            ruleset: battle.ruleset,
            hp_observations: None,
            cands: vec![Candidate {
                id: "opponent".to_string(),
                weight: None,
                sets: Vec::new(),
                refs: Some(refs),
            }],
            alive: vec![0],
            fallback: None,
            fallback_visible: None,
            pinned: true,
            fallback_policy: FallbackPolicy::Layered,
            prior: None,
            pick_draws: None,
            prior_governed_slots: Vec::new(),
            fallback_lead_weights: None,
            loadout_draws: None,
            fallback_cands: None,
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
        self.prior_governed_slots.clear();
        self.fallback_lead_weights = None;
        self.loadout_draws = None;
        self.fallback_cands = None;
        self.synced = None;
    }

    fn active_prior(&self) -> Option<&BeliefPrior> {
        self.prior
            .as_deref()
            .filter(|p| !p.is_empty() && !self.pinned && self.alive.is_empty())
    }

    pub fn has_prior(&self) -> bool {
        self.prior.as_deref().is_some_and(|p| !p.is_empty())
    }

    pub fn condition_pick_prior(
        &mut self,
        dex: &Dex,
        prior: &PickPrior,
        own_preview: &[PreviewMon],
        obs: &Observer,
    ) {
        if self.pinned {
            self.pick_draws = None;
            return;
        }
        let roster: Vec<String> = obs.mons().iter()
            .map(|m| dex.species.key(m.species).to_string()).collect();
        let appeared = obs.mons().iter().enumerate()
            .fold(0u8, |mask, (i, m)| mask | (u8::from(m.appeared) << i));
        let plans = self.cands.iter().map(|candidate| {
            prior.condition(&candidate.id, own_preview, obs.opp(), &roster, appeared)
        }).collect();
        self.pick_draws = Some((obs.revision(), plans));
    }

    pub fn clear_pick_draws(&mut self) {
        self.pick_draws = None;
    }

    /// Condition a pinned (open-sheet) belief's hidden bench sampling on a
    /// 60-action canonical preview policy `policy` (e.g., from a Nash equilibrium
    /// over cores), mixed with `smoothing` over all 3-mon subsets containing
    /// `obs`'s already-revealed opponent mons.
    pub fn condition_canonical_pick_policy(
        &mut self,
        policy: &[f64],
        smoothing: f64,
        obs: &Observer,
    ) {
        let appeared = obs
            .mons()
            .iter()
            .enumerate()
            .fold(0u8, |mask, (i, m)| mask | (u8::from(m.appeared) << i));
        let dist = crate::pick_prior::PickDistribution::from_canonical_policy(
            policy, appeared, smoothing,
        );
        let plans = (0..self.cands.len())
            .map(|_| {
                crate::pick_prior::PickDistribution::from_canonical_policy(
                    policy, appeared, smoothing,
                )
            })
            .collect();
        if dist.is_some() {
            self.pick_draws = Some((obs.revision(), plans));
        } else {
            self.pick_draws = None;
        }
    }

    /// True if an active `pick_draws` distribution is installed for `obs.revision()`.
    pub fn has_active_pick_draws(&self, obs: &Observer) -> bool {
        self.pick_draws.as_ref().is_some_and(|(rev, plans)| {
            *rev == obs.revision()
                && self
                    .alive
                    .iter()
                    .any(|&i| plans.get(i).is_some_and(Option::is_some))
        })
    }

    /// True if an opponent preview prior (`pick_draws` or off-pool fallback
    /// `fallback_lead_weights`) is active for `obs.revision()`.
    pub fn has_preview_prior(&self, obs: &Observer) -> bool {
        self.has_active_pick_draws(obs)
            || (self.synced == Some(obs.revision()) && self.fallback_lead_weights.is_some())
    }

    pub(crate) fn sample_preview_prior(
        &self,
        pick: Option<usize>,
        obs: &Observer,
        actions: &[nc2000_engine::battle::SearchChoice],
        rng: &mut SplitMix64,
    ) -> Option<usize> {
        if let Some(chosen) = self
            .pick_draws
            .as_ref()
            .filter(|(revision, _)| *revision == obs.revision())
            .and_then(|(_, plans)| pick.and_then(|i| plans[i].as_ref()))
            .and_then(|dist| dist.sample_preview(actions, rng))
        {
            return Some(chosen);
        }
        if pick.is_none() && self.synced == Some(obs.revision()) {
            if let Some(lw) = &self.fallback_lead_weights {
                if actions.is_empty() {
                    return None;
                }
                let mut scores = Vec::with_capacity(actions.len());
                let mut raw_sum = 0.0f64;
                for action in actions {
                    let s = match action {
                        nc2000_engine::battle::SearchChoice::Team(slots) => {
                            let a = slots[0];
                            if !(1..=6).contains(&a) {
                                0.0
                            } else {
                                let lead_w = lw[(a - 1) as usize];
                                let team_w: f64 = slots
                                    .iter()
                                    .filter(|&&slot| (1..=6).contains(&slot))
                                    .map(|&slot| lw[(slot - 1) as usize])
                                    .sum();
                                if team_w > 0.0 { lead_w / team_w } else { 0.0 }
                            }
                        }
                        _ => 0.0,
                    };
                    raw_sum += s;
                    scores.push(s);
                }
                let n = actions.len() as f64;
                let mut u = rng.next_f64();
                for (i, raw) in scores.into_iter().enumerate() {
                    let norm = if raw_sum > 0.0 { raw / raw_sum } else { 1.0 / n };
                    let p = 0.85 * norm + 0.15 / n;
                    u -= p;
                    if u < 0.0 {
                        return Some(i);
                    }
                }
                return Some(actions.len() - 1);
            }
        }
        None
    }

    /// Exact probability vector over full legal opponent Team Preview `actions`,
    /// matching `sample_preview_prior(pick, obs, actions, rng)`.
    pub(crate) fn preview_prior_probs(
        &self,
        pick: Option<usize>,
        obs: &Observer,
        actions: &[nc2000_engine::battle::SearchChoice],
    ) -> Option<Vec<f64>> {
        if let Some(probs) = self
            .pick_draws
            .as_ref()
            .filter(|(revision, _)| *revision == obs.revision())
            .and_then(|(_, plans)| pick.and_then(|i| plans.get(i)?.as_ref()))
            .and_then(|dist| dist.preview_probs(actions))
        {
            return Some(probs);
        }
        if pick.is_none() && self.synced == Some(obs.revision()) {
            if let Some(lw) = &self.fallback_lead_weights {
                if actions.is_empty() {
                    return None;
                }
                let mut scores = Vec::with_capacity(actions.len());
                let mut raw_sum = 0.0f64;
                for action in actions {
                    let s = match action {
                        nc2000_engine::battle::SearchChoice::Team(slots) => {
                            let a = slots[0];
                            if !(1..=6).contains(&a) {
                                0.0
                            } else {
                                let lead_w = lw[(a - 1) as usize];
                                let team_w: f64 = slots
                                    .iter()
                                    .filter(|&&slot| (1..=6).contains(&slot))
                                    .map(|&slot| lw[(slot - 1) as usize])
                                    .sum();
                                if team_w > 0.0 { lead_w / team_w } else { 0.0 }
                            }
                        }
                        _ => 0.0,
                    };
                    raw_sum += s;
                    scores.push(s);
                }
                let n = actions.len() as f64;
                let mut probs: Vec<f64> = scores
                    .into_iter()
                    .map(|raw| {
                        let norm = if raw_sum > 0.0 { raw / raw_sum } else { 1.0 / n };
                        0.85 * norm + 0.15 / n
                    })
                    .collect();
                let sum: f64 = probs.iter().sum();
                if sum > 0.0 {
                    for p in &mut probs {
                        *p /= sum;
                    }
                    return Some(probs);
                }
            }
        }
        None
    }

    /// Exact probability vector over canonical opponent Team Preview choices `opp_canon`,
    /// folded from `opp_legal` using the exact same bench-order equivalence as
    /// `canonical_preview_actions`.
    pub(crate) fn preview_prior_probs_canonical(
        &self,
        plan_pick: TeamPlanPick,
        obs: &Observer,
        opp_legal: &[nc2000_engine::battle::SearchChoice],
        opp_canon: &[(usize, [u8; 3])],
        opp_order_matters: bool,
    ) -> Option<Vec<f64>> {
        let legal_probs = self.preview_prior_probs(plan_pick.pool_index(), obs, opp_legal)?;
        if opp_canon.is_empty() {
            return None;
        }
        let canon_key = |mut t: [u8; 3]| -> [u8; 3] {
            if !opp_order_matters && t[2] != 0 && t[1] > t[2] {
                t.swap(1, 2);
            }
            t
        };
        let mut canon_probs = vec![0.0f64; opp_canon.len()];
        for (&ch, &prob) in opp_legal.iter().zip(legal_probs.iter()) {
            let nc2000_engine::battle::SearchChoice::Team(t) = ch else {
                continue;
            };
            let key = canon_key(t);
            if let Some(pos) = opp_canon.iter().position(|&(_, ct)| canon_key(ct) == key) {
                canon_probs[pos] += prob;
            }
        }
        let sum: f64 = canon_probs.iter().sum();
        if sum > 0.0 {
            for p in &mut canon_probs {
                *p /= sum;
            }
            Some(canon_probs)
        } else {
            None
        }
    }

    /// Which fallback roster slots the installed prior currently governs, in
    /// observer roster order (`false` = today's deterministic filler). Empty
    /// when nothing is governed. A coverage surface for the M18 gate.
    pub fn prior_governed(&self) -> Vec<bool> {
        if self.prior_governed_slots.iter().any(|&g| g) {
            self.prior_governed_slots.clone()
        } else {
            Vec::new()
        }
    }

    /// Per-mon synthesis sources in observer roster order. Meaningful when
    /// `is_fallback()`; callers may inspect it at preview before any reveal.
    pub fn fallback_sources(&self, dex: &Dex, obs: &Observer) -> Vec<FallbackSource> {
        obs.mons().iter().map(|mo| self.fallback_source(dex, mo)).collect()
    }

    pub fn loadout_draw_len(&self, slot: usize) -> usize {
        self.loadout_draws
            .as_deref()
            .and_then(|d| d.get(slot)?.as_ref())
            .map_or(0, |ld| ld.entries.len())
    }

    pub fn fallback_cand_count(&self) -> usize {
        self.fallback_cands
            .as_ref()
            .map_or(0, |pool| pool.entries.len())
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
                .fallback_visible
                .as_deref()
                .or(self.fallback.as_deref())
                .expect("refs: fallback roster not built (call sync first)"),
        }
    }

    /// Sample a consistent candidate (`None` = fallback roster): uniform,
    /// or weight-proportional when the survivors' weights differ. Equal
    /// weights take the uniform draw, so an unweighted pool consumes the rng
    /// exactly as before weights existed.
    pub fn sample(&self, rng: &mut SplitMix64) -> Option<usize> {
        let weight = |i: usize| self.cands[i].weight.unwrap_or(1.0);
        let first = *self.alive.first()?;
        if self.alive.iter().all(|&i| weight(i) == weight(first)) {
            return Some(self.alive[rng.below(self.alive.len())]);
        }
        let total: f64 = self.alive.iter().map(|&i| weight(i)).sum();
        let mut r = rng.next_f64() * total;
        for &i in &self.alive {
            if r < weight(i) {
                return Some(i);
            }
            r -= weight(i);
        }
        self.alive.last().copied()
    }

    pub(crate) fn root_sampling_plan(&self, k: usize, rng: &mut SplitMix64) -> Vec<TeamPlanPick> {
        if k == 0 {
            return Vec::new();
        }
        if !self.alive.is_empty() {
            let weights: Vec<f64> = self
                .alive
                .iter()
                .map(|&i| self.cands[i].weight.unwrap_or(1.0))
                .collect();
            return systematic_resample_indices(&weights, k, rng)
                .into_iter()
                .map(|pos| TeamPlanPick::Pool(self.alive[pos]))
                .collect();
        }
        if !self.pinned {
            if let Some(pool) = &self.fallback_cands {
                if !pool.entries.is_empty() {
                    let weights: Vec<f64> = pool.entries.iter().map(|(_, w)| *w).collect();
                    return systematic_resample_indices(&weights, k, rng)
                        .into_iter()
                        .map(TeamPlanPick::FallbackCand)
                        .collect();
                }
            }
        }
        vec![TeamPlanPick::FallbackDefault; k]
    }

    /// Exact normalized Turn-0 posterior distribution over distinct opponent world picks
    /// `(TeamPlanPick, weight)` for Bayesian Team Preview (zero hidden-info leakage).
    pub(crate) fn preview_world_distribution(
        &self,
        max_worlds: usize,
        rng: &mut SplitMix64,
    ) -> Vec<(TeamPlanPick, f64)> {
        let cap = max_worlds.max(1);
        if !self.alive.is_empty() {
            let mut items: Vec<(TeamPlanPick, f64)> = self
                .alive
                .iter()
                .map(|&i| (TeamPlanPick::Pool(i), self.cands[i].weight.unwrap_or(1.0).max(0.0)))
                .collect();
            let sum: f64 = items.iter().map(|&(_, w)| w).sum();
            if sum <= 0.0 {
                let u = 1.0 / (items.len() as f64);
                for (_, w) in &mut items {
                    *w = u;
                }
            } else {
                for (_, w) in &mut items {
                    *w /= sum;
                }
            }
            if items.len() <= cap {
                return items;
            }
            // Systematic stratified resampling down to `cap` worlds, merging duplicate picks.
            let plan = self.root_sampling_plan(cap, rng);
            let mut merged: Vec<(TeamPlanPick, f64)> = Vec::with_capacity(cap);
            let step = 1.0 / (plan.len() as f64);
            for p in plan {
                if let Some((_, w)) = merged.iter_mut().find(|(k, _)| *k == p) {
                    *w += step;
                } else {
                    merged.push((p, step));
                }
            }
            return merged;
        }
        if !self.pinned {
            if let Some(pool) = &self.fallback_cands {
                if !pool.entries.is_empty() {
                    let mut items: Vec<(TeamPlanPick, f64)> = pool
                        .entries
                        .iter()
                        .enumerate()
                        .map(|(idx, &(_, w))| (TeamPlanPick::FallbackCand(idx), w.max(0.0)))
                        .collect();
                    let sum: f64 = items.iter().map(|&(_, w)| w).sum();
                    if sum <= 0.0 {
                        let u = 1.0 / (items.len() as f64);
                        for (_, w) in &mut items {
                            *w = u;
                        }
                    } else {
                        for (_, w) in &mut items {
                            *w /= sum;
                        }
                    }
                    if items.len() <= cap {
                        return items;
                    }
                    let plan = self.root_sampling_plan(cap, rng);
                    let mut merged: Vec<(TeamPlanPick, f64)> = Vec::with_capacity(cap);
                    let step = 1.0 / (plan.len() as f64);
                    for p in plan {
                        if let Some((_, w)) = merged.iter_mut().find(|(k, _)| *k == p) {
                            *w += step;
                        } else {
                            merged.push((p, step));
                        }
                    }
                    return merged;
                }
            }
        }
        vec![(TeamPlanPick::FallbackDefault, 1.0)]
    }

    fn is_weighted_prior(&self) -> bool {
        self.cands.iter().any(|c| c.weight.is_some())
    }

    fn pool_set_for(&self, dex: &Dex, mo: &MonObs) -> Option<&PokemonSet> {
        let carriers = self.cands.iter().filter_map(|c| {
            c.sets
                .iter()
                .find(|s| dex.species.id(&toid(&s.species)) == Some(mo.species))
                .map(|s| (c, s))
        });
        if !self.is_weighted_prior() {
            return carriers.map(|(_, s)| s).next();
        }
        let loadout = |s: &PokemonSet| {
            let mut moves: Vec<String> = s.moves.iter().map(|m| toid(m)).collect();
            moves.sort();
            (toid(&s.item), moves)
        };
        let mut tally: Vec<((String, Vec<String>), f64, &PokemonSet)> = Vec::new();
        for (c, s) in carriers {
            let key = loadout(s);
            let w = c.weight.unwrap_or(1.0);
            match tally.iter_mut().find(|(k, _, _)| *k == key) {
                Some(entry) => entry.1 += w,
                None => tally.push((key, w, s)),
            }
        }
        let mut best: Option<(f64, &PokemonSet)> = None;
        for (_, w, s) in &tally {
            if best.is_none_or(|(bw, _)| *w > bw) {
                best = Some((*w, *s));
            }
        }
        best.map(|(_, s)| s)
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
        let mut out = battle.clone();
        self.determinize_in_place(dex, &mut out, obs, pick, rng);
        out
    }

    /// The input must be reset to the observed battle before imputation.
    pub(crate) fn determinize_in_place(
        &self,
        dex: &Dex,
        out: &mut Battle,
        obs: &Observer,
        pick: Option<usize>,
        rng: &mut SplitMix64,
    ) {
        let sampled = self.sample_fallback_refs(pick, rng, out, obs.opp());
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
        self.determinize_with_refs_in_place(dex, out, obs, pick, refs, rng);
    }

    pub(crate) fn determinize_in_place_planned(
        &self,
        dex: &Dex,
        out: &mut Battle,
        obs: &Observer,
        plan_pick: TeamPlanPick,
        rng: &mut SplitMix64,
    ) {
        match plan_pick {
            TeamPlanPick::Pool(i) => {
                let refs = self.cands[i]
                    .refs
                    .as_deref()
                    .expect("determinize_with: candidate is preview-inconsistent");
                self.determinize_with_refs_in_place(dex, out, obs, Some(i), refs, rng);
            }
            TeamPlanPick::FallbackCand(idx)
                if !self.pinned
                    && self
                        .fallback_cands
                        .as_ref()
                        .is_some_and(|pool| idx < pool.entries.len()) =>
            {
                let refs = &self.fallback_cands.as_ref().unwrap().entries[idx].0;
                self.determinize_with_refs_in_place(dex, out, obs, None, refs, rng);
            }
            TeamPlanPick::FallbackCand(_) | TeamPlanPick::FallbackDefault => {
                self.determinize_in_place(dex, out, obs, None, rng);
            }
        }
    }

    pub(crate) fn sample_hidden_party(
        &self,
        obs: &Observer,
        pick: Option<usize>,
        opp_side: &nc2000_engine::state::Side,
        preview_level_cap: Option<u32>,
        rng: &mut SplitMix64,
    ) -> ([u8; 6], usize, bool) {
        let roster_len = opp_side.roster.len().min(6);
        let party_len = opp_side.party.len().min(6);
        let mut party_key = [0u8; 6];
        for pos in 0..party_len {
            party_key[pos] = opp_side.party[pos];
        }
        if party_len >= roster_len {
            return (party_key, party_len, false);
        }
        let mut appeared = [false; 6];
        for s in 0..roster_len {
            appeared[s] = opp_side.roster[s].previously_switched_in > 0
                || opp_side.roster[s].is_active;
        }
        let mut hidden_positions = [0usize; 6];
        let mut hidden_len = 0usize;
        for pos in 0..party_len {
            if !appeared[party_key[pos] as usize] {
                hidden_positions[hidden_len] = pos;
                hidden_len += 1;
            }
        }
        if hidden_len == 0 {
            return (party_key, party_len, false);
        }

        let mut pool = [0u8; 6];
        let mut pool_len = 0usize;
        for s in 0..roster_len {
            if !appeared[s] {
                pool[pool_len] = s as u8;
                pool_len += 1;
            }
        }
        let public = (0..roster_len)
            .fold(0u8, |mask, slot| mask | (u8::from(appeared[slot]) << slot));
        let level_of = |mask: u8| -> u32 {
            obs.mons()
                .iter()
                .enumerate()
                .take(roster_len)
                .filter(|(slot, _)| mask & (1 << slot) != 0)
                .map(|(_, mon)| u32::from(mon.level))
                .sum()
        };
        let collect_legal = |cap: u32| -> ([u8; 64], usize) {
            let mut legal = [0u8; 64];
            let mut n = 0usize;
            for mask in 0..(1u8 << roster_len) {
                if mask.count_ones() as usize == party_len
                    && (mask & public) == public
                    && level_of(mask) <= cap
                {
                    legal[n] = mask;
                    n += 1;
                }
            }
            assert!(n > 0, "public reveals have no level-legal selection");
            (legal, n)
        };

        let distribution = if party_len == 3 {
            self.pick_draws
                .as_ref()
                .filter(|(revision, _)| *revision == obs.revision())
                .and_then(|(_, plans)| pick.and_then(|i| plans[i].as_ref()))
        } else {
            None
        };

        if let Some(distribution) = distribution {
            let mask = distribution.sample(rng);
            let mut selected = [0u8; 6];
            let mut sel_len = 0usize;
            for &slot in &pool[..pool_len] {
                if mask & (1 << slot) != 0 {
                    selected[sel_len] = slot;
                    sel_len += 1;
                }
            }
            if sel_len == hidden_len {
                pool = selected;
                pool_len = sel_len;
            }
            for &pos in &hidden_positions[..hidden_len] {
                let idx = rng.below(pool_len);
                let new_slot = pool[idx];
                pool[idx] = pool[pool_len - 1];
                pool_len -= 1;
                party_key[pos] = new_slot;
            }
            if let Some(cap) = preview_level_cap {
                let cur_mask = (0..party_len).fold(0u8, |m, p| m | (1 << party_key[p]));
                if level_of(cur_mask) > cap {
                    let (legal, legal_len) = collect_legal(cap);
                    let mask = distribution
                        .sample_allowed(&legal[..legal_len], rng)
                        .unwrap_or_else(|| legal[rng.below(legal_len)]);
                    pool_len = 0;
                    for slot in 0..roster_len {
                        if mask & (1 << slot) != 0 && !appeared[slot] {
                            pool[pool_len] = slot as u8;
                            pool_len += 1;
                        }
                    }
                    for &pos in &hidden_positions[..hidden_len] {
                        let idx = rng.below(pool_len);
                        let new_slot = pool[idx];
                        pool[idx] = pool[pool_len - 1];
                        pool_len -= 1;
                        party_key[pos] = new_slot;
                    }
                }
            }
            return (party_key, party_len, true);
        }

        if pick.is_none() && self.synced == Some(obs.revision()) {
            if let Some(lw) = &self.fallback_lead_weights {
                if party_len == 3 {
                    let mut legal = [0u8; 64];
                    let mut legal_len = 0usize;
                    for mask in 0..(1u8 << roster_len) {
                        if mask.count_ones() as usize == 3
                            && (mask & public) == public
                            && preview_level_cap.is_none_or(|cap| level_of(mask) <= cap)
                        {
                            legal[legal_len] = mask;
                            legal_len += 1;
                        }
                    }
                    assert!(legal_len > 0, "public reveals have no level-legal selection");
                    if hidden_positions[0] == 0 {
                        let mut cands = [(0u8, 0u8, 0.0f64); 192];
                        let mut n_cands = 0usize;
                        let mut total_w = 0.0f64;
                        for &mask in &legal[..legal_len] {
                            let sum_m: f64 = (0..roster_len)
                                .filter(|&x| (mask & (1 << x)) != 0)
                                .map(|x| lw[x])
                                .sum();
                            for s0 in 0..roster_len {
                                if (mask & (1 << s0)) != 0 && !appeared[s0] {
                                    let w = if sum_m > 0.0 { lw[s0] / sum_m } else { 1.0 };
                                    cands[n_cands] = (mask, s0 as u8, w);
                                    n_cands += 1;
                                    total_w += w;
                                }
                            }
                        }
                        let mut u = rng.next_f64() * total_w;
                        let mut chosen = cands[n_cands - 1];
                        for &c in &cands[..n_cands] {
                            u -= c.2;
                            if u < 0.0 {
                                chosen = c;
                                break;
                            }
                        }
                        let (chosen_mask, s0, _) = chosen;
                        party_key[0] = s0;
                        let mut rem_pool = [0u8; 6];
                        let mut rem_len = 0usize;
                        for s in 0..roster_len {
                            if (chosen_mask & (1 << s)) != 0 && !appeared[s] && s as u8 != s0 {
                                rem_pool[rem_len] = s as u8;
                                rem_len += 1;
                            }
                        }
                        for &pos in &hidden_positions[1..hidden_len] {
                            let idx = rng.below(rem_len);
                            let slot = rem_pool[idx];
                            rem_pool[idx] = rem_pool[rem_len - 1];
                            rem_len -= 1;
                            party_key[pos] = slot;
                        }
                    } else {
                        let l0 = party_key[0] as usize;
                        let mut cands = [(0u8, 0.0f64); 64];
                        let mut total_w = 0.0f64;
                        for (i, &mask) in legal[..legal_len].iter().enumerate() {
                            let sum_m: f64 = (0..roster_len)
                                .filter(|&x| (mask & (1 << x)) != 0)
                                .map(|x| lw[x])
                                .sum();
                            let w = if sum_m > 0.0 { lw[l0] / sum_m } else { 1.0 };
                            cands[i] = (mask, w);
                            total_w += w;
                        }
                        let mut u = rng.next_f64() * total_w;
                        let mut chosen_mask = cands[legal_len - 1].0;
                        for &(mask, w) in &cands[..legal_len] {
                            u -= w;
                            if u < 0.0 {
                                chosen_mask = mask;
                                break;
                            }
                        }
                        let mut rem_pool = [0u8; 6];
                        let mut rem_len = 0usize;
                        for s in 0..roster_len {
                            if (chosen_mask & (1 << s)) != 0 && !appeared[s] {
                                rem_pool[rem_len] = s as u8;
                                rem_len += 1;
                            }
                        }
                        for &pos in &hidden_positions[..hidden_len] {
                            let idx = rng.below(rem_len);
                            let slot = rem_pool[idx];
                            rem_pool[idx] = rem_pool[rem_len - 1];
                            rem_len -= 1;
                            party_key[pos] = slot;
                        }
                    }
                    return (party_key, party_len, true);
                } else if hidden_positions[0] == 0 {
                    let legal_opt = preview_level_cap.map(&collect_legal);
                    let mut lead_cands = [0u8; 6];
                    let mut lead_len = 0usize;
                    let mut lead_total = 0.0f64;
                    for &s in &pool[..pool_len] {
                        let ok = match &legal_opt {
                            Some((legal, legal_len)) => {
                                legal[..*legal_len].iter().any(|&m| (m & (1 << s)) != 0)
                            }
                            None => true,
                        };
                        if ok {
                            lead_cands[lead_len] = s;
                            lead_len += 1;
                            lead_total += lw[s as usize];
                        }
                    }
                    let s0 = if lead_len > 0 && lead_total > 0.0 {
                        let mut u = rng.next_f64() * lead_total;
                        let mut pick_s = lead_cands[lead_len - 1];
                        for &s in &lead_cands[..lead_len] {
                            u -= lw[s as usize];
                            if u < 0.0 {
                                pick_s = s;
                                break;
                            }
                        }
                        pick_s
                    } else {
                        pool[rng.below(pool_len)]
                    };
                    party_key[0] = s0;
                    if let Some(pos_in_pool) = pool[..pool_len].iter().position(|&s| s == s0) {
                        pool[pos_in_pool] = pool[pool_len - 1];
                        pool_len -= 1;
                    }
                    for &pos in &hidden_positions[1..hidden_len] {
                        let idx = rng.below(pool_len);
                        let new_slot = pool[idx];
                        pool[idx] = pool[pool_len - 1];
                        pool_len -= 1;
                        party_key[pos] = new_slot;
                    }
                    if let Some((legal, legal_len)) = legal_opt {
                        let cap = preview_level_cap.unwrap();
                        let cur_mask = (0..party_len).fold(0u8, |m, p| m | (1 << party_key[p]));
                        if level_of(cur_mask) > cap {
                            let mut sub = [0u8; 64];
                            let mut sub_len = 0usize;
                            for &m in &legal[..legal_len] {
                                if (m & (1 << s0)) != 0 {
                                    sub[sub_len] = m;
                                    sub_len += 1;
                                }
                            }
                            let mask = if sub_len > 0 {
                                sub[rng.below(sub_len)]
                            } else {
                                legal[rng.below(legal_len)]
                            };
                            let mut rem_pool = [0u8; 6];
                            let mut rem_len = 0usize;
                            for slot in 0..roster_len {
                                if mask & (1 << slot) != 0 && !appeared[slot] && slot as u8 != s0 {
                                    rem_pool[rem_len] = slot as u8;
                                    rem_len += 1;
                                }
                            }
                            for &pos in &hidden_positions[1..hidden_len] {
                                let idx = rng.below(rem_len);
                                let new_slot = rem_pool[idx];
                                rem_pool[idx] = rem_pool[rem_len - 1];
                                rem_len -= 1;
                                party_key[pos] = new_slot;
                            }
                        }
                    }
                    return (party_key, party_len, true);
                }
            }
        }

        for &pos in &hidden_positions[..hidden_len] {
            let idx = rng.below(pool_len);
            let new_slot = pool[idx];
            pool[idx] = pool[pool_len - 1];
            pool_len -= 1;
            party_key[pos] = new_slot;
        }
        if let Some(cap) = preview_level_cap {
            let cur_mask = (0..party_len).fold(0u8, |m, p| m | (1 << party_key[p]));
            if level_of(cur_mask) > cap {
                let (legal, legal_len) = collect_legal(cap);
                let mask = legal[rng.below(legal_len)];
                pool_len = 0;
                for slot in 0..roster_len {
                    if mask & (1 << slot) != 0 && !appeared[slot] {
                        pool[pool_len] = slot as u8;
                        pool_len += 1;
                    }
                }
                for &pos in &hidden_positions[..hidden_len] {
                    let idx = rng.below(pool_len);
                    let new_slot = pool[idx];
                    pool[idx] = pool[pool_len - 1];
                    pool_len -= 1;
                    party_key[pos] = new_slot;
                }
            }
        }
        (party_key, party_len, true)
    }

    fn determinize_with_refs_in_place(
        &self,
        dex: &Dex,
        out: &mut Battle,
        obs: &Observer,
        pick: Option<usize>,
        refs: &[Pokemon],
        rng: &mut SplitMix64,
    ) {
        audit_battle_hidden(out);

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
        out.roll_quick_claw();

        let opp = obs.opp();
        let roster_len = out.sides[opp].roster.len();

        let (party_key, party_len, changed) = self.sample_hidden_party(
            obs,
            pick,
            &out.sides[opp],
            out.preview_level_caps[opp],
            rng,
        );
        if changed {
            // party members carry their display index; the rest are
            // parked canonically (same assignment ⇒ same state key)
            for pos in 0..party_len {
                let slot = party_key[pos] as usize;
                out.sides[opp].party[pos] = slot as u8;
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

        // ---- per-mon set imputation (all 6 roster mons, picked or not)
        let exact_hp = out.is_castle();
        for slot in 0..roster_len {
            let mon = &mut out.sides[opp].roster[slot];
            let mo = &obs.mons()[slot];
            let projected = crate::hp_evidence::project(dex, &refs[slot], mo, rng);
            impute_mon(mon, projected.as_ref().unwrap_or(&refs[slot]), mo, dex, exact_hp);
            if mo.appeared {
                if let Some(&(cur, den)) = self.hp_observations.as_ref().and_then(|h| h.get(slot)) {
                    mon.hp = crate::import::impute_hp_mode(cur, den, mon.maxhp, exact_hp);
                }
            }
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
        self.scrub_pending_move(dex, out, opp, rng);

        out.battle_mask = out.recompute_battle_mask(dex);
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
        let new_id = if let Some(forced) = out
            .get_locked_move(dex, active)
            .or_else(|| out.get_semi_locked_move(dex, active))
            .and_then(|k| dex.moves.id(k))
        {
            forced
        } else if let Some(encored) = dex
            .conds_id("encore")
            .and_then(|c| out.poke(active).volatile(c))
            .and_then(|v| v.get_move())
            .filter(|&m| {
                out.poke(active)
                    .move_slots
                    .iter()
                    .any(|s| s.id == m && s.pp > 0)
            })
        {
            encored
        } else {
            let usable: Vec<MoveId> = out
                .poke(active)
                .move_slots
                .iter()
                .filter(|s| s.pp > 0 && !s.disabled)
                .map(|s| s.id)
                .collect();
            if usable.is_empty() {
                dex.moves.id("struggle").expect("struggle interned")
            } else {
                usable[rng.below(usable.len())]
            }
        };
        let speed = out.get_pokemon_action_speed(dex, active) as f64;
        for i in pending {
            let old_id = match out.queue[i].choice {
                ActionKind::Move { move_id, .. } => move_id,
                _ => unreachable!(),
            };
            out.queue[i].speed = speed;
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
            if dex.moves.key(new_id) == "pursuit" {
                if let Some(target) = out.active_id(1 - opp) {
                    out.before_turn_callback(dex, new_id, active, target);
                }
            }
        }
    }

    // ------------------------------------------------------------- filter

    fn compute_fallback_lead_weights(&self, dex: &Dex, obs: &Observer) -> Option<[f64; 6]> {
        let prior = self.active_prior()?;
        let n = obs.mons().len();
        if !(1..=6).contains(&n) {
            return None;
        }
        let mut spec = [None; 6];
        let mut n_spec = 0usize;
        let mut sum_spec = 0.0f64;
        for (s, mo) in obs.mons().iter().enumerate() {
            if let Some(p) = prior
                .species(dex.species.key(mo.species))
                .and_then(|sp| sp.lead)
                .filter(|w| w.is_finite() && *w >= 0.0)
            {
                spec[s] = Some(p);
                n_spec += 1;
                sum_spec += p;
            }
        }
        if n_spec == 0 || spec[..n].iter().all(|x| matches!(x, Some(v) if *v == 0.0)) {
            return None;
        }
        let u = n - n_spec;
        let unspec_w = if u > 0 {
            ((1.0 - sum_spec).max(0.0) / u as f64).max(1.0 / n as f64)
        } else {
            0.0
        };
        let mut raw = [0.0f64; 6];
        for s in 0..n {
            raw[s] = spec[s].unwrap_or(unspec_w);
        }
        let sum_raw: f64 = raw[..n].iter().sum();
        if !(sum_raw > 0.0) {
            return None;
        }
        let min_w = raw[..n].iter().copied().fold(f64::INFINITY, f64::min);
        let max_w = raw[..n].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (max_w - min_w) <= 1e-12 {
            return None;
        }
        let mut lw = [0.0f64; 6];
        for s in 0..n {
            lw[s] = 0.95 * (raw[s] / sum_raw) + 0.05 / (n as f64);
        }
        Some(lw)
    }

    fn refilter(&mut self, dex: &Dex, obs: &Observer) {
        self.ruleset = obs.ruleset();
        self.alive = (0..self.cands.len())
            .filter(|&i| match &self.cands[i].refs {
                Some(refs) => consistent(dex, refs, obs.mons()),
                None => false,
            })
            .collect();
        if self.alive.is_empty() {
            self.fallback_lead_weights = self.compute_fallback_lead_weights(dex, obs);
            self.prior_governed_slots = if let Some(prior) = self.active_prior() {
                let appeared_count = obs.mons().iter().filter(|m| m.appeared).count();
                let mut governed: Vec<bool> = obs
                    .mons()
                    .iter()
                    .enumerate()
                    .map(|(slot, mo)| {
                        let Some(sp) = prior.species(dex.species.key(mo.species)) else {
                            return false;
                        };
                        let rev = distinct_revealed_moves(dex, mo);
                        let open_slots = 4usize.saturating_sub(rev.len());
                        let has_moves = open_slots > 0
                            && !legal_prior_moves(dex, sp, mo, self.ruleset).is_empty();
                        let has_items = !legal_prior_items(dex, sp, obs.mons(), slot).is_empty();
                        let has_lead = !mo.appeared
                            && appeared_count < 3
                            && sp.lead.is_some()
                            && self.fallback_lead_weights.is_some();
                        has_moves || has_items || has_lead
                    })
                    .collect();
                if !governed.iter().any(|&g| g) {
                    governed.clear();
                }
                governed
            } else {
                Vec::new()
            };
            let roster = self.build_fallback(dex, obs);
            self.loadout_draws = self.build_loadout_draws(dex, obs, &roster);
            self.fallback_cands = if !self.pinned
                && self.fallback_policy == FallbackPolicy::Bayesian
                && (obs.mons().iter().any(|m| m.appeared) || !self.prior_governed_slots.is_empty())
            {
                self.build_fallback_cands(dex, obs, &roster)
            } else {
                None
            };
            self.fallback_visible = obs
                .mons()
                .iter()
                .any(|m| m.hp_type_mask != u16::MAX || m.observed_maxhp.is_some())
                .then(|| {
                    roster
                        .iter()
                        .zip(obs.mons())
                        .map(|(p, mo)| {
                            crate::hp_evidence::project_deterministic_with_maxhp(
                                dex,
                                p,
                                mo.gender,
                                mo.hp_type_mask,
                                mo.observed_maxhp,
                            )
                            .unwrap_or_else(|| p.clone())
                        })
                        .collect()
                });
            self.fallback = Some(roster);
        } else {
            self.prior_governed_slots.clear();
            self.fallback_lead_weights = None;
            self.loadout_draws = None;
            self.fallback_cands = None;
        }
        self.synced = Some(obs.revision());
    }

    // ------------------------------------------------- M18 marginal sampling

    fn early_perish_draw(&self, dex: &Dex, mo: &MonObs, mon: &Pokemon) -> Option<LoadoutDraw> {
        let trap = dex.moves.id("meanlook")?;
        let perish = dex.moves.id("perishsong")?;
        if !mo.appeared
            || !self.pool_set_for(dex, mo).is_some_and(|s| s.moves.iter().any(|m| toid(m) == "perishsong"))
            || [trap, perish].iter().all(|&id| mon.base_move_slots.iter().any(|m| m.id == id))
            || !format_move_legal_at_level_for(
                dex.species.key(mo.species),
                "meanlook",
                mo.level,
                self.ruleset,
            )
            || !format_move_legal_at_level_for(
                dex.species.key(mo.species),
                "perishsong",
                mo.level,
                self.ruleset,
            )
        {
            return None;
        }
        let mut total = 0.0;
        let mut joint = 0.0;
        for candidate in &self.cands {
            for source in &candidate.sets {
                if dex.species.id(&toid(&source.species)) != Some(mo.species) {
                    continue;
                }
                let moves: Vec<_> = source.moves.iter().filter_map(|m| dex.moves.id(&toid(m))).collect();
                if !mo.revealed_moves.iter().all(|&r| moves.iter().any(|&m| move_matches(dex, m, r))) {
                    continue;
                }
                let weight = candidate.weight.unwrap_or(1.0);
                total += weight;
                if moves.contains(&trap) && moves.contains(&perish) {
                    joint += weight;
                }
            }
        }
        if joint <= 0.0 {
            return None;
        }
        let mut trapped = mon.clone();
        for id in [perish, trap] {
            if trapped.base_move_slots.iter().any(|m| m.id == id) {
                continue;
            }
            if trapped.base_move_slots.len() < 4 {
                trapped.base_move_slots.push(fresh_move_slot(dex, id, self.ruleset.is_castle()));
            } else {
                let slot = trapped.base_move_slots.iter().enumerate().rev().find(|(_, m)| {
                    m.id != perish && m.id != trap && !mo.revealed_moves.iter().any(|&r| move_matches(dex, m.id, r))
                })?.0;
                trapped.base_move_slots[slot] = fresh_move_slot(dex, id, self.ruleset.is_castle());
            }
        }
        trapped.move_slots = trapped.base_move_slots;
        let mut entries = vec![(trapped, joint)];
        if total > joint {
            entries.push((mon.clone(), total - joint));
        }
        Some(LoadoutDraw { entries, total })
    }

    fn build_loadout_draws(&self, dex: &Dex, obs: &Observer, roster: &[Pokemon]) -> Option<Vec<Option<LoadoutDraw>>> {
        if self.pinned {
            return None;
        }
        let has_prior = self.active_prior().is_some();
        if !has_prior
            && !matches!(
                self.fallback_policy,
                FallbackPolicy::BayesianPointMass | FallbackPolicy::Bayesian
            )
        {
            return None;
        }
        let draws: Vec<_> = obs
            .mons()
            .iter()
            .enumerate()
            .zip(roster)
            .map(|((slot, mo), mon)| match self.fallback_policy {
                FallbackPolicy::Bayesian => self.bayesian_loadout_draw(dex, mo, slot, obs.mons(), mon),
                FallbackPolicy::BayesianPointMass => self
                    .prior_loadout_draw(dex, mo, slot, obs.mons(), mon)
                    .or_else(|| self.early_perish_draw(dex, mo, mon)),
                _ => self.prior_loadout_draw(dex, mo, slot, obs.mons(), mon),
            })
            .collect();
        draws.iter().any(Option::is_some).then_some(draws)
    }

    #[inline]
    pub(crate) fn predict_plan_draws(
        &self,
        plan_pick: TeamPlanPick,
        battle: &Battle,
        opp: usize,
        rng: &mut SplitMix64,
    ) -> Option<u64> {
        match plan_pick {
            TeamPlanPick::Pool(_) => Some(0),
            TeamPlanPick::FallbackCand(idx)
                if !self.pinned
                    && self
                        .fallback_cands
                        .as_ref()
                        .is_some_and(|pool| idx < pool.entries.len()) =>
            {
                Some((idx as u64) + 1)
            }
            TeamPlanPick::FallbackCand(_) | TeamPlanPick::FallbackDefault => {
                self.predict_fallback_draws(None, battle, opp, rng)
            }
        }
    }

    /// Predicts the fallback loadout draw indices for `BlindSearch::det_cache`
    /// while advancing `rng` identically to `sample_fallback_refs`.
    #[inline]
    pub(crate) fn predict_fallback_draws(
        &self,
        pick: Option<usize>,
        battle: &Battle,
        opp: usize,
        rng: &mut SplitMix64,
    ) -> Option<u64> {
        if pick.is_some() || self.pinned {
            return Some(0);
        }
        if let Some(pool) = &self.fallback_cands {
            let idx = pool.draw_index(rng);
            return Some((idx as u64) + 1);
        }
        let active = |slot: usize| match self.fallback_policy {
            FallbackPolicy::BayesianPointMass | FallbackPolicy::Bayesian => {
                let r = &battle.sides[opp].roster[slot];
                !r.fainted && (r.is_active || battle.active_id(opp).is_none())
            }
            _ => true,
        };
        if self.loadout_draws.as_ref().is_none_or(|plans| {
            !plans
                .iter()
                .enumerate()
                .any(|(slot, plan)| plan.is_some() && active(slot))
        }) {
            return Some(0);
        }
        let fallback = self.fallback.as_deref()?;
        let mut key = 0u64;
        for slot in 0..fallback.len() {
            if !active(slot) {
                continue;
            }
            if let Some(draw) = self.loadout_draws.as_ref().and_then(|d| d.get(slot)?.as_ref()) {
                let idx = draw.draw_index(rng);
                key |= ((idx as u64 + 1) & 0xFF) << (slot * 8);
            }
        }
        Some(key)
    }

    /// The per-determinization fallback roster, or `None` to use the stored
    /// one unchanged. Consumes rng **only** when it actually samples.
    fn sample_fallback_refs(
        &self,
        pick: Option<usize>,
        rng: &mut SplitMix64,
        battle: &Battle,
        opp: usize,
    ) -> Option<Vec<Pokemon>> {
        if pick.is_some() || self.pinned {
            return None;
        }
        if let Some(pool) = &self.fallback_cands {
            let idx = pool.draw_index(rng);
            return Some(pool.entries[idx].0.clone());
        }
        let active = |slot: usize| match self.fallback_policy {
            FallbackPolicy::BayesianPointMass | FallbackPolicy::Bayesian => {
                let r = &battle.sides[opp].roster[slot];
                !r.fainted && (r.is_active || battle.active_id(opp).is_none())
            }
            _ => true,
        };
        if self.loadout_draws.as_ref().is_none_or(|plans| {
            !plans.iter().enumerate().any(|(slot, plan)| plan.is_some() && active(slot))
        }) {
            return None;
        }
        let mut roster = self.fallback.as_deref()?.to_vec();
        for (slot, mon) in roster.iter_mut().enumerate() {
            if !active(slot) {
                continue;
            }
            if let Some(draw) = self.loadout_draws.as_ref().and_then(|d| d[slot].as_ref()) {
                *mon = draw.draw(rng).clone();
            }
        }
        Some(roster)
    }
    // ----------------------------------------------------------- fallback

    /// Per-mon synthesized imputation roster for a non-pool opponent.
    fn build_fallback(&self, dex: &Dex, obs: &Observer) -> Vec<Pokemon> {
        let sets: Vec<PokemonSet> = obs
            .mons()
            .iter()
            .enumerate()
            .map(|(slot, mo)| match self.fallback_policy {
                FallbackPolicy::BayesianPointMass | FallbackPolicy::Bayesian => {
                    self.bayesian_fallback_set(dex, mo, slot, obs.mons())
                }
                _ => self.fallback_set_at(dex, mo, slot, obs.mons()),
            })
            .collect();
        let mut roster = self.build_fallback_from_sets(dex, obs, &sets);
        if self.fallback_policy == FallbackPolicy::Bayesian || self.active_prior().is_some() {
            enforce_roster_item_clause(
                dex,
                obs.mons(),
                self.active_prior(),
                &mut roster,
                obs.ruleset(),
            );
        }
        roster
    }

    fn build_fallback_from_sets(
        &self,
        dex: &Dex,
        obs: &Observer,
        sets: &[PokemonSet],
    ) -> Vec<Pokemon> {
        let ruleset = obs.ruleset();
        let mut roster =
            match Battle::from_fixture_with_ruleset(dex, "1,2,3,4", sets, sets, ruleset) {
                Ok(b) => b.sides[0].roster.clone(),
                Err(_) => {
                    // defensive second stage: minimal sets cannot fail (species /
                    // level / gender come from the live battle, no parsed strings)
                    let minimal: Vec<PokemonSet> = obs
                        .mons()
                        .iter()
                        .map(|mo| match self.fallback_policy {
                            FallbackPolicy::LegacyMetaOnly => legacy_base_set(dex, mo, ruleset),
                            FallbackPolicy::Layered
                            | FallbackPolicy::BayesianPointMass
                            | FallbackPolicy::Bayesian => base_set(dex, mo, ruleset),
                        })
                        .collect();
                    Battle::from_fixture_with_ruleset(dex, "1,2,3,4", &minimal, &minimal, ruleset)
                        .expect("minimal fallback set must construct")
                        .sides[0]
                        .roster
                        .clone()
                }
            };
        if matches!(
            self.fallback_policy,
            FallbackPolicy::Bayesian | FallbackPolicy::BayesianPointMass
        ) {
            for (slot, mon) in roster.iter_mut().enumerate() {
                if let Some(mo) = obs.mons().get(slot) {
                    let poss = mo.hp_type_mask
                        & crate::hp_evidence::possible_types_with_maxhp(
                            dex,
                            mo.species,
                            mo.level,
                            mo.gender,
                            mo.observed_maxhp,
                        );
                    let needs_hp_proj = poss.count_ones() == 1
                        && !crate::hp_evidence::consistent(dex, mon, mo.hp_type_mask);
                    let needs_maxhp_proj = mo.observed_maxhp.is_some_and(|m| mon.base_maxhp != m);
                    if needs_hp_proj || needs_maxhp_proj {
                        if let Some(proj) = crate::hp_evidence::project_deterministic_with_maxhp(
                            dex,
                            mon,
                            mo.gender,
                            mo.hp_type_mask,
                            mo.observed_maxhp,
                        ) {
                            *mon = proj;
                        }
                    }
                }
            }
        } else {
            for (slot, mon) in roster.iter_mut().enumerate() {
                if let Some(mo) = obs.mons().get(slot) {
                    if let Some(obs_hp) = mo.observed_maxhp {
                        if mon.base_maxhp != obs_hp {
                            if let Some(adj) = crate::hp_evidence::project_deterministic_with_maxhp(
                                dex,
                                mon,
                                mo.gender,
                                mo.hp_type_mask,
                                Some(obs_hp),
                            ) {
                                *mon = adj;
                            }
                        }
                    }
                }
            }
        }
        roster
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
            FallbackPolicy::BayesianPointMass | FallbackPolicy::Bayesian
                if community_rental_sets_for(dex, self.ruleset)
                    .iter()
                    .any(|s| dex.species.id(&toid(&s.species)) == Some(mo.species))
                    || supplemental_prior_teams_for(dex, self.ruleset).iter().any(|t| {
                        t.sets
                            .iter()
                            .any(|s| dex.species.id(&toid(&s.species)) == Some(mo.species))
                    }) =>
            {
                FallbackSource::CommunityRental
            }
            FallbackPolicy::Layered
                if community_rental_sets_for(dex, self.ruleset)
                    .iter()
                    .any(|s| dex.species.id(&toid(&s.species)) == Some(mo.species)) =>
            {
                FallbackSource::CommunityRental
            }
            FallbackPolicy::Layered
            | FallbackPolicy::BayesianPointMass
            | FallbackPolicy::Bayesian => FallbackSource::Learnset,
        }
    }

    /// Soft-likelihood weighted candidate sets `(set, team_weight)` for
    /// `mo.species` across `self.cands` and the supplemental/rental priors.
    ///
    /// Each candidate team $T_m$ with prior weight $\pi_m$ receives soft-likelihood
    /// weight $W_m(\mathcal{O}) = \pi_m \exp(\beta \cdot M_{\text{teammates}}(T_m, \mathcal{O}))$,
    /// where $M_{\text{teammates}}$ scores how many of the opponent's other
    /// observed roster slots match species, revealed moves, and items in $T_m$.
    fn bayesian_prior_carriers<'a>(
        &'a self,
        dex: &Dex,
        mo: &MonObs,
        slot: usize,
        obs_mons: &[MonObs],
    ) -> Vec<(&'a PokemonSet, f64)> {
        let level_aware = self.fallback_policy == FallbackPolicy::Bayesian;
        let team_soft_weight = |sets: &[PokemonSet], base_w: f64, carrier_set: &PokemonSet| -> f64 {
            let mut match_score = 0.0f64;
            for other in obs_mons {
                if other.species == mo.species {
                    continue;
                }
                if let Some(s) = sets
                    .iter()
                    .find(|s| dex.species.id(&toid(&s.species)) == Some(other.species))
                {
                    match_score += 1.0;
                    if level_aware && s.level == other.level {
                        match_score += 0.5;
                    }
                    if !other.revealed_moves.is_empty() {
                        let all_rev_match = other.revealed_moves.iter().all(|&rev| {
                            s.moves
                                .iter()
                                .filter_map(|m| dex.moves.id(&toid(m)))
                                .any(|m| move_matches(dex, m, rev))
                        });
                        match_score += if all_rev_match { 1.5 } else { -0.5 };
                    }
                    if let Some(orig) = other.item.original {
                        if dex.items.id(&toid(&s.item)) == orig {
                            match_score += 0.5;
                        }
                    }
                }
            }
            let level_mult = if level_aware {
                if carrier_set.level == mo.level {
                    1.0
                } else {
                    0.15
                }
            } else {
                1.0
            };
            let poss_hp = mo.hp_type_mask
                & crate::hp_evidence::possible_types_with_maxhp(
                    dex,
                    mo.species,
                    mo.level,
                    mo.gender,
                    mo.observed_maxhp,
                );
            let hp_mult = if carrier_set
                .moves
                .iter()
                .any(|m| explicit_hp_contradicted(dex, m, poss_hp, mo.hp_type_mask))
            {
                0.02
            } else if mo.hp_type_mask != u16::MAX {
                let explicit_hp = carrier_set.moves.iter().find_map(|m| {
                    let mid = dex.moves.id(&toid(m))?;
                    let key = dex.moves.key(mid);
                    (key != "hiddenpower" && key.starts_with("hiddenpower")).then(|| {
                        let ty = dex.moves.get(mid).move_type.as_str();
                        crate::hp_evidence::TYPES.iter().position(|t| *t == ty)
                    })?
                });
                match explicit_hp {
                    Some(idx) if (poss_hp & (1u16 << idx)) != 0 => 2.0,
                    Some(_) => 0.02,
                    None => 1.0,
                }
            } else {
                1.0
            };
            let carrier_item = dex.items.id(&toid(&carrier_set.item));
            let item_mult = if carrier_item.is_some() == mo.preview_has_item
                && !Observer::is_item_excluded_for_slot(dex, obs_mons, slot, carrier_item)
            {
                1.5
            } else {
                1.0
            };
            base_w * level_mult * hp_mult * item_mult * (0.6 * match_score).exp()
        };

        let mut out: Vec<(&'a PokemonSet, f64)> = Vec::new();
        for c in &self.cands {
            if let Some(s) = c
                .sets
                .iter()
                .find(|s| dex.species.id(&toid(&s.species)) == Some(mo.species))
            {
                let base_w = c.weight.unwrap_or(1.0);
                out.push((s, team_soft_weight(&c.sets, base_w, s)));
            }
        }
        let cands_has_species = !out.is_empty();
        if self.is_weighted_prior() || self.cands.len() >= 20 || !cands_has_species {
            for t in supplemental_prior_teams_for(dex, self.ruleset) {
                if let Some(s) = t
                    .sets
                    .iter()
                    .find(|s| dex.species.id(&toid(&s.species)) == Some(mo.species))
                {
                    out.push((s, team_soft_weight(&t.sets, t.weight, s)));
                }
            }
            for rental_sets in community_rental_teams_for(dex, self.ruleset) {
                if let Some(s) = rental_sets
                    .iter()
                    .find(|s| dex.species.id(&toid(&s.species)) == Some(mo.species))
                {
                    out.push((s, team_soft_weight(rental_sets, 0.10, s)));
                }
            }
        }
        out
    }

    fn resolve_non_excluded_item(
        &self,
        dex: &Dex,
        obs_mons: &[MonObs],
        slot: usize,
        mo: &MonObs,
        preferred: &str,
        carriers: &[(&PokemonSet, f64)],
    ) -> String {
        if let Some(known) = mo.item.original {
            return known.map(|id| dex.items.key(id).to_string()).unwrap_or_default();
        }
        if !mo.preview_has_item {
            return String::new();
        }
        let allowed = |key: &str| -> Option<String> {
            let id = dex.items.id(&toid(key))?;
            (!crate::observe::Observer::is_item_excluded_for_slot(dex, obs_mons, slot, Some(id)))
                .then(|| dex.items.key(id).to_string())
        };
        if !preferred.is_empty() {
            if let Some(k) = allowed(preferred) {
                return k;
            }
        }
        let mut by_item: Vec<(u16, String, f64)> = Vec::new();
        for &(s, w) in carriers {
            let Some(id) = dex.items.id(&toid(&s.item)) else { continue };
            if crate::observe::Observer::is_item_excluded_for_slot(dex, obs_mons, slot, Some(id)) {
                continue;
            }
            match by_item.iter_mut().find(|(iid, _, _)| *iid == id.0) {
                Some((_, _, sum)) => *sum += w,
                None => by_item.push((id.0, dex.items.key(id).to_string(), w)),
            }
        }
        by_item.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((_, k, _)) = by_item.into_iter().next() {
            return k;
        }
        for cand in item_fallback_order(self.ruleset) {
            if let Some(k) = allowed(cand) {
                return k;
            }
        }
        if let Some(k) = allowed(preferred) {
            return k;
        }
        preferred.to_string()
    }

    fn bayesian_fallback_set(
        &self,
        dex: &Dex,
        mo: &MonObs,
        slot: usize,
        obs_mons: &[MonObs],
    ) -> PokemonSet {
        let carriers = self.bayesian_prior_carriers(dex, mo, slot, obs_mons);
        let species_key = dex.species.key(mo.species);
        let prior_sp = self.active_prior().and_then(|p| p.species(species_key));
        let poss_hp = mo.hp_type_mask
            & if self.ruleset.is_castle() {
                crate::hp_evidence::possible_types_with_maxhp(
                    dex,
                    mo.species,
                    mo.level,
                    mo.gender,
                    mo.observed_maxhp,
                )
            } else {
                crate::hp_evidence::possible_types(dex, mo.species, mo.gender)
            };
        let matches_moves = |s: &PokemonSet| {
            !s.moves
                .iter()
                .any(|m| explicit_hp_contradicted(dex, m, poss_hp, mo.hp_type_mask))
                && mo.revealed_moves.iter().all(|&revealed| {
                    s.moves
                        .iter()
                        .filter_map(|m| dex.moves.id(&toid(m)))
                        .any(|m| move_matches(dex, m, revealed))
                })
        };
        let matches_item = |s: &PokemonSet| {
            let item = dex.items.id(&toid(&s.item));
            item.is_some() == mo.preview_has_item
                && match mo.item.original {
                    Some(known) => known == item,
                    None => true,
                }
        };

        let mut moves: Vec<String> = distinct_revealed_moves(dex, mo)
            .into_iter()
            .map(|m| dex.moves.key(m).to_string())
            .collect();
        let prior_items = prior_sp
            .map(|sp| legal_prior_items(dex, sp, obs_mons, slot))
            .unwrap_or_default();
        if let Some(sp) = prior_sp.filter(|sp| !sp.moves.is_empty()) {
            let mut has_hp = moves.iter().any(|m| toid(m).starts_with("hiddenpower"));
            for (_, key, _) in legal_prior_moves(dex, sp, mo, self.ruleset) {
                if moves.len() >= 4 {
                    break;
                }
                let is_hp = key.starts_with("hiddenpower");
                if is_hp && has_hp {
                    continue;
                }
                if is_hp {
                    has_hp = true;
                }
                moves.push(key);
            }
        }

        let chosen_prior: Option<&PokemonSet> = if carriers.is_empty() {
            None
        } else {
            let has_move_match = carriers.iter().any(|(s, _)| matches_moves(s));
            if has_move_match {
                let has_both = carriers
                    .iter()
                    .any(|(s, _)| matches_moves(s) && matches_item(s));
                if self.fallback_policy == FallbackPolicy::Bayesian {
                    let mut marginal_w: HashMap<String, f64> = HashMap::new();
                    let mut tally_total_w = 0.0f64;
                    // (sorted_mvs, total_weight, best_rep_weight, best_rep_set)
                    let mut tally: Vec<(Vec<String>, f64, f64, &PokemonSet)> = Vec::new();
                    for &(s, w) in &carriers {
                        if !matches_moves(s) || (has_both && !matches_item(s)) {
                            continue;
                        }
                        let mut mvs: Vec<String> = s
                            .moves
                            .iter()
                            .map(|m| toid(m))
                            .filter(|m| {
                                format_move_legal_at_level_for(species_key, m, mo.level, self.ruleset)
                                    && !explicit_hp_contradicted(dex, m, poss_hp, mo.hp_type_mask)
                            })
                            .collect();
                        mvs.sort();
                        mvs.dedup();
                        tally_total_w += w;
                        for mv in &mvs {
                            *marginal_w.entry(mv.clone()).or_insert(0.0) += w;
                        }
                        match tally.iter_mut().find(|(k, _, _, _)| *k == mvs) {
                            Some(entry) => {
                                entry.1 += w;
                                if w > entry.2 {
                                    entry.2 = w;
                                    entry.3 = s;
                                }
                            }
                            None => tally.push((mvs, w, w, s)),
                        }
                    }
                    let is_rev_mv = |mk: &str| {
                        dex.moves.id(mk).is_some_and(|id| {
                            mo.revealed_moves
                                .iter()
                                .any(|&rev| move_matches(dex, id, rev))
                        })
                    };
                    let score_entry = |mvs: &[String], exact_w: f64| -> f64 {
                        let gate = exact_w / (exact_w + 0.04 * tally_total_w.max(1e-9));
                        let unrev_marg: f64 = mvs
                            .iter()
                            .filter(|m| !is_rev_mv(m))
                            .map(|m| marginal_w.get(m).copied().unwrap_or(0.0))
                            .sum();
                        exact_w + 0.15 * gate * unrev_marg
                    };
                    tally
                        .into_iter()
                        .max_by(|a, b| {
                            let sa = score_entry(&a.0, a.1);
                            let sb = score_entry(&b.0, b.1);
                            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(_, _, _, s)| s)
                } else {
                    let loadout_key = |s: &PokemonSet| {
                        let mut mvs: Vec<String> = s
                            .moves
                            .iter()
                            .map(|m| toid(m))
                            .filter(|m| {
                                format_move_legal_at_level_for(species_key, m, mo.level, self.ruleset)
                            })
                            .collect();
                        mvs.sort();
                        (toid(&s.item), mvs)
                    };
                    let mut tally: Vec<((String, Vec<String>), f64, &PokemonSet)> = Vec::new();
                    for &(s, w) in &carriers {
                        if !matches_moves(s) || (has_both && !matches_item(s)) {
                            continue;
                        }
                        let key = loadout_key(s);
                        match tally.iter_mut().find(|(k, _, _)| *k == key) {
                            Some(entry) => entry.1 += w,
                            None => tally.push((key, w, s)),
                        }
                    }
                    tally
                        .into_iter()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(_, _, s)| s)
                }
            } else {
                // Novel move combination: score sets by exponential overlap with revealed moves
                carriers
                    .iter()
                    .max_by(|(sa, wa), (sb, wb)| {
                        let overlap_a = mo
                            .revealed_moves
                            .iter()
                            .filter(|&&rev| {
                                sa.moves
                                    .iter()
                                    .filter_map(|m| dex.moves.id(&toid(m)))
                                    .any(|m| move_matches(dex, m, rev))
                            })
                            .count() as f64;
                        let overlap_b = mo
                            .revealed_moves
                            .iter()
                            .filter(|&&rev| {
                                sb.moves
                                    .iter()
                                    .filter_map(|m| dex.moves.id(&toid(m)))
                                    .any(|m| move_matches(dex, m, rev))
                            })
                            .count() as f64;
                        let score_a = wa * (4.0 * overlap_a).exp() * if matches_item(sa) { 1.5 } else { 1.0 };
                        let score_b = wb * (4.0 * overlap_b).exp() * if matches_item(sb) { 1.5 } else { 1.0 };
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|&(s, _)| s)
            }
        };

        let mut set = chosen_prior
            .cloned()
            .unwrap_or_else(|| base_set(dex, mo, self.ruleset));
        set.level = mo.level;
        set.gender = Some(match mo.gender.as_str() {
            "" => "N".to_string(),
            g => g.to_string(),
        });
        set.name = String::new();

        let has_exact_move_match = carriers.iter().any(|(s, _)| matches_moves(s));
        if moves.len() < 4 && has_exact_move_match {
            for name in chosen_prior.map(|s| s.moves.clone()).unwrap_or_default() {
                if moves.len() >= 4 {
                    break;
                }
                let Some(id) = dex.moves.id(&toid(&name)) else { continue };
                if prior_sp.is_some()
                    && dex.moves.key(id).starts_with("hiddenpower")
                    && moves.iter().any(|m| toid(m).starts_with("hiddenpower"))
                {
                    continue;
                }
                if format_move_legal_at_level_for(species_key, dex.moves.key(id), mo.level, self.ruleset)
                    && !explicit_hp_contradicted(dex, dex.moves.key(id), poss_hp, mo.hp_type_mask)
                    && !mo.revealed_moves.iter().any(|&m| move_matches(dex, id, m))
                    && !moves.iter().any(|m| dex.moves.id(&toid(m)) == Some(id))
                {
                    moves.push(name);
                }
            }
        }

        // Conditional move co-occurrence + non-dominated learnset backoff for any remaining open slots
        // (fires when a novel move combination was revealed or when a template move
        // was level-filtered).
        if moves.len() < 4 && !carriers.is_empty() {
            let mut move_scores: Vec<(MoveId, String, f64, f64, f64)> = Vec::new();
            let mut max_overlap = 0usize;
            let mut total_carrier_w = 0.0f64;
            let mut max_overlap_w = 0.0f64;
            let castle_filter_hp_carriers = self.ruleset.is_castle()
                && carriers.iter().any(|(s, _)| {
                    !s.moves
                        .iter()
                        .any(|m| explicit_hp_contradicted(dex, m, poss_hp, mo.hp_type_mask))
                });
            for &(s, w) in &carriers {
                if castle_filter_hp_carriers
                    && s.moves
                        .iter()
                        .any(|m| explicit_hp_contradicted(dex, m, poss_hp, mo.hp_type_mask))
                {
                    continue;
                }
                total_carrier_w += w;
                let ov = mo
                    .revealed_moves
                    .iter()
                    .filter(|&&rev| {
                        s.moves
                            .iter()
                            .filter_map(|m| dex.moves.id(&toid(m)))
                            .any(|m| move_matches(dex, m, rev))
                    })
                    .count();
                if ov > max_overlap {
                    max_overlap = ov;
                    max_overlap_w = w;
                } else if ov == max_overlap {
                    max_overlap_w += w;
                }
            }
            for &(s, w) in &carriers {
                if castle_filter_hp_carriers
                    && s.moves
                        .iter()
                        .any(|m| explicit_hp_contradicted(dex, m, poss_hp, mo.hp_type_mask))
                {
                    continue;
                }
                let ov = mo
                    .revealed_moves
                    .iter()
                    .filter(|&&rev| {
                        s.moves
                            .iter()
                            .filter_map(|m| dex.moves.id(&toid(m)))
                            .any(|m| move_matches(dex, m, rev))
                    })
                    .count();
                let overlap = ov as f64;
                let set_w = w * (4.0 * overlap).exp() * if matches_item(s) { 1.5 } else { 1.0 };
                let in_max = if ov == max_overlap { w } else { 0.0 };
                for name in &s.moves {
                    let Some(id) = dex.moves.id(&toid(name)) else { continue };
                    if !format_move_legal_at_level_for(species_key, dex.moves.key(id), mo.level, self.ruleset)
                        || explicit_hp_contradicted(dex, dex.moves.key(id), poss_hp, mo.hp_type_mask)
                        || mo.revealed_moves.iter().any(|&m| move_matches(dex, id, m))
                        || moves.iter().any(|m| dex.moves.id(&toid(m)) == Some(id))
                    {
                        continue;
                    }
                    match move_scores.iter_mut().find(|(mid, _, _, _, _)| *mid == id) {
                        Some(entry) => {
                            entry.2 += set_w;
                            entry.3 += in_max;
                            entry.4 += w;
                        }
                        None => move_scores.push((id, name.clone(), set_w, in_max, w)),
                    }
                }
            }
            move_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            if self.fallback_policy == FallbackPolicy::Bayesian {
                if moves.len() + move_scores.len() < 4 {
                    let floor_scale = move_scores
                        .last()
                        .map(|x| x.2 * 0.15)
                        .unwrap_or(1.0)
                        .max(1e-4);
                    for (id, name, quality) in non_dominated_learnset_moves(dex, mo, self.ruleset) {
                        if mo.revealed_moves.iter().any(|&m| move_matches(dex, id, m))
                            || moves.iter().any(|m| dex.moves.id(&toid(m)) == Some(id))
                            || move_scores.iter().any(|(mid, _, _, _, _)| *mid == id)
                        {
                            continue;
                        }
                        move_scores.push((id, name, floor_scale * quality, 0.0, 0.0));
                    }
                    move_scores
                        .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                }
                // Prioritise essential synergy partners (>=85% conditional presence in max-overlap
                // carriers, ordered by Bayesian lift) before remaining co-occurrence fillers.
                if max_overlap_w > 0.0 && total_carrier_w > 0.0 {
                    move_scores.sort_by(|a, b| {
                        let pa_max = a.3 / max_overlap_w;
                        let pb_max = b.3 / max_overlap_w;
                        let a_ess = pa_max >= 0.85;
                        let b_ess = pb_max >= 0.85;
                        match (a_ess, b_ess) {
                            (true, true) => {
                                let la = pa_max / (a.4 / total_carrier_w).max(0.05).powf(0.3);
                                let lb = pb_max / (b.4 / total_carrier_w).max(0.05).powf(0.3);
                                lb.partial_cmp(&la).unwrap_or(std::cmp::Ordering::Equal)
                            }
                            (true, false) => std::cmp::Ordering::Less,
                            (false, true) => std::cmp::Ordering::Greater,
                            (false, false) => {
                                b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                            }
                        }
                    });
                }
            }
            for (_, name, _, _, _) in move_scores {
                if moves.len() >= 4 {
                    break;
                }
                moves.push(name);
            }
        } else if self.fallback_policy == FallbackPolicy::Bayesian && moves.len() < 4 {
            for (id, name, _) in non_dominated_learnset_moves(dex, mo, self.ruleset) {
                if moves.len() >= 4 {
                    break;
                }
                if !mo.revealed_moves.iter().any(|&m| move_matches(dex, id, m))
                    && !moves.iter().any(|m| dex.moves.id(&toid(m)) == Some(id))
                {
                    moves.push(name);
                }
            }
        }

        if moves.is_empty() {
            moves.push(legal_fallback_move(dex, mo, self.ruleset));
        }
        moves.truncate(4);
        set.moves = moves;
        apply_low_attack_spread_if_special(dex, &mut set, mo);
        let pool_item = std::mem::take(&mut set.item);
        let preferred_item = prior_items
            .first()
            .map(|(id, _)| dex.items.key(*id))
            .unwrap_or(pool_item.as_str());
        set.item = self.resolve_non_excluded_item(dex, obs_mons, slot, mo, preferred_item, &carriers);
        set
    }

    fn mon_loadout_sig(p: &Pokemon) -> (Option<u16>, Vec<u16>, [i32; 5], u8) {
        let mut mvs: Vec<u16> = p.base_move_slots.iter().map(|s| s.id.0).collect();
        mvs.sort_unstable();
        (p.item.map(|i| i.0), mvs, p.base_stored_stats, p.base_hp_type.0)
    }

    fn append_secondary_item_variant(
        dex: &Dex,
        obs_mons: &[MonObs],
        slot: usize,
        mo: &MonObs,
        carriers: &[(&PokemonSet, f64)],
        entries: &mut Vec<(Pokemon, f64)>,
    ) {
        if !mo.preview_has_item
            || mo.item.original.is_some()
            || entries.is_empty()
            || entries.len() >= 4
        {
            return;
        }
        let primary = &entries[0].0;
        let primary_item = primary.item;
        let mut by_item: Vec<(ItemId, f64)> = Vec::new();
        for &(s, w) in carriers {
            let Some(id) = dex.items.id(&toid(&s.item)) else { continue };
            if Observer::is_item_excluded_for_slot(dex, obs_mons, slot, Some(id)) {
                continue;
            }
            match by_item.iter_mut().find(|(iid, _)| *iid == id) {
                Some((_, sum)) => *sum += w,
                None => by_item.push((id, w)),
            }
        }
        by_item.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_item_w = by_item
            .iter()
            .find(|(id, _)| Some(*id) == primary_item)
            .map(|(_, w)| *w)
            .or_else(|| by_item.first().map(|(_, w)| *w))
            .unwrap_or(0.0);
        let sec = by_item.iter().find(|&&(id, _)| Some(id) != primary_item).copied();
        let (sec_id, rel_w) = if let Some((id, w)) = sec {
            let ratio = if top_item_w > 0.0 {
                (w / top_item_w).clamp(0.18, 0.45)
            } else {
                0.25
            };
            (id, ratio)
        } else {
            const ALT_STAPLES: &[&str] =
                &["leftovers", "miracleberry", "mintberry", "goldberry", "przcureberry"];
            let Some(id) = ALT_STAPLES.iter().filter_map(|k| dex.items.id(k)).find(|&id| {
                Some(id) != primary_item
                    && !Observer::is_item_excluded_for_slot(dex, obs_mons, slot, Some(id))
            }) else {
                return;
            };
            (id, 0.25)
        };
        let alt_mon = clone_mon_with_item(primary, Some(sec_id));
        let alt_sig = Self::mon_loadout_sig(&alt_mon);
        let alt_w = entries[0].1 * rel_w;
        match entries
            .iter_mut()
            .find(|(existing, _)| Self::mon_loadout_sig(existing) == alt_sig)
        {
            Some((_, w)) => *w += alt_w,
            None => entries.push((alt_mon, alt_w)),
        }
    }

    fn prior_loadout_draw(
        &self,
        dex: &Dex,
        mo: &MonObs,
        slot: usize,
        obs_mons: &[MonObs],
        fallback_mon: &Pokemon,
    ) -> Option<LoadoutDraw> {
        const MAX_SLOT_LOADOUTS: usize = 12;
        let species_key = dex.species.key(mo.species);
        let sp = self.active_prior()?.species(species_key)?;
        let rev = distinct_revealed_moves(dex, mo);
        let open_slots = 4usize.saturating_sub(rev.len());
        let legal_moves = if open_slots > 0 {
            legal_prior_moves(dex, sp, mo, self.ruleset)
        } else {
            Vec::new()
        };
        let legal_items = legal_prior_items(dex, sp, obs_mons, slot);
        let has_prior_moves = open_slots > 0 && !legal_moves.is_empty();
        let has_prior_items = !legal_items.is_empty();
        if !has_prior_moves && !has_prior_items {
            return None;
        }

        let carriers = self.bayesian_prior_carriers(dex, mo, slot, obs_mons);
        let strict_item = |source: &PokemonSet| {
            let item = dex.items.id(&toid(&source.item));
            item.is_some() == mo.preview_has_item
                && mo.item.original.is_none_or(|known| known == item)
        };
        let move_sig = |p: &Pokemon| {
            let mut mvs: Vec<u16> = p.base_move_slots.iter().map(|s| s.id.0).collect();
            mvs.sort_unstable();
            (p.item.map(|i| i.0), mvs, p.base_stored_stats, p.base_hp_type.0)
        };
        let project_consistent = |mut mon: Pokemon| -> Pokemon {
            if !crate::hp_evidence::consistent(dex, &mon, mo.hp_type_mask)
                || mo.observed_maxhp.is_some_and(|m| mon.base_maxhp != m)
            {
                if let Some(proj) = crate::hp_evidence::project_deterministic_with_maxhp(
                    dex,
                    &mon,
                    mo.gender,
                    mo.hp_type_mask,
                    mo.observed_maxhp,
                ) {
                    mon = proj;
                }
            }
            mon
        };
        let build_mon_from_moves = |move_keys: &[String], item_key: &str| -> Pokemon {
            let mut move_ids: Vec<MoveId> =
                move_keys.iter().filter_map(|k| dex.moves.id(&toid(k))).collect();
            move_ids.sort_unstable_by_key(|m| m.0);
            let best_carrier = carriers
                .iter()
                .max_by(|(sa, wa), (sb, wb)| {
                    let ov_a = sa
                        .moves
                        .iter()
                        .filter_map(|m| dex.moves.id(&toid(m)))
                        .filter(|id| move_ids.iter().any(|mid| move_matches(dex, *id, *mid)))
                        .count();
                    let ov_b = sb
                        .moves
                        .iter()
                        .filter_map(|m| dex.moves.id(&toid(m)))
                        .filter(|id| move_ids.iter().any(|mid| move_matches(dex, *id, *mid)))
                        .count();
                    ov_a.cmp(&ov_b)
                        .then_with(|| strict_item(sa).cmp(&strict_item(sb)))
                        .then_with(|| wa.partial_cmp(wb).unwrap_or(std::cmp::Ordering::Equal))
                })
                .map(|&(s, _)| s);
            let mut set = best_carrier
                .cloned()
                .unwrap_or_else(|| base_set(dex, mo, self.ruleset));
            set.level = mo.level;
            set.gender = Some(match mo.gender.as_str() {
                "" => "N".to_string(),
                g => g.to_string(),
            });
            set.name.clear();
            set.moves = move_keys.to_vec();
            apply_low_attack_spread_if_special(dex, &mut set, mo);
            set.item = item_key.to_string();

            let try_build = |mut s: PokemonSet| -> Option<Pokemon> {
                let has_explicit_typed_hp = s.moves.iter().any(|m| {
                    let k = toid(m);
                    k.starts_with("hiddenpower") && k != "hiddenpower"
                });
                let hp_ok = |mon: &Pokemon| {
                    crate::hp_evidence::consistent(dex, mon, mo.hp_type_mask)
                        || (!has_explicit_typed_hp && hp_compatible_with_obs(dex, mon, mo))
                };
                let finish = |mon: Pokemon| -> Option<Pokemon> {
                    if mon.gender != mo.gender {
                        return None;
                    }
                    let maxhp_ok = mo.observed_maxhp.is_none_or(|m| mon.base_maxhp == m);
                    if crate::hp_evidence::consistent(dex, &mon, mo.hp_type_mask) && maxhp_ok {
                        return Some(mon);
                    }
                    if (mo.hp_type_mask != 0 && mo.hp_type_mask != u16::MAX) || !maxhp_ok {
                        if let Some(proj) = crate::hp_evidence::project_deterministic_with_maxhp(
                            dex,
                            &mon,
                            mo.gender,
                            mo.hp_type_mask,
                            mo.observed_maxhp,
                        ) {
                            if proj.gender == mo.gender
                                && crate::hp_evidence::consistent(dex, &proj, mo.hp_type_mask)
                                && mo.observed_maxhp.is_none_or(|m| proj.base_maxhp == m)
                            {
                                return Some(proj);
                            }
                        }
                    }
                    if hp_ok(&mon) && maxhp_ok { Some(mon) } else { None }
                };
                if let Ok(built) = Battle::from_fixture_with_ruleset(
                    dex,
                    "1,2,3,4",
                    &[s.clone()],
                    &[s.clone()],
                    self.ruleset,
                ) {
                    if let Some(mon) = finish(built.sides[0].roster[0].clone()) {
                        return Some(mon);
                    }
                }
                s.ivs = None;
                let built = Battle::from_fixture_with_ruleset(
                    dex,
                    "1,2,3,4",
                    &[s.clone()],
                    &[s],
                    self.ruleset,
                )
                .ok()?;
                finish(built.sides[0].roster[0].clone())
            };

            let mon = if let Some(built) = try_build(set) {
                built
            } else {
                let mut cloned = clone_mon_with_item(
                    fallback_mon,
                    (!item_key.is_empty())
                        .then(|| dex.items.id(&toid(item_key)))
                        .flatten(),
                );
                let mut slots = MoveSlots::default();
                for k in move_keys {
                    if let Some(id) = dex.moves.id(&toid(k)) {
                        if slots.len() < 4 {
                            slots.push(fresh_move_slot(dex, id, self.ruleset.is_castle()));
                        }
                    }
                }
                if !slots.is_empty() {
                    cloned.base_move_slots = slots;
                    cloned.move_slots = slots;
                }
                cloned
            };
            let mon = project_consistent(mon);
            align_moves_with_reference(dex, mon, fallback_mon)
        };

        let fb_item_key = fallback_mon
            .item
            .map(|id| dex.items.key(id).to_string())
            .unwrap_or_default();
        let default_item_key = if !legal_items.is_empty() {
            dex.items.key(legal_items[0].0).to_string()
        } else {
            self.resolve_non_excluded_item(dex, obs_mons, slot, mo, &fb_item_key, &carriers)
        };

        let mut move_archetypes: Vec<(Pokemon, f64)> = Vec::new();
        if has_prior_moves {
            let rev_keys: Vec<String> = rev.iter().map(|&m| dex.moves.key(m).to_string()).collect();
            let rev_has_hp = rev_keys.iter().any(|k| k.starts_with("hiddenpower"));
            let m_cap = legal_moves.len().min(10);
            let capped = &legal_moves[..m_cap];
            let mut is_hp = [false; 10];
            let mut non_hp_count = 0usize;
            for (j, (_, k, _)) in capped.iter().enumerate() {
                is_hp[j] = k.starts_with("hiddenpower");
                if !is_hp[j] {
                    non_hp_count += 1;
                }
            }
            let max_legal_from_capped =
                non_hp_count + usize::from(!rev_has_hp && non_hp_count < m_cap);

            if max_legal_from_capped >= open_slots {
                let n_masks = 1usize << m_cap;
                let mut dp = [0.0f64; 1024];
                dp[0] = 1.0;
                for step in 0..open_slots {
                    let mut next_dp = [0.0f64; 1024];
                    for mask in 0..n_masks {
                        let w_mask = dp[mask];
                        if !(w_mask > 0.0) || mask.count_ones() as usize != step {
                            continue;
                        }
                        let mask_has_hp = rev_has_hp
                            || (0..m_cap).any(|j| (mask & (1 << j)) != 0 && is_hp[j]);
                        let mut rem_w = 0.0f64;
                        for j in 0..m_cap {
                            if (mask & (1 << j)) == 0 && !(mask_has_hp && is_hp[j]) {
                                rem_w += capped[j].2;
                            }
                        }
                        if rem_w > 0.0 {
                            for j in 0..m_cap {
                                if (mask & (1 << j)) == 0 && !(mask_has_hp && is_hp[j]) {
                                    next_dp[mask | (1 << j)] += w_mask * (capped[j].2 / rem_w);
                                }
                            }
                        }
                    }
                    dp = next_dp;
                }

                let carrier_exact_4: Vec<[MoveId; 4]> = carriers
                    .iter()
                    .filter_map(|(s, _)| {
                        if s.moves.len() != 4 {
                            return None;
                        }
                        let mut ids = [MoveId(0); 4];
                        for (i, m) in s.moves.iter().enumerate() {
                            ids[i] = dex.moves.id(&toid(m))?;
                        }
                        Some(ids)
                    })
                    .collect();

                let mut mask_weights: Vec<(usize, f64)> = Vec::new();
                for mask in 0..n_masks {
                    if mask.count_ones() as usize != open_slots || !(dp[mask] > 0.0) {
                        continue;
                    }
                    let mut w = dp[mask];
                    if !carrier_exact_4.is_empty() && rev.len() + open_slots == 4 {
                        let mut full = Vec::with_capacity(4);
                        full.extend_from_slice(&rev);
                        for j in 0..m_cap {
                            if (mask & (1 << j)) != 0 {
                                full.push(capped[j].0);
                            }
                        }
                        let matches_carrier = carrier_exact_4.iter().any(|c_ids| {
                            full.iter()
                                .all(|&fid| c_ids.iter().any(|&cid| move_matches(dex, fid, cid)))
                                && c_ids
                                    .iter()
                                    .all(|&cid| full.iter().any(|&fid| move_matches(dex, fid, cid)))
                        });
                        if matches_carrier {
                            w *= 1.20;
                        }
                    }
                    mask_weights.push((mask, w));
                }
                mask_weights.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });

                let trap_id = dex.moves.id("meanlook");
                let web_id = dex.moves.id("spiderweb");
                let perish_id = dex.moves.id("perishsong");
                let is_joint_trap_mask = |mask: usize| -> bool {
                    let has_id = |target: Option<MoveId>| -> bool {
                        target.is_some_and(|tid| {
                            rev.iter().any(|&r| r == tid)
                                || (0..m_cap)
                                    .any(|j| (mask & (1 << j)) != 0 && capped[j].0 == tid)
                        })
                    };
                    (has_id(trap_id) || has_id(web_id)) && has_id(perish_id)
                };

                let mut top_masks: Vec<(usize, f64)> = mask_weights.iter().copied().take(6).collect();
                if !top_masks.is_empty()
                    && !top_masks.iter().any(|&(m, _)| is_joint_trap_mask(m))
                {
                    if let Some(&trap_cand) =
                        mask_weights.iter().find(|&&(m, _)| is_joint_trap_mask(m))
                    {
                        if top_masks.len() >= 6 {
                            top_masks[5] = trap_cand;
                        } else {
                            top_masks.push(trap_cand);
                        }
                    }
                }

                for (mask, w) in top_masks {
                    let mut mvs = rev_keys.clone();
                    for j in 0..m_cap {
                        if (mask & (1 << j)) != 0 {
                            mvs.push(capped[j].1.clone());
                        }
                    }
                    let mon = build_mon_from_moves(&mvs, &default_item_key);
                    let sig = move_sig(&mon);
                    match move_archetypes
                        .iter_mut()
                        .find(|(existing, _)| move_sig(existing) == sig)
                    {
                        Some((_, ew)) => *ew += w,
                        None => move_archetypes.push((mon, w)),
                    }
                }
            } else {
                // Fewer distinct non-conflicting legal prior moves than open slots:
                // lock all legal prior moves first and backfill only remaining slots.
                let mut locked_ids = rev.clone();
                let mut locked_keys = rev_keys.clone();
                let mut has_hp = rev_has_hp;
                for &(id, ref key, _) in &legal_moves {
                    if locked_ids.len() >= 4 {
                        break;
                    }
                    let hp = key.starts_with("hiddenpower");
                    if hp && has_hp {
                        continue;
                    }
                    if hp {
                        has_hp = true;
                    }
                    locked_ids.push(id);
                    locked_keys.push(key.clone());
                }
                let rem_slots = 4usize.saturating_sub(locked_ids.len());
                if rem_slots == 0 {
                    move_archetypes
                        .push((build_mon_from_moves(&locked_keys, &default_item_key), 1.0));
                } else {
                    let mut fill_scores: Vec<(MoveId, String, f64)> = Vec::new();
                    for &(s, w) in &carriers {
                        let c_ids: Vec<MoveId> = s
                            .moves
                            .iter()
                            .filter_map(|m| dex.moves.id(&toid(m)))
                            .collect();
                        let ov = locked_ids
                            .iter()
                            .filter(|&&lid| c_ids.iter().any(|&cid| move_matches(dex, cid, lid)))
                            .count() as f64;
                        let set_w =
                            w * (4.0 * ov).exp() * if strict_item(s) { 1.5 } else { 1.0 };
                        for id in c_ids {
                            let mk = dex.moves.key(id);
                            if !format_move_legal_at_level_for(species_key, mk, mo.level, self.ruleset)
                                || locked_ids.iter().any(|&lid| move_matches(dex, id, lid))
                                || (has_hp && mk.starts_with("hiddenpower"))
                            {
                                continue;
                            }
                            match fill_scores.iter_mut().find(|(mid, _, _)| *mid == id) {
                                Some(e) => e.2 += set_w,
                                None => fill_scores.push((id, mk.to_string(), set_w)),
                            }
                        }
                    }
                    fill_scores.sort_by(|a, b| {
                        b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if fill_scores.len() < rem_slots + 2 {
                        let floor_scale = fill_scores
                            .last()
                            .map(|x| x.2 * 0.15)
                            .unwrap_or(1.0)
                            .max(1e-4);
                        for (id, name, quality) in non_dominated_learnset_moves(dex, mo, self.ruleset) {
                            let mk = dex.moves.key(id);
                            if locked_ids.iter().any(|&lid| move_matches(dex, id, lid))
                                || (has_hp && mk.starts_with("hiddenpower"))
                                || fill_scores.iter().any(|(mid, _, _)| *mid == id)
                            {
                                continue;
                            }
                            fill_scores.push((id, name, floor_scale * quality));
                        }
                        fill_scores.sort_by(|a, b| {
                            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    let pool_len = fill_scores.len().min(rem_slots + 3);
                    if pool_len >= rem_slots {
                        let mut combos: Vec<(Vec<usize>, f64)> = Vec::new();
                        for mask in 0..(1usize << pool_len) {
                            if mask.count_ones() as usize != rem_slots {
                                continue;
                            }
                            let idxs: Vec<usize> =
                                (0..pool_len).filter(|&i| (mask & (1 << i)) != 0).collect();
                            let w: f64 = idxs.iter().map(|&i| fill_scores[i].2).product();
                            if w > 0.0 {
                                combos.push((idxs, w));
                            }
                        }
                        combos.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        for (idxs, w) in combos.into_iter().take(3) {
                            let mut mvs = locked_keys.clone();
                            for i in idxs {
                                mvs.push(fill_scores[i].1.clone());
                            }
                            let mon = build_mon_from_moves(&mvs, &default_item_key);
                            let sig = move_sig(&mon);
                            match move_archetypes
                                .iter_mut()
                                .find(|(existing, _)| move_sig(existing) == sig)
                            {
                                Some((_, ew)) => *ew += w,
                                None => move_archetypes.push((mon, w)),
                            }
                        }
                    }
                    if move_archetypes.is_empty() {
                        move_archetypes
                            .push((project_consistent(fallback_mon.clone()), 1.0));
                    }
                }
            }
        }

        let mut entries: Vec<(Pokemon, f64)> = Vec::new();
        if !legal_items.is_empty() {
            if move_archetypes.is_empty() {
                move_archetypes.push((project_consistent(fallback_mon.clone()), 1.0));
            }
            move_archetypes.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let k_it = legal_items.len().min(4);
            let top_items = &legal_items[..k_it];
            let item_sum: f64 = top_items.iter().map(|(_, w)| *w).sum();
            if !(item_sum > 0.0) {
                return None;
            }
            let (ref mon0, mw0) = move_archetypes[0];
            for &(item_id, iw) in top_items {
                let cand = clone_mon_with_item(mon0, Some(item_id));
                let w = mw0 * (iw / item_sum);
                let sig = move_sig(&cand);
                match entries
                    .iter_mut()
                    .find(|(existing, _)| move_sig(existing) == sig)
                {
                    Some((_, ew)) => *ew += w,
                    None => entries.push((cand, w)),
                }
            }
            let mut extra_pairs: Vec<(Pokemon, f64)> = Vec::new();
            for (mon_i, mw_i) in move_archetypes.iter().skip(1) {
                for &(item_id, iw) in top_items {
                    extra_pairs.push((
                        clone_mon_with_item(mon_i, Some(item_id)),
                        *mw_i * (iw / item_sum),
                    ));
                }
            }
            extra_pairs.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (cand, w) in extra_pairs.into_iter().take(MAX_SLOT_LOADOUTS.saturating_sub(k_it)) {
                let sig = move_sig(&cand);
                match entries
                    .iter_mut()
                    .find(|(existing, _)| move_sig(existing) == sig)
                {
                    Some((_, ew)) => *ew += w,
                    None => entries.push((cand, w)),
                }
            }
            entries.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            entries = move_archetypes;
            entries.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            Self::append_secondary_item_variant(dex, obs_mons, slot, mo, &carriers, &mut entries);
        }

        let total: f64 = entries.iter().map(|(_, w)| *w).sum();
        (!entries.is_empty() && total > 0.0).then_some(LoadoutDraw { entries, total })
    }

    fn bayesian_loadout_draw(
        &self,
        dex: &Dex,
        mo: &MonObs,
        slot: usize,
        obs_mons: &[MonObs],
        fallback_mon: &Pokemon,
    ) -> Option<LoadoutDraw> {
        if let Some(draw) = self.prior_loadout_draw(dex, mo, slot, obs_mons, fallback_mon) {
            return Some(draw);
        }
        if !obs_mons.iter().any(|m| m.appeared) {
            return None;
        }
        let carriers = self.bayesian_prior_carriers(dex, mo, slot, obs_mons);
        if carriers.is_empty() && !self.is_weighted_prior() && self.cands.len() < 20 {
            return None;
        }
        let species_key = dex.species.key(mo.species);
        let strict_item = |source: &PokemonSet| {
            let item = dex.items.id(&toid(&source.item));
            item.is_some() == mo.preview_has_item
                && mo.item.original.is_none_or(|known| known == item)
        };
        let resolve_item = |source_item: &str, _use_strict: bool| -> String {
            self.resolve_non_excluded_item(dex, obs_mons, slot, mo, source_item, &carriers)
        };
        let move_sig = |p: &Pokemon| {
            let mut mvs: Vec<u16> = p.base_move_slots.iter().map(|s| s.id.0).collect();
            mvs.sort_unstable();
            (p.item.map(|i| i.0), mvs, p.base_stored_stats, p.base_hp_type.0)
        };
        let fallback_sig = move_sig(fallback_mon);
        let build_candidate_mon = |mut set: PokemonSet| -> Option<Pokemon> {
            set.level = mo.level;
            set.name.clear();
            let gender_str = match mo.gender.as_str() {
                "" => "N".to_string(),
                g => g.to_string(),
            };
            let has_explicit_typed_hp = set.moves.iter().any(|m| {
                let k = toid(m);
                k.starts_with("hiddenpower") && k != "hiddenpower"
            });
            let hp_ok = |mon: &Pokemon| {
                crate::hp_evidence::consistent(dex, mon, mo.hp_type_mask)
                    || (!has_explicit_typed_hp && hp_compatible_with_obs(dex, mon, mo))
            };
            let finish = |mon: Pokemon| -> Option<Pokemon> {
                if mon.gender != mo.gender {
                    return None;
                }
                let maxhp_ok = mo.observed_maxhp.is_none_or(|m| mon.base_maxhp == m);
                if hp_ok(&mon) && maxhp_ok {
                    return Some(mon);
                }
                if (mo.hp_type_mask != 0 && mo.hp_type_mask != u16::MAX) || !maxhp_ok {
                    if let Some(proj) = crate::hp_evidence::project_deterministic_with_maxhp(
                        dex,
                        &mon,
                        mo.gender,
                        mo.hp_type_mask,
                        mo.observed_maxhp,
                    ) {
                        if proj.gender == mo.gender
                            && hp_ok(&proj)
                            && mo.observed_maxhp.is_none_or(|m| proj.base_maxhp == m)
                        {
                            return Some(proj);
                        }
                    }
                }
                None
            };
            if let Ok(built) = Battle::from_fixture_with_ruleset(
                dex,
                "1,2,3,4",
                &[set.clone()],
                &[set.clone()],
                self.ruleset,
            ) {
                if let Some(mon) = finish(built.sides[0].roster[0].clone()) {
                    return Some(mon);
                }
            }
            set.gender = Some(gender_str);
            set.ivs = None;
            let built = Battle::from_fixture_with_ruleset(
                dex,
                "1,2,3,4",
                &[set.clone()],
                &[set],
                self.ruleset,
            )
            .ok()?;
            finish(built.sides[0].roster[0].clone())
        };
        let add_secondary_item_variant = |entries: &mut Vec<(Pokemon, f64)>| {
            Self::append_secondary_item_variant(dex, obs_mons, slot, mo, &carriers, entries);
        };

        let poss_hp = mo.hp_type_mask
            & crate::hp_evidence::possible_types_with_maxhp(
                dex,
                mo.species,
                mo.level,
                mo.gender,
                mo.observed_maxhp,
            );
        let move_consistent: Vec<(&PokemonSet, f64)> = carriers
            .iter()
            .copied()
            .filter(|(source, _)| {
                !source.moves.is_empty()
                    && source.moves.len() <= 4
                    && source.moves.iter().all(|m| {
                        let k = toid(m);
                        format_move_legal_at_level_for(species_key, &k, mo.level, self.ruleset)
                            && !explicit_hp_contradicted(dex, &k, poss_hp, mo.hp_type_mask)
                    })
                    && mo.revealed_moves.iter().all(|&revealed| {
                        source
                            .moves
                            .iter()
                            .filter_map(|m| dex.moves.id(&toid(m)))
                            .any(|m| move_matches(dex, m, revealed))
                    })
            })
            .collect();

        let mut entries: Vec<(Pokemon, f64)> = Vec::new();

        if !move_consistent.is_empty() {
            // Tier A: Aggregate soft-likelihood weights by canonical sorted 4-move set
            // before constructing Pokemon fixtures, keeping diverse archetypes and a
            // secondary non-excluded item variant for root multi-world hedging.
            let use_strict_item = move_consistent.iter().any(|(s, _)| strict_item(s));
            let mut marginal_w: HashMap<String, f64> = HashMap::new();
            let mut tally_total_w = 0.0f64;
            // (sorted_moves, total_weight, best_rep_weight, best_rep_set)
            let mut tally: Vec<(Vec<String>, f64, f64, &PokemonSet)> = Vec::new();
            for &(source, weight) in &move_consistent {
                if use_strict_item && !strict_item(source) {
                    continue;
                }
                let mut mvs: Vec<String> = source.moves.iter().map(|m| toid(m)).collect();
                mvs.sort();
                mvs.dedup();
                tally_total_w += weight;
                for mv in &mvs {
                    *marginal_w.entry(mv.clone()).or_insert(0.0) += weight;
                }
                match tally.iter_mut().find(|(k, _, _, _)| *k == mvs) {
                    Some(entry) => {
                        entry.1 += weight;
                        if weight > entry.2 {
                            entry.2 = weight;
                            entry.3 = source;
                        }
                    }
                    None => tally.push((mvs, weight, weight, source)),
                }
            }
            let is_rev_mv = |mk: &str| {
                dex.moves.id(mk).is_some_and(|id| {
                    mo.revealed_moves
                        .iter()
                        .any(|&rev| move_matches(dex, id, rev))
                })
            };
            for (mvs, exact_w, _, _) in &mut tally {
                if !mvs.is_empty() {
                    let gate = *exact_w / (*exact_w + 0.04 * tally_total_w.max(1e-9));
                    let unrev_marg: f64 = mvs
                        .iter()
                        .filter(|m| !is_rev_mv(m))
                        .map(|m| marginal_w.get(m).copied().unwrap_or(0.0))
                        .sum::<f64>();
                    *exact_w += 0.15 * gate * unrev_marg;
                }
            }
            tally.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((top_mvs, top_w, _, top_rep)) = tally.first().cloned() {
                let locked_staples: Vec<String> =
                    if !mo.revealed_moves.is_empty() && tally_total_w > 0.0 {
                        top_mvs
                            .iter()
                            .filter(|m| {
                                let mid = dex.moves.id(m);
                                mid.is_none_or(|id| {
                                    !mo.revealed_moves
                                        .iter()
                                        .any(|&rev| move_matches(dex, id, rev))
                                }) && marginal_w.get(*m).copied().unwrap_or(0.0)
                                    >= 0.85 * tally_total_w
                            })
                            .cloned()
                            .collect()
                    } else {
                        Vec::new()
                    };
                let is_joint_trap = |mvs: &[String]| {
                    mvs.iter().any(|m| m == "meanlook" || m == "spiderweb")
                        && mvs.iter().any(|m| m == "perishsong")
                };
                let mut selected: Vec<(Vec<String>, f64, &PokemonSet)> =
                    vec![(top_mvs.clone(), top_w, top_rep)];
                let max_tier_a = if mo.appeared { 4 } else { 3 };
                for (mvs, w, _, rep) in tally.iter().skip(1) {
                    if selected.len() >= max_tier_a {
                        break;
                    }
                    let diff_moves = mvs.iter().filter(|m| !top_mvs.contains(m)).count();
                    let keeps_staples = locked_staples.iter().all(|req| mvs.contains(req));
                    if *w >= top_w * 0.15 && diff_moves >= 1 && keeps_staples {
                        selected.push((mvs.clone(), *w, *rep));
                    }
                }

                // Ensure Mean Look + Perish Song joint trap mass is never pruned away
                // if present in consistent carriers.
                if !selected.iter().any(|(mvs, _, _)| is_joint_trap(mvs)) {
                    let trap_total: f64 = tally
                        .iter()
                        .filter(|(mvs, _, _, _)| is_joint_trap(mvs))
                        .map(|(_, w, _, _)| *w)
                        .sum();
                    if trap_total > 0.0 {
                        if let Some((k, _, _, rep)) =
                            tally.iter().find(|(mvs, _, _, _)| is_joint_trap(mvs))
                        {
                            if selected.len() >= 4 {
                                selected.pop();
                            }
                            selected.push((k.clone(), trap_total, *rep));
                        }
                    }
                }

                for (_, weight, source) in selected {
                    let mut set = source.clone();
                    set.item = resolve_item(&source.item, use_strict_item);
                    let Some(mon) = build_candidate_mon(set) else {
                        continue;
                    };
                    let sig = move_sig(&mon);
                    let mon = if sig == fallback_sig {
                        fallback_mon.clone()
                    } else {
                        align_moves_with_reference(dex, mon, fallback_mon)
                    };
                    match entries.iter_mut().find(|(existing, _)| move_sig(existing) == sig) {
                        Some((_, w)) => *w += weight,
                        None => entries.push((mon, weight)),
                    }
                }
                add_secondary_item_variant(&mut entries);
            }
        } else if mo.revealed_moves.len() < 4 {
            // Tier B: Novel move combination (or unseen prior species) — conditional
            // move co-occurrence + non-dominated learnset backoff over open slots.
            let base_moves: Vec<String> = distinct_revealed_moves(dex, mo)
                .into_iter()
                .map(|m| dex.moves.key(m).to_string())
                .collect();
            let open_slots = 4usize.saturating_sub(base_moves.len());
            let mut max_overlap = 0usize;
            let mut total_carrier_w = 0.0f64;
            let mut max_overlap_w = 0.0f64;
            let castle_filter_hp_carriers = self.ruleset.is_castle()
                && carriers.iter().any(|(s, _)| {
                    !s.moves
                        .iter()
                        .any(|m| explicit_hp_contradicted(dex, m, poss_hp, mo.hp_type_mask))
                });
            for &(s, w) in &carriers {
                if castle_filter_hp_carriers
                    && s.moves
                        .iter()
                        .any(|m| explicit_hp_contradicted(dex, m, poss_hp, mo.hp_type_mask))
                {
                    continue;
                }
                total_carrier_w += w;
                let ov = mo
                    .revealed_moves
                    .iter()
                    .filter(|&&rev| {
                        s.moves
                            .iter()
                            .filter_map(|m| dex.moves.id(&toid(m)))
                            .any(|m| move_matches(dex, m, rev))
                    })
                    .count();
                if ov > max_overlap {
                    max_overlap = ov;
                    max_overlap_w = w;
                } else if ov == max_overlap {
                    max_overlap_w += w;
                }
            }

            // (MoveId, display_name, co_occurrence_score, w_in_max_overlap, w_in_all)
            let mut move_scores: Vec<(MoveId, String, f64, f64, f64)> = Vec::new();
            for &(s, w) in &carriers {
                if castle_filter_hp_carriers
                    && s.moves
                        .iter()
                        .any(|m| explicit_hp_contradicted(dex, m, poss_hp, mo.hp_type_mask))
                {
                    continue;
                }
                let ov = mo
                    .revealed_moves
                    .iter()
                    .filter(|&&rev| {
                        s.moves
                            .iter()
                            .filter_map(|m| dex.moves.id(&toid(m)))
                            .any(|m| move_matches(dex, m, rev))
                    })
                    .count();
                let overlap = ov as f64;
                let set_w = w * (4.0 * overlap).exp() * if strict_item(s) { 1.5 } else { 1.0 };
                for name in &s.moves {
                    let Some(id) = dex.moves.id(&toid(name)) else { continue };
                    if !format_move_legal_at_level_for(species_key, dex.moves.key(id), mo.level, self.ruleset)
                        || explicit_hp_contradicted(dex, dex.moves.key(id), poss_hp, mo.hp_type_mask)
                        || mo.revealed_moves.iter().any(|&m| move_matches(dex, id, m))
                    {
                        continue;
                    }
                    let in_max = if ov == max_overlap { w } else { 0.0 };
                    match move_scores.iter_mut().find(|(mid, _, _, _, _)| *mid == id) {
                        Some(entry) => {
                            entry.2 += set_w;
                            entry.3 += in_max;
                            entry.4 += w;
                        }
                        None => move_scores.push((id, name.clone(), set_w, in_max, w)),
                    }
                }
            }
            move_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

            if move_scores.len() < (open_slots + 2).min(6) {
                let floor_scale = move_scores
                    .last()
                    .map(|x| x.2 * 0.15)
                    .unwrap_or(1.0)
                    .max(1e-4);
                for (id, name, quality) in non_dominated_learnset_moves(dex, mo, self.ruleset) {
                    if mo.revealed_moves.iter().any(|&m| move_matches(dex, id, m))
                        || move_scores.iter().any(|(mid, _, _, _, _)| *mid == id)
                    {
                        continue;
                    }
                    move_scores.push((id, name, floor_scale * quality, 0.0, 0.0));
                }
                move_scores
                    .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            }

            let pool_len = move_scores.len().min(open_slots + 3);
            if pool_len > open_slots {
                let template_set = carriers
                    .iter()
                    .max_by(|(sa, wa), (sb, wb)| {
                        let ov_a = mo
                            .revealed_moves
                            .iter()
                            .filter(|&&rev| {
                                sa.moves
                                    .iter()
                                    .filter_map(|m| dex.moves.id(&toid(m)))
                                    .any(|m| move_matches(dex, m, rev))
                            })
                            .count() as f64;
                        let ov_b = mo
                            .revealed_moves
                            .iter()
                            .filter(|&&rev| {
                                sb.moves
                                    .iter()
                                    .filter_map(|m| dex.moves.id(&toid(m)))
                                    .any(|m| move_matches(dex, m, rev))
                            })
                            .count() as f64;
                        let sc_a = wa * (4.0 * ov_a).exp() * if strict_item(sa) { 1.5 } else { 1.0 };
                        let sc_b = wb * (4.0 * ov_b).exp() * if strict_item(sb) { 1.5 } else { 1.0 };
                        sc_a.partial_cmp(&sc_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|&(s, _)| s.clone())
                    .unwrap_or_else(|| base_set(dex, mo, self.ruleset));

                // Identify moves in the MAP prefix that are near-universal staples or essential
                // synergy partners of the matched revealed moves (>= 85% conditional presence in
                // max-overlap carriers), locking up to `open_slots - 1` of them in priority order
                // of Bayesian lift over unconditional carrier frequency.
                let locked_partners: Vec<usize> =
                    if open_slots >= 2 && max_overlap_w > 0.0 && total_carrier_w > 0.0 {
                        let mut cands: Vec<(usize, f64)> = (0..open_slots.min(pool_len))
                            .filter_map(|i| {
                                let p_max = move_scores[i].3 / max_overlap_w;
                                let p_uncond = (move_scores[i].4 / total_carrier_w).max(0.05);
                                (p_max >= 0.85).then_some((i, p_max / p_uncond.powf(0.3)))
                            })
                            .collect();
                        cands.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        cands
                            .into_iter()
                            .take(open_slots - 1)
                            .map(|(i, _)| i)
                            .collect()
                    } else {
                        Vec::new()
                    };

                let mut combos: Vec<(Vec<usize>, f64)> = Vec::new();
                for mask in 0..(1usize << pool_len) {
                    if mask.count_ones() as usize != open_slots {
                        continue;
                    }
                    let idxs: Vec<usize> =
                        (0..pool_len).filter(|&i| (mask & (1 << i)) != 0).collect();
                    let w: f64 = idxs.iter().map(|&i| move_scores[i].2).product();
                    if w > 0.0 {
                        combos.push((idxs, w));
                    }
                }
                combos.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                if let Some(best_w) = combos.first().map(|x| x.1) {
                    let max_novel = if mo.appeared { 3 } else { 2 };
                    for (idxs, w) in combos
                        .into_iter()
                        .filter(|(idxs, w)| {
                            *w >= best_w * 0.20
                                && locked_partners.iter().all(|req| idxs.contains(req))
                        })
                        .take(max_novel)
                    {
                        let mut set = template_set.clone();
                        let mut mvs = base_moves.clone();
                        for idx in idxs {
                            mvs.push(move_scores[idx].1.clone());
                        }
                        mvs.truncate(4);
                        set.moves = mvs;
                        apply_low_attack_spread_if_special(dex, &mut set, mo);
                        let pool_item = std::mem::take(&mut set.item);
                        set.item = resolve_item(&pool_item, false);
                        let Some(mon) = build_candidate_mon(set) else {
                            continue;
                        };
                        let sig = move_sig(&mon);
                        let mon = if sig == fallback_sig {
                            fallback_mon.clone()
                        } else {
                            align_moves_with_reference(dex, mon, fallback_mon)
                        };
                        match entries
                            .iter_mut()
                            .find(|(existing, _)| move_sig(existing) == sig)
                        {
                            Some((_, ew)) => *ew += w,
                            None => entries.push((mon, w)),
                        }
                    }
                    add_secondary_item_variant(&mut entries);
                }
            }
        } else if mo.revealed_moves.len() == 4 && mo.preview_has_item && mo.item.original.is_none()
        {
            entries.push((fallback_mon.clone(), 1.0));
            add_secondary_item_variant(&mut entries);
        }

        if self.fallback_policy == FallbackPolicy::Bayesian && mo.appeared && entries.len() < 4 {
            if entries.is_empty() && mo.revealed_moves.len() == 4 {
                entries.push((fallback_mon.clone(), 1.0));
            }
            if !entries.is_empty() && entries.len() < 4 {
                let primary = &entries[0].0;
                if primary.set_evs[5] > 0 || primary.set_ivs[5] > 2 {
                    let base_spe = dex.species.get(mo.species).base_stats.spe;
                    let has_slow_move = primary.base_move_slots.iter().any(|m| {
                        let k = dex.moves.key(m.id);
                        matches!(k, "counter" | "mirrorcoat")
                            || (k == "batonpass" && base_spe <= 75)
                    });
                    if has_slow_move || (mo.revealed_moves.len() == 4 && entries.len() <= 2) {
                        let slow_mon = with_reduced_speed_spread(dex, primary);
                        let slow_sig = move_sig(&slow_mon);
                        if !entries.iter().any(|(existing, _)| move_sig(existing) == slow_sig) {
                            let rel_w = if has_slow_move { 0.12 } else { 0.06 };
                            let w = entries[0].1 * rel_w;
                            entries.push((slow_mon, w));
                        }
                    }
                }
            }
        }

        if entries.is_empty()
            || (entries.len() == 1 && move_sig(&entries[0].0) == fallback_sig)
        {
            return self.early_perish_draw(dex, mo, fallback_mon);
        }
        let total = entries.iter().map(|(_, w)| *w).sum();
        Some(LoadoutDraw { entries, total })
    }

    fn build_fallback_cands(
        &self,
        dex: &Dex,
        obs: &Observer,
        roster: &[Pokemon],
    ) -> Option<FallbackCandPool> {
        let obs_mons = obs.mons();
        let n_mons = obs_mons.len().min(roster.len());
        if n_mons == 0 {
            return None;
        }
        let slot_opts: Vec<Vec<(Pokemon, f64)>> = (0..n_mons)
            .map(|slot| {
                let project_if_prior = |p: Pokemon| -> Pokemon {
                    if (!self.prior_governed_slots.is_empty()
                        && !crate::hp_evidence::consistent(dex, &p, obs_mons[slot].hp_type_mask))
                        || obs_mons[slot].observed_maxhp.is_some_and(|m| p.base_maxhp != m)
                    {
                        crate::hp_evidence::project_deterministic_with_maxhp(
                            dex,
                            &p,
                            obs_mons[slot].gender,
                            obs_mons[slot].hp_type_mask,
                            obs_mons[slot].observed_maxhp,
                        )
                        .unwrap_or(p)
                    } else {
                        p
                    }
                };
                let mut opts: Vec<(Pokemon, f64)> = self
                    .loadout_draws
                    .as_ref()
                    .and_then(|d| d.get(slot)?.as_ref())
                    .map(|ld| {
                        ld.entries
                            .iter()
                            .filter(|(p, w)| {
                                *w > 0.0
                                    && !Observer::is_item_excluded_for_slot(
                                        dex, obs_mons, slot, p.item,
                                    )
                            })
                            .map(|(p, w)| {
                                (
                                    project_if_prior(align_moves_with_reference(
                                        dex,
                                        p.clone(),
                                        &roster[slot],
                                    )),
                                    *w,
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if opts.is_empty() {
                    opts.push((project_if_prior(roster[slot].clone()), 1.0));
                }
                opts
            })
            .collect();

        // Source 1: Near-Pool Soft Shell Repair (matches >= n_mons - 1 slots on species+level)
        let mut source1: Vec<(Vec<Pokemon>, f64, usize)> = Vec::new();
        if n_mons >= 2 && obs_mons.iter().any(|m| m.appeared) {
            let min_shell = n_mons.saturating_sub(1).max(2);
            let mut raw_cands: Vec<(f64, &[PokemonSet], Option<&[Pokemon]>)> = Vec::new();
            let extra_iter = (self.is_weighted_prior() || self.cands.len() >= 20)
                .then(|| {
                    supplemental_prior_teams_for(dex, obs.ruleset())
                        .iter()
                        .map(|t| (t.weight, t.sets.as_slice(), None))
                })
                .into_iter()
                .flatten();
            for (base_w, sets, prebuilt) in self
                .cands
                .iter()
                .map(|c| (c.weight.unwrap_or(1.0), c.sets.as_slice(), c.refs.as_deref()))
                .chain(extra_iter)
            {
                let mut used_set = vec![false; sets.len()];
                let mut shell_matches = 0usize;
                for mo in obs_mons.iter().take(n_mons) {
                    if let Some(idx) = sets.iter().enumerate().position(|(i, s)| {
                        !used_set[i]
                            && s.level == mo.level
                            && dex.species.id(&toid(&s.species)) == Some(mo.species)
                    }) {
                        used_set[idx] = true;
                        shell_matches += 1;
                    }
                }
                if shell_matches >= min_shell {
                    let score = base_w * (1.2 * shell_matches as f64).exp();
                    raw_cands.push((score, sets, prebuilt));
                }
            }
            raw_cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            raw_cands.truncate(16);

            let move_item_sig = |p: &Pokemon| -> (Option<u16>, Vec<u16>) {
                let mut mvs: Vec<u16> = p.base_move_slots.iter().map(|s| s.id.0).collect();
                mvs.sort_unstable();
                (p.item.map(|i| i.0), mvs)
            };
            let prior_slot_sigs: Vec<Vec<(Option<u16>, Vec<u16>)>> = (0..n_mons)
                .map(|slot| {
                    let governed = self.prior_governed_slots.get(slot).copied().unwrap_or(false)
                        && self
                            .loadout_draws
                            .as_ref()
                            .and_then(|d| d.get(slot)?.as_ref())
                            .is_some();
                    if governed {
                        slot_opts[slot]
                            .iter()
                            .map(|(p, _)| move_item_sig(p))
                            .collect()
                    } else {
                        Vec::new()
                    }
                })
                .collect();
            let mon_consistent_for_slot = |slot: usize, p: &Pokemon| -> bool {
                let mo = &obs_mons[slot];
                p.species == mo.species
                    && p.level == mo.level
                    && p.gender == mo.gender
                    && mo.observed_maxhp.is_none_or(|m| p.base_maxhp == m)
                    && p.item.is_some() == mo.preview_has_item
                    && mo.item.original.is_none_or(|known| known == p.item)
                    && !Observer::is_item_excluded_for_slot(dex, obs_mons, slot, p.item)
                    && crate::hp_evidence::consistent(dex, p, mo.hp_type_mask)
                    && mo
                        .revealed_moves
                        .iter()
                        .all(|&rev| p.base_move_slots.iter().any(|s| move_matches(dex, s.id, rev)))
                    && (prior_slot_sigs[slot].is_empty()
                        || prior_slot_sigs[slot].contains(&move_item_sig(p)))
            };

            let mut s1_sigs: Vec<TeamSig> = Vec::new();
            let min_consistent = n_mons.saturating_sub(2).max(2);
            for (raw_score, sets, prebuilt) in raw_cands {
                let owned_roster;
                let team_roster: &[Pokemon] = if let Some(refs) = prebuilt {
                    refs
                } else if let Ok(b) =
                    Battle::from_fixture_with_ruleset(dex, "1,2,3,4", sets, sets, obs.ruleset())
                {
                    owned_roster = b.sides[0].roster.clone();
                    &owned_roster
                } else {
                    continue;
                };
                let mut used_src = vec![false; team_roster.len()];
                let mut matched_idx: Vec<Option<usize>> = vec![None; n_mons];
                for slot in 0..n_mons {
                    if let Some(i) = team_roster
                        .iter()
                        .enumerate()
                        .position(|(i, p)| !used_src[i] && mon_consistent_for_slot(slot, p))
                    {
                        used_src[i] = true;
                        matched_idx[slot] = Some(i);
                    }
                }
                let k_consistent = matched_idx.iter().filter(|m| m.is_some()).count();
                if k_consistent < min_consistent {
                    continue;
                }

                let mut used_items: Vec<ItemId> = Vec::with_capacity(n_mons);
                for mo in obs_mons.iter().take(n_mons) {
                    if let Some(Some(id)) = mo.item.original {
                        if !used_items.contains(&id) {
                            used_items.push(id);
                        }
                    }
                }
                let mut repaired: Vec<Option<Pokemon>> = vec![None; n_mons];
                for slot in 0..n_mons {
                    let Some(idx) = matched_idx[slot] else { continue };
                    let p = &team_roster[idx];
                    let item_ok = obs_mons[slot].item.original.is_some()
                        || p.item.is_none_or(|id| !used_items.contains(&id));
                    if item_ok {
                        if obs_mons[slot].item.original.is_none() {
                            if let Some(id) = p.item {
                                used_items.push(id);
                            }
                        }
                        repaired[slot] =
                            Some(align_moves_with_reference(dex, p.clone(), &roster[slot]));
                    }
                }
                let mut repair_slots: Vec<usize> =
                    (0..n_mons).filter(|&s| repaired[s].is_none()).collect();
                repair_slots.sort_by_key(|&s| (!obs_mons[s].appeared, s));
                for slot in repair_slots {
                    let mo = &obs_mons[slot];
                    let chosen = slot_opts[slot]
                        .iter()
                        .find(|(p, _)| {
                            !Observer::is_item_excluded_for_slot(dex, obs_mons, slot, p.item)
                                && (mo.item.original.is_some()
                                    || p.item.is_none_or(|id| !used_items.contains(&id)))
                        })
                        .map(|(p, _)| p.clone())
                        .unwrap_or_else(|| {
                            let base = &slot_opts[slot][0].0;
                            if mo.preview_has_item && mo.item.original.is_none() {
                                if let Some(alt_id) = choose_fallback_item_for_slot(
                                    dex,
                                    obs_mons,
                                    slot,
                                    &used_items,
                                    obs.ruleset(),
                                ) {
                                    return clone_mon_with_item(base, Some(alt_id));
                                }
                            }
                            base.clone()
                        });
                    if mo.item.original.is_none() {
                        if let Some(id) = chosen.item {
                            if !used_items.contains(&id) {
                                used_items.push(id);
                            }
                        }
                    }
                    repaired[slot] =
                        Some(align_moves_with_reference(dex, chosen, &roster[slot]));
                }
                let team: Vec<Pokemon> = repaired.into_iter().map(Option::unwrap).collect();
                let w = raw_score * (0.3 * k_consistent as f64).exp();
                let sig = team_signature(&team);
                if let Some(pos) = s1_sigs.iter().position(|existing| *existing == sig) {
                    source1[pos].1 += w;
                    source1[pos].2 = source1[pos].2.max(k_consistent);
                } else {
                    s1_sigs.push(sig);
                    source1.push((team, w, k_consistent));
                }
            }
            source1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            source1.truncate(4);
        }

        // Source 2: Latin-Hypercube Quantile Coupling enforcing global Item Clause
        let num_strata: usize = if !self.prior_governed_slots.is_empty() { 16 } else { 8 };
        let mut slot_order: Vec<usize> = (0..n_mons).collect();
        slot_order.sort_by_key(|&s| {
            (
                obs_mons[s].item.original.is_none(),
                !obs_mons[s].appeared,
                slot_opts[s].len(),
                s,
            )
        });
        let slot_cdfs: Vec<Vec<f64>> = (0..n_mons)
            .map(|slot| {
                let total: f64 = slot_opts[slot]
                    .iter()
                    .map(|(_, w)| (*w).max(0.0))
                    .sum::<f64>()
                    .max(1e-12);
                let mut cum = 0.0;
                slot_opts[slot]
                    .iter()
                    .map(|(_, w)| {
                        cum += (*w).max(0.0) / total;
                        cum
                    })
                    .collect()
            })
            .collect();
        let mut source2: Vec<(Vec<Pokemon>, f64)> = Vec::with_capacity(num_strata);
        let mut s2_sigs: Vec<TeamSig> = Vec::with_capacity(num_strata);
        for b in 0..num_strata {
            let mut used_items: Vec<ItemId> = Vec::with_capacity(n_mons);
            for mo in obs_mons.iter().take(n_mons) {
                if let Some(Some(id)) = mo.item.original {
                    if !used_items.contains(&id) {
                        used_items.push(id);
                    }
                }
            }
            let mut team_slots: Vec<Option<Pokemon>> = vec![None; n_mons];
            for &slot in &slot_order {
                let mo = &obs_mons[slot];
                let opts = &slot_opts[slot];
                let k_stratum = if b == 0 {
                    0
                } else {
                    1 + ((b - 1 + slot * 2) % (num_strata - 1))
                };
                let q = (k_stratum as f64 + 0.5) / (num_strata as f64);
                let pref_idx = if k_stratum + 1 == num_strata && opts.len() > 1 {
                    opts.len() - 1
                } else {
                    slot_cdfs[slot]
                        .iter()
                        .position(|&c| q < c)
                        .unwrap_or(opts.len() - 1)
                };
                let chosen = if mo.item.original.is_some()
                    || opts[pref_idx]
                        .0
                        .item
                        .is_none_or(|id| !used_items.contains(&id))
                {
                    opts[pref_idx].0.clone()
                } else if let Some((alt_p, _)) = opts
                    .iter()
                    .find(|(p, _)| p.item.is_none_or(|id| !used_items.contains(&id)))
                {
                    alt_p.clone()
                } else if mo.preview_has_item && mo.item.original.is_none() {
                    if let Some(alt_id) = choose_fallback_item_for_slot(
                        dex,
                        obs_mons,
                        slot,
                        &used_items,
                        obs.ruleset(),
                    ) {
                        clone_mon_with_item(&opts[pref_idx].0, Some(alt_id))
                    } else {
                        opts[pref_idx].0.clone()
                    }
                } else {
                    opts[pref_idx].0.clone()
                };
                if mo.item.original.is_none() {
                    if let Some(id) = chosen.item {
                        if !used_items.contains(&id) {
                            used_items.push(id);
                        }
                    }
                }
                team_slots[slot] = Some(chosen);
            }
            let team: Vec<Pokemon> = team_slots.into_iter().map(Option::unwrap).collect();
            let sig = team_signature(&team);
            let w = 1.0 / (num_strata as f64);
            if let Some(pos) = s2_sigs.iter().position(|existing| *existing == sig) {
                source2[pos].1 += w;
            } else {
                s2_sigs.push(sig);
                source2.push((team, w));
            }
        }

        let s2_sum: f64 = source2.iter().map(|(_, w)| *w).sum();
        if s2_sum > 0.0 {
            for (_, w) in &mut source2 {
                *w /= s2_sum;
            }
        } else {
            source2.clear();
        }

        let mut merged: Vec<(Vec<Pokemon>, f64)> = Vec::new();
        let mut merged_sigs: Vec<TeamSig> = Vec::new();
        let mut push_team = |team: Vec<Pokemon>, weight: f64| {
            let sig = team_signature(&team);
            if let Some(pos) = merged_sigs.iter().position(|s| *s == sig) {
                merged[pos].1 += weight;
            } else {
                merged_sigs.push(sig);
                merged.push((team, weight));
            }
        };

        let has_s2 = !source2.is_empty();
        for (team, w) in source2 {
            push_team(team, w);
        }
        if !source1.is_empty() {
            let max_k = source1.iter().map(|x| x.2).max().unwrap_or(0);
            let s1_sum: f64 = source1.iter().map(|x| x.1).sum();
            if s1_sum > 0.0 {
                let target_mass = if !has_s2 {
                    1.0
                } else {
                    let alpha = if max_k >= n_mons.saturating_sub(1) {
                        0.55
                    } else {
                        0.30
                    };
                    alpha / (1.0 - alpha)
                };
                for (team, w, _) in source1 {
                    push_team(team, w * (target_mass / s1_sum));
                }
            }
        }

        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(num_strata);
        if merged.is_empty() {
            return None;
        }
        let total = merged.iter().map(|(_, w)| *w).sum();
        (total > 0.0).then_some(FallbackCandPool {
            entries: merged,
            total,
        })
    }

    #[cfg(test)]
    fn fallback_set(&self, dex: &Dex, mo: &MonObs) -> PokemonSet {
        self.fallback_set_at(dex, mo, 0, std::slice::from_ref(mo))
    }

    fn fallback_set_at(&self, dex: &Dex, mo: &MonObs, slot: usize, obs_mons: &[MonObs]) -> PokemonSet {
        if matches!(
            self.fallback_policy,
            FallbackPolicy::BayesianPointMass | FallbackPolicy::Bayesian
        ) {
            return self.bayesian_fallback_set(dex, mo, slot, obs_mons);
        }
        let pool_set = self.pool_set_for(dex, mo);
        // A separate prior, not a pool candidate: it can fill hidden fields
        // without pretending an exact rental-team identity survived the
        // preview filter. Pool pedigree remains first, preserving existing
        // same-species behavior.
        let source = self.fallback_source(dex, mo);
        let prior = pool_set.or_else(|| match source {
            FallbackSource::CommunityRental => community_rental_sets_for(dex, self.ruleset)
                .iter()
                .find(|s| dex.species.id(&toid(&s.species)) == Some(mo.species)),
            _ => None,
        });
        let mut set = prior.cloned().unwrap_or_else(|| match self.fallback_policy {
            // Deliberately empty: the engine safely enumerates implicit
            // Struggle, matching the pre-M17d control without a fake move.
            FallbackPolicy::LegacyMetaOnly => legacy_base_set(dex, mo, self.ruleset),
            FallbackPolicy::Layered
            | FallbackPolicy::BayesianPointMass
            | FallbackPolicy::Bayesian => base_set(dex, mo, self.ruleset),
        });
        set.level = mo.level;
        set.gender = Some(match mo.gender.as_str() {
            "" => "N".to_string(),
            g => g.to_string(),
        });
        set.name = String::new();
        let species_key = dex.species.key(mo.species);
        let prior_sp = self.active_prior().and_then(|p| p.species(species_key));
        let mut moves: Vec<String> = if prior_sp.is_some() {
            distinct_revealed_moves(dex, mo)
                .into_iter()
                .map(|m| dex.moves.key(m).to_string())
                .collect()
        } else {
            mo.revealed_moves
                .iter()
                .map(|&m| dex.moves.key(m).to_string())
                .collect()
        };
        if let Some(sp) = prior_sp {
            let mut has_hp = moves.iter().any(|m| toid(m).starts_with("hiddenpower"));
            for (_, key, _) in legal_prior_moves(dex, sp, mo, self.ruleset) {
                if moves.len() >= 4 {
                    break;
                }
                let is_hp = key.starts_with("hiddenpower");
                if is_hp && has_hp {
                    continue;
                }
                if is_hp {
                    has_hp = true;
                }
                moves.push(key);
            }
        }
        for name in prior.map(|s| s.moves.clone()).unwrap_or_default() {
            if moves.len() >= 4 {
                break;
            }
            let Some(id) = dex.moves.id(&toid(&name)) else { continue };
            if prior_sp.is_some()
                && dex.moves.key(id).starts_with("hiddenpower")
                && moves.iter().any(|m| toid(m).starts_with("hiddenpower"))
            {
                continue;
            }
            if format_move_legal_at_level_for(
                species_key,
                dex.moves.key(id),
                mo.level,
                self.ruleset,
            )
                && !mo.revealed_moves.iter().any(|&m| move_matches(dex, id, m))
                && !moves.iter().any(|m| dex.moves.id(&toid(m)) == Some(id))
            {
                moves.push(name);
            }
        }
        if moves.is_empty() && self.fallback_policy != FallbackPolicy::LegacyMetaOnly {
            moves.push(legal_fallback_move(dex, mo, self.ruleset));
        }
        moves.truncate(4);
        set.moves = moves;
        let pool_item = std::mem::take(&mut set.item);
        let prior_items = prior_sp
            .map(|sp| legal_prior_items(dex, sp, obs_mons, slot))
            .unwrap_or_default();
        set.item = if let Some(&(id, _)) = prior_items.first() {
            self.resolve_non_excluded_item(dex, obs_mons, slot, mo, dex.items.key(id), &[])
        } else {
            match mo.item.original {
                Some(Some(x)) => dex.items.key(x).to_string(),
                Some(None) => String::new(),
                None if mo.preview_has_item => pool_item, // pool set's, may be ""
                None => String::new(),
            }
        };
        set
    }
}

fn legacy_base_set(dex: &Dex, mo: &MonObs, ruleset: Ruleset) -> PokemonSet {
    let mut set = base_set(dex, mo, ruleset);
    set.moves.clear();
    set
}

/// Species/level/gender-only set (always constructible: every field comes
/// from the live battle, none is parsed). Max stat exp / DVs, the format
/// norm.
fn base_set(dex: &Dex, mo: &MonObs, ruleset: Ruleset) -> PokemonSet {
    let evs = ["hp", "atk", "def", "spa", "spd", "spe"]
        .iter()
        .map(|k| (k.to_string(), 255u16))
        .collect();
    PokemonSet {
        name: String::new(),
        species: dex.species.key(mo.species).to_string(),
        item: String::new(),
        ability: "No Ability".to_string(),
        moves: vec![legal_fallback_move(dex, mo, ruleset)],
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

fn set_has_attack_dependency(dex: &Dex, moves: &[String]) -> bool {
    moves.iter().any(|m| {
        let k = toid(m);
        k.starts_with("hiddenpower")
            || matches!(k.as_str(), "metronome" | "mimic" | "mirrormove" | "transform")
            || dex
                .moves
                .id(&k)
                .is_some_and(|id| dex.moves.get(id).category == "Physical")
    })
}

/// Apply Gen 2 special/status minimum-Attack optimization to synthesized sets that have
/// no Physical moves, Hidden Power, or move-calling moves, preserving gender and HP DV.
fn apply_low_attack_spread_if_special(dex: &Dex, set: &mut PokemonSet, mo: &MonObs) {
    if dex.species.key(mo.species) == "unown" || set_has_attack_dependency(dex, &set.moves) {
        return;
    }
    let mut evs = set.evs.take().unwrap_or_else(|| {
        ["hp", "atk", "def", "spa", "spd", "spe"]
            .iter()
            .map(|k| (k.to_string(), 255u16))
            .collect()
    });
    evs.insert("atk".to_string(), 0);
    set.evs = Some(evs);
    if mo.gender.as_str() != "M" {
        let ivs: std::collections::BTreeMap<String, u8> = [
            ("hp", 31),
            ("atk", 3),
            ("def", 31),
            ("spa", 31),
            ("spd", 31),
            ("spe", 31),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        set.ivs = Some(ivs);
    } else {
        set.ivs = None;
    }
}

/// Build a reduced-Speed EV/IV spread variant (`spe ev = 0`, LSB-preserving `spe DV in {0, 1}`)
/// keeping HP DV, Hidden Power type, and gender identical.
fn with_reduced_speed_spread(dex: &Dex, mon: &Pokemon) -> Pokemon {
    let mut variant = mon.clone();
    variant.set_evs[5] = 0;
    let spe_dv_new = (mon.set_ivs[5] / 2) & 1;
    variant.set_ivs[5] = 2 * spe_dv_new;
    let base_spe = i32::from(dex.species.get(mon.species).base_stats.spe);
    let spe_stat = (2 * base_spe + variant.set_ivs[5] + variant.set_evs[5] / 4)
        * i32::from(mon.level)
        / 100
        + 5;
    variant.base_stored_stats[4] = spe_stat;
    if !variant.transformed {
        variant.stored_stats[4] = spe_stat;
        variant.speed = spe_stat;
    }
    let atk = variant.set_ivs[1] / 2;
    let def = variant.set_ivs[2] / 2;
    let spc = variant.set_ivs[3] / 2;
    let hp_power =
        (5 * ((spc >> 3) + 2 * (spe_dv_new >> 3) + 4 * (def >> 3) + 8 * (atk >> 3)) + spc % 4)
            / 2
            + 31;
    variant.base_hp_power = hp_power;
    if !variant.transformed {
        variant.hp_power = hp_power;
    }
    variant
}

/// A full-PP move slot, matching what `Battle::from_fixture_with_ruleset` builds
/// for a set's move (`64` for base-40 PP moves on Castle PBS vs `61` on PS Stadium 2).
fn fresh_move_slot(dex: &Dex, id: MoveId, is_castle: bool) -> MoveSlot {
    let ms = dex.move_static(id);
    let pp_ups = if ms.no_pp_boosts { 0 } else { 3 };
    let mut pp = ms.pp * (5 + pp_ups) / 5;
    if ms.pp == 40 && !is_castle {
        pp -= pp_ups;
    }
    MoveSlot { id, pp, maxpp: pp, disabled: false, used: false, shared: true }
}

/// Empirical priors predate the no-OHKO format lens and can contain a move
/// that is illegal at the observed level. Filter every filler through the
/// active ruleset's acceptance table (`CastleRules` on Castle PBS, else
/// `format_learnsets()`).
fn format_move_legal_at_level_for(
    species_id: &str,
    move_id: &str,
    level: u8,
    ruleset: Ruleset,
) -> bool {
    let base = if move_id.starts_with("hiddenpower") { "hiddenpower" } else { move_id };
    if ruleset.learnset_policy() == LearnsetPolicy::CastleJpEvents {
        let cr = CastleRules::embedded();
        if cr.is_banned_species(species_id) || cr.is_ohko_move(base) {
            return false;
        }
        let Some(learnset) = cr.species(species_id) else { return false };
        return learnset.allows(base)
            && learnset.move_min_level.get(base).is_none_or(|&min| level as i64 >= min);
    }
    let Some(learnset) = format_learnsets().species(species_id) else { return false };
    learnset.allows(base)
        && learnset.move_min_level.get(base).is_none_or(|&min| level as i64 >= min)
}

#[cfg(test)]
fn format_move_legal_at_level(species_id: &str, move_id: &str, level: u8) -> bool {
    format_move_legal_at_level_for(species_id, move_id, level, Ruleset::PsStadium2)
}

/// Deterministic nonempty set for a format-legal species absent from every
/// empirical prior: the lexicographically first move accepted at its
/// observed level by the active ruleset's learnset lens.
fn legal_fallback_move(dex: &Dex, mo: &MonObs, ruleset: Ruleset) -> String {
    let species = dex.species.key(mo.species);
    let moves: Option<&[String]> = if ruleset.learnset_policy() == LearnsetPolicy::CastleJpEvents {
        CastleRules::embedded()
            .species(species)
            .map(|l| l.moves.as_slice())
    } else {
        format_learnsets()
            .species(species)
            .map(|l| l.moves.as_slice())
    };
    moves
        .and_then(|list| {
            list.iter().find(|m| {
                dex.moves.id(m).is_some()
                    && format_move_legal_at_level_for(species, m, mo.level, ruleset)
            })
        })
        .cloned()
        // Defensive malformed/off-format input: still never reintroduce the
        // empty-set/implicit-Struggle failure this function closes.
        .unwrap_or_else(|| "return".to_string())
}

/// Non-dominated legal moves from `mo.species`'s learnset at `mo.level`, ranked
/// by competitive utility / STAB expected power as a backoff prior when empirical
/// co-occurrence carriers provide fewer than 6 distinct filler moves.
fn non_dominated_learnset_moves(
    dex: &Dex,
    mo: &MonObs,
    ruleset: Ruleset,
) -> Vec<(MoveId, String, f64)> {
    const ZERO_OR_DOMINATED: &[&str] = &[
        "splash", "teleport", "falseswipe", "payday", "frustration", "growl", "leer",
        "tailwhip", "harden", "withdraw", "kinesis", "flash", "poisongas", "poisonpowder",
        "supersonic", "stringshot", "snore", "sweetscent", "defensecurl", "sharpen",
        "meditate", "rage", "constrict", "absorb", "megadrain", "ember", "watergun",
        "thundershock", "peck", "pound", "scratch", "tackle", "vinewhip", "bubble",
        "confusion", "rockthrow", "karatechop", "gust", "smog", "lick", "leechlife",
    ];
    const HIGH_VALUE_STATUS: &[(&str, f64)] = &[
        ("rest", 3.0),
        ("sleeptalk", 2.4),
        ("curse", 2.6),
        ("swordsdance", 2.4),
        ("batonpass", 2.5),
        ("meanlook", 2.4),
        ("spiderweb", 2.4),
        ("perishsong", 2.5),
        ("destinybond", 2.3),
        ("encore", 2.4),
        ("thunderwave", 2.3),
        ("stunspore", 2.1),
        ("sleeppowder", 2.6),
        ("lovelykiss", 2.7),
        ("spore", 2.9),
        ("hypnosis", 2.2),
        ("toxic", 2.2),
        ("confuseray", 1.9),
        ("reflect", 2.1),
        ("lightscreen", 2.1),
        ("substitute", 2.1),
        ("recover", 2.8),
        ("softboiled", 2.8),
        ("milkdrink", 2.8),
        ("moonlight", 2.3),
        ("morningsun", 2.3),
        ("synthesis", 2.3),
        ("roar", 2.2),
        ("whirlwind", 2.2),
        ("protect", 2.0),
        ("agility", 2.0),
        ("amnesia", 1.9),
        ("bellydrum", 2.2),
        ("charm", 2.0),
        ("painsplit", 2.1),
        ("safeguard", 1.7),
        ("healbell", 2.3),
        ("leechseed", 2.2),
        ("spikes", 2.5),
    ];
    let species_key = dex.species.key(mo.species);
    let moves: &[String] = if ruleset.learnset_policy() == LearnsetPolicy::CastleJpEvents {
        let Some(learnset) = CastleRules::embedded().species(species_key) else {
            return Vec::new();
        };
        &learnset.moves
    } else {
        let Some(learnset) = format_learnsets().species(species_key) else {
            return Vec::new();
        };
        &learnset.moves
    };
    let sp_types = &dex.species.get(mo.species).types;
    let mut out: Vec<(MoveId, String, f64)> = Vec::new();
    for m in moves {
        if ZERO_OR_DOMINATED.contains(&m.as_str()) {
            continue;
        }
        let Some(id) = dex.moves.id(m) else { continue };
        if !format_move_legal_at_level_for(species_key, m, mo.level, ruleset) {
            continue;
        }
        let md = dex.moves.get(id);
        let score = if let Some(&(_, w)) = HIGH_VALUE_STATUS.iter().find(|&&(k, _)| k == m.as_str())
        {
            w
        } else if m == "explosion" || m == "selfdestruct" {
            2.6
        } else if m == "counter" || m == "mirrorcoat" || m == "seismictoss" || m == "superfang" {
            2.0
        } else if m == "rapidspin" || m == "pursuit" || m == "thief" {
            1.9
        } else if md.base_power > 0 {
            let stab = if sp_types.iter().any(|t| t == &md.move_type) {
                1.35
            } else {
                1.0
            };
            let acc = md
                .accuracy
                .as_f64()
                .map(|p| (p / 100.0).clamp(0.5, 1.0))
                .unwrap_or(1.0);
            ((md.base_power as f64 * stab * acc) / 45.0).clamp(0.4, 3.0)
        } else {
            0.5
        };
        out.push((id, m.clone(), score));
    }
    out.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
fn build_refs(dex: &Dex, sets: &[PokemonSet], mons: &[MonObs]) -> Option<Vec<Pokemon>> {
    build_refs_with_ruleset(dex, sets, mons, Ruleset::PsStadium2)
}

/// Construct a pool team's reference mons and align them to the opponent's
/// observed roster slots by (species, level, gender). `None` =
/// preview-inconsistent (public detail mismatch or unbuildable team).
fn build_refs_with_ruleset(
    dex: &Dex,
    sets: &[PokemonSet],
    mons: &[MonObs],
    ruleset: Ruleset,
) -> Option<Vec<Pokemon>> {
    if sets.len() != mons.len() {
        return None;
    }
    let built = Battle::from_fixture_with_ruleset(dex, "1,2,3,4", sets, sets, ruleset).ok()?;
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

fn consistent(dex: &Dex, refs: &[Pokemon], mons: &[MonObs]) -> bool {
    refs.iter().zip(mons).enumerate().all(|(slot, (r, mo))| {
        mo.observed_maxhp.is_none_or(|m| r.base_maxhp == m)
            && crate::hp_evidence::consistent(dex, r, mo.hp_type_mask)
            && !crate::observe::Observer::is_item_excluded_for_slot_with_refs(
                dex,
                Some(refs),
                mons,
                slot,
                r.item,
            )
            && mo
                .revealed_moves
                .iter()
                .all(|m| r.base_move_slots.iter().any(|s| move_matches(dex, s.id, *m)))
            && mo.sketch_overlay.is_none_or(|copied| {
                // Castle Sketch fails if the user already knew the copied move (moveexec.rs:1290),
                // so any submitted candidate carrying the sketched move is impossible.
                !r.base_move_slots.iter().any(|s| move_matches(dex, s.id, copied))
            })
    })
}

// -------------------------------------------------------------- imputation

/// Overwrite one opponent mon's hidden set-level fields with the reference
/// mon's, keeping everything public exactly.
///
/// TOTAL destructure of `Pokemon` on purpose (the `state_key` trick): adding
/// a state field breaks this fn until the field is triaged public (keep) /
/// hidden (impute). Caller refreshes `handler_mask` afterwards.
fn impute_mon(dst: &mut Pokemon, refm: &Pokemon, mo: &MonObs, dex: &Dex, exact_hp: bool) {
    let orig_hp = dst.hp;
    let orig_maxhp = dst.maxhp;
    let public_hp = if exact_hp {
        0
    } else {
        crate::import::announce_hp(orig_hp, orig_maxhp, 100)
    };
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
        hp,                       // public bucket, reconstructed for sampled max HP
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
    // Only the announced HP bucket is public on PS; Castle reveals exact HP directly.
    // Unseen mons are full.
    let appeared = *previously_switched_in > 0 || *is_active || mo.appeared;
    if appeared {
        if exact_hp {
            *hp = if orig_hp <= 0 {
                0
            } else if orig_maxhp > 0 && orig_hp >= orig_maxhp {
                *maxhp
            } else {
                orig_hp.clamp(1, (*maxhp).max(1))
            };
        } else {
            *hp = crate::import::impute_hp(public_hp, 100, *maxhp);
        }
    } else {
        *hp = *maxhp;
    }

    // ---- moves: candidate's set; revealed slots keep their live usage
    let old_base = *base_move_slots;
    let mut nb = MoveSlots::default();
    for rs in refm.base_move_slots.iter() {
        if mo.revealed_moves.iter().any(|m| move_matches(dex, rs.id, *m)) {
            match old_base
                .iter()
                .find(|s| s.id == rs.id || move_matches(dex, s.id, rs.id))
            {
                Some(os) => nb.push(MoveSlot {
                    id: rs.id,
                    shared: true,
                    ..*os
                }),
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
    // Castle Sketch overlay (public, permanent across switches): preserve the
    // copied move's live PP/disabled slot instead of restoring spent Sketch.
    if let Some(sketched) = mo.sketch_overlay {
        if let Some(sketch_id) = dex.moves.id("sketch") {
            if let Some(pos) = (0..nb.len()).find(|&i| nb[i].id == sketch_id) {
                if let Some(live) = old_base.iter().find(|s| s.id == sketched) {
                    nb[pos] = MoveSlot { shared: true, ..*live };
                } else {
                    let ms = dex.move_static(sketched);
                    let pp_ups = if ms.no_pp_boosts { 0 } else { 3 };
                    let maxpp = ms.pp * (5 + pp_ups) / 5;
                    nb[pos] = MoveSlot {
                        id: sketched,
                        pp: maxpp,
                        maxpp,
                        disabled: false,
                        used: false,
                        shared: true,
                    };
                }
            }
        }
    }
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
        ruleset: _,            // public ruleset configuration
        preview_level_caps: _,
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
        quick_claw_rolls: _,   // hidden per-turn coin(s) → resampled by determinize
        triple_kick_mult: _,   // public mid-turn state (0 between turns)
        recoil_ko_side: _,     // public mid-turn state (None between turns)
        bp_switched_this_turn: _, // public mid-turn state ([false, false] between turns)
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
            ruleset: Ruleset::PsStadium2,
            hp_observations: None,
            cands: pool
                .teams
                .into_iter()
                .map(|t| Candidate { id: t.id, weight: None, sets: t.sets, refs: None })
                .collect(),
            alive: Vec::new(),
            fallback: None,
            fallback_visible: None,
            pinned: false,
            fallback_policy: FallbackPolicy::Layered,
            prior: None,
            pick_draws: None,
            prior_governed_slots: Vec::new(),
            fallback_lead_weights: None,
            loadout_draws: None,
            fallback_cands: None,
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
            hp_type_mask: u16::MAX,
            revealed_moves: revealed.iter().map(|m| dex.moves.id(m).unwrap()).collect(),
            item: ItemObs::default(),
            excluded_items: 0,
            appeared: false,
            observed_maxhp: None,
            sketch_overlay: None,
        }
    }

    #[test]
    fn default_policy_anticipates_perish() {
        let (dex, pool, mut battle, mut obs) = loadout_fixture();
        assert_eq!(FallbackPolicy::default(), FallbackPolicy::Bayesian);
        let mut belief = Belief::new(&dex, &pool, &obs);
        assert_eq!(belief.fallback_policy, FallbackPolicy::Bayesian);
        battle.sides[1].roster[0].is_active = true;
        obs.observe(&battle, &dex);
        belief.sync(&dex, &obs);
        assert!(belief.loadout_draws.as_ref().unwrap()[0].is_some());
    }

    fn loadout_fixture() -> (Dex, MetaPool, Battle, Observer) {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();
        let mut set = pool
            .teams
            .iter()
            .flat_map(|t| &t.sets)
            .find(|s| {
                toid(&s.species) == "misdreavus" && s.moves.iter().any(|m| toid(m) == "meanlook")
            })
            .unwrap()
            .clone();
        set.name = "Misdreavus".into();
        let battle = Battle::from_fixture(&dex, "1,2,3,4", &[set.clone()], &[set]).unwrap();
        let obs = Observer::new(&battle, 0);
        (dex, pool, battle, obs)
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

    /// The crawl is an archive, the pool is not: nine of its 28 entries do
    /// not belong to the played format — eight the validator rejects, one
    /// the source names for another rule set — and none of their sets may
    /// reach the imputation pool. Derived from the embedded JSON rather
    /// than a hard-coded list, so a data edit moves the test with it; only
    /// the counts and the species those entries owned are pinned.
    #[test]
    fn the_rental_pool_takes_only_teams_the_format_accepts() {
        let dex = test_dex();
        let db: CommunityRentalDb = serde_json::from_str(COMMUNITY_RENTALS_JSON).unwrap();
        let (legal, rejected): (Vec<_>, Vec<_>) = db.teams.into_iter().partition(|t| {
            !names_another_ruleset(&t.archetype)
                && team_is_format_legal(&dex, &t.sets, Ruleset::PsStadium2)
        });
        assert_eq!(rejected.len(), 9, "entries excluded from the pool");
        assert_eq!(
            rejected.iter().filter(|t| names_another_ruleset(&t.archetype)).count(),
            2,
            "entries named for another rule set (Little Cup, 指振り)"
        );

        let pool = community_rental_sets(&dex);
        let kept: Vec<&PokemonSet> = legal.iter().flat_map(|t| &t.sets).collect();
        assert_eq!(pool.len(), kept.len(), "pool size");
        for (got, want) in pool.iter().zip(kept) {
            assert_eq!(got.species, want.species);
            assert_eq!(got.moves, want.moves);
        }

        // Set-level too, since the pool is consumed one set at a time.
        for set in pool {
            assert!((50..=55).contains(&set.level), "{} L{}", set.species, set.level);
            for mv in &set.moves {
                assert!(
                    format_move_legal_at_level(&toid(&set.species), &toid(mv), set.level),
                    "{} carries {mv}",
                    set.species
                );
            }
        }

        // The species the rejected entries were the sole source of: a Little
        // Cup Chansey, a Fissure Dugtrio, the links page's Ditto, and a
        // Granbull whose whole moveset was Metronome.
        for sid in ["chansey", "dugtrio", "ditto", "granbull"] {
            assert!(
                pool.iter().all(|s| toid(&s.species) != sid),
                "{sid} is still in the pool"
            );
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

        // Vaporeon is absent from the meta pool but present in the community
        // rentals, so its first rental set supplies the filler.
        assert!(belief
            .cands
            .iter()
            .flat_map(|c| &c.sets)
            .all(|s| toid(&s.species) != "vaporeon"));
        let rental_vaporeon = community_rental_sets(&dex)
            .iter()
            .find(|s| toid(&s.species) == "vaporeon")
            .unwrap();
        // At its own level: Hydro Pump is a L52 level-up move, so asking at
        // L50 would (correctly) drop it — the filler is legality-filtered
        // per level, not copied wholesale.
        let lv = rental_vaporeon.level;
        let vaporeon = belief.fallback_set(&dex, &mon(&dex, "vaporeon", lv, &[]));
        assert_eq!(vaporeon.moves, rental_vaporeon.moves);
        assert_eq!(
            belief.fallback_source(&dex, &mon(&dex, "vaporeon", lv, &[])),
            FallbackSource::CommunityRental
        );
        let revealed_vaporeon =
            belief.fallback_set(&dex, &mon(&dex, "vaporeon", lv, &["toxic"]));
        assert_eq!(toid(&revealed_vaporeon.moves[0]), "toxic");
        assert_eq!(revealed_vaporeon.moves.len(), 4);

        // Chansey reaches the bot ONLY through the Little Cup entry, which
        // the format rejects — so it is not a rental species at all and gets
        // the learnset layer, not a L5 set's moves.
        assert_eq!(
            belief.fallback_source(&dex, &mon(&dex, "chansey", 50, &[])),
            FallbackSource::Learnset
        );

        // Reveals lead and survive even when neither empirical source has
        // the species. Zero reveal gets the legal deterministic default.
        assert!(belief
            .cands
            .iter()
            .flat_map(|c| &c.sets)
            .chain(community_rental_sets(&dex))
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
    fn a_weighted_prior_falls_back_to_the_heaviest_loadout() {
        let dex = test_dex();
        let mut belief = test_belief();
        let carriers: Vec<usize> = (0..belief.cands.len())
            .filter(|&i| belief.cands[i].sets.iter().any(|s| toid(&s.species) == "snorlax"))
            .collect();
        let set_of = |b: &Belief, i: usize| {
            b.cands[i].sets.iter().find(|s| toid(&s.species) == "snorlax").unwrap().clone()
        };
        let first = set_of(&belief, carriers[0]);
        let other = carriers
            .iter()
            .copied()
            .find(|&i| set_of(&belief, i).moves != first.moves)
            .expect("the test pool carries two Snorlax loadouts");
        let other_set = set_of(&belief, other);
        let level = first.level;

        // Unweighted: first in file order, as shipped.
        let plain = belief.fallback_set(&dex, &mon(&dex, "snorlax", level, &[]));
        assert_eq!(plain.moves, first.moves);

        // Weighted: the other loadout outweighs every copy of the first.
        for c in belief.cands.iter_mut() {
            c.weight = Some(1.0);
        }
        belief.cands[other].weight = Some(1000.0);
        let heavy = belief.fallback_set(&dex, &mon(&dex, "snorlax", other_set.level, &[]));
        let mut want: Vec<String> = other_set.moves.iter().map(|m| toid(m)).collect();
        let mut got: Vec<String> = heavy.moves.iter().map(|m| toid(m)).collect();
        want.sort();
        got.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn candidate_sampling_follows_weights_and_equal_weights_draw_as_before() {
        let mut belief = test_belief();
        belief.alive = vec![0, 1];
        let draws = |b: &Belief| {
            let mut rng = SplitMix64::new(7);
            (0..4000).map(|_| b.sample(&mut rng).unwrap()).collect::<Vec<_>>()
        };
        let unweighted = draws(&belief);
        belief.cands[0].weight = Some(2.5);
        belief.cands[1].weight = Some(2.5);
        assert_eq!(draws(&belief), unweighted, "equal weights must keep the uniform draw");
        belief.cands[0].weight = Some(3.0);
        belief.cands[1].weight = Some(1.0);
        let share = draws(&belief).iter().filter(|&&i| i == 0).count() as f64 / 4000.0;
        assert!((share - 0.75).abs() < 0.03, "weighted share {share}");
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

    #[test]
    fn prior_items_marginal_sampling_and_dominance() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(META_POOL_JSON).unwrap();
        let item_set = |species: &str, item: &str| {
            let mut s = plain_set(species);
            s.item = item.to_string();
            s
        };
        let custom: Vec<PokemonSet> = [
            ("Bulbasaur", "Gold Berry"),
            ("Ivysaur", "Miracle Berry"),
            ("Venusaur", "Mint Berry"),
            ("Charmander", ""),
            ("Charmeleon", "Quick Claw"),
            ("Charizard", "Leftovers"),
        ]
        .into_iter()
        .map(|(sp, it)| item_set(sp, it))
        .collect();
        let custom_p1: Vec<PokemonSet> = [
            "Snorlax", "Zapdos", "Raikou", "Cloyster", "Tyranitar", "Gengar",
        ]
        .into_iter()
        .map(plain_set)
        .collect();
        let battle =
            Battle::from_fixture(&dex, "1,2,3,4", &custom_p1, &custom).unwrap();
        let mut obs = Observer::new(&battle, 0);

        let prior_json = r#"{
            "format": "nc2000-belief-prior", "version": 1,
            "species": {
                "bulbasaur": {"moves": {"razorleaf": 0.9, "sleeppowder": 0.8, "bodyslam": 0.7, "synthesis": 0.6},
                              "items": {"leftovers": 0.8, "miracleberry": 0.2}},
                "ivysaur": {"moves": {"razorleaf": 0.9, "sleeppowder": 0.8, "bodyslam": 0.7, "synthesis": 0.6},
                            "items": {"leftovers": 0.8, "miracleberry": 0.2}},
                "charmander": {"moves": {"flamethrower": 0.9, "fireblast": 0.8, "bodyslam": 0.7, "rest": 0.6},
                               "items": {"leftovers": 0.9, "goldberry": 0.1}},
                "charmeleon": {"moves": {"flamethrower": 0.9, "fireblast": 0.8, "bodyslam": 0.7, "rest": 0.6},
                               "items": {}}
            }
        }"#;
        let prior = Arc::new(BeliefPrior::from_json(prior_json));
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.set_prior(prior.clone());
        belief.sync(&dex, &obs);

        let leftovers = dex.items.id("leftovers").unwrap();
        let miracleberry = dex.items.id("miracleberry").unwrap();
        let goldberry = dex.items.id("goldberry").unwrap();
        let mut bulb_lefties = 0usize;
        let mut bulb_miracle = 0usize;
        for seed in 0..200 {
            let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(seed + 1));
            // !preview_has_item on slot 3 (Charmander) must dominate sp.items.
            assert_eq!(det.sides[1].roster[3].item, None);
            // Empty sp.items on slot 4 (Charmeleon) falls back cleanly to a non-None item.
            assert!(det.sides[1].roster[4].item.is_some());
            // Global Item Clause across all unrevealed preview_has_item slots.
            let mut seen = Vec::new();
            for mon in &det.sides[1].roster {
                if let Some(id) = mon.item {
                    assert!(!seen.contains(&id), "duplicate item {id:?} under BeliefPrior");
                    seen.push(id);
                }
            }
            match det.sides[1].roster[0].item {
                Some(id) if id == leftovers => bulb_lefties += 1,
                Some(id) if id == miracleberry => bulb_miracle += 1,
                other => panic!("unexpected bulbasaur item {other:?}"),
            }
        }
        assert!(bulb_lefties > bulb_miracle && bulb_miracle > 0);

        // Reveal Gold Berry on slot 0 (item.original) and Leftovers on slot 2 (cross-slot Item Clause)
        // plus chip+upkeep on slot 1 to exclude Leftovers passively on slot 1.
        let mut battle_t1 = battle.clone();
        battle_t1
            .apply_choices(
                &dex,
                [
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                ],
            )
            .unwrap();
        obs.observe(&battle_t1, &dex);
        obs.ingest_line("|-item|p2a: Bulbasaur|Gold Berry", &dex);
        obs.ingest_line("|switch|p2a: Ivysaur|Ivysaur, L50, M|100/100", &dex);
        obs.ingest_line("|-damage|p2a: Ivysaur|90/100", &dex);
        obs.ingest_line("|upkeep", &dex);
        belief.sync(&dex, &obs);
        for seed in 0..32 {
            let det = belief.determinize(&dex, &battle_t1, &obs, &mut SplitMix64::new(seed + 1));
            assert_eq!(det.sides[1].roster[0].item, Some(goldberry));
            assert_ne!(det.sides[1].roster[1].item, Some(leftovers));
            assert_eq!(det.sides[1].roster[1].item, Some(miracleberry));
        }

        // 100%-excluded sp.items falls back cleanly instead of producing None.
        let mut custom2 = custom.clone();
        custom2[2].item = "Leftovers".to_string();
        let mut battle2 =
            Battle::from_fixture(&dex, "1,2,3,4", &custom_p1, &custom2).unwrap();
        battle2
            .apply_choices(
                &dex,
                [
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                ],
            )
            .unwrap();
        let mut obs2 = Observer::new(&battle2, 0);
        obs2.observe(&battle2, &dex);
        obs2.ingest_line("|switch|p2a: Venusaur|Venusaur, L50, M|100/100", &dex);
        obs2.ingest_line("|-heal|p2a: Venusaur|100/100|[from] item: Leftovers", &dex);
        obs2.ingest_line("|switch|p2a: Bulbasaur|Bulbasaur, L50, M|100/100", &dex);
        let mut b2 = Belief::new(&dex, &pool, &obs2);
        b2.set_prior(Arc::new(BeliefPrior::from_json(
            r#"{"species":{"bulbasaur":{"items":{"leftovers":1.0}}}}"#,
        )));
        b2.sync(&dex, &obs2);
        for seed in 0..16 {
            let det = b2.determinize(&dex, &battle2, &obs2, &mut SplitMix64::new(seed + 1));
            let item = det.sides[1].roster[0].item;
            assert!(item.is_some() && item != Some(leftovers));
        }
    }

    #[test]
    fn prior_lead_biases_party_and_respects_level_caps() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(META_POOL_JSON).unwrap();
        let lv_set = |species: &str, level: u8| {
            let mut s = plain_set(species);
            s.level = level;
            s
        };
        let custom: Vec<PokemonSet> = [
            lv_set("Bulbasaur", 55),
            lv_set("Ivysaur", 55),
            lv_set("Venusaur", 55),
            lv_set("Charmander", 50),
            lv_set("Charmeleon", 50),
            lv_set("Charizard", 50),
        ]
        .into_iter()
        .collect();
        let custom_p1: Vec<PokemonSet> = [
            "Snorlax", "Zapdos", "Raikou", "Cloyster", "Tyranitar", "Gengar",
        ]
        .into_iter()
        .map(plain_set)
        .collect();
        let battle =
            Battle::from_fixture(&dex, "1,2,3,4", &custom_p1, &custom).unwrap();
        let obs = Observer::new(&battle, 0);

        // 1. Lead weights bias hidden party[0] (when party_len == 3 & hidden) and strictly obey preview_level_caps <= 155.
        let mut battle_hidden_party = battle.clone();
        battle_hidden_party.sides[1].party.truncate(3);
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.set_prior(Arc::new(BeliefPrior::from_json(
            r#"{
                "species": {
                    "bulbasaur": {"lead": 0.9},
                    "ivysaur": {"lead": 0.9},
                    "venusaur": {"lead": 0.9},
                    "charmander": {"lead": 0.01},
                    "charmeleon": {"lead": 0.01},
                    "charizard": {"lead": 0.0}
                }
            }"#,
        )));
        belief.sync(&dex, &obs);
        let mut high_lv_leads = 0usize;
        for seed in 0..300 {
            let det = belief.determinize(
                &dex,
                &battle_hidden_party,
                &obs,
                &mut SplitMix64::new(seed + 1),
            );
            let total_lv: u32 = det.sides[1]
                .party
                .iter()
                .map(|&s| u32::from(det.sides[1].roster[s as usize].level))
                .sum();
            assert!(total_lv <= 155, "level cap violated: {total_lv}");
            if det.sides[1].party[0] < 3 {
                high_lv_leads += 1;
            }
        }
        assert!(
            high_lv_leads > 200,
            "lead weights should strongly bias party[0] toward slots 0..3, got {high_lv_leads}/300"
        );

        // sample_preview_prior respects legal actions under preview_level_caps.
        let custom_ref = &custom;
        let actions: Vec<nc2000_engine::battle::SearchChoice> = (1..=6u8)
            .flat_map(|a| {
                (1..=6u8).flat_map(move |b| {
                    (1..=6u8).filter_map(move |c| {
                        if a == b || a == c || b == c {
                            return None;
                        }
                        let lv: u32 = [a, b, c]
                            .iter()
                            .map(|&s| u32::from(custom_ref[(s - 1) as usize].level))
                            .sum();
                        (lv <= 155).then_some(nc2000_engine::battle::SearchChoice::Team([a, b, c]))
                    })
                })
            })
            .collect();
        let mut preview_high_leads = 0usize;
        for seed in 0..200 {
            let idx = belief
                .sample_preview_prior(None, &obs, &actions, &mut SplitMix64::new(seed + 1))
                .expect("sample_preview_prior must return Some when fallback_lead_weights is active");
            if let nc2000_engine::battle::SearchChoice::Team([a, _, _]) = actions[idx] {
                if a <= 3 {
                    preview_high_leads += 1;
                }
            }
        }
        assert!(preview_high_leads > 110, "got {preview_high_leads}/200");

        // 2. Luce bench posterior after lead reveal (slot 3 leads at L50, slot 0 has lead 0.9, slot 4 has lead 0.0).
        let mut battle_t1 = battle.clone();
        battle_t1
            .apply_choices(
                &dex,
                [
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                    Some(nc2000_engine::battle::SearchChoice::Team([4, 5, 6])),
                ],
            )
            .unwrap();
        let mut obs_t1 = Observer::new(&battle_t1, 0);
        obs_t1.observe(&battle_t1, &dex);
        let mut belief_t1 = Belief::new(&dex, &pool, &obs_t1);
        belief_t1.set_prior(Arc::new(BeliefPrior::from_json(
            r#"{
                "species": {
                    "bulbasaur": {"lead": 0.9},
                    "ivysaur": {"lead": 0.9},
                    "venusaur": {"lead": 0.9},
                    "charmander": {"lead": 0.05},
                    "charmeleon": {"lead": 0.05},
                    "charizard": {"lead": 0.0}
                }
            }"#,
        )));
        belief_t1.sync(&dex, &obs_t1);
        let mut saw_l55_pair = 0usize;
        for seed in 0..120 {
            let det = belief_t1.determinize(&dex, &battle_t1, &obs_t1, &mut SplitMix64::new(seed + 1));
            assert_eq!(det.sides[1].party[0], 3, "appeared lead must stay at party[0]");
            let total_lv: u32 = det.sides[1]
                .party
                .iter()
                .map(|&s| u32::from(det.sides[1].roster[s as usize].level))
                .sum();
            assert!(total_lv <= 155);
            if det.sides[1].party[1..].iter().any(|&s| s < 3) {
                saw_l55_pair += 1;
            }
        }
        assert!(saw_l55_pair > 0);

        // 3. "lead": 0 on ALL 6 slots must be total (never divide by zero / panic).
        let mut zero_belief = Belief::new(&dex, &pool, &obs);
        zero_belief.set_prior(Arc::new(BeliefPrior::from_json(
            r#"{
                "species": {
                    "bulbasaur": {"lead": 0.0},
                    "ivysaur": {"lead": 0.0},
                    "venusaur": {"lead": 0.0},
                    "charmander": {"lead": 0.0},
                    "charmeleon": {"lead": 0.0},
                    "charizard": {"lead": 0.0}
                }
            }"#,
        )));
        zero_belief.sync(&dex, &obs);
        for seed in 0..16 {
            let det = zero_belief.determinize(
            &dex,
            &battle_hidden_party,
            &obs,
            &mut SplitMix64::new(seed + 1),
        );
            let total_lv: u32 = det.sides[1]
                .party
                .iter()
                .map(|&s| u32::from(det.sides[1].roster[s as usize].level))
                .sum();
            assert!(total_lv <= 155);
        }
    }

    #[test]
    fn fallback_cand_pool_and_bench_diversification_active_under_prior() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(META_POOL_JSON).unwrap();
        let custom: Vec<PokemonSet> = [
            "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard",
        ]
        .into_iter()
        .map(plain_set)
        .collect();
        let custom_p1: Vec<PokemonSet> = [
            "Snorlax", "Zapdos", "Raikou", "Cloyster", "Tyranitar", "Gengar",
        ]
        .into_iter()
        .map(plain_set)
        .collect();
        let mut battle =
            Battle::from_fixture(&dex, "1,2,3,4", &custom_p1, &custom).unwrap();
        battle
            .apply_choices(
                &dex,
                [
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                ],
            )
            .unwrap();
        let mut obs = Observer::new(&battle, 0);
        obs.observe(&battle, &dex);

        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.set_prior(Arc::new(BeliefPrior::from_json(
            r#"{
                "species": {
                    "bulbasaur": {"moves": {"razorleaf": 0.9, "sleeppowder": 0.7, "swordsdance": 0.5, "bodyslam": 0.4, "synthesis": 0.3}},
                    "ivysaur": {"moves": {"razorleaf": 0.9, "sleeppowder": 0.7, "swordsdance": 0.5, "bodyslam": 0.4, "synthesis": 0.3}},
                    "charizard": {"moves": {"flamethrower": 0.9, "earthquake": 0.8, "rockslide": 0.7, "bellydrum": 0.6, "fireblast": 0.5}}
                }
            }"#,
        )));
        belief.sync(&dex, &obs);
        assert!(belief.fallback_cand_count() > 1);
        let plan = belief.root_sampling_plan(16, &mut SplitMix64::new(7));
        assert!(plan.iter().any(|p| matches!(p, TeamPlanPick::FallbackCand(_))));

        // Unappeared bench slots (e.g. slot 1 Ivysaur, slot 5 Charizard) must diversify across determinizations.
        let mut bench1 = std::collections::HashSet::new();
        let mut bench5 = std::collections::HashSet::new();
        for seed in 0..32 {
            let mut rng_probe = SplitMix64::new(seed + 1);
            let key = belief.predict_fallback_draws(None, &battle, 1, &mut rng_probe);
            assert!(key.is_some());
            let mut rng_actual = SplitMix64::new(seed + 1);
            let _ = belief.sample_fallback_refs(None, &mut rng_actual, &battle, 1);
            assert_eq!(rng_probe.0, rng_actual.0);

            let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(seed + 1));
            let m1: Vec<u16> = det.sides[1].roster[1]
                .base_move_slots
                .iter()
                .map(|m| m.id.0)
                .collect();
            let m5: Vec<u16> = det.sides[1].roster[5]
                .base_move_slots
                .iter()
                .map(|m| m.id.0)
                .collect();
            bench1.insert(m1);
            bench5.insert(m5);
        }
        assert!(bench1.len() > 1, "unappeared slot 1 should diversify");
        assert!(bench5.len() > 1, "unappeared slot 5 should diversify");
    }

    #[test]
    fn non_matching_species_prior_is_bit_identical() {
        let (dex, pool, battle, obs) = m18_fixture(true);
        let plain = Belief::new(&dex, &pool, &obs);
        let mut with_other = Belief::new(&dex, &pool, &obs);
        with_other.set_prior(Arc::new(BeliefPrior::from_json(
            r#"{"species":{"mewtwo":{"moves":{"psychic":1.0,"icebeam":0.9},"items":{"leftovers":1.0},"lead":0.9}}}"#,
        )));
        with_other.sync(&dex, &obs);
        assert!(with_other.prior_governed().is_empty());
        assert_eq!(plain.fallback_cand_count(), with_other.fallback_cand_count());
        let mut r1 = SplitMix64::new(123);
        let mut r2 = SplitMix64::new(123);
        assert_eq!(
            plain.root_sampling_plan(8, &mut r1).len(),
            with_other.root_sampling_plan(8, &mut r2).len()
        );
        assert_eq!(r1.0, r2.0);
        for seed in 0..16 {
            let mut a = SplitMix64::new(seed + 1);
            let mut b = SplitMix64::new(seed + 1);
            let d1 = plain.determinize(&dex, &battle, &obs, &mut a);
            let d2 = with_other.determinize(&dex, &battle, &obs, &mut b);
            assert_eq!(a.0, b.0);
            for slot in 0..6 {
                assert_eq!(
                    d1.sides[1].roster[slot].base_move_slots,
                    d2.sides[1].roster[slot].base_move_slots
                );
                assert_eq!(d1.sides[1].roster[slot].item, d2.sides[1].roster[slot].item);
            }
        }
    }

    #[test]
    fn partial_prior_support_locks_moves_and_backfills_remaining() {
        let (dex, pool, battle, obs) = m18_fixture(false);
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.set_prior(Arc::new(BeliefPrior::from_json(
            r#"{"species":{"bulbasaur":{"moves":{"razorleaf":0.8,"sleeppowder":0.2,"fissure":1.0}}}}"#,
        )));
        belief.sync(&dex, &obs);
        for seed in 0..32 {
            let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(seed + 1));
            let moves: Vec<String> = det.sides[1].roster[0]
                .base_move_slots
                .iter()
                .map(|m| dex.moves.key(m.id).to_string())
                .collect();
            assert_eq!(moves.len(), 4);
            assert!(moves.iter().any(|m| m == "razorleaf"));
            assert!(moves.iter().any(|m| m == "sleeppowder"));
            assert!(!moves.iter().any(|m| m == "fissure"));
        }
    }

    #[test]
    fn supplemental_prior_teams_are_all_format_legal_and_cover_baton_pass_learners() {
        let dex = test_dex();
        let raw: SupplementalPriorDb = serde_json::from_str(SUPPLEMENTAL_PRIOR_JSON).unwrap();
        let legal = supplemental_prior_teams(&dex);
        assert_eq!(
            legal.len(),
            raw.teams.len(),
            "every embedded supplemental prior team must pass validate_team"
        );
        for bp_species in [
            "jolteon",
            "scizor",
            "scyther",
            "espeon",
            "girafarig",
            "smeargle",
            "umbreon",
            "vaporeon",
            "flareon",
            "mrmime",
            "aipom",
            "ariados",
            "ledian",
            "farfetchd",
            "venomoth",
            "eevee",
            "venonat",
            "ledyba",
            "spinarak",
        ] {
            let has_bp_set = legal.iter().flat_map(|t| &t.sets).any(|s| {
                toid(&s.species) == bp_species
                    && s.moves.iter().any(|m| toid(m) == "batonpass")
                    && s.moves.len() == 4
            });
            assert!(
                has_bp_set,
                "supplemental prior must provide a 4-move Baton Pass set for {bp_species}"
            );
        }
    }

    #[test]
    fn bayesian_fallback_infers_baton_pass_from_setup_reveals_and_teammate_context() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();

        // Custom high-speed Baton Pass team (off-pool because of custom composition)
        let custom: Vec<PokemonSet> = [
            ("Jolteon", "Leftovers", 50, &["Agility", "Baton Pass", "Substitute", "Thunderbolt"][..]),
            ("Marowak", "Thick Club", 55, &["Earthquake", "Rock Slide", "Swords Dance", "Body Slam"][..]),
            ("Scizor", "Miracle Berry", 50, &["Swords Dance", "Agility", "Baton Pass", "Steel Wing"][..]),
            ("Espeon", "Gold Berry", 50, &["Growth", "Baton Pass", "Psychic", "Substitute"][..]),
            ("Snorlax", "Mint Berry", 50, &["Double-Edge", "Earthquake", "Rest", "Curse"][..]),
            ("Suicune", "Focus Band", 50, &["Surf", "Ice Beam", "Rest", "Roar"][..]),
        ]
        .into_iter()
        .map(|(sp, item, lv, mvs)| PokemonSet {
            name: sp.to_string(),
            species: sp.to_string(),
            item: item.to_string(),
            ability: "No Ability".to_string(),
            moves: mvs.iter().map(|m| m.to_string()).collect(),
            level: lv,
            evs: None,
            ivs: None,
            happiness: None,
            gender: Some(if sp == "Suicune" { "N" } else { "M" }.to_string()),
        })
        .collect();

        let mut battle =
            Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[0].sets, &custom).unwrap();
        let mut obs = Observer::new(&battle, 0);
        let mut belief = Belief::new(&dex, &pool, &obs);
        assert!(belief.is_fallback());

        // Scizor and Espeon (absent from belief-pool-v3) immediately get 4-move sets with Baton Pass
        let scizor_ref = &belief.refs(None)[2];
        assert_eq!(scizor_ref.base_move_slots.len(), 4);
        assert!(scizor_ref
            .base_move_slots
            .iter()
            .any(|m| dex.moves.key(m.id) == "batonpass"));
        let espeon_ref = &belief.refs(None)[3];
        assert_eq!(espeon_ref.base_move_slots.len(), 4);
        assert!(espeon_ref
            .base_move_slots
            .iter()
            .any(|m| dex.moves.key(m.id) == "batonpass"));

        // Jolteon switches in and reveals Agility: posterior loadouts must carry Baton Pass
        battle.sides[1].roster[0].is_active = true;
        obs.observe(&battle, &dex);
        obs.ingest_line("|move|p2a: Jolteon|Agility|p1a: Target", &dex);
        belief.sync(&dex, &obs);

        let jolteon_ref = &belief.refs(None)[0];
        assert!(
            jolteon_ref
                .base_move_slots
                .iter()
                .any(|m| dex.moves.key(m.id) == "batonpass"),
            "Jolteon fallback ref must retain Baton Pass after revealing Agility"
        );
        for seed in 0..64 {
            let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(seed));
            let moves: Vec<&str> = det.sides[1].roster[0]
                .base_move_slots
                .iter()
                .map(|m| dex.moves.key(m.id))
                .collect();
            assert!(
                moves.contains(&"agility") && moves.contains(&"batonpass"),
                "determinized Jolteon moves after Agility must contain batonpass: {moves:?}"
            );
        }

        // Novel move combination backoff: Jolteon reveals Agility + Attract (no exact set match)
        // Conditional co-occurrence with Agility still retains Baton Pass without move-name rules!
        let novel_jolteon = mon(&dex, "jolteon", 50, &["agility", "attract"]);
        let novel_set = belief.fallback_set(&dex, &novel_jolteon);
        assert_eq!(novel_set.moves.len(), 4);
        assert!(
            novel_set.moves.iter().any(|m| toid(m) == "batonpass"),
            "novel Agility + Attract Jolteon must retain Baton Pass via co-occurrence: {:?}",
            novel_set.moves
        );

        // Novel move combination on Misdreavus: Mean Look + Attract retains Perish Song via co-occurrence
        let novel_misdreavus = mon(&dex, "misdreavus", 50, &["meanlook", "attract"]);
        let novel_mis_set = belief.fallback_set(&dex, &novel_misdreavus);
        assert_eq!(novel_mis_set.moves.len(), 4);
        assert!(
            novel_mis_set.moves.iter().any(|m| toid(m) == "perishsong"),
            "novel Mean Look + Attract Misdreavus must retain Perish Song via co-occurrence: {:?}",
            novel_mis_set.moves
        );

        // Tier A verification: Jolteon loadout_draws contains multiple distinct Agility+BatonPass loadouts
        let tier_a_draw = belief.loadout_draws.as_ref().unwrap()[0].as_ref().unwrap();
        assert!(
            (2..=4).contains(&tier_a_draw.entries.len()),
            "Tier A should retain 2..=4 top loadouts, got {}",
            tier_a_draw.entries.len()
        );
        // Verify predict_fallback_draws advances RNG bit-identically to determinize_in_place
        for seed in 0..32 {
            let mut rng_probe = SplitMix64::new(seed);
            let fb_key = belief.predict_fallback_draws(None, &battle, 1, &mut rng_probe);
            assert!(fb_key.is_some());
            let mut rng_actual = SplitMix64::new(seed);
            let _ = belief.sample_fallback_refs(None, &mut rng_actual, &battle, 1);
            assert_eq!(
                rng_probe.0, rng_actual.0,
                "predict_fallback_draws and sample_fallback_refs RNG must match"
            );
        }

        // Tier B verification: Jolteon reveals Attract (novel move combination Agility + Attract)
        obs.ingest_line("|move|p2a: Jolteon|Attract|p1a: Target", &dex);
        belief.sync(&dex, &obs);
        let tier_b_draw = belief.loadout_draws.as_ref().unwrap()[0]
            .as_ref()
            .expect("Tier B must construct LoadoutDraw for novel move combination");
        assert!(
            (2..=4).contains(&tier_b_draw.entries.len()),
            "Tier B should construct 2..=4 novel completions, got {}",
            tier_b_draw.entries.len()
        );
        for (p, _) in &tier_b_draw.entries {
            let mvs: Vec<&str> = p
                .base_move_slots
                .iter()
                .map(|m| dex.moves.key(m.id))
                .collect();
            assert_eq!(mvs.len(), 4);
            assert!(
                mvs.contains(&"agility") && mvs.contains(&"attract") && mvs.contains(&"batonpass"),
                "every Tier B completion for Agility+Attract Jolteon must retain Baton Pass: {mvs:?}"
            );
        }
    }

    #[test]
    fn unrevealed_bench_ignores_hp_observations_and_scrub_enforces_locked_move() {
        use nc2000_engine::battle::{EffectHandle, SearchChoice};
        use nc2000_engine::state::{Action, ActionKind};
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();
        let p1 = pool.teams[0].sets.clone();
        let mut p2 = pool.teams[0].sets.clone();
        p2[0].moves = vec![
            "Outrage".into(),
            "Earthquake".into(),
            "Rest".into(),
            "Sleep Talk".into(),
        ];
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        battle
            .apply_choices(&dex, [Some(SearchChoice::Team([1, 2, 3])), Some(SearchChoice::Team([1, 2, 3]))])
            .unwrap();
        let mut obs = Observer::new(&battle, 0);
        obs.observe(&battle, &dex);
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.set_hp_observations(vec![
            (100, 100),
            (1, 100),
            (100, 100),
            (100, 100),
            (100, 100),
            (100, 100),
        ]);
        let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(7));
        assert_eq!(det.sides[1].roster[1].hp, det.sides[1].roster[1].maxhp);

        let opp_act = battle.active_id(1).unwrap();
        let outrage = dex.moves.id("outrage").unwrap();
        let earthquake = dex.moves.id("earthquake").unwrap();
        battle.add_volatile(
            &dex,
            opp_act,
            "lockedmove",
            Some(opp_act),
            EffectHandle::MoveEff(outrage),
        );
        if let Some(lm) = battle.poke_mut(opp_act).volatile_mut(dex.conds_id("lockedmove").unwrap()) {
            lm.duration = Some(2);
        }
        battle.queue.push(Action {
            choice: ActionKind::Move {
                move_id: earthquake,
                target_loc: 1,
                original_target: None,
                source_effect: None,
            },
            order: 200,
            priority: 0.0,
            fractional_priority: 0.0,
            speed: 1.0,
            pokemon: Some(opp_act),
        });
        let det_locked = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(9));
        let queued = det_locked
            .queue
            .iter()
            .find(|q| q.pokemon == Some(opp_act))
            .expect("queued foe action");
        match queued.choice {
            ActionKind::Move { move_id, .. } => assert_eq!(move_id, outrage),
            _ => panic!("expected move action"),
        }
    }

    #[test]
    fn near_pool_soft_shell_repair_retains_exact_untouched_slots() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();
        let p1 = pool.teams[0].sets.clone();
        let mut p2 = pool.teams[1].sets.clone();
        // Replace slot 0 with an off-pool species at L50, leaving slots 1..=5 identical to pool team 1.
        let untouched_species: Vec<String> = p2[1..]
            .iter()
            .map(|s| s.species.clone())
            .collect();
        p2[0] = PokemonSet {
            name: "Pinsir".into(),
            species: "Pinsir".into(),
            item: "Leftovers".into(),
            ability: "No Ability".into(),
            moves: vec![
                "Swords Dance".into(),
                "Submission".into(),
                "Body Slam".into(),
                "Hidden Power Bug".into(),
            ],
            level: 50,
            evs: None,
            ivs: None,
            happiness: Some(255),
            gender: Some("M".into()),
        };
        // Avoid item duplication with pool.teams[1] slots 1..=5
        p2[0].item = "BrightPowder".into();
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        battle
            .apply_choices(
                &dex,
                [
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                ],
            )
            .unwrap();
        let mut obs = Observer::new(&battle, 0);
        obs.observe(&battle, &dex);
        let belief = Belief::new(&dex, &pool, &obs);
        assert!(belief.is_fallback());
        assert!(
            belief.fallback_cand_count() > 0,
            "in-battle near-pool fallback must populate fallback_cands (untouched={untouched_species:?})"
        );
        let fc = belief.fallback_cands.as_ref().unwrap();
        // Verify at least one high-weight entry matches pool.teams[1] exact items+moves on all 5 untouched slots
        let expected_refs = build_refs(&dex, &pool.teams[1].sets[1..], &obs.mons()[1..]);
        let _ = expected_refs;
        let matching_entry = fc.entries.iter().any(|(team, _)| {
            (1..team.len()).all(|s| {
                let mut m_team: Vec<u16> = team[s].base_move_slots.iter().map(|m| m.id.0).collect();
                m_team.sort_unstable();
                let mut m_true: Vec<u16> =
                    battle.sides[1].roster[s].base_move_slots.iter().map(|m| m.id.0).collect();
                m_true.sort_unstable();
                team[s].species == battle.sides[1].roster[s].species
                    && team[s].item == battle.sides[1].roster[s].item
                    && m_team == m_true
            })
        });
        assert!(
            matching_entry,
            "near-pool soft shell repair must retain exact untouched pool slots 1..=5"
        );
    }

    #[test]
    fn global_item_clause_and_turn_zero_invariance_hold_with_exclusions() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();
        let p1 = pool.teams[0].sets.clone();
        let mut p2 = pool.teams[2].sets.clone();
        // Replace slot 5 with an off-pool species so the team exits the pool cleanly.
        p2[5] = PokemonSet {
            name: "Pinsir".into(),
            species: "Pinsir".into(),
            item: "BrightPowder".into(),
            ability: "No Ability".into(),
            moves: vec![
                "Swords Dance".into(),
                "Submission".into(),
                "Body Slam".into(),
                "Hidden Power Bug".into(),
            ],
            level: 50,
            evs: None,
            ivs: None,
            happiness: Some(255),
            gender: Some("M".into()),
        };
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        let obs_t0 = Observer::new(&battle, 0);
        let belief_t0 = Belief::new(&dex, &pool, &obs_t0);
        // Choice A / Turn 0 invariance: no appeared mon yet -> no fallback_cands or loadout_draws.
        assert_eq!(belief_t0.fallback_cand_count(), 0);
        for slot in 0..6 {
            assert_eq!(belief_t0.loadout_draw_len(slot), 0);
        }

        battle
            .apply_choices(
                &dex,
                [
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                ],
            )
            .unwrap();
        let mut obs = Observer::new(&battle, 0);
        obs.observe(&battle, &dex);
        // Simulate taking chip damage on p2a without any Leftovers heal event -> excludes Leftovers on slot 0.
        let p2a = battle.sides[1].roster[0].name.clone();
        obs.ingest_line(&format!("|-damage|p2a: {p2a}|90/100"), &dex);
        obs.ingest_line("|upkeep", &dex);
        let belief = Belief::new(&dex, &pool, &obs);
        assert!(belief.is_fallback());
        assert!(belief.fallback_cand_count() > 1);
        let leftovers = dex.items.id("leftovers").unwrap();
        let any_bench_diverse = (1..6).any(|slot| belief.loadout_draw_len(slot) > 1);
        assert!(
            any_bench_diverse,
            "in-battle unappeared bench should have multi-loadout diversity"
        );

        for seed in 0..64 {
            let mut rng_probe = SplitMix64::new(seed);
            let fb_key = belief.predict_fallback_draws(None, &battle, 1, &mut rng_probe);
            assert!(fb_key.is_some());
            let mut rng_actual = SplitMix64::new(seed);
            let _ = belief.sample_fallback_refs(None, &mut rng_actual, &battle, 1);
            assert_eq!(rng_probe.0, rng_actual.0);

            let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(seed));
            let mut seen_items = Vec::new();
            for (slot, mon) in det.sides[1].roster.iter().enumerate() {
                if let Some(id) = mon.item {
                    assert!(
                        !seen_items.contains(&id),
                        "duplicate item {} across determinized roster on seed {seed}",
                        dex.items.key(id)
                    );
                    seen_items.push(id);
                }
                if slot == 0 {
                    assert_ne!(
                        mon.item,
                        Some(leftovers),
                        "active slot 0 must not receive excluded Leftovers"
                    );
                }
            }
        }
    }

    #[test]
    fn systematic_resample_respects_skewed_weights_and_skips_zero() {
        let weights = [0.0, 0.7, 0.2, 0.1, 0.0];
        let k = 24;
        for seed in 0..16 {
            let mut rng = SplitMix64::new(seed);
            let picked = systematic_resample_indices(&weights, k, &mut rng);
            assert_eq!(picked.len(), k);
            let mut counts = [0usize; 5];
            for idx in picked {
                counts[idx] += 1;
            }
            assert_eq!(counts[0], 0);
            assert_eq!(counts[4], 0);
            for i in 1..4 {
                let exp = k as f64 * weights[i];
                let lo = exp.floor() as usize;
                let hi = exp.ceil() as usize;
                assert!(
                    counts[i] >= lo && counts[i] <= hi,
                    "seed {seed}: slot {i} count {} not in [{lo}, {hi}]",
                    counts[i]
                );
            }
        }
    }

    #[test]
    fn all_gen2_learnset_species_have_empirical_prior_coverage_and_valid_bayesian_draws() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();
        let sup = supplemental_prior_teams(&dex);
        let ren = community_rental_teams(&dex);
        let raw_learnsets: Value = serde_json::from_str(LEARNSETS_JSON).unwrap();
        let mut uncovered = Vec::new();
        for species_key in raw_learnsets["species"].as_object().unwrap().keys() {
            let sid = dex.species.id(species_key).unwrap();
            let in_pool = pool.teams.iter().any(|t| {
                t.sets
                    .iter()
                    .any(|s| dex.species.id(&toid(&s.species)) == Some(sid))
            });
            let in_sup = sup.iter().any(|t| {
                t.sets
                    .iter()
                    .any(|s| dex.species.id(&toid(&s.species)) == Some(sid))
            });
            let in_ren = ren.iter().any(|sets| {
                sets.iter()
                    .any(|s| dex.species.id(&toid(&s.species)) == Some(sid))
            });
            if !(in_pool || in_sup || in_ren) {
                uncovered.push(species_key.to_string());
            }
        }
        assert!(
            uncovered.is_empty(),
            "every format-legal Gen 2 species must have prior coverage; uncovered: {uncovered:?}"
        );

        // Verify a previously zero-prior species (Dunsparce) now determinizes empirical moves and item in Bayesian mode.
        let p1 = pool.teams[0].sets.clone();
        let mut p2 = pool.teams[0].sets.clone();
        p2[0] = plain_set("Dunsparce");
        p2[0].item = "Leftovers".into();
        p2[0].moves = vec![
            "Glare".into(),
            "Screech".into(),
            "Return".into(),
            "Flamethrower".into(),
        ];
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        battle
            .apply_choices(
                &dex,
                [
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                ],
            )
            .unwrap();
        let mut obs = Observer::new(&battle, 0);
        obs.observe(&battle, &dex);
        obs.ingest_line("|move|p2a: Dunsparce|Glare|p1a: Target", &dex);
        let belief = Belief::new(&dex, &pool, &obs);
        assert!(belief.is_fallback());
        let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(1));
        let det_moves: Vec<&str> = det.sides[1].roster[0]
            .base_move_slots
            .iter()
            .map(|s| dex.moves.key(s.id))
            .collect();
        assert!(
            det_moves.contains(&"screech") && det_moves.contains(&"flamethrower"),
            "Dunsparce empirical prior should predict Screech + Flamethrower: {det_moves:?}"
        );
    }

    #[test]
    fn synthetic_special_sets_optimize_low_attack_and_preserve_spread_signatures() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();
        let mut mo = mon(&dex, "alakazam", 50, &["psychic", "thunderwave", "recover", "encore"]);
        mo.gender = Gender::F;
        let mut set = plain_set("Alakazam");
        set.moves = vec![
            "Psychic".into(),
            "Thunder Wave".into(),
            "Recover".into(),
            "Encore".into(),
        ];
        set.gender = Some("F".into());
        apply_low_attack_spread_if_special(&dex, &mut set, &mo);
        assert_eq!(set.evs.as_ref().unwrap()["atk"], 0);
        assert_eq!(set.ivs.as_ref().unwrap()["atk"], 3);

        let p1 = pool.teams[0].sets.clone();
        let fast_mon = &Battle::from_fixture(&dex, "1,2,3,4", &p1, &p1).unwrap().sides[0].roster[0];
        let slow_mon = with_reduced_speed_spread(&dex, fast_mon);
        assert_eq!(slow_mon.set_evs[5], 0);
        assert!(slow_mon.set_ivs[5] <= 2);
        assert!(slow_mon.base_stored_stats[4] < fast_mon.base_stored_stats[4]);
        assert_eq!(slow_mon.base_hp_type, fast_mon.base_hp_type);
        assert_eq!(slow_mon.gender, fast_mon.gender);
        assert_ne!(
            team_signature(std::slice::from_ref(fast_mon)),
            team_signature(std::slice::from_ref(&slow_mon)),
            "team_signature must distinguish different EV/IV stat spreads"
        );
    }

    #[test]
    fn bayesian_loadout_draw_diversifies_reduced_speed_spread_for_slow_pivots() {
        let dex = test_dex();
        let pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();
        let p1 = pool.teams[0].sets.clone();
        let mut p2 = pool.teams[0].sets.clone();
        p2[0] = plain_set("Umbreon");
        p2[0].name = "SlowUmbreon".into();
        p2[0].item = "Leftovers".into();
        p2[0].moves = vec![
            "Curse".into(),
            "Baton Pass".into(),
            "Mean Look".into(),
            "Moonlight".into(),
        ];
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        battle
            .apply_choices(
                &dex,
                [
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                ],
            )
            .unwrap();
        let mut obs = Observer::new(&battle, 0);
        obs.observe(&battle, &dex);
        obs.ingest_line("|move|p2a: SlowUmbreon|Curse|p2a: SlowUmbreon", &dex);
        obs.ingest_line("|move|p2a: SlowUmbreon|Baton Pass|p2a: SlowUmbreon", &dex);
        obs.ingest_line("|move|p2a: SlowUmbreon|Mean Look|p1a: Target", &dex);
        obs.ingest_line("|move|p2a: SlowUmbreon|Moonlight|p2a: SlowUmbreon", &dex);
        let belief = Belief::new(&dex, &pool, &obs);
        assert!(belief.is_fallback());
        let ld = belief.loadout_draws.as_ref().unwrap()[0]
            .as_ref()
            .expect("Umbreon slow pivot must construct LoadoutDraw");
        let distinct_speeds: Vec<i32> = ld
            .entries
            .iter()
            .map(|(m, _)| m.base_stored_stats[4])
            .collect();
        let min_spe = *distinct_speeds.iter().min().unwrap();
        let max_spe = *distinct_speeds.iter().max().unwrap();
        assert!(
            min_spe < max_spe,
            "slow pivot LoadoutDraw must include both max-Speed and reduced-Speed spread: {distinct_speeds:?}"
        );
    }

    #[test]
    fn marginal_tiebreak_does_not_inflate_tiny_weight_outlier_via_revealed_or_shared_moves() {
        let dex = test_dex();
        let mut pool: MetaPool = serde_json::from_str(include_str!(
            "../../../data/belief-pool-v3/belief-pool.json"
        ))
        .unwrap();
        let p1 = pool.teams[0].sets.clone();
        // Inject a tiny-weight 1-off outlier sharing revealed Protect + common Mean Look / Perish Song.
        let mut outlier_team = pool.teams[0].clone();
        outlier_team.id = "tiny-outlier".into();
        outlier_team.weight = Some(1e-5);
        outlier_team.sets[0] = plain_set("Misdreavus");
        outlier_team.sets[0].item = "BrightPowder".into();
        outlier_team.sets[0].moves = vec![
            "Protect".into(),
            "Mean Look".into(),
            "Perish Song".into(),
            "Spite".into(),
        ];
        pool.teams.push(outlier_team);

        let mut p2 = p1.clone();
        p2[0] = plain_set("Misdreavus");
        p2[0].name = "OffPoolMissy".into();
        p2[0].item = "Leftovers".into();
        p2[0].moves = vec![
            "Protect".into(),
            "Mean Look".into(),
            "Perish Song".into(),
            "Confuse Ray".into(),
        ];
        // Make p2 off-pool by replacing slot 5 with an uncommon species.
        p2[5] = plain_set("Porygon2");
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        battle
            .apply_choices(
                &dex,
                [
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                    Some(nc2000_engine::battle::SearchChoice::Team([1, 2, 3])),
                ],
            )
            .unwrap();
        let mut obs = Observer::new(&battle, 0);
        obs.observe(&battle, &dex);
        obs.ingest_line("|move|p2a: OffPoolMissy|Protect|p2a: OffPoolMissy", &dex);
        let belief = Belief::new(&dex, &pool, &obs);
        assert!(belief.is_fallback());
        let ld = belief.loadout_draws.as_ref().unwrap()[0].as_ref().unwrap();
        let total_w: f64 = ld.entries.iter().map(|(_, w)| *w).sum();
        let spite_id = dex.moves.id("spite").unwrap();
        let spite_w: f64 = ld
            .entries
            .iter()
            .filter(|(m, _)| m.base_move_slots.iter().any(|s| s.id == spite_id))
            .map(|(_, w)| *w)
            .sum();
        assert!(
            spite_w / total_w < 0.02,
            "1e-5 outlier set must not borrow marginal mass from revealed/shared staples: ratio={}",
            spite_w / total_w
        );
    }

    #[test]
    fn explicit_typed_hidden_power_contradicted_by_mask_or_gender_is_strictly_excluded() {
        let dex = test_dex();
        let grass_idx = crate::hp_evidence::TYPES
            .iter()
            .position(|&t| t == "Grass")
            .unwrap();
        let ice_idx = crate::hp_evidence::TYPES
            .iter()
            .position(|&t| t == "Ice")
            .unwrap();
        let only_grass = 1u16 << grass_idx;
        assert!(explicit_hp_contradicted(
            &dex,
            "Hidden Power Ice",
            only_grass,
            only_grass
        ));
        assert!(!explicit_hp_contradicted(
            &dex,
            "Hidden Power Grass",
            only_grass,
            only_grass
        ));
        // Female Raichu (1:1 gender ratio, Attack DV <= 7) cannot legally have Fighting HP (idx 0, requires Atk DV >= 8? Check against possible_types).
        let raichu = dex.species.id("raichu").unwrap();
        let fem_poss =
            crate::hp_evidence::possible_types(&dex, raichu, nc2000_engine::state::Gender::F);
        for (idx, ty) in crate::hp_evidence::TYPES.iter().enumerate() {
            let mv_name = format!("Hidden Power {ty}");
            let impossible = (fem_poss & (1u16 << idx)) == 0;
            assert_eq!(
                explicit_hp_contradicted(&dex, &mv_name, fem_poss, u16::MAX),
                impossible,
                "mismatch for {mv_name} under female DV mask"
            );
        }
        let _ = ice_idx;
    }
}
