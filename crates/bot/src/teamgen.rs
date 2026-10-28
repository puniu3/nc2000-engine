//! M11a/PSRO metagame-research machinery: mutation operators over the
//! *non-dominated* legal set space plus gauntlet fitness evaluation.
//!
//! Provable Dominance Pruning (with Pre-Adaptation Unlock):
//! 1. DVs / EVs / Gender: deterministic in canonicalization (`EvTweak` and
//!    `DvTweak` removed from random mutation). Every stat gets 255 Stat Exp
//!    and 15 DV except special/status-only sets without Hidden Power or
//!    move-calling moves, which deterministically get `Atk EV = 0, Atk DV = 1`
//!    (minimizing confusion self-hit while keeping `HP DV = 15` and `F` gender).
//! 2. Items: excludes the 21 zero-effect items (balls, stones, bugged Gen 2
//!    Dragon Scale), restricts the 5 species-locked items to their species,
//!    restricts the 17 type-boosting items to holders carrying a damaging move
//!    of that type, deduplicates Pink/Polkadot Bow, and excludes dominated
//!    berries (Berry/Berry Juice when Gold Berry is free; single-status berries
//!    and non-Rest Mint Berry when Miracle Berry is free). Also supports
//!    swapping held items between teammates to cross Item Clause valleys.
//! 3. Species: excludes the 90 unconditionally strictly dominated pre-evolutions
//!    from random draws and promotes any set on a pre-evolution to a same-type,
//!    strictly higher-stat evolution whenever all carried moves are legal at
//!    that level and the evolved species is not already on the team. Sets are
//!    derived *ab initio* from the non-dominated learnset (never copied from
//!    `meta-pool.json`).
//! 4. Moves & Pre-Adaptation: excludes zero-effect moves (`Splash`, `Teleport`,
//!    `False Swipe`, `Pay Day`), exact functional aliases (`Frustration` when
//!    `Return` is legal, `Roar`/`Whirlwind`, `Mean Look`/`Spider Web`, etc.),
//!    and moves strictly dominated within the species's own learnset at its
//!    level (same type, >= priority/crit/accuracy/PP, strictly higher BP, same
//!    or strictly better secondary, no new drawback). Dependent combo moves
//!    (`Sleep Talk`/`Snore` requiring `Rest`, `Dream Eater`/`Nightmare`
//!    requiring a sleep move, `Defense Curl` requiring `Rollout` when `+2 Def`
//!    is available) unlock only when their pre-adaptation move is present.
//! 5. Levels: transitions over the 65 Pareto-optimal (slack-free) legal 6-level
//!    allocations under `Max Total Level = 155` or swaps levels between slots.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::state::Battle;
use nc2000_engine::validate::{canonicalize_team, validate_team, Learnsets};

use crate::blind::BlindAgent;
use crate::preview::MetaPool;
use crate::rng::SplitMix64;
use crate::runner::{play_game, GameResult};
use crate::smmcts::{RmAgent, RmConfig, SelRule};
use crate::Agent;

/// Gen-2 Hidden Power types (typed move ids exist for each).
const HP_TYPES: [&str; 16] = [
    "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
    "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark",
];

/// Items with zero in-battle effect in Gen 2 Stadium 2 (balls, evolution items,
/// and bugged Dragon Scale). `Mail` is kept as the single Thief-blocking item.
const ZERO_EFFECT_ITEMS: [&str; 21] = [
    "fastball",
    "heavyball",
    "levelball",
    "loveball",
    "lureball",
    "moonball",
    "sportball",
    "friendball",
    "greatball",
    "masterball",
    "pokeball",
    "ultraball",
    "firestone",
    "leafstone",
    "moonstone",
    "sunstone",
    "thunderstone",
    "waterstone",
    "upgrade",
    "dragonscale",
    "",
];

/// Species-specific items: `(item_id, allowed_species_ids)`.
const SPECIES_ITEMS: [(&str, &[&str]); 5] = [
    ("thickclub", &["cubone", "marowak"]),
    ("lightball", &["pikachu"]),
    ("metalpowder", &["ditto"]),
    ("luckypunch", &["chansey"]),
    ("stick", &["farfetchd"]),
];

/// Type-boosting items (1.1x): `(item_id, MoveType)`.
const TYPE_BOOST_ITEMS: [(&str, &str); 18] = [
    ("charcoal", "Fire"),
    ("mysticwater", "Water"),
    ("magnet", "Electric"),
    ("miracleseed", "Grass"),
    ("nevermeltice", "Ice"),
    ("twistedspoon", "Psychic"),
    ("blackbelt", "Fighting"),
    ("blackglasses", "Dark"),
    ("dragonfang", "Dragon"),
    ("hardstone", "Rock"),
    ("metalcoat", "Steel"),
    ("pinkbow", "Normal"),
    ("polkadotbow", "Normal"),
    ("poisonbarb", "Poison"),
    ("sharpbeak", "Flying"),
    ("silverpowder", "Bug"),
    ("softsand", "Ground"),
    ("spelltag", "Ghost"),
];

/// Status move strict dominance pairs `(dominated, dominator)` when both are
/// legal on the same species at the same level.
const STATUS_DOMINANCE: [(&str, &str); 27] = [
    ("growl", "charm"),
    ("leer", "tailwhip"),
    ("sharpen", "swordsdance"),
    ("meditate", "swordsdance"),
    ("harden", "withdraw"),
    ("harden", "acidarmor"),
    ("harden", "barrier"),
    ("withdraw", "acidarmor"),
    ("withdraw", "barrier"),
    ("kinesis", "sandattack"),
    ("kinesis", "smokescreen"),
    ("flash", "sandattack"),
    ("flash", "smokescreen"),
    ("smokescreen", "sandattack"),
    ("poisongas", "poisonpowder"),
    ("poisongas", "toxic"),
    ("poisonpowder", "toxic"),
    ("sing", "hypnosis"),
    ("sing", "lovelykiss"),
    ("sing", "sleeppowder"),
    ("sing", "spore"),
    ("hypnosis", "lovelykiss"),
    ("hypnosis", "sleeppowder"),
    ("hypnosis", "spore"),
    ("sleeppowder", "spore"),
    ("lovelykiss", "spore"),
    ("lovelykiss", "sleeppowder"),
];

const STATUS_DOMINANCE_EXTRA: [(&str, &str); 29] = [
    ("supersonic", "sweetkiss"),
    ("supersonic", "confuseray"),
    ("sweetkiss", "confuseray"),
    ("meanlook", "spiderweb"),
    ("roar", "whirlwind"),
    ("detect", "protect"),
    ("snore", "sleeptalk"),
    ("flash", "doubleteam"),
    ("kinesis", "doubleteam"),
    ("smokescreen", "doubleteam"),
    ("sandattack", "doubleteam"),
    ("tailwhip", "screech"),
    ("leer", "screech"),
    ("tailwhip", "curse"),
    ("leer", "curse"),
    ("growl", "curse"),
    ("sweetscent", "curse"),
    ("harden", "curse"),
    ("withdraw", "curse"),
    ("harden", "reflect"),
    ("withdraw", "reflect"),
    ("tailwhip", "reflect"),
    ("leer", "reflect"),
    ("growl", "reflect"),
    ("stringshot", "icywind"),
    ("stringshot", "thunderwave"),
    ("stringshot", "stunspore"),
    ("stringshot", "glare"),
    ("stringshot", "scaryface"),
];

/// Damaging moves with unique mechanics that prevent generic BP/Acc/PP
/// dominance pruning.
const UNIQUE_DAMAGE_MOVES: [&str; 33] = [
    "rapidspin",
    "pursuit",
    "beatup",
    "thief",
    "dynamicpunch",
    "bide",
    "counter",
    "mirrorcoat",
    "flail",
    "reversal",
    "present",
    "magnitude",
    "psywave",
    "rollout",
    "whirlpool",
    "firespin",
    "bind",
    "wrap",
    "clamp",
    "solarbeam",
    "fly",
    "dig",
    "hyperbeam",
    "outrage",
    "thrash",
    "petaldance",
    "ancientpower",
    "icywind",
    "mudslap",
    "zapcannon",
    "triattack",
    "superfang",
    "seismictoss",
];

const FIXED_OR_PRIORITY_MOVES: [&str; 8] = [
    "nightshade",
    "sonicboom",
    "dragonrage",
    "machpunch",
    "quickattack",
    "extremespeed",
    "vitalthrow",
    "bonemerang",
];

// -------------------------------------------------------------- operators

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutOp {
    /// Replace (or, when fewer than 4, sometimes add) one move with another
    /// from the species' non-dominated, pre-adaptation-unlocked learnset.
    MoveSwap,
    /// Replace held item from the non-dominated item lens for this slot, or
    /// swap items with a teammate when legal for both.
    ItemSwap,
    /// Transition across Pareto-optimal (slack-free) team level allocations or
    /// swap levels between two teammates.
    LevelTweak,
    /// Retained for enum compatibility; DVs are deterministically optimized.
    DvTweak,
    /// Retained for enum compatibility; EVs are deterministically optimized.
    EvTweak,
    /// New species from the non-dominated legal species pool (Species Clause
    /// honored); set derived ab initio from non-dominated moves/items.
    SpeciesSwap,
    /// Retained for enum compatibility; pool-set copying is retired.
    MonReplace,
}

impl MutOp {
    pub const ALL: [MutOp; 4] = [
        MutOp::MoveSwap,
        MutOp::ItemSwap,
        MutOp::LevelTweak,
        MutOp::SpeciesSwap,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MutOp::MoveSwap => "move-swap",
            MutOp::ItemSwap => "item-swap",
            MutOp::LevelTweak => "level-tweak",
            MutOp::DvTweak => "dv-tweak",
            MutOp::EvTweak => "ev-tweak",
            MutOp::SpeciesSwap => "species-swap",
            MutOp::MonReplace => "mon-replace",
        }
    }
}

/// A validator-clean neighbor of the parent team.
pub struct Proposal {
    pub op: MutOp,
    pub slot: usize,
    /// Canonical team JSON (six set objects).
    pub team: Vec<Value>,
}

/// Optional niche / diversity constraint to prevent convergence ("carcinization")
/// into a single meta archetype across parallel lineages.
#[derive(Clone, Debug, Default)]
pub struct NicheConstraint {
    /// Slots `0..anchor_slots` cannot change their species (nor promote away
    /// from their species), though their moves, items, and levels evolve freely.
    pub anchor_slots: usize,
    /// Species ids that may not appear on the team.
    pub forbidden_species: BTreeSet<String>,
    /// Optional cap `(staple_set, max_count)` on how many team members may
    /// belong to `staple_set`.
    pub staple_cap: Option<(BTreeSet<String>, usize)>,
}

impl NicheConstraint {
    pub fn allows_team(&self, raw: &[Value], parent: Option<&[Value]>) -> bool {
        if let Some(p) = parent {
            for i in 0..self.anchor_slots.min(raw.len()).min(p.len()) {
                if species_id(&raw[i]) != species_id(&p[i]) {
                    return false;
                }
            }
        }
        let mut staple_count = 0usize;
        for set in raw {
            let sid = species_id(set);
            if self.forbidden_species.contains(&sid) {
                return false;
            }
            if let Some((staples, _)) = &self.staple_cap {
                if staples.contains(&sid) {
                    staple_count += 1;
                }
            }
        }
        if let Some((_, max_c)) = &self.staple_cap {
            if staple_count > *max_c {
                return false;
            }
        }
        true
    }
}

// -------------------------------------------------------------- pool teams

pub struct PoolTeamRaw {
    pub id: String,
    pub tier: String,
    /// Raw set objects exactly as the pool file carries them.
    pub sets: Vec<Value>,
}

// ---------------------------------------------------------------- TeamGen

pub struct TeamGen {
    ls: Learnsets,
    /// Non-dominated format-legal species ids (sorted; 156 of 246).
    species_ids: Vec<String>,
    /// For each species id, same-type strictly higher-stat species ids (sorted
    /// by base-stat total descending) for set-level dominance promotion.
    species_dominators: BTreeMap<String, Vec<String>>,
    /// Every item display name in the format lens (sorted by id).
    all_items: Vec<String>,
    /// Slack-free Pareto-optimal sorted 6-level profiles under sum <= 155,
    /// grouped by macrostate `k = count(level == 55)` in `0..=4` so two-stage
    /// Macro -> Micro sampling gives each Ace-count macrostate equal 20% weight.
    pareto_by_aces: Vec<Vec<[i64; 6]>>,
    teams: Vec<PoolTeamRaw>,
}

impl TeamGen {
    /// `learnsets_json` = data/learnsets-gen2.json, `pool_json` =
    /// data/meta-pool-v0/meta-pool.json (or any `{teams:[...]}` catalog).
    pub fn new(dex: &Dex, learnsets_json: &str, pool_json: &str) -> Result<TeamGen, String> {
        let ls = Learnsets::from_json(learnsets_json)?;
        let v: Value = serde_json::from_str(learnsets_json).map_err(|e| e.to_string())?;
        let mut all_species: Vec<String> = v["species"]
            .as_object()
            .ok_or("learnsets: missing species table")?
            .keys()
            .filter(|id| dex.species.id(id).is_some())
            .cloned()
            .collect();
        all_species.sort();

        // Compute same-type strict base-stat dominators and unconditional
        // species dominance (156 of 246 kept).
        let special_item_holders: BTreeSet<&str> =
            ["pikachu", "cubone", "chansey", "ditto", "farfetchd"].into_iter().collect();
        let mut species_dominators: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut unconditionally_dominated: BTreeSet<String> = BTreeSet::new();

        let larval_evolutions: BTreeMap<&str, &str> = [
            ("caterpie", "butterfree"),
            ("metapod", "butterfree"),
            ("weedle", "beedrill"),
            ("kakuna", "beedrill"),
            ("magikarp", "gyarados"),
            ("poliwhirl", "poliwrath"),
            ("remoraid", "octillery"),
            ("onix", "steelix"),
        ]
        .into_iter()
        .collect();

        for a in &all_species {
            if special_item_holders.contains(a.as_str()) {
                continue;
            }
            let sp_a = dex.species.get(dex.species.id(a).unwrap());
            let ls_a = ls.species(a).unwrap();
            let moves_a: BTreeSet<&String> = ls_a.moves.iter().collect();
            let mut doms: Vec<(i32, String)> = Vec::new();
            if let Some(&evo) = larval_evolutions.get(a.as_str()) {
                if let Some(sp_id) = dex.species.id(evo) {
                    let sp_b = dex.species.get(sp_id);
                    let bst = (sp_b.base_stats.hp as i32)
                        + (sp_b.base_stats.atk as i32)
                        + (sp_b.base_stats.def as i32)
                        + (sp_b.base_stats.spa as i32)
                        + (sp_b.base_stats.spd as i32)
                        + (sp_b.base_stats.spe as i32);
                    doms.push((bst + 1000, evo.to_string()));
                    unconditionally_dominated.insert(a.clone());
                }
            }
            for b in &all_species {
                if a == b {
                    continue;
                }
                let sp_b = dex.species.get(dex.species.id(b).unwrap());
                let ls_b = ls.species(b).unwrap();
                let moves_b: BTreeSet<&String> = ls_b.moves.iter().collect();
                let is_direct_evo = moves_a.len() >= 3 && moves_a.is_subset(&moves_b);
                if sp_a.types != sp_b.types && !is_direct_evo {
                    continue;
                }
                let bs_a = [
                    sp_a.base_stats.hp,
                    sp_a.base_stats.atk,
                    sp_a.base_stats.def,
                    sp_a.base_stats.spa,
                    sp_a.base_stats.spd,
                    sp_a.base_stats.spe,
                ];
                let bs_b = [
                    sp_b.base_stats.hp,
                    sp_b.base_stats.atk,
                    sp_b.base_stats.def,
                    sp_b.base_stats.spa,
                    sp_b.base_stats.spd,
                    sp_b.base_stats.spe,
                ];
                if bs_b.iter().zip(&bs_a).all(|(x, y)| x >= y)
                    && bs_b.iter().zip(&bs_a).any(|(x, y)| x > y)
                {
                    let bst: i32 = bs_b.iter().map(|&x| x as i32).sum();
                    // Prefer direct evolution first (e.g. Charmeleon -> Charizard)
                    let prio = if is_direct_evo { bst + 500 } else { bst };
                    doms.push((prio, b.clone()));
                    if ls_b.min_level <= 55 && moves_a.is_subset(&moves_b) {
                        unconditionally_dominated.insert(a.clone());
                    }
                }
            }
            doms.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
            if !doms.is_empty() {
                species_dominators.insert(a.clone(), doms.into_iter().map(|(_, s)| s).collect());
            }
        }

        let species_ids: Vec<String> = all_species
            .into_iter()
            .filter(|s| !unconditionally_dominated.contains(s))
            .collect();

        let all_items: Vec<String> = dex.items.values.iter().map(|d| d.name.clone()).collect();
        let pareto_levels = compute_pareto_levels();
        let mut pareto_by_aces: Vec<Vec<[i64; 6]>> = vec![Vec::new(); 5];
        for lv in pareto_levels {
            let k = lv.iter().filter(|&&x| x == 55).count().min(4);
            pareto_by_aces[k].push(lv);
        }
        pareto_by_aces.retain(|bucket| !bucket.is_empty());

        let pv: Value = serde_json::from_str(pool_json).map_err(|e| e.to_string())?;
        let mut teams = Vec::new();
        for t in pv["teams"].as_array().ok_or("pool: missing teams")? {
            let id = t["id"].as_str().ok_or("pool team without id")?.to_string();
            let tier = t["tier"].as_str().unwrap_or("").to_string();
            let sets = t["sets"].as_array().ok_or("pool team without sets")?.clone();
            teams.push(PoolTeamRaw { id, tier, sets });
        }

        Ok(TeamGen {
            ls,
            species_ids,
            species_dominators,
            all_items,
            pareto_by_aces,
            teams,
        })
    }

    pub fn teams(&self) -> &[PoolTeamRaw] {
        &self.teams
    }

    /// Non-dominated legal species count (156 under Gen 2 Stadium 2).
    pub fn non_dominated_species(&self) -> &[String] {
        &self.species_ids
    }

    /// A pool team's raw set JSON (canonize before mutating from it).
    pub fn team_json(&self, i: usize) -> Vec<Value> {
        self.teams[i].sets.clone()
    }

    /// Canonicalize + strict-validate a raw team. `None` = illegal (the
    /// canonicalizer could not repair it) — never emit such a team.
    pub fn canonize(&self, dex: &Dex, raw: &[Value]) -> Option<Vec<Value>> {
        let text = serde_json::to_string(raw).ok()?;
        let r = canonicalize_team(dex, &self.ls, &text);
        if r["ok"] != json!(true) {
            return None;
        }
        let team = r["team"].as_array()?.clone();
        // Strict re-validation: canonical output must carry zero findings.
        let v = validate_team(dex, &self.ls, &serde_json::to_string(&team).ok()?);
        if v["ok"] != json!(true) || !v["findings"].as_array()?.is_empty() {
            return None;
        }
        Some(team)
    }

    /// Deterministic dominance cleanup applied to mutated/generated teams
    /// before `canonize`:
    /// - promotes any set on a pre-evolution to a same-type higher-stat
    ///   dominator when all carried moves are legal at its level and the
    ///   dominator is not already on the team;
    /// - sets optimal DVs/EVs (`255` across the board, or `Atk EV = 0` &
    ///   `Atk DV = 1` on physical-less sets without Hidden Power / callers);
    /// - repairs any orphaned pre-adaptation moves, dominated moves, or
    ///   invalidated/dominated items.
    pub fn optimize_team_dominance(&self, dex: &Dex, raw: &mut [Value], rng: &mut SplitMix64) {
        for slot in 0..raw.len() {
            // 1. Set-level species dominance promotion
            let sid = species_id(&raw[slot]);
            let mut level = set_level(&raw[slot]);
            let cur_moves = move_ids(&raw[slot]);
            let is_uncond_dom = !self.species_ids.contains(&sid);
            if let Some(doms) = self.species_dominators.get(&sid) {
                let teammates: Vec<String> = (0..raw.len())
                    .filter(|&i| i != slot)
                    .map(|i| species_id(&raw[i]))
                    .collect();
                for dom in doms {
                    if teammates.contains(dom) {
                        continue;
                    }
                    let Some(ls_dom) = self.ls.species(dom) else { continue };
                    if ls_dom.min_level > level {
                        if is_uncond_dom {
                            if let Some(donor) = (0..raw.len()).find(|&i| {
                                i != slot
                                    && set_level(&raw[i]) >= ls_dom.min_level
                                    && self
                                        .ls
                                        .species(&species_id(&raw[i]))
                                        .is_none_or(|l| l.min_level.max(50) <= level)
                            }) {
                                let donor_lv = set_level(&raw[donor]);
                                raw[donor]["level"] = json!(level);
                                raw[slot]["level"] = json!(donor_lv);
                                level = donor_lv;
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }
                    let all_legal = cur_moves.iter().all(|m| {
                        let b = base_id(m);
                        ls_dom.allows(&b)
                            && ls_dom.move_min_level.get(b.as_str()).is_none_or(|&mn| level >= mn)
                    });
                    if all_legal || is_uncond_dom {
                        let dom_name = dex.species.get(dex.species.id(dom).unwrap()).name.clone();
                        raw[slot]["species"] = json!(dom_name.clone());
                        raw[slot]["name"] = json!(dom_name);
                        break;
                    }
                }
            }

            // 2. Clean up any orphaned pre-adaptation moves (e.g. Sleep Talk without Rest)
            self.repair_orphaned_moves(dex, raw, slot, rng);

            // 3. Clean up item if it became zero-effect/dominated after move or species change
            let cur_item = toid(raw[slot]["item"].as_str().unwrap_or(""));
            if !cur_item.is_empty() {
                let valid_items = self.non_dominated_items(dex, raw, slot, true);
                if !valid_items.iter().any(|it| toid(it) == cur_item) {
                    let cands = self.non_dominated_items(dex, raw, slot, false);
                    if let Some(pick) = cands.get(rng.below(cands.len().max(1))) {
                        raw[slot]["item"] = json!(pick);
                    }
                }
            }

            // 4. Deterministic DV / EV optimization
            optimize_set_dvs_evs(dex, &mut raw[slot]);
        }
    }

    /// One mutation attempt from a CANONICAL parent (6 sets). `None` = the
    /// draw produced an illegal/unchanged team; the caller retries
    /// (`propose_valid`). Deterministic given `rng`.
    pub fn propose(&self, dex: &Dex, parent: &[Value], rng: &mut SplitMix64) -> Option<Proposal> {
        assert_eq!(parent.len(), 6, "mutation parents must be 6-mon teams");
        // Weighted selection over the 4 active non-dominated operators:
        // MoveSwap 40%, ItemSwap 20%, SpeciesSwap 25%, LevelTweak 15%
        let roll = rng.below(20);
        let op = if roll < 8 {
            MutOp::MoveSwap
        } else if roll < 12 {
            MutOp::ItemSwap
        } else if roll < 17 {
            MutOp::SpeciesSwap
        } else {
            MutOp::LevelTweak
        };
        let slot = rng.below(6);
        let mut raw = parent.to_vec();
        match op {
            MutOp::MoveSwap => self.mut_move_swap(dex, &mut raw, slot, rng)?,
            MutOp::ItemSwap => self.mut_item_swap(dex, &mut raw, slot, rng)?,
            MutOp::LevelTweak => self.mut_level_tweak(&mut raw, slot, rng)?,
            MutOp::SpeciesSwap => self.mut_species_swap(dex, &mut raw, slot, rng)?,
            MutOp::DvTweak | MutOp::EvTweak | MutOp::MonReplace => unreachable!(),
        }
        self.optimize_team_dominance(dex, &mut raw, rng);
        let team = self.canonize(dex, &raw)?;
        if team_key(&team) == team_key(parent) {
            return None;
        }
        Some(Proposal { op, slot, team })
    }

    /// `propose` with up to `tries` fresh draws.
    pub fn propose_valid(
        &self,
        dex: &Dex,
        parent: &[Value],
        rng: &mut SplitMix64,
        tries: usize,
    ) -> Option<Proposal> {
        (0..tries).find_map(|_| self.propose(dex, parent, rng))
    }

    /// `propose` honoring a `NicheConstraint` (anchored species slots,
    /// forbidden species, or staple overlap cap). Supports neutral 1-to-3 step
    /// compound transitions using the exact same unbiased atomic operators.
    pub fn propose_constrained(
        &self,
        dex: &Dex,
        parent: &[Value],
        rng: &mut SplitMix64,
        constraint: &NicheConstraint,
        tries: usize,
    ) -> Option<Proposal> {
        for _ in 0..tries {
            let step_roll = rng.below(20);
            let num_steps = if step_roll < 12 {
                1
            } else if step_roll < 18 {
                2
            } else {
                3
            };
            let mut raw = parent.to_vec();
            let mut first_op = MutOp::MoveSwap;
            let mut first_slot = 0usize;
            let mut applied = 0usize;
            for _ in 0..num_steps {
                let roll = rng.below(20);
                let op = if roll < 8 {
                    MutOp::MoveSwap
                } else if roll < 12 {
                    MutOp::ItemSwap
                } else if roll < 17 {
                    MutOp::SpeciesSwap
                } else {
                    MutOp::LevelTweak
                };
                let slot = if op == MutOp::SpeciesSwap {
                    if constraint.anchor_slots >= 6 {
                        continue;
                    }
                    constraint.anchor_slots + rng.below(6 - constraint.anchor_slots)
                } else {
                    rng.below(6)
                };
                let ok = match op {
                    MutOp::MoveSwap => self.mut_move_swap(dex, &mut raw, slot, rng),
                    MutOp::ItemSwap => self.mut_item_swap(dex, &mut raw, slot, rng),
                    MutOp::LevelTweak => self.mut_level_tweak(&mut raw, slot, rng),
                    MutOp::SpeciesSwap => self.mut_species_swap(dex, &mut raw, slot, rng),
                    MutOp::DvTweak | MutOp::EvTweak | MutOp::MonReplace => unreachable!(),
                };
                if ok.is_some() {
                    if applied == 0 {
                        first_op = op;
                        first_slot = slot;
                    }
                    applied += 1;
                }
            }
            if applied == 0 {
                continue;
            }
            // Preserve anchored species slots during dominance promotion
            let saved_anchors: Vec<String> = (0..constraint.anchor_slots.min(6))
                .map(|i| species_id(&parent[i]))
                .collect();
            self.optimize_team_dominance(dex, &mut raw, rng);
            for (i, orig_sid) in saved_anchors.iter().enumerate() {
                if species_id(&raw[i]) != *orig_sid {
                    let orig_name = dex.species.get(dex.species.id(orig_sid).unwrap()).name.clone();
                    raw[i]["species"] = json!(orig_name.clone());
                    raw[i]["name"] = json!(orig_name);
                }
            }
            if !constraint.allows_team(&raw, Some(parent)) {
                continue;
            }
            let Some(team) = self.canonize(dex, &raw) else { continue };
            if team_key(&team) == team_key(parent) {
                continue;
            }
            return Some(Proposal {
                op: first_op,
                slot: first_slot,
                team,
            });
        }
        None
    }

    /// Generate a random non-dominated legal team whose first `anchors.len()`
    /// slots use the specified species ids (with slot 0 given a Lv55 ace slot
    /// when legal) and that satisfies `constraint`.
    pub fn random_team_constrained(
        &self,
        dex: &Dex,
        rng: &mut SplitMix64,
        anchors: &[String],
        constraint: &NicheConstraint,
        tries: usize,
    ) -> Option<Vec<Value>> {
        for _ in 0..tries {
            let mut sids: Vec<String> = Vec::new();
            for a in anchors {
                let id = toid(a);
                if self.ls.species(&id).is_none() || dex.species.id(&id).is_none() {
                    return None;
                }
                if !sids.contains(&id) {
                    sids.push(id);
                }
            }
            for _ in 0..80 {
                if sids.len() == 6 {
                    break;
                }
                let s = self.species_ids[rng.below(self.species_ids.len())].clone();
                if sids.contains(&s) || constraint.forbidden_species.contains(&s) {
                    continue;
                }
                if let Some((staples, max_c)) = &constraint.staple_cap {
                    let cur_c = sids.iter().filter(|x| staples.contains(*x)).count();
                    if staples.contains(&s) && cur_c >= *max_c {
                        continue;
                    }
                }
                sids.push(s);
            }
            if sids.len() != 6 {
                continue;
            }
            let levels = self.sample_pareto_level_profile(rng);
            // Check floor compatibility; if a non-anchor species has floor > assigned level, swap with a higher slot
            let mut assigned_levels = levels;
            let mut floor_ok = true;
            for i in 0..6 {
                let fl = self.ls.species(&sids[i]).map_or(50, |l| l.min_level.max(50));
                if fl > assigned_levels[i] {
                    if let Some(donor) = (anchors.len()..6)
                        .chain(0..anchors.len())
                        .find(|&j| assigned_levels[j] >= fl && self.ls.species(&sids[j]).map_or(50, |l| l.min_level.max(50)) <= assigned_levels[i])
                    {
                        assigned_levels.swap(i, donor);
                    } else {
                        floor_ok = false;
                        break;
                    }
                }
            }
            if !floor_ok {
                continue;
            }
            let mut raw: Vec<Value> = Vec::new();
            let mut held: Vec<String> = Vec::new();
            let mut ok = true;
            for i in 0..6 {
                let Some(mut set) = self.derive_set(dex, &sids[i], assigned_levels[i], &held, rng) else {
                    ok = false;
                    break;
                };
                set["level"] = json!(assigned_levels[i]);
                let item = toid(set["item"].as_str().unwrap_or(""));
                if !item.is_empty() {
                    held.push(item);
                }
                raw.push(set);
            }
            if !ok {
                continue;
            }
            self.optimize_team_dominance(dex, &mut raw, rng);
            // Restore anchor species if promoted
            for (i, a) in anchors.iter().enumerate() {
                let aid = toid(a);
                let aname = dex.species.get(dex.species.id(&aid).unwrap()).name.clone();
                raw[i]["species"] = json!(aname.clone());
                raw[i]["name"] = json!(aname);
            }
            if !constraint.allows_team(&raw, None) {
                continue;
            }
            if let Some(team) = self.canonize(dex, &raw) {
                return Some(team);
            }
        }
        None
    }

    // -------------------------------------- non-dominated move & item lenses

    /// Non-dominated legal base moves for `sid` at `level` given `cur_moves`
    /// (which unlocks pre-adapted combo moves like `sleeptalk` when `rest` is
    /// present, `dreameater`/`nightmare` when a sleep move is present, and
    /// `defensecurl` when `rollout` is present).
    pub fn non_dominated_moves(
        &self,
        dex: &Dex,
        sid: &str,
        level: i64,
        cur_moves: &[String],
    ) -> Vec<String> {
        let Some(l) = self.ls.species(sid) else {
            return Vec::new();
        };
        let legal: Vec<String> = l
            .moves
            .iter()
            .filter(|m| {
                dex.moves.id(m).is_some()
                    && l.move_min_level.get(m.as_str()).is_none_or(|&mn| level >= mn)
            })
            .cloned()
            .collect();
        let legal_set: BTreeSet<&str> = legal.iter().map(String::as_str).collect();
        let cur_bases: BTreeSet<String> = cur_moves.iter().map(|m| base_id(m)).collect();

        let has_rest = cur_bases.contains("rest");
        let has_sleep = cur_bases.iter().any(|m| {
            matches!(
                m.as_str(),
                "spore" | "sleeppowder" | "lovelykiss" | "hypnosis" | "sing"
            )
        });
        let has_rollout = cur_bases.contains("rollout");
        let has_inaccurate = cur_bases.iter().any(|m| {
            matches!(m.as_str(), "fissure" | "guillotine" | "horndrill")
                || dex
                    .moves
                    .id(m)
                    .and_then(|id| dex.moves.get(id).accuracy.as_i64())
                    .is_some_and(|acc| acc < 100)
        });

        let mut out = Vec::new();
        for a in &legal {
            let a_str = a.as_str();
            // 1. Unconditional zero-effect / non-competitive moves in trainer battles
            if matches!(a_str, "splash" | "teleport" | "falseswipe" | "payday") {
                continue;
            }
            // 2. Frustration is identical to Return at min happiness and 0 BP at canonical happiness=255
            if a_str == "frustration" && legal_set.contains("return") {
                continue;
            }
            // 3. Pre-adaptation gating (unlocks only when prerequisite move is carried)
            if matches!(a_str, "sleeptalk" | "snore") && !has_rest {
                continue;
            }
            if matches!(a_str, "dreameater" | "nightmare") && !has_sleep {
                continue;
            }
            if matches!(a_str, "lockon" | "mindreader") && !has_inaccurate {
                continue;
            }
            if a_str == "defensecurl"
                && !has_rollout
                && (legal_set.contains("barrier")
                    || legal_set.contains("acidarmor")
                    || legal_set.contains("withdraw")
                    || legal_set.contains("curse")
                    || legal_set.contains("reflect"))
            {
                continue;
            }
            // 4. Status move strict dominance & exact aliases
            if STATUS_DOMINANCE
                .iter()
                .chain(STATUS_DOMINANCE_EXTRA.iter())
                .any(|&(lo, hi)| a_str == lo && legal_set.contains(hi))
            {
                continue;
            }
            if (a_str == "synthesis" || a_str == "moonlight") && legal_set.contains("morningsun") {
                continue;
            }
            if a_str == "synthesis" && legal_set.contains("moonlight") {
                continue;
            }
            // 5. Self-Destruct (200 BP) strictly dominated by Explosion (250 BP)
            if a_str == "selfdestruct" && legal_set.contains("explosion") {
                continue;
            }

            // 6. Damaging move dominance within the species's legal learnset
            //    (ignoring PP differences as long as the dominator has >= 5 PP,
            //    and allowing a +20 BP jump to override minor <= 30% secondaries).
            let Some(ma_id) = dex.moves.id(a_str) else { continue };
            let ma = dex.moves.get(ma_id);
            let bp_a = if a_str == "return" { 102 } else { ma.base_power };
            if ma.category != "Status"
                && bp_a > 0
                && !UNIQUE_DAMAGE_MOVES.contains(&a_str)
                && !FIXED_OR_PRIORITY_MOVES.contains(&a_str)
            {
                // Ultra-low-BP starter moves (BP <= 35, e.g. Poison Sting 15, Lick 20,
                // Leech Life 20, Smog 20, Absorb 20, Peck 35, Vine Whip 35, Constrict 10)
                // deal less than neutral Return (102) even with STAB + 2x SE (3x <= 105).
                if bp_a <= 35 && ma.multihit.is_none() && legal_set.contains("return") {
                    continue;
                }
                let dominated = legal.iter().any(|b| {
                    let b_str = b.as_str();
                    if a_str == b_str {
                        return false;
                    }
                    let Some(mb_id) = dex.moves.id(b_str) else { return false };
                    let mb = dex.moves.get(mb_id);
                    let bp_b = if b_str == "return" { 102 } else { mb.base_power };
                    if mb.category == "Status" || bp_b <= bp_a || mb.move_type != ma.move_type {
                        return false;
                    }
                    if mb.priority < ma.priority || mb.crit_ratio < ma.crit_ratio {
                        return false;
                    }
                    // Treat never-miss 60 BP moves (Swift, Feint Attack) as 100% accuracy
                    // when a same-type move is at least +20 BP stronger.
                    let acc_a = if bp_b >= bp_a + 20 {
                        ma.accuracy.as_i64().unwrap_or(100)
                    } else {
                        ma.accuracy.as_i64().unwrap_or(101)
                    };
                    let acc_b = mb.accuracy.as_i64().unwrap_or(101);
                    if acc_b < acc_a {
                        return false;
                    }
                    // Dominator B cannot introduce recoil (unless A has the same recoil), self-KO, two-turn charge, recharge, or lock-in
                    if (mb.recoil.is_some() && mb.recoil != ma.recoil)
                        || mb.extra.contains_key("selfdestruct")
                        || mb.extra.contains_key("self")
                        || mb.volatile_status.is_some()
                        || mb.extra.contains_key("ohko")
                        || mb.extra.contains_key("damage")
                        || matches!(
                            b_str,
                            "solarbeam"
                                | "skullbash"
                                | "skyattack"
                                | "razorwind"
                                | "fly"
                                | "dig"
                                | "hyperbeam"
                                | "outrage"
                                | "thrash"
                                | "petaldance"
                                | "futuresight"
                                | "explosion"
                                | "selfdestruct"
                        )
                    {
                        return false;
                    }
                    if ma.drain.is_some() && mb.drain != ma.drain && bp_b < bp_a + 25 {
                        return false;
                    }
                    if ma.multihit.is_some() && bp_a >= 25 {
                        return false;
                    }
                    // Secondary effect check: if B is >= +20 BP stronger and A's secondary
                    // is at most a minor (<= 30%) proc, B dominates A; otherwise B must
                    // match A's secondary with >= chance.
                    if let Some(sec_a) = &ma.secondary {
                        let chance_a = sec_a["chance"].as_i64().unwrap_or(100);
                        if !(bp_b >= bp_a + 20 && chance_a <= 30) {
                            let Some(sec_b) = &mb.secondary else { return false };
                            if sec_a["status"] != sec_b["status"]
                                || sec_a["volatileStatus"] != sec_b["volatileStatus"]
                                || sec_a["boosts"] != sec_b["boosts"]
                                || sec_b["chance"].as_i64().unwrap_or(100) < chance_a
                            {
                                return false;
                            }
                        }
                    }
                    mb.pp >= 5
                });
                if dominated {
                    continue;
                }
            }
            out.push(a.clone());
        }
        out
    }

    /// Non-dominated Hidden Power types for `sid` at `level`: excludes types
    /// where the species already learns a no-drawback 100%-accurate move of
    /// that type with BP > 70 and PP >= 5.
    fn non_dominated_hp_types(&self, dex: &Dex, sid: &str, level: i64) -> Vec<String> {
        let Some(l) = self.ls.species(sid) else {
            return Vec::new();
        };
        let legal: Vec<&String> = l
            .moves
            .iter()
            .filter(|m| {
                dex.moves.id(m).is_some()
                    && l.move_min_level.get(m.as_str()).is_none_or(|&mn| level >= mn)
            })
            .collect();

        let mut out = Vec::new();
        for &t in &HP_TYPES {
            let hp_id = format!("hiddenpower{t}");
            let Some(mid) = dex.moves.id(&hp_id) else { continue };
            let hp_type_name = &dex.moves.get(mid).move_type;
            let dominated = legal.iter().any(|&m| {
                let m_str = m.as_str();
                let Some(id) = dex.moves.id(m_str) else { return false };
                let md = dex.moves.get(id);
                let bp = if m_str == "return" { 102 } else { md.base_power };
                md.category != "Status"
                    && &md.move_type == hp_type_name
                    && bp > 70
                    && md.accuracy.as_i64().unwrap_or(101) >= 100
                    && md.pp >= 5
                    && md.recoil.is_none()
                    && !md.extra.contains_key("selfdestruct")
                    && !matches!(
                        m_str,
                        "solarbeam"
                            | "skullbash"
                            | "skyattack"
                            | "razorwind"
                            | "fly"
                            | "dig"
                            | "hyperbeam"
                            | "outrage"
                            | "thrash"
                            | "petaldance"
                            | "futuresight"
                            | "explosion"
                            | "selfdestruct"
                    )
            });
            if !dominated {
                out.push(hp_id);
            }
        }
        if out.is_empty() {
            HP_TYPES.iter().map(|t| format!("hiddenpower{t}")).collect()
        } else {
            out
        }
    }

    /// Non-dominated legal items for `raw[slot]` given teammates' held items.
    /// When `include_current` is true, `raw[slot]`'s current item is not
    /// filtered out for identity.
    fn non_dominated_items(
        &self,
        dex: &Dex,
        raw: &[Value],
        slot: usize,
        include_current: bool,
    ) -> Vec<String> {
        let sid = species_id(&raw[slot]);
        let cur_item = toid(raw[slot]["item"].as_str().unwrap_or(""));
        let held = held_items(raw, slot);
        let moves = move_ids(&raw[slot]);
        let has_rest = moves.iter().any(|m| m == "rest");

        // Damaging move types carried by this Pokemon
        let mut dmg_types: BTreeSet<String> = BTreeSet::new();
        for m in &moves {
            if let Some(mid) = dex.moves.id(m) {
                let md = dex.moves.get(mid);
                if md.category != "Status" && (md.base_power > 0 || m == "return") {
                    dmg_types.insert(md.move_type.clone());
                }
            }
        }

        let gold_free = !held.contains(&"goldberry".to_string());
        let juice_free = !held.contains(&"berryjuice".to_string());
        let miracle_free = !held.contains(&"miracleberry".to_string());
        let pinkbow_free = !held.contains(&"pinkbow".to_string());
        let thickclub_free = !held.contains(&"thickclub".to_string());

        let mut out = Vec::new();
        for item_name in &self.all_items {
            let id = toid(item_name);
            if !include_current && id == cur_item {
                continue;
            }
            if held.contains(&id) {
                continue;
            }
            if ZERO_EFFECT_ITEMS.contains(&id.as_str()) || id == "mail" {
                continue;
            }
            // Species-specific items
            if let Some((_, allowed)) = SPECIES_ITEMS.iter().find(|&&(it, _)| it == id) {
                if !allowed.contains(&sid.as_str()) {
                    continue;
                }
            }
            // Type-boosting items require at least one damaging move of that type
            if let Some(&(_, req_type)) = TYPE_BOOST_ITEMS.iter().find(|&&(it, _)| it == id) {
                if !dmg_types.contains(req_type) {
                    continue;
                }
                if id == "polkadotbow" && pinkbow_free {
                    continue;
                }
                if id == "softsand"
                    && matches!(sid.as_str(), "cubone" | "marowak")
                    && thickclub_free
                {
                    continue;
                }
            }
            // Dominated HP berries: Berry (10 HP) < Berry Juice (20 HP) < Gold Berry (30 HP)
            if matches!(id.as_str(), "berry" | "berryjuice") && gold_free {
                continue;
            }
            if id == "berry" && juice_free {
                continue;
            }
            // Dominated status berries
            if id == "mintberry" && !has_rest && miracle_free {
                continue;
            }
            if matches!(
                id.as_str(),
                "przcureberry" | "psncureberry" | "burntberry" | "iceberry" | "bitterberry"
            ) && miracle_free
            {
                continue;
            }
            out.push(item_name.clone());
        }
        out
    }

    /// Replace any orphaned dependent moves (`sleeptalk`/`snore` without
    /// `rest`, `dreameater`/`nightmare` without a sleep move, `defensecurl`
    /// without `rollout`) or dominated moves on `raw[slot]`.
    fn repair_orphaned_moves(
        &self,
        dex: &Dex,
        raw: &mut [Value],
        slot: usize,
        rng: &mut SplitMix64,
    ) {
        let sid = species_id(&raw[slot]);
        let level = set_level(&raw[slot]);
        let cur = move_ids(&raw[slot]);
        let has_rest = cur.iter().any(|m| m == "rest");
        let has_sleep = cur.iter().any(|m| {
            matches!(
                m.as_str(),
                "spore" | "sleeppowder" | "lovelykiss" | "hypnosis" | "sing"
            )
        });
        let non_dom_init = self.non_dominated_moves(dex, &sid, level, &cur);

        let mut changed = false;
        let mut new_moves = cur.clone();
        for i in 0..new_moves.len() {
            let m = new_moves[i].as_str();
            let b = base_id(m);
            let orphaned = (matches!(m, "sleeptalk" | "snore") && !has_rest)
                || (matches!(m, "dreameater" | "nightmare") && !has_sleep)
                || !non_dom_init.contains(&b);
            if orphaned {
                let cands = self.non_dominated_moves(dex, &sid, level, &new_moves);
                let cur_bases: BTreeSet<String> = new_moves.iter().map(|x| base_id(x)).collect();
                let avail: Vec<&String> =
                    cands.iter().filter(|c| !cur_bases.contains(*c)).collect();
                if let Some(&pick) = avail.get(rng.below(avail.len().max(1))) {
                    let chosen = if pick == "hiddenpower" {
                        let hp = self.non_dominated_hp_types(dex, &sid, level);
                        hp[rng.below(hp.len())].clone()
                    } else {
                        pick.clone()
                    };
                    new_moves[i] = chosen;
                    changed = true;
                }
            }
        }
        while new_moves.len() < 4 {
            let cands = self.non_dominated_moves(dex, &sid, level, &new_moves);
            let cur_bases: BTreeSet<String> = new_moves.iter().map(|x| base_id(x)).collect();
            let avail: Vec<&String> =
                cands.iter().filter(|c| !cur_bases.contains(*c)).collect();
            if let Some(&pick) = avail.get(rng.below(avail.len().max(1))) {
                let chosen = if pick == "hiddenpower" {
                    let hp = self.non_dominated_hp_types(dex, &sid, level);
                    hp[rng.below(hp.len())].clone()
                } else {
                    pick.clone()
                };
                new_moves.push(chosen);
                changed = true;
            } else {
                break;
            }
        }
        if changed {
            raw[slot]["moves"] =
                json!(new_moves.iter().map(|m| move_name(dex, m)).collect::<Vec<_>>());
        }
    }

    // ------------------------------------------------------ op internals

    fn mut_move_swap(
        &self,
        dex: &Dex,
        raw: &mut [Value],
        slot: usize,
        rng: &mut SplitMix64,
    ) -> Option<()> {
        let sid = species_id(&raw[slot]);
        let level = set_level(&raw[slot]);
        let cur = move_ids(&raw[slot]);
        let cur_bases: Vec<String> = cur.iter().map(|m| base_id(m)).collect();
        let non_dom = self.non_dominated_moves(dex, &sid, level, &cur);
        let cands: Vec<&String> = non_dom.iter().filter(|m| !cur_bases.contains(*m)).collect();
        if cands.is_empty() {
            return None;
        }
        let pick = cands[rng.below(cands.len())].clone();
        let display = if pick == "hiddenpower" {
            let typed = self.non_dominated_hp_types(dex, &sid, level);
            if typed.is_empty() {
                return None;
            }
            move_name(dex, &typed[rng.below(typed.len())])
        } else {
            move_name(dex, &pick)
        };
        let moves = raw[slot]["moves"].as_array_mut()?;
        if moves.is_empty() {
            return None;
        }
        if moves.len() < 4 && rng.below(2) == 0 {
            moves.push(json!(display));
        } else {
            // If `pick == "hiddenpower"`, replace any existing hiddenpower slot first
            let hp_slot = cur.iter().position(|m| m.starts_with("hiddenpower"));
            let p = if pick == "hiddenpower" {
                hp_slot.unwrap_or_else(|| rng.below(moves.len()))
            } else {
                rng.below(moves.len())
            };
            moves[p] = json!(display);
        }
        Some(())
    }

    fn mut_item_swap(
        &self,
        dex: &Dex,
        raw: &mut [Value],
        slot: usize,
        rng: &mut SplitMix64,
    ) -> Option<()> {
        // With 35% probability, try swapping items between `slot` and a teammate
        // so key items (Leftovers, Miracle Berry, etc.) can move across slots
        // without dropping them first.
        if rng.below(20) < 7 {
            let mut mates: Vec<usize> = (0..raw.len()).filter(|&i| i != slot).collect();
            let start = rng.below(mates.len());
            mates.rotate_left(start);
            let item_a = raw[slot]["item"].as_str().unwrap_or("").to_string();
            for other in mates {
                let item_b = raw[other]["item"].as_str().unwrap_or("").to_string();
                if toid(&item_a) == toid(&item_b) {
                    continue;
                }
                raw[slot]["item"] = json!("");
                raw[other]["item"] = json!("");
                let ok_a = item_b.is_empty()
                    || self
                        .non_dominated_items(dex, raw, slot, true)
                        .iter()
                        .any(|it| toid(it) == toid(&item_b));
                let ok_b = item_a.is_empty()
                    || self
                        .non_dominated_items(dex, raw, other, true)
                        .iter()
                        .any(|it| toid(it) == toid(&item_a));
                if ok_a && ok_b {
                    raw[slot]["item"] = json!(item_b);
                    raw[other]["item"] = json!(item_a);
                    return Some(());
                }
                raw[slot]["item"] = json!(item_a.clone());
                raw[other]["item"] = json!(item_b);
            }
        }

        let cands = self.non_dominated_items(dex, raw, slot, false);
        if cands.is_empty() {
            return None;
        }
        raw[slot]["item"] = json!(cands[rng.below(cands.len())]);
        Some(())
    }

    /// Two-stage Macro -> Micro draw over slack-free Pareto-optimal 6-level profiles:
    /// 1. Macro draw: choose the Ace-count bucket `k = count(level == 55) in 0..=4` uniformly.
    /// 2. Micro draw: choose a Pareto-optimal profile within bucket `k` uniformly.
    pub fn sample_pareto_level_profile(&self, rng: &mut SplitMix64) -> [i64; 6] {
        let bucket = &self.pareto_by_aces[rng.below(self.pareto_by_aces.len())];
        bucket[rng.below(bucket.len())]
    }

    fn mut_level_tweak(
        &self,
        raw: &mut [Value],
        slot: usize,
        rng: &mut SplitMix64,
    ) -> Option<()> {
        let floors: Vec<i64> = (0..raw.len())
            .map(|i| {
                let sid = species_id(&raw[i]);
                self.ls.species(&sid).map_or(50, |l| l.min_level.max(50))
            })
            .collect();
        let cur_levels: Vec<i64> = (0..raw.len()).map(|i| set_level(&raw[i])).collect();

        // 50% chance: swap levels between `slot` and another slot with a different level
        if rng.below(2) == 0 {
            let mut others: Vec<usize> = (0..raw.len())
                .filter(|&i| {
                    i != slot
                        && cur_levels[i] != cur_levels[slot]
                        && floors[slot] <= cur_levels[i]
                        && floors[i] <= cur_levels[slot]
                })
                .collect();
            if !others.is_empty() {
                let other = others.swap_remove(rng.below(others.len()));
                raw[slot]["level"] = json!(cur_levels[other]);
                raw[other]["level"] = json!(cur_levels[slot]);
                maximize_slack_levels(raw, &floors);
                return Some(());
            }
        }

        // Two-stage Macro -> Micro transition to a Pareto-optimal (slack-free) 6-level profile,
        // preserving existing level rank ordering across teammates (breaking ties randomly).
        let profile = self.sample_pareto_level_profile(rng);
        let mut order: Vec<usize> = (0..raw.len()).collect();
        for i in (1..order.len()).rev() {
            let j = rng.below(i + 1);
            order.swap(i, j);
        }
        order.sort_by(|&a, &b| {
            floors[b]
                .cmp(&floors[a])
                .then(cur_levels[b].cmp(&cur_levels[a]))
        });
        for (rank, &idx) in order.iter().enumerate() {
            if floors[idx] > profile[rank] {
                return None;
            }
            raw[idx]["level"] = json!(profile[rank]);
        }
        let new_levels: Vec<i64> = (0..raw.len()).map(|i| set_level(&raw[i])).collect();
        if new_levels == cur_levels {
            return None;
        }
        Some(())
    }

    fn mut_species_swap(
        &self,
        dex: &Dex,
        raw: &mut [Value],
        slot: usize,
        rng: &mut SplitMix64,
    ) -> Option<()> {
        let cur = species_id(&raw[slot]);
        let teammates: Vec<String> = (0..raw.len())
            .filter(|&i| i != slot)
            .map(|i| species_id(&raw[i]))
            .collect();
        let sid = &self.species_ids[rng.below(self.species_ids.len())];
        if *sid == cur || teammates.contains(sid) {
            return None;
        }
        let level_hint = set_level(&raw[slot]);
        let min_lv = self.ls.species(sid)?.min_level.max(50);
        if min_lv > level_hint {
            // Try swapping level with a teammate that has >= min_lv
            let donor = (0..raw.len()).find(|&i| {
                i != slot
                    && set_level(&raw[i]) >= min_lv
                    && self
                        .ls
                        .species(&species_id(&raw[i]))
                        .is_none_or(|l| l.min_level.max(50) <= level_hint)
            })?;
            let donor_lv = set_level(&raw[donor]);
            raw[donor]["level"] = json!(level_hint);
            raw[slot]["level"] = json!(donor_lv);
        }
        let level = set_level(&raw[slot]);
        let held = held_items(raw, slot);
        raw[slot] = self.derive_set(dex, sid, level, &held, rng)?;
        Some(())
    }

    /// Derive a validator-clean set for `sid` *ab initio* from its
    /// non-dominated learnset and non-dominated items (never copying
    /// `meta-pool.json` templates).
    fn derive_set(
        &self,
        dex: &Dex,
        sid: &str,
        level_hint: i64,
        held: &[String],
        rng: &mut SplitMix64,
    ) -> Option<Value> {
        let l = self.ls.species(sid)?;
        let level = level_hint.max(l.min_level).clamp(50, 55);
        let moves = self.sample_non_dominated_moveset(dex, sid, level, rng)?;
        let species_name = dex.species.get(dex.species.id(sid)?).name.clone();

        let mut set = json!({
            "name": species_name,
            "species": species_name,
            "item": "",
            "ability": "No Ability",
            "moves": moves,
            "nature": "Serious",
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "level": level,
        });

        // Pick a non-dominated item for this newly constructed set
        let mut dummy_team = vec![set.clone()];
        for h in held {
            dummy_team.push(json!({"species": "Snorlax", "item": h, "moves": ["Return"], "level": 50}));
        }
        let item_cands = self.non_dominated_items(dex, &dummy_team, 0, false);
        if !item_cands.is_empty() {
            set["item"] = json!(item_cands[rng.below(item_cands.len())]);
        }
        optimize_set_dvs_evs(dex, &mut set);
        Some(set)
    }

    /// Sample up to 4 distinct moves from the species's non-dominated move pool
    /// at `level`, sequentially updating `picked` so pre-adaptation moves
    /// (like `Rest` -> `Sleep Talk` or `Hypnosis` -> `Dream Eater`) unlock
    /// naturally during construction. Ensures at least one non-self-KO
    /// damaging move when available.
    fn sample_non_dominated_moveset(
        &self,
        dex: &Dex,
        sid: &str,
        level: i64,
        rng: &mut SplitMix64,
    ) -> Option<Vec<String>> {
        let mut picked: Vec<String> = Vec::new();
        let mut picked_bases: BTreeSet<String> = BTreeSet::new();

        // Step 1: ensure at least 1 damaging move (not Explosion/Self-Destruct) if the species has any
        let initial_cands = self.non_dominated_moves(dex, sid, level, &picked);
        let dmg_cands: Vec<&String> = initial_cands
            .iter()
            .filter(|m| {
                let m_str = m.as_str();
                if m_str == "hiddenpower" {
                    return true;
                }
                let Some(mid) = dex.moves.id(m_str) else { return false };
                let md = dex.moves.get(mid);
                md.category != "Status"
                    && (md.base_power > 0 || m_str == "return")
                    && !md.extra.contains_key("selfdestruct")
                    && !matches!(m_str, "explosion" | "selfdestruct")
            })
            .collect();

        if !dmg_cands.is_empty() {
            let base = dmg_cands[rng.below(dmg_cands.len())].clone();
            let chosen = if base == "hiddenpower" {
                let hp = self.non_dominated_hp_types(dex, sid, level);
                hp[rng.below(hp.len())].clone()
            } else {
                base.clone()
            };
            picked_bases.insert(base);
            picked.push(chosen);
        }

        // Step 2: draw remaining slots sequentially from non_dominated_moves (pre-adaptation aware)
        for _ in picked.len()..4 {
            let cands = self.non_dominated_moves(dex, sid, level, &picked);
            let avail: Vec<&String> =
                cands.iter().filter(|m| !picked_bases.contains(*m)).collect();
            if avail.is_empty() {
                break;
            }
            let base = avail[rng.below(avail.len())].clone();
            let chosen = if base == "hiddenpower" {
                let hp = self.non_dominated_hp_types(dex, sid, level);
                hp[rng.below(hp.len())].clone()
            } else {
                base.clone()
            };
            picked_bases.insert(base);
            picked.push(chosen);
        }

        if picked.is_empty() {
            return None;
        }
        Some(picked.iter().map(|m| move_name(dex, m)).collect())
    }

    // --------------------------------------------------- research helpers

    /// Deliberate weakening for smoke validation (proves the hill-climb has
    /// signal, not strength): strip every item, then downgrade the strongest
    /// damaging move of `downgrades` random slots to the weakest legal
    /// damaging move — all through the same canonize/validate pipeline as
    /// the mutation operators.
    pub fn weaken(
        &self,
        dex: &Dex,
        team: &[Value],
        rng: &mut SplitMix64,
        downgrades: usize,
    ) -> Option<Vec<Value>> {
        let mut raw = team.to_vec();
        for set in raw.iter_mut() {
            set["item"] = json!("");
        }
        for _ in 0..downgrades {
            let slot = rng.below(6);
            let sid = species_id(&raw[slot]);
            let Some(l) = self.ls.species(&sid) else { continue };
            let level = set_level(&raw[slot]);
            let sp_types = &dex.species.get(dex.species.id(&sid)?).types;
            let score = |m: &str| -> Option<f64> {
                let md = dex.moves.get(dex.moves.id(m).or_else(|| dex.moves.id(&base_id(m)))?);
                if md.category == "Status" || md.base_power == 0 {
                    return None;
                }
                let stab = if sp_types.contains(&md.move_type) { 1.5 } else { 1.0 };
                Some(md.base_power as f64 * stab)
            };
            let cur = move_ids(&raw[slot]);
            let cur_bases: Vec<String> = cur.iter().map(|m| base_id(m)).collect();
            let Some((target, _)) = cur
                .iter()
                .enumerate()
                .filter_map(|(i, m)| score(m).map(|s| (i, s)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
            else {
                continue;
            };
            let mut weak: Vec<(f64, &String)> = l
                .moves
                .iter()
                .filter(|m| {
                    !cur_bases.contains(m)
                        && dex.moves.id(m).is_some()
                        && l.move_min_level.get(m.as_str()).is_none_or(|&mn| level >= mn)
                })
                .filter_map(|m| score(m).map(|s| (s, m)))
                .collect();
            weak.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(b.1)));
            if let Some((_, m)) = weak.first() {
                raw[slot]["moves"][target] = json!(move_name(dex, m));
            }
        }
        self.canonize(dex, &raw)
    }

    /// One attempt at a random legal team: 6 distinct non-dominated species
    /// on a random Pareto-optimal level profile (or 55/55/50/50/50/50), sets
    /// derived ab initio from non-dominated learnsets and items.
    pub fn random_team(&self, dex: &Dex, rng: &mut SplitMix64) -> Option<Vec<Value>> {
        let mut sids: Vec<String> = Vec::new();
        for _ in 0..40 {
            if sids.len() == 6 {
                break;
            }
            let s = self.species_ids[rng.below(self.species_ids.len())].clone();
            if !sids.contains(&s) {
                sids.push(s);
            }
        }
        if sids.len() != 6 {
            return None;
        }
        sids.sort_by(|a, b| {
            let fa = self.ls.species(a).map_or(50, |l| l.min_level);
            let fb = self.ls.species(b).map_or(50, |l| l.min_level);
            fb.cmp(&fa).then(a.cmp(b))
        });
        let levels = self.sample_pareto_level_profile(rng);
        let mut raw: Vec<Value> = Vec::new();
        let mut held: Vec<String> = Vec::new();
        for (i, sid) in sids.iter().enumerate() {
            if self.ls.species(sid).map_or(50, |l| l.min_level) > levels[i] {
                return None;
            }
            let mut set = self.derive_set(dex, sid, levels[i], &held, rng)?;
            set["level"] = json!(levels[i]);
            let item = toid(set["item"].as_str().unwrap_or(""));
            if !item.is_empty() {
                held.push(item);
            }
            raw.push(set);
        }
        self.optimize_team_dominance(dex, &mut raw, rng);
        self.canonize(dex, &raw)
    }

    pub fn random_team_valid(
        &self,
        dex: &Dex,
        rng: &mut SplitMix64,
        tries: usize,
    ) -> Option<Vec<Value>> {
        (0..tries).find_map(|_| self.random_team(dex, rng))
    }
}

// -------------------------------------------- Pareto level & DV/EV helpers

/// Enumerate all 65 slack-free (Pareto-optimal) sorted 6-level profiles
/// `[l0 >= l1 >= l2 >= l3 >= l4 >= l5]` in `50..=55` satisfying the NC2000
/// validator rules (`l3+l4+l5 <= 155` and `l0+l4+l5 <= 155`) such that no
/// slot's level can be incremented without shrinking the set of legal 3-mon
/// selections.
fn compute_pareto_levels() -> Vec<[i64; 6]> {
    let mut valid = Vec::new();
    for l0 in (50..=55).rev() {
        for l1 in (50..=l0).rev() {
            for l2 in (50..=l1).rev() {
                for l3 in (50..=l2).rev() {
                    for l4 in (50..=l3).rev() {
                        for l5 in (50..=l4).rev() {
                            if l3 + l4 + l5 <= 155 && l0 + l4 + l5 <= 155 {
                                valid.push([l0, l1, l2, l3, l4, l5]);
                            }
                        }
                    }
                }
            }
        }
    }
    let mask_of = |lv: &[i64; 6]| -> u32 {
        let mut mask = 0u32;
        let mut bit = 0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                for k in (j + 1)..6 {
                    if lv[i] + lv[j] + lv[k] <= 155 {
                        mask |= 1 << bit;
                    }
                    bit += 1;
                }
            }
        }
        mask
    };
    let mut maximal = Vec::new();
    for &lv in &valid {
        let m = mask_of(&lv);
        let dominated = (0..6).any(|idx| {
            if lv[idx] >= 55 {
                return false;
            }
            let mut lv2 = lv;
            lv2[idx] += 1;
            lv2.sort_unstable_by(|a, b| b.cmp(a));
            lv2[3] + lv2[4] + lv2[5] <= 155
                && lv2[0] + lv2[4] + lv2[5] <= 155
                && mask_of(&lv2) == m
        });
        if !dominated {
            maximal.push(lv);
        }
    }
    maximal
}

/// Greedily eliminate any free level slack on `raw` without changing the set
/// of legal 3-Pokemon combinations.
fn maximize_slack_levels(raw: &mut [Value], floors: &[i64]) {
    let mask_of = |lv: &[i64; 6]| -> u32 {
        let mut mask = 0u32;
        let mut bit = 0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                for k in (j + 1)..6 {
                    if lv[i] + lv[j] + lv[k] <= 155 {
                        mask |= 1 << bit;
                    }
                    bit += 1;
                }
            }
        }
        mask
    };
    let mut lv = [50i64; 6];
    for i in 0..6 {
        lv[i] = set_level(&raw[i]).max(floors[i]);
    }
    let orig_mask = mask_of(&lv);
    loop {
        let mut improved = false;
        for i in 0..6 {
            if lv[i] < 55 {
                let mut cand = lv;
                cand[i] += 1;
                let mut sorted = cand;
                sorted.sort_unstable();
                if sorted[0] + sorted[1] + sorted[2] <= 155
                    && sorted[5] + sorted[0] + sorted[1] <= 155
                    && mask_of(&cand) == orig_mask
                {
                    lv = cand;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    for i in 0..6 {
        raw[i]["level"] = json!(lv[i]);
    }
}

/// Deterministic DV/EV optimization for a single set:
/// - Always sets HP/Def/SpA/SpD/Spe EVs to 255.
/// - If the set has no Physical damaging moves, no Hidden Power, and no
///   move-calling moves (`metronome`, `mimic`, `mirrormove`, `transform`),
///   sets `Atk EV = 0` and `ivs = {hp: 31, atk: 3, def: 31, spa: 31, spd: 31, spe: 31}`
///   (`Atk DV = 1` keeps `HP DV = 15`, minimizes confusion self-hit, and selects
///   `F` gender on dual-gender species).
/// - Otherwise sets `Atk EV = 255` and removes explicit `ivs` so `canonicalize_team`
///   assigns max DVs (or the exact Hidden Power DVs for typed Hidden Power).
fn optimize_set_dvs_evs(dex: &Dex, set: &mut Value) {
    let moves = move_ids(set);
    let has_hp = moves.iter().any(|m| m.starts_with("hiddenpower"));
    let has_caller = moves.iter().any(|m| {
        matches!(
            m.as_str(),
            "metronome" | "mimic" | "mirrormove" | "transform"
        )
    });
    let has_physical = moves.iter().any(|m| {
        dex.moves
            .id(m)
            .is_some_and(|id| dex.moves.get(id).category == "Physical")
    });

    if !has_hp && !has_caller && !has_physical && species_id(set) != "unown" {
        set["evs"] = json!({"hp": 255, "atk": 0, "def": 255, "spa": 255, "spd": 255, "spe": 255});
        set["ivs"] = json!({"hp": 31, "atk": 3, "def": 31, "spa": 31, "spd": 31, "spe": 31});
        if let Some(obj) = set.as_object_mut() {
            obj.remove("gender");
            obj.remove("shiny");
        }
    } else {
        set["evs"] =
            json!({"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255});
        if let Some(obj) = set.as_object_mut() {
            obj.remove("ivs");
            obj.remove("gender");
            obj.remove("shiny");
        }
    }
}

// -------------------------------------------------------- JSON set helpers

fn species_id(set: &Value) -> String {
    toid(set["species"].as_str().unwrap_or(""))
}

fn set_level(set: &Value) -> i64 {
    match set["level"].as_i64() {
        Some(l) if l != 0 => l,
        _ => 55,
    }
}

fn move_ids(set: &Value) -> Vec<String> {
    set["moves"]
        .as_array()
        .map(|a| a.iter().filter_map(|m| m.as_str()).map(toid).collect())
        .unwrap_or_default()
}

/// Typed Hidden Power collapses onto the base id for learnset checks.
fn base_id(move_id: &str) -> String {
    if move_id.starts_with("hiddenpower") && move_id != "hiddenpower" {
        "hiddenpower".into()
    } else {
        move_id.into()
    }
}

fn move_name(dex: &Dex, id: &str) -> String {
    match dex.moves.id(id) {
        Some(mid) => dex.moves.get(mid).name.clone(),
        None => id.to_string(),
    }
}

/// Item ids held by every slot except `slot` (Item Clause universe).
fn held_items(raw: &[Value], slot: usize) -> Vec<String> {
    (0..raw.len())
        .filter(|&i| i != slot)
        .map(|i| toid(raw[i]["item"].as_str().unwrap_or("")))
        .filter(|it| !it.is_empty())
        .collect()
}

/// Stable identity of a canonical team (both sides always come from
/// `canonize`, so key order is consistent).
pub fn team_key(team: &[Value]) -> String {
    serde_json::to_string(team).unwrap()
}

/// Canonical team JSON -> engine sets (extra canonical-only fields like
/// `nature` are ignored by `PokemonSet`).
pub fn to_sets(team: &[Value]) -> Result<Vec<PokemonSet>, String> {
    serde_json::from_value(Value::Array(team.to_vec())).map_err(|e| e.to_string())
}

// ------------------------------------------------------------------ fitness

#[derive(Clone, Debug)]
pub struct EvalCfg {
    /// Seed-paired games per gauntlet member (rounded up to even).
    pub games_per_opponent: u32,
    /// skuct iterations per decision, both sides (preview = live search).
    pub agent_iters: u32,
    pub max_turns: u16,
    pub threads: usize,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Candidate mean score over every game (win 1 / tie 0.5 / loss 0).
    pub score: f64,
    pub per_opponent: Vec<f64>,
    pub games: usize,
}

struct EvalJob {
    opp: usize,
    battle_seed: String,
    cand_seed: u64,
    opp_seed: u64,
    cand_is_p1: bool,
}

/// Candidate fitness vs a gauntlet: seed-paired skuct-vs-skuct duels (both
/// orientations on the same battle seed), full games from team preview.
/// Deterministic for a given `cfg.seed` at any `cfg.threads`.
pub fn gauntlet_eval(
    dex: &Dex,
    candidate: &[PokemonSet],
    gauntlet: &[Vec<PokemonSet>],
    cfg: &EvalCfg,
) -> EvalResult {
    let games = (cfg.games_per_opponent + cfg.games_per_opponent % 2) as usize;
    let mut jobs = Vec::with_capacity(gauntlet.len() * games);
    for opp in 0..gauntlet.len() {
        let mut rng =
            SplitMix64::new(cfg.seed ^ (opp as u64 + 1).wrapping_mul(0x9FB2_1C65_1E98_DF25));
        let mut battle_seed = String::new();
        for g in 0..games {
            if g % 2 == 0 {
                battle_seed = rng.battle_seed();
            }
            jobs.push(EvalJob {
                opp,
                battle_seed: battle_seed.clone(),
                cand_seed: rng.next(),
                opp_seed: rng.next(),
                cand_is_p1: g % 2 == 0,
            });
        }
    }

    let cursor = AtomicUsize::new(0);
    let mut results: Vec<(usize, f64)> = Vec::with_capacity(jobs.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..cfg.threads.max(1) {
            let (jobs, cursor) = (&jobs, &cursor);
            handles.push(scope.spawn(move || {
                let mut out: Vec<(usize, (usize, f64))> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let job = &jobs[i];
                    let skuct = |seed: u64| {
                        RmAgent::new(
                            RmConfig {
                                iterations: cfg.agent_iters,
                                rule: SelRule::Ucb,
                                ..Default::default()
                            },
                            seed,
                        )
                    };
                    let mut cand_agent = skuct(job.cand_seed);
                    let mut opp_agent = skuct(job.opp_seed);
                    let (t1, t2) = if job.cand_is_p1 {
                        (candidate, gauntlet[job.opp].as_slice())
                    } else {
                        (gauntlet[job.opp].as_slice(), candidate)
                    };
                    let mut b = Battle::from_fixture(dex, &job.battle_seed, t1, t2).unwrap();
                    b.set_log_enabled(false);
                    let res = if job.cand_is_p1 {
                        play_game(dex, &mut b, &mut [&mut cand_agent, &mut opp_agent], cfg.max_turns)
                    } else {
                        play_game(dex, &mut b, &mut [&mut opp_agent, &mut cand_agent], cfg.max_turns)
                    }
                    .unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let score = if job.cand_is_p1 { p1_score } else { 1.0 - p1_score };
                    out.push((i, (job.opp, score)));
                }
                out
            }));
        }
        let mut all: Vec<(usize, (usize, f64))> = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all.sort_by_key(|r| r.0);
        results = all.into_iter().map(|(i, (_, s))| (jobs[i].opp, s)).collect();
    });

    // job-ordered summation -> bit-stable means at any thread count
    let mut per: Vec<(f64, usize)> = vec![(0.0, 0); gauntlet.len()];
    let mut total = 0.0;
    for &(opp, s) in &results {
        per[opp].0 += s;
        per[opp].1 += 1;
        total += s;
    }
    EvalResult {
        score: total / results.len() as f64,
        per_opponent: per.iter().map(|&(s, n)| s / n.max(1) as f64).collect(),
        games: results.len(),
    }
}

/// Configuration for weighted blind or skuct gauntlet evaluation (used in
/// PSRO / evolutionary Best-Response runs under `blind:ITERS:0.4`).
#[derive(Clone)]
pub struct WeightedEvalCfg {
    pub games_per_opponent: u32,
    pub agent_iters: u32,
    pub c: f64,
    pub blind_pool: Option<Arc<MetaPool>>,
    pub max_turns: u16,
    pub threads: usize,
    pub seed: u64,
}

/// Evaluate `candidate` against `gauntlet` with optional `weights` (if empty,
/// uniform weights are used) under either `BlindAgent` (when `cfg.blind_pool`
/// is `Some`) or `RmAgent` (`skuct`), with seed-paired common random numbers.
pub fn gauntlet_eval_weighted(
    dex: &Dex,
    candidate: &[PokemonSet],
    gauntlet: &[Vec<PokemonSet>],
    weights: &[f64],
    cfg: &WeightedEvalCfg,
) -> EvalResult {
    let games = (cfg.games_per_opponent + cfg.games_per_opponent % 2) as usize;
    let mut jobs = Vec::with_capacity(gauntlet.len() * games);
    for opp in 0..gauntlet.len() {
        let mut rng =
            SplitMix64::new(cfg.seed ^ (opp as u64 + 1).wrapping_mul(0x9FB2_1C65_1E98_DF25));
        let mut battle_seed = String::new();
        for g in 0..games {
            if g % 2 == 0 {
                battle_seed = rng.battle_seed();
            }
            jobs.push(EvalJob {
                opp,
                battle_seed: battle_seed.clone(),
                cand_seed: rng.next(),
                opp_seed: rng.next(),
                cand_is_p1: g % 2 == 0,
            });
        }
    }

    let cursor = AtomicUsize::new(0);
    let mut results: Vec<(usize, f64)> = Vec::with_capacity(jobs.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..cfg.threads.max(1) {
            let (jobs, cursor) = (&jobs, &cursor);
            handles.push(scope.spawn(move || {
                let mut out: Vec<(usize, (usize, f64))> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let job = &jobs[i];
                    let rm_cfg = RmConfig {
                        iterations: cfg.agent_iters,
                        rule: SelRule::Ucb,
                        c: cfg.c,
                        ..Default::default()
                    };
                    let make_agent = |seed: u64| -> Box<dyn Agent> {
                        if let Some(pool) = &cfg.blind_pool {
                            Box::new(BlindAgent::new(rm_cfg.clone(), pool.clone(), None, seed))
                        } else {
                            Box::new(RmAgent::new(rm_cfg.clone(), seed))
                        }
                    };
                    let mut cand_agent = make_agent(job.cand_seed);
                    let mut opp_agent = make_agent(job.opp_seed);
                    let (t1, t2) = if job.cand_is_p1 {
                        (candidate, gauntlet[job.opp].as_slice())
                    } else {
                        (gauntlet[job.opp].as_slice(), candidate)
                    };
                    let mut b = Battle::from_fixture(dex, &job.battle_seed, t1, t2).unwrap();
                    b.set_log_enabled(cfg.blind_pool.is_some());
                    let res = if job.cand_is_p1 {
                        play_game(
                            dex,
                            &mut b,
                            &mut [cand_agent.as_mut(), opp_agent.as_mut()],
                            cfg.max_turns,
                        )
                    } else {
                        play_game(
                            dex,
                            &mut b,
                            &mut [opp_agent.as_mut(), cand_agent.as_mut()],
                            cfg.max_turns,
                        )
                    }
                    .unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let score = if job.cand_is_p1 { p1_score } else { 1.0 - p1_score };
                    out.push((i, (job.opp, score)));
                }
                out
            }));
        }
        let mut all: Vec<(usize, (usize, f64))> = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all.sort_by_key(|r| r.0);
        results = all.into_iter().map(|(i, (_, s))| (jobs[i].opp, s)).collect();
    });

    let mut per: Vec<(f64, usize)> = vec![(0.0, 0); gauntlet.len()];
    for &(opp, s) in &results {
        per[opp].0 += s;
        per[opp].1 += 1;
    }
    let per_opponent: Vec<f64> = per.iter().map(|&(s, n)| s / n.max(1) as f64).collect();
    let score = if weights.len() == gauntlet.len() {
        let wsum: f64 = weights.iter().sum();
        per_opponent
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| s * w)
            .sum::<f64>()
            / wsum.max(1e-12)
    } else {
        per_opponent.iter().sum::<f64>() / per_opponent.len().max(1) as f64
    };
    EvalResult {
        score,
        per_opponent,
        games: results.len(),
    }
}
