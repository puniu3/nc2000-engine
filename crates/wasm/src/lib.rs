//! M9a wasm bridge: the JS-facing API over the engine + bot crates.
//!
//! Design: coarse-grained JSON-string API (all structured data crosses the
//! boundary as JSON text) — cheap to evolve, trivial to consume, and the
//! per-call volume is tiny next to search work. Four classes, exported under
//! clean JS names:
//!
//! - `Dex` — the loaded game data. Embedded copy (`new Dex()`) or
//!   caller-supplied JSON (`Dex.fromJson(text)`).
//! - `Battle` — one battle, protocol log ON by default (the GUI narrates
//!   from `takeNewLog`). Choices go in as PS-canonical strings (the same
//!   strings `legalChoices` returns), exactly the fixture/inputLog format.
//! - `Searcher` — persistent stepped skuct search over ONE decision point
//!   (state-keyed UCB argmax — the perfect-info flagship). The M9c ponder
//!   worker pumps `step(n)` in slices and reads `best()` when the move is
//!   wanted; create a fresh `Searcher` whenever the battle advances.
//! - `BlindSearcher` (M10c) — the imperfect-info agent: one instance per
//!   GAME (it accumulates the M10a observer/belief), `observe()` fed the
//!   current battle at each decision point (which also snapshots a fresh
//!   stepped `BlindSearch` — same `step`/`best` ponder shape as `Searcher`),
//!   baked-table preview via the belief when the opponent's pool identity
//!   has publicly resolved, `beliefInfo()` for the UI. `pinOpponent` (M12)
//!   switches it to the open-team-sheet product policy: belief pinned to
//!   the opponent's true sets (picks stay hidden), preview by direct
//!   signature lookup.
//! - `PreviewTables` — the M8 baked team-preview equilibria. JS fetches
//!   `meta-pool.json` + `pair-*.json` and feeds them in as strings;
//!   `resolve` returns the matchup's mixed/argmax policies, `sample` draws
//!   a seeded pick as a ready-to-apply choice string.
//!
//! Objects share the dex via `Rc` (wasm is single-threaded), so JS passes
//! the dex only at construction time.

use std::rc::Rc;

use wasm_bindgen::prelude::*;

use nc2000_bot::preview::{MetaPool, TableSet};
use nc2000_bot::{
    baked_preview_pick, open_preview_pick, Belief, BlindSearch, Observer, ProtocolAgent,
    RmConfig, SkuctSearch, SplitMix64,
};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::{Category, Dex};
use nc2000_engine::state::{Battle, Pokemon, RequestKind, Status, BOOST_NAMES};
use nc2000_engine::validate::{canonicalize_team, validate_team, Learnsets};

/// data/gen2stadium2.json baked into the binary (~416 KB, ~150 KB gzipped
/// over the wire) — one fetch fewer and no path plumbing for the common
/// case; `Dex.fromJson` exists for callers that want to ship it separately.
const EMBEDDED_DEX: &str = include_str!("../../../data/gen2stadium2.json");

/// data/learnsets-gen2.json baked in too (~124 KB raw, ~13 KB gzipped —
/// well under the dex's embed precedent, and the validator is useless
/// without it); `Validator.fromJson` exists for callers shipping their own.
const EMBEDDED_LEARNSETS: &str = include_str!("../../../data/learnsets-gen2.json");

fn js_err(e: impl std::fmt::Debug) -> JsError {
    JsError::new(&format!("{e:?}"))
}

// -------------------------------------------------------------------- Dex

#[wasm_bindgen(js_name = Dex)]
pub struct WasmDex {
    dex: Rc<Dex>,
}

#[wasm_bindgen(js_class = Dex)]
impl WasmDex {
    /// The embedded gen2stadium2 dex.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmDex, JsError> {
        Self::from_json(EMBEDDED_DEX)
    }

    /// Construct from caller-supplied dex JSON (`data/gen2stadium2.json`
    /// contents).
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(text: &str) -> Result<WasmDex, JsError> {
        let dex = Dex::from_json(text).map_err(js_err)?;
        Ok(WasmDex { dex: Rc::new(dex) })
    }
}

// -------------------------------------------------------------- Validator

/// M14a team validator + canonicalizer (`nc2000_engine::validate`): checks
/// custom teams against the gen2nc2000 rules the way PS's TeamValidator
/// does, from embedded data only (dex + learnsets — no network). Move
/// legality is a FLAT per-species set: cross-move compatibility constraints
/// are deliberately not encoded, so a small superset of true PS legality is
/// accepted (never the reverse — oracle-certified by
/// `tools/validate-oracle.js`).
#[wasm_bindgen(js_name = Validator)]
pub struct WasmValidator {
    dex: Rc<Dex>,
    learnsets: Learnsets,
}

#[wasm_bindgen(js_class = Validator)]
impl WasmValidator {
    /// Validator over the embedded learnsets (`data/learnsets-gen2.json`).
    #[wasm_bindgen(constructor)]
    pub fn new(dex: &WasmDex) -> Result<WasmValidator, JsError> {
        Self::from_json(dex, EMBEDDED_LEARNSETS)
    }

    /// Construct from caller-supplied learnsets JSON.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(dex: &WasmDex, learnsets_json: &str) -> Result<WasmValidator, JsError> {
        let learnsets = Learnsets::from_json(learnsets_json).map_err(|e| JsError::new(&e))?;
        Ok(WasmValidator { dex: dex.dex.clone(), learnsets })
    }

    /// Validate a team (the same JSON array the `Battle` constructor takes).
    /// Returns findings JSON `{ok, errors, findings: [{severity, code,
    /// mon, slot, ...params}]}` — `severity` is `"error"` (PS's validator
    /// rejects it) or `"fix"` (legal, but PS/the engine silently
    /// canonicalize it; `canonicalizeTeam` applies these). Machine-readable
    /// codes only — M14b renders the messages.
    #[wasm_bindgen(js_name = validateTeam)]
    pub fn validate_team(&self, team_json: &str) -> String {
        validate_team(&self.dex, &self.learnsets, team_json).to_string()
    }

    /// Canonicalize a team: apply every `fix` plus the derivable errors
    /// (HP DV, SpD:=SpA mirrors, gender/shiny from DVs, typed-Hidden-Power
    /// DV spreads, EV clamps/fills, nickname repairs, duplicate moves).
    /// Returns `{ok, team, applied, errors}` — `team` is ready for the
    /// `Battle` constructor when `ok`; `errors` lists what remains
    /// (species/level/move legality, clauses — user intent, not auto-fixed).
    #[wasm_bindgen(js_name = canonicalizeTeam)]
    pub fn canonicalize_team(&self, team_json: &str) -> String {
        canonicalize_team(&self.dex, &self.learnsets, team_json).to_string()
    }
}

// ----------------------------------------------------------------- Battle

#[wasm_bindgen(js_name = Battle)]
pub struct WasmBattle {
    dex: Rc<Dex>,
    battle: Battle,
    log_cursor: usize,
}

#[wasm_bindgen(js_class = Battle)]
impl WasmBattle {
    /// `p1_team` / `p2_team`: JSON arrays of PokemonSet (the fixture
    /// `p1team` / meta-pool `sets` shape). `seed`: PS `Gen5RNG.getSeed()`
    /// format — four decimal 16-bit limbs, comma-joined (e.g. `"1,2,3,4"`).
    /// The battle starts at team preview with the protocol log ON.
    #[wasm_bindgen(constructor)]
    pub fn new(
        dex: &WasmDex,
        p1_team: &str,
        p2_team: &str,
        seed: &str,
    ) -> Result<WasmBattle, JsError> {
        let p1: Vec<PokemonSet> = serde_json::from_str(p1_team).map_err(js_err)?;
        let p2: Vec<PokemonSet> = serde_json::from_str(p2_team).map_err(js_err)?;
        let battle = Battle::from_fixture(&dex.dex, seed, &p1, &p2).map_err(js_err)?;
        Ok(WasmBattle { dex: dex.dex.clone(), battle, log_cursor: 0 })
    }

    /// JSON `[bool, bool]` — which sides owe a choice. `[false, false]`
    /// means the battle has ended.
    #[wasm_bindgen(js_name = needsChoice)]
    pub fn needs_choice(&self) -> String {
        serde_json::to_string(&self.battle.needs_choice()).unwrap()
    }

    /// JSON array of choice objects for `side` (empty ⇔ the side owes
    /// nothing). Every object carries `input` — the PS-canonical string to
    /// pass back to `applyChoice` — plus UI metadata per kind:
    /// move: id/name/type/category/basePower/accuracy/pp/maxpp/target;
    /// switch: pos/species/name/level/hp/maxhp/status;
    /// team: slots (1-based display positions, lead first).
    #[wasm_bindgen(js_name = legalChoices)]
    pub fn legal_choices(&mut self, side: usize) -> String {
        let choices = self.battle.legal_choices(&self.dex, side);
        let arr: Vec<serde_json::Value> =
            choices.iter().map(|&c| choice_json(&self.battle, &self.dex, side, c)).collect();
        serde_json::to_string(&arr).unwrap()
    }

    /// Submit one side's choice as its PS-canonical string (`"move surf"` /
    /// `"switch 3"` / `"team 1, 3, 5"` / `"pass"`). When the last owing side
    /// submits, the battle advances to the next request point (or ends).
    #[wasm_bindgen(js_name = applyChoice)]
    pub fn apply_choice(&mut self, side: usize, input: &str) -> Result<(), JsError> {
        self.battle.choose(&self.dex, side, input).map_err(js_err)
    }

    /// `"p1"` / `"p2"` / `"tie"`, or `null` while the battle is running.
    pub fn outcome(&self) -> Option<String> {
        self.battle.outcome().map(|o| {
            match o {
                Outcome::P1Win => "p1",
                Outcome::P2Win => "p2",
                Outcome::Tie => "tie",
            }
            .to_string()
        })
    }

    pub fn turn(&self) -> u16 {
        self.battle.turn
    }

    /// Current PRNG seed in PS format — the native≡wasm parity invariant
    /// (seed equality = RNG-consumption-order equality).
    pub fn seed(&self) -> String {
        self.battle.prng.seed_str()
    }

    /// Full render-ready state as JSON: per side (active index, party in
    /// display order with hp/status/boosts/moves+pp/item/types), field
    /// (weather, pseudo-weathers), side conditions, request kinds, turn.
    /// Semantics follow the `play` example's panel — full information, both
    /// sides; the GUI decides what the viewer may see.
    #[wasm_bindgen(js_name = stateView)]
    pub fn state_view(&self) -> String {
        state_view_json(&self.battle, &self.dex).to_string()
    }

    /// Protocol log lines appended since the last call (JSON array of raw
    /// PS protocol lines, `|split|`-structure included). The GUI narrates
    /// from these; the `play` example's renderer is the semantics reference.
    #[wasm_bindgen(js_name = takeNewLog)]
    pub fn take_new_log(&mut self) -> String {
        let lines = &self.battle.log[self.log_cursor..];
        let out = serde_json::to_string(lines).unwrap();
        self.log_cursor = self.battle.log.len();
        out
    }

    /// Search mode: disable the protocol log (the human-facing battle keeps
    /// it ON; searchers clone the battle and disable it themselves).
    #[wasm_bindgen(js_name = setLogEnabled)]
    pub fn set_log_enabled(&mut self, on: bool) {
        self.battle.set_log_enabled(on);
    }
}

// --------------------------------------------------------------- Searcher

/// Default per-decision searcher settings = the gate-measured `skuct`
/// configuration (`RmConfig` defaults with rule = Ucb).
fn skuct_config(c: Option<f64>, hp_buckets: Option<i32>) -> RmConfig {
    RmConfig {
        rule: nc2000_bot::smmcts::SelRule::Ucb,
        c: c.unwrap_or(1.0),
        hp_buckets: hp_buckets.map(|b| b as i64).unwrap_or(16),
        ..RmConfig::default()
    }
}

#[wasm_bindgen(js_name = Searcher)]
pub struct WasmSearcher {
    dex: Rc<Dex>,
    search: SkuctSearch,
    side: usize,
}

#[wasm_bindgen(js_class = Searcher)]
impl WasmSearcher {
    /// Snapshot the battle's CURRENT decision point and search it for
    /// `side`. The searcher stays valid (and keeps improving under `step`)
    /// until the battle advances — then create a fresh one. `seed` drives
    /// the searcher's own RNG (chance resampling + tie-breaking);
    /// `c` (UCB exploration, default 1.0) and `hpBuckets` (state-key HP
    /// abstraction, default 16) are the gate-measured skuct defaults.
    #[wasm_bindgen(constructor)]
    pub fn new(
        battle: &WasmBattle,
        side: usize,
        seed: u32,
        c: Option<f64>,
        hp_buckets: Option<i32>,
    ) -> WasmSearcher {
        let cfg = skuct_config(c, hp_buckets);
        let search = SkuctSearch::new(&battle.battle, &battle.dex, cfg, seed as u64);
        WasmSearcher { dex: battle.dex.clone(), search, side }
    }

    /// Pump `n` iterations (each: clone root + fresh chance seed + one
    /// select/expand/rollout/backprop pass), then return control to JS.
    /// Returns total iterations run so far. The ponder loop calls this in
    /// small slices (e.g. 250) to stay responsive.
    pub fn step(&mut self, n: u32) -> u32 {
        self.search.step(&self.dex, n)
    }

    pub fn iterations(&self) -> u32 {
        self.search.iterations()
    }

    /// Current best choice (argmax root visits — the skuct play rule) as a
    /// ready-to-apply input string. `null` if the side owes nothing.
    pub fn best(&self) -> Option<String> {
        self.search.best(self.side).map(|c| c.to_input(&self.dex))
    }

    /// JSON: `{iterations, preview, actions: [{input, visits, mean, frac}]}`
    /// — the root visit distribution (frac = visits share) plus per-action
    /// mean value in [0,1] from this side's perspective. Sorted by visits,
    /// descending.
    #[wasm_bindgen(js_name = rootPolicy)]
    pub fn root_policy(&self) -> String {
        serde_json::json!({
            "iterations": self.search.iterations(),
            "preview": self.search.is_preview(),
            "actions": policy_rows(
                &self.dex,
                self.search.actions(self.side),
                self.search.visits(self.side),
                &self.search.means(self.side),
            ),
        })
        .to_string()
    }
}

/// Root visit rows for `rootPolicy` (shared by `Searcher`/`BlindSearcher`).
/// Forced spots (one legal action) bypass the visit statistics inside the
/// search; report the trivial point mass instead of zeros. Same for a
/// searcher that has not stepped yet: uniform.
fn policy_rows(
    dex: &Dex,
    acts: &[SearchChoice],
    visits: &[u32],
    means: &[f64],
) -> Vec<serde_json::Value> {
    let total: u32 = visits.iter().sum();
    let fallback = if acts.is_empty() { 0.0 } else { 1.0 / acts.len() as f64 };
    let mut rows: Vec<serde_json::Value> = acts
        .iter()
        .zip(visits.iter().zip(means))
        .map(|(&a, (&n, &m))| {
            serde_json::json!({
                "input": a.to_input(dex),
                "visits": n,
                "mean": m,
                "frac": if total > 0 { n as f64 / total as f64 } else { fallback },
            })
        })
        .collect();
    rows.sort_by(|a, b| b["visits"].as_u64().cmp(&a["visits"].as_u64()));
    rows
}

// ---------------------------------------------------------- BlindSearcher

#[wasm_bindgen(js_name = BlindSearcher)]
pub struct WasmBlindSearcher {
    dex: Rc<Dex>,
    cfg: RmConfig,
    side: usize,
    tables: TableSet,
    observer: Observer,
    belief: Belief,
    /// Open-team-sheet mode (M12, `pinOpponent`): the belief is pinned to
    /// the opponent's true sets and preview resolves both sides by
    /// signature (public) instead of through pool identification.
    pinned: bool,
    rng: SplitMix64,
    search: Option<BlindSearch>,
    baked: Option<SearchChoice>,
}

#[wasm_bindgen(js_class = BlindSearcher)]
impl WasmBlindSearcher {
    /// The M10 imperfect-info agent for `side`, one instance per GAME:
    /// it accumulates what that side legitimately observes (public state
    /// diffs + protocol log — run the observed battle log-ON) and keeps a
    /// belief over the meta pool (`pool_json` = `meta-pool.json` contents).
    /// Construct right after the battle (at team preview); feed baked pair
    /// tables with `addPair` for belief-mediated table previews; then per
    /// decision point call `observe(battle)` and pump `step(n)` / read
    /// `best()` exactly like `Searcher`. `seed` drives the agent's own RNG
    /// (determinization sampling, equilibrium sampling, tie-breaking).
    #[wasm_bindgen(constructor)]
    pub fn new(
        battle: &WasmBattle,
        side: usize,
        pool_json: &str,
        seed: u32,
        c: Option<f64>,
        hp_buckets: Option<i32>,
    ) -> Result<WasmBlindSearcher, JsError> {
        let pool: MetaPool = serde_json::from_str(pool_json).map_err(js_err)?;
        let tables = TableSet::from_pool(&battle.dex, &pool);
        let observer = Observer::new(&battle.battle, side);
        let belief = Belief::new(&battle.dex, &pool, &observer);
        Ok(WasmBlindSearcher {
            dex: battle.dex.clone(),
            cfg: skuct_config(c, hp_buckets),
            side,
            tables,
            observer,
            belief,
            pinned: false,
            rng: SplitMix64::new(seed as u64),
            search: None,
            baked: None,
        })
    }

    /// Open-team-sheet mode (M12 product policy): pin the belief to the
    /// opponent's TRUE team (`team_json` = the same JSON array the Battle
    /// was constructed with — pool or custom). Determinizations then equal
    /// the truth except for hidden picks: which 3 of 6 + lead stay hidden
    /// until revealed (unseen pick identities are resampled, the mid-turn
    /// pending-move tell is scrubbed). Preview switches to the direct
    /// table lookup — both sheets are public, so the baked pair answers
    /// whenever the matchup is baked, collision pair included. Call once,
    /// right after construction (at team preview).
    #[wasm_bindgen(js_name = pinOpponent)]
    pub fn pin_opponent(&mut self, team_json: &str) -> Result<(), JsError> {
        let sets: Vec<PokemonSet> = serde_json::from_str(team_json).map_err(js_err)?;
        self.belief = Belief::pinned(&self.dex, "opponent", &sets, &self.observer);
        self.pinned = true;
        Ok(())
    }

    /// Feed one baked pair table (`data/preview-tables-v0/pair-*.json`
    /// contents) for the belief-mediated preview lookup.
    #[wasm_bindgen(js_name = addPair)]
    pub fn add_pair(&mut self, pair_json: &str) -> Result<(), JsError> {
        self.tables.add_pair_json(pair_json).map_err(|e| JsError::new(&e))
    }

    /// Feed the current battle at a decision point: ingest everything that
    /// became visible, re-filter the belief, and snapshot a fresh stepped
    /// search (any previous decision's search is dropped). At a team
    /// preview where the belief is a singleton and the pair is baked, the
    /// M8 mixed table plays instead: `bakedPreview()`/`best()` return the
    /// table pick immediately and `step` is a no-op.
    pub fn observe(&mut self, battle: &WasmBattle) {
        self.observer.observe(&battle.battle, &self.dex);
        self.belief.sync(&self.dex, &self.observer);
        self.baked = None;
        let seed = self.rng.next();
        let search =
            BlindSearch::new(&battle.battle, &self.dex, self.cfg.clone(), self.side, seed);
        if search.is_preview() {
            // Pinned (open sheet): both sheets public — resolve the pair
            // directly by signature, no identification condition. Else the
            // M10 belief-mediated lookup (singleton identification only).
            let pick = if self.pinned {
                open_preview_pick(&self.tables, &battle.battle, self.side, &mut self.rng)
            } else {
                baked_preview_pick(
                    &self.tables,
                    &self.belief,
                    &battle.battle,
                    self.side,
                    &mut self.rng,
                )
            };
            if let Some(c) = pick {
                if search.actions().contains(&c) {
                    self.baked = Some(c);
                }
            }
        }
        self.search = Some(search);
    }

    /// The baked-table preview pick when it applies at the current decision
    /// point (ready-to-apply input string), else `null` — the caller then
    /// pumps the search.
    #[wasm_bindgen(js_name = bakedPreview)]
    pub fn baked_preview(&self) -> Option<String> {
        self.baked.map(|c| c.to_input(&self.dex))
    }

    /// Pump `n` blind iterations (each on a fresh belief determinization).
    /// Returns total iterations run at this decision point. No-op when the
    /// baked preview already decided.
    pub fn step(&mut self, n: u32) -> Result<u32, JsError> {
        let search = self.search.as_mut().ok_or_else(|| {
            JsError::new("BlindSearcher.step before observe")
        })?;
        if self.baked.is_some() {
            return Ok(search.iterations());
        }
        Ok(search.step(&self.dex, &self.belief, &self.observer, n))
    }

    pub fn iterations(&self) -> u32 {
        self.search.as_ref().map_or(0, |s| s.iterations())
    }

    /// Current best choice (baked table pick, or argmax visits over the
    /// global information-set root) as a ready-to-apply input string.
    /// `null` if the side owes nothing (or before `observe`).
    pub fn best(&self) -> Option<String> {
        if let Some(c) = self.baked {
            return Some(c.to_input(&self.dex));
        }
        self.search.as_ref()?.best().map(|c| c.to_input(&self.dex))
    }

    /// JSON: `{iterations, preview, baked, actions: [...]}` — same row
    /// shape as `Searcher.rootPolicy`, over the global root stats. A baked
    /// preview reports its point mass.
    #[wasm_bindgen(js_name = rootPolicy)]
    pub fn root_policy(&self) -> String {
        let (iterations, preview) = match &self.search {
            Some(s) => (s.iterations(), s.is_preview()),
            None => (0, false),
        };
        if let Some(c) = self.baked {
            return serde_json::json!({
                "iterations": iterations,
                "preview": preview,
                "baked": true,
                "actions": [{
                    "input": c.to_input(&self.dex),
                    "visits": 0,
                    "mean": 0.5,
                    "frac": 1.0,
                }],
            })
            .to_string();
        }
        let rows = match &self.search {
            Some(s) => policy_rows(&self.dex, s.actions(), s.visits(), &s.means()),
            None => Vec::new(),
        };
        serde_json::json!({
            "iterations": iterations,
            "preview": preview,
            "baked": false,
            "actions": rows,
        })
        .to_string()
    }

    /// JSON: `{count, fallback, candidates: [pool team ids still alive]}` —
    /// the bot's current read of the opponent's team. `count` 1 = publicly
    /// identified; `fallback` true = no pool team is consistent (a custom
    /// team; imputation runs on a synthesized roster).
    #[wasm_bindgen(js_name = beliefInfo)]
    pub fn belief_info(&self) -> String {
        let candidates: Vec<&str> =
            self.belief.alive().iter().map(|&i| self.belief.candidate_id(i)).collect();
        serde_json::json!({
            "count": self.belief.candidate_count(),
            "fallback": self.belief.is_fallback(),
            "candidates": candidates,
        })
        .to_string()
    }
}

// ------------------------------------------------------- ProtocolSearcher

/// M15a: the imperfect-info agent driven by PLAYER-VISIBLE information only
/// — PS protocol lines + our request JSON — instead of a `Battle` object.
/// One instance per GAME (like `BlindSearcher`): feed every battle line of
/// our player stream with `pushLines`, feed each `|request|` JSON with
/// `onRequest` (which synthesizes a battle consistent with all public info,
/// hidden opponent fields imputed from the M10 belief), then pump
/// `step`/`best` exactly like the other searchers. `best` is always
/// projected onto the PS-legal choice set (incl. the Max Total Level
/// preview rule the engine does not enforce).
#[wasm_bindgen(js_name = ProtocolSearcher)]
pub struct WasmProtocolSearcher {
    dex: Rc<Dex>,
    agent: ProtocolAgent,
}

#[wasm_bindgen(js_class = ProtocolSearcher)]
impl WasmProtocolSearcher {
    /// `side`: 0 = p1. `pool_json` = `meta-pool.json` contents (belief
    /// prior for genuinely-hidden opponents). `seed` drives determinization
    /// sampling / tie-breaking.
    #[wasm_bindgen(constructor)]
    pub fn new(
        dex: &WasmDex,
        side: usize,
        pool_json: &str,
        seed: u32,
        c: Option<f64>,
        hp_buckets: Option<i32>,
    ) -> Result<WasmProtocolSearcher, JsError> {
        let pool: MetaPool = serde_json::from_str(pool_json).map_err(js_err)?;
        let cfg = skuct_config(c, hp_buckets);
        let agent = ProtocolAgent::new(&dex.dex, side, pool, cfg, seed as u64);
        Ok(WasmProtocolSearcher { dex: dex.dex.clone(), agent })
    }

    /// Our exact team, as submitted to PS (same JSON array shape as the
    /// `Battle` constructor). Required before the first request.
    #[wasm_bindgen(js_name = setOwnTeam)]
    pub fn set_own_team(&mut self, team_json: &str) -> Result<(), JsError> {
        let sets: Vec<PokemonSet> = serde_json::from_str(team_json).map_err(js_err)?;
        self.agent.set_own_team(sets);
        Ok(())
    }

    /// Open-team-sheet mode: pin the opponent's true sets (the M12 product
    /// policy). Without this the belief runs pool identification (the parked
    /// full-blind policy — the M15 product policy for genuinely hidden
    /// sets).
    #[wasm_bindgen(js_name = pinOpponent)]
    pub fn pin_opponent(&mut self, team_json: &str) -> Result<(), JsError> {
        let sets: Vec<PokemonSet> = serde_json::from_str(team_json).map_err(js_err)?;
        self.agent.pin_opponent(sets);
        Ok(())
    }

    /// Feed one baked pair table for table-answered previews.
    #[wasm_bindgen(js_name = addPair)]
    pub fn add_pair(&mut self, pair_json: &str) -> Result<(), JsError> {
        self.agent.add_pair_json(pair_json).map_err(|e| JsError::new(&e))
    }

    /// Feed player-visible battle protocol lines (JSON array of strings —
    /// the lines of our player stream, `|request|` lines excluded).
    #[wasm_bindgen(js_name = pushLines)]
    pub fn push_lines(&mut self, lines_json: &str) -> Result<(), JsError> {
        let lines: Vec<String> = serde_json::from_str(lines_json).map_err(js_err)?;
        for line in &lines {
            self.agent.push_line(&self.dex, line);
        }
        Ok(())
    }

    /// Feed the request JSON at a decision point. Returns `false` for a
    /// `wait` request (nothing owed), `true` when a decision is pending
    /// (the searcher is then ready to `step`/`best`).
    #[wasm_bindgen(js_name = onRequest)]
    pub fn on_request(&mut self, request_json: &str) -> Result<bool, JsError> {
        self.agent.on_request(&self.dex, request_json).map_err(|e| JsError::new(&e))
    }

    /// The baked-table preview pick when one applies, else `null`.
    #[wasm_bindgen(js_name = bakedPreview)]
    pub fn baked_preview(&self) -> Option<String> {
        self.agent.baked_preview(&self.dex)
    }

    /// Pump `n` blind iterations (fresh belief determinization each) on the
    /// synthesized battle. No-op when a baked preview / forced choice
    /// already decided.
    pub fn step(&mut self, n: u32) -> Result<u32, JsError> {
        self.agent.step(&self.dex, n).map_err(|e| JsError::new(&e))
    }

    pub fn iterations(&self) -> u32 {
        self.agent.iterations()
    }

    /// Current best choice as a ready-to-submit PS choice string, projected
    /// onto the request-legal set.
    pub fn best(&mut self) -> Option<String> {
        self.agent.best(&self.dex)
    }

    /// JSON `{iterations, baked, forced, actions: [...]}`.
    #[wasm_bindgen(js_name = rootPolicy)]
    pub fn root_policy(&self) -> String {
        self.agent.root_policy(&self.dex)
    }

    /// JSON `{count, fallback, candidates}` — the belief's current read.
    #[wasm_bindgen(js_name = beliefInfo)]
    pub fn belief_info(&self) -> String {
        self.agent.belief_info()
    }

    /// Full state view of the CURRENT synthesized battle (same shape as
    /// `Battle.stateView`) — the M15a gate-b surface: every public field is
    /// asserted against PS's true state by the harness.
    #[wasm_bindgen(js_name = stateView)]
    pub fn state_view(&self) -> Result<String, JsError> {
        let battle = self
            .agent
            .battle()
            .ok_or_else(|| JsError::new("stateView before onRequest"))?;
        Ok(state_view_json(battle, &self.dex).to_string())
    }

    /// `[legality_drift, projections]` counters (target: zeros).
    pub fn metrics(&self) -> String {
        serde_json::json!({
            "legalityDrift": self.agent.legality_drift,
            "projections": self.agent.projections,
        })
        .to_string()
    }
}

// ---------------------------------------------------------- PreviewTables

#[wasm_bindgen(js_name = PreviewTables)]
pub struct WasmTables {
    dex: Rc<Dex>,
    set: TableSet,
}

#[wasm_bindgen(js_class = PreviewTables)]
impl WasmTables {
    /// `pool_json`: the `data/meta-pool-v0/meta-pool.json` contents (JS
    /// fetches it). Builds the team-signature index; pair tables are fed in
    /// afterwards with `addPair`.
    #[wasm_bindgen(constructor)]
    pub fn new(dex: &WasmDex, pool_json: &str) -> Result<WasmTables, JsError> {
        let pool: MetaPool = serde_json::from_str(pool_json).map_err(js_err)?;
        let set = TableSet::from_pool(&dex.dex, &pool);
        Ok(WasmTables { dex: dex.dex.clone(), set })
    }

    /// Feed one baked pair table (`data/preview-tables-v0/pair-*.json`
    /// contents). Errors if the pair references teams outside the pool.
    #[wasm_bindgen(js_name = addPair)]
    pub fn add_pair(&mut self, pair_json: &str) -> Result<(), JsError> {
        self.set.add_pair_json(pair_json).map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen(js_name = pairCount)]
    pub fn pair_count(&self) -> usize {
        self.set.len()
    }

    /// Resolve the battle's matchup (both rosters matched against the pool
    /// by full-set signature). JSON `{found: false}` for unknown matchups
    /// (the caller falls back to `Searcher` preview); otherwise
    /// `{found, teamA, teamB, iAmA, value, mixed: [{slots, input, p}...],
    /// argmax: {slots, input}}` with `mixed` restricted to p > 0, sorted
    /// descending, and `value` the side-a equilibrium value.
    pub fn resolve(&self, battle: &WasmBattle, side: usize) -> String {
        let Some((tab, i_am_a)) = self.set.lookup(&battle.battle, side) else {
            return r#"{"found":false}"#.to_string();
        };
        let actions = self.set.actions();
        let (p, amax) = if i_am_a {
            (&tab.sol.p_a, tab.sol.argmax_a)
        } else {
            (&tab.sol.p_b, tab.sol.argmax_b)
        };
        let mut mixed: Vec<serde_json::Value> = p
            .iter()
            .enumerate()
            .filter(|(_, &pr)| pr > 0.0)
            .map(|(i, &pr)| {
                serde_json::json!({
                    "slots": actions[i],
                    "input": SearchChoice::Team(actions[i]).to_input(&self.dex),
                    "p": pr,
                })
            })
            .collect();
        mixed.sort_by(|a, b| b["p"].as_f64().partial_cmp(&a["p"].as_f64()).unwrap());
        serde_json::json!({
            "found": true,
            "teamA": tab.team_a,
            "teamB": tab.team_b,
            "iAmA": i_am_a,
            "value": tab.sol.value,
            "mixed": mixed,
            "argmax": {
                "slots": actions[amax],
                "input": SearchChoice::Team(actions[amax]).to_input(&self.dex),
            },
        })
        .to_string()
    }

    /// Sample a preview pick from the mixed equilibrium with a caller seed
    /// (ready-to-apply input string), or `null` for unknown matchups.
    /// Deterministic per seed — one fresh seed per preview decision.
    pub fn sample(&self, battle: &WasmBattle, side: usize, seed: u32) -> Option<String> {
        let (tab, i_am_a) = self.set.lookup(&battle.battle, side)?;
        let p = if i_am_a { &tab.sol.p_a } else { &tab.sol.p_b };
        let mut rng = SplitMix64::new(seed as u64);
        let u = rng.next_f64();
        let mut acc = 0.0;
        let mut pick = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        for (i, &pr) in p.iter().enumerate() {
            acc += pr;
            if u < acc {
                pick = i;
                break;
            }
        }
        let triple = self.set.actions()[pick];
        Some(SearchChoice::Team(triple).to_input(&self.dex))
    }

    /// The argmax-policy pick (pure best reply to the opponent's
    /// equilibrium), or `null` for unknown matchups.
    pub fn argmax(&self, battle: &WasmBattle, side: usize) -> Option<String> {
        let (tab, i_am_a) = self.set.lookup(&battle.battle, side)?;
        let amax = if i_am_a { tab.sol.argmax_a } else { tab.sol.argmax_b };
        let triple = self.set.actions()[amax];
        Some(SearchChoice::Team(triple).to_input(&self.dex))
    }
}

// ------------------------------------------------------------ view builders

fn choice_json(battle: &Battle, dex: &Dex, side: usize, c: SearchChoice) -> serde_json::Value {
    let input = c.to_input(dex);
    match c {
        SearchChoice::Move(id) => {
            let ms = dex.move_static(id);
            let (pp, maxpp) = battle
                .active_id(side)
                .and_then(|aid| {
                    battle
                        .poke(aid)
                        .move_slots
                        .iter()
                        .find(|s| s.id == id)
                        .map(|s| (s.pp, s.maxpp))
                })
                .unwrap_or((-1, -1));
            serde_json::json!({
                "kind": "move",
                "input": input,
                "id": dex.moves.key(id),
                "name": ms.name,
                "type": dex.type_name(ms.move_type),
                "category": category_str(ms.category),
                "basePower": ms.base_power,
                "pp": pp,
                "maxpp": maxpp,
                "target": ms.target,
            })
        }
        SearchChoice::Switch(pos) => {
            let s = &battle.sides[side];
            let p = &s.roster[s.party[(pos - 1) as usize] as usize];
            serde_json::json!({
                "kind": "switch",
                "input": input,
                "pos": pos,
                "species": dex.species.get(p.species).name,
                "name": p.name.as_str(),
                "level": p.level,
                "hp": p.hp.max(0),
                "maxhp": p.maxhp,
                "status": status_str(p.status),
            })
        }
        SearchChoice::Team(slots) => serde_json::json!({
            "kind": "team",
            "input": input,
            "slots": slots,
        }),
        SearchChoice::Pass => serde_json::json!({ "kind": "pass", "input": input }),
    }
}

fn category_str(c: Category) -> &'static str {
    match c {
        Category::Physical => "Physical",
        Category::Special => "Special",
        Category::Status => "Status",
    }
}

fn status_str(s: Status) -> &'static str {
    match s {
        Status::None => "",
        other => other.as_str(),
    }
}

fn request_str(r: Option<RequestKind>) -> &'static str {
    match r {
        Some(RequestKind::TeamPreview) => "teampreview",
        Some(RequestKind::Move) => "move",
        Some(RequestKind::Switch) => "switch",
        Some(RequestKind::Wait) | None => "",
    }
}

fn poke_json(dex: &Dex, p: &Pokemon) -> serde_json::Value {
    let mut boosts = serde_json::Map::new();
    for (&b, name) in p.boosts.iter().zip(BOOST_NAMES) {
        boosts.insert(name.to_string(), serde_json::json!(b));
    }
    let moves: Vec<serde_json::Value> = p
        .move_slots
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": dex.moves.key(s.id),
                "name": dex.move_static(s.id).name,
                "pp": s.pp,
                "maxpp": s.maxpp,
                "disabled": s.disabled,
            })
        })
        .collect();
    let types: Vec<&str> = p.types.iter().map(|t| dex.type_name(t)).collect();
    let volatiles: Vec<&str> =
        p.volatiles.iter().map(|(id, _)| dex.conds_key(*id)).collect();
    serde_json::json!({
        "species": dex.species.get(p.species).name,
        "name": p.name.as_str(),
        "level": p.level,
        "gender": p.gender.as_str(),
        "hp": p.hp.max(0),
        "maxhp": p.maxhp,
        "fainted": p.fainted,
        "status": status_str(p.status),
        "boosts": boosts,
        "moves": moves,
        "item": p.item.map(|it| dex.items.get(it).name.clone()),
        "types": types,
        "volatiles": volatiles,
        "trapped": p.trapped,
    })
}

fn state_view_json(battle: &Battle, dex: &Dex) -> serde_json::Value {
    let sides: Vec<serde_json::Value> = (0..2)
        .map(|n| {
            let side = &battle.sides[n];
            // party in display order (preview: 6 mons; battle: the picked 3)
            let party: Vec<serde_json::Value> = side
                .party
                .iter()
                .map(|&slot| poke_json(dex, &side.roster[slot as usize]))
                .collect();
            // display index of the active mon, if any
            let active = side
                .active
                .and_then(|slot| side.party.iter().position(|&x| x == slot));
            let conditions: Vec<&str> =
                side.side_conditions.iter().map(|(id, _)| dex.conds_key(*id)).collect();
            serde_json::json!({
                "name": side.name,
                "active": active,
                "party": party,
                "pokemonLeft": side.pokemon_left,
                "sideConditions": conditions,
                "request": request_str(side.request_state()),
            })
        })
        .collect();
    let field = serde_json::json!({
        "weather": battle.field.weather.map(|w| dex.conds_key(w)),
        "pseudoWeather": battle
            .field
            .pseudo_weather
            .iter()
            .map(|(id, _)| dex.conds_key(*id))
            .collect::<Vec<_>>(),
    });
    serde_json::json!({
        "turn": battle.turn,
        "sides": sides,
        "field": field,
        "outcome": battle.outcome().map(|o| match o {
            Outcome::P1Win => "p1",
            Outcome::P2Win => "p2",
            Outcome::Tie => "tie",
        }),
    })
}

// ------------------------------------------------------------------ misc

/// Derive a PS-format battle seed from a small integer (convenience for
/// demos/tests; any "a,b,c,d" 16-bit-limb string works directly).
#[wasm_bindgen(js_name = deriveBattleSeed)]
pub fn derive_battle_seed(seed: u32) -> String {
    SplitMix64::new(seed as u64).battle_seed()
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Raw team JSON straight from a fixture file (PokemonSet is
    /// deserialize-only; the wasm API consumes team JSON as text anyway).
    fn fixture_teams() -> (String, String) {
        let path = conformance::fixture::repo_root()
            .join("fixtures/corpus-v1/full/battle-001.json");
        let text = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        (v["p1team"].to_string(), v["p2team"].to_string())
    }

    #[test]
    fn embedded_dex_loads() {
        let _ = WasmDex::new().map_err(|_| "dex").unwrap();
    }

    /// M14a: the embedded-learnset validator accepts fixture teams verbatim
    /// and flags an injected illegal move; canonicalize round-trips into the
    /// Battle constructor.
    #[test]
    fn validator_flow() {
        let dex = WasmDex::new().map_err(|_| "dex").unwrap();
        let v = WasmValidator::new(&dex).map_err(|_| "validator").unwrap();
        let (p1, p2) = fixture_teams();
        for team in [&p1, &p2] {
            let r: serde_json::Value =
                serde_json::from_str(&v.validate_team(team)).unwrap();
            assert_eq!(r["ok"], true, "{r}");
        }
        // corrupt: give the first mon a move outside its learnset
        let mut t: serde_json::Value = serde_json::from_str(&p1).unwrap();
        t[0]["moves"][0] = serde_json::json!("Spikes");
        let r: serde_json::Value =
            serde_json::from_str(&v.validate_team(&t.to_string())).unwrap();
        assert_eq!(r["ok"], false, "{r}");
        assert!(r["findings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["code"] == "move-illegal"));
        // canonicalize a sloppy but fixable team and construct a battle
        let mut t: serde_json::Value = serde_json::from_str(&p1).unwrap();
        t[0]["ability"] = serde_json::json!("");
        t[0]["gender"] = serde_json::Value::Null;
        let c: serde_json::Value =
            serde_json::from_str(&v.canonicalize_team(&t.to_string())).unwrap();
        assert_eq!(c["ok"], true, "{c}");
        let fixed = c["team"].to_string();
        let _ = WasmBattle::new(&dex, &fixed, &p2, "1,2,3,4")
            .map_err(|_| "battle from canonicalized team")
            .unwrap();
    }

    #[test]
    fn battle_runs_and_searcher_picks_legal() {
        let dex = WasmDex::new().map_err(|_| "dex").unwrap();
        let (p1, p2) = fixture_teams();
        let mut b = WasmBattle::new(&dex, &p1, &p2, "1,2,3,4")
            .map_err(|_| "battle")
            .unwrap();
        // preview log exists
        let log0: Vec<String> = serde_json::from_str(&b.take_new_log()).unwrap();
        assert!(!log0.is_empty());
        // play a few decisions with the stepped searcher
        for _ in 0..6 {
            if b.outcome().is_some() {
                break;
            }
            let needs: [bool; 2] = serde_json::from_str(&b.needs_choice()).unwrap();
            let mut picks: Vec<(usize, String)> = Vec::new();
            for side in 0..2 {
                if !needs[side] {
                    continue;
                }
                let legal: Vec<serde_json::Value> =
                    serde_json::from_str(&b.legal_choices(side)).unwrap();
                assert!(!legal.is_empty());
                let mut s = WasmSearcher::new(&b, side, 7 + side as u32, None, None);
                s.step(60);
                let best = s.best().unwrap();
                assert!(
                    legal.iter().any(|c| c["input"] == best.as_str()),
                    "searcher best {best} not in legal set"
                );
                picks.push((side, best));
            }
            for (side, input) in picks {
                b.apply_choice(side, &input).map_err(|_| "apply").unwrap();
            }
        }
        // state view parses and has both sides
        let view: serde_json::Value = serde_json::from_str(&b.state_view()).unwrap();
        assert_eq!(view["sides"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn blind_searcher_flow() {
        let root = conformance::fixture::repo_root();
        let pool_json =
            std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
        let pool: serde_json::Value = serde_json::from_str(&pool_json).unwrap();
        let team = |i: usize| pool["teams"][i]["sets"].to_string();

        let dex = WasmDex::new().map_err(|_| "dex").unwrap();
        let mut b = WasmBattle::new(&dex, &team(0), &team(1), "1,2,3,4")
            .map_err(|_| "battle")
            .unwrap();
        let mut bs = WasmBlindSearcher::new(&b, 1, &pool_json, 7, None, None)
            .map_err(|_| "blind searcher")
            .unwrap();
        let pair_json = std::fs::read_to_string(
            root.join("fixtures/preview-tables-test/pair-00-01.json"),
        )
        .unwrap();
        bs.add_pair(&pair_json).map_err(|_| "pair").unwrap();

        // preview: the opponent (side 0 = pool team 0) identifies publicly
        // and the pair is baked -> table pick, no stepping needed
        bs.observe(&b);
        let info: serde_json::Value = serde_json::from_str(&bs.belief_info()).unwrap();
        assert_eq!(info["count"], 1);
        assert_eq!(info["fallback"], false);
        assert_eq!(info["candidates"][0], pool["teams"][0]["id"]);
        let baked = bs.baked_preview().expect("baked preview pick");
        assert_eq!(bs.best().as_deref(), Some(baked.as_str()));
        let legal: Vec<serde_json::Value> =
            serde_json::from_str(&b.legal_choices(1)).unwrap();
        assert!(legal.iter().any(|c| c["input"] == baked.as_str()));
        let pol: serde_json::Value = serde_json::from_str(&bs.root_policy()).unwrap();
        assert_eq!(pol["baked"], true);

        // into battle: stepped blind search returns a legal pick
        // (team 0 is 52/52/52/51/51/51 — slots 1,2,4 sum 155 = Max Total Level)
        b.apply_choice(0, "team 1, 2, 4").map_err(|_| "apply").unwrap();
        b.apply_choice(1, &baked).map_err(|_| "apply").unwrap();
        bs.observe(&b);
        assert!(bs.baked_preview().is_none());
        let done = bs.step(40).map_err(|_| "step").unwrap();
        assert_eq!(done, 40);
        let best = bs.best().expect("in-battle best");
        let legal: Vec<serde_json::Value> =
            serde_json::from_str(&b.legal_choices(1)).unwrap();
        assert!(legal.iter().any(|c| c["input"] == best.as_str()));

        // fallback belief: an off-pool opponent flips is_fallback and the
        // search still plays
        let (p1, _) = fixture_teams();
        let mut fb = WasmBattle::new(&dex, &p1, &team(1), "1,2,3,4")
            .map_err(|_| "battle")
            .unwrap();
        let mut bfs = WasmBlindSearcher::new(&fb, 1, &pool_json, 9, None, None)
            .map_err(|_| "blind searcher")
            .unwrap();
        bfs.observe(&fb);
        let info: serde_json::Value = serde_json::from_str(&bfs.belief_info()).unwrap();
        assert_eq!(info["fallback"], true);
        assert_eq!(info["count"], 0);
        assert!(bfs.baked_preview().is_none());
        bfs.step(30).map_err(|_| "step").unwrap();
        let best = bfs.best().expect("fallback best");
        let legal: Vec<serde_json::Value> =
            serde_json::from_str(&fb.legal_choices(1)).unwrap();
        assert!(legal.iter().any(|c| c["input"] == best.as_str()));
    }

    /// M12 open-team-sheet mode: `pinOpponent` pins the belief to the
    /// opponent's true sets (pool or custom) and preview resolves by
    /// direct signature lookup.
    #[test]
    fn pinned_open_sheet_flow() {
        let root = conformance::fixture::repo_root();
        let pool_json =
            std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
        let pool: serde_json::Value = serde_json::from_str(&pool_json).unwrap();
        let team = |i: usize| pool["teams"][i]["sets"].to_string();
        let dex = WasmDex::new().map_err(|_| "dex").unwrap();

        // pool opponent + baked pair: preview answers from the table
        let mut b = WasmBattle::new(&dex, &team(0), &team(1), "1,2,3,4")
            .map_err(|_| "battle")
            .unwrap();
        let mut bs = WasmBlindSearcher::new(&b, 1, &pool_json, 7, None, None)
            .map_err(|_| "blind searcher")
            .unwrap();
        bs.pin_opponent(&team(0)).map_err(|_| "pin").unwrap();
        let pair_json = std::fs::read_to_string(
            root.join("fixtures/preview-tables-test/pair-00-01.json"),
        )
        .unwrap();
        bs.add_pair(&pair_json).map_err(|_| "pair").unwrap();
        bs.observe(&b);
        let info: serde_json::Value = serde_json::from_str(&bs.belief_info()).unwrap();
        assert_eq!(info["count"], 1);
        assert_eq!(info["fallback"], false);
        let baked = bs.baked_preview().expect("open-sheet baked preview pick");
        let legal: Vec<serde_json::Value> =
            serde_json::from_str(&b.legal_choices(1)).unwrap();
        assert!(legal.iter().any(|c| c["input"] == baked.as_str()));

        // into battle: the pinned blind search returns a legal pick
        // (team 0 is 52/52/52/51/51/51 — slots 1,2,4 sum 155 = Max Total Level)
        b.apply_choice(0, "team 1, 2, 4").map_err(|_| "apply").unwrap();
        b.apply_choice(1, &baked).map_err(|_| "apply").unwrap();
        bs.observe(&b);
        assert!(bs.baked_preview().is_none());
        bs.step(40).map_err(|_| "step").unwrap();
        let best = bs.best().expect("in-battle best");
        let legal: Vec<serde_json::Value> =
            serde_json::from_str(&b.legal_choices(1)).unwrap();
        assert!(legal.iter().any(|c| c["input"] == best.as_str()));

        // custom (off-pool) opponent pinned: a REAL singleton belief —
        // never fallback — with live-search preview (no table for the
        // matchup)
        let (p1, _) = fixture_teams();
        let mut fb = WasmBattle::new(&dex, &p1, &team(1), "1,2,3,4")
            .map_err(|_| "battle")
            .unwrap();
        let mut bfs = WasmBlindSearcher::new(&fb, 1, &pool_json, 9, None, None)
            .map_err(|_| "blind searcher")
            .unwrap();
        bfs.pin_opponent(&p1).map_err(|_| "pin").unwrap();
        bfs.observe(&fb);
        let info: serde_json::Value = serde_json::from_str(&bfs.belief_info()).unwrap();
        assert_eq!(info["fallback"], false);
        assert_eq!(info["count"], 1);
        assert!(bfs.baked_preview().is_none());
        bfs.step(30).map_err(|_| "step").unwrap();
        let best = bfs.best().expect("pinned custom best");
        let legal: Vec<serde_json::Value> =
            serde_json::from_str(&fb.legal_choices(1)).unwrap();
        assert!(legal.iter().any(|c| c["input"] == best.as_str()));
    }
}
