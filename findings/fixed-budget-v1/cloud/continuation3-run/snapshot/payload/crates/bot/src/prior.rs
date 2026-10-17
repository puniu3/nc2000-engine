//! M18 work item 3 — the community belief prior's interpreter.
//!
//! Reads the `nc2000-belief-prior` format that `examples/count_belief_prior.rs`
//! emits (per-`(species, move)` carry-marginals, optional per-species item
//! marginals and lead probability) and hands `belief::Belief` a resolved,
//! in-range table.
//!
//! # Totality is the whole contract
//!
//! `docs/community-belief-prior-design.md` puts the safety of hand-edited,
//! unreviewed community data on one invariant: *certified code dominates the
//! data*. Reveals, legality and HP imputation are code and always win, so the
//! worst a bad number can do is play weakly against an unrevealed set. That
//! argument only holds if the reader itself cannot fail — a table that
//! crashes the bot is not "suboptimal play", it is an outage. So every entry
//! point here is **total**: it returns a table for any input bytes, including
//! bytes that are not JSON.
//!
//! The premise is explicitly non-adversarial (local machine, well-intentioned
//! non-technical contributors), so this is malformed-robustness, not
//! hardening: no size limits, no recursion guards, no signature checks. The
//! failure mode being engineered against is a typo, not an attacker.
//!
//! Concretely, and in the design doc's words — *clamp/normalise, ignore
//! unknown keys, default on missing/garbage, never crash*:
//!
//! - unparseable bytes / non-object root / missing `species` → the **empty**
//!   table, which the sampler treats as "no prior loaded", i.e. today's
//!   behaviour;
//! - probabilities are **clamped** to `[0, 1]`; they are deliberately *not*
//!   renormalised, because these are marginals over four slots (a species
//!   sums to ~4.0, not 1.0) and the weighted draw needs no normalisation;
//! - a non-numeric probability is dropped, not fatal; numeric strings and a
//!   trailing `%` are accepted, since that is what a human hand-edit looks
//!   like;
//! - a zero or negative weight means *never*, and is dropped from the table
//!   rather than kept as a zero-weight draw candidate;
//! - unknown keys anywhere (`note`, the counter's per-species `n`, a
//!   misspelt field) are ignored in silence;
//! - everything skipped is recorded in [`BeliefPrior::warnings`] so a human
//!   can see what the machine did not understand, without any of it being an
//!   error.
//!
//! # Precedence
//!
//! [`BeliefPrior::overlay`] implements the doc's rule exactly: **per species,
//! not global** (a file mentioning only Snorlax changes only Snorlax) and
//! **within a species, wholesale** (the incoming `moves` map replaces the
//! resident one outright — a per-move merge would produce a hybrid neither
//! the editor nor a reader could predict).

use std::collections::BTreeMap;
use std::path::Path;

use nc2000_engine::dex::toid;

/// The `format` string a well-formed table carries.
pub const FORMAT: &str = "nc2000-belief-prior";
/// The format revision this interpreter was written against.
pub const VERSION: u64 = 1;
/// Conventional auto-pickup path, relative to the repo root. Note that the
/// tracked sample lives next to it as `belief-prior-v0.sample.json`: shipping
/// a table AT this path would silently activate M18 for every hidden-team
/// game, and the design requires the no-file default to stay unchanged.
pub const DEFAULT_PATH: &str = "data/belief-prior-v0.json";

/// How many warnings are retained before the tail is summarised. A human
/// reading a diagnostic does not need the thousandth instance of it.
const MAX_WARNINGS: usize = 32;

/// One species' marginals. `moves` and `items` are sorted by id and carry
/// strictly positive, `[0, 1]`-clamped weights.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeciesPrior {
    /// `P(this species carries this move)`, by move id.
    pub moves: Vec<(String, f64)>,
    /// `P(this species holds this item)`, by item id. Unused by the M18
    /// sampler (item identity is already reveal-dominated and the fallback
    /// set supplies one); parsed so a table that carries them round-trips.
    pub items: Vec<(String, f64)>,
    /// `P(this species leads)`. Parsed, currently unconsumed — pick identity
    /// is resampled uniformly by the determinizer.
    pub lead: Option<f64>,
}

/// A resolved belief-prior table. Construct it with [`BeliefPrior::from_json`]
/// or [`BeliefPrior::load`]; both are total.
#[derive(Clone, Debug, Default)]
pub struct BeliefPrior {
    species: BTreeMap<String, SpeciesPrior>,
    warnings: Vec<String>,
    skipped: usize,
}

impl BeliefPrior {
    /// The table that changes nothing. Equivalent to "no prior file loaded".
    pub fn empty() -> BeliefPrior {
        BeliefPrior::default()
    }

    pub fn is_empty(&self) -> bool {
        self.species.is_empty()
    }

    pub fn len(&self) -> usize {
        self.species.len()
    }

    /// Marginals for a PS species id (`dex.species.key(..)`), or `None` when
    /// the table says nothing about it — which the sampler reads as "keep
    /// today's deterministic filler for this species".
    pub fn species(&self, id: &str) -> Option<&SpeciesPrior> {
        self.species.get(id)
    }

    pub fn species_ids(&self) -> impl Iterator<Item = &str> {
        self.species.keys().map(String::as_str)
    }

    /// Everything the reader did not understand. Never an error — a caller
    /// may print these, and must not treat them as a load failure.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Total entries dropped (species, moves and items alike). `warnings()`
    /// is capped; this is not.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Per-species sum of move marginals — the design doc's editing
    /// invariant. Complete 4-move sets give ~4.0; a reveal-derived table sits
    /// near 2.5; a sum above ~4.5 claims more than four moves per set. The
    /// interpreter deliberately does **not** enforce it (enforcement would
    /// break totality and silently rewrite the owner's numbers); it only
    /// reports it.
    pub fn move_sum(&self, id: &str) -> Option<f64> {
        self.species
            .get(id)
            .map(|s| s.moves.iter().map(|(_, p)| *p).sum())
    }

    /// Mean of [`Self::move_sum`] over the table — the coverage diagnostic in
    /// one number.
    pub fn mean_move_sum(&self) -> f64 {
        if self.species.is_empty() {
            return 0.0;
        }
        self.species
            .values()
            .map(|s| s.moves.iter().map(|(_, p)| *p).sum::<f64>())
            .sum::<f64>()
            / self.species.len() as f64
    }

    /// Apply `other` on top of this table with the design doc's precedence:
    /// per species (species `other` never mentions keep their entry here) and
    /// wholesale within a species (`other`'s entry replaces this one entirely,
    /// never merging move by move).
    pub fn overlay(&mut self, other: BeliefPrior) {
        for (id, entry) in other.species {
            self.species.insert(id, entry);
        }
        for w in other.warnings {
            self.note(w);
        }
        self.skipped += other.skipped;
    }

    /// Read a table from JSON text. Total: any bytes yield a table, and bytes
    /// that make no sense yield the empty one.
    pub fn from_json(text: &str) -> BeliefPrior {
        let mut out = BeliefPrior::default();
        let root: serde_json::Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(e) => {
                out.note(format!("not JSON ({e}); ignoring the file entirely"));
                return out;
            }
        };
        let Some(obj) = root.as_object() else {
            out.note("top level is not a JSON object; ignoring the file".to_string());
            return out;
        };
        if let Some(f) = obj.get("format").and_then(|v| v.as_str()) {
            if toid(f) != toid(FORMAT) {
                out.note(format!(
                    "format is {f:?}, expected {FORMAT:?}; reading it anyway"
                ));
            }
        }
        if let Some(v) = obj.get("version").and_then(|v| v.as_u64()) {
            if v > VERSION {
                out.note(format!(
                    "version {v} is newer than {VERSION}; unknown fields will be ignored"
                ));
            }
        }
        let Some(species) = obj.get("species").and_then(|v| v.as_object()) else {
            out.note("no `species` object; nothing to apply".to_string());
            return out;
        };
        for (raw_key, raw_entry) in species {
            let id = toid(raw_key);
            if id.is_empty() {
                out.skip(format!("species key {raw_key:?} has no id characters"));
                continue;
            }
            let Some(entry) = raw_entry.as_object() else {
                out.skip(format!("species {id}: entry is not an object"));
                continue;
            };
            let moves = out.read_marginals(&id, "moves", entry.get("moves"));
            let items = out.read_marginals(&id, "items", entry.get("items"));
            let lead = entry.get("lead").and_then(as_prob);
            if moves.is_empty() && items.is_empty() && lead.is_none() {
                out.skip(format!("species {id}: nothing usable in the entry"));
                continue;
            }
            if let Some(prev) = out.species.insert(id.clone(), SpeciesPrior { moves, items, lead })
            {
                // Two keys that normalise to one id ("Snorlax" and
                // "snorlax"). Last wins, matching the wholesale rule.
                let _ = prev;
                out.skip(format!("species {id}: duplicate key, later entry wins"));
            }
        }
        out
    }

    /// Read a table from a file. Total: a missing or unreadable file yields
    /// the empty table plus a warning, never an error.
    pub fn load(path: &Path) -> BeliefPrior {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let mut prior = BeliefPrior::from_json(&text);
                if prior.is_empty() {
                    prior.note(format!("{} produced no usable species", path.display()));
                }
                prior
            }
            Err(e) => {
                let mut out = BeliefPrior::default();
                out.note(format!("cannot read {}: {e}", path.display()));
                out
            }
        }
    }

    /// The conventional auto-pickup path under `root`. Absent ⇒ the empty
    /// table with **no** warning: not having a prior is the shipped default,
    /// not a problem worth reporting.
    pub fn load_conventional(root: &Path) -> BeliefPrior {
        let path = root.join(DEFAULT_PATH);
        if path.exists() {
            BeliefPrior::load(&path)
        } else {
            BeliefPrior::empty()
        }
    }

    // ------------------------------------------------------------ internals

    fn read_marginals(
        &mut self,
        species: &str,
        field: &str,
        value: Option<&serde_json::Value>,
    ) -> Vec<(String, f64)> {
        let Some(value) = value else { return Vec::new() };
        let Some(map) = value.as_object() else {
            self.skip(format!("species {species}: `{field}` is not an object"));
            return Vec::new();
        };
        let mut out: BTreeMap<String, f64> = BTreeMap::new();
        for (raw_key, raw_p) in map {
            let key = toid(raw_key);
            if key.is_empty() {
                self.skip(format!("species {species}: {field} key {raw_key:?} has no id"));
                continue;
            }
            let Some(p) = as_prob(raw_p) else {
                self.skip(format!(
                    "species {species}: {field}.{key} = {raw_p} is not a probability"
                ));
                continue;
            };
            if p <= 0.0 {
                // A zero marginal is a statement ("never"), and the way to
                // honour it is to leave the move out of the draw entirely.
                continue;
            }
            // Two spellings of one id keep the stronger claim, so a hand-edit
            // that adds "Body Slam" next to an existing "bodyslam" cannot
            // silently weaken the entry.
            let slot = out.entry(key).or_insert(0.0);
            *slot = slot.max(p);
        }
        out.into_iter().collect()
    }

    fn note(&mut self, msg: String) {
        if self.warnings.len() < MAX_WARNINGS {
            self.warnings.push(msg);
        } else if self.warnings.len() == MAX_WARNINGS {
            self.warnings.push("... further warnings suppressed".to_string());
        }
    }

    fn skip(&mut self, msg: String) {
        self.skipped += 1;
        self.note(msg);
    }
}

/// Coerce one JSON value into a `[0, 1]` probability. Accepts numbers, and
/// the two shapes a hand-edit actually produces: a quoted number, and a
/// trailing `%`. Everything else — booleans, nulls, arrays, prose, NaN — is
/// not a probability and is reported rather than guessed at.
fn as_prob(v: &serde_json::Value) -> Option<f64> {
    let x = match v {
        serde_json::Value::Number(n) => n.as_f64()?,
        serde_json::Value::String(s) => {
            let s = s.trim();
            match s.strip_suffix('%') {
                Some(head) => head.trim().parse::<f64>().ok()? / 100.0,
                None => s.parse::<f64>().ok()?,
            }
        }
        _ => return None,
    };
    if !x.is_finite() {
        return None;
    }
    Some(x.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snorlax(text: &str) -> BeliefPrior {
        BeliefPrior::from_json(text)
    }

    #[test]
    fn a_well_formed_table_round_trips() {
        let p = snorlax(
            r#"{"format":"nc2000-belief-prior","version":1,"note":"free text",
                "species":{"snorlax":{
                    "moves":{"bodyslam":0.82,"curse":0.61},
                    "items":{"leftovers":0.7},
                    "lead":0.31}}}"#,
        );
        assert_eq!(p.len(), 1);
        let s = p.species("snorlax").unwrap();
        assert_eq!(s.moves, [("bodyslam".to_string(), 0.82), ("curse".to_string(), 0.61)]);
        assert_eq!(s.items, [("leftovers".to_string(), 0.7)]);
        assert_eq!(s.lead, Some(0.31));
        assert!(p.warnings().is_empty(), "{:?}", p.warnings());
        assert!((p.move_sum("snorlax").unwrap() - 1.43).abs() < 1e-9);
    }

    #[test]
    fn garbage_bytes_degrade_to_the_empty_table_instead_of_panicking() {
        for text in [
            "",
            "not json at all",
            "[1,2,3]",
            "null",
            "42",
            r#"{"species": 7}"#,
            r#"{"species":{"snorlax":"oops"}}"#,
            r#"{"species":{"snorlax":{"moves":[1,2]}}}"#,
            r#"{"species":{"":{"moves":{"rest":1}}}}"#,
            "{\"species\":{\"snorlax\":{\"moves\":{\"rest\":",
        ] {
            let p = BeliefPrior::from_json(text);
            assert!(p.is_empty(), "{text:?} produced {} species", p.len());
            assert!(!p.warnings().is_empty(), "{text:?} warned about nothing");
        }
        // A well-formed but empty table is not garbage: nothing to apply and
        // nothing to complain about.
        let none = BeliefPrior::from_json(r#"{"species":{}}"#);
        assert!(none.is_empty() && none.warnings().is_empty());
    }

    #[test]
    fn probabilities_are_clamped_never_renormalised() {
        let p = snorlax(
            r#"{"species":{"snorlax":{"moves":{
                 "bodyslam": 3.5, "curse": -1, "rest": "0.5", "sleeptalk": "55%",
                 "earthquake": "NaN", "hyperbeam": "inf", "return": "nope",
                 "lovelykiss": true, "doubleedge": null}}}}"#,
        );
        let s = p.species("snorlax").unwrap();
        let got: BTreeMap<&str, f64> =
            s.moves.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        assert_eq!(got.get("bodyslam"), Some(&1.0), "clamped up");
        assert_eq!(got.get("rest"), Some(&0.5), "numeric string");
        assert_eq!(got.get("sleeptalk"), Some(&0.55), "percent string");
        assert!(!got.contains_key("curse"), "negative dropped");
        assert!(!got.contains_key("return"), "prose dropped");
        assert!(!got.contains_key("lovelykiss"), "bool dropped");
        assert!(!got.contains_key("doubleedge"), "null dropped");
        // JSON has no NaN/Infinity literal, so non-finite can only arrive as
        // a string — which `f64::from_str` happily accepts, hence the guard.
        assert!(!got.contains_key("earthquake"), "NaN dropped");
        assert!(!got.contains_key("hyperbeam"), "infinity dropped");
        // The sum is left exactly as the numbers say — no renormalisation.
        assert!((p.move_sum("snorlax").unwrap() - 2.05).abs() < 1e-9);
    }

    #[test]
    fn unknown_keys_are_ignored_including_the_counters_per_species_n() {
        // `count_belief_prior` emits `n` per species; the format sample does
        // not mention it. Ignoring it is the point of the rule.
        let p = snorlax(
            r#"{"format":"nc2000-belief-prior","version":9,"whatever":{"x":1},
                "species":{"Snorlax":{"moves":{"Body Slam":0.8},"n":192,
                                      "mvoes":{"typo":1.0}}}}"#,
        );
        let s = p.species("snorlax").expect("key normalised through toid");
        assert_eq!(s.moves, [("bodyslam".to_string(), 0.8)]);
        // A newer version warns but still reads.
        assert!(p.warnings().iter().any(|w| w.contains("version 9")));
        // A misspelt field is not an object of probabilities, but it is also
        // not fatal, and the species survives.
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn duplicate_spellings_keep_the_stronger_claim() {
        let p = snorlax(
            r#"{"species":{"snorlax":{"moves":{"Body Slam":0.9,"bodyslam":0.2}}}}"#,
        );
        assert_eq!(
            p.species("snorlax").unwrap().moves,
            [("bodyslam".to_string(), 0.9)]
        );
    }

    #[test]
    fn overlay_is_per_species_and_wholesale_within_one() {
        let mut base = snorlax(
            r#"{"species":{"snorlax":{"moves":{"bodyslam":0.8,"curse":0.6}},
                           "machamp":{"moves":{"crosschop":0.9}}}}"#,
        );
        let owner = snorlax(r#"{"species":{"snorlax":{"moves":{"lovelykiss":1.0}}}}"#);
        base.overlay(owner);
        // Named species: replaced outright, not merged move by move.
        assert_eq!(
            base.species("snorlax").unwrap().moves,
            [("lovelykiss".to_string(), 1.0)]
        );
        // Unnamed species: untouched.
        assert_eq!(
            base.species("machamp").unwrap().moves,
            [("crosschop".to_string(), 0.9)]
        );
    }

    #[test]
    fn a_missing_file_is_the_empty_table_not_an_error() {
        let p = BeliefPrior::load(Path::new("/nonexistent/belief-prior.json"));
        assert!(p.is_empty());
        assert!(p.warnings().iter().any(|w| w.contains("cannot read")));
        let none = BeliefPrior::load_conventional(Path::new("/nonexistent"));
        assert!(none.is_empty());
        assert!(none.warnings().is_empty(), "the default absence is not a warning");
    }

    #[test]
    fn the_shipped_reference_table_parses_clean() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root.join("data/belief-prior-v0.sample.json");
        let p = BeliefPrior::load(&path);
        assert!(!p.is_empty(), "sample table is empty: {:?}", p.warnings());
        assert_eq!(p.skipped(), 0, "sample table warned: {:?}", p.warnings());
        // Complete-set provenance: the editing invariant holds.
        assert!(
            (p.mean_move_sum() - 4.0).abs() < 0.01,
            "mean move-probability sum {} (complete sets => 4.0)",
            p.mean_move_sum()
        );
    }

    #[test]
    fn the_conventional_path_is_absent_so_the_default_stays_unchanged() {
        // Shipping a table AT `data/belief-prior-v0.json` would silently turn
        // M18 on for every hidden-team game. The sample deliberately sits
        // beside it; activating the prior is the owner's explicit act.
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        assert!(
            !root.join(DEFAULT_PATH).exists(),
            "{DEFAULT_PATH} is tracked; the no-prior default is no longer the default"
        );
        assert!(BeliefPrior::load_conventional(&root).is_empty());
    }
}
