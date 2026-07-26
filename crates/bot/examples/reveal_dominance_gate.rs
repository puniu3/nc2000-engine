//! M18 work item 4 — the class-A certification gate.
//!
//! The design (`docs/community-belief-prior-design.md`) accepts hand-edited,
//! unreviewed community data on one argument: *certified code dominates the
//! data*, so the worst a wrong prior can do is class-B — lose games against
//! sets it mispredicts — and it can never reach the class-A surface, a
//! single-decision error a competent observer sharing the same information
//! horizon can point to. That argument is only worth as much as the list of
//! structural invariants behind it, and this replays the 570-battle spectator
//! corpus to check them in batch:
//!
//! **(a) Reveal-dominance.** No imputed slot ever contradicts a revealed
//! fact. A move the opponent was publicly seen using is in every imputed set;
//! a publicly known item is the imputed item; the imputed mon is the species,
//! level and gender team preview announced. Losing to a landmine is class B;
//! *re-stepping the same landmine after the reveal* is the class-A bug, and
//! this is the invariant that forbids it.
//!
//! **(b) The `%`-HP path is the fixed one.** Every announced foe HP
//! re-announces to the number the stream actually printed. The 2026-07-21 bug
//! this pins assumed a /48 pixel bar on a /100 HP-Percentage-Mod stream,
//! inflating every foe HP by ~2.08x, and the bot then refused kills it had.
//!
//! Plus the two structural guards the sampler must not break: at most four
//! move slots, and never zero (an empty set is implicit Struggle, which reads
//! as a free win to the search).
//!
//! Both are checked on the synthesized decision battle AND on `--dets`
//! independent determinizations of it, because with a prior loaded the
//! unrevealed slots are redrawn per determinization — the determinizations
//! are what the search actually sees, and they are the thing item 2 changed.
//!
//! **Satisfiability first.** A reveal channel can record more than four
//! distinct moves for one mon — the live `Observer` counts the plain `|move|`
//! *release* of a two-turn move that Metronome charged, because that line
//! carries no `[from]` (`corpus.rs::collect_set_evidence` fixes exactly this
//! for the offline evidence path, `-prepare`-anchored, citing battle 215).
//! When that happens the invariant is not violated, it is *unsatisfiable*: a
//! set has four slots, so one of the five "reveals" is false and no
//! imputation can honour them all. Failing there would say nothing about the
//! imputation, so such a mon is counted and named under `over_revealed` and
//! its move assertions are skipped. The defect is upstream of everything this
//! gate certifies, and it is reported, not hidden.
//!
//! Ship bar: zero violations. Exits 1 otherwise, naming coordinates.
//!
//!   cargo run --release -p nc2000-bot --example reveal_dominance_gate -- \
//!     --corpus tmp/corpus-spectator --battles 0-569 --dets 8 \
//!     --prior data/belief-prior-v0.sample.json
//!   ... --no-prior      # the shipped default, as a control

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use nc2000_bot::corpus::{
    corpus_files, load_battle, load_sources, reconstruct_context_with_prior, SetSources,
};
use nc2000_bot::import::{announce_hp, impute_hp, ProtocolTracker};
use nc2000_bot::observe::move_matches;
use nc2000_bot::preview::MetaPool;
use nc2000_bot::prior::BeliefPrior;
use nc2000_bot::rng::SplitMix64;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, Pokemon};

// ------------------------------------------------------------------ tallies

#[derive(Default, Clone)]
struct Tally {
    battles: usize,
    decisions: usize,
    reconstruct_skips: usize,
    /// Decisions whose belief fell through to synthesis (the M18 surface).
    fallback: usize,
    /// Decisions where the prior governed at least one roster slot.
    prior_active: usize,
    /// Roster slots the prior governed, summed over decisions.
    governed_slots: usize,
    /// Per-invariant assertion counts (denominators for the zero above).
    n_move: usize,
    n_item: usize,
    n_identity: usize,
    n_hp: usize,
    n_slots: usize,
    /// Announced buckets with no reachable HP (maxhp < den): the round trip
    /// is not defined there, so they are counted out rather than asserted.
    hp_unreachable: usize,
    /// Mons the reveal channel credits with more than four distinct moves,
    /// which makes reveal-dominance unsatisfiable for them (see the module
    /// doc). Named, not asserted.
    over_revealed: usize,
    over_reveal_sites: std::collections::BTreeSet<String>,
    /// Determinizations whose imputed maxhp moved the announced percentage.
    /// A diagnostic, not a violation: `determinize` is documented to keep the
    /// public HP *amount* while imputing a candidate's max HP.
    det_hp_drift: usize,
    violations: Vec<String>,
}

impl Tally {
    fn merge(&mut self, o: Tally) {
        self.battles += o.battles;
        self.decisions += o.decisions;
        self.reconstruct_skips += o.reconstruct_skips;
        self.fallback += o.fallback;
        self.prior_active += o.prior_active;
        self.governed_slots += o.governed_slots;
        self.n_move += o.n_move;
        self.n_item += o.n_item;
        self.n_identity += o.n_identity;
        self.n_hp += o.n_hp;
        self.n_slots += o.n_slots;
        self.hp_unreachable += o.hp_unreachable;
        self.over_revealed += o.over_revealed;
        self.over_reveal_sites.extend(o.over_reveal_sites);
        self.det_hp_drift += o.det_hp_drift;
        self.violations.extend(o.violations);
    }

    fn fail(&mut self, what: String) {
        if self.violations.len() < 200 {
            self.violations.push(what);
        }
    }
}

/// Where a violation was found, so a failure is reproducible rather than a
/// bare count.
struct Site<'a> {
    battle: usize,
    turn: u16,
    side: usize,
    slot: usize,
    stage: &'a str,
}

impl std::fmt::Display for Site<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "battle {} turn {} p{} foe-slot {} [{}]",
            self.battle,
            self.turn,
            self.side + 1,
            self.slot,
            self.stage
        )
    }
}

// ------------------------------------------------------------ the invariants

/// (a) + the structural guards, over one imputed opponent roster.
#[allow(clippy::too_many_arguments)]
fn check_imputation(
    dex: &Dex,
    tally: &mut Tally,
    battle: &Battle,
    obs: &nc2000_bot::observe::Observer,
    battle_idx: usize,
    turn: u16,
    side: usize,
    stage: &str,
) {
    let opp = 1 - side;
    for (slot, mo) in obs.mons().iter().enumerate() {
        let Some(p) = battle.sides[opp].roster.get(slot) else {
            tally.fail(format!(
                "{}: observer knows a mon the roster does not",
                Site { battle: battle_idx, turn, side, slot, stage }
            ));
            continue;
        };
        let site = Site { battle: battle_idx, turn, side, slot, stage };

        // ---- identity: species / level / gender are preview-public
        tally.n_identity += 1;
        if p.base_species != mo.species || p.level != mo.level || p.gender != mo.gender {
            tally.fail(format!(
                "{site}: imputed {} L{} {:?}, preview announced {} L{} {:?}",
                dex.species.key(p.base_species),
                p.level,
                p.gender,
                dex.species.key(mo.species),
                mo.level,
                mo.gender
            ));
        }

        // ---- the <=4 clamp, and never the empty set
        tally.n_slots += 1;
        if p.base_move_slots.is_empty() || p.base_move_slots.len() > 4 {
            tally.fail(format!(
                "{site}: {} imputed move slots",
                p.base_move_slots.len()
            ));
        }

        // ---- reveal-dominance over the item
        tally.n_item += 1;
        let expected = match (mo.item.current, mo.item.original) {
            (Some(known), _) => Some(known),
            (None, Some(orig)) => Some(orig),
            (None, None) => None,
        };
        if let Some(want) = expected {
            if p.item != want {
                tally.fail(format!(
                    "{site}: item known to be {:?}, imputed {:?}",
                    want.map(|i| dex.items.key(i)),
                    p.item.map(|i| dex.items.key(i))
                ));
            }
        }

        // ---- reveal-dominance over moves, when the reveals are satisfiable
        let mut distinct: Vec<nc2000_engine::dex::MoveId> = Vec::new();
        for &m in &mo.revealed_moves {
            if !distinct.iter().any(|&d| move_matches(dex, d, m)) {
                distinct.push(m);
            }
        }
        if distinct.len() > 4 {
            tally.over_revealed += 1;
            tally.over_reveal_sites.insert(format!(
                "battle {} p{} foe-slot {} {}: {} distinct reveals {:?} for a 4-slot set",
                battle_idx,
                side + 1,
                slot,
                dex.species.key(mo.species),
                distinct.len(),
                distinct.iter().map(|&m| dex.moves.key(m)).collect::<Vec<_>>()
            ));
            continue;
        }
        for &m in &mo.revealed_moves {
            tally.n_move += 1;
            let present = p
                .base_move_slots
                .iter()
                .any(|s| move_matches(dex, s.id, m))
                || p.move_slots.iter().any(|s| move_matches(dex, s.id, m));
            if !present {
                tally.fail(format!(
                    "{site}: {} was revealed and is absent from the imputed set {}",
                    dex.moves.key(m),
                    slot_names(dex, p)
                ));
            }
        }

    }
}

fn slot_names(dex: &Dex, p: &Pokemon) -> String {
    let names: Vec<&str> = p.base_move_slots.iter().map(|s| dex.moves.key(s.id)).collect();
    format!("{names:?}")
}

// ------------------------------------------------------------------ per file

#[allow(clippy::too_many_arguments)]
fn process_battle(
    dex: &Dex,
    src: &SetSources,
    pool: &MetaPool,
    path: &std::path::Path,
    battle_idx: usize,
    dets: u32,
    base_seed: u64,
    prior: Option<&Arc<BeliefPrior>>,
) -> Tally {
    let mut tally = Tally { battles: 1, ..Default::default() };
    let cb = load_battle(path);

    for (di, d) in cb.decisions.iter().enumerate() {
        let seed = base_seed
            ^ (battle_idx as u64).wrapping_mul(0x9E37_79B9_7F4A)
            ^ (di as u64).wrapping_mul(0xBF58_476D)
            ^ d.side as u64;
        let Some(rec) = reconstruct_context_with_prior(
            dex,
            src,
            pool.clone(),
            &cb.lines,
            &cb.evidence,
            d,
            seed,
            prior.cloned(),
        ) else {
            tally.reconstruct_skips += 1;
            continue;
        };
        let agent = rec.agent;
        let (Some(battle), Some(belief), Some(obs)) =
            (agent.battle(), agent.belief(), agent.observer())
        else {
            tally.reconstruct_skips += 1;
            continue;
        };
        tally.decisions += 1;
        if belief.is_fallback() {
            tally.fallback += 1;
        }
        let governed = belief.prior_governed().iter().filter(|&&g| g).count();
        tally.governed_slots += governed;
        if governed > 0 {
            tally.prior_active += 1;
        }

        // ---- (a) on the synthesized decision battle
        check_imputation(dex, &mut tally, battle, obs, battle_idx, d.turn, d.side, "synth");

        // ---- (b) the %-HP path, against an independent tracker pass. This
        // re-reads the protocol rather than trusting the reconstruction's own
        // bookkeeping: a gate that shares the code under test proves nothing.
        let mut tr = ProtocolTracker::new(d.side);
        for ln in &cb.lines[..=d.cut] {
            tr.push_line(dex, ln);
        }
        let (foes, _) = tr.snapshot(1 - d.side);
        let opp = 1 - d.side;
        for (slot, m) in foes.iter().enumerate() {
            let Some(p) = battle.sides[opp].roster.get(slot) else { continue };
            let site = Site {
                battle: battle_idx,
                turn: d.turn,
                side: d.side,
                slot,
                stage: "hp",
            };
            if m.fainted {
                tally.n_hp += 1;
                if p.hp != 0 || !p.fainted {
                    tally.fail(format!("{site}: announced fainted, imputed hp {}", p.hp));
                }
                continue;
            }
            if p.maxhp < m.hp_den {
                // Fewer HP than announcement buckets: most percentages have
                // no HP amount at all, so the round trip is undefined.
                tally.hp_unreachable += 1;
                continue;
            }
            tally.n_hp += 1;
            let want = impute_hp(m.hp_num, m.hp_den, p.maxhp);
            if p.hp != want {
                tally.fail(format!(
                    "{site}: hp {} but the certified imputation of {}/{} over max {} is {want}",
                    p.hp, m.hp_num, m.hp_den, p.maxhp
                ));
                continue;
            }
            let back = announce_hp(p.hp, p.maxhp, m.hp_den);
            if back != m.hp_num {
                tally.fail(format!(
                    "{site}: stream announced {}/{}, imputed hp {} of max {} re-announces as {back}/{}",
                    m.hp_num, m.hp_den, p.hp, p.maxhp, m.hp_den
                ));
            }
        }

        // ---- (a) again on the determinizations the search actually sees
        let mut rng = SplitMix64::new(seed ^ 0xD1B5_4A32_D192_ED03);
        for k in 0..dets {
            let det = belief.determinize(dex, battle, obs, &mut rng);
            check_imputation(
                dex,
                &mut tally,
                &det,
                obs,
                battle_idx,
                d.turn,
                d.side,
                "determinized",
            );
            if k == 0 {
                for (slot, m) in foes.iter().enumerate() {
                    let Some(p) = det.sides[opp].roster.get(slot) else { continue };
                    if m.fainted || p.maxhp < m.hp_den {
                        continue;
                    }
                    if announce_hp(p.hp, p.maxhp, m.hp_den) != m.hp_num {
                        tally.det_hp_drift += 1;
                    }
                }
            }
        }
    }
    tally
}

// --------------------------------------------------------------------- main

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let range = arg_s(&args, "--battles", "0-569");
    let dets = arg(&args, "--dets", 8) as u32;
    let seed = arg(&args, "--seed", 1) as u64;
    let threads = arg(
        &args,
        "--threads",
        std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(4),
    );
    let no_prior = args.iter().any(|a| a == "--no-prior");
    let prior_path = arg_s(&args, "--prior", "data/belief-prior-v0.sample.json");

    let (lo, hi) = {
        let mut it = range.split('-');
        let lo: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let hi: usize = it.next().unwrap_or("569").parse().unwrap_or(569);
        (lo, hi)
    };

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool = nc2000_bot::preview::load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));

    let prior = if no_prior {
        eprintln!("prior: none (the shipped default)");
        None
    } else {
        let p = BeliefPrior::load(&root.join(&prior_path));
        for w in p.warnings() {
            eprintln!("prior: {w}");
        }
        eprintln!(
            "prior: {prior_path} — {} species, mean move-probability sum {:.2}, {} entries skipped",
            p.len(),
            p.mean_move_sum(),
            p.skipped()
        );
        Some(Arc::new(p))
    };

    let files: Vec<(usize, std::path::PathBuf)> = corpus_files(&root.join(&corpus))
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i >= lo && *i <= hi)
        .collect();
    eprintln!("battles {} (index {lo}-{hi})  dets {dets}  threads {threads}", files.len());

    let total = Mutex::new(Tally::default());
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= files.len() {
                    return;
                }
                let (battle_idx, path) = &files[j];
                let t = process_battle(
                    &dex,
                    &src,
                    &pool,
                    path,
                    *battle_idx,
                    dets,
                    seed,
                    prior.as_ref(),
                );
                total.lock().unwrap().merge(t);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(50) {
                    eprintln!("  {n}/{} battles", files.len());
                }
            });
        }
    });

    let t = total.into_inner().unwrap();
    let checks = t.n_move + t.n_item + t.n_identity + t.n_hp + t.n_slots;
    println!(
        "{}",
        serde_json::json!({
            "battles": t.battles,
            "decisions": t.decisions,
            "reconstruct_skips": t.reconstruct_skips,
            "fallback_decisions": t.fallback,
            "prior_active_decisions": t.prior_active,
            "prior_governed_slots": t.governed_slots,
            "dets_per_decision": dets,
            "checks": {
                "revealed_move": t.n_move,
                "item": t.n_item,
                "identity": t.n_identity,
                "hp_reannounce": t.n_hp,
                "slot_count": t.n_slots,
                "total": checks,
            },
            "hp_buckets_unreachable": t.hp_unreachable,
            "over_revealed_mon_checks": t.over_revealed,
            "over_revealed_mons": t.over_reveal_sites.len(),
            "det_hp_pct_drift": t.det_hp_drift,
            "violations": t.violations.len(),
            "prior": if no_prior { "none".to_string() } else { prior_path.clone() },
        })
    );
    for n in &t.over_reveal_sites {
        println!("NOTE unsatisfiable reveals (upstream of this gate) — {n}");
    }
    for v in t.violations.iter().take(50) {
        println!("VIOLATION {v}");
    }
    if t.violations.is_empty() {
        eprintln!("class-A gate PASS — {checks} assertions, 0 violations");
    } else {
        eprintln!(
            "class-A gate FAIL — {} violations out of {checks} assertions",
            t.violations.len()
        );
        std::process::exit(1);
    }
}
