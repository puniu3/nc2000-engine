//! Stall reachability census (M17e e-5 scheduling research; read-only
//! consumer of enumerate/corpus APIs — no solver changes).
//!
//! Question: at a heal/stall root (default: the b455 Snorlax-vs-Skarmory
//! anchor), does the reachable decision-state set per layer diverge or
//! converge, and where are the chokepoints? Concretely measured, per BFS
//! layer (one joint decision step = one layer):
//!
//!   raw      distinct `state_key128` states (turn included — states can
//!            never recur across layers, so this is the unrolled count);
//!   norm     distinct states after zeroing `Battle::turn` before hashing —
//!            the turn-quotient count. `hits` = expansions avoided because
//!            the normalized key was already expanded in an earlier layer
//!            (a lower bound on quotient merging: volatile-internal
//!            absolute-turn stamps, if any, still split);
//!   rest     states with a side at full HP and asleep (just-Rested proxy),
//!            with a structural projection census at the end (HP kept vs
//!            HP dropped) over ALL such states;
//!   pp       min/max total PP (both actives' 4 slots) in the layer;
//!   term     terminal states hit from this layer + absorbed probability
//!            mass under a uniform-over-cells policy (weak proxy, labeled).
//!
//! Layers whose expansion was cut by the run budget or frontier cap are
//! flagged PARTIAL — their counts are lower bounds.
//!
//! Usage: stall_census [--corpus DIR] [--battle N] [--side S] [--turn T]
//!        [--work N] [--leaf-cap N] [--max-depth N] [--frontier-cap N]
//!        [--no-quotient-skip] [--out CSV]

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write as _;
use std::time::Instant;

use nc2000_bot::corpus::{
    complete_active_moves_from_future, corpus_files, load_battle, load_sources, reconstruct,
};
use nc2000_bot::eval::{eval01, EvalWeights};
use nc2000_engine::battle::enumerate::enumerate_step;
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, Status, DK};

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn norm_key(b: &Battle) -> u128 {
    let mut c = b.clone();
    c.turn = 0;
    c.state_key128()
}

/// Structural projection of both actives: everything the post-Rest lattice
/// claim says should determine the state (species/level/status/clocks/
/// boosts/PP/disabled/volatile ids+ints), with HP optionally dropped.
/// Approximate on purpose (volatile EffectStates reduced to Time/Counter
/// ints) — used only relative to itself for census ratios.
fn proj_key(b: &Battle, dex: &Dex, keep_hp: bool) -> u64 {
    let mut s = String::new();
    for side in 0..2 {
        let Some(id) = b.active_id(side) else { continue };
        let p = b.poke(id);
        s.push_str(&format!("|S{side} sp{} L{}", dex.species.key(p.species), p.level));
        if keep_hp {
            s.push_str(&format!(" hp{}/{}", p.hp, p.maxhp));
        }
        s.push_str(&format!(
            " st{} t{} c{}",
            p.status.as_str(),
            p.status_state.get_int(DK::Time),
            p.status_state.get_int(DK::Counter)
        ));
        s.push_str(&format!(" b{:?}", p.boosts));
        for m in p.move_slots.iter() {
            s.push_str(&format!(" {}:{}{}", dex.moves.key(m.id), m.pp, if m.disabled { "d" } else { "" }));
        }
        let mut vols: Vec<String> = p
            .volatiles
            .iter()
            .map(|(cid, es)| {
                format!("{:?}:{}:{}", cid, es.get_int(DK::Time), es.get_int(DK::Counter))
            })
            .collect();
        vols.sort();
        s.push_str(&format!(" v[{}]", vols.join(",")));
    }
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn pp_total(b: &Battle) -> i32 {
    (0..2)
        .filter_map(|s| b.active_id(s))
        .map(|id| b.poke(id).move_slots.iter().map(|m| m.pp).sum::<i32>())
        .sum()
}

fn is_post_rest(b: &Battle) -> bool {
    (0..2).filter_map(|s| b.active_id(s)).any(|id| {
        let p = b.poke(id);
        p.hp == p.maxhp && matches!(p.status, Status::Slp)
    })
}

fn repo_root() -> std::path::PathBuf {
    if let Ok(root) = std::env::var("NC2000_REPO_ROOT") {
        return root.into();
    }
    let current = std::env::current_dir().unwrap();
    if current.join("data/gen2stadium2.json").is_file() {
        return current;
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let battle_idx: usize = arg_s(&args, "--battle", "455").parse().unwrap();
    let side: usize = arg_s(&args, "--side", "0").parse().unwrap();
    let turn: u16 = arg_s(&args, "--turn", "39").parse().unwrap();
    let work: usize = arg_s(&args, "--work", "3000000").parse().unwrap();
    let leaf_cap: usize = arg_s(&args, "--leaf-cap", "100000").parse().unwrap();
    let max_depth: usize = arg_s(&args, "--max-depth", "40").parse().unwrap();
    let frontier_cap: usize = arg_s(&args, "--frontier-cap", "30000").parse().unwrap();
    let quotient_skip = !args.iter().any(|a| a == "--no-quotient-skip");
    // --quot turn|proj: expansion-dedupe key. `proj` walks the structural-
    // projection quotient (turn + damage-bookkeeping fields dropped, HP
    // kept) — behavior-preserving ONLY when neither side carries a
    // bookkeeping observer (Counter/Mirror Coat/Bide/Flail/Reversal/Rage);
    // the harness checks and warns.
    let quot_mode = arg_s(&args, "--quot", "turn");
    let out_path = arg_s(&args, "--out", "tmp/stall-census.csv");

    let root = repo_root();
    let dex_json = std::fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap();
    let dex = Dex::from_json(&dex_json).unwrap();
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");

    let corpus = std::path::PathBuf::from(corpus);
    let corpus = if corpus.is_absolute() { corpus } else { root.join(corpus) };
    let files = corpus_files(&corpus);
    let path = &files[battle_idx];
    let cb = load_battle(path);
    let d = cb
        .decisions
        .iter()
        .find(|d| d.side == side && d.turn == turn)
        .unwrap_or_else(|| {
            eprintln!("no decision side {side} turn {turn}; available:");
            for d in &cb.decisions {
                eprintln!("  side {} turn {} {:?}", d.side, d.turn, d.action);
            }
            std::process::exit(1);
        });
    let mut b0 = reconstruct(&dex, &src, &pool_path, &cb.lines, &cb.evidence, d, 1)
        .expect("reconstruction failed");
    if args.iter().any(|arg| arg == "--oracle-future-moves") {
        complete_active_moves_from_future(&dex, &mut b0, &cb.lines);
    }

    let w = EvalWeights::default();
    if quot_mode == "proj" {
        let observers = ["counter", "mirrorcoat", "bide", "flail", "reversal", "rage"];
        for sd in 0..2 {
            if let Some(id) = b0.active_id(sd) {
                for m in b0.poke(id).move_slots.iter() {
                    if observers.contains(&dex.moves.key(m.id)) {
                        println!("WARNING: side {sd} carries {} — proj quotient NOT behavior-preserving here", dex.moves.key(m.id));
                    }
                }
            }
        }
    }
    println!("root b{battle_idx} T{turn} s{side}  eval {:.3}  pp_total {}", eval01(&b0, &dex, &w), pp_total(&b0));
    for sd in 0..2 {
        if let Some(id) = b0.active_id(sd) {
            let p = b0.poke(id);
            let moves: Vec<String> = p
                .move_slots
                .iter()
                .map(|m| format!("{}({})", dex.moves.key(m.id), m.pp))
                .collect();
            println!(
                "  side {sd}: {} {}/{} {} [{}]",
                dex.species.key(p.species),
                p.hp,
                p.maxhp,
                p.status.as_str(),
                moves.join(" ")
            );
        }
    }

    // ---- BFS ------------------------------------------------------------
    struct Node {
        b: Battle,
        mass: f64,
    }
    let mut frontier: Vec<Node> = vec![Node { b: b0.clone(), mass: 1.0 }];
    // normalized key -> depth first EXPANDED at
    let mut norm_expanded: HashMap<u128, usize> = HashMap::new();
    let mut runs_total = 0usize;
    let mut absorbed_mass = 0.0f64;
    let mut lost_mass = 0.0f64; // capped cells + dropped frontier states
    let mut rest_norm: HashMap<u128, ()> = HashMap::new();
    let mut rest_proj_hp: HashMap<u64, ()> = HashMap::new();
    let mut rest_proj_nohp: HashMap<u64, ()> = HashMap::new();
    let mut pp_census: HashMap<i32, usize> = HashMap::new(); // pp_total -> distinct norm states
    let mut pp_seen: HashMap<u128, ()> = HashMap::new();
    let t0 = Instant::now();

    let mut csv = String::from(
        "depth,raw,norm_new,quot_hits,rest,term_states,term_mass,pp_min,pp_max,runs,partial\n",
    );
    println!(
        "\n{:>5} {:>8} {:>8} {:>9} {:>6} {:>10} {:>10} {:>9} {:>10} {:>8}",
        "depth", "raw", "norm_new", "quot_hits", "rest", "term_mass", "lost_mass", "pp_range", "runs", "wall_s"
    );

    let mut depth = 0usize;
    while depth < max_depth && !frontier.is_empty() && runs_total < work {
        depth += 1;
        let mut next: HashMap<u128, Node> = HashMap::new();
        let mut raw = 0usize;
        let mut norm_new = 0usize;
        let mut quot_hits = 0usize;
        let mut rest_n = 0usize;
        let mut term_states = 0usize;
        let mut term_mass = 0.0f64;
        let (mut pp_min, mut pp_max) = (i32::MAX, i32::MIN);
        let mut partial = false;
        let layer_runs0 = runs_total;

        for node in frontier.drain(..) {
            if runs_total >= work {
                partial = true;
                lost_mass += node.mass;
                continue;
            }
            let nk = norm_key(&node.b);
            let qk = if quot_mode == "proj" { proj_key(&node.b, &dex, true) as u128 } else { nk };
            let pp = pp_total(&node.b);
            pp_min = pp_min.min(pp);
            pp_max = pp_max.max(pp);
            if pp_seen.insert(qk, ()).is_none() {
                *pp_census.entry(pp).or_insert(0) += 1;
            }
            if is_post_rest(&node.b) {
                rest_n += 1;
                rest_norm.insert(nk, ());
                rest_proj_hp.insert(proj_key(&node.b, &dex, true), ());
                rest_proj_nohp.insert(proj_key(&node.b, &dex, false), ());
            }
            if quotient_skip {
                if let Some(&d0) = norm_expanded.get(&qk) {
                    let _ = d0;
                    quot_hits += 1;
                    continue;
                }
            }
            norm_expanded.insert(qk, depth);
            norm_new += 1;

            // expand: all joint action cells, uniform cell mass
            let needs = node.b.needs_choice();
            let mut probe = node.b.clone();
            let acts = |probe: &mut Battle, side: usize, need: bool| -> Vec<Option<SearchChoice>> {
                if need {
                    probe.legal_choices(&dex, side).into_iter().map(Some).collect()
                } else {
                    vec![None]
                }
            };
            let a0 = acts(&mut probe, 0, needs[0]);
            let a1 = acts(&mut probe, 1, needs[1]);
            if a0.is_empty() || a1.is_empty() {
                lost_mass += node.mass;
                continue;
            }
            let cell_mass = node.mass / (a0.len() * a1.len()) as f64;
            for &c0 in &a0 {
                for &c1 in &a1 {
                    let Some(step) = enumerate_step(&dex, &node.b, [c0, c1], leaf_cap) else {
                        lost_mass += cell_mass;
                        continue;
                    };
                    runs_total += step.runs;
                    for l in step.leaves {
                        let m = cell_mass * l.prob;
                        if l.battle.outcome().is_some() {
                            term_states += 1;
                            term_mass += m;
                            absorbed_mass += m;
                            continue;
                        }
                        let k = if quot_mode == "proj" {
                            proj_key(&l.battle, &dex, true) as u128
                        } else {
                            l.battle.state_key128()
                        };
                        match next.get_mut(&k) {
                            Some(n) => n.mass += m,
                            None => {
                                if next.len() >= frontier_cap {
                                    partial = true;
                                    lost_mass += m;
                                } else {
                                    next.insert(k, Node { b: l.battle, mass: m });
                                }
                            }
                        }
                    }
                    if runs_total >= work {
                        partial = true;
                    }
                }
            }
        }
        raw += next.len();
        let layer_runs = runs_total - layer_runs0;
        let ppr = if pp_min > pp_max { "-".into() } else { format!("{pp_min}-{pp_max}") };
        println!(
            "{:>5} {:>8} {:>8} {:>9} {:>6} {:>10.4} {:>10.4} {:>9} {:>10} {:>8.1}{}",
            depth,
            raw,
            norm_new,
            quot_hits,
            rest_n,
            term_mass,
            lost_mass,
            ppr,
            layer_runs,
            t0.elapsed().as_secs_f64(),
            if partial { "  PARTIAL" } else { "" }
        );
        csv.push_str(&format!(
            "{depth},{raw},{norm_new},{quot_hits},{rest_n},{term_states},{term_mass:.6},{},{},{layer_runs},{}\n",
            if pp_min > pp_max { -1 } else { pp_min },
            if pp_min > pp_max { -1 } else { pp_max },
            partial as u8
        ));
        frontier = next.into_values().collect();
        if partial && runs_total >= work {
            break;
        }
    }

    println!("\ntotal runs {runs_total}  wall {:.1}s", t0.elapsed().as_secs_f64());
    println!("absorbed (uniform-policy) mass {absorbed_mass:.4}, lost/truncated mass {lost_mass:.4}");
    println!("quotient graph: {} expanded normalized states", norm_expanded.len());
    println!(
        "post-Rest census: {} normalized states, {} proj(with-HP), {} proj(no-HP)  [compression x{:.1}]",
        rest_norm.len(),
        rest_proj_hp.len(),
        rest_proj_nohp.len(),
        rest_norm.len() as f64 / rest_proj_nohp.len().max(1) as f64
    );
    let mut pps: Vec<(i32, usize)> = pp_census.into_iter().collect();
    pps.sort();
    println!("\nPP-layer census (distinct normalized states per total-PP):");
    for (pp, n) in pps {
        println!("  pp {pp:>4}: {n}");
    }

    std::fs::create_dir_all("tmp").ok();
    let mut f = std::fs::File::create(&out_path).expect("csv");
    f.write_all(csv.as_bytes()).unwrap();
    println!("\ncsv: {out_path}");
}
