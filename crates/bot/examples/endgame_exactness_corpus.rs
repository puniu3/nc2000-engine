//! M17e step 3 — eval vs certified endgame brackets on HUMAN-GAME positions.
//!
//! Same measurement as `endgame_exactness`, but positions come from the
//! 570-battle spectator corpus (real endgames reached by real play) instead
//! of random-legal self-play — owner decision 2026-07-21: similar positions
//! recur in live games, so anchor the eval where it will actually be asked.
//!
//! Position reconstruction is `human_agreement`'s fabrication path (tracker
//! over the protocol prefix → set fabrication from rentals/pool/learnsets →
//! synthesized full battle via ProtocolAgent::on_request), WITHOUT running
//! the search. The exact value is therefore exact FOR THE IMPUTED
//! DETERMINIZATION — the same full-info state family the eval scores, so
//! the comparison is apples-to-apples; it is not the true hidden-set game.
//! (Fabrication helpers are copied from examples/human_agreement.rs —
//! examples cannot import each other; dedup into bot::corpus when a third
//! user appears.)
//!
//! Reports certified-tight comparisons plus PROVEN bracket violations
//! (eval outside a certified interval, any width — zero playouts involved).
//!
//! Usage: endgame_exactness_corpus [--corpus DIR] [--battles LO-HI]
//!        [--hp-cap N] [--work N] [--per-battle N] [--out CSV]

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::time::Instant;

use nc2000_bot::eval::{eval01, EvalWeights};
use nc2000_bot::exact::{ExactConfig, ExactSolver};
use nc2000_bot::import::{MonSnapshot, ProtocolAgent, ProtocolTracker};
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_engine::dex::{toid, Dex, SpeciesId};
use nc2000_engine::state::{Battle, Status};

fn cfg() -> RmConfig {
    RmConfig { rule: SelRule::Ucb, c: 1.0, hp_buckets: 16, ..RmConfig::default() }
}

fn plain(key: &str) -> String {
    if key.starts_with("hiddenpower") { "hiddenpower".into() } else { key.into() }
}

// -- fabrication machinery: verbatim from examples/human_agreement.rs ------

struct SetSources {
    by_species: HashMap<SpeciesId, Vec<serde_json::Value>>,
    learnsets: HashMap<String, Vec<String>>,
    hp_dvs: HashMap<String, (i64, i64)>,
}

fn load_sources(dex: &Dex, root: &std::path::Path) -> SetSources {
    let mut by_species: HashMap<SpeciesId, Vec<serde_json::Value>> = HashMap::new();
    let mut add_sets = |sets: &[serde_json::Value]| {
        for s in sets {
            if let Some(sp) = s["species"].as_str() {
                if let Some(id) = dex.species.id(&toid(sp)) {
                    by_species.entry(id).or_default().push(s.clone());
                }
            }
        }
    };
    let rentals: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("data/community-rentals-v0/teams.json")).unwrap(),
    )
    .unwrap();
    if let Some(teams) = rentals.as_array() {
        for t in teams {
            if let Some(sets) = t["sets"].as_array() {
                add_sets(sets);
            }
        }
    }
    let pool: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap(),
    )
    .unwrap();
    if let Some(teams) = pool["teams"].as_array() {
        for t in teams {
            if let Some(sets) = t["sets"].as_array() {
                add_sets(sets);
            }
        }
    }
    let ls: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap(),
    )
    .unwrap();
    let mut learnsets = HashMap::new();
    if let Some(sp) = ls["species"].as_object() {
        for (k, v) in sp {
            let moves: Vec<String> = v["moves"]
                .as_array()
                .map(|a| a.iter().filter_map(|m| m.as_str().map(String::from)).collect())
                .unwrap_or_default();
            learnsets.insert(k.clone(), moves);
        }
    }
    let mut hp_dvs = HashMap::new();
    if let Some(o) = ls["hpDvs"].as_object() {
        for (k, v) in o {
            hp_dvs
                .insert(k.clone(), (v["atk"].as_i64().unwrap_or(15), v["def"].as_i64().unwrap_or(15)));
        }
    }
    SetSources { by_species, learnsets, hp_dvs }
}

#[derive(Clone, Debug)]
enum HumanAction {
    Move(String),
    Switch(String),
}

#[derive(Clone, Debug)]
struct Decision {
    side: usize,
    turn: u16,
    cut: usize,
    action: HumanAction,
}

fn extract_decisions(lines: &[String]) -> Vec<Decision> {
    let mut out = Vec::new();
    let mut turn_open = false;
    let mut turn = 0u16;
    let mut cut = 0usize;
    let mut fainted = [false; 2];
    let mut decided = [false; 2];
    for (i, ln) in lines.iter().enumerate() {
        let p: Vec<&str> = ln.split('|').collect();
        if p.len() < 2 {
            continue;
        }
        let side_of = |s: &str| -> usize { if s.as_bytes().get(1) == Some(&b'2') { 1 } else { 0 } };
        match p[1] {
            "turn" => {
                turn_open = true;
                turn = p[2].parse().unwrap_or(0);
                cut = i;
                fainted = [false; 2];
                decided = [false; 2];
            }
            "faint" => fainted[side_of(p[2])] = true,
            "cant" => {
                let s = side_of(p[2]);
                if turn_open && !decided[s] {
                    decided[s] = true;
                }
            }
            "switch" => {
                let s = side_of(p[2]);
                if turn_open && !decided[s] {
                    if !fainted[s] {
                        let species = p[3].split(',').next().unwrap_or("").trim();
                        out.push(Decision {
                            side: s,
                            turn,
                            cut,
                            action: HumanAction::Switch(toid(species)),
                        });
                    }
                    decided[s] = true;
                }
            }
            "move" => {
                let s = side_of(p[2]);
                let from = ln.contains("[from]");
                if turn_open && !decided[s] && !from {
                    out.push(Decision {
                        side: s,
                        turn,
                        cut,
                        action: HumanAction::Move(plain(&toid(p[3]))),
                    });
                    decided[s] = true;
                }
            }
            _ => {}
        }
    }
    out
}

fn fabricate_set(
    dex: &Dex,
    src: &SetSources,
    m: &MonSnapshot,
    eaten: bool,
) -> (serde_json::Value, &'static str) {
    let sp_key = dex.species.key(m.species).to_string();
    let sp_name = dex.species.get(m.species).name.clone();
    let revealed: Vec<String> = m.uses.iter().map(|(id, _)| plain(dex.moves.key(*id))).collect();
    let nick = if m.name.is_empty() { sp_name.clone() } else { m.name.clone() };
    let gender = match format!("{:?}", m.gender).as_str() {
        "M" => "M",
        "F" => "F",
        _ => "",
    };

    let cands = src.by_species.get(&m.species).map(|v| v.as_slice()).unwrap_or(&[]);
    let norm_moves = |set: &serde_json::Value| -> Vec<String> {
        set["moves"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| plain(&toid(s)))).collect())
            .unwrap_or_default()
    };
    let mut best: Option<(usize, bool, usize)> = None;
    for (i, c) in cands.iter().enumerate() {
        let cm = norm_moves(c);
        let overlap = revealed.iter().filter(|r| cm.contains(r)).count();
        let full = overlap == revealed.len();
        let key = (overlap, full, usize::MAX - i);
        if best.map_or(true, |(o, f, bi)| key > (o, f, usize::MAX - (usize::MAX - bi))) {
            best = Some((overlap, full, i));
        }
    }

    let mut set;
    let provenance;
    match best {
        Some((_, true, i)) if !revealed.is_empty() || !cands.is_empty() => {
            set = cands[i].clone();
            provenance = "cand-full";
        }
        Some((_, false, i)) => {
            set = cands[i].clone();
            let cm: Vec<String> = set["moves"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let mut moves: Vec<String> = Vec::new();
            for r in &revealed {
                let disp = dex
                    .moves
                    .id(r)
                    .map(|id| dex.moves.get(id).name.clone())
                    .unwrap_or_else(|| r.clone());
                moves.push(disp);
            }
            for c in cm {
                if moves.len() >= 4 {
                    break;
                }
                if !moves.iter().any(|x| plain(&toid(x)) == plain(&toid(&c))) {
                    moves.push(c);
                }
            }
            set["moves"] = serde_json::json!(moves);
            provenance = "cand-fill";
        }
        _ => {
            let legal = src.learnsets.get(&sp_key).cloned().unwrap_or_default();
            let mut moves: Vec<String> = Vec::new();
            let mut ivs = serde_json::json!({});
            for r in &revealed {
                if r == "hiddenpower" {
                    let (a, d) = *src.hp_dvs.get("ice").unwrap_or(&(15, 13));
                    ivs = serde_json::json!({"atk": a * 2, "def": d * 2});
                    moves.push("Hidden Power".into());
                } else if let Some(id) = dex.moves.id(r) {
                    moves.push(dex.moves.get(id).name.clone());
                }
            }
            let mut scored: Vec<(u16, String, String)> = legal
                .iter()
                .filter_map(|k| {
                    let id = dex.moves.id(k)?;
                    let md = dex.moves.get(id);
                    if md.category == "Status" || k.starts_with("hiddenpower") {
                        return None;
                    }
                    Some((md.base_power, md.move_type.clone(), md.name.clone()))
                })
                .collect();
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            let mut seen_types: Vec<String> = Vec::new();
            for (_, ty, name) in scored {
                if moves.len() >= 4 {
                    break;
                }
                if moves.iter().any(|x| plain(&toid(x)) == plain(&toid(&name))) {
                    continue;
                }
                if seen_types.contains(&ty) {
                    continue;
                }
                seen_types.push(ty.clone());
                moves.push(name);
            }
            set = serde_json::json!({
                "species": sp_name, "item": "Leftovers", "ability": "No Ability",
                "moves": moves, "nature": "Serious",
                "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs": ivs,
            });
            provenance = "learnset-pad";
        }
    }
    set["name"] = serde_json::json!(nick);
    set["level"] = serde_json::json!(m.level);
    set["gender"] = serde_json::json!(gender);
    if eaten {
        set["item"] = serde_json::json!("");
    }
    (set, provenance)
}

fn details(dex: &Dex, m: &MonSnapshot) -> String {
    let name = &dex.species.get(m.species).name;
    let mut d = format!("{}, L{}", name, m.level);
    match format!("{:?}", m.gender).as_str() {
        "M" => d.push_str(", M"),
        "F" => d.push_str(", F"),
        _ => {}
    }
    d
}

// -- position reconstruction (human_agreement's per-decision body through
// on_request; no search step) ---------------------------------------------

fn reconstruct(
    dex: &Dex,
    src: &SetSources,
    pool_path: &std::path::Path,
    lines: &[String],
    eaten: &[(usize, String)],
    d: &Decision,
    seed: u64,
) -> Option<Battle> {
    let mut tr = ProtocolTracker::new(d.side);
    for ln in &lines[..=d.cut] {
        tr.push_line(dex, ln);
    }
    let (mons, active_slot) = tr.snapshot(d.side);
    let active_slot = active_slot?;
    if mons[active_slot].choiceless {
        return None;
    }

    let mut own_sets_json = Vec::new();
    for m in &mons {
        let ate = eaten.iter().any(|(s, n)| *s == d.side && !m.name.is_empty() && *n == m.name);
        let (set, _prov) = fabricate_set(dex, src, m, ate);
        own_sets_json.push(set);
    }
    let own_sets: Vec<nc2000_engine::battle::PokemonSet> =
        serde_json::from_value(serde_json::json!(own_sets_json.clone())).ok()?;
    let maxhps: Vec<i32> = Battle::from_fixture(dex, "1,2,3,4", &own_sets, &own_sets)
        .ok()?
        .sides[0]
        .roster
        .iter()
        .map(|p| p.maxhp)
        .collect();

    let mut picked: Vec<usize> = Vec::new();
    picked.push(active_slot);
    for (i, m) in mons.iter().enumerate() {
        if i != active_slot && m.appeared {
            picked.push(i);
        }
    }
    for (i, m) in mons.iter().enumerate() {
        if picked.len() >= 3 {
            break;
        }
        if !m.appeared {
            picked.push(i);
        }
    }
    if picked.len() < 3 {
        return None;
    }
    picked.truncate(3);

    let act = &mons[active_slot];
    let act_set_moves: Vec<String> = own_sets_json[active_slot]["moves"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let encore = act.vols.iter().find(|(k, _)| k == "encore").and_then(|(_, m)| *m);
    let disable = act.vols.iter().find(|(k, _)| k == "disable").and_then(|(_, m)| *m);
    let trapped = act.vols.iter().any(|(k, _)| k == "meanlook" || k == "partiallytrapped");
    let uses_of = |key: &str| -> i32 {
        act.uses
            .iter()
            .find(|(id, _)| plain(dex.moves.key(*id)) == plain(key))
            .map(|(_, n)| *n)
            .unwrap_or(0)
    };
    let req_moves: Vec<serde_json::Value> = act_set_moves
        .iter()
        .map(|name| {
            let key = toid(name);
            let pk = plain(&key);
            let (pp, maxpp) = dex
                .moves
                .id(&key)
                .map(|id| {
                    let mx = dex.moves.get(id).pp as i32 * 8 / 5;
                    ((mx - uses_of(&key)).max(0), mx)
                })
                .unwrap_or((5, 5));
            let disabled = match (encore, disable) {
                (Some(em), _) => plain(dex.moves.key(em)) != pk,
                (None, Some(dm)) => plain(dex.moves.key(dm)) == pk,
                _ => false,
            };
            serde_json::json!({"id": pk, "move": name, "pp": pp, "maxpp": maxpp,
                "target": "normal", "disabled": disabled})
        })
        .collect();
    let req_mons: Vec<serde_json::Value> = picked
        .iter()
        .map(|&i| {
            let m = &mons[i];
            let cond = if m.fainted {
                "0 fnt".to_string()
            } else {
                let hp = ((m.hp_frac * maxhps[i] as f64).round() as i32).clamp(1, maxhps[i]);
                let st = m.status.as_str();
                if st.is_empty() || m.status == Status::Fnt {
                    format!("{}/{}", hp, maxhps[i])
                } else {
                    format!("{}/{} {}", hp, maxhps[i], st)
                }
            };
            let nick = if m.name.is_empty() {
                dex.species.get(m.species).name.clone()
            } else {
                m.name.clone()
            };
            let ident = format!("p{}: {}", d.side + 1, nick);
            serde_json::json!({"ident": ident, "details": details(dex, m),
                "condition": cond, "active": i == active_slot,
                "item": own_sets_json[i]["item"].as_str().map(toid).unwrap_or_default()})
        })
        .collect();
    let req = serde_json::json!({
        "active": [{"moves": req_moves, "trapped": trapped}],
        "side": {"name": format!("p{}", d.side + 1), "id": format!("p{}", d.side + 1),
                 "pokemon": req_mons},
        "rqid": d.cut as u64,
    })
    .to_string();

    let mut agent = ProtocolAgent::new(dex, d.side, load_meta_pool(pool_path), cfg(), seed);
    agent.set_own_team(own_sets);
    for ln in &lines[..=d.cut] {
        agent.push_line(dex, ln);
    }
    agent.on_request(dex, &req).ok()?;
    agent.battle().cloned()
}

// ------------------------------------------------------------------- main

fn alive(b: &Battle, side: usize) -> usize {
    b.sides[side].party.iter().filter(|&&s| !b.sides[side].roster[s as usize].fainted).count()
}

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

struct Row {
    battle: usize,
    side: usize,
    turn: u16,
    human: String,
    exact: f64,
    width: f64,
    horizon: u16,
    eval: f64,
    alive0: usize,
    alive1: usize,
    total_hp: u64,
    desc: String,
}

const SOLVED_W: f64 = 0.05;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let range = arg_s(&args, "--battles", "0-49");
    let hp_cap: u64 = arg_s(&args, "--hp-cap", "150").parse().unwrap();
    let work: usize = arg_s(&args, "--work", "1000000").parse().unwrap();
    let per_battle: usize = arg_s(&args, "--per-battle", "2").parse().unwrap();
    let out_path = arg_s(&args, "--out", "tmp/endgame-exactness-corpus.csv");
    let (lo, hi) = {
        let mut it = range.split('-');
        (
            it.next().unwrap_or("0").parse::<usize>().unwrap_or(0),
            it.next().unwrap_or("49").parse::<usize>().unwrap_or(49),
        )
    };

    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let weights = EvalWeights::default();

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(root.join(&corpus))
        .unwrap_or_else(|e| panic!("corpus dir {corpus}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".raw.log"))
        .collect();
    files.sort();
    let files: Vec<(usize, std::path::PathBuf)> =
        files.into_iter().enumerate().filter(|(i, _)| *i >= lo && *i <= hi).collect();
    println!(
        "corpus battles {} (index {lo}-{hi}), hp-cap {hp_cap}, work {work}, per-battle {per_battle}",
        files.len()
    );

    let mut solver = ExactSolver::new(
        &dex,
        ExactConfig { work_budget: work, ..ExactConfig::default() },
    );
    let mut seen: HashSet<u64> = HashSet::new();
    let mut rows: Vec<Row> = Vec::new();
    let mut reconstructed = 0usize;
    let mut attempted = 0usize;
    let mut aborted = 0usize;
    let t0 = Instant::now();

    for (bi, path) in &files {
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let lines: Vec<String> = text
            .lines()
            .filter(|l| {
                let t = l.split('|').nth(1).unwrap_or("");
                !matches!(t, "j" | "l" | "t:" | "init" | "title" | "")
            })
            .map(String::from)
            .collect();
        let decisions = extract_decisions(&lines);
        let mut eaten: Vec<(usize, String)> = Vec::new();
        for ln in &lines {
            if ln.contains("|-enditem|") && ln.contains("[eat]") {
                let p: Vec<&str> = ln.split('|').collect();
                if let Some(subj) = p.get(2) {
                    let side = if subj.as_bytes().get(1) == Some(&b'2') { 1 } else { 0 };
                    if let Some(nick) = subj.split(':').nth(1) {
                        eaten.push((side, nick.trim().to_string()));
                    }
                }
            }
        }

        let mut battle_attempts = 0usize;
        // walk decisions from the END: endgames live there
        for d in decisions.iter().rev() {
            if battle_attempts >= per_battle {
                break;
            }
            let Some(b) = reconstruct(&dex, &src, &pool_path, &lines, &eaten, d, 1) else {
                continue;
            };
            reconstructed += 1;
            let (a0, a1) = (alive(&b, 0), alive(&b, 1));
            let total_hp: u64 = b
                .sides
                .iter()
                .flat_map(|s| s.party.iter().map(|&sl| s.roster[sl as usize].hp as u64))
                .sum();
            if !(a0 <= 2 && a1 <= 2 && total_hp <= hp_cap) {
                continue;
            }
            if !seen.insert(b.state_key()) {
                continue;
            }
            attempted += 1;
            battle_attempts += 1;
            let runs0 = solver.stats.chance_runs;
            let ts = Instant::now();
            let solved = solver.solve(&b);
            let dt = ts.elapsed().as_secs_f64();
            let human = match &d.action {
                HumanAction::Move(k) => format!("move {k}"),
                HumanAction::Switch(sp) => format!("switch {sp}"),
            };
            let desc = {
                let name = |side: usize| {
                    let s = &b.sides[side];
                    s.party
                        .iter()
                        .filter(|&&sl| !s.roster[sl as usize].fainted)
                        .map(|&sl| {
                            let p = &s.roster[sl as usize];
                            format!("{}({}/{})", dex.species.get(p.species).name, p.hp, p.maxhp)
                        })
                        .collect::<Vec<_>>()
                        .join("+")
                };
                format!("b{bi} T{} s{} {} vs {}", d.turn, d.side, name(0), name(1))
            };
            println!(
                "  b{bi} T{} hp{total_hp} {a0}v{a1}: {} ({} runs, {dt:.0}s)",
                d.turn,
                match &solved {
                    None => "ABORT".to_string(),
                    Some(c) => format!("[{:.3},{:.3}] w{:.3} h{}", c.lo, c.hi, c.width(), c.horizon),
                },
                solver.stats.chance_runs - runs0
            );
            match solved {
                None => aborted += 1,
                Some(c) => rows.push(Row {
                    battle: *bi,
                    side: d.side,
                    turn: d.turn,
                    human,
                    exact: c.mid(),
                    width: c.width(),
                    horizon: c.horizon,
                    eval: eval01(&b, &dex, &weights),
                    alive0: a0,
                    alive1: a1,
                    total_hp,
                    desc,
                }),
            }
        }
    }

    // ---- report
    let tight: Vec<&Row> = rows.iter().filter(|r| r.width <= SOLVED_W).collect();
    println!(
        "\nreconstructed {reconstructed}, attempted {attempted}: bracketed {} (tight {}), aborted {aborted}",
        rows.len(),
        tight.len()
    );
    println!(
        "solver exact-memo {} chance-runs {} worst-gap {:.2e}; wall {:.0}s",
        solver.stats.states,
        solver.stats.chance_runs,
        solver.stats.worst_gap,
        t0.elapsed().as_secs_f64()
    );

    let mut viols: Vec<(f64, &Row)> = rows
        .iter()
        .map(|r| {
            let (lo, hi) = (r.exact - r.width / 2.0, r.exact + r.width / 2.0);
            ((r.eval - hi).max(lo - r.eval).max(0.0), r)
        })
        .filter(|(v, _)| *v > 0.02)
        .collect();
    viols.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\nproven bracket violations (>0.02): {}", viols.len());
    for (v, r) in viols.iter().take(15) {
        println!(
            "  margin {v:.3}: eval {:.3} vs [{:.3},{:.3}]  {} (human: {})",
            r.eval,
            r.exact - r.width / 2.0,
            r.exact + r.width / 2.0,
            r.desc,
            r.human
        );
    }

    if !tight.is_empty() {
        let k = tight.len() as f64;
        let bias = tight.iter().map(|r| r.eval - r.exact).sum::<f64>() / k;
        let mae = tight.iter().map(|r| (r.eval - r.exact).abs()).sum::<f64>() / k;
        println!("\ncertified-tight n {}: bias {bias:+.4} MAE {mae:.4}", tight.len());
        for r in tight.iter().take(10) {
            println!("  exact {:.3}±{:.3} eval {:.3}  {}", r.exact, r.width / 2.0, r.eval, r.desc);
        }
    }

    std::fs::create_dir_all("tmp").ok();
    let mut f = std::fs::File::create(&out_path).expect("csv");
    writeln!(f, "battle,side,turn,human,exact,width,horizon,eval,alive0,alive1,total_hp,desc")
        .unwrap();
    for r in &rows {
        writeln!(
            f,
            "{},{},{},\"{}\",{:.6},{:.6},{},{:.6},{},{},{},\"{}\"",
            r.battle,
            r.side,
            r.turn,
            r.human,
            r.exact,
            r.width,
            r.horizon,
            r.eval,
            r.alive0,
            r.alive1,
            r.total_hp,
            r.desc
        )
        .unwrap();
    }
    println!("\ncsv: {out_path}");
}
