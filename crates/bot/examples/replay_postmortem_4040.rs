//! Postmortem for ladder battle-4040 (EnHayaku vs puniu3, 2026-08-16): with
//! Gengar as its LAST mon the bot selected Perish Song on turns 24/25/26/32/
//! 42/43/45 and Destiny Bond on turn 27 — eight moves the Stadium 2 rule
//! fails outright (`moveexec.rs` destinybond/perishsong onPrepareHit: the
//! move returns False when `pokemon_left == 1`). Its only live option was
//! Ice Punch.
//!
//! Reproduces every p2 decision of the game through the real corpus
//! reconstruction + `ProtocolAgent` pipeline at the shipped ladder budget,
//! and prints, per decision:
//!   * the mask's verdict (`smmcts::dominated_actions`) — does the bot even
//!     know the move is dead?
//!   * the full root policy: visits, mean value, dominated flag,
//!   * what `best()` would submit, versus what was actually played.
//!
//! Usage:
//!   cargo run --release -p nc2000-bot --example replay_postmortem_4040 -- \
//!     [--log tmp/corpus-4040/battle-4040.raw.log] [--iters 30000] \
//!     [--seed 1] [--seeds N] [--from-turn 1] [--side 1]

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::corpus::{
    load_battle, load_sources, reconstruct_context_with_pool, HumanAction,
};
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_bot::smmcts::dominated_actions;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn describe(dex: &Dex, b: &Battle, side: usize) -> String {
    let Some(id) = b.active_id(side) else { return "-".into() };
    let p = b.poke(id);
    let hp = 100.0 * p.hp as f64 / p.maxhp as f64;
    let st = format!("{:?}", p.status);
    let moves: Vec<String> = p
        .move_slots
        .iter()
        .map(|s| format!("{}({})", dex.moves.key(s.id), s.pp))
        .collect();
    format!(
        "{} {:.0}% {} left={} [{}]",
        dex.species.key(p.species),
        hp,
        if st == "None" { String::new() } else { st },
        b.sides[side].pokemon_left,
        moves.join(" ")
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let log = arg(&args, "--log")
        .unwrap_or("tmp/corpus-4040/battle-4040.raw.log")
        .to_string();
    let iters: u32 = arg(&args, "--iters").unwrap_or("30000").parse().unwrap();
    let seed0: u64 = arg(&args, "--seed").unwrap_or("1").parse().unwrap();
    let seeds: u64 = arg(&args, "--seeds").unwrap_or("1").parse().unwrap();
    let from_turn: u16 = arg(&args, "--from-turn").unwrap_or("1").parse().unwrap();
    let only: Vec<u16> = arg(&args, "--turns")
        .map(|s| s.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_default();
    let terse = args.iter().any(|a| a == "--terse");
    // Open team sheet (`ps-client --mode open`, and the shipped product
    // policy): re-pose the same reconstructed position to a fresh agent whose
    // belief is PINNED to the opponent's true sets.
    let open: Option<Vec<nc2000_engine::battle::PokemonSet>> = arg(&args, "--open").map(|p| {
        serde_json::from_str(&std::fs::read_to_string(p).expect("open team file")).expect("sets")
    });
    let side: usize = arg(&args, "--side").unwrap_or("1").parse().unwrap();

    let dex = load_dex();
    let root = repo_root();
    let src = load_sources(&dex, &root);
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let battle = load_battle(&std::path::Path::new(&log).to_path_buf());

    println!("log: {log}   decisions: {}", battle.decisions.len());
    println!("budget: {iters} iters x {seeds} seed(s) from {seed0}\n");

    for d in battle
        .decisions
        .iter()
        .filter(|d| d.side == side && d.turn >= from_turn)
        .filter(|d| only.is_empty() || only.contains(&d.turn))
    {
        let played = match &d.action {
            HumanAction::Move(m) => format!("move {m}"),
            HumanAction::Switch(s) => format!("switch {s}"),
        };
        let mut header_done = false;
        for k in 0..seeds {
            let seed = seed0 + k;
            let Some(rec) =
                reconstruct_context_with_pool(&dex, &src, pool.clone(), &battle.lines, &battle.evidence, d, seed)
            else {
                println!("turn {:>2}: reconstruction refused", d.turn);
                break;
            };
            let mut agent = rec.agent;
            if let Some(sets) = &open {
                let spec = agent.to_position_spec(&dex).expect("position spec");
                let cfg = RmConfig { rule: SelRule::Ucb, c: 1.0, hp_buckets: 16, ..RmConfig::default() };
                let mut pinned = ProtocolAgent::new(&dex, side, pool.clone(), cfg, seed);
                pinned.pin_opponent(sets.clone());
                match pinned.set_position(&dex, &spec) {
                    Ok(()) => agent = pinned,
                    Err(e) => {
                        println!("turn {:>2}: pinned position refused: {e}", d.turn);
                        break;
                    }
                }
            }
            let b = agent.battle().cloned().expect("battle");
            if !header_done {
                println!("--- turn {:>2}  played: {played}", d.turn);
                println!("    me : {}", describe(&dex, &b, side));
                println!("    foe: {}", describe(&dex, &b, 1 - side));
                let dom = dominated_actions(&b, &dex, side);
                if dom.is_empty() {
                    println!("    mask: (nothing refused)");
                } else {
                    for (c, why) in &dom {
                        println!("    mask: REFUSE {:?} — {why}", c);
                    }
                }
                header_done = true;
            }
            agent.step(&dex, iters).expect("search");
            let policy: serde_json::Value =
                serde_json::from_str(&agent.root_policy(&dex)).unwrap();
            let best = agent.best(&dex).unwrap_or_default();
            let rows: Vec<String> = policy["actions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    format!(
                        "{}={}({:.3})",
                        r["input"].as_str().unwrap_or("?"),
                        r["visits"].as_u64().unwrap_or(0),
                        r["mean"].as_f64().unwrap_or(0.0)
                    )
                })
                .collect();
            if terse {
                println!("turn {} iters {iters} seed {seed} best {best}", d.turn);
            } else {
                println!("    s{seed}: best={best:<22} | {}", rows.join("  "));
            }
        }
        if header_done {
            println!();
        }
    }
}
