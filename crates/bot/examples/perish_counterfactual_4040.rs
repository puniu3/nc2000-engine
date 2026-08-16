//! What the eight auto-failing turns of battle-4040 actually cost.
//!
//! Rolls the turn-24 position forward under fixed policies and counts wins.
//! Side 1 (the bot) is down to Gengar; side 0 has Heracross 79%, Skarmory
//! 35% and a frozen Snorlax at 11%. The player's complaint was that Ice
//! Punch crit-spam had a chance ("突破の目") that the Perish Songs threw
//! away, so both halves are measured:
//!
//!   * `always-ice-punch`   — the line the player wanted.
//!   * `as-played`          — the recorded sequence, Perish Songs and all.
//!   * `--rest-at 0`        — the same run with the opponent forbidden to
//!                            Rest, which separates "the Rest loop is the
//!                            wall" from "the damage never got there".
//!
//! The opponent is scripted (it never switched in the real game either):
//! Rest at or below `--rest-at` when awake and Rest has PP, Sleep Talk when
//! asleep, Megahorn otherwise. `--foe max-damage` swaps in the repo's
//! MaxDamage baseline as a second reading.
//!
//! Usage:
//!   cargo run --release -p nc2000-bot --example perish_counterfactual_4040 -- \
//!     [--turn 24] [--trials 20000] [--rest-at 0.55] [--foe script|max-damage]

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::agent::{Agent, MaxDamageAgent};
use nc2000_bot::corpus::{load_battle, load_sources, reconstruct};
use nc2000_bot::runner::{play_game, GameResult};
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::prng::{BattleRng, Prng};
use nc2000_engine::state::{Battle, Status};

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// Gengar's side: play `script` in order, then Ice Punch forever; whenever
/// the named move is illegal or out of PP, fall through to Ice Punch, then
/// to the first legal action.
struct ScriptedGengar {
    script: Vec<&'static str>,
    at: usize,
}

impl ScriptedGengar {
    fn new(script: Vec<&'static str>) -> Self {
        ScriptedGengar { script, at: 0 }
    }
}

fn pick_move(dex: &Dex, choices: &[SearchChoice], key: &str) -> Option<SearchChoice> {
    choices.iter().copied().find(|c| match c {
        SearchChoice::Move(id) => dex.moves.key(*id) == key,
        _ => false,
    })
}

impl Agent for ScriptedGengar {
    fn name(&self) -> String {
        "scripted-gengar".into()
    }

    fn choose(
        &mut self,
        _b: &Battle,
        dex: &Dex,
        _side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let want = self.script.get(self.at).copied().unwrap_or("icepunch");
        self.at += 1;
        pick_move(dex, choices, want)
            .or_else(|| pick_move(dex, choices, "icepunch"))
            .unwrap_or(choices[0])
    }
}

/// The opponent as it actually played: never switches, Rests when hurt,
/// Sleep Talks while asleep, Megahorns otherwise.
struct ScriptedFoe {
    rest_at: f64,
}

impl Agent for ScriptedFoe {
    fn name(&self) -> String {
        "scripted-foe".into()
    }

    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let Some(id) = b.active_id(side) else { return choices[0] };
        let p = b.poke(id);
        if p.status == Status::Slp {
            if let Some(c) = pick_move(dex, choices, "sleeptalk") {
                return c;
            }
        }
        let frac = p.hp as f64 / p.maxhp as f64;
        if frac <= self.rest_at {
            if let Some(c) = pick_move(dex, choices, "rest") {
                return c;
            }
        }
        for key in ["megahorn", "doubleedge", "drillpeck", "toxic"] {
            if let Some(c) = pick_move(dex, choices, key) {
                return c;
            }
        }
        choices[0]
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let turn: u16 = arg(&args, "--turn").unwrap_or("24").parse().unwrap();
    let trials: u64 = arg(&args, "--trials").unwrap_or("20000").parse().unwrap();
    let rest_at: f64 = arg(&args, "--rest-at").unwrap_or("0.55").parse().unwrap();
    let foe_kind = arg(&args, "--foe").unwrap_or("script").to_string();

    let dex = load_dex();
    let root = repo_root();
    let src = load_sources(&dex, &root);
    let pool = root.join("data/meta-pool-v0/meta-pool.json");
    let log = root.join("tmp/corpus-4040/battle-4040.raw.log");
    let battle = load_battle(&log);
    let d = battle
        .decisions
        .iter()
        .find(|d| d.side == 1 && d.turn == turn)
        .expect("no p2 decision at that turn");
    let base = reconstruct(&dex, &src, &pool, &battle.lines, &battle.evidence, d, 1)
        .expect("reconstruction");

    println!("turn {turn}: side-1 pokemon_left = {}", base.sides[1].pokemon_left);
    if let Some(id) = base.active_id(1) {
        let pp: Vec<String> = base
            .poke(id)
            .move_slots
            .iter()
            .map(|m| format!("{}({})", dex.moves.key(m.id), m.pp))
            .collect();
        println!("  active: {}", pp.join(" "));
    }
    for s in 0..2 {
        let ids: Vec<String> = base.sides[s]
            .party
            .iter()
            .map(|&slot| {
                let p = &base.sides[s].roster[slot as usize];
                format!(
                    "{} {:.0}%{}",
                    dex.species.key(p.species),
                    100.0 * p.hp as f64 / p.maxhp as f64,
                    if p.status == Status::None { String::new() } else { format!(" {:?}", p.status) }
                )
            })
            .collect();
        println!("  side {s}: {}", ids.join(", "));
    }
    println!("foe policy: {foe_kind} (rest at <= {:.0}% HP)\n", rest_at * 100.0);

    // The recorded line from turn 24 on, exactly as submitted.
    const AS_PLAYED: [&str; 22] = [
        "perishsong", "perishsong", "perishsong", "destinybond", "icepunch", "icepunch",
        "icepunch", "icepunch", "perishsong", "icepunch", "icepunch", "icepunch", "icepunch",
        "icepunch", "icepunch", "icepunch", "icepunch", "icepunch", "perishsong", "perishsong",
        "icepunch", "perishsong",
    ];

    // AS_PLAYED is indexed from turn 24, so a later start point replays the
    // tail from that turn — not the whole sequence over again.
    let from = (turn.saturating_sub(24) as usize).min(AS_PLAYED.len());
    let arms: Vec<(&str, Vec<&'static str>)> =
        vec![("always-ice-punch", vec![]), ("as-played", AS_PLAYED[from..].to_vec())];

    for (name, script) in arms {
        let mut wins = 0u64;
        let mut losses = 0u64;
        let mut ties = 0u64;
        let mut turns_survived = 0u64;
        let mut froze = 0u64;
        let mut crit = 0u64;
        for t in 0..trials {
            let mut b = base.clone();
            b.set_log_enabled(true);
            b.prng = BattleRng::seeded(
                Prng::from_seed_str(&format!("{},{},{},{}", t, t + 7, t + 13, t + 29))
                    .expect("seed"),
            );
            let start = b.turn;
            let mut me: Box<dyn Agent> = Box::new(ScriptedGengar::new(script.clone()));
            let mut foe: Box<dyn Agent> = if foe_kind == "max-damage" {
                Box::new(MaxDamageAgent::new())
            } else {
                Box::new(ScriptedFoe { rest_at })
            };
            // Inline play loop instead of `play_game`: the engine clears
            // `log` every turn, so the diagnostics have to drain it as it goes.
            let mut agents: [&mut dyn Agent; 2] = [foe.as_mut(), me.as_mut()];
            let mut saw_frz = false;
            let mut saw_crit = false;
            let r = loop {
                if let Some(o) = b.outcome() {
                    break Ok::<GameResult, ()>(GameResult::Outcome(o));
                }
                if b.turn > 1000 {
                    break Ok(GameResult::TurnCapped);
                }
                let mut picks = [None, None];
                for s in 0..2 {
                    let cs = b.legal_choices(&dex, s);
                    if !cs.is_empty() {
                        picks[s] = Some(agents[s].choose(&b, &dex, s, &cs));
                    }
                }
                b.apply_choices(&dex, picks).expect("apply");
                for l in &b.log {
                    if l.starts_with("|-status|p1a") && l.ends_with("frz") {
                        saw_frz = true;
                    }
                    if l.starts_with("|-crit|p1a") {
                        saw_crit = true;
                    }
                }
            };
            if saw_frz {
                froze += 1;
            }
            if saw_crit {
                crit += 1;
            }
            turns_survived += (b.turn.saturating_sub(start)) as u64;
            if b.log.iter().any(|l| l.starts_with("-status|p1a") && l.ends_with("frz")) {
                froze += 1;
            }
            if b.log.iter().any(|l| l.starts_with("-crit|p1a")) {
                crit += 1;
            }
            match r {
                Ok(GameResult::Outcome(Outcome::P2Win)) => wins += 1,
                Ok(GameResult::Outcome(Outcome::P1Win)) => losses += 1,
                _ => ties += 1,
            }
            let _ = &play_game;
        }
        let p = wins as f64 / trials as f64;
        let se = (p * (1.0 - p) / trials as f64).sqrt();
        println!(
            "{name:<18} win {wins:>6}/{trials}  = {:.4} ± {:.4}   (loss {losses}, tie/cap {ties}, mean turns {:.1})",
            p,
            1.96 * se,
            turns_survived as f64 / trials as f64
        );
        println!(
            "                   games with a NEW p1 freeze: {:.3}   games with >=1 crit: {:.3}",
            froze as f64 / trials as f64,
            crit as f64 / trials as f64
        );
    }
}
