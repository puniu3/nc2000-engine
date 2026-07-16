//! Interactive CLI battle — human vs bot, or bot vs bot spectate.
//!
//!   cargo run --release -p nc2000-bot --example play -- human mcts:1000
//!   cargo run --release -p nc2000-bot --example play -- mcts:300 maxdamage   # spectate
//!
//! Options: --seed S (default 1) --team N --foe-team N (pool indices,
//! default random from seed) --max-turns M (default 500)
//!
//! Agent specs: human | random | maxdamage | mcts[:iters[:c]] | rm[:iters]
//!               | blind[:iters]  (M10b imperfect-info skuct)
//!
//! NOTE: search agents except `blind` read the full battle state — your
//! moves, PP, and DVs included — so those bots play with perfect
//! information. `blind` restricts itself to public observations + the meta
//! pool prior (a fixture-pool opponent falls back to a synthesized belief).

use std::io::Write as _;

use std::sync::Arc;

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{
    Agent, BlindAgent, MaxDamageAgent, MctsAgent, MctsConfig, RandomAgent, RmAgent, RmConfig,
    SplitMix64, TableSet,
};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::{Category, Dex};
use nc2000_engine::state::{Battle, Pokemon, Status};

// ------------------------------------------------------------------ input

fn read_line_or_quit() -> String {
    print!("> ");
    std::io::stdout().flush().unwrap();
    let mut s = String::new();
    if std::io::stdin().read_line(&mut s).unwrap_or(0) == 0 {
        println!("(eof)");
        std::process::exit(0);
    }
    let s = s.trim().to_string();
    if s == "q" || s == "quit" {
        println!("bye");
        std::process::exit(0);
    }
    s
}

// ------------------------------------------------------------ human agent

struct HumanAgent;

impl Agent for HumanAgent {
    fn name(&self) -> String {
        "human".into()
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if matches!(choices[0], SearchChoice::Team(_)) {
            return preview_prompt(battle, dex, side, choices);
        }
        panel(battle, dex, side);
        let labels: Vec<String> =
            choices.iter().map(|c| choice_label(battle, dex, side, *c)).collect();
        println!("Your choices:");
        for (i, l) in labels.iter().enumerate() {
            println!("  {}) {}", i + 1, l);
        }
        loop {
            let s = read_line_or_quit();
            if let Ok(n) = s.parse::<usize>() {
                if n >= 1 && n <= choices.len() {
                    return choices[n - 1];
                }
            }
            println!("pick 1-{} (q to quit)", choices.len());
        }
    }
}

fn preview_prompt(
    battle: &Battle,
    dex: &Dex,
    side: usize,
    choices: &[SearchChoice],
) -> SearchChoice {
    let you = &battle.sides[side];
    let foe = &battle.sides[1 - side];
    println!("\n=== Team preview ===");
    println!("Foe team:");
    for (i, &slot) in foe.party.iter().enumerate() {
        let p = &foe.roster[slot as usize];
        println!("  {}) {} L{}", i + 1, p.name, p.level);
    }
    println!("Your team:");
    for (i, &slot) in you.party.iter().enumerate() {
        let p = &you.roster[slot as usize];
        let moves: Vec<&str> = p
            .base_move_slots
            .iter()
            .map(|m| dex.move_static(m.id).name.as_str())
            .collect();
        let item = p
            .item
            .map(|it| format!(" @ {}", dex.items.get(it).name))
            .unwrap_or_default();
        println!("  {}) {} L{}{}  [{}]", i + 1, p.name, p.level, item, moves.join(", "));
    }
    println!("Pick 3, lead first (e.g. '1 3 5'):");
    loop {
        let s = read_line_or_quit();
        let nums: Vec<u8> = s
            .split(|c: char| !c.is_ascii_digit())
            .filter(|t| !t.is_empty())
            .filter_map(|t| t.parse().ok())
            .collect();
        if nums.len() == 3 {
            let pick = SearchChoice::Team([nums[0], nums[1], nums[2]]);
            if choices.contains(&pick) {
                return pick;
            }
        }
        println!("need 3 distinct positions 1-{}", you.party.len());
    }
}

// -------------------------------------------------------------- rendering

fn status_str(p: &Pokemon) -> String {
    match p.status {
        Status::None => String::new(),
        s => format!(" {}", s.as_str()),
    }
}

fn boosts_str(p: &Pokemon) -> String {
    const NAMES: [&str; 7] = ["Atk", "Def", "SpA", "SpD", "Spe", "Acc", "Eva"];
    let parts: Vec<String> = p
        .boosts
        .iter()
        .zip(NAMES)
        .filter(|(&b, _)| b != 0)
        .map(|(&b, n)| format!("{n}{b:+}"))
        .collect();
    if parts.is_empty() {
        String::new()
    } else {
        format!("  [{}]", parts.join(" "))
    }
}

fn hp_exact(p: &Pokemon) -> String {
    format!("{}/{}{}", p.hp.max(0), p.maxhp, status_str(p))
}

fn hp_pct(p: &Pokemon) -> String {
    if p.hp <= 0 {
        return "fnt".into();
    }
    let pct = ((p.hp as f64 / p.maxhp as f64) * 100.0).round().max(1.0);
    format!("{pct:.0}%{}", status_str(p))
}

/// State panel shown before each human decision.
fn panel(battle: &Battle, dex: &Dex, side: usize) {
    let _ = dex;
    let you = &battle.sides[side];
    let foe = &battle.sides[1 - side];
    println!("\n---------------- turn {} ----------------", battle.turn);
    if let Some(id) = battle.active_id(1 - side) {
        let p = battle.poke(id);
        println!(
            " Foe: {} L{}  {}{}   ({} left)",
            p.name,
            p.level,
            hp_pct(p),
            boosts_str(p),
            foe.pokemon_left
        );
    }
    if let Some(id) = battle.active_id(side) {
        let p = battle.poke(id);
        println!(" You: {} L{}  {}{}", p.name, p.level, hp_exact(p), boosts_str(p));
    }
    let bench: Vec<String> = you
        .party
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, &slot)| {
            let p = &you.roster[slot as usize];
            format!("{}) {} {}", i + 1, p.name, if p.fainted { "fnt".into() } else { hp_exact(p) })
        })
        .collect();
    if !bench.is_empty() {
        println!(" Bench: {}", bench.join("   "));
    }
    println!("------------------------------------------");
}

fn choice_label(battle: &Battle, dex: &Dex, side: usize, c: SearchChoice) -> String {
    match c {
        SearchChoice::Move(id) => {
            let ms = dex.move_static(id);
            let cat = match ms.category {
                Category::Physical => "Phys",
                Category::Special => "Spec",
                Category::Status => "Status",
            };
            let bp = if ms.base_power > 0 { format!(" {}BP", ms.base_power) } else { String::new() };
            let pp = battle
                .active_id(side)
                .and_then(|aid| {
                    battle.poke(aid).move_slots.iter().find(|s| s.id == id).map(|s| {
                        format!("  PP {}/{}", s.pp, s.maxpp)
                    })
                })
                .unwrap_or_default();
            format!("{}  [{} {}{}]{}", ms.name, dex.type_name(ms.move_type), cat, bp, pp)
        }
        SearchChoice::Switch(pos) => {
            let s = &battle.sides[side];
            let p = &s.roster[s.party[(pos - 1) as usize] as usize];
            format!("switch -> {}  {}", p.name, hp_exact(p))
        }
        SearchChoice::Pass => "pass".into(),
        SearchChoice::Team(t) => format!("team {:?}", t),
    }
}

// ------------------------------------------------------- protocol log view

struct LogView {
    cursor: usize,
}

impl LogView {
    /// Render log lines added since the last flush. `viewer`: Some(side) =
    /// that side's knowledge (foe HP as %, own HP exact); None = spectator
    /// (real HP everywhere).
    fn flush(&mut self, battle: &Battle, viewer: Option<usize>) {
        let lines = &battle.log;
        let mut i = self.cursor;
        while i < lines.len() {
            let line = &lines[i];
            // |split|pN: next two lines are the secret (real HP) then shared
            // (/48-scaled) variants of the same event.
            if let Some(rest) = line.strip_prefix("|split|p") {
                let split_side = rest.bytes().next().map(|b| (b - b'1') as usize).unwrap_or(0);
                let use_secret = match viewer {
                    Some(v) => v == split_side,
                    None => true,
                };
                let pick = if use_secret { lines.get(i + 1) } else { lines.get(i + 2) };
                if let Some(l) = pick {
                    if let Some(txt) = render_line(l, viewer) {
                        println!("{txt}");
                    }
                }
                i += 3;
                continue;
            }
            if let Some(txt) = render_line(line, viewer) {
                println!("{txt}");
            }
            i += 1;
        }
        self.cursor = i;
    }
}

/// "p1a: Gastly" -> (side, "Gastly")
fn parse_ref(r: &str) -> (usize, &str) {
    let side = if r.starts_with("p2") { 1 } else { 0 };
    let name = r.split_once(": ").map(|(_, n)| n).unwrap_or(r);
    (side, name)
}

fn who(r: &str, viewer: Option<usize>) -> String {
    let (side, name) = parse_ref(r);
    match viewer {
        Some(v) if v == side => name.to_string(),
        Some(_) => format!("Foe {name}"),
        None => format!("P{} {name}", side + 1),
    }
}

/// "102/211 par" / "0 fnt" -> display form; % for a foe mon under a viewer.
fn fmt_hp_token(token: &str, mon_side: usize, viewer: Option<usize>) -> String {
    let (hp, status) = token.split_once(' ').unwrap_or((token, ""));
    let exact = match viewer {
        Some(v) => v == mon_side,
        None => true,
    };
    let core = if exact {
        hp.to_string()
    } else if let Some((num, den)) = hp.split_once('/') {
        match (num.parse::<f64>(), den.parse::<f64>()) {
            (Ok(n), Ok(d)) if d > 0.0 => format!("{:.0}%", (n / d * 100.0).round().max(1.0)),
            _ => hp.to_string(),
        }
    } else {
        hp.to_string() // "0" (fainted)
    };
    if status.is_empty() {
        core
    } else {
        format!("{core} {status}")
    }
}

fn strip_effect(e: &str) -> &str {
    e.strip_prefix("move: ").unwrap_or(e)
}

fn render_line(line: &str, viewer: Option<usize>) -> Option<String> {
    let parts: Vec<&str> = line.split('|').collect();
    if parts.len() < 2 || parts[1].is_empty() {
        return None;
    }
    let tag = parts[1];
    let arg = |i: usize| parts.get(i).copied().unwrap_or("");
    // trailing "[from] xxx" annotation, if any
    let from = parts
        .iter()
        .find(|p| p.starts_with("[from]"))
        .map(|p| format!("  ({})", strip_effect(p.trim_start_matches("[from] "))))
        .unwrap_or_default();
    let out = match tag {
        "turn" => format!("\n======== Turn {} ========", arg(2)),
        "move" => format!("{} used {}!", who(arg(2), viewer), arg(3)),
        "switch" | "drag" => {
            let (side, _) = parse_ref(arg(2));
            let verb = if tag == "drag" { "was dragged out" } else { "was sent out" };
            format!("{} {} ({})", who(arg(2), viewer), verb, fmt_hp_token(arg(4), side, viewer))
        }
        "-damage" | "-heal" | "-sethp" => {
            let (side, _) = parse_ref(arg(2));
            format!("  {}: {}{}", who(arg(2), viewer), fmt_hp_token(arg(3), side, viewer), from)
        }
        "faint" => format!("{} fainted!", who(arg(2), viewer)),
        "-status" => {
            let verb = match arg(3) {
                "brn" => "was burned",
                "par" => "was paralyzed",
                "slp" => "fell asleep",
                "frz" => "was frozen solid",
                "psn" => "was poisoned",
                "tox" => "was badly poisoned",
                s => return Some(format!("  {} status: {s}", who(arg(2), viewer))),
            };
            format!("  {} {verb}!{from}", who(arg(2), viewer))
        }
        "-curestatus" => format!("  {} was cured of {}!", who(arg(2), viewer), arg(3)),
        "-boost" | "-unboost" => {
            let stat = match arg(3) {
                "atk" => "Attack",
                "def" => "Defense",
                "spa" => "Sp. Atk",
                "spd" => "Sp. Def",
                "spe" => "Speed",
                "accuracy" => "accuracy",
                "evasion" => "evasion",
                s => s,
            };
            let n: i32 = arg(4).parse().unwrap_or(1);
            let dir = if tag == "-boost" { "rose" } else { "fell" };
            let adv = if n >= 2 { " sharply" } else { "" };
            format!("  {}'s {stat} {dir}{adv}!", who(arg(2), viewer))
        }
        "cant" => {
            let why = match arg(3) {
                "slp" => "is fast asleep".into(),
                "par" => "is fully paralyzed".into(),
                "frz" => "is frozen solid".into(),
                "flinch" => "flinched".into(),
                "recharge" => "must recharge".into(),
                r => format!("can't move ({})", strip_effect(r)),
            };
            format!("{} {why}!", who(arg(2), viewer))
        }
        "-crit" => "  A critical hit!".into(),
        "-supereffective" => "  It's super effective!".into(),
        "-resisted" => "  It's not very effective...".into(),
        "-immune" => format!("  It doesn't affect {}...", who(arg(2), viewer)),
        "-miss" => format!("  {}'s attack missed!", who(arg(2), viewer)),
        "-fail" => "  But it failed!".into(),
        "-nothing" => "  But nothing happened!".into(),
        "-hitcount" => format!("  Hit {} time(s)!", arg(3)),
        "-prepare" => format!("{} is preparing {}...", who(arg(2), viewer), arg(3)),
        "-mustrecharge" => format!("  {} must recharge!", who(arg(2), viewer)),
        "-start" => format!("  {}: {} started{}", who(arg(2), viewer), strip_effect(arg(3)), from),
        "-end" => format!("  {}: {} ended{}", who(arg(2), viewer), strip_effect(arg(3)), from),
        "-activate" => format!("  {}: {}", who(arg(2), viewer), strip_effect(arg(3))),
        "-singlemove" | "-singleturn" => {
            format!("  {}: {}", who(arg(2), viewer), strip_effect(arg(3)))
        }
        "-transform" => format!("  {} transformed into {}!", who(arg(2), viewer), who(arg(3), viewer)),
        "-copyboost" => format!("  {} copied {}'s stat changes!", who(arg(2), viewer), who(arg(3), viewer)),
        "-clearallboost" => "  All stat changes were eliminated!".into(),
        "-weather" => {
            if parts.iter().any(|p| *p == "[upkeep]") {
                return None; // once per turn is enough; the start line suffices
            }
            match arg(2) {
                "none" => "  The weather cleared.".into(),
                w => format!("  Weather: {w}"),
            }
        }
        "-sidestart" => format!("  {}: {} started", arg(2), strip_effect(arg(3))),
        "-sideend" => format!("  {}: {} ended", arg(2), strip_effect(arg(3))),
        "-item" => format!("  {} holds {}{}", who(arg(2), viewer), arg(3), from),
        "-enditem" => format!("  {} used up its {}{}", who(arg(2), viewer), arg(3), from),
        "-message" => format!("  {}", arg(2)),
        "-hint" => format!("  ({})", arg(2)),
        "win" => format!("\n******** {} wins! ********", arg(2)),
        "tie" => "\n******** Tie ********".into(),
        // init/noise lines
        "player" | "teamsize" | "gen" | "gametype" | "tier" | "rule" | "start" | "clearpoke"
        | "poke" | "teampreview" | "upkeep" | "t:" | "rated" | "-anim" => return None,
        _ => format!("  . {line}"), // unknown: keep visible, semi-raw
    };
    Some(out)
}

// ------------------------------------------------------------------ main

#[derive(Clone)]
enum Spec {
    Human,
    Random,
    MaxDamage,
    Mcts { iterations: u32, c: f64, uniform: bool },
    Rm { iterations: u32 },
    Blind { iterations: u32 },
}

impl Spec {
    fn parse(s: &str) -> Spec {
        let p: Vec<&str> = s.split(':').collect();
        match p[0] {
            "human" => Spec::Human,
            "random" => Spec::Random,
            "maxdamage" => Spec::MaxDamage,
            // mcts = M6 heavy playout; mcts5 = M5 uniform-rollout baseline
            "mcts" | "mcts5" => Spec::Mcts {
                iterations: p.get(1).and_then(|v| v.parse().ok()).unwrap_or(1000),
                c: p.get(2).and_then(|v| v.parse().ok()).unwrap_or(1.0),
                uniform: p[0] == "mcts5",
            },
            // M7 state-keyed tree + RM-solved mixed root play
            "rm" => Spec::Rm {
                iterations: p.get(1).and_then(|v| v.parse().ok()).unwrap_or(1000),
            },
            // M10b imperfect-info skuct (public info + meta-pool prior only)
            "blind" => Spec::Blind {
                iterations: p.get(1).and_then(|v| v.parse().ok()).unwrap_or(1000),
            },
            other => {
                eprintln!("unknown agent: {other}");
                std::process::exit(2);
            }
        }
    }

    fn build(&self, seed: u64, dex: &Dex) -> Box<dyn Agent> {
        match self {
            Spec::Human => Box::new(HumanAgent),
            Spec::Random => Box::new(RandomAgent::new(seed)),
            Spec::MaxDamage => Box::new(MaxDamageAgent::new()),
            Spec::Mcts { iterations, c, uniform } => {
                let cfg = if *uniform {
                    MctsConfig::uniform(*iterations, *c)
                } else {
                    MctsConfig { iterations: *iterations, c: *c, ..Default::default() }
                };
                Box::new(MctsAgent::new(cfg, seed))
            }
            Spec::Rm { iterations } => Box::new(RmAgent::new(
                RmConfig { iterations: *iterations, ..Default::default() },
                seed,
            )),
            Spec::Blind { iterations } => {
                let pool = Arc::new(load_meta_pool(
                    &repo_root().join("data/meta-pool-v0/meta-pool.json"),
                ));
                let tables =
                    TableSet::load(dex, &pool, &repo_root().join("data/preview-tables-v0"));
                Box::new(BlindAgent::new(
                    RmConfig { iterations: *iterations, ..Default::default() },
                    pool,
                    Some(tables),
                    seed,
                ))
            }
        }
    }

    fn is_human(&self) -> bool {
        matches!(self, Spec::Human)
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: play <p1: human|random|maxdamage|mcts[:it[:c]]|rm[:it]> <p2: ...> [--seed S] [--team N] [--foe-team N] [--max-turns M]");
        std::process::exit(2);
    }
    let specs = [Spec::parse(&args[0]), Spec::parse(&args[1])];
    let seed: u64 = flag(&args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(1);
    let max_turns: u16 = flag(&args, "--max-turns").map(|v| v.parse().unwrap()).unwrap_or(500);

    let dex = load_dex();
    let teams = load_team_pool();
    let mut rng = SplitMix64::new(seed);
    let t1 = flag(&args, "--team")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| rng.below(teams.len()));
    let t2 = flag(&args, "--foe-team")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| rng.below(teams.len()));
    let battle_seed = rng.battle_seed();
    println!("teams: p1=#{t1} p2=#{t2}   battle seed {battle_seed}   (--seed {seed})");

    let mut battle = Battle::from_fixture(&dex, &battle_seed, &teams[t1], &teams[t2]).unwrap();
    let mut agents: Vec<Box<dyn Agent>> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| s.build(seed ^ (i as u64 + 0xB07), &dex))
        .collect();

    // viewer: the human side if exactly one human plays, else spectator view
    let humans: Vec<usize> = (0..2).filter(|&i| specs[i].is_human()).collect();
    let viewer = if humans.len() == 1 { Some(humans[0]) } else { None };

    let mut view = LogView { cursor: 0 };
    loop {
        view.flush(&battle, viewer);
        if let Some(o) = battle.outcome() {
            let _ = o;
            break;
        }
        if battle.turn > max_turns {
            println!("(turn cap {max_turns} reached — calling it a tie)");
            break;
        }
        // humans decide first, then bots think
        let mut order = [0usize, 1];
        if specs[1].is_human() && !specs[0].is_human() {
            order = [1, 0];
        }
        let mut picks = [None, None];
        for &s in &order {
            let cs = battle.legal_choices(&dex, s);
            if cs.is_empty() {
                continue;
            }
            if !specs[s].is_human() && !humans.is_empty() && cs.len() > 1 {
                println!("({} is thinking...)", agents[s].name());
            }
            picks[s] = Some(agents[s].choose(&battle, &dex, s, &cs));
        }
        battle.apply_choices(&dex, picks).unwrap();
    }
    match battle.outcome() {
        Some(Outcome::P1Win) => println!("result: P1 ({}) wins", agents[0].name()),
        Some(Outcome::P2Win) => println!("result: P2 ({}) wins", agents[1].name()),
        Some(Outcome::Tie) => println!("result: tie"),
        None => println!("result: unfinished"),
    }
}

fn load_team_pool() -> Vec<Vec<PokemonSet>> {
    let root = repo_root().join("fixtures/corpus-v1");
    let mut teams = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&root.join(corpus)) {
            let fx = Fixture::load(&path).unwrap();
            teams.push(fx.p1team);
            teams.push(fx.p2team);
        }
    }
    teams
}
