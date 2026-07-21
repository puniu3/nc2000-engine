//! Postmortem for the 2026-07-21 sohechan-vs-puniu3 report ("the bot
//! Explodes with its final Pokemon"): turn 11, the bot's LAST mon (Gengar,
//! full HP, par) faced a full-HP Starmie with TWO unrevealed opponent picks
//! alive, and chose Explosion — an unconditional immediate loss (the user
//! faints; we are out of mons regardless of the damage; had Starmie been
//! the last foe, the Self-KO clause loses too).
//!
//! The reporter runs a fork of the pre-migration client (gen2nc2000-era
//! header, --iters 1000 default), so the leading suspect is a clone that
//! predates 694efb1 (synthesize left pokemon_left at roster size — no
//! terminal was reachable in search, the battle-3634 mechanism). This
//! harness replays the decision through the CURRENT pipeline at 1000 (the
//! fork's budget) and 10000 (shipped default) iterations to establish
//! whether current master still produces the blunder.

use nc2000_bot::blind::BlindSearch;
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;

fn load_dex() -> Dex {
    conformance::load_dex()
}

fn cfg() -> RmConfig {
    RmConfig { rule: SelRule::Ucb, c: 1.0, hp_buckets: 16, ..RmConfig::default() }
}

/// Own team in |poke| preview order. Revealed this game: Typhlosion (Sunny
/// Day / Fire Blast / Thunder Punch, Miracle Berry eaten), Gengar (Zap
/// Cannon / Explosion). Unrevealed slots filled from the 2026-07-20
/// reconstruction of the same roster (replay_analysis.rs); the decision
/// under test needs only Gengar's revealed Zap Cannon / Explosion.
fn own_team() -> Vec<PokemonSet> {
    serde_json::from_str(
        r#"[
        {"name":"Snorlax","species":"Snorlax","item":"Leftovers","ability":"No Ability",
         "moves":["Body Slam","Curse","Rest","Earthquake"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":55},
        {"name":"Marowak","species":"Marowak","item":"Thick Club","ability":"No Ability",
         "moves":["Earthquake","Rock Slide","Hidden Power Bug","Swords Dance"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
         "ivs":{"hp":30,"atk":26,"def":26,"spa":30,"spd":30,"spe":30},"gender":"M","level":50},
        {"name":"Porygon2","species":"Porygon2","item":"Mint Berry","ability":"No Ability",
         "moves":["Recover","Thunderbolt","Ice Beam","Curse"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"level":50},
        {"name":"Gengar","species":"Gengar","item":"Gold Berry","ability":"No Ability",
         "moves":["Zap Cannon","Explosion","Mean Look","Destiny Bond"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
        {"name":"Skarmory","species":"Skarmory","item":"Sharp Beak","ability":"No Ability",
         "moves":["Drill Peck","Whirlwind","Curse","Rest"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
        {"name":"Typhlosion","species":"Typhlosion","item":"Miracle Berry","ability":"No Ability",
         "moves":["Fire Blast","Thunder Punch","Dynamic Punch","Sunny Day"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":55}
    ]"#,
    )
    .unwrap()
}

/// Public lines through |turn|11 (|j|/|t:|/blank stripped).
const LOG: &str = r#"|player|p1|sohechan|1|
|player|p2|puniu3|1|
|gen|2
|tier|[Gen 2] Nintendo Cup 2000 No OHKO Stadium2 Strict
|clearpoke
|poke|p1|Snorlax, L55, M|item
|poke|p1|Marowak, L55, M|item
|poke|p1|Gengar, L50, M|item
|poke|p1|Starmie, L50|item
|poke|p1|Jolteon, L50, F|item
|poke|p1|Miltank, L50, F|item
|poke|p2|Snorlax, L55, M|item
|poke|p2|Marowak, L50, M|item
|poke|p2|Porygon2, L50|item
|poke|p2|Gengar, L50, M|item
|poke|p2|Skarmory, L50, M|item
|poke|p2|Typhlosion, L55, M|item
|teampreview|3
|teamsize|p1|3
|teamsize|p2|3
|start
|switch|p1a: Starmie|Starmie, L50|100/100
|switch|p2a: Typhlosion|Typhlosion, L55, M|100/100
|turn|1
|move|p2a: Typhlosion|Sunny Day|p2a: Typhlosion
|-weather|SunnyDay
|move|p1a: Starmie|Hydro Pump|p2a: Typhlosion
|-supereffective|p2a: Typhlosion
|-damage|p2a: Typhlosion|67/100
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|2
|move|p2a: Typhlosion|Fire Blast|p1a: Starmie
|-resisted|p1a: Starmie
|-damage|p1a: Starmie|53/100
|move|p1a: Starmie|Thunder Wave|p2a: Typhlosion
|-status|p2a: Typhlosion|par
|-weather|SunnyDay|[upkeep]
|-enditem|p2a: Typhlosion|Miracle Berry|[eat]
|-curestatus|p2a: Typhlosion|par|[msg]
|upkeep
|turn|3
|move|p2a: Typhlosion|Thunder Punch|p1a: Starmie
|-supereffective|p1a: Starmie
|-damage|p1a: Starmie|5/100
|move|p1a: Starmie|Thunder Wave|p2a: Typhlosion
|-status|p2a: Typhlosion|par
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|4
|move|p1a: Starmie|Recover|p1a: Starmie
|-heal|p1a: Starmie|55/100
|move|p2a: Typhlosion|Fire Blast|p1a: Starmie
|-resisted|p1a: Starmie
|-damage|p1a: Starmie|8/100
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|5
|move|p1a: Starmie|Recover|p1a: Starmie
|-heal|p1a: Starmie|58/100
|move|p2a: Typhlosion|Fire Blast|p1a: Starmie
|-resisted|p1a: Starmie
|-damage|p1a: Starmie|10/100
|-weather|none
|upkeep
|turn|6
|move|p1a: Starmie|Hydro Pump|p2a: Typhlosion
|-supereffective|p2a: Typhlosion
|-damage|p2a: Typhlosion|0 fnt
|faint|p2a: Typhlosion
|switch|p2a: Marowak|Marowak, L50, M|100/100
|upkeep
|turn|7
|move|p1a: Starmie|Hydro Pump|p2a: Marowak
|-supereffective|p2a: Marowak
|-damage|p2a: Marowak|0 fnt
|faint|p2a: Marowak
|switch|p2a: Gengar|Gengar, L50, M|100/100
|upkeep
|turn|8
|move|p1a: Starmie|Thunder Wave|p2a: Gengar
|-status|p2a: Gengar|par
|move|p2a: Gengar|Zap Cannon|p1a: Starmie|[miss]
|-miss|p2a: Gengar
|upkeep
|turn|9
|move|p1a: Starmie|Recover|p1a: Starmie
|-heal|p1a: Starmie|60/100
|cant|p2a: Gengar|par
|upkeep
|turn|10
|move|p1a: Starmie|Recover|p1a: Starmie
|-heal|p1a: Starmie|100/100
|cant|p2a: Gengar|par
|upkeep
|turn|11"#;

/// The move request PS would send p2 at turn 11 (Gengar active, par, full;
/// Zap Cannon used once; berry unspent this game — Gold Berry intact).
const REQ_T11: &str = r#"{
  "active":[{"moves":[
    {"id":"zapcannon","move":"Zap Cannon","pp":7,"maxpp":8,"target":"normal","disabled":false},
    {"id":"explosion","move":"Explosion","pp":8,"maxpp":8,"target":"normal","disabled":false},
    {"id":"meanlook","move":"Mean Look","pp":8,"maxpp":8,"target":"normal","disabled":false},
    {"id":"destinybond","move":"Destiny Bond","pp":8,"maxpp":8,"target":"self","disabled":false}
  ],"trapped":false}],
  "side":{"name":"puniu3","id":"p2","pokemon":[
    {"ident":"p2: Gengar","details":"Gengar, L50, M","condition":"999/999 par","active":true,"item":"goldberry"},
    {"ident":"p2: Typhlosion","details":"Typhlosion, L55, M","condition":"0 fnt","active":false,"item":""},
    {"ident":"p2: Marowak","details":"Marowak, L50, M","condition":"0 fnt","active":false,"item":"thickclub"}
  ]},
  "rqid":21
}"#;

fn show(label: &str, seed: u64, s: &BlindSearch, dex: &Dex) {
    let acts = s.actions();
    let visits = s.visits();
    let means = s.means();
    let total: u32 = visits.iter().sum();
    let mut rows: Vec<(String, u32, f64)> = acts
        .iter()
        .zip(visits.iter().zip(means.iter()))
        .map(|(&a, (&n, &m))| (a.to_input(dex), n, m))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let best = rows.first().map(|r| r.0.clone()).unwrap_or_default();
    print!("{label} seed {seed}: best={best:<18}");
    for (input, n, m) in &rows {
        print!("  {input}={n}/{total} ({m:.3})");
    }
    println!();
}

fn main() {
    let dex = load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let lines: Vec<&str> = LOG.lines().collect();

    for iters in [1000u32, 10000] {
        let mut census: std::collections::BTreeMap<String, u32> = Default::default();
        let mut first = true;
        for seed in 1u64..=50 {
            let mut agent =
                ProtocolAgent::new(&dex, 1, load_meta_pool(&pool_path), cfg(), seed);
            agent.set_own_team(own_team());
            for line in &lines {
                agent.push_line(&dex, line);
            }
            agent.on_request(&dex, REQ_T11).unwrap();
            let b = agent.battle().unwrap().clone();

            if first {
                first = false;
                println!(
                    "[{iters}] synthesized turn {} pokemon_left: p1={} p2={} (truth: 3 / 1)",
                    b.turn, b.sides[0].pokemon_left, b.sides[1].pokemon_left
                );
                println!("[{iters}] belief: {}", agent.belief_info());
                for input in ["move explosion", "move zapcannon", "move destinybond"] {
                    let mut probe = b.clone();
                    probe.choose(&dex, 1, input).unwrap();
                    if !probe.ended && probe.needs_choice()[0] {
                        let foe = probe.legal_choices(&dex, 0)[0].to_input(&dex);
                        probe.choose(&dex, 0, &foe).unwrap();
                    }
                    println!(
                        "  [{iters}] probe {input:<18} -> ended={} winner={:?} own_left={}",
                        probe.ended, probe.winner, probe.sides[1].pokemon_left
                    );
                }
            }

            agent.step(&dex, iters).unwrap();
            let best = agent.best(&dex).unwrap();
            *census.entry(best).or_default() += 1;
            if seed <= 5 {
                show(&format!("[{iters}]"), seed, agent.search().expect("search"), &dex);
            }
        }
        println!("[{iters}] argmax census over 50 seeds: {census:?}\n");
    }
}
