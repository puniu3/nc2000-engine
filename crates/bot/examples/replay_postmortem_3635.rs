//! Postmortem for ladder battle-3635 (sobby vs puniu3, 2026-07-21): at turn
//! 5 the bot (p2) had Alakazam (100%) in on the opponent's Starmie (27%,
//! slower, just Thunder-Punched for ~49%) and switched to Snorlax instead of
//! finishing with Thunder Punch — Starmie recovered to full and the bot's
//! Snorlax was later forced into a Self-Destruct trade.
//!
//! Same reproduction shape as `replay_postmortem`: the real ProtocolAgent
//! pipeline (post-fix), against a PREFIX arm that re-plants the pre-fix
//! `pokemon_left = 6` roster count on the synthesized battle.

use nc2000_bot::blind::BlindSearch;
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;

fn cfg() -> RmConfig {
    RmConfig { rule: SelRule::Ucb, c: 1.0, hp_buckets: 16, ..RmConfig::default() }
}

fn own_team() -> Vec<PokemonSet> {
    // |poke| preview order; revealed moves + plausible fills
    serde_json::from_str(
        r#"[
        {"name":"Alakazam","species":"Alakazam","item":"","ability":"No Ability",
         "moves":["Thunder Punch","Psychic","Recover","Encore"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"F","level":55},
        {"name":"Heracross","species":"Heracross","item":"Focus Band","ability":"No Ability",
         "moves":["Megahorn","Earthquake","Rest","Curse"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":55},
        {"name":"Graveler","species":"Graveler","item":"Berserk Gene","ability":"No Ability",
         "moves":["Earthquake","Rock Slide","Explosion","Curse"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
        {"name":"Golem","species":"Golem","item":"Hard Stone","ability":"No Ability",
         "moves":["Earthquake","Rock Slide","Explosion","Roar"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
        {"name":"Cloyster","species":"Cloyster","item":"Mystic Water","ability":"No Ability",
         "moves":["Spikes","Explosion","Surf","Ice Beam"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
        {"name":"Snorlax","species":"Snorlax","item":"Leftovers","ability":"No Ability",
         "moves":["Earthquake","Self-Destruct","Body Slam","Rest"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
    ]"#,
    )
    .unwrap()
}

/// battle-3635 public lines, up to the turn-5 decision point.
const LOG: &str = r#"|player|p1|sobby|102|
|player|p2|puniu3|1|
|gen|2
|tier|[Gen 2] Nintendo Cup 2000 No OHKO Stadium2 Strict
|clearpoke
|poke|p1|Starmie, L55|item
|poke|p1|Raikou, L55|item
|poke|p1|Cloyster, L50, M|item
|poke|p1|Snorlax, L50, M|item
|poke|p1|Skarmory, L50, M|item
|poke|p1|Heracross, L50, M|item
|poke|p2|Alakazam, L55, F|item
|poke|p2|Heracross, L55, M|item
|poke|p2|Graveler, L50, M|item
|poke|p2|Golem, L50, M|item
|poke|p2|Cloyster, L50, M|item
|poke|p2|Snorlax, L50, M|item
|teampreview|3
|teamsize|p1|3
|teamsize|p2|3
|start
|switch|p1a: Starmie|Starmie, L55|100/100
|switch|p2a: Cloyster|Cloyster, L50, M|100/100
|turn|1
|move|p1a: Starmie|Toxic|p2a: Cloyster
|-status|p2a: Cloyster|tox
|move|p2a: Cloyster|Spikes|p1a: Starmie
|-sidestart|p1: sobby|Spikes
|-damage|p2a: Cloyster|95/100 tox|[from] psn
|upkeep
|turn|2
|switch|p1a: Snorlax|Snorlax, L50, M|100/100
|-damage|p1a: Snorlax|88/100|[from] Spikes
|move|p2a: Cloyster|Explosion|p1a: Snorlax
|-damage|p1a: Snorlax|0 fnt
|faint|p2a: Cloyster
|faint|p1a: Snorlax
|switch|p2a: Snorlax|Snorlax, L50, M|100/100
|switch|p1a: Starmie|Starmie, L55|100/100
|-damage|p1a: Starmie|88/100|[from] Spikes
|upkeep
|turn|3
|switch|p1a: Skarmory|Skarmory, L50, M|100/100
|switch|p2a: Alakazam|Alakazam, L55, F|100/100
|upkeep
|turn|4
|switch|p1a: Starmie|Starmie, L55|88/100
|-damage|p1a: Starmie|76/100|[from] Spikes
|move|p2a: Alakazam|Thunder Punch|p1a: Starmie
|-supereffective|p1a: Starmie
|-damage|p1a: Starmie|27/100
|upkeep
|turn|5"#;

/// The move request PS would send p2 at turn 5.
const REQ_T5: &str = r#"{
  "active":[{"moves":[
    {"id":"thunderpunch","move":"Thunder Punch","pp":22,"maxpp":24,"target":"normal","disabled":false},
    {"id":"psychic","move":"Psychic","pp":16,"maxpp":16,"target":"normal","disabled":false},
    {"id":"recover","move":"Recover","pp":32,"maxpp":32,"target":"self","disabled":false},
    {"id":"encore","move":"Encore","pp":8,"maxpp":8,"target":"normal","disabled":false}
  ],"trapped":false}],
  "side":{"name":"puniu3","id":"p2","pokemon":[
    {"ident":"p2: Alakazam","details":"Alakazam, L55, F","condition":"999/999","active":true,"item":""},
    {"ident":"p2: Cloyster","details":"Cloyster, L50, M","condition":"0 fnt","active":false,"item":"mysticwater"},
    {"ident":"p2: Snorlax","details":"Snorlax, L50, M","condition":"999/999","active":false,"item":"leftovers"}
  ]},
  "rqid":9
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
    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");

    let mut fixed: std::collections::BTreeMap<String, u32> = Default::default();
    let mut prefix: std::collections::BTreeMap<String, u32> = Default::default();
    for seed in 1u64..=20 {
        let mut agent = ProtocolAgent::new(&dex, 1, load_meta_pool(&pool_path), cfg(), seed);
        agent.set_own_team(own_team());
        for line in LOG.lines() {
            agent.push_line(&dex, line);
        }
        agent.on_request(&dex, REQ_T5).unwrap();
        let b = agent.battle().unwrap().clone();
        if seed == 1 {
            println!(
                "synthesized turn {} pokemon_left: p1={} p2={} (truth: 2 / 2)",
                b.turn, b.sides[0].pokemon_left, b.sides[1].pokemon_left
            );
            for &slot in b.sides[0].party.iter() {
                let p = &b.sides[0].roster[slot as usize];
                let moves: Vec<&str> =
                    p.move_slots.iter().map(|ms| dex.moves.key(ms.id)).collect();
                println!(
                    "  imputed opp {}: hp {}/{} item {:?} moves {:?}",
                    dex.species.key(p.species),
                    p.hp,
                    p.maxhp,
                    p.item.map(|i| dex.items.key(i)),
                    moves
                );
            }
        }

        agent.step(&dex, 1000).unwrap();
        *fixed.entry(agent.best(&dex).unwrap()).or_default() += 1;
        if seed <= 3 {
            show("FIXED ", seed, agent.search().unwrap(), &dex);
        }

        // pre-fix emulation: roster-count pokemon_left
        let mut pb = b.clone();
        pb.sides[0].pokemon_left = 6;
        pb.sides[1].pokemon_left = 6;
        let mut s = BlindSearch::new(&pb, &dex, cfg(), 1, seed);
        s.step(&dex, agent.belief().unwrap(), agent.observer().unwrap(), 1000);
        let best = s.best().map(|c| c.to_input(&dex)).unwrap_or_default();
        *prefix.entry(best).or_default() += 1;
        if seed <= 3 {
            show("PREFIX", seed, &s, &dex);
        }
    }
    println!("argmax census over 20 seeds:");
    println!("  fixed:   {fixed:?}");
    println!("  pre-fix: {prefix:?}");
}
