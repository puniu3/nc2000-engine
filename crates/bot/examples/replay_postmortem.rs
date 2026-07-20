//! Postmortem for ladder battle-3634 (sobou vs puniu3, 2026-07-21): at turn
//! 10 the bot (p2, Electrode at full HP) faced the opponent's LAST mon
//! (Heracross, 3%) and chose Explosion over the certain-win Thunderbolt,
//! losing by the Self-KO clause.
//!
//! Reproduces the decision inside the real ProtocolAgent pipeline (tracker →
//! synthesize → BlindSearch at the shipped skuct config), then re-runs the
//! same search on a copy whose `pokemon_left` is forced to the true values.
//! Own sets are reconstructed (revealed moves + plausible fills); the
//! decision under test only needs Electrode's revealed Thunderbolt/Explosion.
//!
//! Pre-fix (synthesize leaving pokemon_left at the 6-mon roster count) the
//! PIPELINE arm showed ended=false in both probe directions and a flat root
//! (certain win 0.497 vs certain loss 0.478); post-fix both arms agree:
//! Thunderbolt ~0.91, Explosion 0.00.

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
    // wasm skuct_config: what ps-client shipped
    RmConfig { rule: SelRule::Ucb, c: 1.0, hp_buckets: 16, ..RmConfig::default() }
}

fn own_team() -> Vec<PokemonSet> {
    // MUST be in |poke| preview order (= the order the team was submitted in)
    serde_json::from_str(
        r#"[
        {"name":"Charizard","species":"Charizard","item":"Charcoal","ability":"No Ability",
         "moves":["Fire Blast","Earthquake","Belly Drum","Rock Slide"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":55},
        {"name":"Miltank","species":"Miltank","item":"Leftovers","ability":"No Ability",
         "moves":["Body Slam","Milk Drink","Heal Bell","Earthquake"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"F","level":55},
        {"name":"Venusaur","species":"Venusaur","item":"Miracle Berry","ability":"No Ability",
         "moves":["Razor Leaf","Sleep Powder","Leech Seed","Synthesis"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"F","level":50},
        {"name":"Golem","species":"Golem","item":"Hard Stone","ability":"No Ability",
         "moves":["Earthquake","Rock Slide","Explosion","Curse"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
        {"name":"Cloyster","species":"Cloyster","item":"Mystic Water","ability":"No Ability",
         "moves":["Surf","Spikes","Explosion","Toxic"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
        {"name":"Electrode","species":"Electrode","item":"","ability":"No Ability",
         "moves":["Thunderbolt","Explosion","Light Screen","Screech"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"level":50}
    ]"#,
    )
    .unwrap()
}

/// battle-3634 public lines, up to the turn-10 decision point.
const LOG: &str = r#"|player|p1|sobou|266|
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
|poke|p2|Charizard, L55, M|item
|poke|p2|Miltank, L55, F|item
|poke|p2|Venusaur, L50, F|item
|poke|p2|Golem, L50, M|item
|poke|p2|Cloyster, L50, M|item
|poke|p2|Electrode, L50|item
|teampreview|3
|teamsize|p1|3
|teamsize|p2|3
|start
|switch|p1a: Raikou|Raikou, L55|100/100
|switch|p2a: Cloyster|Cloyster, L50, M|100/100
|turn|1
|move|p1a: Raikou|Thunderbolt|p2a: Cloyster
|-crit|p2a: Cloyster
|-supereffective|p2a: Cloyster
|-damage|p2a: Cloyster|0 fnt
|faint|p2a: Cloyster
|switch|p2a: Electrode|Electrode, L50|100/100
|upkeep
|turn|2
|switch|p1a: Heracross|Heracross, L50, M|100/100
|move|p2a: Electrode|Light Screen|p2a: Electrode
|-sidestart|p2: puniu3|move: Light Screen
|upkeep
|turn|3
|switch|p2a: Charizard|Charizard, L55, M|100/100
|move|p1a: Heracross|Earthquake|p2a: Charizard
|-immune|p2a: Charizard
|upkeep
|turn|4
|switch|p1a: Raikou|Raikou, L55|100/100
|move|p2a: Charizard|Fire Blast|p1a: Raikou
|-damage|p1a: Raikou|60/100
|upkeep
|turn|5
|move|p1a: Raikou|Thunderbolt|p2a: Charizard
|-supereffective|p2a: Charizard
|-damage|p2a: Charizard|64/100
|move|p2a: Charizard|Belly Drum|p2a: Charizard
|-damage|p2a: Charizard|14/100
|-boost|p2a: Charizard|atk|6
|upkeep
|turn|6
|move|p2a: Charizard|Earthquake|p1a: Raikou
|-supereffective|p1a: Raikou
|-damage|p1a: Raikou|0 fnt
|faint|p1a: Raikou
|switch|p1a: Heracross|Heracross, L50, M|100/100
|-sideend|p2: puniu3|move: Light Screen
|upkeep
|turn|7
|move|p2a: Charizard|Fire Blast|p1a: Heracross
|-supereffective|p1a: Heracross
|-damage|p1a: Heracross|3/100
|move|p1a: Heracross|Megahorn|p2a: Charizard
|-crit|p2a: Charizard
|-resisted|p2a: Charizard
|-damage|p2a: Charizard|0 fnt
|faint|p2a: Charizard
|switch|p2a: Electrode|Electrode, L50|100/100
|upkeep
|turn|8
|switch|p1a: Skarmory|Skarmory, L50, M|100/100
|move|p2a: Electrode|Thunderbolt|p1a: Skarmory
|-supereffective|p1a: Skarmory
|-damage|p1a: Skarmory|29/100
|-heal|p1a: Skarmory|35/100|[from] item: Leftovers
|upkeep
|turn|9
|move|p2a: Electrode|Thunderbolt|p1a: Skarmory
|-supereffective|p1a: Skarmory
|-damage|p1a: Skarmory|0 fnt
|faint|p1a: Skarmory
|switch|p1a: Heracross|Heracross, L50, M|3/100
|upkeep
|turn|10"#;

/// The move request PS would send p2 at turn 10.
const REQ_T10: &str = r#"{
  "active":[{"moves":[
    {"id":"thunderbolt","move":"Thunderbolt","pp":22,"maxpp":24,"target":"normal","disabled":false},
    {"id":"explosion","move":"Explosion","pp":8,"maxpp":8,"target":"normal","disabled":false},
    {"id":"lightscreen","move":"Light Screen","pp":47,"maxpp":48,"target":"allySide","disabled":false},
    {"id":"screech","move":"Screech","pp":64,"maxpp":64,"target":"normal","disabled":false}
  ],"trapped":false}],
  "side":{"name":"puniu3","id":"p2","pokemon":[
    {"ident":"p2: Electrode","details":"Electrode, L50","condition":"999/999","active":true,"item":""},
    {"ident":"p2: Cloyster","details":"Cloyster, L50, M","condition":"0 fnt","active":false,"item":"mysticwater"},
    {"ident":"p2: Charizard","details":"Charizard, L55, M","condition":"0 fnt","active":false,"item":"charcoal"}
  ]},
  "rqid":19
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

    let mut census: std::collections::BTreeMap<String, u32> = Default::default();
    let mut first = true;
    for seed in 1u64..=50 {
        let mut agent = ProtocolAgent::new(&dex, 1, load_meta_pool(&pool_path), cfg(), seed);
        agent.set_own_team(own_team());
        for line in LOG.lines() {
            agent.push_line(&dex, line);
        }
        agent.on_request(&dex, REQ_T10).unwrap();
        let b = agent.battle().unwrap().clone();

        if first {
            first = false;
            println!(
                "synthesized turn {} pokemon_left: p1={} p2={} (truth: 1 / 1)",
                b.turn, b.sides[0].pokemon_left, b.sides[1].pokemon_left
            );
            println!("belief: {}", agent.belief_info());
            // dynamics probe: what the model thinks Explosion / Thunderbolt do
            for input in ["move explosion", "move thunderbolt"] {
                for patch in [false, true] {
                    let mut probe = b.clone();
                    if patch {
                        probe.sides[0].pokemon_left = 1;
                        probe.sides[1].pokemon_left = 1;
                    }
                    probe.choose(&dex, 1, input).unwrap();
                    if !probe.ended && probe.needs_choice()[0] {
                        let foe = probe.legal_choices(&dex, 0)[0].to_input(&dex);
                        probe.choose(&dex, 0, &foe).unwrap();
                    }
                    println!(
                        "  {} {input:<16} -> ended={} winner={:?}",
                        if patch { "patched" } else { "model  " },
                        probe.ended,
                        probe.winner
                    );
                }
            }
        }

        // the shipped pipeline's own search
        agent.step(&dex, 1000).unwrap();
        let best = agent.best(&dex).unwrap();
        *census.entry(best).or_default() += 1;
        if seed <= 5 {
            let s = agent_search(&agent);
            show("PIPELINE", seed, s, &dex);

            // counterfactual: same synthesis, true pokemon_left
            let mut pb = b.clone();
            pb.sides[0].pokemon_left = 1;
            pb.sides[1].pokemon_left = 1;
            let mut s = BlindSearch::new(&pb, &dex, cfg(), 1, seed);
            s.step(&dex, agent.belief().unwrap(), agent.observer().unwrap(), 1000);
            show("TRUELEFT", seed, &s, &dex);
        }
    }
    println!("argmax census over 50 seeds (shipped pipeline): {census:?}");
}

fn agent_search(agent: &ProtocolAgent) -> &BlindSearch {
    agent.search().expect("search built")
}
