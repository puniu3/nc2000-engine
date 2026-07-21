//! Postmortem for the 2026-07-21 12:36 sohehe-vs-puniu3 report (frozen
//! Snorlax stay-in) — played on the e18827f ladder build (self-KO guard,
//! 10k iters; no certain_noop mask, no L1 damage fixes).
//!
//! Three decision classes in one game:
//!   T4:  Snorlax (own) vs Gengar — Double-Edge is IMMUNE (Normal→Ghost,
//!        public knowledge) yet the live bot clicked it 7 times while
//!        Curse (free setup) and switch Porygon2 were available.
//!   T19: Snorlax 97% FROZEN vs a fresh Marowak (+0) — the reported class:
//!        an immobilized active facing a setup sweeper, with an effective
//!        answer (Porygon2, Ice Beam) on the bench. Correct play: switch.
//!   T21: same but Marowak already at +4 — the cost of staying is now
//!        near-lethal and public.
//!
//! Reproduces each through the CURRENT pipeline (tracker → synthesize →
//! BlindSearch, shipped config) at the ladder budget (10k) and 1k, 30
//! seeds: argmax census + root visit/mean detail for the stay-vs-switch
//! arms.

use nc2000_bot::blind::BlindSearch;
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;

fn cfg() -> RmConfig {
    RmConfig { rule: SelRule::Ucb, c: 1.0, hp_buckets: 16, ..RmConfig::default() }
}

/// Own team in |poke| preview order. Snorlax's 4 moves are all revealed in
/// this game (Curse / Double-Edge / Sleep Talk / Rest @ Leftovers);
/// Porygon2 revealed Ice Beam (rest from the 2026-07-20 reconstruction).
fn own_team() -> Vec<PokemonSet> {
    serde_json::from_str(
        r#"[
        {"name":"Snorlax","species":"Snorlax","item":"Leftovers","ability":"No Ability",
         "moves":["Curse","Double-Edge","Sleep Talk","Rest"],
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

const PREAMBLE: &str = r#"|player|p1|sohehe|1|
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
|switch|p2a: Marowak|Marowak, L50, M|100/100
|turn|1
|move|p1a: Starmie|Hydro Pump|p2a: Marowak
|-supereffective|p2a: Marowak
|-damage|p2a: Marowak|0 fnt
|faint|p2a: Marowak
|switch|p2a: Snorlax|Snorlax, L55, M|100/100
|upkeep
|turn|2
|switch|p1a: Gengar|Gengar, L50, M|100/100
|move|p2a: Snorlax|Curse|p2a: Snorlax
|-unboost|p2a: Snorlax|spe|1
|-boost|p2a: Snorlax|atk|1
|-boost|p2a: Snorlax|def|1
|upkeep
|turn|3
|move|p1a: Gengar|Destiny Bond|p1a: Gengar
|-singlemove|p1a: Gengar|Destiny Bond
|move|p2a: Snorlax|Curse|p2a: Snorlax
|-unboost|p2a: Snorlax|spe|1
|-boost|p2a: Snorlax|atk|1
|-boost|p2a: Snorlax|def|1
|upkeep
|turn|4"#;

/// T4 → T19 continuation (the game's middle, |t:| lines stripped).
const MID: &str = r#"|move|p1a: Gengar|Ice Punch|p2a: Snorlax
|-damage|p2a: Snorlax|89/100
|move|p2a: Snorlax|Double-Edge|p1a: Gengar
|-immune|p1a: Gengar
|-heal|p2a: Snorlax|95/100|[from] item: Leftovers
|upkeep
|turn|5
|move|p1a: Gengar|Destiny Bond|p1a: Gengar
|-singlemove|p1a: Gengar|Destiny Bond
|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax
|-heal|p2a: Snorlax|100/100|[from] item: Leftovers
|upkeep
|turn|6
|move|p1a: Gengar|Ice Punch|p2a: Snorlax
|-damage|p2a: Snorlax|90/100
|move|p2a: Snorlax|Double-Edge|p1a: Gengar
|-immune|p1a: Gengar
|-heal|p2a: Snorlax|96/100|[from] item: Leftovers
|upkeep
|turn|7
|move|p1a: Gengar|Ice Punch|p2a: Snorlax
|-damage|p2a: Snorlax|84/100
|move|p2a: Snorlax|Double-Edge|p1a: Gengar
|-immune|p1a: Gengar
|-heal|p2a: Snorlax|90/100|[from] item: Leftovers
|upkeep
|turn|8
|move|p1a: Gengar|Ice Punch|p2a: Snorlax
|-damage|p2a: Snorlax|80/100
|move|p2a: Snorlax|Double-Edge|p1a: Gengar
|-immune|p1a: Gengar
|-heal|p2a: Snorlax|86/100|[from] item: Leftovers
|upkeep
|turn|9
|move|p1a: Gengar|Ice Punch|p2a: Snorlax
|-damage|p2a: Snorlax|76/100
|move|p2a: Snorlax|Rest|p2a: Snorlax
|-status|p2a: Snorlax|slp|[from] move: Rest
|-heal|p2a: Snorlax|100/100 slp|[silent]
|upkeep
|turn|10
|move|p1a: Gengar|Thunderbolt|p2a: Snorlax
|-damage|p2a: Snorlax|87/100 slp
|cant|p2a: Snorlax|slp
|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax
|move|p2a: Snorlax|Double-Edge|p1a: Gengar|[from] Sleep Talk
|-immune|p1a: Gengar
|-heal|p2a: Snorlax|93/100 slp|[from] item: Leftovers
|upkeep
|turn|11
|move|p1a: Gengar|Thunderbolt|p2a: Snorlax
|-crit|p2a: Snorlax
|-damage|p2a: Snorlax|66/100 slp
|cant|p2a: Snorlax|slp
|-heal|p2a: Snorlax|72/100 slp|[from] item: Leftovers
|upkeep
|turn|12
|move|p1a: Gengar|Thunderbolt|p2a: Snorlax
|-damage|p2a: Snorlax|59/100 slp
|-curestatus|p2a: Snorlax|slp|[msg]
|move|p2a: Snorlax|Rest|p2a: Snorlax
|-status|p2a: Snorlax|slp|[from] move: Rest
|-heal|p2a: Snorlax|100/100 slp|[silent]
|upkeep
|turn|13
|switch|p1a: Marowak|Marowak, L55, M|100/100
|cant|p2a: Snorlax|slp
|upkeep
|turn|14
|move|p1a: Marowak|Swords Dance|p1a: Marowak
|-boost|p1a: Marowak|atk|2
|cant|p2a: Snorlax|slp
|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax
|move|p2a: Snorlax|Curse|p2a: Snorlax|[from] Sleep Talk
|-unboost|p2a: Snorlax|spe|1
|-boost|p2a: Snorlax|atk|1
|-boost|p2a: Snorlax|def|1
|upkeep
|turn|15
|switch|p1a: Gengar|Gengar, L50, M|100/100
|-curestatus|p2a: Snorlax|slp|[msg]
|move|p2a: Snorlax|Double-Edge|p1a: Gengar
|-immune|p1a: Gengar
|upkeep
|turn|16
|move|p1a: Gengar|Ice Punch|p2a: Snorlax
|-damage|p2a: Snorlax|89/100
|move|p2a: Snorlax|Double-Edge|p1a: Gengar
|-immune|p1a: Gengar
|-heal|p2a: Snorlax|95/100|[from] item: Leftovers
|upkeep
|turn|17
|move|p1a: Gengar|Ice Punch|p2a: Snorlax
|-damage|p2a: Snorlax|85/100
|-status|p2a: Snorlax|frz
|cant|p2a: Snorlax|frz
|-heal|p2a: Snorlax|91/100 frz|[from] item: Leftovers
|upkeep
|turn|18
|switch|p1a: Marowak|Marowak, L55, M|100/100
|cant|p2a: Snorlax|frz
|-heal|p2a: Snorlax|97/100 frz|[from] item: Leftovers
|upkeep
|turn|19"#;

/// T19 → T21 continuation.
const TAIL: &str = r#"|move|p1a: Marowak|Swords Dance|p1a: Marowak
|-boost|p1a: Marowak|atk|2
|cant|p2a: Snorlax|frz
|-heal|p2a: Snorlax|100/100 frz|[from] item: Leftovers
|upkeep
|turn|20
|move|p1a: Marowak|Swords Dance|p1a: Marowak
|-boost|p1a: Marowak|atk|2
|cant|p2a: Snorlax|frz
|upkeep
|turn|21"#;

fn req(hp_pct: i32, status: &str, curse_pp: i32, de_pp: i32, st_pp: i32, rest_pp: i32, maxhp: i32) -> String {
    let hp = ((hp_pct as f64 / 100.0) * maxhp as f64).round() as i32;
    let cond = if status.is_empty() {
        format!("{hp}/{maxhp}")
    } else {
        format!("{hp}/{maxhp} {status}")
    };
    format!(
        r#"{{
  "active":[{{"moves":[
    {{"id":"curse","move":"Curse","pp":{curse_pp},"maxpp":16,"target":"normal","disabled":false}},
    {{"id":"doubleedge","move":"Double-Edge","pp":{de_pp},"maxpp":24,"target":"normal","disabled":false}},
    {{"id":"sleeptalk","move":"Sleep Talk","pp":{st_pp},"maxpp":16,"target":"self","disabled":false}},
    {{"id":"rest","move":"Rest","pp":{rest_pp},"maxpp":16,"target":"self","disabled":false}}
  ],"trapped":false}}],
  "side":{{"name":"puniu3","id":"p2","pokemon":[
    {{"ident":"p2: Snorlax","details":"Snorlax, L55, M","condition":"{cond}","active":true,"item":"leftovers"}},
    {{"ident":"p2: Marowak","details":"Marowak, L50, M","condition":"0 fnt","active":false,"item":"thickclub"}},
    {{"ident":"p2: Porygon2","details":"Porygon2, L50","condition":"999/999","active":false,"item":"mintberry"}}
  ]}},
  "rqid":9
}}"#
    )
}

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
    print!("{label} seed {seed}:");
    for (input, n, m) in &rows {
        print!("  {input}={n}/{total} ({m:.3})");
    }
    println!();
}

fn run_case(dex: &Dex, pool_path: &std::path::Path, label: &str, lines: &[&str], req: &str, iters: u32) {
    let mut census: std::collections::BTreeMap<String, u32> = Default::default();
    for seed in 1u64..=30 {
        let mut agent = ProtocolAgent::new(dex, 1, load_meta_pool(pool_path), cfg(), seed);
        agent.set_own_team(own_team());
        for line in lines {
            agent.push_line(dex, line);
        }
        agent.on_request(dex, req).unwrap();
        if seed == 1 {
            let b = agent.battle().unwrap();
            println!("[{label}] belief: {}", agent.belief_info());
            // counterfactual: how much does the eval fear the foe's boosts?
            {
                use nc2000_bot::eval::{eval01, EvalWeights};
                let w = EvalWeights::default();
                let mut nb = b.clone();
                if let Some(foe) = nb.active_id(0) {
                    nb.poke_mut(foe).boosts = [0; 7];
                }
                let ours = 1.0 - eval01(b, dex, &w);
                let ours_nb = 1.0 - eval01(&nb, dex, &w);
                println!("[{label}] eval01(ours) {:.3} | foe boosts zeroed {:.3} | boost fear {:+.3}",
                    ours, ours_nb, ours_nb - ours);
            }
            if let Some(foe) = b.active_id(0) {
                let p = b.poke(foe);
                let mv: Vec<&str> = p.move_slots.iter().map(|ms| dex.moves.key(ms.id)).collect();
                println!("[{label}] imputed foe active {:?} L{} moves {:?} boosts {:?}",
                    dex.species.key(p.species), p.level, mv, p.boosts);
            }
        }
        if std::env::var("NC_PROBE_ONLY").is_ok() {
            return;
        }
        agent.step(dex, iters).unwrap();
        let best = agent.best(dex).unwrap();
        *census.entry(best).or_default() += 1;
        if seed <= 3 {
            show(label, seed, agent.search().expect("search"), dex);
        }
    }
    println!("[{label}] argmax census over 30 seeds at {iters}: {census:?}\n");
}

fn main() {
    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");

    // Snorlax L55 maxhp via a probe battle
    let team = own_team();
    let probe = nc2000_engine::state::Battle::from_fixture(&dex, "1,2,3,4", &team, &team).unwrap();
    let maxhp = probe.sides[0].roster[0].maxhp;
    println!("Snorlax L55 maxhp {maxhp}");

    let pre: Vec<&str> = PREAMBLE.lines().collect();
    let mut t19 = pre.clone();
    t19.extend(MID.lines());
    let mut t21 = t19.clone();
    t21.extend(TAIL.lines());

    for iters in [1000u32, 10000] {
        run_case(&dex, &pool_path, "T4 healthy-vs-Gengar", &pre,
            &req(100, "", 14, 24, 16, 16, maxhp), iters);
        run_case(&dex, &pool_path, "T19 frz-vs-Marowak+0", &t19,
            &req(97, "frz", 14, 18, 13, 14, maxhp), iters);
        run_case(&dex, &pool_path, "T21 frz-vs-Marowak+4", &t21,
            &req(100, "frz", 14, 18, 13, 14, maxhp), iters);
    }
}
