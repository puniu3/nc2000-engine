//! Postmortem for ladder battle-3623 (sohehe vs puniu3, 2026-07-20), the two
//! decisions the 2026-07-20 audit flagged as "root chose a move its own eval
//! ranked LAST" (handoff §8):
//!
//!   T6:  Typhlosion (100%, par) vs Zapdos (51%)  — played Dynamic Punch
//!        (audit eval 0.053) over Fire Blast (0.365).
//!   T12: Marowak (+4 Atk, 100%) vs Cloyster (100%, Reflect up) — played
//!        Hidden Power [Bug] (0.314) over the strictly-dominating Earthquake
//!        (0.667: same category, same screen, same accuracy, double power).
//!
//! The live game ran the PRE-fix binary: synthesize left `pokemon_left` at 6
//! (no terminal reachable → eval-only rollouts) and impute_hp read the HP%
//! bar as /48 pixels (foe HP ~2.08x inflated) — both fixed in 694efb1. This
//! harness replays both decision points through the CURRENT ProtocolAgent
//! pipeline (tracker → synthesize → BlindSearch, shipped skuct config, the
//! ladder client's 1000-iteration budget) across 50 searcher seeds.
//!
//! Question answered: does the current bot still produce either blunder
//! (live bug), or do the fixes account for them (close as fixed/noise)?
//! The T12 request also exercises the generic-`hiddenpower` request id
//! against a typed own set (Hidden Power Bug) — the untested path from the
//! audit's §8.
//!
//! The move-slot mapping suspicion ("slot 2 never used") is closed
//! statically, not here: submission is by move NAME end-to-end —
//! `best()` → `ps_input` → "move <id>" (typed HP normalized to plain) →
//! ps-client `/choose move <id>|rqid`. No slot index exists in the move
//! path, and M15b gate a (0 choice rejections) certifies the names match
//! PS's request slots.

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

/// Own team in |poke| preview order (= submitted order). Sets reconstructed
/// during the 2026-07-20 forensics (see replay_analysis.rs; Fire Blast and
/// Surf damage in the log sanity-match these stats). Skarmory never appears
/// and its set is a plausible fill; Snorlax is L55 per the preview line.
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
         "moves":["Zap Cannon","Mean Look","Perish Song","Destiny Bond"],
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

/// battle-3623 public lines (|j|/|t:|/blank lines stripped), through |turn|6.
const LOG_T6: &str = r#"|player|p1|sohehe|1|
|player|p2|puniu3|1|
|gen|2
|tier|[Gen 2] Nintendo Cup 2000 No OHKO Stadium2 Strict
|clearpoke
|poke|p1|Zapdos, L55|item
|poke|p1|Cloyster, L50, M|item
|poke|p1|Marowak, L50, M|item
|poke|p1|Snorlax, L50, M|item
|poke|p1|Misdreavus, L50, F|item
|poke|p1|Blissey, L50, F|item
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
|switch|p1a: Zapdos|Zapdos, L55|100/100
|switch|p2a: Typhlosion|Typhlosion, L55, M|100/100
|turn|1
|move|p1a: Zapdos|Thunder Wave|p2a: Typhlosion
|-status|p2a: Typhlosion|par
|-enditem|p2a: Typhlosion|Miracle Berry|[eat]
|-curestatus|p2a: Typhlosion|par|[msg]
|move|p2a: Typhlosion|Fire Blast|p1a: Zapdos
|-damage|p1a: Zapdos|60/100
|-heal|p1a: Zapdos|66/100|[from] item: Leftovers
|upkeep
|turn|2
|switch|p2a: Gengar|Gengar, L50, M|100/100
|move|p1a: Zapdos|Thunder Wave|p2a: Gengar
|-status|p2a: Gengar|par
|-heal|p1a: Zapdos|72/100|[from] item: Leftovers
|upkeep
|turn|3
|move|p1a: Zapdos|Thunderbolt|p2a: Gengar
|-damage|p2a: Gengar|40/100 par
|move|p2a: Gengar|Zap Cannon|p1a: Zapdos|[miss]
|-miss|p2a: Gengar
|-heal|p1a: Zapdos|78/100|[from] item: Leftovers
|-enditem|p2a: Gengar|Gold Berry|[eat]
|-heal|p2a: Gengar|58/100 par|[from] item: Gold Berry
|upkeep
|turn|4
|move|p1a: Zapdos|Thunderbolt|p2a: Gengar
|-damage|p2a: Gengar|0 fnt
|faint|p2a: Gengar
|switch|p2a: Typhlosion|Typhlosion, L55, M|100/100
|-heal|p1a: Zapdos|84/100|[from] item: Leftovers
|upkeep
|turn|5
|move|p1a: Zapdos|Thunder Wave|p2a: Typhlosion
|-status|p2a: Typhlosion|par
|move|p2a: Typhlosion|Fire Blast|p1a: Zapdos
|-damage|p1a: Zapdos|45/100
|-heal|p1a: Zapdos|51/100|[from] item: Leftovers
|upkeep
|turn|6"#;

/// Continuation from |turn|6 through |turn|12.
const LOG_T6_TO_T12: &str = r#"|move|p1a: Zapdos|Thunderbolt|p2a: Typhlosion
|-damage|p2a: Typhlosion|57/100 par
|move|p2a: Typhlosion|Dynamic Punch|p1a: Zapdos
|-resisted|p1a: Zapdos
|-damage|p1a: Zapdos|41/100
|-start|p1a: Zapdos|confusion
|-heal|p1a: Zapdos|47/100|[from] item: Leftovers
|upkeep
|turn|7
|-activate|p1a: Zapdos|confusion
|-damage|p1a: Zapdos|37/100|[from] confusion
|move|p2a: Typhlosion|Fire Blast|p1a: Zapdos|[miss]
|-miss|p2a: Typhlosion
|-heal|p1a: Zapdos|43/100|[from] item: Leftovers
|upkeep
|turn|8
|-end|p1a: Zapdos|confusion
|move|p1a: Zapdos|Thunderbolt|p2a: Typhlosion
|-damage|p2a: Typhlosion|16/100 par
|cant|p2a: Typhlosion|par
|-heal|p1a: Zapdos|49/100|[from] item: Leftovers
|upkeep
|turn|9
|move|p1a: Zapdos|Thunderbolt|p2a: Typhlosion
|-damage|p2a: Typhlosion|0 fnt
|faint|p2a: Typhlosion
|switch|p2a: Marowak|Marowak, L50, M|100/100
|-heal|p1a: Zapdos|55/100|[from] item: Leftovers
|upkeep
|turn|10
|switch|p1a: Cloyster|Cloyster, L50, M|100/100
|move|p2a: Marowak|Swords Dance|p2a: Marowak
|-boost|p2a: Marowak|atk|2
|upkeep
|turn|11
|move|p1a: Cloyster|Reflect|p1a: Cloyster
|-sidestart|p1: sohehe|Reflect
|move|p2a: Marowak|Swords Dance|p2a: Marowak
|-boost|p2a: Marowak|atk|2
|upkeep
|turn|12"#;

/// The move request PS would send p2 at turn 6 (Typhlosion active, par;
/// Fire Blast used twice; both berries eaten).
const REQ_T6: &str = r#"{
  "active":[{"moves":[
    {"id":"fireblast","move":"Fire Blast","pp":6,"maxpp":8,"target":"normal","disabled":false},
    {"id":"thunderpunch","move":"Thunder Punch","pp":24,"maxpp":24,"target":"normal","disabled":false},
    {"id":"dynamicpunch","move":"Dynamic Punch","pp":8,"maxpp":8,"target":"normal","disabled":false},
    {"id":"sunnyday","move":"Sunny Day","pp":8,"maxpp":8,"target":"all","disabled":false}
  ],"trapped":false}],
  "side":{"name":"puniu3","id":"p2","pokemon":[
    {"ident":"p2: Typhlosion","details":"Typhlosion, L55, M","condition":"999/999 par","active":true,"item":""},
    {"ident":"p2: Gengar","details":"Gengar, L50, M","condition":"0 fnt","active":false,"item":""},
    {"ident":"p2: Marowak","details":"Marowak, L50, M","condition":"999/999","active":false,"item":"thickclub"}
  ]},
  "rqid":11
}"#;

/// The move request PS would send p2 at turn 12 (Marowak active at +4,
/// Swords Dance used twice). NOTE: PS delivers the typed own set's Hidden
/// Power as the GENERIC id `hiddenpower` — this request is also the test of
/// that path against the typed "Hidden Power Bug" own slot.
const REQ_T12: &str = r#"{
  "active":[{"moves":[
    {"id":"earthquake","move":"Earthquake","pp":16,"maxpp":16,"target":"normal","disabled":false},
    {"id":"rockslide","move":"Rock Slide","pp":16,"maxpp":16,"target":"normal","disabled":false},
    {"id":"hiddenpower","move":"Hidden Power","pp":24,"maxpp":24,"target":"normal","disabled":false},
    {"id":"swordsdance","move":"Swords Dance","pp":46,"maxpp":48,"target":"self","disabled":false}
  ],"trapped":false}],
  "side":{"name":"puniu3","id":"p2","pokemon":[
    {"ident":"p2: Marowak","details":"Marowak, L50, M","condition":"999/999","active":true,"item":"thickclub"},
    {"ident":"p2: Gengar","details":"Gengar, L50, M","condition":"0 fnt","active":false,"item":""},
    {"ident":"p2: Typhlosion","details":"Typhlosion, L55, M","condition":"0 fnt","active":false,"item":""}
  ]},
  "rqid":23
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

fn run_case(
    dex: &Dex,
    pool_path: &std::path::Path,
    label: &str,
    lines: &[&str],
    req: &str,
    probes: &[&str],
    iters: u32,
    seeds: u64,
) {
    let mut census: std::collections::BTreeMap<String, u32> = Default::default();
    let mut first = true;
    for seed in 1u64..=seeds {
        let mut agent = ProtocolAgent::new(dex, 1, load_meta_pool(pool_path), cfg(), seed);
        agent.set_own_team(own_team());
        for line in lines {
            agent.push_line(dex, line);
        }
        agent.on_request(dex, req).unwrap();
        let b = agent.battle().unwrap().clone();

        if first {
            first = false;
            println!(
                "[{label}] synthesized turn {} pokemon_left: p1={} p2={}",
                b.turn, b.sides[0].pokemon_left, b.sides[1].pokemon_left
            );
            println!("[{label}] belief: {}", agent.belief_info());
            // own active's synthesized move ids (typed-HP retention check)
            let own = b.active_id(1).unwrap();
            let ids: Vec<&str> = b
                .poke(own)
                .move_slots
                .iter()
                .map(|ms| dex.moves.key(ms.id))
                .collect();
            println!("[{label}] own active synthesized moves: {ids:?}");
            // dynamics probe: one sample of what the model thinks each move does
            for input in probes {
                let mut probe = b.clone();
                if let Err(e) = probe.choose(dex, 1, input) {
                    println!("  [{label}] probe {input:<18} -> REJECTED: {e:?}");
                    continue;
                }
                if !probe.ended && probe.needs_choice()[0] {
                    let foe = probe.legal_choices(dex, 0)[0].to_input(dex);
                    probe.choose(dex, 0, &foe).unwrap();
                }
                let foe_frac = probe
                    .active_id(0)
                    .map(|id| {
                        let p = probe.poke(id);
                        format!("{:.0}%", 100.0 * p.hp as f64 / p.maxhp as f64)
                    })
                    .unwrap_or_else(|| "-".into());
                println!(
                    "  [{label}] probe {input:<18} -> foe active {} ended={}",
                    foe_frac, probe.ended
                );
            }
        }

        agent.step(dex, iters).unwrap();
        let best = agent.best(dex).unwrap();
        *census.entry(best).or_default() += 1;
        if seed <= 5 {
            show(label, seed, agent.search().expect("search built"), dex);
        }
    }
    println!("[{label}] argmax census over {seeds} seeds at {iters} iters: {census:?}\n");
}

fn main() {
    let dex = load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");

    let t6_lines: Vec<&str> = LOG_T6.lines().collect();
    run_case(
        &dex,
        &pool_path,
        "T6 ",
        &t6_lines,
        REQ_T6,
        &["move fireblast", "move dynamicpunch", "move thunderpunch"],
        1000,
        50,
    );
    // budget scaling: does the flat root sharpen at product-scale budgets?
    for iters in [3000u32, 10000] {
        run_case(&dex, &pool_path, "T6+", &t6_lines, REQ_T6, &[], iters, 10);
    }

    let mut t12_lines = t6_lines.clone();
    t12_lines.extend(LOG_T6_TO_T12.lines());
    run_case(
        &dex,
        &pool_path,
        "T12",
        &t12_lines,
        REQ_T12,
        &["move earthquake", "move hiddenpowerbug", "move rockslide"],
        1000,
        50,
    );
}
