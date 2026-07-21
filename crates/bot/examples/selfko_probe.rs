//! M16 — last-mon self-destruct probe (bug report 2026-07-21: a cloned-repo
//! user saw the bot Explode with its final Pokemon).
//!
//! True-state path (SkuctSearch — the play example / web worker agent, no
//! importer involved). Three scenario classes, all "our Electrode is our
//! LAST mon, holding Explosion":
//!   (a) foe's LAST mon at ~3%   — winnable: Thunderbolt wins, Explosion is
//!       a Self-KO-clause loss (the 3634 shape);
//!   (b) foe's LAST mon healthy  — losing-ish 1v1;
//!   (c) foe has TWO mons left   — hopeless.
//! For each: verify the terminal semantics (what Explosion actually does),
//! then argmax census over 30 searcher seeds at 1k and 30k iterations
//! (ladder-old and product budgets).
//!
//! Expected healthy behavior: (a) never Explosion; (b)/(c) Explosion is an
//! immediate certain loss while any other action keeps nonzero equity, so
//! it should not be argmax — if it is, the root is treating terminal loss
//! as indistinguishable from fighting on.

use nc2000_bot::smmcts::{RmConfig, SelRule, SkuctSearch};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, Status};

fn cfg() -> RmConfig {
    RmConfig { rule: SelRule::Ucb, c: 1.0, hp_buckets: 16, ..RmConfig::default() }
}

fn team() -> Vec<PokemonSet> {
    // the 3634 postmortem team — engine-legal, has Electrode w/ Explosion
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

/// Faint a roster slot outright and fix the side's bookkeeping.
fn faint_slot(b: &mut Battle, side: usize, slot: usize) {
    let id = nc2000_engine::state::PokeId { side: side as u8, slot: slot as u8 };
    let p = b.poke_mut(id);
    p.hp = 0;
    p.status = Status::Fnt;
    p.fainted = true;
    b.sides[side].pokemon_left -= 1;
}

fn scenario(dex: &Dex, which: char) -> Battle {
    let t = team();
    let mut b = Battle::from_fixture(dex, "11,22,33,44", &t, &t).unwrap();
    b.set_log_enabled(false);
    // us: Electrode lead + Golem + Cloyster; them: Charizard lead + Miltank + Venusaur
    b.choose(dex, 0, "team 6, 4, 5").unwrap();
    b.choose(dex, 1, "team 1, 3, 4").unwrap();
    // our bench dies: Electrode is the last mon
    let (g, c) = (3usize, 4usize); // roster slots of Golem / Cloyster
    faint_slot(&mut b, 0, g);
    faint_slot(&mut b, 0, c);
    match which {
        'a' => {
            // foe last mon at ~3%
            faint_slot(&mut b, 1, 2); // Venusaur
            faint_slot(&mut b, 1, 3); // Golem
            let foe = b.active_id(1).unwrap();
            b.poke_mut(foe).hp = (b.poke(foe).maxhp / 30).max(1);
        }
        'b' => {
            faint_slot(&mut b, 1, 2);
            faint_slot(&mut b, 1, 3);
        }
        _ => {
            faint_slot(&mut b, 1, 3);
        }
    }
    b
}

fn terminal_check(dex: &Dex, b: &Battle, label: &str) {
    let mut p = b.clone();
    p.choose(dex, 0, "move explosion").unwrap();
    if !p.ended && p.needs_choice()[1] {
        let foe = p.legal_choices(dex, 1)[0].to_input(dex);
        p.choose(dex, 1, &foe).unwrap();
    }
    println!(
        "  [{label}] explosion terminal: ended={} winner={:?} own_left={} foe_left={}",
        p.ended, p.winner, p.sides[0].pokemon_left, p.sides[1].pokemon_left
    );
}

fn main() {
    let dex = conformance::load_dex();
    for which in ['a', 'b', 'c'] {
        let b = scenario(&dex, which);
        let desc = match which {
            'a' => "foe last mon ~3% (winnable)",
            'b' => "foe last mon healthy (1v1)",
            _ => "foe 2 mons left (hopeless)",
        };
        println!("== scenario {which}: {desc} ==");
        terminal_check(&dex, &b, "term");
        for iters in [1000u32, 30000] {
            let mut census: std::collections::BTreeMap<String, u32> = Default::default();
            let mut detail = String::new();
            for seed in 1u64..=30 {
                let mut s = SkuctSearch::new(&b, &dex, cfg(), seed);
                s.step(&dex, iters);
                let best = s.best(0).map(|c| c.to_input(&dex)).unwrap_or_default();
                *census.entry(best).or_default() += 1;
                if seed == 1 {
                    let acts = s.actions(0);
                    let vis = s.visits(0);
                    let means = s.means(0);
                    let mut rows: Vec<(String, u32, f64)> = acts
                        .iter()
                        .zip(vis.iter().zip(means.iter()))
                        .map(|(&a, (&n, &m))| (a.to_input(&dex), n, m))
                        .collect();
                    rows.sort_by(|x, y| y.1.cmp(&x.1));
                    for (a, n, m) in rows {
                        detail.push_str(&format!("  {a}={n} ({m:.3})"));
                    }
                }
            }
            println!("  iters {iters:>6}: census {census:?}");
            println!("    seed1 root:{detail}");
        }
        println!();
    }
}
