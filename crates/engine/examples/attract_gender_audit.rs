//! Isolate gender in actual Attract turns, then validate legal catalog changes.
//! cargo run -p nc2000-engine --example attract_gender_audit -- OLD_POOL NEW_POOL NEW_REPORT
use nc2000_engine::{
    battle::{
        enumerate::enumerate_step,
        moveexec::{get_damage_synthetic, synthetic_move},
        PokemonSet, SearchChoice,
    },
    dex::{toid, Dex},
    state::{Battle, Gender},
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};

fn battle(dex: &Dex, sets: &Value, lead: usize) -> Battle {
    let team: Vec<PokemonSet> = serde_json::from_value(sets.clone()).unwrap();
    let mut selection = None;
    for i in 0..team.len() {
        for j in i + 1..team.len() {
            if i != lead
                && j != lead
                && u16::from(team[lead].level) + u16::from(team[i].level) + u16::from(team[j].level)
                    <= 155
            {
                selection.get_or_insert((i, j));
            }
        }
    }
    let (i, j) = selection.unwrap();
    let mut b = Battle::from_fixture(dex, "1,2,3,4", &team, &team).unwrap();
    for side in 0..2 {
        b.choose(dex, side, &format!("team {},{},{}", lead + 1, i + 1, j + 1))
            .unwrap();
    }
    b.set_log_enabled(false);
    b
}
fn damage(dex: &Dex, b: &mut Battle, name: &str, atk: i8, def: i8) -> i64 {
    let id = b.active_id(0).unwrap();
    b.poke_mut(id).boosts[0] = atk;
    b.poke_mut(id).boosts[1] = def;
    let mut fake = synthetic_move(dex, 40);
    fake.base_move_type = dex
        .move_static(dex.moves.id(&toid(name)).unwrap())
        .move_type;
    get_damage_synthetic(b, dex, id, id, fake).unwrap() as i64
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: attract_gender_audit OLD_POOL NEW_POOL NEW_REPORT"
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dex_text = fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap();
    let dex = Dex::from_json(&dex_text).unwrap();
    let ls =
        Learnsets::from_json(&fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap())
            .unwrap();
    let old_bytes = fs::read(root.join(&args[1])).unwrap();
    let new_bytes = fs::read(root.join(&args[2])).unwrap();
    let old: Value = serde_json::from_slice(&old_bytes).unwrap();
    let new: Value = serde_json::from_slice(&new_bytes).unwrap();
    // Controlled causal intervention: the identical battle has only gender
    // changed. Synthetic genderless Venusaur/Ninetales are negative controls,
    // not proposed legal sets. The actual proposed sets are validated below.
    let mon = |sp: &str, mv: &str| {
        serde_json::from_value::<PokemonSet>(json!({
            "name":sp,"species":sp,"moves":[mv],"level":50,
            "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
            "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255}
        }))
        .unwrap()
    };
    let own = vec![
        mon("Ninetales", "Attract"),
        mon("Snorlax", "Rest"),
        mon("Starmie", "Recover"),
    ];
    let foe = vec![
        mon("Venusaur", "Growth"),
        mon("Snorlax", "Rest"),
        mon("Starmie", "Recover"),
    ];
    let mut base = Battle::from_fixture(&dex, "1,2,3,4", &own, &foe).unwrap();
    for side in 0..2 {
        base.choose(&dex, side, "team 1,2,3").unwrap();
    }
    base.set_log_enabled(false);
    let mut matrix = Vec::new();
    for source_gender in [Gender::M, Gender::F, Gender::N] {
        for target_gender in [Gender::M, Gender::F, Gender::N] {
            let mut b = base.clone();
            let src = b.active_id(0).unwrap();
            let dst = b.active_id(1).unwrap();
            b.poke_mut(src).gender = source_gender;
            b.poke_mut(dst).gender = target_gender;
            let choices = [
                Some(SearchChoice::Move(dex.moves.id("attract").unwrap())),
                Some(SearchChoice::Move(dex.moves.id("growth").unwrap())),
            ];
            let step = enumerate_step(&dex, &b, choices, 20000).unwrap();
            let mut infatuated = 0.0;
            let mut acted = 0.0;
            let mut mass = 0.0;
            for leaf in &step.leaves {
                mass += leaf.prob;
                let target = leaf.battle.poke(dst);
                if target.has_volatile(dex.conds_id("attract").unwrap()) {
                    infatuated += leaf.prob;
                }
                if target.boosts[2] > 0 {
                    acted += leaf.prob;
                }
            }
            let opposite = matches!(
                (source_gender, target_gender),
                (Gender::M, Gender::F) | (Gender::F, Gender::M)
            );
            assert!((mass - 1.0).abs() < 1e-10);
            assert!((infatuated - if opposite { 1.0 } else { 0.0 }).abs() < 1e-10);
            assert!((acted - if opposite { 0.5 } else { 1.0 }).abs() < 1e-10);
            let label = |g: Gender| if g == Gender::N { "N" } else { g.as_str() };
            matrix.push(json!({"sourceGender":label(source_gender),"targetGender":label(target_gender),
                "infatuatedProbability":infatuated,"targetActsProbability":acted,"leaves":step.leaves.len()}));
        }
    }
    let mut rows = Vec::new();
    let mut checks = 0;
    assert_eq!(
        old["teams"].as_array().unwrap().len(),
        new["teams"].as_array().unwrap().len()
    );
    for (before, after) in old["teams"]
        .as_array()
        .unwrap()
        .iter()
        .zip(new["teams"].as_array().unwrap())
    {
        assert_eq!(before["id"], after["id"]);
        assert_eq!(before["drawWeight"], after["drawWeight"]);
        for sets in [&before["sets"], &after["sets"]] {
            let verdict = validate_team(&dex, &ls, &sets.to_string());
            assert_eq!(verdict["ok"], true, "{verdict}");
            assert!(!verdict["findings"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| matches!(f["code"].as_str(), Some("dv-gender" | "gender-species"))));
        }
        for (slot, (a, z)) in before["sets"]
            .as_array()
            .unwrap()
            .iter()
            .zip(after["sets"].as_array().unwrap())
            .enumerate()
        {
            if a == z {
                continue;
            }
            assert_eq!(a["gender"], "M");
            assert_eq!(z["gender"], "F");
            let mut restored = z.clone();
            restored["gender"] = a["gender"].clone();
            restored["ivs"]["atk"] = a["ivs"]["atk"].clone();
            assert_eq!(restored, *a, "unexpected non-gender/Attack-IV change");
            let mut b = battle(&dex, &before["sets"], slot);
            let mut n = battle(&dex, &after["sets"], slot);
            let bp = b.poke(b.active_id(0).unwrap());
            let np = n.poke(n.active_id(0).unwrap());
            assert_eq!(bp.gender, Gender::M);
            assert_eq!(np.gender, Gender::F);
            assert_eq!(bp.maxhp, np.maxhp);
            assert_eq!(bp.stored_stats[1..], np.stored_stats[1..]);
            assert!(np.stored_stats[0] <= bp.stored_stats[0]);
            let (attack_before, attack_after, hp) =
                (bp.stored_stats[0], np.stored_stats[0], bp.maxhp);
            let mut damages = Vec::new();
            for mv in a["moves"].as_array().unwrap() {
                let name = mv.as_str().unwrap();
                damages.push(json!({"move":name,"neutralBefore":damage(&dex,&mut b,name,0,0),
                    "neutralAfter":damage(&dex,&mut n,name,0,0),"swaggerBefore":damage(&dex,&mut b,name,2,0),
                    "swaggerAfter":damage(&dex,&mut n,name,2,0)}));
                for atk in -6..=6 {
                    for def in -6..=6 {
                        assert!(
                            damage(&dex, &mut n, name, atk, def)
                                <= damage(&dex, &mut b, name, atk, def)
                        );
                        checks += 1;
                    }
                }
            }
            rows.push(json!({"baseId":before["id"],"slot":slot+1,"species":a["species"],"hp":hp,
                "genderBefore":"M","genderAfter":"F","attackDvBefore":a["ivs"]["atk"].as_i64().unwrap()/2,
                "attackDvAfter":z["ivs"]["atk"].as_i64().unwrap()/2,
                "attackBefore":attack_before,"attackAfter":attack_after,"damage":damages}));
        }
    }
    let report = json!({"format":"nc2000-attract-gender-mechanics-v1",
        "baselineSha256":format!("{:x}",Sha256::digest(&old_bytes)),"optimizedSha256":format!("{:x}",Sha256::digest(&new_bytes)),
        "matrixControl":"Identical stats, moves, items, seed and field; gender intervention only. Genderless Ninetales/Venusaur are synthetic negative controls, not legal catalog proposals.",
        "matrix":matrix,"validatedTeams":new["teams"].as_array().unwrap().len(),"changedSets":rows.len(),
        "selfHitComparisons":checks,"selfHitIncreases":0,"unchanged":"HP, non-Attack stats, moves, items, EVs, levels, team IDs, draw weights",
        "rows":rows});
    let out = root.join(&args[3]);
    fs::create_dir_all(out.parent().unwrap()).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .unwrap()
        .write_all((serde_json::to_string_pretty(&report).unwrap() + "\n").as_bytes())
        .unwrap();
    println!("9 exact Attract cases; {} teams validated; {} legal gender changes; {} self-hit comparisons",new["teams"].as_array().unwrap().len(),rows.len(),checks);
}
