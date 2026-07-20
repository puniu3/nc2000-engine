//! Ad-hoc damage analysis for a 2026-07-20 ladder replay. Throwaway.
//! Uses the real engine damage core (get_damage_synthetic) so Reflect,
//! Thick Club, Swords Dance boosts, STAB and type effectiveness are all
//! applied exactly as in a real battle. Max roll = no_damage_variance;
//! min roll = floor(max * 217/255).

use conformance::load_dex;
use nc2000_engine::battle::moveexec::{get_active_move, get_damage_synthetic};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::{toid, Category, Dex};
use nc2000_engine::state::{Battle, PokeId};

fn set(json: &str) -> PokemonSet {
    serde_json::from_str(json).unwrap()
}

fn start(dex: &Dex, p1: &[PokemonSet], p2: &[PokemonSet]) -> Battle {
    let mut b = Battle::from_fixture(dex, "1,2,3,4", p1, p2).unwrap();
    b.set_log_enabled(false);
    b.choose(dex, 0, "team 1,2,3").unwrap();
    b.choose(dex, 1, "team 1,2,3").unwrap();
    b
}

/// (max_roll, min_roll) damage of `mv` from `att` to `def` in the current
/// battle state. Deterministic (no crit, top roll then scaled down).
fn dmg(b: &mut Battle, dex: &Dex, att: PokeId, def: PokeId, mv: &str) -> (i32, i32) {
    let mid = dex.moves.id(&toid(mv)).unwrap_or_else(|| panic!("no move {mv}"));
    let mut fake = get_active_move(dex, mid);
    fake.no_damage_variance = true;
    fake.will_crit = Some(false);
    // Hidden Power: base_power comes from the basePowerCallback (hp_power);
    // its type/category are set in onModifyMove which get_damage does not
    // run, so plant them here from the attacker's rolled DVs.
    if dex.moves.key(mid) == "hiddenpower" {
        let a = b.poke(att);
        fake.move_type = a.hp_type;
        fake.base_move_type = a.hp_type;
        // gen-2: physical/special by type; Bug is physical.
        let special = matches!(
            dex.type_name(a.hp_type),
            "Fire" | "Water" | "Grass" | "Electric" | "Psychic" | "Ice" | "Dragon" | "Dark"
        );
        fake.category = if special { Category::Special } else { Category::Physical };
    }
    let max = get_damage_synthetic(b, dex, att, def, fake).unwrap_or(0.0) as i32;
    let min = ((max as f64) * 217.0 / 255.0).floor() as i32;
    (max, min)
}

fn line(b: &mut Battle, dex: &Dex, att: PokeId, def: PokeId, mv: &str, tag: &str) {
    let maxhp = b.poke(def).maxhp;
    let cur = b.poke(def).hp;
    let (mx, mn) = dmg(b, dex, att, def, mv);
    let pct_hi = 100.0 * mx as f64 / maxhp as f64;
    let pct_lo = 100.0 * mn as f64 / maxhp as f64;
    let ko = if mn >= cur {
        "KO (even min roll)"
    } else if mx >= cur {
        "KO possible (high roll)"
    } else {
        "no KO"
    };
    println!(
        "    {mv:<16} {tag:<10} {mn:>3}-{mx:>3} dmg  ({pct_lo:>4.0}-{pct_hi:>4.0}% of {maxhp} maxhp; def@{cur})  -> {ko}"
    );
}

fn main() {
    let dex = load_dex();

    let marowak = set(r#"{"name":"Marowak","species":"Marowak","item":"Thick Club","ability":"No Ability","moves":["Earthquake","Rock Slide","Hidden Power Bug","Swords Dance"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":30,"atk":26,"def":26,"spa":30,"spd":30,"spe":30},"gender":"M"}"#);
    let cloyster = set(r#"{"name":"Cloyster","species":"Cloyster","item":"Gold Berry","ability":"No Ability","moves":["Surf","Ice Beam","Reflect","Spikes"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"M"}"#);
    let zapdos = set(r#"{"name":"Zapdos","species":"Zapdos","item":"Leftovers","ability":"No Ability","moves":["Thunderbolt","Hidden Power Ice","Thunder Wave","Whirlwind"],"level":55,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":30,"atk":30,"def":26,"spa":30,"spd":30,"spe":30},"gender":"N"}"#);
    let gengar = set(r#"{"name":"Gengar","species":"Gengar","item":"Gold Berry","ability":"No Ability","moves":["Zap Cannon","Mean Look","Perish Song","Destiny Bond"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"M"}"#);
    let typhlosion = set(r#"{"name":"Typhlosion","species":"Typhlosion","item":"Miracle Berry","ability":"No Ability","moves":["Fire Blast","Thunder Punch","Dynamic Punch","Sunny Day"],"level":55,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"M"}"#);
    let snorlax = set(r#"{"name":"Snorlax","species":"Snorlax","item":"Leftovers","ability":"No Ability","moves":["Body Slam","Curse","Rest","Earthquake"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"M"}"#);
    let porygon2 = set(r#"{"name":"Porygon2","species":"Porygon2","item":"Mint Berry","ability":"No Ability","moves":["Recover","Thunderbolt","Ice Beam","Curse"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"N"}"#);

    // ============ Scenario A: Marowak (p2) vs Cloyster (p1) ============
    println!("===== A. MAROWAK vs CLOYSTER (Marowak @Thick Club, HP Bug set) =====");
    {
        let p1 = vec![cloyster.clone(), snorlax.clone(), porygon2.clone()];
        let p2 = vec![marowak.clone(), snorlax.clone(), porygon2.clone()];
        let mut b = start(&dex, &p1, &p2);
        let cl = b.active_id(0).unwrap();
        let mw = b.active_id(1).unwrap();
        println!(
            "  Marowak atk stat (unboosted, w/ Thick Club): {}",
            b.get_stat(&dex, mw, 0, false, false, false)
        );
        println!("  Cloyster maxhp {} / def {}", b.poke(cl).maxhp, b.get_stat(&dex, cl, 1, false, false, false));
        for &boost in &[0i8, 2, 4, 6] {
            for &reflect in &[false, true] {
                b.poke_mut(mw).boosts[0] = boost;
                // reset & maybe (re)add Reflect on Cloyster's side (p1=0)
                b.sides[0].side_conditions.clear();
                if reflect {
                    b.add_side_condition(&dex, 0, "reflect", Some(cl), nc2000_engine::battle::EffectHandle::None);
                }
                let rtag = if reflect { "Reflect" } else { "no-Reflect" };
                println!("  +{boost} Atk, {rtag}:");
                line(&mut b, &dex, mw, cl, "Earthquake", rtag);
                line(&mut b, &dex, mw, cl, "Rock Slide", rtag);
                line(&mut b, &dex, mw, cl, "Hidden Power Bug", rtag);
            }
        }
        // Cloyster Surf back on Marowak (2HKO check; Marowak 2x weak)
        b.poke_mut(mw).boosts[0] = 0;
        b.sides[0].side_conditions.clear();
        println!("  Cloyster Surf vs Marowak (Marowak maxhp {}):", b.poke(mw).maxhp);
        line(&mut b, &dex, cl, mw, "Surf", "");
    }

    // ============ Scenario B: Gengar (p2) vs Zapdos (p1) ============
    println!("\n===== B. GENGAR vs ZAPDOS =====");
    {
        let p1 = vec![zapdos.clone(), snorlax.clone(), porygon2.clone()];
        let p2 = vec![gengar.clone(), snorlax.clone(), porygon2.clone()];
        let mut b = start(&dex, &p1, &p2);
        let zap = b.active_id(0).unwrap();
        let gen = b.active_id(1).unwrap();
        println!("  Gengar Zap Cannon vs Zapdos (Zapdos maxhp {}):", b.poke(zap).maxhp);
        line(&mut b, &dex, gen, zap, "Zap Cannon", "neut");
        println!("  Zapdos Thunderbolt vs Gengar (Gengar maxhp {}):", b.poke(gen).maxhp);
        line(&mut b, &dex, zap, gen, "Thunderbolt", "neut");
    }

    // ============ Scenario C: Typhlosion (p2) vs Zapdos (p1) ============
    println!("\n===== C. TYPHLOSION vs ZAPDOS (log sanity: FB did 100->60 ~40%) =====");
    {
        let p1 = vec![zapdos.clone(), snorlax.clone(), porygon2.clone()];
        let p2 = vec![typhlosion.clone(), snorlax.clone(), porygon2.clone()];
        let mut b = start(&dex, &p1, &p2);
        let zap = b.active_id(0).unwrap();
        let typ = b.active_id(1).unwrap();
        println!("  Typhlosion Fire Blast vs Zapdos (Zapdos maxhp {}):", b.poke(zap).maxhp);
        line(&mut b, &dex, typ, zap, "Fire Blast", "neut");
        println!("  Zapdos Thunderbolt vs Typhlosion (Typh maxhp {}):", b.poke(typ).maxhp);
        line(&mut b, &dex, zap, typ, "Thunderbolt", "neut");
    }
}
