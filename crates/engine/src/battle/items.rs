//! Ported item callbacks (all 38 gen2stadium2 NC2000 items — see
//! reference/merged-items.txt).

use crate::dex::{Dex, HitEffect, ItemId};
use crate::state::*;

use super::dmg::HealEffect;
use super::events::EvTarget;
use super::{EffectHandle, RV};

/// The 1.1x type-boost items.
fn type_boost(item_key: &str) -> Option<&'static str> {
    Some(match item_key {
        "blackbelt" => "Fighting",
        "blackglasses" => "Dark",
        "charcoal" => "Fire",
        "dragonfang" => "Dragon",
        "hardstone" => "Rock",
        "magnet" => "Electric",
        "metalcoat" => "Steel",
        "miracleseed" => "Grass",
        "mysticwater" => "Water",
        "nevermeltice" => "Ice",
        "pinkbow" => "Normal",
        "poisonbarb" => "Poison",
        "polkadotbow" => "Normal",
        "sharpbeak" => "Flying",
        "silverpowder" => "Bug",
        "softsand" => "Ground",
        "spelltag" => "Ghost",
        "twistedspoon" => "Psychic",
        _ => return None,
    })
}

const KINGSROCK_MOVES: [&str; 91] = [
    "absorb", "aeroblast", "barrage", "beatup", "bide", "bonerush", "bonemerang", "cometpunch",
    "counter", "crabhammer", "crosschop", "cut", "dig", "doublekick", "doubleslap", "doubleedge",
    "dragonrage", "drillpeck", "eggbomb", "explosion", "extremespeed", "falseswipe", "feintattack",
    "flail", "fly", "frustration", "furyattack", "furycutter", "furyswipes", "gigadrain",
    "hiddenpower", "highjumpkick", "hornattack", "hydropump", "jumpkick", "karatechop",
    "leechlife", "machpunch", "magnitude", "megadrain", "megakick", "megapunch", "megahorn",
    "mirrorcoat", "nightshade", "outrage", "payday", "peck", "petaldance", "pinmissile", "pound",
    "present", "pursuit", "psywave", "quickattack", "rage", "rapidspin", "razorleaf", "razorwind",
    "return", "reversal", "rockthrow", "rollout", "scratch", "seismictoss", "selfdestruct",
    "skullbash", "skyattack", "slam", "slash", "snore", "solarbeam", "sonicboom", "spikecannon",
    "strength", "struggle", "submission", "superfang", "surf", "swift", "tackle", "takedown",
    "thief", "thrash", "triplekick", "twineedle", "visegrip", "vinewhip", "watergun", "waterfall",
    "wingattack",
];

#[allow(clippy::too_many_arguments)]
pub fn dispatch_item(
    b: &mut Battle,
    dex: &Dex,
    item: ItemId,
    cb: &str,
    target: EvTarget,
    source: Option<PokeId>,
    source_effect: EffectHandle,
    relay: RV,
) -> RV {
    let key = dex.items.key(item).to_string();
    let tpoke = target.poke();
    let _ = source;

    // 1.1x type-boost items
    if let (Some(ty), "onModifyDamage") = (type_boost(&key), cb) {
        let move_type = b.active_move.as_ref().map(|m| m.move_type.clone()).unwrap_or_default();
        if move_type == ty {
            return RV::Num(relay.as_num() * 1.1);
        }
        return RV::Undef;
    }

    match (key.as_str(), cb) {
        // ------------------------------------------------------- crit items
        ("scopelens", "onModifyCritRatio") => RV::Num(relay.as_num() + 1.0),
        ("stick", "onModifyCritRatio") => {
            let user = tpoke.unwrap();
            if dex.species.key(b.poke(user).species) == "farfetchd" {
                return RV::Num(3.0);
            }
            RV::Undef
        }
        ("luckypunch", "onModifyCritRatio") => {
            let user = tpoke.unwrap();
            if dex.species.get(b.poke(user).species).name == "Chansey" {
                return RV::Num(3.0);
            }
            RV::Undef
        }
        // ---------------------------------------------------- brightpowder
        ("brightpowder", "onModifyAccuracy") => match relay {
            RV::Num(acc) => RV::Num(acc - 20.0),
            _ => RV::Undef,
        },
        // ------------------------------------------------------- focusband
        ("focusband", "onDamage") => {
            let t = tpoke.unwrap();
            let damage = relay.as_num();
            let is_move = matches!(source_effect, EffectHandle::MoveEff(_));
            if b.prng.random_chance(30, 256) && damage >= b.poke(t).hp as f64 && is_move {
                let ts = b.poke_str(t);
                b.add(&["-activate", &ts, "item: Focus Band"]);
                return RV::Num(b.poke(t).hp as f64 - 1.0);
            }
            RV::Undef
        }
        // ------------------------------------------------------- kingsrock
        ("kingsrock", "onModifyMove") => {
            let move_key = b
                .active_move
                .as_ref()
                .and_then(|m| m.id)
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            if KINGSROCK_MOVES.contains(&move_key.as_str()) {
                if let Some(am) = b.active_move.as_mut() {
                    am.secondaries.push(HitEffect {
                        chance: Some(12),
                        volatile_status: Some("flinch".to_string()),
                        kingsrock: true,
                        ..Default::default()
                    });
                }
            }
            RV::Undef
        }
        // ------------------------------------------------------- leftovers
        ("leftovers", "onResidual") => {
            let t = tpoke.unwrap();
            let amount = b.poke(t).base_maxhp as f64 / 16.0;
            b.heal(dex, amount, None, None, HealEffect::Effect(EffectHandle::None));
            RV::Undef
        }
        // ------------------------------------------------------ berryjuice
        ("berryjuice", "onResidual") => {
            let t = tpoke.unwrap();
            if b.poke(t).hp as f64 <= b.poke(t).maxhp as f64 / 2.0 {
                let ok = b
                    .run_event(
                        dex,
                        "TryHeal",
                        EvTarget::Poke(t),
                        None,
                        EffectHandle::Item(item),
                        Some(RV::Num(20.0)),
                        false,
                        false,
                    )
                    .truthy();
                if ok && b.use_item(dex, t, None, EffectHandle::None) {
                    b.heal(dex, 20.0, None, None, HealEffect::Effect(EffectHandle::None));
                }
            }
            RV::Undef
        }
        // ---------------------------------------------------- heal berries
        ("berry", "onResidual") | ("goldberry", "onResidual") => {
            let t = tpoke.unwrap();
            if b.poke(t).hp as f64 <= b.poke(t).maxhp as f64 / 2.0 {
                b.eat_item(dex, t, false, None, EffectHandle::None);
            }
            RV::Undef
        }
        ("berry", "onTryEatItem") | ("goldberry", "onTryEatItem") => {
            let t = tpoke.unwrap();
            let amount = if key == "berry" { 10.0 } else { 30.0 };
            let ok = b
                .run_event(
                    dex,
                    "TryHeal",
                    EvTarget::Poke(t),
                    None,
                    EffectHandle::Item(item),
                    Some(RV::Num(amount)),
                    false,
                    false,
                )
                .truthy();
            if !ok {
                return RV::False;
            }
            RV::Undef
        }
        ("berry", "onEat") | ("goldberry", "onEat") => {
            let amount = if key == "berry" { 10.0 } else { 30.0 };
            b.heal(dex, amount, None, None, HealEffect::Effect(EffectHandle::None));
            RV::Undef
        }
        // -------------------------------------------------- status berries
        ("bitterberry", "onUpdate") => {
            let t = tpoke.unwrap();
            if dex.conds_id("confusion").map(|c| b.poke(t).has_volatile(c)).unwrap_or(false) {
                b.eat_item(dex, t, false, None, EffectHandle::None);
            }
            RV::Undef
        }
        ("bitterberry", "onEat") => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "confusion");
            RV::Undef
        }
        ("burntberry", "onUpdate") | ("iceberry", "onUpdate") | ("mintberry", "onUpdate")
        | ("przcureberry", "onUpdate") | ("psncureberry", "onUpdate") => {
            let t = tpoke.unwrap();
            let status = b.poke(t).status;
            let fire = match key.as_str() {
                "burntberry" => status == Status::Frz,
                "iceberry" => status == Status::Brn,
                "mintberry" => status == Status::Slp,
                "przcureberry" => status == Status::Par,
                _ => status == Status::Psn || status == Status::Tox,
            };
            if fire {
                b.eat_item(dex, t, false, None, EffectHandle::None);
            }
            RV::Undef
        }
        ("burntberry", "onEat") | ("iceberry", "onEat") | ("mintberry", "onEat")
        | ("przcureberry", "onEat") | ("psncureberry", "onEat") => {
            let t = tpoke.unwrap();
            let status = b.poke(t).status;
            let fire = match key.as_str() {
                "burntberry" => status == Status::Frz,
                "iceberry" => status == Status::Brn,
                "mintberry" => status == Status::Slp,
                "przcureberry" => status == Status::Par,
                _ => status == Status::Psn || status == Status::Tox,
            };
            if fire {
                b.cure_status(dex, t, false);
            }
            RV::Undef
        }
        ("miracleberry", "onUpdate") => {
            let t = tpoke.unwrap();
            let has_conf =
                dex.conds_id("confusion").map(|c| b.poke(t).has_volatile(c)).unwrap_or(false);
            if b.poke(t).status != Status::None || has_conf {
                b.eat_item(dex, t, false, None, EffectHandle::None);
            }
            RV::Undef
        }
        ("miracleberry", "onEat") => {
            let t = tpoke.unwrap();
            b.cure_status(dex, t, false);
            b.remove_volatile(dex, t, "confusion");
            RV::Undef
        }
        // ---------------------------------------------------- mysteryberry
        ("mysteryberry", "onUpdate") => {
            let t = tpoke.unwrap();
            if b.poke(t).hp <= 0 {
                return RV::Undef;
            }
            let slot_idx = b.poke(t).last_move.and_then(|lm| {
                b.poke(t).move_slots.iter().position(|s| s.id == lm)
            });
            if let Some(i) = slot_idx {
                if b.poke(t).move_slots[i].pp == 0 {
                    b.add_volatile(dex, t, "leppaberry", None, EffectHandle::None);
                    let lb = dex.conds_id("leppaberry").unwrap();
                    if let Some(vs) = b.poke_mut(t).volatile_mut(lb) {
                        vs.slot_ref = Some(i);
                    }
                    b.eat_item(dex, t, false, None, EffectHandle::None);
                }
            }
            RV::Undef
        }
        ("mysteryberry", "onEat") => {
            let t = tpoke.unwrap();
            let lb = dex.conds_id("leppaberry").unwrap();
            let slot_idx = if b.poke(t).has_volatile(lb) {
                let idx = b.poke(t).volatile(lb).and_then(|v| v.slot_ref);
                b.remove_volatile(dex, t, "leppaberry");
                idx
            } else {
                // lowest-pp slot
                let mut best: Option<usize> = None;
                let mut pp = 99;
                for (i, s) in b.poke(t).move_slots.iter().enumerate() {
                    if s.pp < pp {
                        best = Some(i);
                        pp = s.pp;
                    }
                }
                best
            };
            if let Some(i) = slot_idx {
                let (move_id, shared) = {
                    let p = b.poke_mut(t);
                    let s = &mut p.move_slots[i];
                    s.pp += 5;
                    if s.pp > s.maxpp {
                        s.pp = s.maxpp;
                    }
                    (s.id, s.shared)
                };
                if shared {
                    let pp = b.poke(t).move_slots[i].pp;
                    if let Some(base) =
                        b.poke_mut(t).base_move_slots.iter_mut().find(|m| m.id == move_id)
                    {
                        base.pp = pp;
                    }
                }
                let ts = b.poke_str(t);
                let move_name = dex.move_static(move_id).name.clone();
                b.add(&["-activate", &ts, "item: Mystery Berry", &move_name]);
            }
            RV::Undef
        }
        // ------------------------------------------------------------ mail
        ("mail", "onTakeItem") => {
            let Some(am) = b.active_move.as_ref() else { return RV::False };
            let move_key = am.id.map(|m| dex.moves.key(m)).unwrap_or("");
            if !matches!(move_key, "knockoff" | "thief" | "covet") {
                return RV::False;
            }
            RV::Undef
        }
        // ----------------------------------------------------- berserkgene
        ("berserkgene", "onUpdate") => {
            let t = tpoke.unwrap();
            if b.use_item(dex, t, None, EffectHandle::None) {
                b.add_volatile(dex, t, "confusion", None, EffectHandle::None);
            }
            RV::Undef
        }
        _ => panic!("unported item callback: {key} {cb}"),
    }
}
