//! Ported item callbacks (all 38 gen2stadium2 NC2000 items — see
//! reference/merged-items.txt).

use crate::dex::{Cb, Dex, HitEffect, ItemCbOp, ItemId};
use crate::state::*;

use super::dmg::HealEffect;
use super::events::{ev, EvTarget};
use super::{EffectHandle, RV};

#[cold]
fn unported_item_cb(dex: &Dex, item: ItemId, cb: Cb) -> ! {
    panic!("unported item callback: {} {}", dex.items.key(item), dex.cb_key(cb));
}

#[inline]
fn heal_berry_try_eat(b: &mut Battle, dex: &Dex, item: ItemId, target: PokeId, amount: f64) -> RV {
    let ok = b
        .run_event(
            dex,
            &ev::TryHeal,
            EvTarget::Poke(target),
            None,
            EffectHandle::Item(item),
            Some(RV::Num(amount)),
            false,
            false,
        )
        .truthy();
    if !ok { RV::False } else { RV::Undef }
}

#[inline]
fn status_berry_update(b: &mut Battle, dex: &Dex, target: PokeId, pred: impl FnOnce(Status) -> bool) -> RV {
    if pred(b.poke(target).status) {
        b.eat_item(dex, target, false, None, EffectHandle::None);
    }
    RV::Undef
}

#[inline]
fn status_berry_eat(b: &mut Battle, dex: &Dex, target: PokeId, pred: impl FnOnce(Status) -> bool) -> RV {
    if pred(b.poke(target).status) {
        b.cure_status(dex, target, false);
    }
    RV::Undef
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_item(
    b: &mut Battle,
    dex: &Dex,
    item: ItemId,
    cb: Cb,
    target: EvTarget,
    source: Option<PokeId>,
    source_effect: EffectHandle,
    relay: RV,
) -> RV {
    let tpoke = target.poke();
    let _ = source;

    match dex.items.get(item).cb_ops[cb.0 as usize] {
        ItemCbOp::TypeBoostModifyDamage => {
            if let Some(ty) = dex.items.get(item).type_boost {
                let move_type = b.active_move.as_ref().map(|m| m.move_type);
                if move_type == Some(ty) {
                    return RV::Num(relay.as_num() * 1.1);
                }
            }
            RV::Undef
        }
        // ------------------------------------------------------- crit items
        ItemCbOp::ScopeLensCrit => RV::Num(relay.as_num() + 1.0),
        ItemCbOp::StickCrit => {
            let user = tpoke.unwrap();
            if Some(b.poke(user).species) == dex.known_species.farfetchd {
                return RV::Num(3.0);
            }
            RV::Undef
        }
        ItemCbOp::LuckyPunchCrit => {
            let user = tpoke.unwrap();
            if Some(b.poke(user).species) == dex.known_species.chansey {
                return RV::Num(3.0);
            }
            RV::Undef
        }
        // ---------------------------------------------------- brightpowder
        ItemCbOp::BrightPowderAccuracy => match relay {
            RV::Num(acc) => RV::Num(acc - 20.0),
            _ => RV::Undef,
        },
        // ------------------------------------------------------- focusband
        ItemCbOp::FocusBandDamage => {
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
        ItemCbOp::KingsRockModifyMove => {
            let has_kr = b
                .active_move
                .as_ref()
                .and_then(|m| m.id)
                .is_some_and(|m| dex.move_static(m).kingsrock);
            if has_kr {
                if let Some(am) = b.active_move.as_mut() {
                    am.secondaries.push(HitEffect {
                        chance: Some(12),
                        volatile_status: Some("flinch"),
                        kingsrock: true,
                        ..Default::default()
                    });
                }
            }
            RV::Undef
        }
        // ------------------------------------------------------- leftovers
        ItemCbOp::LeftoversResidual => {
            let t = tpoke.unwrap();
            let amount = b.poke(t).base_maxhp as f64 / 16.0;
            b.heal(dex, amount, None, None, HealEffect::Effect(EffectHandle::None));
            RV::Undef
        }
        // ------------------------------------------------------ berryjuice
        ItemCbOp::BerryJuiceResidual => {
            let t = tpoke.unwrap();
            if b.poke(t).hp as f64 <= b.poke(t).maxhp as f64 / 2.0 {
                let ok = b
                    .run_event(
                        dex,
                        &ev::TryHeal,
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
        ItemCbOp::BerryResidual => {
            let t = tpoke.unwrap();
            if b.poke(t).hp as f64 <= b.poke(t).maxhp as f64 / 2.0 {
                b.eat_item(dex, t, false, None, EffectHandle::None);
            }
            RV::Undef
        }
        ItemCbOp::BerryTryEat => heal_berry_try_eat(b, dex, item, tpoke.unwrap(), 10.0),
        ItemCbOp::GoldBerryTryEat => heal_berry_try_eat(b, dex, item, tpoke.unwrap(), 30.0),
        ItemCbOp::BerryEat => {
            b.heal(dex, 10.0, None, None, HealEffect::Effect(EffectHandle::None));
            RV::Undef
        }
        ItemCbOp::GoldBerryEat => {
            b.heal(dex, 30.0, None, None, HealEffect::Effect(EffectHandle::None));
            RV::Undef
        }
        // -------------------------------------------------- status berries
        ItemCbOp::BitterBerryUpdate => {
            let t = tpoke.unwrap();
            if crate::cond_id!(dex, "confusion").map(|c| b.poke(t).has_volatile(c)).unwrap_or(false) {
                b.eat_item(dex, t, false, None, EffectHandle::None);
            }
            RV::Undef
        }
        ItemCbOp::BitterBerryEat => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "confusion");
            RV::Undef
        }
        ItemCbOp::BurntBerryUpdate => status_berry_update(b, dex, tpoke.unwrap(), |s| s == Status::Frz),
        ItemCbOp::IceBerryUpdate => status_berry_update(b, dex, tpoke.unwrap(), |s| s == Status::Brn),
        ItemCbOp::PrzCureBerryUpdate => status_berry_update(b, dex, tpoke.unwrap(), |s| s == Status::Par),
        ItemCbOp::PsnCureBerryUpdate => {
            status_berry_update(b, dex, tpoke.unwrap(), |s| s == Status::Psn || s == Status::Tox)
        }
        // gen2stadium2nc2000: Mint/Miracle Berry trigger onBeforeMove
        // (priority 11, before the sleep check) + onResidual, not onUpdate.
        ItemCbOp::MintBerryCheck => status_berry_update(b, dex, tpoke.unwrap(), |s| s == Status::Slp),
        ItemCbOp::BurntBerryEat => status_berry_eat(b, dex, tpoke.unwrap(), |s| s == Status::Frz),
        ItemCbOp::IceBerryEat => status_berry_eat(b, dex, tpoke.unwrap(), |s| s == Status::Brn),
        ItemCbOp::MintBerryEat => status_berry_eat(b, dex, tpoke.unwrap(), |s| s == Status::Slp),
        ItemCbOp::PrzCureBerryEat => status_berry_eat(b, dex, tpoke.unwrap(), |s| s == Status::Par),
        ItemCbOp::PsnCureBerryEat => {
            status_berry_eat(b, dex, tpoke.unwrap(), |s| s == Status::Psn || s == Status::Tox)
        }
        ItemCbOp::MiracleBerryCheck => {
            let t = tpoke.unwrap();
            let has_conf =
                crate::cond_id!(dex, "confusion").map(|c| b.poke(t).has_volatile(c)).unwrap_or(false);
            if b.poke(t).status != Status::None || has_conf {
                b.eat_item(dex, t, false, None, EffectHandle::None);
            }
            RV::Undef
        }
        ItemCbOp::MiracleBerryEat => {
            let t = tpoke.unwrap();
            b.cure_status(dex, t, false);
            b.remove_volatile(dex, t, "confusion");
            RV::Undef
        }
        // ---------------------------------------------------- mysteryberry
        ItemCbOp::MysteryBerryUpdate => {
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
                    let lb = crate::cond_id!(dex, "leppaberry").unwrap();
                    if let Some(vs) = b.poke_mut(t).volatile_mut(lb) {
                        vs.slot_ref = Some(i as u8);
                    }
                    b.eat_item(dex, t, false, None, EffectHandle::None);
                }
            }
            RV::Undef
        }
        ItemCbOp::MysteryBerryEat => {
            let t = tpoke.unwrap();
            let lb = crate::cond_id!(dex, "leppaberry").unwrap();
            let slot_idx = if b.poke(t).has_volatile(lb) {
                let idx = b.poke(t).volatile(lb).and_then(|v| v.slot_ref.map(usize::from));
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
        ItemCbOp::MailTakeItem => {
            let Some(am) = b.active_move.as_ref() else { return RV::False };
            let move_key = am.id.map(|m| dex.moves.key(m)).unwrap_or("");
            if !matches!(move_key, "knockoff" | "thief" | "covet") {
                return RV::False;
            }
            RV::Undef
        }
        // ----------------------------------------------------- berserkgene
        ItemCbOp::BerserkGeneUpdate => {
            let t = tpoke.unwrap();
            if b.use_item(dex, t, None, EffectHandle::None) {
                b.add_volatile(dex, t, "confusion", None, EffectHandle::None);
            }
            RV::Undef
        }
        ItemCbOp::Unported => unported_item_cb(dex, item, cb),
    }
}
