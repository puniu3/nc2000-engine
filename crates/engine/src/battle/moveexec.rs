//! The move pipeline: gen2 `runMove` → base `useMove` → gen3 `useMoveInner`
//! → gen2stadium2 `tryMoveHit` → gen2 `moveHit` → gen2stadium2 `getDamage`.

use crate::dex::{Accuracy, Category, Dex, HitEffect, Multihit, MoveId};
use crate::state::*;

use super::conditions::DamageEffect;
use super::events::EvTarget;
use super::{clamp_int_range, EffectHandle, RV};

pub fn move_has_callback(dex: &Dex, m: MoveId, callback_name: &str) -> bool {
    dex.move_static(m).callbacks.iter().any(|c| c == callback_name)
}

/// Active-move-aware callback check: synthetic/modified ActiveMoves carry
/// their own callback list (curse deletes onHit; bide/futuresight synthetics
/// have none), which overrides the dex static list.
pub fn active_move_has_callback(b: &Battle, dex: &Dex, m: MoveId, callback_name: &str) -> bool {
    if let Some(am) = &b.active_move {
        if am.id == Some(m) {
            return am.has_callbacks.iter().any(|c| c == callback_name);
        }
    }
    move_has_callback(dex, m, callback_name)
}

/// Build an ActiveMove from the dex (PS getActiveMove).
pub fn get_active_move(dex: &Dex, id: MoveId) -> ActiveMove {
    let ms = dex.move_static(id);
    ActiveMove {
        id: Some(id),
        name: ms.name.clone(),
        move_type: ms.move_type.clone(),
        base_move_type: ms.move_type.clone(),
        category: ms.category,
        base_power: ms.base_power,
        accuracy: ms.accuracy,
        priority: ms.priority,
        target: ms.target.clone(),
        crit_ratio: ms.crit_ratio,
        will_crit: ms.will_crit,
        status: ms.status.clone(),
        volatile_status: ms.volatile_status.clone(),
        side_condition: ms.side_condition.clone(),
        weather: ms.weather.clone(),
        pseudo_weather: ms.pseudo_weather.clone(),
        boosts: ms.boosts.clone(),
        has_boosts: ms.has_boosts,
        heal: ms.heal,
        drain: ms.drain,
        recoil: ms.recoil,
        struggle_recoil: ms.struggle_recoil,
        multihit: ms.multihit.clone(),
        secondaries: ms.secondaries.clone(),
        self_effect: ms.self_effect.clone(),
        damage: ms.damage.clone(),
        ohko: ms.ohko,
        selfdestruct: ms.selfdestruct,
        self_switch: ms.self_switch.clone(),
        force_switch: ms.force_switch,
        ignore_immunity: ms.ignore_immunity,
        ignore_accuracy: ms.ignore_accuracy,
        ignore_evasion: ms.ignore_evasion,
        ignore_positive_evasion: ms.ignore_positive_evasion,
        ignore_offensive: ms.ignore_offensive,
        ignore_defensive: ms.ignore_defensive,
        sleep_usable: ms.sleep_usable,
        no_damage_variance: ms.no_damage_variance,
        always_hit: ms.always_hit,
        thaws_target: ms.thaws_target,
        stalling_move: ms.stalling_move,
        non_ghost_target: ms.non_ghost_target.clone(),
        flags: ms.flags.clone(),
        has_callbacks: ms.callbacks.clone(),
        hit: 0,
        last_hit: false,
        total_damage: None,
        source_effect: None,
        is_confusion_self_hit: false,
        spread_hit: false,
        magnitude: None,
        allies: None,
        status_roll: None,
        on_hit_suppressed: false,
        infiltrates: false,
        move_hit_data: Vec::new(),
    }
}

/// The confusion self-hit fake move.
pub fn synthetic_move(base_power: i32) -> ActiveMove {
    ActiveMove {
        id: None,
        name: String::new(),
        move_type: "???".into(),
        base_move_type: "???".into(),
        category: Category::Physical,
        base_power,
        accuracy: Accuracy::AlwaysHits,
        priority: 0,
        target: "normal".into(),
        crit_ratio: 0,
        will_crit: Some(false),
        status: None,
        volatile_status: None,
        side_condition: None,
        weather: None,
        pseudo_weather: None,
        boosts: Vec::new(),
        has_boosts: false,
        heal: None,
        drain: None,
        recoil: None,
        struggle_recoil: false,
        multihit: None,
        secondaries: Vec::new(),
        self_effect: None,
        damage: None,
        ohko: false,
        selfdestruct: false,
        self_switch: None,
        force_switch: false,
        ignore_immunity: false,
        ignore_accuracy: false,
        ignore_evasion: false,
        ignore_positive_evasion: false,
        ignore_offensive: false,
        ignore_defensive: false,
        sleep_usable: false,
        no_damage_variance: true,
        always_hit: false,
        thaws_target: false,
        stalling_move: false,
        non_ghost_target: None,
        flags: Vec::new(),
        has_callbacks: Vec::new(),
        hit: 0,
        last_hit: false,
        total_damage: None,
        source_effect: None,
        is_confusion_self_hit: true,
        spread_hit: false,
        magnitude: None,
        allies: None,
        status_roll: None,
        on_hit_suppressed: false,
        infiltrates: false,
        move_hit_data: Vec::new(),
    }
}

/// gen2stadium2 dex type iteration order (dex.types.names()).
const TYPE_NAMES: [&str; 17] = [
    "Fire", "Ice", "Steel", "Electric", "Ghost", "Grass", "Dark", "Bug", "Dragon", "Fighting",
    "Flying", "Ground", "Normal", "Poison", "Psychic", "Rock", "Water",
];

/// damageCallback dispatch (counter/mirrorcoat/psywave/superfang).
fn damage_callback(b: &mut Battle, dex: &Dex, m: MoveId, source: PokeId, target: PokeId) -> DamageResult {
    match dex.moves.key(m) {
        "counter" | "mirrorcoat" => {
            let want_physical = dex.moves.key(m) == "counter";
            let Some(last) = b.get_last_attacked_by(source).cloned() else {
                return DamageResult::False;
            };
            if last.move_id == MoveId(u16::MAX) || !last.this_turn {
                return DamageResult::False;
            }
            let Some(last_move_of_source) = b.poke(last.source).last_move else {
                return DamageResult::False;
            };
            if last.move_id != last_move_of_source {
                return DamageResult::False;
            }
            let cat = dex.move_static(last.move_id).category;
            let is_physical = cat == Category::Physical;
            if is_physical == want_physical && cat != Category::Status {
                return DamageResult::Damage(2.0 * last.damage as f64);
            }
            DamageResult::False
        }
        "psywave" => {
            let level = b.poke(source).level as u32;
            let hi = level + level / 2;
            DamageResult::Damage(b.prng.random_range(1, hi) as f64)
        }
        "superfang" => {
            let d = clamp_int_range(b.poke(target).hp as f64 / 2.0, Some(1.0), None);
            DamageResult::Damage(d)
        }
        other => panic!("unported damageCallback: {other}"),
    }
}

/// basePowerCallback dispatch. None = JS null (move fails).
fn base_power_callback(
    b: &mut Battle,
    dex: &Dex,
    m: MoveId,
    source: PokeId,
    target: PokeId,
    base_power: i32,
) -> Option<i32> {
    match dex.moves.key(m) {
        "flail" | "reversal" => {
            let p = b.poke(source);
            let ratio = ((p.hp as f64 * 48.0 / p.maxhp as f64).floor() as i32).max(1);
            Some(match ratio {
                r if r < 2 => 200,
                r if r < 5 => 150,
                r if r < 10 => 100,
                r if r < 17 => 80,
                r if r < 33 => 40,
                _ => 20,
            })
        }
        "frustration" => {
            let bp = (255 - b.poke(source).happiness as i32) * 10 / 25;
            if bp == 0 { None } else { Some(bp) }
        }
        "return" => {
            let bp = b.poke(source).happiness as i32 * 10 / 25;
            if bp == 0 { None } else { Some(bp) }
        }
        "triplekick" => {
            let hit = b.active_move.as_ref().map(|am| am.hit).unwrap_or(0);
            Some(10 * hit)
        }
        "pursuit" => {
            let t = b.poke(target);
            if t.being_called_back || t.switch_flag.is_set() {
                Some(base_power * 2)
            } else {
                Some(base_power)
            }
        }
        "beatup" => {
            let allies_len = b
                .active_move
                .as_ref()
                .and_then(|am| am.allies.as_ref())
                .map(|a| a.len())
                .unwrap_or(0);
            if allies_len == 0 { None } else { Some(10) }
        }
        "hiddenpower" => Some(b.poke(source).hp_power),
        "furycutter" => {
            let fc = dex.conds_id("furycutter").unwrap();
            let hit = b.active_move.as_ref().map(|am| am.hit).unwrap_or(0);
            if !b.poke(source).has_volatile(fc) || hit == 1 {
                b.add_volatile(dex, source, "furycutter", None, EffectHandle::None);
            }
            let mult = b.poke(source).volatile(fc).map(|v| v.get_int("multiplier")).unwrap_or(1);
            Some(clamp_int_range(base_power as f64 * mult as f64, Some(1.0), Some(160.0)) as i32)
        }
        "rollout" => {
            let ro = dex.conds_id("rollout").unwrap();
            let dc = dex.conds_id("defensecurl").unwrap();
            let mut bp = base_power as i64;
            let has_rollout = b.poke(source).has_volatile(ro);
            if has_rollout {
                let (hit_count, contact) = {
                    let v = b.poke(source).volatile(ro).unwrap();
                    (v.get_int("hitCount"), v.get_int("contactHitCount"))
                };
                if hit_count != 0 {
                    bp *= 1i64 << contact;
                }
                if b.poke(source).status != Status::Slp {
                    let v = b.poke_mut(source).volatile_mut(ro).unwrap();
                    v.set_int("hitCount", hit_count + 1);
                    v.set_int("contactHitCount", contact + 1);
                    if hit_count + 1 < 5 {
                        v.duration = Some(2);
                    }
                }
            }
            if b.poke(source).has_volatile(dc) {
                bp *= 2;
            }
            Some(bp as i32)
        }
        other => panic!("unported basePowerCallback: {other}"),
    }
}

/// beforeMoveCallback dispatch (bide). Returns true → abort runMove silently.
fn before_move_callback(
    b: &mut Battle,
    dex: &Dex,
    m: MoveId,
    pokemon: PokeId,
    _target: Option<PokeId>,
) -> bool {
    match dex.moves.key(m) {
        "bide" => {
            let bide = dex.conds_id("bide").unwrap();
            b.poke(pokemon).has_volatile(bide)
        }
        other => panic!("unported beforeMoveCallback: {other}"),
    }
}

/// Dispatch a move's own callbacks used as effect handlers.
pub fn dispatch_move_callback(
    b: &mut Battle,
    dex: &Dex,
    m: MoveId,
    callback_name: &str,
    target: EvTarget,
    source: Option<PokeId>,
    relay: RV,
) -> RV {
    let key = dex.moves.key(m).to_string();
    let move_eff = EffectHandle::MoveEff(m);
    let tpoke = target.poke();
    match (key.as_str(), callback_name) {
        // ---------------------------------------------------- onModifyMove
        ("struggle", "onModifyMove") => {
            if let Some(am) = b.active_move.as_mut() {
                am.move_type = "???".into();
            }
            RV::Undef
        }
        ("thunder", "onModifyMove") => {
            // target?.effectiveWeather() — the foe (= the event's source)
            let weather = match source {
                Some(t) => b.effective_weather(t),
                None => String::new(),
            };
            if let Some(am) = b.active_move.as_mut() {
                match weather.as_str() {
                    "raindance" => am.accuracy = Accuracy::AlwaysHits,
                    "sunnyday" => am.accuracy = Accuracy::Pct(50),
                    _ => {}
                }
            }
            RV::Undef
        }
        ("hiddenpower", "onModifyMove") => {
            let user = tpoke.unwrap();
            let hp_type = b.poke(user).hp_type.clone();
            const SPECIAL: [&str; 8] =
                ["Fire", "Water", "Grass", "Ice", "Electric", "Dark", "Psychic", "Dragon"];
            if let Some(am) = b.active_move.as_mut() {
                am.move_type = hp_type.clone();
                am.category = if SPECIAL.contains(&hp_type.as_str()) {
                    Category::Special
                } else {
                    Category::Physical
                };
            }
            RV::Undef
        }
        ("curse", "onModifyMove") => {
            let user = tpoke.unwrap();
            if !b.poke(user).has_type("Ghost") {
                let non_ghost = b
                    .active_move
                    .as_ref()
                    .and_then(|am| am.non_ghost_target.clone())
                    .unwrap_or_else(|| "self".to_string());
                if let Some(am) = b.active_move.as_mut() {
                    am.volatile_status = None;
                    am.has_callbacks.retain(|c| c != "onHit");
                    am.self_effect = Some(HitEffect {
                        boosts: vec![(0, 1), (1, 1), (4, -1)],
                        ..Default::default()
                    });
                    am.target = non_ghost;
                }
            } else {
                let sub_blocked = source
                    .map(|t| {
                        dex.conds_id("substitute")
                            .map(|c| b.poke(t).has_volatile(c))
                            .unwrap_or(false)
                    })
                    .unwrap_or(false);
                if sub_blocked {
                    if let Some(am) = b.active_move.as_mut() {
                        am.volatile_status = None;
                        am.has_callbacks.retain(|c| c != "onHit");
                    }
                }
            }
            RV::Undef
        }
        ("magnitude", "onModifyMove") => {
            let i = b.prng.random(100);
            let (mag, bp) = match i {
                0..=4 => (4, 10),
                5..=14 => (5, 30),
                15..=34 => (6, 50),
                35..=64 => (7, 70),
                65..=84 => (8, 90),
                85..=94 => (9, 110),
                _ => (10, 150),
            };
            if let Some(am) = b.active_move.as_mut() {
                am.magnitude = Some(mag);
                am.base_power = bp;
            }
            RV::Undef
        }
        ("present", "onModifyMove") => {
            let rand = b.prng.random(10);
            if let Some(am) = b.active_move.as_mut() {
                if rand < 2 {
                    am.heal = Some((1, 4));
                    am.infiltrates = true;
                } else if rand < 6 {
                    am.base_power = 40;
                } else if rand < 9 {
                    am.base_power = 80;
                } else {
                    am.base_power = 120;
                }
            }
            RV::Undef
        }
        ("beatup", "onModifyMove") => {
            let user = tpoke.unwrap();
            let side = &b.sides[user.side as usize];
            let allies: Vec<PokeId> = side
                .party
                .iter()
                .map(|&slot| PokeId { side: user.side, slot })
                .filter(|&id| !b.poke(id).fainted && b.poke(id).status == Status::None)
                .collect();
            if let Some(am) = b.active_move.as_mut() {
                am.move_type = "???".into();
                am.category = Category::Special;
                am.multihit = Some(Multihit::Fixed(allies.len() as i32));
                am.allies = Some(allies);
            }
            RV::Undef
        }
        ("rollout", "onModifyMove") => {
            let user = tpoke.unwrap();
            let ro = dex.conds_id("rollout").unwrap();
            if b.poke(user).has_volatile(ro) || b.poke(user).status == Status::Slp || source.is_none()
            {
                return RV::Undef;
            }
            b.add_volatile(dex, user, "rollout", None, EffectHandle::None);
            let src_eff = b.active_move.as_ref().and_then(|am| am.source_effect);
            if src_eff.is_some() {
                if let Some(t) = source {
                    let loc = if t.side == user.side { -1 } else { 1 };
                    b.poke_mut(user).last_move_target_loc = Some(loc);
                }
            }
            RV::Undef
        }
        // ------------------------------------------------ onUseMoveMessage
        ("magnitude", "onUseMoveMessage") => {
            let user = tpoke.unwrap();
            let ps = b.poke_str(user);
            let mag = b.active_move.as_ref().and_then(|am| am.magnitude).unwrap_or(0);
            let mag_str = mag.to_string();
            b.add(&["-activate", &ps, "move: Magnitude", &mag_str]);
            RV::Undef
        }
        // ------------------------------------------------------ onTryMove (charge)
        ("dig", "onTryMove") | ("fly", "onTryMove") | ("razorwind", "onTryMove")
        | ("skullbash", "onTryMove") | ("skyattack", "onTryMove") | ("solarbeam", "onTryMove") => {
            let attacker = tpoke.unwrap();
            if b.remove_volatile(dex, attacker, &key) {
                return RV::Undef;
            }
            let ps = b.poke_str(attacker);
            let move_name = dex.move_static(m).name.clone();
            b.add(&["-prepare", &ps, &move_name]);
            if key == "skullbash" {
                b.boost(dex, &[(1, 1)], Some(attacker), Some(attacker), move_eff);
            }
            if key == "solarbeam" {
                let w = b.effective_weather(attacker);
                if w == "sunnyday" {
                    b.attr_last_move(&["[still]"]);
                    let ps = b.poke_str(attacker);
                    let defender_str = source.map(|d| b.poke_str(d)).unwrap_or_default();
                    b.add_move(&["-anim", &ps, &move_name, &defender_str]);
                    return RV::Undef;
                }
            }
            if !b
                .run_event(dex, "ChargeMove", EvTarget::Poke(attacker), source, move_eff, None, false, false)
                .truthy()
            {
                return RV::Undef;
            }
            b.add_volatile(dex, attacker, "twoturnmove", source, EffectHandle::None);
            RV::Null
        }
        // ---------------------------------------------------- onPrepareHit
        ("dig", "onPrepareHit") | ("fly", "onPrepareHit") | ("razorwind", "onPrepareHit")
        | ("skullbash", "onPrepareHit") | ("skyattack", "onPrepareHit")
        | ("solarbeam", "onPrepareHit") => {
            // (target, source): source = the attacker
            let attacker = source.unwrap();
            RV::from_bool(b.poke(attacker).status != Status::Slp)
        }
        ("protect", "onPrepareHit") | ("detect", "onPrepareHit") | ("endure", "onPrepareHit") => {
            let user = tpoke.unwrap();
            if !b.queue_will_act() {
                return RV::False;
            }
            b.run_event(dex, "StallMove", EvTarget::Poke(user), None, EffectHandle::None, None, false, false)
        }
        ("destinybond", "onPrepareHit") | ("perishsong", "onPrepareHit") => {
            let user = tpoke.unwrap();
            if b.sides[user.side as usize].pokemon_left == 1 {
                let name = dex.move_static(m).name.clone();
                b.hint(
                    &format!("In Pokemon Stadium 2, {name} fails if it is being used by your last Pokemon."),
                    false,
                );
                return RV::False;
            }
            RV::Undef
        }
        // ----------------------------------------------------------- onTry
        ("rest", "onTry") => {
            let user = tpoke.unwrap();
            if b.poke(user).hp < b.poke(user).maxhp {
                return RV::Undef;
            }
            let ps = b.poke_str(user);
            b.add(&["-fail", &ps]);
            RV::Null
        }
        ("sleeptalk", "onTry") | ("snore", "onTry") => {
            let user = tpoke.unwrap();
            RV::from_bool(b.poke(user).status == Status::Slp)
        }
        ("splash", "onTry") => RV::Undef, // gravity: no such pseudo-weather in gen2
        ("teleport", "onTry") => RV::False, // data constant `onTry: false`
        ("futuresight", "onTry") => {
            let user = tpoke.unwrap();
            let foe = source.unwrap();
            if !b.add_slot_condition(dex, foe, "futuremove", Some(user), move_eff).truthy() {
                return RV::False;
            }
            // damage precompute on a synthetic 80 BP special '???' move
            let mut fake = get_active_move(dex, m);
            fake.base_power = 80;
            fake.category = Category::Special;
            fake.move_type = "???".into();
            fake.will_crit = Some(false);
            fake.damage = None;
            fake.has_callbacks = Vec::new();
            let damage = get_damage_synthetic(b, dex, user, foe, fake);
            let fm = dex.conds_id("futuremove").unwrap();
            let loc = StateLoc::SlotCond(foe.side, 0, fm);
            if let Some(st) = b.state_at_mut(loc) {
                st.duration = Some(3);
                st.set("move", crate::state::Scalar::Str("futuresight".into()));
                st.source = Some(user);
                st.future_damage = damage;
            }
            let ps = b.poke_str(user);
            b.add(&["-start", &ps, "Future Sight"]);
            RV::Null
        }
        // -------------------------------------------------------- onTryHit
        ("substitute", "onTryHit") => {
            let user = tpoke.unwrap();
            let sub = dex.conds_id("substitute").unwrap();
            if b.poke(user).has_volatile(sub) {
                let ps = b.poke_str(user);
                b.add(&["-fail", &ps, "move: Substitute"]);
                return RV::Str(String::new()); // NOT_FAIL
            }
            if b.poke(user).hp as f64 <= b.poke(user).maxhp as f64 / 4.0 || b.poke(user).maxhp == 1 {
                let ps = b.poke_str(user);
                b.add(&["-fail", &ps, "move: Substitute", "[weak]"]);
                return RV::Str(String::new()); // NOT_FAIL
            }
            RV::Undef
        }
        ("curse", "onTryHit") => {
            let user = source.unwrap();
            let t = tpoke.unwrap();
            if !b.poke(user).has_type("Ghost") {
                if let Some(am) = b.active_move.as_mut() {
                    am.volatile_status = None;
                    am.has_callbacks.retain(|c| c != "onHit");
                    am.self_effect = Some(HitEffect {
                        boosts: vec![(4, -1), (0, 1), (1, 1)],
                        ..Default::default()
                    });
                }
            } else {
                let has_vs = b.active_move.as_ref().map(|am| am.volatile_status.is_some()).unwrap_or(false);
                let curse = dex.conds_id("curse").unwrap();
                if has_vs && b.poke(t).has_volatile(curse) {
                    return RV::False;
                }
            }
            RV::Undef
        }
        ("disable", "onTryHit") => {
            let t = tpoke.unwrap();
            match b.poke(t).last_move {
                None => RV::False,
                Some(lm) if dex.moves.key(lm) == "struggle" => RV::False,
                _ => RV::Undef,
            }
        }
        ("foresight", "onTryHit") => {
            let t = tpoke.unwrap();
            let fs = dex.conds_id("foresight").unwrap();
            if b.poke(t).has_volatile(fs) {
                return RV::False;
            }
            RV::Undef
        }
        ("lockon", "onTryHit") | ("mindreader", "onTryHit") => {
            let t = tpoke.unwrap();
            let fs = dex.conds_id("foresight").unwrap();
            let lo = dex.conds_id("lockon").unwrap();
            if b.poke(t).has_volatile(fs) || b.poke(t).has_volatile(lo) {
                return RV::False;
            }
            RV::Undef
        }
        ("roar", "onTryHit") | ("whirlwind", "onTryHit") => {
            for action in &b.queue {
                match &action.choice {
                    ActionKind::Move { .. } => return RV::False,
                    ActionKind::Switch { insta: false, .. } => return RV::False,
                    _ => {}
                }
            }
            RV::Undef
        }
        ("splash", "onTryHit") => {
            b.add(&["-nothing"]);
            RV::Undef
        }
        ("swagger", "onTryHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            if b.poke(t).boosts[0] >= 6 || b.get_stat(dex, t, 0, false, true, false) == 999 {
                let ps = b.poke_str(user);
                b.add(&["-miss", &ps]);
                return RV::Null;
            }
            RV::Undef
        }
        ("sleeptalk", "onTryHit") => {
            let user = tpoke.unwrap();
            let cl = dex.conds_id("choicelock");
            let en = dex.conds_id("encore").unwrap();
            let locked = cl.map(|c| b.poke(user).has_volatile(c)).unwrap_or(false)
                || b.poke(user).has_volatile(en);
            RV::from_bool(!locked)
        }
        // --------------------------------------------------- onTryImmunity
        ("attract", "onTryImmunity") => {
            let t = tpoke.unwrap();
            let s = source.unwrap();
            let tg = b.poke(t).gender.clone();
            let sg = b.poke(s).gender.clone();
            RV::from_bool((tg == "M" && sg == "F") || (tg == "F" && sg == "M"))
        }
        ("dreameater", "onTryImmunity") => {
            let t = tpoke.unwrap();
            let sub = dex.conds_id("substitute").unwrap();
            RV::from_bool(b.poke(t).status == Status::Slp && !b.poke(t).has_volatile(sub))
        }
        ("leechseed", "onTryImmunity") => {
            let t = tpoke.unwrap();
            RV::from_bool(!b.poke(t).has_type("Grass"))
        }
        // ------------------------------------------------------ onMoveFail
        ("highjumpkick", "onMoveFail") | ("jumpkick", "onMoveFail") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            if b.run_type_immunity(dex, t, "Fighting") {
                let damage = {
                    let saved = b.active_move.clone();
                    let d = b.get_damage(dex, user, t, true);
                    b.active_move = saved;
                    d
                };
                let d = match damage {
                    DamageResult::Damage(d) => d,
                    DamageResult::Zero => 0.0,
                    _ => panic!("Couldn't get {key} recoil"),
                };
                let amount = clamp_int_range(d / 8.0, Some(1.0), None);
                b.damage(dex, amount, Some(user), Some(user), DamageEffect::Effect(move_eff), false);
            }
            RV::Undef
        }
        ("outrage", "onMoveFail") | ("petaldance", "onMoveFail") | ("thrash", "onMoveFail") => {
            let user = source.unwrap();
            b.add_volatile(dex, user, "lockedmove", None, EffectHandle::None);
            RV::Undef
        }
        // ---------------------------------------------------------- onDamage
        ("falseswipe", "onDamage") => {
            let t = tpoke.unwrap();
            let damage = relay.as_num();
            if damage >= b.poke(t).hp as f64 {
                return RV::Num(b.poke(t).hp as f64 - 1.0);
            }
            RV::Undef
        }
        // ------------------------------------------------------ onHitField
        ("haze", "onHitField") => {
            b.add(&["-clearallboost"]);
            for id in b.get_all_active(false) {
                b.poke_mut(id).boosts = [0; 7];
                b.remove_volatile(dex, id, "brnattackdrop");
                b.remove_volatile(dex, id, "parspeeddrop");
            }
            RV::Undef
        }
        ("perishsong", "onHitField") => {
            let user = source.unwrap();
            let mut result = false;
            let mut message = false;
            for id in b.get_all_active(false) {
                let invuln = b.run_event(dex, "Invulnerability", EvTarget::Poke(id), Some(user), move_eff, None, false, false);
                if invuln == RV::False {
                    let ss = b.poke_str(user);
                    let ts = b.poke_str(id);
                    b.add(&["-miss", &ss, &ts]);
                    result = true;
                } else if b.run_event(dex, "TryHit", EvTarget::Poke(id), Some(user), move_eff, None, false, false) == RV::Null {
                    result = true;
                } else {
                    let ps_cond = dex.conds_id("perishsong").unwrap();
                    if !b.poke(id).has_volatile(ps_cond) {
                        b.add_volatile(dex, id, "perishsong", None, EffectHandle::None);
                        let ts = b.poke_str(id);
                        b.add(&["-start", &ts, "perish3", "[silent]"]);
                        result = true;
                        message = true;
                    }
                }
            }
            if !result {
                return RV::False;
            }
            if message {
                b.add(&["-fieldactivate", "move: Perish Song"]);
            }
            RV::Undef
        }
        // ------------------------------------------------------------ onHit
        ("bellydrum", "onHit") => {
            let t = tpoke.unwrap();
            if b.poke(t).boosts[0] >= 6 || b.poke(t).hp as f64 <= b.poke(t).maxhp as f64 / 2.0 {
                return RV::False;
            }
            let maxhp = b.poke(t).maxhp as f64;
            b.direct_damage(dex, maxhp / 2.0, Some(t), None, EffectHandle::None);
            // max-out atk in +2 steps, stopping when 999 is reached
            let original = b.poke(t).boosts[0];
            let mut current = original as i32;
            let mut loop_stage;
            while current < 6 {
                loop_stage = current;
                current += 1;
                if current < 6 {
                    current += 1;
                }
                b.poke_mut(t).boosts[0] = loop_stage as i8;
                if b.get_stat(dex, t, 0, false, true, false) < 999 {
                    b.poke_mut(t).boosts[0] = current as i8;
                    continue;
                }
                b.poke_mut(t).boosts[0] = (current - 1) as i8;
                break;
            }
            let boosts = b.poke(t).boosts[0] - original;
            b.poke_mut(t).boosts[0] = original;
            b.boost(dex, &[(0, boosts)], None, None, EffectHandle::None);
            RV::Undef
        }
        ("conversion", "onHit") => {
            let t = tpoke.unwrap();
            let mut possible: Vec<String> = Vec::new();
            for slot in b.poke(t).move_slots.clone() {
                let ms = dex.move_static(slot.id);
                if dex.moves.key(slot.id) != "curse" && !b.poke(t).has_type(&ms.move_type) {
                    possible.push(ms.move_type.clone());
                }
            }
            if possible.is_empty() {
                return RV::False;
            }
            let ty = possible[b.prng.sample_index(possible.len())].clone();
            b.set_type(t, vec![ty.clone()]);
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "typechange", &ty]);
            RV::Undef
        }
        ("conversion2", "onHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            let Some(last) = b.poke(t).last_move else { return RV::False };
            let attack_type = if dex.moves.key(last) == "struggle" {
                "Normal".to_string()
            } else {
                dex.move_static(last).move_type.clone()
            };
            let mut possible: Vec<&str> = Vec::new();
            for ty in TYPE_NAMES {
                let Some(tc) = dex.typechart.get(&crate::dex::toid(ty)) else { continue };
                let taken = tc.damage_taken.get(&attack_type).copied().unwrap_or(0);
                if taken == 2 || taken == 3 {
                    possible.push(ty);
                }
            }
            if possible.is_empty() {
                return RV::False;
            }
            let ty = possible[b.prng.sample_index(possible.len())].to_string();
            b.set_type(user, vec![ty.clone()]);
            let us = b.poke_str(user);
            b.add(&["-start", &us, "typechange", &ty]);
            RV::Undef
        }
        ("detect", "onHit") | ("protect", "onHit") | ("endure", "onHit") => {
            let t = tpoke.unwrap();
            b.add_volatile(dex, t, "stall", None, EffectHandle::None);
            RV::Undef
        }
        ("healbell", "onHit") => {
            let user = source.unwrap();
            let us = b.poke_str(user);
            b.add(&["-cureteam", &us, "[from] move: Heal Bell"]);
            let side_n = user.side as usize;
            let party = b.sides[side_n].party.clone();
            for slot in party {
                let id = PokeId { side: user.side, slot };
                b.pokemon_clear_status(dex, id);
            }
            RV::Undef
        }
        ("lockon", "onHit") | ("mindreader", "onHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            b.add_volatile(dex, user, "lockon", Some(t), EffectHandle::None);
            let us = b.poke_str(user);
            let of = format!("[of] {}", b.poke_str(t));
            let name = format!("move: {}", dex.move_static(m).name);
            b.add(&["-activate", &us, &name, &of]);
            RV::Undef
        }
        ("meanlook", "onHit") | ("spiderweb", "onHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            b.add_volatile_linked(dex, t, "trapped", Some(user), move_eff, "trapper")
        }
        ("metronome", "onHit") => {
            let user = tpoke.unwrap();
            let own: Vec<MoveId> = b.poke(user).move_slots.iter().map(|s| s.id).collect();
            let mut pool: Vec<(i32, MoveId)> = Vec::new();
            for i in 0..dex.moves.len() {
                let mid = MoveId(i as u16);
                let md = dex.moves.get(mid);
                if md.flags.contains_key("metronome") && !own.contains(&mid) {
                    pool.push((md.num, mid));
                }
            }
            pool.sort_by_key(|(num, _)| *num);
            if pool.is_empty() {
                return RV::False;
            }
            let pick = pool[b.prng.sample_index(pool.len())].1;
            b.use_move(dex, pick, user, None, None);
            RV::Undef
        }
        ("mimic", "onHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            let sub = dex.conds_id("substitute").unwrap();
            if b.poke(user).transformed || b.poke(t).last_move.is_none() || b.poke(t).has_volatile(sub)
            {
                return RV::False;
            }
            let last = b.poke(t).last_move.unwrap();
            let user_moves: Vec<MoveId> = b.poke(user).move_slots.iter().map(|s| s.id).collect();
            if dex.move_static(last).has_flag("failmimic") || user_moves.contains(&last) {
                return RV::False;
            }
            let mimic_id = dex.moves.id("mimic").unwrap();
            let Some(idx) = b.poke(user).move_slots.iter().position(|s| s.id == mimic_id) else {
                return RV::False;
            };
            let ms = dex.move_static(last);
            let pp_ups = if ms.no_pp_boosts { 0 } else { 3 };
            let mut maxpp = ms.pp * (5 + pp_ups) / 5;
            if ms.pp == 40 {
                maxpp -= pp_ups;
            }
            b.poke_mut(user).move_slots[idx] = MoveSlot {
                id: last,
                pp: ms.pp.min(5),
                maxpp,
                disabled: false,
                used: false,
                shared: false,
            };
            let us = b.poke_str(user);
            let name = ms.name.clone();
            b.add(&["-activate", &us, "move: Mimic", &name]);
            RV::Undef
        }
        ("mirrormove", "onHit") => {
            let user = tpoke.unwrap();
            const NO_MIRROR: [&str; 6] =
                ["metronome", "mimic", "mirrormove", "sketch", "sleeptalk", "transform"];
            let Some(foe) = b.active_id(1 - user.side as usize) else { return RV::False };
            let Some(last) = b.poke(foe).last_move else { return RV::False };
            let last_key = dex.moves.key(last);
            let own: Vec<MoveId> = b.poke(user).move_slots.iter().map(|s| s.id).collect();
            if NO_MIRROR.contains(&last_key) || own.contains(&last) {
                return RV::False;
            }
            b.use_move(dex, last, user, None, None);
            RV::Undef
        }
        ("moonlight", "onHit") | ("morningsun", "onHit") | ("synthesis", "onHit") => {
            let t = tpoke.unwrap();
            let w = b.effective_weather(t);
            let amount = if w == "sunnyday" {
                b.poke(t).maxhp as f64
            } else if matches!(w.as_str(), "raindance" | "sandstorm") {
                b.poke(t).base_maxhp as f64 / 4.0
            } else {
                b.poke(t).base_maxhp as f64 / 2.0
            };
            b.heal(dex, amount, None, None, super::dmg::HealEffect::Effect(EffectHandle::None));
            RV::Undef
        }
        ("painsplit", "onHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            let target_hp = b.poke(t).hp;
            let average = ((target_hp + b.poke(user).hp) / 2).max(1);
            let target_change = target_hp - average;
            b.set_hp(t, (b.poke(t).hp - target_change) as f64);
            let ts = b.poke_str(t);
            let (secret, shared) = b.get_health(t);
            let side_id = format!("p{}", t.side + 1);
            b.add_split(
                &side_id,
                &["-sethp", &ts, &secret, "[from] move: Pain Split", "[silent]"],
                &["-sethp", &ts, &shared, "[from] move: Pain Split", "[silent]"],
            );
            b.set_hp(user, average as f64);
            let us = b.poke_str(user);
            let (secret, shared) = b.get_health(user);
            let side_id = format!("p{}", user.side + 1);
            b.add_split(
                &side_id,
                &["-sethp", &us, &secret, "[from] move: Pain Split"],
                &["-sethp", &us, &shared, "[from] move: Pain Split"],
            );
            RV::Undef
        }
        ("payday", "onHit") => {
            b.add(&["-fieldactivate", "move: Pay Day"]);
            RV::Undef
        }
        ("psychup", "onHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            let boosts = b.poke(t).boosts;
            b.poke_mut(user).boosts = boosts;
            let us = b.poke_str(user);
            let ts = b.poke_str(t);
            b.add(&["-copyboost", &us, &ts, "[from] move: Psych Up"]);
            RV::Undef
        }
        ("rest", "onHit") => {
            let t = tpoke.unwrap();
            if b.poke(t).status != Status::Slp {
                if !b.set_status(dex, t, "slp", source, move_eff, false).truthy() {
                    return RV::Undef;
                }
            } else {
                let ts = b.poke_str(t);
                b.add(&["-status", &ts, "slp", "[from] move: Rest"]);
            }
            {
                let p = b.poke_mut(t);
                p.status_state.set_int("time", 3);
                p.status_state.set_int("startTime", 3);
                p.status_state.source = Some(t);
            }
            let maxhp = b.poke(t).maxhp as f64;
            b.heal(dex, maxhp, None, None, super::dmg::HealEffect::Effect(EffectHandle::None));
            RV::Undef
        }
        ("sketch", "onHit") => {
            b.add(&["-nothing"]);
            RV::Undef
        }
        ("sleeptalk", "onHit") => {
            let user = tpoke.unwrap();
            let mut moves: Vec<MoveId> = Vec::new();
            for slot in &b.poke(user).move_slots {
                let ms = dex.move_static(slot.id);
                if !ms.has_flag("nosleeptalk") && !ms.has_flag("charge") {
                    moves.push(slot.id);
                }
            }
            if moves.is_empty() {
                return RV::False;
            }
            let pick = moves[b.prng.sample_index(moves.len())];
            b.use_move(dex, pick, user, None, None);
            RV::Undef
        }
        ("spite", "onHit") => {
            let t = tpoke.unwrap();
            let roll = b.prng.random_range(2, 6) as i32;
            if let Some(last) = b.poke(t).last_move {
                if b.deduct_pp(t, last, roll) != 0 {
                    let ts = b.poke_str(t);
                    let move_key = dex.moves.key(last).to_string();
                    let roll_str = roll.to_string();
                    b.add(&["-activate", &ts, "move: Spite", &move_key, &roll_str]);
                    return RV::Undef;
                }
            }
            RV::False
        }
        ("transform", "onHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            RV::from_bool(b.transform_into(dex, user, t))
        }
        ("triattack", "onHit") => {
            const STATUSES: [&str; 3] = ["par", "frz", "brn"];
            let pick = STATUSES[b.prng.sample_index(3)];
            if let Some(am) = b.active_move.as_mut() {
                am.status_roll = Some(pick.to_string());
            }
            RV::Undef
        }
        ("batonpass", "onHit") => {
            let t = tpoke.unwrap();
            if !b.can_switch(t.side) {
                b.attr_last_move(&["[still]"]);
                let ts = b.poke_str(t);
                b.add(&["-fail", &ts]);
                return RV::Str(String::new()); // NOT_FAIL
            }
            RV::Undef
        }
        ("furycutter", "onHit") => {
            let user = source.unwrap();
            b.add_volatile(dex, user, "furycutter", None, EffectHandle::None);
            RV::Undef
        }
        ("curse", "onHit") => {
            let user = source.unwrap();
            let maxhp = b.poke(user).maxhp as f64;
            b.direct_damage(dex, maxhp / 2.0, Some(user), Some(user), EffectHandle::None);
            RV::Undef
        }
        // ----------------------------------------------- sub-block onHit
        ("thief", "secondary.onHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            if b.poke(user).item.is_some() {
                return RV::Undef;
            }
            let Some(item) = b.take_item(dex, t, Some(user)) else { return RV::Undef };
            if !b.set_item(dex, user, item) {
                b.poke_mut(t).item = Some(item);
                return RV::Undef;
            }
            let us = b.poke_str(user);
            let item_name = dex.items.get(item).name.clone();
            let of = format!("[of] {}", b.poke_str(t));
            b.add(&["-item", &us, &item_name, "[from] move: Thief", &of]);
            RV::Undef
        }
        ("triattack", "secondary.onHit") => {
            let t = tpoke.unwrap();
            let user = source.unwrap();
            let roll = b.active_move.as_ref().and_then(|am| am.status_roll.clone());
            if let Some(status) = roll {
                b.try_set_status(dex, t, &status, Some(user), EffectHandle::None);
            }
            RV::Undef
        }
        ("rapidspin", "self.onHit") => {
            let user = tpoke.unwrap();
            rapid_spin_cleanup(b, dex, user);
            RV::Undef
        }
        ("batonpass", "self.onHit") => {
            let user = tpoke.unwrap();
            b.poke_mut(user).skip_before_switch_out = true;
            RV::Undef
        }
        ("rapidspin", "onAfterHit") => {
            let user = source.unwrap();
            rapid_spin_cleanup(b, dex, user);
            RV::Undef
        }
        ("rollout", "onAfterMove") => {
            // rolloutstorage transfer: hitCount==5 && contactHitCount<5 is
            // unreachable in gen2 (both counters increment together).
            RV::Undef
        }
        _ => panic!("unported move callback: {key} {callback_name}"),
    }
}

/// futuremove.onEnd — Future Sight's delayed hit. Runs the MODERN
/// trySpreadMoveHit pipeline (gen4 hitStep overrides) against the stored
/// fixed damage; see reference/merged-conditions.txt futuremove + PS
/// sim/battle-actions.ts.
pub fn resolve_future_move(b: &mut Battle, dex: &Dex, state: StateLoc, target: PokeId) {
    let (source, damage) = {
        let Some(st) = b.state_at(state) else { return };
        (st.source, st.future_damage)
    };
    let Some(source) = source else { return };
    if b.poke(target).fainted || target == source {
        let what = if b.poke(target).fainted { "fainted" } else { "the user" };
        b.hint(&format!("Future Sight did not hit because the target is {what}."), false);
        return;
    }
    let ts = b.poke_str(target);
    b.add(&["-end", &ts, "move: Future Sight"]);
    b.remove_volatile(dex, target, "protect");
    b.remove_volatile(dex, target, "endure");

    let fs_id = dex.moves.id("futuresight").unwrap();
    let mut mv = get_active_move(dex, fs_id);
    mv.accuracy = Accuracy::Pct(90);
    mv.base_power = 0;
    mv.damage = Some(crate::dex::FixedDamage::Amount(damage.unwrap_or(0.0) as i32));
    mv.category = Category::Special;
    mv.move_type = "???".into();
    mv.will_crit = None;
    mv.crit_ratio = 0;
    mv.secondaries = Vec::new();
    mv.self_effect = None;
    mv.volatile_status = None;
    mv.flags = vec!["metronome".into(), "futuremove".into()];
    mv.has_callbacks = Vec::new();
    mv.ignore_immunity = false;
    let move_eff = EffectHandle::MoveEff(fs_id);

    b.set_active_move(mv, Some(source), Some(target));
    // singleEvent Try / PrepareHit: no callbacks on the hit move.
    let prep = b.run_event(dex, "PrepareHit", EvTarget::Poke(source), Some(target), move_eff, None, false, false);
    if !prep.truthy() {
        if prep == RV::False {
            let ss = b.poke_str(source);
            b.add(&["-fail", &ss]);
            b.attr_last_move(&["[still]"]);
        }
        b.clear_active_move(true);
        return;
    }
    'steps: {
        // 0. Invulnerability (gen4 override)
        let invuln = b.run_event(dex, "Invulnerability", EvTarget::Poke(target), Some(source), move_eff, None, false, false);
        if invuln == RV::False {
            b.attr_last_move(&["[miss]"]);
            let ss = b.poke_str(source);
            let ts = b.poke_str(target);
            b.add(&["-miss", &ss, &ts]);
            break 'steps;
        }
        // 1. type immunity: '???' — always passes.
        // 2. TryHit event
        let tryhit = b.run_event(dex, "TryHit", EvTarget::Poke(target), Some(source), move_eff, None, false, false);
        if !tryhit.truthy() {
            if tryhit == RV::False {
                let ss = b.poke_str(source);
                b.add(&["-fail", &ss]);
                b.attr_last_move(&["[still]"]);
            }
            break 'steps;
        }
        // 3. TryImmunity single: none.
        // 4. accuracy (gen4 override; boost tables, modern d100 roll)
        const BOOST_TABLE: [f64; 7] = [1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0, 7.0 / 3.0, 8.0 / 3.0, 3.0];
        let mut accuracy = 90.0f64;
        {
            let acc_boost = b.poke(source).boosts[5].clamp(-6, 6);
            if acc_boost > 0 {
                accuracy *= BOOST_TABLE[acc_boost as usize];
            } else {
                accuracy /= BOOST_TABLE[(-acc_boost) as usize];
            }
            // evasion (foresight zeroes positive evasion)
            let mut eva = b.poke(target).boosts[6].clamp(-6, 6);
            let fs_cond = dex.conds_id("foresight").unwrap();
            if eva > 0 && b.poke(target).has_volatile(fs_cond) {
                eva = 0;
            }
            if eva > 0 {
                accuracy /= BOOST_TABLE[eva as usize];
            } else if eva < 0 {
                accuracy *= BOOST_TABLE[(-eva) as usize];
            }
        }
        let rv = b.run_event(
            dex,
            "ModifyAccuracy",
            EvTarget::Poke(target),
            Some(source),
            move_eff,
            Some(RV::Num(accuracy)),
            false,
            false,
        );
        accuracy = rv.as_num();
        let acc_rv = b.run_event(
            dex,
            "Accuracy",
            EvTarget::Poke(target),
            Some(source),
            move_eff,
            Some(RV::Num(accuracy)),
            false,
            false,
        );
        let hit = match acc_rv {
            RV::True => true,
            RV::Num(a) => (b.prng.random(100) as f64) < a,
            _ => false,
        };
        if !hit {
            b.attr_last_move(&["[miss]"]);
            let ss = b.poke_str(source);
            let ts = b.poke_str(target);
            b.add(&["-miss", &ss, &ts]);
            break 'steps;
        }
        // 7. move hit loop (single hit)
        b.active_move.as_mut().unwrap().total_damage = Some(0);
        b.poke_mut(source).last_damage = 0;
        b.active_move.as_mut().unwrap().hit = 1;
        b.active_move.as_mut().unwrap().last_hit = true;
        // spreadMoveHit: TryPrimaryHit (substitute), getDamage, spreadDamage,
        // Hit event; no self/secondaries/forceSwitch.
        let mut cur_target = Some(target);
        let rv = b.run_event(dex, "TryPrimaryHit", EvTarget::Poke(target), Some(source), move_eff, None, false, false);
        if rv == RV::Num(0.0) {
            cur_target = None;
        } else if !rv.truthy() {
            cur_target = None;
        }
        let mut dealt: Option<f64> = None;
        if let Some(t) = cur_target {
            let calc = b.get_damage(dex, source, t, false);
            if let DamageResult::Damage(d) = calc {
                dealt = b.damage(dex, d, Some(t), Some(source), DamageEffect::Effect(move_eff), false);
            }
            b.run_event(dex, "Hit", EvTarget::Poke(t), Some(source), move_eff, None, false, false);
            if let Some(d) = dealt {
                let am = b.active_move.as_mut().unwrap();
                am.total_damage = Some(am.total_damage.unwrap_or(0) + d as i64);
                // gen < 5 DamagingHit: no handlers in this format.
            }
        }
        b.each_event(dex, "Update", None);
        b.faint_messages(dex, false);
        if b.ended {
            b.clear_active_move(true);
            return;
        }
        if cur_target.is_some() {
            b.got_attacked(target, Some(fs_id), dealt, source);
        }
        if dealt.is_some() {
            b.each_event(dex, "Update", None);
            b.single_event(dex, "AfterMoveSecondary", move_eff, StateLoc::None, EvTarget::Poke(target), Some(source), move_eff, None);
            b.run_event(dex, "AfterMoveSecondary", EvTarget::Poke(target), Some(source), move_eff, None, false, false);
        }
    }
    b.clear_active_move(true);
    // checkWin()
    if !b.ended {
        if b.sides[0].pokemon_left == 0 {
            b.win(Some(1));
        } else if b.sides[1].pokemon_left == 0 {
            b.win(Some(0));
        }
    }
}

/// Rapid Spin's hazard/leech-seed/trap cleanup (onAfterHit + self.onHit run
/// the same code; the second invocation is a silent no-op).
fn rapid_spin_cleanup(b: &mut Battle, dex: &Dex, pokemon: PokeId) {
    if b.remove_volatile(dex, pokemon, "leechseed") {
        let ps = b.poke_str(pokemon);
        let of = format!("[of] {ps}");
        b.add(&["-end", &ps, "Leech Seed", "[from] move: Rapid Spin", &of]);
    }
    if let Some(spikes) = dex.conds_id("spikes") {
        if b.remove_side_condition(dex, pokemon.side, spikes) {
            let side_str = b.side_str(pokemon.side);
            let of = format!("[of] {}", b.poke_str(pokemon));
            b.add(&["-sideend", &side_str, "Spikes", "[from] move: Rapid Spin", &of]);
        }
    }
    let pt = dex.conds_id("partiallytrapped").unwrap();
    if b.poke(pokemon).has_volatile(pt) {
        b.remove_volatile(dex, pokemon, "partiallytrapped");
    }
}

impl Battle {
    pub fn set_active_move(&mut self, mv: ActiveMove, pokemon: Option<PokeId>, target: Option<PokeId>) {
        self.active_move = Some(mv);
        self.active_pokemon = pokemon;
        self.active_target = target.or(pokemon);
    }

    pub fn clear_active_move(&mut self, failed: bool) {
        if let Some(am) = &self.active_move {
            if !failed {
                self.last_move_id = am.id;
            }
            self.active_move = None;
            self.active_pokemon = None;
            self.active_target = None;
        }
    }

    // ----------------------------------------------------------- targeting

    /// battle.getRandomTarget (singles).
    pub fn get_random_target(&self, mv_target: &str, pokemon: PokeId) -> Option<PokeId> {
        match mv_target {
            "self" | "all" | "allySide" | "allyTeam" | "adjacentAllyOrSelf" => Some(pokemon),
            "adjacentAlly" => None,
            _ => self.active_id(1 - pokemon.side as usize),
        }
    }

    /// battle.validTargetLoc (singles slice).
    pub fn valid_target_loc(&self, target_loc: i8, _source: PokeId, target_type: &str) -> bool {
        if target_loc == 0 {
            return true;
        }
        if target_loc.abs() > 1 {
            return false;
        }
        let is_self = target_loc == -1;
        let is_foe = target_loc == 1;
        let is_adjacent = is_foe || (is_self && false) || target_loc == -1 && false;
        // singles: adjacency = the foe slot only (|loc| == 1, loc != selfLoc);
        // selfLoc is -1, so isAdjacent === (targetLoc == 1).
        let is_adjacent = is_adjacent || target_loc == 1;
        match target_type {
            "randomNormal" | "scripted" | "normal" => is_adjacent,
            "adjacentAlly" => is_adjacent && !is_foe,
            "adjacentAllyOrSelf" => (is_adjacent && !is_foe) || is_self,
            "adjacentFoe" => is_adjacent && is_foe,
            "any" => !is_self,
            _ => false,
        }
    }

    /// battle.getTarget.
    pub fn get_target(&self, mv: &ActiveMove, pokemon: PokeId, target_loc: i8) -> Option<PokeId> {
        // Fails if the target is the user and the move can't target its own position
        let self_loc = -1i8;
        if matches!(mv.target.as_str(), "adjacentAlly" | "any" | "normal")
            && target_loc == self_loc
            && !self.has_twoturn_volatile(pokemon)
        {
            if mv.has_flag("futuremove") {
                return Some(pokemon);
            }
            return self.get_random_target(&mv.target, pokemon);
        }
        if mv.target != "randomNormal" && self.valid_target_loc(target_loc, pokemon, &mv.target) {
            let target = if target_loc == 1 {
                self.active_id(1 - pokemon.side as usize)
            } else {
                self.active_id(pokemon.side as usize)
            };
            if let Some(t) = target {
                if self.poke(t).fainted {
                    if t.side == pokemon.side {
                        // fainted ally: attack shouldn't retarget
                        return Some(t);
                    }
                } else {
                    return Some(t);
                }
            }
        }
        self.get_random_target(&mv.target, pokemon)
    }

    fn has_twoturn_volatile(&self, pokemon: PokeId) -> bool {
        // twoturnmove / iceball / rollout volatiles — ids live in the state bag.
        self.poke(pokemon)
            .volatiles
            .iter()
            .any(|(_, st)| st.id == "twoturnmove" || st.id == "iceball" || st.id == "rollout")
    }

    // ------------------------------------------------------------ runMove

    /// gen2 actions.runMove.
    pub fn run_move(
        &mut self,
        dex: &Dex,
        move_id: MoveId,
        pokemon: PokeId,
        target_loc: i8,
        source_effect: Option<MoveId>,
    ) {
        let mut mv = get_active_move(dex, move_id);
        let mut target = self.get_target(&mv, pokemon, target_loc);
        if source_effect.is_none() && dex.moves.key(move_id) != "struggle" {
            let changed = self.run_event(
                dex,
                "OverrideAction",
                EvTarget::Poke(pokemon),
                target,
                EffectHandle::MoveEff(move_id),
                None,
                false,
                false,
            );
            if let RV::Str(new_id) = changed {
                if let Some(nm) = dex.moves.id(&new_id) {
                    mv = get_active_move(dex, nm);
                    target = self.get_random_target(&mv.target, pokemon);
                }
            }
        }
        // everything below acts on the (possibly overridden) move
        let move_id = mv.id.unwrap();
        if target.is_none() {
            target = self.get_random_target(&mv.target, pokemon);
        }

        let move_eff = EffectHandle::MoveEff(mv.id.unwrap());
        self.set_active_move(mv, Some(pokemon), target);

        if self.poke(pokemon).move_this_turn.is_some() {
            // already moved — sanity path
            self.clear_active_move(true);
            return;
        }
        let before = self.run_event(
            dex,
            "BeforeMove",
            EvTarget::Poke(pokemon),
            target,
            move_eff,
            None,
            false,
            false,
        );
        if !before.truthy() {
            self.run_event(dex, "MoveAborted", EvTarget::Poke(pokemon), target, move_eff, None, false, false);
            self.clear_active_move(true);
            // This is only run for sleep and fully paralysed.
            self.run_event(dex, "AfterMoveSelf", EvTarget::Poke(pokemon), target, move_eff, None, false, false);
            return;
        }
        // beforeMoveCallback (bide)
        if active_move_has_callback(self, dex, move_id, "beforeMoveCallback")
            && before_move_callback(self, dex, move_id, pokemon, target)
        {
            self.clear_active_move(true);
            return;
        }
        self.poke_mut(pokemon).last_damage = 0;
        let locked = self.get_locked_move(dex, pokemon).or_else(|| self.get_semi_locked_move(dex, pokemon));
        if locked.is_none() {
            let deducted = self.deduct_pp(pokemon, move_id, 1);
            if deducted == 0 && dex.moves.key(move_id) != "struggle" {
                let ps = self.poke_str(pokemon);
                let move_name = dex.move_static(move_id).name.clone();
                self.add(&["cant", &ps, "nopp", &move_name]);
                self.clear_active_move(true);
                return;
            }
        }
        self.move_used(dex, pokemon, move_id, None);
        self.use_move(dex, move_id, pokemon, target, source_effect);
        self.single_event(
            dex,
            "AfterMove",
            move_eff,
            StateLoc::None,
            EvTarget::Poke(pokemon),
            target,
            move_eff,
            None,
        );
        // gen2: if !move.selfSwitch && foe active has hp → AfterMoveSelf
        let self_switch = self.active_move.as_ref().and_then(|m| m.self_switch.clone());
        let move_self_switch = dex.move_static(move_id).self_switch.is_some();
        let _ = self_switch;
        let foe_hp = self
            .active_id(1 - pokemon.side as usize)
            .map(|f| self.poke(f).hp > 0)
            .unwrap_or(false);
        if !move_self_switch && foe_hp {
            self.run_event(dex, "AfterMoveSelf", EvTarget::Poke(pokemon), target, move_eff, None, false, false);
        }
    }

    /// pokemon.getSemiLockedMove.
    pub fn get_semi_locked_move(&mut self, dex: &Dex, id: PokeId) -> Option<String> {
        let rv = self.priority_event(dex, "SemiLockMove", EvTarget::Poke(id), None, EffectHandle::None, None);
        match rv {
            RV::Str(s) => Some(s),
            _ => None,
        }
    }

    /// base useMove wrapper (moveThisTurnResult bookkeeping).
    pub fn use_move(
        &mut self,
        dex: &Dex,
        move_id: MoveId,
        pokemon: PokeId,
        target: Option<PokeId>,
        source_effect: Option<MoveId>,
    ) -> bool {
        self.poke_mut(pokemon).move_this_turn_result = MoveResult::Undef;
        let old_result = self.poke(pokemon).move_this_turn_result;
        let move_result = self.use_move_inner(dex, move_id, pokemon, target, source_effect);
        if old_result == self.poke(pokemon).move_this_turn_result {
            self.poke_mut(pokemon).move_this_turn_result =
                if move_result { MoveResult::True } else { MoveResult::False };
        }
        move_result
    }

    /// gen3 useMoveInner.
    fn use_move_inner(
        &mut self,
        dex: &Dex,
        move_id: MoveId,
        pokemon: PokeId,
        target: Option<PokeId>,
        source_effect: Option<MoveId>,
    ) -> bool {
        let mut source_effect_h: EffectHandle = source_effect
            .map(EffectHandle::MoveEff)
            .unwrap_or(EffectHandle::None);
        if source_effect_h.is_none() {
            let cur = self.current_effect();
            if !cur.is_none() {
                source_effect_h = cur;
            }
        }

        let mut mv = get_active_move(dex, move_id);
        self.poke_mut(pokemon).last_move_used = Some(move_id);
        if let Some(am) = &self.active_move {
            mv.priority = am.priority;
        }
        let base_target = mv.target.clone();
        let mut target = target;
        if target.is_none() {
            target = self.get_random_target(&mv.target, pokemon);
        }
        if mv.target == "self" || mv.target == "allies" {
            target = Some(pokemon);
        }
        if !source_effect_h.is_none() {
            if let EffectHandle::MoveEff(se) = source_effect_h {
                mv.source_effect = Some(se);
            }
        }
        let move_eff = EffectHandle::MoveEff(move_id);
        self.set_active_move(mv, Some(pokemon), target);

        // ModifyMove single + run events
        self.single_event(
            dex,
            "ModifyMove",
            move_eff,
            StateLoc::None,
            EvTarget::Poke(pokemon),
            target,
            move_eff,
            None,
        );
        if self.active_move.as_ref().map(|m| m.target.clone()).unwrap_or_default() != base_target {
            target = self.get_random_target(
                &self.active_move.as_ref().unwrap().target.clone(),
                pokemon,
            );
        }
        self.run_event(dex, "ModifyMove", EvTarget::Poke(pokemon), target, move_eff, None, false, false);
        if self.poke(pokemon).fainted {
            return false;
        }

        // |move| line
        let ps = self.poke_str(pokemon);
        let move_name = self.active_move.as_ref().unwrap().name.clone();
        let target_str = match target {
            Some(t) => self.poke_str(t),
            None => "null".to_string(),
        };
        let mut last_part = target_str;
        if let EffectHandle::MoveEff(se) = source_effect_h {
            last_part = format!("{last_part}|[from] {}", dex.move_static(se).name);
        }
        self.add_move(&["move", &ps, &move_name, &last_part]);

        if target.is_none() {
            self.attr_last_move(&["[notarget]"]);
            let ps = self.poke_str(pokemon);
            self.add(&["-notarget", &ps]);
            return false;
        }

        // getMoveTargets (singles single-target path + field/side path)
        let mv_target_kind = self.active_move.as_ref().unwrap().target.clone();
        let is_field_move = matches!(
            mv_target_kind.as_str(),
            "all" | "foeSide" | "allySide" | "allyTeam"
        );

        // TryMove events
        if !self
            .single_event(dex, "TryMove", move_eff, StateLoc::None, EvTarget::Poke(pokemon), target, move_eff, None)
            .truthy()
            || !self
                .run_event(dex, "TryMove", EvTarget::Poke(pokemon), target, move_eff, None, false, false)
                .truthy()
        {
            return false;
        }

        self.single_event(
            dex,
            "UseMoveMessage",
            move_eff,
            StateLoc::None,
            EvTarget::Poke(pokemon),
            target,
            move_eff,
            None,
        );

        {
            let am = self.active_move.as_mut().unwrap();
            // (data always carries an explicit bool; PS's undefined-default
            // logic resolved at export time)
            let _ = &am.ignore_immunity;
        }

        // gen3: selfdestruct === 'always' faints the user up front (the
        // stadium2 tryMoveHit faint below is then a no-op; side.lastMove
        // bookkeeping still happens there).
        if self.active_move.as_ref().unwrap().selfdestruct {
            self.pokemon_faint(pokemon, Some(pokemon), move_eff);
        }

        let mut move_result = false;
        let damage: MoveOutcome;
        if is_field_move {
            damage = self.try_move_hit(dex, target.unwrap(), pokemon);
            if !matches!(damage, MoveOutcome::Fail) {
                move_result = true;
            }
        } else {
            // single-target: getMoveTargets
            let selected = target.unwrap();
            let mut t = selected;
            if self.poke(t).fainted && t.side != pokemon.side {
                // retarget
                match self.get_random_target(&mv_target_kind, pokemon) {
                    Some(nt) => t = nt,
                    None => {
                        self.attr_last_move(&["[notarget]"]);
                        let ps = self.poke_str(pokemon);
                        self.add(&["-notarget", &ps]);
                        return false;
                    }
                }
            }
            let futuremove = self.active_move.as_ref().unwrap().has_flag("futuremove");
            if self.poke(t).fainted && !futuremove {
                self.attr_last_move(&["[notarget]"]);
                let ps = self.poke_str(pokemon);
                self.add(&["-notarget", &ps]);
                return false;
            }
            if t != selected {
                let tstr = self.poke_str(t);
                self.retarget_last_move(&tstr);
            }
            target = Some(t);
            damage = self.try_move_hit(dex, t, pokemon);
            if !matches!(damage, MoveOutcome::Fail) {
                move_result = true;
            }
        }
        if !self.poke(pokemon).hp.is_positive() {
            self.pokemon_faint(pokemon, Some(pokemon), EffectHandle::MoveEff(move_id));
        }
        let _ = damage;

        if !move_result {
            self.single_event(
                dex,
                "MoveFail",
                move_eff,
                StateLoc::None,
                target.map(EvTarget::Poke).unwrap_or(EvTarget::Battle),
                Some(pokemon),
                move_eff,
                None,
            );
            return false;
        }

        self.single_event(
            dex,
            "AfterMoveSecondarySelf",
            move_eff,
            StateLoc::None,
            EvTarget::Poke(pokemon),
            target,
            move_eff,
            None,
        );
        self.run_event(
            dex,
            "AfterMoveSecondarySelf",
            EvTarget::Poke(pokemon),
            target,
            move_eff,
            None,
            false,
            false,
        );
        true
    }

    // --------------------------------------------------------- tryMoveHit

    /// gen2stadium2 tryMoveHit. Returns the PS damage value collapsed to an
    /// outcome + number.
    pub fn try_move_hit(&mut self, dex: &Dex, target: PokeId, pokemon: PokeId) -> MoveOutcome {
        const POS_BOOST: [f64; 7] = [1.0, 1.33, 1.66, 2.0, 2.33, 2.66, 3.0];
        const NEG_BOOST: [f64; 7] = [1.0, 0.75, 0.6, 0.5, 0.43, 0.36, 0.33];

        let move_id = self.active_move.as_ref().unwrap().id.unwrap();
        let move_eff = EffectHandle::MoveEff(move_id);

        if self.active_move.as_ref().unwrap().selfdestruct {
            self.pokemon_faint(pokemon, Some(pokemon), move_eff);
            // self-KO clause bookkeeping
            self.sides[target.side as usize].last_move = None;
            self.sides[pokemon.side as usize].last_move = Some(move_id);
        }

        let hit = self.single_event(
            dex,
            "PrepareHit",
            move_eff,
            StateLoc::None,
            EvTarget::Poke(target),
            Some(pokemon),
            move_eff,
            None,
        );
        if !hit.truthy() {
            if hit == RV::False {
                let ts = self.poke_str(target);
                self.add(&["-fail", &ts]);
            }
            return MoveOutcome::Fail;
        }
        self.run_event(dex, "PrepareHit", EvTarget::Poke(pokemon), Some(target), move_eff, None, false, false);

        if !self
            .single_event(dex, "Try", move_eff, StateLoc::None, EvTarget::Poke(pokemon), Some(target), move_eff, None)
            .truthy()
        {
            return MoveOutcome::Fail;
        }

        let mv_target = self.active_move.as_ref().unwrap().target.clone();
        if matches!(mv_target.as_str(), "all" | "foeSide" | "allySide" | "allyTeam") {
            let hit_result = if mv_target == "all" {
                self.run_event(dex, "TryHitField", EvTarget::Poke(target), Some(pokemon), move_eff, None, false, false)
            } else {
                self.run_event(dex, "TryHitSide", EvTarget::Side(target.side), Some(pokemon), move_eff, None, false, false)
            };
            if !hit_result.truthy() {
                if hit_result == RV::False {
                    let ps = self.poke_str(pokemon);
                    self.add(&["-fail", &ps]);
                    self.attr_last_move(&["[still]"]);
                }
                return MoveOutcome::Fail;
            }
            return self.move_hit(dex, Some(target), pokemon, MoveHitData::Primary, false, false);
        }

        let invuln = self.run_event(dex, "Invulnerability", EvTarget::Poke(target), Some(pokemon), move_eff, None, false, false);
        if invuln == RV::False {
            self.attr_last_move(&["[miss]"]);
            let ps = self.poke_str(pokemon);
            self.add(&["-miss", &ps]);
            return MoveOutcome::Fail;
        }

        if !self.run_move_immunity_of(dex, target, true) {
            return MoveOutcome::Fail;
        }

        let try_immunity = self.single_event(
            dex,
            "TryImmunity",
            move_eff,
            StateLoc::None,
            EvTarget::Poke(target),
            Some(pokemon),
            move_eff,
            None,
        );
        if try_immunity == RV::False {
            let ts = self.poke_str(target);
            self.add(&["-immune", &ts]);
            return MoveOutcome::Fail;
        }

        let try_hit = self.run_event(dex, "TryHit", EvTarget::Poke(target), Some(pokemon), move_eff, None, false, false);
        if !try_hit.truthy() {
            if try_hit == RV::False {
                let ts = self.poke_str(target);
                self.add(&["-fail", &ts]);
            }
            return MoveOutcome::Fail;
        }

        // accuracy
        let (mut accuracy, ohko, ignore_accuracy, ignore_evasion, ignore_positive_evasion, always_hit) = {
            let am = self.active_move.as_ref().unwrap();
            (
                am.accuracy,
                am.ohko,
                am.ignore_accuracy,
                am.ignore_evasion,
                am.ignore_positive_evasion,
                am.always_hit,
            )
        };
        if always_hit {
            accuracy = Accuracy::AlwaysHits;
        } else {
            accuracy = self.run_accuracy_event(dex, target, pokemon, accuracy);
        }
        let mut acc_num: Option<f64> = match accuracy {
            Accuracy::AlwaysHits => None,
            Accuracy::Pct(a) => Some(a as f64),
        };
        if let Some(a) = acc_num {
            let mut acc = (a * 255.0 / 100.0).floor();
            if ohko {
                let plevel = self.poke(pokemon).level as f64;
                let tlevel = self.poke(target).level as f64;
                if plevel >= tlevel {
                    acc += (plevel - tlevel) * 2.0;
                    acc = acc.min(255.0);
                } else {
                    let ts = self.poke_str(target);
                    self.add(&["-immune", &ts, "[ohko]"]);
                    return MoveOutcome::Fail;
                }
            }
            if !ignore_accuracy {
                let boost = self.poke(pokemon).boosts[5].clamp(-6, 6);
                if boost > 0 {
                    acc *= POS_BOOST[boost as usize];
                } else {
                    acc *= NEG_BOOST[(-boost) as usize];
                }
            }
            if !ignore_evasion {
                let boost = self.poke(target).boosts[6].clamp(-6, 6);
                if boost > 0 && !ignore_positive_evasion {
                    acc *= NEG_BOOST[boost as usize];
                } else if boost < 0 {
                    acc *= POS_BOOST[(-boost) as usize];
                }
            }
            acc = acc.floor().min(255.0);
            acc = acc.max(1.0);
            acc_num = Some(acc);
        } else {
            // accuracy true: run Accuracy event again (PS quirk)
            let _ = self.run_accuracy_event(dex, target, pokemon, Accuracy::AlwaysHits);
        }
        // ModifyAccuracy
        if let Some(a) = acc_num {
            let rv = self.run_event(
                dex,
                "ModifyAccuracy",
                EvTarget::Poke(target),
                Some(pokemon),
                move_eff,
                Some(RV::Num(a)),
                false,
                false,
            );
            acc_num = Some(rv.as_num().max(0.0));
        }
        if always_hit {
            acc_num = None;
        } else if let Some(a) = acc_num {
            let rv = self.run_accuracy_event(dex, target, pokemon, Accuracy::Pct(a as i32));
            acc_num = match rv {
                Accuracy::AlwaysHits => None,
                Accuracy::Pct(v) => Some(v as f64),
            };
        }
        if let Some(a) = acc_num {
            if a != 255.0 && !self.prng.random_chance(a as u32, 256) {
                self.attr_last_move(&["[miss]"]);
                let ps = self.poke_str(pokemon);
                self.add(&["-miss", &ps]);
                return MoveOutcome::Fail;
            }
        }

        self.active_move.as_mut().unwrap().total_damage = Some(0);
        self.poke_mut(pokemon).last_damage = 0;

        let multihit = self.active_move.as_ref().unwrap().multihit.clone();
        let damage: MoveOutcome;
        if let Some(mh) = multihit {
            let mut hits = match mh {
                Multihit::Fixed(n) => n,
                Multihit::Range(2, 5) => {
                    const TABLE: [i32; 8] = [2, 2, 2, 3, 3, 3, 4, 5];
                    TABLE[self.prng.sample_index(8)]
                }
                Multihit::Range(lo, hi) => self.prng.random_range(lo as u32, hi as u32 + 1) as i32,
            };
            hits = hits.max(0);
            let mut null_damage = true;
            let mut mh_damage = MoveOutcome::Fail;
            let is_sleep_usable = {
                let am = self.active_move.as_ref().unwrap();
                am.sleep_usable
                    || am
                        .source_effect
                        .map(|se| dex.move_static(se).sleep_usable)
                        .unwrap_or(false)
            };
            let mut i = 0;
            while i < hits && self.poke(target).hp > 0 && self.poke(pokemon).hp > 0 {
                if self.poke(pokemon).status == Status::Slp && !is_sleep_usable {
                    break;
                }
                {
                    let am = self.active_move.as_mut().unwrap();
                    am.hit = i + 1;
                    am.last_hit = am.hit == hits;
                }
                let move_damage = self.move_hit(dex, Some(target), pokemon, MoveHitData::Primary, false, false);
                if matches!(move_damage, MoveOutcome::Fail) {
                    break;
                }
                null_damage = false;
                let dmg_num = move_damage.num().unwrap_or(0.0);
                mh_damage = MoveOutcome::Damage(dmg_num);
                let am = self.active_move.as_mut().unwrap();
                am.total_damage = Some(am.total_damage.unwrap_or(0) + dmg_num as i64);
                self.each_event(dex, "Update", None);
                i += 1;
            }
            if i == 0 {
                return MoveOutcome::Damage(1.0);
            }
            if null_damage {
                mh_damage = MoveOutcome::Fail;
            }
            damage = mh_damage;
            let ts = self.poke_str(target);
            let hits_str = i.to_string();
            self.add(&["-hitcount", &ts, &hits_str]);
        } else {
            damage = self.move_hit(dex, Some(target), pokemon, MoveHitData::Primary, false, false);
            self.active_move.as_mut().unwrap().total_damage = damage.num().map(|n| n as i64);
        }

        if self.active_move.as_ref().unwrap().category != Category::Status {
            self.got_attacked(target, Some(move_id), damage.num(), pokemon);
        }
        if self.active_move.as_ref().unwrap().ohko {
            self.add(&["-ohko"]);
        }

        self.single_event(
            dex,
            "AfterMoveSecondary",
            move_eff,
            StateLoc::None,
            EvTarget::Poke(target),
            Some(pokemon),
            move_eff,
            None,
        );
        self.run_event(dex, "AfterMoveSecondary", EvTarget::Poke(target), Some(pokemon), move_eff, None, false, false);

        let (recoil, total_damage) = {
            let am = self.active_move.as_ref().unwrap();
            (am.recoil, am.total_damage.unwrap_or(0))
        };
        if let Some((num, den)) = recoil {
            if total_damage > 0
                && (self.sides[pokemon.side as usize].pokemon_left > 1
                    || self.sides[target.side as usize].pokemon_left > 1
                    || self.poke(target).hp > 0)
            {
                // gen3 calcRecoilDamage
                let amount = clamp_int_range(
                    (total_damage as f64 * num as f64 / den as f64).floor(),
                    Some(1.0),
                    None,
                );
                self.damage(dex, amount, Some(pokemon), Some(target), DamageEffect::Recoil, false);
            }
        }
        damage
    }

    fn run_accuracy_event(
        &mut self,
        dex: &Dex,
        target: PokeId,
        pokemon: PokeId,
        accuracy: Accuracy,
    ) -> Accuracy {
        let move_eff = self
            .active_move
            .as_ref()
            .and_then(|m| m.id)
            .map(EffectHandle::MoveEff)
            .unwrap_or(EffectHandle::None);
        let relay = match accuracy {
            Accuracy::AlwaysHits => RV::True,
            Accuracy::Pct(a) => RV::Num(a as f64),
        };
        let rv = self.run_event(
            dex,
            "Accuracy",
            EvTarget::Poke(target),
            Some(pokemon),
            move_eff,
            Some(relay),
            false,
            false,
        );
        match rv {
            RV::True => Accuracy::AlwaysHits,
            RV::Num(n) => Accuracy::Pct(n as i32),
            _ => Accuracy::Pct(0),
        }
    }

    /// pokemon.runImmunity on the current active move.
    fn run_move_immunity_of(&mut self, dex: &Dex, target: PokeId, message: bool) -> bool {
        // gen2 tryMoveHit sets ignoreImmunity default for Status moves; our
        // data already carries explicit values.
        self.run_move_immunity(dex, target, message)
    }

    // ------------------------------------------------------------ moveHit

    /// gen2 moveHit. `hit_data` selects primary move data vs secondary/self
    /// sub-blocks.
    pub fn move_hit(
        &mut self,
        dex: &Dex,
        target: Option<PokeId>,
        pokemon: PokeId,
        hit_data: MoveHitData,
        is_secondary: bool,
        is_self: bool,
    ) -> MoveOutcome {
        let move_id = self.active_move.as_ref().unwrap().id;
        let move_eff = move_id.map(EffectHandle::MoveEff).unwrap_or(EffectHandle::None);
        let mut target = target;

        let mv_target_kind = self.active_move.as_ref().unwrap().target.clone();

        // TryHit single events
        let mut hit_result = RV::True;
        if mv_target_kind == "all" && !is_self {
            hit_result = self.single_event(dex, "TryHitField", move_eff, StateLoc::None, EvTarget::Battle, Some(pokemon), move_eff, None);
        } else if (mv_target_kind == "foeSide" || mv_target_kind == "allySide") && !is_self {
            hit_result = self.single_event(
                dex,
                "TryHitSide",
                move_eff,
                StateLoc::None,
                target.map(|t| EvTarget::Side(t.side)).unwrap_or(EvTarget::Battle),
                Some(pokemon),
                move_eff,
                None,
            );
        } else if let Some(t) = target {
            // only the PRIMARY moveData TryHit runs as a singleEvent on the
            // move; secondary/self blocks have no onTryHit in M1.
            if matches!(hit_data, MoveHitData::Primary) {
                hit_result = self.single_event(dex, "TryHit", move_eff, StateLoc::None, EvTarget::Poke(t), Some(pokemon), move_eff, None);
            }
        }
        if !hit_result.truthy() {
            if hit_result == RV::False {
                let ts = target.map(|t| self.poke_str(t)).unwrap_or_default();
                self.add(&["-fail", &ts]);
            }
            return MoveOutcome::Fail;
        }

        if target.is_some() && !is_secondary && !is_self {
            let rv = self.run_event(dex, "TryPrimaryHit", EvTarget::Poke(target.unwrap()), Some(pokemon), move_eff, None, false, false);
            if rv == RV::Num(0.0) {
                // special Substitute flag
                target = None;
            } else if !rv.truthy() {
                return MoveOutcome::Fail;
            }
        }
        // (isSecondary && !moveData.self → hitResult forced true — no-op here)

        // Extract the effective move data AFTER the TryHit events (curse's
        // onTryHit mutates the active move's self/volatileStatus).
        let md = self.hit_effect_data(hit_data.clone());

        let mut damage: MoveOutcome = MoveOutcome::Undefined;
        if let Some(t) = target {
            // PS `didSomething: boolean | number | null` with `||` chaining.
            let mut did_something = Tri::False;

            // getDamage only computes for the primary move data (sub-blocks
            // have no base power / category of their own in gen2 M1).
            if matches!(hit_data, MoveHitData::Primary) {
                let calc = self.get_damage(dex, pokemon, t, false);
                match calc {
                    DamageResult::Damage(_) | DamageResult::Zero => {
                        let d = match calc {
                            DamageResult::Damage(d) => d,
                            _ => 0.0,
                        };
                        if !self.poke(t).fainted {
                            let dealt = self.damage(dex, d, Some(t), Some(pokemon), DamageEffect::Effect(move_eff), false);
                            match dealt {
                                Some(v) => {
                                    damage = MoveOutcome::Damage(v);
                                    did_something = Tri::True;
                                }
                                None => {
                                    return MoveOutcome::Fail;
                                }
                            }
                        } else {
                            damage = MoveOutcome::Damage(d);
                            did_something = Tri::True;
                        }
                    }
                    DamageResult::Undefined => {
                        damage = MoveOutcome::Undefined;
                    }
                    DamageResult::False | DamageResult::Null => {
                        if calc == DamageResult::False && !is_secondary && !is_self {
                            let ts = self.poke_str(t);
                            self.add(&["-fail", &ts]);
                        }
                        return MoveOutcome::Fail;
                    }
                }
            }

            // boosts
            if md.has_boosts && !self.poke(t).fainted {
                let r = match self.boost(dex, &md.boosts, Some(t), Some(pokemon), move_eff) {
                    Some(true) => Tri::True,
                    Some(false) => Tri::False,
                    None => Tri::Null,
                };
                did_something = did_something.or(r);
            }
            // heal
            if let Some((hn, hd)) = md.heal {
                if !self.poke(t).fainted {
                    let maxhp = self.poke(t).maxhp as f64;
                    let amount = (maxhp * hn as f64 / hd as f64).round();
                    let healed = self.pokemon_heal(t, amount);
                    match healed {
                        None => {
                            let ts = self.poke_str(t);
                            self.add(&["-fail", &ts]);
                            return MoveOutcome::Fail;
                        }
                        Some(_) => {
                            let ts = self.poke_str(t);
                            let (secret, shared) = self.get_health(t);
                            let side_id = format!("p{}", t.side + 1);
                            self.add_split(&side_id, &["-heal", &ts, &secret], &["-heal", &ts, &shared]);
                            did_something = Tri::True;
                        }
                    }
                }
            }
            // status
            if let Some(st) = &md.status {
                let r = self.try_set_status(dex, t, st, Some(pokemon), move_eff);
                let move_status = self.active_move.as_ref().unwrap().status.is_some();
                if !r.truthy() && move_status {
                    // return hitResult (false/null) — no further processing
                    return MoveOutcome::Fail;
                }
                did_something = did_something.or(Tri::from_rv(&r));
            }
            // volatileStatus
            if let Some(vs) = &md.volatile_status {
                let r = self.add_volatile(dex, t, vs, Some(pokemon), move_eff);
                did_something = did_something.or(Tri::from_rv(&r));
            }
            // sideCondition
            if let Some(sc) = &md.side_condition {
                let r = self.add_side_condition(dex, t.side, sc, Some(pokemon), move_eff);
                did_something = did_something.or(Tri::from_rv(&r));
            }
            // weather
            if let Some(w) = &md.weather {
                let key = crate::dex::toid(w);
                let r = self.set_weather(dex, &key, Some(pokemon), move_eff);
                did_something = did_something.or(Tri::from_rv(&r));
            }
            // forceSwitch / selfSwitch presence defers the fail message
            if md.force_switch && self.can_switch(t.side) {
                did_something = Tri::True;
            }
            if md.self_switch && self.can_switch(pokemon.side) {
                did_something = Tri::True;
            }

            // Hit events (hitResult = null before them)
            let mut hit_event = Tri::Null;
            if mv_target_kind == "all" && !is_self {
                if move_id.map(|m| move_has_callback(dex, m, "onHitField")).unwrap_or(false) {
                    let r = self.single_event(dex, "HitField", move_eff, StateLoc::None, EvTarget::Poke(t), Some(pokemon), move_eff, None);
                    hit_event = Tri::from_rv(&r);
                }
            } else if (mv_target_kind == "foeSide" || mv_target_kind == "allySide") && !is_self {
                if move_id.map(|m| move_has_callback(dex, m, "onHitSide")).unwrap_or(false) {
                    let r = self.single_event(dex, "HitSide", move_eff, StateLoc::None, EvTarget::Side(t.side), Some(pokemon), move_eff, None);
                    hit_event = Tri::from_rv(&r);
                }
            } else {
                match &hit_data {
                    MoveHitData::Primary => {
                        if move_id
                            .map(|m| active_move_has_callback(self, dex, m, "onHit"))
                            .unwrap_or(false)
                        {
                            let r = self.single_event(dex, "Hit", move_eff, StateLoc::None, EvTarget::Poke(t), Some(pokemon), move_eff, None);
                            hit_event = Tri::from_rv(&r);
                        }
                    }
                    MoveHitData::Sub(h) if h.has_on_hit => {
                        // singleEvent('Hit', subBlock, ...) — dispatch by the
                        // owning move + block kind.
                        let name = if is_self { "self.onHit" } else { "secondary.onHit" };
                        self.effect_stack.push(EffectFrame { effect: move_eff, state: StateLoc::None });
                        self.event_stack.push(EventFrame {
                            id: "Hit".to_string(),
                            target: Some(t),
                            source: Some(pokemon),
                            effect: move_eff,
                            modifier: 1.0,
                        });
                        self.event_depth += 1;
                        let r = dispatch_move_callback(
                            self,
                            dex,
                            move_id.unwrap(),
                            name,
                            EvTarget::Poke(t),
                            Some(pokemon),
                            RV::True,
                        );
                        self.event_depth -= 1;
                        self.event_stack.pop();
                        self.effect_stack.pop();
                        if r != RV::Undef {
                            hit_event = Tri::from_rv(&r);
                        } else {
                            hit_event = Tri::True;
                        }
                    }
                    _ => {}
                }
                if !is_self && !is_secondary {
                    self.run_event(dex, "Hit", EvTarget::Poke(t), Some(pokemon), move_eff, None, false, false);
                }
                if matches!(hit_data, MoveHitData::Primary)
                    && move_id
                        .map(|m| active_move_has_callback(self, dex, m, "onAfterHit"))
                        .unwrap_or(false)
                {
                    let r = self.single_event(dex, "AfterHit", move_eff, StateLoc::None, EvTarget::Poke(t), Some(pokemon), move_eff, None);
                    hit_event = Tri::from_rv(&r);
                }
            }

            let has_self = md.self_effect.is_some();
            let move_selfdestruct = self.active_move.as_ref().unwrap().selfdestruct;
            if hit_event != Tri::True
                && did_something != Tri::True
                && !has_self
                && !move_selfdestruct
            {
                if !is_self && !is_secondary && (hit_event == Tri::False || did_something == Tri::False) {
                    let ts = self.poke_str(t);
                    self.add(&["-fail", &ts]);
                }
                return MoveOutcome::Fail;
            }
        }

        // moveData.self
        if let Some(self_block) = md.self_effect.clone() {
            // All self drops grab a random number (in-game RNG behavior)
            if !is_secondary && !self_block.boosts.is_empty() {
                self.prng.random(100);
            }
            self.move_hit(
                dex,
                Some(pokemon),
                pokemon,
                MoveHitData::Sub(self_block),
                is_secondary,
                true,
            );
        }
        // secondaries
        if let Some(t) = target {
            if self.poke(t).hp > 0 && !md.secondaries.is_empty() {
                let try_sec = self.run_event(dex, "TrySecondaryHit", EvTarget::Poke(t), Some(pokemon), move_eff, None, false, false);
                if try_sec.truthy() {
                    for secondary in md.secondaries.clone() {
                        // brn/frz immunity if target shares the move's type
                        let move_type = self.active_move.as_ref().unwrap().move_type.clone();
                        if let Some(st) = &secondary.status {
                            if (st == "brn" || st == "frz") && self.poke(t).has_type(&move_type) {
                                continue;
                            }
                        }
                        if secondary.volatile_status.as_deref() == Some("flinch")
                            && matches!(self.poke(t).status, Status::Slp | Status::Frz)
                            && !secondary.kingsrock
                        {
                            continue;
                        }
                        let (is_multi, last_hit) = {
                            let am = self.active_move.as_ref().unwrap();
                            (am.multihit.is_some(), am.last_hit)
                        };
                        if !is_multi || last_hit {
                            match secondary.chance {
                                None => {
                                    self.move_hit(dex, Some(t), pokemon, MoveHitData::Sub(secondary.clone()), true, is_self);
                                }
                                Some(chance) => {
                                    let effect_chance = ((chance as f64) * 255.0 / 100.0).floor();
                                    if self.prng.random_chance(effect_chance as u32, 256) {
                                        self.move_hit(dex, Some(t), pokemon, MoveHitData::Sub(secondary.clone()), true, is_self);
                                    } else if effect_chance == 255.0 {
                                        self.hint("In Gen 2, moves with a 100% secondary effect chance will not trigger in 1/256 uses.", false);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // forceSwitch drag
        if let Some(t) = target {
            if self.poke(t).hp > 0 && self.poke(pokemon).hp > 0 && md.force_switch && self.can_switch(t.side) {
                let rv = self.run_event(dex, "DragOut", EvTarget::Poke(t), Some(pokemon), move_eff, None, false, false);
                if rv.truthy() {
                    let pos = self.poke(t).position;
                    self.drag_in(dex, t.side, pos);
                } else if rv == RV::False {
                    let ts = self.poke_str(t);
                    self.add(&["-fail", &ts]);
                }
            }
        }
        // selfSwitch flag
        if md.self_switch && self.poke(pokemon).hp > 0 {
            self.poke_mut(pokemon).switch_flag = SwitchFlag::Move(move_id.unwrap());
        }
        damage
    }

    /// Effective move data of a hit (primary move fields or sub-block).
    fn hit_effect_data(&self, hit: MoveHitData) -> EffectiveMoveData {
        match hit {
            MoveHitData::Primary => {
                let am = self.active_move.as_ref().unwrap();
                EffectiveMoveData {
                    has_boosts: am.has_boosts,
                    boosts: am.boosts.clone(),
                    status: am.status.clone(),
                    volatile_status: am.volatile_status.clone(),
                    side_condition: am.side_condition.clone(),
                    weather: am.weather.clone(),
                    heal: am.heal,
                    self_effect: am.self_effect.clone(),
                    secondaries: am.secondaries.clone(),
                    force_switch: am.force_switch,
                    self_switch: am.self_switch.is_some(),
                }
            }
            MoveHitData::Sub(h) => EffectiveMoveData {
                has_boosts: !h.boosts.is_empty(),
                boosts: h.boosts.clone(),
                status: h.status.clone(),
                volatile_status: h.volatile_status.clone(),
                side_condition: None,
                weather: None,
                heal: None,
                self_effect: h.self_effect.as_deref().cloned(),
                secondaries: Vec::new(),
                force_switch: false,
                self_switch: false,
            },
        }
    }

    // ---------------------------------------------------------- getDamage

    /// gen2stadium2 getDamage on the current active move.
    pub fn get_damage(
        &mut self,
        dex: &Dex,
        source: PokeId,
        target: PokeId,
        suppress_messages: bool,
    ) -> DamageResult {
        let move_eff = self
            .active_move
            .as_ref()
            .and_then(|m| m.id)
            .map(EffectHandle::MoveEff)
            .unwrap_or(EffectHandle::None);

        // immunity
        if !self.run_move_immunity(dex, target, true) {
            return DamageResult::False;
        }
        let (ohko, damage_kind, category, mut base_power, will_crit, crit_ratio_data, selfdestruct, is_confusion, no_damage_variance, ignore_offensive, ignore_defensive, mv_type) = {
            let am = self.active_move.as_ref().unwrap();
            (
                am.ohko,
                am.damage.clone(),
                am.category,
                am.base_power,
                am.will_crit,
                am.crit_ratio,
                am.selfdestruct,
                am.is_confusion_self_hit,
                am.no_damage_variance,
                am.ignore_offensive,
                am.ignore_defensive,
                am.move_type.clone(),
            )
        };
        if ohko {
            return DamageResult::Damage(self.poke(target).maxhp as f64);
        }
        // damageCallback (counter/mirrorcoat/psywave/superfang)
        if let Some(m) = self.active_move.as_ref().and_then(|m| m.id) {
            if active_move_has_callback(self, dex, m, "damageCallback") {
                return damage_callback(self, dex, m, source, target);
            }
        }
        if let Some(dk) = damage_kind {
            return match dk {
                crate::dex::FixedDamage::Level => DamageResult::Damage(self.poke(source).level as f64),
                crate::dex::FixedDamage::Amount(n) => DamageResult::Damage(n as f64),
            };
        }
        // basePowerCallback
        if let Some(m) = self.active_move.as_ref().and_then(|m| m.id) {
            if active_move_has_callback(self, dex, m, "basePowerCallback") {
                match base_power_callback(self, dex, m, source, target, base_power) {
                    Some(bp) => base_power = bp,
                    // JS `null`/false base power propagates like 0-but-not-0
                    None => return DamageResult::Null,
                }
            }
        }
        if base_power == 0 {
            return DamageResult::Undefined;
        }
        base_power = clamp_int_range(base_power as f64, Some(1.0), None) as i32;

        // crit
        let crit_ratio = self
            .run_event(
                dex,
                "ModifyCritRatio",
                EvTarget::Poke(source),
                Some(target),
                move_eff,
                Some(RV::Num(crit_ratio_data as f64)),
                false,
                false,
            )
            .as_num();
        let crit_ratio = clamp_int_range(crit_ratio, Some(0.0), Some(5.0)) as usize;
        const CRIT_MULT: [u32; 6] = [0, 16, 8, 4, 3, 2];
        let mut is_crit = will_crit.unwrap_or(false);
        if will_crit.is_none() && crit_ratio > 0 {
            is_crit = self.prng.random_chance(1, CRIT_MULT[crit_ratio]);
        }
        if is_crit
            && self
                .run_event(dex, "CriticalHit", EvTarget::Poke(target), None, move_eff, None, false, false)
                .truthy()
        {
            let slot = self.slot_str(target);
            self.active_move.as_mut().unwrap().hit_data_mut(slot).0 = true;
        }

        // BasePower event (after crit calc)
        let bp_rv = if is_confusion {
            // temporarily restore base move type
            let base_type = self.active_move.as_ref().unwrap().base_move_type.clone();
            self.active_move.as_mut().unwrap().move_type = base_type;
            let rv = self.run_event(
                dex,
                "BasePower",
                EvTarget::Poke(source),
                Some(target),
                move_eff,
                Some(RV::Num(base_power as f64)),
                true,
                false,
            );
            self.active_move.as_mut().unwrap().move_type = "???".into();
            rv
        } else {
            self.run_event(
                dex,
                "BasePower",
                EvTarget::Poke(source),
                Some(target),
                move_eff,
                Some(RV::Num(base_power as f64)),
                true,
                false,
            )
        };
        let base_power = bp_rv.as_num();
        if base_power == 0.0 {
            return DamageResult::Zero;
        }
        let base_power = clamp_int_range(base_power, Some(1.0), None);

        let mut level = self.poke(source).level as f64;
        // Beat Up: level + '-activate' (stat overrides follow the stat fetch)
        let beatup_first = self
            .active_move
            .as_ref()
            .and_then(|m| m.allies.as_ref())
            .and_then(|a| a.first().copied());
        if let Some(first) = beatup_first {
            let ss = self.poke_str(source);
            let of = format!("[of] {}", self.poke(first).name);
            self.add(&["-activate", &ss, "move: Beat Up", &of]);
            level = self.poke(first).level as f64;
        }
        let is_physical = category == Category::Physical;
        let atk_stat = if is_physical { 0 } else { 2 };
        let def_stat = if is_physical { 1 } else { 3 };

        let mut unboosted = false;
        let mut noburndrop = false;
        if is_crit {
            if !suppress_messages {
                let ts = self.poke_str(target);
                self.add(&["-crit", &ts]);
            }
            if self.poke(source).boosts[atk_stat] <= self.poke(target).boosts[def_stat] {
                unboosted = true;
                noburndrop = true;
            }
        }

        let mut attack = self.get_stat(dex, source, atk_stat, unboosted, noburndrop, false) as f64;
        let mut defense = self.get_stat(dex, target, def_stat, unboosted, false, false) as f64;

        // Beat Up: base-stat overrides, then allies.shift()
        if let Some(first) = beatup_first {
            attack = dex.species.get(self.poke(first).species).base_stats.atk as f64;
            defense = dex.species.get(self.poke(target).species).base_stats.def as f64;
            if let Some(allies) = self.active_move.as_mut().and_then(|m| m.allies.as_mut()) {
                allies.remove(0);
            }
        }

        if ignore_offensive {
            attack = self.get_stat(dex, source, atk_stat, true, true, false) as f64;
        }
        if ignore_defensive {
            defense = self.get_stat(dex, target, def_stat, true, true, false) as f64;
        }

        // stadium2 rollover fix
        if attack >= 256.0 || defense >= 256.0 {
            attack = clamp_int_range(
                (clamp_int_range(attack, Some(1.0), Some(999.0)) / 4.0).floor(),
                Some(1.0),
                None,
            );
            defense = clamp_int_range(
                (clamp_int_range(defense, Some(1.0), Some(999.0)) / 4.0).floor(),
                Some(1.0),
                None,
            );
        }

        if selfdestruct && def_stat == 1 {
            defense = clamp_int_range((defense / 2.0).floor(), Some(1.0), None);
        }

        let mut damage = level * 2.0;
        damage = (damage / 5.0).floor();
        damage += 2.0;
        damage *= base_power;
        damage *= attack;
        damage = (damage / defense).floor();
        damage = (damage / 50.0).floor();
        if is_crit {
            damage *= 2.0;
        }
        let rv = self.run_event(
            dex,
            "ModifyDamage",
            EvTarget::Poke(source),
            Some(target),
            move_eff,
            Some(RV::Num(damage)),
            false,
            false,
        );
        damage = rv.as_num().floor();
        damage = clamp_int_range(damage, Some(1.0), Some(997.0));
        damage += 2.0;

        // weather modifiers
        let weather = self.field_weather_key.clone();
        let move_key = self.active_move.as_ref().unwrap().id.map(|m| dex.moves.key(m).to_string()).unwrap_or_default();
        if (mv_type == "Water" && weather == "raindance") || (mv_type == "Fire" && weather == "sunnyday") {
            damage = (damage * 1.5).floor();
        } else if ((mv_type == "Fire" || move_key == "solarbeam") && weather == "raindance")
            || (mv_type == "Water" && weather == "sunnyday")
        {
            damage = (damage / 2.0).floor();
        }

        // STAB
        if mv_type != "???" && self.poke(source).has_type(&mv_type) {
            damage += (damage / 2.0).floor();
        }

        // type effectiveness
        let total_type_mod = self.run_effectiveness(dex, target);
        if total_type_mod > 0 {
            if !suppress_messages {
                let ts = self.poke_str(target);
                self.add(&["-supereffective", &ts]);
            }
            damage *= 2.0;
            if total_type_mod >= 2 {
                damage *= 2.0;
            }
        }
        if total_type_mod < 0 {
            if !suppress_messages {
                let ts = self.poke_str(target);
                self.add(&["-resisted", &ts]);
            }
            damage = (damage / 2.0).floor();
            if total_type_mod <= -2 {
                damage = (damage / 2.0).floor();
            }
        }

        // random factor
        if !no_damage_variance && damage > 1.0 {
            damage *= self.prng.random_range(217, 256) as f64;
            damage = (damage / 255.0).floor();
        }

        if base_power != 0.0 && damage.floor() == 0.0 {
            return DamageResult::Damage(1.0);
        }
        DamageResult::Damage(damage)
    }
}

/// The confusion self-hit entry point (getDamage on a synthetic move without
/// disturbing the current active move… PS passes the fake move object into
/// actions.getDamage, which reads only the move argument — but our getDamage
/// reads battle.active_move, so swap it in and out).
pub fn get_damage_synthetic(
    b: &mut Battle,
    dex: &Dex,
    source: PokeId,
    target: PokeId,
    fake: ActiveMove,
) -> Option<f64> {
    let saved = b.active_move.take();
    let saved_pokemon = b.active_pokemon;
    let saved_target = b.active_target;
    b.active_move = Some(fake);
    let result = b.get_damage(dex, source, target, false);
    b.active_move = saved;
    b.active_pokemon = saved_pokemon;
    b.active_target = saved_target;
    match result {
        DamageResult::Damage(d) => Some(d),
        DamageResult::Zero => Some(0.0),
        _ => None,
    }
}

/// PS getDamage return: number | 0 | undefined | false | null.
#[derive(Clone, Debug, PartialEq)]
pub enum DamageResult {
    Damage(f64),
    Zero,
    Undefined,
    False,
    Null,
}

/// PS moveHit/tryMoveHit outcome: number | undefined (success, no damage) |
/// false (fail).
#[derive(Clone, Debug, PartialEq)]
pub enum MoveOutcome {
    Damage(f64),
    Undefined,
    Fail,
}

impl MoveOutcome {
    pub fn num(&self) -> Option<f64> {
        match self {
            MoveOutcome::Damage(d) => Some(*d),
            _ => None,
        }
    }
}

/// Which move data block a moveHit call applies.
#[derive(Clone, Debug)]
pub enum MoveHitData {
    Primary,
    Sub(HitEffect),
}

/// JS truthiness tri-state (false | null | true) with `||` chaining.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tri {
    False,
    Null,
    True,
}

impl Tri {
    /// JS `a || b`: a if truthy else b.
    fn or(self, b: Tri) -> Tri {
        if self == Tri::True {
            self
        } else {
            b
        }
    }

    fn from_rv(rv: &RV) -> Tri {
        match rv {
            RV::Null | RV::Undef => Tri::Null,
            RV::False => Tri::False,
            _ => Tri::True,
        }
    }
}

/// Flattened view of the effective move data for a hit.
struct EffectiveMoveData {
    has_boosts: bool,
    boosts: Vec<(usize, i8)>,
    status: Option<String>,
    volatile_status: Option<String>,
    side_condition: Option<String>,
    weather: Option<String>,
    heal: Option<(i32, i32)>,
    self_effect: Option<HitEffect>,
    secondaries: Vec<HitEffect>,
    force_switch: bool,
    self_switch: bool,
}
