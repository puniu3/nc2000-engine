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
        flags: ms.flags.clone(),
        has_callbacks: ms.callbacks.clone(),
        hit: 0,
        last_hit: false,
        total_damage: None,
        source_effect: None,
        is_confusion_self_hit: false,
        spread_hit: false,
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
        flags: Vec::new(),
        has_callbacks: Vec::new(),
        hit: 0,
        last_hit: false,
        total_damage: None,
        source_effect: None,
        is_confusion_self_hit: true,
        spread_hit: false,
        move_hit_data: Vec::new(),
    }
}

/// Dispatch a move's own callbacks used as effect handlers (only pure-data-
/// adjacent ones for M1; everything else panics for divergence-driven porting).
pub fn dispatch_move_callback(
    b: &mut Battle,
    dex: &Dex,
    m: MoveId,
    callback_name: &str,
    _target: EvTarget,
    _source: Option<PokeId>,
    _relay: RV,
) -> RV {
    let key = dex.moves.key(m);
    match (key, callback_name) {
        ("struggle", "onModifyMove") => {
            if let Some(am) = b.active_move.as_mut() {
                am.move_type = "???".into();
            }
            RV::Undef
        }
        _ => panic!("unported move callback: {key} {callback_name}"),
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
        // beforeMoveCallback: no gen2 M1 moves carry one.
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

        // Extract the effective move data (primary or sub-block)
        let md = self.hit_effect_data(hit_data.clone());

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
                if matches!(hit_data, MoveHitData::Primary)
                    && move_id.map(|m| move_has_callback(dex, m, "onHit")).unwrap_or(false)
                {
                    let r = self.single_event(dex, "Hit", move_eff, StateLoc::None, EvTarget::Poke(t), Some(pokemon), move_eff, None);
                    hit_event = Tri::from_rv(&r);
                }
                if !is_self && !is_secondary {
                    self.run_event(dex, "Hit", EvTarget::Poke(t), Some(pokemon), move_eff, None, false, false);
                }
                // onAfterHit: M2
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
        // damageCallback moves are M2.
        if let Some(dk) = damage_kind {
            return match dk {
                crate::dex::FixedDamage::Level => DamageResult::Damage(self.poke(source).level as f64),
                crate::dex::FixedDamage::Amount(n) => DamageResult::Damage(n as f64),
            };
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

        let level = self.poke(source).level as f64;
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
