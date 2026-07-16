//! Battle driver — the faithful port of PS's battle loop for gen2stadium2
//! NC2000 (singles, no abilities, team preview, picked team size 3).
//!
//! Architecture decisions (fixed by the project plan, do not re-litigate):
//! - No dynamic event broadcast: hooks dispatch through match on condition id
//!   (see `conditions.rs`); handler collection/ordering mirrors PS
//!   `findEventHandlers`/`resolvePriority` exactly (see `events.rs`).
//! - PRNG consumption order must match PS exactly — snapshot parity asserts
//!   this via `prng_seed` at every snapshot point.
//! - Methods take `dex: &Dex` explicitly; `Battle` stays a plain clonable
//!   value (search-friendly).

pub mod actions;
pub mod choices;
pub mod conditions;
pub mod dmg;
pub mod essence;
pub mod events;
pub mod fieldfx;
pub mod items;
pub mod moveexec;
pub mod pokemon;
pub mod search;
pub mod turn;

pub use search::{Outcome, SearchChoice};

use crate::dex::{toid, CondId, Dex, EffectType, ItemId, MoveId};
use crate::prng::Prng;
use crate::state::*;

#[derive(Debug)]
pub enum EngineError {
    /// The engine has not been ported far enough to run this battle.
    Unimplemented(&'static str),
    InvalidChoice(String),
}

/// A player's team as delivered by the fixture (canonical validated sets).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PokemonSet {
    pub name: String,
    pub species: String,
    #[serde(default)]
    pub item: String,
    #[serde(default)]
    pub ability: String,
    pub moves: Vec<String>,
    pub level: u8,
    #[serde(default)]
    pub evs: Option<std::collections::BTreeMap<String, u16>>,
    #[serde(default)]
    pub ivs: Option<std::collections::BTreeMap<String, u8>>,
    #[serde(default)]
    pub happiness: Option<u8>,
    #[serde(default)]
    pub gender: Option<String>,
}

/// Which effect a piece of behavior belongs to (PS `Effect`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectHandle {
    None,
    Cond(CondId),
    MoveEff(MoveId),
    Item(ItemId),
    Format,
}

impl EffectHandle {
    pub fn is_none(&self) -> bool {
        matches!(self, EffectHandle::None)
    }
}

/// Relay variable (PS runEvent relayVar):
/// undefined | null | false | true | number | string (move ids from LockMove).
#[derive(Clone, Debug, PartialEq)]
pub enum RV {
    Undef,
    Null,
    False,
    True,
    Num(f64),
    Str(String),
}

impl RV {
    pub fn truthy(&self) -> bool {
        match self {
            RV::Undef | RV::Null | RV::False => false,
            RV::True => true,
            RV::Num(n) => *n != 0.0,
            RV::Str(s) => !s.is_empty(),
        }
    }

    pub fn as_num(&self) -> f64 {
        match self {
            RV::Num(n) => *n,
            RV::True => 1.0,
            _ => 0.0,
        }
    }

    pub fn from_bool(b: bool) -> RV {
        if b {
            RV::True
        } else {
            RV::False
        }
    }
}

/// PS `clampIntRange(num, min, max)` (lib/utils): truncates then clamps.
pub fn clamp_int_range(num: f64, min: Option<f64>, max: Option<f64>) -> f64 {
    let mut num = num.trunc();
    if num.is_nan() {
        num = 0.0;
    }
    if let Some(min) = min {
        if num < min {
            num = min;
        }
    }
    if let Some(max) = max {
        if num > max {
            num = max;
        }
    }
    num
}

/// `dex.trunc` for gens ≤4: JS `num >>> 0` (ToUint32).
pub fn tr(num: f64) -> f64 {
    if !num.is_finite() {
        return 0.0;
    }
    let n = num.trunc();
    let m = n.rem_euclid(4294967296.0);
    m
}

impl Battle {
    // ------------------------------------------------------------ logging

    pub fn add(&mut self, parts: &[&str]) {
        if !self.log_enabled {
            return;
        }
        self.log.push(format!("|{}", parts.join("|")));
    }

    pub fn add_split(&mut self, side_id: &str, secret: &[&str], shared: &[&str]) {
        if !self.log_enabled {
            return;
        }
        self.log.push(format!("|split|{side_id}"));
        self.add(secret);
        self.add(shared);
    }

    pub fn add_move(&mut self, parts: &[&str]) {
        if !self.log_enabled {
            return;
        }
        self.last_move_line = self.log.len() as i64;
        self.log.push(format!("|{}", parts.join("|")));
    }

    /// Toggle protocol-log recording (search mode: off). Disabling drops the
    /// accumulated log (construction lines included — nothing reads them in
    /// search mode) and clears the move-line cursor so `attr_last_move` can
    /// never touch stale lines.
    pub fn set_log_enabled(&mut self, on: bool) {
        self.log_enabled = on;
        if !on {
            self.last_move_line = -1;
            self.log = Vec::new();
        }
    }

    pub fn attr_last_move(&mut self, args: &[&str]) {
        if self.last_move_line < 0 {
            return;
        }
        let idx = self.last_move_line as usize;
        if args.contains(&"[still]") {
            // If no animation plays, the target should never be known
            let mut parts: Vec<String> = self.log[idx].split('|').map(String::from).collect();
            if parts.len() > 4 {
                parts[4] = String::new();
                self.log[idx] = parts.join("|");
            }
        }
        self.log[idx] = format!("{}|{}", self.log[idx], args.join("|"));
    }

    pub fn retarget_last_move(&mut self, new_target: &str) {
        if self.last_move_line < 0 {
            return;
        }
        let idx = self.last_move_line as usize;
        let mut parts: Vec<String> = self.log[idx].split('|').map(String::from).collect();
        if parts.len() > 4 {
            parts[4] = new_target.to_string();
            self.log[idx] = parts.join("|");
        }
    }

    /// PS `hint(text, once?, side?)` — we only ever emit non-side hints; the
    /// `once` behavior needs a seen-set which lives outside snapshots in PS
    /// (not part of state parity but part of LOG parity — implement via log
    /// scan: PS `hints` is a Set persisting for the battle).
    pub fn hint(&mut self, text: &str, once: bool) {
        if !self.log_enabled {
            return;
        }
        let line = format!("|-hint|{text}");
        if once && self.log.iter().any(|l| l == &line) {
            return;
        }
        // PS checks its hints Set only when `once` is passed... actually PS
        // checks `this.hints.has(hint)` ALWAYS and adds only when once=true.
        // So a non-once hint repeats unless an identical once-hint was stored.
        // We approximate: `once` hints emit at most once; non-once hints
        // always emit. The only NC2000-reachable hints are once=true
        // (residualdmg) and repeatable 1/256 hints (never stored).
        self.log.push(line);
    }

    // ------------------------------------------------------------ helpers

    pub fn poke(&self, id: PokeId) -> &Pokemon {
        &self.sides[id.side as usize].roster[id.slot as usize]
    }

    pub fn poke_mut(&mut self, id: PokeId) -> &mut Pokemon {
        &mut self.sides[id.side as usize].roster[id.slot as usize]
    }

    pub fn side(&self, n: usize) -> &Side {
        &self.sides[n]
    }

    pub fn active_id(&self, side: usize) -> Option<PokeId> {
        self.sides[side].active.map(|slot| PokeId { side: side as u8, slot })
    }

    pub fn foe_active_id(&self, side: usize) -> Option<PokeId> {
        self.active_id(1 - side)
    }

    /// PS `pokemon.fullname`: "p1: Abra".
    pub fn fullname(&self, id: PokeId) -> String {
        format!("p{}: {}", id.side + 1, self.poke(id).name)
    }

    /// PS `side.toString()`: "p1: P1". Log-only; empty when the log is off.
    pub fn side_str(&self, side_n: u8) -> String {
        if !self.log_enabled {
            return String::new();
        }
        format!("p{}: {}", side_n + 1, self.sides[side_n as usize].name)
    }

    /// queue.willAct(): any move/switch/instaswitch queued.
    pub fn queue_will_act(&self) -> bool {
        self.queue
            .iter()
            .any(|a| matches!(a.choice, ActionKind::Move { .. } | ActionKind::Switch { .. }))
    }

    /// queue.willMove(pokemon).
    pub fn queue_will_move(&self, pokemon: PokeId) -> bool {
        if self.poke(pokemon).fainted {
            return false;
        }
        self.queue
            .iter()
            .any(|a| matches!(a.choice, ActionKind::Move { .. }) && a.pokemon == Some(pokemon))
    }

    /// PS `pokemon.getSlot()`: "p1a".
    pub fn slot_str(&self, id: PokeId) -> String {
        let p = self.poke(id);
        let letter = (b'a' + p.position) as char;
        format!("p{}{}", id.side + 1, letter)
    }

    /// `pokemon.getSlot()` as (side, position) — the compact form stored in
    /// effect states (rendered "p1a" at essence time).
    pub fn slot_of(&self, id: PokeId) -> (u8, u8) {
        (id.side, self.poke(id).position)
    }

    /// getAtSlot on the compact (side, position) form.
    pub fn poke_at_slot_pos(&self, slot: (u8, u8)) -> Option<PokeId> {
        let id = self.active_id(slot.0 as usize)?;
        if self.poke(id).position == slot.1 {
            Some(id)
        } else {
            None
        }
    }

    /// PS `battle.getAtSlot("p2a")` — the active pokemon at that slot.
    pub fn poke_at_slot(&self, slot: &str) -> Option<PokeId> {
        let bytes = slot.as_bytes();
        if bytes.len() < 3 || bytes[0] != b'p' {
            return None;
        }
        let side = (bytes[1] - b'1') as usize;
        let position = bytes[2] - b'a';
        let id = self.active_id(side)?;
        if self.poke(id).position == position {
            Some(id)
        } else {
            None
        }
    }

    /// PS `pokemon.toString()`: active → "p1a: Abra", else fullname.
    /// Feeds protocol-log lines only — returns empty when the log is off so
    /// search stepping skips the formatting entirely.
    pub fn poke_str(&self, id: PokeId) -> String {
        if !self.log_enabled {
            return String::new();
        }
        let p = self.poke(id);
        if p.is_active {
            format!("{}: {}", self.slot_str(id), p.name)
        } else {
            self.fullname(id)
        }
    }

    /// PS `getFieldPositionValue`: side.n + sides.length * position.
    pub fn field_position_value(&self, id: PokeId) -> usize {
        id.side as usize + 2 * self.poke(id).position as usize
    }

    pub fn get_all_active(&self, include_fainted: bool) -> Vec<PokeId> {
        let mut out = Vec::new();
        for side in 0..2 {
            if let Some(id) = self.active_id(side) {
                if include_fainted || !self.poke(id).fainted {
                    out.push(id);
                }
            }
        }
        out
    }

    /// All pokemon in PS `getAllPokemon` order (side.pokemon display order).
    pub fn get_all_pokemon(&self) -> Vec<PokeId> {
        let mut out = Vec::new();
        for side in 0..2 {
            for &slot in &self.sides[side].party {
                out.push(PokeId { side: side as u8, slot });
            }
        }
        out
    }

    /// PS initEffectState: assign effectOrder if the state has an id and an
    /// active/attached target.
    pub fn init_effect_state(&mut self, mut state: EffectState, target_active: bool) -> EffectState {
        if !state.id.is_empty() && target_active {
            state.effect_order = self.effect_order;
            self.effect_order += 1;
        } else {
            state.effect_order = 0;
        }
        state
    }

    // -------------------------------------------------------- effect info

    pub fn effect_id<'d>(&self, dex: &'d Dex, effect: EffectHandle) -> &'d str {
        match effect {
            EffectHandle::Cond(c) => dex.conds_key(c),
            EffectHandle::MoveEff(m) => dex.moves.key(m),
            EffectHandle::Item(i) => dex.items.key(i),
            EffectHandle::Format => "gen2nc2000",
            EffectHandle::None => "",
        }
    }

    /// PS `effect.fullname` (what `-damage ... [from]` prints).
    pub fn effect_fullname(&self, dex: &Dex, effect: EffectHandle) -> String {
        match effect {
            EffectHandle::Cond(c) => dex.cond_display_name(c).to_string(),
            EffectHandle::MoveEff(m) => format!("move: {}", dex.move_static(m).name),
            EffectHandle::Item(i) => format!("item: {}", dex.items.get(i).name),
            EffectHandle::Format => "format: [Gen 2] NC 2000".to_string(),
            EffectHandle::None => String::new(),
        }
    }

    /// The active move's display name (synthetic moves have none).
    pub fn active_move_name(&self, dex: &Dex) -> String {
        self.active_move
            .as_ref()
            .and_then(|m| m.id)
            .map(|m| dex.move_static(m).name.clone())
            .unwrap_or_default()
    }

    /// PS `effect.name`.
    pub fn effect_name(&self, dex: &Dex, effect: EffectHandle) -> String {
        match effect {
            EffectHandle::Cond(c) => dex.cond_display_name(c).to_string(),
            EffectHandle::MoveEff(m) => dex.move_static(m).name.clone(),
            EffectHandle::Item(i) => dex.items.get(i).name.clone(),
            EffectHandle::Format => "[Gen 2] NC 2000".to_string(),
            EffectHandle::None => String::new(),
        }
    }

    pub fn effect_type(&self, dex: &Dex, effect: EffectHandle) -> EffectType {
        match effect {
            EffectHandle::Cond(c) => dex.cond_effect_type(c),
            EffectHandle::MoveEff(_) => EffectType::Move,
            EffectHandle::Item(_) => EffectType::Item,
            EffectHandle::Format => EffectType::Format,
            EffectHandle::None => EffectType::Condition,
        }
    }

    // -------------------------------------------------------- construction

    /// Constructs a battle in team-preview state from a fixture's seed and
    /// canonical teams. Mirrors PS `new Battle({formatid, seed}) + setPlayer×2`.
    pub fn from_fixture(
        dex: &Dex,
        seed: &str,
        p1: &[PokemonSet],
        p2: &[PokemonSet],
    ) -> Result<Battle, EngineError> {
        let prng = Prng::from_seed_str(seed)
            .ok_or_else(|| EngineError::InvalidChoice(format!("bad seed: {seed}")))?;
        let mut battle = Battle {
            prng,
            turn: 0,
            request_state: RequestState::None,
            mid_turn: false,
            started: false,
            ended: false,
            winner: None,
            field: Field::default(),
            sides: [Side::empty("P1"), Side::empty("P2")],
            queue: Default::default(),
            faint_queue: Vec::new(),
            log: Vec::new(),
            log_enabled: true,
            effect_order: 0,
            event_depth: 0,
            last_move_line: -1,
            last_successful_move_this_turn: None,
            last_damage: 0,
            quick_claw_roll: false,
            speed_order: [0, 1],
            format_data: EffectState { id: EffId::Format, ..Default::default() },
            sent_log_pos: 0,
            event_stack: Vec::new(),
            effect_stack: Vec::new(),
            active_move: None,
            active_pokemon: None,
            active_target: None,
            last_move_id: None,
            pending_boosts: None,
            listener_pool: Default::default(),
            battle_mask: crate::dex::CbMask::EMPTY,
        };
        // formatData/field states get effectOrder slots in PS construction
        // order: formatData (id set, no target → order 0? target absent →
        // effectOrder 0), field weather/terrain states (id '' → 0). None
        // consume the counter. Rule pseudo-weathers DO (id + Field target).
        battle.add(&["gametype", "singles"]);

        // Rules with runtime handlers become pseudo-weathers at construction.
        for rule in ["maxtotallevel", "stadiumsleepclause", "freezeclausemod"] {
            let cid = dex.conds_id(rule).expect("rule condition interned");
            let state = EffectState { id: EffId::Cond(cid), ..Default::default() };
            let state = battle.init_effect_state(state, true);
            battle.field.pseudo_weather.push((cid, state));
        }

        battle.set_player(dex, 0, p1)?;
        battle.set_player(dex, 1, p2)?;
        battle.battle_mask = battle.recompute_battle_mask(dex);
        Ok(battle)
    }

    fn set_player(&mut self, dex: &Dex, side_n: usize, team: &[PokemonSet]) -> Result<(), EngineError> {
        let name = if side_n == 0 { "P1" } else { "P2" };
        let mut side = Side::empty(name);
        for set in team {
            let slot = side.roster.len() as u8;
            let pokemon = self.new_pokemon(dex, set, side_n as u8, slot)?;
            side.roster.push(pokemon);
            side.party.push(slot);
            let pos = side.party.len() - 1;
            side.roster[slot as usize].position = pos as u8;
            side.pokemon_left += 1;
        }
        // PS Side constructor: pokemonLeft = pokemon.length (addPokemon also
        // increments — PS quirk: ctor sets pokemonLeft = this.pokemon.length
        // AFTER adds, overwriting; net effect = team size).
        side.pokemon_left = side.roster.len() as i32;
        self.sides[side_n] = side;
        self.add(&["player", &format!("p{}", side_n + 1), name, "", ""]);
        if side_n == 1 {
            self.start(dex);
        }
        Ok(())
    }

    fn new_pokemon(
        &mut self,
        dex: &Dex,
        set: &PokemonSet,
        side: u8,
        _slot: u8,
    ) -> Result<Pokemon, EngineError> {
        let species_id = dex
            .species
            .id(&toid(&set.species))
            .ok_or_else(|| EngineError::InvalidChoice(format!("unknown species {}", set.species)))?;
        let species = dex.species.get(species_id);

        let name = if set.name.is_empty() || set.name == set.species {
            species.name.clone()
        } else {
            set.name.clone()
        };
        let name = PokeName::new(&name.chars().take(20).collect::<String>());

        // gender: set.gender || species.gender || battle.sample(['M','F'])
        let gender = match set.gender.as_deref() {
            Some("M") => Gender::M,
            Some("F") => Gender::F,
            Some("N") => Gender::N,
            _ => match species.gender.as_deref() {
                Some("M") => Gender::M,
                Some("F") => Gender::F,
                Some("N") => Gender::N,
                _ => {
                    let pick = self.prng.sample_index(2);
                    if pick == 0 { Gender::M } else { Gender::F }
                }
            },
        };

        let happiness = set.happiness.unwrap_or(255);

        let mut base_move_slots = MoveSlots::default();
        for mv in &set.moves {
            let move_id = dex
                .moves
                .id(&toid(mv))
                .ok_or_else(|| EngineError::InvalidChoice(format!("unknown move {mv}")))?;
            let ms = dex.move_static(move_id);
            let pp_ups = if ms.no_pp_boosts { 0 } else { 3 };
            // calculatePP: pp * (5 + ppUps) / 5; gen<=2 && pp==40 → -= ppUps
            let mut pp = ms.pp * (5 + pp_ups) / 5;
            if ms.pp == 40 {
                pp -= pp_ups;
            }
            base_move_slots.push(MoveSlot { id: move_id, pp, maxpp: pp, disabled: false, used: false, shared: true });
        }

        // stat math (battle.statModify, nature neutral, tr = u32-trunc)
        let stat_keys = ["hp", "atk", "def", "spa", "spd", "spe"];
        let mut ivs = [31i32; 6];
        let mut evs = [0i32; 6];
        for (i, key) in stat_keys.iter().enumerate() {
            if let Some(m) = &set.ivs {
                if let Some(&v) = m.get(*key) {
                    ivs[i] = v as i32;
                }
            }
            if let Some(m) = &set.evs {
                if let Some(&v) = m.get(*key) {
                    evs[i] = v as i32;
                }
            }
            ivs[i] = ivs[i].clamp(0, 31) & 30; // gen<=2: DVs = even IVs
            evs[i] = evs[i].clamp(0, 255);
        }
        let bs = &species.base_stats;
        let bases = [bs.hp, bs.atk, bs.def, bs.spa, bs.spd, bs.spe];
        let level = set.level as i32;
        let mut stats = [0i32; 6];
        for i in 0..6 {
            let base = bases[i] as f64;
            let iv = ivs[i] as f64;
            let ev_term = tr(evs[i] as f64 / 4.0);
            stats[i] = if i == 0 {
                tr(tr(2.0 * base + iv + ev_term + 100.0) * level as f64 / 100.0 + 10.0) as i32
            } else {
                tr(tr(2.0 * base + iv + ev_term) * level as f64 / 100.0 + 5.0) as i32
            };
        }

        let item = if set.item.is_empty() { None } else { dex.items.id(&toid(&set.item)) };

        // gen 2 hidden power from DVs (ivs already clamped & 30)
        const HP_TYPES: [&str; 16] = [
            "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel",
            "Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark",
        ];
        let atk_dv = ivs[1] / 2;
        let def_dv = ivs[2] / 2;
        let spe_dv = ivs[5] / 2;
        let spc_dv = ivs[3] / 2;
        let hp_type = dex
            .type_id(HP_TYPES[(4 * (atk_dv % 4) + (def_dv % 4)) as usize])
            .expect("hidden power type interned");
        let hp_power = (5 * ((spc_dv >> 3) + 2 * (spe_dv >> 3) + 4 * (def_dv >> 3) + 8 * (atk_dv >> 3))
            + (spc_dv % 4))
            / 2
            + 31;

        let mut p = Pokemon {
            species: species_id,
            base_species: species_id,
            name,
            level: set.level,
            gender,
            happiness,
            set_ivs: ivs,
            set_evs: evs,
            base_move_slots: base_move_slots.clone(),
            hp_type,
            hp_power,
            base_hp_type: hp_type,
            base_hp_power: hp_power,
            base_stored_stats: [stats[1], stats[2], stats[3], stats[4], stats[5]],
            stored_stats: [stats[1], stats[2], stats[3], stats[4], stats[5]],
            base_maxhp: stats[0],
            maxhp: stats[0],
            hp: stats[0],
            status: Status::None,
            status_state: EffectState::default(),
            boosts: [0; 7],
            move_slots: base_move_slots,
            item,
            last_item: None,
            item_state: EffectState {
                id: item.map(EffId::Item).unwrap_or_default(),
                ..Default::default()
            },
            types: dex.species_types(species_id),
            volatiles: Default::default(),
            handler_mask: item.map(|i| dex.items.get(i).mask).unwrap_or(crate::dex::CbMask::EMPTY),
            transformed: false,
            fainted: false,
            faint_queued: false,
            is_active: false,
            is_started: false,
            position: 0,
            active_turns: 0,
            active_move_actions: 0,
            newly_switched: true,
            being_called_back: false,
            dragged_in: None,
            previously_switched_in: 0,
            switch_flag: SwitchFlag::No,
            force_switch_flag: false,
            skip_before_switch_out: false,
            trapped: false,
            maybe_trapped: false,
            last_move: None,
            last_move_encore: None,
            last_move_used: None,
            last_move_target_loc: None,
            move_this_turn: None,
            move_this_turn_result: MoveResult::Undef,
            move_last_turn_result: MoveResult::Undef,
            hurt_this_turn: None,
            stats_raised_this_turn: false,
            stats_lowered_this_turn: false,
            used_item_this_turn: false,
            last_damage: 0,
            attacked_by: Default::default(),
            times_attacked: 0,
            speed: stats[5],
        };
        // item/status states: PS initEffectState with target → effectOrder is
        // only assigned when the pokemon is active (it isn't at construction).
        let _ = side;
        let _ = &mut p;
        Ok(p)
    }

    /// PS `pokemon.details`: "Name, L51, M". Log-only.
    pub fn details(&self, dex: &Dex, id: PokeId) -> String {
        if !self.log_enabled {
            return String::new();
        }
        let p = self.poke(id);
        let species = dex.species.get(p.species);
        let mut d = species.name.clone();
        if p.level != 100 {
            d.push_str(&format!(", L{}", p.level));
        }
        if p.gender != Gender::N {
            d.push_str(&format!(", {}", p.gender.as_str()));
        }
        d
    }

    /// PS `pokemon.getHealth` → (secret, shared) strings. Log-only.
    pub fn get_health(&self, id: PokeId) -> (String, String) {
        if !self.log_enabled {
            return (String::new(), String::new());
        }
        let p = self.poke(id);
        if p.hp <= 0 {
            return ("0 fnt".into(), "0 fnt".into());
        }
        let secret = format!("{}/{}", p.hp, p.maxhp);
        let pixels = {
            let px = (48 * p.hp) / p.maxhp; // floor
            if px == 0 {
                1
            } else {
                px
            }
        };
        let mut shared = format!("{pixels}/48");
        let mut secret = secret;
        if p.status != Status::None {
            secret.push_str(&format!(" {}", p.status.as_str()));
            shared.push_str(&format!(" {}", p.status.as_str()));
        }
        (secret, shared)
    }

    /// `battle.start()` — both players set.
    fn start(&mut self, dex: &Dex) {
        self.started = true;
        self.add(&["gen", "2"]);
        self.add(&["tier", "[Gen 2] NC 2000"]);
        // rule onBegin log lines, in ruleset order
        self.add(&["rule", "Stadium Sleep Clause: Limit one foe put to sleep"]);
        self.add(&["rule", "Freeze Clause Mod: Limit one foe frozen"]);
        self.add(&["rule", "Species Clause: Limit one of each Pokémon"]);
        self.add(&["rule", "Item Clause: Limit 1 of each item"]);
        self.add(&["rule", "Endless Battle Clause: Forcing endless battles is banned"]);
        self.add(&["rule", "Event Moves Clause: Event-only moves are banned"]);
        self.add(&["rule", "Beat Up Nicknames Mod: Beat Up will not reveal any party members"]);
        // runPickTeam → Team Preview onTeamPreview
        self.add(&["clearpoke"]);
        for id in self.get_all_pokemon() {
            let details = self.details(dex, id);
            let side_id = format!("p{}", id.side + 1);
            let item = if self.poke(id).item.is_some() { "item" } else { "" };
            self.add(&["poke", &side_id, &details, item]);
        }
        self.make_request(dex, RequestState::TeamPreview);
        // queue.addChoice({choice:'start'})
        self.queue.push(Action {
            choice: ActionKind::Start,
            order: 2,
            priority: 0.0,
            fractional_priority: 0.0,
            speed: 1.0,
            pokemon: None,
        });
        self.mid_turn = true;
        // requestState is set → turnLoop deferred until choices commit
    }
}

impl Side {
    pub fn empty(name: &'static str) -> Side {
        Side {
            name,
            roster: Default::default(),
            party: Default::default(),
            active: None,
            pokemon_left: 0,
            total_fainted: 0,
            side_conditions: Default::default(),
            slot_conditions: Default::default(),
            handler_mask: crate::dex::CbMask::EMPTY,
            last_move: None,
            fainted_this_turn: None,
            fainted_last_turn: None,
            request: None,
            choice: Choice::default(),
        }
    }
}

