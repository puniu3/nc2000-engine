use nc2000_bot::{observe::move_matches, preview::MetaPool, rng::SplitMix64, Belief, Observer};
use nc2000_engine::{dex::Dex, state::{Battle, MoveSlot, MoveSlots}, validate::Learnsets};
use serde_json::Value;

pub struct MoveModels {
    pub worlds: Vec<Battle>,
    pub names: Vec<String>,
    pub excluded: Vec<String>,
}

fn base_name(name: &str) -> &str {
    if name.starts_with("hiddenpower") { "hiddenpower" } else { name }
}

fn regular(sources: &Value, level: u8, generation: char) -> bool {
    sources.as_array().is_some_and(|sources| sources.iter().any(|source| {
        let Some(source) = source.as_str() else { return false; };
        if !source.starts_with(generation) { return false; }
        source.get(1..) == Some("M") || source.get(1..) == Some("T") ||
            source.get(1..2) == Some("L") && source[2..].parse::<u8>().is_ok_and(|n| n <= level)
    }))
}

pub fn generate(dex: &Dex, learnsets: &Learnsets, sources: &Value, pool: &MetaPool,
    battle: &Battle, obs: &Observer, seed: u64) -> Option<MoveModels> {
    if battle.sides.iter().any(|s| s.pokemon_left != 1) { return None; }
    let active = battle.active_id(obs.opp())?;
    let observation = &obs.mons()[active.slot as usize];
    if observation.revealed_moves.len() != 3 { return None; }
    let belief = Belief::new(dex, pool, obs);
    let mut rng = SplitMix64::new(seed);
    let imputed = belief.determinize(dex, battle, obs, &mut rng);
    let mon = imputed.poke(active);
    if mon.transformed || mon.move_slots.iter().any(|m| !m.shared) { return None; }
    let revealed = |slot: &MoveSlot| observation.revealed_moves.iter().any(|&m| move_matches(dex, slot.id, m));
    let known: Vec<_> = mon.base_move_slots.iter().filter(|m| revealed(m)).copied().collect();
    if known.len() != 3 { return None; }
    let species = dex.species.key(observation.species);
    let format = learnsets.species(species)?;
    let routes = &sources["species"][species];
    let post_transfer = known.iter().all(|slot|
        regular(&routes[base_name(dex.moves.key(slot.id))], observation.level, '2'));
    let mut names = Vec::new();
    let mut excluded = Vec::new();
    for name in &format.moves {
        if known.iter().any(|m| base_name(dex.moves.key(m.id)) == name) { continue; }
        let level_ok = format.move_min_level.get(name).is_none_or(|&n| n <= i64::from(observation.level));
        let route_ok = regular(&routes[name], observation.level, '2') ||
            post_transfer && regular(&routes[name], observation.level, '1');
        if level_ok && route_ok { names.push(name.clone()); } else { excluded.push(name.clone()); }
    }
    if let Some(slot) = mon.base_move_slots.iter().find(|m| !revealed(m)) {
        let name = base_name(dex.moves.key(slot.id));
        let index = names.iter().position(|n| n == name)?;
        let name = names.remove(index);
        names.insert(0, name);
    }
    names.push("<empty>".into());
    let mut worlds = Vec::new();
    for name in &names {
        let mut world = imputed.clone();
        let replacement = if name == "<empty>" { None } else {
            let typed;
            let move_name = if name == "hiddenpower" {
                typed = format!("hiddenpower{}", dex.type_name(mon.base_hp_type).to_ascii_lowercase());
                typed.as_str()
            } else { name };
            let id = dex.moves.id(move_name)?;
            let spec = dex.move_static(id);
            let boosts = if spec.no_pp_boosts { 0 } else { 3 };
            let pp = spec.pp * (5 + boosts) / 5 - if spec.pp == 40 { boosts } else { 0 };
            Some(MoveSlot { id, pp, maxpp: pp, disabled: false, used: false, shared: true })
        };
        let mut slots = MoveSlots::default();
        let mut inserted = false;
        for slot in &mon.base_move_slots {
            if revealed(slot) { slots.push(*slot); }
            else if let Some(slot) = replacement {
                assert!(!inserted);
                slots.push(slot);
                inserted = true;
            }
        }
        if !inserted {
            if let Some(slot) = replacement { slots.push(slot); }
        }
        world.poke_mut(active).base_move_slots = slots;
        world.poke_mut(active).move_slots = slots;
        world.refresh_poke_mask(dex, active);
        world.battle_mask = world.recompute_battle_mask(dex);
        worlds.push(world);
    }
    Some(MoveModels { worlds, names, excluded })
}
