use nc2000_engine::{
    battle::SearchChoice,
    dex::Dex,
    state::{Battle, RequestState, Status},
};

fn allowed_move(name: &str) -> bool {
    matches!(
        name,
        "rockslide"
            | "earthquake"
            | "hiddenpowerice"
            | "hiddenpowerflying"
            | "thunderbolt"
            | "thunder"
            | "thundershock"
            | "drillpeck"
            | "peck"
            | "swift"
            | "hyperbeam"
            | "steelwing"
            | "rocksmash"
            | "mudslap"
            | "snore"
            | "swordsdance"
            | "agility"
            | "doubleteam"
            | "flash"
            | "rest"
    )
}

pub fn rejection(
    dex: &Dex,
    battle: &Battle,
    choices: [Option<SearchChoice>; 2],
) -> Option<&'static str> {
    if battle.request_state != RequestState::Move
        || battle.mid_turn
        || !battle.queue.is_empty()
        || !battle.faint_queue.is_empty()
    {
        return Some("decision_boundary");
    }
    if battle.field.weather.is_some() {
        return Some("weather");
    }
    if battle.field.pseudo_weather.iter().any(|(id, _)| {
        !matches!(
            dex.conds_key(*id),
            "maxtotallevel" | "stadiumsleepclause" | "freezeclausemod"
        )
    }) {
        return Some("field_effect");
    }
    for (s, side) in battle.sides.iter().enumerate() {
        if side.pokemon_left != 1 {
            return Some("multiple_pokemon");
        }
        if !side.side_conditions.is_empty() || !side.slot_conditions.is_empty() {
            return Some("side_effect");
        }
        let Some(active) = side.active else {
            return Some("no_active");
        };
        let mon = &side.roster[active as usize];
        if mon.fainted || mon.faint_queued || mon.hp <= 0 || mon.hp > mon.maxhp {
            return Some("invalid_hp");
        }
        if !matches!(
            mon.status,
            Status::None | Status::Slp | Status::Par | Status::Frz
        ) {
            return Some("residual_status");
        }
        if mon
            .volatiles
            .iter()
            .any(|(id, _)| !matches!(dex.conds_key(*id), "focusenergy" | "stall" | "mustrecharge"))
        {
            return Some("volatile_effect");
        }
        if let Some(item) = mon.item {
            let name = dex.items.key(item);
            if !dex.items.get(item).callbacks.is_empty()
                && !matches!(
                    name,
                    "thickclub"
                        | "brightpowder"
                        | "focusband"
                        | "berry"
                        | "goldberry"
                        | "berryjuice"
                        | "przcureberry"
                        | "mintberry"
                        | "miracleberry"
                        | "leftovers"
                        | "quickclaw"
                        | "kingsrock"
                        | "scopelens"
                        | "blackbelt"
                        | "blackglasses"
                        | "charcoal"
                        | "dragonfang"
                        | "hardstone"
                        | "magnet"
                        | "metalcoat"
                        | "miracleseed"
                        | "mysticwater"
                        | "nevermeltice"
                        | "pinkbow"
                        | "poisonbarb"
                        | "polkadotbow"
                        | "sharpbeak"
                        | "silverpowder"
                        | "softsand"
                        | "spelltag"
                        | "twistedspoon"
                        | "stick"
                        | "luckypunch"
                        | "bitterberry"
                        | "burntberry"
                        | "iceberry"
                        | "psncureberry"
                        | "mysteryberry"
                        | "mail"
                )
            {
                return Some("item_callback");
            }
        }
        let Some(SearchChoice::Move(id)) = choices[s] else {
            return Some("non_move");
        };
        let name = dex.moves.key(id);
        if name == "sleeptalk" {
            if mon.move_slots.iter().any(|slot| {
                let name = dex.moves.key(slot.id);
                name != "sleeptalk" && !allowed_move(name)
            }) {
                return Some("called_move");
            }
        } else if !allowed_move(name) {
            return Some("selected_move");
        }
    }
    None
}
