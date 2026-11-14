//! M15a protocol→state importer: corpus-replay certification.
//!
//! Every conformance fixture battle is replayed as ONE PLAYER would see it:
//! the protocol lines are channel-filtered (own secret lines kept, foe
//! secret lines dropped — the exact `|split|` semantics of a PS player
//! stream), a PS-shaped request JSON is built from the snapshot's own-side
//! truth (which is precisely what PS grants a player), and the importer
//! synthesizes a battle at every decision point. Asserted against the
//! omniscient snapshot truth:
//!
//! - own side EXACT: display order, hp/maxhp, status, per-move PP, item,
//!   boosts, faint flags;
//! - opponent: status/boosts exact, HP within the announced percentage bucket,
//!   PP marks of revealed moves exact + no unseen usage (a missed reveal
//!   fails here), faint flags;
//! - side conditions + weather key sets, turn, request kind;
//! - the fixture's actually-played choice is legal in the synthesized
//!   battle (RandomPlayerAI choices are PS-legal by construction — this is
//!   the native cousin of the PS-side zero-rejection gate);
//! - a short blind search runs on the synthesized battle without panicking
//!   (fresh determinization per iteration — belief/importer coherence).
//!
//! Both belief modes run: full-blind (fixture opponents are off-pool →
//! fallback roster) and pinned open-sheet (M12 policy).

use super::common;
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::position::{synthesize_spec, PositionSpec};
use nc2000_bot::preview::MetaPool;
use nc2000_bot::smmcts::RmConfig;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::state::{Battle, RequestState, Status, BOOST_NAMES};
use serde_json::Value;

#[test]
fn imported_trapping_preserves_both_sides_switch_legality() {
    use nc2000_engine::battle::SearchChoice;
    use nc2000_engine::validate::canonicalize_team;
    let dex = common::dex();
    let pool = common::meta_pool_v0();
    let learnsets = common::learnsets();
    for trap in ["Mean Look", "Spider Web", "Wrap", "Fire Spin"] {
        let team = serde_json::json!([
            {"species":"Smeargle","name":"Smeargle","level":50,"gender":"M",
             "moves":[trap,"Protect","Perish Song","Baton Pass"]},
            {"species":"Snorlax","name":"Snorlax","level":50,"gender":"M",
             "moves":["Rest","Body Slam","Earthquake","Curse"]},
            {"species":"Skarmory","name":"Skarmory","level":50,"gender":"M",
             "moves":["Whirlwind","Rest","Drill Peck","Toxic"]},
            {"species":"Starmie","level":50,"moves":["Surf"]},
            {"species":"Tauros","level":50,"moves":["Body Slam"]},
            {"species":"Machamp","level":50,"moves":["Cross Chop"]}
        ]);
        let canonical = canonicalize_team(&dex, &learnsets, &team.to_string());
        assert_eq!(canonical["ok"], true, "{canonical}");
        let sets: Vec<PokemonSet> = serde_json::from_value(canonical["team"].clone()).unwrap();
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &sets, &sets).unwrap();
        b.choose(&dex, 0, "team 1,2,3").unwrap();
        b.choose(&dex, 1, "team 2,1,3").unwrap();
        let check = |b: &Battle, stage: &str| {
            for observer in 0..2 {
                if !b.needs_choice()[observer] {
                    continue;
                }
                let mut agent =
                    ProtocolAgent::new(&dex, observer, pool.clone(), RmConfig::default(), 7);
                agent.set_own_team(sets.clone());
                agent.pin_opponent(&dex, sets.clone()).unwrap();
                let mut filter = SplitFilter::new(observer);
                for line in &b.log {
                    if filter.visible(line) {
                        agent.push_line(&dex, line);
                    }
                }
                let genders = sets
                    .iter()
                    .map(|s| (s.species.clone(), "M".into()))
                    .collect();
                let levels = sets.iter().map(|s| (s.species.clone(), 50)).collect();
                let mut request: Value = serde_json::from_str(&build_request(
                    &dex,
                    &b.essence(&dex),
                    observer,
                    &genders,
                    &levels,
                    true,
                ))
                .unwrap();
                if b.request_state == RequestState::Move {
                    request["active"][0]["trapped"] =
                        serde_json::json!(b.poke(b.active_id(observer).unwrap()).trapped);
                }
                assert!(agent.on_request(&dex, &request.to_string()).unwrap());
                let mut imported = agent.battle().unwrap().clone();
                let mut truth = b.clone();
                for side in 0..2 {
                    if truth.sides[side].request_state().is_none() {
                        continue;
                    }
                    let switches = |b: &mut Battle| {
                        let mut result: Vec<_> = b
                            .legal_choices(&dex, side)
                            .into_iter()
                            .filter_map(|c| {
                                let SearchChoice::Switch(pos) = c else {
                                    return None;
                                };
                                let slot = b.sides[side].party[pos as usize - 1];
                                Some(
                                    dex.species
                                        .key(b.sides[side].roster[slot as usize].species)
                                        .to_owned(),
                                )
                            })
                            .collect();
                        result.sort();
                        result
                    };
                    assert_eq!(
                        switches(&mut imported),
                        switches(&mut truth),
                        "{trap}, {stage}, observer {observer}, actor {side}"
                    );
                }
            }
        };
        check(&b, "before trapping");
        for _ in 0..10 {
            b.choose(&dex, 0, &format!("move {}", toid(trap))).unwrap();
            b.choose(&dex, 1, "move curse").unwrap();
            if b.poke(b.active_id(1).unwrap()).trapped {
                break;
            }
        }
        assert!(b.poke(b.active_id(1).unwrap()).trapped, "trap did not land");
        check(&b, "trapped");
        let mut bp = b.clone();
        bp.choose(&dex, 0, "move batonpass").unwrap();
        bp.choose(&dex, 1, "move curse").unwrap();
        assert_eq!(bp.request_state, nc2000_engine::state::RequestState::Switch);
        check(&bp, "baton pass switch prompt");
        bp.choose(&dex, 0, "switch 2").unwrap();
        if matches!(trap, "Mean Look" | "Spider Web") {
            assert!(
                bp.poke(bp.active_id(1).unwrap()).trapped,
                "{trap} must survive baton pass"
            );
            check(&bp, "baton pass kept trapping");
        } else {
            assert!(
                !bp.poke(bp.active_id(1).unwrap()).trapped,
                "{trap} binding must end on baton pass"
            );
            check(&bp, "baton pass released binding");
        }
        let mut knocked_out = b.clone();
        let source = knocked_out.active_id(0).unwrap();
        knocked_out.poke_mut(source).hp = 1;
        knocked_out.choose(&dex, 0, "move perishsong").unwrap();
        knocked_out.choose(&dex, 1, "move earthquake").unwrap();
        assert!(knocked_out.poke(source).fainted);
        check(&knocked_out, "forced replacement after source fainted");
        knocked_out.choose(&dex, 0, "switch 2").unwrap();
        assert!(!knocked_out.poke(knocked_out.active_id(1).unwrap()).trapped);
        check(&knocked_out, "source replaced after fainting");
        if matches!(trap, "Wrap" | "Fire Spin") {
            let mut expired = b.clone();
            for _ in 0..6 {
                expired.choose(&dex, 0, "move protect").unwrap();
                expired.choose(&dex, 1, "move curse").unwrap();
                if !expired.poke(expired.active_id(1).unwrap()).trapped {
                    break;
                }
            }
            assert!(!expired.poke(expired.active_id(1).unwrap()).trapped);
            check(&expired, "binding expired");
        }
        b.choose(&dex, 0, "switch 2").unwrap();
        b.choose(&dex, 1, "move curse").unwrap();
        assert!(!b.poke(b.active_id(1).unwrap()).trapped);
        check(&b, "source switched out");
    }
}

/// The announced HP bucket under HP Percentage Mod: ceil(100*hp/maxhp),
/// not-quite-full 100 knocked down to 99 (mirrors `Battle::get_health`).
fn px_of(hp: i64, maxhp: i64) -> i64 {
    if hp <= 0 {
        return 0;
    }
    let pct = (100 * hp + maxhp - 1) / maxhp;
    if pct == 100 && hp < maxhp {
        99
    } else {
        pct
    }
}

fn maxpp_of(dex: &Dex, id: &str) -> i64 {
    let mid = dex.moves.id(id).unwrap();
    let ms = dex.move_static(mid);
    let pp_ups = if ms.no_pp_boosts { 0 } else { 3 };
    let mut pp = ms.pp * (5 + pp_ups) / 5;
    if ms.pp == 40 {
        pp -= pp_ups;
    }
    pp as i64
}

/// `|split|` channel filter: own secret line kept, foe secret dropped.
struct SplitFilter {
    side: usize,
    skip_next: bool,
    keep_next_drop_after: bool,
}

impl SplitFilter {
    fn new(side: usize) -> SplitFilter {
        SplitFilter { side, skip_next: false, keep_next_drop_after: false }
    }

    /// Returns whether `line` is visible to this player.
    fn visible(&mut self, line: &str) -> bool {
        if self.skip_next {
            self.skip_next = false;
            return false;
        }
        if self.keep_next_drop_after {
            self.keep_next_drop_after = false;
            self.skip_next = true;
            return true;
        }
        if let Some(rest) = line.strip_prefix("|split|p") {
            let split_side = rest.as_bytes()[0] - b'1';
            if split_side as usize == self.side {
                self.keep_next_drop_after = true; // secret is ours
            } else {
                self.skip_next = true; // their secret; the shared line follows
            }
            return false;
        }
        true
    }
}

/// PS-shaped request JSON from the snapshot's own-side truth.
fn build_request(
    dex: &Dex,
    snap: &Value,
    side: usize,
    genders: &std::collections::HashMap<String, String>,
    levels: &std::collections::HashMap<String, u8>,
    our_choice_now: bool,
) -> String {
    let kind = snap["requestState"].as_str().unwrap_or("");
    let sd = &snap["sides"][side];
    let mut pokemon = Vec::new();
    for p in sd["pokemon"].as_array().unwrap() {
        let ident = p["ident"].as_str().unwrap();
        let name = ident.split_once(": ").map(|(_, n)| n).unwrap_or("");
        // PS request details always show the BASE forme; a transformed mon's
        // snapshot `species` is the copy target, so resolve the display from
        // the ident (fixture nicknames are the species name) instead.
        let disp = if p["transformed"].as_bool() == Some(true) {
            name.to_string()
        } else {
            let species = dex.species.id(p["species"].as_str().unwrap()).unwrap();
            dex.species.get(species).name.clone()
        };
        let level = levels.get(name).copied().unwrap_or(100);
        let gender = genders.get(name).cloned().unwrap_or_default();
        let mut details = format!("{disp}, L{level}");
        if !gender.is_empty() {
            details.push_str(&format!(", {gender}"));
        }
        let hp = p["hp"].as_i64().unwrap();
        let maxhp = p["maxhp"].as_i64().unwrap();
        let status = p["status"].as_str().unwrap_or("");
        let fainted = p["fainted"].as_bool().unwrap_or(false);
        let condition = if fainted || hp <= 0 {
            "0 fnt".to_string()
        } else if status.is_empty() {
            format!("{hp}/{maxhp}")
        } else {
            format!("{hp}/{maxhp} {status}")
        };
        let moves: Vec<String> = p["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap().to_string())
            .collect();
        pokemon.push(serde_json::json!({
            "ident": ident,
            "details": details,
            "condition": condition,
            "active": p["active"].as_bool().unwrap_or(false),
            "moves": moves,
            "item": p["item"].as_str().unwrap_or(""),
        }));
    }
    match kind {
        "teampreview" => serde_json::json!({
            "teamPreview": true,
            "maxChosenTeamSize": 3,
            "side": {"pokemon": pokemon},
        })
        .to_string(),
        "move" => {
            let active = sd["pokemon"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["active"].as_bool().unwrap_or(false));
            let moves: Vec<Value> = active
                .map(|p| {
                    let v = &p["volatiles"];
                    let locked_id = if v.get("mustrecharge").is_some() {
                        Some("recharge".to_string())
                    } else if let Some(s) = v.get("lockedmove").and_then(|x| x["move"].as_str()) {
                        Some(s.to_string())
                    } else if let Some(s) = v.get("twoturnmove").and_then(|x| x["move"].as_str()) {
                        Some(s.to_string())
                    } else if v.get("rollout").is_some() {
                        Some("rollout".to_string())
                    } else if v.get("bide").is_some() {
                        Some("bide".to_string())
                    } else {
                        None
                    };
                    if let Some(id) = locked_id {
                        vec![serde_json::json!({ "id": id })]
                    } else {
                        p["moves"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|m| {
                                let id = m["id"].as_str().unwrap();
                                serde_json::json!({
                                    "id": id,
                                    "pp": m["pp"].as_i64().unwrap_or(0),
                                    "maxpp": maxpp_of(dex, id),
                                    "disabled": m["disabled"].as_bool().unwrap_or(false),
                                })
                            })
                            .collect()
                    }
                })
                .unwrap_or_default();
            // PS computes `trapped` from state; mirror the same conditions
            // from the snapshot's volatiles
            let trapped = active
                .map(|p| {
                    let v = &p["volatiles"];
                    v.get("trapped").is_some() || v.get("partiallytrapped").is_some()
                })
                .unwrap_or(false);
            serde_json::json!({
                "active": [{"moves": moves, "trapped": trapped}],
                "side": {"pokemon": pokemon},
            })
            .to_string()
        }
        "switch" if our_choice_now => serde_json::json!({
            "forceSwitch": [true],
            "side": {"pokemon": pokemon},
        })
        .to_string(),
        _ => serde_json::json!({"wait": true, "side": {"pokemon": pokemon}}).to_string(),
    }
}

#[derive(Default)]
struct Stats {
    decisions: u32,
    mismatches: u32,
    illegal_choices: u32,
    vol_diffs: u32,
    sleep_sources: u32,
    /// Decision points exported to a `PositionSpec` and rebuilt from it.
    roundtrips: u32,
    /// Round trips the spec could not describe (counted, never skipped
    /// silently — a shape the solver cannot express is the one thing this
    /// gate exists to find).
    roundtrip_gaps: u32,
    notes: Vec<String>,
    vol_notes: Vec<String>,
}

impl Stats {
    fn miss(&mut self, ctx: &str, what: String) {
        self.mismatches += 1;
        if self.notes.len() < 40 {
            self.notes.push(format!("{ctx}: {what}"));
        }
    }
}

fn compare(
    dex: &Dex,
    b: &Battle,
    snap: &Value,
    side: usize,
    ctx: &str,
    stats: &mut Stats,
) {
    if b.turn as i64 != snap["turn"].as_i64().unwrap() {
        stats.miss(ctx, format!("turn {} vs {}", b.turn, snap["turn"]));
    }
    let kind = match b.request_state {
        RequestState::TeamPreview => "teampreview",
        RequestState::Move => "move",
        RequestState::Switch => "switch",
        RequestState::None => "",
    };
    if kind != snap["requestState"].as_str().unwrap_or("") {
        stats.miss(ctx, format!("requestState {kind} vs {}", snap["requestState"]));
    }
    // weather
    let w = b.field.weather.map(|c| dex.conds_key(c)).unwrap_or("");
    let tw = snap["field"]["weather"].as_str().unwrap_or("");
    if w != tw {
        stats.miss(ctx, format!("weather {w:?} vs {tw:?}"));
    }
    for s in 0..2 {
        // pokemon_left (terminal detection depends on it: win + Self-KO)
        if let Some(left) = snap["sides"][s]["pokemonLeft"].as_i64() {
            if b.sides[s].pokemon_left as i64 != left {
                stats.miss(
                    ctx,
                    format!("side {s} pokemonLeft {} vs {left}", b.sides[s].pokemon_left),
                );
            }
        }
        // side conditions (key sets)
        let mut mine: Vec<&str> =
            b.sides[s].side_conditions.iter().map(|(c, _)| dex.conds_key(*c)).collect();
        let mut truth: Vec<String> = snap["sides"][s]["sideConditions"]
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        mine.sort_unstable();
        truth.sort();
        if mine != truth.iter().map(|s| s.as_str()).collect::<Vec<_>>() {
            stats.miss(ctx, format!("side {s} conditions {mine:?} vs {truth:?}"));
        }
        let tmons = snap["sides"][s]["pokemon"].as_array().unwrap();
        // own display order must match exactly
        if s == side {
            let truth_order: Vec<&str> =
                tmons.iter().map(|p| p["species"].as_str().unwrap()).collect();
            let mine_order: Vec<&str> = b.sides[s]
                .party
                .iter()
                .map(|&sl| dex.species.key(b.sides[s].roster[sl as usize].species))
                .collect();
            if truth_order != mine_order {
                stats.miss(ctx, format!("own order {mine_order:?} vs {truth_order:?}"));
            }
        }
        for tp in tmons {
            let species = tp["species"].as_str().unwrap();
            // snapshot species reflects an active Transform — match the
            // current species first, base species otherwise
            let Some(mon) = b.sides[s]
                .roster
                .iter()
                .find(|p| dex.species.key(p.species) == species)
                .or_else(|| {
                    b.sides[s]
                        .roster
                        .iter()
                        .find(|p| dex.species.key(p.base_species) == species)
                })
            else {
                stats.miss(ctx, format!("side {s} missing {species}"));
                continue;
            };
            let thp = tp["hp"].as_i64().unwrap();
            let tmax = tp["maxhp"].as_i64().unwrap();
            let tstatus = tp["status"].as_str().unwrap_or("");
            let tfaint = tp["fainted"].as_bool().unwrap_or(false);
            if mon.fainted != tfaint {
                stats.miss(ctx, format!("{species} fainted {} vs {tfaint}", mon.fainted));
                continue;
            }
            // boosts (exact, both sides)
            for (i, name) in BOOST_NAMES.iter().enumerate() {
                let tb = tp["boosts"][name].as_i64().unwrap_or(0);
                if mon.boosts[i] as i64 != tb {
                    stats.miss(ctx, format!("{species} boost {name} {} vs {tb}", mon.boosts[i]));
                }
            }
            let mstatus = if mon.status == Status::Fnt { "fnt" } else { mon.status.as_str() };
            if !tfaint && mstatus != tstatus {
                stats.miss(ctx, format!("{species} status {mstatus:?} vs {tstatus:?}"));
            }
            if s == side {
                // own side exact
                if (mon.hp.max(0) as i64, mon.maxhp as i64) != (thp, tmax) {
                    stats.miss(
                        ctx,
                        format!("{species} hp {}/{} vs {thp}/{tmax}", mon.hp, mon.maxhp),
                    );
                }
                let titem = tp["item"].as_str().unwrap_or("");
                let mitem = mon.item.map(|i| dex.items.key(i)).unwrap_or("");
                if mitem != titem {
                    stats.miss(ctx, format!("{species} item {mitem:?} vs {titem:?}"));
                }
                if !mon.transformed && tp["transformed"].as_bool() != Some(true) {
                    for tm in tp["moves"].as_array().unwrap() {
                        let id = tm["id"].as_str().unwrap();
                        let tpp = tm["pp"].as_i64().unwrap();
                        let Some(ms) =
                            mon.move_slots.iter().find(|m| dex.moves.key(m.id) == id)
                        else {
                            stats.miss(ctx, format!("{species} missing move {id}"));
                            continue;
                        };
                        if ms.pp as i64 != tpp {
                            stats.miss(ctx, format!("{species} {id} pp {} vs {tpp}", ms.pp));
                        }
                    }
                }
            } else if !tfaint {
                // opponent: HP within the announced percentage bucket
                let tpx = px_of(thp, tmax);
                let mpx = px_of(mon.hp as i64, mon.maxhp as i64);
                if tpx != mpx {
                    stats.miss(ctx, format!("{species} hp bucket {mpx} vs {tpx}"));
                }
                // PP marks: revealed usage exact, no unseen usage. Skip
                // transformed / Mimic-overlaid movesets (both directions).
                let overlaid = mon.transformed
                    || tp["transformed"].as_bool() == Some(true)
                    || mon.move_slots.iter().any(|m| !m.shared)
                    || tp["volatiles"].get("mimic").is_some()
                    || tp["volatiles"].get("transform").is_some();
                if !overlaid {
                    for tm in tp["moves"].as_array().unwrap() {
                        let id = tm["id"].as_str().unwrap();
                        let tpp = tm["pp"].as_i64().unwrap();
                        let tused = maxpp_of(dex, id) - tpp;
                        let Some(ms) =
                            mon.move_slots.iter().find(|m| dex.moves.key(m.id) == id)
                        else {
                            if tused > 0 {
                                stats.miss(
                                    ctx,
                                    format!("{species} used move {id} not in imputation"),
                                );
                            }
                            continue;
                        };
                        let mused = (ms.maxpp - ms.pp) as i64;
                        if mused != tused {
                            stats.miss(
                                ctx,
                                format!("{species} {id} used {mused} vs {tused}"),
                            );
                        }
                    }
                }
                // volatile key sets: diagnostic only (hidden inference —
                // locked/stall — is best-effort by design)
                let mut mv: Vec<&str> =
                    mon.volatiles.iter().map(|(c, _)| dex.conds_key(*c)).collect();
                let mut tv: Vec<String> = tp["volatiles"]
                    .as_object()
                    .map(|o| o.keys().cloned().collect())
                    .unwrap_or_default();
                mv.sort_unstable();
                tv.sort();
                if mv != tv.iter().map(|s| s.as_str()).collect::<Vec<_>>() {
                    stats.vol_diffs += 1;
                    if stats.vol_notes.len() < 4000 {
                        stats.vol_notes.push(format!("{ctx}: {species} vols {mv:?} vs {tv:?}"));
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn replay(dex: &Dex, fixture: &Value, side: usize, pinned: bool, stats: &mut Stats) {
    let own_key = if side == 0 { "p1team" } else { "p2team" };
    let opp_key = if side == 0 { "p2team" } else { "p1team" };
    let own_sets: Vec<PokemonSet> = serde_json::from_value(fixture[own_key].clone()).unwrap();
    let opp_sets: Vec<PokemonSet> = serde_json::from_value(fixture[opp_key].clone()).unwrap();

    let pool_ref = common::meta_pool_v0();
    let cfg = RmConfig { horizon: 6, ..RmConfig::default() };
    let opp_ref = opp_sets.clone();
    let mut agent = ProtocolAgent::new(dex, side, pool_ref.clone(), cfg, 7 + side as u64);
    agent.set_own_team(own_sets);
    if pinned {
        agent.pin_opponent(dex, opp_sets).unwrap();
    }

    let snaps = fixture["snapshots"].as_array().unwrap();
    let choices = fixture["choices"].as_array().unwrap();

    // own preview details (level+gender) from our own |poke| lines
    let mut genders = std::collections::HashMap::new();
    let mut levels = std::collections::HashMap::new();
    for snap in snaps {
        for line in snap["log"].as_array().unwrap() {
            let line = line.as_str().unwrap();
            if let Some(rest) = line.strip_prefix(&format!("|poke|p{}|", side + 1)) {
                let details = rest.split('|').next().unwrap_or("");
                let mut name = details;
                let mut level = 100u8;
                let mut gender = String::new();
                for (i, part) in details.split(", ").enumerate() {
                    if i == 0 {
                        name = part;
                    } else if let Some(l) = part.strip_prefix('L') {
                        level = l.parse().unwrap_or(100);
                    } else if part == "M" || part == "F" {
                        gender = part.to_string();
                    }
                }
                genders.insert(name.to_string(), gender);
                levels.insert(name.to_string(), level);
            }
        }
    }

    let mut filter = SplitFilter::new(side);
    // (protocol side, nickname) -> was the latest sleep on that mon Rest?
    let mut sleep_rest: std::collections::HashMap<(usize, String), bool> =
        std::collections::HashMap::new();
    let mut fed = 0usize; // snapshots whose log has been fed
    for ch in choices {
        let ci = ch["index"].as_i64().unwrap();
        let ch_side = ch["side"].as_str().unwrap();
        let choice = ch["choice"].as_str().unwrap();
        if choice == "undo" {
            continue;
        }
        // state before this choice = last snapshot with afterLine < ci
        let mut j = 0;
        for (k, s) in snaps.iter().enumerate() {
            if s["afterLine"].as_i64().unwrap() < ci {
                j = k;
            }
        }
        while fed <= j {
            for line in snaps[fed]["log"].as_array().unwrap() {
                let line = line.as_str().unwrap();
                if filter.visible(line) {
                    if line.starts_with("|-status|") && line.contains("|slp") {
                        let f: Vec<&str> = line.split('|').collect();
                        let who = f.get(2).copied().unwrap_or("");
                        let s = if who.starts_with("p2") { 1 } else { 0 };
                        let nick = who.split(": ").nth(1).unwrap_or("").to_string();
                        sleep_rest.insert((s, nick), line.contains("move: Rest"));
                    }
                    agent.push_line(dex, line);
                }
            }
            fed += 1;
        }
        let us = format!("p{}", side + 1);
        if ch_side != us {
            continue;
        }
        let snap = &snaps[j];
        let req = build_request(dex, snap, side, &genders, &levels, true);
        stats.decisions += 1;
        let ctx = format!(
            "{} side {side} pinned {pinned} choice #{ci} turn {}",
            fixture["meta"]["index"], snap["turn"]
        );
        match agent.on_request(dex, &req) {
            Ok(true) => {}
            Ok(false) => {
                stats.miss(&ctx, "wait request at own decision".to_string());
                continue;
            }
            Err(e) => {
                stats.miss(&ctx, format!("on_request error: {e}"));
                continue;
            }
        }
        let battle = agent.battle().unwrap().clone();
        compare(dex, &battle, snap, side, &ctx, stats);
        let has_slp = (0..2usize).any(|s| {
            battle.sides[s].roster.iter().any(|p| p.status == Status::Slp)
        });
        let sample_heavy = stats.decisions % 8 == 1 || has_slp;
        if sample_heavy {
            roundtrip_position(
                dex,
                &agent,
                pool_ref,
                pinned.then_some(opp_ref.as_slice()),
                &ctx,
                stats,
            );
        }
        for s in 0..2usize {
            for slot in 0..battle.sides[s].roster.len() {
                let p = &battle.sides[s].roster[slot];
                if p.status != Status::Slp {
                    continue;
                }
                let name = dex.species.key(p.species).to_string();
                let Some((_, &rest)) = sleep_rest
                    .iter()
                    .find(|((ls, nick), _)| *ls == s && toid(nick) == name)
                else {
                    continue;
                };
                stats.sleep_sources += 1;
                let src_side = p.status_state.source.map(|x| x.side as usize);
                let ally_sourced = src_side == Some(s);
                if ally_sourced != rest {
                    stats.miss(
                        &ctx,
                        format!(
                            "{name} sleep source: protocol says {}, synthesis says {}",
                            if rest { "Rest (ally)" } else { "foe" },
                            if ally_sourced { "Rest (ally)" } else { "foe" }
                        ),
                    );
                }
            }
        }
        // the actually-played choice must be legal in the synthesized battle
        let mut bb = battle.clone();
        let legal: Vec<String> =
            bb.legal_choices(dex, side).iter().map(|c| c.to_input(dex)).collect();
        let normalized = normalize_choice(dex, choice);
        if !legal.contains(&normalized) {
            stats.illegal_choices += 1;
            if stats.notes.len() < 40 {
                stats
                    .notes
                    .push(format!("{ctx}: played {normalized:?} not in {legal:?}"));
            }
        }
        if sample_heavy {
            // synthesized battle is searchable (fresh determinization per step)
            agent.step(dex, 1).unwrap();
            let _ = agent.best(dex);
        }
    }
}

/// Solver round trip: export the decision point as a `PositionSpec`, send it
/// through JSON (the only form the browser ever sees), rebuild the whole
/// information set from it, and re-synthesize with the same seed. Same
/// tracker + same observer + same belief must give the same battle, down to
/// the essence the conformance fixtures are written against.
///
/// This is what lets a hand-entered position claim to be the same kind of
/// object as a live one: every field the importer reads is proven to survive
/// the trip, on real battles, at every decision point.
fn roundtrip_position(
    dex: &Dex,
    agent: &ProtocolAgent,
    pool: &MetaPool,
    pinned: Option<&[PokemonSet]>,
    ctx: &str,
    stats: &mut Stats,
) {
    const SEED: u64 = 0x5061_7274; // any fixed value: both sides must roll alike
    let Some(spec) = agent.to_position_spec(dex) else {
        stats.roundtrip_gaps += 1;
        return;
    };
    stats.roundtrips += 1;
    let json = match serde_json::to_string(&spec) {
        Ok(j) => j,
        Err(e) => return stats.miss(ctx, format!("spec serialize: {e}")),
    };
    let spec = match PositionSpec::parse(&json) {
        Ok(s) => s,
        Err(e) => return stats.miss(ctx, format!("spec reparse: {e}")),
    };
    // re-serializing the reparsed spec must be a fixed point (no field is
    // read into a shape that writes back differently)
    match serde_json::to_string(&spec) {
        Ok(j2) if j2 == json => {}
        Ok(_) => stats.miss(ctx, "spec is not a serialization fixed point".to_string()),
        Err(e) => return stats.miss(ctx, format!("spec re-serialize: {e}")),
    }
    let live = match agent.resynthesize(dex, SEED) {
        Ok(b) => b,
        Err(e) => return stats.miss(ctx, format!("resynthesize: {e}")),
    };
    let rebuilt = match synthesize_spec(dex, &spec, pool, pinned, SEED) {
        Ok(b) => b,
        Err(e) => return stats.miss(ctx, format!("synthesize_spec: {e}")),
    };
    if live.state_key128() != rebuilt.state_key128() {
        let (a, b) = (live.essence(dex), rebuilt.essence(dex));
        let detail = if a == b {
            "state keys differ but essences match".to_string()
        } else {
            first_essence_diff(&a, &b)
        };
        stats.miss(ctx, format!("position round trip diverged: {detail}"));
    }
}

/// The first differing path between two essences — a whole essence dump is
/// unreadable and a bare "differs" is unactionable.
fn first_essence_diff(a: &Value, b: &Value) -> String {
    fn walk(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
        if out.len() >= 3 || a == b {
            return;
        }
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => {
                let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
                keys.sort();
                keys.dedup();
                for k in keys {
                    walk(
                        x.get(k).unwrap_or(&Value::Null),
                        y.get(k).unwrap_or(&Value::Null),
                        &format!("{path}.{k}"),
                        out,
                    );
                }
            }
            (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
                for (i, (xa, yb)) in x.iter().zip(y).enumerate() {
                    walk(xa, yb, &format!("{path}[{i}]"), out);
                }
            }
            _ => out.push(format!("{path}: {a} != {b}")),
        }
    }
    let mut out = Vec::new();
    walk(a, b, "", &mut out);
    out.join("; ")
}

/// Fixture choices are already id-based ("move thief", "team 2, 4, 5",
/// "switch 2"); normalize spacing.
fn normalize_choice(dex: &Dex, c: &str) -> String {
    if let Some(rest) = c.strip_prefix("move ") {
        if let Some(mid) = dex.moves.id(&toid(rest)) {
            return format!("move {}", dex.moves.key(mid));
        }
    }
    if let Some(rest) = c.strip_prefix("team ") {
        let slots: Vec<String> =
            rest.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        return format!("team {}", slots.join(", "));
    }
    c.to_string()
}

#[test]
fn corpus_replay_both_modes() {
    let dex = common::dex();
    let mut stats = Stats::default();
    let corpus = common::corpus_json_values();
    let n_fixtures = corpus.len();
    for (idx, (_, fixture)) in corpus.iter().enumerate() {
        for side in 0..2 {
            replay(dex, fixture, side, false, &mut stats);
            if idx % 5 == 0 {
                replay(dex, fixture, side, true, &mut stats);
            }
        }
    }
    if stats.mismatches > 0 || stats.illegal_choices > 0 || stats.roundtrip_gaps > 0 {
        eprintln!(
            "corpus replay: {n_fixtures} fixtures, {} decisions, {} mismatches, {} illegal, \
             {} volatile-set diffs (diagnostic), {} sleep sources checked, \
             {} position round trips ({} inexpressible)",
            stats.decisions, stats.mismatches, stats.illegal_choices, stats.vol_diffs,
            stats.sleep_sources, stats.roundtrips, stats.roundtrip_gaps
        );
        for n in &stats.notes {
            eprintln!("  {n}");
        }
    }
    assert_eq!(stats.mismatches, 0, "public-field mismatches");
    assert_eq!(stats.illegal_choices, 0, "played choices must be legal in synthesis");
    assert!(stats.sleep_sources > 0, "no sleep reached the source check — gate is vacuous");
    assert!(stats.roundtrips > 0, "no position round trip ran — gate is vacuous");
    assert_eq!(stats.roundtrip_gaps, 0, "decision points a PositionSpec could not describe");
}
