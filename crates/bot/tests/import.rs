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

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::RmConfig;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::state::{Battle, RequestState, Status, BOOST_NAMES};
use serde_json::Value;

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

    let pool = load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json"));
    let cfg = RmConfig { horizon: 6, ..RmConfig::default() };
    let mut agent = ProtocolAgent::new(dex, side, pool, cfg, 7 + side as u64);
    agent.set_own_team(own_sets);
    if pinned {
        agent.pin_opponent(opp_sets);
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
        // synthesized battle is searchable (fresh determinization per step)
        agent.step(dex, 2).unwrap();
        let _ = agent.best(dex);
    }
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
    let dex = load_dex();
    let mut stats = Stats::default();
    let mut n_fixtures = 0;
    for pool_dir in ["full", "puredata"] {
        let dir = repo_root().join("fixtures/corpus-v1").join(pool_dir);
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map(|x| x == "json").unwrap_or(false)
                    && !p.to_string_lossy().contains("DIVERGED")
            })
            .collect();
        files.sort();
        for f in files {
            let fixture: Value =
                serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
            n_fixtures += 1;
            for side in 0..2 {
                for pinned in [false, true] {
                    replay(&dex, &fixture, side, pinned, &mut stats);
                }
            }
        }
    }
    eprintln!(
        "corpus replay: {n_fixtures} fixtures, {} decisions, {} mismatches, {} illegal, \
         {} volatile-set diffs (diagnostic)",
        stats.decisions, stats.mismatches, stats.illegal_choices, stats.vol_diffs
    );
    for n in &stats.notes {
        eprintln!("  {n}");
    }
    for n in &stats.vol_notes {
        eprintln!("  vol: {n}");
    }
    assert_eq!(stats.mismatches, 0, "public-field mismatches");
    assert_eq!(stats.illegal_choices, 0, "played choices must be legal in synthesis");
}
