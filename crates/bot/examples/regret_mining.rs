//! M17a two-stage search-regret miner over spectator logs.
//!
//! Screen every reconstructed decision with independently seeded blind
//! searches.  Each search is checkpointed at the deployed 10k budget and
//! then continued to the 30k oracle budget, so the product and oracle arms
//! share the exact same search prefix.  The reference is the modal product
//! action, not the logged human action: the corpus is being used to find bot
//! search errors rather than human mistakes.
//!
//! Candidates are only discoveries.  Confirm the top rows with fresh seeds
//! and equal root allocation inside one shared tree; this gives both
//! compared actions the full budget without letting the simultaneous
//! opponent condition its root policy on our action, and reports a
//! paired-search confidence interval.
//!
//! Screen:
//!   cargo run --release -p nc2000-bot --example regret_mining -- \
//!     --mode screen --corpus tmp/corpus-spectator --battles 0-96 \
//!     --product-iters 10000 --oracle-iters 30000 --samples 3 \
//!     --threads 8 --out tmp/regret-screen-0.jsonl
//! `--decisions LO-HI` restricts the per-battle decision index for focused
//! reconstruction/search diagnostics.
//!
//! Confirm after merging screen shards:
//!   cargo run --release -p nc2000-bot --example regret_mining -- \
//!     --mode confirm --input tmp/regret-screen.jsonl --top 100 \
//!     --iters 60000 --samples 8 --threads 8 \
//!     --out tmp/regret-confirm.jsonl
//!
//! Live decision-log v2 uses the exact submitted team, exact request, and
//! accumulated player-visible protocol captured by `tools/ps-client.js`:
//!   cargo run --release -p nc2000-bot --example regret_mining -- \
//!     --mode live-screen --input tmp/decisions.jsonl \
//!     --iters 60000 --samples 3 --out tmp/live-regret-screen.jsonl
//!   cargo run --release -p nc2000-bot --example regret_mining -- \
//!     --mode live-confirm --input tmp/live-regret-screen.jsonl \
//!     --live-log tmp/decisions.jsonl --iters 60000 --samples 8 \
//!     --out tmp/live-regret-confirm.jsonl

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nc2000_bot::blind::BlindSearch;
use nc2000_bot::corpus::{
    corpus_files, load_battle, load_sources, plain, reconstruct_agent_with_pool, CorpusBattle,
    HumanAction, SetSources,
};
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::regret::{mean, paired_regret};
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::state::{Battle, Status};

fn arg(args: &[String], key: &str, default: usize) -> usize {
    match args.iter().position(|a| a == key) {
        None => default,
        Some(index) => args
            .get(index + 1)
            .unwrap_or_else(|| panic!("{key} requires a value"))
            .parse()
            .unwrap_or_else(|_| panic!("invalid usize for {key}")),
    }
}

fn arg_u32(args: &[String], key: &str, default: u32) -> u32 {
    match args.iter().position(|a| a == key) {
        None => default,
        Some(index) => args
            .get(index + 1)
            .unwrap_or_else(|| panic!("{key} requires a value"))
            .parse()
            .unwrap_or_else(|_| panic!("invalid u32 for {key}")),
    }
}

fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    match args.iter().position(|a| a == key) {
        None => default,
        Some(index) => args
            .get(index + 1)
            .unwrap_or_else(|| panic!("{key} requires a value"))
            .parse()
            .unwrap_or_else(|_| panic!("invalid u64 for {key}")),
    }
}

fn arg_f(args: &[String], key: &str, default: f64) -> f64 {
    let value = match args.iter().position(|a| a == key) {
        None => default,
        Some(index) => args
            .get(index + 1)
            .unwrap_or_else(|| panic!("{key} requires a value"))
            .parse()
            .unwrap_or_else(|_| panic!("invalid float for {key}")),
    };
    assert!(value.is_finite(), "{key} must be finite");
    value
}

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    match args.iter().position(|a| a == key) {
        None => default.to_string(),
        Some(index) => args
            .get(index + 1)
            .filter(|value| !value.starts_with("--"))
            .unwrap_or_else(|| panic!("{key} requires a value"))
            .clone(),
    }
}

fn default_samples(mode: &str) -> usize {
    match mode {
        "screen" | "live-screen" => 3,
        _ => 8,
    }
}

fn validate_confirmation_budget(kind: &str, iters: u32, samples: usize) {
    assert!(iters > 0, "{kind} iterations must be positive");
    assert!(
        samples >= 2,
        "{kind} needs at least two independent searches"
    );
    assert!(
        iters.is_multiple_of(2) || samples.is_multiple_of(2),
        "{kind} forbids odd iterations with odd samples: alternating forced-action order \
         would be imbalanced"
    );
}

fn validate_screen_budget(product_iters: u32, oracle_iters: u32, samples: usize) {
    assert!(product_iters > 0, "product budget must be positive");
    assert!(samples > 0, "screen needs at least one independent search");
    assert!(
        oracle_iters >= product_iters,
        "oracle budget must cover product budget"
    );
}

fn range(s: &str) -> (usize, usize) {
    let mut p = s.split('-');
    let lo: usize = p
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("invalid range {s:?}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid range {s:?}"));
    let hi: usize = p
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("invalid range {s:?}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid range {s:?}"));
    assert!(p.next().is_none() && lo <= hi, "invalid range {s:?}");
    (lo, hi)
}

fn cfg() -> RmConfig {
    RmConfig {
        rule: SelRule::Ucb,
        c: 1.0,
        hp_buckets: 16,
        ..RmConfig::default()
    }
}

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn job_seed(base: u64, battle: usize, decision: usize, rep: usize, domain: u64) -> u64 {
    mix64(
        base ^ domain
            ^ (battle as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ (decision as u64).wrapping_mul(0xABC9_8388_FB8F_AC03)
            ^ (rep as u64).wrapping_mul(0x8CB9_2BA7_2F3D_8DD7),
    )
}

fn file_key(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn fnv1a64_update(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_update(0xcbf2_9ce4_8422_2325, bytes)
}

fn corpus_source_id(files: &[PathBuf]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for path in files {
        let name = file_key(path);
        let contents = std::fs::read(path)
            .unwrap_or_else(|e| panic!("read corpus file {}: {e}", path.display()));
        for part in [name.as_bytes(), contents.as_slice()] {
            hash = fnv1a64_update(hash, &(part.len() as u64).to_le_bytes());
            hash = fnv1a64_update(hash, part);
        }
    }
    format!("fnv1a64:{hash:016x}:{}files", files.len())
}

fn human_label(action: &HumanAction) -> String {
    match action {
        HumanAction::Move(key) => format!("move {key}"),
        HumanAction::Switch(species) => format!("switch {species}"),
    }
}

/// Semantic corpus key: moves use PS's plain Hidden Power id; switches use
/// species rather than the request-local numeric party position.
fn choice_label(b: &Battle, dex: &Dex, side: usize, choice: SearchChoice) -> String {
    match choice {
        SearchChoice::Move(id) => format!("move {}", plain(dex.moves.key(id))),
        SearchChoice::Switch(pos) => {
            let slot = b.sides[side]
                .party
                .get(pos as usize - 1)
                .copied()
                .unwrap_or(0);
            let species = b.sides[side].roster[slot as usize].species;
            format!("switch {}", dex.species.key(species))
        }
        other => other.to_input(dex),
    }
}

fn action_class(dex: &Dex, label: &str) -> String {
    if label.starts_with("switch ") {
        return "switch".into();
    }
    label
        .strip_prefix("move ")
        .and_then(|key| dex.moves.id(key))
        .map(|id| dex.moves.get(id).category.clone())
        .unwrap_or_else(|| "other".into())
}

fn modal(labels: &[String]) -> (String, f64) {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for label in labels {
        *counts.entry(label).or_default() += 1;
    }
    let (label, count) = counts
        .into_iter()
        .max_by(|(la, ca), (lb, cb)| ca.cmp(cb).then_with(|| lb.cmp(la)))
        .unwrap_or(("", 0));
    (label.to_string(), count as f64 / labels.len().max(1) as f64)
}

fn tags(b: &Battle, dex: &Dex, side: usize) -> Vec<String> {
    let mut out = Vec::new();
    out.push(if b.turn <= 10 {
        "phase:early".into()
    } else if b.turn <= 30 {
        "phase:mid".into()
    } else {
        "phase:late".into()
    });
    out.push(format!(
        "alive:{}v{}",
        b.sides[side].pokemon_left,
        b.sides[1 - side].pokemon_left
    ));
    if let (Some(me), Some(foe)) = (b.active_id(side), b.active_id(1 - side)) {
        out.push(format!(
            "active:{}-{}",
            dex.species.key(b.poke(me).species),
            dex.species.key(b.poke(foe).species)
        ));
        for (who, id) in [("self", me), ("foe", foe)] {
            let p = b.poke(id);
            let hp = p.hp as f64 / p.maxhp.max(1) as f64;
            let hp_tag = if hp <= 0.25 {
                "low"
            } else if hp <= 0.6 {
                "mid"
            } else {
                "high"
            };
            out.push(format!("{who}-hp:{hp_tag}"));
            if p.status != Status::None {
                out.push(format!("{who}-status:{}", p.status.as_str()));
            }
            if p.boosts.iter().any(|&stage| stage != 0) {
                out.push(format!("{who}:boosted"));
            }
            for key in [
                "substitute",
                "confusion",
                "leechseed",
                "curse",
                "meanlook",
                "partiallytrapped",
                "perishsong",
                "encore",
                "disable",
            ] {
                if dex.conds_id(key).is_some_and(|id| p.has_volatile(id)) {
                    out.push(format!("{who}:{key}"));
                }
            }
        }
    }
    for (who, s) in [("self", side), ("foe", 1 - side)] {
        for key in ["spikes", "reflect", "lightscreen"] {
            if dex
                .conds_id(key)
                .is_some_and(|id| b.sides[s].has_side_condition(id))
            {
                out.push(format!("{who}:{key}"));
            }
        }
    }
    if b.field.weather.is_some() {
        out.push("weather".into());
    }
    out
}

#[derive(Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[allow(dead_code)]
struct LiveLogRow {
    version: u32,
    #[serde(rename = "type")]
    kind: String,
    room: String,
    battle: usize,
    decision: usize,
    rqid: u64,
    side: usize,
    turn: u16,
    server: String,
    format: String,
    mode: String,
    driver: String,
    iterations: u32,
    seed: i64,
    team_label: String,
    own_team: Vec<PokemonSet>,
    request: serde_json::Value,
    protocol_reset: bool,
    protocol_delta: Vec<String>,
    submitted: String,
    root_policy: serde_json::Value,
    state_view_kind: String,
    state_view: serde_json::Value,
    #[serde(skip)]
    input_line: usize,
}

struct LiveRoom {
    rows: Vec<LiveLogRow>,
}

struct LiveLog {
    source_id: String,
    rooms: Vec<LiveRoom>,
}

fn validate_live_row(path: &str, line_no: usize, row: &LiveLogRow) {
    let bad = |why: &str| panic!("{path}:{line_no}: invalid decision-log v2 row: {why}");
    if row.version != 2 || row.kind != "decision" {
        bad("expected version=2 type=decision");
    }
    if row.room.is_empty() {
        bad("room is empty");
    }
    if row.side > 1 {
        bad("side must be 0 or 1");
    }
    if row.format != "gen2nintendocup2000noohkostadium2strict" {
        bad("unsupported format");
    }
    if !matches!(row.mode.as_str(), "blind" | "open") {
        bad("mode must be blind or open");
    }
    if !matches!(row.driver.as_str(), "search" | "random") {
        bad("driver must be search or random");
    }
    if row.own_team.is_empty() {
        bad("ownTeam is empty");
    }
    if !row.request.is_object() {
        bad("request must be an object");
    }
    if row.request["rqid"].as_u64() != Some(row.rqid) {
        bad("request.rqid differs from rqid");
    }
    let expected_side = if row.side == 0 { "p1" } else { "p2" };
    if row.request["side"]["id"].as_str() != Some(expected_side) {
        bad("request.side.id differs from side");
    }
    if row.submitted.trim().is_empty() {
        bad("submitted is empty");
    }
    if row.protocol_delta.iter().any(|line| !line.starts_with('|')) {
        bad("protocolDelta contains a non-protocol line");
    }
    if row.state_view_kind != "diagnostic-imputed" {
        bad("unknown stateViewKind");
    }
}

/// Parse once, validate every non-empty line, and deduplicate before protocol
/// accumulation. Rooms retain input order even when several games' logs are
/// interleaved in one JSONL file.
fn parse_live_rooms(path: &str, text: &str) -> Vec<LiveRoom> {
    let mut rooms = Vec::<LiveRoom>::new();
    let mut room_indexes = HashMap::<String, usize>::new();
    let mut seen = HashMap::<(String, u64), String>::new();
    let mut duplicates = 0usize;
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let line_no = line_index + 1;
        let mut row: LiveLogRow =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("{path}:{line_no}: {e}"));
        row.input_line = line_no;
        validate_live_row(path, line_no, &row);
        let dedupe_key = (row.room.clone(), row.rqid);
        if let Some(first) = seen.get(&dedupe_key) {
            assert!(
                first == line,
                "{path}:{line_no}: conflicting duplicate room+rqid {}+{}",
                row.room,
                row.rqid
            );
            duplicates += 1;
            continue;
        }
        seen.insert(dedupe_key, line.to_string());
        let room_index = match room_indexes.get(&row.room).copied() {
            Some(index) => index,
            None => {
                let index = rooms.len();
                rooms.push(LiveRoom { rows: Vec::new() });
                room_indexes.insert(row.room.clone(), index);
                index
            }
        };
        rooms[room_index].rows.push(row);
    }
    if duplicates != 0 {
        eprintln!("deduplicated {duplicates} repeated room+rqid rows");
    }
    rooms
}

fn load_live_log(path: &str) -> LiveLog {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let source_id = format!("fnv1a64:{:016x}", fnv1a64(text.as_bytes()));
    let rooms = parse_live_rooms(path, &text);
    LiveLog { source_id, rooms }
}

fn request_is_preview(request: &serde_json::Value) -> bool {
    request["teamPreview"].as_bool().unwrap_or(false)
}

fn apply_protocol_delta(protocol: &mut Vec<String>, reset: bool, delta: &[String]) {
    if reset {
        protocol.clear();
    }
    protocol.extend(delta.iter().cloned());
}

fn submitted_label(request: &serde_json::Value, submitted: &str) -> Result<String, String> {
    let submitted = submitted.trim();
    if let Some(raw) = submitted.strip_prefix("move ") {
        let token = raw.split_whitespace().next().unwrap_or("");
        if token.is_empty() {
            return Err("empty move choice".into());
        }
        let key = match token.parse::<usize>() {
            Ok(slot) => request["active"][0]["moves"]
                .as_array()
                .and_then(|moves| moves.get(slot.checked_sub(1)?))
                .and_then(|m| m["id"].as_str())
                .ok_or_else(|| format!("move slot {slot} absent from request"))?,
            Err(_) => token,
        };
        return Ok(format!("move {}", plain(&toid(key))));
    }
    if let Some(raw) = submitted.strip_prefix("switch ") {
        let slot: usize = raw
            .split_whitespace()
            .next()
            .unwrap_or("")
            .parse()
            .map_err(|_| format!("invalid switch choice {submitted:?}"))?;
        let details = request["side"]["pokemon"]
            .as_array()
            .and_then(|mons| mons.get(slot.checked_sub(1)?))
            .and_then(|m| m["details"].as_str())
            .ok_or_else(|| format!("switch slot {slot} absent from request"))?;
        let species = details.split(',').next().unwrap_or("").trim();
        if species.is_empty() {
            return Err(format!("switch slot {slot} has no species"));
        }
        return Ok(format!("switch {}", toid(species)));
    }
    if submitted == "pass" {
        return Ok("pass".into());
    }
    Err(format!("unsupported submitted choice {submitted:?}"))
}

fn live_seed(base: u64, row: &LiveLogRow, rep: usize, domain: u64) -> u64 {
    let room_hash = fnv1a64(row.room.as_bytes());
    mix64(
        base ^ domain
            ^ room_hash
            ^ row.rqid.wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ (rep as u64).wrapping_mul(0x8CB9_2BA7_2F3D_8DD7),
    )
}

fn reconstruct_live_agent(
    dex: &Dex,
    pool: &MetaPool,
    row: &LiveLogRow,
    protocol: &[String],
    seed: u64,
) -> Result<ProtocolAgent, String> {
    let mut agent = ProtocolAgent::new(dex, row.side, pool.clone(), cfg(), seed);
    agent.set_own_team(row.own_team.clone());
    for line in protocol {
        agent.push_line(dex, line);
    }
    let request = serde_json::to_string(&row.request).map_err(|e| e.to_string())?;
    if !agent.on_request(dex, &request)? {
        return Err("wait-request".into());
    }
    Ok(agent)
}

struct LoadedBattle {
    index: usize,
    path: PathBuf,
    data: CorpusBattle,
}

#[allow(clippy::too_many_arguments)]
fn screen_decision(
    dex: &Dex,
    src: &SetSources,
    pool: &MetaPool,
    corpus_fingerprint: &str,
    loaded: &LoadedBattle,
    di: usize,
    product_iters: u32,
    oracle_iters: u32,
    samples: usize,
    base_seed: u64,
) -> serde_json::Value {
    let d = &loaded.data.decisions[di];
    let base = serde_json::json!({
        "mode": "screen", "corpus_fingerprint":corpus_fingerprint,
        "battle": loaded.index, "file": file_key(&loaded.path),
        "decision": di, "side": d.side, "turn": d.turn,
    });
    if oracle_iters < product_iters || samples == 0 {
        return merge(base, serde_json::json!({"skip":"bad-budget"}));
    }

    let mut labels: Vec<String> = Vec::new();
    let mut classes: Vec<String> = Vec::new();
    let mut seed_means: Vec<Vec<f64>> = Vec::new();
    let mut seed_visits: Vec<Vec<u32>> = Vec::new();
    let mut product_choices = Vec::new();
    let mut oracle_choices = Vec::new();
    let mut position_tags = Vec::new();

    for rep in 0..samples {
        let synth_seed = job_seed(base_seed, loaded.index, di, rep, 0x51A7_0001);
        let Some(agent) = reconstruct_agent_with_pool(
            dex,
            src,
            pool.clone(),
            &loaded.data.lines,
            &loaded.data.evidence,
            d,
            synth_seed,
        ) else {
            return merge(base, serde_json::json!({"skip":"reconstruct"}));
        };
        let battle = agent.battle().unwrap();
        let belief = agent.belief().unwrap();
        let observer = agent.observer().unwrap();
        let mut search = BlindSearch::new(
            battle,
            dex,
            cfg(),
            d.side,
            job_seed(base_seed, loaded.index, di, rep, 0x51A7_1001),
        );
        let rep_labels: Vec<String> = search
            .actions()
            .iter()
            .map(|&a| choice_label(battle, dex, d.side, a))
            .collect();
        if rep == 0 {
            labels = rep_labels;
            classes = labels.iter().map(|a| action_class(dex, a)).collect();
            seed_means = vec![Vec::new(); labels.len()];
            seed_visits = vec![Vec::new(); labels.len()];
            position_tags = tags(battle, dex, d.side);
        } else if rep_labels != labels {
            return merge(base, serde_json::json!({"skip":"action-set-drift"}));
        }
        if labels.len() <= 1 {
            return merge(
                base,
                serde_json::json!({
                    "skip":"trivial", "n_actions": labels.len()
                }),
            );
        }

        search.step(dex, belief, observer, product_iters);
        let product = search.best().unwrap();
        product_choices.push(choice_label(battle, dex, d.side, product));
        search.step(dex, belief, observer, oracle_iters - product_iters);
        let oracle = search.best().unwrap();
        oracle_choices.push(choice_label(battle, dex, d.side, oracle));
        let means = search.means();
        for i in 0..labels.len() {
            seed_means[i].push(means[i]);
            seed_visits[i].push(search.visits()[i]);
        }
    }

    let (reference, product_stability) = modal(&product_choices);
    let (candidate, oracle_stability) = modal(&oracle_choices);
    let reference_idx = labels.iter().position(|a| a == &reference).unwrap();
    let candidate_idx = labels.iter().position(|a| a == &candidate).unwrap();
    let regret = paired_regret(&seed_means[candidate_idx], &seed_means[reference_idx]).mean;
    let human = human_label(&d.action);
    let actions: Vec<serde_json::Value> = (0..labels.len())
        .map(|i| {
            serde_json::json!({
                "action": labels[i], "class": classes[i], "mean": mean(&seed_means[i]),
                "min_visits": seed_visits[i].iter().copied().min().unwrap_or(0),
                "total_visits": seed_visits[i].iter().sum::<u32>(),
                "seed_means": seed_means[i], "seed_visits": seed_visits[i],
            })
        })
        .collect();
    merge(
        base,
        serde_json::json!({
            "human": human, "human_in_set": labels.contains(&human),
            "reference": reference, "candidate": candidate,
            "reference_class": classes[reference_idx],
            "candidate_class": classes[candidate_idx],
            "regret": regret,
            "product_stability": product_stability, "oracle_stability": oracle_stability,
            "product_choices": product_choices, "oracle_choices": oracle_choices,
            "actions": actions, "tags": position_tags,
            "product_iters": product_iters, "oracle_iters": oracle_iters, "samples": samples,
        }),
    )
}

fn live_base(input: &str, source_id: &str, row: &LiveLogRow) -> serde_json::Value {
    serde_json::json!({
        "mode":"live-screen", "source":"live-decision-log-v2",
        "input_file":file_key(Path::new(input)), "input_fingerprint":source_id,
        "input_line":row.input_line,
        "room":row.room, "rqid":row.rqid, "battle":row.battle,
        "decision":row.decision, "side":row.side, "turn":row.turn,
        "driver":row.driver, "live_mode":row.mode,
        "submitted_raw":row.submitted,
    })
}

#[allow(clippy::too_many_arguments)]
fn live_screen_decision(
    dex: &Dex,
    pool: &MetaPool,
    input: &str,
    source_id: &str,
    row: &LiveLogRow,
    protocol: &[String],
    iters: u32,
    samples: usize,
    base_seed: u64,
) -> serde_json::Value {
    let base = live_base(input, source_id, row);
    if row.driver == "random" {
        return merge(base, serde_json::json!({"skip":"random"}));
    }
    if row.mode == "open" {
        // v2 records only ownTeam. Replaying an open-sheet product action
        // without its pinned opponent team would silently compare policies
        // from different information sets.
        return merge(
            base,
            serde_json::json!({"skip":"open-opponent-team-unavailable"}),
        );
    }
    if request_is_preview(&row.request) {
        return merge(base, serde_json::json!({"skip":"preview"}));
    }
    if row.request["wait"].as_bool().unwrap_or(false) {
        return merge(base, serde_json::json!({"skip":"wait"}));
    }
    let submitted = match submitted_label(&row.request, &row.submitted) {
        Ok(label) => label,
        Err(detail) => {
            return merge(
                base,
                serde_json::json!({"skip":"submitted-invalid", "detail":detail}),
            );
        }
    };

    let mut labels = Vec::<String>::new();
    let mut classes = Vec::<String>::new();
    let mut seed_means = Vec::<Vec<f64>>::new();
    let mut seed_visits = Vec::<Vec<u32>>::new();
    let mut oracle_choices = Vec::<String>::new();
    let mut position_tags = Vec::<String>::new();
    for rep in 0..samples {
        let synth_seed = live_seed(base_seed, row, rep, 0x11E0_0001);
        let agent = match reconstruct_live_agent(dex, pool, row, protocol, synth_seed) {
            Ok(agent) => agent,
            Err(detail) => {
                return merge(
                    base,
                    serde_json::json!({"skip":"reconstruct", "detail":detail}),
                );
            }
        };
        let battle = agent.battle().unwrap();
        let belief = agent.belief().unwrap();
        let observer = agent.observer().unwrap();
        let mut search = BlindSearch::new(
            battle,
            dex,
            cfg(),
            row.side,
            live_seed(base_seed, row, rep, 0x11E0_1001),
        );
        let rep_labels: Vec<String> = search
            .actions()
            .iter()
            .map(|&action| choice_label(battle, dex, row.side, action))
            .collect();
        if rep == 0 {
            labels = rep_labels;
            classes = labels
                .iter()
                .map(|label| action_class(dex, label))
                .collect();
            seed_means = vec![Vec::new(); labels.len()];
            seed_visits = vec![Vec::new(); labels.len()];
            position_tags = tags(battle, dex, row.side);
        } else if rep_labels != labels {
            return merge(base, serde_json::json!({"skip":"action-set-drift"}));
        }
        if labels.len() <= 1 {
            return merge(
                base,
                serde_json::json!({"skip":"trivial", "n_actions":labels.len()}),
            );
        }
        search.step(dex, belief, observer, iters);
        let Some(best) = search.best() else {
            return merge(base, serde_json::json!({"skip":"no-best"}));
        };
        oracle_choices.push(choice_label(battle, dex, row.side, best));
        let means = search.means();
        for index in 0..labels.len() {
            seed_means[index].push(means[index]);
            seed_visits[index].push(search.visits()[index]);
        }
    }

    let Some(reference_idx) = labels.iter().position(|label| label == &submitted) else {
        return merge(
            base,
            serde_json::json!({
                "skip":"submitted-missing", "submitted":submitted, "actions":labels,
            }),
        );
    };
    let (candidate, oracle_stability) = modal(&oracle_choices);
    let candidate_idx = labels.iter().position(|label| label == &candidate).unwrap();
    let estimate = paired_regret(&seed_means[candidate_idx], &seed_means[reference_idx]);
    let actions: Vec<serde_json::Value> = (0..labels.len())
        .map(|index| {
            serde_json::json!({
                "action":labels[index], "class":classes[index],
                "mean":mean(&seed_means[index]),
                "min_visits":seed_visits[index].iter().copied().min().unwrap_or(0),
                "total_visits":seed_visits[index].iter().sum::<u32>(),
                "seed_means":seed_means[index], "seed_visits":seed_visits[index],
            })
        })
        .collect();
    merge(
        base,
        serde_json::json!({
            "human":submitted, "submitted":submitted,
            "reference":submitted, "candidate":candidate,
            "reference_class":classes[reference_idx],
            "candidate_class":classes[candidate_idx],
            "regret":estimate.mean,
            "oracle_stability":oracle_stability, "oracle_choices":oracle_choices,
            "actions":actions, "tags":position_tags,
            "oracle_iters":iters, "samples":samples,
            "logged_iterations":row.iterations,
        }),
    )
}

fn merge(mut a: serde_json::Value, b: serde_json::Value) -> serde_json::Value {
    a.as_object_mut()
        .unwrap()
        .extend(b.as_object().unwrap().clone());
    a
}

fn write_rows(path: &str, mut rows: Vec<serde_json::Value>) {
    rows.sort_by(|a, b| {
        (
            a["rank"].as_u64().unwrap_or(u64::MAX),
            a["battle"].as_u64().unwrap_or(u64::MAX),
            a["decision"].as_u64().unwrap_or(u64::MAX),
            a["input_line"].as_u64().unwrap_or(u64::MAX),
        )
            .cmp(&(
                b["rank"].as_u64().unwrap_or(u64::MAX),
                b["battle"].as_u64().unwrap_or(u64::MAX),
                b["decision"].as_u64().unwrap_or(u64::MAX),
                b["input_line"].as_u64().unwrap_or(u64::MAX),
            ))
            .then_with(|| a["room"].as_str().cmp(&b["room"].as_str()))
            .then_with(|| a["rqid"].as_u64().cmp(&b["rqid"].as_u64()))
    });
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for row in rows {
        writeln!(out, "{row}").unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_live_screen(
    dex: &Dex,
    root: &Path,
    input: &str,
    iters: u32,
    samples: usize,
    threads: usize,
    seed: u64,
    out_path: &str,
) {
    assert!(iters > 0, "live screen iterations must be positive");
    assert!(
        samples > 0,
        "live screen needs at least one independent search"
    );
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let live_log = load_live_log(input);
    let source_id = live_log.source_id;
    let rooms = live_log.rooms;
    let decisions: usize = rooms.iter().map(|room| room.rows.len()).sum();
    eprintln!(
        "live-screen rooms {} decisions {decisions} iters {iters} samples {samples} threads {threads}",
        rooms.len()
    );
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let room_index = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(room) = rooms.get(room_index) else {
                    return;
                };
                let mut protocol = Vec::<String>::new();
                let mut local = Vec::with_capacity(room.rows.len());
                for row in &room.rows {
                    apply_protocol_delta(&mut protocol, row.protocol_reset, &row.protocol_delta);
                    local.push(live_screen_decision(
                        dex, &pool, input, &source_id, row, &protocol, iters, samples, seed,
                    ));
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_multiple_of(25) || n == decisions {
                        eprintln!("  {n}/{decisions} decisions");
                    }
                }
                rows.lock().unwrap().extend(local);
            });
        }
    });
    write_rows(out_path, rows.into_inner().unwrap());
    eprintln!("done -> {out_path}");
}

#[allow(clippy::too_many_arguments)]
fn run_screen(
    dex: &Dex,
    root: &Path,
    corpus: &str,
    battle_range: (usize, usize),
    decision_range: (usize, usize),
    product_iters: u32,
    oracle_iters: u32,
    samples: usize,
    threads: usize,
    per_battle: usize,
    seed: u64,
    out_path: &str,
) {
    validate_screen_budget(product_iters, oracle_iters, samples);
    let src = load_sources(dex, root);
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let files = corpus_files(&root.join(corpus));
    let corpus_fingerprint = corpus_source_id(&files);
    let loaded: Vec<LoadedBattle> = files
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i >= battle_range.0 && *i <= battle_range.1)
        .map(|(index, path)| {
            let data = load_battle(&path);
            LoadedBattle { index, path, data }
        })
        .collect();
    let decisions: usize = loaded
        .iter()
        .map(|b| {
            b.data
                .decisions
                .iter()
                .take(per_battle)
                .enumerate()
                .filter(|(di, _)| *di >= decision_range.0 && *di <= decision_range.1)
                .count()
        })
        .sum();
    eprintln!(
        "screen battles {} decisions {decisions} product {product_iters} oracle {oracle_iters} \
         samples {samples} threads {threads}",
        loaded.len()
    );

    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= loaded.len() {
                    return;
                }
                let b = &loaded[j];
                let mut local = Vec::new();
                for di in (0..b.data.decisions.len().min(per_battle))
                    .filter(|di| *di >= decision_range.0 && *di <= decision_range.1)
                {
                    local.push(screen_decision(
                        dex,
                        &src,
                        &pool,
                        &corpus_fingerprint,
                        b,
                        di,
                        product_iters,
                        oracle_iters,
                        samples,
                        seed,
                    ));
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_multiple_of(100) || n == decisions {
                        eprintln!("  {n}/{decisions} decisions");
                    }
                }
                rows.lock().unwrap().extend(local);
            });
        }
    });
    write_rows(out_path, rows.into_inner().unwrap());
    eprintln!("done -> {out_path}");
}

#[derive(Clone)]
struct Candidate {
    rank: usize,
    row: serde_json::Value,
}

fn validate_screen_row(path: &str, line_no: usize, row: &serde_json::Value) {
    let bad = |why: &str| panic!("{path}:{line_no}: invalid screen row: {why}");
    if row["corpus_fingerprint"].as_str().is_none_or(str::is_empty)
        || row["file"].as_str().is_none_or(str::is_empty)
    {
        bad("corpus_fingerprint/file missing");
    }
    if row["battle"].as_u64().is_none()
        || row["decision"].as_u64().is_none()
        || row["side"].as_u64().is_none_or(|side| side > 1)
        || row["turn"].as_u64().is_none()
    {
        bad("decision coordinates missing or invalid");
    }
    for field in [
        "human",
        "reference",
        "candidate",
        "reference_class",
        "candidate_class",
    ] {
        if row[field].as_str().is_none() {
            bad(&format!("{field} missing"));
        }
    }
    if row["tags"].as_array().is_none()
        || row["actions"].as_array().is_none()
        || row["regret"].as_f64().is_none()
        || row["product_iters"].as_u64().is_none_or(|n| n == 0)
        || row["oracle_iters"].as_u64().is_none_or(|n| n == 0)
        || row["samples"].as_u64().is_none_or(|n| n == 0)
    {
        bad("search statistics missing or invalid");
    }
}

fn hypothesis_key(row: &serde_json::Value, live: bool) -> String {
    if live {
        serde_json::json!([
            row["input_fingerprint"],
            row["input_line"],
            row["room"],
            row["rqid"],
            row["battle"],
            row["decision"],
            row["side"],
            row["turn"],
            row["reference"],
            row["candidate"],
        ])
        .to_string()
    } else {
        serde_json::json!([
            row["corpus_fingerprint"],
            row["file"],
            row["battle"],
            row["decision"],
            row["side"],
            row["turn"],
            row["reference"],
            row["candidate"],
        ])
        .to_string()
    }
}

fn same_hypothesis_payload(a: &serde_json::Value, b: &serde_json::Value, live: bool) -> bool {
    if !live {
        return a == b;
    }
    let mut a = a.clone();
    let mut b = b.clone();
    a.as_object_mut().unwrap().remove("input_file");
    b.as_object_mut().unwrap().remove("input_file");
    a == b
}

fn dedupe_hypotheses(
    path: &str,
    rows: Vec<(usize, serde_json::Value)>,
    live: bool,
) -> Vec<serde_json::Value> {
    let mut seen = HashMap::<String, (usize, usize)>::new();
    let mut out = Vec::<serde_json::Value>::new();
    let mut duplicates = 0usize;
    for (line_no, row) in rows {
        let key = hypothesis_key(&row, live);
        if let Some(&(first_line, first_index)) = seen.get(&key) {
            assert!(
                same_hypothesis_payload(&out[first_index], &row, live),
                "{path}:{line_no}: conflicting duplicate hypothesis (first at line {first_line})"
            );
            duplicates += 1;
            continue;
        }
        seen.insert(key, (line_no, out.len()));
        out.push(row);
    }
    if duplicates != 0 {
        eprintln!("deduplicated {duplicates} repeated hypotheses from {path}");
    }
    out
}

fn cmp_field_str(a: &serde_json::Value, b: &serde_json::Value, field: &str) -> std::cmp::Ordering {
    a[field].as_str().cmp(&b[field].as_str())
}

fn cmp_field_u64(a: &serde_json::Value, b: &serde_json::Value, field: &str) -> std::cmp::Ordering {
    a[field].as_u64().cmp(&b[field].as_u64())
}

fn cmp_offline_identity(a: &serde_json::Value, b: &serde_json::Value) -> std::cmp::Ordering {
    cmp_field_str(a, b, "corpus_fingerprint")
        .then_with(|| cmp_field_str(a, b, "file"))
        .then_with(|| cmp_field_u64(a, b, "battle"))
        .then_with(|| cmp_field_u64(a, b, "decision"))
        .then_with(|| cmp_field_u64(a, b, "side"))
        .then_with(|| cmp_field_u64(a, b, "turn"))
        .then_with(|| cmp_field_str(a, b, "reference"))
        .then_with(|| cmp_field_str(a, b, "candidate"))
}

fn cmp_live_identity(a: &serde_json::Value, b: &serde_json::Value) -> std::cmp::Ordering {
    cmp_field_str(a, b, "input_fingerprint")
        .then_with(|| cmp_field_u64(a, b, "input_line"))
        .then_with(|| cmp_field_str(a, b, "room"))
        .then_with(|| cmp_field_u64(a, b, "rqid"))
        .then_with(|| cmp_field_u64(a, b, "battle"))
        .then_with(|| cmp_field_u64(a, b, "decision"))
        .then_with(|| cmp_field_u64(a, b, "side"))
        .then_with(|| cmp_field_u64(a, b, "turn"))
        .then_with(|| cmp_field_str(a, b, "reference"))
        .then_with(|| cmp_field_str(a, b, "candidate"))
}

#[derive(Clone)]
struct LiveSnapshot {
    row: LiveLogRow,
    protocol: Vec<String>,
}

fn live_snapshots(rooms: Vec<LiveRoom>, wanted: &HashSet<usize>) -> HashMap<usize, LiveSnapshot> {
    let mut snapshots = HashMap::new();
    for room in rooms {
        let mut protocol = Vec::<String>::new();
        for row in room.rows {
            apply_protocol_delta(&mut protocol, row.protocol_reset, &row.protocol_delta);
            if wanted.contains(&row.input_line) {
                snapshots.insert(
                    row.input_line,
                    LiveSnapshot {
                        row,
                        protocol: protocol.clone(),
                    },
                );
            }
        }
    }
    snapshots
}

fn live_confirm_base(candidate: &Candidate) -> serde_json::Value {
    let row = &candidate.row;
    serde_json::json!({
        "mode":"live-confirm", "rank":candidate.rank,
        "source":"live-decision-log-v2", "input_file":row["input_file"],
        "input_fingerprint":row["input_fingerprint"],
        "input_line":row["input_line"], "room":row["room"], "rqid":row["rqid"],
        "battle":row["battle"], "decision":row["decision"],
        "side":row["side"], "turn":row["turn"],
    })
}

fn validate_live_screen_row(path: &str, line_no: usize, row: &serde_json::Value) {
    let bad = |why: &str| panic!("{path}:{line_no}: invalid live-screen row: {why}");
    if row["input_file"].as_str().is_none() {
        bad("input_file missing");
    }
    if row["input_fingerprint"].as_str().is_none() {
        bad("input_fingerprint missing");
    }
    if row["input_line"].as_u64().is_none_or(|line| line == 0) {
        bad("input_line must be positive");
    }
    if row["room"].as_str().is_none_or(str::is_empty) || row["rqid"].as_u64().is_none() {
        bad("room/rqid missing");
    }
    if row["battle"].as_u64().is_none()
        || row["decision"].as_u64().is_none()
        || row["side"].as_u64().is_none_or(|side| side > 1)
        || row["turn"].as_u64().is_none()
    {
        bad("decision coordinates missing or invalid");
    }
    for field in [
        "human",
        "submitted",
        "reference",
        "candidate",
        "reference_class",
        "candidate_class",
    ] {
        if row[field].as_str().is_none() {
            bad(&format!("{field} missing"));
        }
    }
    if row["tags"].as_array().is_none()
        || row["actions"].as_array().is_none()
        || row["regret"].as_f64().is_none()
        || row["oracle_iters"].as_u64().is_none_or(|n| n == 0)
        || row["samples"].as_u64().is_none_or(|n| n == 0)
    {
        bad("search statistics missing or invalid");
    }
}

#[allow(clippy::too_many_arguments)]
fn confirm_live_candidate(
    dex: &Dex,
    pool: &MetaPool,
    candidate: &Candidate,
    snapshot: Option<&LiveSnapshot>,
    source_id: &str,
    iters: u32,
    samples: usize,
    base_seed: u64,
) -> serde_json::Value {
    let screen = &candidate.row;
    let base = live_confirm_base(candidate);
    let Some(snapshot) = snapshot else {
        return merge(base, serde_json::json!({"skip":"input-line-missing"}));
    };
    let live = &snapshot.row;
    if screen["input_fingerprint"].as_str() != Some(source_id)
        || screen["room"].as_str() != Some(&live.room)
        || screen["rqid"].as_u64() != Some(live.rqid)
    {
        return merge(base, serde_json::json!({"skip":"source-mismatch"}));
    }
    let Some(reference) = screen["reference"].as_str() else {
        return merge(base, serde_json::json!({"skip":"reference-missing"}));
    };
    let Some(proposed) = screen["candidate"].as_str() else {
        return merge(base, serde_json::json!({"skip":"candidate-missing"}));
    };
    let mut reference_values = Vec::with_capacity(samples);
    let mut candidate_values = Vec::with_capacity(samples);
    for rep in 0..samples {
        let synth_seed = live_seed(base_seed, live, rep, 0x11C0_0001);
        let agent = match reconstruct_live_agent(dex, pool, live, &snapshot.protocol, synth_seed) {
            Ok(agent) => agent,
            Err(detail) => {
                return merge(
                    base,
                    serde_json::json!({"skip":"reconstruct", "detail":detail}),
                );
            }
        };
        let battle = agent.battle().unwrap();
        let belief = agent.belief().unwrap();
        let observer = agent.observer().unwrap();
        let mut choice_battle = battle.clone();
        let actions = choice_battle.legal_choices(dex, live.side);
        let labels: Vec<String> = actions
            .iter()
            .map(|&action| choice_label(battle, dex, live.side, action))
            .collect();
        let Some(reference_index) = labels.iter().position(|label| label == reference) else {
            return merge(base, serde_json::json!({"skip":"reference-missing"}));
        };
        let Some(candidate_index) = labels.iter().position(|label| label == proposed) else {
            return merge(base, serde_json::json!({"skip":"candidate-missing"}));
        };
        let mut search = BlindSearch::new(
            battle,
            dex,
            cfg(),
            live.side,
            live_seed(base_seed, live, rep, 0x11C0_1001),
        );
        for iteration in 0..iters {
            let order = if (iteration + rep as u32).is_multiple_of(2) {
                [reference_index, candidate_index]
            } else {
                [candidate_index, reference_index]
            };
            for index in order {
                search.step_forced(dex, belief, observer, index);
            }
        }
        let means = search.means();
        reference_values.push(means[reference_index]);
        candidate_values.push(means[candidate_index]);
    }
    let estimate = paired_regret(&candidate_values, &reference_values);
    merge(
        base,
        serde_json::json!({
            "human":screen["human"], "submitted":screen["submitted"],
            "reference":reference, "candidate":proposed,
            "reference_class":screen["reference_class"],
            "candidate_class":screen["candidate_class"], "tags":screen["tags"],
            "discovery_regret":screen["regret"],
            "regret":estimate.mean, "ci95":estimate.ci95, "lower95":estimate.lower95,
            "reference_values":reference_values, "candidate_values":candidate_values,
            "iters_per_action":iters, "samples":samples,
            "estimand":"shared-opponent-root-equal-allocation",
        }),
    )
}

fn bind_offline_candidate(files: &[PathBuf], candidate: &Candidate) {
    let row = &candidate.row;
    let battle = row["battle"].as_u64().unwrap() as usize;
    let decision = row["decision"].as_u64().unwrap() as usize;
    let path = files.get(battle).unwrap_or_else(|| {
        panic!(
            "confirm rank {}: battle index {battle} is absent from corpus",
            candidate.rank
        )
    });
    let expected_file = file_key(path);
    assert_eq!(
        row["file"].as_str(),
        Some(expected_file.as_str()),
        "confirm rank {}: corpus filename mismatch at battle {battle}",
        candidate.rank
    );
    let data = load_battle(path);
    let point = data.decisions.get(decision).unwrap_or_else(|| {
        panic!(
            "confirm rank {}: decision {decision} is absent from battle {battle}",
            candidate.rank
        )
    });
    let expected_human = human_label(&point.action);
    assert!(
        row["side"].as_u64() == Some(point.side as u64)
            && row["turn"].as_u64() == Some(point.turn as u64)
            && row["human"].as_str() == Some(expected_human.as_str()),
        "confirm rank {}: filename/decision/side/turn/action coordinates do not bind",
        candidate.rank
    );
}

#[allow(clippy::too_many_arguments)]
fn confirm_candidate(
    dex: &Dex,
    src: &SetSources,
    pool: &MetaPool,
    files: &[PathBuf],
    candidate: &Candidate,
    iters: u32,
    samples: usize,
    base_seed: u64,
) -> serde_json::Value {
    let row = &candidate.row;
    let bi = row["battle"].as_u64().unwrap() as usize;
    let di = row["decision"].as_u64().unwrap() as usize;
    let Some(path) = files.get(bi) else {
        return serde_json::json!({
            "mode":"confirm", "rank":candidate.rank, "battle":bi, "decision":di,
            "skip":"battle-missing"
        });
    };
    let data = load_battle(path);
    let Some(d) = data.decisions.get(di) else {
        return serde_json::json!({
            "mode":"confirm", "rank":candidate.rank, "battle":bi, "decision":di,
            "skip":"decision-missing"
        });
    };
    let reference = row["reference"].as_str().unwrap_or("");
    let proposed = row["candidate"].as_str().unwrap_or("");
    let mut reference_values = Vec::new();
    let mut candidate_values = Vec::new();

    for rep in 0..samples {
        let synth_seed = job_seed(base_seed, bi, di, rep, 0xC0F1_0001);
        let Some(agent) = reconstruct_agent_with_pool(
            dex,
            src,
            pool.clone(),
            &data.lines,
            &data.evidence,
            d,
            synth_seed,
        ) else {
            return merge(
                row.clone(),
                serde_json::json!({
                    "mode":"confirm", "rank":candidate.rank, "skip":"reconstruct"
                }),
            );
        };
        let battle = agent.battle().unwrap();
        let belief = agent.belief().unwrap();
        let observer = agent.observer().unwrap();
        let mut choice_battle = battle.clone();
        let actions = choice_battle.legal_choices(dex, d.side);
        let labels: Vec<String> = actions
            .iter()
            .map(|&a| choice_label(battle, dex, d.side, a))
            .collect();
        let Some(ri) = labels.iter().position(|a| a == reference) else {
            return merge(
                row.clone(),
                serde_json::json!({
                    "mode":"confirm", "rank":candidate.rank, "skip":"reference-missing"
                }),
            );
        };
        let Some(ci) = labels.iter().position(|a| a == proposed) else {
            return merge(
                row.clone(),
                serde_json::json!({
                    "mode":"confirm", "rank":candidate.rank, "skip":"candidate-missing"
                }),
            );
        };
        let search_seed = job_seed(base_seed, bi, di, rep, 0xC0F1_1001);
        let mut search = BlindSearch::new(battle, dex, cfg(), d.side, search_seed);
        for iteration in 0..iters {
            // Alternate within-pair order so neither arm is systematically
            // measured against the less-developed shared opponent root.
            let order = if (iteration + rep as u32).is_multiple_of(2) {
                [ri, ci]
            } else {
                [ci, ri]
            };
            for index in order {
                search.step_forced(dex, belief, observer, index);
            }
        }
        let means = search.means();
        reference_values.push(means[ri]);
        candidate_values.push(means[ci]);
    }
    let estimate = paired_regret(&candidate_values, &reference_values);
    serde_json::json!({
        "mode":"confirm", "rank":candidate.rank,
        "corpus_fingerprint":row["corpus_fingerprint"],
        "battle":bi, "file":row["file"], "decision":di,
        "side":row["side"], "turn":row["turn"], "human":row["human"],
        "reference":reference, "candidate":proposed,
        "reference_class":row["reference_class"], "candidate_class":row["candidate_class"],
        "tags":row["tags"], "discovery_regret":row["regret"],
        "regret":estimate.mean, "ci95":estimate.ci95, "lower95":estimate.lower95,
        "reference_values":reference_values, "candidate_values":candidate_values,
        "iters_per_action":iters, "samples":samples,
        "estimand":"shared-opponent-root-equal-allocation",
    })
}

#[allow(clippy::too_many_arguments)]
fn run_confirm(
    dex: &Dex,
    root: &Path,
    corpus: &str,
    input: &str,
    top: usize,
    candidate_range: (usize, usize),
    min_regret: f64,
    iters: u32,
    samples: usize,
    threads: usize,
    seed: u64,
    out_path: &str,
) {
    validate_confirmation_budget("confirmation", iters, samples);
    let files = corpus_files(&root.join(corpus));
    let corpus_fingerprint = corpus_source_id(&files);
    let text = std::fs::read_to_string(input).unwrap_or_else(|e| panic!("read {input}: {e}"));
    let parsed: Vec<(usize, serde_json::Value)> = text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .filter_map(|(line_index, line)| {
            let line_no = line_index + 1;
            let row: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| panic!("{input}:{line_no}: {e}"));
            assert!(row.is_object(), "{input}:{line_no}: row must be an object");
            if row["mode"] != "screen" || row.get("skip").is_some() {
                return None;
            }
            validate_screen_row(input, line_no, &row);
            assert_eq!(
                row["corpus_fingerprint"].as_str(),
                Some(corpus_fingerprint.as_str()),
                "{input}:{line_no}: screen row belongs to a different corpus"
            );
            Some((line_no, row))
        })
        .collect();
    let mut rows = dedupe_hypotheses(input, parsed, false);
    rows.retain(|row| {
        row["reference"] != row["candidate"] && row["regret"].as_f64().unwrap() >= min_regret
    });
    rows.sort_by(|a, b| {
        b["regret"]
            .as_f64()
            .unwrap()
            .total_cmp(&a["regret"].as_f64().unwrap())
            .then_with(|| cmp_offline_identity(a, b))
    });
    rows.truncate(top);
    let candidates: Vec<Candidate> = rows
        .into_iter()
        .enumerate()
        .filter(|(rank, _)| *rank >= candidate_range.0 && *rank <= candidate_range.1)
        .map(|(rank, row)| Candidate { rank, row })
        .collect();
    for candidate in &candidates {
        bind_offline_candidate(&files, candidate);
    }
    let src = load_sources(dex, root);
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    eprintln!(
        "confirm corpus {corpus_fingerprint} candidates {} iters {iters} samples {samples} \
         threads {threads}",
        candidates.len()
    );

    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let out = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= candidates.len() {
                    return;
                }
                let row = confirm_candidate(
                    dex,
                    &src,
                    &pool,
                    &files,
                    &candidates[j],
                    iters,
                    samples,
                    seed,
                );
                out.lock().unwrap().push(row);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(10) || n == candidates.len() {
                    eprintln!("  {n}/{} candidates", candidates.len());
                }
            });
        }
    });
    write_rows(out_path, out.into_inner().unwrap());
    eprintln!("done -> {out_path}");
}

#[allow(clippy::too_many_arguments)]
fn run_live_confirm(
    dex: &Dex,
    root: &Path,
    live_log: &str,
    input: &str,
    top: usize,
    candidate_range: (usize, usize),
    min_regret: f64,
    iters: u32,
    samples: usize,
    threads: usize,
    seed: u64,
    out_path: &str,
) {
    validate_confirmation_budget("live confirmation", iters, samples);
    let loaded_live_log = load_live_log(live_log);
    let source_id = loaded_live_log.source_id;
    let text = std::fs::read_to_string(input).unwrap_or_else(|e| panic!("read {input}: {e}"));
    let mut source_matches = 0usize;
    let mut source_mismatches = 0usize;
    let parsed: Vec<(usize, serde_json::Value)> = text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .filter_map(|(line_index, line)| {
            let line_no = line_index + 1;
            let row: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| panic!("{input}:{line_no}: {e}"));
            assert!(row.is_object(), "{input}:{line_no}: row must be an object");
            if row["mode"] != "live-screen" || row.get("skip").is_some() {
                return None;
            }
            validate_live_screen_row(input, line_no, &row);
            if row["input_fingerprint"].as_str() != Some(source_id.as_str()) {
                source_mismatches += 1;
                return None;
            }
            source_matches += 1;
            Some((line_no, row))
        })
        .collect();
    assert!(
        source_matches != 0 || source_mismatches == 0,
        "live-confirm: no live-screen rows match decision log fingerprint {source_id} \
         ({source_mismatches} rows belong to other logs)"
    );
    eprintln!(
        "live-confirm source {source_id}: {source_matches} matching rows, \
         {source_mismatches} rows from other logs ignored"
    );
    let mut rows = dedupe_hypotheses(input, parsed, true);
    rows.retain(|row| {
        row["reference"] != row["candidate"] && row["regret"].as_f64().unwrap() >= min_regret
    });
    rows.sort_by(|a, b| {
        b["regret"]
            .as_f64()
            .unwrap()
            .total_cmp(&a["regret"].as_f64().unwrap())
            .then_with(|| cmp_live_identity(a, b))
    });
    rows.truncate(top);
    let candidates: Vec<Candidate> = rows
        .into_iter()
        .enumerate()
        .filter(|(rank, _)| *rank >= candidate_range.0 && *rank <= candidate_range.1)
        .map(|(rank, row)| Candidate { rank, row })
        .collect();
    let wanted: HashSet<usize> = candidates
        .iter()
        .filter_map(|candidate| candidate.row["input_line"].as_u64().map(|n| n as usize))
        .collect();
    let snapshots = live_snapshots(loaded_live_log.rooms, &wanted);
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    eprintln!(
        "live-confirm candidates {} iters {iters} samples {samples} threads {threads}",
        candidates.len()
    );
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let out = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let index = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(candidate) = candidates.get(index) else {
                    return;
                };
                let input_line = candidate.row["input_line"].as_u64().map(|n| n as usize);
                let snapshot = input_line.and_then(|line| snapshots.get(&line));
                let row = confirm_live_candidate(
                    dex, &pool, candidate, snapshot, &source_id, iters, samples, seed,
                );
                out.lock().unwrap().push(row);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(10) || n == candidates.len() {
                    eprintln!("  {n}/{} candidates", candidates.len());
                }
            });
        }
    });
    write_rows(out_path, out.into_inner().unwrap());
    eprintln!("done -> {out_path}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = arg_s(&args, "--mode", "screen");
    let root = conformance::fixture::repo_root();
    let dex = conformance::load_dex();
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let threads = arg(
        &args,
        "--threads",
        std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(4),
    );
    let samples = arg(&args, "--samples", default_samples(&mode));
    let seed = arg_u64(&args, "--seed", 1);
    let out = arg_s(
        &args,
        "--out",
        match mode.as_str() {
            "screen" => "tmp/regret-screen.jsonl",
            "confirm" => "tmp/regret-confirm.jsonl",
            "live-screen" => "tmp/live-regret-screen.jsonl",
            "live-confirm" => "tmp/live-regret-confirm.jsonl",
            _ => "tmp/regret.jsonl",
        },
    );
    match mode.as_str() {
        "screen" => run_screen(
            &dex,
            &root,
            &corpus,
            range(&arg_s(&args, "--battles", "0-569")),
            range(&arg_s(&args, "--decisions", "0-999999")),
            arg_u32(&args, "--product-iters", 10_000),
            arg_u32(&args, "--oracle-iters", 30_000),
            samples,
            threads,
            arg(&args, "--per-battle", usize::MAX),
            seed,
            &out,
        ),
        "confirm" => run_confirm(
            &dex,
            &root,
            &corpus,
            &arg_s(&args, "--input", "tmp/regret-screen.jsonl"),
            arg(&args, "--top", 100),
            range(&arg_s(&args, "--candidates", "0-99")),
            arg_f(&args, "--min-regret", 0.0),
            arg_u32(&args, "--iters", 60_000),
            samples,
            threads,
            seed,
            &out,
        ),
        "live-screen" => run_live_screen(
            &dex,
            &root,
            &arg_s(&args, "--input", "tmp/decisions.jsonl"),
            arg_u32(&args, "--iters", 60_000),
            samples,
            threads,
            seed,
            &out,
        ),
        "live-confirm" => run_live_confirm(
            &dex,
            &root,
            &arg_s(&args, "--live-log", "tmp/decisions.jsonl"),
            &arg_s(&args, "--input", "tmp/live-regret-screen.jsonl"),
            arg(&args, "--top", 100),
            range(&arg_s(&args, "--candidates", "0-99")),
            arg_f(&args, "--min-regret", 0.0),
            arg_u32(&args, "--iters", 60_000),
            samples,
            threads,
            seed,
            &out,
        ),
        other => panic!("unknown --mode {other}; use screen|confirm|live-screen|live-confirm"),
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    fn v2_row(room: &str, rqid: u64, reset: bool, delta: &[&str]) -> serde_json::Value {
        serde_json::json!({
            "version":2, "type":"decision", "room":room,
            "battle":0, "decision":rqid, "rqid":rqid, "side":0, "turn":1,
            "server":"ws://127.0.0.1:8123/",
            "format":"gen2nintendocup2000noohkostadium2strict",
            "mode":"blind", "driver":"search", "iterations":10000, "seed":1000,
            "teamLabel":"pool:0",
            "ownTeam":[{"name":"Pikachu", "species":"Pikachu",
                        "moves":["Splash"], "level":50}],
            "request":{"rqid":rqid, "side":{"id":"p1", "pokemon":[]},
                       "active":[{"moves":[{"id":"splash"}]}]},
            "protocolReset":reset, "protocolDelta":delta,
            "submitted":"move splash", "rootPolicy":null,
            "stateViewKind":"diagnostic-imputed", "stateView":null,
        })
    }

    fn screen_row() -> serde_json::Value {
        serde_json::json!({
            "mode":"screen", "corpus_fingerprint":"fnv1a64:test:1files",
            "battle":0, "file":"battle.raw.log", "decision":1, "side":0, "turn":1,
            "human":"move splash", "reference":"move splash", "candidate":"move tackle",
            "reference_class":"Status", "candidate_class":"Physical", "regret":0.25,
            "tags":[], "actions":[], "product_iters":1, "oracle_iters":2, "samples":1,
        })
    }

    #[test]
    fn cli_numbers_and_ranges_fail_closed() {
        let overflow = vec!["x".into(), "--iters".into(), "4294967296".into()];
        assert!(std::panic::catch_unwind(|| arg_u32(&overflow, "--iters", 1)).is_err());
        let invalid = vec!["x".into(), "--samples".into(), "many".into()];
        assert!(std::panic::catch_unwind(|| arg(&invalid, "--samples", 3)).is_err());
        assert!(std::panic::catch_unwind(|| range("4")).is_err());
        assert!(std::panic::catch_unwind(|| range("5-4")).is_err());
    }

    #[test]
    fn budgets_reject_zero_and_odd_by_odd_confirmation() {
        assert!(std::panic::catch_unwind(|| validate_screen_budget(0, 30, 3)).is_err());
        assert!(std::panic::catch_unwind(|| validate_confirmation_budget("test", 3, 3)).is_err());
        validate_confirmation_budget("test", 3, 2);
        validate_confirmation_budget("test", 4, 3);
    }

    #[test]
    fn hypothesis_duplicates_are_exact_or_fatal() {
        let row = screen_row();
        let deduped =
            dedupe_hypotheses("synthetic", vec![(1, row.clone()), (2, row.clone())], false);
        assert_eq!(deduped, [row.clone()]);

        let mut conflict = row.clone();
        conflict["regret"] = serde_json::json!(0.5);
        assert!(std::panic::catch_unwind(|| {
            dedupe_hypotheses("synthetic", vec![(1, row), (2, conflict)], false)
        })
        .is_err());
    }

    #[test]
    fn offline_screen_schema_is_strict() {
        let row = screen_row();
        validate_screen_row("synthetic", 1, &row);
        let mut malformed = row;
        malformed["file"] = serde_json::Value::Null;
        assert!(
            std::panic::catch_unwind(|| { validate_screen_row("synthetic", 1, &malformed) })
                .is_err()
        );
    }

    #[test]
    fn corpus_fingerprint_binds_names_and_contents_not_location() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let left = std::env::temp_dir().join(format!("nc2000-regret-fp-{nonce}-a"));
        let right = std::env::temp_dir().join(format!("nc2000-regret-fp-{nonce}-b"));
        std::fs::create_dir_all(&left).unwrap();
        std::fs::create_dir_all(&right).unwrap();
        let left_file = left.join("battle.raw.log");
        let right_file = right.join("battle.raw.log");
        std::fs::write(&left_file, b"|turn|1\n").unwrap();
        std::fs::write(&right_file, b"|turn|1\n").unwrap();
        let original = corpus_source_id(std::slice::from_ref(&left_file));
        assert_eq!(
            original,
            corpus_source_id(std::slice::from_ref(&right_file))
        );
        std::fs::write(&right_file, b"|turn|2\n").unwrap();
        assert_ne!(
            original,
            corpus_source_id(std::slice::from_ref(&right_file))
        );
        std::fs::remove_dir_all(left).unwrap();
        std::fs::remove_dir_all(right).unwrap();
    }

    #[test]
    fn live_screen_uses_screen_sample_default() {
        assert_eq!(default_samples("screen"), 3);
        assert_eq!(default_samples("live-screen"), 3);
        assert_eq!(default_samples("confirm"), 8);
        assert_eq!(default_samples("live-confirm"), 8);
    }

    #[test]
    fn v2_schema_matches_logger_and_rejects_unknown_fields() {
        let row = v2_row("battle-test-1", 1, false, &["|turn|1"]);
        let parsed: LiveLogRow = serde_json::from_value(row.clone()).unwrap();
        validate_live_row("synthetic", 1, &parsed);

        let mut extra = row;
        extra["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<LiveLogRow>(extra).is_err());
    }

    #[test]
    fn parser_deduplicates_before_reset_accumulation() {
        let first = v2_row("battle-test-1", 1, false, &["|turn|1"]);
        let rows = [
            first.clone(),
            first,
            v2_row("battle-test-1", 2, true, &["|turn|2"]),
        ];
        let text = rows
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        let rooms = parse_live_rooms("synthetic", &text);
        assert_eq!(rooms.len(), 1);
        assert_eq!(rooms[0].rows.len(), 2);
        let mut protocol = Vec::new();
        for row in &rooms[0].rows {
            apply_protocol_delta(&mut protocol, row.protocol_reset, &row.protocol_delta);
        }
        assert_eq!(protocol, ["|turn|2"]);
    }

    #[test]
    fn parser_rejects_conflicting_room_rqid_duplicates() {
        let rows = [
            v2_row("battle-test-1", 1, false, &["|turn|1"]),
            v2_row("battle-test-1", 1, true, &["|turn|2"]),
        ];
        let text = rows
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(std::panic::catch_unwind(|| parse_live_rooms("synthetic", &text)).is_err());
    }

    #[test]
    fn submitted_choices_are_normalized_to_semantic_labels() {
        let request = serde_json::json!({
            "active":[{"moves":[
                {"id":"hiddenpowerice"}, {"id":"bellydrum"}
            ]}],
            "side":{"pokemon":[
                {"details":"Snorlax, L50, M"}, {"details":"Starmie, L55"}
            ]}
        });
        assert_eq!(
            submitted_label(&request, "move 1").unwrap(),
            "move hiddenpower"
        );
        assert_eq!(
            submitted_label(&request, "move bellydrum").unwrap(),
            "move bellydrum"
        );
        assert_eq!(
            submitted_label(&request, "switch 2").unwrap(),
            "switch starmie"
        );
    }

    #[test]
    fn protocol_reset_replaces_prior_accumulation() {
        let mut protocol = vec!["|turn|1".to_string()];
        apply_protocol_delta(&mut protocol, false, &["|move|p1a: A|Rest".to_string()]);
        assert_eq!(protocol.len(), 2);
        apply_protocol_delta(&mut protocol, true, &["|turn|2".to_string()]);
        assert_eq!(protocol, ["|turn|2"]);
    }
}
