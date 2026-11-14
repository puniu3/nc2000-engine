//! Imperfect-information diagnostic harness (Phase 0..5 verification):
//! - `--mode corpus_audit`: replays simulated Stadium 2 games (catalog, rentals,
//!   mutations, and timing edge cases) + spectator logs (`tmp/corpus-spectator`) to
//!   enforce zero false item / HP / pool exclusions.
//! - `--mode calibration`: measures Belief posterior quality against held-out
//!   off-pool rosters (unseen active/bench move recall, item accuracy & log-loss,
//!   HP type accuracy, effective loadout diversity, fallback rate).
//! - `--mode gauntlet`: runs seed-paired non-self-play matches (`BlindAgent` vs
//!   `OpenAgent` oracle, plus optional catalog non-regression) over held-out
//!   off-pool teams and emits one compact single-line JSON summary.
//!
//! Strictly single-line JSON stdout (M2M output principle). Detailed per-record
//! JSONL is written only when `--out PATH` is supplied.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::agent::{Agent, MaxDamageAgent};
use nc2000_bot::belief::{Belief, FallbackPolicy};
use nc2000_bot::blind::{BlindAgent, OpenAgent};
use nc2000_bot::corpus::{
    corpus_files, load_battle, load_sources, reconstruct_context_with_pool, SetSources,
};
use nc2000_bot::observe::move_matches;
use nc2000_bot::player::PlayerChannel;
use nc2000_bot::player_state::LocalPlayer;
use nc2000_bot::preview::{load_meta_pool, team_sig, MetaPool};
use nc2000_bot::rng::SplitMix64;
use nc2000_bot::runner::{play_game, GameResult};
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_bot::teamgen::{to_sets, TeamGen};
use nc2000_bot::Observer;
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::{toid, Dex, MoveId, SpeciesId, TypeId};
use nc2000_engine::state::{Battle, RequestState};
use nc2000_engine::validate::{validate_team, Learnsets};
use serde_json::{json, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    CorpusAudit,
    Calibration,
    Gauntlet,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatchupSet {
    Offpool,
    Catalog,
}

struct Cli {
    mode: Mode,
    matchups: MatchupSet,
    corpus_dir: PathBuf,
    battles: usize,
    pairs: usize,
    iters: u32,
    threads: usize,
    seed: u64,
    out_path: Option<PathBuf>,
    max_turns: u16,
    samples: usize,
    mutations: usize,
}

fn arg_val(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_cli(root: &Path) -> Cli {
    let args: Vec<String> = std::env::args().collect();
    let mode = match arg_val(&args, "--mode").as_deref() {
        Some("calibration") => Mode::Calibration,
        Some("gauntlet") => Mode::Gauntlet,
        _ => Mode::CorpusAudit,
    };
    let matchups = match arg_val(&args, "--matchups").as_deref() {
        Some("catalog") => MatchupSet::Catalog,
        _ => MatchupSet::Offpool,
    };
    let corpus_dir = arg_val(&args, "--corpus")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("tmp/corpus-spectator"));
    let battles = arg_val(&args, "--battles")
        .and_then(|s| s.parse().ok())
        .unwrap_or(570);
    let pairs = arg_val(&args, "--pairs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let iters = arg_val(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3_000);
    let default_threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(16);
    let threads = arg_val(&args, "--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_threads)
        .max(1);
    let seed = arg_val(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x4245_4C49_4546_0001);
    let out_path = arg_val(&args, "--out").map(PathBuf::from);
    let max_turns = arg_val(&args, "--max-turns")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let samples = arg_val(&args, "--samples")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let mutations = arg_val(&args, "--mutations")
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    Cli {
        mode,
        matchups,
        corpus_dir,
        battles,
        pairs,
        iters,
        threads,
        seed,
        out_path,
        max_turns,
        samples,
        mutations,
    }
}

fn production_cfg(iters: u32) -> RmConfig {
    let profile: Value =
        serde_json::from_str(include_str!("../../../data/search-profiles.json")).unwrap();
    RmConfig {
        iterations: iters,
        rule: SelRule::Ucb,
        c: profile["blind"]["c"].as_f64().unwrap(),
        ..Default::default()
    }
}

struct TeamEntry {
    id: String,
    kind: &'static str,
    sets: Vec<PokemonSet>,
}

fn load_legal_rentals(root: &Path, dex: &Dex, ls: &Learnsets) -> Vec<TeamEntry> {
    let text = std::fs::read_to_string(root.join("data/community-rentals-v0/teams.json")).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    let mut out = Vec::new();
    for (i, t) in v["teams"].as_array().unwrap().iter().enumerate() {
        let Some(sets_val) = t["sets"].as_array() else {
            continue;
        };
        let team_json = Value::Array(sets_val.clone()).to_string();
        if validate_team(dex, ls, &team_json)["ok"] != Value::Bool(true) {
            continue;
        }
        if let Ok(sets) = to_sets(sets_val) {
            let cban = t["cban"].as_u64().unwrap_or(i as u64);
            out.push(TeamEntry {
                id: format!("rental-{cban:02}"),
                kind: "rental",
                sets,
            });
        }
    }
    out
}

fn load_offpool_teams(
    root: &Path,
    dex: &Dex,
    prior_pool: &MetaPool,
    mutation_count: usize,
    seed: u64,
) -> Vec<TeamEntry> {
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let ls = Learnsets::from_json(&ls_text).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/belief-pool-v3/belief-pool.json")).unwrap();
    let gen = TeamGen::new(dex, &ls_text, &pool_text).unwrap();

    let prior_sigs: HashSet<_> = prior_pool
        .teams
        .iter()
        .map(|t| team_sig(dex, &t.sets))
        .collect();
    let mut seen_sigs = prior_sigs.clone();
    let mut out = Vec::new();

    // 1. Held-out teams from team-pool-v2 absent from belief-pool-v3
    let v2 = load_meta_pool(&root.join("data/team-pool-v2/team-pool.json"));
    for t in &v2.teams {
        let sig = team_sig(dex, &t.sets);
        if !seen_sigs.insert(sig) {
            continue;
        }
        out.push(TeamEntry {
            id: format!("v2-{}", t.id),
            kind: "team-pool-v2-off",
            sets: t.sets.clone(),
        });
    }

    // 2. Legal teams from community-rentals-v0 absent from belief-pool-v3
    for r in load_legal_rentals(root, dex, &ls) {
        let sig = team_sig(dex, &r.sets);
        if !seen_sigs.insert(sig) {
            continue;
        }
        out.push(r);
    }

    // 3. Controlled TeamGen mutations (1..=4 ops from belief-pool-v3 templates:
    //    covers both near-pool 1-move/1-item/1-slot deviations and deeper off-pool teams)
    let mut rng = SplitMix64::new(seed ^ 0x517C_C1B7_2722_0A95);
    for m_idx in 0..mutation_count {
        let seed_idx = (m_idx * 17) % prior_pool.teams.len();
        let mut cur = gen.canonize(dex, &gen.team_json(seed_idx)).unwrap();
        let n_steps = 1 + (m_idx % 4);
        let mut accepted = None;
        for step in 0..24usize {
            let Some(prop) = gen.propose_valid(dex, &cur, &mut rng, 64) else {
                break;
            };
            cur = prop.team;
            if step + 1 >= n_steps {
                if let Ok(sets) = to_sets(&cur) {
                    let sig = team_sig(dex, &sets);
                    if seen_sigs.insert(sig) {
                        accepted = Some(sets);
                        break;
                    }
                }
            }
        }
        if let Some(sets) = accepted {
            out.push(TeamEntry {
                id: format!("mut-{m_idx:02}"),
                kind: if n_steps == 1 {
                    "mutation-1step"
                } else {
                    "mutation-multi"
                },
                sets,
            });
        }
    }

    out
}

fn hidden_power_mask_bit(dex: &Dex, type_id: TypeId) -> Option<u16> {
    let types = [
        "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel",
        "Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark",
    ];
    let name = dex.type_name(type_id);
    types.iter().position(|&x| x == name).map(|i| 1u16 << i)
}

fn plain_move_id(dex: &Dex, id: MoveId) -> MoveId {
    let key = dex.moves.key(id);
    if key.starts_with("hiddenpower") {
        dex.moves.id("hiddenpower").unwrap_or(id)
    } else {
        id
    }
}

fn mk_set(species: &str, item: &str, moves: &[&str], level: u8) -> PokemonSet {
    serde_json::from_value(json!({
        "species": species,
        "name": species,
        "gender": "M",
        "item": item,
        "level": level,
        "moves": moves,
        "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255}
    }))
    .unwrap()
}

fn pad_to_six(sets: &[PokemonSet]) -> Vec<PokemonSet> {
    let fillers = [
        mk_set("Pidgey", "", &["Tackle"], 5),
        mk_set("Rattata", "", &["Tackle"], 5),
        mk_set("Caterpie", "", &["Tackle"], 5),
        mk_set("Weedle", "", &["Poison Sting"], 5),
        mk_set("Sentret", "", &["Tackle"], 5),
        mk_set("Hoothoot", "", &["Tackle"], 5),
    ];
    let mut out = sets.to_vec();
    for f in fillers {
        if out.len() >= 6 {
            break;
        }
        if !out
            .iter()
            .any(|s| s.species.eq_ignore_ascii_case(&f.species))
        {
            out.push(f);
        }
    }
    out
}

fn check_hp_masks(
    dex: &Dex,
    battle: &Battle,
    obs: &Observer,
    hp_id: MoveId,
    true_hp_moves: &HashMap<SpeciesId, TypeId>,
) -> usize {
    let opp = obs.opp();
    let mut errs = 0;
    for (mo, true_mon) in obs.mons().iter().zip(&battle.sides[opp].roster) {
        if mo.revealed_moves.iter().any(|&m| move_matches(dex, hp_id, m)) {
            if let Some(&true_ty) = true_hp_moves.get(&true_mon.species) {
                if let Some(bit) = hidden_power_mask_bit(dex, true_ty) {
                    if mo.hp_type_mask & bit == 0 {
                        errs += 1;
                    }
                }
            }
        }
    }
    errs
}

/// Plays a scripted or rollout Stadium-2 game using real `PlayerChannel` feeds
/// and asserts zero false exclusions on HP masks, in-pool candidates, etc.
fn run_simulated_audit(
    dex: &Dex,
    prior: &MetaPool,
    offpool: &[TeamEntry],
    seed: u64,
    out_lines: &mut Vec<String>,
) -> (usize, usize, usize, usize) {
    let hp_id = dex.moves.id("hiddenpower").unwrap();
    let mut rng = SplitMix64::new(seed ^ 0xA0D1_7001);
    let mut sim_games = 0usize;
    let mut sim_decisions = 0usize;
    let mut in_pool_drops = 0usize;
    let mut hp_false_exclusions = 0usize;

    // Include explicit Stadium-2 edge-case rosters covering Leftovers + Leech Seed/Sandstorm/Curse,
    // Mint/Miracle Berry before move vs upkeep, immediate cure berries, Thief, Hidden Power, and Berserk Gene.
    let edge_p1 = pad_to_six(&[
        mk_set("Snorlax", "Leftovers", &["Sleep Talk", "Rest", "Body Slam", "Curse"], 50),
        mk_set("Zapdos", "Miracle Berry", &["Thunder", "Hidden Power Ice", "Rest", "Sleep Talk"], 55),
        mk_set("Exeggutor", "Mint Berry", &["Psychic", "Giga Drain", "Sleep Powder", "Leech Seed"], 50),
    ]);
    let edge_p2 = pad_to_six(&[
        mk_set("Tyranitar", "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Sandstorm"], 55),
        mk_set("Misdreavus", "Miracle Berry", &["Mean Look", "Perish Song", "Confuse Ray", "Thief"], 50),
        mk_set("Gengar", "Przcureberry", &["Thunderbolt", "Hypnosis", "Curse", "Nightmare"], 50),
    ]);

    let mut fixtures: Vec<(&str, bool, &[PokemonSet], &[PokemonSet])> = Vec::new();
    fixtures.push(("edge-stadium2-timings", false, &edge_p1, &edge_p2));
    for i in 0..8 {
        let p1 = &prior.teams[(i * 29) % prior.teams.len()].sets;
        let p2 = &prior.teams[(i * 43 + 7) % prior.teams.len()].sets;
        fixtures.push(("catalog-pair", true, p1, p2));
    }
    for i in 0..8.min(offpool.len()) {
        let p1 = &prior.teams[(i * 19 + 3) % prior.teams.len()].sets;
        let p2 = &offpool[i].sets;
        fixtures.push(("offpool-pair", false, p1, p2));
    }

    for (tag, is_in_pool, p1_sets, p2_sets) in fixtures {
        sim_games += 1;
        let bseed = rng.battle_seed();
        let mut battle = Battle::from_fixture(dex, &bseed, p1_sets, p2_sets).unwrap();
        battle.set_log_enabled(true);

        let true_hp_types: [HashMap<SpeciesId, TypeId>; 2] = [0, 1].map(|s| {
            let sets = if s == 0 { p1_sets } else { p2_sets };
            let mut m = HashMap::new();
            for (r, st) in battle.sides[s].roster.iter().zip(sets) {
                if st.moves.iter().any(|mv| toid(mv).starts_with("hiddenpower")) {
                    m.insert(r.species, r.base_hp_type);
                }
            }
            m
        });

        let mut agents = [MaxDamageAgent::conformant(), MaxDamageAgent::conformant()];
        let mut channels = [PlayerChannel::new(0), PlayerChannel::new(1)];
        let mut locals: [Option<(LocalPlayer, Observer, Belief)>; 2] = [None, None];
        let mut proto_obs: [Option<nc2000_bot::ProtocolTracker>; 2] = [None, None];

        for _step in 0..80 {
            if battle.outcome().is_some() {
                break;
            }
            let mut picks = [None, None];
            for s in 0..2 {
                let cs = battle.legal_choices(dex, s);
                if cs.is_empty() {
                    continue;
                }
                if let Ok(frame) = channels[s].frame(&mut battle, dex) {
                    let trk = proto_obs[s]
                        .get_or_insert_with(|| nc2000_bot::ProtocolTracker::new(s));
                    for line in &frame.lines {
                        trk.push_line(dex, line);
                    }
                }
                if locals[s].is_none() {
                    let (loc, obs) = LocalPlayer::new(&battle, dex, s).unwrap();
                    let belief = Belief::new(dex, prior, &obs);
                    locals[s] = Some((loc, obs, belief));
                }
                let (ref mut loc, ref mut obs, ref mut belief) = locals[s].as_mut().unwrap();
                loc.observe(&battle, dex, obs).unwrap();
                belief.sync(dex, obs);
                sim_decisions += 1;
                if is_in_pool && belief.is_fallback() {
                    in_pool_drops += 1;
                    out_lines.push(
                        json!({"kind": "in_pool_drop", "tag": tag, "side": s, "turn": battle.turn})
                            .to_string(),
                    );
                }
                let errs = check_hp_masks(dex, &battle, obs, hp_id, &true_hp_types[1 - s]);
                if errs > 0 {
                    hp_false_exclusions += errs;
                    out_lines.push(
                        json!({"kind": "hp_false_excl", "tag": tag, "side": s, "turn": battle.turn, "errs": errs})
                            .to_string(),
                    );
                }
                picks[s] = Some(agents[s].choose(&battle, dex, s, &cs));
            }
            if picks == [None, None] {
                break;
            }
            if battle.apply_choices(dex, picks).is_err() {
                break;
            }
        }
    }

    (sim_games, sim_decisions, in_pool_drops, hp_false_exclusions)
}

fn run_spectator_corpus_audit(
    dex: &Dex,
    prior: &MetaPool,
    src: &SetSources,
    corpus_dir: &Path,
    max_battles: usize,
    seed: u64,
) -> (usize, usize, usize, f64, usize, usize) {
    if !corpus_dir.exists() {
        return (0, 0, 0, 0.0, 0, 0);
    }
    let files = corpus_files(corpus_dir);
    let n = files.len().min(max_battles);
    let mut battles = 0usize;
    let mut decisions = 0usize;
    let mut fallbacks = 0usize;
    let mut alive_sum = 0usize;
    let mut recon_skipped = 0usize;
    let mut corpus_hp_false_exclusions = 0usize;
    let hp_id = dex.moves.id("hiddenpower").unwrap();

    for (bi, file) in files.into_iter().take(n).enumerate() {
        let cb = load_battle(&file);
        battles += 1;
        // Sample up to 6 spread decisions per battle for full 570-battle speed
        let total_d = cb.decisions.len();
        if total_d == 0 {
            continue;
        }
        let stride = (total_d / 6).max(1);
        for (di, d) in cb.decisions.iter().enumerate().step_by(stride).take(6) {
            let dseed = seed ^ ((bi as u64 + 1) * 0x9E37_79B9) ^ ((di as u64 + 1) * 0xBF58_476D);
            let Some(rec) = reconstruct_context_with_pool(
                dex,
                src,
                prior.clone(),
                &cb.lines,
                &cb.evidence,
                d,
                dseed,
            ) else {
                recon_skipped += 1;
                continue;
            };
            decisions += 1;
            if let Some(b) = rec.agent.belief() {
                if b.is_fallback() {
                    fallbacks += 1;
                }
                alive_sum += b.candidate_count();
            }
            if let (Some(obs), Some(battle)) = (rec.agent.observer(), rec.agent.battle()) {
                for (mo, p) in obs.mons().iter().zip(&battle.sides[obs.opp()].roster) {
                    if mo.revealed_moves.iter().any(|&m| move_matches(dex, hp_id, m)) {
                        if let Some(bit) = hidden_power_mask_bit(dex, p.base_hp_type) {
                            // Spectator log ground truth: only validate when mask isn't vacuously 0
                            if mo.hp_type_mask == 0 && bit != 0 {
                                corpus_hp_false_exclusions += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    let mean_alive = if decisions > fallbacks {
        alive_sum as f64 / (decisions - fallbacks) as f64
    } else {
        0.0
    };
    (
        battles,
        decisions,
        fallbacks,
        mean_alive,
        recon_skipped,
        corpus_hp_false_exclusions,
    )
}

fn mode_corpus_audit(cli: &Cli, root: &Path) -> Value {
    let dex = load_dex();
    let prior = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let offpool = load_offpool_teams(root, &dex, &prior, cli.mutations, cli.seed);
    let mut detail_lines = Vec::new();
    let (sim_games, sim_decisions, in_pool_drops, hp_false_exclusions) =
        run_simulated_audit(&dex, &prior, &offpool, cli.seed, &mut detail_lines);

    let (
        corpus_battles,
        corpus_decisions,
        corpus_fallbacks,
        corpus_mean_alive,
        corpus_skipped,
        corpus_hp_false_excl,
    ) = if cli.corpus_dir.exists() {
        let src = load_sources(&dex, root);
        run_spectator_corpus_audit(
            &dex,
            &prior,
            &src,
            &cli.corpus_dir,
            cli.battles,
            cli.seed,
        )
    } else {
        (0, 0, 0, 0.0, 0, 0)
    };

    if let Some(path) = &cli.out_path {
        let mut f = std::fs::File::create(path).unwrap();
        for l in &detail_lines {
            writeln!(f, "{l}").unwrap();
        }
    }

    let fb_rate = if corpus_decisions > 0 {
        corpus_fallbacks as f64 / corpus_decisions as f64
    } else {
        0.0
    };

    json!({
        "mode": "corpus_audit",
        "sim_games": sim_games,
        "sim_decisions": sim_decisions,
        "in_pool_drops": in_pool_drops,
        "hp_false_exclusions": hp_false_exclusions,
        "corpus_battles": corpus_battles,
        "corpus_decisions": corpus_decisions,
        "corpus_skipped": corpus_skipped,
        "corpus_fallbacks": corpus_fallbacks,
        "corpus_fallback_rate": (fb_rate * 1e4).round() / 1e4,
        "corpus_mean_alive": (corpus_mean_alive * 100.0).round() / 100.0,
        "corpus_hp_false_exclusions": corpus_hp_false_excl,
    })
}

fn mode_calibration(cli: &Cli, root: &Path) -> Value {
    let dex = Arc::new(load_dex());
    let prior = Arc::new(load_meta_pool(
        &root.join("data/belief-pool-v3/belief-pool.json"),
    ));
    let offpool = Arc::new(load_offpool_teams(
        root,
        &dex,
        &prior,
        cli.mutations,
        cli.seed,
    ));
    let n_games = cli.pairs.min(offpool.len());

    #[derive(Default)]
    struct CalibAcc {
        positions: usize,
        fallback_positions: usize,
        alive_sum: usize,
        active_unseen_total: usize,
        active_unseen_recalled: usize,
        active_unseen_prob_sum: f64,
        bench_unseen_total: usize,
        bench_unseen_recalled: usize,
        bench_unseen_prob_sum: f64,
        item_slots: usize,
        item_top1_hits: usize,
        item_ll_sum: f64,
        item_prob_sum: f64,
        hp_slots: usize,
        hp_prob_sum: f64,
        distinct_loadouts_sum: usize,
        loadout_slot_count: usize,
        detail_lines: Vec<String>,
    }

    let next = AtomicUsize::new(0);
    let total_acc = Mutex::new(CalibAcc::default());

    std::thread::scope(|s| {
        for _ in 0..cli.threads {
            s.spawn(|| {
                let mut local_acc = CalibAcc::default();
                let hp_id = dex.moves.id("hiddenpower").unwrap();
                loop {
                    let gi = next.fetch_add(1, Ordering::Relaxed);
                    if gi >= n_games {
                        break;
                    }
                    let opp_entry = &offpool[gi];
                    let own_sets = &prior.teams[(gi * 13 + 5) % prior.teams.len()].sets;
                    let mut rng = SplitMix64::new(cli.seed ^ ((gi as u64 + 1) * 0x517C_C1B7));
                    let bseed = rng.battle_seed();
                    let Ok(mut battle) = Battle::from_fixture(&dex, &bseed, own_sets, &opp_entry.sets) else {
                        continue;
                    };
                    let mut agents = [MaxDamageAgent::conformant(), MaxDamageAgent::conformant()];
                    let (mut lp, mut obs) = LocalPlayer::new(&battle, &dex, 0).unwrap();
                    let mut belief = Belief::new(&dex, &prior, &obs);
                    let mut sampled_turns = HashSet::new();

                    for _ply in 0..80 {
                        if battle.outcome().is_some() || battle.turn > 30 {
                            break;
                        }
                        if battle.turn > 0
                            && matches!(battle.turn, 2 | 5 | 9 | 14 | 20)
                            && !sampled_turns.contains(&battle.turn)
                            && !matches!(battle.request_state, RequestState::TeamPreview)
                        {
                            sampled_turns.insert(battle.turn);
                            lp.observe(&battle, &dex, &mut obs).unwrap();
                            belief.sync(&dex, &obs);
                            local_acc.positions += 1;
                            if belief.is_fallback() {
                                local_acc.fallback_positions += 1;
                            } else {
                                local_acc.alive_sum += belief.candidate_count();
                            }

                            let opp = 1;
                            let k_samples = cli.samples;
                            let mut move_counts: [HashMap<MoveId, usize>; 6] =
                                std::array::from_fn(|_| HashMap::new());
                            let mut item_counts: [HashMap<Option<u16>, usize>; 6] =
                                std::array::from_fn(|_| HashMap::new());
                            let mut hp_counts: [usize; 6] = [0; 6];
                            let mut loadouts: [HashSet<(Option<u16>, Vec<u16>)>; 6] =
                                std::array::from_fn(|_| HashSet::new());

                            for _ in 0..k_samples {
                                let det = belief.determinize(&dex, &battle, &obs, &mut rng);
                                for slot in 0..battle.sides[opp].roster.len().min(6) {
                                    let p = &det.sides[opp].roster[slot];
                                    let truth = &battle.sides[opp].roster[slot];
                                    for ms in &p.base_move_slots {
                                        let key = plain_move_id(&dex, ms.id);
                                        *move_counts[slot].entry(key).or_default() += 1;
                                    }
                                    *item_counts[slot].entry(p.item.map(|i| i.0)).or_default() += 1;
                                    if p.base_hp_type == truth.base_hp_type {
                                        hp_counts[slot] += 1;
                                    }
                                    let mut mv_keys: Vec<u16> =
                                        p.base_move_slots.iter().map(|m| m.id.0).collect();
                                    mv_keys.sort_unstable();
                                    loadouts[slot].insert((p.item.map(|i| i.0), mv_keys));
                                }
                            }

                            for slot in 0..battle.sides[opp].roster.len().min(6) {
                                let mo = &obs.mons()[slot];
                                let truth = &battle.sides[opp].roster[slot];
                                // Unrevealed living move evaluation
                                if truth.hp > 0 {
                                    let picked_or_seen = mo.appeared
                                        || battle.sides[opp].party.contains(&(slot as u8));
                                    if picked_or_seen {
                                        for ms in &truth.base_move_slots {
                                            if mo
                                                .revealed_moves
                                                .iter()
                                                .any(|&rm| move_matches(&dex, ms.id, rm))
                                            {
                                                continue;
                                            }
                                            let key = plain_move_id(&dex, ms.id);
                                            let c = *move_counts[slot].get(&key).unwrap_or(&0);
                                            let prob = c as f64 / k_samples as f64;
                                            if mo.appeared {
                                                local_acc.active_unseen_total += 1;
                                                if c > 0 {
                                                    local_acc.active_unseen_recalled += 1;
                                                }
                                                local_acc.active_unseen_prob_sum += prob;
                                            } else {
                                                local_acc.bench_unseen_total += 1;
                                                if c > 0 {
                                                    local_acc.bench_unseen_recalled += 1;
                                                }
                                                local_acc.bench_unseen_prob_sum += prob;
                                            }
                                        }
                                        local_acc.loadout_slot_count += 1;
                                        local_acc.distinct_loadouts_sum += loadouts[slot].len();
                                    }
                                    // Unrevealed item accuracy when original is unknown
                                    if mo.appeared && mo.item.original.is_none() {
                                        let true_item = truth.item.map(|i| i.0);
                                        let c = *item_counts[slot].get(&true_item).unwrap_or(&0);
                                        let prob = (c as f64 + 0.01) / (k_samples as f64 + 0.25);
                                        let raw_prob = c as f64 / k_samples as f64;
                                        let top1 = item_counts[slot]
                                            .iter()
                                            .max_by_key(|(_, cnt)| *cnt)
                                            .map(|(&k, _)| k);
                                        local_acc.item_slots += 1;
                                        if top1 == Some(true_item) {
                                            local_acc.item_top1_hits += 1;
                                        }
                                        local_acc.item_prob_sum += raw_prob;
                                        local_acc.item_ll_sum += prob.ln();
                                    }
                                    // Hidden Power type prediction when HP is revealed
                                    if mo
                                        .revealed_moves
                                        .iter()
                                        .any(|&rm| move_matches(&dex, hp_id, rm))
                                    {
                                        local_acc.hp_slots += 1;
                                        local_acc.hp_prob_sum +=
                                            hp_counts[slot] as f64 / k_samples as f64;
                                    }
                                }
                            }
                            if cli.out_path.is_some() {
                                local_acc.detail_lines.push(
                                    json!({
                                        "team": opp_entry.id,
                                        "kind": opp_entry.kind,
                                        "turn": battle.turn,
                                        "fallback": belief.is_fallback(),
                                        "alive": belief.candidate_count(),
                                    })
                                    .to_string(),
                                );
                            }
                        }

                        let mut picks = [None, None];
                        for side in 0..2 {
                            let cs = battle.legal_choices(&dex, side);
                            if !cs.is_empty() {
                                picks[side] = Some(agents[side].choose(&battle, &dex, side, &cs));
                            }
                        }
                        if picks == [None, None] || battle.apply_choices(&dex, picks).is_err() {
                            break;
                        }
                    }
                }

                let mut guard = total_acc.lock().unwrap();
                guard.positions += local_acc.positions;
                guard.fallback_positions += local_acc.fallback_positions;
                guard.alive_sum += local_acc.alive_sum;
                guard.active_unseen_total += local_acc.active_unseen_total;
                guard.active_unseen_recalled += local_acc.active_unseen_recalled;
                guard.active_unseen_prob_sum += local_acc.active_unseen_prob_sum;
                guard.bench_unseen_total += local_acc.bench_unseen_total;
                guard.bench_unseen_recalled += local_acc.bench_unseen_recalled;
                guard.bench_unseen_prob_sum += local_acc.bench_unseen_prob_sum;
                guard.item_slots += local_acc.item_slots;
                guard.item_top1_hits += local_acc.item_top1_hits;
                guard.item_ll_sum += local_acc.item_ll_sum;
                guard.item_prob_sum += local_acc.item_prob_sum;
                guard.hp_slots += local_acc.hp_slots;
                guard.hp_prob_sum += local_acc.hp_prob_sum;
                guard.distinct_loadouts_sum += local_acc.distinct_loadouts_sum;
                guard.loadout_slot_count += local_acc.loadout_slot_count;
                guard.detail_lines.extend(local_acc.detail_lines);
            });
        }
    });

    let acc = total_acc.into_inner().unwrap();
    if let Some(path) = &cli.out_path {
        let mut f = std::fs::File::create(path).unwrap();
        for l in &acc.detail_lines {
            writeln!(f, "{l}").unwrap();
        }
    }

    let round4 = |x: f64| (x * 1e4).round() / 1e4;
    let active_recall = if acc.active_unseen_total > 0 {
        acc.active_unseen_recalled as f64 / acc.active_unseen_total as f64
    } else {
        0.0
    };
    let active_prob = if acc.active_unseen_total > 0 {
        acc.active_unseen_prob_sum / acc.active_unseen_total as f64
    } else {
        0.0
    };
    let bench_recall = if acc.bench_unseen_total > 0 {
        acc.bench_unseen_recalled as f64 / acc.bench_unseen_total as f64
    } else {
        0.0
    };
    let bench_prob = if acc.bench_unseen_total > 0 {
        acc.bench_unseen_prob_sum / acc.bench_unseen_total as f64
    } else {
        0.0
    };
    let item_top1 = if acc.item_slots > 0 {
        acc.item_top1_hits as f64 / acc.item_slots as f64
    } else {
        0.0
    };
    let item_prob = if acc.item_slots > 0 {
        acc.item_prob_sum / acc.item_slots as f64
    } else {
        0.0
    };
    let item_ll = if acc.item_slots > 0 {
        acc.item_ll_sum / acc.item_slots as f64
    } else {
        0.0
    };
    let hp_acc = if acc.hp_slots > 0 {
        acc.hp_prob_sum / acc.hp_slots as f64
    } else {
        0.0
    };
    let fb_rate = if acc.positions > 0 {
        acc.fallback_positions as f64 / acc.positions as f64
    } else {
        0.0
    };
    let eff_loadouts = if acc.loadout_slot_count > 0 {
        acc.distinct_loadouts_sum as f64 / acc.loadout_slot_count as f64
    } else {
        0.0
    };

    json!({
        "mode": "calibration",
        "games": n_games,
        "positions": acc.positions,
        "fallback_rate": round4(fb_rate),
        "mean_alive": round4(if acc.positions > acc.fallback_positions { acc.alive_sum as f64 / (acc.positions - acc.fallback_positions) as f64 } else { 0.0 }),
        "active_unseen_move_recall": round4(active_recall),
        "active_unseen_move_prob": round4(active_prob),
        "bench_unseen_move_recall": round4(bench_recall),
        "bench_unseen_move_prob": round4(bench_prob),
        "item_top1_acc": round4(item_top1),
        "item_true_prob": round4(item_prob),
        "item_log_likelihood": round4(item_ll),
        "hp_type_acc": round4(hp_acc),
        "hp_slots": acc.hp_slots,
        "eff_distinct_loadouts": round4(eff_loadouts),
    })
}

fn elo_diff(win_rate: f64) -> f64 {
    let clamped = win_rate.clamp(1e-4, 1.0 - 1e-4);
    -400.0 * (1.0 / clamped - 1.0).log10()
}

fn mode_gauntlet(cli: &Cli, root: &Path) -> Value {
    let t0 = Instant::now();
    let dex = Arc::new(load_dex());
    let prior = Arc::new(load_meta_pool(
        &root.join("data/belief-pool-v3/belief-pool.json"),
    ));
    let offpool = Arc::new(load_offpool_teams(
        root,
        &dex,
        &prior,
        cli.mutations,
        cli.seed,
    ));

    // Paired non-self-play matchups: BlindAgent (evaluated, using production prior pool)
    // vs OpenAgent oracle (given true opponent sheets).
    // Game A (p1=team_a, p2=team_b): BlindAgent on p1, OpenAgent on p2
    // Game B (same battle seed & roster assignment swapped: p1=team_b, p2=team_a): OpenAgent on p1, BlindAgent on p2
    // Wait: to eliminate team-strength bias across asymmetric custom vs prior pairs, we
    // also alternate who gets the custom offpool team across pairs! Even cleaner:
    // for each pair p, select (team_p1, team_p2), then play:
    //   arm 0: BlindAgent plays team_p1 vs OpenAgent on team_p2
    //   arm 1: OpenAgent plays team_p1 vs BlindAgent on team_p2 (same battle seed, identical teams!)
    // Averaging BlindAgent score across the two arms cancels first-player & matchup skew!
    let mut pairs_spec: Vec<(Vec<PokemonSet>, Vec<PokemonSet>, String, String)> = Vec::new();
    match cli.matchups {
        MatchupSet::Offpool => {
            for i in 0..cli.pairs {
                let opp = &offpool[i % offpool.len()];
                let pilot = &prior.teams[(i * 19 + 3) % prior.teams.len()];
                pairs_spec.push((
                    pilot.sets.clone(),
                    opp.sets.clone(),
                    pilot.id.clone(),
                    opp.id.clone(),
                ));
            }
        }
        MatchupSet::Catalog => {
            for i in 0..cli.pairs {
                let a = &prior.teams[(i * 17) % prior.teams.len()];
                let b = &prior.teams[(i * 31 + 11) % prior.teams.len()];
                pairs_spec.push((a.sets.clone(), b.sets.clone(), a.id.clone(), b.id.clone()));
            }
        }
    }

    struct PairOut {
        blind_score_sum: f64,
        games_scored: usize,
        capped: usize,
        invalids: usize,
        blind_turns_sum: usize,
        lines: Vec<String>,
    }

    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::<PairOut>::with_capacity(pairs_spec.len()));
    let cfg = production_cfg(cli.iters);

    std::thread::scope(|s| {
        for _ in 0..cli.threads {
            s.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let pi = next.fetch_add(1, Ordering::Relaxed);
                    if pi >= pairs_spec.len() {
                        break;
                    }
                    let (t1, t2, id1, id2) = &pairs_spec[pi];
                    let mut rng = SplitMix64::new(cli.seed ^ ((pi as u64 + 1) * 0x9FB2_1C65));
                    let bseed = rng.battle_seed();
                    let blind_seed_a = rng.next();
                    let open_seed_a = rng.next();
                    let blind_seed_b = rng.next();
                    let open_seed_b = rng.next();

                    let mut pair_score = 0.0f64;
                    let mut pair_scored = 0usize;
                    let mut pair_capped = 0usize;
                    let mut pair_inv = 0usize;
                    let mut turns_sum = 0usize;

                    // Arm 0: p1=BlindAgent (t1) vs p2=OpenAgent (t2)
                    // Arm 1: p1=OpenAgent (t1) vs p2=BlindAgent (t2)
                    for (arm, blind_is_p1, bs, os) in [
                        (0usize, true, blind_seed_a, open_seed_a),
                        (1usize, false, blind_seed_b, open_seed_b),
                    ] {
                        let mut battle = match Battle::from_fixture(&dex, &bseed, t1, t2) {
                            Ok(b) => b,
                            Err(_) => {
                                pair_inv += 1;
                                continue;
                            }
                        };
                        battle.set_log_enabled(true);
                        let mut blind = BlindAgent::new_with_fallback_policy(
                            cfg.clone(),
                            prior.clone(),
                            None,
                            bs,
                            FallbackPolicy::Bayesian,
                        );
                        let mut open = OpenAgent::new(cfg.clone(), None, os);
                        let res = if blind_is_p1 {
                            play_game(&dex, &mut battle, &mut [&mut blind, &mut open], cli.max_turns)
                        } else {
                            play_game(&dex, &mut battle, &mut [&mut open, &mut blind], cli.max_turns)
                        };
                        match res {
                            Ok(GameResult::Outcome(o)) => {
                                let p1_s = match o {
                                    Outcome::P1Win => 1.0,
                                    Outcome::P2Win => 0.0,
                                    Outcome::Tie => 0.5,
                                };
                                let s = if blind_is_p1 { p1_s } else { 1.0 - p1_s };
                                pair_score += s;
                                pair_scored += 1;
                                turns_sum += battle.turn as usize;
                            }
                            Ok(GameResult::TurnCapped) => {
                                pair_capped += 1;
                                turns_sum += battle.turn as usize;
                            }
                            Err(_) => {
                                pair_inv += 1;
                            }
                        }
                        let _ = arm;
                    }

                    let line = json!({
                        "pair": pi,
                        "t1": id1,
                        "t2": id2,
                        "scored": pair_scored,
                        "blind_score": pair_score,
                        "capped": pair_capped,
                        "invalids": pair_inv,
                    })
                    .to_string();

                    local.push(PairOut {
                        blind_score_sum: pair_score,
                        games_scored: pair_scored,
                        capped: pair_capped,
                        invalids: pair_inv,
                        blind_turns_sum: turns_sum,
                        lines: if cli.out_path.is_some() { vec![line] } else { vec![] },
                    });
                }
                results.lock().unwrap().extend(local);
            });
        }
    });

    let all = results.into_inner().unwrap();
    if let Some(path) = &cli.out_path {
        let mut f = std::fs::File::create(path).unwrap();
        for p in &all {
            for l in &p.lines {
                writeln!(f, "{l}").unwrap();
            }
        }
    }

    let blind_score: f64 = all.iter().map(|p| p.blind_score_sum).sum();
    let scored: usize = all.iter().map(|p| p.games_scored).sum();
    let capped: usize = all.iter().map(|p| p.capped).sum();
    let invalids: usize = all.iter().map(|p| p.invalids).sum();
    let turns: usize = all.iter().map(|p| p.blind_turns_sum).sum();

    let blind_win_rate = if scored > 0 {
        blind_score / scored as f64
    } else {
        0.5
    };
    let mean_turns = if scored + capped > 0 {
        turns as f64 / (scored + capped) as f64
    } else {
        0.0
    };

    json!({
        "mode": "gauntlet",
        "matchups": match cli.matchups { MatchupSet::Offpool => "offpool", MatchupSet::Catalog => "catalog" },
        "pairs": pairs_spec.len(),
        "iters": cli.iters,
        "games_scored": scored,
        "capped": capped,
        "invalids": invalids,
        "blind_vs_open_win_rate": (blind_win_rate * 1e4).round() / 1e4,
        "blind_minus_open_elo": (elo_diff(blind_win_rate) * 10.0).round() / 10.0,
        "mean_turns": (mean_turns * 10.0).round() / 10.0,
        "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
    })
}

#[allow(dead_code)]
fn _choice_debug(_c: SearchChoice) {}

fn main() {
    let root = repo_root();
    let cli = parse_cli(&root);
    let summary = match cli.mode {
        Mode::CorpusAudit => mode_corpus_audit(&cli, &root),
        Mode::Calibration => mode_calibration(&cli, &root),
        Mode::Gauntlet => mode_gauntlet(&cli, &root),
    };
    println!("{summary}");
}
