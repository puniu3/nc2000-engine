use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

use conformance::fixture::repo_root;
use nc2000_bot::player::action_input;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::SplitMix64;
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WorkerSpec {
    program: PathBuf,
    args: Vec<String>,
    #[serde(default)]
    artifacts: Vec<PathBuf>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: String,
    seed: u64,
    games: usize,
    threads: usize,
    pool: PathBuf,
    #[serde(default = "default_dex")]
    dex: PathBuf,
    agents: [WorkerSpec; 2],
    record: bool,
    #[serde(default)]
    crn_agent_seeds: bool,
}

fn default_dex() -> PathBuf {
    repo_root().join("data/gen2stadium2.json")
}

struct Worker {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Worker {
    fn spawn(spec: &WorkerSpec) -> Result<Self, String> {
        let mut child = Command::new(&spec.program)
            .args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", spec.program.display()))?;
        let input = BufWriter::new(child.stdin.take().unwrap());
        let output = BufReader::new(child.stdout.take().unwrap());
        Ok(Self {
            child,
            input,
            output,
        })
    }

    fn call(&mut self, command: &Value) -> Result<Value, String> {
        serde_json::to_writer(&mut self.input, command).map_err(|e| e.to_string())?;
        writeln!(self.input).map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        if self
            .output
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
        {
            return Err(format!("worker exited: {:?}", self.child.try_wait()));
        }
        serde_json::from_str(&line).map_err(|e| format!("worker response: {e}: {line}"))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn fingerprint(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn play(
    dex: &Dex,
    config: &Config,
    teams: &[Vec<PokemonSet>],
    workers: &mut [Worker; 2],
    game: usize,
) -> Result<Value, String> {
    let pair = game / 2;
    let swap = game % 2;
    let mut schedule = SplitMix64(SplitMix64(
        config
            .seed
            .wrapping_add((pair as u64).wrapping_mul(0x9E3779B97F4A7C15)),
    ).next());
    let team_ids = [schedule.below(teams.len()), schedule.below(teams.len())];
    let battle_seed = schedule.battle_seed();
    let agent_seeds = [schedule.next() as u32; 2];
    let mut battle =
        Battle::from_fixture(dex, &battle_seed, &teams[team_ids[0]], &teams[team_ids[1]])
            .map_err(|e| format!("battle init: {e:?}"))?;
    for side in 0..2 {
        let agent = side ^ swap;
        let reply = workers[agent].call(&json!({
            "op": "new", "side": side, "battle": battle, "seed": agent_seeds[agent],
        }))?;
        if reply["ready"] != true {
            return Err(format!("worker not ready: {reply}"));
        }
    }
    let mut frames = Vec::new();
    let mut timing: [Vec<u64>; 2] = [Vec::new(), Vec::new()];
    let started = Instant::now();
    loop {
        if let Some(outcome) = battle.outcome() {
            let score_p1 = match outcome {
                Outcome::P1Win => 1.0,
                Outcome::P2Win => 0.0,
                Outcome::Tie => 0.5,
            };
            return Ok(json!({
                "type": "game", "game": game, "pair": pair, "swap": swap,
                "team_ids": team_ids, "battle_seed": battle_seed, "agent_seeds": agent_seeds,
                "score": if swap == 0 { score_p1 } else { 1.0 - score_p1 },
                "outcome": match outcome { Outcome::P1Win => "p1", Outcome::P2Win => "p2", Outcome::Tie => "tie" },
                "turns": battle.turn, "capped": false,
                "elapsed_ns": started.elapsed().as_nanos() as u64,
                "decision_ns": timing, "terminal_key": battle.state_key(), "terminal_seed": battle.prng.seed_str(),
                "frames": if config.record { Some(frames) } else { None },
            }));
        }
        if battle.turn > 1100 {
            return Err("engine failed to terminate before turn 1100".into());
        }
        let state = serde_json::to_value(&battle).map_err(|e| e.to_string())?;
        let state_bytes = serde_json::to_vec(&state).map_err(|e| e.to_string())?;
        let state_hash = format!("{:x}", Sha256::digest(&state_bytes));
        let key = battle.state_key();
        let mut picks = [None, None];
        for side in 0..2 {
            let choices = battle.legal_choices(dex, side);
            if choices.is_empty() {
                continue;
            }
            let agent = side ^ swap;
            let decision_started = Instant::now();
            let response = workers[agent].call(&json!({"op": "choose", "battle": state}))?;
            if response["key"] != key { return Err("transferred state key mismatch".into()); }
            let elapsed = decision_started.elapsed().as_nanos() as u64;
            let action = response["action"]
                .as_str()
                .ok_or("worker returned no action")?;
            let choice = choices
                .iter()
                .copied()
                .find(|&c| action_input(dex, c) == action)
                .ok_or_else(|| {
                    format!("side {side} turn {}: illegal action {action}", battle.turn)
                })?;
            timing[agent].push(elapsed);
            if config.record {
                frames.push(json!({"side": side, "agent": agent, "turn": battle.turn, "state_sha256": state_hash, "state_key": key, "legal_actions": choices.iter().map(|&a| action_input(dex, a)).collect::<Vec<_>>(), "response": response}));
            }
            picks[side] = Some(choice);
        }
        if picks == [None, None] {
            return Err("no choices in unfinished battle".into());
        }
        battle
            .apply_choices(dex, picks)
            .map_err(|e| format!("apply choices: {e:?}"))?;
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: learning_arena CONFIG.json OUT.jsonl [--resume]".into());
    }
    let config: Config =
        serde_json::from_str(&std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if config.schema != "nc2000-native-deployed-duel-v1" || config.games == 0
        || config.games % 2 != 0 || config.threads == 0 || !config.record || !config.crn_agent_seeds
    {
        return Err("native duel requires paired recorded games with common seeds".into());
    }
    let worker_hashes: Vec<Value> = config
        .agents
        .iter()
        .map(|spec| {
            let artifacts: Result<Vec<_>, String> = spec
                .artifacts
                .iter()
                .map(|p| Ok(json!({"path": p, "hash": fingerprint(p)?})))
                .collect();
            Ok(json!({"program": fingerprint(&spec.program)?, "artifacts": artifacts?}))
        })
        .collect::<Result<_, String>>()?;
    let metadata = json!({
        "type": "manifest", "config": config,
        "hashes": {"arena": fingerprint(&std::env::current_exe().map_err(|e| e.to_string())?)?,
            "pool": fingerprint(&config.pool)?, "dex": fingerprint(&config.dex)?,
            "agents": worker_hashes},
    });
    let resume = args.iter().any(|a| a == "--resume");
    let mut completed = std::collections::BTreeSet::new();
    let output = if resume {
        let mut rows = BufReader::new(File::open(&args[2]).map_err(|e| e.to_string())?).lines();
        let existing: Value = serde_json::from_str(
            &rows
                .next()
                .ok_or("empty resume file")?
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if existing != metadata {
            return Err("resume manifest differs from current inputs/binaries".into());
        }
        for row in rows {
            let row: Value = serde_json::from_str(&row.map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            let game = row["game"].as_u64().ok_or("resume row has no game")? as usize;
            let valid_kind = row["type"] == "game" && row["capped"] == false;
            if !valid_kind || game >= config.games || !completed.insert(game) {
                return Err("invalid/duplicate resume row".into());
            }
        }
        OpenOptions::new()
            .append(true)
            .open(&args[2])
            .map_err(|e| e.to_string())?
    } else {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])
            .map_err(|e| e.to_string())?;
        serde_json::to_writer(&mut file, &metadata).map_err(|e| e.to_string())?;
        writeln!(file).map_err(|e| e.to_string())?;
        file
    };
    let jobs: Vec<_> = (0..config.games)
        .filter(|g| !completed.contains(g))
        .collect();
    if jobs.is_empty() {
        eprintln!("all {} games already complete", config.games);
        return Ok(());
    }
    let dex = Arc::new(
        Dex::from_json(&std::fs::read_to_string(&config.dex).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?,
    );
    let teams: Arc<Vec<_>> = Arc::new(
        load_meta_pool(&config.pool)
            .teams
            .iter()
            .map(|t| t.sets.clone())
            .collect(),
    );
    if teams.len() < 2 {
        return Err("pool needs at least two teams".into());
    }
    let jobs = Arc::new(jobs);
    let next = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let config = Arc::new(config);
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();
    for _ in 0..config.threads.min(jobs.len()) {
        let (dex, teams, jobs, next, failed, config, tx) = (
            dex.clone(),
            teams.clone(),
            jobs.clone(),
            next.clone(),
            failed.clone(),
            config.clone(),
            tx.clone(),
        );
        handles.push(std::thread::spawn(move || {
            let result = (|| {
                let mut workers = [
                    Worker::spawn(&config.agents[0])?,
                    Worker::spawn(&config.agents[1])?,
                ];
                while !failed.load(Ordering::Relaxed) {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= jobs.len() {
                        break;
                    }
                    let game = jobs[index];
                    let row = play(&dex, &config, &teams, &mut workers, game)
                        .map_err(|e| format!("game {game}: {e}"))?;
                    tx.send(Ok(row)).map_err(|e| e.to_string())?;
                }
                Ok::<_, String>(())
            })();
            if let Err(error) = result {
                failed.store(true, Ordering::Relaxed);
                let _ = tx.send(Err(error));
            }
        }));
    }
    drop(tx);
    let mut writer = BufWriter::new(output);
    let mut count = completed.len();
    let mut error = None;
    for result in rx {
        match result {
            Ok(row) => {
                serde_json::to_writer(&mut writer, &row).map_err(|e| e.to_string())?;
                writeln!(writer).map_err(|e| e.to_string())?;
                writer.flush().map_err(|e| e.to_string())?;
                count += 1;
                eprintln!("{count}/{} game={} score={} turns={}",
                    config.games, row["game"], row["score"], row["turns"]);
            }
            Err(e) => {
                eprintln!("ERROR {e}");
                error = Some(e);
            }
        }
    }
    for handle in handles {
        handle.join().map_err(|_| "worker thread panicked")?;
    }
    if let Some(error) = error {
        return Err(error);
    }
    if count != config.games {
        return Err(format!("incomplete: {count}/{}", config.games));
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
