use std::io::{BufRead, Write};
use std::time::Instant;

use conformance::fixture::repo_root;
use nc2000_bot::player::PlayerFrame;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{ProtocolAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    New {
        side: usize,
        team: Vec<PokemonSet>,
        seed: u64,
    },
    Choose {
        frame: PlayerFrame,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--features" | "--describe" | "--prune-root" | "--shared-search" | "--pick-preview" => index += 1,
            "--iters"
            | "--c"
            | "--pool"
            | "--dex"
            | "--model"
            | "--leaf-model"
            | "--pick-prior"
            | "--leaf-preview-iters"
            | "--shared-iters"
            | "--temperature" => {
                if index + 1 == args.len() {
                    return Err(format!("missing value for {}", args[index]).into());
                }
                index += 2;
            }
            name => return Err(format!("unknown option {name}").into()),
        }
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let iterations: u32 = flag("--iters")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(30_000);
    if iterations == 0 {
        return Err("--iters must be positive".into());
    }
    let ucb_c: f64 = flag("--c").map(|s| s.parse()).transpose()?.unwrap_or(1.0);
    if !ucb_c.is_finite() || ucb_c <= 0.0 {
        return Err("--c must be positive and finite".into());
    }
    let search_config = RmConfig {
        c: ucb_c,
        ..RmConfig::default()
    };
    let leaf_preview_iterations: u32 = flag("--leaf-preview-iters")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(0);
    let pool_path = flag("--pool")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo_root().join("data/meta-pool-v0/meta-pool.json"));
    let dex_path = flag("--dex")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo_root().join("data/gen2stadium2.json"));
    let dex = Dex::from_json(&std::fs::read_to_string(dex_path)?)?;
    let pool = load_meta_pool(&pool_path);
    if args.iter().any(|a| a == "--describe") {
        println!(
            "{}",
            serde_json::to_string(&nc2000_bot::learning::Vocabulary::from_dex(&dex))?
        );
        return Ok(());
    }
    let features = args.iter().any(|a| a == "--features");
    let prune_root = args.iter().any(|a| a == "--prune-root");
    let shared_search = args.iter().any(|a| a == "--shared-search");
    let shared_iterations: u32 = flag("--shared-iters")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(iterations);
    if shared_iterations == 0 || (flag("--shared-iters").is_some() && !shared_search) {
        return Err("--shared-iters requires --shared-search and a positive budget".into());
    }
    let temperature: f64 = flag("--temperature")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(0.0);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("temperature must be finite and nonnegative".into());
    }
    let model = flag("--model")
        .map(|path| {
            let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            nc2000_bot::learned::PolicyValue::from_json(&text, &dex)
        })
        .transpose()?;
    let leaf_model = flag("--leaf-model")
        .map(|path| {
            let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            nc2000_bot::learned::PolicyValue::from_json(&text, &dex)
        })
        .transpose()?;
    let pick_prior = flag("--pick-prior").map(|path| {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        nc2000_bot::pick_prior::PickPrior::from_json(&text).map(std::sync::Arc::new)
    }).transpose()?;
    let pick_preview = args.iter().any(|arg| arg == "--pick-preview");
    if pick_preview && (pick_prior.is_none() || model.is_some()) {
        return Err("--pick-preview requires a pick prior and search".into());
    }
    if model.is_some() && leaf_model.is_some() {
        return Err("choose either --model or --leaf-model".into());
    }
    if model.is_some() && prune_root {
        return Err("--prune-root requires search".into());
    }
    if shared_search && (model.is_some() || leaf_model.is_some() || prune_root) {
        return Err("--shared-search is a separate search arm".into());
    }
    if leaf_preview_iterations > 0 && leaf_model.is_none() {
        return Err("--leaf-preview-iters requires --leaf-model".into());
    }
    let mut agent: Option<ProtocolAgent> = None;
    let mut policy_rng = SplitMix64::new(0);
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let command: Command = serde_json::from_str(&line?)?;
        let response = match command {
            Command::New { side, team, seed } => {
                if side > 1 {
                    return Err("side must be 0 or 1".into());
                }
                let mut next =
                    ProtocolAgent::new(&dex, side, pool.clone(), search_config.clone(), seed);
                next.set_own_team(team);
                if let Some(prior) = &pick_prior {
                    next.set_pick_prior(prior.clone());
                }
                next.set_pick_preview(pick_preview);
                agent = Some(next);
                policy_rng = SplitMix64::new(seed ^ 0xC6BC279692B5C323);
                json!({"ready": true, "iterations": iterations})
            }
            Command::Choose { frame } => {
                let started = Instant::now();
                let agent = agent.as_mut().ok_or("choose before new")?;
                for line in &frame.lines {
                    agent.push_line(&dex, line);
                }
                if !agent.on_request(&dex, &frame.request.to_string())? {
                    return Err("choose on wait request".into());
                }
                if prune_root {
                    agent.prune_root()?;
                }
                let encoded = if features || model.is_some() {
                    Some(nc2000_bot::learning::observation(agent, &dex)?)
                } else {
                    None
                };
                let prediction = model
                    .as_ref()
                    .map(|model| model.predict(encoded.as_ref().unwrap()))
                    .transpose()?;
                let mut log_prob = None;
                let mut leaf_calls = 0;
                let mut shared_metrics = None;
                let mut shared_policy = None;
                let action = if let Some((logits, _)) = &prediction {
                    let encoded = encoded.as_ref().unwrap();
                    let eligible: Vec<usize> = logits
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| {
                            encoded.actions[*i].eligible
                                && frame.legal_actions.contains(&encoded.actions[*i].input)
                        })
                        .map(|(i, _)| i)
                        .collect();
                    let best = eligible
                        .iter()
                        .copied()
                        .max_by(|&i, &j| logits[i].total_cmp(&logits[j]))
                        .ok_or("model has no legal action")?;
                    let best = if temperature > 0.0 {
                        let peak = logits[best] as f64;
                        let weights: Vec<f64> = eligible
                            .iter()
                            .map(|&i| ((logits[i] as f64 - peak) / temperature).exp())
                            .collect();
                        let total: f64 = weights.iter().sum();
                        let mut draw = policy_rng.next_f64() * total;
                        let mut selected = weights.len() - 1;
                        for (i, &weight) in weights.iter().enumerate() {
                            draw -= weight;
                            if draw < 0.0 {
                                selected = i;
                                break;
                            }
                        }
                        log_prob = Some((weights[selected] / total).ln());
                        eligible[selected]
                    } else {
                        best
                    };
                    encoded.actions[best].input.clone()
                } else if shared_search
                    && !agent.search().is_some_and(|s| s.is_preview())
                    && frame.legal_actions.len() > 1
                {
                    let mut search = nc2000_bot::shared_search::SharedSearch::new(
                        agent.battle().ok_or("no battle")?,
                        &dex,
                        agent.side(),
                        search_config.clone(),
                        policy_rng.next(),
                    );
                    search.step(
                        &dex,
                        agent.belief().ok_or("no belief")?,
                        agent.observer().ok_or("no observer")?,
                        shared_iterations,
                    );
                    let action = nc2000_bot::player::action_input(&dex, search.best());
                    shared_metrics = Some(
                        json!({"nodes": search.node_count(), "mean_depth": search.mean_depth()}),
                    );
                    shared_policy = Some(json!({
                        "iterations": shared_iterations,
                        "actions": search.root_policy().iter().map(|&(action, visits, mean)| json!({
                            "input": nc2000_bot::player::action_input(&dex, action), "visits": visits, "mean": mean,
                        })).collect::<Vec<_>>(),
                    }));
                    action
                } else {
                    if leaf_preview_iterations > 0 && agent.search().is_some_and(|s| s.is_preview())
                    {
                        agent.step(&dex, leaf_preview_iterations)?;
                    } else if let Some(model) = &leaf_model {
                        let root_observer = agent.observer().ok_or("no observer")?.clone();
                        let belief = agent.belief().ok_or("no belief")?;
                        let fallback = belief.is_fallback();
                        let candidates = belief.candidate_count();
                        let side = agent.side();
                        let mut error = None;
                        agent.step_with_leaf(&dex, iterations, &mut |sim, _, _| {
                            leaf_calls += 1;
                            let mut observed = root_observer.clone();
                            observed.observe(sim, &dex);
                            let input = nc2000_bot::learning::state_observation(
                                sim,
                                &dex,
                                &observed,
                                side,
                                fallback,
                                candidates,
                                &[],
                                &[],
                            );
                            match model.predict_value(&input) {
                                Ok(value) => {
                                    if side == 0 {
                                        value as f64
                                    } else {
                                        1.0 - value as f64
                                    }
                                }
                                Err(message) => {
                                    error = Some(message);
                                    0.5
                                }
                            }
                        })?;
                        if let Some(error) = error {
                            return Err(error.into());
                        }
                    } else {
                        agent.step(&dex, iterations)?;
                    }
                    agent.best(&dex).ok_or("no selected action")?
                };
                if !frame.legal_actions.contains(&action) {
                    return Err(format!("selected illegal action {action}").into());
                }
                let elapsed_ns = started.elapsed().as_nanos() as u64;
                let mut response = json!({
                    "action": action,
                    "elapsed_ns": elapsed_ns,
                    "iterations": if shared_metrics.is_some() { shared_iterations } else { agent.iterations() },
                    "legality_drift": agent.legality_drift,
                    "projections": agent.projections,
                    "leaf_calls": leaf_calls,
                });
                if features {
                    response["hp_evidence_enabled"] = json!(cfg!(feature = "hp-type-evidence"));
                    response["hp_type_masks"] = json!(agent.observer().map(|o| o.mons().iter().map(|m| m.hp_type_mask).collect::<Vec<_>>()));
                    response["observation"] = serde_json::to_value(encoded)?;
                    response["root_policy"] = match shared_policy {
                        Some(policy) => policy,
                        None => serde_json::from_str(&agent.root_policy(&dex))?,
                    };
                }
                if let Some(metrics) = shared_metrics {
                    response["shared_search"] = metrics;
                }
                if let Some((logits, value)) = prediction {
                    response["logits"] = serde_json::to_value(logits)?;
                    response["value"] = json!(value);
                    response["log_prob"] = json!(log_prob);
                    response["temperature"] = json!(temperature);
                }
                response
            }
        };
        serde_json::to_writer(&mut output, &response)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}
