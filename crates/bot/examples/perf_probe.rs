use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use nc2000_bot::belief::Belief;
use nc2000_bot::blind::BlindSearch;
use nc2000_bot::observe::Observer;
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::smmcts::{SelRule, SkuctSearch};
use nc2000_bot::{Agent, RmAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::{json, Value};

#[allow(dead_code)]
mod combo_common;

struct Position {
    battle: Battle,
    observer: Observer,
    tag: String,
}

fn positions(dex: &Dex, pool: &MetaPool) -> Vec<Position> {
    let mut rng = SplitMix64::new(0x20261002);
    let mut out = Vec::new();
    for game in 0..8 {
        let combo = game >= 6;
        let a = if combo {
            pool.teams.iter().position(|t| t.sets.iter().any(combo_common::is_trapper)).unwrap()
        } else { rng.below(pool.teams.len()) };
        let b = rng.below(pool.teams.len());
        let mut battle = Battle::from_fixture(dex, &rng.battle_seed(), &pool.teams[a].sets, &pool.teams[b].sets).unwrap();
        let mut observer = Observer::new(&battle, 0);
        let cfg = RmConfig { iterations: 200, c: 0.4, rule: SelRule::Ucb, ..Default::default() };
        let mut agents: Vec<Box<dyn Agent>> = (0..2).map(|s| {
            Box::new(combo_common::Forced {
                inner: Box::new(RmAgent::new(cfg.clone(), rng.next())),
                lead: if combo && s == 0 { pool.teams[a].sets.iter().position(combo_common::is_trapper).map(|i| i as u8 + 1) } else { None },
                specialist: combo,
            }) as Box<dyn Agent>
        }).collect();
        let mut captured_switch = false;
        for decision in 0..80 {
            if battle.outcome().is_some() { break; }
            observer.observe(&battle, dex);
            let acts = [0, 1].map(|s| battle.legal_choices(dex, s));
            let forced = !acts[0].is_empty() && acts[0].iter().all(|c| matches!(c, SearchChoice::Switch(_) | SearchChoice::Pass));
            if !acts[0].is_empty() && ([0, 1, 4, 10, 25].contains(&decision) || (forced && !captured_switch)) {
                captured_switch |= forced;
                let tag = if decision == 0 { "preview" } else if forced { "replacement" } else if combo { "perish" } else { "ordinary" };
                let mut snapshot = battle.clone();
                snapshot.set_log_enabled(false);
                out.push(Position { battle: snapshot, observer: observer.clone(), tag: tag.into() });
            }
            let choices = [0, 1].map(|s| if acts[s].is_empty() { None } else { Some(agents[s].choose(&battle, dex, s, &acts[s])) });
            battle.apply_choices(dex, choices).unwrap();
        }
    }
    let mut healers = Vec::new();
    for set in pool.teams.iter().flat_map(|t| &t.sets) {
        if set.level <= 50 && set.moves.iter().any(|m| combo_common::norm(m) == "rest")
            && !healers.iter().any(|p: &nc2000_engine::battle::PokemonSet| p.species == set.species) {
            healers.push(set.clone());
        }
        if healers.len() == 3 { break; }
    }
    assert_eq!(healers.len(),3);
    for size in [1, 3] {
        let team = healers[..size].to_vec();
        let mut battle = Battle::from_fixture(dex, &rng.battle_seed(), &team, &team).unwrap();
        let choices = [0, 1].map(|s| battle.legal_choices(dex,s).first().copied());
        battle.apply_choices(dex,choices).unwrap();
        for s in 0..2 {
            let id = battle.active_id(s).unwrap();
            battle.poke_mut(id).hp = battle.poke(id).maxhp / 2;
        }
        let mut observer = Observer::new(&battle,0);
        for _ in 0..3 {
            observer.observe(&battle,dex);
            let mut snapshot = battle.clone();
            snapshot.set_log_enabled(false);
            out.push(Position { battle:snapshot, observer:observer.clone(), tag:"recovery".into() });
            battle.apply_choices(dex,[Some(SearchChoice::Move(dex.moves.id("rest").unwrap()));2]).unwrap();
        }
    }
    out
}

fn measure(p: &Position, dex: &Dex, pool: &MetaPool, mode: &str, iters: u32, id: usize) -> Value {
    let mut belief = Belief::new(dex, pool, &p.observer);
    belief.sync(dex, &p.observer);
    let nobp = mode.ends_with("-nobp");
    let with_res = mode.ends_with("-res");
    let base_mode = mode
        .strip_suffix("-nobp")
        .or_else(|| mode.strip_suffix("-res"))
        .or_else(|| mode.strip_suffix("-nores"))
        .unwrap_or(mode);
    let playout = nc2000_bot::Playout::Heavy {
        eps: 0.2,
        turns: 8,
        weights: nc2000_bot::EvalWeights {
            residual_scale: if with_res { 1.0 } else { 0.0 },
            ..nc2000_bot::EvalWeights::default()
        },
    };
    let cfg = RmConfig {
        iterations: iters,
        c: 0.4,
        rule: SelRule::Ucb,
        playout,
        rollout_baton_pass: !nobp,
        ..Default::default()
    };
    let start = Instant::now();
    let signature = match base_mode {
        "blind" | "budget" => {
            let mut s = BlindSearch::new(&p.battle, dex, cfg, 0, id as u64 + 71);
            let mut completed = 0;
            if base_mode == "budget" {
                while start.elapsed().as_secs_f64() < iters as f64 {
                    s.step(dex,&belief,&p.observer,64);
                    completed += 64;
                }
            } else {
                s.step(dex, &belief, &p.observer, iters);
                completed = iters;
            }
            let seconds = start.elapsed().as_secs_f64();
            return json!({"id":id,"tag":p.tag,"seconds":seconds,"completed":completed,"state":p.battle.state_key().to_string(),"signature":format!("{:?}/{:?}/{}",s.best(),s.root_matrix(),s.node_count())});
        }
        "skuct" => {
            let mut s = SkuctSearch::new(&p.battle, dex, cfg, id as u64 + 71);
            s.step(dex, iters);
            let seconds = start.elapsed().as_secs_f64();
            return json!({"id":id,"tag":p.tag,"seconds":seconds,"state":p.battle.state_key().to_string(),"signature":format!("{:?}/{:?}/{:?}/{}",s.best(0),s.visits(0),s.means(0),s.node_count())});
        }
        "clone" => { for _ in 0..iters { black_box(p.battle.clone()); } 0 }
        "hash" => { for _ in 0..iters { black_box(p.battle.state_key_bucketed(16)); } 0 }
        "determinize" => {
            let mut rng = SplitMix64::new(71);
            for _ in 0..iters { black_box(belief.determinize(dex, &p.battle, &p.observer, &mut rng)); }
            0
        }
        "rollout" => {
            let mut rng = SplitMix64::new(71);
            let mut sum = 0.0;
            for _ in 0..iters {
                let mut b = p.battle.clone();
                b.reseed(rng.next());
                let cap = b.turn.saturating_add(cfg.horizon);
                sum += nc2000_bot::mcts::playout_value(&mut b,dex,&cfg.playout,cap,&mut rng,cfg.rollout_rules());
            }
            sum.to_bits()
        }
        "rollout-cutoff" => {
            let mut rng = SplitMix64::new(71);
            let mut terminal = 0u64;
            let mut leaf_eval = 0u64;
            for _ in 0..iters {
                let mut b = belief.determinize(dex, &p.battle, &p.observer, &mut rng);
                let cap = b.turn.saturating_add(cfg.horizon);
                let v = nc2000_bot::mcts::playout_value(&mut b, dex, &cfg.playout, cap, &mut rng, cfg.rollout_rules());
                if v == 0.0 || v == 1.0 {
                    terminal += 1;
                } else {
                    leaf_eval += 1;
                }
            }
            return json!({"id":id,"tag":p.tag,"turn":p.battle.turn,"terminal":terminal,"leaf_eval":leaf_eval,"leaf_eval_frac":leaf_eval as f64 / (terminal + leaf_eval) as f64});
        }
        "step" => {
            let mut root = p.battle.clone();
            let actions = [0,1].map(|s| root.legal_choices(dex,s).first().copied());
            let mut checksum = 0;
            for i in 0..iters {
                let mut b = p.battle.clone();
                b.reseed(i as u64 + 71);
                b.apply_choices(dex,actions).unwrap();
                checksum ^= b.turn as u64;
                black_box(&b);
            }
            checksum
        }
        _ => panic!("mode must be blind, skuct, clone, hash, determinize, rollout, or step"),
    };
    json!({"id":id,"tag":p.tag,"seconds":start.elapsed().as_secs_f64(),"state":p.battle.state_key().to_string(),"signature":signature})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("blind");
    let iters: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(27000);
    let threads: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
    let limit: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
    assert!(iters > 0 && threads > 0 && limit > 0);
    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let pool = Arc::new(load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json")));
    let mut states = positions(&dex, &pool);
    if mode == "budget" {
        let index = states.iter().position(|p| p.tag == "ordinary" && p.battle.turn >= 4).unwrap();
        states = vec![states.remove(index)];
    }
    states.truncate(limit);
    eprintln!("ready: {} states, {mode}, {iters} iterations, {threads} threads, pid {}",states.len(),std::process::id());
    black_box(measure(&states[0], &dex, &pool, if mode == "budget" { "blind" } else { mode }, iters.min(1000), 0));
    let next = AtomicUsize::new(0);
    let start = Instant::now();
    let rows = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads).map(|_| scope.spawn(|| {
            let mut rows = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= states.len() { break; }
                rows.push(measure(&states[i], &dex, &pool, mode, iters, i));
            }
            rows
        })).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    let wall = start.elapsed().as_secs_f64();
    let mut rows = rows;
    rows.sort_by_key(|r|r["id"].as_u64().unwrap());
    println!("{}",json!({"schema":"nc2000-perf-v1","mode":mode,"iterations":iters,"threads":threads,"wall_seconds":wall,"rows":rows}));
}
