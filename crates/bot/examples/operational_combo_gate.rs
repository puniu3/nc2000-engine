use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    preview::{load_meta_pool, MetaPool},
    smmcts::SelRule,
    Agent, Belief, BlindAgent, FallbackPolicy, Observer, RmConfig, SplitMix64,
};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    dex::Dex,
    state::Battle,
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use std::{
    io::Write,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
mod combo_common;

struct Measured {
    inner: BlindAgent,
    decisions: usize,
    fallback: usize,
    exposed: usize,
    missing: usize,
}
impl Agent for Measured {
    fn name(&self) -> String {
        "measured".into()
    }
    fn choose(&mut self, b: &Battle, dex: &Dex, side: usize, cs: &[SearchChoice]) -> SearchChoice {
        let choice = self.inner.choose(b, dex, side, cs);
        self.decisions += 1;
        let belief = self.inner.belief().unwrap();
        let obs = self.inner.observer().unwrap();
        self.fallback += usize::from(belief.is_fallback());
        if let Some(foe) = b.active_id(1 - side) {
            if dex.species.key(b.poke(foe).species) == "misdreavus" {
                let mo = obs
                    .mons()
                    .iter()
                    .find(|m| m.species == b.poke(foe).species)
                    .unwrap();
                let has = |key| mo.revealed_moves.iter().any(|&m| dex.moves.key(m) == key);
                if has("meanlook") && !has("perishsong") && belief.is_fallback() {
                    self.exposed += 1;
                    let sim = belief.determinize(dex, b, obs, &mut SplitMix64::new(771));
                    let p = sim.poke(sim.active_id(1 - side).unwrap());
                    self.missing += usize::from(
                        !p.move_slots
                            .iter()
                            .any(|m| dex.moves.key(m.id) == "perishsong"),
                    );
                }
            }
        }
        choice
    }
}
struct Opponent {
    inner: BlindAgent,
    forced: bool,
}
impl Agent for Opponent {
    fn name(&self) -> String {
        "opponent".into()
    }
    fn choose(&mut self, b: &Battle, dex: &Dex, side: usize, cs: &[SearchChoice]) -> SearchChoice {
        let mut ordinary = || self.inner.choose(b, dex, side, cs);
        if !self.forced {
            return ordinary();
        }
        if matches!(cs.first(), Some(SearchChoice::Team(_))) {
            let SearchChoice::Team(t) = ordinary() else {
                unreachable!()
            };
            return cs
                .iter()
                .copied()
                .filter(|c| matches!(c,SearchChoice::Team(u) if u[0]==1))
                .max_by_key(|c| match c {
                    SearchChoice::Team(u) => u.iter().filter(|p| t.contains(p)).count(),
                    _ => 0,
                })
                .unwrap();
        }
        let Some(id) = b.active_id(side) else {
            return ordinary();
        };
        let me = b.poke(id);
        if dex.species.key(me.species) != "misdreavus" || me.fainted {
            return ordinary();
        }
        let Some(foe_id) = b.active_id(1 - side) else {
            return ordinary();
        };
        let foe = b.poke(foe_id);
        if foe.fainted {
            return ordinary();
        }
        let count = |p: &nc2000_engine::state::Pokemon| {
            dex.conds_id("perishsong")
                .and_then(|c| p.volatile(c))
                .and_then(|v| v.duration)
        };
        let mv = |key| {
            dex.moves
                .id(key)
                .map(SearchChoice::Move)
                .filter(|a| cs.contains(a))
        };
        if count(me) == Some(1) {
            let picked = ordinary();
            if matches!(picked, SearchChoice::Switch(_)) {
                return picked;
            }
            if let Some(s) = cs.iter().find(|c| matches!(c, SearchChoice::Switch(_))) {
                return *s;
            }
        }
        if count(foe).is_none() && b.sides[side].pokemon_left > 1 && !me.trapped {
            if foe.trapped || b.sides[1 - side].pokemon_left == 1 {
                if let Some(a) = mv("perishsong") {
                    return a;
                }
            } else if let Some(a) = mv("meanlook") {
                return a;
            }
        }
        if count(foe).is_some() {
            if me.last_move != dex.moves.id("protect") {
                if let Some(a) = mv("protect") {
                    return a;
                }
            }
            if me.hp * 2 < me.maxhp {
                if let Some(a) = mv("rest") {
                    return a;
                }
            }
            for key in ["destinybond", "confuseray", "zapcannon", "meanlook"] {
                if let Some(a) = mv(key) {
                    return a;
                }
            }
        }
        ordinary()
    }
}
fn weight_pick(weights: &[f64], rng: &mut SplitMix64) -> usize {
    let mut x = rng.next_f64() * weights.iter().sum::<f64>();
    for (i, w) in weights.iter().enumerate() {
        x -= w;
        if x < 0.0 {
            return i;
        }
    }
    weights.len() - 1
}
fn digest(lines: &[String]) -> u64 {
    lines
        .iter()
        .flat_map(|s| s.bytes().chain([10]))
        .fold(0xcbf29ce484222325, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        })
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str, d: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
            .unwrap_or_else(|| d.into())
    };
    let root = repo_root();
    let dex = load_dex();
    let iterations: u32 = arg("--iters", "27000").parse().unwrap();
    let games: usize = arg("--games", "64").parse().unwrap();
    let start: usize = arg("--start", "0").parse().unwrap();
    let seed: u64 = arg("--seed", "202610041").parse().unwrap();
    let mode = arg("--mode", "stress");
    assert!(["stress", "natural", "onpool"].contains(&mode.as_str()));
    let candidate = if args.iter().any(|a| a == "--same") {
        FallbackPolicy::Layered
    } else {
        FallbackPolicy::RevealedLoadout
    };
    let prior = Arc::new(load_meta_pool(
        &root.join("data/belief-pool-v3/belief-pool.json"),
    ));
    let catalog: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("data/team-pool-v2/team-pool.json")).unwrap(),
    )
    .unwrap();
    let own: MetaPool = serde_json::from_value(catalog.clone()).unwrap();
    let weights: Vec<f64> = catalog["teams"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["drawWeight"].as_f64().unwrap())
        .collect();
    let opponents = if mode == "onpool" {
        own.clone()
    } else {
        load_meta_pool(&root.join("findings/operational-combo-validation-v1/opponents.json"))
    };
    let learnsets = Learnsets::from_json(
        &std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap(),
    )
    .unwrap();
    for t in &opponents.teams {
        let v = validate_team(&dex, &learnsets, &serde_json::to_string(&t.sets).unwrap());
        assert_eq!(v["ok"], true, "{}: {}", t.id, v);
        let b = Battle::from_fixture(&dex, "1,2,3,4", &own.teams[0].sets, &t.sets).unwrap();
        let obs = Observer::new(&b, 0);
        let bel = Belief::new(&dex, &prior, &obs);
        assert_eq!(bel.is_fallback(), mode != "onpool", "{}", t.id);
    }
    let cfg = RmConfig {
        iterations,
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let cursor = AtomicUsize::new(start);
    let out = Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(arg("--out", "tmp/operational-validation/games.jsonl")).unwrap(),
    ));
    std::thread::scope(|scope| {
        for _ in 0..arg("--threads", "8").parse::<usize>().unwrap() {
            scope.spawn(|| loop {
                let job=cursor.fetch_add(1,Ordering::Relaxed);if job>=start+games {break;}
                let mut rng=SplitMix64::new(seed^(job as u64).wrapping_mul(0x9e3779b97f4a7c15));
                let own_i=weight_pick(&weights,&mut rng);
                let opp_i=if mode=="onpool" {weight_pick(&weights,&mut rng)} else {job%opponents.teams.len()};
                let battle_seed=rng.battle_seed();let bot_seed=rng.next();let opp_seed=rng.next();
                let side=(job/opponents.teams.len()+job)%2;
                let teams:[&[PokemonSet];2]=if side==0 {[&own.teams[own_i].sets,&opponents.teams[opp_i].sets]} else {[&opponents.teams[opp_i].sets,&own.teams[own_i].sets]};
                let mut arms=Vec::new();
                for policy in [FallbackPolicy::Layered,candidate] {
                    let started=std::time::Instant::now();
                    let mut battle=Battle::from_fixture(&dex,&battle_seed,teams[0],teams[1]).unwrap();battle.set_log_enabled(true);
                    let mut me=Measured {inner:BlindAgent::new_with_fallback_policy(cfg.clone(),prior.clone(),None,bot_seed,policy),decisions:0,fallback:0,exposed:0,missing:0};
                    let mut opp=Opponent {inner:BlindAgent::new(cfg.clone(),prior.clone(),None,opp_seed),forced:mode=="stress"};
                    let agents:[&mut dyn Agent;2]=if side==0 {[&mut me,&mut opp]} else {[&mut opp,&mut me]};
                    let (score,turns,tally)=combo_common::play(&dex,&mut battle,agents,500);
                    arms.push(json!({"elapsed_seconds":started.elapsed().as_secs_f64(),"policy":policy.id(),"score":if side==0 {score}else{1.0-score},"turns":turns,"terminal":battle.outcome().is_some(),"trajectory":format!("{:016x}",digest(&battle.log)),"decisions":me.decisions,"fallback":me.fallback,"exposed":me.exposed,"missing":me.missing,"own":tally[side].json(),"opponent":tally[1-side].json()}));
                }
                let row=json!({"job":job,"mode":mode,"iterations":iterations,"c":0.4,"seed":seed,"battle_seed":battle_seed,"bot_seed":bot_seed,"opp_seed":opp_seed,"side":side,"own_team":own.teams[own_i].id,"opponent_team":opponents.teams[opp_i].id,"a":arms[0],"b":arms[1]});
                let mut f=out.lock().unwrap();writeln!(f,"{row}").unwrap();f.flush().unwrap();eprintln!("completed {job}");
            });
        }
    });
}
