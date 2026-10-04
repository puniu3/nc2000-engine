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
    trace: Option<std::io::BufWriter<std::fs::File>>,
    cfg: RmConfig,
    inner: BlindAgent,
    decisions: usize,
    fallback: usize,
    exposed: usize,
    missing: usize,
    early_exposed: usize,
    early_combo_samples: usize,
}
impl Agent for Measured {
    fn name(&self) -> String {
        "measured".into()
    }
    fn choose(&mut self, b: &Battle, dex: &Dex, side: usize, cs: &[SearchChoice]) -> SearchChoice {
        let search_seed = self.inner.search_rng_state();
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
                if mo.appeared && !has("meanlook") && belief.is_fallback() {
                    self.early_exposed += 1;
                    let mut rng = SplitMix64::new(771);
                    for _ in 0..64 {
                        let sim = belief.determinize(dex, b, obs, &mut rng);
                        let p = sim.poke(sim.active_id(1 - side).unwrap());
                        self.early_combo_samples += usize::from(["meanlook", "perishsong"].iter()
                            .all(|key| p.move_slots.iter().any(|m| dex.moves.key(m.id) == *key)));
                    }
                }
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
        if let Some(file) = self.trace.as_mut() {
            let model = belief.determinize(dex, b, obs, &mut SplitMix64::new(771));
            let mut root = Vec::new();
            if cs.len() > 1 {
                let mut search = nc2000_bot::blind::BlindSearch::new(
                    b,
                    dex,
                    self.cfg.clone(),
                    side,
                    search_seed,
                );
                search.step(dex, belief, obs, self.cfg.iterations);
                assert_eq!(
                    search.best(),
                    Some(choice),
                    "trace search must reproduce the live decision"
                );
                root = search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(dex),"visits":search.visits()[i],"mean":search.means()[i],"dominated":search.dominated()[i]})).collect();
            }
            let row = json!({"decision":self.decisions,"turn":b.turn,"side":side,"state_key":format!("{:032x}",b.state_key128()),"seed":search_seed,"choice":choice.to_input(dex),"root":root,"truth":board(b,dex),"model":board(&model,dex),"reveals":obs.mons().iter().map(|m|json!({"species":dex.species.key(m.species),"moves":m.revealed_moves.iter().map(|m|dex.moves.key(*m)).collect::<Vec<_>>(),"item":format!("{:?}",m.item)})).collect::<Vec<_>>()});
            writeln!(file, "{row}").unwrap();
            file.flush().unwrap();
        }
        choice
    }
}
fn board(b: &Battle, dex: &Dex) -> Value {
    json!((0..2).map(|s|json!({"active":b.sides[s].active,"party":b.sides[s].party.iter().copied().collect::<Vec<_>>(),"roster":b.sides[s].roster.iter().map(|p|json!({"species":dex.species.key(p.species),"hp":p.hp,"maxhp":p.maxhp,"status":p.status.as_str(),"item":p.item.map(|i|dex.items.key(i)),"boosts":format!("{:?}",p.boosts),"trapped":p.trapped,"moves":p.move_slots.iter().map(|m|json!({"move":dex.moves.key(m.id),"pp":m.pp})).collect::<Vec<_>>(),"volatiles":p.volatiles.iter().map(|(c,v)|json!([dex.conds_key(*c),v.duration])).collect::<Vec<_>>()})).collect::<Vec<_>>()})).collect::<Vec<_>>())
}
struct Recorder<'a> {
    inner: &'a mut dyn Agent,
    inputs: Arc<Mutex<Vec<Value>>>,
}
impl Agent for Recorder<'_> {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn choose(&mut self, b: &Battle, dex: &Dex, side: usize, cs: &[SearchChoice]) -> SearchChoice {
        let pick = self.inner.choose(b, dex, side, cs);
        self.inputs.lock().unwrap().push(json!({"key":format!("{:032x}",b.state_key128()),"turn":b.turn,"side":side,"choice":pick.to_input(dex)}));
        pick
    }
}
struct Opponent {
    inner: BlindAgent,
    forced: bool,
    adaptive: bool,
    opened: bool,
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
        if self.adaptive {
            if !self.opened {
                self.opened = true;
                let lock = dex.moves.id("meanlook").map(SearchChoice::Move);
                if let Some(lock) = lock.filter(|c| cs.contains(c)) { return lock; }
            }
            return ordinary();
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
    assert!(["stress", "adaptive", "natural", "onpool"].contains(&mode.as_str()));
    let trace_dir = arg("--trace-dir", "");
    if !trace_dir.is_empty() {
        std::fs::create_dir_all(&trace_dir).unwrap();
    }
    let only: Option<Vec<usize>> = args
        .iter()
        .position(|a| a == "--only")
        .map(|i| args[i + 1].split(',').map(|s| s.parse().unwrap()).collect());
    let candidate = if args.iter().any(|a| a == "--same") {
        FallbackPolicy::Layered
    } else if args.iter().any(|a| a == "--early-perish") {
        FallbackPolicy::EarlyPerish
    } else if args.iter().any(|a| a == "--retain-perish") {
        FallbackPolicy::RetainPerish
    } else {
        FallbackPolicy::RevealedLoadout
    };
    let triad = args.iter().any(|a| a == "--triad");
    assert!(!(triad && args.iter().any(|a| a == "--retain-perish")));
    let only_arm = arg("--arm", "");
    assert!(["", "a", "b", "c"].contains(&only_arm.as_str()));
    let mut policies = if triad {
        if args.iter().any(|a| a == "--same") { vec![FallbackPolicy::Layered;3] }
        else if args.iter().any(|a| a == "--early-perish") { vec![FallbackPolicy::Layered,FallbackPolicy::RetainPerish,FallbackPolicy::EarlyPerish] }
        else { vec![FallbackPolicy::Layered,FallbackPolicy::RevealedLoadout,FallbackPolicy::RevealedLoadoutDraw] }
    } else { vec![FallbackPolicy::Layered,candidate] };
    if !only_arm.is_empty() {
        assert!(only_arm != "c" || triad);
        policies = vec![policies[(only_arm.as_bytes()[0] - b'a') as usize]];
    }
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
                if only.as_ref().is_some_and(|jobs|!jobs.contains(&job)) { continue; }
                let mut rng=SplitMix64::new(seed^(job as u64).wrapping_mul(0x9e3779b97f4a7c15));
                let own_i=weight_pick(&weights,&mut rng);
                let opp_i=if mode=="onpool" {weight_pick(&weights,&mut rng)} else {job%opponents.teams.len()};
                let battle_seed=rng.battle_seed();let bot_seed=rng.next();let opp_seed=rng.next();
                let side=(job/opponents.teams.len()+job)%2;
                let teams:[&[PokemonSet];2]=if side==0 {[&own.teams[own_i].sets,&opponents.teams[opp_i].sets]} else {[&opponents.teams[opp_i].sets,&own.teams[own_i].sets]};
                let mut arms=Vec::new();
                for &policy in &policies {
                    let started=std::time::Instant::now();
                    let mut battle=Battle::from_fixture(&dex,&battle_seed,teams[0],teams[1]).unwrap();battle.set_log_enabled(true);
                    let trace=if trace_dir.is_empty() {None} else {Some(std::io::BufWriter::new(std::fs::File::create(format!("{trace_dir}/{job}-{}.jsonl",policy.id())).unwrap()))};
                    let mut me=Measured {trace,cfg:cfg.clone(),inner:BlindAgent::new_with_fallback_policy(cfg.clone(),prior.clone(),None,bot_seed,policy),decisions:0,fallback:0,exposed:0,missing:0,early_exposed:0,early_combo_samples:0};
                    let mut opp=Opponent {inner:BlindAgent::new_with_fallback_policy(cfg.clone(),prior.clone(),None,opp_seed,FallbackPolicy::Layered),forced:mode=="stress"||mode=="adaptive",adaptive:mode=="adaptive",opened:false};
                    let inputs=Arc::new(Mutex::new(Vec::new()));
                    let (score,turns,tally)=if trace_dir.is_empty() {
                        let agents:[&mut dyn Agent;2]=if side==0 {[&mut me,&mut opp]} else {[&mut opp,&mut me]};
                        combo_common::play(&dex,&mut battle,agents,500)
                    } else {
                        let mut mine=Recorder{inner:&mut me,inputs:inputs.clone()};
                        let mut theirs=Recorder{inner:&mut opp,inputs:inputs.clone()};
                        let agents:[&mut dyn Agent;2]=if side==0 {[&mut mine,&mut theirs]} else {[&mut theirs,&mut mine]};
                        let result=combo_common::play(&dex,&mut battle,agents,500);
                        std::fs::write(format!("{trace_dir}/{job}-{}-replay.json",policy.id()),serde_json::to_string(&json!({"teams":teams,"battle_seed":battle_seed,"inputs":*inputs.lock().unwrap(),"log":battle.log})).unwrap()).unwrap();
                        result
                    };
                    arms.push(json!({"elapsed_seconds":started.elapsed().as_secs_f64(),"policy":policy.id(),"score":if side==0 {score}else{1.0-score},"turns":turns,"terminal":battle.outcome().is_some(),"trajectory":format!("{:016x}",digest(&battle.log)),"decisions":me.decisions,"fallback":me.fallback,"exposed":me.exposed,"missing":me.missing,"early_exposed":me.early_exposed,"early_combo_samples":me.early_combo_samples,"own":tally[side].json(),"opponent":tally[1-side].json(),"opponent_bond_moves":battle.log.iter().filter(|l| l.starts_with(&format!("|move|p{}",2-side)) && l.contains("|Destiny Bond|")).count(),"opponent_switch_events":battle.log.iter().filter(|l|l.starts_with(&format!("|switch|p{}",2-side))).count()}));
                }
                let mut row=json!({"implementation":if args.iter().any(|a|a=="--early-perish") {"early-perish-v1"} else if args.iter().any(|a|a=="--retain-perish") {"retain-perish-v1"} else {"compatible-gender-v1"},"job":job,"mode":mode,"iterations":iterations,"c":0.4,"ucb_c":0.4,"seed":seed,"battle_seed":battle_seed,"bot_seed":bot_seed,"opp_seed":opp_seed,"side":side,"own_team":own.teams[own_i].id,"opponent_team":opponents.teams[opp_i].id,"a":arms.first(),"b":arms.get(1)});
                if !only_arm.is_empty() {
                    row.as_object_mut().unwrap().remove("a");row.as_object_mut().unwrap().remove("b");
                    row[&only_arm]=arms[0].clone();
                } else if triad {row["c"]=arms[2].clone();}
                let mut f=out.lock().unwrap();writeln!(f,"{row}").unwrap();f.flush().unwrap();eprintln!("completed {job}");
            });
        }
    });
}
