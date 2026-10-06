use nc2000_bot::{
    preview::load_meta_pool,
    smmcts::{RmConfig, SelRule},
    Belief, BlindSearch, Observer,
};
use nc2000_engine::{battle::{enumerate::enumerate_step_partial, Outcome, PokemonSet}, state::Battle};
use serde_json::{json, Value};
use std::{
    io::BufRead,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};

#[derive(Clone)]
struct Case {
    battle: Battle,
    observer: Observer,
    game: Value,
    decision: usize,
    side: usize,
    exposed: bool,
    order: u128,
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(args.len() == 3 || (args.len() == 4 && (args[3] == "--audit-exclusions" || args[3].starts_with("--audit-case="))),
        "corpus output [--audit-exclusions | --audit-case=game:decision:side]");
    let audit = args.len() == 4;
    let focus = args.get(3).and_then(|s| s.strip_prefix("--audit-case=")).map(|s| {
        let keys = s.split(':').map(|v| v.parse::<usize>().unwrap()).collect::<Vec<_>>();
        assert_eq!(keys.len(), 3);
        keys
    });
    let dex = conformance::load_dex();
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let cfg = RmConfig {
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let mut cases = Vec::new();
    let mut decisions = 0;
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        b.set_log_enabled(true);
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
            for side in [0, 1] {
                if !b.needs_choice()[side] || legal[side].len() < 2 || b.active_id(side).is_none() {
                    continue;
                }
                decisions += 1;
                let search = BlindSearch::new(&b, &dex, cfg.clone(), side, 0);
                let exposed =
                    search.dominated().iter().any(|&x| x) && search.dominated().iter().any(|&x| !x);
                let order = b
                    .state_key128()
                    .wrapping_mul(0x9e3779b97f4a7c15u128)
                    .wrapping_add(side as u128);
                cases.push(Case {
                    battle: b.clone(),
                    observer: observers[side].clone(),
                    game: game["game"].clone(),
                    decision,
                    side,
                    exposed,
                    order,
                });
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| legal[s][i as usize])),
            )
            .unwrap();
            for s in [0, 1] {
                observers[s].observe(&b, &dex);
            }
        }
    }
    let exposed = cases.iter().filter(|c| c.exposed).count();
    cases.sort_by_key(|c| c.order);
    let mut quotas = if audit { [0, usize::MAX] } else { [8usize, 48] };
    cases.retain(|c| {
        if focus.as_ref().is_some_and(|f| c.game.as_u64() != Some(f[0] as u64) || c.decision != f[1] || c.side != f[2]) { return false; }
        let i = usize::from(c.exposed);
        if quotas[i] == 0 {
            false
        } else {
            quotas[i] -= 1;
            true
        }
    });
    let detail = focus.as_ref().map(|_| {
        assert_eq!(cases.len(), 1);
        let c = &cases[0];
        let mut base = c.battle.clone();
        let actions = [0, 1].map(|side| base.legal_choices(&dex, side));
        let mut cells = Vec::new();
        for &own in &actions[c.side] {
            for &reply in &actions[1-c.side] {
                let joint = if c.side == 0 { [own, reply] } else { [reply, own] };
                let result = enumerate_step_partial(&dex, &c.battle, joint.map(Some), 100_000);
                let mut lower = 0.0;
                let mut unknown = result.pending.iter().map(|p| p.mass).sum::<f64>();
                let mut losses = 0.0;
                let mut successors = std::collections::BTreeMap::<String, f64>::new();
                for leaf in &result.leaves {
                    *successors.entry(leaf.battle.state_key128().to_string()).or_default() += leaf.prob;
                    let value = leaf.battle.outcome().map(|outcome| match outcome {
                        Outcome::P1Win => if c.side == 0 {1.0} else {0.0},
                        Outcome::P2Win => if c.side == 1 {1.0} else {0.0},
                        Outcome::Tie => 0.5,
                    });
                    if let Some(v) = value { lower += leaf.prob*v; losses += leaf.prob*(1.0-v); }
                    else { unknown += leaf.prob; }
                }
                assert!((lower + losses + unknown - 1.0).abs() < 1e-9);
                cells.push(json!({"own":own.to_input(&dex),"reply":reply.to_input(&dex),"lower":lower,"upper":(lower+unknown).min(1.0),"runs":result.runs,"complete":!result.exhausted && result.pending.is_empty(),"successors":successors}));
            }
        }
        let active = [0, 1].map(|side| {
            let p = c.battle.poke(c.battle.active_id(side).unwrap());
            json!({"species":dex.species.key(p.species),"hp":p.hp,"maxhp":p.maxhp,"status":format!("{:?}",p.status),"alive":c.battle.sides[side].pokemon_left})
        });
        json!({"scope":"True-state one-step bounds; nonterminal and unfinished branches stay [0,1]. Not a public-information guarantee.","active":active,"cells":cells,"masked":nc2000_bot::smmcts::dominated_actions(&c.battle,&dex,c.side).iter().map(|(a,r)|json!({"action":a.to_input(&dex),"reason":r})).collect::<Vec<_>>()})
    });
    let next = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(||loop {
        let i=next.fetch_add(1,Ordering::Relaxed);if i>=cases.len(){break;}let c=&cases[i];let seed=0x20261007u64^(c.order as u64);
        let belief=Belief::new(&dex,&pool,&c.observer);let mut arms=Vec::new();
        for prune in [false,true].into_iter().filter(|&prune| !audit || !prune) {
            let mut search=BlindSearch::new(&c.battle,&dex,cfg.clone(),c.side,seed);if prune {search.prune_dominated();}
            let started=Instant::now();let mut checkpoints=Vec::new();
            for budget in [3000,27000] {
                search.step(&dex,&belief,&c.observer,budget-search.iterations());
                checkpoints.push(json!({"iterations":budget,"best":search.best().unwrap().to_input(&dex),"visits":search.visits(),"means":search.means(),"nodes":search.node_count(),"elapsed_ms":started.elapsed().as_secs_f64()*1000.}));
            }
            if prune {assert!(search.visits().iter().zip(search.dominated()).all(|(&n,&masked)|!masked || n==0));}
            arms.push(json!({"prune":prune,"dominated":search.dominated(),"actions":search.actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"checkpoints":checkpoints}));
        }
        if !c.exposed {for j in 0..2 {let a=&arms[0]["checkpoints"][j];let b=&arms[1]["checkpoints"][j];for k in ["iterations","best","visits","means","nodes"] {assert_eq!(a[k],b[k]);}}}
        rows.lock().unwrap().push(json!({"index":i,"game":c.game,"decision":c.decision,"side":c.side,"exposed":c.exposed,"state_key":c.battle.state_key128().to_string(),"seed":seed,"arms":arms}));
        eprintln!("root {}/{} complete",i+1,cases.len());
    });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|r| r["index"].as_u64().unwrap());
    let mut output = json!({"decisions":decisions,"exposed":exposed,"sampled":rows.len(),"rows":rows});
    if let Some(detail) = detail { output["detail"] = detail; }
    std::fs::write(&args[2], serde_json::to_vec(&output).unwrap()).unwrap();
}
