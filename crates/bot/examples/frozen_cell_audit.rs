use nc2000_bot::{frozen::FrozenPolicy, root_policy::{analyze_bayesian_root, RootCell, RootType}, Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Deserialize)]
struct Decision { key: String, choices: [Option<usize>; 2] }
#[derive(Deserialize)]
struct Game { seed: String, teams: [Vec<PokemonSet>; 2], decisions: Vec<Decision> }
#[derive(Deserialize)]
struct Request { id: String, corpus: String, expected: Value, trials: u32, evaluation_seed: u64 }

fn snapshot(search: &BlindSearch, dex: &Dex) -> Value {
    json!({"visits":search.visits(),"joint_probe_visits":search.joint_probe_visits(),"means":search.means(),
        "choice":search.best().map(|a|a.to_input(dex)),"dominated":search.dominated(),
        "candidate_policy":search.mixed_policy().map(|m|m.policy),
        "joint":search.root_matrix().into_iter().map(|(own,reply,samples,mean)|
            json!({"own":own,"reply":reply.to_input(dex),"samples":samples,"mean":mean})).collect::<Vec<_>>()})
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "requests.json output.json");
    let requests: Vec<Request> = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let dex = conformance::load_dex();
    let mut output = Vec::new();
    for request in requests {
        let game: Game = serde_json::from_str(&request.corpus).unwrap();
        let side = request.expected["side"].as_u64().unwrap() as usize;
        let seed = request.expected["seed"].as_u64().unwrap();
        let iterations = request.expected["iterations"].as_u64().unwrap() as u32;
        let probes = iterations*2/5;
        let mut battle = Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
        let mut observers = [0,1].map(|s|Observer::new(&battle,s));
        let mut beliefs = [0,1].map(|s|Belief::pinned_from_battle(&battle,&observers[s]));
        for (i, decision) in game.decisions.iter().enumerate() {
            assert_eq!(battle.state_key128().to_string(), decision.key);
            if i+1 == game.decisions.len() { break; }
            let legal = [0,1].map(|s|battle.legal_choices(&dex,s));
            battle.apply_choices(&dex,[0,1].map(|s|decision.choices[s].map(|a|legal[s][a]))).unwrap();
            for s in 0..2 { observers[s].observe(&battle,&dex); beliefs[s].sync(&dex,&observers[s]); }
        }
        let actions = [0,1].map(|s|battle.legal_choices(&dex,s));
        let mut search = BlindSearch::new(&battle,&dex,RmConfig { c:0.4,..Default::default() },side,seed);
        let started = Instant::now();
        let used = search.finish_with_mixed_probes(&dex,&beliefs[side],&observers[side],iterations,probes);
        let search_ms = started.elapsed().as_secs_f64()*1000.;
        assert_eq!(u64::from(used),request.expected["probes_used"].as_u64().unwrap());
        let before = snapshot(&search,&dex);
        for (key,value) in before.as_object().unwrap() { assert_eq!(&request.expected[key],value,"{} {key}",request.id); }
        let mut variants = Vec::new();
        for (name, policy) in [("most",FrozenPolicy::MostVisited),("sample",FrozenPolicy::SampleVisits)] {
            let started = Instant::now();
            let mut cells = Vec::new();
            let mut evidence = Vec::new();
            for (own, &a) in actions[side].iter().enumerate() {
                for (reply, &b) in actions[1-side].iter().enumerate() {
                    let pair = if side == 0 { [Some(a),Some(b)] } else { [Some(b),Some(a)] };
                    let mut sum = 0.;
                    let mut square_sum = 0.;
                    let mut terminal = 0;
                    let mut prefix = 0u64;
                    for trial in 0..request.trials {
                        let trial_seed = request.evaluation_seed ^ (u64::from(trial)).wrapping_mul(0x94d049bb133111eb)
                            ^ ((own*actions[1-side].len()+reply) as u64).wrapping_mul(0xa24baed4963ee407);
                        let result = search.evaluate_frozen_joint(&dex,&beliefs[side],&observers[side],pair,[policy;2],trial_seed,&mut |_|{});
                        let reward = if side == 0 {result.reward0} else {1.-result.reward0};
                        sum += reward;
                        square_sum += reward*reward;
                        terminal += u32::from(result.terminal);
                        prefix += u64::from(result.prefix_choices);
                    }
                    let mean = sum/f64::from(request.trials);
                    cells.push(json!({"own":own,"reply":reply,"sum":sum,"square_sum":square_sum,
                        "mean":mean,"terminal":terminal,"prefix_choices":prefix}));
                    evidence.push(RootCell {own,reply,samples:request.trials,mean});
                }
            }
            let samples = request.trials * cells.len() as u32;
            let types = [RootType {samples,replies:actions[1-side].len(),cells:evidence}];
            let eligible = search.root_policy_evidence().0;
            let analysis = analyze_bayesian_root(&eligible,samples,&types);
            variants.push(json!({"policy":name,"trials_per_cell":request.trials,"cells":cells,"analysis":analysis,
                "milliseconds":started.elapsed().as_secs_f64()*1000.}));
        }
        assert_eq!(before,snapshot(&search,&dex));
        assert_eq!(search.iterations(),iterations);
        output.push(json!({"id":request.id,"search_ms":search_ms,"search":before,"variants":variants}));
        eprintln!("{} complete",request.id);
    }
    std::fs::write(&args[2],serde_json::to_vec(&output).unwrap()).unwrap();
}
