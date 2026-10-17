//! Experimental strength/cost comparison, using the standard paired duel.
//! smab_duel eval:4x4|net:4x4|mcts:3000 B GAMES SEED THREADS OUT [--crn]
#[path = "smab_duel/search.rs"]
mod search;

use nc2000_bot::{
    agent::Agent,
    duel::{run_duel, DuelSpec},
    eval::EvalWeights,
    preview::load_meta_pool,
    smmcts::{RmAgent, RmConfig, SelRule},
    value::{sample_mixed, shared_preview, Leaf, ValueCodec, ValueNet},
    SplitMix64,
};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    dex::Dex,
    state::Battle,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::BufRead,
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Default)]
struct Metrics {
    samples_ns: Vec<u64>,
    counts: BTreeMap<String, u64>,
    depths: BTreeMap<usize, u64>,
    caps: u64,
}
struct Candidate {
    leaf: Arc<Leaf>,
    ks: Vec<usize>,
    rng: SplitMix64,
    metrics: Arc<Mutex<Metrics>>,
}
impl Agent for Candidate {
    fn name(&self) -> String {
        format!("smab:{:?}", self.ks)
    }
    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let result =
            search::choose_policy(b, dex, &self.leaf, &self.ks, side, self.rng.next(), 100000);
        let acts = nc2000_bot::value::legal_actions(b, dex);
        assert_eq!(
            acts[side].iter().map(|a| a.unwrap()).collect::<Vec<_>>(),
            choices
        );
        assert_eq!(result.policy.len(), choices.len());
        let mut metrics = self.metrics.lock().unwrap();
        *metrics.depths.entry(result.depth).or_default() += 1;
        metrics.caps += result.capped as u64;
        for (key, n) in result.counts.as_object().unwrap() {
            *metrics.counts.entry(key.clone()).or_default() += n.as_u64().unwrap();
        }
        drop(metrics);
        choices[sample_mixed(&result.policy, 0., &mut self.rng)]
    }
}
struct Shared {
    inner: Box<dyn Agent>,
    metrics: Arc<Mutex<Metrics>>,
}
impl Agent for Shared {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if matches!(choices[0], SearchChoice::Team(_)) {
            return shared_preview(b, side, choices);
        }
        if choices.len() == 1 {
            return choices[0];
        }
        let start = Instant::now();
        let pick = self.inner.choose(b, dex, side, choices);
        self.metrics
            .lock()
            .unwrap()
            .samples_ns
            .push(start.elapsed().as_nanos() as u64);
        pick
    }
}
fn build(
    spec: &str,
    eval: Arc<Leaf>,
    net: Arc<Leaf>,
    metrics: Arc<Mutex<Metrics>>,
) -> Box<dyn Fn(u64) -> Box<dyn Agent> + Sync> {
    let (kind, budget) = spec.split_once(':').unwrap();
    let iters = if kind == "mcts" {
        budget.parse().unwrap()
    } else {
        0
    };
    let ks: Vec<usize> = if kind == "mcts" {
        vec![]
    } else {
        budget.split('x').map(|k| k.parse().unwrap()).collect()
    };
    let leaf = if kind == "eval" { eval } else { net };
    let candidate = kind != "mcts";
    assert!(matches!(kind, "mcts" | "eval" | "net"));
    Box::new(move |seed| {
        let inner: Box<dyn Agent> = if candidate {
            Box::new(Candidate {
                leaf: leaf.clone(),
                ks: ks.clone(),
                rng: SplitMix64::new(seed),
                metrics: metrics.clone(),
            })
        } else {
            Box::new(RmAgent::new(
                RmConfig {
                    iterations: iters,
                    c: 0.4,
                    rule: SelRule::Ucb,
                    ..Default::default()
                },
                seed,
            ))
        };
        Box::new(Shared {
            inner,
            metrics: metrics.clone(),
        })
    })
}
fn metrics(m: &Mutex<Metrics>) -> Value {
    let mut m = m.lock().unwrap();
    m.samples_ns.sort_unstable();
    let n = m.samples_ns.len();
    let pct = |p: usize| {
        if n == 0 {
            0.
        } else {
            m.samples_ns[(n * p).div_ceil(100).saturating_sub(1).min(n - 1)] as f64 / 1e6
        }
    };
    json!({"decisions":n,"mean_ms":m.samples_ns.iter().sum::<u64>() as f64/n.max(1) as f64/1e6,"p50_ms":pct(50),"p95_ms":pct(95),"p99_ms":pct(99),"work_caps":m.caps,"completed_depths":m.depths,"counts":m.counts,"samples_ns":m.samples_ns})
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let dex = conformance::load_dex();
    let eval = Arc::new(Leaf::Eval(EvalWeights::default()));
    let net = Arc::new(Leaf::Net(
        Arc::new(ValueNet::load(std::path::Path::new("tmp/fvi/model-d3.json")).unwrap()),
        Arc::new(ValueCodec::new(&dex)),
    ));
    if args[1] == "check" {
        let roots: Value =
            serde_json::from_slice(&std::fs::read("findings/smab-screen-v1/roots.json").unwrap())
                .unwrap();
        let selected: Vec<_> = roots["roots"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 4 == 0)
            .map(|(_, r)| r)
            .collect();
        let mut checks = Vec::new();
        for line in std::io::BufReader::new(std::fs::File::open(&args[2]).unwrap()).lines() {
            let g: Value = serde_json::from_str(&line.unwrap()).unwrap();
            let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(g["teams"].clone()).unwrap();
            let mut b =
                Battle::from_fixture(&dex, g["seed"].as_str().unwrap(), &teams[0], &teams[1])
                    .unwrap();
            b.set_log_enabled(false);
            for (di, d) in g["decisions"].as_array().unwrap().iter().enumerate() {
                assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
                if selected
                    .iter()
                    .any(|r| r["game"] == g["game"] && r["decision"] == di)
                {
                    for (name, leaf) in [("eval", &eval), ("net", &net)] {
                        for ks in [vec![4, 4], vec![2, 2, 2]] {
                            let result = search::check_position(&b, &dex, leaf, &ks);
                            checks.push(json!({"game":g["game"],"decision":di,"leaf":name,"ks":ks,"check":result}));
                        }
                    }
                }
                let acts = [0, 1].map(|s| b.legal_choices(&dex, s));
                b.apply_choices(
                    &dex,
                    [0, 1].map(|s| d["choices"][s].as_u64().map(|i| acts[s][i as usize])),
                )
                .unwrap();
            }
        }
        assert_eq!(checks.len(), 12);
        std::fs::write(&args[3], serde_json::to_vec_pretty(&checks).unwrap()).unwrap();
        return;
    }
    assert!(args.len() >= 7, "A B games seed threads output [--crn]");
    let a = Arc::new(Mutex::new(Metrics::default()));
    let b = Arc::new(Mutex::new(Metrics::default()));
    let ba = build(&args[1], eval.clone(), net.clone(), a.clone());
    let bb = build(&args[2], eval, net, b.clone());
    let mut spec = DuelSpec::new(args[3].parse().unwrap(), args[4].parse().unwrap());
    spec.threads = args[5].parse().unwrap();
    spec.max_turns = 300;
    spec.progress = true;
    spec.crn_agent_seeds = args.iter().any(|s| s == "--crn");
    let teams: Vec<_> = load_meta_pool(std::path::Path::new("data/team-pool-v2/team-pool.json"))
        .teams
        .into_iter()
        .map(|t| t.sets)
        .collect();
    let st = run_duel(&dex, &teams, &*ba, &*bb, spec);
    let result = json!({"a":args[1],"b":args[2],"seed":spec.base_seed,"threads":spec.threads,"max_turns":spec.max_turns,"crn_agent_seeds":spec.crn_agent_seeds,"games":st.games,"pairs":st.pairs,"score":st.score,"ci95":st.ci95,"wins":st.wins,"losses":st.losses,"ties":st.ties,"turn_caps":st.turn_caps,"pair_scores":st.pair_scores,"game_scores":st.game_scores,"avg_turns":st.avg_turns,"wall_seconds":st.secs,"a_all_choice_ms":st.a_ms_per_move,"b_all_choice_ms":st.b_ms_per_move,"a_metrics":metrics(&a),"b_metrics":metrics(&b)});
    let out = &args[6];
    std::fs::write(out, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    println!(
        "{}",
        json!({"a":args[1],"b":args[2],"score":st.score,"ci95":st.ci95,"a_ms":result["a_metrics"]["mean_ms"],"b_ms":result["b_metrics"]["mean_ms"],"turn_caps":st.turn_caps})
    );
}
