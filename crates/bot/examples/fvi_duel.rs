//! S1 strength gate: seed-paired, side-swapped duels over the training team
//! pools with a shared team preview (`value::shared_preview`), so only
//! in-battle play differs between arms.
//!
//!   fvi_duel --a net:MODEL.json[:K] --b eval1ply[:K] | skuct:ITERS[:C]
//!            [--games 400] [--seed 1] [--threads N] [--max-turns 500]

use std::sync::Arc;

use nc2000_bot::agent::Agent;
use nc2000_bot::duel::{run_duel, DuelSpec};
use nc2000_bot::eval::EvalWeights;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmAgent, RmConfig, SelRule};
use nc2000_bot::value::{shared_preview, Leaf, ValueAgent, ValueCodec, ValueNet};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn arg_s(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

/// A search agent with the duel's shared preview instead of its own.
struct SharedPreview(Box<dyn Agent>);

impl Agent for SharedPreview {
    fn name(&self) -> String {
        self.0.name()
    }
    fn choose(&mut self, b: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> SearchChoice {
        if b.turn == 0 {
            shared_preview(b, side, choices)
        } else {
            self.0.choose(b, dex, side, choices)
        }
    }
}

enum Arm {
    Net(usize),
    Eval(usize),
    Skuct(u32, f64),
}

fn parse(spec: &str, nets: &mut Vec<String>) -> Arm {
    let p: Vec<&str> = spec.split(':').collect();
    match p[0] {
        "net" => {
            nets.push(p[1].to_string());
            Arm::Net(p.get(2).map_or(16, |k| k.parse().unwrap()))
        }
        "eval1ply" => Arm::Eval(p.get(1).map_or(16, |k| k.parse().unwrap())),
        "skuct" => Arm::Skuct(
            p.get(1).map_or(27000, |k| k.parse().unwrap()),
            p.get(2).map_or(1.0, |c| c.parse().unwrap()),
        ),
        other => panic!("unknown arm {other}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let codec = Arc::new(ValueCodec::new(&dex));
    let mut net_paths = Vec::new();
    let a = parse(&arg_s(&args, "--a", "eval1ply"), &mut net_paths);
    let b = parse(&arg_s(&args, "--b", "skuct:27000"), &mut net_paths);
    let mut nets = net_paths.iter().map(|p| Arc::new(ValueNet::load(&root.join(p)).expect("model")));
    let teams: Vec<Vec<PokemonSet>> = ["data/team-pool-v2/team-pool.json", "data/belief-pool-v3/belief-pool.json"]
        .iter()
        .flat_map(|p| load_meta_pool(&root.join(p)).teams.into_iter().map(|t| t.sets))
        .collect();
    let mut spec = DuelSpec::new(
        arg_s(&args, "--games", "400").parse().unwrap(),
        arg_s(&args, "--seed", "1").parse().unwrap(),
    );
    spec.threads = arg_s(&args, "--threads", "10").parse().unwrap();
    spec.max_turns = arg_s(&args, "--max-turns", "500").parse().unwrap();
    spec.progress = true;
    let build = |arm: &Arm, net: Option<Arc<ValueNet>>| -> Box<dyn Fn(u64) -> Box<dyn Agent> + Sync> {
        match *arm {
            Arm::Net(k) => {
                let (net, codec) = (net.unwrap(), codec.clone());
                Box::new(move |s| {
                    Box::new(ValueAgent::new(Leaf::Net(net.clone(), codec.clone()), k, s)) as Box<dyn Agent>
                })
            }
            Arm::Eval(k) => Box::new(move |s| {
                Box::new(ValueAgent::new(Leaf::Eval(EvalWeights::default()), k, s)) as Box<dyn Agent>
            }),
            Arm::Skuct(iterations, c) => Box::new(move |s| {
                Box::new(SharedPreview(Box::new(RmAgent::new(
                    RmConfig { iterations, rule: SelRule::Ucb, c, hp_buckets: 16, ..Default::default() },
                    s,
                )))) as Box<dyn Agent>
            }),
        }
    };
    let net_a = matches!(a, Arm::Net(_)).then(|| nets.next().unwrap());
    let net_b = matches!(b, Arm::Net(_)).then(|| nets.next().unwrap());
    let ba = build(&a, net_a);
    let bb = build(&b, net_b);
    let st = run_duel(&dex, &teams, &*ba, &*bb, spec);
    println!(
        "A {} vs B {}: score {:.3} ± {:.3} ({}W {}L {}T, caps {}) | ms/move A {:.1} B {:.1} | {:.0}s",
        arg_s(&args, "--a", "eval1ply"),
        arg_s(&args, "--b", "skuct:27000"),
        st.score,
        st.ci95,
        st.wins,
        st.losses,
        st.ties,
        st.turn_caps,
        st.a_ms_per_move,
        st.b_ms_per_move,
        st.secs
    );
}
