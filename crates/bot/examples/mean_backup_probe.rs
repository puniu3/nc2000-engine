mod rng { pub use nc2000_bot::rng::*; }
#[path = "../src/exp3.rs"]
mod exp3;
use rng::SplitMix64;
use serde_json::json;
use std::time::Instant;

struct Node {
    visits: [Vec<u32>; 2],
    rewards: [Vec<f64>; 2],
    bandits: [exp3::Exp3Ix; 2],
    count: u32,
    total: f64,
}

impl Node {
    fn new() -> Self {
        Self { visits:[vec![0;2],vec![0;3]], rewards:[vec![0.;2],vec![0.;3]],
            bandits:[exp3::Exp3Ix::new(2),exp3::Exp3Ix::new(3)], count:0,total:0. }
    }

    fn ucb(&self, side: usize, rng: &mut SplitMix64) -> usize {
        let untried: Vec<_> = self.visits[side].iter().enumerate().filter(|(_,n)|**n==0).map(|(i,_)|i).collect();
        if !untried.is_empty() { return untried[rng.below(untried.len())]; }
        let log = f64::from(self.count).ln();
        let mut best = 0;
        let mut value = f64::NEG_INFINITY;
        for (i,&n) in self.visits[side].iter().enumerate() {
            let score = self.rewards[side][i]/f64::from(n)+0.4*(log/f64::from(n)).sqrt();
            if score > value { best=i;value=score; }
        }
        best
    }
}

fn simulate(nodes: &mut [Node], exp3: bool, average: bool, stochastic: bool, rng: &mut SplitMix64, p: f64) -> f64 {
    let Some((node, children)) = nodes.split_first_mut() else {
        return if stochastic {f64::from(rng.next_f64()<p)} else {p};
    };
    let mut draws = if exp3 {Some([node.bandits[0].sample(rng),node.bandits[1].sample(rng)])} else {None};
    let chosen = [0,1].map(|s|draws.as_ref().map_or_else(||node.ucb(s,rng),|d|d[s].action));
    let continues = chosen == [1,1];
    let raw = if continues {
        simulate(children,exp3,average,stochastic,rng,p)
    } else if chosen[1]==2 || chosen==[0,0] {
        if stochastic {f64::from(rng.next_f64()<p)} else {p}
    } else {0.};
    let update = if average && continues && !children.is_empty() {
        children[0].total/f64::from(children[0].count)
    } else {raw};
    node.count+=1;
    node.total+=raw;
    for side in 0..2 {
        node.visits[side][chosen[side]]+=1;
        node.rewards[side][chosen[side]]+=if side==0 {update} else {1.-update};
    }
    if let Some([a,b]) = draws.take() {
        node.bandits[0].observe(a,1.-update);
        node.bandits[1].observe(b,update);
    }
    raw
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(),2,"output.json");
    let budgets = [27000u32,108000,432000];
    let p = 0.91461181640625;
    let mut rows = Vec::new();
    for levels in [1,7] {
        for stochastic in [false,true] {
            for exp3 in [false,true] {
                for average in [false,true] {
                    for seed_index in 0..16u64 {
                        let seed = 2026121901 ^ seed_index.wrapping_mul(0x94d049bb133111eb);
                        let mut rng = SplitMix64::new(seed);
                        let mut nodes: Vec<_> = (0..levels).map(|_|Node::new()).collect();
                        let started = Instant::now();
                        let mut done = 0;
                        for budget in budgets {
                            while done < budget {simulate(&mut nodes,exp3,average,stochastic,&mut rng,p);done+=1;}
                            rows.push(json!({"levels":levels,"stochastic":stochastic,"exp3":exp3,"average":average,
                                "seed":seed,"iterations":budget,"milliseconds":started.elapsed().as_secs_f64()*1000.,
                                "nodes":nodes.iter().map(|n|json!({"visits":n.visits,"count":n.count,"total":n.total})).collect::<Vec<_>>()}));
                        }
                    }
                    eprintln!("levels={levels} stochastic={stochastic} exp3={exp3} average={average} complete");
                }
            }
        }
    }
    std::fs::write(&args[1],serde_json::to_vec(&json!({"payoff":p,"rows":rows})).unwrap()).unwrap();
}
