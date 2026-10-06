use nc2000_bot::bounds::{BoundConfig, FixedPolicySolver};
use nc2000_engine::{
    battle::{enumerate::enumerate_step_partial, Outcome, PokemonSet, SearchChoice},
    dex::Dex,
    state::Battle,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::BufRead};

type Key = (bool, bool, String);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    game: usize,
    decision: usize,
    first_opponent: String,
    cap: usize,
    targets: Vec<Key>,
    side: usize,
    #[serde(default)]
    first: Option<String>,
    priority: Vec<String>,
    calls: usize,
    work_budget: usize,
    node_budget: usize,
}

fn key(dex: &Dex, b: &Battle) -> Key {
    let damage = !b.damage_bookkeeping_observable(dex);
    let claw = !b.quick_claw_roll_observable(dex);
    let value = if claw {
        b.state_key128_without_quick_claw_roll(dex, damage)
    } else if damage {
        b.state_key128_without_damage_bookkeeping()
    } else {
        b.state_key128()
    };
    (damage, claw, value.to_string())
}

struct Census<'a> {
    dex: &'a Dex,
    cap: usize,
    nodes: BTreeMap<Key, (Battle, Value)>,
    runs: usize,
}

impl Census<'_> {
    fn kernel(&mut self, b: &Battle, choices: [Option<SearchChoice>; 2], depth: usize) -> Value {
        let step = enumerate_step_partial(self.dex, b, choices, self.cap);
        self.runs += step.runs;
        let mut leaves = BTreeMap::<Key, f64>::new();
        for leaf in step.leaves {
            let k = self.visit(leaf.battle, depth + 1);
            *leaves.entry(k).or_default() += leaf.prob;
        }
        json!({"choices": choices.map(|a| a.map(|a| a.to_input(self.dex))),
            "runs": step.runs, "complete": !step.exhausted, "pending_scripts": step.pending.len(),
            "pending": step.pending.iter().map(|p| p.mass).sum::<f64>(),
            "leaves": leaves.into_iter().map(|(key, mass)| json!({"key":key,"mass":mass})).collect::<Vec<_>>()})
    }

    fn visit(&mut self, mut b: Battle, depth: usize) -> Key {
        assert!(depth <= 3);
        let k = key(self.dex, &b);
        if self.nodes.contains_key(&k) {
            return k;
        }
        let terminal = b.outcome().map(|o| match o {
            Outcome::Tie => 0.5,
            Outcome::P1Win => 1.,
            Outcome::P2Win => 0.,
        });
        let needs = b.needs_choice();
        let legal = [0, 1].map(|s| {
            if needs[s] {
                b.legal_choices(self.dex, s)
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>()
            } else {
                vec![None]
            }
        });
        assert_eq!(key(self.dex, &b), k);
        let mut row = json!({"key":k, "state_key":b.state_key128().to_string(), "state":b.essence(self.dex),
            "living":[b.sides[0].pokemon_left,b.sides[1].pokemon_left], "terminal":terminal,
            "legal":legal.each_ref().map(|aa| aa.iter().map(|a| a.map(|a| a.to_input(self.dex))).collect::<Vec<_>>()),
            "active":([0,1].map(|s| b.active_id(s).map(|id| json!({"species":self.dex.species.key(b.poke(id).species),
                "hp":b.poke(id).hp,"maxhp":b.poke(id).maxhp})))) });
        if terminal.is_none()
            && needs != [true, true]
            && needs != [false, false]
            && legal.iter().all(|aa| aa.len() == 1)
        {
            row["forced"] = self.kernel(&b, [legal[0][0], legal[1][0]], depth);
        }
        row["duel"] = json!(
            terminal.is_none()
                && needs == [true, true]
                && b.sides.iter().all(|s| s.pokemon_left == 1)
        );
        self.nodes.insert(k.clone(), (b, row));
        k
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus manifest output");
    let raw = std::fs::read(&args[2]).unwrap();
    let m: Manifest = serde_json::from_slice(&raw).unwrap();
    assert!(m.cap > 0 && m.side < 2 && m.work_budget > 0 && m.node_budget > 0);
    let corpus = std::fs::read(&args[1]).unwrap();
    let game = corpus
        .lines()
        .map(|line| serde_json::from_str::<Value>(&line.unwrap()).unwrap())
        .find(|g| g["game"] == m.game)
        .unwrap();
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let dex = conformance::load_dex();
    let mut b =
        Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
    for d in game["decisions"]
        .as_array()
        .unwrap()
        .iter()
        .take(m.decision)
    {
        assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
        let aa = [0, 1].map(|s| b.legal_choices(&dex, s));
        b.apply_choices(
            &dex,
            [0, 1].map(|s| d["choices"][s].as_u64().map(|i| aa[s][i as usize])),
        )
        .unwrap();
    }
    assert_eq!(
        b.state_key128().to_string(),
        game["decisions"][m.decision]["key"].as_str().unwrap()
    );
    assert_eq!(b.needs_choice(), [true, true]);
    let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
    let first = *legal[1]
        .iter()
        .find(|a| a.to_input(&dex) == m.first_opponent)
        .unwrap();
    let mut census = Census {
        dex: &dex,
        cap: m.cap,
        nodes: BTreeMap::new(),
        runs: 0,
    };
    let kernels: Vec<_> = legal[0]
        .iter()
        .map(|a| census.kernel(&b, [Some(*a), Some(first)], 0))
        .collect();
    let priority: Vec<_> = m
        .priority
        .iter()
        .map(|s| SearchChoice::Move(dex.moves.id(s.strip_prefix("move ").unwrap()).unwrap()))
        .collect();
    assert!(!priority.is_empty());
    let mut solved = Vec::new();
    for target in &m.targets {
        let (state, row) = &census.nodes[target];
        assert_eq!(row["duel"], true);
        let mut state = state.clone();
        let replies = state.legal_choices(&dex, 1 - m.side);
        let cfg = BoundConfig {
            work_budget: m.work_budget,
            node_budget: m.node_budget,
            eps: 1e-6,
            dead_quick_claw_quotient: true,
            retain_eager_prefix: true,
            aggregate_lazy_children: true,
            chance_fair_period: 8,
            ..BoundConfig::default()
        };
        let mut solver = if let Some(name) = &m.first {
            let first = state
                .legal_choices(&dex, m.side)
                .into_iter()
                .find(|a| a.to_input(&dex) == *name)
                .unwrap();
            FixedPolicySolver::new_with_first_action(
                &dex,
                cfg,
                m.side,
                priority.clone(),
                &state,
                first,
            )
        } else {
            FixedPolicySolver::new(&dex, cfg, m.side, priority.clone())
        };
        let mut rounds = Vec::new();
        for i in 0..m.calls {
            let reply = replies[i % replies.len()];
            let report = solver.solve_reply(&state, reply, None);
            let matrix = solver.restricted_root_payoffs(&state).unwrap();
            assert_eq!(matrix.actions[m.side].len(), 1);
            assert_eq!(
                matrix.actions[1 - m.side],
                replies.iter().copied().map(Some).collect::<Vec<_>>()
            );
            let bounds = if m.side == 0 {
                [matrix.value.lo, matrix.value.hi]
            } else {
                [1. - matrix.value.hi, 1. - matrix.value.lo]
            };
            let cells: Vec<_> = replies
                .iter()
                .zip(&matrix.cells)
                .map(|(a, v)| json!({"reply":a.to_input(&dex),"side_zero_bounds":[v.lo,v.hi]}))
                .collect();
            rounds.push(json!({"reply":reply.to_input(&dex),"runs":report.runs,"total_runs":solver.stats().runs,
                "fixed_side_bounds":bounds,"reply_side_zero_bounds":[report.bounds.lo,report.bounds.hi],
                "stop":format!("{:?}",report.stop),"nodes":solver.node_count(),"root_cells":cells}));
        }
        eprintln!(
            "duel {target:?} {:?}",
            rounds.last().map(|r| &r["fixed_side_bounds"])
        );
        solved.push(json!({"key":target,"rounds":rounds}));
    }
    let output = json!({"manifest_sha256":format!("{:x}",Sha256::digest(raw)),
        "corpus_sha256":format!("{:x}",Sha256::digest(corpus)),"state_key":b.state_key128().to_string(),
        "state":b.essence(&dex),"legal":legal.map(|aa|aa.into_iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>()),
        "census_runs":census.runs,"kernels":kernels,
        "nodes":census.nodes.into_values().map(|(_,row)|row).collect::<Vec<_>>(),"solved":solved});
    std::fs::write(&args[3], serde_json::to_vec(&output).unwrap()).unwrap();
}
