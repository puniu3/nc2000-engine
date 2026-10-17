//! A constructive, finite-horizon pure-win certificate, independent of matrix solving.
//! Opponent actions are unrestricted; unresolved chance and horizon tails are losses
//! in the lower bound. The policy observes only foe species and its own past actions.
use nc2000_engine::{
    battle::{enumerate::enumerate_step_partial, Outcome, PokemonSet, SearchChoice},
    dex::Dex,
    state::Battle,
};
use serde_json::{json, Value};
use std::{collections::HashMap, io::Write, time::Instant};

#[allow(dead_code)]
mod view {
    include!("theoretical_headroom.rs");
    pub fn state(b: &Battle, d: &Dex) -> Value {
        state_view_json(b, d)
    }
}

struct Proof<'a> {
    dex: &'a Dex,
    cache: HashMap<(u128, u8, u8), [f64; 2]>,
    runs: usize,
    cap: usize,
    truncated: usize,
    conservative_discards: usize,
    initial_hp: i32,
    out: std::fs::File,
}
impl Proof<'_> {
    fn value(&mut self, b: &Battle, remaining: u8, depth: u8) -> [f64; 2] {
        if let Some(o) = b.outcome() {
            let w = if o == Outcome::P1Win { 1. } else { 0. };
            return [w, w];
        }
        // Count these branches as losses, without asserting that they really lose.
        // This is a deliberately conservative pure-win lower certificate.
        let foe_gengar = b.active_id(1).is_some_and(|id| {
            !b.poke(id).fainted && self.dex.species.key(b.poke(id).species) == "gengar"
        });
        if b.active_id(0).is_some_and(|id| {
            b.poke(id).status != nc2000_engine::state::Status::None
                || (foe_gengar && b.poke(id).hp < self.initial_hp)
        }) || (remaining == 0 && foe_gengar && b.sides[1].pokemon_left == 2)
        {
            self.conservative_discards += 1;
            return [0., 1.];
        }
        if depth == 0 || self.runs >= self.cap {
            self.truncated += 1;
            return [0., 1.];
        }
        let key = (b.state_key128(), remaining, depth);
        if let Some(v) = self.cache.get(&key) {
            return *v;
        }
        let mut legal_battle = b.clone();
        let legal = [0, 1].map(|s| legal_battle.legal_choices(self.dex, s));
        let eq = SearchChoice::Move(self.dex.moves.id("earthquake").unwrap());
        let curse = SearchChoice::Move(self.dex.moves.id("curse").unwrap());
        let gengar = b.active_id(1).is_some_and(|id| {
            !b.poke(id).fainted && self.dex.species.key(b.poke(id).species) == "gengar"
        });
        let own: Vec<(Option<SearchChoice>, f64, u8)> = if legal[0].is_empty() {
            vec![(None, 1., remaining)]
        } else if remaining > 1 && gengar && legal[0].contains(&eq) && legal[0].contains(&curse) {
            vec![
                (Some(eq), 1. / remaining as f64, 0),
                (Some(curse), 1. - 1. / remaining as f64, remaining - 1),
            ]
        } else {
            vec![(
                Some(if legal[0].contains(&eq) {
                    eq
                } else {
                    legal[0][0]
                }),
                1.,
                0,
            )]
        };
        let replies: Vec<Option<SearchChoice>> = if legal[1].is_empty() {
            vec![None]
        } else {
            legal[1].iter().copied().map(Some).collect()
        };
        let mut value = [1f64, 1f64];
        let mut rows = Vec::new();
        for reply in replies {
            let mut row = [0., 0.];
            let mut cells = Vec::new();
            for &(action, weight, next) in &own {
                let e = enumerate_step_partial(self.dex, b, [action, reply], 4096);
                self.runs += e.runs;
                let pending: f64 = e.pending.iter().map(|p| p.mass).sum();
                let mut cell = [0., pending];
                let mut children: HashMap<u128, (f64, Battle)> = HashMap::new();
                for leaf in e.leaves {
                    let k = leaf.battle.state_key128();
                    children
                        .entry(k)
                        .and_modify(|x| x.0 += leaf.prob)
                        .or_insert((leaf.prob, leaf.battle));
                }
                let mut children: Vec<_> = children.into_iter().collect();
                children.sort_by_key(|x| x.0);
                let mut edges = Vec::new();
                for (child_key, (mass, child)) in children {
                    let v = self.value(&child, next, depth - 1);
                    for k in 0..2 {
                        cell[k] += mass * v[k];
                    }
                    edges.push(json!({"key":child_key.to_string(),"remaining":next,"depth":depth-1,"prob":mass,"win_bounds":v,"outcome":child.outcome().map(|o|format!("{o:?}"))}));
                }
                for k in 0..2 {
                    row[k] += weight * cell[k];
                }
                cells.push(json!({"action":action.map(|a|a.to_input(self.dex)),"weight":weight,"pending":pending,"children":edges,"win_bounds":cell}));
            }
            for k in 0..2 {
                value[k] = value[k].min(row[k]);
            }
            rows.push(
                json!({"reply":reply.map(|a|a.to_input(self.dex)),"cells":cells,"win_bounds":row}),
            );
        }
        self.cache.insert(key, value);
        writeln!(self.out,"{}",json!({"key":key.0.to_string(),"remaining":remaining,"depth":depth,"policy":own.iter().map(|(a,p,n)|json!({"action":a.map(|a|a.to_input(self.dex)),"prob":p,"next_remaining":n})).collect::<Vec<_>>(),"replies":rows,"win_bounds":value})).unwrap();
        value
    }
}
fn upper_against_thunderbolt(
    dex: &Dex,
    b: &Battle,
    depth: u8,
    first: bool,
    runs: &mut usize,
    nodes: &mut Vec<Value>,
) -> f64 {
    if let Some(o) = b.outcome() {
        return if o == Outcome::P1Win { 1. } else { 0. };
    }
    if depth == 0 {
        return 1.;
    }
    let mut c = b.clone();
    let legal = [0, 1].map(|s| c.legal_choices(dex, s));
    let thunderbolt = SearchChoice::Move(dex.moves.id("thunderbolt").unwrap());
    let curse = SearchChoice::Move(dex.moves.id("curse").unwrap());
    let reply = if legal[1].is_empty() {
        None
    } else {
        Some(if legal[1].contains(&thunderbolt) {
            thunderbolt
        } else {
            legal[1][0]
        })
    };
    let actions = if first {
        assert!(legal[0].contains(&curse));
        vec![Some(curse)]
    } else if legal[0].is_empty() {
        vec![None]
    } else {
        legal[0].iter().copied().map(Some).collect()
    };
    let mut upper = 0f64;
    let mut cells = Vec::new();
    for action in actions {
        let e = enumerate_step_partial(dex, b, [action, reply], 4096);
        *runs += e.runs;
        let pending: f64 = e.pending.iter().map(|p| p.mass).sum();
        let mut cell = pending;
        let mut edges = Vec::new();
        for leaf in e.leaves {
            let v = upper_against_thunderbolt(dex, &leaf.battle, depth - 1, false, runs, nodes);
            cell += leaf.prob * v;
            edges.push(json!({"key":leaf.battle.state_key128().to_string(),"prob":leaf.prob,"win_upper":v,"outcome":leaf.battle.outcome().map(|o|format!("{o:?}"))}));
        }
        upper = upper.max(cell);
        cells.push(json!({"action":action.map(|a|a.to_input(dex)),"reply":reply.map(|a|a.to_input(dex)),"pending":pending,"children":edges,"win_upper":cell}));
    }
    nodes.push(json!({"key":b.state_key128().to_string(),"depth":depth,"first":first,"cells":cells,"win_upper":upper}));
    upper
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 4, "fixture output-directory depth");
    let input: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let out = std::path::Path::new(&args[2]);
    std::fs::create_dir(out).unwrap();
    let dex = conformance::load_dex();
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(input["teams"].clone()).unwrap();
    let mut b =
        Battle::from_fixture(&dex, input["seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
    let mut trace = Vec::new();
    let decisions = input["decisions"].as_array().unwrap();
    for d in &decisions[..decisions.len() - 1] {
        assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
        let actions = [0, 1].map(|s| {
            d["choices"][s]
                .as_u64()
                .map(|i| b.legal_choices(&dex, s)[i as usize])
        });
        trace.push(json!({"state":view::state(&b,&dex),"seed":b.prng.seed_str(),"actions":actions.map(|a|a.map(|a|a.to_input(&dex)))}));
        b.apply_choices(&dex, actions).unwrap();
    }
    assert_eq!(
        b.state_key128().to_string(),
        decisions.last().unwrap()["key"].as_str().unwrap()
    );
    let mut upper_runs = 0;
    let mut upper_nodes = Vec::new();
    let upper = upper_against_thunderbolt(&dex, &b, 3, true, &mut upper_runs, &mut upper_nodes);
    std::fs::write(out.join("curse-upper.json"),serde_json::to_string_pretty(&json!({"pure_win_upper":upper,"runs":upper_runs,"nodes":upper_nodes,"policy":"Opponent uses Thunderbolt whenever legal, otherwise its first legal action; own first move Curse, every future own action free; unfinished tails count as wins."})).unwrap()).unwrap();
    std::fs::write(
        out.join("replay.json"),
        serde_json::to_string_pretty(
            &json!({"trace":trace,"root":view::state(&b,&dex),"seed":b.prng.seed_str()}),
        )
        .unwrap(),
    )
    .unwrap();
    // Independent seeded trajectories exercise the deployed WASM engine later.
    let mut parity = std::fs::File::create(out.join("parity.jsonl")).unwrap();
    for trial in 0..768u64 {
        let seed = 910001 + trial * 7919;
        let mut c = b.clone();
        c.reseed(seed);
        let mut remaining = (trial % 8 + 1) as u8;
        let opponent = (trial / 8) % 6;
        let mut records = Vec::new();
        for decision in 0..24 {
            if c.outcome().is_some() {
                break;
            }
            let legal = [0, 1].map(|s| c.legal_choices(&dex, s));
            let eq = SearchChoice::Move(dex.moves.id("earthquake").unwrap());
            let curse = SearchChoice::Move(dex.moves.id("curse").unwrap());
            let gengar = c.active_id(1).is_some_and(|id| {
                !c.poke(id).fainted && dex.species.key(c.poke(id).species) == "gengar"
            });
            let a = if legal[0].is_empty() {
                None
            } else if gengar && remaining > 1 && legal[0].contains(&curse) {
                remaining -= 1;
                Some(curse)
            } else {
                remaining = 0;
                Some(if legal[0].contains(&eq) {
                    eq
                } else {
                    legal[0][0]
                })
            };
            let wanted = match opponent {
                0 => "move thunderbolt",
                1 => "move destinybond",
                2 => "switch 3",
                3 if decision % 2 == 0 => "move destinybond",
                3 => "move thunderbolt",
                4 if decision < 6 => "move destinybond",
                4 => "move thunderbolt",
                _ => "",
            };
            let reply = if legal[1].is_empty() {
                None
            } else if opponent == 5 {
                Some(legal[1][(trial as usize + decision) % legal[1].len()])
            } else {
                Some(
                    *legal[1]
                        .iter()
                        .find(|a| a.to_input(&dex) == wanted)
                        .unwrap_or(&legal[1][0]),
                )
            };
            let actions = [a, reply];
            records.push(json!({"state":view::state(&c,&dex),"seed":c.prng.seed_str(),"actions":actions.map(|a|a.map(|a|a.to_input(&dex)))}));
            c.apply_choices(&dex, actions).unwrap();
        }
        writeln!(parity,"{}",json!({"trial":trial,"chance_seed":seed,"opponent_template":opponent,"deadline":trial%8+1,"records":records,"final":view::state(&c,&dex),"final_seed":c.prng.seed_str()})).unwrap();
    }
    let start = Instant::now();
    let mut proof = Proof {
        dex: &dex,
        cache: HashMap::new(),
        runs: 0,
        cap: 3_000_000,
        truncated: 0,
        conservative_discards: 0,
        initial_hp: b.poke(b.active_id(0).unwrap()).hp,
        out: std::fs::File::create(out.join("nodes.jsonl")).unwrap(),
    };
    let value = proof.value(&b, 8, args[3].parse().unwrap());
    let result = json!({"pure_win_bounds":value,"runs":proof.runs,"nodes":proof.cache.len(),"truncated":proof.truncated,"conservative_discards":proof.conservative_discards,"seconds":start.elapsed().as_secs_f64(),"horizon":args[3],"policy":"Uniform attack deadline 1..8 while facing Gengar; attack immediately on a different foe; after the first attack keep attacking. Only own history and public foe species are used.","numerical_safety_margin":1e-8});
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(&result).unwrap(),
    )
    .unwrap();
    println!("{result}");
}
