use nc2000_engine::{battle::{Outcome, PokemonSet, SearchChoice}, dex::Dex, state::Battle};
use nc2000_engine::battle::enumerate::enumerate_step_partial;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::BufRead;

#[derive(Deserialize)]
struct Decision { key: String, choices: [Option<usize>; 2] }
#[derive(Deserialize)]
struct Game { game: usize, seed: String, teams: [Vec<PokemonSet>; 2], decisions: Vec<Decision> }
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    game: usize, decision: usize, target_side: usize, first_choice: String,
    policy_side: usize, priority: Vec<String>, rest_at_percent: Option<i64>,
    horizon: usize, work_budget: usize, cell_cap: usize,
    #[serde(default)]
    guarded_keys: bool,
}

struct Witness<'a> {
    dex: &'a Dex, cfg: Config, runs: usize,
    memo: HashMap<(bool, bool, u128, usize), usize>, nodes: Vec<Value>,
}

impl Witness<'_> {
    fn selected(&self, b: &Battle, side: usize, legal: &[Option<SearchChoice>], depth: usize) -> Vec<Option<SearchChoice>> {
        if legal == [None] { return legal.to_vec(); }
        if depth == 0 && side == self.cfg.target_side {
            let selected: Vec<_> = legal.iter().copied().filter(|a|a.unwrap().to_input(self.dex)==self.cfg.first_choice).collect();
            assert_eq!(selected.len(),1);
            return selected;
        }
        if side != self.cfg.policy_side { return legal.to_vec(); }
        let names: Vec<_> = legal.iter().map(|a|a.map(|a|a.to_input(self.dex))).collect();
        if self.cfg.rest_at_percent.is_some_and(|pct|b.active_id(side).is_some_and(|id| {
            let p=b.poke(id);i64::from(p.hp)*100<=i64::from(p.maxhp)*pct
        })) {
            if let Some(i)=names.iter().position(|a|a.as_deref()==Some("move rest")) { return vec![legal[i]]; }
        }
        for name in &self.cfg.priority {
            if let Some(i)=names.iter().position(|a|a.as_ref()==Some(name)) { return vec![legal[i]]; }
        }
        vec![legal[0]]
    }

    fn visit(&mut self, mut battle: Battle, depth: usize) -> usize {
        let key=battle.state_key128();
        let damage=self.cfg.guarded_keys && !battle.damage_bookkeeping_observable(self.dex);
        let claw=self.cfg.guarded_keys && !battle.quick_claw_roll_observable(self.dex);
        let semantic=if claw {battle.state_key128_without_quick_claw_roll(self.dex,damage)}
            else if damage {battle.state_key128_without_damage_bookkeeping()} else {key};
        let cache_key=(damage,claw,semantic,depth);
        if let Some(&id)=self.memo.get(&cache_key) { return id; }
        let terminal=battle.outcome().map(|o|match o {
            Outcome::Tie=>0.5, Outcome::P1Win=>f64::from(self.cfg.target_side==0),
            Outcome::P2Win=>f64::from(self.cfg.target_side==1),
        });
        let mut record=json!({"state_key":key.to_string(),"depth":depth,
            "cache_key":[damage,claw,semantic.to_string()]});
        if let Some(value)=terminal {
            record["kind"]=json!("terminal");record["bounds"]=json!([value,value]);
        } else if depth>=self.cfg.horizon || self.runs>=self.cfg.work_budget {
            record["kind"]=json!(if depth>=self.cfg.horizon {"horizon"} else {"budget"});
            record["bounds"]=json!([0.,1.]);
        } else {
            let legal=[0,1].map(|s| {
                if battle.needs_choice()[s] { battle.legal_choices(self.dex,s).into_iter().map(Some).collect::<Vec<_>>() }
                else { vec![None] }
            });
            let selected=[0,1].map(|s|self.selected(&battle,s,&legal[s],depth));
            let maximizing=self.cfg.policy_side!=self.cfg.target_side;
            let mut bounds=if maximizing {[0f64,0f64]} else {[1f64,1f64]};
            let mut branches=Vec::new();
            for &a in &selected[0] { for &b in &selected[1] {
                let choices=[a,b];
                let remaining=self.cfg.work_budget.saturating_sub(self.runs);
                let mut children=Vec::new();
                let mut branch=[0.,1.];
                let mut pending=1.;
                let mut executions=0;
                if remaining>0 {
                    let step=enumerate_step_partial(self.dex,&battle,choices,self.cfg.cell_cap.min(remaining));
                    executions=step.runs;self.runs+=executions;
                    pending=step.pending.iter().map(|p|p.mass).sum::<f64>();
                    branch=[0.,pending];
                    let mass=step.leaves.iter().map(|l|l.prob).sum::<f64>()+pending;
                    assert!((mass-1.).abs()<1e-8);
                    for leaf in step.leaves {
                        let child=self.visit(leaf.battle,depth+1);
                        for k in 0..2 { branch[k]+=leaf.prob*self.nodes[child]["bounds"][k].as_f64().unwrap(); }
                        children.push(json!({"mass":leaf.prob,"node":child}));
                    }
                    let pad=1e-12*(children.len()+2) as f64;
                    branch=[(branch[0]-pad).max(0.),(branch[1]+pad).min(1.)];
                }
                for k in 0..2 {
                    bounds[k]=if maximizing {bounds[k].max(branch[k])} else {bounds[k].min(branch[k])};
                }
                branches.push(json!({"choices":choices.map(|a|a.map(|a|a.to_input(self.dex))),
                    "runs":executions,"pending":pending,"children":children,"bounds":branch}));
            }}
            record["kind"]=json!("decision");record["maximizing"]=json!(maximizing);
            record["legal"]=json!(legal.map(|aa|aa.into_iter().map(|a|a.map(|a|a.to_input(self.dex))).collect::<Vec<_>>()));
            record["selected"]=json!(selected.map(|aa|aa.into_iter().map(|a|a.map(|a|a.to_input(self.dex))).collect::<Vec<_>>()));
            record["branches"]=json!(branches);record["bounds"]=json!(bounds);
        }
        let id=self.nodes.len();self.nodes.push(record);self.memo.insert(cache_key,id);id
    }
}

fn main() {
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),4,"corpus config output");
    let cfg:Config=serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    assert!(cfg.target_side<2 && cfg.policy_side<2 && cfg.horizon>0 && cfg.work_budget>0 && cfg.cell_cap>0);
    let dex=conformance::load_dex();
    let game=std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines()
        .map(|l|serde_json::from_str::<Game>(&l.unwrap()).unwrap()).find(|g|g.game==cfg.game).unwrap();
    let mut battle=Battle::from_fixture(&dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();
    for record in game.decisions.iter().take(cfg.decision) {
        assert_eq!(battle.state_key128().to_string(),record.key);
        let legal=[0,1].map(|s|battle.legal_choices(&dex,s));
        battle.apply_choices(&dex,[0,1].map(|s|record.choices[s].map(|i|legal[s][i]))).unwrap();
    }
    assert_eq!(battle.state_key128().to_string(),game.decisions[cfg.decision].key);
    let mut witness=Witness {dex:&dex,cfg,runs:0,memo:HashMap::new(),nodes:Vec::new()};
    let root=witness.visit(battle,0);
    let output=json!({"config":witness.cfg,"runs":witness.runs,"root":root,
        "bounds":witness.nodes[root]["bounds"],"nodes":witness.nodes});
    std::fs::write(&args[3],serde_json::to_vec(&output).unwrap()).unwrap();
    eprintln!("runs={} nodes={} bounds={}",witness.runs,witness.nodes.len(),output["bounds"]);
}
