use super::*;

#[derive(Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    game:usize, decision:usize, output_game:usize,
    side:usize, original_slot:u8, replacement_slot:u8,
    extra_steps:Vec<[Option<String>;2]>,
}

fn record(dex:&Dex,b:&mut Battle,choices:[Option<SearchChoice>;2])->Value {
    let legal=[0,1].map(|side|b.legal_choices(dex,side));
    let indices=[0,1].map(|side|match choices[side] {
        Some(action)=>Some(legal[side].iter().position(|&a|a==action).expect("fork action must be legal")),
        None=>{assert!(legal[side].is_empty());None},
    });
    json!({"key":b.state_key128().to_string(),"choices":indices})
}

pub(super) fn execute(args:&[String]) {
    assert_eq!(args.len(),5,"corpus fork-corpus config output-directory");
    let raw=std::fs::read(&args[1]).unwrap();let config_raw=std::fs::read(&args[3]).unwrap();
    let cfg:Config=serde_json::from_slice(&config_raw).unwrap();assert!(cfg.side<2 && cfg.decision>0);
    let mut output=std::str::from_utf8(&raw).unwrap().lines().map(|line|serde_json::from_str::<Value>(line).unwrap())
        .find(|game|game["game"].as_u64().unwrap() as usize==cfg.game).unwrap();
    let game:Game=serde_json::from_value(output.clone()).unwrap();let dex=conformance::load_dex();
    let mut original=Battle::from_fixture(&dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();original.set_log_enabled(true);
    let mut fork=original.clone();let mut decisions=Vec::new();
    for (index,step) in game.decisions[..cfg.decision].iter().enumerate() {
        assert_eq!(original.state_key128().to_string(),step.key);
        let legal=[0,1].map(|side|original.legal_choices(&dex,side));
        let choices=[0,1].map(|side|step.choices[side].map(|i|legal[side][i]));
        let mut altered=choices;
        if index==0 {
            let Some(SearchChoice::Team(mut slots))=altered[cfg.side] else {panic!("fork requires preview")};
            let selected=slots.iter().position(|&s|s==cfg.original_slot+1).unwrap();
            assert!(cfg.original_slot==cfg.replacement_slot || !slots.contains(&(cfg.replacement_slot+1)));
            slots[selected]=cfg.replacement_slot+1;altered[cfg.side]=Some(SearchChoice::Team(slots));
        }
        decisions.push(record(&dex,&mut fork,altered));
        original.apply_choices(&dex,choices).unwrap();fork.apply_choices(&dex,altered).unwrap();
        assert_eq!(fork.log,original.log,"fork changed prefix public history");
    }
    assert_eq!(original.state_key128().to_string(),game.decisions[cfg.decision].key);
    for state in [&original,&fork] {for slot in [cfg.original_slot,cfg.replacement_slot] {
        let mon=&state.sides[cfg.side].roster[slot as usize];
        assert!(!mon.is_active && mon.previously_switched_in==0 && !mon.fainted && mon.hp==mon.maxhp);
    }}
    let fork_root=fork.state_key128().to_string();
    let common_log=sha(&serde_json::to_vec(&fork.log).unwrap());
    for step in &cfg.extra_steps {
        let choices=[0,1].map(|side|step[side].as_ref().map(|name|fork.legal_choices(&dex,side).into_iter()
            .find(|&action|nc2000_bot::player::action_input(&dex,action)==*name).expect("unknown fork action")));
        decisions.push(record(&dex,&mut fork,choices));fork.apply_choices(&dex,choices).unwrap();
    }
    let decision=decisions.len();assert!(fork.outcome().is_none());
    let choices=[0,1].map(|side|fork.legal_choices(&dex,side).first().copied());
    decisions.push(record(&dex,&mut fork,choices));
    output["game"]=json!(cfg.output_game);output["decisions"]=json!(decisions);output["samples"]=json!([decision]);
    output["outcome"]=Value::Null;output["prefix_only"]=json!(true);
    output["interpretation"]=json!("Counterfactual legal prefix. Final decision choices are placeholders for reconstruction, not measured bot choices; no terminal outcome is claimed.");
    let corpus=serde_json::to_string(&output).unwrap()+"\n";
    let replayed:Game=serde_json::from_value(output).unwrap();let rebuilt=reconstruct(&dex,&replayed,decision);
    assert_eq!(rebuilt.state_key128(),fork.state_key128());assert_eq!(rebuilt.essence(&dex),fork.essence(&dex));
    let evidence=json!({"schema":"nc2000-hidden-selection-fork-v1","config":cfg,"config_sha256":sha(&config_raw),
        "original_state_key":original.state_key128().to_string(),"fork_root_key":fork_root,
        "common_public_log_sha256":common_log,"state_key":fork.state_key128().to_string(),"state":fork.essence(&dex),
        "output_decision":decision,"output_corpus_sha256":sha(corpus.as_bytes()),"manifest":manifest(&dex,&raw),
        "generator_sha256":sha(&std::fs::read("crates/bot/examples/fixed_reply_counterpolicy/fork_corpus.rs").unwrap())});
    let directory=std::path::Path::new(&args[4]);std::fs::create_dir(directory).unwrap();
    std::fs::write(directory.join("corpus.jsonl"),corpus).unwrap();
    std::fs::write(directory.join("fork.json"),serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
}
