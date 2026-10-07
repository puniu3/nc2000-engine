#[allow(dead_code)]
mod study {
    include!("hp_information_duel.rs");
    use nc2000_bot::eval::{eval01, expected_hit_fraction, EvalWeights};
    use std::io::Write;
    use std::sync::Mutex;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Study {
        schema: String,
        fixture: Protocol,
        repeats: usize,
        samples: usize,
        interpretation: String,
    }

    fn mon(dex: &Dex, p: &nc2000_engine::state::Pokemon) -> Value {
        json!({"species":dex.species.key(p.species),"moves":move_keys(dex,p),"item":p.item.map(|id|dex.items.key(id)),"hp_type":dex.type_name(p.base_hp_type),"hp_power":p.base_hp_power,"stats":p.base_stored_stats,"hp":p.hp,"maxhp":p.maxhp})
    }

    fn estimate(dex: &Dex, b: &Battle, side: usize) -> Value {
        let att = b.active_id(1 - side).unwrap();
        let def = b.active_id(side).unwrap();
        let moves=b.poke(att).move_slots.iter().map(|s|json!({"move":dex.moves.key(s.id),"damage":expected_hit_fraction(b,dex,att,def,s.id,true)*f64::from(b.poke(def).hp)})).collect::<Vec<_>>();
        let score = eval01(b, dex, &EvalWeights::default());
        json!({"foe":mon(dex,b.poke(att)),"own":mon(dex,b.poke(def)),"foe_damage":moves,"static_eval":if side==0 {score}else{1.0-score}})
    }

    pub fn run() {
        assert!(cfg!(feature = "hp-type-evidence"));
        let args = std::env::args().collect::<Vec<_>>();
        assert!(
            (3..=4).contains(&args.len()),
            "hp_corrected_evidence_roots PROTOCOL OUTPUT.jsonl [--evidence]"
        );
        let evidence = args.get(3).is_some_and(|a| a == "--evidence");
        assert!(args.len() == 3 || evidence);
        let study: Study = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
        assert_eq!(study.schema, "nc2000-hp-corrected-evidence-roots-v1");
        assert!(study.repeats > 0 && study.samples > 0 && !study.interpretation.is_empty());
        let p = &study.fixture;
        assert_eq!(hash(Path::new(&p.prior)), p.prior_sha256);
        let dex =
            Dex::from_json(&std::fs::read_to_string("data/gen2stadium2.json").unwrap()).unwrap();
        let learnsets =
            Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
                .unwrap();
        for case in &p.cases {
            for team in &case.teams {
                assert_eq!(
                    validate_team(&dex, &learnsets, &serde_json::to_string(team).unwrap())["ok"],
                    true
                );
            }
        }
        let pool = load_meta_pool(Path::new(&p.prior));
        let out = Mutex::new(std::io::BufWriter::new(
            std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&args[2])
                .unwrap(),
        ));
        let manifest = json!({"manifest":true,"evidence":evidence,"protocol_sha256":hash(Path::new(&args[1])),"sources":sources(),"probe_sha256":hash(Path::new("crates/bot/examples/hp_corrected_evidence_roots.rs")),"binary_sha256":hash(&std::env::current_exe().unwrap())});
        writeln!(out.lock().unwrap(), "{manifest}").unwrap();
        let jobs = p
            .cases
            .iter()
            .enumerate()
            .flat_map(|(ci, c)| {
                (0..c.pairs)
                    .flat_map(move |trial| (0..study.repeats).map(move |rep| (ci, trial, rep)))
            })
            .collect::<Vec<_>>();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..p.threads {
                scope.spawn(||loop {
                    let job=next.fetch_add(1,Ordering::Relaxed);
                    let Some(&(ci,trial,rep))=jobs.get(job) else {break};
                    let (mut b,battle_seed,agents)=initial(&dex,p,ci,trial);
                    let side=trial%2;
                    let mut obs=Observer::new(&b,side);
                    let mut belief=Belief::new(&dex,&pool,&obs);
                    let mut history=Vec::new();
                    for prefix in &p.cases[ci].prefix {
                        obs.observe(&b,&dex);
                        let view=if evidence {obs.clone()}else{obs.without_hp_type_evidence()};
                        belief.sync(&dex,&view);
                        let choices=[0,1].map(|s|Some(*b.legal_choices(&dex,s).iter().find(|a|a.to_input(&dex)==prefix[s^side]).unwrap()));
                        history.push(json!({"key":b.state_key128().to_string(),"choices":choices.map(|a|a.unwrap().to_input(&dex))}));
                        b.apply_choices(&dex,choices).unwrap();
                    }
                    obs.observe(&b,&dex);
                        let view=if evidence {obs.clone()}else{obs.without_hp_type_evidence()};
                        belief.sync(&dex,&view);
                    assert!(b.outcome().is_none() && b.legal_choices(&dex,side).len()>1);
                    let search_seed=SplitMix64::new(agents[side]^(rep as u64).wrapping_mul(0x94d049bb133111eb)).next();
                    let mut models=std::collections::BTreeMap::<String,usize>::new();
                    for seed in 0..study.samples {
                        let sample=belief.determinize(&dex,&b,&view,&mut SplitMix64::new(seed as u64));
                        let record=json!({"active":estimate(&dex,&sample,side),"roster":sample.sides[1-side].roster.iter().map(|m|mon(&dex,m)).collect::<Vec<_>>(),"party":sample.sides[1-side].party.to_vec()});
                        *models.entry(serde_json::to_string(&record).unwrap()).or_default()+=1;
                    }
                    let started=Instant::now();
                    let mut search=BlindSearch::new(&b,&dex,RmConfig{iterations:p.iterations,c:p.c,..Default::default()},side,search_seed);
                    search.step(&dex,&belief,&view,p.iterations);
                    let means=search.means();
                    let actions=search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(&dex),"visits":search.visits()[i],"mean":means[i],"dominated":search.dominated()[i]})).collect::<Vec<_>>();
                    assert_eq!(search.visits().iter().sum::<u32>(),p.iterations);
                    let row=json!({"case":p.cases[ci].id,"trial":trial,"rep":rep,"side":side,"battle_seed":battle_seed,"search_seed":search_seed,"history":history,"key":b.state_key128().to_string(),"observation":observation(&view,&dex),"raw_observation":observation(&obs,&dex),"active_slot":b.sides[1-side].active,"truth":estimate(&dex,&b,side),"models":models,"fallback":belief.is_fallback(),"iterations":search.iterations(),"best":search.best().unwrap().to_input(&dex),"actions":actions,"seconds":started.elapsed().as_secs_f64()});
                    let mut out=out.lock().unwrap();writeln!(out,"{row}").unwrap();out.flush().unwrap();
                    eprintln!("{}/{} {} trial={trial} rep={rep}",job+1,jobs.len(),p.cases[ci].id);
                });
            }
        });
    }
}

fn main() {
    study::run();
}
