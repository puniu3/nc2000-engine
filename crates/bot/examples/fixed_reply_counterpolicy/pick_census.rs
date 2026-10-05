use super::*;
use nc2000_bot::{Belief,Observer};
use nc2000_bot::rng::SplitMix64;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Config {game:usize,decision:usize,side:usize,samples:usize,seed:u64}

pub(super) fn execute(args:&[String]) {
    assert_eq!(args.len(),5,"corpus pick-census config output");
    let raw=std::fs::read(&args[1]).unwrap();let cfg_raw=std::fs::read(&args[3]).unwrap();
    let cfg:Config=serde_json::from_slice(&cfg_raw).unwrap();assert!(cfg.side<2 && cfg.samples>0);
    let dex=conformance::load_dex();
    let game=std::str::from_utf8(&raw).unwrap().lines().map(|line|serde_json::from_str::<Game>(line).unwrap())
        .find(|game|game.game==cfg.game).unwrap();
    let mut b=Battle::from_fixture(&dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();b.set_log_enabled(true);
    let mut obs=Observer::new(&b,cfg.side);
    let pool_path="data/belief-pool-v3/belief-pool.json";
    let mut beliefs=[Belief::pinned_from_battle(&b,&obs),
        Belief::new(&dex,&nc2000_bot::preview::load_meta_pool(Path::new(pool_path)),&obs),
        Belief::new(&dex,&nc2000_bot::preview::MetaPool {teams:vec![]},&obs)];
    for step in &game.decisions[..cfg.decision] {
        assert_eq!(b.state_key128().to_string(),step.key);
        let legal=[0,1].map(|s|b.legal_choices(&dex,s));
        b.apply_choices(&dex,[0,1].map(|s|step.choices[s].map(|i|legal[s][i]))).unwrap();
        obs.observe(&b,&dex);for belief in &mut beliefs {belief.sync(&dex,&obs);}
    }
    let expected=reconstruct(&dex,&game,cfg.decision);
    assert_eq!(b.essence(&dex),expected.essence(&dex));assert_eq!(b.state_key128(),expected.state_key128());
    let opp=1-cfg.side;let cap=b.preview_level_caps[opp];let mut rows=Vec::new();
    for (mode,belief) in ["pinned","blind","fallback"].into_iter().zip(&beliefs) {
        let mut rng=SplitMix64::new(cfg.seed);let mut counts=BTreeMap::<Vec<u8>,usize>::new();
        for _ in 0..cfg.samples {
            let det=belief.determinize(&dex,&b,&obs,&mut rng);
            let party=det.sides[opp].party.to_vec();
            for (position,&slot) in b.sides[opp].party.iter().enumerate() {
                if obs.mons()[slot as usize].appeared {assert_eq!(party[position],slot);}
            }
            *counts.entry(party).or_default()+=1;
        }
        let selections=counts.into_iter().map(|(party,samples)| {
            let level:u32=party.iter().map(|&slot|u32::from(obs.mons()[slot as usize].level)).sum();
            json!({"party":party,"species":party.iter().map(|&s|dex.species.key(obs.mons()[s as usize].species)).collect::<Vec<_>>(),
                "total_level":level,"legal":cap.is_none_or(|limit|level<=limit),"samples":samples})
        }).collect::<Vec<_>>();
        rows.push(json!({"mode":mode,"alive":belief.alive().len(),"fallback":belief.is_fallback(),"selections":selections}));
    }
    let paths=["crates/bot/examples/fixed_reply_counterpolicy/pick_census.rs","crates/bot/src/belief.rs","crates/bot/src/pick_prior.rs","crates/bot/src/observe.rs"];
    let output=json!({"schema":"nc2000-hidden-pick-census-v1","config":cfg,"config_sha256":sha(&cfg_raw),
        "state_key":b.state_key128().to_string(),"cap":cap,"rows":rows,
        "public_roster":obs.mons().iter().map(|m|json!({"species":dex.species.key(m.species),"level":m.level,"appeared":m.appeared})).collect::<Vec<_>>(),
        "manifest":manifest(&dex,&raw),"source_hashes":paths.map(|p|(p,sha(&std::fs::read(p).unwrap()))),"pool_sha256":sha(&std::fs::read(pool_path).unwrap())});
    std::fs::write(&args[4],serde_json::to_vec_pretty(&output).unwrap()).unwrap();
}
