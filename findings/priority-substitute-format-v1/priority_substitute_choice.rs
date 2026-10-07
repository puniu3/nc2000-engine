use conformance::load_dex;
use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{battle::{PokemonSet, SearchChoice}, dex::Dex, state::Battle};
use serde_json::{json, Value};

fn set(species: &str, moves: &[&str], defense: u8, special_defense: u8) -> PokemonSet {
    serde_json::from_value(json!({"name":species,"species":species,"level":50,
        "gender":if species=="Electrode" {"N"} else {"M"},"moves":moves,"evs":{"hp":255,"atk":255,"def":defense,
        "spa":special_defense,"spd":special_defense,"spe":255},
        "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}})).unwrap()
}

fn step(b: &mut Battle, dex: &Dex, side: usize, own: &str, foe: &str) {
    let mut joint = [None; 2];
    for (s, input) in [(side, own), (1-side, foe)] {
        joint[s] = Some(*b.legal_choices(dex, s).iter().find(|a| a.to_input(dex)==input).unwrap_or_else(|| panic!("illegal prefix action {input} at turn {}",b.turn)));
    }
    b.apply_choices(dex, joint).unwrap();
}

fn pp(b: &Battle, dex: &Dex, side: usize, key: &str) -> i32 {
    b.poke(b.active_id(side).unwrap()).move_slots.iter().find(|m| dex.moves.key(m.id)==key).unwrap().pp
}

fn switch_to(b: &mut Battle, dex: &Dex, side: usize, species: &str) -> String {
    b.legal_choices(dex,side).into_iter().find(|choice|match choice {
        SearchChoice::Switch(pos)=>dex.species.key(b.sides[side].roster[b.sides[side].party[(*pos-1) as usize] as usize].species)==species,
        _=>false,
    }).unwrap().to_input(dex)
}

fn position(dex: &Dex, side: usize) -> (Battle, Battle) {
    let own = vec![set("Electrode", &["Explosion"],255,255),
        set("Gengar", &["Haze", "Dream Eater", "Nightmare", "Explosion"],255,255),
        set("Jolteon", &["Substitute", "Thunderbolt", "Agility"],0,255),
        set("Machamp", &["Karate Chop"],255,255),
        set("Pidgeot", &["Wing Attack"],255,255),set("Lapras", &["Surf"],255,255)];
    let foe = vec![set("Magikarp", &["Splash"],255,255),set("Golem", &["Explosion"],255,255),
        set("Scizor", &["Quick Attack", "Swords Dance", "Protect"],255,0),
        set("Venusaur", &["Razor Leaf"],255,255),set("Arcanine", &["Flamethrower"],255,255),
        set("Espeon", &["Psychic"],255,255)];
    let teams = if side==0 {[own,foe]} else {[foe,own]};
    let learnsets=nc2000_engine::validate::Learnsets::from_json(include_str!("../../data/learnsets-gen2.json")).unwrap();
    for team in &teams {
        let verdict=nc2000_engine::validate::validate_team(dex,&learnsets,&serde_json::to_string(team).unwrap());
        assert_eq!(verdict["ok"],true,"{verdict}");
    }
    let preview = Battle::from_fixture(dex,"1,2,3,4",&teams[0],&teams[1]).unwrap();
    let mut b = preview.clone();
    b.choose(dex,side,"team 1,2,3").unwrap();b.choose(dex,1-side,"team 1,2,3").unwrap();
    assert!(b.sides.iter().all(|s|s.roster.len()==6 && s.party.len()==3 && s.pokemon_left==3));
    step(&mut b,dex,side,"move explosion","move splash");
    assert!(b.sides.iter().all(|s|s.pokemon_left==2));
    let own=switch_to(&mut b,dex,side,"gengar");let foe=switch_to(&mut b,dex,1-side,"golem");
    step(&mut b,dex,side,&own,&foe);
    step(&mut b,dex,side,"move haze","move explosion");
    assert_eq!(b.sides[1-side].pokemon_left,1);
    let foe=switch_to(&mut b,dex,1-side,"scizor");
    let mut joint=[None;2];
    joint[1-side]=Some(*b.legal_choices(dex,1-side).iter().find(|a|a.to_input(dex)==foe).unwrap());
    assert!(b.legal_choices(dex,side).is_empty());b.apply_choices(dex,joint).unwrap();
    while pp(&b,dex,1-side,"protect")>1 {step(&mut b,dex,side,"move haze","move protect");}
    while pp(&b,dex,1-side,"quickattack")>2 {
        let own = if pp(&b,dex,side,"haze")>0 {"move haze"} else {"move dreameater"};
        step(&mut b,dex,side,own,"move quickattack");
    }
    let input=switch_to(&mut b,dex,side,"jolteon");
    step(&mut b,dex,side,&input,"move swordsdance");
    while pp(&b,dex,1-side,"swordsdance")>2 {
        step(&mut b,dex,side,"move agility","move swordsdance");
    }
    let input=switch_to(&mut b,dex,side,"gengar");
    step(&mut b,dex,side,&input,"move swordsdance");
    step(&mut b,dex,side,"move explosion","move protect");
    let cs = b.legal_choices(dex,side);assert_eq!(cs.len(),1);
    assert!(b.legal_choices(dex,1-side).is_empty());
    let mut joint = [None;2];joint[side]=Some(cs[0]);b.apply_choices(dex,joint).unwrap();
    step(&mut b,dex,side,"move substitute","move swordsdance");
    let me=b.active_id(side).unwrap();let foe=b.active_id(1-side).unwrap();
    assert!(b.sides.iter().all(|s|s.pokemon_left==1));
    assert!(b.poke(me).has_volatile(dex.conds_id("substitute").unwrap()));
    assert_eq!(pp(&b,dex,1-side,"quickattack"),2);
    assert_eq!(pp(&b,dex,1-side,"swordsdance"),0);assert_eq!(pp(&b,dex,1-side,"protect"),0);
    assert_eq!(b.poke(foe).hp,b.poke(foe).maxhp);
    assert!(b.get_pokemon_action_speed(dex,me)>b.get_pokemon_action_speed(dex,foe));
    assert_eq!(b.legal_choices(dex,1-side),vec![SearchChoice::Move(dex.moves.id("quickattack").unwrap())]);
    (preview,b)
}

fn main() {
    let dex=load_dex();
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),2);
    let p:Value=serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    for side in 0..2 {
        let (preview,b)=position(&dex,side);
        let own=b.poke(b.active_id(side).unwrap());let foe=b.poke(b.active_id(1-side).unwrap());
        println!("{}",json!({"kind":"fixture","side":side,"turn":b.turn,"key":b.state_key128().to_string(),
            "team_validation_ok":true,"registered":[6,6],"picked":[3,3],"own_hp":own.hp,"own_maxhp":own.maxhp,"foe_hp":foe.hp,"foe_boosts":format!("{:?}",foe.boosts),
            "log":b.log}));
        let mut obs=Observer::new(&preview,side);
        let mut belief=Belief::pinned_from_battle(&preview,&obs);
        obs.observe(&b,&dex);belief.sync_checked(&dex,&obs).unwrap();
        for seed in p["budgets"]["search_seeds"].as_array().unwrap() {
            let mut search=BlindSearch::new(&b,&dex,RmConfig {c:0.4,..Default::default()},side,seed.as_u64().unwrap());
            search.step(&dex,&belief,&obs,p["budgets"]["search_iterations"].as_u64().unwrap() as u32);
            let sub=SearchChoice::Move(dex.moves.id("substitute").unwrap());
            let old=(0..search.actions().len()).filter(|&i|search.actions()[i]!=sub && !search.dominated()[i])
                .max_by_key(|&i|search.visits()[i]).unwrap();
            println!("{}",json!({"kind":"search","side":side,"seed":seed,"actions":search.actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),
                "visits":search.visits(),"means":search.means(),"mask":search.dominated(),"before_choice":search.actions()[old].to_input(&dex),
                "after_choice":search.best().unwrap().to_input(&dex),"nodes":search.node_count()}));
        }
    }
}
