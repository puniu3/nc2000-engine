use super::*;
use nc2000_engine::battle::SearchChoice;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    side: usize,
    original_slot: u8,
    replacement_slot: u8,
}

pub(super) fn replay(dex:&Dex, game:&Game, decision:usize, cfg:&Config)->(Battle,Value) {
    assert!(cfg.side<2 && decision>0);
    let mut original=Battle::from_fixture(dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();
    original.set_log_enabled(true);
    let mut variant=original.clone();
    let roster_len=original.sides[cfg.side].roster.len();
    assert!((cfg.original_slot as usize)<roster_len && (cfg.replacement_slot as usize)<roster_len);
    for (index,record) in game.decisions[..decision].iter().enumerate() {
        assert_eq!(original.state_key128().to_string(),record.key);
        let legal=[0,1].map(|side|original.legal_choices(dex,side));
        let choices=[0,1].map(|side|record.choices[side].map(|i|legal[side][i]));
        let mut altered=choices;
        if index==0 {
            let Some(SearchChoice::Team(mut slots))=altered[cfg.side] else {panic!("first decision must be preview")};
            let selected=slots.iter().position(|&slot|slot==cfg.original_slot+1).unwrap();
            assert!(cfg.original_slot==cfg.replacement_slot || !slots.contains(&(cfg.replacement_slot+1)));
            slots[selected]=cfg.replacement_slot+1;
            altered[cfg.side]=Some(SearchChoice::Team(slots));
        }
        for side in 0..2 {
            let legal=variant.legal_choices(dex,side);
            if let Some(action)=altered[side] {assert!(legal.contains(&action),"illegal hidden-selection replay");}
            else {assert!(legal.is_empty());}
        }
        original.apply_choices(dex,choices).unwrap();
        variant.apply_choices(dex,altered).unwrap();
        assert_eq!(variant.log,original.log,"hidden selection changed public history");
    }
    assert_eq!(original.state_key128().to_string(),game.decisions[decision].key);
    for battle in [&original,&variant] {
        for slot in [cfg.original_slot,cfg.replacement_slot] {
            let mon=&battle.sides[cfg.side].roster[slot as usize];
            assert!(!mon.is_active && mon.previously_switched_in==0 && !mon.fainted);
            assert_eq!(mon.hp,mon.maxhp);
        }
    }
    let side=&variant.sides[cfg.side];
    let total_level:u32=side.party.iter().map(|&slot|u32::from(side.roster[slot as usize].level)).sum();
    assert!(variant.preview_level_caps[cfg.side].is_none_or(|cap|total_level<=cap));
    let evidence=json!({"side":cfg.side,"original_slot":cfg.original_slot,"replacement_slot":cfg.replacement_slot,
        "species":dex.species.key(side.roster[cfg.replacement_slot as usize].species),
        "party":side.party.as_slice(),"total_level":total_level,"cap":variant.preview_level_caps[cfg.side],
        "original_state_key":original.state_key128().to_string(),"state_key":variant.state_key128().to_string(),
        "public_log_sha256":sha(&serde_json::to_vec(&variant.log).unwrap()),"same_public_log":true,
        "state":variant.essence(dex)});
    variant.set_log_enabled(false);
    (variant,evidence)
}
