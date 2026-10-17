use nc2000_bot::{compiled_preview::key, preview::load_meta_pool};
use nc2000_engine::{battle::SearchChoice, state::Battle};
use std::path::Path;
#[test]
fn preview_identity_has_no_hidden_opponent_fields_and_keeps_own_order() {
 let dex=conformance::load_dex();
 let pool=load_meta_pool(Path::new("../../data/team-pool-v2/team-pool.json"));
 let mut b=Battle::from_fixture(&dex,"1,2,3,4",&pool.teams[0].sets,&pool.teams[1].sets).unwrap();
 let original=key(&b,0).unwrap();
 b.sides[1].roster[0].item=None;
 b.sides[1].roster[0].base_move_slots=b.sides[1].roster[1].base_move_slots.clone();
 b.sides[1].roster[0].set_ivs=[0;6];
 b.sides[1].roster[0].happiness=0;
 assert_eq!(original,key(&b,0).unwrap());
 b.sides[1].roster[0].level-=1; assert_ne!(original,key(&b,0).unwrap());
 b.sides[1].roster[0].level+=1;
 b.sides[0].roster.swap(1,2); assert_ne!(original,key(&b,0).unwrap());
 b.sides[0].roster.swap(1,2);
 b.sides[0].roster[0].set_ivs[0]-=1;assert_ne!(original,key(&b,0).unwrap());
 b.sides[0].roster[0].set_ivs[0]+=1;
 let picks=[0,1].map(|side|b.legal_choices(&dex,side).into_iter().find(|c|matches!(c,SearchChoice::Team(_))));
 b.apply_choices(&dex,picks).unwrap();assert!(key(&b,0).is_none());
 assert!(key(&b,2).is_none());
}
