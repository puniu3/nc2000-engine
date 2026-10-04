use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    smmcts::{dominated_actions_with, MaskRules, SelRule},
    Belief, BlindSearch, Observer, ProtocolAgent, RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step, Outcome, SearchChoice},
    prng::{BattleRng, Prng},
};
use serde_json::json;

#[path = "own_recovery_support/mod.rs"]
mod fixture;

fn main() {
    let dex = load_dex();
    let rules = MaskRules {
        own_recovery_wait: !std::env::args().any(|a| a == "--baseline"),
        ..Default::default()
    };
    let (preview, b, teams) = fixture::position(&dex);
    println!(
        "{}",
        json!({"type":"fixture","teams":teams,"log":b.log,
        "hp":[b.poke(b.active_id(0).unwrap()).hp,b.poke(b.active_id(1).unwrap()).hp],
        "mask":dominated_actions_with(&b,&dex,0,rules).iter().map(|(a,r)|json!([a.to_input(&dex),r])).collect::<Vec<_>>()})
    );
    if std::env::args().any(|s| s == "--fixture") {
        return;
    }
    let mut probe = b.clone();
    for a in probe.legal_choices(&dex, 0) {
        for reply in probe.legal_choices(&dex, 1) {
            let out = enumerate_step(&dex, &b, [Some(a), Some(reply)], 100000).unwrap();
            let loss: f64 = out
                .leaves
                .iter()
                .filter(|l| l.battle.outcome() == Some(Outcome::P2Win))
                .map(|l| l.prob)
                .sum();
            println!(
                "{}",
                json!({"type":"first_turn","action":a.to_input(&dex),"reply":reply.to_input(&dex),"loss":loss,"runs":out.runs})
            );
        }
    }
    for first in probe.legal_choices(&dex, 0) {
        let mut counts = [0usize; 4];
        for seed in 92001..93025 {
            let mut play = b.clone();
            play.set_log_enabled(false);
            play.prng = BattleRng::seeded(Prng::new(seed));
            let mut recovered = false;
            for turn in 0..100 {
                let me = play.poke(play.active_id(0).unwrap());
                recovered |= me.hp == me.maxhp;
                let own = if turn == 0 {
                    first
                } else {
                    SearchChoice::Move(
                        dex.moves
                            .id(if recovered { "return" } else { "thunderwave" })
                            .unwrap(),
                    )
                };
                let legal = play.legal_choices(&dex, 0);
                let own = if legal.contains(&own) { own } else { legal[0] };
                let replies = play.legal_choices(&dex, 1);
                let counter = SearchChoice::Move(dex.moves.id("counter").unwrap());
                let foe = if replies.contains(&counter) {
                    counter
                } else {
                    replies[0]
                };
                play.apply_choices(&dex, [Some(own), Some(foe)]).unwrap();
                if play.outcome().is_some() {
                    break;
                }
            }
            counts[match play.outcome() {
                Some(Outcome::P1Win) => 0,
                Some(Outcome::P2Win) => 1,
                Some(Outcome::Tie) => 2,
                None => 3,
            }] += 1;
        }
        println!(
            "{}",
            json!({"type":"continuation","first":first.to_input(&dex),"seed_range":[92001,93025],"win_loss_tie_unresolved":counts,
            "policy":"After the first move, wait with Thunder Wave until full HP, then Return. Opponent always Counter. Exhausted moves fall back to first legal action. Cap 100 turns."})
        );
    }
    let mut control = b.clone();
    let own = control.active_id(0).unwrap();
    control.poke_mut(own).hp = control.poke(own).maxhp;
    for a in control.legal_choices(&dex, 0) {
        let reply = SearchChoice::Move(dex.moves.id("counter").unwrap());
        let out = enumerate_step(&dex, &control, [Some(a), Some(reply)], 100000).unwrap();
        let loss: f64 = out
            .leaves
            .iter()
            .filter(|l| l.battle.outcome() == Some(Outcome::P2Win))
            .map(|l| l.prob)
            .sum();
        assert_eq!(loss, 0.0);
        println!(
            "{}",
            json!({"type":"control","case":"own_full_hp","action":a.to_input(&dex),"loss":loss})
        );
    }
    for blind in [false, true] {
        let mut obs = Observer::new(&preview, 0);
        let mut belief = if blind {
            Belief::new(
                &dex,
                &nc2000_bot::preview::load_meta_pool(
                    &repo_root().join("data/belief-pool-v3/belief-pool.json"),
                ),
                &obs,
            )
        } else {
            Belief::pinned_from_battle(&preview, &obs)
        };
        obs.observe(&b, &dex);
        belief.sync(&dex, &obs);
        for seed in 91001..91009 {
            let mut search = BlindSearch::new(
                &b,
                &dex,
                RmConfig {
                    c: 0.4,
                    rule: SelRule::Ucb,
                    mask_rules: rules,
                    ..Default::default()
                },
                0,
                seed,
            );
            search.step(&dex, &belief, &obs, 27000);
            let raw = (0..search.actions().len())
                .max_by_key(|&i| search.visits()[i])
                .unwrap();
            println!(
                "{}",
                json!({"type":"search","blind":blind,"seed":seed,"choice":search.best().unwrap().to_input(&dex),"raw":search.actions()[raw].to_input(&dex),
                "actions":search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(&dex),"visits":search.visits()[i],"mean":search.means()[i],"excluded":search.dominated()[i]})).collect::<Vec<_>>()})
            );
        }
    }
    for seed in 94001..94009 {
        let pool = nc2000_bot::preview::load_meta_pool(
            &repo_root().join("data/belief-pool-v3/belief-pool.json"),
        );
        let mut agent = ProtocolAgent::new(
            &dex,
            0,
            pool,
            RmConfig {
                c: 0.4,
                rule: SelRule::Ucb,
                mask_rules: rules,
                ..Default::default()
            },
            seed,
        );
        agent.set_own_team(teams[0].clone());
        let frame = nc2000_bot::player::PlayerChannel::new(0)
            .frame(&mut b.clone(), &dex)
            .unwrap();
        for line in frame.lines {
            agent.push_line(&dex, &line);
        }
        assert!(agent.on_request(&dex, &frame.request.to_string()).unwrap());
        agent.step(&dex, 27000).unwrap();
        let choice = agent.best(&dex).unwrap();
        let search = agent.search().unwrap();
        let raw = (0..search.actions().len())
            .max_by_key(|&i| search.visits()[i])
            .unwrap();
        println!(
            "{}",
            json!({"type":"protocol_search","seed":seed,"choice":choice,"raw":search.actions()[raw].to_input(&dex),
            "actions":search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(&dex),"visits":search.visits()[i],"mean":search.means()[i],"excluded":search.dominated()[i]})).collect::<Vec<_>>()})
        );
    }
}
