use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    preview::load_meta_pool, smmcts::SelRule, Belief, BlindSearch, Observer, ProtocolAgent,
    RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step, Outcome, PokemonSet, SearchChoice},
    dex::Dex,
    state::{Battle, MoveSlot, MoveSlots},
};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct Bound {
    lo: f64,
    hi: f64,
    all_win: bool,
}

struct Certificate<'a> {
    dex: &'a Dex,
    memo: BTreeMap<(u128, u8), Bound>,
    runs: usize,
    nodes: usize,
    cutoffs: usize,
}

impl Certificate<'_> {
    fn value(&mut self, b: &Battle, depth: u8) -> Bound {
        if let Some(outcome) = b.outcome() {
            let v = match outcome {
                Outcome::P1Win => 1.0,
                Outcome::P2Win => 0.0,
                Outcome::Tie => 0.5,
            };
            return Bound {
                lo: v,
                hi: v,
                all_win: outcome == Outcome::P1Win,
            };
        }
        if depth == 0 {
            self.cutoffs += 1;
            return Bound {
                lo: 0.0,
                hi: 1.0,
                all_win: false,
            };
        }
        let key = (b.state_key128(), depth);
        if let Some(&bound) = self.memo.get(&key) {
            return bound;
        }
        self.nodes += 1;
        let needs = b.needs_choice();
        let mut copy = b.clone();
        let own = if needs[0] {
            let enemy = b.active_id(1).unwrap();
            let key = if self.dex.species.key(b.poke(enemy).species) == "gengar" {
                "snore"
            } else {
                "nightshade"
            };
            Some(action(&mut copy, self.dex, 0, &format!("move {key}")))
        } else {
            None
        };
        let replies = if needs[1] {
            copy.legal_choices(self.dex, 1)
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>()
        } else {
            vec![None]
        };
        assert!(!replies.is_empty());
        let mut best = Bound {
            lo: 1.0,
            hi: 1.0,
            all_win: true,
        };
        for reply in replies {
            let step =
                enumerate_step(self.dex, b, [own, reply], 4096).expect("chance enumeration cap");
            self.runs += step.runs;
            assert!(self.runs <= 100000, "proof work budget");
            assert!((step.leaves.iter().map(|l| l.prob).sum::<f64>() - 1.0).abs() < 1e-12);
            let mut value = Bound {
                lo: 0.0,
                hi: 0.0,
                all_win: true,
            };
            for leaf in step.leaves {
                let v = self.value(&leaf.battle, depth - 1);
                value.lo += leaf.prob * v.lo;
                value.hi += leaf.prob * v.hi;
                value.all_win &= v.all_win;
            }
            best.lo = best.lo.min(value.lo);
            best.hi = best.hi.min(value.hi);
            best.all_win &= value.all_win;
        }
        self.memo.insert(key, best);
        best
    }
}

fn prove(dex: &Dex, b: &Battle) -> serde_json::Value {
    let mut control = Certificate {
        dex,
        memo: BTreeMap::new(),
        runs: 0,
        nodes: 0,
        cutoffs: 0,
    };
    let unknown = control.value(b, 0);
    assert_eq!((unknown.lo, unknown.hi, unknown.all_win), (0.0, 1.0, false));
    for (winner, expected) in [(Some(0), 1.0), (Some(1), 0.0), (None, 0.5)] {
        let mut end = b.clone();
        end.win(winner);
        let value = control.value(&end, 0);
        assert_eq!((value.lo, value.hi), (expected, expected));
    }
    let mut cert = Certificate {
        dex,
        memo: BTreeMap::new(),
        runs: 0,
        nodes: 0,
        cutoffs: 0,
    };
    let wait = cert.value(b, 14);
    let mut copy = b.clone();
    let attack = action(&mut copy, dex, 0, "move nightshade");
    let replies = copy.legal_choices(dex, 1);
    let mut upper: f64 = 1.0;
    let mut rows = Vec::new();
    for reply in replies {
        let step = enumerate_step(dex, b, [Some(attack), Some(reply)], 4096).unwrap();
        assert!((step.leaves.iter().map(|l| l.prob).sum::<f64>() - 1.0).abs() < 1e-12);
        let loss = step
            .leaves
            .iter()
            .filter(|l| l.battle.outcome() == Some(Outcome::P2Win))
            .map(|l| l.prob)
            .sum::<f64>();
        upper = upper.min(1.0 - loss);
        rows.push(json!({"reply":reply.to_input(dex),"immediate_loss_mass":loss,"runs":step.runs,"chance_leaves":step.leaves.len()}));
    }
    assert!(
        wait.lo - 1e-10 > upper + 1e-10,
        "bounds must strictly separate"
    );
    json!({"type":"certificate","wait_policy":"Snore while opposing Gengar remains; Night Shade against Magikarp. All legal opposing replies and engine chance branches enumerated.",
        "decision_depth":14,"wait_lower":wait.lo,"wait_upper":wait.hi,"every_leaf_wins":wait.all_win,"attack_upper":upper,"root_attack_replies":rows,
        "nodes":cert.nodes,"runs":cert.runs,"depth_cutoffs":cert.cutoffs,"rounding_margin":1e-10,
        "unknown_leaf_bounds":[0.0,1.0],"leaf_evaluation":"terminal outcomes only; no heuristic or opponent rollout", "memoization":"state_key128 and remaining depth","controls":"zero depth stays unknown; win/loss/tie terminal values checked"})
}

fn hidden_bench_controls(dex: &Dex, b: &Battle) -> serde_json::Value {
    let learnsets = nc2000_engine::validate::Learnsets::from_json(
        &std::fs::read_to_string(repo_root().join("data/learnsets-gen2.json")).unwrap(),
    )
    .unwrap();
    let all = &learnsets.species("magikarp").unwrap().moves;
    assert_eq!(all, &["bubble", "flail", "reversal", "splash", "tackle"]);
    let mut frontier = BTreeMap::from([(b.state_key128(), b.clone())]);
    let mut runs = 0;
    for _ in 0..7 {
        let mut next = BTreeMap::new();
        for mut current in frontier.into_values() {
            let joint = [
                Some(action(&mut current, dex, 0, "move snore")),
                Some(action(&mut current, dex, 1, "move destinybond")),
            ];
            let step = enumerate_step(dex, &current, joint, 4096).unwrap();
            runs += step.runs;
            assert!((step.leaves.iter().map(|l| l.prob).sum::<f64>() - 1.0).abs() < 1e-12);
            for leaf in step.leaves {
                next.insert(leaf.battle.state_key128(), leaf.battle);
            }
        }
        frontier = next;
    }
    let mut boundary = Vec::new();
    for mut current in frontier.into_values() {
        let joint = [
            Some(action(&mut current, dex, 0, "move snore")),
            Some(action(&mut current, dex, 1, "move struggle")),
        ];
        let step = enumerate_step(dex, &current, joint, 4096).unwrap();
        runs += step.runs;
        assert!((step.leaves.iter().map(|l| l.prob).sum::<f64>() - 1.0).abs() < 1e-12);
        for leaf in step.leaves {
            let mut state = leaf.battle;
            assert_eq!(state.needs_choice(), [false, true]);
            assert_eq!(state.sides[1].pokemon_left, 1);
            let replies = state.legal_choices(dex, 1);
            assert_eq!(replies.len(), 1);
            let replacement = enumerate_step(dex, &state, [None, Some(replies[0])], 4096).unwrap();
            runs += replacement.runs;
            assert!((replacement.leaves.iter().map(|l| l.prob).sum::<f64>() - 1.0).abs() < 1e-12);
            for leaf in replacement.leaves {
                let me = leaf.battle.poke(leaf.battle.active_id(0).unwrap());
                assert!(me.hp > 0);
                assert_eq!(me.boosts, [0; 7]);
                assert_eq!(me.status, b.poke(b.active_id(0).unwrap()).status);
                boundary.push(leaf.battle);
            }
        }
    }
    let min_hp = boundary
        .iter()
        .map(|s| s.poke(s.active_id(0).unwrap()).hp)
        .min()
        .unwrap();
    let base = &boundary[0];
    let own = base.active_id(0).unwrap();
    let foe = base.active_id(1).unwrap();
    assert_eq!(dex.species.key(base.poke(foe).species), "magikarp");
    let damage = base.poke(own).level as i32;
    let hits = (base.poke(foe).hp + damage - 1) / damage;
    assert_eq!(hits, 3);
    assert!(
        base.poke(own)
            .move_slots
            .iter()
            .find(|m| dex.moves.key(m.id) == "nightshade")
            .unwrap()
            .pp
            >= hits
    );
    let mut worst = 0;
    let mut cases = Vec::new();
    for key in all {
        let mut worst_for_move = 0;
        for hit in 0..hits {
            for speed_stage in [0, -1, -2] {
                let mut state = base.clone();
                state.poke_mut(own).hp = min_hp;
                state.poke_mut(own).boosts[4] = speed_stage;
                let foe_hp = base.poke(foe).hp - hit * damage;
                state.poke_mut(foe).hp = foe_hp;
                let id = dex.moves.id(key).unwrap();
                let mut slots = MoveSlots::default();
                slots.push(MoveSlot {
                    id,
                    pp: 20,
                    maxpp: 20,
                    disabled: false,
                    used: false,
                    shared: true,
                });
                state.poke_mut(foe).move_slots = slots;
                state.poke_mut(foe).base_move_slots = slots;
                let joint = [
                    Some(action(&mut state, dex, 0, "move nightshade")),
                    Some(action(&mut state, dex, 1, &format!("move {key}"))),
                ];
                let step = enumerate_step(dex, &state, joint, 4096).unwrap();
                runs += step.runs;
                assert!(runs <= 100000);
                assert!((step.leaves.iter().map(|l| l.prob).sum::<f64>() - 1.0).abs() < 1e-12);
                for leaf in step.leaves {
                    let me = leaf.battle.poke(own);
                    assert_eq!(leaf.battle.poke(foe).hp, (foe_hp - damage).max(0));
                    assert_eq!(me.status, base.poke(own).status);
                    for stat in [0, 1, 2, 3, 5, 6] {
                        assert_eq!(me.boosts[stat], base.poke(own).boosts[stat]);
                    }
                    assert!(me.boosts[4] >= speed_stage - 1);
                    worst_for_move = worst_for_move.max(min_hp - me.hp);
                }
            }
        }
        worst = worst.max(worst_for_move);
        cases.push(json!({"move":key,"maximum_incoming_damage":worst_for_move}));
    }
    assert!(min_hp - hits * worst > 0);
    json!({"type":"hidden_bench_controls","all_format_legal_moves":all,"minimum_hp_after_gengar":min_hp,
        "night_shade_damage":damage,"hits_to_ko":hits,"maximum_damage_per_reply":worst,
        "hp_lower_bound_after_three_replies":min_hp-hits*worst,"runs":runs,"move_cases":cases,
        "proof":"Every learnable opposing move at each of the three target HP values and each reachable pre-action Speed stage is enumerated. Only own Speed may change, by at most one stage per reply. Three worst-case hits cannot KO the user; every Night Shade removes 55 HP. Thus any unknown move subset, including event moves, loses.",
        "scope":"Conservative HP bound over hidden bench moves. Abstract HP/Speed states are proof controls, separate from the legally played primary fixture."})
}

fn action(b: &mut Battle, dex: &Dex, side: usize, key: &str) -> SearchChoice {
    b.legal_choices(dex, side)
        .into_iter()
        .find(|a| a.to_input(dex) == key)
        .unwrap()
}

fn step(b: &mut Battle, dex: &Dex, inputs: &mut Vec<[Option<String>; 2]>, own: &str, foe: &str) {
    inputs.push([Some(own.into()), Some(foe.into())]);
    let joint = [Some(action(b, dex, 0, own)), Some(action(b, dex, 1, foe))];
    b.apply_choices(dex, joint).unwrap();
}

fn switch_to(b: &mut Battle, dex: &Dex, side: usize, species: &str) -> SearchChoice {
    b.legal_choices(dex, side)
        .into_iter()
        .find(|c| {
            let SearchChoice::Switch(pos) = c else {
                return false;
            };
            dex.species.key(
                b.sides[side].roster[b.sides[side].party[(*pos - 1) as usize] as usize].species,
            ) == species
        })
        .unwrap()
}

fn forced_switch(
    b: &mut Battle,
    dex: &Dex,
    inputs: &mut Vec<[Option<String>; 2]>,
    side: usize,
    species: &str,
) {
    let pick = switch_to(b, dex, side, species);
    let mut joint = [None, None];
    joint[side] = Some(pick);
    inputs.push(joint.map(|a| a.map(|a| a.to_input(dex))));
    assert_eq!(
        b.needs_choice(),
        if side == 0 {
            [true, false]
        } else {
            [false, true]
        }
    );
    b.apply_choices(dex, joint).unwrap();
}

fn harmless_reply(b: &Battle, dex: &Dex) -> String {
    let foe = b.poke(b.active_id(1).unwrap());
    let key = foe
        .move_slots
        .iter()
        .find(|m| m.pp > 0 && dex.moves.key(m.id) != "destinybond")
        .unwrap();
    format!("move {}", dex.moves.key(key.id))
}

fn main() {
    let dex = load_dex();
    let set = |species: &str, level: u8, moves: &[&str]| -> PokemonSet {
        serde_json::from_value(
            json!({"species":species,"name":species,"level":level,"moves":moves,
            "item":"","ability":"No Ability","happiness":255,
            "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
            "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}}),
        )
        .unwrap()
    };
    let mut own = [
        set(
            "Gengar",
            55,
            &["Mean Look", "Return", "Snore", "Night Shade"],
        ),
        set("Electrode", 50, &["Explosion"]),
        set("Voltorb", 50, &["Explosion"]),
        set("Magikarp", 50, &["Splash"]),
        set("Pikachu", 50, &["Thunder Shock"]),
        set("Abra", 50, &["Teleport"]),
    ];
    own[0].evs.as_mut().unwrap().insert("spe".into(), 0);
    let foe = [
        set("Electrode", 50, &["Explosion"]),
        set(
            "Gengar",
            50,
            &["Mean Look", "Sleep Talk", "Snore", "Destiny Bond"],
        ),
        set("Magikarp", 50, &["Splash", "Tackle", "Flail"]),
        set("Pikachu", 50, &["Thunder Shock"]),
        set("Abra", 50, &["Teleport"]),
        set("Rattata", 50, &["Tackle"]),
    ];
    let learnsets = nc2000_engine::validate::Learnsets::from_json(
        &std::fs::read_to_string(repo_root().join("data/learnsets-gen2.json")).unwrap(),
    )
    .unwrap();
    for p in own.iter().chain(foe.iter()) {
        let ls = learnsets
            .species(&nc2000_engine::dex::toid(&p.species))
            .unwrap();
        for m in &p.moves {
            assert!(
                ls.allows(&nc2000_engine::dex::toid(m)),
                "{} cannot learn {}",
                p.species,
                m
            );
        }
    }
    for team in [&own, &foe] {
        let validation = nc2000_engine::validate::validate_team(
            &dex,
            &learnsets,
            &serde_json::to_string(team).unwrap(),
        );
        assert_eq!(validation["ok"], true, "{validation}");
    }
    let preview = Battle::from_fixture(&dex, "1,2,3,4", &own, &foe).unwrap();
    let mut b = preview.clone();
    let mut inputs = vec![[Some("team 1,2,3".into()), Some("team 1,2,3".into())]];
    b.choose(&dex, 0, "team 1,2,3").unwrap();
    b.choose(&dex, 1, "team 1,2,3").unwrap();
    step(&mut b, &dex, &mut inputs, "move snore", "move explosion");
    forced_switch(&mut b, &dex, &mut inputs, 1, "magikarp");
    step(&mut b, &dex, &mut inputs, "move snore", "move splash");
    let switch = switch_to(&mut b, &dex, 1, "gengar").to_input(&dex);
    step(&mut b, &dex, &mut inputs, "move snore", &switch);
    let switch = switch_to(&mut b, &dex, 0, "electrode").to_input(&dex);
    step(&mut b, &dex, &mut inputs, &switch, "move snore");
    step(&mut b, &dex, &mut inputs, "move explosion", "move snore");
    forced_switch(&mut b, &dex, &mut inputs, 0, "voltorb");
    step(&mut b, &dex, &mut inputs, "move explosion", "move snore");
    forced_switch(&mut b, &dex, &mut inputs, 0, "gengar");
    step(&mut b, &dex, &mut inputs, "move snore", "move destinybond");
    for (move_key, count) in [("nightshade", 3), ("meanlook", 8), ("return", 32)] {
        for _ in 0..count {
            let reply = harmless_reply(&b, &dex);
            step(
                &mut b,
                &dex,
                &mut inputs,
                &format!("move {move_key}"),
                &reply,
            );
        }
    }
    while b
        .poke(b.active_id(1).unwrap())
        .move_slots
        .iter()
        .any(|m| m.pp > 0 && dex.moves.key(m.id) != "destinybond")
    {
        let reply = harmless_reply(&b, &dex);
        step(&mut b, &dex, &mut inputs, "move snore", &reply);
    }
    let enemy = b.poke(b.active_id(1).unwrap());
    assert_eq!(enemy.hp, 1);
    assert!(enemy.trapped);
    for m in &enemy.move_slots {
        assert_eq!(
            m.pp,
            if dex.moves.key(m.id) == "destinybond" {
                7
            } else {
                0
            }
        );
    }
    assert!(!enemy.has_volatile(dex.conds_id("destinybond").unwrap()));
    assert_eq!(b.sides[0].pokemon_left, 1);
    assert_eq!(b.sides[1].pokemon_left, 2);
    println!(
        "{}",
        json!({"type":"fixture","teams":[own.to_vec(),foe.to_vec()],"log":b.log,"inputs":inputs,"rng":b.prng.seed_str(),"validated_six_mon_teams":true,
        "sides":(0..2).map(|s|{let p=b.poke(b.active_id(s).unwrap());json!({"hp":p.hp,"maxhp":p.maxhp,"trapped":p.trapped,"speed":p.speed,"moves":p.move_slots.iter().map(|m|json!({"move":dex.moves.key(m.id),"pp":m.pp})).collect::<Vec<_>>()})}).collect::<Vec<_>>(),
        "mask":nc2000_bot::smmcts::dominated_actions(&b,&dex,0).iter().map(|(a,r)|json!([a.to_input(&dex),r])).collect::<Vec<_>>() })
    );
    if std::env::args().any(|a| a == "--fixture") {
        return;
    }
    println!("{}", prove(&dex, &b));
    println!("{}", hidden_bench_controls(&dex, &b));
    if std::env::args().any(|a| a == "--proof") {
        return;
    }
    let pool = load_meta_pool(&repo_root().join("data/belief-pool-v3/belief-pool.json"));
    for blind in [false, true] {
        let mut obs = Observer::new(&preview, 0);
        let mut belief = if blind {
            Belief::new(&dex, &pool, &obs)
        } else {
            Belief::pinned_from_battle(&preview, &obs)
        };
        obs.observe(&b, &dex);
        belief.sync(&dex, &obs);
        for seed in 99001..99009 {
            let mut search = BlindSearch::new(
                &b,
                &dex,
                RmConfig {
                    iterations: 27000,
                    c: 0.4,
                    rule: SelRule::Ucb,
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
                json!({"type":"search","blind":blind,"seed":seed,"best":search.best().unwrap().to_input(&dex),"raw":search.actions()[raw].to_input(&dex),"actions":search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(&dex),"visits":search.visits()[i],"mean":search.means()[i],"masked":search.dominated()[i]})).collect::<Vec<_>>() })
            );
        }
    }
    for seed in 99001..99009 {
        let mut agent = ProtocolAgent::new(
            &dex,
            0,
            pool.clone(),
            RmConfig {
                iterations: 27000,
                c: 0.4,
                rule: SelRule::Ucb,
                ..Default::default()
            },
            seed,
        );
        agent.set_own_team(own.to_vec());
        let frame = nc2000_bot::player::PlayerChannel::new(0)
            .frame(&mut b.clone(), &dex)
            .unwrap();
        for line in frame.lines {
            agent.push_line(&dex, &line);
        }
        assert!(agent.on_request(&dex, &frame.request.to_string()).unwrap());
        agent.step(&dex, 27000).unwrap();
        let best = agent.best(&dex).unwrap();
        let search = agent.search().unwrap();
        let raw = (0..search.actions().len())
            .max_by_key(|&i| search.visits()[i])
            .unwrap();
        println!(
            "{}",
            json!({"type":"protocol_search","seed":seed,"best":best,"raw":search.actions()[raw].to_input(&dex),
            "actions":search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(&dex),"visits":search.visits()[i],"mean":search.means()[i],"masked":search.dominated()[i]})).collect::<Vec<_>>() })
        );
    }
}
