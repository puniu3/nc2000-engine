//! PoC verification & benchmark for the Turn-Economy ("Golf") evaluator (`tempo.rs`).
//!
//! Modes:
//!   - Default (`scenarios`): Runs canonical Gen 2 / NC2000 simple scenarios to test
//!     whether the single-currency turn model cleanly explains advantage/disadvantage
//!     and selects sensible 1-ply equilibrium actions.
//!   - `duel [--games N] [--vs greedy|skuct:300|skuct:3000]`: Runs a paired arena
//!     duel of `TempoAgent` vs baseline bots and prints a single-line JSON summary.

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::duel::{run_duel, DuelSpec};
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmAgent, RmConfig, SelRule};
use nc2000_bot::tempo::{
    analyze_turn_matrix, evaluate_tempo, format_tempo_explanation, TempoAgent,
};
use nc2000_bot::{Agent, MaxDamageAgent};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, EffectState, Status, DK};
use serde_json::json;

fn mk(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(json!({
        "name": species,
        "species": species,
        "item": item,
        "ability": "No Ability",
        "moves": moves,
        "level": level,
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
        "happiness": 255,
        "gender": "M"
    }))
    .unwrap()
}

fn start_battle(dex: &Dex, p1: &[PokemonSet], p2: &[PokemonSet]) -> Battle {
    let mut b = Battle::from_fixture(dex, "1,2,3,4", p1, p2).unwrap();
    let pick1 = match p1.len() {
        1 => "team 1",
        2 => "team 1,2",
        _ => "team 1,2,3",
    };
    let pick2 = match p2.len() {
        1 => "team 1",
        2 => "team 1,2",
        _ => "team 1,2,3",
    };
    b.choose(dex, 0, pick1).unwrap();
    b.choose(dex, 1, pick2).unwrap();
    b
}

fn print_decision(dex: &Dex, b: &Battle, side: usize, samples: usize) {
    let mut b_tmp = b.clone();
    let cs = b_tmp.legal_choices(dex, side);
    let m = analyze_turn_matrix(b, dex, side, &cs, samples, 42);
    let opp_names: Vec<String> = m.opp_choices.iter().map(|c| c.to_input(dex)).collect();
    println!(
        "  P{} 1-ply Eq: WinProb={:.1}%, ΔG={:+.2}t | Opp choices: {:?}",
        side + 1,
        m.eq_win_prob * 100.0,
        m.eq_turn_adv,
        opp_names
    );
    for (i, c) in m.own_choices.iter().enumerate() {
        let cells: Vec<String> = (0..m.opp_choices.len())
            .map(|j| {
                format!(
                    "{:+.2}t({:.0}%)",
                    m.turn_matrix[i * m.opp_choices.len() + j],
                    m.win_matrix[i * m.opp_choices.len() + j] * 100.0
                )
            })
            .collect();
        println!(
            "    {:>18} [p={:4.0}%]: {}",
            c.to_input(dex),
            m.own_policy[i] * 100.0,
            cells.join(" | ")
        );
    }
}

fn run_scenarios(dex: &Dex) {
    // =========================================================================
    // Case 1: Matchup & Speed (±0.5t) — Mirror with Lv55 vs Lv50 Zapdos
    // =========================================================================
    println!("=== Case 1: Speed & Pure Race (Lv55 Zapdos vs Lv50 Zapdos) ===");
    {
        let p1 = vec![mk("Zapdos", 55, "Leftovers", &["Thunderbolt", "Hidden Power Ice"])];
        let p2 = vec![mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice"])];
        let b = start_battle(dex, &p1, &p2);
        let ev = evaluate_tempo(&b, dex);
        print!("{}", format_tempo_explanation(&b, dex, &ev));
        // Symmetry check: equal Lv50 mirror must be ΔG = 0.00t, WinProb = 50.0%
        let b_eq = start_battle(dex, &p2, &p2);
        let ev_eq = evaluate_tempo(&b_eq, dex);
        println!(
            "  Mirror sanity (Lv50 vs Lv50): G=({:.2}t, {:.2}t), ΔG={:+.2}t, WinProb={:.1}%\n",
            ev_eq.g_turns[0],
            ev_eq.g_turns[1],
            ev_eq.turn_advantage,
            ev_eq.win_prob * 100.0
        );
    }

    // =========================================================================
    // Case 2: Attack vs Recover (35% Wall vs ~50% Even vs 75% Breaker)
    // =========================================================================
    println!("=== Case 2: Attack vs 50% Recover (Starmie Recover vs Weak / ~50% / Strong) ===");
    {
        // 2a: Weak attacker (<50%): Vaporeon Surf (~25%) vs Starmie Recover/Psychic
        let p1_weak = vec![mk("Vaporeon", 50, "", &["Surf"])];
        let p2_starmie = vec![mk("Starmie", 50, "", &["Psychic", "Recover"])];
        let b_weak = start_battle(dex, &p1_weak, &p2_starmie);
        let ev_weak = evaluate_tempo(&b_weak, dex);
        println!("[2a: ~20-25% dmg vs Recover]");
        print!("{}", format_tempo_explanation(&b_weak, dex, &ev_weak));

        // 2b: ~50% Double-Edge / Body Slam vs Starmie Recover (at 50% HP -> must Recover = 1-for-1!)
        let p1_half = vec![mk("Snorlax", 50, "", &["Double-Edge"])];
        let mut b_half = start_battle(dex, &p1_half, &p2_starmie);
        if let Some(id) = b_half.active_id(1) {
            let p = b_half.poke_mut(id);
            p.hp = p.maxhp / 2;
        }
        let ev_half = evaluate_tempo(&b_half, dex);
        println!("[2b: ~50% Double-Edge vs 50%-HP Starmie Recover]");
        print!("{}", format_tempo_explanation(&b_half, dex, &ev_half));
        print_decision(dex, &b_half, 1, 24);

        // 2c: Breaker (>65%): Zapdos Thunderbolt (~85-100%) vs Starmie Recover
        let p1_strong = vec![mk("Zapdos", 50, "", &["Thunderbolt"])];
        let b_strong = start_battle(dex, &p1_strong, &p2_starmie);
        let ev_strong = evaluate_tempo(&b_strong, dex);
        println!("[2c: >80% Thunderbolt vs Starmie Recover]");
        print!("{}", format_tempo_explanation(&b_strong, dex, &ev_strong));
        println!();
    }

    // =========================================================================
    // Case 3: Asymmetric Trade (1/3 dealt vs 1/2 taken) -> Staying vs Switching
    // =========================================================================
    println!("=== Case 3: 1/3 Dealt vs 1/2 Taken (Stay vs Switch to Counter) ===");
    {
        // P1: Snorlax (Body Slam ~35% vs Machamp) + Zapdos (Drill Peck ~60% vs Machamp)
        // P2: Machamp (Cross Chop ~65% vs Snorlax) + Raikou (Snorlax EQ handles Raikou)
        let p1 = vec![
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Drill Peck", "Thunderbolt"]),
        ];
        let p2 = vec![
            mk("Machamp", 50, "Leftovers", &["Cross Chop", "Rock Slide"]),
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Rest"]),
        ];
        let b = start_battle(dex, &p1, &p2);
        let ev = evaluate_tempo(&b, dex);
        print!("{}", format_tempo_explanation(&b, dex, &ev));
        // Also compare exact 1/3 dealt vs 1/2 taken HP state against baseline!
        let mut b_traded = b.clone();
        if let Some(id0) = b_traded.active_id(0) {
            let p = b_traded.poke_mut(id0);
            p.hp = p.maxhp / 2; // took 1/2
        }
        if let Some(id1) = b_traded.active_id(1) {
            let p = b_traded.poke_mut(id1);
            p.hp = (p.maxhp * 2) / 3; // dealt 1/3
        }
        let ev_traded = evaluate_tempo(&b_traded, dex);
        println!(
            "  After 1/3 dealt vs 1/2 taken: ΔG {:+.2}t -> {:+.2}t (net {:+.2}t)",
            ev.turn_advantage,
            ev_traded.turn_advantage,
            ev_traded.turn_advantage - ev.turn_advantage
        );
        print_decision(dex, &b, 0, 24);
        println!();
    }

    // =========================================================================
    // Case 4: Setup (積み) — Investing 1 turn to shrink multi-mon TTK
    // =========================================================================
    println!("=== Case 4: Setup (Marowak Swords Dance: Safe vs Doomed) ===");
    {
        // 4a: Safe setup — Marowak (Thick Club) vs passive Raikou/Snorlax (takes ~40% or less, sweeps!)
        let p1 = vec![
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance"]),
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake"]),
        ];
        let p2 = vec![
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice"]),
        ];
        let b = start_battle(dex, &p1, &p2);
        let ev_pre = evaluate_tempo(&b, dex);
        println!("[4a: Before Swords Dance]");
        print!("{}", format_tempo_explanation(&b, dex, &ev_pre));
        // Simulate 1 turn: Marowak uses Swords Dance, Snorlax uses Body Slam
        let mut b_after = b.clone();
        b_after.choose(dex, 0, "move swordsdance").unwrap();
        b_after.choose(dex, 1, "move bodyslam").unwrap();
        let ev_post = evaluate_tempo(&b_after, dex);
        println!("[4a: After 1 turn (SD vs Body Slam)]");
        print!("{}", format_tempo_explanation(&b_after, dex, &ev_post));
        print_decision(dex, &b, 0, 24);

        // 4b: Doomed setup — Marowak facing Starmie Surf (OHKO'd before recouping!)
        let p2_starmie = vec![
            mk("Starmie", 50, "Leftovers", &["Surf", "Recover"]),
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Rest"]),
        ];
        let b_doom = start_battle(dex, &p1, &p2_starmie);
        println!("[4b: Marowak facing faster OHKO Starmie (must NOT SD!)]");
        print_decision(dex, &b_doom, 0, 24);
        println!();
    }

    // =========================================================================
    // Case 5: Phazing (積み流し) — Skarmory Whirlwind erasing +2 Snorlax
    // =========================================================================
    println!("=== Case 5: Phazing (+2 Curse Snorlax vs Skarmory Whirlwind) ===");
    {
        let p1 = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice"]),
        ];
        let p2 = vec![
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest"]),
            mk("Machamp", 50, "Leftovers", &["Cross Chop", "Earthquake"]),
        ];
        let mut b = start_battle(dex, &p1, &p2);
        // Put Snorlax at +2 Atk / +2 Def (as if it cursed twice before Skarmory came in).
        if let Some(id) = b.active_id(0) {
            let p = b.poke_mut(id);
            p.boosts[0] = 2;
            p.boosts[1] = 2;
        }
        let ev = evaluate_tempo(&b, dex);
        print!("{}", format_tempo_explanation(&b, dex, &ev));
        print_decision(dex, &b, 1, 24);
        println!();
    }

    // =========================================================================
    // Case 6: 3v3 Role Preservation (役割破壊) — Losing Zapdos vs Skarmory
    // =========================================================================
    println!("=== Case 6: 3v3 Role Preservation (Zapdos alive vs fainted vs Skarmory) ===");
    {
        let p1 = vec![
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest"]),
        ];
        let p2 = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Rest"]),
        ];
        let b = start_battle(dex, &p1, &p2);
        let ev_alive = evaluate_tempo(&b, dex);
        println!("[6a: Full 3v3 — Cloyster facing Raikou (should switch to Snorlax!)]");
        print!("{}", format_tempo_explanation(&b, dex, &ev_alive));
        print_decision(dex, &b, 0, 16);

        // Now knock out P1's Zapdos (bench slot 2) -> Skarmory becomes a massive wall!
        let mut b_no_zap = b.clone();
        let zap_roster = b_no_zap.sides[0].party[2] as usize;
        b_no_zap.sides[0].roster[zap_roster].hp = 0;
        b_no_zap.sides[0].roster[zap_roster].status = Status::Fnt;
        b_no_zap.sides[0].pokemon_left = 2;
        let ev_dead = evaluate_tempo(&b_no_zap, dex);
        println!("[6b: Same board EXCEPT P1 Zapdos fainted (Role Destruction!)]");
        print!("{}", format_tempo_explanation(&b_no_zap, dex, &ev_dead));
        println!();
    }

    // =========================================================================
    // Case 7: Status & Perish Trap (Sleep / Paralysis / Spikes / Perish Song)
    // =========================================================================
    println!("=== Case 7: Status & Hazards in Turn Currency ===");
    {
        let p1 = vec![
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake"]),
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice"]),
        ];
        let p2 = vec![
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice"]),
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake"]),
        ];
        let base = start_battle(dex, &p1, &p2);
        let ev_base = evaluate_tempo(&base, dex);

        let mut b_slp = base.clone();
        if let Some(id) = b_slp.active_id(1) {
            let p = b_slp.poke_mut(id);
            p.status = Status::Slp;
            p.status_state.set_int(DK::Time, 4); // 3 sleep turns
        }
        let ev_slp = evaluate_tempo(&b_slp, dex);

        let mut b_par = base.clone();
        if let Some(id) = b_par.active_id(1) {
            b_par.poke_mut(id).status = Status::Par;
        }
        let ev_par = evaluate_tempo(&b_par, dex);

        let mut b_spk = base.clone();
        if let Some(scid) = dex.conds_id("spikes") {
            let mut st = EffectState::default();
            st.set_int(DK::Layers, 1);
            b_spk.sides[1].side_conditions.push((scid, st));
        }
        let ev_spk = evaluate_tempo(&b_spk, dex);

        println!(
            "  Baseline:           G=({:.2}t, {:.2}t) ΔG={:+.2}t (Win {:.1}%)",
            ev_base.g_turns[0], ev_base.g_turns[1], ev_base.turn_advantage, ev_base.win_prob * 100.0
        );
        println!(
            "  Foe Zapdos Asleep:  G=({:.2}t, {:.2}t) ΔG={:+.2}t (shift {:+.2}t)",
            ev_slp.g_turns[0],
            ev_slp.g_turns[1],
            ev_slp.turn_advantage,
            ev_slp.turn_advantage - ev_base.turn_advantage
        );
        println!(
            "  Foe Zapdos Par:     G=({:.2}t, {:.2}t) ΔG={:+.2}t (shift {:+.2}t)",
            ev_par.g_turns[0],
            ev_par.g_turns[1],
            ev_par.turn_advantage,
            ev_par.turn_advantage - ev_base.turn_advantage
        );
        println!(
            "  Spikes on Foe Side: G=({:.2}t, {:.2}t) ΔG={:+.2}t (shift {:+.2}t)\n",
            ev_spk.g_turns[0],
            ev_spk.g_turns[1],
            ev_spk.turn_advantage,
            ev_spk.turn_advantage - ev_base.turn_advantage
        );
    }

    // =========================================================================
    // Case 8: Explosion / Self-Destruct Simultaneous-Move Nash Mix
    // =========================================================================
    println!("=== Case 8: Snorlax Self-Destruct vs [Zapdos, Gengar] Nash Matrix ===");
    {
        let p1 = vec![
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake", "Self-Destruct"]),
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice"]),
        ];
        let p2 = vec![
            mk("Zapdos", 55, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest"]),
            mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch"]),
        ];
        let mut b = start_battle(dex, &p1, &p2);
        // Snorlax chipped down to 45% HP -> high incentive to boom Zapdos, but Gengar can switch in!
        if let Some(id) = b.active_id(0) {
            let p = b.poke_mut(id);
            p.hp = (p.maxhp * 45) / 100;
        }
        let ev = evaluate_tempo(&b, dex);
        print!("{}", format_tempo_explanation(&b, dex, &ev));
        print_decision(dex, &b, 0, 24);
    }
}

fn run_duel_bench(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let games: usize = flag("--games").map_or(40, |v| v.parse().unwrap());
    let samples: usize = flag("--samples").map_or(8, |v| v.parse().unwrap());
    let vs = flag("--vs").unwrap_or_else(|| "greedy".to_string());
    let seed: u64 = flag("--seed").map_or(20261123, |v| v.parse().unwrap());

    let root = repo_root();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let teams: Vec<Vec<PokemonSet>> = pool.teams.iter().map(|t| t.sets.clone()).collect();

    let build_a = move |s: u64| -> Box<dyn Agent> {
        Box::new(TempoAgent::with_samples(s, samples, false))
    };
    let vs_clone = vs.clone();
    let build_b = move |s: u64| -> Box<dyn Agent> {
        if vs_clone == "greedy" {
            Box::new(MaxDamageAgent::conformant())
        } else if vs_clone == "maxdamage" {
            Box::new(MaxDamageAgent::new())
        } else if let Some(rest) = vs_clone.strip_prefix("skuct:") {
            let iters: u32 = rest.parse().unwrap_or(300);
            let cfg = RmConfig {
                iterations: iters,
                c: 0.4,
                rule: SelRule::Ucb,
                ..RmConfig::default()
            };
            Box::new(RmAgent::new(cfg, s))
        } else {
            panic!("unknown --vs spec: {vs_clone}");
        }
    };

    let mut spec = DuelSpec::new(games, seed);
    spec.max_turns = 200;
    let stats = run_duel(dex, &teams, &build_a, &build_b, spec);
    println!(
        "{}",
        json!({
            "arm_a": format!("tempo:{samples}"),
            "arm_b": vs,
            "games": stats.games,
            "wins": stats.wins,
            "losses": stats.losses,
            "ties": stats.ties,
            "turn_caps": stats.turn_caps,
            "score": (stats.score * 10000.0).round() / 10000.0,
            "ci95": if stats.ci95.is_finite() { (stats.ci95 * 10000.0).round() / 10000.0 } else { 0.0 },
            "avg_turns": (stats.avg_turns * 10.0).round() / 10.0,
            "a_ms_per_move": (stats.a_ms_per_move * 100.0).round() / 100.0,
            "b_ms_per_move": (stats.b_ms_per_move * 100.0).round() / 100.0,
        })
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dex = load_dex();
    match args.first().map(String::as_str) {
        Some("duel") => run_duel_bench(&dex, &args[1..]),
        _ => run_scenarios(&dex),
    }
}
