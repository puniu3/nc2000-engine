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
use nc2000_bot::preview::{action_index, load_meta_pool, preview_actions, MetaPool};
use nc2000_bot::runner::{play_game, GameResult};
use nc2000_bot::smmcts::{RmAgent, RmConfig, SelRule};
use nc2000_bot::tempo::{
    analyze_turn_matrix, analyze_turn_matrix_variant, evaluate_tempo, format_tempo_explanation,
    search_tempo_mcts, BlindCmctsAgent, BlindCmctsConfig, BlindGolfAgent, BlindGolfConfig,
    SkuctGolfAgent, SkuctGolfConfig, TempoAgent, TempoLeafMode, TempoMctsAgent, TempoMctsConfig,
    TempoVariant,
};
use nc2000_bot::{Agent, BlindAgent, MaxDamageAgent, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, EffectState, Status, DK};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

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

struct FixedPreviewWrap(Box<dyn Agent>);

impl Agent for FixedPreviewWrap {
    fn name(&self) -> String {
        self.0.name()
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[nc2000_engine::battle::SearchChoice],
    ) -> nc2000_engine::battle::SearchChoice {
        if matches!(choices.first(), Some(nc2000_engine::battle::SearchChoice::Team(_))) {
            let d = nc2000_engine::battle::SearchChoice::Team([1, 2, 3]);
            return if choices.contains(&d) { d } else { choices[0] };
        }
        self.0.choose(battle, dex, side, choices)
    }

    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[nc2000_engine::battle::SearchChoice],
    ) -> Vec<f64> {
        if matches!(choices.first(), Some(nc2000_engine::battle::SearchChoice::Team(_))) {
            let pick = self.choose(battle, dex, side, choices);
            return choices.iter().map(|&c| if c == pick { 1.0 } else { 0.0 }).collect();
        }
        self.0.root_policy(battle, dex, side, choices)
    }
}

fn parse_leaf_rollout(rest: &str) -> Option<(TempoLeafMode, u16, u32)> {
    let (leaf, turns, iters, _) = parse_leaf_rollout_adaptive(rest)?;
    Some((leaf, turns, iters))
}

fn parse_leaf_rollout_adaptive(rest: &str) -> Option<(TempoLeafMode, u16, u32, bool)> {
    // Parses e.g. `golf-0r:1000`, `hybrid-0r:a243000`, `hybrid-2r:486000`, `bs-8r:1000`
    let (head, it_s) = rest.split_once(':')?;
    let (adaptive, num_s) = if let Some(s) = it_s.strip_prefix('a') {
        (true, s)
    } else {
        (false, it_s)
    };
    let iters: u32 = num_s.parse().ok()?;
    let (leaf_s, r_s) = head.split_once('-')?;
    let leaf = match leaf_s {
        "golf" | "v2" => TempoLeafMode::FastGolf,
        "hybrid" | "hy" => TempoLeafMode::HybridGolf,
        "s3log" => TempoLeafMode::Stage3Log,
        "s3mlp" => TempoLeafMode::Stage3Mlp,
        "azv" => TempoLeafMode::AzValue,
        "cfv" => TempoLeafMode::CfValue,
        "bs" | "blindspot" => TempoLeafMode::EvalBlindspot,
        "eval" => TempoLeafMode::EvalStatic,
        _ => return None,
    };
    let turns: u16 = r_s.strip_suffix('r')?.parse().ok()?;
    Some((leaf, turns, iters, adaptive))
}

fn build_agent(spec: &str, seed: u64, fixed_preview: bool) -> Box<dyn Agent> {
    let raw: Box<dyn Agent> = if spec == "greedy" {
        Box::new(MaxDamageAgent::conformant())
    } else if spec == "maxdamage" {
        Box::new(MaxDamageAgent::new())
    } else if let Some(rest) = spec.strip_prefix("skuct:") {
        let iters: u32 = rest.parse().unwrap_or(300);
        let cfg = RmConfig {
            iterations: iters,
            c: 0.4,
            rule: SelRule::Ucb,
            ..RmConfig::default()
        };
        Box::new(RmAgent::new(cfg, seed))
    } else if let Some(rest) = spec.strip_prefix("skuct-") {
        let (leaf, rollout_turns, iterations) =
            parse_leaf_rollout(rest).unwrap_or_else(|| panic!("bad skuct-* spec: {spec}"));
        Box::new(SkuctGolfAgent::new(
            SkuctGolfConfig {
                iterations,
                c: 0.4,
                leaf,
                rollout_turns,
                fixed_preview,
            },
            seed,
        ))
    } else if let Some(rest) = spec.strip_prefix("cmcts-") {
        let (leaf, rollout_turns, iterations) =
            parse_leaf_rollout(rest).unwrap_or_else(|| panic!("bad cmcts-* spec: {spec}"));
        Box::new(TempoMctsAgent::new(
            TempoMctsConfig {
                iterations,
                c: 0.4,
                max_depth: 64,
                leaf,
                deterministic: true,
                fixed_preview,
                minimax_backup: true,
                argmax_root: true,
                rollout_turns,
            },
            seed,
        ))
    } else if let Some(rest) = spec.strip_prefix("nmcts-") {
        let (leaf, rollout_turns, iterations) =
            parse_leaf_rollout(rest).unwrap_or_else(|| panic!("bad nmcts-* spec: {spec}"));
        Box::new(TempoMctsAgent::new(
            TempoMctsConfig {
                iterations,
                c: 0.4,
                max_depth: 64,
                leaf,
                deterministic: false,
                fixed_preview,
                minimax_backup: true,
                argmax_root: false,
                rollout_turns,
            },
            seed,
        ))
    } else if let Some(rest) = spec.strip_prefix("amcts-") {
        let (leaf, rollout_turns, iterations) =
            parse_leaf_rollout(rest).unwrap_or_else(|| panic!("bad amcts-* spec: {spec}"));
        Box::new(TempoMctsAgent::new(
            TempoMctsConfig {
                iterations,
                c: 0.4,
                max_depth: 64,
                leaf,
                deterministic: true,
                fixed_preview,
                minimax_backup: false,
                argmax_root: true,
                rollout_turns,
            },
            seed,
        ))
    } else if let Some(rest) = spec.strip_prefix("1ply-full:") {
        let k: usize = rest.parse().unwrap_or(8);
        let mut a = TempoAgent::with_variant(seed, k, false, TempoVariant::Full);
        a.fixed_preview = fixed_preview;
        Box::new(a)
    } else if let Some(rest) = spec.strip_prefix("1ply-raw:") {
        let k: usize = rest.parse().unwrap_or(8);
        let mut a = TempoAgent::with_variant(seed, k, false, TempoVariant::Raw);
        a.fixed_preview = fixed_preview;
        Box::new(a)
    } else if let Some(rest) = spec.strip_prefix("mcts-full:") {
        let iters: u32 = rest.parse().unwrap_or(300);
        Box::new(TempoMctsAgent::new(
            TempoMctsConfig {
                iterations: iters,
                leaf: TempoLeafMode::TempoFull,
                fixed_preview,
                ..TempoMctsConfig::default()
            },
            seed,
        ))
    } else if let Some(rest) = spec.strip_prefix("mcts-raw:") {
        let iters: u32 = rest.parse().unwrap_or(300);
        Box::new(TempoMctsAgent::new(
            TempoMctsConfig {
                iterations: iters,
                leaf: TempoLeafMode::TempoRaw,
                fixed_preview,
                ..TempoMctsConfig::default()
            },
            seed,
        ))
    } else if let Some(rest) = spec.strip_prefix("mcts-v2:") {
        let iters: u32 = rest.parse().unwrap_or(300);
        Box::new(TempoMctsAgent::new(
            TempoMctsConfig {
                iterations: iters,
                leaf: TempoLeafMode::FastGolf,
                fixed_preview,
                ..TempoMctsConfig::default()
            },
            seed,
        ))
    } else if let Some(rest) = spec.strip_prefix("mcts-eval:") {
        let iters: u32 = rest.parse().unwrap_or(300);
        Box::new(TempoMctsAgent::new(
            TempoMctsConfig {
                iterations: iters,
                leaf: TempoLeafMode::EvalStatic,
                fixed_preview,
                ..TempoMctsConfig::default()
            },
            seed,
        ))
    } else if let Some(rest) = spec.strip_prefix("mcts-blindspot:") {
        let iters: u32 = rest.parse().unwrap_or(300);
        Box::new(TempoMctsAgent::new(
            TempoMctsConfig {
                iterations: iters,
                leaf: TempoLeafMode::EvalBlindspot,
                fixed_preview,
                ..TempoMctsConfig::default()
            },
            seed,
        ))
    } else {
        panic!("unknown agent spec: {spec}");
    };
    if fixed_preview {
        Box::new(FixedPreviewWrap(raw))
    } else {
        raw
    }
}

fn run_one_duel(
    dex: &Dex,
    teams: &[Vec<PokemonSet>],
    arm_a: &str,
    arm_b: &str,
    games: usize,
    seed: u64,
    crn: bool,
    fixed_preview: bool,
    threads: usize,
) -> serde_json::Value {
    let a_str = arm_a.to_string();
    let b_str = arm_b.to_string();
    let build_a = move |s: u64| build_agent(&a_str, s, fixed_preview);
    let build_b = move |s: u64| build_agent(&b_str, s, fixed_preview);
    let mut spec = DuelSpec::new(games, seed);
    spec.max_turns = 200;
    spec.crn_agent_seeds = crn;
    spec.threads = threads.min(10);
    let stats = run_duel(dex, teams, &build_a, &build_b, spec);
    json!({
        "arm_a": arm_a,
        "arm_b": arm_b,
        "games": stats.games,
        "wins": stats.wins,
        "losses": stats.losses,
        "ties": stats.ties,
        "turn_caps": stats.turn_caps,
        "score": (stats.score * 10000.0).round() / 10000.0,
        "ci95": if stats.ci95.is_finite() { (stats.ci95 * 10000.0).round() / 10000.0 } else { 0.0 },
        "avg_turns": (stats.avg_turns * 10.0).round() / 10.0,
        "a_ms": (stats.a_ms_per_move * 100.0).round() / 100.0,
        "b_ms": (stats.b_ms_per_move * 100.0).round() / 100.0,
    })
}

fn run_duel_bench(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let games: usize = flag("--games").map_or(60, |v| v.parse().unwrap());
    let samples: usize = flag("--samples").map_or(8, |v| v.parse().unwrap());
    let arm_a = flag("--arm-a").unwrap_or_else(|| format!("1ply-full:{samples}"));
    let arm_b = flag("--arm-b")
        .or_else(|| flag("--vs"))
        .unwrap_or_else(|| "greedy".to_string());
    let seed: u64 = flag("--seed").map_or(20261123, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let crn = has("--crn");
    let fixed_preview = has("--fixed-preview");

    let root = repo_root();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let teams: Vec<Vec<PokemonSet>> = pool.teams.iter().map(|t| t.sets.clone()).collect();
    let row = run_one_duel(dex, &teams, &arm_a, &arm_b, games, seed, crn, fixed_preview, threads);
    println!("{row}");
}

/// Probe whether deeper search fixes the 1-ply HP/boost cliffs on `Raw` without static workarounds.
fn run_cliff_probe(dex: &Dex) -> serde_json::Value {
    // Case 6a: Cloyster facing Raikou (`switch 2` = Snorlax vs dying to Thunderbolt)
    let p1_6 = vec![
        mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion"]),
        mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Rest"]),
        mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest"]),
    ];
    let p2_6 = vec![
        mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest"]),
        mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest"]),
        mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Rest"]),
    ];
    let mut b6 = start_battle(dex, &p1_6, &p2_6);
    let cs6 = b6.legal_choices(dex, 0);

    // Case 4b: Marowak facing faster Starmie (must NOT click Swords Dance)
    let p1_4 = vec![
        mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance"]),
        mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake"]),
    ];
    let p2_4 = vec![
        mk("Starmie", 50, "Leftovers", &["Surf", "Recover"]),
        mk("Snorlax", 50, "Leftovers", &["Body Slam", "Rest"]),
    ];
    let mut b4 = start_battle(dex, &p1_4, &p2_4);
    let cs4 = b4.legal_choices(dex, 0);

    let mut rows = Vec::new();
    for (v_name, variant, leaf) in [
        ("raw", TempoVariant::Raw, TempoLeafMode::TempoRaw),
        ("full", TempoVariant::Full, TempoLeafMode::TempoFull),
    ] {
        // 1-ply
        let m6 = analyze_turn_matrix_variant(&b6, dex, 0, &cs6, 24, 42, variant);
        let m4 = analyze_turn_matrix_variant(&b4, dex, 0, &cs4, 24, 42, variant);
        let sw_i = cs6.iter().position(|c| c.to_input(dex) == "switch 2").unwrap();
        let exp_i = cs6.iter().position(|c| c.to_input(dex) == "move explosion").unwrap();
        let surf_i = cs6.iter().position(|c| c.to_input(dex) == "move surf").unwrap();
        let sd_i = cs4.iter().position(|c| c.to_input(dex) == "move swordsdance").unwrap();
        let eq_i = cs4.iter().position(|c| c.to_input(dex) == "move earthquake").unwrap();
        rows.push(json!({
            "variant": v_name,
            "budget": "1ply",
            "case6_p_switch_lax": (m6.own_policy[sw_i] * 1000.0).round() / 1000.0,
            "case6_p_explosion": (m6.own_policy[exp_i] * 1000.0).round() / 1000.0,
            "case6_p_surf_stay": (m6.own_policy[surf_i] * 1000.0).round() / 1000.0,
            "case6_depth": 1.0,
            "case4b_p_sd": (m4.own_policy[sd_i] * 1000.0).round() / 1000.0,
            "case4b_p_eq": (m4.own_policy[eq_i] * 1000.0).round() / 1000.0,
        }));

        for &iters in &[100u32, 300, 900, 2700] {
            let cfg = TempoMctsConfig {
                iterations: iters,
                leaf,
                ..TempoMctsConfig::default()
            };
            let r6 = search_tempo_mcts(&b6, dex, 0, &cs6, cfg, 42);
            let r4 = search_tempo_mcts(&b4, dex, 0, &cs4, cfg, 42);
            rows.push(json!({
                "variant": v_name,
                "budget": format!("mcts:{iters}"),
                "case6_p_switch_lax": (r6.own_policy[sw_i] * 1000.0).round() / 1000.0,
                "case6_p_explosion": (r6.own_policy[exp_i] * 1000.0).round() / 1000.0,
                "case6_p_surf_stay": (r6.own_policy[surf_i] * 1000.0).round() / 1000.0,
                "case6_depth": (r6.mean_depth * 100.0).round() / 100.0,
                "case4b_p_sd": (r4.own_policy[sd_i] * 1000.0).round() / 1000.0,
                "case4b_p_eq": (r4.own_policy[eq_i] * 1000.0).round() / 1000.0,
            }));
        }
    }
    json!(rows)
}

fn run_scale_suite(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let games: usize = flag("--games").map_or(100, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026112301, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());

    let root = repo_root();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let teams: Vec<Vec<PokemonSet>> = pool.teams.iter().map(|t| t.sets.clone()).collect();

    // 1. Self-scaling ladder (fixed preview = false, same fast preview across all Tempo arms)
    let self_ladder = [
        ("mcts-full:100", "1ply-full:8"),
        ("mcts-full:300", "mcts-full:100"),
        ("mcts-full:900", "mcts-full:300"),
        ("mcts-full:2700", "mcts-full:900"),
    ];
    let mut self_rows = Vec::new();
    for (idx, &(a, b)) in self_ladder.iter().enumerate() {
        self_rows.push(run_one_duel(
            dex,
            &teams,
            a,
            b,
            games,
            seed + idx as u64,
            true,
            false,
            threads,
        ));
    }

    // 2. Fixed-anchor ladder vs skuct:300 (and skuct:1000 at 900/2700)
    let anchor_ladder = [
        ("1ply-full:8", "skuct:300"),
        ("mcts-full:100", "skuct:300"),
        ("mcts-full:300", "skuct:300"),
        ("mcts-full:900", "skuct:300"),
        ("mcts-full:2700", "skuct:300"),
        ("mcts-full:900", "skuct:1000"),
        ("mcts-full:2700", "skuct:1000"),
    ];
    let mut anchor_rows = Vec::new();
    for (idx, &(a, b)) in anchor_ladder.iter().enumerate() {
        anchor_rows.push(run_one_duel(
            dex,
            &teams,
            a,
            b,
            games,
            seed + 100 + idx as u64,
            false,
            false,
            threads,
        ));
    }

    // 3. Hypothesis 2: Raw vs Full ablation across search depths (CRN paired, fixed preview so only in-battle differs!)
    let raw_vs_full = [
        ("1ply-raw:8", "1ply-full:8"),
        ("mcts-raw:100", "mcts-full:100"),
        ("mcts-raw:300", "mcts-full:300"),
        ("mcts-raw:900", "mcts-full:900"),
        ("mcts-raw:2700", "mcts-full:2700"),
    ];
    let mut ablation_rows = Vec::new();
    for (idx, &(a, b)) in raw_vs_full.iter().enumerate() {
        ablation_rows.push(run_one_duel(
            dex,
            &teams,
            a,
            b,
            games,
            seed + 200 + idx as u64,
            true,
            true,
            threads,
        ));
    }

    // 4. Leaf control inside the exact same zero-rollout DUCT tree (TempoFull vs EvalStatic / EvalBlindspot)
    let leaf_controls = [
        ("mcts-full:300", "mcts-eval:300"),
        ("mcts-full:300", "mcts-blindspot:300"),
        ("mcts-full:900", "mcts-eval:900"),
        ("mcts-full:900", "mcts-blindspot:900"),
    ];
    let mut control_rows = Vec::new();
    for (idx, &(a, b)) in leaf_controls.iter().enumerate() {
        control_rows.push(run_one_duel(
            dex,
            &teams,
            a,
            b,
            games,
            seed + 300 + idx as u64,
            true,
            true,
            threads,
        ));
    }

    let cliff_probe = run_cliff_probe(dex);

    let report = json!({
        "self_ladder": self_rows,
        "anchor_ladder": anchor_rows,
        "raw_vs_full": ablation_rows,
        "leaf_controls": control_rows,
        "cliff_probe": cliff_probe,
    });
    println!("{report}");
}

fn run_v2_bench(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let games: usize = flag("--games").map_or(160, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026112350, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());

    let root = repo_root();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let teams: Vec<Vec<PokemonSet>> = pool.teams.iter().map(|t| t.sets.clone()).collect();

    // 1. Equal-iteration & equal-tree comparisons (fixed preview, CRN paired)
    let equal_iter_pairs = [
        ("mcts-v2:300", "mcts-full:300"),
        ("mcts-v2:300", "mcts-eval:300"),
        ("mcts-v2:300", "mcts-blindspot:300"),
        ("mcts-v2:900", "mcts-eval:900"),
        ("mcts-v2:900", "mcts-blindspot:900"),
        ("mcts-v2:2700", "mcts-eval:2700"),
        ("mcts-v2:2700", "mcts-blindspot:2700"),
    ];
    let mut equal_iter = Vec::new();
    for (idx, &(a, b)) in equal_iter_pairs.iter().enumerate() {
        equal_iter.push(run_one_duel(
            dex,
            &teams,
            a,
            b,
            games,
            seed + idx as u64,
            true,
            true,
            threads,
        ));
    }

    // 2. Self-scaling of FastGolf v2 up to 8100 iters (fixed preview, CRN)
    let self_pairs = [
        ("mcts-v2:900", "mcts-v2:300"),
        ("mcts-v2:2700", "mcts-v2:900"),
        ("mcts-v2:8100", "mcts-v2:2700"),
    ];
    let mut self_scale = Vec::new();
    for (idx, &(a, b)) in self_pairs.iter().enumerate() {
        self_scale.push(run_one_duel(
            dex,
            &teams,
            a,
            b,
            games,
            seed + 100 + idx as u64,
            true,
            true,
            threads,
        ));
    }

    // 3. Full-game vs shipped SM-MCTS (`skuct:300`, `skuct:1000`, `skuct:3000`) at equal/comparable wall-clock!
    let vs_skuct_pairs = [
        ("mcts-v2:900", "skuct:300"),
        ("mcts-v2:2700", "skuct:300"),
        ("mcts-v2:2700", "skuct:1000"),
        ("mcts-v2:8100", "skuct:1000"),
        ("mcts-v2:8100", "skuct:3000"),
    ];
    let mut vs_skuct = Vec::new();
    for (idx, &(a, b)) in vs_skuct_pairs.iter().enumerate() {
        vs_skuct.push(run_one_duel(
            dex,
            &teams,
            a,
            b,
            games,
            seed + 200 + idx as u64,
            false,
            false,
            threads,
        ));
    }

    println!(
        "{}",
        json!({
            "equal_iter": equal_iter,
            "self_scale": self_scale,
            "vs_skuct": vs_skuct,
        })
    );
}

/// Post-mortem autopsy: play `games` between `mcts-v2:300` and `mcts-eval:300`, log states where
/// `mcts-v2` lost despite thinking it was ahead (or disagreed sharply with `eval.rs`), to `/tmp/v2_autopsy.txt`.
fn run_autopsy(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let games: usize = flag("--games").map_or(40, |v| v.parse().unwrap());
    let arm_a = flag("--arm-a").unwrap_or_else(|| "mcts-v2:300".to_string());
    let arm_b = flag("--arm-b").unwrap_or_else(|| "mcts-eval:300".to_string());
    let root = repo_root();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let teams: Vec<Vec<PokemonSet>> = pool.teams.iter().map(|t| t.sets.clone()).collect();

    let mut sched = nc2000_bot::SplitMix64::new(2026112399);
    let mut dump = String::new();
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut blindspot_counts = [0usize; 6];

    for g in 0..games {
        let t0 = sched.below(teams.len());
        let t1 = sched.below(teams.len());
        let bseed = sched.battle_seed();
        let mut b = Battle::from_fixture(dex, &bseed, &teams[t0], &teams[t1]).unwrap();
        let tp0 = b.legal_choices(dex, 0)[0];
        let tp1 = b.legal_choices(dex, 1)[0];
        b.apply_choices(dex, [Some(tp0), Some(tp1)]).unwrap();

        let mut a0 = build_agent(&arm_a, 1000 + g as u64, true);
        let mut a1 = build_agent(&arm_b, 2000 + g as u64, true);
        let mut trace = Vec::new();
        let mut reached_endgame_21 = false;
        let mut reached_endgame_11 = false;

        for _ in 0..160 {
            if b.outcome().is_some() {
                break;
            }
            let (table, cb) = nc2000_engine::compact::CompactBattle::from_battle(&b, dex);
            let n_tot = cb.sides[0].alive_mask.count_ones() + cb.sides[1].alive_mask.count_ones();
            if n_tot <= 3 { reached_endgame_21 = true; }
            if n_tot <= 2 { reached_endgame_11 = true; }
            let fg = nc2000_bot::tempo::evaluate_fast_golf(&cb, &table, None);
            let ev = nc2000_bot::eval::compact_eval_leaf_blindspot(&cb, &table, &nc2000_bot::EvalWeights::default());
            let cs0 = b.legal_choices(dex, 0);
            let cs1 = b.legal_choices(dex, 1);
            let p0 = if cs0.is_empty() { None } else { Some(a0.choose(&b, dex, 0, &cs0)) };
            let p1 = if cs1.is_empty() { None } else { Some(a1.choose(&b, dex, 1, &cs1)) };

            let diff = fg.win_prob - ev;
            if diff.abs() > 0.25 {
                let alive0 = cb.sides[0].alive_mask.count_ones();
                let alive1 = cb.sides[1].alive_mask.count_ones();
                let desc_side = |s: usize| -> String {
                    let mut parts = Vec::new();
                    for slot in 0..3 {
                        if let Some(&ri) = b.sides[s].party.get(slot) {
                            let p = &b.sides[s].roster[ri as usize];
                            if p.hp > 0 {
                                let act = if cb.sides[s].active_idx as usize == slot { "*" } else { "" };
                                let st = if p.status != Status::None { format!(" {:?}", p.status) } else { String::new() };
                                let bst: String = p.boosts.iter().enumerate()
                                    .filter(|(_, &v)| v != 0)
                                    .map(|(i, v)| format!("b{i}:{v:+}"))
                                    .collect::<Vec<_>>().join(",");
                                let bst_s = if bst.is_empty() { String::new() } else { format!(" [{bst}]") };
                                parts.push(format!("{act}{}({}/{}{st}{bst_s})", p.name.as_str(), p.hp, p.maxhp));
                            }
                        }
                    }
                    parts.join(", ")
                };
                trace.push((
                    b.turn,
                    fg.g_turns[0],
                    fg.g_turns[1],
                    fg.win_prob,
                    ev,
                    alive0,
                    alive1,
                    p0.map(|c| c.to_input(dex)).unwrap_or_default(),
                    p1.map(|c| c.to_input(dex)).unwrap_or_default(),
                    desc_side(0),
                    desc_side(1),
                ));
            }
            if b.apply_choices(dex, [p0, p1]).is_err() {
                break;
            }
        }

        let won = b.outcome() == Some(nc2000_engine::battle::Outcome::P1Win);
        if won {
            wins += 1;
            if reached_endgame_11 { blindspot_counts[0] += 1; }
        } else {
            losses += 1;
            if reached_endgame_11 { blindspot_counts[1] += 1; }
            if reached_endgame_21 { blindspot_counts[3] += 1; }
            if !trace.is_empty() {
                // Find turn with max optimism (fg - ev) in lost games
                if let Some(worst) = trace.iter().max_by(|a, b| (a.3 - a.4).total_cmp(&(b.3 - b.4))) {
                    if worst.3 - worst.4 > 0.25 {
                        if worst.5 < worst.6 { blindspot_counts[2] += 1; }
                        dump.push_str(&format!(
                            "=== Game {g} LOST (t={}, eg11={}) | FastGolf G=({:.2}t,{:.2}t) wp={:.1}% vs blindspot={:.1}% (gap={:+.1}pp) ===\n  P1(v2) [{}, picked {}]: {}\n  P2(bs) [{}, picked {}]: {}\n\n",
                            worst.0, reached_endgame_11, worst.1, worst.2, worst.3 * 100.0, worst.4 * 100.0, (worst.3 - worst.4) * 100.0,
                            worst.5, worst.7, worst.9, worst.6, worst.8, worst.10
                        ));
                    }
                }
            }
        }
    }
    std::fs::write("/tmp/v2_autopsy.txt", &dump).unwrap();
    println!(
        "{}",
        json!({
            "games": games,
            "v2_wins": wins,
            "v2_losses": losses,
            "eg11_wins": blindspot_counts[0],
            "eg11_losses": blindspot_counts[1],
            "eg21_losses": blindspot_counts[3],
            "autopsy_entries": dump.matches("=== Game").count(),
            "under_piece_cases": blindspot_counts[2]
        })
    );
}

/// Fast offline calibration & phase-breakdown benchmark:
/// Plays `games` self-play games in parallel (default 240 games with `mcts-eval:300`), collects all
/// non-terminal states with ground-truth game outcomes, and reports Brier, LogLoss, AUC, and sharpness
/// across `FastGolf`, `eval.rs`, and `eval_blindspot` split by piece-count bucket.
fn run_calib(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let games: usize = flag("--games").map_or(240, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026112377, |v| v.parse().unwrap());
    let arm_a = flag("--arm-a")
        .or_else(|| flag("--arm"))
        .unwrap_or_else(|| "mcts-eval:300".to_string());
    let arm_b = flag("--arm-b").unwrap_or_else(|| arm_a.clone());
    let pool_kind = flag("--pool").unwrap_or_else(|| "v0".to_string());
    let root = repo_root();
    let teams: Vec<Vec<PokemonSet>> = if pool_kind == "v2" {
        let cat_txt = std::fs::read_to_string(root.join("data/team-pool-v2/team-pool.json"))
            .expect("read team-pool-v2");
        let catalog: WeightedCatalog = serde_json::from_str(&cat_txt).expect("parse team-pool-v2");
        catalog.teams.into_iter().map(|t| t.sets).collect()
    } else {
        let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
        pool.teams.iter().map(|t| t.sets.clone()).collect()
    };

    let mut sched = nc2000_bot::SplitMix64::new(seed);
    let mut jobs = Vec::with_capacity(games);
    for g in 0..games {
        let t0 = sched.below(teams.len());
        let t1 = sched.below(teams.len());
        let bseed = sched.battle_seed();
        jobs.push((g, t0, t1, bseed));
    }

    #[derive(Clone)]
    struct Sample {
        game: usize,
        bucket: usize, // 0:1v1, 1:2v1, 2:2v2, 3:3v1/3v2, 4:3v3
        y: f64,
        g0: f64,
        g1: f64,
        fg_p: f64,
        hy_p: f64,
        ev_p: f64,
        bs_p: f64,
        desc: String,
    }

    let chunk_size = (games + threads - 1) / threads.max(1);
    let all_samples: Vec<Sample> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for chunk in jobs.chunks(chunk_size.max(1)) {
            let dex_ref = dex;
            let teams_ref = &teams;
            let arm_a_ref = &arm_a;
            let arm_b_ref = &arm_b;
            handles.push(s.spawn(move || {
                let w = nc2000_bot::EvalWeights::default();
                let mut local = Vec::new();
                for &(g, t0, t1, ref bseed) in chunk {
                    let mut b = Battle::from_fixture(dex_ref, bseed, &teams_ref[t0], &teams_ref[t1]).unwrap();
                    let tp0 = b.legal_choices(dex_ref, 0)[0];
                    let tp1 = b.legal_choices(dex_ref, 1)[0];
                    b.apply_choices(dex_ref, [Some(tp0), Some(tp1)]).unwrap();
                    let mut a0 = build_agent(arm_a_ref, 30000 + g as u64, true);
                    let mut a1 = build_agent(arm_b_ref, 40000 + g as u64, true);
                    let mut game_rows: Vec<(usize, f64, f64, f64, f64, f64, f64, String)> = Vec::new();
                    for _ in 0..160 {
                        if b.outcome().is_some() {
                            break;
                        }
                        let (table, cb) = nc2000_engine::compact::CompactBattle::from_battle(&b, dex_ref);
                        let n0 = cb.sides[0].alive_mask.count_ones();
                        let n1 = cb.sides[1].alive_mask.count_ones();
                        let bucket = match (n0.min(n1), n0.max(n1)) {
                            (1, 1) => 0,
                            (1, 2) => 1,
                            (2, 2) => 2,
                            (3, 3) => 4,
                            _ => 3,
                        };
                        let fg = nc2000_bot::tempo::evaluate_fast_golf(&cb, &table, None);
                        let hy = nc2000_bot::tempo::evaluate_hybrid_golf(&cb, &table, None);
                        let ev = nc2000_bot::eval::compact_eval01(&cb, &table, &w);
                        let bs = nc2000_bot::eval::compact_eval_leaf_blindspot(&cb, &table, &w);
                        let desc = if (fg.win_prob - bs).abs() > 0.18 || (bucket == 1 && (fg.win_prob - bs).abs() > 0.10) {
                            let fmt_s = |side: usize| -> String {
                                let mut ps = Vec::new();
                                for slot in 0..3 {
                                    if let Some(&ri) = b.sides[side].party.get(slot) {
                                        let p = &b.sides[side].roster[ri as usize];
                                        if p.hp > 0 {
                                            let act = if cb.sides[side].active_idx as usize == slot && cb.sides[side].request != 2 { "*" } else { "" };
                                            let cmon = &cb.mons[side * 3 + slot];
                                            let st = if p.status != Status::None { format!(" {:?}(c={})", p.status, cmon.status_counter()) } else { String::new() };
                                            let it = p.item.map(|id| format!(" @{}", dex_ref.items.get(id).name)).unwrap_or_default();
                                            let vf = if cmon.vol_flags != 0 || cmon.sub_hp() > 0 { format!(" v={:#x}/sub={}", cmon.vol_flags, cmon.sub_hp()) } else { String::new() };
                                            let bst: String = p.boosts.iter().enumerate()
                                                .filter(|(_, &v)| v != 0)
                                                .map(|(i, v)| format!("b{i}:{v:+}"))
                                                .collect::<Vec<_>>().join(",");
                                            let bs_s = if bst.is_empty() { String::new() } else { format!(" [{bst}]") };
                                            let mvs: String = p.move_slots.iter()
                                                .map(|m| format!("{}:{}", dex_ref.move_static(m.id).name, m.pp))
                                                .collect::<Vec<_>>().join("/");
                                            ps.push(format!("#{slot}:{act}{}({}/{}{it}{st}{vf}{bs_s} [{mvs}])", p.name.as_str(), p.hp, p.maxhp));
                                        }
                                    }
                                }
                                ps.join(", ")
                            };
                            let dbg = nc2000_bot::tempo::format_fast_golf_debug(&cb, &table);
                            format!("g={g} t={} {}v{} req=({},{})\n  P1: {}\n  P2: {}\n{}", b.turn, n0, n1, cb.sides[0].request, cb.sides[1].request, fmt_s(0), fmt_s(1), dbg)
                        } else {
                            String::new()
                        };
                        game_rows.push((bucket, fg.g_turns[0], fg.g_turns[1], fg.win_prob, hy, ev, bs, desc));

                        let cs0 = b.legal_choices(dex_ref, 0);
                        let cs1 = b.legal_choices(dex_ref, 1);
                        let p0 = if cs0.is_empty() { None } else { Some(a0.choose(&b, dex_ref, 0, &cs0)) };
                        let p1 = if cs1.is_empty() { None } else { Some(a1.choose(&b, dex_ref, 1, &cs1)) };
                        if b.apply_choices(dex_ref, [p0, p1]).is_err() {
                            break;
                        }
                    }
                    let y = match b.outcome() {
                        Some(nc2000_engine::battle::Outcome::P1Win) => 1.0,
                        Some(nc2000_engine::battle::Outcome::P2Win) => 0.0,
                        _ => continue,
                    };
                    for (bucket, g0, g1, fg_p, hy_p, ev_p, bs_p, desc) in game_rows {
                        local.push(Sample { game: g, bucket, y, g0, g1, fg_p, hy_p, ev_p, bs_p, desc });
                    }
                }
                local
            }));
        }
        let mut merged = Vec::new();
        for h in handles {
            merged.extend(h.join().unwrap());
        }
        merged
    });

    let metrics = |pts: &[(f64, f64)]| -> serde_json::Value {
        let n = pts.len();
        if n == 0 {
            return json!({"n": 0});
        }
        let mut brier = 0.0;
        let mut ll = 0.0;
        let mut sharp = 0.0;
        let mut ext_err = 0usize;
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for &(p, y) in pts {
            let pc = p.clamp(1e-5, 1.0 - 1e-5);
            brier += (pc - y) * (pc - y);
            ll -= y * pc.ln() + (1.0 - y) * (1.0 - pc).ln();
            sharp += (pc - 0.5).abs();
            if (pc - y).abs() > 0.75 {
                ext_err += 1;
            }
            if y > 0.5 { pos.push(pc); } else { neg.push(pc); }
        }
        // Exact AUC via sorted ranks
        let mut ranked: Vec<(f64, f64)> = pts.to_vec();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut rank_sum_pos = 0.0;
        let mut i = 0;
        while i < ranked.len() {
            let mut j = i + 1;
            while j < ranked.len() && (ranked[j].0 - ranked[i].0).abs() <= 1e-12 {
                j += 1;
            }
            let avg_rank = 0.5 * ((i + 1 + j) as f64);
            for k in i..j {
                if ranked[k].1 > 0.5 {
                    rank_sum_pos += avg_rank;
                }
            }
            i = j;
        }
        let n_pos = pos.len() as f64;
        let n_neg = neg.len() as f64;
        let auc = if n_pos > 0.0 && n_neg > 0.0 {
            (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
        } else {
            0.5
        };
        json!({
            "n": n,
            "brier": (brier / n as f64 * 10000.0).round() / 10000.0,
            "logloss": (ll / n as f64 * 10000.0).round() / 10000.0,
            "auc": (auc * 10000.0).round() / 10000.0,
            "sharp": (sharp / n as f64 * 1000.0).round() / 1000.0,
            "ext75": ((ext_err as f64 / n as f64) * 1000.0).round() / 10.0,
        })
    };

    // Grid-search optimal Brownian `var_per_turn` per bucket for FastGolf `g1 - g0`
    let best_var_brier = |rows: &[&Sample]| -> (f64, f64) {
        let mut best_v = 0.5;
        let mut best_b = f64::INFINITY;
        for step in 2..=80 {
            let v = step as f64 * 0.25;
            let mut bsum = 0.0;
            for r in rows {
                let sigma = (v * (r.g0 + r.g1).max(1.0)).sqrt();
                let z = (r.g1 - r.g0) / sigma;
                // Abramowitz-Stegun normal CDF
                let t = 1.0 / (1.0 + 0.2316419 * z.abs());
                let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
                let pdf = (-0.5 * z * z).exp() * 0.3989422804014327;
                let p = if z >= 0.0 { 1.0 - pdf * poly } else { pdf * poly };
                bsum += (p - r.y) * (p - r.y);
            }
            let b = bsum / rows.len().max(1) as f64;
            if b < best_b {
                best_b = b;
                best_v = v;
            }
        }
        ((best_v * 100.0).round() / 100.0, (best_b * 10000.0).round() / 10000.0)
    };

    // Grid-search optimal log-odds blend `(w_fg, w_bs)` per bucket for `HybridGolf`
    let best_blend = |rows: &[&Sample]| -> (f64, f64, f64) {
        let logit = |p: f64| -> f64 {
            let q = p.clamp(0.000_5, 0.999_5);
            (q / (1.0 - q)).ln()
        };
        let mut best_pair = (1.0, 0.0);
        let mut best_b = f64::INFINITY;
        for a_i in 0..=24 {
            let w_fg = a_i as f64 * 0.05;
            for b_i in 0..=20 {
                let w_bs = b_i as f64 * 0.05;
                if a_i == 0 && b_i == 0 {
                    continue;
                }
                let mut bsum = 0.0;
                for r in rows {
                    let z = w_fg * logit(r.fg_p) + w_bs * logit(r.bs_p);
                    let p = 1.0 / (1.0 + (-z).exp());
                    bsum += (p - r.y) * (p - r.y);
                }
                let b = bsum / rows.len().max(1) as f64;
                if b < best_b {
                    best_b = b;
                    best_pair = (w_fg, w_bs);
                }
            }
        }
        (
            (best_pair.0 * 100.0).round() / 100.0,
            (best_pair.1 * 100.0).round() / 100.0,
            (best_b * 10000.0).round() / 10000.0,
        )
    };

    let bucket_names = ["1v1", "2v1", "2v2", "3v2_3v1", "3v3"];
    let mut by_bucket = serde_json::Map::new();
    for b_idx in 0..5 {
        let sub: Vec<&Sample> = all_samples.iter().filter(|s| s.bucket == b_idx).collect();
        let fg_pts: Vec<(f64, f64)> = sub.iter().map(|s| (s.fg_p, s.y)).collect();
        let hy_pts: Vec<(f64, f64)> = sub.iter().map(|s| (s.hy_p, s.y)).collect();
        let ev_pts: Vec<(f64, f64)> = sub.iter().map(|s| (s.ev_p, s.y)).collect();
        let bs_pts: Vec<(f64, f64)> = sub.iter().map(|s| (s.bs_p, s.y)).collect();
        let (opt_v, opt_b) = best_var_brier(&sub);
        let (opt_wfg, opt_wbs, opt_blend_b) = best_blend(&sub);
        by_bucket.insert(
            bucket_names[b_idx].to_string(),
            json!({
                "fg": metrics(&fg_pts),
                "hy": metrics(&hy_pts),
                "ev": metrics(&ev_pts),
                "bs": metrics(&bs_pts),
                "fg_opt_var": opt_v,
                "fg_opt_brier": opt_b,
                "opt_blend": [opt_wfg, opt_wbs, opt_blend_b],
            }),
        );
    }

    let fg_all: Vec<(f64, f64)> = all_samples.iter().map(|s| (s.fg_p, s.y)).collect();
    let hy_all: Vec<(f64, f64)> = all_samples.iter().map(|s| (s.hy_p, s.y)).collect();
    let ev_all: Vec<(f64, f64)> = all_samples.iter().map(|s| (s.ev_p, s.y)).collect();
    let bs_all: Vec<(f64, f64)> = all_samples.iter().map(|s| (s.bs_p, s.y)).collect();

    // Dump deduplicated (1 worst state per game) blunders for 2v1 and 2v2 separately!
    let dump_bucket = |b_idx: usize, path: &str, limit: usize| {
        let mut by_game: std::collections::BTreeMap<usize, &Sample> = std::collections::BTreeMap::new();
        for s in all_samples.iter().filter(|s| s.bucket == b_idx && !s.desc.is_empty()) {
            let err = (s.fg_p - s.y).abs() - (s.bs_p - s.y).abs();
            by_game
                .entry(s.game)
                .and_modify(|cur| {
                    let cur_err = (cur.fg_p - cur.y).abs() - (cur.bs_p - cur.y).abs();
                    if err > cur_err {
                        *cur = s;
                    }
                })
                .or_insert(s);
        }
        let mut worst: Vec<&Sample> = by_game.into_values().collect();
        worst.sort_by(|a, b| {
            let ea = (a.fg_p - a.y).abs() - (a.bs_p - a.y).abs();
            let eb = (b.fg_p - b.y).abs() - (b.bs_p - b.y).abs();
            eb.total_cmp(&ea)
        });
        let mut dump = String::new();
        for s in worst.iter().take(limit) {
            dump.push_str(&format!(
                "=== y={:.0} | fg={:.1}% G=({:.2}t,{:.2}t) | bs={:.1}% | ev={:.1}% | {}\n",
                s.y, s.fg_p * 100.0, s.g0, s.g1, s.bs_p * 100.0, s.ev_p * 100.0, s.desc
            ));
        }
        std::fs::write(path, &dump).unwrap();
    };
    dump_bucket(0, "/tmp/v2_calib_1v1.txt", 25);
    dump_bucket(1, "/tmp/v2_calib_2v1.txt", 25);
    dump_bucket(2, "/tmp/v2_calib_2v2.txt", 20);
    dump_bucket(3, "/tmp/v2_calib_3v2.txt", 15);

    println!(
        "{}",
        json!({
            "games": games,
            "samples": all_samples.len(),
            "overall": {
                "fg": metrics(&fg_all),
                "hy": metrics(&hy_all),
                "ev": metrics(&ev_all),
                "bs": metrics(&bs_all),
            },
            "buckets": by_bucket,
        })
    );
}

/// Controlled rollout-necessity & search-engine ablation benchmark (100% symmetric fixed preview!).
fn run_rollout_bench(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let games: usize = flag("--games").map_or(160, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026112500, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());

    let root = repo_root();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let teams: Vec<Vec<PokemonSet>> = pool.teams.iter().map(|t| t.sets.clone()).collect();

    // 1. Inside EXACT `SkuctSearch` engine (`step_with_leaf`), symmetric fixed preview:
    //    Tests whether `FastGolf` needs rollouts (`0r` vs `2r` vs `4r` vs `8r`) vs `eval_blindspot` (`0r` vs shipped `8r`).
    let skuct_engine_pairs = [
        ("skuct-bs-0r:1000", "skuct:1000"),
        ("skuct-golf-0r:1000", "skuct:1000"),
        ("skuct-golf-2r:1000", "skuct:1000"),
        ("skuct-golf-4r:1000", "skuct:1000"),
        ("skuct-golf-8r:1000", "skuct:1000"),
        ("skuct-golf-0r:1000", "skuct-golf-8r:1000"),
        ("skuct-golf-0r:2200", "skuct:1000"),
    ];
    let mut skuct_engine = Vec::new();
    for (idx, &(a, b)) in skuct_engine_pairs.iter().enumerate() {
        skuct_engine.push(run_one_duel(
            dex,
            &teams,
            a,
            b,
            games,
            seed + idx as u64,
            true,
            true,
            threads,
        ));
    }

    // 2. Inside CompactBattle-native Minimax-UCT (`cmcts`, 5x faster than Battle-based `SkuctSearch`!):
    //    Measures step-by-step search upgrades & rollout necessity at equal wall-clock (~10 ms vs `skuct:1000`).
    let cmcts_pairs = [
        ("mcts-v2:5000", "skuct:1000"),
        ("cmcts-golf-0r:5000", "skuct:1000"),
        ("cmcts-bs-0r:7500", "skuct:1000"),
        ("cmcts-golf-2r:3500", "skuct:1000"),
        ("cmcts-golf-4r:2500", "skuct:1000"),
        ("cmcts-golf-8r:1600", "skuct:1000"),
        ("cmcts-golf-0r:5000", "cmcts-golf-8r:1600"),
    ];
    let mut cmcts = Vec::new();
    for (idx, &(a, b)) in cmcts_pairs.iter().enumerate() {
        cmcts.push(run_one_duel(
            dex,
            &teams,
            a,
            b,
            games,
            seed + 100 + idx as u64,
            true,
            true,
            threads,
        ));
    }

    println!(
        "{}",
        json!({
            "skuct_engine": skuct_engine,
            "cmcts_engine": cmcts,
        })
    );
}

#[derive(Clone, Deserialize)]
struct WeightedCatalog {
    teams: Vec<WeightedTeam>,
}

#[derive(Clone, Deserialize)]
struct WeightedTeam {
    #[allow(dead_code)]
    id: String,
    sets: Vec<PokemonSet>,
    #[serde(rename = "drawWeight")]
    draw_weight: Option<f64>,
}

struct TimedAgentWrap {
    inner: Box<dyn Agent>,
    preview_override: Option<[u8; 3]>,
    searched_battle_ns: Vec<u64>,
}

impl TimedAgentWrap {
    fn new(inner: Box<dyn Agent>) -> Self {
        Self { inner, preview_override: None, searched_battle_ns: Vec::new() }
    }

    fn with_preview_override(inner: Box<dyn Agent>, triple: [u8; 3]) -> Self {
        Self { inner, preview_override: Some(triple), searched_battle_ns: Vec::new() }
    }
}

impl Agent for TimedAgentWrap {
    fn requires_player_log(&self) -> bool {
        self.inner.requires_player_log()
    }

    fn name(&self) -> String {
        self.inner.name()
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let is_preview = !choices.is_empty() && matches!(choices[0], SearchChoice::Team(_));
        let record = choices.len() > 1 && !is_preview;
        let t0 = if record { Some(Instant::now()) } else { None };
        let pick = self.inner.choose(battle, dex, side, choices);
        if let Some(t) = t0 {
            self.searched_battle_ns.push(t.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        }
        if is_preview {
            if let Some(ov) = self.preview_override {
                let sc = SearchChoice::Team(ov);
                if choices.contains(&sc) {
                    return sc;
                }
            }
        }
        pick
    }
}

fn one_hot_preview_policy(triple: [u8; 3]) -> Option<Vec<f64>> {
    if triple[2] == 0 {
        return None;
    }
    let acts = preview_actions();
    let idx = action_index(&acts, triple)?;
    let mut pol = vec![0.0; acts.len()];
    pol[idx] = 1.0;
    Some(pol)
}

fn build_blind_arm(
    spec: &str,
    pool: Arc<MetaPool>,
    seed: u64,
    preview_triple: [u8; 3],
) -> Box<dyn Agent> {
    if let Some(rest) = spec.strip_prefix("blind-default:") {
        if rest == "adaptive" {
            let cfg = RmConfig {
                iterations: nc2000_bot::adaptive_budget::BATTLE_CAP,
                rule: SelRule::Ucb,
                c: 0.4,
                ..Default::default()
            };
            let mut ag = BlindAgent::new(cfg, pool, None, seed).with_adaptive_battle_budget();
            if let Some(pol) = one_hot_preview_policy(preview_triple) {
                ag = ag.with_own_preview_policy(pol);
            } else {
                ag = ag.with_preview_iterations(240_000);
            }
            Box::new(ag)
        } else {
            let iters: u32 = rest.parse().unwrap_or_else(|_| panic!("bad spec: {spec}"));
            let cfg = RmConfig { iterations: iters, rule: SelRule::Ucb, c: 0.4, ..Default::default() };
            let mut ag = BlindAgent::new(cfg, pool, None, seed);
            if let Some(pol) = one_hot_preview_policy(preview_triple) {
                ag = ag.with_own_preview_policy(pol);
            } else {
                ag = ag.with_preview_iterations(240_000);
            }
            Box::new(ag)
        }
    } else if let Some(rest) = spec.strip_prefix("blind-shipped:") {
        if rest == "adaptive" {
            let cfg = RmConfig {
                iterations: nc2000_bot::adaptive_budget::BATTLE_CAP,
                rule: SelRule::Ucb,
                c: 0.4,
                settlement: nc2000_bot::accounting::LeafSettlementMode::BlindSpotLeaf,
                ..Default::default()
            };
            let mut ag = BlindAgent::new(cfg, pool, None, seed).with_adaptive_battle_budget();
            if let Some(pol) = one_hot_preview_policy(preview_triple) {
                ag = ag.with_own_preview_policy(pol);
            } else {
                ag = ag.with_preview_iterations(1);
            }
            Box::new(ag)
        } else {
            let iters: u32 = rest.parse().unwrap_or_else(|_| panic!("bad spec: {spec}"));
            let cfg = RmConfig {
                iterations: iters,
                rule: SelRule::Ucb,
                c: 0.4,
                settlement: nc2000_bot::accounting::LeafSettlementMode::BlindSpotLeaf,
                ..Default::default()
            };
            let mut ag = BlindAgent::new(cfg, pool, None, seed);
            if let Some(pol) = one_hot_preview_policy(preview_triple) {
                ag = ag.with_own_preview_policy(pol);
            } else {
                ag = ag.with_preview_iterations(1);
            }
            Box::new(ag)
        }
    } else if let Some(rest) = spec.strip_prefix("bcmcts2-") {
        let mut parts = rest.split('@');
        let base_spec = parts.next().unwrap();
        let (leaf, rollout_turns, iterations, adaptive) =
            parse_leaf_rollout_adaptive(base_spec).unwrap_or_else(|| panic!("bad bcmcts2 spec: {spec}"));
        let mut cfg = BlindCmctsConfig {
            iterations,
            c: 0.4,
            max_depth: 250,
            leaf,
            rollout_turns,
            max_worlds: 24,
            preview_iterations: 27_000,
            own_preview_triple: Some(preview_triple),
            adaptive,
            adaptive_interval: 13_500,
            adaptive_min_gap: 0.01,
            stage2_root: true,
            ..BlindCmctsConfig::default()
        };
        for m in parts {
            if m == "sp" { cfg.sempool = true; }
            else if m == "mw" { cfg.mature_window = true; }
            else if m == "puct" { cfg.use_puct = true; }
            else if let Some(v) = m.strip_prefix("cp") { cfg.c_puct = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("fpu") { cfg.fpu_red = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("peps") { cfg.puct_eps = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("lam") { cfg.backup_lambda = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("tau") { cfg.backup_tau = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("gap") { cfg.adaptive_min_gap = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("int") { cfg.adaptive_interval = v.parse().unwrap(); }
            else if m == "eg" { cfg.endgame_runs = 8_192; }
            else if let Some(v) = m.strip_prefix("eg") { cfg.endgame_runs = v.parse().unwrap(); }
            else if m == "ver" { cfg.verify_iters = 24_000; }
            else if let Some(v) = m.strip_prefix("ver") { cfg.verify_iters = v.parse().unwrap(); }
            else if m == "cfp" { cfg.use_cf_policy = true; }
            else if let Some(v) = m.strip_prefix('k') { cfg.contested_trees = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix('t') { cfg.leaf_temp = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix('c') { cfg.c = v.parse().unwrap(); }
            else { panic!("unknown modifier @{m} in {spec}"); }
        }
        Box::new(BlindCmctsAgent::new(cfg, pool, seed))
    } else if let Some(rest) = spec.strip_prefix("bcmcts-") {
        let mut parts = rest.split('@');
        let base_spec = parts.next().unwrap();
        let (leaf, rollout_turns, iterations, adaptive) =
            parse_leaf_rollout_adaptive(base_spec).unwrap_or_else(|| panic!("bad bcmcts spec: {spec}"));
        let mut cfg = BlindCmctsConfig {
            iterations,
            c: 0.4,
            max_depth: 250,
            leaf,
            rollout_turns,
            max_worlds: 24,
            preview_iterations: 27_000,
            own_preview_triple: Some(preview_triple),
            adaptive,
            adaptive_interval: 13_500,
            adaptive_min_gap: 0.01,
            stage2_root: false,
            ..BlindCmctsConfig::default()
        };
        for m in parts {
            if m == "sp" { cfg.sempool = true; }
            else if m == "mw" { cfg.mature_window = true; }
            else if m == "puct" { cfg.use_puct = true; }
            else if let Some(v) = m.strip_prefix("cp") { cfg.c_puct = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("fpu") { cfg.fpu_red = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("peps") { cfg.puct_eps = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("lam") { cfg.backup_lambda = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("tau") { cfg.backup_tau = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("gap") { cfg.adaptive_min_gap = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("int") { cfg.adaptive_interval = v.parse().unwrap(); }
            else if m == "eg" { cfg.endgame_runs = 8_192; }
            else if let Some(v) = m.strip_prefix("eg") { cfg.endgame_runs = v.parse().unwrap(); }
            else if m == "ver" { cfg.verify_iters = 24_000; }
            else if let Some(v) = m.strip_prefix("ver") { cfg.verify_iters = v.parse().unwrap(); }
            else if m == "cfp" { cfg.use_cf_policy = true; }
            else if let Some(v) = m.strip_prefix('k') { cfg.contested_trees = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix('t') { cfg.leaf_temp = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix('c') { cfg.c = v.parse().unwrap(); }
            else { panic!("unknown modifier @{m} in {spec}"); }
        }
        Box::new(BlindCmctsAgent::new(cfg, pool, seed))
    } else if let Some(rest) = spec.strip_prefix("blind-") {
        let (leaf, rollout_turns, iterations) =
            parse_leaf_rollout(rest).unwrap_or_else(|| panic!("bad blind spec: {spec}"));
        Box::new(
            BlindGolfAgent::new(
                BlindGolfConfig {
                    iterations,
                    c: 0.4,
                    leaf,
                    rollout_turns,
                    preview_iterations: 27_000,
                    own_preview_triple: Some(preview_triple),
                },
                pool,
                seed,
            ),
        )
    } else {
        panic!("unknown blind arm spec: {spec} (expected blind-shipped:N or blind-golf-Xr:N)")
    }
}

fn bootstrap_ci(vals: &[f64], seed: u64) -> (f64, f64, f64) {
    let n = vals.len();
    if n == 0 {
        return (0.5, 0.0, 1.0);
    }
    let mean = vals.iter().sum::<f64>() / (n as f64);
    let mut rng = SplitMix64::new(seed ^ 0xA5A5_5A5A_F0F0_0F0F);
    const B: usize = 2000;
    let mut boot = Vec::with_capacity(B);
    for _ in 0..B {
        let mut s = 0.0;
        for _ in 0..n {
            s += vals[rng.below(n)];
        }
        boot.push(s / (n as f64));
    }
    boot.sort_by(|a, b| a.total_cmp(b));
    (mean, boot[B * 25 / 1000], boot[B * 975 / 1000])
}

fn percentile_ms(ns: &[u64], pct: usize) -> f64 {
    if ns.is_empty() {
        return 0.0;
    }
    let mut v = ns.to_vec();
    v.sort_unstable();
    let idx = ((v.len() - 1) * pct) / 100;
    (v[idx] as f64) / 1e6
}

fn elo_diff(score: f64) -> f64 {
    let p = score.clamp(1e-4, 1.0 - 1e-4);
    -400.0 * ((1.0 / p) - 1.0).log10()
}

/// Production-protocol imperfect-information H2H (`team-pool-v2` drawWeight + `belief-pool-v3` +
/// shared `preview_iters` blind preview + seat-swapped paired schedules).
fn run_blind_duel(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let arm_a = flag("--arm-a").unwrap_or_else(|| "blind-golf-2r:9000".to_string());
    let arm_b = flag("--arm-b").unwrap_or_else(|| "blind-shipped:9000".to_string());
    let pairs: usize = flag("--pairs").map_or(128, |v| v.parse().unwrap());
    let pair_start: usize = flag("--pair-start").map_or(0, |v| v.parse().unwrap());
    let preview_iters: u32 = flag("--preview-iters").map_or(27_000, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(20261017003, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let preview_cache = flag("--preview-cache").map(PathBuf::from);
    let out_path = flag("--out").map(PathBuf::from);
    let max_turns: u16 = flag("--max-turns").map_or(500, |v| v.parse().unwrap());
    let catalog_path = flag("--catalog").unwrap_or_else(|| "data/team-pool-v2/team-pool.json".to_string());
    let unseen_only = args.iter().any(|a| a == "--unseen-only");
    let preview_only = args.iter().any(|a| a == "--preview-only");
    let total_pairs_needed = pair_start + pairs;

    let root = repo_root();
    let cat_txt = std::fs::read_to_string(root.join(&catalog_path))
        .unwrap_or_else(|e| panic!("read catalog {catalog_path}: {e}"));
    let mut catalog: WeightedCatalog = serde_json::from_str(&cat_txt)
        .unwrap_or_else(|e| panic!("parse catalog {catalog_path}: {e}"));
    if unseen_only {
        catalog.teams.retain(|t| (canonical_team_hash(&t.sets) % 5) == 0);
        assert!(
            !catalog.teams.is_empty(),
            "unseen-only filter left 0 teams in {catalog_path}"
        );
    }
    let pool = Arc::new(load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json")));

    let weights: Vec<f64> = catalog
        .teams
        .iter()
        .map(|t| t.draw_weight.unwrap_or(1.0))
        .collect();
    let total_w: f64 = weights.iter().sum();
    let sample_w = |rng: &mut SplitMix64| -> usize {
        let mut r = rng.next_f64() * total_w;
        for (i, &w) in weights.iter().enumerate() {
            if r < w {
                return i;
            }
            r -= w;
        }
        weights.len() - 1
    };

    #[derive(Clone)]
    struct PairSched {
        pair_idx: usize,
        ta: usize,
        tb: usize,
        bseed: String,
        pseed_a: u64,
        pseed_b: u64,
        sseed_p1: u64,
        sseed_p2: u64,
        prev_a: [u8; 3],
        prev_b: [u8; 3],
    }

    let mut rng = SplitMix64::new(seed);
    let mut scheds: Vec<PairSched> = Vec::with_capacity(total_pairs_needed);
    for pair_idx in 0..total_pairs_needed {
        let ta = sample_w(&mut rng);
        let tb = sample_w(&mut rng);
        let bseed = rng.battle_seed();
        let pseed_a = rng.next();
        let pseed_b = rng.next();
        let sseed_p1 = rng.next();
        let sseed_p2 = rng.next();
        scheds.push(PairSched {
            pair_idx,
            ta,
            tb,
            bseed,
            pseed_a,
            pseed_b,
            sseed_p1,
            sseed_p2,
            prev_a: [0; 3],
            prev_b: [0; 3],
        });
    }

    // Load or compute shared blind Team Preview picks (identical across arms and seat swaps!).
    let mut cached_prev: HashMap<usize, ([u8; 3], [u8; 3])> = HashMap::new();
    if let Some(ref p) = preview_cache {
        if p.exists() {
            if let Ok(txt) = std::fs::read_to_string(p) {
                if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                    if v["seed"].as_u64() == Some(seed)
                        && v["preview_iters"].as_u64() == Some(u64::from(preview_iters))
                    {
                        if let Some(arr) = v["entries"].as_array() {
                            for e in arr {
                                if let (Some(idx), Some(pa), Some(pb)) = (
                                    e["pair_idx"].as_u64(),
                                    e["prev_a"].as_array(),
                                    e["prev_b"].as_array(),
                                ) {
                                    if pa.len() == 3 && pb.len() == 3 {
                                        cached_prev.insert(
                                            idx as usize,
                                            (
                                                [
                                                    pa[0].as_u64().unwrap() as u8,
                                                    pa[1].as_u64().unwrap() as u8,
                                                    pa[2].as_u64().unwrap() as u8,
                                                ],
                                                [
                                                    pb[0].as_u64().unwrap() as u8,
                                                    pb[1].as_u64().unwrap() as u8,
                                                    pb[2].as_u64().unwrap() as u8,
                                                ],
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let missing: Vec<usize> = (pair_start..total_pairs_needed)
        .filter(|i| !cached_prev.contains_key(i))
        .collect();
    if !missing.is_empty() {
        let cursor = AtomicUsize::new(0);
        let computed: Vec<(usize, [u8; 3], [u8; 3])> = std::thread::scope(|s| {
            let mut hs = Vec::new();
            for _ in 0..threads.min(missing.len()).max(1) {
                let (cat_r, pool_r, sched_r, miss_r, cur_r) =
                    (&catalog, &pool, &scheds, &missing, &cursor);
                hs.push(s.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let k = cur_r.fetch_add(1, Ordering::Relaxed);
                        if k >= miss_r.len() {
                            break;
                        }
                        let pi = miss_r[k];
                        let sc = &sched_r[pi];
                        let pick_side = |side: usize, pseed: u64| -> [u8; 3] {
                            let mut b = Battle::from_fixture(
                                dex,
                                &sc.bseed,
                                &cat_r.teams[sc.ta].sets,
                                &cat_r.teams[sc.tb].sets,
                            )
                            .unwrap();
                            b.set_log_enabled(true);
                            let cs = b.legal_choices(dex, side);
                            let cfg = RmConfig {
                                iterations: preview_iters,
                                rule: SelRule::Ucb,
                                c: 0.4,
                                ..Default::default()
                            };
                            let mut ag = BlindAgent::new(cfg, pool_r.clone(), None, pseed);
                            match ag.choose(&b, dex, side, &cs) {
                                SearchChoice::Team(t) => t,
                                other => panic!("expected team choice, got {other:?}"),
                            }
                        };
                        let pa = pick_side(0, sc.pseed_a);
                        let pb = pick_side(1, sc.pseed_b);
                        out.push((pi, pa, pb));
                    }
                    out
                }));
            }
            let mut all = Vec::new();
            for h in hs {
                all.extend(h.join().unwrap());
            }
            all
        });
        for (pi, pa, pb) in computed {
            cached_prev.insert(pi, (pa, pb));
        }
        if let Some(ref p) = preview_cache {
            let mut entries: Vec<Value> = cached_prev
                .iter()
                .map(|(&idx, &(pa, pb))| json!({"pair_idx": idx, "prev_a": pa, "prev_b": pb}))
                .collect();
            entries.sort_by_key(|e| e["pair_idx"].as_u64().unwrap());
            let doc = json!({"seed": seed, "preview_iters": preview_iters, "entries": entries});
            std::fs::write(p, serde_json::to_string(&doc).unwrap()).unwrap();
        }
    }
    for sc in &mut scheds[pair_start..total_pairs_needed] {
        let (pa, pb) = cached_prev[&sc.pair_idx];
        sc.prev_a = pa;
        sc.prev_b = pb;
    }
    if preview_only {
        println!(
            "{}",
            json!({"cmd":"preview-only","seed":seed,"preview_iters":preview_iters,"pair_start":pair_start,"pairs":pairs,"cached":cached_prev.len()})
        );
        return;
    }

    let t0 = Instant::now();
    let total_games = pairs * 2;
    let cursor = AtomicUsize::new(0);
    struct GameOut {
        game_idx: usize,
        pair_idx: usize,
        a_is_p1: bool,
        a_score: f64,
        turns: u16,
        turn_cap: bool,
        a_ns: Vec<u64>,
        b_ns: Vec<u64>,
    }
    let results: Vec<GameOut> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(total_games).max(1) {
            let (cat_r, pool_r, sched_r, cur_r, aa, ab) =
                (&catalog, &pool, &scheds, &cursor, &arm_a, &arm_b);
            hs.push(s.spawn(move || {
                let mut local = Vec::new();
                loop {
                    let local_gi = cur_r.fetch_add(1, Ordering::Relaxed);
                    if local_gi >= total_games {
                        break;
                    }
                    let pi = pair_start + local_gi / 2;
                    let gi = pi * 2 + (local_gi % 2);
                    let a_is_p1 = gi % 2 == 0;
                    let sc = &sched_r[pi];
                    let (spec1, spec2) = if a_is_p1 { (aa.as_str(), ab.as_str()) } else { (ab.as_str(), aa.as_str()) };
                    let mut p1 = TimedAgentWrap::new(build_blind_arm(spec1, pool_r.clone(), sc.sseed_p1, sc.prev_a));
                    let mut p2 = TimedAgentWrap::new(build_blind_arm(spec2, pool_r.clone(), sc.sseed_p2, sc.prev_b));
                    let mut b = Battle::from_fixture(
                        dex,
                        &sc.bseed,
                        &cat_r.teams[sc.ta].sets,
                        &cat_r.teams[sc.tb].sets,
                    )
                    .unwrap();
                    b.set_log_enabled(true);
                    let res = play_game(dex, &mut b, &mut [&mut p1, &mut p2], max_turns).unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        _ => 0.5,
                    };
                    let a_score = if a_is_p1 { p1_score } else { 1.0 - p1_score };
                    let (a_ns, b_ns) = if a_is_p1 {
                        (p1.searched_battle_ns, p2.searched_battle_ns)
                    } else {
                        (p2.searched_battle_ns, p1.searched_battle_ns)
                    };
                    local.push(GameOut {
                        game_idx: gi,
                        pair_idx: pi,
                        a_is_p1,
                        a_score,
                        turns: b.turn,
                        turn_cap: matches!(res, GameResult::TurnCapped),
                        a_ns,
                        b_ns,
                    });
                }
                local
            }));
        }
        let mut merged = Vec::with_capacity(total_games);
        for h in hs {
            merged.extend(h.join().unwrap());
        }
        merged
    });

    let by_idx: HashMap<usize, GameOut> =
        results.into_iter().map(|r| (r.game_idx, r)).collect();
    if let Some(ref op) = out_path {
        use std::io::Write;
        if let Some(parent) = op.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).unwrap();
            }
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(op).unwrap());
        for pi in pair_start..total_pairs_needed {
            for gi in [pi * 2, pi * 2 + 1] {
                let g = &by_idx[&gi];
                let row = json!({
                    "game_idx": g.game_idx,
                    "pair_idx": g.pair_idx,
                    "a_is_p1": g.a_is_p1,
                    "a_score": g.a_score,
                    "turns": g.turns,
                    "turn_cap": g.turn_cap,
                    "a_ns": g.a_ns,
                    "b_ns": g.b_ns,
                });
                serde_json::to_writer(&mut f, &row).unwrap();
                writeln!(f).unwrap();
            }
        }
        f.flush().unwrap();
    }
    let mut pair_scores = Vec::with_capacity(pairs);
    let (mut wins, mut losses, mut ties, mut turn_caps, mut turns_sum) = (0usize, 0usize, 0usize, 0usize, 0u64);
    let mut a_ns_all = Vec::new();
    let mut b_ns_all = Vec::new();
    for pi in pair_start..total_pairs_needed {
        let g0 = &by_idx[&(pi * 2)];
        let g1 = &by_idx[&(pi * 2 + 1)];
        pair_scores.push(0.5 * (g0.a_score + g1.a_score));
        for g in [g0, g1] {
            if g.a_score == 1.0 {
                wins += 1;
            } else if g.a_score == 0.0 {
                losses += 1;
            } else {
                ties += 1;
            }
            if g.turn_cap {
                turn_caps += 1;
            }
            turns_sum += g.turns as u64;
            a_ns_all.extend_from_slice(&g.a_ns);
            b_ns_all.extend_from_slice(&g.b_ns);
        }
    }
    let split_pairs = pair_scores.iter().filter(|&&s| s != 0.5).count();
    let (mean, ci_lo, ci_hi) = bootstrap_ci(&pair_scores, seed);
    let med_a = percentile_ms(&a_ns_all, 50);
    let med_b = percentile_ms(&b_ns_all, 50);
    println!(
        "{}",
        json!({
            "cmd": "blind-duel",
            "arm_a": arm_a,
            "arm_b": arm_b,
            "pairs": pairs,
            "games": total_games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "turn_caps": turn_caps,
            "split_pairs": split_pairs,
            "score": (mean * 10000.0).round() / 10000.0,
            "ci95": [(ci_lo * 10000.0).round() / 10000.0, (ci_hi * 10000.0).round() / 10000.0],
            "elo_diff": (elo_diff(mean) * 10.0).round() / 10.0,
            "avg_turns": (turns_sum as f64 / total_games as f64 * 10.0).round() / 10.0,
            "a_median_ms": (med_a * 100.0).round() / 100.0,
            "b_median_ms": (med_b * 100.0).round() / 100.0,
            "time_ratio": ((med_a / med_b.max(1e-9)) * 1000.0).round() / 1000.0,
            "a_p95_ms": (percentile_ms(&a_ns_all, 95) * 100.0).round() / 100.0,
            "b_p95_ms": (percentile_ms(&b_ns_all, 95) * 100.0).round() / 100.0,
            "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        })
    );
}

/// Generate strong `BlindAgent` self-play trajectories on `team-pool-v2` (with `belief-pool-v3`)
/// extracting the 20 strictly antisymmetric Stage-3 features at every multi-mon turn for offline fitting.
fn run_collect_stage3(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let games: usize = flag("--games").map_or(200, |v| v.parse().unwrap());
    let game_start: usize = flag("--game-start").map_or(0, |v| v.parse().unwrap());
    let iters: u32 = flag("--iters").map_or(3000, |v| v.parse().unwrap());
    let preview_iters: u32 = flag("--preview-iters").map_or(3000, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026112401, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let out_path = PathBuf::from(flag("--out").unwrap_or_else(|| "/tmp/stage3_data.jsonl".to_string()));
    let max_turns: u16 = flag("--max-turns").map_or(500, |v| v.parse().unwrap());

    let root = repo_root();
    let cat_txt = std::fs::read_to_string(root.join("data/team-pool-v2/team-pool.json"))
        .expect("read team-pool-v2");
    let catalog: WeightedCatalog = serde_json::from_str(&cat_txt).expect("parse team-pool-v2");
    let pool = Arc::new(load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json")));
    let weights: Vec<f64> = catalog.teams.iter().map(|t| t.draw_weight.unwrap_or(1.0)).collect();
    let total_w: f64 = weights.iter().sum();
    let sample_w = |rng: &mut SplitMix64| -> usize {
        let mut r = rng.next_f64() * total_w;
        for (i, &w) in weights.iter().enumerate() {
            if r < w {
                return i;
            }
            r -= w;
        }
        weights.len() - 1
    };

    struct Job {
        g: usize,
        ta: usize,
        tb: usize,
        bseed: String,
        s1: u64,
        s2: u64,
        eps_seed: u64,
    }
    let total_needed = game_start + games;
    let mut rng = SplitMix64::new(seed);
    let mut jobs = Vec::with_capacity(games);
    for g in 0..total_needed {
        let ta = sample_w(&mut rng);
        let tb = sample_w(&mut rng);
        let bseed = rng.battle_seed();
        let s1 = rng.next();
        let s2 = rng.next();
        let eps_seed = rng.next();
        if g >= game_start {
            jobs.push(Job { g, ta, tb, bseed, s1, s2, eps_seed });
        }
    }

    let t0 = Instant::now();
    let cursor = AtomicUsize::new(0);
    let rows_all: Vec<String> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(jobs.len()).max(1) {
            let (cat_r, pool_r, jobs_r, cur_r) = (&catalog, &pool, &jobs, &cursor);
            hs.push(s.spawn(move || {
                let mut local_lines = Vec::new();
                loop {
                    let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                    if idx >= jobs_r.len() {
                        break;
                    }
                    let job = &jobs_r[idx];
                    let mut b = Battle::from_fixture(
                        dex,
                        &job.bseed,
                        &cat_r.teams[job.ta].sets,
                        &cat_r.teams[job.tb].sets,
                    )
                    .unwrap();
                    b.set_log_enabled(true);
                    let cfg_p = RmConfig {
                        iterations: preview_iters,
                        rule: SelRule::Ucb,
                        c: 0.4,
                        ..Default::default()
                    };
                    let cfg_b = RmConfig {
                        iterations: iters,
                        rule: SelRule::Ucb,
                        c: 0.4,
                        ..Default::default()
                    };
                    let mut a0 = BlindAgent::new(cfg_b.clone(), pool_r.clone(), None, job.s1)
                        .with_preview_iterations(cfg_p.iterations);
                    let mut a1 = BlindAgent::new(cfg_b, pool_r.clone(), None, job.s2)
                        .with_preview_iterations(cfg_p.iterations);
                    let mut erng = SplitMix64::new(job.eps_seed);

                    // Resolve Team Preview first (with 15% random legal lead diversity for broad state coverage).
                    let tp0_cs = b.legal_choices(dex, 0);
                    let tp1_cs = b.legal_choices(dex, 1);
                    let tp0 = if erng.next_f64() < 0.15 {
                        tp0_cs[erng.below(tp0_cs.len())]
                    } else {
                        a0.choose(&b, dex, 0, &tp0_cs)
                    };
                    let tp1 = if erng.next_f64() < 0.15 {
                        tp1_cs[erng.below(tp1_cs.len())]
                    } else {
                        a1.choose(&b, dex, 1, &tp1_cs)
                    };
                    if b.apply_choices(dex, [Some(tp0), Some(tp1)]).is_err() {
                        continue;
                    }

                    let mut states: Vec<(u16, nc2000_bot::tempo::Stage3Features)> = Vec::new();
                    for _ in 0..max_turns {
                        if b.outcome().is_some() {
                            break;
                        }
                        // Record strictly AFTER the Turn 1–2 exploration window and ONLY on idle
                        // `[Move, Move]` turns (matching `8r` rollout cutoffs; excludes double-KO switches).
                        if b.turn >= 3 && b.is_idle_move_request() {
                            let (table, cb) = nc2000_engine::compact::CompactBattle::from_battle(&b, dex);
                            let n0 = cb.sides[0].alive_mask.count_ones();
                            let n1 = cb.sides[1].alive_mask.count_ones();
                            if n0 + n1 >= 3 {
                                let feat = nc2000_bot::tempo::extract_stage3_features(&cb, &table, None);
                                states.push((b.turn, feat));
                            }
                        }
                        let cs0 = b.legal_choices(dex, 0);
                        let cs1 = b.legal_choices(dex, 1);
                        let pick_safe_rand = |side: usize, cs: &[SearchChoice], erng: &mut SplitMix64| -> SearchChoice {
                            let safe: Vec<SearchChoice> = cs
                                .iter()
                                .copied()
                                .filter(|&c| !nc2000_bot::tempo::is_public_noop_choice(&b, dex, side, c))
                                .collect();
                            let pool_c = if safe.is_empty() { cs } else { &safe };
                            pool_c[erng.below(pool_c.len())]
                        };
                        let p0 = if cs0.is_empty() {
                            None
                        } else if b.turn <= 2 && cs0.len() > 1 && erng.next_f64() < 0.08 {
                            Some(pick_safe_rand(0, &cs0, &mut erng))
                        } else {
                            Some(a0.choose(&b, dex, 0, &cs0))
                        };
                        let p1 = if cs1.is_empty() {
                            None
                        } else if b.turn <= 2 && cs1.len() > 1 && erng.next_f64() < 0.08 {
                            Some(pick_safe_rand(1, &cs1, &mut erng))
                        } else {
                            Some(a1.choose(&b, dex, 1, &cs1))
                        };
                        if b.apply_choices(dex, [p0, p1]).is_err() {
                            break;
                        }
                    }
                    let y = match b.outcome() {
                        Some(Outcome::P1Win) => 1.0,
                        Some(Outcome::P2Win) => 0.0,
                        Some(Outcome::Tie) => 0.5,
                        None => continue,
                    };
                    let n_states = states.len();
                    let game_turns = b.turn;
                    for (turn, f) in states {
                        let row = json!({
                            "game": job.g,
                            "ta": job.ta,
                            "tb": job.tb,
                            "turn": turn,
                            "game_turns": game_turns,
                            "n_states": n_states,
                            "y": y,
                            "n0": f.n0,
                            "n1": f.n1,
                            "relay_elig": f.relay_elig,
                            "g_stag": (f64::from(f.g_stag) * 10000.0).round() / 10000.0,
                            "fg_p": (f.fg_p * 100000.0).round() / 100000.0,
                            "bs_p": (f.bs_p * 100000.0).round() / 100000.0,
                            "hy_p": (f.hy_p * 100000.0).round() / 100000.0,
                            "phi": f.phi.iter().map(|&x| (f64::from(x) * 100000.0).round() / 100000.0).collect::<Vec<f64>>(),
                        });
                        local_lines.push(serde_json::to_string(&row).unwrap());
                    }
                }
                local_lines
            }));
        }
        let mut merged = Vec::new();
        for h in hs {
            merged.extend(h.join().unwrap());
        }
        merged
    });

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap();
        }
    }
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
        for l in &rows_all {
            writeln!(f, "{l}").unwrap();
        }
        f.flush().unwrap();
    }
    println!(
        "{}",
        json!({
            "cmd": "collect-stage3",
            "games": games,
            "game_start": game_start,
            "iters": iters,
            "rows": rows_all.len(),
            "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        })
    );
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AzPolRec {
    team_hash: u64,
    opp_hash: u64,
    game_id: u32,
    turn: u16,
    n_acts: u8,
    noop_mask: u8,
    req: u8,
    side: u8,
    n0: u8,
    n1: u8,
    v_root: f32,
    ctx: [f32; nc2000_bot::compact_az::AZ_POLICY_CTX_DIM],
    af: [[f32; nc2000_bot::compact_az::AZ_POLICY_ACT_DIM]; 6],
    visits: [u32; 6],
    q: [f32; 6],
}
const _: () = assert!(std::mem::size_of::<AzPolRec>() == 960);

#[repr(C)]
#[derive(Clone, Copy)]
struct AzValRec {
    team_hash_0: u64,
    team_hash_1: u64,
    game_id: u32,
    turn: u16,
    n0: u8,
    n1: u8,
    logit_fg: f32,
    logit_bs: f32,
    v_search: f32,
    z: f32,
    x0: [f32; nc2000_bot::compact_az::AZ_VALUE_SIDE_DIM],
    x1: [f32; nc2000_bot::compact_az::AZ_VALUE_SIDE_DIM],
}
const _: () = assert!(std::mem::size_of::<AzValRec>() == 840);

fn canonical_team_hash(sets: &[PokemonSet]) -> u64 {
    let mut keys: Vec<String> = sets
        .iter()
        .map(|s| {
            let mut mv = s.moves.clone();
            mv.sort();
            format!("{}|{}|{}|{}", s.species, s.level, s.item, mv.join(","))
        })
        .collect();
    keys.sort();
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for k in &keys {
        for b in k.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x1000_0000_01B3);
        }
        h ^= 0xFF;
        h = h.wrapping_mul(0x1000_0000_01B3);
    }
    h
}

fn write_pod_slice<T: Copy>(path: &PathBuf, items: &[T]) {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap();
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let byte_len = std::mem::size_of_val(items);
    if byte_len > 0 {
        let bytes = unsafe { std::slice::from_raw_parts(items.as_ptr() as *const u8, byte_len) };
        f.write_all(bytes).unwrap();
    }
    f.flush().unwrap();
}

/// Fast unified policy+value collector using `CompactBattle`-native full-info teacher (`skuct` or `az`)
/// across 50% `team-pool-v2` (drawWeight) + 50% `belief-pool-v3` (520 diverse teams).
/// Outputs zero-copy binary shards `<out_prefix>.pol.bin` and `<out_prefix>.val.bin`.
fn run_collect_az(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let games: usize = flag("--games").map_or(200, |v| v.parse().unwrap());
    let game_start: usize = flag("--game-start").map_or(0, |v| v.parse().unwrap());
    let iters: u32 = flag("--iters").map_or(12_000, |v| v.parse().unwrap());
    let mode = flag("--mode").unwrap_or_else(|| "skuct".to_string());
    let az_rollout: u16 = flag("--rollout").map_or(0, |v| v.parse().unwrap());
    let az_cp: f64 = flag("--cp").map_or(1.0, |v| v.parse().unwrap());
    let az_fpu: f64 = flag("--fpu").map_or(0.09, |v| v.parse().unwrap());
    let az_lam: f64 = flag("--lam").map_or(0.0, |v| v.parse().unwrap());
    let az_tau: f64 = flag("--tau").map_or(4.0, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026112501, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let out_prefix = flag("--out").unwrap_or_else(|| "/tmp/az_data".to_string());
    let max_turns: u16 = flag("--max-turns").map_or(250, |v| v.parse().unwrap());

    let root = repo_root();
    let cat_txt = std::fs::read_to_string(root.join("data/team-pool-v2/team-pool.json"))
        .expect("read team-pool-v2");
    let catalog: WeightedCatalog = serde_json::from_str(&cat_txt).expect("parse team-pool-v2");
    let bpool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let tp_hashes: Vec<u64> = catalog.teams.iter().map(|t| canonical_team_hash(&t.sets)).collect();
    let bp_hashes: Vec<u64> = bpool.teams.iter().map(|t| canonical_team_hash(&t.sets)).collect();
    let weights: Vec<f64> = catalog.teams.iter().map(|t| t.draw_weight.unwrap_or(1.0)).collect();
    let total_w: f64 = weights.iter().sum();
    let sample_tp = |rng: &mut SplitMix64| -> usize {
        let mut r = rng.next_f64() * total_w;
        for (i, &w) in weights.iter().enumerate() {
            if r < w {
                return i;
            }
            r -= w;
        }
        weights.len() - 1
    };

    struct Job {
        g: usize,
        use_bp_a: bool,
        ta: usize,
        use_bp_b: bool,
        tb: usize,
        bseed: String,
        s_seed: u64,
    }
    let total_needed = game_start + games;
    let mut rng = SplitMix64::new(seed);
    let mut jobs = Vec::with_capacity(games);
    for g in 0..total_needed {
        // 50% team-pool-v2, 50% belief-pool-v3 (520 diverse teams) per side independently.
        let use_bp_a = rng.next_f64() < 0.50;
        let ta = if use_bp_a { rng.below(bpool.teams.len()) } else { sample_tp(&mut rng) };
        let use_bp_b = rng.next_f64() < 0.50;
        let tb = if use_bp_b { rng.below(bpool.teams.len()) } else { sample_tp(&mut rng) };
        let bseed = rng.battle_seed();
        let s_seed = rng.next();
        if g >= game_start {
            jobs.push(Job { g, use_bp_a, ta, use_bp_b, tb, bseed, s_seed });
        }
    }

    let t0 = Instant::now();
    let cursor = AtomicUsize::new(0);
    let mode_az = mode == "az";
    let (all_pol, all_val, finished_games) = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(jobs.len()).max(1) {
            let (cat_r, bp_r, tph_r, bph_r, jobs_r, cur_r) =
                (&catalog, &bpool, &tp_hashes, &bp_hashes, &jobs, &cursor);
            hs.push(s.spawn(move || {
                let mut pols: Vec<AzPolRec> = Vec::new();
                let mut vals: Vec<AzValRec> = Vec::new();
                let mut done_g = 0usize;
                loop {
                    let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                    if idx >= jobs_r.len() {
                        break;
                    }
                    let job = &jobs_r[idx];
                    let sets_a = if job.use_bp_a { &bp_r.teams[job.ta].sets } else { &cat_r.teams[job.ta].sets };
                    let sets_b = if job.use_bp_b { &bp_r.teams[job.tb].sets } else { &cat_r.teams[job.tb].sets };
                    let th0 = if job.use_bp_a { bph_r[job.ta] } else { tph_r[job.ta] };
                    let th1 = if job.use_bp_b { bph_r[job.tb] } else { tph_r[job.tb] };

                    let mut b = match Battle::from_fixture(dex, &job.bseed, sets_a, sets_b) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    b.set_log_enabled(false);
                    let mut grng = SplitMix64::new(job.s_seed);

                    // Fast realistic preview: 35% uniform legal triple, 65% best-of-16 FastGolf pick.
                    let tp0_cs = b.legal_choices(dex, 0);
                    let tp1_cs = b.legal_choices(dex, 1);
                    let pick_tp = |side: usize, cs: &[SearchChoice], opp_cs: &[SearchChoice], grng: &mut SplitMix64| -> SearchChoice {
                        if cs.len() <= 1 || grng.next_f64() < 0.35 {
                            return cs[grng.below(cs.len())];
                        }
                        let opp_sample = [
                            opp_cs[grng.below(opp_cs.len())],
                            opp_cs[grng.below(opp_cs.len())],
                        ];
                        let mut best_c = cs[0];
                        let mut best_sc = f64::NEG_INFINITY;
                        for _ in 0..12 {
                            let cand = cs[grng.below(cs.len())];
                            let mut sum = 0.0;
                            for &oc in &opp_sample {
                                let mut sim = b.clone();
                                let picks = if side == 0 { [Some(cand), Some(oc)] } else { [Some(oc), Some(cand)] };
                                if sim.apply_choices(dex, picks).is_ok() {
                                    let (tbl, cb) = nc2000_engine::compact::CompactBattle::from_battle(&sim, dex);
                                    let fg = nc2000_bot::tempo::evaluate_fast_golf(&cb, &tbl, None);
                                    sum += if side == 0 { fg.win_prob } else { 1.0 - fg.win_prob };
                                }
                            }
                            if sum > best_sc {
                                best_sc = sum;
                                best_c = cand;
                            }
                        }
                        best_c
                    };
                    let tp0 = pick_tp(0, &tp0_cs, &tp1_cs, &mut grng);
                    let tp1 = pick_tp(1, &tp1_cs, &tp0_cs, &mut grng);
                    if b.apply_choices(dex, [Some(tp0), Some(tp1)]).is_err() {
                        continue;
                    }
                    if !nc2000_engine::compact::is_compact_tree_supported(&b, dex) {
                        continue;
                    }
                    let (table, mut cb) = nc2000_engine::compact::CompactBattle::from_battle(&b, dex);
                    let cache = nc2000_bot::tempo::TempoMatchCache::build(&cb, &table);

                    let mut g_pols: Vec<AzPolRec> = Vec::new();
                    let mut g_vals: Vec<AzValRec> = Vec::new();
                    let mut outcome_opt: Option<f32> = None;

                    for _ in 0..(max_turns as usize * 3) {
                        if cb.outcome != 0 {
                            outcome_opt = Some(match cb.outcome {
                                1 => 1.0,
                                2 => 0.0,
                                _ => 0.5,
                            });
                            break;
                        }
                        if let Some(v) = nc2000_bot::eval::compact_turn_limit_outcome(&cb, &table) {
                            outcome_opt = Some(v as f32);
                            break;
                        }
                        if cb.turn > max_turns {
                            break;
                        }
                        let mut a_buf = [[0u8; 7]; 2];
                        let n0_a = if cb.sides[0].request > 0 { cb.legal_actions(&table, 0, &mut a_buf[0]) } else { 0 };
                        let n1_a = if cb.sides[1].request > 0 { cb.legal_actions(&table, 1, &mut a_buf[1]) } else { 0 };
                        if n0_a == 0 && n1_a == 0 {
                            break;
                        }
                        if n0_a <= 1 && n1_a <= 1 {
                            let p0 = if n0_a == 1 { a_buf[0][0] } else { 7 };
                            let p1 = if n1_a == 1 { a_buf[1][0] } else { 7 };
                            cb.apply_actions(&table, [p0, p1]);
                            continue;
                        }

                        let step_seed = grng.next();
                        let res = if mode_az {
                            nc2000_bot::compact_az::search_compact_az_teacher(
                                &cb, &table, &cache, iters, az_rollout, az_cp, az_fpu, 0.02, az_lam, az_tau, step_seed,
                            )
                        } else {
                            nc2000_bot::compact_az::search_compact_teacher(&cb, &table, iters, step_seed)
                        };

                        let alive0 = cb.sides[0].alive_mask.count_ones() as u8;
                        let alive1 = cb.sides[1].alive_mask.count_ones() as u8;

                        // Record Policy samples for each acting side with >= 2 choices.
                        for side in 0..2usize {
                            let k = (res.n_acts[side] as usize).min(6);
                            if k < 2 {
                                continue;
                            }
                            let noop = nc2000_bot::compact_az::compact_interior_noop_mask(
                                &cb, &table, side, &res.acts[side], k,
                            );
                            let (ctx, af7) = nc2000_bot::compact_az::extract_policy_features(
                                &cb, &table, Some(&cache), side, &res.acts[side], k,
                            );
                            let mut af6 = [[0.0f32; nc2000_bot::compact_az::AZ_POLICY_ACT_DIM]; 6];
                            af6.copy_from_slice(&af7[..6]);
                            let mut vis6 = [0u32; 6];
                            vis6.copy_from_slice(&res.visits[side][..6]);
                            let mut q6 = [0.5f32; 6];
                            for i in 0..k {
                                q6[i] = res.q_side[side][i] as f32;
                            }
                            let (th_me, th_opp) = if side == 0 { (th0, th1) } else { (th1, th0) };
                            g_pols.push(AzPolRec {
                                team_hash: th_me,
                                opp_hash: th_opp,
                                game_id: job.g as u32,
                                turn: cb.turn,
                                n_acts: k as u8,
                                noop_mask: noop,
                                req: cb.sides[side].request,
                                side: side as u8,
                                n0: alive0,
                                n1: alive1,
                                v_root: (if side == 0 { res.v_root0 } else { 1.0 - res.v_root0 }) as f32,
                                ctx,
                                af: af6,
                                visits: vis6,
                                q: q6,
                            });
                        }

                        // Record Value sample on idle multi-mon states (n0 + n1 >= 3).
                        if cb.sides[0].request == 1 && cb.sides[1].request == 1 && alive0 + alive1 >= 3 {
                            let (x0, x1, lfg, lbs) =
                                nc2000_bot::compact_az::extract_value_features(&cb, &table, Some(&cache));
                            g_vals.push(AzValRec {
                                team_hash_0: th0,
                                team_hash_1: th1,
                                game_id: job.g as u32,
                                turn: cb.turn,
                                n0: alive0,
                                n1: alive1,
                                logit_fg: lfg,
                                logit_bs: lbs,
                                v_search: res.v_root0 as f32,
                                z: 0.5,
                                x0,
                                x1,
                            });
                        }

                        // Execute moves: proportional-to-visits^0.65 sampling on 12% of Turns 1..=16
                        // (tactical diversity without uniform blunders corrupting outcome z).
                        let mut play_acts = [7u8; 2];
                        for side in 0..2usize {
                            let k = (res.n_acts[side] as usize).min(6);
                            if k == 0 {
                                continue;
                            }
                            if k == 1 {
                                play_acts[side] = res.acts[side][0];
                                continue;
                            }
                            if cb.turn <= 12 && grng.next_f64() < 0.12 {
                                let noop = nc2000_bot::compact_az::compact_interior_noop_mask(
                                    &cb, &table, side, &res.acts[side], k,
                                );
                                let mut wts = [0.0f64; 6];
                                let mut w_sum = 0.0f64;
                                for i in 0..k {
                                    if ((noop >> i) & 1) == 0 && res.visits[side][i] > 0 {
                                        let wt = f64::from(res.visits[side][i]).powf(0.65);
                                        wts[i] = wt;
                                        w_sum += wt;
                                    }
                                }
                                if w_sum > 0.0 {
                                    let mut r = grng.next_f64() * w_sum;
                                    let mut picked = res.best_acts[side];
                                    for i in 0..k {
                                        if wts[i] > 0.0 {
                                            r -= wts[i];
                                            if r <= 0.0 {
                                                picked = res.acts[side][i];
                                                break;
                                            }
                                        }
                                    }
                                    play_acts[side] = picked;
                                } else {
                                    play_acts[side] = res.best_acts[side];
                                }
                            } else {
                                play_acts[side] = res.best_acts[side];
                            }
                        }
                        cb.apply_actions(&table, play_acts);
                    }

                    if let Some(z) = outcome_opt {
                        done_g += 1;
                        pols.extend_from_slice(&g_pols);
                        for mut v in g_vals {
                            v.z = z;
                            vals.push(v);
                        }
                    }
                }
                (pols, vals, done_g)
            }));
        }
        let mut ap = Vec::new();
        let mut av = Vec::new();
        let mut dg = 0usize;
        for h in hs {
            let (p, v, d) = h.join().unwrap();
            ap.extend(p);
            av.extend(v);
            dg += d;
        }
        (ap, av, dg)
    });

    write_pod_slice(&PathBuf::from(format!("{out_prefix}.pol.bin")), &all_pol);
    write_pod_slice(&PathBuf::from(format!("{out_prefix}.val.bin")), &all_val);
    println!(
        "{}",
        json!({
            "cmd": "collect-az",
            "mode": mode,
            "games": games,
            "finished_games": finished_games,
            "game_start": game_start,
            "iters": iters,
            "pol_rows": all_pol.len(),
            "val_rows": all_val.len(),
            "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        })
    );
}

/// Fast 2-minute calibration probe comparing candidate rule-engine-guaranteed `CF-Oracle`
/// continuation policies against `V_ref` (`M = 16` full `bs-8r@puct:1500` games to exact `cb.outcome`)
/// on unseen `belief-pool-v3` states + CRN state-perturbation pairs `(s, do(Δs)(s))`.
fn run_cf_probe(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let n_states: usize = flag("--states").map_or(120, |v| v.parse().unwrap());
    let ref_m: usize = flag("--ref-m").map_or(12, |v| v.parse().unwrap());
    let ref_iters: u32 = flag("--ref-iters").map_or(1000, |v| v.parse().unwrap());
    let cand_m: usize = flag("--cand-m").map_or(8, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026112601, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());

    let root = repo_root();
    let bpool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let mut rng = SplitMix64::new(seed);

    // 1. Sample `n_states` realistic midgame states (100% belief-pool-v3) + CRN perturbation `s'`.
    struct ProbeItem {
        table: nc2000_engine::compact::MatchTable,
        cb: nc2000_engine::compact::CompactBattle,
        cb_p: nc2000_engine::compact::CompactBattle,
        kind: u8,
        pair_base_seed: u64,
        is_stall: bool,
    }
    let mut items: Vec<ProbeItem> = Vec::with_capacity(n_states);
    while items.len() < n_states {
        let ta = rng.below(bpool.teams.len());
        let tb = rng.below(bpool.teams.len());
        let bseed = rng.battle_seed();
        let Ok(mut b) = Battle::from_fixture(dex, &bseed, &bpool.teams[ta].sets, &bpool.teams[tb].sets) else {
            continue;
        };
        b.set_log_enabled(false);
        let tp0_cs = b.legal_choices(dex, 0);
        let tp1_cs = b.legal_choices(dex, 1);
        let tp0 = tp0_cs[rng.below(tp0_cs.len())];
        let tp1 = tp1_cs[rng.below(tp1_cs.len())];
        if b.apply_choices(dex, [Some(tp0), Some(tp1)]).is_err()
            || !nc2000_engine::compact::is_compact_tree_supported(&b, dex)
        {
            continue;
        }
        let (table, mut cb) = nc2000_engine::compact::CompactBattle::from_battle(&b, dex);
        let cache = nc2000_bot::tempo::TempoMatchCache::build(&cb, &table);
        let target_turn = (3 + rng.below(26)) as u16;
        for _ in 0..80 {
            if cb.outcome != 0 || cb.turn >= target_turn {
                break;
            }
            let mut a0 = [0u8; 7];
            let mut a1 = [0u8; 7];
            let n0 = if cb.sides[0].request > 0 { cb.legal_actions(&table, 0, &mut a0).min(6) } else { 0 };
            let n1 = if cb.sides[1].request > 0 { cb.legal_actions(&table, 1, &mut a1).min(6) } else { 0 };
            if n0 == 0 && n1 == 0 {
                break;
            }
            let sseed = rng.next();
            let res = nc2000_bot::compact_az::search_compact_puct_bs(&cb, &table, &cache, 200, sseed);
            cb.apply_actions(&table, res.best_acts);
        }
        if cb.outcome != 0 || cb.sides[0].request != 1 || cb.sides[1].request != 1 {
            continue;
        }
        let n0 = cb.sides[0].alive_mask.count_ones();
        let n1 = cb.sides[1].alive_mask.count_ones();
        if n0 + n1 < 3 {
            continue;
        }
        let Some((cb_p, kind)) = nc2000_bot::compact_az::apply_cf_perturbation(&cb, &table, &mut rng) else {
            continue;
        };
        let (x0, _, _, _) = nc2000_bot::compact_az::extract_value_features(&cb, &table, Some(&cache));
        let is_stall = x0[91] > 0.30;
        items.push(ProbeItem {
            table,
            cb,
            cb_p,
            kind,
            pair_base_seed: rng.next(),
            is_stall,
        });
    }

    // 2. Evaluate `V_ref`, baselines, and 5 candidate `CF-Oracle` continuation policies in parallel.
    #[derive(Clone, Default)]
    struct Row {
        is_stall: bool,
        kind: u8,
        v_ref: f64,
        vp_ref: f64,
        hy: f64,
        hyp: f64,
        azv: f64,
        azvp: f64,
        bs: f64,
        bsp: f64,
        // 5 candidates: [pnet_t0.35, pnet_t0.50, pnet_t0.70, two_tier_200_4t_t0.50, puct_bs_200_full]
        c_v: [f64; 5],
        c_vp: [f64; 5],
        c_turns: [f64; 5],
        c_caps: [u32; 5],
        c_us: [f64; 5],
    }

    let cands: [(&str, u16, u32, f32, usize, bool); 5] = [
        ("pnet_t050_m64", 0, 0, 0.50, 64, false),
        ("2tier_bs_64x2t_m32", 2, 64, 0.50, 32, false),
        ("2tier_cfv_64x2t_m32", 2, 64, 0.50, 32, true),
        ("2tier_cfv_96x4t_m32", 4, 96, 0.50, 32, true),
        ("puct_cfv_128_full_m16", 200, 128, 0.50, 16, true),
    ];

    let cursor = AtomicUsize::new(0);
    let t_start = Instant::now();
    let rows: Vec<Row> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(items.len()).max(1) {
            let (it_r, cur_r) = (&items, &cursor);
            hs.push(s.spawn(move || {
                let mut local = Vec::new();
                let ew = nc2000_bot::EvalWeights::default();
                loop {
                    let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                    if idx >= it_r.len() {
                        break;
                    }
                    let it = &it_r[idx];
                    let cache = nc2000_bot::tempo::TempoMatchCache::build(&it.cb, &it.table);
                    let mut r = Row {
                        is_stall: it.is_stall,
                        kind: it.kind,
                        hy: nc2000_bot::tempo::evaluate_hybrid_golf(&it.cb, &it.table, Some(&cache)),
                        hyp: nc2000_bot::tempo::evaluate_hybrid_golf(&it.cb_p, &it.table, Some(&cache)),
                        azv: nc2000_bot::compact_az::evaluate_az_value(&it.cb, &it.table, Some(&cache), 1.0),
                        azvp: nc2000_bot::compact_az::evaluate_az_value(&it.cb_p, &it.table, Some(&cache), 1.0),
                        bs: nc2000_bot::eval::compact_eval_leaf_blindspot(&it.cb, &it.table, &ew),
                        bsp: nc2000_bot::eval::compact_eval_leaf_blindspot(&it.cb_p, &it.table, &ew),
                        ..Row::default()
                    };
                    // Ground-truth reference: `ref_m` CRN-paired full games (`bs-8r@puct:ref_iters`)
                    let mut sum_r = 0.0;
                    let mut sum_rp = 0.0;
                    for m in 0..ref_m {
                        let pseed = it.pair_base_seed ^ (m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        let (v0, _, _) = nc2000_bot::compact_az::cf_rollout_to_outcome(
                            &it.cb, &it.table, &cache, 200, ref_iters, 0.35, 200, pseed,
                        );
                        let (v1, _, _) = nc2000_bot::compact_az::cf_rollout_to_outcome(
                            &it.cb_p, &it.table, &cache, 200, ref_iters, 0.35, 200, pseed,
                        );
                        sum_r += v0;
                        sum_rp += v1;
                    }
                    r.v_ref = sum_r / (ref_m as f64);
                    r.vp_ref = sum_rp / (ref_m as f64);

                    for (ci, &(_, st, si, temp, cm, u_cfv)) in cands.iter().enumerate() {
                        let tc0 = Instant::now();
                        let mut s0 = 0.0;
                        let mut s1 = 0.0;
                        let mut tsum = 0.0;
                        let mut caps = 0u32;
                        for m in 0..cm {
                            let pseed = it.pair_base_seed
                                ^ ((ci as u64 + 1) * 0xBF58_476D_1CE4_E5B9)
                                ^ (m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                            let (v0, dt0, c0) = nc2000_bot::compact_az::cf_rollout_to_outcome_mode(
                                &it.cb, &it.table, &cache, st, si, temp, 200, pseed, u_cfv,
                            );
                            let (v1, dt1, c1) = nc2000_bot::compact_az::cf_rollout_to_outcome_mode(
                                &it.cb_p, &it.table, &cache, st, si, temp, 200, pseed, u_cfv,
                            );
                            s0 += v0;
                            s1 += v1;
                            tsum += f64::from(dt0) + f64::from(dt1);
                            caps += u32::from(c0) + u32::from(c1);
                        }
                        r.c_v[ci] = s0 / (cm as f64);
                        r.c_vp[ci] = s1 / (cm as f64);
                        r.c_turns[ci] = tsum / ((2 * cm) as f64);
                        r.c_caps[ci] = caps;
                        r.c_us[ci] = tc0.elapsed().as_secs_f64() * 1e6; // total us per CRN pair!
                    }
                    local.push(r);
                }
                local
            }));
        }
        let mut merged = Vec::with_capacity(items.len());
        for h in hs {
            merged.extend(h.join().unwrap());
        }
        merged
    });

    let corr = |xs: &[f64], ys: &[f64]| -> f64 {
        let n = xs.len() as f64;
        if n < 2.0 { return 0.0; }
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let (mut num, mut vx, mut vy) = (0.0, 0.0, 0.0);
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let dx = x - mx;
            let dy = y - my;
            num += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
        if vx > 1e-12 && vy > 1e-12 { num / (vx.sqrt() * vy.sqrt()) } else { 0.0 }
    };

    let summarize = |name: &str, pred: &[(f64, f64)], pair_ms: f64, avg_turns: f64, cap_rate: f64| -> Value {
        let mut xs = Vec::with_capacity(rows.len() * 2);
        let mut ys = Vec::with_capacity(rows.len() * 2);
        let mut d_pred = Vec::with_capacity(rows.len());
        let mut d_ref = Vec::with_capacity(rows.len());
        let (mut sig_ok, mut sig_tie, mut sig_tot) = (0usize, 0usize, 0usize);
        let mut mse = 0.0;
        for (r, &(p0, p1)) in rows.iter().zip(pred.iter()) {
            xs.push(p0);
            xs.push(p1);
            ys.push(r.v_ref);
            ys.push(r.vp_ref);
            mse += (p0 - r.v_ref).powi(2) + (p1 - r.vp_ref).powi(2);
            let dp = p1 - p0;
            let dr = r.vp_ref - r.v_ref;
            d_pred.push(dp);
            d_ref.push(dr);
            if dr.abs() >= 0.15 {
                sig_tot += 1;
                if (dp > 1e-9 && dr > 0.0) || (dp < -1e-9 && dr < 0.0) {
                    sig_ok += 1;
                } else if dp.abs() <= 1e-9 {
                    sig_tie += 1;
                }
            }
        }
        let rmse = (mse / (xs.len().max(1) as f64)).sqrt();
        let non_tie = sig_tot.saturating_sub(sig_tie);
        json!({
            "name": name,
            "r_val": (corr(&xs, &ys) * 10000.0).round() / 10000.0,
            "r_delta": (corr(&d_pred, &d_ref) * 10000.0).round() / 10000.0,
            "sign_acc": if non_tie > 0 { ((sig_ok as f64 / non_tie as f64) * 1000.0).round() / 10.0 } else { 0.0 },
            "sign_ok_tie_tot": [sig_ok, sig_tie, sig_tot],
            "rmse": (rmse * 10000.0).round() / 10000.0,
            "pair_ms": (pair_ms * 100.0).round() / 100.0,
            "avg_turns": (avg_turns * 10.0).round() / 10.0,
            "cap_pct": (cap_rate * 1000.0).round() / 10.0,
        })
    };

    let mut report = Vec::new();
    let bs_p: Vec<(f64, f64)> = rows.iter().map(|r| (r.bs, r.bsp)).collect();
    let hy_p: Vec<(f64, f64)> = rows.iter().map(|r| (r.hy, r.hyp)).collect();
    let az_p: Vec<(f64, f64)> = rows.iter().map(|r| (r.azv, r.azvp)).collect();
    report.push(summarize("baseline_blindspot", &bs_p, 0.0, 0.0, 0.0));
    report.push(summarize("baseline_hybrid_golf", &hy_p, 0.0, 0.0, 0.0));
    report.push(summarize("baseline_gen1_azv", &az_p, 0.0, 0.0, 0.0));
    for (ci, &(cname, _, _, _, cm, _)) in cands.iter().enumerate() {
        let cp: Vec<(f64, f64)> = rows.iter().map(|r| (r.c_v[ci], r.c_vp[ci])).collect();
        let avg_ms = rows.iter().map(|r| r.c_us[ci]).sum::<f64>() / (rows.len().max(1) as f64) / 1000.0;
        let avg_t = rows.iter().map(|r| r.c_turns[ci]).sum::<f64>() / (rows.len().max(1) as f64);
        let cap_r = rows.iter().map(|r| r.c_caps[ci] as f64).sum::<f64>() / ((rows.len() * 2 * cm).max(1) as f64);
        report.push(summarize(cname, &cp, avg_ms, avg_t, cap_r));
    }
    let stall_cnt = rows.iter().filter(|r| r.is_stall).count();
    println!(
        "{}",
        json!({
            "cmd": "cf-probe",
            "states": rows.len(),
            "stall_states": stall_cnt,
            "ref_m": ref_m,
            "ref_iters": ref_iters,
            "cand_m": cand_m,
            "wall_s": (t_start.elapsed().as_secs_f64() * 10.0).round() / 10.0,
            "results": report,
        })
    );
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CfPairRec {
    team_hash_0: u64,
    team_hash_1: u64,
    game_id: u32,
    turn: u16,
    n0: u8,
    n1: u8,
    kind: u8,
    _pad: [u8; 3],
    v_cf: f32,
    vp_cf: f32,
    z_game: f32,
    x0: [f32; nc2000_bot::compact_az::CF_RAW_VALUE_SIDE_DIM],
    x1: [f32; nc2000_bot::compact_az::CF_RAW_VALUE_SIDE_DIM],
    xp0: [f32; nc2000_bot::compact_az::CF_RAW_VALUE_SIDE_DIM],
    xp1: [f32; nc2000_bot::compact_az::CF_RAW_VALUE_SIDE_DIM],
}
const _: () = assert!(
    std::mem::size_of::<CfPairRec>() == 40 + 4 * 4 * nc2000_bot::compact_az::CF_RAW_VALUE_SIDE_DIM
);

/// Counterfactual Rule-Engine-Guaranteed CRN Pair & Action-Intervention Collector (`collect-cf`).
/// Uses 100% `belief-pool-v3` (ZERO `team-pool-v2` overlap!). Teams with `hash % 5 != 0` are
/// training data; `hash % 5 == 0` are strictly held out for offline unseen-team certification.
fn run_collect_cf(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let games: usize = flag("--games").map_or(2000, |v| v.parse().unwrap());
    let game_start: usize = flag("--game-start").map_or(0, |v| v.parse().unwrap());
    let states_per_game: usize = flag("--states-per-game").map_or(6, |v| v.parse().unwrap());
    let crn_m: usize = flag("--crn-m").map_or(32, |v| v.parse().unwrap());
    let search_turns: u16 = flag("--search-turns").map_or(4, |v| v.parse().unwrap());
    let search_iters: u32 = flag("--search-iters").map_or(96, |v| v.parse().unwrap());
    let pnet_temp: f32 = flag("--pnet-temp").map_or(0.50, |v| v.parse().unwrap());
    let play_iters: u32 = flag("--play-iters").map_or(200, |v| v.parse().unwrap());
    let act_m: usize = flag("--act-m").map_or(24, |v| v.parse().unwrap());
    let use_cfv = args.iter().any(|a| a == "--use-cfv");
    let seed: u64 = flag("--seed").map_or(2026112701, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let out_prefix = flag("--out").unwrap_or_else(|| "/tmp/cf_data".to_string());

    let root = repo_root();
    let bpool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let bp_hashes: Vec<u64> = bpool.teams.iter().map(|t| canonical_team_hash(&t.sets)).collect();

    struct Job {
        g: usize,
        ta: usize,
        tb: usize,
        bseed: String,
        s_seed: u64,
    }
    let total_needed = game_start + games;
    let mut rng = SplitMix64::new(seed);
    let mut jobs = Vec::with_capacity(games);
    for g in 0..total_needed {
        let ta = rng.below(bpool.teams.len());
        let tb = rng.below(bpool.teams.len());
        let bseed = rng.battle_seed();
        let s_seed = rng.next();
        if g >= game_start {
            jobs.push(Job { g, ta, tb, bseed, s_seed });
        }
    }

    let t0 = Instant::now();
    let cursor = AtomicUsize::new(0);
    let (all_pairs, all_pols, finished_games) = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(jobs.len()).max(1) {
            let (bp_r, bph_r, jobs_r, cur_r) = (&bpool, &bp_hashes, &jobs, &cursor);
            hs.push(s.spawn(move || {
                let mut pairs_out: Vec<CfPairRec> = Vec::new();
                let mut pols_out: Vec<AzPolRec> = Vec::new();
                let mut done_g = 0usize;
                loop {
                    let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                    if idx >= jobs_r.len() {
                        break;
                    }
                    let job = &jobs_r[idx];
                    let sets_a = &bp_r.teams[job.ta].sets;
                    let sets_b = &bp_r.teams[job.tb].sets;
                    let th0 = bph_r[job.ta];
                    let th1 = bph_r[job.tb];
                    let Ok(mut b) = Battle::from_fixture(dex, &job.bseed, sets_a, sets_b) else {
                        continue;
                    };
                    b.set_log_enabled(false);
                    let mut grng = SplitMix64::new(job.s_seed);

                    // Realistic diverse Team Preview (35% uniform, 65% best-of-12 FastGolf).
                    let tp0_cs = b.legal_choices(dex, 0);
                    let tp1_cs = b.legal_choices(dex, 1);
                    let pick_tp = |side: usize, cs: &[SearchChoice], opp_cs: &[SearchChoice], grng: &mut SplitMix64| -> SearchChoice {
                        if cs.len() <= 1 || grng.next_f64() < 0.35 {
                            return cs[grng.below(cs.len())];
                        }
                        let opp_sample = [opp_cs[grng.below(opp_cs.len())], opp_cs[grng.below(opp_cs.len())]];
                        let mut best_c = cs[0];
                        let mut best_sc = f64::NEG_INFINITY;
                        for _ in 0..10 {
                            let cand = cs[grng.below(cs.len())];
                            let mut sum = 0.0;
                            for &oc in &opp_sample {
                                let mut sim = b.clone();
                                let picks = if side == 0 { [Some(cand), Some(oc)] } else { [Some(oc), Some(cand)] };
                                if sim.apply_choices(dex, picks).is_ok() {
                                    let (tbl, cb) = nc2000_engine::compact::CompactBattle::from_battle(&sim, dex);
                                    let fg = nc2000_bot::tempo::evaluate_fast_golf(&cb, &tbl, None);
                                    sum += if side == 0 { fg.win_prob } else { 1.0 - fg.win_prob };
                                }
                            }
                            if sum > best_sc {
                                best_sc = sum;
                                best_c = cand;
                            }
                        }
                        best_c
                    };
                    let tp0 = pick_tp(0, &tp0_cs, &tp1_cs, &mut grng);
                    let tp1 = pick_tp(1, &tp1_cs, &tp0_cs, &mut grng);
                    if b.apply_choices(dex, [Some(tp0), Some(tp1)]).is_err()
                        || !nc2000_engine::compact::is_compact_tree_supported(&b, dex)
                    {
                        continue;
                    }
                    let (table, mut cb) = nc2000_engine::compact::CompactBattle::from_battle(&b, dex);
                    let cache = nc2000_bot::tempo::TempoMatchCache::build(&cb, &table);

                    // Play out full trajectory with `play_iters` search (`--use-cfv` or `bs-8r@puct`) + early exploration,
                    // recording candidate idle multi-mon states (including late & stall states; cap raised to 360 turns!).
                    let mut snapshots: Vec<(nc2000_engine::compact::CompactBattle, bool)> = Vec::with_capacity(64);
                    let mut outcome_opt: Option<f32> = None;
                    for _ in 0..900 {
                        if cb.outcome != 0 {
                            outcome_opt = Some(match cb.outcome { 1 => 1.0, 2 => 0.0, _ => 0.5 });
                            break;
                        }
                        if let Some(v) = nc2000_bot::eval::compact_turn_limit_outcome(&cb, &table) {
                            outcome_opt = Some(v as f32);
                            break;
                        }
                        if cb.turn > 360 {
                            break;
                        }
                        let mut a_buf = [[0u8; 7]; 2];
                        let n0 = if cb.sides[0].request > 0 { cb.legal_actions(&table, 0, &mut a_buf[0]).min(6) } else { 0 };
                        let n1 = if cb.sides[1].request > 0 { cb.legal_actions(&table, 1, &mut a_buf[1]).min(6) } else { 0 };
                        if n0 == 0 && n1 == 0 {
                            break;
                        }
                        if n0 <= 1 && n1 <= 1 {
                            let p0 = if n0 == 1 { a_buf[0][0] } else { 7 };
                            let p1 = if n1 == 1 { a_buf[1][0] } else { 7 };
                            cb.apply_actions(&table, [p0, p1]);
                            continue;
                        }
                        let al0 = cb.sides[0].alive_mask.count_ones();
                        let al1 = cb.sides[1].alive_mask.count_ones();
                        if cb.sides[0].request == 1 && cb.sides[1].request == 1 && al0 + al1 >= 4 && cb.turn >= 2 {
                            let is_high_lev = cb.turn >= 25 || (al0 == 2 && al1 == 2);
                            snapshots.push((cb, is_high_lev));
                        }
                        let sseed = grng.next();
                        let res = if use_cfv {
                            nc2000_bot::compact_az::search_compact_cfv_puct(&cb, &table, &cache, play_iters, sseed)
                        } else {
                            nc2000_bot::compact_az::search_compact_puct_bs(&cb, &table, &cache, play_iters, sseed)
                        };
                        let mut play_acts = res.best_acts;
                        if cb.turn <= 10 && grng.next_f64() < 0.14 {
                            for side in 0..2usize {
                                let k = (res.n_acts[side] as usize).min(6);
                                if k >= 2 {
                                    let noop = nc2000_bot::compact_az::compact_interior_noop_mask(&cb, &table, side, &res.acts[side], k);
                                    let mut safe = [0u8; 6];
                                    let mut ns = 0usize;
                                    for i in 0..k {
                                        if ((noop >> i) & 1) == 0 && res.visits[side][i] > 0 {
                                            safe[ns] = res.acts[side][i];
                                            ns += 1;
                                        }
                                    }
                                    if ns > 0 {
                                        play_acts[side] = safe[grng.below(ns)];
                                    }
                                }
                            }
                        }
                        cb.apply_actions(&table, play_acts);
                    }
                    if snapshots.is_empty() {
                        continue;
                    }
                    done_g += 1;

                    // Prioritize high-leverage stall / late / 2v2 snapshots (up to half of `states_per_game`)
                    // while keeping uniform midgame diversity!
                    let mut hi_snaps: Vec<nc2000_engine::compact::CompactBattle> = Vec::new();
                    let mut lo_snaps: Vec<nc2000_engine::compact::CompactBattle> = Vec::new();
                    for (scb, hi) in snapshots {
                        if hi { hi_snaps.push(scb); } else { lo_snaps.push(scb); }
                    }
                    for v in [&mut hi_snaps, &mut lo_snaps] {
                        for i in (1..v.len()).rev() {
                            let j = grng.below(i + 1);
                            v.swap(i, j);
                        }
                    }
                    let mut picked: Vec<nc2000_engine::compact::CompactBattle> = Vec::with_capacity(states_per_game);
                    let hi_quota = (states_per_game / 2).min(hi_snaps.len());
                    picked.extend_from_slice(&hi_snaps[..hi_quota]);
                    for &s_cb in &lo_snaps {
                        if picked.len() >= states_per_game { break; }
                        picked.push(s_cb);
                    }
                    for &s_cb in &hi_snaps[hi_quota..] {
                        if picked.len() >= states_per_game { break; }
                        picked.push(s_cb);
                    }

                    for (s_i, &s_cb) in picked.iter().enumerate() {
                        let Some((sp_cb, kind)) =
                            nc2000_bot::compact_az::apply_cf_perturbation(&s_cb, &table, &mut grng)
                        else {
                            continue;
                        };
                        let base_pseed = grng.next();
                        let mut sum0 = 0.0;
                        let mut sum1 = 0.0;
                        for m in 0..crn_m {
                            let pseed = base_pseed ^ (m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                            let (v0, _, _) = nc2000_bot::compact_az::cf_rollout_to_outcome_mode(
                                &s_cb, &table, &cache, search_turns, search_iters, pnet_temp, 200, pseed, use_cfv,
                            );
                            let (v1, _, _) = nc2000_bot::compact_az::cf_rollout_to_outcome_mode(
                                &sp_cb, &table, &cache, search_turns, search_iters, pnet_temp, 200, pseed, use_cfv,
                            );
                            sum0 += v0;
                            sum1 += v1;
                        }
                        let v_cf = (sum0 / (crn_m as f64)) as f32;
                        let vp_cf = (sum1 / (crn_m as f64)) as f32;
                        let z_game = outcome_opt.unwrap_or(v_cf);
                        let (x0, x1) = nc2000_bot::compact_az::extract_raw_value_features(&s_cb, &table, Some(&cache));
                        let (xp0, xp1) = nc2000_bot::compact_az::extract_raw_value_features(&sp_cb, &table, Some(&cache));
                        pairs_out.push(CfPairRec {
                            team_hash_0: th0,
                            team_hash_1: th1,
                            game_id: job.g as u32,
                            turn: s_cb.turn,
                            n0: s_cb.sides[0].alive_mask.count_ones() as u8,
                            n1: s_cb.sides[1].alive_mask.count_ones() as u8,
                            kind,
                            _pad: [0; 3],
                            v_cf,
                            vp_cf,
                            z_game,
                            x0,
                            x1,
                            xp0,
                            xp1,
                        });

                        // Also run 1-sided CRN action intervention `do(a_i)` on the first 2 sampled states per game!
                        if act_m > 0 && s_i < 2 {
                            let side = grng.below(2);
                            let opp = 1 - side;
                            let mut my_a = [0u8; 7];
                            let mut opp_a = [0u8; 7];
                            let km = s_cb.legal_actions(&table, side, &mut my_a).min(6);
                            let ko = s_cb.legal_actions(&table, opp, &mut opp_a).min(6);
                            if km >= 2 && ko >= 1 {
                                let ((p_me, m_me), (p_op, m_op)) = if side == 0 {
                                    nc2000_bot::compact_az::policy_priors_both(&s_cb, &table, Some(&cache), &my_a, km, &opp_a, ko, 0.0)
                                } else {
                                    let (r0, r1) = nc2000_bot::compact_az::policy_priors_both(&s_cb, &table, Some(&cache), &opp_a, ko, &my_a, km, 0.0);
                                    (r1, r0)
                                };
                                let safe_me: Vec<usize> = (0..km).filter(|&i| ((m_me >> i) & 1) == 0).collect();
                                let safe_op: Vec<usize> = (0..ko).filter(|&i| ((m_op >> i) & 1) == 0).collect();
                                if safe_me.len() >= 2 && !safe_op.is_empty() {
                                    // q_eval = 0.75 * P_net(opp) + 0.25 * Uniform(safe_opp)
                                    let u_op = 1.0 / (safe_op.len() as f64);
                                    let mut q_op = [0.0f64; 6];
                                    let mut q_sum = 0.0;
                                    for &j in &safe_op {
                                        let w = 0.75 * f64::from(p_op[j]) + 0.25 * u_op;
                                        q_op[j] = w;
                                        q_sum += w;
                                    }
                                    let mut q_acc = [0.0f64; 6];
                                    let mut v_cnt = [0u32; 6];
                                    let act_base_seed = grng.next();
                                    for m in 0..act_m {
                                        let m_seed = act_base_seed ^ (m as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                                        let mut mrng = SplitMix64::new(m_seed);
                                        let mut r = mrng.next_f64() * q_sum;
                                        let mut opp_j = safe_op[0];
                                        for &j in &safe_op {
                                            r -= q_op[j];
                                            if r <= 0.0 {
                                                opp_j = j;
                                                break;
                                            }
                                        }
                                        let a_opp = opp_a[opp_j];
                                        let step_seed = mrng.next();
                                        let roll_seed = mrng.next();
                                        for &i in &safe_me {
                                            let mut sim = s_cb;
                                            let mut prng = nc2000_engine::prng::Prng::new(step_seed);
                                            let qc = Battle::roll_quick_claw_with(table.ruleset, &mut prng);
                                            sim.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
                                            sim.prng_seed = prng.seed();
                                            let acts = if side == 0 { [my_a[i], a_opp] } else { [a_opp, my_a[i]] };
                                            sim.apply_actions(&table, acts);
                                            let (v0, _, _) = nc2000_bot::compact_az::cf_rollout_to_outcome_mode(
                                                &sim, &table, &cache, 2, search_iters, pnet_temp, 200, roll_seed, use_cfv,
                                            );
                                            let mine = if side == 0 { v0 } else { 1.0 - v0 };
                                            q_acc[i] += mine;
                                            v_cnt[i] += 1;
                                        }
                                    }
                                    let (ctx, af7) = nc2000_bot::compact_az::extract_policy_features(
                                        &s_cb, &table, Some(&cache), side, &my_a, km,
                                    );
                                    let mut af6 = [[0.0f32; nc2000_bot::compact_az::AZ_POLICY_ACT_DIM]; 6];
                                    af6.copy_from_slice(&af7[..6]);
                                    let mut q6 = [0.0f32; 6];
                                    let mut max_q = 0.0f32;
                                    for &i in &safe_me {
                                        let qi = (q_acc[i] / f64::from(v_cnt[i].max(1))) as f32;
                                        q6[i] = qi;
                                        if qi > max_q {
                                            max_q = qi;
                                        }
                                    }
                                    let (th_me, th_opp) = if side == 0 { (th0, th1) } else { (th1, th0) };
                                    pols_out.push(AzPolRec {
                                        team_hash: th_me,
                                        opp_hash: th_opp,
                                        game_id: job.g as u32,
                                        turn: s_cb.turn,
                                        n_acts: km as u8,
                                        noop_mask: m_me,
                                        req: s_cb.sides[side].request,
                                        side: side as u8,
                                        n0: s_cb.sides[0].alive_mask.count_ones() as u8,
                                        n1: s_cb.sides[1].alive_mask.count_ones() as u8,
                                        v_root: max_q,
                                        ctx,
                                        af: af6,
                                        visits: v_cnt,
                                        q: q6,
                                    });
                                }
                            }
                        }
                    }
                }
                (pairs_out, pols_out, done_g)
            }));
        }
        let mut ap = Vec::new();
        let mut a_pol = Vec::new();
        let mut dg = 0usize;
        for h in hs {
            let (p, pl, d) = h.join().unwrap();
            ap.extend(p);
            a_pol.extend(pl);
            dg += d;
        }
        (ap, a_pol, dg)
    });

    write_pod_slice(&PathBuf::from(format!("{out_prefix}.cfpair.bin")), &all_pairs);
    if !all_pols.is_empty() {
        write_pod_slice(&PathBuf::from(format!("{out_prefix}.cfpol.bin")), &all_pols);
    }
    println!(
        "{}",
        json!({
            "cmd": "collect-cf",
            "games": games,
            "finished_games": finished_games,
            "game_start": game_start,
            "crn_m": crn_m,
            "search_turns": search_turns,
            "search_iters": search_iters,
            "pair_rows": all_pairs.len(),
            "pol_rows": all_pols.len(),
            "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        })
    );
}

fn parse_preview_spec(spec: &str) -> Result<Result<u32, nc2000_bot::preview_search::PreviewSearchConfig>, String> {
    use nc2000_bot::preview_search::{PreviewCellReadout, PreviewPolicyMode, PreviewSearchConfig};
    if let Some(rest) = spec.strip_prefix("shipped:") {
        let iters: u32 = rest.parse().map_err(|e| format!("bad shipped iters {spec}: {e}"))?;
        return Ok(Ok(iters));
    }
    if let Some(rest) = spec.strip_prefix("cfprev:") {
        let mut parts = rest.split('@');
        let iters_s = parts.next().unwrap_or("120000");
        let total_iters: u32 = if let Some(k) = iters_s.strip_suffix('k') {
            k.parse::<u32>().map_err(|e| format!("bad iters {spec}: {e}"))? * 1000
        } else {
            iters_s.parse::<u32>().map_err(|e| format!("bad iters {spec}: {e}"))?
        };
        let mut cfg = PreviewSearchConfig {
            total_iters,
            ..PreviewSearchConfig::default()
        };
        let mut dust_ratio = 0.08f64;
        let mut min_mix_margin = 0.015f64;
        let mut lcb_z = 1.0f64;
        let mut mode_tag = "lcb";
        for m in parts {
            if m == "stage1" { mode_tag = "stage1"; }
            else if m == "pure" { mode_tag = "pure"; }
            else if m == "mix" { mode_tag = "mix"; }
            else if m == "lcb" { mode_tag = "lcb"; }
            else if m == "raw" { cfg.cell_readout = PreviewCellReadout::RawMean; }
            else if m == "late" { cfg.cell_readout = PreviewCellReadout::LateWindowMean; }
            else if m == "t1mm" { cfg.cell_readout = PreviewCellReadout::Turn1Minimax; }
            else if m == "cfp" { cfg.use_cf_policy = true; }
            else if let Some(v) = m.strip_prefix("do") { cfg.do_rounds = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("di") { cfg.do_iters = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("dm") { cfg.do_margin = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("cp") { cfg.c_puct = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("fpu") { cfg.fpu_red = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("mm") { min_mix_margin = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("dr") { dust_ratio = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("lz") { lcb_z = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix("pg") { cfg.polish_gap = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix('k') { cfg.support_k = v.parse().unwrap(); }
            else if let Some(v) = m.strip_prefix('c') { cfg.coarse_iters = v.parse().unwrap(); }
            else { return Err(format!("unknown cfprev modifier @{m} in {spec}")); }
        }
        cfg.policy_mode = match mode_tag {
            "stage1" => PreviewPolicyMode::Stage1Only,
            "pure" => PreviewPolicyMode::PureMaximin,
            "mix" => PreviewPolicyMode::MixedNash { dust_ratio },
            _ => PreviewPolicyMode::LcbGatedMixed {
                min_mix_margin,
                dust_ratio,
                lcb_z,
            },
        };
        return Ok(Err(cfg));
    }
    Err(format!("unknown preview spec: {spec} (expected shipped:N or cfprev:N[@...])"))
}

fn run_single_live_preview(
    dex: &Dex,
    battle: &Battle,
    side: usize,
    own_sets: &[PokemonSet],
    pool: &Arc<MetaPool>,
    spec: &str,
    seed: u64,
) -> ([u8; 3], u64, bool, usize) {
    let parsed = parse_preview_spec(spec).unwrap_or_else(|e| panic!("{e}"));
    let (mut lp, mut obs) =
        nc2000_bot::player_state::LocalPlayer::new_with_own_sets(battle, dex, side, own_sets.to_vec())
            .expect("LocalPlayer::new_with_own_sets");
    lp.observe(battle, dex, &mut obs).expect("observe turn 0");
    let mut belief = nc2000_bot::belief::Belief::new(dex, pool, &obs);
    belief.sync(dex, &obs);
    let mut rng = SplitMix64::new(seed);
    let pos = lp.position(dex, &mut belief, &obs, &mut rng).expect("position turn 0");
    let t0 = Instant::now();
    match parsed {
        Ok(iters) => {
            let ch = nc2000_bot::preview_search::solve_shipped_preview(
                &pos, dex, &belief, &obs, side, iters, rng,
            );
            let ns = t0.elapsed().as_nanos() as u64;
            let SearchChoice::Team(tr) = ch else { panic!("expected Team choice") };
            (tr, ns, false, 0)
        }
        Err(cfg) => {
            let res = nc2000_bot::preview_search::solve_cf_preview(
                &pos, dex, &belief, &obs, side, &cfg, &mut rng,
            );
            let ns = t0.elapsed().as_nanos() as u64;
            let SearchChoice::Team(tr) = res.choice else { panic!("expected Team choice") };
            (tr, ns, res.used_mix, res.do_admitted_me + res.do_admitted_opp)
        }
    }
}

/// Turn-1 `CF-V_net` v2 diagnostic on strictly unseen `belief-pool-v3` (`hash % 5 == 0`) 6v6 matchups:
/// measures Stage-1 speed, correlation/RMSE vs `cfv-0r` ground truth, Recall@K of ground-truth top
/// picks inside `S_me`, and lead-archetype residuals.
fn run_preview_diag(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let pairs: usize = flag("--pairs").map_or(24, |v| v.parse().unwrap());
    let gt_iters: u32 = flag("--gt-iters").map_or(512, |v| v.parse().unwrap());
    let sample_cells: usize = flag("--cells").map_or(24, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026113001, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());

    let root = repo_root();
    let pool_path = root.join("data/belief-pool-v3/belief-pool.json");
    let pool = Arc::new(load_meta_pool(&pool_path));
    let mut unseen: Vec<usize> = (0..pool.teams.len())
        .filter(|&i| (canonical_team_hash(&pool.teams[i].sets) % 5) == 0)
        .collect();
    let mut rng = SplitMix64::new(seed);
    let mut matchups = Vec::with_capacity(pairs);
    for p in 0..pairs {
        let a = unseen[rng.below(unseen.len())];
        let mut b = unseen[rng.below(unseen.len())];
        if b == a && unseen.len() > 1 {
            b = unseen[(a + 1) % unseen.len()];
        }
        matchups.push((p, a, b, rng.next()));
    }
    unseen.shrink_to_fit();

    let cursor = AtomicUsize::new(0);
    #[derive(Default)]
    struct DiagAcc {
        v0: Vec<f64>,
        vgt: Vec<f64>,
        arch_res: [Vec<f64>; 6], // 0=normal, 1=sleep, 2=spikes, 3=bp, 4=boom, 5=pursuit
        s1_ms: Vec<f64>,
        s2_ms: Vec<f64>,
        do_adm: usize,
        mix_cnt: usize,
        top1_in_sme: usize,
        top3_hits: usize,
        top3_tot: usize,
    }

    let t0 = Instant::now();
    let parts: Vec<DiagAcc> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(pairs).max(1) {
            let (pool_r, m_r, cur_r) = (&pool, &matchups, &cursor);
            hs.push(s.spawn(move || {
                let mut acc = DiagAcc::default();
                let cfg_s2 = nc2000_bot::preview_search::PreviewSearchConfig {
                    total_iters: 120_000,
                    ..Default::default()
                };
                loop {
                    let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                    if idx >= m_r.len() {
                        break;
                    }
                    let (_, ia, ib, mseed) = m_r[idx];
                    let ta = &pool_r.teams[ia].sets;
                    let tb = &pool_r.teams[ib].sets;
                    let mut b = Battle::from_fixture(dex, "1,2,3,4", ta, tb).unwrap();
                    if !nc2000_engine::compact::is_compact_preview_supported(&b, dex) {
                        continue;
                    }
                    let (mut lp, mut obs) =
                        nc2000_bot::player_state::LocalPlayer::new_with_own_sets(&b, dex, 0, ta.clone()).unwrap();
                    lp.observe(&b, dex, &mut obs).unwrap();
                    let mut belief = nc2000_bot::belief::Belief::new(dex, pool_r, &obs);
                    belief.sync(dex, &obs);
                    let mut lrng = SplitMix64::new(mseed);
                    let pos = lp.position(dex, &mut belief, &obs, &mut lrng).unwrap();
                    let res = nc2000_bot::preview_search::solve_cf_preview(
                        &pos, dex, &belief, &obs, 0, &cfg_s2, &mut lrng,
                    );
                    acc.s1_ms.push(res.stage1_ms);
                    acc.s2_ms.push(res.total_ms);
                    acc.do_adm += res.do_admitted_me + res.do_admitted_opp;
                    if res.used_mix {
                        acc.mix_cnt += 1;
                    }

                    // Sample `sample_cells` legal (own, opp) Turn-1 picks and compare 0-ply `CF-V_net` vs `gt_iters` `cfv-0r`.
                    let roster = nc2000_engine::compact::CompactPreviewRoster::from_battle(&b, dex, None);
                    let pcache = nc2000_bot::tempo::PreviewTempoCache::build_for_preview_roster(
                        &roster,
                        [b.sides[0].party.len(), b.sides[1].party.len()],
                    );
                    let legal0 = b.legal_choices(dex, 0);
                    let legal1 = b.legal_choices(dex, 1);
                    let mut c0 = Vec::new();
                    for ch in &legal0 {
                        if let SearchChoice::Team(t) = *ch {
                            if t[2] == 0 || t[1] < t[2] { c0.push(t); }
                        }
                    }
                    let mut c1 = Vec::new();
                    for ch in &legal1 {
                        if let SearchChoice::Team(t) = *ch {
                            if t[2] == 0 || t[1] < t[2] { c1.push(t); }
                        }
                    }
                    if c0.is_empty() || c1.is_empty() {
                        continue;
                    }
                    // Evaluate `sample_cells` random pairs for V_0 vs V_gt + archetype residuals.
                    for ci in 0..sample_cells {
                        let p0 = c0[lrng.below(c0.len())];
                        let p1 = c1[lrng.below(c1.len())];
                        let (tbl, mut cb) = roster.build_turn1([p0, p1], 0);
                        cb.quick_claw_roll = 0;
                        let slc = pcache.slice_for(&cb);
                        let v0 = nc2000_bot::compact_az::evaluate_cf_value(&cb, &tbl, Some(&slc), 1.0);
                        let gt = nc2000_bot::compact_az::search_compact_cfv_puct(
                            &cb, &tbl, &slc, gt_iters, mseed ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                        );
                        acc.v0.push(v0);
                        acc.vgt.push(gt.v_root0);
                        // Classify p0 lead archetype.
                        let lead_r = b.sides[0].party[(p0[0] as usize) - 1] as usize;
                        let lead_m = &b.sides[0].roster[lead_r];
                        let mut bucket = 0usize;
                        for sl in &lead_m.base_move_slots {
                            match dex.moves.key(sl.id) {
                                "spore" | "sleeppowder" | "lovelykiss" | "hypnosis" | "sing" => { bucket = 1; break; }
                                "spikes" => { bucket = 2; break; }
                                "batonpass" => { bucket = 3; break; }
                                "explosion" | "selfdestruct" => { bucket = 4; break; }
                                "pursuit" | "counter" | "mirrorcoat" => { bucket = 5; break; }
                                _ => {}
                            }
                        }
                        acc.arch_res[bucket].push(gt.v_root0 - v0);
                    }
                    // Also check Recall@12: evaluate 12 random own picks against one fixed opponent pick under V_gt.
                    let opp_pick = c1[lrng.below(c1.len())];
                    let mut scored0: Vec<(usize, f64)> = (0..c0.len())
                        .map(|i| {
                            let (tbl, mut cb) = roster.build_turn1([c0[i], opp_pick], 0);
                            cb.quick_claw_roll = 0;
                            let slc = pcache.slice_for(&cb);
                            (i, nc2000_bot::compact_az::evaluate_cf_value(&cb, &tbl, Some(&slc), 1.0))
                        })
                        .collect();
                    scored0.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let top12: Vec<usize> = scored0.iter().take(12).map(|&(i, _)| i).collect();
                    let mut gt_scores: Vec<(usize, f64)> = (0..c0.len())
                        .step_by((c0.len() / 15).max(1))
                        .map(|i| {
                            let (tbl, mut cb) = roster.build_turn1([c0[i], opp_pick], 0);
                            cb.quick_claw_roll = 0;
                            let slc = pcache.slice_for(&cb);
                            let r = nc2000_bot::compact_az::search_compact_cfv_puct(
                                &cb, &tbl, &slc, 192, mseed ^ ((i as u64) + 1000),
                            );
                            (i, r.v_root0)
                        })
                        .collect();
                    gt_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
                    if let Some(&(best_i, _)) = gt_scores.first() {
                        if top12.contains(&best_i) {
                            acc.top1_in_sme += 1;
                        }
                    }
                    for &(gi, _) in gt_scores.iter().take(3) {
                        acc.top3_tot += 1;
                        if top12.contains(&gi) {
                            acc.top3_hits += 1;
                        }
                    }
                }
                acc
            }));
        }
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut all_v0 = Vec::new();
    let mut all_vgt = Vec::new();
    let mut arch: [Vec<f64>; 6] = Default::default();
    let mut s1_ms = Vec::new();
    let mut s2_ms = Vec::new();
    let mut do_adm = 0usize;
    let mut mix_cnt = 0usize;
    let mut top1_in = 0usize;
    let mut top3_h = 0usize;
    let mut top3_t = 0usize;
    for p in parts {
        all_v0.extend(p.v0);
        all_vgt.extend(p.vgt);
        for k in 0..6 { arch[k].extend(p.arch_res[k].iter().copied()); }
        s1_ms.extend(p.s1_ms);
        s2_ms.extend(p.s2_ms);
        do_adm += p.do_adm;
        mix_cnt += p.mix_cnt;
        top1_in += p.top1_in_sme;
        top3_h += p.top3_hits;
        top3_t += p.top3_tot;
    }
    let n = all_v0.len().max(1) as f64;
    let m0 = all_v0.iter().sum::<f64>() / n;
    let mg = all_vgt.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var0 = 0.0;
    let mut varg = 0.0;
    let mut mse = 0.0;
    for (&a, &b) in all_v0.iter().zip(&all_vgt) {
        cov += (a - m0) * (b - mg);
        var0 += (a - m0).powi(2);
        varg += (b - mg).powi(2);
        mse += (a - b).powi(2);
    }
    let r = cov / (var0.sqrt() * varg.sqrt()).max(1e-12);
    let rmse = (mse / n).sqrt();
    s1_ms.sort_by(|a, b| a.total_cmp(b));
    s2_ms.sort_by(|a, b| a.total_cmp(b));
    let med = |v: &[f64]| if v.is_empty() { 0.0 } else { v[v.len() / 2] };
    let p95 = |v: &[f64]| if v.is_empty() { 0.0 } else { v[(v.len() - 1) * 95 / 100] };
    let arch_names = ["normal", "sleep", "spikes", "bp", "boom", "pursuit"];
    let mut arch_json = serde_json::Map::new();
    for k in 0..6 {
        let cnt = arch[k].len();
        let mean_r = if cnt > 0 { arch[k].iter().sum::<f64>() / (cnt as f64) } else { 0.0 };
        arch_json.insert(
            arch_names[k].into(),
            json!({"n": cnt, "mean_gt_minus_v0": (mean_r * 10000.0).round() / 10000.0}),
        );
    }
    println!(
        "{}",
        json!({
            "cmd": "preview-diag",
            "matchups": s1_ms.len(),
            "cells": all_v0.len(),
            "gt_iters": gt_iters,
            "r_v0_vs_gt": (r * 10000.0).round() / 10000.0,
            "rmse": (rmse * 10000.0).round() / 10000.0,
            "recall12_top1": format!("{}/{}", top1_in, s1_ms.len()),
            "recall12_top3": format!("{}/{}", top3_h, top3_t),
            "stage1_med_ms": (med(&s1_ms) * 100.0).round() / 100.0,
            "stage2_120k_med_ms": (med(&s2_ms) * 100.0).round() / 100.0,
            "stage2_120k_p95_ms": (p95(&s2_ms) * 100.0).round() / 100.0,
            "mix_rate": (mix_cnt as f64 / s1_ms.len().max(1) as f64 * 1000.0).round() / 1000.0,
            "do_admitted_total": do_adm,
            "arch_residuals": arch_json,
            "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        })
    );
}

/// Strictly NO-CACHE live Team Preview H2H duel (`preview-duel`).
/// Isolates Team Preview strength by giving both seats identical in-battle engines (or tests
/// end-to-end when `--battle-a` != `--battle-b`), with fresh live preview search on EVERY game.
fn run_preview_duel(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let preview_a = flag("--preview-a").unwrap_or_else(|| "cfprev:120k@lcb".to_string());
    let preview_b = flag("--preview-b").unwrap_or_else(|| "shipped:729000".to_string());
    let battle_shared = flag("--battle");
    let battle_a = flag("--battle-a")
        .or_else(|| battle_shared.clone())
        .unwrap_or_else(|| "bcmcts-cfv-0r:27000@puct@lam0".to_string());
    let battle_b = flag("--battle-b")
        .or(battle_shared)
        .unwrap_or_else(|| "bcmcts-cfv-0r:27000@puct@lam0".to_string());
    let pairs: usize = flag("--pairs").map_or(32, |v| v.parse().unwrap());
    let pair_start: usize = flag("--pair-start").map_or(0, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026113001, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let out_path = flag("--out").map(PathBuf::from);
    let max_turns: u16 = flag("--max-turns").map_or(500, |v| v.parse().unwrap());
    let catalog_path = flag("--catalog").unwrap_or_else(|| "data/team-pool-v2/team-pool.json".to_string());
    let unseen_only = args.iter().any(|a| a == "--unseen-only");
    let offpool = args.iter().any(|a| a == "--offpool");
    let total_pairs_needed = pair_start + pairs;

    let root = repo_root();
    let cat_txt = std::fs::read_to_string(root.join(&catalog_path))
        .unwrap_or_else(|e| panic!("read catalog {catalog_path}: {e}"));
    let mut catalog: WeightedCatalog = serde_json::from_str(&cat_txt)
        .unwrap_or_else(|e| panic!("parse catalog {catalog_path}: {e}"));
    if unseen_only {
        catalog.teams.retain(|t| (canonical_team_hash(&t.sets) % 5) == 0);
        assert!(!catalog.teams.is_empty());
    }
    let pool = Arc::new(load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json")));
    let public_shell = |sets: &[PokemonSet]| -> Vec<(String, u8)> {
        let mut v: Vec<(String, u8)> = sets
            .iter()
            .map(|s| (s.species.to_ascii_lowercase(), s.level))
            .collect();
        v.sort_unstable();
        v
    };
    let pool_shells: Vec<Vec<(String, u8)>> = pool.teams.iter().map(|t| public_shell(&t.sets)).collect();

    let weights: Vec<f64> = catalog.teams.iter().map(|t| t.draw_weight.unwrap_or(1.0)).collect();
    let w_sum: f64 = weights.iter().sum();
    let mut draw_rng = SplitMix64::new(seed);
    let draw_team = |rng: &mut SplitMix64| -> usize {
        let mut r = rng.next_f64() * w_sum;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    };

    struct PSchedule {
        pair_idx: usize,
        ta: usize,
        tb: usize,
        bseed: String,
        pseed_p1_g0: u64,
        pseed_p2_g0: u64,
        pseed_p1_g1: u64,
        pseed_p2_g1: u64,
        sseed_p1: u64,
        sseed_p2: u64,
    }
    let mut scheds = Vec::with_capacity(total_pairs_needed);
    for pair_idx in 0..total_pairs_needed {
        let ta = draw_team(&mut draw_rng);
        let mut tb = draw_team(&mut draw_rng);
        if tb == ta && catalog.teams.len() > 1 {
            tb = (ta + 1) % catalog.teams.len();
        }
        let s1 = (draw_rng.next() & 0xFFFF) as u16;
        let s2 = (draw_rng.next() & 0xFFFF) as u16;
        let s3 = (draw_rng.next() & 0xFFFF) as u16;
        let s4 = (draw_rng.next() & 0xFFFF) as u16;
        scheds.push(PSchedule {
            pair_idx,
            ta,
            tb,
            bseed: format!("{s1},{s2},{s3},{s4}"),
            pseed_p1_g0: draw_rng.next(),
            pseed_p2_g0: draw_rng.next(),
            pseed_p1_g1: draw_rng.next(),
            pseed_p2_g1: draw_rng.next(),
            sseed_p1: draw_rng.next(),
            sseed_p2: draw_rng.next(),
        });
    }

    let t0 = Instant::now();
    let total_games = pairs * 2;
    let cursor = AtomicUsize::new(0);
    struct PGameOut {
        game_idx: usize,
        pair_idx: usize,
        a_is_p1: bool,
        a_score: f64,
        turns: u16,
        turn_cap: bool,
        a_prev_ns: u64,
        b_prev_ns: u64,
        a_pick: [u8; 3],
        b_pick: [u8; 3],
        a_used_mix: bool,
        a_do_adm: usize,
        a_ns: Vec<u64>,
        b_ns: Vec<u64>,
    }

    let results: Vec<PGameOut> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(total_games).max(1) {
            let (cat_r, pool_r, ps_r, sched_r, cur_r, pa, pb, ba, bb) = (
                &catalog, &pool, &pool_shells, &scheds, &cursor, &preview_a, &preview_b, &battle_a, &battle_b,
            );
            hs.push(s.spawn(move || {
                let mut local = Vec::new();
                loop {
                    let local_gi = cur_r.fetch_add(1, Ordering::Relaxed);
                    if local_gi >= total_games {
                        break;
                    }
                    let pi = pair_start + local_gi / 2;
                    let gi = pi * 2 + (local_gi % 2);
                    let a_is_p1 = gi % 2 == 0;
                    let sc = &sched_r[pi];
                    let sets1 = &cat_r.teams[sc.ta].sets;
                    let sets2 = &cat_r.teams[sc.tb].sets;
                    let (pspec1, pspec2) = if a_is_p1 { (pa.as_str(), pb.as_str()) } else { (pb.as_str(), pa.as_str()) };
                    let (bspec1, bspec2) = if a_is_p1 { (ba.as_str(), bb.as_str()) } else { (bb.as_str(), ba.as_str()) };
                    let (pseed1, pseed2) = if a_is_p1 {
                        (sc.pseed_p1_g0, sc.pseed_p2_g0)
                    } else {
                        (sc.pseed_p1_g1, sc.pseed_p2_g1)
                    };

                    // Optional `--offpool`: strip all pool teams matching opponent's public
                    // `(species, level)` shell so Turn-0 `Belief::is_fallback() == true`.
                    let make_pool = |opp_sets: &[PokemonSet]| -> Arc<MetaPool> {
                        if !offpool {
                            return pool_r.clone();
                        }
                        let mut sh: Vec<(String, u8)> = opp_sets
                            .iter()
                            .map(|s| (s.species.to_ascii_lowercase(), s.level))
                            .collect();
                        sh.sort_unstable();
                        let filtered = pool_r
                            .teams
                            .iter()
                            .enumerate()
                            .filter(|&(idx, _)| ps_r[idx] != sh)
                            .map(|(_, t)| t.clone())
                            .collect();
                        Arc::new(MetaPool { teams: filtered })
                    };
                    let pool1 = make_pool(sets2);
                    let pool2 = make_pool(sets1);

                    let mut b = Battle::from_fixture(dex, &sc.bseed, sets1, sets2).unwrap();
                    b.set_log_enabled(true);
                    let (tr1, pns1, mix1, adm1) =
                        run_single_live_preview(dex, &b, 0, sets1, &pool1, pspec1, pseed1);
                    let (tr2, pns2, mix2, adm2) =
                        run_single_live_preview(dex, &b, 1, sets2, &pool2, pspec2, pseed2);

                    let mut p1 = TimedAgentWrap::with_preview_override(
                        build_blind_arm(bspec1, pool1, sc.sseed_p1, tr1),
                        tr1,
                    );
                    let mut p2 = TimedAgentWrap::with_preview_override(
                        build_blind_arm(bspec2, pool2, sc.sseed_p2, tr2),
                        tr2,
                    );
                    let res = play_game(dex, &mut b, &mut [&mut p1, &mut p2], max_turns).unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        _ => 0.5,
                    };
                    let a_score = if a_is_p1 { p1_score } else { 1.0 - p1_score };
                    let (a_prev_ns, b_prev_ns, a_pick, b_pick, a_used_mix, a_do_adm, a_ns, b_ns) =
                        if a_is_p1 {
                            (pns1, pns2, tr1, tr2, mix1, adm1, p1.searched_battle_ns, p2.searched_battle_ns)
                        } else {
                            (pns2, pns1, tr2, tr1, mix2, adm2, p2.searched_battle_ns, p1.searched_battle_ns)
                        };
                    local.push(PGameOut {
                        game_idx: gi,
                        pair_idx: pi,
                        a_is_p1,
                        a_score,
                        turns: b.turn,
                        turn_cap: matches!(res, GameResult::TurnCapped),
                        a_prev_ns,
                        b_prev_ns,
                        a_pick,
                        b_pick,
                        a_used_mix,
                        a_do_adm,
                        a_ns,
                        b_ns,
                    });
                }
                local
            }));
        }
        let mut merged = Vec::with_capacity(total_games);
        for h in hs {
            merged.extend(h.join().unwrap());
        }
        merged
    });

    let by_idx: HashMap<usize, PGameOut> = results.into_iter().map(|r| (r.game_idx, r)).collect();
    if let Some(ref op) = out_path {
        use std::io::Write;
        if let Some(parent) = op.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).unwrap();
            }
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(op).unwrap());
        for pi in pair_start..total_pairs_needed {
            for gi in [pi * 2, pi * 2 + 1] {
                let g = &by_idx[&gi];
                let row = json!({
                    "game_idx": g.game_idx,
                    "pair_idx": g.pair_idx,
                    "a_is_p1": g.a_is_p1,
                    "a_score": g.a_score,
                    "turns": g.turns,
                    "turn_cap": g.turn_cap,
                    "a_prev_ns": g.a_prev_ns,
                    "b_prev_ns": g.b_prev_ns,
                    "a_pick": g.a_pick,
                    "b_pick": g.b_pick,
                    "a_used_mix": g.a_used_mix,
                    "a_do_adm": g.a_do_adm,
                    "a_ns": g.a_ns,
                    "b_ns": g.b_ns,
                });
                serde_json::to_writer(&mut f, &row).unwrap();
                writeln!(f).unwrap();
            }
        }
        f.flush().unwrap();
    }

    let mut pair_scores = Vec::with_capacity(pairs);
    let (mut wins, mut losses, mut ties, mut turn_caps, mut turns_sum) = (0usize, 0usize, 0usize, 0usize, 0u64);
    let mut a_prev_all = Vec::new();
    let mut b_prev_all = Vec::new();
    let mut mix_cnt = 0usize;
    let mut pick_diffs = 0usize;
    for pi in pair_start..total_pairs_needed {
        let g0 = &by_idx[&(pi * 2)];
        let g1 = &by_idx[&(pi * 2 + 1)];
        pair_scores.push(0.5 * (g0.a_score + g1.a_score));
        // Compare A's P1 pick (g0.a_pick) vs B's P1 pick (g1.b_pick) on identical roster!
        if g0.a_pick != g1.b_pick { pick_diffs += 1; }
        if g1.a_pick != g0.b_pick { pick_diffs += 1; }
        for g in [g0, g1] {
            if g.a_score == 1.0 { wins += 1; }
            else if g.a_score == 0.0 { losses += 1; }
            else { ties += 1; }
            if g.turn_cap { turn_caps += 1; }
            turns_sum += u64::from(g.turns);
            a_prev_all.push(g.a_prev_ns);
            b_prev_all.push(g.b_prev_ns);
            if g.a_used_mix { mix_cnt += 1; }
        }
    }
    let (mean, ci_lo, ci_hi) = bootstrap_ci(&pair_scores, seed);
    let med_ap = percentile_ms(&a_prev_all, 50);
    let med_bp = percentile_ms(&b_prev_all, 50);
    let p95_ap = percentile_ms(&a_prev_all, 95);
    let p95_bp = percentile_ms(&b_prev_all, 95);
    println!(
        "{}",
        json!({
            "cmd": "preview-duel",
            "preview_a": preview_a,
            "preview_b": preview_b,
            "battle_a": battle_a,
            "battle_b": battle_b,
            "offpool": offpool,
            "pairs": pairs,
            "games": total_games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "turn_caps": turn_caps,
            "score": (mean * 10000.0).round() / 10000.0,
            "ci95": [(ci_lo * 10000.0).round() / 10000.0, (ci_hi * 10000.0).round() / 10000.0],
            "elo_diff": (elo_diff(mean) * 10.0).round() / 10.0,
            "avg_turns": (turns_sum as f64 / total_games.max(1) as f64 * 10.0).round() / 10.0,
            "a_prev_med_ms": (med_ap * 100.0).round() / 100.0,
            "b_prev_med_ms": (med_bp * 100.0).round() / 100.0,
            "prev_time_ratio": (med_ap / med_bp.max(1e-9) * 1000.0).round() / 1000.0,
            "a_prev_p95_ms": (p95_ap * 100.0).round() / 100.0,
            "b_prev_p95_ms": (p95_bp * 100.0).round() / 100.0,
            "a_mix_rate": (mix_cnt as f64 / total_games.max(1) as f64 * 1000.0).round() / 1000.0,
            "pick_diff_rate": (pick_diffs as f64 / total_games.max(1) as f64 * 1000.0).round() / 1000.0,
            "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        })
    );
}

/// Counter-Pick Exploitability Audit (`preview-exploit`):
/// Evaluates the root preview policy distribution `p` of multiple preview strategies
/// against a double-estimator (zero winner's-curse) full-information opponent counter-pick oracle:
/// 1. Candidate opponent counters are screened with BOTH `CF-V_net` v2 AND independent `linear.rs`
///    `compact_eval_leaf_blindspot` PLUS 1 candidate per distinct opponent lead (de-biasing screen).
/// 2. `j_BR = argmin_{p1} E_{p0 ~ p}[V_select(p0, p1)]` is chosen using seed 1 (`oracle_iters / 2`).
/// 3. Unbiased worst-case value `E_{p0 ~ p}[V_eval(p0, j_BR)]` is evaluated on an INDEPENDENT seed 2
///    (`oracle_iters`, with proper Berserk Gene confusion resampling).
/// Use `shipped:N` for a single live pick (apples-to-apples deterministic worst-case) or `shipped5:N`
/// for a 5-seed empirical randomized distribution.
fn run_preview_exploit(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let arms_str = flag("--arms").unwrap_or_else(|| {
        "cfprev:120k@lcb@t1mm,cfprev:120k@mix@t1mm,cfprev:120k@pure@t1mm,cfprev:0@stage1,shipped:729000,shipped5:729000".to_string()
    });
    let arms: Vec<String> = arms_str.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let pairs: usize = flag("--pairs").map_or(32, |v| v.parse().unwrap());
    let oracle_iters: u32 = flag("--oracle-iters").map_or(768, |v| v.parse().unwrap());
    let opp_cands_k: usize = flag("--opp-cands").map_or(16, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026120201, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let catalog_path = flag("--catalog").unwrap_or_else(|| "data/team-pool-v2/team-pool.json".to_string());
    let unseen_only = args.iter().any(|a| a == "--unseen-only");

    let root = repo_root();
    let cat_txt = std::fs::read_to_string(root.join(&catalog_path)).unwrap();
    let mut catalog: WeightedCatalog = serde_json::from_str(&cat_txt).unwrap();
    if unseen_only {
        catalog.teams.retain(|t| (canonical_team_hash(&t.sets) % 5) == 0);
        assert!(!catalog.teams.is_empty());
    }
    let pool = Arc::new(load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json")));
    let weights: Vec<f64> = catalog.teams.iter().map(|t| t.draw_weight.unwrap_or(1.0)).collect();
    let w_sum: f64 = weights.iter().sum();
    let mut rng = SplitMix64::new(seed);
    let mut matchups = Vec::with_capacity(pairs);
    for pi in 0..pairs {
        let mut draw_t = |r: &mut SplitMix64| -> usize {
            let mut x = r.next_f64() * w_sum;
            for (i, &w) in weights.iter().enumerate() {
                x -= w;
                if x <= 0.0 { return i; }
            }
            weights.len() - 1
        };
        let ta = draw_t(&mut rng);
        let mut tb = draw_t(&mut rng);
        if tb == ta && catalog.teams.len() > 1 { tb = (ta + 1) % catalog.teams.len(); }
        matchups.push((pi, ta, tb, rng.next()));
    }

    let n_arms = arms.len();
    let cursor = AtomicUsize::new(0);
    #[derive(Clone, Default)]
    struct ArmAcc {
        worst_ev: Vec<f64>,
        supp_sizes: Vec<f64>,
        mix_cnt: usize,
        ms: Vec<f64>,
    }

    let t0 = Instant::now();
    let parts: Vec<Vec<ArmAcc>> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(pairs).max(1) {
            let (cat_r, pool_r, m_r, cur_r, arms_r) = (&catalog, &pool, &matchups, &cursor, &arms);
            hs.push(s.spawn(move || {
                let mut acc = vec![ArmAcc::default(); arms_r.len()];
                let lin_w = nc2000_bot::eval::EvalWeights::default();
                let select_iters = (oracle_iters / 2).max(192);
                loop {
                    let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                    if idx >= m_r.len() { break; }
                    let (_, ia, ib, mseed) = m_r[idx];
                    let ta = &cat_r.teams[ia].sets;
                    let tb = &cat_r.teams[ib].sets;
                    let mut b = Battle::from_fixture(dex, "1,2,3,4", ta, tb).unwrap();
                    if !nc2000_engine::compact::is_compact_preview_supported(&b, dex) {
                        continue;
                    }
                    b.set_log_enabled(true);
                    let legal0 = b.legal_choices(dex, 0);
                    let opp_cands_all = nc2000_bot::preview_search::canonical_preview_triples(dex, &mut b, 1);
                    if opp_cands_all.is_empty() { continue; }

                    let (mut lp, mut obs) =
                        nc2000_bot::player_state::LocalPlayer::new_with_own_sets(&b, dex, 0, ta.clone()).unwrap();
                    lp.observe(&b, dex, &mut obs).unwrap();
                    let mut belief = nc2000_bot::belief::Belief::new(dex, pool_r, &obs);
                    belief.sync(dex, &obs);

                    let mut arm_dists: Vec<Vec<([u8; 3], f64)>> = Vec::with_capacity(arms_r.len());
                    for (ai, spec) in arms_r.iter().enumerate() {
                        let (eff_spec, n_seeds) = if let Some(rest) = spec.strip_prefix("shipped5:") {
                            (format!("shipped:{rest}"), 5usize)
                        } else {
                            (spec.clone(), 1usize)
                        };
                        let parsed = parse_preview_spec(&eff_spec).unwrap_or_else(|e| panic!("{e}"));
                        match parsed {
                            Ok(iters) => {
                                let mut counts: HashMap<[u8; 3], usize> = HashMap::new();
                                let mut dt_ms = 0.0;
                                for sk in 0..n_seeds {
                                    let mut lrng = SplitMix64::new(mseed ^ ((ai as u64 + 1) * 0x9E37_79B9) ^ (sk as u64 * 0x517C_C1B7));
                                    let pos = lp.position(dex, &mut belief, &obs, &mut lrng).unwrap();
                                    let ts = Instant::now();
                                    let ch = nc2000_bot::preview_search::solve_shipped_preview(
                                        &pos, dex, &belief, &obs, 0, iters, lrng,
                                    );
                                    dt_ms += ts.elapsed().as_secs_f64() * 1000.0;
                                    if let SearchChoice::Team(tr) = ch {
                                        *counts.entry(tr).or_insert(0) += 1;
                                    }
                                }
                                acc[ai].ms.push(dt_ms / (n_seeds as f64));
                                let tot = n_seeds.max(1) as f64;
                                let dist: Vec<([u8; 3], f64)> = counts.into_iter().map(|(t, c)| (t, c as f64 / tot)).collect();
                                if dist.len() > 1 { acc[ai].mix_cnt += 1; }
                                acc[ai].supp_sizes.push(dist.len() as f64);
                                arm_dists.push(dist);
                            }
                            Err(cfg) => {
                                let mut lrng = SplitMix64::new(mseed ^ ((ai as u64 + 1) * 0x9E37_79B9));
                                let pos = lp.position(dex, &mut belief, &obs, &mut lrng).unwrap();
                                let ts = Instant::now();
                                let res = nc2000_bot::preview_search::solve_cf_preview(
                                    &pos, dex, &belief, &obs, 0, &cfg, &mut lrng,
                                );
                                acc[ai].ms.push(ts.elapsed().as_secs_f64() * 1000.0);
                                let mut dist: Vec<([u8; 3], f64)> = Vec::new();
                                for (li, &p) in res.policy_full.iter().enumerate() {
                                    if p > 1e-6 {
                                        if let Some(SearchChoice::Team(tr)) = legal0.get(li) {
                                            dist.push((*tr, p));
                                        }
                                    }
                                }
                                if dist.is_empty() {
                                    if let SearchChoice::Team(tr) = res.choice { dist.push((tr, 1.0)); }
                                }
                                let s: f64 = dist.iter().map(|x| x.1).sum();
                                for x in &mut dist { x.1 /= s.max(1e-12); }
                                if dist.len() > 1 { acc[ai].mix_cnt += 1; }
                                acc[ai].supp_sizes.push(dist.len() as f64);
                                arm_dists.push(dist);
                            }
                        }
                    }

                    // De-biased counter-pick candidate screening: combine `CF-V_net` v2, independent
                    // `linear.rs` (`compact_eval_leaf_blindspot`), and 1 best counter per opponent lead.
                    let roster = nc2000_engine::compact::CompactPreviewRoster::from_battle(&b, dex, None);
                    let pcache = nc2000_bot::tempo::PreviewTempoCache::build_for_preview_roster(
                        &roster,
                        [b.sides[0].party.len(), b.sides[1].party.len()],
                    );
                    let mut own_union: Vec<[u8; 3]> = Vec::new();
                    for dist in &arm_dists {
                        for &(tr, _) in dist {
                            if !own_union.contains(&tr) { own_union.push(tr); }
                        }
                    }
                    let mut v0_cache: HashMap<([u8; 3], [u8; 3]), (f64, f64)> = HashMap::new();
                    for &p0 in &own_union {
                        for &p1 in &opp_cands_all {
                            let (tbl, mut cb) = roster.build_turn1([p0, p1], 0);
                            cb.quick_claw_roll = 0;
                            let slc = pcache.slice_for(&cb);
                            let v_cf = nc2000_bot::compact_az::evaluate_cf_value(&cb, &tbl, Some(&slc), 1.0);
                            let v_lin = nc2000_bot::eval::compact_eval_leaf_blindspot(&cb, &tbl, &lin_w);
                            v0_cache.insert((p0, p1), (v_cf, v_lin));
                        }
                    }
                    let half_k = (opp_cands_k / 2).max(4);
                    let mut opp_eval_set: Vec<[u8; 3]> = Vec::new();
                    for dist in &arm_dists {
                        let mut by_cf: Vec<([u8; 3], f64)> = opp_cands_all
                            .iter()
                            .map(|&p1| (p1, dist.iter().map(|&(p0, pr)| pr * v0_cache[&(p0, p1)].0).sum()))
                            .collect();
                        by_cf.sort_by(|a, b| a.1.total_cmp(&b.1));
                        for &(p1, _) in by_cf.iter().take(half_k) {
                            if !opp_eval_set.contains(&p1) { opp_eval_set.push(p1); }
                        }
                        let mut by_lin: Vec<([u8; 3], f64)> = opp_cands_all
                            .iter()
                            .map(|&p1| (p1, dist.iter().map(|&(p0, pr)| pr * v0_cache[&(p0, p1)].1).sum()))
                            .collect();
                        by_lin.sort_by(|a, b| a.1.total_cmp(&b.1));
                        for &(p1, _) in by_lin.iter().take(half_k) {
                            if !opp_eval_set.contains(&p1) { opp_eval_set.push(p1); }
                        }
                        // Ensure >= 1 strongest counter per opponent lead (from both CF and linear screens).
                        for ld in 1u8..=6 {
                            if let Some(&(best_p1, _)) = by_cf.iter().find(|&&(p1, _)| p1[0] == ld) {
                                if !opp_eval_set.contains(&best_p1) { opp_eval_set.push(best_p1); }
                            }
                            if let Some(&(best_p1, _)) = by_lin.iter().find(|&&(p1, _)| p1[0] == ld) {
                                if !opp_eval_set.contains(&best_p1) { opp_eval_set.push(best_p1); }
                            }
                        }
                    }

                    // Stage A (Selection seed 1): evaluate `own_union × opp_eval_set` to identify `j_BR`.
                    let mut sel_v: HashMap<([u8; 3], [u8; 3]), f64> = HashMap::new();
                    for (oi, &p0) in own_union.iter().enumerate() {
                        for (oj, &p1) in opp_eval_set.iter().enumerate() {
                            let oseed = mseed ^ 0xD1B5_4A32_D192_ED03 ^ ((oi as u64) << 16) ^ (oj as u64);
                            let v = nc2000_bot::preview_search::eval_turn1_oracle_cell(
                                &roster, &pcache, [p0, p1], select_iters, oseed,
                            );
                            sel_v.insert((p0, p1), v);
                        }
                    }

                    // Stage B (Unbiased Evaluation seed 2): evaluate ONLY each arm's chosen `j_BR` on a fresh seed.
                    let mut eval_cache: HashMap<([u8; 3], [u8; 3]), f64> = HashMap::new();
                    for (ai, dist) in arm_dists.iter().enumerate() {
                        let mut best_j = opp_eval_set[0];
                        let mut best_sel = f64::INFINITY;
                        for &p1 in &opp_eval_set {
                            let ev: f64 = dist.iter().map(|&(p0, pr)| pr * sel_v[&(p0, p1)]).sum();
                            if ev < best_sel {
                                best_sel = ev;
                                best_j = p1;
                            }
                        }
                        let mut unbiased_ev = 0.0;
                        for &(p0, pr) in dist {
                            let v2 = *eval_cache.entry((p0, best_j)).or_insert_with(|| {
                                let k_hash = (p0[0] as u64)
                                    | ((p0[1] as u64) << 4)
                                    | ((p0[2] as u64) << 8)
                                    | ((best_j[0] as u64) << 12)
                                    | ((best_j[1] as u64) << 16)
                                    | ((best_j[2] as u64) << 20);
                                let eseed = mseed ^ 0x94D0_49BB_1331_11EB ^ k_hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                                nc2000_bot::preview_search::eval_turn1_oracle_cell(
                                    &roster, &pcache, [p0, best_j], oracle_iters, eseed,
                                )
                            });
                            unbiased_ev += pr * v2;
                        }
                        acc[ai].worst_ev.push(unbiased_ev);
                    }
                }
                acc
            }));
        }
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut merged = vec![ArmAcc::default(); n_arms];
    for p in parts {
        for ai in 0..n_arms {
            merged[ai].worst_ev.extend_from_slice(&p[ai].worst_ev);
            merged[ai].supp_sizes.extend_from_slice(&p[ai].supp_sizes);
            merged[ai].mix_cnt += p[ai].mix_cnt;
            merged[ai].ms.extend_from_slice(&p[ai].ms);
        }
    }
    let mut rows = Vec::new();
    for ai in 0..n_arms {
        let n = merged[ai].worst_ev.len().max(1) as f64;
        let w_mean = merged[ai].worst_ev.iter().sum::<f64>() / n;
        let s_mean = merged[ai].supp_sizes.iter().sum::<f64>() / n;
        let mut ms_s = merged[ai].ms.clone();
        ms_s.sort_by(|a, b| a.total_cmp(b));
        let med_ms = if ms_s.is_empty() { 0.0 } else { ms_s[ms_s.len() / 2] };
        rows.push(json!({
            "arm": arms[ai],
            "matchups": merged[ai].worst_ev.len(),
            "worst_case_ev": (w_mean * 10000.0).round() / 10000.0,
            "avg_support": (s_mean * 100.0).round() / 100.0,
            "mix_rate": (merged[ai].mix_cnt as f64 / n * 1000.0).round() / 1000.0,
            "med_ms": (med_ms * 100.0).round() / 100.0,
        }));
    }
    println!(
        "{}",
        json!({
            "cmd": "preview-exploit",
            "oracle_iters": oracle_iters,
            "opp_cands": opp_cands_k,
            "arms": rows,
            "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        })
    );
}

/// Oracle Ceiling & Calibration Diagnostic (`preview-ceiling`):
/// For `pairs` matchups, evaluates the full union subgame (`top-20` Stage-1 + all tested arms' picks)
/// with an independent `cfv-0r` oracle, computes exact Nash `(p*, q*)`, and measures:
/// 1. Theoretical H2H ceiling of exact Nash `p*` vs `shipped:729000`
/// 2. Actual H2H of `cfprev` arms vs `shipped:729000` AND vs exact Nash `p*`
/// 3. Exploitability gap `V* - min_j V(p, j)` for every arm vs exact Nash `V*`
/// 4. Support recall of exact Nash support inside `select_diverse_support(k=12)`
/// 5. Fraction of matchups where `shipped:729000` is within `0.02` of `V*` (Nash indifference ceiling).
fn run_preview_ceiling(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let pairs: usize = flag("--pairs").map_or(24, |v| v.parse().unwrap());
    let oracle_iters: u32 = flag("--oracle-iters").map_or(384, |v| v.parse().unwrap());
    let seed: u64 = flag("--seed").map_or(2026120101, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let catalog_path = flag("--catalog").unwrap_or_else(|| "data/team-pool-v2/team-pool.json".to_string());
    let unseen_only = args.iter().any(|a| a == "--unseen-only");

    let root = repo_root();
    let cat_txt = std::fs::read_to_string(root.join(&catalog_path)).unwrap();
    let mut catalog: WeightedCatalog = serde_json::from_str(&cat_txt).unwrap();
    if unseen_only {
        catalog.teams.retain(|t| (canonical_team_hash(&t.sets) % 5) == 0);
    }
    let pool = Arc::new(load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json")));
    let weights: Vec<f64> = catalog.teams.iter().map(|t| t.draw_weight.unwrap_or(1.0)).collect();
    let w_sum: f64 = weights.iter().sum();
    let mut rng = SplitMix64::new(seed);
    let mut matchups = Vec::with_capacity(pairs);
    for pi in 0..pairs {
        let mut draw_t = |r: &mut SplitMix64| -> usize {
            let mut x = r.next_f64() * w_sum;
            for (i, &w) in weights.iter().enumerate() {
                x -= w;
                if x <= 0.0 { return i; }
            }
            weights.len() - 1
        };
        let ta = draw_t(&mut rng);
        let mut tb = draw_t(&mut rng);
        if tb == ta && catalog.teams.len() > 1 { tb = (ta + 1) % catalog.teams.len(); }
        matchups.push((pi, ta, tb, rng.next()));
    }

    let arm_specs = [
        "cfprev:0@stage1",
        "cfprev:120k@lcb@late",
        "cfprev:180k@lcb@t1mm",
        "cfprev:240k@lcb@t1mm",
        "cfprev:240k@lcb@late",
        "cfprev:240k@lcb@t1mm@k14",
        "cfprev:360k@lcb@t1mm@k14",
        "shipped:729000",
    ];
    let n_arms = arm_specs.len();
    let cursor = AtomicUsize::new(0);
    let t0 = Instant::now();

    #[derive(Default)]
    struct RowAcc {
        n: usize,
        nash_vs_ship: f64,
        ship_near_nash: usize,
        arm_vs_ship: Vec<f64>,
        arm_vs_nash: Vec<f64>,
        arm_regret_to_vstar: Vec<f64>,
        arm_worst_gap_to_vstar: Vec<f64>,
        s1_top12_mass_of_nash: f64,
    }

    let parts: Vec<RowAcc> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(pairs).max(1) {
            let (cat_r, pool_r, m_r, cur_r) = (&catalog, &pool, &matchups, &cursor);
            hs.push(s.spawn(move || {
                let mut acc = RowAcc {
                    arm_vs_ship: vec![0.0; n_arms],
                    arm_vs_nash: vec![0.0; n_arms],
                    arm_regret_to_vstar: vec![0.0; n_arms],
                    arm_worst_gap_to_vstar: vec![0.0; n_arms],
                    ..Default::default()
                };
                loop {
                    let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                    if idx >= m_r.len() { break; }
                    let (_, ia, ib, mseed) = m_r[idx];
                    let ta = &cat_r.teams[ia].sets;
                    let tb = &cat_r.teams[ib].sets;
                    let mut b = Battle::from_fixture(dex, "1,2,3,4", ta, tb).unwrap();
                    if !nc2000_engine::compact::is_compact_preview_supported(&b, dex) { continue; }
                    b.set_log_enabled(true);
                    let legal0 = b.legal_choices(dex, 0);
                    let legal1 = b.legal_choices(dex, 1);
                    let c0 = nc2000_bot::preview_search::canonical_preview_triples(dex, &mut b, 0);
                    let c1 = nc2000_bot::preview_search::canonical_preview_triples(dex, &mut b, 1);
                    if c0.is_empty() || c1.is_empty() { continue; }

                    let get_dist = |side: usize, sets: &[PokemonSet], legal: &[SearchChoice], spec: &str, s_seed: u64| -> Vec<([u8; 3], f64)> {
                        let (mut lp, mut obs) =
                            nc2000_bot::player_state::LocalPlayer::new_with_own_sets(&b, dex, side, sets.to_vec()).unwrap();
                        lp.observe(&b, dex, &mut obs).unwrap();
                        let mut belief = nc2000_bot::belief::Belief::new(dex, pool_r, &obs);
                        belief.sync(dex, &obs);
                        let mut lrng = SplitMix64::new(s_seed);
                        let pos = lp.position(dex, &mut belief, &obs, &mut lrng).unwrap();
                        match parse_preview_spec(spec).unwrap() {
                            Ok(iters) => {
                                let ch = nc2000_bot::preview_search::solve_shipped_preview(
                                    &pos, dex, &belief, &obs, side, iters, lrng,
                                );
                                if let SearchChoice::Team(t) = ch { vec![(t, 1.0)] } else { vec![] }
                            }
                            Err(cfg) => {
                                let res = nc2000_bot::preview_search::solve_cf_preview(
                                    &pos, dex, &belief, &obs, side, &cfg, &mut lrng,
                                );
                                let mut d = Vec::new();
                                for (li, &p) in res.policy_full.iter().enumerate() {
                                    if p > 1e-6 {
                                        if let Some(SearchChoice::Team(t)) = legal.get(li) {
                                            d.push((*t, p));
                                        }
                                    }
                                }
                                if d.is_empty() {
                                    if let SearchChoice::Team(t) = res.choice { d.push((t, 1.0)); }
                                }
                                let sm: f64 = d.iter().map(|x| x.1).sum();
                                for x in &mut d { x.1 /= sm.max(1e-12); }
                                d
                            }
                        }
                    };

                    let mut dists0 = Vec::with_capacity(n_arms);
                    let mut dists1 = Vec::with_capacity(n_arms);
                    for (ai, &sp) in arm_specs.iter().enumerate() {
                        dists0.push(get_dist(0, ta, &legal0, sp, mseed ^ ((ai as u64 + 1) * 0x9E37_79B9)));
                        dists1.push(get_dist(1, tb, &legal1, sp, mseed ^ ((ai as u64 + 101) * 0x9E37_79B9)));
                    }

                    // Build Stage-1 full matrix to pick top-20 rows & cols + union with all arms' support.
                    let roster = nc2000_engine::compact::CompactPreviewRoster::from_battle(&b, dex, None);
                    let pcache = nc2000_bot::tempo::PreviewTempoCache::build_for_preview_roster(
                        &roster,
                        [b.sides[0].party.len(), b.sides[1].party.len()],
                    );
                    let mut v0_mat = vec![0.5f64; c0.len() * c1.len()];
                    for (i, &p0) in c0.iter().enumerate() {
                        for (j, &p1) in c1.iter().enumerate() {
                            let (tbl, mut cb) = roster.build_turn1([p0, p1], 0);
                            cb.quick_claw_roll = 0;
                            let slc = pcache.slice_for(&cb);
                            v0_mat[i * c1.len() + j] = nc2000_bot::compact_az::evaluate_cf_value(&cb, &tbl, Some(&slc), 1.0);
                        }
                    }
                    let s1_sol = nc2000_bot::preview_search::solve_bayesian_rm_plus(
                        c0.len(),
                        &[(1.0, c1.len(), &v0_mat)],
                        1000,
                    );
                    let mut ord0: Vec<usize> = (0..c0.len()).collect();
                    ord0.sort_by(|&a, &b| s1_sol.row_vs_q[b].total_cmp(&s1_sol.row_vs_q[a]));
                    let top12_s1: Vec<[u8; 3]> = ord0.iter().take(12).map(|&i| c0[i]).collect();

                    let mut ord1: Vec<usize> = (0..c1.len()).collect();
                    ord1.sort_by(|&a, &b| s1_sol.col_vs_p[0][a].total_cmp(&s1_sol.col_vs_p[0][b]));

                    let mut rows_u: Vec<[u8; 3]> = ord0.iter().take(18).map(|&i| c0[i]).collect();
                    let mut cols_u: Vec<[u8; 3]> = ord1.iter().take(18).map(|&j| c1[j]).collect();
                    for d in &dists0 { for &(t, _) in d { if !rows_u.contains(&t) { rows_u.push(t); } } }
                    for d in &dists1 { for &(t, _) in d { if !cols_u.contains(&t) { cols_u.push(t); } } }

                    // Evaluate full `rows_u × cols_u` with `oracle_iters` `cfv-0r`.
                    let nr = rows_u.len();
                    let nc = cols_u.len();
                    let mut omat = vec![0.5f64; nr * nc];
                    for r in 0..nr {
                        for c in 0..nc {
                            let oseed = mseed ^ 0xA5A5_1234_5678_9ABC ^ ((r as u64) << 16) ^ (c as u64);
                            omat[r * nc + c] = nc2000_bot::preview_search::eval_turn1_oracle_cell(
                                &roster, &pcache, [rows_u[r], cols_u[c]], oracle_iters, oseed,
                            );
                        }
                    }
                    let osol = nc2000_bot::preview_search::solve_bayesian_rm_plus(
                        nr,
                        &[(1.0, nc, &omat)],
                        2000,
                    );
                    let eval_pair = |d0: &[([u8; 3], f64)], d1: &[([u8; 3], f64)]| -> f64 {
                        let mut s = 0.0;
                        for &(t0, p0) in d0 {
                            let r = rows_u.iter().position(|&x| x == t0).unwrap();
                            for &(t1, p1) in d1 {
                                let c = cols_u.iter().position(|&x| x == t1).unwrap();
                                s += p0 * p1 * omat[r * nc + c];
                            }
                        }
                        s
                    };
                    let nash0: Vec<([u8; 3], f64)> = rows_u.iter().enumerate().filter(|&(r, _)| osol.own_policy[r] > 1e-6).map(|(r, &t)| (t, osol.own_policy[r])).collect();
                    let nash1: Vec<([u8; 3], f64)> = cols_u.iter().enumerate().filter(|&(c, _)| osol.opp_policies[0][c] > 1e-6).map(|(c, &t)| (t, osol.opp_policies[0][c])).collect();

                    let ship_idx = n_arms - 1;
                    let v_n_vs_s = 0.5 * (eval_pair(&nash0, &dists1[ship_idx]) + (1.0 - eval_pair(&dists0[ship_idx], &nash1)));
                    acc.nash_vs_ship += v_n_vs_s;
                    if (v_n_vs_s - 0.5).abs() < 0.02 { acc.ship_near_nash += 1; }
                    let mass12: f64 = nash0.iter().filter(|&&(t, _)| top12_s1.contains(&t)).map(|&(_, p)| p).sum();
                    acc.s1_top12_mass_of_nash += mass12;

                    for ai in 0..n_arms {
                        let vs_s = 0.5 * (eval_pair(&dists0[ai], &dists1[ship_idx]) + (1.0 - eval_pair(&dists0[ship_idx], &dists1[ai])));
                        let vs_n = 0.5 * (eval_pair(&dists0[ai], &nash1) + (1.0 - eval_pair(&nash0, &dists1[ai])));
                        // Side 0 worst-case gap to V*
                        let mut w0 = f64::INFINITY;
                        for c in 0..nc {
                            let ev: f64 = dists0[ai].iter().map(|&(t0, p0)| {
                                let r = rows_u.iter().position(|&x| x == t0).unwrap();
                                p0 * omat[r * nc + c]
                            }).sum();
                            if ev < w0 { w0 = ev; }
                        }
                        let mut w1 = f64::INFINITY;
                        for r in 0..nr {
                            let ev: f64 = dists1[ai].iter().map(|&(t1, p1)| {
                                let c = cols_u.iter().position(|&x| x == t1).unwrap();
                                p1 * (1.0 - omat[r * nc + c])
                            }).sum();
                            if ev < w1 { w1 = ev; }
                        }
                        let avg_wgap = 0.5 * ((osol.value - w0) + ((1.0 - osol.value) - w1));
                        acc.arm_vs_ship[ai] += vs_s;
                        acc.arm_vs_nash[ai] += vs_n;
                        acc.arm_regret_to_vstar[ai] += 0.5 - vs_n;
                        acc.arm_worst_gap_to_vstar[ai] += avg_wgap;
                    }
                    acc.n += 1;
                }
                acc
            }));
        }
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut tot_n = 0usize;
    let mut nash_vs_ship = 0.0;
    let mut ship_near = 0usize;
    let mut mass12 = 0.0;
    let mut avs = vec![0.0; n_arms];
    let mut avn = vec![0.0; n_arms];
    let mut awg = vec![0.0; n_arms];
    for p in parts {
        tot_n += p.n;
        nash_vs_ship += p.nash_vs_ship;
        ship_near += p.ship_near_nash;
        mass12 += p.s1_top12_mass_of_nash;
        for ai in 0..n_arms {
            avs[ai] += p.arm_vs_ship[ai];
            avn[ai] += p.arm_vs_nash[ai];
            awg[ai] += p.arm_worst_gap_to_vstar[ai];
        }
    }
    let nf = tot_n.max(1) as f64;
    let rows: Vec<Value> = (0..n_arms).map(|ai| json!({
        "arm": arm_specs[ai],
        "vs_shipped_ev": (avs[ai] / nf * 10000.0).round() / 10000.0,
        "vs_shipped_elo": (elo_diff(avs[ai] / nf) * 10.0).round() / 10.0,
        "vs_exact_nash_ev": (avn[ai] / nf * 10000.0).round() / 10000.0,
        "exploit_gap_to_vstar": (awg[ai] / nf * 10000.0).round() / 10000.0,
    })).collect();
    println!("{}", json!({
        "cmd": "preview-ceiling",
        "matchups": tot_n,
        "oracle_iters": oracle_iters,
        "exact_nash_vs_shipped_ev": (nash_vs_ship / nf * 10000.0).round() / 10000.0,
        "exact_nash_vs_shipped_elo": (elo_diff(nash_vs_ship / nf) * 10.0).round() / 10.0,
        "shipped_within_2pp_of_nash": (ship_near as f64 / nf * 1000.0).round() / 1000.0,
        "s1_top12_nash_mass_recall": (mass12 / nf * 1000.0).round() / 1000.0,
        "arms": rows,
        "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
    }));
}

/// Micro-benchmark `extract_raw_value_features` + `evaluate_cf_value` latency (`us/leaf`)
/// and emit deterministic golden `f64::to_bits()` hashes for 0-ULP backward-parity checks.
fn run_cfv_bench(dex: &Dex, args: &[String]) {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let loops: usize = flag("--loops").map_or(50_000, |v| v.parse().unwrap());
    let root = repo_root();
    let bpool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let mut rng = SplitMix64::new(2026120199);

    let mut cases: Vec<(
        nc2000_engine::compact::MatchTable,
        nc2000_engine::compact::CompactBattle,
        nc2000_bot::tempo::TempoMatchCache,
    )> = Vec::new();
    while cases.len() < 16 {
        let ta = rng.below(bpool.teams.len());
        let tb = rng.below(bpool.teams.len());
        let bseed = rng.battle_seed();
        let Ok(mut b) = Battle::from_fixture(dex, &bseed, &bpool.teams[ta].sets, &bpool.teams[tb].sets) else {
            continue;
        };
        b.set_log_enabled(false);
        let tp0_cs = b.legal_choices(dex, 0);
        let tp1_cs = b.legal_choices(dex, 1);
        let tp0 = tp0_cs[rng.below(tp0_cs.len())];
        let tp1 = tp1_cs[rng.below(tp1_cs.len())];
        if b.apply_choices(dex, [Some(tp0), Some(tp1)]).is_err()
            || !nc2000_engine::compact::is_compact_tree_supported(&b, dex)
        {
            continue;
        }
        let (table, mut cb) = nc2000_engine::compact::CompactBattle::from_battle(&b, dex);
        let cache = nc2000_bot::tempo::TempoMatchCache::build(&cb, &table);
        if cases.len() % 2 == 1 {
            // Advance a few turns for midgame coverage.
            for _ in 0..8 {
                if cb.outcome != 0 {
                    break;
                }
                let res = nc2000_bot::compact_az::search_compact_cfv_puct(&cb, &table, &cache, 32, rng.next());
                cb.apply_actions(&table, res.best_acts);
            }
            if cb.outcome != 0 || cb.sides[0].alive_mask.count_ones() + cb.sides[1].alive_mask.count_ones() < 4 {
                continue;
            }
        }
        cb.quick_claw_roll = 0;
        cases.push((table, cb, cache));
    }

    let golden_bits: Vec<u64> = cases
        .iter()
        .map(|(t, cb, c)| nc2000_bot::compact_az::evaluate_cf_value(cb, t, Some(c), 1.0).to_bits())
        .collect();
    let mut golden_xor = 0u64;
    for (i, &bits) in golden_bits.iter().enumerate() {
        golden_xor ^= bits.rotate_left((i * 5) as u32);
    }

    // Warmup
    let mut sink_f = 0.0f64;
    for i in 0..1000 {
        let (t, cb, c) = &cases[i & 15];
        let (x0, x1) = nc2000_bot::compact_az::extract_raw_value_features(cb, t, Some(c));
        sink_f += f64::from(x0[0] + x1[0]);
        sink_f += nc2000_bot::compact_az::evaluate_cf_value(cb, t, Some(c), 1.0);
    }

    let t_ext = Instant::now();
    let mut ext_acc = 0.0f32;
    for i in 0..loops {
        let (t, cb, c) = &cases[i & 15];
        let (x0, x1) = nc2000_bot::compact_az::extract_raw_value_features(cb, t, Some(c));
        ext_acc += x0[i % nc2000_bot::compact_az::CF_RAW_VALUE_SIDE_DIM]
            + x1[i % nc2000_bot::compact_az::CF_RAW_VALUE_SIDE_DIM];
    }
    let ext_us = t_ext.elapsed().as_secs_f64() * 1e6 / (loops as f64);

    let t_eval = Instant::now();
    let mut eval_acc = 0.0f64;
    for i in 0..loops {
        let (t, cb, c) = &cases[i & 15];
        eval_acc += nc2000_bot::compact_az::evaluate_cf_value(cb, t, Some(c), 1.0);
    }
    let eval_us = t_eval.elapsed().as_secs_f64() * 1e6 / (loops as f64);
    std::hint::black_box((sink_f, ext_acc, eval_acc));

    println!(
        "{}",
        json!({
            "cmd": "cfv-bench",
            "side_dim": nc2000_bot::compact_az::CF_RAW_VALUE_SIDE_DIM,
            "loops": loops,
            "extract_us": (ext_us * 1000.0).round() / 1000.0,
            "eval_total_us": (eval_us * 1000.0).round() / 1000.0,
            "nn_only_us": ((eval_us - ext_us) * 1000.0).round() / 1000.0,
            "golden_xor": format!("{golden_xor:#018x}"),
            "golden_bits_0_3": &golden_bits[..4],
        })
    );
}

fn species_level_signature(sets: &[PokemonSet]) -> Vec<(String, u8)> {
    let mut sig: Vec<(String, u8)> = sets
        .iter()
        .map(|s| (s.species.to_ascii_lowercase(), s.level))
        .collect();
    sig.sort();
    sig
}

/// Fixed-preview wrapper around `BlindAgent` for optional `--t1-blind-frac > 0` ablations.
struct FixedPreviewBlindAgent {
    pick: SearchChoice,
    inner: BlindAgent,
}

impl Agent for FixedPreviewBlindAgent {
    fn name(&self) -> String {
        "fixed-preview-blind".to_string()
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if battle.request_state == nc2000_engine::state::RequestState::TeamPreview {
            return self.pick;
        }
        self.inner.choose(battle, dex, side, choices)
    }
}

fn play_blind6v6_fixed_pair(
    dex: &Dex,
    sets_a: &[PokemonSet],
    sets_b: &[PokemonSet],
    pick_a: [u8; 3],
    pick_b: [u8; 3],
    iters: u32,
    seed: u64,
) -> f64 {
    let mut rng = SplitMix64::new(seed);
    let bseed = rng.battle_seed();
    let Ok(mut b) = Battle::from_fixture(dex, &bseed, sets_a, sets_b) else {
        return 0.5;
    };
    b.set_log_enabled(true);
    let pool_a = Arc::new(MetaPool {
        teams: vec![nc2000_bot::preview::MetaTeam {
            id: "opp_b".into(),
            sets: sets_b.to_vec(),
            weight: Some(1.0),
        }],
    });
    let pool_b = Arc::new(MetaPool {
        teams: vec![nc2000_bot::preview::MetaTeam {
            id: "opp_a".into(),
            sets: sets_a.to_vec(),
            weight: Some(1.0),
        }],
    });
    let cfg = RmConfig {
        iterations: iters,
        rule: SelRule::Ucb,
        c: 0.4,
        settlement: nc2000_bot::accounting::LeafSettlementMode::CfValuePuct,
        ..Default::default()
    };
    let mut a0 = FixedPreviewBlindAgent {
        pick: SearchChoice::Team(pick_a),
        inner: BlindAgent::new(cfg.clone(), pool_a, None, rng.next()),
    };
    let mut a1 = FixedPreviewBlindAgent {
        pick: SearchChoice::Team(pick_b),
        inner: BlindAgent::new(cfg, pool_b, None, rng.next()),
    };
    let mut agents: [&mut dyn Agent; 2] = [&mut a0, &mut a1];
    match play_game(dex, &mut b, &mut agents, 300) {
        Ok(GameResult::Outcome(Outcome::P1Win)) => 1.0,
        Ok(GameResult::Outcome(Outcome::P2Win)) => 0.0,
        _ => 0.5,
    }
}

#[derive(Deserialize)]
struct PoolSetsOnlyFile {
    teams: Vec<PoolSetsOnlyEntry>,
}

#[derive(Deserialize)]
struct PoolSetsOnlyEntry {
    sets: Vec<PokemonSet>,
}

/// Zero-leakage `CF-V_net v3` CRN Midgame + Turn-1 Compositional Pair Collector (`collect-cf-v3`).
///
/// - Loads `data/belief-pool-v3/belief-pool.json` and STRICTLY excludes every team matching ANY
///   `data/team-pool-v4/team-pool.json` team by canonical order-insensitive hash OR species+levels
///   shell signature.
/// - Splits clean teams by `canonical_team_hash % 5`: `!= 0` (train) vs `== 0` (pure unseen val),
///   drawing `(1 - val_frac)` games from `train x train` and `val_frac` from `val x val` (zero cross-leakage).
/// - Labels BOTH Midgame CF pairs (`turn >= 2`, `alive >= 4`) and Turn-1 Compositional CF pairs
///   (`turn == 1`, 65% 1-bench-slot swap `kind=10` / 35% lead swap `kind=11`) with CRN full-game
///   `cfv-0r` (`search_compact_cfv_puct`) + conditional Rao-Blackwellized `n0+n1 <= 3` `linear.rs` exit
///   (excluding live `Endure`/`FlailReversal`/`FocusBand`/`DestinyBond`/`PerishSong`).
/// - Supports `--shard` / `--game-start` / `--games` / `--threads` / `--out` for direct `barn` dispatch.
fn run_collect_cf_v3(dex: &Dex, args: &[String]) {
    use nc2000_bot::compact_az::{cf_full_game_rollout, CfPairRecV3};
    use nc2000_bot::preview_search::canonical_preview_triples;
    use nc2000_bot::tempo::PreviewTempoCache;
    use nc2000_engine::compact::CompactPreviewRoster;
    use std::collections::HashSet;

    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let has_flag = |name: &str| args.iter().any(|a| a == name);

    let games: usize = flag("--games").map_or(200, |v| v.parse().unwrap());
    let shard: Option<usize> = flag("--shard").map(|v| v.parse().unwrap());
    let game_start: usize = match flag("--game-start") {
        Some(v) => v.parse().unwrap(),
        None => shard.unwrap_or(0) * games,
    };
    let mid_states: usize = flag("--mid-states").map_or(4, |v| v.parse().unwrap());
    let mid_iters: u32 = flag("--mid-iters").map_or(128, |v| v.parse().unwrap());
    let mid_m: usize = flag("--mid-m").map_or(8, |v| v.parse().unwrap());
    let t1_pairs: usize = flag("--t1-pairs").map_or(2, |v| v.parse().unwrap());
    let t1_iters: u32 = flag("--t1-iters").map_or(256, |v| v.parse().unwrap());
    let t1_m: usize = flag("--t1-m").map_or(8, |v| v.parse().unwrap());
    let t1_blind_frac: f64 = flag("--t1-blind-frac").map_or(0.0, |v| v.parse().unwrap());
    let t1_blind_iters: u32 = flag("--t1-blind-iters").map_or(1000, |v| v.parse().unwrap());
    let play_iters: u32 = flag("--play-iters").map_or(192, |v| v.parse().unwrap());
    let max_delta_turns: u16 = flag("--max-delta-turns").map_or(160, |v| v.parse().unwrap());
    let val_frac: f64 = flag("--val-frac").map_or(0.15, |v| v.parse().unwrap());
    let use_rb_exit = !has_flag("--no-rb-exit");
    let seed: u64 = flag("--seed").map_or(2026120101, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap());
    let out_arg = flag("--out").unwrap_or_else(|| "/tmp/cf_v3_data".to_string());

    let root = repo_root();
    let bpool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let tp4_txt = std::fs::read_to_string(root.join("data/team-pool-v4/team-pool.json"))
        .expect("read data/team-pool-v4/team-pool.json");
    let tp4: PoolSetsOnlyFile = serde_json::from_str(&tp4_txt).expect("parse team-pool-v4");

    let tp4_hashes: HashSet<u64> = tp4
        .teams
        .iter()
        .map(|t| canonical_team_hash(&t.sets))
        .collect();
    let tp4_shells: HashSet<Vec<(String, u8)>> = tp4
        .teams
        .iter()
        .map(|t| species_level_signature(&t.sets))
        .collect();

    let mut excluded_exact = 0usize;
    let mut excluded_shell_only = 0usize;
    let mut train_pool: Vec<(usize, u64)> = Vec::new();
    let mut val_pool: Vec<(usize, u64)> = Vec::new();
    for (ti, team) in bpool.teams.iter().enumerate() {
        let h = canonical_team_hash(&team.sets);
        let sh = species_level_signature(&team.sets);
        let exact_hit = tp4_hashes.contains(&h);
        let shell_hit = tp4_shells.contains(&sh);
        if exact_hit {
            excluded_exact += 1;
            continue;
        }
        if shell_hit {
            excluded_shell_only += 1;
            continue;
        }
        if (h % 5) == 0 {
            val_pool.push((ti, h));
        } else {
            train_pool.push((ti, h));
        }
    }
    assert!(
        !train_pool.is_empty() && !val_pool.is_empty(),
        "clean pools must be non-empty: train={}, val={}",
        train_pool.len(),
        val_pool.len()
    );

    #[derive(Clone, Copy)]
    struct JobV3 {
        g: usize,
        ta: usize,
        tb: usize,
        th0: u64,
        th1: u64,
        s_seed: u64,
    }
    let mix_u64 = |a: u64, b: u64| -> u64 {
        let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };

    let mut jobs = Vec::with_capacity(games);
    for offset in 0..games {
        let g = game_start + offset;
        let mut jrng = SplitMix64::new(mix_u64(seed, g as u64));
        let is_val = jrng.next_f64() < val_frac;
        let src = if is_val { &val_pool } else { &train_pool };
        let (ta, th0) = src[jrng.below(src.len())];
        let (tb, th1) = src[jrng.below(src.len())];
        let s_seed = jrng.next();
        jobs.push(JobV3 { g, ta, tb, th0, th1, s_seed });
    }

    #[derive(Default)]
    struct WorkerAcc {
        pairs: Vec<CfPairRecV3>,
        finished_games: usize,
        mid_rows: usize,
        t1_bench_rows: usize,
        t1_lead_rows: usize,
        blind_rows: usize,
        train_rows: usize,
        val_rows: usize,
        total_rollouts: usize,
        turn_cap_cutoffs: usize,
        rb_exits: usize,
        sum_rollout_turns: u64,
    }

    let t0 = Instant::now();
    let cursor = AtomicUsize::new(0);
    let acc: WorkerAcc = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(jobs.len()).max(1) {
            let (bp_r, jobs_r, cur_r) = (&bpool, &jobs, &cursor);
            hs.push(s.spawn(move || {
                let mut w = WorkerAcc::default();
                loop {
                    let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                    if idx >= jobs_r.len() {
                        break;
                    }
                    let job = jobs_r[idx];
                    let sets_a = &bp_r.teams[job.ta].sets;
                    let sets_b = &bp_r.teams[job.tb].sets;
                    let mut grng = SplitMix64::new(job.s_seed);
                    let bseed = grng.battle_seed();
                    let Ok(mut b_tp) = Battle::from_fixture(dex, &bseed, sets_a, sets_b) else {
                        continue;
                    };
                    b_tp.set_log_enabled(false);
                    if !nc2000_engine::compact::is_compact_preview_supported(&b_tp, dex) {
                        continue;
                    }

                    let roster = CompactPreviewRoster::from_battle(&b_tp, dex, None);
                    let plen0 = sets_a.len().min(6);
                    let plen1 = sets_b.len().min(6);
                    let pcache = PreviewTempoCache::build_for_preview_roster(&roster, [plen0, plen1]);
                    let c0 = canonical_preview_triples(dex, &mut b_tp, 0);
                    let c1 = canonical_preview_triples(dex, &mut b_tp, 1);
                    if c0.is_empty() || c1.is_empty() {
                        continue;
                    }

                    // Quick 0-ply CF-V_net matrix for 50% tactical top-k sampling (~1.8 ms for 48x48).
                    let mut row_avg = vec![0.0f64; c0.len()];
                    let mut col_avg = vec![0.0f64; c1.len()];
                    let need_tactical = t1_pairs > 0 || mid_states > 0;
                    if need_tactical {
                        for (i, &p0) in c0.iter().enumerate() {
                            for (j, &p1) in c1.iter().enumerate() {
                                let (tbl, mut cb1) = roster.build_turn1([p0, p1], 0);
                                cb1.quick_claw_roll = 0;
                                let slc = pcache.slice_for(&cb1);
                                let v = nc2000_bot::compact_az::evaluate_cf_value(
                                    &cb1,
                                    &tbl,
                                    Some(&slc),
                                    1.0,
                                );
                                row_avg[i] += v;
                                col_avg[j] += 1.0 - v;
                            }
                        }
                    }
                    let mut top0: Vec<usize> = (0..c0.len()).collect();
                    top0.sort_by(|&a, &b| row_avg[b].total_cmp(&row_avg[a]));
                    top0.truncate(12.min(c0.len()));
                    let mut top1: Vec<usize> = (0..c1.len()).collect();
                    top1.sort_by(|&a, &b| col_avg[b].total_cmp(&col_avg[a]));
                    top1.truncate(12.min(c1.len()));

                    let is_val_game = (job.th0 % 5) == 0 && (job.th1 % 5) == 0;
                    let sample_pick = |cs: &[[u8; 3]], top: &[usize], rng: &mut SplitMix64| -> [u8; 3] {
                        if rng.next_f64() < 0.50 || top.is_empty() {
                            cs[rng.below(cs.len())]
                        } else {
                            cs[top[rng.below(top.len())]]
                        }
                    };

                    // ---------------------------------------------------------
                    // A. Turn-1 Compositional CF Pairs (65% bench swap / 35% lead swap)
                    // ---------------------------------------------------------
                    for _ in 0..t1_pairs {
                        let p0 = sample_pick(&c0, &top0, &mut grng);
                        let p1 = sample_pick(&c1, &top1, &mut grng);
                        let pert_side = grng.below(2);
                        let (base_p, legal_cs) = if pert_side == 0 { (p0, &c0) } else { (p1, &c1) };
                        // Strictly NC2000-legal 1-bench-slot swaps (same lead, shares 1 bench slot)
                        let mut bench_swaps: Vec<[u8; 3]> = Vec::new();
                        let mut lead_swaps: Vec<[u8; 3]> = Vec::new();
                        let mut base_sorted = base_p;
                        base_sorted.sort_unstable();
                        for &cand in legal_cs {
                            if cand == base_p {
                                continue;
                            }
                            if cand[0] == base_p[0] {
                                let shares = cand[1] == base_p[1]
                                    || cand[1] == base_p[2]
                                    || cand[2] == base_p[1]
                                    || cand[2] == base_p[2];
                                if shares {
                                    bench_swaps.push(cand);
                                }
                            } else {
                                let mut cs = cand;
                                cs.sort_unstable();
                                if cs == base_sorted {
                                    lead_swaps.push(cand);
                                }
                            }
                        }
                        let (alt_p, kind) = if !bench_swaps.is_empty()
                            && (lead_swaps.is_empty() || grng.next_f64() < 0.65)
                        {
                            // Legal 1-bench-slot swap (kind = 10)
                            (bench_swaps[grng.below(bench_swaps.len())], 10u8)
                        } else if !lead_swaps.is_empty() {
                            // Legal lead swap (kind = 11)
                            (lead_swaps[grng.below(lead_swaps.len())], 11u8)
                        } else {
                            continue;
                        };
                        let picks_a = [p0, p1];
                        let picks_b = if pert_side == 0 { [alt_p, p1] } else { [p0, alt_p] };

                        let (tbl_a, mut cb_a) = roster.build_turn1(picks_a, 0);
                        cb_a.quick_claw_roll = 0;
                        let slc_a = pcache.slice_for(&cb_a);
                        let (tbl_b, mut cb_b) = roster.build_turn1(picks_b, 0);
                        cb_b.quick_claw_roll = 0;
                        let slc_b = pcache.slice_for(&cb_b);

                        let (x0, x1) = nc2000_bot::compact_az::extract_raw_value_features(
                            &cb_a,
                            &tbl_a,
                            Some(&slc_a),
                        );
                        let (xp0, xp1) = nc2000_bot::compact_az::extract_raw_value_features(
                            &cb_b,
                            &tbl_b,
                            Some(&slc_b),
                        );

                        let use_blind = t1_blind_frac > 0.0 && grng.next_f64() < t1_blind_frac;
                        let base_pseed = grng.next();
                        let mut sum0 = 0.0f64;
                        let mut sum1 = 0.0f64;
                        for m in 0..t1_m {
                            let pseed =
                                base_pseed ^ (m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                            if use_blind {
                                sum0 += play_blind6v6_fixed_pair(
                                    dex,
                                    sets_a,
                                    sets_b,
                                    picks_a[0],
                                    picks_a[1],
                                    t1_blind_iters,
                                    pseed,
                                );
                                sum1 += play_blind6v6_fixed_pair(
                                    dex,
                                    sets_a,
                                    sets_b,
                                    picks_b[0],
                                    picks_b[1],
                                    t1_blind_iters,
                                    pseed,
                                );
                                w.total_rollouts += 2;
                            } else {
                                let (v0, t_el0, cap0, rb0) = cf_full_game_rollout(
                                    &cb_a,
                                    &tbl_a,
                                    &slc_a,
                                    t1_iters,
                                    max_delta_turns,
                                    pseed,
                                    use_rb_exit,
                                );
                                let (v1, t_el1, cap1, rb1) = cf_full_game_rollout(
                                    &cb_b,
                                    &tbl_b,
                                    &slc_b,
                                    t1_iters,
                                    max_delta_turns,
                                    pseed,
                                    use_rb_exit,
                                );
                                sum0 += v0;
                                sum1 += v1;
                                w.total_rollouts += 2;
                                w.sum_rollout_turns += u64::from(t_el0) + u64::from(t_el1);
                                w.turn_cap_cutoffs += usize::from(cap0) + usize::from(cap1);
                                w.rb_exits += usize::from(rb0) + usize::from(rb1);
                            }
                        }
                        let v_cf = (sum0 / (t1_m.max(1) as f64)) as f32;
                        let vp_cf = (sum1 / (t1_m.max(1) as f64)) as f32;
                        w.pairs.push(CfPairRecV3 {
                            team_hash_0: job.th0,
                            team_hash_1: job.th1,
                            game_id: job.g as u32,
                            turn: 1,
                            n0: cb_a.sides[0].alive_mask.count_ones() as u8,
                            n1: cb_a.sides[1].alive_mask.count_ones() as u8,
                            kind,
                            is_blind: u8::from(use_blind),
                            _pad: [0; 2],
                            v_cf,
                            vp_cf,
                            z_game: v_cf,
                            x0,
                            x1,
                            xp0,
                            xp1,
                        });
                        if kind == 10 {
                            w.t1_bench_rows += 1;
                        } else {
                            w.t1_lead_rows += 1;
                        }
                        if use_blind {
                            w.blind_rows += 1;
                        }
                        if is_val_game {
                            w.val_rows += 1;
                        } else {
                            w.train_rows += 1;
                        }
                    }

                    // ---------------------------------------------------------
                    // B. Midgame Trajectory + CRN Full-Game `cfv-0r` CF Pairs
                    // ---------------------------------------------------------
                    if mid_states > 0 {
                        let p0 = sample_pick(&c0, &top0, &mut grng);
                        let p1 = sample_pick(&c1, &top1, &mut grng);
                        let (table, mut cb) = roster.build_turn1([p0, p1], grng.next());
                        let cache = pcache.slice_for(&cb);
                        let mut snapshots: Vec<(nc2000_engine::compact::CompactBattle, bool)> =
                            Vec::with_capacity(64);
                        let mut outcome_opt: Option<f32> = None;

                        for _ in 0..720 {
                            if cb.outcome != 0 {
                                outcome_opt = Some(match cb.outcome {
                                    1 => 1.0,
                                    2 => 0.0,
                                    _ => 0.5,
                                });
                                break;
                            }
                            if let Some(v) = nc2000_bot::eval::compact_turn_limit_outcome(&cb, &table) {
                                outcome_opt = Some(v as f32);
                                break;
                            }
                            if cb.turn > 300 {
                                break;
                            }
                            let is_idle = cb.sides[0].request == 1 && cb.sides[1].request == 1;
                            if is_idle {
                                let mut prng = nc2000_engine::prng::Prng::new(grng.next());
                                let qc = Battle::roll_quick_claw_with(table.ruleset, &mut prng);
                                cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
                                cb.prng_seed = prng.seed();
                            }
                            let mut a_buf = [[0u8; 7]; 2];
                            let n0 = if cb.sides[0].request > 0 {
                                cb.legal_actions(&table, 0, &mut a_buf[0]).min(6)
                            } else {
                                0
                            };
                            let n1 = if cb.sides[1].request > 0 {
                                cb.legal_actions(&table, 1, &mut a_buf[1]).min(6)
                            } else {
                                0
                            };
                            if n0 == 0 && n1 == 0 {
                                break;
                            }
                            if n0 <= 1 && n1 <= 1 {
                                let p0_a = if n0 == 1 { a_buf[0][0] } else { 7 };
                                let p1_a = if n1 == 1 { a_buf[1][0] } else { 7 };
                                cb.apply_actions(&table, [p0_a, p1_a]);
                                continue;
                            }
                            let al0 = cb.sides[0].alive_mask.count_ones();
                            let al1 = cb.sides[1].alive_mask.count_ones();
                            if is_idle && al0 + al1 >= 4 && cb.turn >= 2 {
                                let is_hi = cb.turn >= 25 || (al0 == 2 && al1 == 2);
                                snapshots.push((cb, is_hi));
                            }
                            let sseed = grng.next();
                            let res = nc2000_bot::compact_az::search_compact_cfv_puct(
                                &cb, &table, &cache, play_iters, sseed,
                            );
                            let mut play_acts = res.best_acts;
                            if cb.turn <= 10 && grng.next_f64() < 0.14 {
                                for side in 0..2usize {
                                    let k = (res.n_acts[side] as usize).min(6);
                                    if k >= 2 {
                                        let noop =
                                            nc2000_bot::compact_az::compact_interior_noop_mask(
                                                &cb,
                                                &table,
                                                side,
                                                &res.acts[side],
                                                k,
                                            );
                                        let mut safe = [0u8; 6];
                                        let mut ns = 0usize;
                                        for i in 0..k {
                                            if ((noop >> i) & 1) == 0 && res.visits[side][i] > 0 {
                                                safe[ns] = res.acts[side][i];
                                                ns += 1;
                                            }
                                        }
                                        if ns > 0 {
                                            play_acts[side] = safe[grng.below(ns)];
                                        }
                                    }
                                }
                            }
                            cb.apply_actions(&table, play_acts);
                        }

                        if !snapshots.is_empty() {
                            let mut hi_snaps = Vec::new();
                            let mut lo_snaps = Vec::new();
                            for (scb, hi) in snapshots {
                                if hi {
                                    hi_snaps.push(scb);
                                } else {
                                    lo_snaps.push(scb);
                                }
                            }
                            for v in [&mut hi_snaps, &mut lo_snaps] {
                                for i in (1..v.len()).rev() {
                                    let j = grng.below(i + 1);
                                    v.swap(i, j);
                                }
                            }
                            let mut picked = Vec::with_capacity(mid_states);
                            let hi_quota = (mid_states / 2).min(hi_snaps.len());
                            picked.extend_from_slice(&hi_snaps[..hi_quota]);
                            for &scb in &lo_snaps {
                                if picked.len() >= mid_states {
                                    break;
                                }
                                picked.push(scb);
                            }
                            for &scb in &hi_snaps[hi_quota..] {
                                if picked.len() >= mid_states {
                                    break;
                                }
                                picked.push(scb);
                            }

                            for &s_cb in &picked {
                                let Some((sp_cb, kind)) =
                                    nc2000_bot::compact_az::apply_cf_perturbation(
                                        &s_cb, &table, &mut grng,
                                    )
                                else {
                                    continue;
                                };
                                let base_pseed = grng.next();
                                let mut sum0 = 0.0f64;
                                let mut sum1 = 0.0f64;
                                for m in 0..mid_m {
                                    let pseed = base_pseed
                                        ^ (m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                                    let (v0, t_el0, cap0, rb0) = cf_full_game_rollout(
                                        &s_cb,
                                        &table,
                                        &cache,
                                        mid_iters,
                                        max_delta_turns,
                                        pseed,
                                        use_rb_exit,
                                    );
                                    let (v1, t_el1, cap1, rb1) = cf_full_game_rollout(
                                        &sp_cb,
                                        &table,
                                        &cache,
                                        mid_iters,
                                        max_delta_turns,
                                        pseed,
                                        use_rb_exit,
                                    );
                                    sum0 += v0;
                                    sum1 += v1;
                                    w.total_rollouts += 2;
                                    w.sum_rollout_turns += u64::from(t_el0) + u64::from(t_el1);
                                    w.turn_cap_cutoffs += usize::from(cap0) + usize::from(cap1);
                                    w.rb_exits += usize::from(rb0) + usize::from(rb1);
                                }
                                let v_cf = (sum0 / (mid_m.max(1) as f64)) as f32;
                                let vp_cf = (sum1 / (mid_m.max(1) as f64)) as f32;
                                let z_game = outcome_opt.unwrap_or(v_cf);
                                let (x0, x1) = nc2000_bot::compact_az::extract_raw_value_features(
                                    &s_cb,
                                    &table,
                                    Some(&cache),
                                );
                                let (xp0, xp1) = nc2000_bot::compact_az::extract_raw_value_features(
                                    &sp_cb,
                                    &table,
                                    Some(&cache),
                                );
                                w.pairs.push(CfPairRecV3 {
                                    team_hash_0: job.th0,
                                    team_hash_1: job.th1,
                                    game_id: job.g as u32,
                                    turn: s_cb.turn,
                                    n0: s_cb.sides[0].alive_mask.count_ones() as u8,
                                    n1: s_cb.sides[1].alive_mask.count_ones() as u8,
                                    kind,
                                    is_blind: 0,
                                    _pad: [0; 2],
                                    v_cf,
                                    vp_cf,
                                    z_game,
                                    x0,
                                    x1,
                                    xp0,
                                    xp1,
                                });
                                w.mid_rows += 1;
                                if is_val_game {
                                    w.val_rows += 1;
                                } else {
                                    w.train_rows += 1;
                                }
                            }
                        }
                    }
                    w.finished_games += 1;
                }
                w
            }));
        }
        let mut tot = WorkerAcc::default();
        for h in hs {
            let p = h.join().unwrap();
            tot.pairs.extend(p.pairs);
            tot.finished_games += p.finished_games;
            tot.mid_rows += p.mid_rows;
            tot.t1_bench_rows += p.t1_bench_rows;
            tot.t1_lead_rows += p.t1_lead_rows;
            tot.blind_rows += p.blind_rows;
            tot.train_rows += p.train_rows;
            tot.val_rows += p.val_rows;
            tot.total_rollouts += p.total_rollouts;
            tot.turn_cap_cutoffs += p.turn_cap_cutoffs;
            tot.rb_exits += p.rb_exits;
            tot.sum_rollout_turns += p.sum_rollout_turns;
        }
        tot
    });

    let out_path = if out_arg.ends_with(".bin") {
        PathBuf::from(&out_arg)
    } else {
        PathBuf::from(format!("{out_arg}.cfpair3.bin"))
    };
    write_pod_slice(&out_path, &acc.pairs);

    let wall_s = t0.elapsed().as_secs_f64();
    let roll_den = acc.total_rollouts.max(1) as f64;
    println!(
        "{}",
        json!({
            "cmd": "collect-cf-v3",
            "side_dim": nc2000_bot::compact_az::CF_RAW_VALUE_SIDE_DIM,
            "rec_bytes": std::mem::size_of::<CfPairRecV3>(),
            "shard": shard,
            "game_start": game_start,
            "games": games,
            "finished_games": acc.finished_games,
            "threads": threads,
            "pool_total": bpool.teams.len(),
            "excluded_tp4_exact": excluded_exact,
            "excluded_tp4_shell": excluded_shell_only,
            "excluded_tp4_total": excluded_exact + excluded_shell_only,
            "clean_train_teams": train_pool.len(),
            "clean_val_teams": val_pool.len(),
            "pair_rows": acc.pairs.len(),
            "mid_rows": acc.mid_rows,
            "t1_bench_rows": acc.t1_bench_rows,
            "t1_lead_rows": acc.t1_lead_rows,
            "t1_frac": ((acc.t1_bench_rows + acc.t1_lead_rows) as f64 / acc.pairs.len().max(1) as f64 * 1000.0).round() / 1000.0,
            "blind_rows": acc.blind_rows,
            "train_rows": acc.train_rows,
            "val_rows": acc.val_rows,
            "mid_iters": mid_iters,
            "mid_m": mid_m,
            "t1_iters": t1_iters,
            "t1_m": t1_m,
            "use_rb_exit": use_rb_exit,
            "rb_exit_rate": (acc.rb_exits as f64 / roll_den * 10000.0).round() / 10000.0,
            "turn_cap_rate": (acc.turn_cap_cutoffs as f64 / roll_den * 10000.0).round() / 10000.0,
            "avg_rollout_turns": (acc.sum_rollout_turns as f64 / roll_den * 10.0).round() / 10.0,
            "games_per_s": (acc.finished_games as f64 / wall_s.max(1e-6) * 100.0).round() / 100.0,
            "pairs_per_s": (acc.pairs.len() as f64 / wall_s.max(1e-6) * 100.0).round() / 100.0,
            "out": out_path.display().to_string(),
            "wall_s": (wall_s * 100.0).round() / 100.0,
        })
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dex = load_dex();
    match args.first().map(String::as_str) {
        Some("duel") => run_duel_bench(&dex, &args[1..]),
        Some("blind-duel") => run_blind_duel(&dex, &args[1..]),
        Some("collect-stage3") => run_collect_stage3(&dex, &args[1..]),
        Some("collect-az") => run_collect_az(&dex, &args[1..]),
        Some("collect-cf") => run_collect_cf(&dex, &args[1..]),
        Some("collect-cf-v3") => run_collect_cf_v3(&dex, &args[1..]),
        Some("cf-probe") => run_cf_probe(&dex, &args[1..]),
        Some("cfv-bench") => run_cfv_bench(&dex, &args[1..]),
        Some("preview-diag") => run_preview_diag(&dex, &args[1..]),
        Some("preview-duel") => run_preview_duel(&dex, &args[1..]),
        Some("preview-exploit") => run_preview_exploit(&dex, &args[1..]),
        Some("preview-ceiling") => run_preview_ceiling(&dex, &args[1..]),
        Some("scale") => run_scale_suite(&dex, &args[1..]),
        Some("v2bench") => run_v2_bench(&dex, &args[1..]),
        Some("rolloutbench") => run_rollout_bench(&dex, &args[1..]),
        Some("autopsy") => run_autopsy(&dex, &args[1..]),
        Some("calib") => run_calib(&dex, &args[1..]),
        _ => run_scenarios(&dex),
    }
}
