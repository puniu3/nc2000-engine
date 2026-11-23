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
    // Parses e.g. `golf-0r:1000`, `golf-4r:2500`, `bs-8r:1000`, `eval-0r:1000`
    let (head, it_s) = rest.split_once(':')?;
    let iters: u32 = it_s.parse().ok()?;
    let (leaf_s, r_s) = head.split_once('-')?;
    let leaf = match leaf_s {
        "golf" | "v2" => TempoLeafMode::FastGolf,
        "bs" | "blindspot" => TempoLeafMode::EvalBlindspot,
        "eval" => TempoLeafMode::EvalStatic,
        _ => return None,
    };
    let turns: u16 = r_s.strip_suffix('r')?.parse().ok()?;
    Some((leaf, turns, iters))
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
    let root = repo_root();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let teams: Vec<Vec<PokemonSet>> = pool.teams.iter().map(|t| t.sets.clone()).collect();

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
                    let mut game_rows: Vec<(usize, f64, f64, f64, f64, f64, String)> = Vec::new();
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
                        game_rows.push((bucket, fg.g_turns[0], fg.g_turns[1], fg.win_prob, ev, bs, desc));

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
                    for (bucket, g0, g1, fg_p, ev_p, bs_p, desc) in game_rows {
                        local.push(Sample { game: g, bucket, y, g0, g1, fg_p, ev_p, bs_p, desc });
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

    let bucket_names = ["1v1", "2v1", "2v2", "3v2_3v1", "3v3"];
    let mut by_bucket = serde_json::Map::new();
    for b_idx in 0..5 {
        let sub: Vec<&Sample> = all_samples.iter().filter(|s| s.bucket == b_idx).collect();
        let fg_pts: Vec<(f64, f64)> = sub.iter().map(|s| (s.fg_p, s.y)).collect();
        let ev_pts: Vec<(f64, f64)> = sub.iter().map(|s| (s.ev_p, s.y)).collect();
        let bs_pts: Vec<(f64, f64)> = sub.iter().map(|s| (s.bs_p, s.y)).collect();
        let (opt_v, opt_b) = best_var_brier(&sub);
        by_bucket.insert(
            bucket_names[b_idx].to_string(),
            json!({
                "fg": metrics(&fg_pts),
                "ev": metrics(&ev_pts),
                "bs": metrics(&bs_pts),
                "fg_opt_var": opt_v,
                "fg_opt_brier": opt_b,
            }),
        );
    }

    let fg_all: Vec<(f64, f64)> = all_samples.iter().map(|s| (s.fg_p, s.y)).collect();
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
    searched_battle_ns: Vec<u64>,
}

impl TimedAgentWrap {
    fn new(inner: Box<dyn Agent>) -> Self {
        Self { inner, searched_battle_ns: Vec::new() }
    }
}

impl Agent for TimedAgentWrap {
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
        let record = choices.len() > 1 && !matches!(choices[0], SearchChoice::Team(_));
        let t0 = if record { Some(Instant::now()) } else { None };
        let pick = self.inner.choose(battle, dex, side, choices);
        if let Some(t) = t0 {
            self.searched_battle_ns.push(t.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        }
        pick
    }
}

fn one_hot_preview_policy(triple: [u8; 3]) -> Vec<f64> {
    let acts = preview_actions();
    let idx = action_index(&acts, triple).expect("canonical preview action");
    let mut pol = vec![0.0; acts.len()];
    pol[idx] = 1.0;
    pol
}

fn build_blind_arm(
    spec: &str,
    pool: Arc<MetaPool>,
    seed: u64,
    preview_triple: [u8; 3],
) -> Box<dyn Agent> {
    if let Some(rest) = spec.strip_prefix("blind-shipped:") {
        let iters: u32 = rest.parse().unwrap_or_else(|_| panic!("bad spec: {spec}"));
        let cfg = RmConfig { iterations: iters, rule: SelRule::Ucb, c: 0.4, ..Default::default() };
        Box::new(
            BlindAgent::new(cfg, pool, None, seed)
                .with_own_preview_policy(one_hot_preview_policy(preview_triple)),
        )
    } else if let Some(rest) = spec.strip_prefix("bcmcts-") {
        let (leaf, rollout_turns, iterations) =
            parse_leaf_rollout(rest).unwrap_or_else(|| panic!("bad bcmcts spec: {spec}"));
        Box::new(BlindCmctsAgent::new(
            BlindCmctsConfig {
                iterations,
                c: 0.4,
                max_depth: 64,
                leaf,
                rollout_turns,
                max_worlds: 24,
                preview_iterations: 27_000,
                own_preview_triple: Some(preview_triple),
            },
            pool,
            seed,
        ))
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
    let total_pairs_needed = pair_start + pairs;

    let root = repo_root();
    let cat_txt = std::fs::read_to_string(root.join("data/team-pool-v2/team-pool.json"))
        .expect("read team-pool-v2");
    let catalog: WeightedCatalog = serde_json::from_str(&cat_txt).expect("parse team-pool-v2");
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dex = load_dex();
    match args.first().map(String::as_str) {
        Some("duel") => run_duel_bench(&dex, &args[1..]),
        Some("blind-duel") => run_blind_duel(&dex, &args[1..]),
        Some("scale") => run_scale_suite(&dex, &args[1..]),
        Some("v2bench") => run_v2_bench(&dex, &args[1..]),
        Some("rolloutbench") => run_rollout_bench(&dex, &args[1..]),
        Some("autopsy") => run_autopsy(&dex, &args[1..]),
        Some("calib") => run_calib(&dex, &args[1..]),
        _ => run_scenarios(&dex),
    }
}
