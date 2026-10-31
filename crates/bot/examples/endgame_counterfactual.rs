//! Counterfactual verification of `EndgameRules` (`fast_win`, `safe_win`,
//! `comeback_loss`, and `early_stop`) under production search settings
//! (`blind` profile: `iterations = 27000`, `c = 0.4`).

use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    belief::Belief,
    blind::BlindSearch,
    eval::EvalWeights,
    mcts::Playout,
    observe::Observer,
    preview::{load_meta_pool, MetaPool},
    rng::SplitMix64,
    smmcts::{EndgameDecisionInfo, EndgameRegime, EndgameRules, RmConfig, SelRule},
};
use nc2000_engine::{
    battle::{Outcome, SearchChoice},
    dex::Dex,
    state::{Battle, RequestState},
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[derive(Clone)]
struct DecisionSnapshot {
    game_id: usize,
    turn: u16,
    side: usize,
    battle: Battle,
    obs: [Observer; 2],
    search_seed: u64,
}

#[derive(Clone, Debug, Default)]
struct RegimeSummary {
    states_total: usize,
    early_stop_triggered: usize,
    early_stop_saved_iters: u64,
    early_stop_exact_match: usize,
    early_stop_opt_score_full: f64,
    early_stop_opt_score_es: f64,
    forks: usize,
    opt_score_legacy_sum: f64,
    opt_score_endgame_sum: f64,
    opt_turns_legacy_sum: f64,
    opt_turns_endgame_sum: f64,
    eps_score_legacy_sum: f64,
    eps_score_endgame_sum: f64,
    fork_wins: usize,
    fork_losses: usize,
    fork_ties: usize,
}

fn production_cfg(iterations: u32, endgame_rules: EndgameRules) -> RmConfig {
    RmConfig {
        iterations,
        rule: SelRule::Ucb,
        c: 0.4,
        hp_buckets: 12,
        playout: Playout::Heavy {
            eps: 0.2,
            turns: 8,
            weights: EvalWeights::default(),
        },
        endgame_rules,
        ..RmConfig::default()
    }
}

fn collect_snapshots(
    dex: &Dex,
    pool: &MetaPool,
    games: usize,
    max_snapshots: usize,
    base_seed: u64,
) -> Vec<DecisionSnapshot> {
    let mut snapshots = Vec::new();
    let mut fast_count = 0usize;
    let mut safe_count = 0usize;
    let mut loss_count = 0usize;
    let mut norm_count = 0usize;
    let per_bucket_cap = (max_snapshots / 3).max(12);

    let scout_cfg = production_cfg(160, EndgameRules::all_decision());
    let mut sched_rng = SplitMix64::new(base_seed);
    for g in 0..games {
        if snapshots.len() >= max_snapshots {
            break;
        }
        let t1 = sched_rng.below(pool.teams.len());
        let t2 = sched_rng.below(pool.teams.len());
        let bseed = sched_rng.battle_seed();
        let mut battle =
            Battle::from_fixture(dex, &bseed, &pool.teams[t1].sets, &pool.teams[t2].sets).unwrap();
        battle.set_log_enabled(true);

        let mut obs = [Observer::new(&battle, 0), Observer::new(&battle, 1)];
        let mut bel = [
            Belief::new(dex, pool, &obs[0]),
            Belief::new(dex, pool, &obs[1]),
        ];
        let mut rng = SplitMix64::new(
            base_seed.wrapping_add((g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let mut took_bucket = [[false; 4]; 2];

        let mut step_guard = 0u32;
        while battle.outcome().is_none() && step_guard < 300 {
            step_guard += 1;
            for s in 0..2 {
                obs[s].observe(&battle, dex);
                bel[s].sync(dex, &obs[s]);
            }
            let c0 = battle.legal_choices(dex, 0);
            let c1 = battle.legal_choices(dex, 1);
            let is_preview = matches!(battle.request_state, RequestState::TeamPreview);
            let mut picks = [None, None];
            for s in 0..2 {
                let choices = if s == 0 { &c0 } else { &c1 };
                if choices.is_empty() {
                    continue;
                }
                if choices.len() == 1 {
                    picks[s] = Some(choices[0]);
                    continue;
                }
                let sseed = rng.next();
                let mut search =
                    BlindSearch::new(&battle, dex, scout_cfg.clone(), s, sseed);
                if is_preview {
                    search.prune_preview_aliases(dex);
                } else {
                    search.prune_rest_aliases(dex);
                }
                search.step(dex, &bel[s], &obs[s], scout_cfg.iterations);
                if !is_preview {
                    let elig_cnt = (0..choices.len())
                        .filter(|&i| !search.dominated()[i])
                        .count();
                    if elig_cnt >= 2 {
                        if let Some(info) = search.endgame_analysis() {
                            let mu = info.mu_star;
                            let want = if mu >= 0.985 && !took_bucket[s][0] && fast_count < per_bucket_cap {
                                took_bucket[s][0] = true;
                                fast_count += 1;
                                true
                            } else if (0.78..0.975).contains(&mu) && !took_bucket[s][1] && safe_count < per_bucket_cap {
                                took_bucket[s][1] = true;
                                safe_count += 1;
                                true
                            } else if mu <= 0.10 && !took_bucket[s][2] && loss_count < per_bucket_cap {
                                took_bucket[s][2] = true;
                                loss_count += 1;
                                true
                            } else if (0.35..0.65).contains(&mu) && !took_bucket[s][3] && norm_count < per_bucket_cap / 3 {
                                took_bucket[s][3] = true;
                                norm_count += 1;
                                true
                            } else {
                                false
                            };
                            if want && snapshots.len() < max_snapshots {
                                snapshots.push(DecisionSnapshot {
                                    game_id: g,
                                    turn: battle.turn,
                                    side: s,
                                    battle: battle.clone(),
                                    obs: obs.clone(),
                                    search_seed: rng.next(),
                                });
                            }
                        }
                    }
                }
                picks[s] = search.best();
            }
            if picks == [None, None] {
                break;
            }
            let _ = battle.apply_choices(dex, picks);
        }
    }
    snapshots
}

fn play_continuation(
    dex: &Dex,
    pool: &MetaPool,
    snap: &DecisionSnapshot,
    forced_my_choice: SearchChoice,
    forced_opp_choice: Option<SearchChoice>,
    cont_iters: u32,
    opp_eps: f64,
    rollout_seed: u64,
) -> (f64, f64) {
    let mut b = snap.battle.clone();
    b.reseed(rollout_seed);
    b.set_log_enabled(true);
    let side = snap.side;
    let opp = 1 - side;

    let mut first = [None, None];
    first[side] = Some(forced_my_choice);
    first[opp] = forced_opp_choice;
    let root_turn = b.turn;
    if b.apply_choices(dex, first).is_err() {
        return (0.5, 1.0);
    }

    let mut obs = snap.obs.clone();
    let mut bel = [
        Belief::new(dex, pool, &obs[0]),
        Belief::new(dex, pool, &obs[1]),
    ];
    let cfg = production_cfg(cont_iters, EndgameRules::none());
    let mut rng = SplitMix64::new(rollout_seed ^ 0xA5A5_5A5A_F0F0_0F0F);
    let mut steps = 0u32;
    while b.outcome().is_none() && steps < 160 {
        steps += 1;
        for s in 0..2 {
            obs[s].observe(&b, dex);
            bel[s].sync(dex, &obs[s]);
        }
        let mut joint = [None, None];
        for s in 0..2 {
            let legal = b.legal_choices(dex, s);
            if legal.is_empty() {
                continue;
            }
            if legal.len() == 1 {
                joint[s] = Some(legal[0]);
                continue;
            }
            let sseed = rng.next();
            let mut search = BlindSearch::new(&b, dex, cfg.clone(), s, sseed);
            search.prune_rest_aliases(dex);
            if s == opp && opp_eps > 0.0 && rng.next_f64() < opp_eps {
                let dom = search.dominated();
                let non_dom: Vec<SearchChoice> = (0..legal.len())
                    .filter(|&i| !dom[i])
                    .map(|i| legal[i])
                    .collect();
                let pool_choices = if non_dom.is_empty() { &legal } else { &non_dom };
                joint[s] = Some(pool_choices[rng.below(pool_choices.len())]);
                continue;
            }
            search.step(dex, &bel[s], &obs[s], cfg.iterations);
            joint[s] = search.best();
        }
        if joint == [None, None] {
            break;
        }
        if b.apply_choices(dex, joint).is_err() {
            break;
        }
    }
    let score = match b.outcome() {
        Some(Outcome::P1Win) => {
            if side == 0 {
                1.0
            } else {
                0.0
            }
        }
        Some(Outcome::P2Win) => {
            if side == 1 {
                1.0
            } else {
                0.0
            }
        }
        Some(Outcome::Tie) | None => 0.5,
    };
    let base_offset = u16::from(matches!(snap.battle.request_state, RequestState::Switch));
    let turns = b
        .turn
        .saturating_sub(root_turn.saturating_add(base_offset))
        as f64
        + 1.0;
    (score, turns)
}

fn evaluate_snapshot(
    dex: &Dex,
    pool: &MetaPool,
    snap: &DecisionSnapshot,
    prod_iters: u32,
    cont_iters: u32,
    rollouts: usize,
) -> (
    EndgameRegime,
    EndgameDecisionInfo,
    bool,
    u32,
    bool,
    Option<(f64, f64)>,
    Option<(f64, f64, f64, f64, f64, f64)>,
) {
    let mut b_tmp = snap.battle.clone();
    let choices = b_tmp.legal_choices(dex, snap.side);
    let bel_my = Belief::new(dex, pool, &snap.obs[snap.side]);
    let cfg = production_cfg(prod_iters, EndgameRules::all_with_early_stop());
    let mut search = BlindSearch::new(&snap.battle, dex, cfg, snap.side, snap.search_seed);
    search.prune_rest_aliases(dex);

    let mut early_stopped = false;
    let mut early_stop_iter = prod_iters;
    let mut early_stop_choice: Option<SearchChoice> = None;
    let chunk = 32u32;
    let mut done = 0u32;
    while done < prod_iters {
        let step_n = (prod_iters - done).min(chunk);
        search.step(dex, &bel_my, &snap.obs[snap.side], step_n);
        done += step_n;
        if !early_stopped && search.should_early_stop() {
            early_stopped = true;
            early_stop_iter = done;
            early_stop_choice = search.best();
        }
    }

    let info = search.endgame_analysis().expect("non-empty root");
    let full_choice = choices[info.chosen_idx];
    let legacy_choice = choices[info.legacy_idx];
    let es_matches = early_stop_choice.map_or(true, |c| c == full_choice);

    let opp = 1 - snap.side;
    let opp_choices = b_tmp.legal_choices(dex, opp);
    let bel_opp = Belief::new(dex, pool, &snap.obs[opp]);
    let opp_cfg = production_cfg(cont_iters.max(400), EndgameRules::none());

    let es_eval = if early_stopped && !es_matches {
        let es_c = early_stop_choice.unwrap();
        let mut s_full = 0.0;
        let mut s_es = 0.0;
        for r in 0..rollouts {
            let rseed = snap.search_seed ^ ((r as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let opp_pick = if opp_choices.is_empty() {
                None
            } else if opp_choices.len() == 1 {
                Some(opp_choices[0])
            } else {
                let mut osearch = BlindSearch::new(
                    &snap.battle,
                    dex,
                    opp_cfg.clone(),
                    opp,
                    rseed ^ 0x1234_5678,
                );
                osearch.prune_rest_aliases(dex);
                osearch.step(dex, &bel_opp, &snap.obs[opp], opp_cfg.iterations);
                osearch.best()
            };
            let (sc_f, _) =
                play_continuation(dex, pool, snap, full_choice, opp_pick, cont_iters, 0.0, rseed);
            let (sc_e, _) =
                play_continuation(dex, pool, snap, es_c, opp_pick, cont_iters, 0.0, rseed);
            s_full += sc_f;
            s_es += sc_e;
        }
        Some((s_full / rollouts as f64, s_es / rollouts as f64))
    } else {
        None
    };

    let fork_eval = if info.chosen_idx != info.legacy_idx {
        let mut opt_leg = 0.0;
        let mut opt_end = 0.0;
        let mut t_leg = 0.0;
        let mut t_end = 0.0;
        let mut eps_leg = 0.0;
        let mut eps_end = 0.0;

        for r in 0..rollouts {
            let rseed = snap.search_seed ^ ((r as u64 + 1).wrapping_mul(0x94D0_49BB_1331_11EB));
            let opp_opt_pick = if opp_choices.is_empty() {
                None
            } else if opp_choices.len() == 1 {
                Some(opp_choices[0])
            } else {
                let mut osearch = BlindSearch::new(
                    &snap.battle,
                    dex,
                    opp_cfg.clone(),
                    opp,
                    rseed ^ 0x5555_AAAA,
                );
                osearch.prune_rest_aliases(dex);
                osearch.step(dex, &bel_opp, &snap.obs[opp], opp_cfg.iterations);
                osearch.best()
            };
            let (s_l, tr_l) = play_continuation(
                dex,
                pool,
                snap,
                legacy_choice,
                opp_opt_pick,
                cont_iters,
                0.0,
                rseed,
            );
            let (s_e, tr_e) = play_continuation(
                dex,
                pool,
                snap,
                full_choice,
                opp_opt_pick,
                cont_iters,
                0.0,
                rseed,
            );
            opt_leg += s_l;
            opt_end += s_e;
            t_leg += tr_l;
            t_end += tr_e;

            let opp_eps_pick = if opp_choices.is_empty() {
                None
            } else if opp_choices.len() == 1 {
                Some(opp_choices[0])
            } else {
                let mut r_rng = SplitMix64::new(rseed ^ 0xCAFE_BABE);
                if r_rng.next_f64() < 0.35 {
                    let probe = BlindSearch::new(&snap.battle, dex, opp_cfg.clone(), opp, rseed);
                    let dom = probe.dominated();
                    let non_dom: Vec<SearchChoice> = (0..opp_choices.len())
                        .filter(|&i| !dom[i])
                        .map(|i| opp_choices[i])
                        .collect();
                    let list = if non_dom.is_empty() {
                        &opp_choices
                    } else {
                        &non_dom
                    };
                    Some(list[r_rng.below(list.len())])
                } else {
                    opp_opt_pick
                }
            };
            let (es_l, _) = play_continuation(
                dex,
                pool,
                snap,
                legacy_choice,
                opp_eps_pick,
                cont_iters,
                0.25,
                rseed,
            );
            let (es_e, _) = play_continuation(
                dex,
                pool,
                snap,
                full_choice,
                opp_eps_pick,
                cont_iters,
                0.25,
                rseed,
            );
            eps_leg += es_l;
            eps_end += es_e;
        }
        let n = rollouts as f64;
        Some((
            opt_leg / n,
            opt_end / n,
            t_leg / n,
            t_end / n,
            eps_leg / n,
            eps_end / n,
        ))
    } else {
        None
    };

    (
        info.regime,
        info,
        early_stopped,
        early_stop_iter,
        es_matches,
        es_eval,
        fork_eval,
    )
}

fn regime_idx(r: EndgameRegime) -> usize {
    match r {
        EndgameRegime::FastWin => 0,
        EndgameRegime::SafeWin => 1,
        EndgameRegime::ComebackLoss => 2,
        EndgameRegime::Normal => 3,
    }
}

fn regime_name(idx: usize) -> &'static str {
    match idx {
        0 => "FastWin",
        1 => "SafeWin",
        2 => "ComebackLoss",
        _ => "Normal",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |key: &str, default: &str| {
        args.iter()
            .position(|s| s == key)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.into())
    };
    let prod_iters: u32 = arg("--iters", "27000").parse().unwrap();
    let cont_iters: u32 = arg("--cont-iters", "400").parse().unwrap();
    let rollouts: usize = arg("--rollouts", "32").parse().unwrap();
    let max_snapshots: usize = arg("--snapshots", "120").parse().unwrap();
    let games: usize = arg("--games", "120").parse().unwrap();
    let threads: usize = arg("--threads", "8").parse().unwrap();
    let seed: u64 = arg("--seed", "20261031").parse().unwrap();

    let dex = Arc::new(load_dex());
    let root = repo_root();
    let pool = Arc::new(load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json")));

    eprintln!(
        "[counterfactual] scouting up to {} snapshots from {} games (seed={})...",
        max_snapshots, games, seed
    );
    let t0 = std::time::Instant::now();
    let snapshots = Arc::new(collect_snapshots(&dex, &pool, games, max_snapshots, seed));
    eprintln!(
        "[counterfactual] collected {} snapshots in {:.1}s; running production search (iters={}, c=0.4, rollouts={}, cont_iters={}, threads={})...",
        snapshots.len(),
        t0.elapsed().as_secs_f64(),
        prod_iters,
        rollouts,
        cont_iters,
        threads
    );

    let next_idx = Arc::new(AtomicUsize::new(0));
    let summaries = Arc::new(Mutex::new([
        RegimeSummary::default(),
        RegimeSummary::default(),
        RegimeSummary::default(),
        RegimeSummary::default(),
    ]));

    std::thread::scope(|scope| {
        for _ in 0..threads {
            let dex = Arc::clone(&dex);
            let pool = Arc::clone(&pool);
            let snapshots = Arc::clone(&snapshots);
            let next_idx = Arc::clone(&next_idx);
            let summaries = Arc::clone(&summaries);
            scope.spawn(move || {
                loop {
                    let i = next_idx.fetch_add(1, Ordering::Relaxed);
                    if i >= snapshots.len() {
                        break;
                    }
                    let snap = &snapshots[i];
                    let (regime, info, es_trig, es_iter, es_match, es_eval, fork_eval) =
                        evaluate_snapshot(&dex, &pool, snap, prod_iters, cont_iters, rollouts);
                    let r_idx = regime_idx(regime);
                    let mut guard = summaries.lock().unwrap();
                    let s = &mut guard[r_idx];
                    s.states_total += 1;
                    if es_trig {
                        s.early_stop_triggered += 1;
                        s.early_stop_saved_iters += u64::from(prod_iters.saturating_sub(es_iter));
                        if es_match {
                            s.early_stop_exact_match += 1;
                        } else if let Some((sf, se)) = es_eval {
                            s.early_stop_opt_score_full += sf;
                            s.early_stop_opt_score_es += se;
                        }
                    }
                    if let Some((opt_l, opt_e, t_l, t_e, eps_l, eps_e)) = fork_eval {
                        s.forks += 1;
                        s.opt_score_legacy_sum += opt_l;
                        s.opt_score_endgame_sum += opt_e;
                        s.opt_turns_legacy_sum += t_l;
                        s.opt_turns_endgame_sum += t_e;
                        s.eps_score_legacy_sum += eps_l;
                        s.eps_score_endgame_sum += eps_e;
                        if opt_e > opt_l + 1e-6 {
                            s.fork_wins += 1;
                        } else if opt_e + 1e-6 < opt_l {
                            s.fork_losses += 1;
                        } else {
                            s.fork_ties += 1;
                        }
                        let leg_m = &info.actions[info.legacy_idx];
                        let ch_m = &info.actions[info.chosen_idx];
                        eprintln!(
                            "  [FORK {:>12}] game={} turn={} side={} | leg={:?} (n={}, mu={:.4}, win={:.3}, loss={:.3}, wt={:.2}, trap={:.3}, m_ev={:.3}) -> ch={:?} (n={}, mu={:.4}, win={:.3}, loss={:.3}, wt={:.2}, trap={:.3}, m_ev={:.3}) | CF opt: {:.3}->{:.3} (turns {:.2}->{:.2}) | CF eps: {:.3}->{:.3}",
                            regime_name(r_idx),
                            snap.game_id,
                            snap.turn,
                            snap.side,
                            leg_m.action,
                            leg_m.visits,
                            leg_m.mean,
                            leg_m.win_rate,
                            leg_m.loss_rate,
                            leg_m.win_turns,
                            leg_m.trap_risk,
                            leg_m.mistake_ev,
                            ch_m.action,
                            ch_m.visits,
                            ch_m.mean,
                            ch_m.win_rate,
                            ch_m.loss_rate,
                            ch_m.win_turns,
                            ch_m.trap_risk,
                            ch_m.mistake_ev,
                            opt_l,
                            opt_e,
                            t_l,
                            t_e,
                            eps_l,
                            eps_e,
                        );
                    }
                }
            });
        }
    });

    let guard = summaries.lock().unwrap();
    println!(
        "\n=== Production Counterfactual Summary (blind:{}:0.4) ===",
        prod_iters
    );
    for idx in 0..4 {
        let s = &guard[idx];
        let name = regime_name(idx);
        if s.forks > 0 {
            let f = s.forks as f64;
            println!(
                "{:<14}: states={:>3}, forks={:>2} ({}W/{}L/{}T) | CF Opt Score: {:.4} -> {:.4} (Δ={:+.4}) | CF Turns: {:.2} -> {:.2} (Δ={:+.2}) | CF Mistake-Opp Score: {:.4} -> {:.4} (Δ={:+.4})",
                name,
                s.states_total,
                s.forks,
                s.fork_wins,
                s.fork_losses,
                s.fork_ties,
                s.opt_score_legacy_sum / f,
                s.opt_score_endgame_sum / f,
                (s.opt_score_endgame_sum - s.opt_score_legacy_sum) / f,
                s.opt_turns_legacy_sum / f,
                s.opt_turns_endgame_sum / f,
                (s.opt_turns_endgame_sum - s.opt_turns_legacy_sum) / f,
                s.eps_score_legacy_sum / f,
                s.eps_score_endgame_sum / f,
                (s.eps_score_endgame_sum - s.eps_score_legacy_sum) / f,
            );
        } else {
            println!(
                "{:<14}: states={:>3}, forks= 0 (100% agreement with legacy argmax-visits -> Δ = 0.0000)",
                name, s.states_total
            );
        }
        if s.early_stop_triggered > 0 {
            let avg_saved = s.early_stop_saved_iters as f64 / s.early_stop_triggered as f64;
            let diff_cnt = s.early_stop_triggered - s.early_stop_exact_match;
            let es_cf = if diff_cnt > 0 {
                format!(
                    ", diff_cf_score={:.4}->{:.4}",
                    s.early_stop_opt_score_full / diff_cnt as f64,
                    s.early_stop_opt_score_es / diff_cnt as f64
                )
            } else {
                String::new()
            };
            println!(
                "  └─ early_stop: triggered={}/{} (saved {:.0} iters = {:.1}% budget), exact_match={}/{}{}",
                s.early_stop_triggered,
                s.states_total,
                avg_saved,
                100.0 * avg_saved / prod_iters as f64,
                s.early_stop_exact_match,
                s.early_stop_triggered,
                es_cf,
            );
        }
    }
}
