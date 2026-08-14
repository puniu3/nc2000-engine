//! M18a PUCT prior slot: contract tests.
//!
//! The slot is a research arm. Its first obligation is to leave the shipped
//! agent alone — `PriorKind::Off` must reproduce decoupled UCB1 exactly, not
//! approximately — and its second is that the prior it computes is a real
//! distribution over the node's legal actions.

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::mcts::action_scores;
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{Agent, PriorKind, RmAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn team_pool() -> Vec<Vec<PokemonSet>> {
    let root = repo_root().join("fixtures/corpus-v1");
    let mut teams = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&root.join(corpus)) {
            let fx = Fixture::load(&path).unwrap();
            teams.push(fx.p1team);
            teams.push(fx.p2team);
        }
    }
    teams
}

fn skuct(iters: u32) -> RmConfig {
    RmConfig { iterations: iters, rule: SelRule::Ucb, ..Default::default() }
}

/// Play one battle with `cfg` on both sides; return every choice made, in
/// order, plus the final state key. Two configurations that select
/// identically produce identical vectors.
fn trace(dex: &Dex, teams: &[Vec<PokemonSet>], t1: usize, t2: usize, seed: &str, cfg: RmConfig) -> (Vec<String>, u64) {
    let mut battle = Battle::from_fixture(dex, seed, &teams[t1], &teams[t2]).unwrap();
    battle.set_log_enabled(false);
    let mut agents: Vec<RmAgent> =
        (0..2).map(|s| RmAgent::new(cfg.clone(), 11 + s as u64 * 13)).collect();
    let mut out = Vec::new();
    while battle.outcome().is_none() && battle.turn <= 200 {
        let mut picks = [None, None];
        for s in 0..2 {
            let cs = battle.legal_choices(dex, s);
            if cs.is_empty() {
                continue;
            }
            let c = agents[s].choose(&battle, dex, s, &cs);
            out.push(format!("{s}:{c:?}"));
            picks[s] = Some(c);
        }
        if picks == [None, None] {
            break;
        }
        battle.apply_choices(dex, picks).unwrap();
    }
    (out, battle.state_key())
}

/// The shipped default must not have the probe armed.
#[test]
fn default_config_leaves_the_slot_off() {
    let d = RmConfig::default();
    assert_eq!(d.prior, PriorKind::Off);
    assert_eq!(d.puct, 0.0);
    assert_eq!(d.prior_status_bonus, 0.0);
}

/// `PriorKind::Off` is the shipped selection rule, move for move — even when
/// the (unread) prior fields are set to values that would change play if the
/// slot ever leaked into the default path. This is what makes the research
/// arm safe to keep in-tree.
#[test]
fn prior_off_is_move_for_move_identical() {
    let dex = load_dex();
    let teams = team_pool();
    let mut rng = SplitMix64::new(3);
    for _ in 0..4 {
        let t1 = rng.below(teams.len());
        let t2 = rng.below(teams.len());
        let seed = rng.battle_seed();
        let shipped = trace(&dex, &teams, t1, t2, &seed, skuct(200));
        let armed_off = trace(
            &dex,
            &teams,
            t1,
            t2,
            &seed,
            RmConfig { puct: 4.0, prior_status_bonus: 0.9, ..skuct(200) },
        );
        assert_eq!(shipped.0, armed_off.0, "PriorKind::Off diverged from the shipped rule");
        assert_eq!(shipped.1, armed_off.1, "final state diverged");
    }
}

/// Step a battle a few turns past team preview into real in-battle decisions.
fn warmed(dex: &Dex, teams: &[Vec<PokemonSet>], seed_n: u64, turns: usize) -> Battle {
    let mut rng = SplitMix64::new(seed_n);
    let t1 = rng.below(teams.len());
    let t2 = rng.below(teams.len());
    let seed = rng.battle_seed();
    let mut battle = Battle::from_fixture(dex, &seed, &teams[t1], &teams[t2]).unwrap();
    battle.set_log_enabled(false);
    let mut agent = RmAgent::new(skuct(50), 1);
    for _ in 0..turns {
        if battle.outcome().is_some() {
            break;
        }
        let mut picks = [None, None];
        for s in 0..2 {
            let cs = battle.legal_choices(dex, s);
            if !cs.is_empty() {
                picks[s] = Some(agent.choose(&battle, dex, s, &cs));
            }
        }
        battle.apply_choices(dex, picks).unwrap();
    }
    battle
}

/// One finite score per legal action, or `None` — never a ragged vector.
#[test]
fn prior_scores_align_with_the_legal_set() {
    let dex = load_dex();
    let teams = team_pool();
    let mut battle = warmed(&dex, &teams, 9, 3);
    let mut checked = 0;
    for s in 0..2 {
        let cs = battle.legal_choices(&dex, s);
        if cs.len() < 2 {
            continue;
        }
        for bonus in [0.0, 0.5] {
            if let Some(scores) = action_scores(&battle, &dex, s, &cs, bonus) {
                assert_eq!(scores.len(), cs.len(), "one score per legal action");
                assert!(scores.iter().all(|v| v.is_finite()));
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no scorable decision reached");
}

/// The status bonus must move mass toward Status-category moves — it is the
/// cluster-2 arm, and a bug that left it inert would read as a null result.
#[test]
fn status_bonus_raises_status_moves() {
    use nc2000_engine::battle::SearchChoice;
    use nc2000_engine::dex::Category;
    let dex = load_dex();
    let teams = team_pool();
    let mut found = false;
    for seed_n in 1..40u64 {
        let mut battle = warmed(&dex, &teams, seed_n, 3);
        for s in 0..2 {
            let cs = battle.legal_choices(&dex, s);
            let has_status = cs.iter().any(|c| {
                matches!(c, SearchChoice::Move(id)
                    if dex.move_static(*id).category == Category::Status)
            });
            if cs.len() < 2 || !has_status {
                continue;
            }
            let (Some(lo), Some(hi)) = (
                action_scores(&battle, &dex, s, &cs, 0.0),
                action_scores(&battle, &dex, s, &cs, 0.5),
            ) else {
                continue;
            };
            for (i, &c) in cs.iter().enumerate() {
                if let SearchChoice::Move(id) = c {
                    if dex.move_static(id).category == Category::Status {
                        assert!(
                            hi[i] >= lo[i],
                            "status bonus lowered a status move's score"
                        );
                        if hi[i] > lo[i] {
                            found = true;
                        }
                    }
                }
            }
        }
        if found {
            break;
        }
    }
    assert!(found, "status bonus never raised a status move");
}

/// The adversarial arm must actually be adversarial: it is the probe that
/// bounds the slot's leverage, so a bug making it behave like the informed
/// prior would silently invalidate the whole measurement.
#[test]
fn inverted_prior_disagrees_with_the_informed_one() {
    let dex = load_dex();
    let teams = team_pool();
    let armed = |k: PriorKind| RmConfig { prior: k, puct: 2.0, ..skuct(300) };
    let am = |p: &[f64]| (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
    let mut disagreements = 0;
    for seed_n in 1..12u64 {
        let mut battle = warmed(&dex, &teams, seed_n, 4);
        for s in 0..2 {
            let cs = battle.legal_choices(&dex, s);
            if cs.len() < 3 || action_scores(&battle, &dex, s, &cs, 0.0).is_none() {
                continue;
            }
            let mut g = RmAgent::new(armed(PriorKind::Greedy), 7);
            let mut i = RmAgent::new(armed(PriorKind::Inverted), 7);
            let pg = g.root_policy(&battle, &dex, s, &cs);
            let pi = i.root_policy(&battle, &dex, s, &cs);
            if am(&pg) != am(&pi) {
                disagreements += 1;
            }
        }
    }
    assert!(disagreements > 0, "inverted prior never changed the top-1 — probe is inert");
}
