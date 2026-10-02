//! `bot::value` encoder and network invariants the training loop relies on.

use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::value::*;
use nc2000_bot::SplitMix64;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn teams() -> Vec<Vec<PokemonSet>> {
    let root = conformance::fixture::repo_root();
    ["data/team-pool-v2/team-pool.json", "data/belief-pool-v3/belief-pool.json"]
        .iter()
        .flat_map(|p| load_meta_pool(&root.join(p)).teams.into_iter().map(|t| t.sets))
        .collect()
}

/// The record of the side-swapped battle: every per-side block exchanged.
fn swap_record(r: &[f32]) -> Vec<f32> {
    let mut out = r.to_vec();
    let blocks = [
        (OFF_IDS, MONS * IDS),
        (OFF_MON, MONS * F_MON),
        (OFF_MDYN, MONS * MOVES * F_MDYN),
        (OFF_VOL, F_VOL),
        (OFF_SIDE, F_SIDE),
        (OFF_PAIR, F_PAIR),
    ];
    for (off, n) in blocks {
        out[off..off + n].copy_from_slice(&r[off + n..off + 2 * n]);
        out[off + n..off + 2 * n].copy_from_slice(&r[off..off + n]);
    }
    out
}

fn random_net(codec: &ValueCodec, rng: &mut SplitMix64) -> ValueNet {
    let d = codec.describe();
    let mut w = |n: usize| (0..n).map(|_| (rng.next_f64() as f32 - 0.5) * 0.4).collect::<Vec<_>>();
    let emb = |rows: usize, dim: usize, w: &mut dyn FnMut(usize) -> Vec<f32>| Embedding {
        rows,
        dim,
        w: w(rows * dim),
    };
    let dense = |n_in: usize, n_out: usize, w: &mut dyn FnMut(usize) -> Vec<f32>| Dense {
        n_in,
        n_out,
        w: w(n_in * n_out),
        b: w(n_out),
        wt: Vec::new(),
    };
    let (ds, di, dm, hm, h1, h2, hh1, hh2) = (8, 4, 8, 12, 16, 12, 24, 12);
    let side = 2 * h2 + F_SIDE + F_VOL;
    ValueNet {
        schema: VALUE_SCHEMA.into(),
        species: emb(d.n_species + 1, ds, &mut w),
        items: emb(d.n_items + 1, di, &mut w),
        moves: emb(d.n_moves + 1, dm, &mut w),
        move_layer: dense(dm + F_MSTAT + F_MDYN, hm, &mut w),
        mon1: dense(ds + di + hm + F_MON, h1, &mut w),
        mon2: dense(h1, h2, &mut w),
        head1: dense(2 * side + F_PAIR + F_GLOBAL, hh1, &mut w),
        head2: dense(hh1, hh2, &mut w),
        head3: dense(hh2, 1, &mut w),
        parity: Vec::new(),
        meta: serde_json::Value::Null,
    }
}

/// Play uniformly random legal games and hand every decision state to `f`.
fn random_states(dex: &Dex, games: usize, seed: u64, mut f: impl FnMut(&Battle)) {
    let teams = teams();
    let mut rng = SplitMix64::new(seed);
    for _ in 0..games {
        let (a, b) = (rng.below(teams.len()), rng.below(teams.len()));
        let mut battle = Battle::from_fixture(dex, &rng.battle_seed(), &teams[a], &teams[b]).unwrap();
        battle.set_log_enabled(false);
        while battle.outcome().is_none() && battle.turn <= 200 {
            if battle.turn > 0 {
                f(&battle);
            }
            let mut picks = [None, None];
            for s in 0..2 {
                let cs = battle.legal_choices(dex, s);
                if !cs.is_empty() {
                    picks[s] = Some(cs[rng.below(cs.len())]);
                }
            }
            battle.apply_choices(dex, picks).unwrap();
        }
    }
}

#[test]
fn every_reached_volatile_and_side_condition_has_a_slot() {
    let dex = conformance::load_dex();
    let mut unknown = std::collections::BTreeSet::new();
    random_states(&dex, 2000, 7, |b| {
        for s in 0..2 {
            if let Some(id) = b.active_id(s) {
                for (cid, _) in &b.poke(id).volatiles {
                    let key = dex.conds_key(*cid);
                    if !VOLATILE_KEYS.contains(&key) {
                        unknown.insert(format!("volatile {key}"));
                    }
                }
            }
            for (cid, _) in &b.sides[s].side_conditions {
                let key = dex.conds_key(*cid);
                if !SIDE_CONDS.contains(&key) {
                    unknown.insert(format!("side condition {key}"));
                }
            }
        }
    });
    assert!(unknown.is_empty(), "unencoded conditions reached: {unknown:?}");
}

#[test]
fn swapping_the_teams_mirrors_the_record() {
    let dex = conformance::load_dex();
    let codec = ValueCodec::new(&dex);
    let teams = teams();
    let mut rng = SplitMix64::new(11);
    let mut checked = 0;
    for _ in 0..40 {
        let (a, b) = (rng.below(teams.len()), rng.below(teams.len()));
        let seed = rng.battle_seed();
        let mut x = Battle::from_fixture(&dex, &seed, &teams[a], &teams[b]).unwrap();
        let mut y = Battle::from_fixture(&dex, &seed, &teams[b], &teams[a]).unwrap();
        let px = [x.legal_choices(&dex, 0), x.legal_choices(&dex, 1)];
        let (i, j) = (rng.below(px[0].len()), rng.below(px[1].len()));
        x.apply_choices(&dex, [Some(px[0][i]), Some(px[1][j])]).unwrap();
        y.apply_choices(&dex, [Some(px[1][j]), Some(px[0][i])]).unwrap();
        let (mut rx, mut ry) = (Vec::new(), Vec::new());
        codec.encode(&x, &dex, &mut rx);
        codec.encode(&y, &dex, &mut ry);
        let sy = swap_record(&ry);
        for (k, (u, v)) in rx.iter().zip(&sy).enumerate() {
            assert!((u - v).abs() < 1e-6, "record column {k}: {u} vs {v}");
        }
        checked += 1;
    }
    assert_eq!(checked, 40);
}

#[test]
fn the_network_is_side_antisymmetric() {
    let dex = conformance::load_dex();
    let codec = ValueCodec::new(&dex);
    let mut rng = SplitMix64::new(3);
    let net = random_net(&codec, &mut rng);
    let mut sparse = net.clone();
    sparse.prepare();
    let mut n = 0;
    random_states(&dex, 20, 5, |b| {
        let mut r = Vec::new();
        codec.encode(b, &dex, &mut r);
        let (l, ls) = (net.logit(&codec, &r), net.logit(&codec, &swap_record(&r)));
        assert!((l + ls).abs() < 1e-4, "logit {l} vs swapped {ls}");
        let lsp = sparse.logit(&codec, &r);
        assert!((l - lsp).abs() < 1e-4, "dense {l} vs sparse forward {lsp}");
        n += 1;
    });
    assert!(n > 100);
}
