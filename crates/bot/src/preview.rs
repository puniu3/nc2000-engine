//! M8 baked team-preview tables: the offline-solved (mixed) preview
//! equilibrium over the meta matchup matrix, plus the agents that consume
//! and probe it.
//!
//! Preview action space: NC2000 previews are ordered picks of 3 from 6
//! (120 per side), but bench order only affects which display slot a random
//! drag-in (Roar/Whirlwind) resolves to — uniform over the eligible bench
//! either way — so win probability is invariant under bench permutation.
//! The baked space is therefore the 60 canonical actions
//! (20 subsets × 3 leads, bench ascending), a 4x cell saving.
//!
//! Per meta-pool pair (row team a, column team b) a `PairTable` records the
//! estimated preview payoff matrix (side-a win rate per joint preview), the
//! refined support, and the RM+-solved mixed equilibrium alongside the
//! argmax policy and both policies' exact counter-picking guarantees on the
//! matrix — the M8 gate quantities.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::rng::SplitMix64;
use crate::smmcts::solve_rm_plus;

// ------------------------------------------------------------ action space

/// Canonical preview actions: `[lead, bench_lo, bench_hi]`, 1-based display
/// positions, subsets enumerated lexicographically, leads in subset order.
pub fn preview_actions() -> Vec<[u8; 3]> {
    let mut out = Vec::with_capacity(60);
    for a in 1..=6u8 {
        for b in a + 1..=6 {
            for c in b + 1..=6 {
                for lead in [a, b, c] {
                    let mut bench = [a, b, c].into_iter().filter(|&x| x != lead);
                    out.push([lead, bench.next().unwrap(), bench.next().unwrap()]);
                }
            }
        }
    }
    debug_assert_eq!(out.len(), 60);
    out
}

/// Canonical index of an arbitrary ordered `Team` triple (lead kept, bench
/// sorted). `None` if malformed.
pub fn action_index(actions: &[[u8; 3]], triple: [u8; 3]) -> Option<usize> {
    let canon = canonical_triple(triple);
    actions.iter().position(|&a| a == canon)
}

/// Lead kept, bench ascending.
pub fn canonical_triple(t: [u8; 3]) -> [u8; 3] {
    if t[1] <= t[2] {
        t
    } else {
        [t[0], t[2], t[1]]
    }
}

// -------------------------------------------------------------- meta pool

#[derive(serde::Deserialize)]
pub struct MetaPool {
    pub teams: Vec<MetaTeam>,
}

#[derive(serde::Deserialize)]
pub struct MetaTeam {
    pub id: String,
    pub sets: Vec<PokemonSet>,
}

pub fn load_meta_pool(path: &Path) -> MetaPool {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

// --------------------------------------------------------- team signatures

/// Order-insensitive identity of a 6-mon team as the engine canonicalizes it:
/// sorted (species, level, item+1, sorted move ids). Computed from a live
/// roster, so table lookup works from battle state alone.
pub type TeamSig = Vec<(u16, u8, u16, [u16; 4])>;

pub fn roster_sig(battle: &Battle, side: usize) -> TeamSig {
    let mut v: TeamSig = battle.sides[side]
        .roster
        .iter()
        .map(|p| {
            let mut mv = [0u16; 4];
            for (i, s) in p.base_move_slots.iter().enumerate() {
                mv[i] = s.id.0;
            }
            mv.sort_unstable();
            (p.species.0, p.level, p.item.map_or(0, |i| i.0 + 1), mv)
        })
        .collect();
    v.sort_unstable();
    v
}

/// Signature of a pool team, via a throwaway battle so canonicalization
/// (string → id, item/move normalization) is exactly the live path.
pub fn team_sig(dex: &Dex, sets: &[PokemonSet]) -> TeamSig {
    let b = Battle::from_fixture(dex, "1,2,3,4", sets, sets)
        .expect("pool team failed to construct");
    roster_sig(&b, 0)
}

// ------------------------------------------------------------ table format

/// A payoff-matrix estimate: per-cell sample count and mean side-a score.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MatrixEst {
    pub rows: usize,
    pub cols: usize,
    pub n: Vec<u32>,
    pub v: Vec<f64>,
}

impl MatrixEst {
    pub fn new(rows: usize, cols: usize) -> MatrixEst {
        MatrixEst { rows, cols, n: vec![0; rows * cols], v: vec![0.5; rows * cols] }
    }

    pub fn record(&mut self, cell: usize, score: f64) {
        self.n[cell] += 1;
        self.v[cell] += (score - self.v[cell]) / self.n[cell] as f64;
    }

    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.v[r * self.cols + c]
    }

    /// Round stored means for compact JSON (4 dp ≫ estimation noise).
    pub fn compact(&mut self) {
        for v in self.v.iter_mut() {
            *v = (*v * 1e4).round() / 1e4;
        }
    }
}

/// Solved preview stage game for one matchup. All strategies live in the
/// canonical 60-action index space; `support` records which actions the
/// refined matrix covers.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PairSolution {
    /// Mixed equilibrium (RM+ average strategy, dust-thresholded), len 60.
    pub p_a: Vec<f64>,
    pub p_b: Vec<f64>,
    /// The argmax policy: best pure reply to the opponent's equilibrium.
    pub argmax_a: usize,
    pub argmax_b: usize,
    /// Side-a equilibrium value on the refined matrix.
    pub value: f64,
    /// Exact counter-picking guarantees on the refined matrix: the payoff
    /// each policy keeps against a best-responding opponent restricted to
    /// the refined support. Gate: mixed ≥ argmax, margin = how much
    /// counter-picking punishes the pure policy.
    pub guarantee_mixed_a: f64,
    pub guarantee_argmax_a: f64,
    pub guarantee_mixed_b: f64,
    pub guarantee_argmax_b: f64,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct BakeCfg {
    pub screen_games: u32,
    pub refine_games: u32,
    pub support: usize,
    pub skuct_iters: u32,
    pub advisor_iters: u32,
    pub advisor_runs: u32,
    pub eps: f64,
    pub max_turns: u16,
    pub seed: u64,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PairTable {
    /// Pool ids; team_a is the row player (matrix v = side-a score).
    pub team_a: String,
    pub team_b: String,
    /// The canonical 60 triples (self-description; identical for all pairs).
    pub actions: Vec<[u8; 3]>,
    /// Full-width screening estimate (cheap rollout-policy games).
    pub screen: MatrixEst,
    /// Refined support (indices into `actions`) per side.
    pub support: [Vec<usize>; 2],
    /// Refined estimate over support×support (skuct self-play games).
    pub refine: MatrixEst,
    pub sol: PairSolution,
    pub cfg: BakeCfg,
    pub secs: f64,
}

// ------------------------------------------------------------- solving

/// Solve the refined stage game and derive the shipped policies + gate
/// numbers. `refine` is |s0|×|s1| over `support`; strategies are embedded
/// into the 60-action space.
pub fn solve_pair(
    refine: &MatrixEst,
    support: &[Vec<usize>; 2],
    threshold: f64,
    sweeps: u32,
) -> PairSolution {
    let (k0, k1) = (support[0].len(), support[1].len());
    assert_eq!((refine.rows, refine.cols), (k0, k1));
    let (mut s0, mut s1) = solve_rm_plus(&refine.v, [k0, k1], sweeps);
    shed_dust(&mut s0, threshold);
    shed_dust(&mut s1, threshold);

    // payoff of each row vs σ1 / each column vs σ0 (side-a perspective)
    let row_u: Vec<f64> = (0..k0)
        .map(|a| (0..k1).map(|b| refine.at(a, b) * s1[b]).sum())
        .collect();
    let col_u: Vec<f64> = (0..k1)
        .map(|b| (0..k0).map(|a| refine.at(a, b) * s0[a]).sum())
        .collect();
    let value: f64 = (0..k0).map(|a| row_u[a] * s0[a]).sum();
    let br_a = argmax(&row_u);
    let br_b = argmin(&col_u); // side b maximizes 1 − u

    // counter-picking guarantees: worst case over the opponent's support
    let g_mixed_a = (0..k1).map(|b| col_u[b]).fold(f64::INFINITY, f64::min);
    let g_argmax_a = (0..k1).map(|b| refine.at(br_a, b)).fold(f64::INFINITY, f64::min);
    let g_mixed_b = 1.0 - (0..k0).map(|a| row_u[a]).fold(f64::NEG_INFINITY, f64::max);
    let g_argmax_b =
        1.0 - (0..k0).map(|a| refine.at(a, br_b)).fold(f64::NEG_INFINITY, f64::max);

    let embed = |s: &[f64], sup: &[usize]| {
        let mut p = vec![0.0; 60];
        for (j, &a) in sup.iter().enumerate() {
            p[a] = s[j];
        }
        p
    };
    PairSolution {
        p_a: embed(&s0, &support[0]),
        p_b: embed(&s1, &support[1]),
        argmax_a: support[0][br_a],
        argmax_b: support[1][br_b],
        value,
        guarantee_mixed_a: g_mixed_a,
        guarantee_argmax_a: g_argmax_a,
        guarantee_mixed_b: g_mixed_b,
        guarantee_argmax_b: g_argmax_b,
    }
}

/// Drop probabilities below `threshold × max` and renormalize — solver dust,
/// not genuine mixing. Gentler than the M7 in-battle threshold because the
/// offline matrix is the ground truth here.
fn shed_dust(p: &mut [f64], threshold: f64) {
    let pmax = p.iter().cloned().fold(0.0, f64::max);
    for v in p.iter_mut() {
        if *v < threshold * pmax {
            *v = 0.0;
        }
    }
    let z: f64 = p.iter().sum();
    for v in p.iter_mut() {
        *v /= z;
    }
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

fn argmin(v: &[f64]) -> usize {
    (0..v.len()).min_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

// ------------------------------------------------------------- table set

/// All baked tables + pool signatures, shared read-only across agents.
pub struct TableSet {
    pub ids: Vec<String>,
    sig_to_idx: HashMap<TeamSig, usize>,
    tables: HashMap<(usize, usize), PairTable>,
    actions: Vec<[u8; 3]>,
}

impl TableSet {
    pub fn load(dex: &Dex, pool: &MetaPool, dir: &Path) -> Arc<TableSet> {
        let mut sig_to_idx = HashMap::new();
        let mut ids = Vec::new();
        for (i, t) in pool.teams.iter().enumerate() {
            ids.push(t.id.clone());
            if sig_to_idx.insert(team_sig(dex, &t.sets), i).is_some() {
                eprintln!("warning: duplicate team signature in pool at {}", t.id);
            }
        }
        let id_idx: HashMap<&str, usize> =
            ids.iter().enumerate().map(|(i, id)| (id.as_str(), i)).collect();
        let mut tables = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("json") {
                    continue;
                }
                let Ok(text) = std::fs::read_to_string(&p) else { continue };
                let Ok(tab) = serde_json::from_str::<PairTable>(&text) else { continue };
                let (Some(&i), Some(&j)) =
                    (id_idx.get(tab.team_a.as_str()), id_idx.get(tab.team_b.as_str()))
                else {
                    eprintln!("warning: {} references unknown teams", p.display());
                    continue;
                };
                tables.insert((i, j), tab);
            }
        }
        Arc::new(TableSet { ids, sig_to_idx, tables, actions: preview_actions() })
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn pairs(&self) -> impl Iterator<Item = (&(usize, usize), &PairTable)> {
        self.tables.iter()
    }

    /// Resolve both sides of a live preview to pool indices, orientation
    /// included: `(table, my_side_is_a)`.
    fn lookup(&self, battle: &Battle, side: usize) -> Option<(&PairTable, bool)> {
        let me = *self.sig_to_idx.get(&roster_sig(battle, side))?;
        let opp = *self.sig_to_idx.get(&roster_sig(battle, 1 - side))?;
        if let Some(t) = self.tables.get(&(me, opp)) {
            return Some((t, true));
        }
        self.tables.get(&(opp, me)).map(|t| (t, false))
    }
}

// ------------------------------------------------------------- baked agent

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewMode {
    Mixed,
    Argmax,
}

/// Plays team preview from the baked tables (mixed equilibrium sample or
/// argmax), everything else via `inner`. Unknown matchups fall back to
/// `inner`'s own preview search.
pub struct BakedPreviewAgent {
    tables: Arc<TableSet>,
    inner: Box<dyn Agent>,
    mode: PreviewMode,
    rng: SplitMix64,
}

impl BakedPreviewAgent {
    pub fn new(
        tables: Arc<TableSet>,
        inner: Box<dyn Agent>,
        mode: PreviewMode,
        seed: u64,
    ) -> Self {
        BakedPreviewAgent { tables, inner, mode, rng: SplitMix64::new(seed) }
    }

    fn table_pick(&mut self, battle: &Battle, side: usize) -> Option<SearchChoice> {
        let (tab, i_am_a) = self.tables.lookup(battle, side)?;
        let idx = match self.mode {
            PreviewMode::Argmax => {
                if i_am_a {
                    tab.sol.argmax_a
                } else {
                    tab.sol.argmax_b
                }
            }
            PreviewMode::Mixed => {
                let p = if i_am_a { &tab.sol.p_a } else { &tab.sol.p_b };
                let u = self.rng.next_f64();
                let mut acc = 0.0;
                let mut pick = argmax(p);
                for (i, &pr) in p.iter().enumerate() {
                    acc += pr;
                    if u < acc {
                        pick = i;
                        break;
                    }
                }
                pick
            }
        };
        Some(SearchChoice::Team(self.tables.actions[idx]))
    }
}

impl Agent for BakedPreviewAgent {
    fn name(&self) -> String {
        let m = match self.mode {
            PreviewMode::Mixed => "baked",
            PreviewMode::Argmax => "bakedarg",
        };
        format!("{m}({})", self.inner.name())
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if matches!(choices[0], SearchChoice::Team(_)) {
            if let Some(c) = self.table_pick(battle, side) {
                debug_assert!(choices.contains(&c), "baked preview outside legal set");
                if choices.contains(&c) {
                    return c;
                }
            }
        }
        self.inner.choose(battle, dex, side, choices)
    }

    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> Vec<f64> {
        if matches!(choices[0], SearchChoice::Team(_)) {
            if let Some((tab, i_am_a)) = self.tables.lookup(battle, side) {
                let (p, amax) = if i_am_a {
                    (&tab.sol.p_a, tab.sol.argmax_a)
                } else {
                    (&tab.sol.p_b, tab.sol.argmax_b)
                };
                let actions = &self.tables.actions;
                return choices
                    .iter()
                    .map(|c| match c {
                        SearchChoice::Team(t) => {
                            // mass sits on the canonical ordering only
                            if canonical_triple(*t) != *t {
                                return 0.0;
                            }
                            let i = action_index(actions, *t).unwrap();
                            match self.mode {
                                PreviewMode::Mixed => p[i],
                                PreviewMode::Argmax => (i == amax) as u8 as f64,
                            }
                        }
                        _ => 0.0,
                    })
                    .collect();
            }
        }
        self.inner.root_policy(battle, dex, side, choices)
    }
}

// ----------------------------------------------------------- counter agent

/// The counter-picking probe: knows the opponent plays the baked table
/// (mixed or argmax) and best-responds at preview using the refined matrix,
/// then plays `inner` in battle. This is the adversary the M8 gate says the
/// mixed equilibrium must lose less to than the argmax policy does.
pub struct CounterPickAgent {
    tables: Arc<TableSet>,
    inner: Box<dyn Agent>,
    /// Which policy the opponent is assumed to play.
    target: PreviewMode,
}

impl CounterPickAgent {
    pub fn new(tables: Arc<TableSet>, inner: Box<dyn Agent>, target: PreviewMode) -> Self {
        CounterPickAgent { tables, inner, target }
    }

    fn br_pick(&self, battle: &Battle, side: usize) -> Option<SearchChoice> {
        let (tab, i_am_a) = self.tables.lookup(battle, side)?;
        let (my_sup, opp_sup) = if i_am_a {
            (&tab.support[0], &tab.support[1])
        } else {
            (&tab.support[1], &tab.support[0])
        };
        // opponent strategy over their support
        let opp_p: Vec<f64> = match self.target {
            PreviewMode::Mixed => {
                let p = if i_am_a { &tab.sol.p_b } else { &tab.sol.p_a };
                opp_sup.iter().map(|&a| p[a]).collect()
            }
            PreviewMode::Argmax => {
                let amax = if i_am_a { tab.sol.argmax_b } else { tab.sol.argmax_a };
                opp_sup.iter().map(|&a| (a == amax) as u8 as f64).collect()
            }
        };
        // my expected payoff per support action against that strategy
        let u: Vec<f64> = (0..my_sup.len())
            .map(|m| {
                (0..opp_sup.len())
                    .map(|o| {
                        let v = if i_am_a { tab.refine.at(m, o) } else { 1.0 - tab.refine.at(o, m) };
                        v * opp_p[o]
                    })
                    .sum()
            })
            .collect();
        Some(SearchChoice::Team(self.tables.actions[my_sup[argmax(&u)]]))
    }
}

impl Agent for CounterPickAgent {
    fn name(&self) -> String {
        let t = match self.target {
            PreviewMode::Mixed => "counter",
            PreviewMode::Argmax => "counterarg",
        };
        format!("{t}({})", self.inner.name())
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if matches!(choices[0], SearchChoice::Team(_)) {
            if let Some(c) = self.br_pick(battle, side) {
                if choices.contains(&c) {
                    return c;
                }
            }
        }
        self.inner.choose(battle, dex, side, choices)
    }
}

// ------------------------------------------------------------- rollout agent

/// The M6 heavy-playout policy as a standalone agent (ε-greedy max-damage) —
/// the cheap self-play policy that fills the full-width screening matrix.
pub struct RolloutAgent {
    playout: crate::mcts::Playout,
    rng: SplitMix64,
}

impl RolloutAgent {
    pub fn new(eps: f64, seed: u64) -> Self {
        RolloutAgent {
            playout: crate::mcts::Playout::Heavy {
                eps,
                turns: 8,
                weights: crate::eval::EvalWeights::default(),
            },
            rng: SplitMix64::new(seed),
        }
    }
}

impl Agent for RolloutAgent {
    fn name(&self) -> String {
        "rollout".into()
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        crate::mcts::playout_pick(battle, dex, &self.playout, side, choices, &mut self.rng)
    }
}

// ------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_space_is_canonical() {
        let acts = preview_actions();
        assert_eq!(acts.len(), 60);
        for (i, a) in acts.iter().enumerate() {
            assert!(a[1] < a[2], "bench not ascending: {a:?}");
            assert_ne!(a[0], a[1]);
            assert_ne!(a[0], a[2]);
            assert_eq!(action_index(&acts, *a), Some(i));
            // the non-canonical ordering maps to the same index
            assert_eq!(action_index(&acts, [a[0], a[2], a[1]]), Some(i));
        }
    }

    #[test]
    fn matching_pennies_mixes() {
        // classic 2x2: pure play is fully counter-picked, mixing holds 0.5
        let refine = MatrixEst {
            rows: 2,
            cols: 2,
            n: vec![100; 4],
            v: vec![1.0, 0.0, 0.0, 1.0],
        };
        let support = [vec![0, 1], vec![0, 1]];
        let sol = solve_pair(&refine, &support, 0.05, 20_000);
        assert!((sol.value - 0.5).abs() < 0.01, "value {}", sol.value);
        assert!(sol.guarantee_mixed_a > 0.45);
        assert!(sol.guarantee_argmax_a < 0.05);
        assert!(sol.guarantee_mixed_a >= sol.guarantee_argmax_a);
        assert!(sol.guarantee_mixed_b >= sol.guarantee_argmax_b);
        assert!((sol.p_a[0] - 0.5).abs() < 0.02);
    }

    #[test]
    fn dominant_action_purifies() {
        // row 0 dominates: solution should be (essentially) pure on it
        let refine = MatrixEst {
            rows: 2,
            cols: 2,
            n: vec![100; 4],
            v: vec![0.8, 0.7, 0.3, 0.2],
        };
        let support = [vec![4, 9], vec![2, 7]];
        let sol = solve_pair(&refine, &support, 0.05, 20_000);
        assert!(sol.p_a[4] > 0.98, "p_a[4] = {}", sol.p_a[4]);
        assert_eq!(sol.argmax_a, 4);
        // column player prefers col 1 (0.7/0.2 vs 0.8/0.3)
        assert_eq!(sol.argmax_b, 7);
        assert!((sol.guarantee_mixed_a - sol.guarantee_argmax_a).abs() < 0.02);
    }
}
