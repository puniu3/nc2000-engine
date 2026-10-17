//! Bounded, analysis-only screen: full sampled Shapley vs double oracle,
//! with/without serialized alpha-beta bounds. No production agent changes.
//! Every (state, remaining depth, joint action, sample) has a fixed seed.
//! Values and pruning refer to that finite sampled game, NOT the real game.
use nc2000_bot::{
    exact::solve_matrix_full,
    value::{legal_actions, Leaf},
};
use nc2000_engine::{battle::SearchChoice, dex::Dex, state::Battle};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{cell::RefCell, collections::HashMap, time::Instant};

const EPS: f64 = 1e-10;
#[derive(Clone, Copy, Debug)]
struct Iv {
    lo: f64,
    hi: f64,
}
impl Iv {
    fn exact(v: f64) -> Self {
        Self { lo: v, hi: v }
    }
    fn full() -> Self {
        Self { lo: 0., hi: 1. }
    }
    fn done(self) -> bool {
        self.hi - self.lo < 1e-13
    }
    fn cut(self, a: f64, b: f64) -> bool {
        self.hi <= a || self.lo >= b
    }
    fn intersect(self, other: Self) -> Self {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        assert!(lo <= hi + 1e-8, "inconsistent interval {self:?} {other:?}");
        Self { lo: lo.min(hi), hi }
    }
}
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

// A generic backend lets the same pruning code be checked on exhaustive
// finite trees, including chance, transpositions and both move orders.
trait Game {
    type State: Clone;
    type Action: Copy;
    fn key(&self, s: &Self::State) -> u128;
    fn terminal(&self, s: &Self::State) -> bool;
    fn leaf(&self, s: &Self::State) -> f64;
    fn actions(&self, s: &Self::State) -> [Vec<Self::Action>; 2];
    fn step(&self, s: &Self::State, a: [Self::Action; 2], seed: u64) -> Self::State;
}
struct Engine<'a> {
    dex: &'a Dex,
    leaf: &'a Leaf,
    scratch: RefCell<Vec<f32>>,
}
impl Game for Engine<'_> {
    type State = Battle;
    type Action = Option<SearchChoice>;
    fn key(&self, s: &Battle) -> u128 {
        s.state_key128()
    }
    fn terminal(&self, s: &Battle) -> bool {
        s.outcome().is_some()
    }
    fn leaf(&self, s: &Battle) -> f64 {
        self.leaf.value(s, self.dex, &mut self.scratch.borrow_mut())
    }
    fn actions(&self, s: &Battle) -> [Vec<Self::Action>; 2] {
        legal_actions(s, self.dex)
    }
    fn step(&self, s: &Battle, a: [Self::Action; 2], seed: u64) -> Battle {
        let mut c = s.clone();
        c.set_log_enabled(false);
        c.reseed(seed);
        c.apply_choices(self.dex, a).unwrap();
        c
    }
}
#[derive(Default, Serialize, Debug)]
struct Counts {
    runs: u64,
    leaves: u64,
    nodes: u64,
    lp: u64,
    lp_fallbacks: u64,
    serial_nodes: u64,
    serial_cuts: u64,
    rejected_replies: u64,
    cell_hits: u64,
    node_hits: u64,
    ordering_runs: u64,
}
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
enum Mode {
    Full,
    Do,
    DoAb,
}
#[derive(Clone)]
struct Solution {
    v: f64,
    x: Vec<f64>,
    y: Vec<f64>,
    matrix: Vec<f64>,
}

// Exhaust all small LP vertices only when the shared simplex returns a
// nontrivial witness gap. Both arms pay this cost; never use an unverified
// midpoint as an exact matrix value. The final guarantees use ALL actions.
fn vertex_policy(a: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut best = (f64::NEG_INFINITY, vec![0.; n]);
    for mask in 1usize..1 << n {
        let rows: Vec<_> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        let k = rows.len();
        if k > m {
            continue;
        }
        for cmask in 1usize..1 << m {
            if cmask.count_ones() as usize != k {
                continue;
            }
            let cols: Vec<_> = (0..m).filter(|j| cmask & (1 << j) != 0).collect();
            let mut eq = vec![vec![0.; k + 2]; k + 1];
            for (r, &j) in cols.iter().enumerate() {
                for (c, &i) in rows.iter().enumerate() {
                    eq[r][c] = a[i * m + j];
                }
                eq[r][k] = -1.;
            }
            for c in 0..k {
                eq[k][c] = 1.;
            }
            eq[k][k + 1] = 1.;
            let mut singular = false;
            for c in 0..=k {
                let pivot = (c..=k)
                    .max_by(|&i, &j| eq[i][c].abs().total_cmp(&eq[j][c].abs()))
                    .unwrap();
                if eq[pivot][c].abs() < 1e-12 {
                    singular = true;
                    break;
                }
                eq.swap(c, pivot);
                let p = eq[c][c];
                for j in c..k + 2 {
                    eq[c][j] /= p;
                }
                for r in 0..=k {
                    if r != c {
                        let f = eq[r][c];
                        for j in c..k + 2 {
                            eq[r][j] -= f * eq[c][j];
                        }
                    }
                }
            }
            if singular || (0..k).any(|i| !eq[i][k + 1].is_finite() || eq[i][k + 1] < -1e-9) {
                continue;
            }
            let mut x = vec![0.; n];
            for (i, &r) in rows.iter().enumerate() {
                x[r] = eq[i][k + 1].max(0.);
            }
            let total = x.iter().sum::<f64>();
            if total <= 0. {
                continue;
            }
            for v in &mut x {
                *v /= total;
            }
            let guarantee = (0..m)
                .map(|j| (0..n).map(|i| x[i] * a[i * m + j]).sum::<f64>())
                .fold(f64::INFINITY, f64::min);
            if guarantee > best.0 {
                best = (guarantee, x);
            }
        }
    }
    assert!(best.0.is_finite());
    best.1
}
fn vertex_solve(a: &[f64], n: usize, m: usize) -> Solution {
    let x = vertex_policy(a, n, m);
    let dual: Vec<_> = (0..m)
        .flat_map(|j| (0..n).map(move |i| 1. - a[i * m + j]))
        .collect();
    let y = vertex_policy(&dual, m, n);
    let lo = (0..m)
        .map(|j| (0..n).map(|i| x[i] * a[i * m + j]).sum::<f64>())
        .fold(f64::INFINITY, f64::min);
    let hi = (0..n)
        .map(|i| (0..m).map(|j| y[j] * a[i * m + j]).sum::<f64>())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(hi - lo < 1e-8, "vertex fallback gap {}", hi - lo);
    Solution {
        v: (lo + hi) * 0.5,
        x,
        y,
        matrix: a.to_vec(),
    }
}
type NKey = (u128, usize, u8);
type CKey = (u128, usize, usize, usize, u8);
struct Search<'a, G: Game> {
    g: &'a G,
    mode: Mode,
    ks: Vec<usize>,
    seed: u64,
    nodes: HashMap<NKey, Iv>,
    cells: HashMap<CKey, Iv>,
    leaves: HashMap<u128, f64>,
    c: Counts,
    start: Instant,
    run_cap: u64,
    seconds: f64,
}
type R<T> = Result<T, &'static str>;
impl<'a, G: Game> Search<'a, G> {
    fn new(g: &'a G, mode: Mode, ks: Vec<usize>, seed: u64, run_cap: u64, seconds: f64) -> Self {
        Self {
            g,
            mode,
            ks,
            seed,
            nodes: HashMap::new(),
            cells: HashMap::new(),
            leaves: HashMap::new(),
            c: Counts::default(),
            start: Instant::now(),
            run_cap,
            seconds,
        }
    }
    fn check(&self) -> R<()> {
        if self.c.runs >= self.run_cap {
            Err("WorkCap")
        } else if self.start.elapsed().as_secs_f64() > self.seconds {
            Err("TimeCap")
        } else {
            Ok(())
        }
    }
    fn leaf(&mut self, s: &G::State) -> f64 {
        let key = self.g.key(s);
        if let Some(&v) = self.leaves.get(&key) {
            return v;
        }
        self.c.leaves += 1;
        let v = self.g.leaf(s);
        assert!((0.0..=1.0).contains(&v));
        self.leaves.insert(key, v);
        v
    }
    fn child(
        &mut self,
        s: &G::State,
        key: u128,
        d: usize,
        a: [G::Action; 2],
        i: usize,
        j: usize,
        k: usize,
    ) -> R<G::State> {
        self.check()?;
        self.c.runs += 1;
        let seed = mix64(
            self.seed
                ^ mix64(key as u64)
                ^ mix64((key >> 64) as u64)
                ^ mix64(d as u64 + 0x100)
                ^ mix64(i as u64 + 0x200)
                ^ mix64(j as u64 + 0x300)
                ^ mix64(k as u64 + 0x400),
        );
        Ok(self.g.step(s, a, seed))
    }
    // kind=0: actual simultaneous game, 1: row moves first (lower),
    // 2: column moves first (upper), at EVERY subsequent stage.
    fn cell(
        &mut self,
        s: &G::State,
        key: u128,
        d: usize,
        acts: &[Vec<G::Action>; 2],
        i: usize,
        j: usize,
        kind: u8,
        alpha: f64,
        beta: f64,
    ) -> R<Iv> {
        let kind = if d == 1 { 0 } else { kind };
        let ck = (key, d, i, j, kind);
        // Full traversal needs no incremental per-cell table.
        let cache_cell = self.mode != Mode::Full || kind != 0;
        let old = if cache_cell {
            self.cells.get(&ck).copied().unwrap_or(Iv::full())
        } else {
            Iv::full()
        };
        if old.done() || (kind != 0 && old.cut(alpha, beta)) {
            self.c.cell_hits += 1;
            return Ok(old);
        }
        let k = self.ks[self.ks.len() - d];
        let mut sum = Iv::exact(0.);
        for n in 0..k {
            let child = self.child(s, key, d, [acts[0][i], acts[1][j]], i, j, n)?;
            let remaining = (k - n - 1) as f64;
            let v = if d == 1 || self.g.terminal(&child) {
                Iv::exact(self.leaf(&child))
            } else if kind == 0 {
                Iv::exact(self.value(&child, d - 1)?)
            } else {
                self.serial(
                    &child,
                    d - 1,
                    kind,
                    alpha * k as f64 - sum.hi - remaining,
                    beta * k as f64 - sum.lo,
                )?
            };
            sum.lo += v.lo;
            sum.hi += v.hi;
            let bound = Iv {
                lo: sum.lo / k as f64,
                hi: (sum.hi + remaining) / k as f64,
            };
            if kind != 0 && bound.cut(alpha, beta) {
                let result = old.intersect(bound);
                if cache_cell {
                    self.cells.insert(ck, result);
                }
                return Ok(result);
            }
        }
        let result = old.intersect(Iv {
            lo: sum.lo / k as f64,
            hi: sum.hi / k as f64,
        });
        if cache_cell {
            self.cells.insert(ck, result);
        }
        Ok(result)
    }
    fn value(&mut self, s: &G::State, d: usize) -> R<f64> {
        if d == 0 || self.g.terminal(s) {
            return Ok(self.leaf(s));
        }
        let nk = (self.g.key(s), d, 0);
        if let Some(v) = self.nodes.get(&nk) {
            self.c.node_hits += 1;
            return Ok(v.lo);
        }
        Ok(self.solve(s, d)?.v)
    }
    fn serial(&mut self, s: &G::State, d: usize, kind: u8, alpha: f64, beta: f64) -> R<Iv> {
        if d == 0 || self.g.terminal(s) {
            return Ok(Iv::exact(self.leaf(s)));
        }
        let key = self.g.key(s);
        let nk = (key, d, kind);
        let old = self.nodes.get(&nk).copied().unwrap_or(Iv::full());
        if old.done() || old.cut(alpha, beta) {
            self.c.node_hits += 1;
            return Ok(old);
        }
        self.c.serial_nodes += 1;
        let acts = self.g.actions(s);
        let lower = kind == 1;
        let (outer, inner) = if lower {
            (acts[0].len(), acts[1].len())
        } else {
            (acts[1].len(), acts[0].len())
        };
        let mut result = if lower { Iv::exact(0.) } else { Iv::exact(1.) };
        for a in 0..outer {
            let mut sub = if lower { Iv::exact(1.) } else { Iv::exact(0.) };
            for b in 0..inner {
                let (i, j) = if lower { (a, b) } else { (b, a) };
                let al = if lower {
                    alpha.max(result.lo)
                } else {
                    alpha.max(sub.lo)
                };
                let be = if lower {
                    beta.min(sub.hi)
                } else {
                    beta.min(result.hi)
                };
                let v = self.cell(s, key, d, &acts, i, j, kind, al, be)?;
                if lower {
                    sub.lo = sub.lo.min(v.lo);
                    sub.hi = sub.hi.min(v.hi);
                } else {
                    sub.lo = sub.lo.max(v.lo);
                    sub.hi = sub.hi.max(v.hi);
                }
                if b + 1 < inner && ((lower && sub.hi <= al) || (!lower && sub.lo >= be)) {
                    self.c.serial_cuts += 1;
                    if lower {
                        sub.lo = 0.;
                    } else {
                        sub.hi = 1.;
                    }
                    break;
                }
            }
            if lower {
                result.lo = result.lo.max(sub.lo);
                result.hi = result.hi.max(sub.hi);
            } else {
                result.lo = result.lo.min(sub.lo);
                result.hi = result.hi.min(sub.hi);
            }
            if a + 1 < outer && ((lower && result.lo >= beta) || (!lower && result.hi <= alpha)) {
                self.c.serial_cuts += 1;
                if lower {
                    result.hi = 1.;
                } else {
                    result.lo = 0.;
                }
                break;
            }
        }
        let result = old.intersect(result);
        self.nodes.insert(nk, result);
        Ok(result)
    }
    fn lp(&mut self, m: &[f64], n: usize, k: usize) -> Solution {
        if n == 1 || k == 1 {
            let (v, [x, y]) = nc2000_bot::value::solve(m, n, k);
            return Solution {
                v,
                x,
                y,
                matrix: m.to_vec(),
            };
        }
        self.c.lp += 1;
        let s = solve_matrix_full(m, n, k);
        if s.gap >= 1e-8 {
            self.c.lp_fallbacks += 1;
            return vertex_solve(m, n, k);
        }
        Solution {
            v: s.value,
            x: s.x,
            y: s.y,
            matrix: m.to_vec(),
        }
    }
    fn solve(&mut self, s: &G::State, d: usize) -> R<Solution> {
        self.check()?;
        self.c.nodes += 1;
        let key = self.g.key(s);
        let acts = self.g.actions(s);
        let n = acts[0].len();
        let m = acts[1].len();
        assert!(n > 0 && m > 0);
        let result = if self.mode == Mode::Full {
            let mut a = vec![0.; n * m];
            for i in 0..n {
                for j in 0..m {
                    a[i * m + j] = self.cell(s, key, d, &acts, i, j, 0, -1., 2.)?.lo;
                }
            }
            self.lp(&a, n, m)
        } else {
            // Charged one-step ordering at internal nodes; use the very same
            // first sample as the deeper game, never consult baseline values.
            let (mut ri, mut cj) = (vec![0], vec![0]);
            let mut row_order: Vec<_> = (0..n).collect();
            let mut col_order: Vec<_> = (0..m).collect();
            if d > 1 {
                let mut h = vec![0.; n * m];
                for i in 0..n {
                    for j in 0..m {
                        let child = self.child(s, key, d, [acts[0][i], acts[1][j]], i, j, 0)?;
                        self.c.ordering_runs += 1;
                        h[i * m + j] = self.leaf(&child);
                    }
                }
                let order = self.lp(&h, n, m);
                row_order.sort_by(|&i, &j| order.x[j].total_cmp(&order.x[i]).then(i.cmp(&j)));
                col_order.sort_by(|&i, &j| order.y[j].total_cmp(&order.y[i]).then(i.cmp(&j)));
                ri[0] = row_order[0];
                cj[0] = col_order[0];
            }
            loop {
                let mut restricted = Vec::new();
                for &i in &ri {
                    for &j in &cj {
                        restricted.push(self.cell(s, key, d, &acts, i, j, 0, -1., 2.)?.lo);
                    }
                }
                let p = self.lp(&restricted, ri.len(), cj.len());
                let mut best_row = None;
                let mut best_row_v = p.v + EPS;
                for &i in &row_order {
                    if ri.contains(&i) {
                        continue;
                    }
                    let v = self.reply(s, key, d, &acts, &cj, &p.y, i, true, best_row_v)?;
                    if let Some(v) = v {
                        if v > best_row_v {
                            best_row_v = v;
                            best_row = Some(i);
                        }
                    }
                }
                let mut best_col = None;
                let mut best_col_v = p.v - EPS;
                for &j in &col_order {
                    if cj.contains(&j) {
                        continue;
                    }
                    let v = self.reply(s, key, d, &acts, &ri, &p.x, j, false, best_col_v)?;
                    if let Some(v) = v {
                        if v < best_col_v {
                            best_col_v = v;
                            best_col = Some(j);
                        }
                    }
                }
                if best_row.is_none() && best_col.is_none() {
                    let mut x = vec![0.; n];
                    let mut y = vec![0.; m];
                    for (&i, &w) in ri.iter().zip(&p.x) {
                        x[i] = w;
                    }
                    for (&j, &w) in cj.iter().zip(&p.y) {
                        y[j] = w;
                    }
                    break Solution {
                        v: p.v,
                        x,
                        y,
                        matrix: Vec::new(),
                    };
                }
                if let Some(i) = best_row {
                    ri.push(i);
                }
                if let Some(j) = best_col {
                    cj.push(j);
                }
                assert!(ri.len() <= n && cj.len() <= m);
            }
        };
        self.nodes.insert((key, d, 0), Iv::exact(result.v));
        Ok(result)
    }
    // A weighted best-response test. Serialized bounds can reject an action;
    // only simultaneous-game values may admit it to the restricted game.
    fn reply(
        &mut self,
        s: &G::State,
        key: u128,
        d: usize,
        acts: &[Vec<G::Action>; 2],
        support: &[usize],
        weights: &[f64],
        a: usize,
        maximize: bool,
        threshold: f64,
    ) -> R<Option<f64>> {
        let mut vals = vec![Iv::full(); support.len()];
        let mut order: Vec<_> = (0..support.len()).filter(|&t| weights[t] > 0.).collect();
        order.sort_by(|&i, &j| weights[j].total_cmp(&weights[i]));
        let sum = |vs: &[Iv]| -> Iv {
            Iv {
                lo: vs.iter().zip(weights).map(|(v, w)| v.lo * w).sum(),
                hi: vs.iter().zip(weights).map(|(v, w)| v.hi * w).sum(),
            }
        };
        for pass in if self.mode == Mode::DoAb && d > 1 {
            0..2
        } else {
            1..2
        } {
            for &t in &order {
                let total = sum(&vals);
                if (maximize && total.hi <= threshold) || (!maximize && total.lo >= threshold) {
                    self.c.rejected_replies += 1;
                    return Ok(None);
                }
                let other = if maximize {
                    total.hi - weights[t] * vals[t].hi
                } else {
                    total.lo - weights[t] * vals[t].lo
                };
                let cutoff = (threshold - other) / weights[t];
                let (i, j) = if maximize {
                    (a, support[t])
                } else {
                    (support[t], a)
                };
                let kind = if pass == 1 {
                    0
                } else if maximize {
                    2
                } else {
                    1
                };
                let v = self.cell(
                    s,
                    key,
                    d,
                    acts,
                    i,
                    j,
                    kind,
                    if maximize { cutoff } else { -1. },
                    if maximize { 2. } else { cutoff },
                )?;
                vals[t] = if pass == 1 {
                    v
                } else if maximize {
                    Iv { lo: 0., hi: v.hi }
                } else {
                    Iv { lo: v.lo, hi: 1. }
                };
            }
        }
        let v = sum(&vals);
        Ok(Some((v.lo + v.hi) * 0.5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vertex_fallback_validates_mixed_degenerate_and_random_games() {
        let p = vertex_solve(&[1., 0., 0., 1.], 2, 2);
        assert!((p.v - 0.5).abs() < 1e-12);
        let p = vertex_solve(&[0.2, 0.2, 0.8, 0.8, 0.8, 0.8], 3, 2);
        assert!((p.v - 0.8).abs() < 1e-12);
        for seed in 0..40 {
            let n = 2 + (seed % 3) as usize;
            let m = 2 + (seed % 4) as usize;
            let a: Vec<_> = (0..n * m)
                .map(|i| (mix64(seed ^ i as u64) % 7) as f64 / 6.)
                .collect();
            let p = vertex_solve(&a, n, m);
            let fast = solve_matrix_full(&a, n, m);
            if fast.gap < 1e-8 {
                assert!((p.v - fast.value).abs() < 1e-8);
            }
        }
    }
    struct Tiny {
        seed: u64,
    }
    impl Game for Tiny {
        type State = (u64, usize);
        type Action = usize;
        fn key(&self, s: &Self::State) -> u128 {
            s.0 as u128 + ((s.1 as u128) << 64)
        }
        fn terminal(&self, s: &Self::State) -> bool {
            s.1 == 0
        }
        fn leaf(&self, s: &Self::State) -> f64 {
            (mix64(s.0 ^ self.seed) % 11) as f64 / 10.
        }
        fn actions(&self, s: &Self::State) -> [Vec<usize>; 2] {
            [0, 1].map(|i| (0..1 + (mix64(s.0 + i) % 3) as usize).collect())
        }
        fn step(&self, s: &Self::State, a: [usize; 2], seed: u64) -> Self::State {
            (
                mix64(s.0 ^ mix64(a[0] as u64 + 31) ^ mix64(a[1] as u64 + 79) ^ seed % 3) % 19,
                s.1 - 1,
            )
        }
    }
    #[test]
    fn exhaustive_chance_games_and_serial_windows() {
        for seed in 0..40 {
            let g = Tiny { seed };
            let root = (seed, 3);
            let ks = vec![2, 2, 2];
            let mut full = Search::new(&g, Mode::Full, ks.clone(), 19, 1000000, 60.);
            let truth = full.solve(&root, 3).unwrap();
            for mode in [Mode::Do, Mode::DoAb] {
                let mut s = Search::new(&g, mode, ks.clone(), 19, 1000000, 60.);
                let v = s.solve(&root, 3).unwrap();
                assert!((v.v - truth.v).abs() < 1e-8, "seed {seed} mode {mode:?}");
                let n = v.x.len();
                let m = v.y.len();
                let lo = (0..m)
                    .map(|j| {
                        (0..n)
                            .map(|i| v.x[i] * truth.matrix[i * m + j])
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min);
                let hi = (0..n)
                    .map(|i| {
                        (0..m)
                            .map(|j| v.y[j] * truth.matrix[i * m + j])
                            .sum::<f64>()
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                assert!(hi - lo < 1e-8);
            }
            for kind in [1, 2] {
                let mut s = Search::new(&g, Mode::Full, ks.clone(), 19, 1000000, 60.);
                let exact = s.serial(&root, 3, kind, -1., 2.).unwrap();
                assert!(exact.done());
                assert!(if kind == 1 {
                    exact.hi <= truth.v + 1e-9
                } else {
                    exact.lo >= truth.v - 1e-9
                });
                for window in [(0.2, 0.21), (0.5, 0.51), (0.8, 0.81)] {
                    let mut t = Search::new(&g, Mode::Full, ks.clone(), 19, 1000000, 60.);
                    let b = t.serial(&root, 3, kind, window.0, window.1).unwrap();
                    assert!(b.lo <= exact.lo + 1e-9 && b.hi >= exact.hi - 1e-9);
                    let final_v = t.serial(&root, 3, kind, -1., 2.).unwrap();
                    assert!((final_v.lo - exact.lo).abs() < 1e-9 && final_v.done());
                }
            }
        }
    }
}

// The copied screen implementation above stays isolated from production and
// from the frozen previous screen. Only the leaf backend and this adapter
// differ. Budget counts include all completed and interrupted deepening work.
pub(super) struct MoveResult {
    pub policy: Vec<f64>,
    pub depth: usize,
    pub capped: bool,
    pub counts: serde_json::Value,
}
pub(super) fn choose_policy(
    b: &Battle,
    dex: &Dex,
    leaf: &Leaf,
    ks: &[usize],
    side: usize,
    seed: u64,
    cap: u64,
) -> MoveResult {
    let g = Engine {
        dex,
        leaf,
        scratch: RefCell::new(Vec::new()),
    };
    let mut search = Search::new(&g, Mode::DoAb, ks.to_vec(), seed, cap, 600.);
    let mut last = None;
    let mut depth = 0;
    let mut capped = false;
    for d in 1..=ks.len() {
        match search.solve(b, d) {
            Ok(s) => {
                last = Some(s);
                depth = d;
            }
            Err("WorkCap") => {
                capped = true;
                break;
            }
            Err(other) => panic!("non-deterministic search stop: {other}"),
        }
    }
    let s = last.expect("work budget must complete at least depth one");
    MoveResult {
        policy: if side == 0 { s.x } else { s.y },
        depth,
        capped,
        counts: json!(search.c),
    }
}
pub(super) fn check_position(
    b: &Battle,
    dex: &Dex,
    leaf: &Leaf,
    ks: &[usize],
) -> serde_json::Value {
    let g = Engine {
        dex,
        leaf,
        scratch: RefCell::new(Vec::new()),
    };
    let mut f = Search::new(&g, Mode::Full, ks.to_vec(), 717, 1000000, 600.);
    let full = f.solve(b, ks.len()).unwrap();
    let mut a = Search::new(&g, Mode::DoAb, ks.to_vec(), 717, 1000000, 600.);
    let mut out = None;
    for d in 1..=ks.len() {
        out = Some(a.solve(b, d).unwrap());
    }
    let ab = out.unwrap();
    let n = full.x.len();
    let m = full.y.len();
    let lo = (0..m)
        .map(|j| {
            (0..n)
                .map(|i| ab.x[i] * full.matrix[i * m + j])
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    let hi = (0..n)
        .map(|i| {
            (0..m)
                .map(|j| ab.y[j] * full.matrix[i * m + j])
                .sum::<f64>()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!((full.v - ab.v).abs() < 1e-7 && hi - lo < 1e-7);
    json!({"value_error":(full.v-ab.v).abs(),"policy_gap":hi-lo,"full_runs":f.c.runs,"ab_runs":a.c.runs})
}
