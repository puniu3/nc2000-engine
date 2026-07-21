//! M17e step 2 — exact endgame solver.
//!
//! Perfect-information subgame value (side-0 win probability; tie = 0.5)
//! by depth-first backward induction over `enumerate_step`'s exact chance
//! distributions. The game is finite and acyclic: `state_key` includes the
//! turn counter and every within-turn step strictly advances the queue, and
//! the engine auto-ties past turn 1000 — so plain memoized DFS is exact,
//! no cycle handling needed. The memo is shared across `solve` calls
//! (endgame positions from one game overlap heavily).
//!
//! Simultaneous decision nodes are zero-sum matrix games, solved by a
//! compact tableau simplex on the classic positive-shift LP; the returned
//! value is certified by a best-response bracket (row guarantee vs column
//! guarantee) whose worst width is reported in `ExactStats` — everything
//! here is deterministic, so re-runs are bit-identical.

use std::collections::HashMap;

use nc2000_engine::battle::enumerate::enumerate_step;
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

#[derive(Clone, Debug)]
pub struct ExactConfig {
    /// Max NEW decision states expanded per `solve` call before giving up.
    pub state_budget: usize,
    /// Per-step chance-leaf cap (passed to `enumerate_step`).
    pub leaf_cap: usize,
    /// Max chance leaves enumerated per `solve` call — the actual work
    /// bound: state count alone lets one unsolvable attempt burn
    /// states × matrix-cells × leaves engine runs before giving up.
    pub work_budget: usize,
    /// Iterative-deepening horizons, in full turns past the root.
    pub horizons: &'static [u16],
    /// Stop deepening once the root interval is at most this wide.
    pub eps: f64,
    /// Stop deepening when one horizon step narrows the interval by less
    /// than this (stall positions never converge — spend nothing more).
    pub stall_gain: f64,
}

impl Default for ExactConfig {
    fn default() -> Self {
        ExactConfig {
            state_budget: 100_000,
            leaf_cap: 100_000,
            work_budget: 2_000_000,
            horizons: &[1, 2, 4, 8, 12],
            eps: 0.02,
            stall_gain: 0.05,
        }
    }
}

/// A certified value bracket: the true (turn-1000-tie) game value lies in
/// `[lo, hi]` — chance handled exactly, lines truncated `horizon` turns
/// past the root contribute their full [0,1] uncertainty. `hi − lo` ≤ eps
/// means solved-for-practical-purposes; width 0 means fully resolved.
#[derive(Clone, Copy, Debug)]
pub struct Certified {
    pub lo: f64,
    pub hi: f64,
    pub horizon: u16,
}

impl Certified {
    pub fn mid(&self) -> f64 {
        (self.lo + self.hi) * 0.5
    }
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExactStats {
    /// Decision states solved (memo size).
    pub states: usize,
    /// Chance leaves enumerated (= engine turn executions).
    pub chance_runs: usize,
    /// Worst certified simplex bracket width seen.
    pub worst_gap: f64,
    /// Largest simultaneous matrix solved.
    pub max_matrix: [usize; 2],
}

pub struct ExactSolver<'d> {
    dex: &'d Dex,
    pub cfg: ExactConfig,
    /// Fully-resolved subgame values (interval width 0: no truncated line
    /// contributes) — true values regardless of horizon, kept forever and
    /// shared across positions.
    exact_memo: HashMap<u64, f64>,
    /// Interval memo for the current (root, horizon) pass only.
    ival_memo: HashMap<u64, (f64, f64)>,
    t_max: u16,
    limit: usize,
    work_limit: usize,
    pub stats: ExactStats,
}

impl<'d> ExactSolver<'d> {
    pub fn new(dex: &'d Dex, cfg: ExactConfig) -> Self {
        ExactSolver {
            dex,
            cfg,
            exact_memo: HashMap::new(),
            ival_memo: HashMap::new(),
            t_max: 0,
            limit: 0,
            work_limit: 0,
            stats: ExactStats::default(),
        }
    }

    /// Certified value bracket for `b` by iterative deepening over
    /// `cfg.horizons`. Returns the tightest bracket obtained before the
    /// budget ran out, `eps` was reached, or deepening stopped paying
    /// (`stall_gain`); `None` if not even the first horizon finished.
    pub fn solve(&mut self, b: &Battle) -> Option<Certified> {
        self.limit = self.exact_memo.len() + self.cfg.state_budget;
        self.work_limit = self.stats.chance_runs + self.cfg.work_budget;
        let mut best: Option<Certified> = None;
        let mut prev_width = f64::INFINITY;
        for &h in self.cfg.horizons {
            self.t_max = b.turn.saturating_add(h);
            self.ival_memo.clear();
            let Some((lo, hi)) = self.ival(b) else { break };
            best = Some(Certified { lo, hi, horizon: h });
            let w = hi - lo;
            if w <= self.cfg.eps || prev_width - w < self.cfg.stall_gain {
                break;
            }
            prev_width = w;
        }
        self.stats.states = self.exact_memo.len();
        best
    }

    fn ival(&mut self, b: &Battle) -> Option<(f64, f64)> {
        if let Some(o) = b.outcome() {
            let v = match o {
                Outcome::P1Win => 1.0,
                Outcome::Tie => 0.5,
                Outcome::P2Win => 0.0,
            };
            return Some((v, v));
        }
        let key = b.state_key();
        if let Some(&v) = self.exact_memo.get(&key) {
            return Some((v, v));
        }
        if b.turn > self.t_max {
            return Some((0.0, 1.0)); // truncated: full uncertainty
        }
        if let Some(&iv) = self.ival_memo.get(&key) {
            return Some(iv);
        }
        if self.exact_memo.len() >= self.limit || self.stats.chance_runs >= self.work_limit {
            return None;
        }

        let needs = b.needs_choice();
        let mut probe = b.clone();
        let acts = |probe: &mut Battle, dex, side: usize, need: bool| -> Vec<Option<SearchChoice>> {
            if need {
                probe.legal_choices(dex, side).into_iter().map(Some).collect()
            } else {
                vec![None]
            }
        };
        let a0 = acts(&mut probe, self.dex, 0, needs[0]);
        let a1 = acts(&mut probe, self.dex, 1, needs[1]);
        let (n0, n1) = (a0.len(), a1.len());
        if n0 == 0 || n1 == 0 {
            return None; // defensive: a side owes a choice but has none
        }

        let mut mlo = vec![0.0f64; n0 * n1];
        let mut mhi = vec![0.0f64; n0 * n1];
        for (i, &c0) in a0.iter().enumerate() {
            for (j, &c1) in a1.iter().enumerate() {
                let leaves = enumerate_step(self.dex, b, [c0, c1], self.cfg.leaf_cap)?;
                self.stats.chance_runs += leaves.len();
                // Merge identical successors before recursing.
                let mut agg: HashMap<u64, (f64, usize)> = HashMap::new();
                for (idx, l) in leaves.iter().enumerate() {
                    let e = agg.entry(l.battle.state_key()).or_insert((0.0, idx));
                    e.0 += l.prob;
                }
                let (mut elo, mut ehi) = (0.0, 0.0);
                for (p, idx) in agg.values() {
                    let (lo, hi) = self.ival(&leaves[*idx].battle)?;
                    elo += p * lo;
                    ehi += p * hi;
                }
                mlo[i * n1 + j] = elo;
                mhi[i * n1 + j] = ehi;
            }
        }

        // The matrix-game value is monotone in every payoff entry, so
        // solving the all-lo and all-hi matrices brackets the true value.
        let mut game = |m: &[f64]| -> f64 {
            if n0 == 1 && n1 == 1 {
                m[0]
            } else if n1 == 1 {
                m.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            } else if n0 == 1 {
                m.iter().cloned().fold(f64::INFINITY, f64::min)
            } else {
                let (v, gap) = solve_matrix(m, n0, n1);
                self.stats.worst_gap = self.stats.worst_gap.max(gap);
                v
            }
        };
        let (lo, hi) = (game(&mlo), game(&mhi));
        if n0 > 1 && n1 > 1 && n0 * n1 > self.stats.max_matrix[0] * self.stats.max_matrix[1] {
            self.stats.max_matrix = [n0, n1];
        }
        if hi - lo < 1e-12 {
            self.exact_memo.insert(key, (lo + hi) * 0.5);
        }
        self.ival_memo.insert(key, (lo, hi));
        Some((lo, hi))
    }
}

/// Value of the zero-sum matrix game `a` (row maximizes, row-major n0×n1),
/// with a certified bracket: returns ((lower+upper)/2, upper−lower) where
/// lower = row strategy's guarantee, upper = column strategy's guarantee.
pub fn solve_matrix(a: &[f64], n0: usize, n1: usize) -> (f64, f64) {
    // Positive shift so the game value is > 0.
    let min = a.iter().cloned().fold(f64::INFINITY, f64::min);
    let shift = min - 1.0;

    // LP (column player, scaled): maximize Σw_j s.t. Σ_j A'[i][j] w_j ≤ 1.
    // Tableau: n1 vars + n0 slacks + rhs; objective row last (stores
    // reduced costs; slack entries at optimum = dual values = row strategy).
    let cols = n1 + n0 + 1;
    let mut tab = vec![0.0f64; (n0 + 1) * cols];
    for i in 0..n0 {
        for j in 0..n1 {
            tab[i * cols + j] = a[i * n1 + j] - shift;
        }
        tab[i * cols + n1 + i] = 1.0;
        tab[i * cols + cols - 1] = 1.0;
    }
    for j in 0..n1 {
        tab[n0 * cols + j] = -1.0;
    }
    let mut basis: Vec<usize> = (n1..n1 + n0).collect();

    const EPS: f64 = 1e-12;
    loop {
        // Bland's rule: entering = lowest-index negative reduced cost.
        let Some(e) = (0..n1 + n0).find(|&j| tab[n0 * cols + j] < -EPS) else { break };
        // Ratio test; Bland tie-break on the leaving basis variable index.
        let mut leave: Option<(usize, f64)> = None;
        for i in 0..n0 {
            let coef = tab[i * cols + e];
            if coef > EPS {
                let ratio = tab[i * cols + cols - 1] / coef;
                let better = match leave {
                    None => true,
                    Some((li, lr)) => {
                        ratio < lr - EPS || (ratio < lr + EPS && basis[i] < basis[li])
                    }
                };
                if better {
                    leave = Some((i, ratio));
                }
            }
        }
        let (r, _) = leave.expect("game LP is bounded");
        // Pivot on (r, e).
        let p = tab[r * cols + e];
        for j in 0..cols {
            tab[r * cols + j] /= p;
        }
        for i in 0..n0 + 1 {
            if i != r {
                let f = tab[i * cols + e];
                if f != 0.0 {
                    for j in 0..cols {
                        tab[i * cols + j] -= f * tab[r * cols + j];
                    }
                }
            }
        }
        basis[r] = e;
    }

    let inv_v: f64 = tab[n0 * cols + cols - 1]; // Σw at optimum = 1/v'
    let vp = 1.0 / inv_v.max(EPS);

    // Recover strategies: column player w from the basis; row player from
    // the slack reduced costs (dual values).
    let mut y = vec![0.0f64; n1];
    for (i, &bv) in basis.iter().enumerate() {
        if bv < n1 {
            y[bv] = tab[i * cols + cols - 1] * vp;
        }
    }
    let mut x = vec![0.0f64; n0];
    for i in 0..n0 {
        x[i] = tab[n0 * cols + n1 + i] * vp;
    }
    // Normalize away accumulated fp drift.
    let (sx, sy) = (x.iter().sum::<f64>(), y.iter().sum::<f64>());
    x.iter_mut().for_each(|v| *v /= sx.max(EPS));
    y.iter_mut().for_each(|v| *v /= sy.max(EPS));

    // Certified bracket in the ORIGINAL payoffs.
    let lower = (0..n1)
        .map(|j| (0..n0).map(|i| x[i] * a[i * n1 + j]).sum::<f64>())
        .fold(f64::INFINITY, f64::min);
    let upper = (0..n0)
        .map(|i| (0..n1).map(|j| y[j] * a[i * n1 + j]).sum::<f64>())
        .fold(f64::NEG_INFINITY, f64::max);
    ((lower + upper) * 0.5, (upper - lower).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::solve_matrix;

    #[test]
    fn matching_pennies() {
        let (v, gap) = solve_matrix(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        assert!((v - 0.5).abs() < 1e-9, "v={v}");
        assert!(gap < 1e-9, "gap={gap}");
    }

    #[test]
    fn dominant_row() {
        // Row 0 dominates; column picks its best response 0.3.
        let (v, gap) = solve_matrix(&[0.7, 0.3, 0.2, 0.1], 2, 2);
        assert!((v - 0.3).abs() < 1e-9, "v={v}");
        assert!(gap < 1e-9, "gap={gap}");
    }

    #[test]
    fn rps_like() {
        // Rock-paper-scissors in win-prob form: value 0.5.
        let a = [0.5, 1.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.5];
        let (v, gap) = solve_matrix(&a, 3, 3);
        assert!((v - 0.5).abs() < 1e-9, "v={v}");
        assert!(gap < 1e-9, "gap={gap}");
    }

    #[test]
    fn asymmetric_2x3() {
        // Known value: max_x min(...) for [[0,1,2],[2,1,0]] is 1.0 at x=(.5,.5).
        let a = [0.0, 1.0, 2.0, 2.0, 1.0, 0.0];
        let (v, gap) = solve_matrix(&a, 2, 3);
        assert!((v - 1.0).abs() < 1e-9, "v={v}");
        assert!(gap < 1e-9, "gap={gap}");
    }
}
