//! Pure-Rust multi-threaded Siamese NNUE Trainer & Rust Weight Exporter for `CF-V_net v3`.
//!
//! Trains the exact 0-ULP antisymmetric Siamese value network (`192 -> 128 -> 64 -> 1`)
//! matching `compact_az::evaluate_cf_value` directly on zero-copy binary `CfPairRecV3` shards.
//!
//! Features:
//! - Strict zero-leakage train/val split enforced by `team_hash_0 % 5` & `team_hash_1 % 5`:
//!   - Train: `team_hash_0 % 5 != 0 && team_hash_1 % 5 != 0`
//!   - Pure Unseen Val: `team_hash_0 % 5 == 0 && team_hash_1 % 5 == 0`
//! - Exact Siamese parameterization `(W1_plus, W1_minus, b1, W2_plus, W2_minus, b2, w_out)`
//!   so `z(x0, x1) == -z(x1, x0)` holds to 0-ULP at every training step.
//! - Loss: `w_s * (w_abs * 0.5*(BCE(z, v_cf) + BCE(z', vp_cf)) + w_cf * MSE(sig(z')-sig(z), vp_cf-v_cf))`
//!   with Turn-1 boost `w_s = w_t1` for `turn == 1` (`1.0` for midgame).
//! - Multi-threaded mini-batch AdamW + cosine LR + global L2 gradient clipping.
//! - Separate unseen validation tracking on **Unseen Midgame** and **Unseen Turn-1**
//!   (`r_val`, `r_delta`, `rmse`, `delta_rmse`, `sign_acc`).
//! - Direct Rust exporter to `compact_cf_weights.rs` preserving `CF_P_*` weights verbatim.
//!
//! M2M Output: emits a single compact JSON summary line on stdout (optional `--log` JSONL).

use nc2000_bot::compact_az::{
    eval_cf_value_logit_from_features, CfPairRecV3, CF_RAW_VALUE_SIDE_DIM,
};
use nc2000_bot::compact_cf_weights::{
    CF_VALUE_H1, CF_VALUE_H2, CF_V_B1, CF_V_B2, CF_V_W1_MINUS_T, CF_V_W1_PLUS_T, CF_V_W2_MINUS_T,
    CF_V_W2_PLUS_T, CF_V_W_OUT,
};
use nc2000_bot::rng::SplitMix64;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const D: usize = CF_RAW_VALUE_SIDE_DIM;
const H1: usize = CF_VALUE_H1;
const H2: usize = CF_VALUE_H2;

#[derive(Clone)]
struct SiameseNet {
    w1_plus: Vec<f32>,  // [D * H1]
    w1_minus: Vec<f32>, // [D * H1]
    b1: [f32; H1],
    w2_plus: Vec<f32>,  // [H1 * H2]
    w2_minus: Vec<f32>, // [H1 * H2]
    b2: [f32; H2],
    w_out: [f32; H2],
}

impl SiameseNet {
    fn zeros() -> Self {
        Self {
            w1_plus: vec![0.0; D * H1],
            w1_minus: vec![0.0; D * H1],
            b1: [0.0; H1],
            w2_plus: vec![0.0; H1 * H2],
            w2_minus: vec![0.0; H1 * H2],
            b2: [0.0; H2],
            w_out: [0.0; H2],
        }
    }

    fn kaiming_init(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut gauss = || -> f32 {
            // Box-Muller
            let u1 = rng.next_f64().max(1e-12);
            let u2 = rng.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        };
        let mut net = Self::zeros();
        let s1 = (1.0 / (D as f32)).sqrt() * 0.7;
        for v in &mut net.w1_plus {
            *v = gauss() * s1;
        }
        for v in &mut net.w1_minus {
            *v = gauss() * s1;
        }
        let s2 = (1.0 / (H1 as f32)).sqrt() * 0.7;
        for v in &mut net.w2_plus {
            *v = gauss() * s2;
        }
        for v in &mut net.w2_minus {
            *v = gauss() * s2;
        }
        let s3 = (1.0 / (H2 as f32)).sqrt();
        for v in &mut net.w_out {
            *v = gauss() * s3;
        }
        net
    }

    fn from_embedded() -> Self {
        let mut net = Self::zeros();
        for k in 0..D {
            for h in 0..H1 {
                net.w1_plus[k * H1 + h] = CF_V_W1_PLUS_T[k][h];
                net.w1_minus[k * H1 + h] = CF_V_W1_MINUS_T[k][h];
            }
        }
        net.b1 = CF_V_B1;
        for h in 0..H1 {
            for j in 0..H2 {
                net.w2_plus[h * H2 + j] = CF_V_W2_PLUS_T[h][j];
                net.w2_minus[h * H2 + j] = CF_V_W2_MINUS_T[h][j];
            }
        }
        net.b2 = CF_V_B2;
        net.w_out = CF_V_W_OUT;
        net
    }

    fn add_assign(&mut self, other: &Self) {
        for (a, b) in self.w1_plus.iter_mut().zip(&other.w1_plus) {
            *a += *b;
        }
        for (a, b) in self.w1_minus.iter_mut().zip(&other.w1_minus) {
            *a += *b;
        }
        for i in 0..H1 {
            self.b1[i] += other.b1[i];
        }
        for (a, b) in self.w2_plus.iter_mut().zip(&other.w2_plus) {
            *a += *b;
        }
        for (a, b) in self.w2_minus.iter_mut().zip(&other.w2_minus) {
            *a += *b;
        }
        for i in 0..H2 {
            self.b2[i] += other.b2[i];
            self.w_out[i] += other.w_out[i];
        }
    }

    fn scale(&mut self, c: f32) {
        for v in &mut self.w1_plus {
            *v *= c;
        }
        for v in &mut self.w1_minus {
            *v *= c;
        }
        for v in &mut self.b1 {
            *v *= c;
        }
        for v in &mut self.w2_plus {
            *v *= c;
        }
        for v in &mut self.w2_minus {
            *v *= c;
        }
        for v in &mut self.b2 {
            *v *= c;
        }
        for v in &mut self.w_out {
            *v *= c;
        }
    }

    fn l2_norm(&self) -> f64 {
        let mut s = 0.0f64;
        for &v in &self.w1_plus {
            s += f64::from(v) * f64::from(v);
        }
        for &v in &self.w1_minus {
            s += f64::from(v) * f64::from(v);
        }
        for &v in &self.b1 {
            s += f64::from(v) * f64::from(v);
        }
        for &v in &self.w2_plus {
            s += f64::from(v) * f64::from(v);
        }
        for &v in &self.w2_minus {
            s += f64::from(v) * f64::from(v);
        }
        for &v in &self.b2 {
            s += f64::from(v) * f64::from(v);
        }
        for &v in &self.w_out {
            s += f64::from(v) * f64::from(v);
        }
        s.sqrt()
    }

    /// Exact forward logit matching `compact_az::evaluate_cf_value` (0-ULP antisymmetry).
    #[inline]
    fn forward_logit(&self, x0: &[f32; D], x1: &[f32; D]) -> f32 {
        let mut a1 = self.b1;
        let mut b1 = [0.0f32; H1];
        for k in 0..D {
            let xs = x0[k] + x1[k];
            if xs != 0.0 {
                let row = &self.w1_plus[k * H1..(k + 1) * H1];
                for h in 0..H1 {
                    a1[h] += xs * row[h];
                }
            }
            let xd = x0[k] - x1[k];
            if xd != 0.0 {
                let row = &self.w1_minus[k * H1..(k + 1) * H1];
                for h in 0..H1 {
                    b1[h] += xd * row[h];
                }
            }
        }
        let mut a2 = self.b2;
        let mut b2 = [0.0f32; H2];
        for h in 0..H1 {
            let h0 = (a1[h] + b1[h]).max(0.0);
            let h1 = (a1[h] - b1[h]).max(0.0);
            let hs = h0 + h1;
            if hs != 0.0 {
                let row = &self.w2_plus[h * H2..(h + 1) * H2];
                for j in 0..H2 {
                    a2[j] += hs * row[j];
                }
            }
            let hd = h0 - h1;
            if hd != 0.0 {
                let row = &self.w2_minus[h * H2..(h + 1) * H2];
                for j in 0..H2 {
                    b2[j] += hd * row[j];
                }
            }
        }
        let mut z = 0.0f32;
        for j in 0..H2 {
            let d = (a2[j] + b2[j]).max(0.0) - (a2[j] - b2[j]).max(0.0);
            z += self.w_out[j] * d;
        }
        z
    }

    /// Forward pass caching activations for exact backprop.
    #[inline]
    fn forward_cached(&self, x0: &[f32; D], x1: &[f32; D]) -> (f32, ForwardCache) {
        let mut a1 = self.b1;
        let mut b1 = [0.0f32; H1];
        for k in 0..D {
            let xs = x0[k] + x1[k];
            if xs != 0.0 {
                let row = &self.w1_plus[k * H1..(k + 1) * H1];
                for h in 0..H1 {
                    a1[h] += xs * row[h];
                }
            }
            let xd = x0[k] - x1[k];
            if xd != 0.0 {
                let row = &self.w1_minus[k * H1..(k + 1) * H1];
                for h in 0..H1 {
                    b1[h] += xd * row[h];
                }
            }
        }
        let mut p0 = [0.0f32; H1];
        let mut p1 = [0.0f32; H1];
        let mut hs = [0.0f32; H1];
        let mut hd = [0.0f32; H1];
        let mut a2 = self.b2;
        let mut b2 = [0.0f32; H2];
        for h in 0..H1 {
            let v0 = a1[h] + b1[h];
            let v1 = a1[h] - b1[h];
            p0[h] = v0;
            p1[h] = v1;
            let h0 = if v0 > 0.0 { v0 } else { 0.0 };
            let h1 = if v1 > 0.0 { v1 } else { 0.0 };
            let s = h0 + h1;
            let d = h0 - h1;
            hs[h] = s;
            hd[h] = d;
            if s != 0.0 {
                let row = &self.w2_plus[h * H2..(h + 1) * H2];
                for j in 0..H2 {
                    a2[j] += s * row[j];
                }
            }
            if d != 0.0 {
                let row = &self.w2_minus[h * H2..(h + 1) * H2];
                for j in 0..H2 {
                    b2[j] += d * row[j];
                }
            }
        }
        let mut q0 = [0.0f32; H2];
        let mut q1 = [0.0f32; H2];
        let mut z = 0.0f32;
        for j in 0..H2 {
            let v0 = a2[j] + b2[j];
            let v1 = a2[j] - b2[j];
            q0[j] = v0;
            q1[j] = v1;
            let g0 = if v0 > 0.0 { v0 } else { 0.0 };
            let g1 = if v1 > 0.0 { v1 } else { 0.0 };
            z += self.w_out[j] * (g0 - g1);
        }
        (z, ForwardCache { p0, p1, hs, hd, q0, q1 })
    }

    /// Accumulate exact gradients into `grad` given upstream `dz = dL/dz`.
    #[inline]
    fn backward_accum(
        &self,
        x0: &[f32; D],
        x1: &[f32; D],
        cache: &ForwardCache,
        dz: f32,
        grad: &mut SiameseNet,
    ) {
        if dz == 0.0 {
            return;
        }
        let mut da2 = [0.0f32; H2];
        let mut db2 = [0.0f32; H2];
        for j in 0..H2 {
            let g0 = if cache.q0[j] > 0.0 { cache.q0[j] } else { 0.0 };
            let g1 = if cache.q1[j] > 0.0 { cache.q1[j] } else { 0.0 };
            grad.w_out[j] += dz * (g0 - g1);
            let dg0 = dz * self.w_out[j];
            let dg1 = -dg0;
            let dq0 = if cache.q0[j] > 0.0 { dg0 } else { 0.0 };
            let dq1 = if cache.q1[j] > 0.0 { dg1 } else { 0.0 };
            let a = dq0 + dq1;
            let b = dq0 - dq1;
            da2[j] = a;
            db2[j] = b;
            grad.b2[j] += a;
        }

        let mut da1 = [0.0f32; H1];
        let mut db1 = [0.0f32; H1];
        for h in 0..H1 {
            let s = cache.hs[h];
            let d = cache.hd[h];
            let w2p_row = &self.w2_plus[h * H2..(h + 1) * H2];
            let w2m_row = &self.w2_minus[h * H2..(h + 1) * H2];
            let g2p_row = &mut grad.w2_plus[h * H2..(h + 1) * H2];
            let g2m_row = &mut grad.w2_minus[h * H2..(h + 1) * H2];
            let mut dhs = 0.0f32;
            let mut dhd = 0.0f32;
            for j in 0..H2 {
                g2p_row[j] += s * da2[j];
                g2m_row[j] += d * db2[j];
                dhs += da2[j] * w2p_row[j];
                dhd += db2[j] * w2m_row[j];
            }
            let dh0 = dhs + dhd;
            let dh1 = dhs - dhd;
            let dp0 = if cache.p0[h] > 0.0 { dh0 } else { 0.0 };
            let dp1 = if cache.p1[h] > 0.0 { dh1 } else { 0.0 };
            let a = dp0 + dp1;
            let b = dp0 - dp1;
            da1[h] = a;
            db1[h] = b;
            grad.b1[h] += a;
        }

        for k in 0..D {
            let xs = x0[k] + x1[k];
            if xs != 0.0 {
                let row = &mut grad.w1_plus[k * H1..(k + 1) * H1];
                for h in 0..H1 {
                    row[h] += xs * da1[h];
                }
            }
            let xd = x0[k] - x1[k];
            if xd != 0.0 {
                let row = &mut grad.w1_minus[k * H1..(k + 1) * H1];
                for h in 0..H1 {
                    row[h] += xd * db1[h];
                }
            }
        }
    }
}

struct ForwardCache {
    p0: [f32; H1],
    p1: [f32; H1],
    hs: [f32; H1],
    hd: [f32; H1],
    q0: [f32; H2],
    q1: [f32; H2],
}

#[inline]
fn sigmoid_f64(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-20.0, 20.0)).exp())
}

#[inline]
fn bce_with_logit(z: f64, y: f64) -> f64 {
    let zc = z.clamp(-20.0, 20.0);
    let abs_z = zc.abs();
    zc.max(0.0) - zc * y + (-abs_z).exp().ln_1p()
}

struct LossCfg {
    w_abs: f64,
    w_cf: f64,
    w_t1: f64,
    w_zgame: f64,
}

#[inline]
fn pair_loss_and_grads(
    z: f32,
    zp: f32,
    rec: &CfPairRecV3,
    cfg: &LossCfg,
) -> (f64, f32, f32) {
    let y = f64::from(rec.v_cf).clamp(1e-5, 1.0 - 1e-5);
    let yp = f64::from(rec.vp_cf).clamp(1e-5, 1.0 - 1e-5);
    let zg = f64::from(rec.z_game).clamp(1e-5, 1.0 - 1e-5);
    let z64 = f64::from(z);
    let zp64 = f64::from(zp);
    let p = sigmoid_f64(z64);
    let pp = sigmoid_f64(zp64);
    let ws = if rec.turn == 1 { cfg.w_t1 } else { 1.0 };

    let l_abs = 0.5 * (bce_with_logit(z64, y) + bce_with_logit(zp64, yp));
    let err_d = (pp - p) - (yp - y);
    let l_cf = err_d * err_d;
    let l_zg = if cfg.w_zgame > 0.0 {
        0.5 * (bce_with_logit(z64, zg) + bce_with_logit(zp64, zg))
    } else {
        0.0
    };
    let loss = ws * (cfg.w_abs * l_abs + cfg.w_cf * l_cf + cfg.w_zgame * l_zg);

    let mut dz = ws * 0.5 * cfg.w_abs * (p - y);
    let mut dzp = ws * 0.5 * cfg.w_abs * (pp - yp);
    if cfg.w_cf > 0.0 {
        let coeff = ws * 2.0 * cfg.w_cf * err_d;
        dz -= coeff * p * (1.0 - p);
        dzp += coeff * pp * (1.0 - pp);
    }
    if cfg.w_zgame > 0.0 {
        dz += ws * 0.5 * cfg.w_zgame * (p - zg);
        dzp += ws * 0.5 * cfg.w_zgame * (pp - zg);
    }
    (loss, dz as f32, dzp as f32)
}

struct AdamW {
    m: SiameseNet,
    v: SiameseNet,
    step: u64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
}

impl AdamW {
    fn new(wd: f64) -> Self {
        Self {
            m: SiameseNet::zeros(),
            v: SiameseNet::zeros(),
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd,
        }
    }

    fn update_slice(
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f64,
        wd: f64,
        b1: f64,
        b2: f64,
        bc1: f64,
        bc2: f64,
        eps: f64,
    ) {
        for i in 0..params.len() {
            let g = f64::from(grads[i]);
            let mi = b1 * f64::from(m[i]) + (1.0 - b1) * g;
            let vi = b2 * f64::from(v[i]) + (1.0 - b2) * g * g;
            m[i] = mi as f32;
            v[i] = vi as f32;
            let m_hat = mi / bc1;
            let v_hat = vi / bc2;
            let mut p = f64::from(params[i]);
            if wd > 0.0 {
                p *= 1.0 - lr * wd;
            }
            p -= lr * m_hat / (v_hat.sqrt() + eps);
            params[i] = p as f32;
        }
    }

    fn step(&mut self, net: &mut SiameseNet, grad: &SiameseNet, lr: f64) {
        self.step += 1;
        let t = self.step as i32;
        let bc1 = 1.0 - self.beta1.powi(t);
        let bc2 = 1.0 - self.beta2.powi(t);
        Self::update_slice(
            &mut net.w1_plus, &grad.w1_plus, &mut self.m.w1_plus, &mut self.v.w1_plus,
            lr, self.wd, self.beta1, self.beta2, bc1, bc2, self.eps,
        );
        Self::update_slice(
            &mut net.w1_minus, &grad.w1_minus, &mut self.m.w1_minus, &mut self.v.w1_minus,
            lr, self.wd, self.beta1, self.beta2, bc1, bc2, self.eps,
        );
        Self::update_slice(
            &mut net.b1, &grad.b1, &mut self.m.b1, &mut self.v.b1,
            lr, 0.0, self.beta1, self.beta2, bc1, bc2, self.eps,
        );
        Self::update_slice(
            &mut net.w2_plus, &grad.w2_plus, &mut self.m.w2_plus, &mut self.v.w2_plus,
            lr, self.wd, self.beta1, self.beta2, bc1, bc2, self.eps,
        );
        Self::update_slice(
            &mut net.w2_minus, &grad.w2_minus, &mut self.m.w2_minus, &mut self.v.w2_minus,
            lr, self.wd, self.beta1, self.beta2, bc1, bc2, self.eps,
        );
        Self::update_slice(
            &mut net.b2, &grad.b2, &mut self.m.b2, &mut self.v.b2,
            lr, 0.0, self.beta1, self.beta2, bc1, bc2, self.eps,
        );
        Self::update_slice(
            &mut net.w_out, &grad.w_out, &mut self.m.w_out, &mut self.v.w_out,
            lr, self.wd, self.beta1, self.beta2, bc1, bc2, self.eps,
        );
    }
}

#[derive(Default, Clone)]
struct EvalMetrics {
    n: usize,
    loss: f64,
    r_val: f64,
    r_delta: f64,
    rmse: f64,
    delta_rmse: f64,
    sign_acc: f64,
    sign_n: usize,
}

impl EvalMetrics {
    fn to_json(&self) -> Value {
        let r4 = |x: f64| if x.is_finite() { (x * 10_000.0).round() / 10_000.0 } else { 0.0 };
        json!({
            "n": self.n,
            "loss": r4(self.loss),
            "r_val": r4(self.r_val),
            "r_delta": r4(self.r_delta),
            "rmse": r4(self.rmse),
            "delta_rmse": r4(self.delta_rmse),
            "sign_acc": r4(self.sign_acc),
            "sign_n": self.sign_n,
        })
    }
}

fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let mx = xs.iter().sum::<f64>() / (n as f64);
    let my = ys.iter().sum::<f64>() / (n as f64);
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom > 1e-12 { cov / denom } else { 0.0 }
}

fn evaluate_split(net: &SiameseNet, recs: &[&CfPairRecV3], cfg: &LossCfg) -> EvalMetrics {
    if recs.is_empty() {
        return EvalMetrics::default();
    }
    let mut loss_sum = 0.0;
    let mut sq_v = 0.0;
    let mut sq_d = 0.0;
    let mut pred_v = Vec::with_capacity(recs.len() * 2);
    let mut true_v = Vec::with_capacity(recs.len() * 2);
    let mut pred_d = Vec::with_capacity(recs.len());
    let mut true_d = Vec::with_capacity(recs.len());
    let mut sign_ok = 0usize;
    let mut sign_n = 0usize;

    for &rec in recs {
        let z = net.forward_logit(&rec.x0, &rec.x1);
        let zp = net.forward_logit(&rec.xp0, &rec.xp1);
        let (l, _, _) = pair_loss_and_grads(z, zp, rec, cfg);
        loss_sum += l;
        let p = sigmoid_f64(f64::from(z));
        let pp = sigmoid_f64(f64::from(zp));
        let y = f64::from(rec.v_cf);
        let yp = f64::from(rec.vp_cf);
        pred_v.push(p);
        true_v.push(y);
        pred_v.push(pp);
        true_v.push(yp);
        sq_v += (p - y).powi(2) + (pp - yp).powi(2);
        let dp = pp - p;
        let dy = yp - y;
        pred_d.push(dp);
        true_d.push(dy);
        sq_d += (dp - dy).powi(2);
        if dy.abs() >= 0.03 {
            sign_n += 1;
            if (dp >= 0.0) == (dy >= 0.0) {
                sign_ok += 1;
            }
        }
    }
    let nf = recs.len() as f64;
    EvalMetrics {
        n: recs.len(),
        loss: loss_sum / nf,
        r_val: pearson_r(&pred_v, &true_v),
        r_delta: pearson_r(&pred_d, &true_d),
        rmse: (sq_v / (2.0 * nf)).sqrt(),
        delta_rmse: (sq_d / nf).sqrt(),
        sign_acc: if sign_n > 0 { sign_ok as f64 / sign_n as f64 } else { 0.0 },
        sign_n,
    }
}

fn load_records(paths: &[PathBuf]) -> Vec<CfPairRecV3> {
    let rec_size = std::mem::size_of::<CfPairRecV3>();
    let mut out = Vec::new();
    for p in paths {
        let mut f = File::open(p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).unwrap();
        assert_eq!(
            bytes.len() % rec_size,
            0,
            "file {} byte len {} is not a multiple of CfPairRecV3 size {}",
            p.display(),
            bytes.len(),
            rec_size
        );
        let count = bytes.len() / rec_size;
        out.reserve(count);
        for i in 0..count {
            let mut rec = std::mem::MaybeUninit::<CfPairRecV3>::uninit();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(i * rec_size),
                    rec.as_mut_ptr() as *mut u8,
                    rec_size,
                );
                out.push(rec.assume_init());
            }
        }
    }
    out
}

fn forward_logit_f64(
    w1p: &[f64],
    w1m: &[f64],
    b1: &[f64; H1],
    w2p: &[f64],
    w2m: &[f64],
    b2: &[f64; H2],
    wout: &[f64; H2],
    x0: &[f32; D],
    x1: &[f32; D],
) -> f64 {
    let mut a1 = *b1;
    let mut bv1 = [0.0f64; H1];
    for k in 0..D {
        let xs = f64::from(x0[k]) + f64::from(x1[k]);
        if xs != 0.0 {
            for h in 0..H1 {
                a1[h] += xs * w1p[k * H1 + h];
            }
        }
        let xd = f64::from(x0[k]) - f64::from(x1[k]);
        if xd != 0.0 {
            for h in 0..H1 {
                bv1[h] += xd * w1m[k * H1 + h];
            }
        }
    }
    let mut a2 = *b2;
    let mut bv2 = [0.0f64; H2];
    for h in 0..H1 {
        let h0 = (a1[h] + bv1[h]).max(0.0);
        let h1 = (a1[h] - bv1[h]).max(0.0);
        let hs = h0 + h1;
        let hd = h0 - h1;
        for j in 0..H2 {
            a2[j] += hs * w2p[h * H2 + j];
            bv2[j] += hd * w2m[h * H2 + j];
        }
    }
    let mut z = 0.0f64;
    for j in 0..H2 {
        z += wout[j] * ((a2[j] + bv2[j]).max(0.0) - (a2[j] - bv2[j]).max(0.0));
    }
    z
}

fn finite_diff_grad_check(rec: &CfPairRecV3, cfg: &LossCfg) -> f64 {
    let net = SiameseNet::kaiming_init(1234567);
    let mut grad = SiameseNet::zeros();
    let (z, c) = net.forward_cached(&rec.x0, &rec.x1);
    let (zp, cp) = net.forward_cached(&rec.xp0, &rec.xp1);
    let (_, dz, dzp) = pair_loss_and_grads(z, zp, rec, cfg);
    net.backward_accum(&rec.x0, &rec.x1, &c, dz, &mut grad);
    net.backward_accum(&rec.xp0, &rec.xp1, &cp, dzp, &mut grad);

    let mut w1p: Vec<f64> = net.w1_plus.iter().map(|&v| f64::from(v)).collect();
    let mut w1m: Vec<f64> = net.w1_minus.iter().map(|&v| f64::from(v)).collect();
    let mut b1 = [0.0f64; H1];
    for i in 0..H1 {
        b1[i] = f64::from(net.b1[i]);
    }
    let mut w2p: Vec<f64> = net.w2_plus.iter().map(|&v| f64::from(v)).collect();
    let mut w2m: Vec<f64> = net.w2_minus.iter().map(|&v| f64::from(v)).collect();
    let mut b2 = [0.0f64; H2];
    let mut wout = [0.0f64; H2];
    for i in 0..H2 {
        b2[i] = f64::from(net.b2[i]);
        wout[i] = f64::from(net.w_out[i]);
    }

    let loss_f64 = |w1p: &[f64], w1m: &[f64], b1: &[f64; H1], w2p: &[f64], w2m: &[f64], b2: &[f64; H2], wout: &[f64; H2]| -> f64 {
        let z0 = forward_logit_f64(w1p, w1m, b1, w2p, w2m, b2, wout, &rec.x0, &rec.x1);
        let z1 = forward_logit_f64(w1p, w1m, b1, w2p, w2m, b2, wout, &rec.xp0, &rec.xp1);
        let y = f64::from(rec.v_cf).clamp(1e-5, 1.0 - 1e-5);
        let yp = f64::from(rec.vp_cf).clamp(1e-5, 1.0 - 1e-5);
        let zg = f64::from(rec.z_game).clamp(1e-5, 1.0 - 1e-5);
        let p = sigmoid_f64(z0);
        let pp = sigmoid_f64(z1);
        let ws = if rec.turn == 1 { cfg.w_t1 } else { 1.0 };
        let l_abs = 0.5 * (bce_with_logit(z0, y) + bce_with_logit(z1, yp));
        let err_d = (pp - p) - (yp - y);
        let l_zg = if cfg.w_zgame > 0.0 {
            0.5 * (bce_with_logit(z0, zg) + bce_with_logit(z1, zg))
        } else {
            0.0
        };
        ws * (cfg.w_abs * l_abs + cfg.w_cf * err_d * err_d + cfg.w_zgame * l_zg)
    };

    let eps = 1e-6f64;
    let mut max_rel = 0.0f64;
    macro_rules! check_fd {
        ($place:expr, $ana:expr) => {{
            let orig = $place;
            $place = orig + eps;
            let lp = loss_f64(&w1p, &w1m, &b1, &w2p, &w2m, &b2, &wout);
            $place = orig - eps;
            let lm = loss_f64(&w1p, &w1m, &b1, &w2p, &w2m, &b2, &wout);
            $place = orig;
            let num = (lp - lm) / (2.0 * eps);
            let a = f64::from($ana);
            let err = (num - a).abs() / (num.abs() + a.abs() + 1e-4);
            if err > max_rel {
                max_rel = err;
            }
        }};
    }

    let mut rng = SplitMix64::new(98765);
    for _ in 0..8 {
        let i = rng.below(w1p.len());
        check_fd!(w1p[i], grad.w1_plus[i]);
        check_fd!(w1m[i], grad.w1_minus[i]);
    }
    for i in 0..8 {
        check_fd!(b1[i], grad.b1[i]);
        let j = rng.below(w2p.len());
        check_fd!(w2p[j], grad.w2_plus[j]);
        check_fd!(w2m[j], grad.w2_minus[j]);
        check_fd!(b2[i], grad.b2[i]);
        check_fd!(wout[i], grad.w_out[i]);
    }
    max_rel
}

fn fmt_f32(v: f32) -> String {
    let v = if v == 0.0 { 0.0 } else { v }; // canonicalize -0.0
    let mut s = format!("{v}");
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    s
}

fn write_matrix<W: Write>(w: &mut W, rows: usize, cols: usize, data: &[f32]) {
    for r in 0..rows {
        write!(w, "    [").unwrap();
        for c in 0..cols {
            if c > 0 {
                write!(w, ", ").unwrap();
            }
            write!(w, "{}", fmt_f32(data[r * cols + c])).unwrap();
        }
        writeln!(w, "],").unwrap();
    }
}

fn write_vec<W: Write>(w: &mut W, data: &[f32]) {
    write!(w, "[").unwrap();
    for (i, &v) in data.iter().enumerate() {
        if i > 0 {
            write!(w, ", ").unwrap();
        }
        write!(w, "{}", fmt_f32(v)).unwrap();
    }
    write!(w, "]").unwrap();
}

/// Export `net` to Rust `compact_cf_weights.rs` format while preserving `CF_P_*` weights verbatim!
fn export_rust_weights(net: &SiameseNet, src_cf_weights: &Path, dst_path: &Path) -> f64 {
    let existing = fs::read_to_string(src_cf_weights)
        .unwrap_or_else(|e| panic!("read {}: {e}", src_cf_weights.display()));
    let marker = "// Auto-generated `CF-P_net`";
    let policy_tail = if let Some(pos) = existing.find(marker) {
        &existing[pos..]
    } else if let Some(pos) = existing.find("pub const CF_POLICY_WEIGHTS_LOADED") {
        &existing[pos..]
    } else {
        panic!("could not locate CF_P_* section in {}", src_cf_weights.display());
    };

    if let Some(parent) = dst_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).unwrap();
        }
    }
    let mut out = BufWriter::new(File::create(dst_path).unwrap());
    writeln!(out, "//! Auto-generated Direct Raw-Board Siamese `CF-V_net v3` weights (192-dim/side,").unwrap();
    writeln!(out, "//! trained on zero-leakage CRN Midgame + Turn-1 Compositional Pairs from").unwrap();
    writeln!(out, "//! `belief-pool-v3` \\ `team-pool-v4`, `hash % 5 != 0`).").unwrap();
    writeln!(out, "//! Column-major `[IN][OUT]` Siamese `(W_PLUS, W_MINUS)` layout for exact 0-ULP antisymmetry.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub const CF_VALUE_WEIGHTS_LOADED: bool = true;").unwrap();
    writeln!(out, "pub const CF_VALUE_SIDE_DIM: usize = {D};").unwrap();
    writeln!(out, "pub const CF_VALUE_H1: usize = {H1};").unwrap();
    writeln!(out, "pub const CF_VALUE_H2: usize = {H2};").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "pub static CF_V_W1_PLUS_T: [[f32; CF_VALUE_H1]; CF_VALUE_SIDE_DIM] = [").unwrap();
    write_matrix(&mut out, D, H1, &net.w1_plus);
    writeln!(out, "];").unwrap();

    writeln!(out, "pub static CF_V_W1_MINUS_T: [[f32; CF_VALUE_H1]; CF_VALUE_SIDE_DIM] = [").unwrap();
    write_matrix(&mut out, D, H1, &net.w1_minus);
    writeln!(out, "];").unwrap();

    write!(out, "pub static CF_V_B1: [f32; CF_VALUE_H1] = ").unwrap();
    write_vec(&mut out, &net.b1);
    writeln!(out, ";").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "pub static CF_V_W2_PLUS_T: [[f32; CF_VALUE_H2]; CF_VALUE_H1] = [").unwrap();
    write_matrix(&mut out, H1, H2, &net.w2_plus);
    writeln!(out, "];").unwrap();

    writeln!(out, "pub static CF_V_W2_MINUS_T: [[f32; CF_VALUE_H2]; CF_VALUE_H1] = [").unwrap();
    write_matrix(&mut out, H1, H2, &net.w2_minus);
    writeln!(out, "];").unwrap();

    write!(out, "pub static CF_V_B2: [f32; CF_VALUE_H2] = ").unwrap();
    write_vec(&mut out, &net.b2);
    writeln!(out, ";").unwrap();
    writeln!(out).unwrap();

    write!(out, "pub static CF_V_W_OUT: [f32; CF_VALUE_H2] = ").unwrap();
    write_vec(&mut out, &net.w_out);
    writeln!(out, ";").unwrap();
    writeln!(out).unwrap();

    out.write_all(policy_tail.as_bytes()).unwrap();
    out.flush().unwrap();

    // Verify exact string round-trip parity of exported f32 literals!
    let mut max_rt_diff = 0.0f64;
    for &v in net
        .w1_plus
        .iter()
        .chain(&net.w1_minus)
        .chain(&net.b1)
        .chain(&net.w2_plus)
        .chain(&net.w2_minus)
        .chain(&net.b2)
        .chain(&net.w_out)
    {
        let parsed: f32 = fmt_f32(v).parse().expect("exported f32 must parse cleanly");
        let diff = (f64::from(parsed) - f64::from(v)).abs();
        if diff > max_rt_diff {
            max_rt_diff = diff;
        }
    }
    max_rt_diff
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let has_flag = |name: &str| args.iter().any(|a| a == name);

    let mut data_paths: Vec<PathBuf> = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == "--data" {
            let mut j = i + 1;
            while j < args.len() && !args[j].starts_with("--") {
                data_paths.push(PathBuf::from(&args[j]));
                j += 1;
            }
            i = j;
        } else if args[i] == "--data-dir" {
            if let Some(dir) = args.get(i + 1) {
                let mut entries: Vec<PathBuf> = fs::read_dir(dir)
                    .unwrap_or_else(|e| panic!("read_dir {dir}: {e}"))
                    .filter_map(|e| e.ok().map(|x| x.path()))
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("bin"))
                    .collect();
                entries.sort();
                data_paths.extend(entries);
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    assert!(
        !data_paths.is_empty(),
        "usage: train_cf_v3 --data <file1.bin> [file2.bin ...] | --data-dir <dir> [--epochs 40] [--export-rs PATH]"
    );

    let epochs: usize = flag("--epochs").map_or(40, |v| v.parse().unwrap());
    let batch_size: usize = flag("--batch-size").map_or(512, |v| v.parse().unwrap());
    let lr_max: f64 = flag("--lr").map_or(2e-3, |v| v.parse().unwrap());
    let lr_min: f64 = flag("--min-lr").map_or(5e-5, |v| v.parse().unwrap());
    let warmup_epochs: usize = flag("--warmup").map_or(2, |v| v.parse().unwrap());
    let wd: f64 = flag("--wd").map_or(1e-4, |v| v.parse().unwrap());
    let grad_clip: f64 = flag("--grad-clip").map_or(2.0, |v| v.parse().unwrap());
    let w_abs: f64 = flag("--w-abs").map_or(1.0, |v| v.parse().unwrap());
    let w_cf: f64 = flag("--w-cf").map_or(4.0, |v| v.parse().unwrap());
    let w_t1: f64 = flag("--w-t1").map_or(1.5, |v| v.parse().unwrap());
    let w_zgame: f64 = flag("--w-zgame").map_or(0.0, |v| v.parse().unwrap());
    let threads: usize = flag("--threads").map_or(10, |v| v.parse().unwrap()).max(1);
    let seed: u64 = flag("--seed").map_or(2026120142, |v| v.parse().unwrap());
    let filter_blind = flag("--filter-blind").unwrap_or_else(|| "open".to_string());
    let init_from_v2 = has_flag("--init-from-v2");
    let export_rs = flag("--export-rs").map(PathBuf::from);
    let cf_weights_src = flag("--cf-weights-src")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("crates/bot/src/compact_cf_weights.rs"));
    let log_path = flag("--log").map(PathBuf::from);

    let cfg = LossCfg { w_abs, w_cf, w_t1, w_zgame };
    let t0 = Instant::now();

    let raw = load_records(&data_paths);
    let mut train_idx: Vec<usize> = Vec::new();
    let mut val_mid_idx: Vec<usize> = Vec::new();
    let mut val_t1_idx: Vec<usize> = Vec::new();
    let mut val_t1_bench_idx: Vec<usize> = Vec::new();
    let mut val_t1_lead_idx: Vec<usize> = Vec::new();
    let mut train_t1_cnt = 0usize;
    let mut train_mid_cnt = 0usize;
    let mut mixed_skipped = 0usize;
    let mut blind_skipped = 0usize;

    for (idx, r) in raw.iter().enumerate() {
        let keep_blind = match filter_blind.as_str() {
            "open" => r.is_blind == 0,
            "blind" => r.is_blind != 0,
            "all" => true,
            other => panic!("unknown --filter-blind {other} (expected open|blind|all)"),
        };
        if !keep_blind {
            blind_skipped += 1;
            continue;
        }
        let h0_val = (r.team_hash_0 % 5) == 0;
        let h1_val = (r.team_hash_1 % 5) == 0;
        if !h0_val && !h1_val {
            train_idx.push(idx);
            if r.turn == 1 {
                train_t1_cnt += 1;
            } else {
                train_mid_cnt += 1;
            }
        } else if h0_val && h1_val {
            if r.turn == 1 {
                val_t1_idx.push(idx);
                if r.kind == 10 {
                    val_t1_bench_idx.push(idx);
                } else if r.kind == 11 {
                    val_t1_lead_idx.push(idx);
                }
            } else {
                val_mid_idx.push(idx);
            }
        } else {
            mixed_skipped += 1;
        }
    }

    assert!(!train_idx.is_empty(), "no training rows after filtering!");

    // Exact finite-difference backprop verification on the first training record (`f64`, eps=1e-6).
    let grad_check_err = finite_diff_grad_check(&raw[train_idx[0]], &cfg);
    assert!(
        grad_check_err < 1e-4,
        "backprop finite-diff check failed: max_rel_err={grad_check_err}"
    );

    // Exact 0-ULP forward parity check: `SiameseNet::from_embedded` vs `compact_az::eval_cf_value_logit_from_features`!
    let emb_net = SiameseNet::from_embedded();
    let mut embedded_parity_diff = 0.0f64;
    for r in raw.iter().take(32) {
        let z_tr = emb_net.forward_logit(&r.x0, &r.x1);
        let z_az = eval_cf_value_logit_from_features(&r.x0, &r.x1);
        let d = (f64::from(z_tr) - f64::from(z_az)).abs();
        if d > embedded_parity_diff {
            embedded_parity_diff = d;
        }
    }
    assert!(
        embedded_parity_diff < 1e-6,
        "SiameseNet::forward_logit mismatch vs compact_az::eval_cf_value_logit_from_features: {embedded_parity_diff}"
    );

    // Also verify embedded-weights forward parity if initialized from v2.
    let mut net = if init_from_v2 {
        emb_net
    } else {
        SiameseNet::kaiming_init(seed)
    };
    let mut opt = AdamW::new(wd);
    let mut rng = SplitMix64::new(seed ^ 0x9E37_79B9_7F4A_7C15);

    let val_mid_refs: Vec<&CfPairRecV3> = val_mid_idx.iter().map(|&i| &raw[i]).collect();
    let val_t1_refs: Vec<&CfPairRecV3> = val_t1_idx.iter().map(|&i| &raw[i]).collect();
    let val_t1_bench_refs: Vec<&CfPairRecV3> = val_t1_bench_idx.iter().map(|&i| &raw[i]).collect();
    let val_t1_lead_refs: Vec<&CfPairRecV3> = val_t1_lead_idx.iter().map(|&i| &raw[i]).collect();

    let mut log_writer = log_path.as_ref().map(|p| {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).unwrap();
            }
        }
        BufWriter::new(File::create(p).unwrap())
    });

    let mut best_net = net.clone();
    let mut best_score = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut best_mid = EvalMetrics::default();
    let mut best_t1 = EvalMetrics::default();
    let mut last_train_loss = 0.0f64;
    let mut any_nan = false;

    for ep in 1..=epochs {
        // Cosine LR with linear warmup.
        let lr = if ep <= warmup_epochs && warmup_epochs > 0 {
            lr_max * (ep as f64 / warmup_epochs as f64)
        } else {
            let denom = epochs.saturating_sub(warmup_epochs).max(1) as f64;
            let prog = ((ep - warmup_epochs) as f64 / denom).clamp(0.0, 1.0);
            lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * prog).cos())
        };

        // Fisher-Yates shuffle of training indices.
        for k in (1..train_idx.len()).rev() {
            let j = rng.below(k + 1);
            train_idx.swap(k, j);
        }

        let mut ep_loss = 0.0f64;
        let mut ep_n = 0usize;

        for batch in train_idx.chunks(batch_size) {
            let bn = batch.len();
            let eff_threads = threads.min(bn.div_ceil(16)).max(1);
            let chunk_sz = bn.div_ceil(eff_threads);

            let (mut grad_sum, batch_loss) = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(eff_threads);
                let net_r = &net;
                let raw_r = &raw;
                let cfg_r = &cfg;
                for sub in batch.chunks(chunk_sz) {
                    handles.push(s.spawn(move || {
                        let mut g = SiameseNet::zeros();
                        let mut lsum = 0.0f64;
                        for &ridx in sub {
                            let rec = &raw_r[ridx];
                            let (z, c) = net_r.forward_cached(&rec.x0, &rec.x1);
                            let (zp, cp) = net_r.forward_cached(&rec.xp0, &rec.xp1);
                            let (l, dz, dzp) = pair_loss_and_grads(z, zp, rec, cfg_r);
                            lsum += l;
                            net_r.backward_accum(&rec.x0, &rec.x1, &c, dz, &mut g);
                            net_r.backward_accum(&rec.xp0, &rec.xp1, &cp, dzp, &mut g);
                        }
                        (g, lsum)
                    }));
                }
                let mut acc_g = SiameseNet::zeros();
                let mut acc_l = 0.0f64;
                for h in handles {
                    let (g, l) = h.join().unwrap();
                    acc_g.add_assign(&g);
                    acc_l += l;
                }
                (acc_g, acc_l)
            });

            ep_loss += batch_loss;
            ep_n += bn;
            grad_sum.scale(1.0 / (bn as f32));
            let gnorm = grad_sum.l2_norm();
            if !gnorm.is_finite() {
                any_nan = true;
                break;
            }
            if grad_clip > 0.0 && gnorm > grad_clip {
                grad_sum.scale((grad_clip / gnorm) as f32);
            }
            opt.step(&mut net, &grad_sum, lr);
        }

        if any_nan {
            break;
        }
        last_train_loss = ep_loss / (ep_n.max(1) as f64);

        let m_mid = evaluate_split(&net, &val_mid_refs, &cfg);
        let m_t1 = evaluate_split(&net, &val_t1_refs, &cfg);
        let comb_val = match (m_mid.n > 0, m_t1.n > 0) {
            (true, true) => 0.5 * m_mid.loss + 0.5 * m_t1.loss,
            (true, false) => m_mid.loss,
            (false, true) => m_t1.loss,
            (false, false) => last_train_loss,
        };
        if comb_val < best_score {
            best_score = comb_val;
            best_epoch = ep;
            best_net = net.clone();
            best_mid = m_mid.clone();
            best_t1 = m_t1.clone();
        }

        if let Some(lw) = log_writer.as_mut() {
            writeln!(
                lw,
                "{}",
                json!({
                    "epoch": ep,
                    "lr": lr,
                    "train_loss": last_train_loss,
                    "val_mid": m_mid.to_json(),
                    "val_t1": m_t1.to_json(),
                })
            )
            .unwrap();
        }
    }

    if let Some(mut lw) = log_writer {
        lw.flush().unwrap();
    }

    // Verify exact 0-ULP antisymmetry on best_net across first 16 records.
    let mut max_asym = 0.0f32;
    for r in raw.iter().take(16) {
        let z_ab = best_net.forward_logit(&r.x0, &r.x1);
        let z_ba = best_net.forward_logit(&r.x1, &r.x0);
        let d = (z_ab + z_ba).abs();
        if d > max_asym {
            max_asym = d;
        }
    }

    let best_t1_bench = evaluate_split(&best_net, &val_t1_bench_refs, &cfg);
    let best_t1_lead = evaluate_split(&best_net, &val_t1_lead_refs, &cfg);

    let export_rt_diff = export_rs
        .as_ref()
        .map(|dst| export_rust_weights(&best_net, &cf_weights_src, dst));

    println!(
        "{}",
        json!({
            "cmd": "train_cf_v3",
            "side_dim": D,
            "files": data_paths.len(),
            "total_rows": raw.len(),
            "train_rows": train_idx.len(),
            "train_mid_rows": train_mid_cnt,
            "train_t1_rows": train_t1_cnt,
            "val_mid_rows": val_mid_idx.len(),
            "val_t1_rows": val_t1_idx.len(),
            "mixed_skipped": mixed_skipped,
            "blind_skipped": blind_skipped,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "any_nan": any_nan,
            "grad_check_rel_err": (grad_check_err * 1e6).round() / 1e6,
            "embedded_parity_diff": embedded_parity_diff,
            "max_antisym_err": max_asym,
            "final_train_loss": (last_train_loss * 10_000.0).round() / 10_000.0,
            "best_val_mid": best_mid.to_json(),
            "best_val_t1": best_t1.to_json(),
            "best_val_t1_bench": best_t1_bench.to_json(),
            "best_val_t1_lead": best_t1_lead.to_json(),
            "export_rs": export_rs.as_ref().map(|p| p.display().to_string()),
            "export_max_rt_diff": export_rt_diff,
            "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        })
    );
}
