//! Learned state value for the game-solving program (docs/GAME-SOLVING-PLAN.md).
//!
//! `ValueCodec` turns a full-information `Battle` into one fixed-length f32
//! record; `ValueNet` maps a record to side 0's win probability. The record
//! layout and the network's forward pass are mirrored by
//! `tools/fvi-train.py` — change both together and bump `VALUE_SCHEMA`.
//!
//! The network is side-antisymmetric by construction: `logit = h(0) − h(1)`,
//! where `h(s)` reads side s's encoding first and the foe's second, so
//! swapping the sides negates the logit exactly.
//!
//! `sampled_backup` is the Shapley backup the program trains on: the
//! matrix-game value of K sampled successors per joint action.

use std::collections::HashMap;

use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::{Accuracy, Category, CondId, Dex, MoveId};
use nc2000_engine::state::{Battle, EffectState, PokeId, Pokemon, Status, DK};
use serde::{Deserialize, Serialize};

use crate::eval::{best_hit_fraction, eval01, EvalWeights};
use crate::exact::solve_matrix_full;
use crate::rng::SplitMix64;

pub const VALUE_SCHEMA: &str = "nc2000-value-v1";

pub const N_TYPES: usize = 18;
pub const MONS: usize = 3;
pub const MOVES: usize = 4;
/// Per-mon ids: species+1, item+1 (0 = none), four moves+1 (0 = empty).
pub const IDS: usize = 6;
pub const F_MON: usize = 60;
pub const F_MDYN: usize = 4;
pub const F_MSTAT: usize = 30;

/// Volatiles the active mon's block encodes. Anything else lands in the
/// trailing "other" slot; `tests/value.rs` fails if self-play ever reaches
/// one, so the list stays complete for this format.
pub const VOLATILE_KEYS: [&str; 40] = [
    "substitute",
    "confusion",
    "leechseed",
    "curse",
    "perishsong",
    "encore",
    "disable",
    "attract",
    "focusenergy",
    "foresight",
    "lockon",
    "nightmare",
    "destinybond",
    "protect",
    "endure",
    "stall",
    "defensecurl",
    "minimize",
    "rage",
    "bide",
    "rollout",
    "iceball",
    "furycutter",
    "lockedmove",
    "twoturnmove",
    "mustrecharge",
    "trapped",
    "trapper",
    "partiallytrapped",
    "residualdmg",
    "parspeeddrop",
    "brnattackdrop",
    "flinch",
    "pursuit",
    "solarbeam",
    "skyattack",
    "razorwind",
    "skullbash",
    "fly",
    "dig",
];
pub const F_VOL_EACH: usize = 4;
/// Active-mon block: seven stat stages, then one entry per volatile key plus
/// the trailing "other" entry.
pub const F_VOL: usize = 7 + (VOLATILE_KEYS.len() + 1) * F_VOL_EACH;
pub const SIDE_CONDS: [&str; 5] = ["spikes", "reflect", "lightscreen", "safeguard", "mist"];
pub const F_SIDE: usize = 5 + SIDE_CONDS.len() * 2 + 2;
pub const F_PAIR: usize = MONS * MONS * 3;
pub const WEATHERS: [&str; 3] = ["raindance", "sunnyday", "sandstorm"];
pub const F_GLOBAL: usize = WEATHERS.len() + 2;

/// Record layout: offsets of each block in one record.
pub const OFF_IDS: usize = 0;
pub const OFF_MON: usize = OFF_IDS + 2 * MONS * IDS;
pub const OFF_MDYN: usize = OFF_MON + 2 * MONS * F_MON;
pub const OFF_VOL: usize = OFF_MDYN + 2 * MONS * MOVES * F_MDYN;
pub const OFF_SIDE: usize = OFF_VOL + 2 * F_VOL;
pub const OFF_PAIR: usize = OFF_SIDE + 2 * F_SIDE;
pub const OFF_GLOBAL: usize = OFF_PAIR + 2 * F_PAIR;
pub const RECORD: usize = OFF_GLOBAL + F_GLOBAL;

/// Layout and fixed tables the trainer needs (`fvi_gen --describe`).
#[derive(Serialize, Deserialize)]
pub struct CodecDescription {
    pub schema: String,
    pub n_species: usize,
    pub n_items: usize,
    pub n_moves: usize,
    pub record: usize,
    pub offsets: HashMap<String, usize>,
    pub dims: HashMap<String, usize>,
    /// Row 0 = empty slot; row id+1 = move id.
    pub move_static: Vec<Vec<f32>>,
}

pub struct ValueCodec {
    vol_ids: Vec<Option<CondId>>,
    side_ids: Vec<Option<CondId>>,
    weather_ids: Vec<Option<CondId>>,
    futuremove: Option<CondId>,
    residualdmg: Option<CondId>,
    move_static: Vec<[f32; F_MSTAT]>,
    n_species: usize,
    n_items: usize,
}

const STAGE: [f64; 13] = [
    2.0 / 8.0,
    2.0 / 7.0,
    2.0 / 6.0,
    2.0 / 5.0,
    2.0 / 4.0,
    2.0 / 3.0,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
];

fn effective_speed(p: &Pokemon, active: bool) -> f64 {
    let mut s = p.stored_stats[4] as f64;
    if active {
        s *= STAGE[(p.boosts[4] as i32 + 6).clamp(0, 12) as usize];
    }
    if p.status == Status::Par {
        s *= 0.25;
    }
    s
}

impl ValueCodec {
    pub fn new(dex: &Dex) -> Self {
        let ids = |keys: &[&str]| keys.iter().map(|k| dex.conds_id(k)).collect::<Vec<_>>();
        let n_moves = dex.moves.len();
        let mut move_static = vec![[0f32; F_MSTAT]; n_moves + 1];
        for (i, row) in move_static.iter_mut().enumerate().skip(1) {
            let ms = dex.move_static(MoveId((i - 1) as u16));
            row[0] = 1.0;
            row[1] = ms.base_power as f32 / 150.0;
            row[2] = match ms.accuracy {
                Accuracy::AlwaysHits => 1.0,
                Accuracy::Pct(p) => p as f32 / 100.0,
            };
            row[3] = f32::from(ms.accuracy == Accuracy::AlwaysHits);
            match ms.category {
                Category::Physical => row[4] = 1.0,
                Category::Special => row[5] = 1.0,
                Category::Status => row[6] = 1.0,
            }
            row[7] = ms.priority as f32 / 3.0;
            row[8] = f32::from(ms.heal.is_some());
            row[9] = f32::from(ms.drain.is_some());
            row[10] = f32::from(ms.recoil.is_some());
            row[11] = ms.pp as f32 / 40.0;
            let t = ms.move_type.0 as usize;
            assert!(t < N_TYPES, "move type index {t} out of range");
            row[12 + t] = 1.0;
        }
        ValueCodec {
            vol_ids: ids(&VOLATILE_KEYS),
            side_ids: ids(&SIDE_CONDS),
            weather_ids: ids(&WEATHERS),
            futuremove: dex.conds_id("futuremove"),
            residualdmg: dex.conds_id("residualdmg"),
            move_static,
            n_species: dex.species.len(),
            n_items: dex.items.len(),
        }
    }

    pub fn describe(&self) -> CodecDescription {
        let offsets = [
            ("ids", OFF_IDS),
            ("mon", OFF_MON),
            ("mdyn", OFF_MDYN),
            ("vol", OFF_VOL),
            ("side", OFF_SIDE),
            ("pair", OFF_PAIR),
            ("global", OFF_GLOBAL),
        ];
        let dims = [
            ("mons", MONS),
            ("moves", MOVES),
            ("ids", IDS),
            ("mon", F_MON),
            ("mdyn", F_MDYN),
            ("mstat", F_MSTAT),
            ("vol", F_VOL),
            ("side", F_SIDE),
            ("pair", F_PAIR),
            ("global", F_GLOBAL),
        ];
        CodecDescription {
            schema: VALUE_SCHEMA.into(),
            n_species: self.n_species,
            n_items: self.n_items,
            n_moves: self.move_static.len() - 1,
            record: RECORD,
            offsets: offsets.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            dims: dims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            move_static: self.move_static.iter().map(|r| r.to_vec()).collect(),
        }
    }

    pub fn move_static(&self, id: usize) -> &[f32; F_MSTAT] {
        &self.move_static[id]
    }

    /// The side's three mons in encoding order: the active slot first, then
    /// the rest of the party in party order.
    fn order(b: &Battle, s: usize) -> [Option<u8>; MONS] {
        let side = &b.sides[s];
        let mut out = [None; MONS];
        let mut n = 0;
        if let Some(a) = side.active {
            out[0] = Some(a);
            n = 1;
        }
        for &slot in side.party.iter() {
            if Some(slot) == side.active || n >= MONS {
                continue;
            }
            out[n] = Some(slot);
            n += 1;
        }
        out
    }

    /// Append one record for `b` to `out`.
    pub fn encode(&self, b: &Battle, dex: &Dex, out: &mut Vec<f32>) {
        let base = out.len();
        out.resize(base + RECORD, 0.0);
        let r = &mut out[base..];
        let order = [Self::order(b, 0), Self::order(b, 1)];
        for s in 0..2 {
            for (m, slot) in order[s].iter().enumerate() {
                let Some(slot) = *slot else { continue };
                let p = &b.sides[s].roster[slot as usize];
                let active = m == 0 && b.sides[s].active == Some(slot);
                let ids = OFF_IDS + (s * MONS + m) * IDS;
                r[ids] = (p.species.0 as usize + 1) as f32;
                r[ids + 1] = p.item.map_or(0.0, |i| (i.0 as usize + 1) as f32);
                for (k, ms) in p.move_slots.iter().enumerate().take(MOVES) {
                    r[ids + 2 + k] = (ms.id.0 as usize + 1) as f32;
                    let d = &mut r[OFF_MDYN + ((s * MONS + m) * MOVES + k) * F_MDYN..][..F_MDYN];
                    d[0] = ms.pp as f32 / ms.maxpp.max(1) as f32;
                    d[1] = f32::from(ms.pp <= 0);
                    d[2] = f32::from(ms.disabled);
                    d[3] = ms.pp as f32 / 32.0;
                }
                let f = &mut r[OFF_MON + (s * MONS + m) * F_MON..][..F_MON];
                self.mon_features(p, active, f);
                if active {
                    let v = &mut r[OFF_VOL + s * F_VOL..][..F_VOL];
                    self.volatile_features(p, v);
                }
            }
            let f = &mut r[OFF_SIDE + s * F_SIDE..][..F_SIDE];
            self.side_features(b, s, f);
        }
        for me in 0..2 {
            let foe = 1 - me;
            let f = &mut r[OFF_PAIR + me * F_PAIR..][..F_PAIR];
            for (i, si) in order[me].iter().enumerate() {
                for (j, sj) in order[foe].iter().enumerate() {
                    let (Some(si), Some(sj)) = (*si, *sj) else { continue };
                    let pi = &b.sides[me].roster[si as usize];
                    let pj = &b.sides[foe].roster[sj as usize];
                    if pi.fainted || pj.fainted || pi.hp <= 0 || pj.hp <= 0 {
                        continue;
                    }
                    let a = PokeId { side: me as u8, slot: si };
                    let d = PokeId { side: foe as u8, slot: sj };
                    let k = i * MONS + j;
                    f[k] = best_hit_fraction(b, dex, a, d, true).min(3.0) as f32;
                    f[MONS * MONS + k] = best_hit_fraction(b, dex, d, a, true).min(3.0) as f32;
                    let (vi, vj) = (effective_speed(pi, i == 0), effective_speed(pj, j == 0));
                    f[2 * MONS * MONS + k] = if vi > vj {
                        1.0
                    } else if vi < vj {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }
        }
        let g = &mut r[OFF_GLOBAL..][..F_GLOBAL];
        for (i, id) in self.weather_ids.iter().enumerate() {
            if id.is_some() && b.field.weather == *id {
                g[i] = 1.0;
                g[WEATHERS.len()] = b.field.weather_state.duration.unwrap_or(0) as f32 / 5.0;
            }
        }
        g[WEATHERS.len() + 1] = b.turn as f32 / 1000.0;
    }

    fn mon_features(&self, p: &Pokemon, active: bool, f: &mut [f32]) {
        let alive = !p.fainted && p.hp > 0;
        f[0] = f32::from(alive);
        f[1] = p.hp as f32 / p.maxhp.max(1) as f32;
        f[2] = p.hp as f32 / 400.0;
        f[3] = p.maxhp as f32 / 400.0;
        f[4] = p.level as f32 / 100.0;
        for k in 0..5 {
            f[5 + k] = p.stored_stats[k] as f32 / 400.0;
        }
        f[10] = effective_speed(p, active) as f32 / 400.0;
        let st = match p.status {
            Status::None => 0,
            Status::Brn => 1,
            Status::Par => 2,
            Status::Slp => 3,
            Status::Frz => 4,
            Status::Psn => 5,
            Status::Tox => 6,
            _ => 0,
        };
        f[11 + st] = 1.0;
        if p.status == Status::Slp {
            f[18] = p.status_state.get_int(DK::Time) as f32 / 4.0;
        }
        if let Some(rd) = self.residualdmg {
            if let Some(vs) = p.volatile(rd) {
                f[19] = vs.get_int(DK::Counter) as f32 / 16.0;
            }
        }
        f[20] = f32::from(p.item.is_some());
        f[21] = f32::from(active);
        f[22] = f32::from(p.transformed);
        for t in p.types.iter() {
            let t = t.0 as usize;
            assert!(t < N_TYPES, "type index {t} out of range");
            f[23 + t] = 1.0;
        }
        let ht = p.hp_type.0 as usize;
        assert!(ht < N_TYPES, "hidden power type index {ht} out of range");
        f[41 + ht] = 1.0;
        f[59] = p.hp_power as f32 / 70.0;
    }

    fn volatile_features(&self, p: &Pokemon, v: &mut [f32]) {
        for k in 0..7 {
            v[k] = p.boosts[k] as f32 / 6.0;
        }
        for (cid, es) in p.volatiles.iter() {
            let idx = self
                .vol_ids
                .iter()
                .position(|id| *id == Some(*cid))
                .unwrap_or(VOLATILE_KEYS.len());
            let e = &mut v[7 + idx * F_VOL_EACH..][..F_VOL_EACH];
            e[0] += 1.0;
            e[1] = Self::clock(es) / 8.0;
            e[2] = es.get_int(DK::Counter).max(es.get_int(DK::HitCount)) as f32 / 16.0;
            e[3] = es.get_int(DK::Hp) as f32 / p.maxhp.max(1) as f32;
        }
    }

    fn clock(es: &EffectState) -> f32 {
        es.duration.map_or_else(|| es.get_int(DK::Time) as f32, |d| d as f32)
    }

    fn side_features(&self, b: &Battle, s: usize, f: &mut [f32]) {
        let side = &b.sides[s];
        use nc2000_engine::state::RequestKind;
        match side.request_state() {
            Some(RequestKind::Move) => f[0] = 1.0,
            Some(RequestKind::Switch) => f[1] = 1.0,
            _ => f[2] = 1.0,
        }
        f[3] = side.pokemon_left as f32 / 3.0;
        let alive = side
            .party
            .iter()
            .filter(|&&slot| {
                let p = &side.roster[slot as usize];
                !p.fainted && p.hp > 0
            })
            .count();
        f[4] = alive as f32 / 3.0;
        for (i, id) in self.side_ids.iter().enumerate() {
            if let Some(es) = id.and_then(|id| side.side_condition(id)) {
                f[5 + 2 * i] = 1.0;
                f[6 + 2 * i] = if i == 0 {
                    es.get_int(DK::Layers).max(1) as f32 / 3.0
                } else {
                    es.duration.unwrap_or(0) as f32 / 5.0
                };
            }
        }
        let fm = 5 + 2 * SIDE_CONDS.len();
        if let Some(fid) = self.futuremove {
            if let Some((_, es)) = side.slot_conditions.iter().find(|(c, _)| *c == fid) {
                f[fm] = 1.0;
                f[fm + 1] = es.duration.unwrap_or(0) as f32 / 3.0;
            }
        }
    }
}

// ------------------------------------------------------------------ network

/// Eight independent partial sums: a single running sum is a serial
/// dependency chain the compiler may not reorder, so it never vectorizes.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]) + tail
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dense {
    pub n_in: usize,
    pub n_out: usize,
    /// Row-major [n_out][n_in].
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl Dense {
    fn check(&self, n_in: usize, name: &str) -> Result<(), String> {
        if self.n_in != n_in
            || self.w.len() != self.n_in * self.n_out
            || self.b.len() != self.n_out
            || self.w.iter().chain(&self.b).any(|x| !x.is_finite())
        {
            return Err(format!("layer {name}: bad shape or values"));
        }
        Ok(())
    }

    #[inline]
    fn forward(&self, x: &[f32], y: &mut [f32], relu: bool) {
        debug_assert_eq!(x.len(), self.n_in);
        for (o, (row, &bias)) in self.w.chunks_exact(self.n_in).zip(&self.b).enumerate() {
            let acc = bias + dot(row, x);
            y[o] = if relu { acc.max(0.0) } else { acc };
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Embedding {
    pub rows: usize,
    pub dim: usize,
    pub w: Vec<f32>,
}

impl Embedding {
    fn row(&self, i: usize) -> &[f32] {
        let i = i.min(self.rows - 1);
        &self.w[i * self.dim..(i + 1) * self.dim]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValueNet {
    pub schema: String,
    pub species: Embedding,
    pub items: Embedding,
    pub moves: Embedding,
    pub move_layer: Dense,
    pub mon1: Dense,
    pub mon2: Dense,
    pub head1: Dense,
    pub head2: Dense,
    pub head3: Dense,
    /// Trainer-side predictions on its first validation records, for the
    /// Rust/PyTorch parity check (`fvi_gen --parity`).
    #[serde(default)]
    pub parity: Vec<(Vec<f32>, f32)>,
    #[serde(default)]
    pub meta: serde_json::Value,
}

impl ValueNet {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let net: ValueNet = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        net.validate()?;
        Ok(net)
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != VALUE_SCHEMA {
            return Err(format!("schema {} != {VALUE_SCHEMA}", self.schema));
        }
        let hm = self.move_layer.n_out;
        self.move_layer.check(self.moves.dim + F_MSTAT + F_MDYN, "move_layer")?;
        self.mon1.check(self.species.dim + self.items.dim + hm + F_MON, "mon1")?;
        self.mon2.check(self.mon1.n_out, "mon2")?;
        let side = 2 * self.mon2.n_out + F_SIDE + F_VOL;
        self.head1.check(2 * side + F_PAIR + F_GLOBAL, "head1")?;
        self.head2.check(self.head1.n_out, "head2")?;
        self.head3.check(self.head2.n_out, "head3")?;
        if self.head3.n_out != 1 {
            return Err("head3 must have one output".into());
        }
        Ok(())
    }

    fn side_vector(&self, codec: &ValueCodec, r: &[f32], s: usize, out: &mut Vec<f32>) {
        let h2 = self.mon2.n_out;
        let hm = self.move_layer.n_out;
        let mut active = vec![0f32; h2];
        let mut bench = vec![0f32; h2];
        let mut mvec = vec![0f32; hm];
        let mut mtmp = vec![0f32; hm];
        let mut min = Vec::with_capacity(self.move_layer.n_in);
        let mut monin = Vec::with_capacity(self.mon1.n_in);
        let mut h1 = vec![0f32; self.mon1.n_out];
        let mut h2v = vec![0f32; h2];
        for m in 0..MONS {
            let mon = &r[OFF_MON + (s * MONS + m) * F_MON..][..F_MON];
            if mon[0] == 0.0 {
                continue;
            }
            let ids = &r[OFF_IDS + (s * MONS + m) * IDS..][..IDS];
            mvec.iter_mut().for_each(|v| *v = 0.0);
            for k in 0..MOVES {
                let id = ids[2 + k] as usize;
                if id == 0 {
                    continue;
                }
                min.clear();
                min.extend_from_slice(self.moves.row(id));
                min.extend_from_slice(codec.move_static(id.min(codec.move_static.len() - 1)));
                min.extend_from_slice(&r[OFF_MDYN + ((s * MONS + m) * MOVES + k) * F_MDYN..][..F_MDYN]);
                self.move_layer.forward(&min, &mut mtmp, true);
                for (a, b) in mvec.iter_mut().zip(&mtmp) {
                    *a += b;
                }
            }
            monin.clear();
            monin.extend_from_slice(self.species.row(ids[0] as usize));
            monin.extend_from_slice(self.items.row(ids[1] as usize));
            monin.extend_from_slice(&mvec);
            monin.extend_from_slice(mon);
            self.mon1.forward(&monin, &mut h1, true);
            self.mon2.forward(&h1, &mut h2v, true);
            if m == 0 {
                active.copy_from_slice(&h2v);
            } else {
                for (a, b) in bench.iter_mut().zip(&h2v) {
                    *a += b;
                }
            }
        }
        out.extend_from_slice(&active);
        out.extend_from_slice(&bench);
        out.extend_from_slice(&r[OFF_SIDE + s * F_SIDE..][..F_SIDE]);
        out.extend_from_slice(&r[OFF_VOL + s * F_VOL..][..F_VOL]);
    }

    fn head(&self, sides: &[Vec<f32>; 2], r: &[f32], me: usize) -> f32 {
        let mut x = Vec::with_capacity(self.head1.n_in);
        x.extend_from_slice(&sides[me]);
        x.extend_from_slice(&sides[1 - me]);
        x.extend_from_slice(&r[OFF_PAIR + me * F_PAIR..][..F_PAIR]);
        x.extend_from_slice(&r[OFF_GLOBAL..][..F_GLOBAL]);
        let mut a = vec![0f32; self.head1.n_out];
        let mut c = vec![0f32; self.head2.n_out];
        let mut o = [0f32; 1];
        self.head1.forward(&x, &mut a, true);
        self.head2.forward(&a, &mut c, true);
        self.head3.forward(&c, &mut o, false);
        o[0]
    }

    /// Side-0 logit for one encoded record.
    pub fn logit(&self, codec: &ValueCodec, r: &[f32]) -> f32 {
        let mut s0 = Vec::with_capacity(self.head1.n_in);
        let mut s1 = Vec::with_capacity(self.head1.n_in);
        self.side_vector(codec, r, 0, &mut s0);
        self.side_vector(codec, r, 1, &mut s1);
        let sides = [s0, s1];
        self.head(&sides, r, 0) - self.head(&sides, r, 1)
    }

    pub fn value(&self, codec: &ValueCodec, r: &[f32]) -> f64 {
        1.0 / (1.0 + (-(self.logit(codec, r) as f64)).exp())
    }
}

// ------------------------------------------------------------------ leaf values

/// Leaf value source for backups: the handcrafted eval (iteration 0) or a
/// trained network.
pub enum Leaf {
    Eval(EvalWeights),
    Net(std::sync::Arc<ValueNet>, std::sync::Arc<ValueCodec>),
}

impl Leaf {
    pub fn value(&self, b: &Battle, dex: &Dex, scratch: &mut Vec<f32>) -> f64 {
        if let Some(o) = b.outcome() {
            return outcome_value(o);
        }
        match self {
            Leaf::Eval(w) => eval01(b, dex, w),
            Leaf::Net(net, codec) => {
                scratch.clear();
                codec.encode(b, dex, scratch);
                net.value(codec, scratch)
            }
        }
    }
}

pub fn outcome_value(o: Outcome) -> f64 {
    match o {
        Outcome::P1Win => 1.0,
        Outcome::P2Win => 0.0,
        Outcome::Tie => 0.5,
    }
}

/// One sampled Shapley backup at a decision state.
pub struct Backup {
    pub actions: [Vec<Option<SearchChoice>>; 2],
    /// Row-major [n0][n1] expected successor values (side 0's view).
    pub matrix: Vec<f64>,
    pub value: f64,
    /// Equilibrium mixtures of the sampled matrix, aligned with `actions`.
    pub mix: [Vec<f64>; 2],
}

pub fn legal_actions(b: &Battle, dex: &Dex) -> [Vec<Option<SearchChoice>>; 2] {
    let needs = b.needs_choice();
    let mut probe = b.clone();
    [0, 1].map(|s| {
        if needs[s] {
            probe.legal_choices(dex, s).into_iter().map(Some).collect()
        } else {
            vec![None]
        }
    })
}

/// Matrix-game value and mixtures with the degenerate one-sided cases
/// solved directly.
pub fn solve(m: &[f64], n0: usize, n1: usize) -> (f64, [Vec<f64>; 2]) {
    if n0 > 1 && n1 > 1 {
        let s = solve_matrix_full(m, n0, n1);
        return (s.value, [s.x, s.y]);
    }
    let pick = |n: usize, better: &dyn Fn(f64, f64) -> bool| {
        let mut best = 0;
        for i in 1..n {
            if better(m[i], m[best]) {
                best = i;
            }
        }
        best
    };
    if n0 == 1 && n1 == 1 {
        (m[0], [vec![1.0], vec![1.0]])
    } else if n1 == 1 {
        let i = pick(n0, &|a, b| a > b);
        let mut x = vec![0.0; n0];
        x[i] = 1.0;
        (m[i], [x, vec![1.0]])
    } else {
        let j = pick(n1, &|a, b| a < b);
        let mut y = vec![0.0; n1];
        y[j] = 1.0;
        (m[j], [vec![1.0], y])
    }
}

/// Sampled Shapley backup: K seeded successors per joint action, leaf
/// values cached by exact state key (deterministic outcomes repeat).
pub fn sampled_backup(
    b: &Battle,
    dex: &Dex,
    leaf: &Leaf,
    k: usize,
    rng: &mut SplitMix64,
    scratch: &mut Vec<f32>,
) -> Option<Backup> {
    let actions = legal_actions(b, dex);
    let (n0, n1) = (actions[0].len(), actions[1].len());
    if n0 == 0 || n1 == 0 {
        return None;
    }
    let mut cache: HashMap<u64, f64> = HashMap::new();
    let mut matrix = vec![0.0; n0 * n1];
    for (i, &c0) in actions[0].iter().enumerate() {
        for (j, &c1) in actions[1].iter().enumerate() {
            let mut acc = 0.0;
            for _ in 0..k {
                let mut c = b.clone();
                c.set_log_enabled(false);
                c.reseed(rng.next());
                c.apply_choices(dex, [c0, c1]).expect("legal choices");
                let key = c.state_key();
                let v = match cache.get(&key) {
                    Some(&v) => v,
                    None => {
                        let v = leaf.value(&c, dex, scratch);
                        cache.insert(key, v);
                        v
                    }
                };
                acc += v;
            }
            matrix[i * n1 + j] = acc / k as f64;
        }
    }
    let (value, mix) = solve(&matrix, n0, n1);
    Some(Backup { actions, matrix, value, mix })
}

/// Sample an index from `p` with probability `eps` of a uniform pick.
pub fn sample_mixed(p: &[f64], eps: f64, rng: &mut SplitMix64) -> usize {
    if p.len() == 1 {
        return 0;
    }
    if rng.next_f64() < eps {
        return rng.below(p.len());
    }
    let total: f64 = p.iter().map(|x| x.max(0.0)).sum();
    let mut u = rng.next_f64() * total;
    for (i, &x) in p.iter().enumerate() {
        u -= x.max(0.0);
        if u <= 0.0 {
            return i;
        }
    }
    p.len() - 1
}

// ------------------------------------------------------------------ agent

/// Team preview shared by every arm of a value duel: a pick that depends only
/// on the preview state and the side, so both games of a side-swapped pair
/// field the same teams whichever agent sits where.
pub fn shared_preview(battle: &Battle, side: usize, choices: &[SearchChoice]) -> SearchChoice {
    let mut rng = SplitMix64::new(battle.state_key() ^ (side as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    choices[rng.below(choices.len())]
}

/// Plays the equilibrium mixture of the 1-ply sampled Shapley backup of its
/// leaf (`eval01` or a network). Team preview: `shared_preview`.
pub struct ValueAgent {
    pub leaf: Leaf,
    pub k: usize,
    rng: SplitMix64,
    scratch: Vec<f32>,
}

impl ValueAgent {
    pub fn new(leaf: Leaf, k: usize, seed: u64) -> Self {
        ValueAgent { leaf, k, rng: SplitMix64::new(seed), scratch: Vec::new() }
    }
}

impl crate::agent::Agent for ValueAgent {
    fn name(&self) -> String {
        match self.leaf {
            Leaf::Eval(_) => format!("value-eval01:k{}", self.k),
            Leaf::Net(..) => format!("value-net:k{}", self.k),
        }
    }

    fn choose(&mut self, battle: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> SearchChoice {
        if battle.turn == 0 || choices.len() == 1 {
            return if battle.turn == 0 { shared_preview(battle, side, choices) } else { choices[0] };
        }
        let bk = sampled_backup(battle, dex, &self.leaf, self.k, &mut self.rng, &mut self.scratch)
            .expect("decision state");
        let i = sample_mixed(&bk.mix[side], 0.0, &mut self.rng);
        let pick = bk.actions[side][i].expect("side owes a choice");
        debug_assert!(choices.contains(&pick));
        pick
    }
}
