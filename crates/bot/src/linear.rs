//! Linear-algebra transition and Bellman-Markov operators on [`CompactBattle`]:
//!
//! 1. **1D Banded Toeplitz Damage & Residual Kernel ([`DmgKernel1D`], [`survival_curve_4`])**:
//!    Exact Gen 2 damage roll + critical hit + accuracy/evasion + BrightPowder +
//!    `Sleep Talk` uniform slot mixture + end-of-turn residual shift operator $T$.
//!    Evaluates $k$-turn survival probabilities $S(k) = (\mathbf{1}^\top T^k \mathbf{e}_{h_0})$
//!    with zero heap allocations.
//! 2. **2D Coupled Schur-Complement Bellman Solver & Rank-3 SVD ([`solve_schur_bellman_2d`], [`SolvedBellmanSvd`])**:
//!    For matchups with active recovery (`Rest` or 50% heals), solves the coupled
//!    Bellman-Shapley fixed-point equation $(I - \gamma P_{\pi^*}) V^* = r_{\pi^*}$ over
//!    the canonical $16 \times 16 \times 3 \times 3$ `(hp0_bin, hp1_bin, sleep0, sleep1)` state space
//!    (where $1/16$ is the exact Gen 2 Leftovers/residual quantum).
//!    Sleeping states ($s_0 + s_1 > 0$) are eliminated in a single bottom-up Schur-complement pass per
//!    sweep, and the awake stage game is solved in closed form ($2 \times 2$ minimax/Nash).
//!    The resulting awake value surface is decomposed into its top-3 singular vectors
//!    $V(b_0, b_1) - 0.5 \approx \sum_{r=1}^3 \sigma_r u_r(b_0) v_r(b_1)$ and cached in a
//!    fixed-size thread-local table.
//! 3. **Exact Stopping-Time Bilinear Race ([`toeplitz_coupled_race`])**:
//!    For non-healing matchups, evaluates the exact discrete-time stopping-time race probability
//!    $\mathbb{P}(\tau_0 > \tau_1) - \mathbb{P}(\tau_1 > \tau_0) = \sum_k (f_1(k) S_0(k) - f_0(k) S_1(k) + f_0(k) f_1(k) \text{spd\_sign})$
//!    from the 1D Toeplitz survival curves $T_0^k \mathbf{1}$ and $T_1^k \mathbf{1}$.
//! 4. **Perish Song / Mean Look Countdown Survival ([`compact_perish_margin`])**:
//!    Evaluates active Perish Song trap survival directly from the Toeplitz survival curve
//!    $S_{\text{trapper}}(c - 1 - \mathbf{1}_{\text{protect}}) \cdot (1 - \mathbf{1}_{\text{phaze}} S_{\text{victim}}(1))$.

use std::cell::RefCell;

use nc2000_engine::compact::{vol_bit, CompactBattle, CompactWeather, MatchTable};
use nc2000_engine::state::Status;

/// Canonical Gen 2 HP grid resolution ($1/16 = 6.25\%$, matching Leftovers $1/16$,
/// Burn/Poison/Leech Seed $2/16$, Substitute $4/16$, and Recover/Soft-Boiled $8/16$).
pub const B16: usize = 16;

/// Fine 1D grid resolution for pure stopping-time convolutions without recovery.
const B32: usize = 32;

/// Discount factor $\gamma$ pulling infinite stall cycles toward the tie value $0.5$.
const GAMMA: f32 = 0.985;

/// Stack-allocated 1D damage distribution kernel (up to 16 `(dmg, prob)` support points).
#[derive(Clone, Copy, Debug)]
pub struct DmgKernel1D {
    pub pts: [(i16, f32); 16],
    pub len: usize,
    pub max_dmg: i16,
    pub mean_dmg: f32,
}

impl DmgKernel1D {
    #[inline]
    pub const fn zero() -> Self {
        Self {
            pts: [(0, 1.0); 16],
            len: 1,
            max_dmg: 0,
            mean_dmg: 0.0,
        }
    }

    #[inline]
    fn push_mass(&mut self, dmg: i16, prob: f32) {
        if prob <= 0.0 {
            return;
        }
        if dmg > self.max_dmg {
            self.max_dmg = dmg;
        }
        self.mean_dmg += (dmg as f32) * prob;
        for i in 0..self.len {
            if self.pts[i].0 == dmg {
                self.pts[i].1 += prob;
                return;
            }
        }
        if self.len < 16 {
            self.pts[self.len] = (dmg, prob);
            self.len += 1;
        } else {
            self.pts[self.len - 1].1 += prob;
        }
    }

    /// Probability that a single attack from this kernel KOs a defender at `def_hp`.
    #[inline]
    pub fn ko_prob(&self, def_hp: i16) -> f32 {
        if def_hp <= 0 {
            return 1.0;
        }
        if self.max_dmg < def_hp {
            return 0.0;
        }
        let mut p = 0.0f32;
        for i in 0..self.len {
            if self.pts[i].0 >= def_hp {
                p += self.pts[i].1;
            }
        }
        p.clamp(0.0, 1.0)
    }

    /// Build the 1D damage kernel for the best damaging move of `(att_side, att_slot)` against `(def_side, def_slot)`.
    #[inline]
    pub fn best_attack(
        cb: &CompactBattle,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
    ) -> Self {
        let att_idx = att_side * 3 + att_slot;
        let def_idx = def_side * 3 + def_slot;
        let att_mon = &cb.mons[att_idx];
        let def_hp = cb.mons[def_idx].hp().max(1);
        let num_moves = table.mons[att_idx].num_moves as usize;
        let is_disabled = (att_mon.vol_flags & vol_bit::DISABLE) != 0;
        let dis_slot = ((att_mon.slot_refs >> 4) & 0x03) as usize;
        let last_mon = cb.sides[att_side].alive_mask.count_ones() <= 1;

        let mut best_kern = Self::zero();
        let mut best_score = 0.0f32;
        for m_i in 0..num_moves {
            if att_mon.pp[m_i] == 0 || (is_disabled && m_i == dis_slot) {
                continue;
            }
            let mspec = &table.move_specs[att_idx][m_i];
            if last_mon && mspec.is_selfdestruct {
                continue;
            }
            let kern = Self::for_move(cb, table, att_side, att_slot, def_side, def_slot, m_i);
            if kern.max_dmg <= 0 {
                continue;
            }
            let p_ko = kern.ko_prob(def_hp);
            let frac = (kern.mean_dmg / (def_hp as f32)).min(1.0);
            let raw_frac = kern.mean_dmg / (table.mons[def_idx].maxhp.max(1) as f32);
            let score = p_ko + frac + 1e-3 * raw_frac;
            if score > best_score {
                best_score = score;
                best_kern = kern;
            }
        }
        best_kern
    }

    /// Build the 1D damage kernel for the best damaging move on the full-HP grid (independent of instantaneous HP).
    #[inline]
    pub fn best_attack_grid(
        cb: &CompactBattle,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
    ) -> Self {
        let att_idx = att_side * 3 + att_slot;
        let def_idx = def_side * 3 + def_slot;
        let att_mon = &cb.mons[att_idx];
        let def_maxhp = table.mons[def_idx].maxhp.max(1);
        let num_moves = table.mons[att_idx].num_moves as usize;
        let is_disabled = (att_mon.vol_flags & vol_bit::DISABLE) != 0;
        let dis_slot = ((att_mon.slot_refs >> 4) & 0x03) as usize;
        let last_mon = cb.sides[att_side].alive_mask.count_ones() <= 1;

        let mut best_kern = Self::zero();
        let mut best_score = 0.0f32;
        for m_i in 0..num_moves {
            if att_mon.pp[m_i] == 0 || (is_disabled && m_i == dis_slot) {
                continue;
            }
            let mspec = &table.move_specs[att_idx][m_i];
            if last_mon && mspec.is_selfdestruct {
                continue;
            }
            let kern = Self::for_move_impl(
                cb, table, att_side, att_slot, def_side, def_slot, m_i, true,
            );
            if kern.max_dmg <= 0 {
                continue;
            }
            let score = kern.mean_dmg / (def_maxhp as f32);
            if score > best_score {
                best_score = score;
                best_kern = kern;
            }
        }
        best_kern
    }

    /// Build the exact `Sleep Talk` mixture kernel: in Gen 2, `Sleep Talk` uniformly selects
    /// one of the 3 other move slots (`m != st_slot`) with weight $1 / (N_{\text{moves}} - 1)$.
    #[inline]
    pub fn sleep_talk(
        cb: &CompactBattle,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
    ) -> Self {
        Self::sleep_talk_impl(cb, table, att_side, att_slot, def_side, def_slot, false)
    }

    #[inline]
    pub fn sleep_talk_grid(
        cb: &CompactBattle,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
    ) -> Self {
        Self::sleep_talk_impl(cb, table, att_side, att_slot, def_side, def_slot, true)
    }

    #[inline]
    fn sleep_talk_impl(
        cb: &CompactBattle,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
        grid_mode: bool,
    ) -> Self {
        let att_idx = att_side * 3 + att_slot;
        let att_mon = &cb.mons[att_idx];
        let num_moves = table.mons[att_idx].num_moves as usize;
        let is_disabled = (att_mon.vol_flags & vol_bit::DISABLE) != 0;
        let dis_slot = ((att_mon.slot_refs >> 4) & 0x03) as usize;

        let mut st_slot = None;
        for m_i in 0..num_moves {
            if att_mon.pp[m_i] > 0
                && !(is_disabled && m_i == dis_slot)
                && table.move_specs[att_idx][m_i].special_kind == 10
            {
                st_slot = Some(m_i);
                break;
            }
        }
        let Some(st_i) = st_slot else {
            return Self::zero();
        };
        if num_moves <= 1 {
            return Self::zero();
        }
        let weight = 1.0 / ((num_moves - 1) as f32);
        let mut mix = Self {
            pts: [(0, 0.0); 16],
            len: 0,
            max_dmg: 0,
            mean_dmg: 0.0,
        };
        for m_i in 0..num_moves {
            if m_i == st_i {
                continue;
            }
            let mspec = &table.move_specs[att_idx][m_i];
            if mspec.is_rest || mspec.is_status || mspec.is_selfdestruct {
                mix.push_mass(0, weight);
                continue;
            }
            let k = Self::for_move_impl(
                cb, table, att_side, att_slot, def_side, def_slot, m_i, grid_mode,
            );
            for j in 0..k.len {
                mix.push_mass(k.pts[j].0, k.pts[j].1 * weight);
            }
        }
        if mix.len == 0 {
            Self::zero()
        } else {
            mix
        }
    }

    /// Build the 1D damage kernel for move slot `m_slot` of `(att_side, att_slot)` against `(def_side, def_slot)`.
    #[inline]
    pub fn for_move(
        cb: &CompactBattle,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
        m_slot: usize,
    ) -> Self {
        Self::for_move_impl(
            cb, table, att_side, att_slot, def_side, def_slot, m_slot, false,
        )
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn for_move_impl(
        cb: &CompactBattle,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
        m_slot: usize,
        grid_mode: bool,
    ) -> Self {
        let att_idx = att_side * 3 + att_slot;
        let def_idx = def_side * 3 + def_slot;
        let mspec = &table.move_specs[att_idx][m_slot];
        let att_mon = &cb.mons[att_idx];
        let def_mon = &cb.mons[def_idx];
        let att_spec = &table.mons[att_idx];
        let def_spec = &table.mons[def_idx];

        let base_power = if mspec.var_bp_kind == 4 {
            let att_hp = if grid_mode {
                att_spec.maxhp / 2
            } else {
                att_mon.hp()
            };
            let ratio =
                ((att_hp as f64 * 48.0 / att_spec.maxhp.max(1) as f64).floor() as i32).max(1);
            if ratio < 2 {
                200
            } else if ratio < 5 {
                150
            } else if ratio < 10 {
                100
            } else if ratio < 17 {
                80
            } else if ratio < 33 {
                40
            } else {
                20
            }
        } else {
            mspec.base_power as i32
        };

        let is_super_fang = mspec.special_kind == 15;
        if mspec.fixed_damage == 0 && !is_super_fang && (mspec.is_status || base_power <= 0) {
            return Self::zero();
        }
        let def_foresight = (def_mon.vol_flags & vol_bit::FORESIGHT) != 0;
        let (type_mod, is_immune) =
            table.effectiveness_ext(mspec.move_type, def_spec, def_foresight);
        if is_immune {
            return Self::zero();
        }

        let mut p_hit =
            cb.hit_probability(table, att_side, att_slot, def_side, def_slot, m_slot) as f32;
        if def_mon.item != 0 && def_spec.is_brightpowder && !mspec.always_hit {
            p_hit *= 235.0 / 256.0;
        }
        if p_hit <= 0.0 {
            return Self::zero();
        }

        if mspec.fixed_damage != 0 || is_super_fang {
            let fdmg = if is_super_fang {
                let d_hp = if grid_mode {
                    def_spec.maxhp / 2
                } else {
                    def_mon.hp()
                };
                (d_hp / 2).max(1)
            } else if mspec.fixed_damage < 0 {
                att_spec.level as i16
            } else {
                mspec.fixed_damage
            };
            let mut k = Self {
                pts: [(0, 0.0); 16],
                len: 0,
                max_dmg: 0,
                mean_dmg: 0.0,
            };
            if p_hit < 1.0 {
                k.push_mass(0, 1.0 - p_hit);
            }
            k.push_mass(fdmg, p_hit);
            return k;
        }

        let hits_mult = (mspec.multihit_mean_x2 as f32 * 0.5).max(1.0);
        const CRIT_DEN: [u32; 6] = [0, 16, 8, 4, 3, 2];
        let scope_bonus = ((att_mon.item != 0) & att_spec.is_scope_lens) as usize;
        let c_stage = (mspec.crit_ratio as usize + scope_bonus).min(5);
        let p_crit = if mspec.special_kind != 13 && c_stage > 0 {
            1.0 / (CRIT_DEN[c_stage] as f32)
        } else {
            0.0
        };

        let compute_base = |is_crit: bool| -> i32 {
            let atk_stat = if mspec.is_physical { 0 } else { 2 };
            let def_stat = if mspec.is_physical { 1 } else { 3 };
            let ignore_boosts = is_crit & (att_mon.boosts[atk_stat] <= def_mon.boosts[def_stat]);
            let mut atk =
                cb.mon_stat(table, att_side, att_slot, atk_stat, ignore_boosts, ignore_boosts) as i32;
            let mut def =
                cb.mon_stat(table, def_side, def_slot, def_stat, ignore_boosts, false) as i32;
            if (atk | def) >= 256 {
                atk = (atk >> 2).max(1);
                def = (def >> 2).max(1);
            }
            if mspec.is_selfdestruct && def_stat == 1 {
                def = (def / 2).max(1);
            }
            let level = att_spec.level as i32;
            let mut d = ((level * 2 / 5 + 2) * base_power * atk / def.max(1)) / 50;
            if is_crit {
                d *= 2;
            }
            if att_mon.item != 0 && att_spec.item_boost_type == mspec.move_type {
                d = d * 110 / 100;
            }
            d = d.min(997) + 2;
            if cb.weather_turns > 0 {
                if cb.weather_kind == CompactWeather::Rain as u8 {
                    if mspec.is_water {
                        d = d * 3 / 2;
                    } else if mspec.is_fire || mspec.special_kind == 23 {
                        d /= 2;
                    }
                } else if cb.weather_kind == CompactWeather::Sun as u8 {
                    if mspec.is_fire {
                        d = d * 3 / 2;
                    } else if mspec.is_water {
                        d /= 2;
                    }
                }
            }
            if (att_spec.type_mask & (1u32 << (mspec.move_type & 31))) != 0 {
                d += d / 2;
            }
            if type_mod > 0 {
                d <<= type_mod as u32;
            } else if type_mod < 0 {
                d >>= (-type_mod) as u32;
            }
            d.max(1)
        };

        let base_norm = compute_base(false);
        let base_crit = if p_crit > 0.0 {
            compute_base(true)
        } else {
            base_norm
        };
        const ROLL_NUMS: [i32; 4] = [221, 231, 242, 253];
        let mut k = Self {
            pts: [(0, 0.0); 16],
            len: 0,
            max_dmg: 0,
            mean_dmg: 0.0,
        };
        if p_hit < 1.0 {
            k.push_mass(0, 1.0 - p_hit);
        }
        let p_norm_roll = p_hit * (1.0 - p_crit) * 0.25;
        for &r in &ROLL_NUMS {
            let dmg = (((base_norm * r / 255).max(1) as f32) * hits_mult).round() as i16;
            k.push_mass(dmg, p_norm_roll);
        }
        if p_crit > 0.0 {
            let p_crit_roll = p_hit * p_crit * 0.25;
            for &r in &ROLL_NUMS {
                let dmg = (((base_crit * r / 255).max(1) as f32) * hits_mult).round() as i16;
                k.push_mass(dmg, p_crit_roll);
            }
        }
        k
    }
}

/// Compute `[S(0), S(1), S(2), S(3)]`, the probability that `def_side`'s active mon
/// stays strictly above `0` HP after $k = 0, 1, 2, 3$ turns of `att_side`'s attack + residuals:
/// $S(k) = (\mathbf{1}^\top T^k \mathbf{e}_{h_0})$.
#[inline]
pub fn survival_curve_4(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    def_side: usize,
    start_hp: i16,
) -> [f64; 4] {
    let (s, _) = survival_curve_6_with_mean_hp(cb, table, att_side, def_side, start_hp);
    [s[0], s[1], s[2], s[3]]
}

/// Compute `([S(0)..=S(6)], mean_hp_frac_at_6)` via exact 1D Toeplitz convolution $T^k \mathbf{e}_{h_0}$.
#[inline]
fn survival_curve_6_with_mean_hp(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    def_side: usize,
    start_hp: i16,
) -> ([f64; 7], f64) {
    if start_hp <= 0 {
        return ([0.0; 7], 0.0);
    }
    let att_slot = cb.sides[att_side].active_idx as usize;
    let def_slot = cb.sides[def_side].active_idx as usize;
    let att_idx = att_side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    let maxhp = table.mons[def_idx].maxhp.max(1);
    let kern_awake = DmgKernel1D::best_attack(cb, table, att_side, att_slot, def_side, def_slot);
    let kern_slp = DmgKernel1D::sleep_talk(cb, table, att_side, att_slot, def_side, def_slot);

    let to_b = |hp: i16| -> usize {
        if hp <= 0 {
            0
        } else {
            (((hp as i32 * B32 as i32 + maxhp as i32 - 1) / maxhp as i32) as usize).clamp(1, B32)
        }
    };
    let b0 = to_b(start_hp.min(maxhp));
    let b_to_hp = |b: usize, step: usize| -> i16 {
        if b == 0 {
            0
        } else if step == 1 && b == b0 {
            start_hp.min(maxhp)
        } else {
            (((b as i32 * maxhp as i32) + (B32 as i32 / 2)) / B32 as i32)
                .clamp(1, maxhp as i32) as i16
        }
    };

    let mut dist = [0.0f32; B32 + 1];
    dist[b0] = 1.0;

    let def_mon = &cb.mons[def_idx];
    let att_mon = &cb.mons[att_idx];
    let def_spec = &table.mons[def_idx];
    let att_spec = &table.mons[att_idx];

    let lefties: i16 = if def_mon.item != 0 && def_spec.is_leftovers {
        (maxhp / 16).max(1)
    } else {
        0
    };
    let leech_drain: i16 = if (def_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
        (maxhp / 8).max(1)
    } else {
        0
    };
    let leech_gain: i16 = if (att_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
        (att_spec.maxhp.max(1) / 8).max(1)
    } else {
        0
    };
    let def_st = def_mon.status_raw();
    let cnt_packed = def_mon.status_counter();
    let is_tox = def_st == Status::Tox as u8
        || ((cnt_packed & 0x10) != 0 && (def_st == Status::Psn as u8 || def_st == Status::Brn as u8));
    let mut tox_counter = (cnt_packed & 0x0F) as i16;
    let psn_brn_dmg: i16 = if def_st == Status::Brn as u8 || def_st == Status::Psn as u8 {
        (maxhp / 8).max(1)
    } else {
        0
    };

    // Defender confusion self-hit & recharge
    let mut def_conf = if (def_mon.vol_flags & vol_bit::CONFUSION) != 0 {
        def_mon.timers[0].saturating_sub(1)
    } else {
        0
    };
    let def_atk = cb.mon_stat(table, def_side, def_slot, 0, false, false) as i32;
    let def_def = cb.mon_stat(table, def_side, def_slot, 1, false, false) as i32;
    let def_conf_dmg =
        (((def_spec.level as i32 * 2 / 5 + 2) * 40 * def_atk / def_def.max(1)) / 50 + 2).max(1)
            as i16;

    // Attacker sleep, freeze, paralysis, confusion, recharge
    let att_berry = att_mon.item != 0 && (att_spec.is_mintberry || att_spec.is_miracleberry);
    let att_miracle = att_mon.item != 0 && att_spec.is_miracleberry;
    let mut att_slp = if att_mon.status_raw() == Status::Slp as u8 && !att_berry {
        att_mon.status_counter().saturating_sub(1)
    } else {
        0
    };
    let mut p_still_frz = if att_mon.status_raw() == Status::Frz as u8 && !att_miracle {
        1.0f32
    } else {
        0.0f32
    };
    let att_par = att_mon.status_raw() == Status::Par as u8 && !att_miracle;
    let mut att_conf = if (att_mon.vol_flags & vol_bit::CONFUSION) != 0 && !att_miracle {
        att_mon.timers[0].saturating_sub(1)
    } else {
        0
    };
    let mut att_rech = (att_mon.slot_refs & 0x80) != 0;
    let mut sub_up = (def_mon.vol_flags & vol_bit::SUBSTITUTE) != 0 && def_mon.hp_pool[1] > 0;

    let mut out = [1.0f64; 7];
    for step in 1..=6usize {
        let mut next = [0.0f32; B32 + 1];
        next[0] = dist[0];

        let status_dot = if is_tox {
            tox_counter += 1;
            (maxhp / 16).max(1) * tox_counter
        } else {
            psn_brn_dmg
        };
        let res_this_turn = lefties + leech_gain - leech_drain - status_dot;

        let p_def_selfhit = if def_conf > 0 {
            def_conf -= 1;
            0.5f32
        } else {
            0.0f32
        };

        let mut p_att = 1.0f32;
        if att_rech {
            att_rech = false;
            p_att = 0.0;
        } else if p_still_frz > 0.0 {
            p_att *= 1.0 - p_still_frz;
            p_still_frz *= 231.0 / 256.0;
        } else {
            if att_conf > 0 {
                att_conf -= 1;
                p_att *= 0.5;
            }
            if att_par {
                p_att *= 0.75;
            }
        }

        let kern = if att_slp > 0 {
            att_slp -= 1;
            &kern_slp
        } else {
            &kern_awake
        };

        let sub_absorbs = sub_up && p_att > 0.0 && kern.max_dmg > 0;
        if sub_absorbs {
            sub_up = false;
        }

        for b in 1..=B32 {
            let p_state = dist[b];
            if p_state <= 1e-7 {
                continue;
            }
            let raw_hp = b_to_hp(b, step);

            // Defender confusion self-hit branch (50% when confused)
            let branches: [(i16, f32); 2] = if p_def_selfhit > 0.0 {
                [
                    (raw_hp, 1.0 - p_def_selfhit),
                    ((raw_hp - def_conf_dmg).max(0), p_def_selfhit),
                ]
            } else {
                [(raw_hp, 1.0), (0, 0.0)]
            };

            for &(hp_after_self, w_self) in &branches {
                if w_self <= 0.0 {
                    continue;
                }
                let pm = p_state * w_self;
                if hp_after_self <= 0 {
                    next[0] += pm;
                    continue;
                }
                if p_att < 1.0 {
                    let p_no_act = pm * (1.0 - p_att);
                    let final_hp = (hp_after_self + res_this_turn).clamp(0, maxhp);
                    next[to_b(final_hp)] += p_no_act;
                }
                if p_att > 0.0 {
                    let pa = pm * p_att;
                    for i in 0..kern.len {
                        let (dmg, prob) = kern.pts[i];
                        let eff_dmg = if sub_absorbs { 0 } else { dmg };
                        let post_hit = (hp_after_self - eff_dmg).max(0);
                        let final_hp = if post_hit <= 0 {
                            0
                        } else {
                            (post_hit + res_this_turn).clamp(0, maxhp)
                        };
                        next[to_b(final_hp)] += pa * prob;
                    }
                }
            }
        }
        dist = next;
        let mut surv = 0.0f32;
        for b in 1..=B32 {
            surv += dist[b];
        }
        out[step] = (surv as f64).clamp(0.0, 1.0);
    }

    let mut alive_mass = 0.0f32;
    let mut weighted_b = 0.0f32;
    for b in 1..=B32 {
        alive_mass += dist[b];
        weighted_b += dist[b] * (b as f32 / B32 as f32);
    }
    let mean_hp_frac = if alive_mass > 1e-6 {
        (weighted_b / alive_mass) as f64
    } else {
        0.0
    };
    (out, mean_hp_frac)
}

// =========================================================================
// 2. 2D Coupled Schur-Complement Bellman Solver & Rank-3 SVD
// =========================================================================

#[derive(Clone, Copy)]
struct BinDist16 {
    bins: [(u8, f32); 8],
    len: usize,
    ko_prob: f32,
}

impl BinDist16 {
    #[inline]
    const fn identity(b: u8) -> Self {
        let mut bins = [(0u8, 0.0f32); 8];
        bins[0] = (b, 1.0);
        Self {
            bins,
            len: 1,
            ko_prob: 0.0,
        }
    }

    #[inline]
    fn push(&mut self, b: u8, prob: f32) {
        if prob <= 0.0 {
            return;
        }
        if b == 0 {
            self.ko_prob += prob;
            return;
        }
        for i in 0..self.len {
            if self.bins[i].0 == b {
                self.bins[i].1 += prob;
                return;
            }
        }
        if self.len < 8 {
            self.bins[self.len] = (b, prob);
            self.len += 1;
        } else {
            self.bins[self.len - 1].1 += prob;
        }
    }
}

#[derive(Clone, Copy)]
struct Side1DOps {
    has_heal: bool,
    heal_is_rest: bool,
    has_rest: bool,
    has_st: bool,
    has_berry: bool,
    half_heal_bin: [u8; B16 + 1],
    /// `eot_bin[cured][b]` -> post-residual bin in `0..=16`
    eot_bin: [[u8; B16 + 1]; 2],
    /// `incoming[kind][b]` where `kind = 0` (no foe attack), `1` (foe awake attack), `2` (foe Sleep Talk)
    incoming: [[BinDist16; B16 + 1]; 3],
}

impl Side1DOps {
    #[inline]
    fn to_b16(hp: i16, maxhp: i16) -> u8 {
        if hp <= 0 {
            0
        } else {
            (((hp as i32 * B16 as i32 + maxhp as i32 - 1) / maxhp as i32) as u8).clamp(1, B16 as u8)
        }
    }

    #[inline]
    fn b16_to_hp(b: usize, maxhp: i16) -> i16 {
        if b == 0 {
            0
        } else {
            (((b as i32 * maxhp as i32) + (B16 as i32 / 2)) / B16 as i32)
                .clamp(1, maxhp as i32) as i16
        }
    }

    fn build(
        cb: &CompactBattle,
        table: &MatchTable,
        side: usize,
        slot: usize,
        foe_side: usize,
        foe_slot: usize,
        foe_k_att: &DmgKernel1D,
        foe_k_st: &DmgKernel1D,
    ) -> Self {
        let idx = side * 3 + slot;
        let foe_idx = foe_side * 3 + foe_slot;
        let mon = &cb.mons[idx];
        let foe_mon = &cb.mons[foe_idx];
        let spec = &table.mons[idx];
        let foe_spec = &table.mons[foe_idx];
        let maxhp = spec.maxhp.max(1);

        let dis = (mon.vol_flags & vol_bit::DISABLE) != 0;
        let dis_slot = ((mon.slot_refs >> 4) & 0x03) as usize;
        let mut has_half_heal = false;
        let mut has_rest = false;
        let mut has_st = false;
        for m_i in 0..spec.num_moves as usize {
            if mon.pp[m_i] > 0 && !(dis && m_i == dis_slot) {
                let ms = &table.move_specs[idx][m_i];
                if ms.is_half_heal {
                    has_half_heal = true;
                } else if ms.is_rest {
                    has_rest = true;
                }
                if ms.special_kind == 10 {
                    has_st = true;
                }
            }
        }
        let perish_active = (mon.vol_flags & vol_bit::PERISH_SONG) != 0;
        let has_berry = mon.item != 0 && (spec.is_mintberry || spec.is_miracleberry);
        let is_frz =
            mon.status_raw() == Status::Frz as u8 && !(mon.item != 0 && spec.is_miracleberry);
        let has_heal = !perish_active && !is_frz && (has_half_heal || has_rest);
        let heal_is_rest = !has_half_heal && has_rest;

        let mut half_heal_bin = [0u8; B16 + 1];
        for b in 1..=B16 {
            let hp = Self::b16_to_hp(b, maxhp);
            let healed = (hp + maxhp / 2).min(maxhp);
            half_heal_bin[b] = Self::to_b16(healed, maxhp);
        }

        let lefties = if mon.item != 0 && spec.is_leftovers {
            (maxhp / 16).max(1)
        } else {
            0
        };
        let leech_drain = if (mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
            (maxhp / 8).max(1)
        } else {
            0
        };
        let leech_gain = if (foe_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
            (foe_spec.maxhp.max(1) / 8).max(1)
        } else {
            0
        };
        let st = mon.status_raw();
        let cnt_packed = mon.status_counter();
        let is_tox = st == Status::Tox as u8
            || ((cnt_packed & 0x10) != 0 && (st == Status::Psn as u8 || st == Status::Brn as u8));
        let tox_c = ((cnt_packed & 0x0F) as i16).max(2);
        let status_dot = if is_tox {
            (maxhp / 16).max(1) * tox_c
        } else if st == Status::Brn as u8 || st == Status::Psn as u8 {
            (maxhp / 8).max(1)
        } else {
            0
        };
        let net_uncured = lefties + leech_gain - leech_drain - status_dot;
        let net_cured = lefties + leech_gain - leech_drain;

        let mut eot_bin = [[0u8; B16 + 1]; 2];
        for b in 1..=B16 {
            let hp = Self::b16_to_hp(b, maxhp);
            eot_bin[0][b] = Self::to_b16((hp + net_uncured).clamp(0, maxhp), maxhp);
            eot_bin[1][b] = Self::to_b16((hp + net_cured).clamp(0, maxhp), maxhp);
        }

        let foe_miracle = foe_mon.item != 0 && foe_spec.is_miracleberry;
        let foe_frz = foe_mon.status_raw() == Status::Frz as u8 && !foe_miracle;
        let foe_par = foe_mon.status_raw() == Status::Par as u8 && !foe_miracle;
        let foe_conf = (foe_mon.vol_flags & vol_bit::CONFUSION) != 0 && !foe_miracle;
        let p_foe_act = (if foe_frz { 25.0 / 256.0 } else { 1.0f32 })
            * (if foe_par { 0.75f32 } else { 1.0f32 })
            * (if foe_conf { 0.50f32 } else { 1.0f32 });

        let mut incoming = [[BinDist16::identity(0); B16 + 1]; 3];
        for b in 1..=B16 {
            incoming[0][b] = BinDist16::identity(b as u8);
            let hp = Self::b16_to_hp(b, maxhp);

            for (kind, kern) in [(1usize, foe_k_att), (2usize, foe_k_st)] {
                let mut d = BinDist16 {
                    bins: [(0, 0.0); 8],
                    len: 0,
                    ko_prob: 0.0,
                };
                if kern.max_dmg <= 0 {
                    d.push(b as u8, 1.0);
                } else {
                    if p_foe_act < 1.0 {
                        d.push(b as u8, 1.0 - p_foe_act);
                    }
                    for i in 0..kern.len {
                        let (dmg, prob) = kern.pts[i];
                        let post_hp = (hp - dmg).max(0);
                        d.push(Self::to_b16(post_hp, maxhp), p_foe_act * prob);
                    }
                }
                incoming[kind][b] = d;
            }
        }

        Self {
            has_heal,
            heal_is_rest,
            has_rest,
            has_st,
            has_berry,
            half_heal_bin,
            eot_bin,
            incoming,
        }
    }
}

/// Rank-3 SVD and Schur-Bellman solution of a 1v1 active pair over `(b0, b1, s0, s1)`.
#[derive(Clone, Copy)]
pub struct SolvedBellmanSvd {
    pub key: u128,
    /// Singular values $\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$ of $V(b_0, b_1, 0, 0) - 0.5$.
    pub sigmas: [f32; 3],
    /// Left singular vectors $u_r(b_0)$ for $r \in \{0, 1, 2\}, b_0 \in 1..=16$.
    pub u_vecs: [[f32; B16]; 3],
    /// Right singular vectors $v_r(b_1)$ for $r \in \{0, 1, 2\}, b_1 \in 1..=16$.
    pub v_vecs: [[f32; B16]; 3],
    /// Fraction of Frobenius energy $\|\sum_{r=1}^3 \sigma_r u_r v_r^\top\|_F^2 / \|V - 0.5\|_F^2 \in [0, 1]$.
    pub rank3_energy: f32,
    /// Full solved Bellman value table $V^*(s_0, s_1, b_0, b_1) \in [0, 1]$.
    pub v_table: [[[[f32; B16 + 1]; B16 + 1]; 3]; 3],
    /// Optimal awake heal bitmask for Side 0 (`(heal_mask0[b0] >> b1) & 1 == 1` iff $Q_{\text{Heal}} > Q_{\text{Att}}$).
    pub heal_mask0: [u32; B16 + 1],
    /// Optimal awake heal bitmask for Side 1 (`(heal_mask1[b1] >> b0) & 1 == 1` iff $Q_{\text{Heal}} > Q_{\text{Att}}$).
    pub heal_mask1: [u32; B16 + 1],
}

impl SolvedBellmanSvd {
    const fn empty() -> Self {
        Self {
            key: 0,
            sigmas: [0.0; 3],
            u_vecs: [[0.0; B16]; 3],
            v_vecs: [[0.0; B16]; 3],
            rank3_energy: 1.0,
            v_table: [[[[0.5; B16 + 1]; B16 + 1]; 3]; 3],
            heal_mask0: [0; B16 + 1],
            heal_mask1: [0; B16 + 1],
        }
    }

    /// Evaluate the Rank-3 SVD approximation of $V(b_0, b_1, 0, 0) \in [0, 1]$ at grid bins $b_0, b_1 \in 1..=16$.
    #[inline]
    pub fn eval_svd_awake(&self, b0: usize, b1: usize) -> f32 {
        if b0 == 0 && b1 == 0 {
            return 0.5;
        }
        if b0 == 0 {
            return 0.0;
        }
        if b1 == 0 {
            return 1.0;
        }
        let i = (b0 - 1).min(B16 - 1);
        let j = (b1 - 1).min(B16 - 1);
        let mut s = 0.5f32;
        for r in 0..3 {
            s += self.sigmas[r] * self.u_vecs[r][i] * self.v_vecs[r][j];
        }
        s.clamp(0.0, 1.0)
    }

    /// Bilinear interpolation of $V^*(s_0, s_1, h_0, h_1) \in [0, 1]$ at exact HP values `(hp0, hp1)`.
    #[inline]
    pub fn eval_interp(&self, hp0: i16, max0: i16, s0: usize, hp1: i16, max1: i16, s1: usize) -> f64 {
        if hp0 <= 0 && hp1 <= 0 {
            return 0.5;
        }
        if hp0 <= 0 {
            return 0.0;
        }
        if hp1 <= 0 {
            return 1.0;
        }
        let s0 = s0.min(2);
        let s1 = s1.min(2);
        let x0 = ((hp0 as f32) * (B16 as f32) / (max0.max(1) as f32)).clamp(0.0, B16 as f32);
        let x1 = ((hp1 as f32) * (B16 as f32) / (max1.max(1) as f32)).clamp(0.0, B16 as f32);

        let b0_lo = (x0.floor() as usize).clamp(1, B16);
        let b0_hi = (x0.ceil() as usize).clamp(1, B16);
        let w0 = (x0 - (b0_lo as f32)).clamp(0.0, 1.0);

        let b1_lo = (x1.floor() as usize).clamp(1, B16);
        let b1_hi = (x1.ceil() as usize).clamp(1, B16);
        let w1 = (x1 - (b1_lo as f32)).clamp(0.0, 1.0);

        let tbl = &self.v_table[s0][s1];
        let v00 = tbl[b0_lo][b1_lo];
        let v01 = tbl[b0_lo][b1_hi];
        let v10 = tbl[b0_hi][b1_lo];
        let v11 = tbl[b0_hi][b1_hi];

        let v0 = v00 + w1 * (v01 - v00);
        let v1 = v10 + w1 * (v11 - v10);
        (v0 + w0 * (v1 - v0)).clamp(0.0, 1.0) as f64
    }

    /// Evaluate a single exact-HP Bellman stage-game backup from `(hp0, hp1, s0, s1)` into `v_table`.
    /// Eliminates first-turn $1/16$-bin quantization error on exact KO thresholds while preserving
    /// exact antisymmetry.
    #[inline]
    pub fn eval_exact_root(
        &self,
        cb: &CompactBattle,
        table: &MatchTable,
        s0: usize,
        slot0: usize,
        s1: usize,
        slot1: usize,
    ) -> f64 {
        let idx = [s0 * 3 + slot0, s1 * 3 + slot1];
        let hp = [cb.mons[idx[0]].hp(), cb.mons[idx[1]].hp()];
        if hp[0] <= 0 && hp[1] <= 0 {
            return 0.5;
        }
        if hp[0] <= 0 {
            return 0.0;
        }
        if hp[1] <= 0 {
            return 1.0;
        }
        let maxhp = [
            table.mons[idx[0]].maxhp.max(1),
            table.mons[idx[1]].maxhp.max(1),
        ];
        let slp = [
            side_sleep_rem(cb, table, s0, slot0),
            side_sleep_rem(cb, table, s1, slot1),
        ];

        let k_att = [
            DmgKernel1D::best_attack(cb, table, s0, slot0, s1, slot1),
            DmgKernel1D::best_attack(cb, table, s1, slot1, s0, slot0),
        ];
        let k_st = [
            DmgKernel1D::sleep_talk(cb, table, s0, slot0, s1, slot1),
            DmgKernel1D::sleep_talk(cb, table, s1, slot1, s0, slot0),
        ];

        let mut has_heal = [false; 2];
        let mut heal_is_rest = [false; 2];
        let mut has_rest = [false; 2];
        let mut has_st = [false; 2];
        let mut has_berry = [false; 2];
        let mut p_act = [1.0f32; 2];
        let mut eot_net = [[0i16; 2]; 2];

        for i in 0..2 {
            let m = &cb.mons[idx[i]];
            let fm = &cb.mons[idx[1 - i]];
            let sp = &table.mons[idx[i]];
            let fsp = &table.mons[idx[1 - i]];
            let dis = (m.vol_flags & vol_bit::DISABLE) != 0;
            let dis_slot = ((m.slot_refs >> 4) & 0x03) as usize;
            let mut hh = false;
            let mut hr = false;
            let mut hst = false;
            for m_i in 0..sp.num_moves as usize {
                if m.pp[m_i] > 0 && !(dis && m_i == dis_slot) {
                    let ms = &table.move_specs[idx[i]][m_i];
                    if ms.is_half_heal {
                        hh = true;
                    } else if ms.is_rest {
                        hr = true;
                    }
                    if ms.special_kind == 10 {
                        hst = true;
                    }
                }
            }
            let perish = (m.vol_flags & vol_bit::PERISH_SONG) != 0;
            let miracle = m.item != 0 && sp.is_miracleberry;
            let berry = m.item != 0 && (sp.is_mintberry || sp.is_miracleberry);
            let st_raw = m.status_raw();
            let is_frz = st_raw == Status::Frz as u8 && !miracle;
            let is_par = st_raw == Status::Par as u8 && !miracle;
            let is_conf =
                (m.vol_flags & vol_bit::CONFUSION) != 0 && m.timers[0] > 1 && !miracle;
            let is_rech = (m.slot_refs & 0x80) != 0;

            has_heal[i] = !perish && !is_frz && !is_rech && (hh || hr);
            heal_is_rest[i] = !hh && hr;
            has_rest[i] = hr;
            has_st[i] = hst;
            has_berry[i] = berry;
            p_act[i] = if is_rech {
                0.0
            } else {
                (if is_frz { 25.0 / 256.0 } else { 1.0f32 })
                    * (if is_par { 0.75f32 } else { 1.0f32 })
                    * (if is_conf { 0.50f32 } else { 1.0f32 })
            };

            let lefties = if m.item != 0 && sp.is_leftovers {
                (maxhp[i] / 16).max(1)
            } else {
                0
            };
            let leech_drain = if (m.vol_flags & vol_bit::LEECH_SEED) != 0 {
                (maxhp[i] / 8).max(1)
            } else {
                0
            };
            let leech_gain = if (fm.vol_flags & vol_bit::LEECH_SEED) != 0 {
                (fsp.maxhp.max(1) / 8).max(1)
            } else {
                0
            };
            let cnt_packed = m.status_counter();
            let is_tox = st_raw == Status::Tox as u8
                || ((cnt_packed & 0x10) != 0
                    && (st_raw == Status::Psn as u8 || st_raw == Status::Brn as u8));
            let tox_c = ((cnt_packed & 0x0F) as i16 + 1).max(2);
            let status_dot = if is_tox {
                (maxhp[i] / 16).max(1) * tox_c
            } else if st_raw == Status::Brn as u8 || st_raw == Status::Psn as u8 {
                (maxhp[i] / 8).max(1)
            } else {
                0
            };
            eot_net[i][0] = lefties + leech_gain - leech_drain - status_dot;
            eot_net[i][1] = lefties + leech_gain - leech_drain;
        }

        let eval_order = |first: usize, a0: u8, a1: u8| -> f32 {
            let second = 1 - first;
            let act = [a0, a1];
            let win_first = if first == 0 { 1.0f32 } else { 0.0f32 };
            let win_second = 1.0f32 - win_first;

            let (hp_fir_mid, ns_fir, cured_fir, k_fir_opt) = if slp[first] > 0 {
                (
                    hp[first],
                    slp[first] - 1,
                    has_rest[first],
                    if has_st[first] {
                        Some(&k_st[first])
                    } else {
                        None
                    },
                )
            } else if act[first] == 1 {
                if heal_is_rest[first] {
                    (
                        maxhp[first],
                        if has_berry[first] { 0 } else { 2 },
                        true,
                        None,
                    )
                } else {
                    (
                        (hp[first] + maxhp[first] / 2).min(maxhp[first]),
                        0,
                        false,
                        None,
                    )
                }
            } else if act[first] == 2 {
                (hp[first], 0, false, None)
            } else {
                (hp[first], 0, false, Some(&k_att[first]))
            };

            let mut sec_pts = [(0i16, 0.0f32); 17];
            let mut sec_len = 0usize;
            let mut ko_sec_prob = 0.0f32;
            let push_pt =
                |pts: &mut [(i16, f32); 17], len: &mut usize, ko: &mut f32, h: i16, p: f32| {
                    if p <= 0.0 {
                        return;
                    }
                    if h <= 0 {
                        *ko += p;
                        return;
                    }
                    for idx_p in 0..*len {
                        if pts[idx_p].0 == h {
                            pts[idx_p].1 += p;
                            return;
                        }
                    }
                    if *len < 17 {
                        pts[*len] = (h, p);
                        *len += 1;
                    } else {
                        pts[*len - 1].1 += p;
                    }
                };

            let pa_fir = p_act[first];
            if let Some(kern) = k_fir_opt {
                if kern.max_dmg <= 0 || pa_fir <= 0.0 {
                    push_pt(&mut sec_pts, &mut sec_len, &mut ko_sec_prob, hp[second], 1.0);
                } else {
                    if pa_fir < 1.0 {
                        push_pt(
                            &mut sec_pts,
                            &mut sec_len,
                            &mut ko_sec_prob,
                            hp[second],
                            1.0 - pa_fir,
                        );
                    }
                    for i in 0..kern.len {
                        let (dmg, prob) = kern.pts[i];
                        push_pt(
                            &mut sec_pts,
                            &mut sec_len,
                            &mut ko_sec_prob,
                            (hp[second] - dmg).max(0),
                            pa_fir * prob,
                        );
                    }
                }
            } else {
                push_pt(&mut sec_pts, &mut sec_len, &mut ko_sec_prob, hp[second], 1.0);
            }

            let mut total_val = ko_sec_prob * win_first;

            for &(hp_sec_mid, p_sec) in &sec_pts[..sec_len] {
                let (hp_sec_post, ns_sec, cured_sec, k_sec_opt) = if slp[second] > 0 {
                    (
                        hp_sec_mid,
                        slp[second] - 1,
                        has_rest[second],
                        if has_st[second] {
                            Some(&k_st[second])
                        } else {
                            None
                        },
                    )
                } else if act[second] == 1 {
                    if heal_is_rest[second] {
                        (
                            maxhp[second],
                            if has_berry[second] { 0 } else { 2 },
                            true,
                            None,
                        )
                    } else {
                        (
                            (hp_sec_mid + maxhp[second] / 2).min(maxhp[second]),
                            0,
                            false,
                            None,
                        )
                    }
                } else if act[second] == 2 {
                    (hp_sec_mid, 0, false, None)
                } else {
                    (hp_sec_mid, 0, false, Some(&k_att[second]))
                };

                let hp_sec_end =
                    (hp_sec_post + eot_net[second][cured_sec as usize]).clamp(0, maxhp[second]);

                let mut fir_pts = [(0i16, 0.0f32); 17];
                let mut fir_len = 0usize;
                let mut ko_fir_prob = 0.0f32;
                let pa_sec = p_act[second];
                if let Some(kern) = k_sec_opt {
                    if kern.max_dmg <= 0 || pa_sec <= 0.0 {
                        push_pt(
                            &mut fir_pts,
                            &mut fir_len,
                            &mut ko_fir_prob,
                            hp_fir_mid,
                            1.0,
                        );
                    } else {
                        if pa_sec < 1.0 {
                            push_pt(
                                &mut fir_pts,
                                &mut fir_len,
                                &mut ko_fir_prob,
                                hp_fir_mid,
                                1.0 - pa_sec,
                            );
                        }
                        for i in 0..kern.len {
                            let (dmg, prob) = kern.pts[i];
                            push_pt(
                                &mut fir_pts,
                                &mut fir_len,
                                &mut ko_fir_prob,
                                (hp_fir_mid - dmg).max(0),
                                pa_sec * prob,
                            );
                        }
                    }
                } else {
                    push_pt(
                        &mut fir_pts,
                        &mut fir_len,
                        &mut ko_fir_prob,
                        hp_fir_mid,
                        1.0,
                    );
                }

                let mut branch_val = ko_fir_prob * win_second;
                for &(hp_fir_post, p_fir) in &fir_pts[..fir_len] {
                    let hp_fir_end =
                        (hp_fir_post + eot_net[first][cured_fir as usize]).clamp(0, maxhp[first]);
                    let cont = if hp_fir_end <= 0 && hp_sec_end <= 0 {
                        0.5f32
                    } else if hp_sec_end <= 0 {
                        win_first
                    } else if hp_fir_end <= 0 {
                        win_second
                    } else {
                        let (nh0, nh1, ns0, ns1) = if first == 0 {
                            (hp_fir_end, hp_sec_end, ns_fir, ns_sec)
                        } else {
                            (hp_sec_end, hp_fir_end, ns_sec, ns_fir)
                        };
                        let v_cont =
                            self.eval_interp(nh0, maxhp[0], ns0, nh1, maxhp[1], ns1) as f32;
                        GAMMA * v_cont + (1.0 - GAMMA) * 0.5
                    };
                    branch_val += p_fir * cont;
                }
                total_val += p_sec * branch_val;
            }
            total_val.clamp(0.0, 1.0)
        };

        let spd0 = cb.mon_action_speed(table, s0, slot0);
        let spd1 = cb.mon_action_speed(table, s1, slot1);
        let n0 = if slp[0] == 0 && has_heal[0] && hp[0] < maxhp[0] {
            2
        } else {
            1
        };
        let n1 = if slp[1] == 0 && has_heal[1] && hp[1] < maxhp[1] {
            2
        } else {
            1
        };
        let eval_pair = |a0: u8, a1: u8| -> f32 {
            if spd0 > spd1 {
                eval_order(0, a0, a1)
            } else if spd1 > spd0 {
                eval_order(1, a0, a1)
            } else {
                0.5 * (eval_order(0, a0, a1) + eval_order(1, a0, a1))
            }
        };
        let mut q = [[0.5f32; 2]; 2];
        for a0 in 0..n0 {
            let p0_ok = if a0 == 1 { p_act[0] } else { 1.0 };
            for a1 in 0..n1 {
                let p1_ok = if a1 == 1 { p_act[1] } else { 1.0 };
                let mut cell = p0_ok * p1_ok * eval_pair(a0 as u8, a1 as u8);
                if p0_ok < 1.0 {
                    cell += (1.0 - p0_ok) * p1_ok * eval_pair(2, a1 as u8);
                }
                if p1_ok < 1.0 {
                    cell += p0_ok * (1.0 - p1_ok) * eval_pair(a0 as u8, 2);
                }
                if p0_ok < 1.0 && p1_ok < 1.0 {
                    cell += (1.0 - p0_ok) * (1.0 - p1_ok) * eval_pair(2, 2);
                }
                q[a0][a1] = cell;
            }
        }
        let (val, _, _) = solve_stage_game_2x2(&q, n0, n1);
        val.clamp(0.0, 1.0) as f64
    }
}

const CACHE_SETS: usize = 256;
const CACHE_WAYS: usize = 4;

struct BellmanCache {
    sets: Vec<[SolvedBellmanSvd; CACHE_WAYS]>,
    hand: [u8; CACHE_SETS],
}

impl BellmanCache {
    fn new() -> Self {
        Self {
            sets: vec![[SolvedBellmanSvd::empty(); CACHE_WAYS]; CACHE_SETS],
            hand: [0u8; CACHE_SETS],
        }
    }
}

thread_local! {
    static BELLMAN_CACHE: RefCell<BellmanCache> = RefCell::new(BellmanCache::new());
}

#[inline]
fn pair_bellman_key(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
) -> u128 {
    let idx0 = s0 * 3 + slot0;
    let idx1 = s1 * 3 + slot1;
    let m0 = &cb.mons[idx0];
    let m1 = &cb.mons[idx1];
    let sp0 = &table.mons[idx0];
    let sp1 = &table.mons[idx1];

    let mut h: u128 = 0x6c62272e07bb014262b821756295c58d;
    let mut mix = |v: u64| {
        h ^= v as u128;
        h = h.wrapping_mul(0x0000000001000000000000000000013b);
    };
    mix(sp0.species.0 as u64);
    mix(sp1.species.0 as u64);
    mix(sp0.level as u64);
    mix(sp1.level as u64);
    mix(sp0.maxhp as u64);
    mix(sp1.maxhp as u64);
    mix(m0.item as u64);
    mix(m1.item as u64);
    for st_i in 0..5 {
        mix(cb.mon_stat(table, s0, slot0, st_i, false, false) as u64);
        mix(cb.mon_stat(table, s0, slot0, st_i, true, true) as u64);
        mix(cb.mon_stat(table, s1, slot1, st_i, false, false) as u64);
        mix(cb.mon_stat(table, s1, slot1, st_i, true, true) as u64);
    }
    mix(cb.mon_action_speed(table, s0, slot0) as u64);
    mix(cb.mon_action_speed(table, s1, slot1) as u64);
    for i in 0..7 {
        mix((m0.boosts[i] as u8) as u64);
        mix((m1.boosts[i] as u8) as u64);
    }
    let st0_has_slp = (m0.status_raw() == Status::Slp as u8) as u64;
    let st1_has_slp = (m1.status_raw() == Status::Slp as u8) as u64;
    let st0_non_slp = if m0.status_raw() == Status::Slp as u8 {
        0
    } else {
        m0.status_raw()
    };
    let st1_non_slp = if m1.status_raw() == Status::Slp as u8 {
        0
    } else {
        m1.status_raw()
    };
    mix(st0_has_slp);
    mix(st1_has_slp);
    mix(st0_non_slp as u64);
    mix(st1_non_slp as u64);
    mix(if st0_non_slp != 0 {
        (m0.status_counter() & 0x1F) as u64
    } else {
        0
    });
    mix(if st1_non_slp != 0 {
        (m1.status_counter() & 0x1F) as u64
    } else {
        0
    });
    mix((m0.vol_flags
        & (vol_bit::LEECH_SEED
            | vol_bit::CONFUSION
            | vol_bit::DISABLE
            | vol_bit::FORESIGHT
            | vol_bit::PERISH_SONG)) as u64);
    mix((m1.vol_flags
        & (vol_bit::LEECH_SEED
            | vol_bit::CONFUSION
            | vol_bit::DISABLE
            | vol_bit::FORESIGHT
            | vol_bit::PERISH_SONG)) as u64);
    mix(((m0.slot_refs >> 4) & 0x03) as u64);
    mix(((m1.slot_refs >> 4) & 0x03) as u64);
    mix((cb.sides[s0].alive_mask.count_ones() <= 1) as u64);
    mix((cb.sides[s1].alive_mask.count_ones() <= 1) as u64);
    mix((cb.sides[s0].reflect_turns > 0) as u64);
    mix((cb.sides[s0].lightscreen_turns > 0) as u64);
    mix((cb.sides[s1].reflect_turns > 0) as u64);
    mix((cb.sides[s1].lightscreen_turns > 0) as u64);
    mix(cb.weather_kind as u64);
    mix((cb.weather_turns > 0) as u64);
    for m_i in 0..4 {
        mix(table.move_specs[idx0][m_i].id.0 as u64);
        mix((m0.pp[m_i] > 0) as u64);
        mix(table.move_specs[idx1][m_i].id.0 as u64);
        mix((m1.pp[m_i] > 0) as u64);
    }
    h | 1
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_ordered_turn(
    first: usize,
    b0: usize,
    b1: usize,
    s0: usize,
    s1: usize,
    act0: u8,
    act1: u8,
    ops: &[Side1DOps; 2],
    v_table: &[[[[f32; B16 + 1]; B16 + 1]; 3]; 3],
) -> f32 {
    let second = 1 - first;
    let b = [b0, b1];
    let s = [s0, s1];
    let act = [act0, act1];

    let win_first = if first == 0 { 1.0f32 } else { 0.0f32 };
    let win_second = 1.0f32 - win_first;

    // Phase 1: `first` acts
    let (b_fir_mid, ns_fir, cured_fir, k_fir) = if s[first] > 0 {
        (
            b[first],
            s[first] - 1,
            ops[first].has_rest,
            if ops[first].has_st { 2usize } else { 0usize },
        )
    } else if act[first] == 1 {
        if ops[first].heal_is_rest {
            (
                B16,
                if ops[first].has_berry { 0 } else { 2 },
                true,
                0usize,
            )
        } else {
            (
                ops[first].half_heal_bin[b[first]] as usize,
                0,
                false,
                0usize,
            )
        }
    } else {
        (b[first], 0, false, 1usize)
    };

    let dist_sec = &ops[second].incoming[k_fir][b[second]];
    let mut total_val = dist_sec.ko_prob * win_first;

    for i_sec in 0..dist_sec.len {
        let (b_sec_mid_u8, p_sec) = dist_sec.bins[i_sec];
        let b_sec_mid = b_sec_mid_u8 as usize;

        // Phase 2: `second` acts at `b_sec_mid`
        let (b_sec_post, ns_sec, cured_sec, k_sec) = if s[second] > 0 {
            (
                b_sec_mid,
                s[second] - 1,
                ops[second].has_rest,
                if ops[second].has_st { 2usize } else { 0usize },
            )
        } else if act[second] == 1 {
            if ops[second].heal_is_rest {
                (
                    B16,
                    if ops[second].has_berry { 0 } else { 2 },
                    true,
                    0usize,
                )
            } else {
                (
                    ops[second].half_heal_bin[b_sec_mid] as usize,
                    0,
                    false,
                    0usize,
                )
            }
        } else {
            (b_sec_mid, 0, false, 1usize)
        };

        let b_sec_end = ops[second].eot_bin[cured_sec as usize][b_sec_post] as usize;
        let dist_fir = &ops[first].incoming[k_sec][b_fir_mid];
        let mut branch_val = dist_fir.ko_prob * win_second;

        for i_fir in 0..dist_fir.len {
            let (b_fir_post_u8, p_fir) = dist_fir.bins[i_fir];
            let b_fir_end =
                ops[first].eot_bin[cured_fir as usize][b_fir_post_u8 as usize] as usize;

            let cont = if b_fir_end == 0 && b_sec_end == 0 {
                0.5f32
            } else if b_sec_end == 0 {
                win_first
            } else if b_fir_end == 0 {
                win_second
            } else {
                let (nb0, nb1, ns0, ns1) = if first == 0 {
                    (b_fir_end, b_sec_end, ns_fir, ns_sec)
                } else {
                    (b_sec_end, b_fir_end, ns_sec, ns_fir)
                };
                GAMMA * v_table[ns0][ns1][nb0][nb1] + (1.0 - GAMMA) * 0.5
            };
            branch_val += p_fir * cont;
        }
        total_val += p_sec * branch_val;
    }

    total_val.clamp(0.0, 1.0)
}

#[inline]
fn solve_stage_game_2x2(q: &[[f32; 2]; 2], n0: usize, n1: usize) -> (f32, bool, bool) {
    if n0 == 1 && n1 == 1 {
        return (q[0][0], false, false);
    }
    if n0 == 2 && n1 == 1 {
        let heal0 = q[1][0] > q[0][0] + 1e-5;
        return (q[0][0].max(q[1][0]), heal0, false);
    }
    if n0 == 1 && n1 == 2 {
        let heal1 = q[0][1] < q[0][0] - 1e-5;
        return (q[0][0].min(q[0][1]), false, heal1);
    }
    let r0 = q[0][0].min(q[0][1]);
    let r1 = q[1][0].min(q[1][1]);
    let maximin = r0.max(r1);

    let c0 = q[0][0].max(q[1][0]);
    let c1 = q[0][1].max(q[1][1]);
    let minimax = c0.min(c1);

    let heal0 = r1 > r0 + 1e-5;
    let heal1 = c1 < c0 - 1e-5;

    if (minimax - maximin).abs() <= 1e-5 {
        return (0.5 * (maximin + minimax), heal0, heal1);
    }
    let denom = q[0][0] + q[1][1] - q[0][1] - q[1][0];
    if denom.abs() <= 1e-6 {
        return (0.5 * (maximin + minimax), heal0, heal1);
    }
    let val = ((q[0][0] * q[1][1] - q[0][1] * q[1][0]) / denom).clamp(maximin, minimax);
    (val, heal0, heal1)
}

fn compute_svd_rank3_16x16(
    awake: &[[f32; B16 + 1]; B16 + 1],
) -> ([f32; 3], [[f32; B16]; 3], [[f32; B16]; 3], f32) {
    let mut res = [[0.0f32; B16]; B16];
    let mut total_sq = 0.0f32;
    for i in 0..B16 {
        for j in 0..B16 {
            let m = awake[i + 1][j + 1] - 0.5;
            res[i][j] = m;
            total_sq += m * m;
        }
    }
    if total_sq <= 1e-10 {
        return ([0.0; 3], [[0.0; B16]; 3], [[0.0; B16]; 3], 1.0);
    }

    let mut sigmas = [0.0f32; 3];
    let mut u_vecs = [[0.0f32; B16]; 3];
    let mut v_vecs = [[0.0f32; B16]; 3];
    let mut captured_sq = 0.0f32;

    for r in 0..3 {
        let mut v = [0.25f32; B16];
        for (j, vj) in v.iter_mut().enumerate() {
            if r == 1 {
                *vj = ((j as f32) - 7.5) * 0.05;
            } else if r == 2 {
                *vj = if j % 2 == 0 { 0.25 } else { -0.25 };
            }
        }
        let mut u = [0.0f32; B16];
        let mut sigma = 0.0f32;

        for _it in 0..20 {
            let mut u_norm = 0.0f32;
            for i in 0..B16 {
                let mut s = 0.0f32;
                for j in 0..B16 {
                    s += res[i][j] * v[j];
                }
                u[i] = s;
                u_norm += s * s;
            }
            u_norm = u_norm.sqrt();
            if u_norm <= 1e-8 {
                sigma = 0.0;
                break;
            }
            let inv_u = 1.0 / u_norm;
            for ui in &mut u {
                *ui *= inv_u;
            }

            let mut v_norm = 0.0f32;
            for j in 0..B16 {
                let mut s = 0.0f32;
                for i in 0..B16 {
                    s += res[i][j] * u[i];
                }
                v[j] = s;
                v_norm += s * s;
            }
            sigma = v_norm.sqrt();
            if sigma <= 1e-8 {
                break;
            }
            let inv_v = 1.0 / sigma;
            for vj in &mut v {
                *vj *= inv_v;
            }
        }
        sigmas[r] = sigma;
        u_vecs[r] = u;
        v_vecs[r] = v;
        captured_sq += sigma * sigma;
        for i in 0..B16 {
            for j in 0..B16 {
                res[i][j] -= sigma * u[i] * v[j];
            }
        }
    }

    let energy = (captured_sq / total_sq).clamp(0.0, 1.0);
    (sigmas, u_vecs, v_vecs, energy)
}

/// Solve (or retrieve from thread-local cache) the 2D Coupled Schur-Complement Bellman fixed point
/// and its Rank-3 SVD over the $16 \times 16 \times 3 \times 3$ state space.
pub fn solve_schur_bellman_2d(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
) -> SolvedBellmanSvd {
    with_solved_bellman(cb, table, s0, slot0, s1, slot1, |s| *s)
}

#[inline]
fn with_solved_bellman<R>(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
    f: impl FnOnce(&SolvedBellmanSvd) -> R,
) -> R {
    let key = pair_bellman_key(cb, table, s0, slot0, s1, slot1);
    let set_idx = ((key as usize) ^ ((key >> 32) as usize)) & (CACHE_SETS - 1);
    let hit_opt = BELLMAN_CACHE.with(|c| {
        let borrow = c.borrow();
        let set = &borrow.sets[set_idx];
        for way in 0..CACHE_WAYS {
            if set[way].key == key {
                return Some(way);
            }
        }
        None
    });
    if let Some(way) = hit_opt {
        return BELLMAN_CACHE.with(|c| {
            let borrow = c.borrow();
            f(&borrow.sets[set_idx][way])
        });
    }

    let k01_att = DmgKernel1D::best_attack_grid(cb, table, s0, slot0, s1, slot1);
    let k10_att = DmgKernel1D::best_attack_grid(cb, table, s1, slot1, s0, slot0);
    let k01_st = DmgKernel1D::sleep_talk_grid(cb, table, s0, slot0, s1, slot1);
    let k10_st = DmgKernel1D::sleep_talk_grid(cb, table, s1, slot1, s0, slot0);

    let ops0 = Side1DOps::build(cb, table, s0, slot0, s1, slot1, &k10_att, &k10_st);
    let ops1 = Side1DOps::build(cb, table, s1, slot1, s0, slot0, &k01_att, &k01_st);
    let ops = [ops0, ops1];

    let max_s0 = if ops0.has_rest || cb.mons[s0 * 3 + slot0].status_raw() == Status::Slp as u8 {
        2usize
    } else {
        0usize
    };
    let max_s1 = if ops1.has_rest || cb.mons[s1 * 3 + slot1].status_raw() == Status::Slp as u8 {
        2usize
    } else {
        0usize
    };
    let max_s_sum = max_s0 + max_s1;

    let spd0 = cb.mon_action_speed(table, s0, slot0);
    let spd1 = cb.mon_action_speed(table, s1, slot1);

    let mut v_work = [[[[0.5f32; B16 + 1]; B16 + 1]; 3]; 3];
    for slp0 in 0..3 {
        for slp1 in 0..3 {
            for b in 1..=B16 {
                v_work[slp0][slp1][0][b] = 0.0;
                v_work[slp0][slp1][b][0] = 1.0;
            }
            v_work[slp0][slp1][0][0] = 0.5;
        }
    }

    let mut heal_mask0 = [0u32; B16 + 1];
    let mut heal_mask1 = [0u32; B16 + 1];

    let eval_cell = |b0: usize,
                     b1: usize,
                     slp0: usize,
                     slp1: usize,
                     a0: u8,
                     a1: u8,
                     vt: &[[[[f32; B16 + 1]; B16 + 1]; 3]; 3]|
     -> f32 {
        if spd0 > spd1 {
            eval_ordered_turn(0, b0, b1, slp0, slp1, a0, a1, &ops, vt)
        } else if spd1 > spd0 {
            eval_ordered_turn(1, b0, b1, slp0, slp1, a0, a1, &ops, vt)
        } else {
            0.5 * (eval_ordered_turn(0, b0, b1, slp0, slp1, a0, a1, &ops, vt)
                + eval_ordered_turn(1, b0, b1, slp0, slp1, a0, a1, &ops, vt))
        }
    };

    for _iter in 0..18 {
        // 1. Bottom-up Schur-complement elimination over the sleeping DAG `s_sum = 1..=max_s_sum`
        for s_sum in 1..=max_s_sum {
            let mut next_level = v_work;
            for slp0 in 0..=max_s0 {
                for slp1 in 0..=max_s1 {
                    if slp0 + slp1 != s_sum {
                        continue;
                    }
                    for b0 in 1..=B16 {
                        let n0 = if slp0 == 0 && ops[0].has_heal && b0 < B16 {
                            2
                        } else {
                            1
                        };
                        for b1 in 1..=B16 {
                            let n1 = if slp1 == 0 && ops[1].has_heal && b1 < B16 {
                                2
                            } else {
                                1
                            };
                            let mut q = [[0.5f32; 2]; 2];
                            for a0 in 0..n0 {
                                for a1 in 0..n1 {
                                    q[a0][a1] =
                                        eval_cell(b0, b1, slp0, slp1, a0 as u8, a1 as u8, &v_work);
                                }
                            }
                            let (val, _, _) = solve_stage_game_2x2(&q, n0, n1);
                            next_level[slp0][slp1][b0][b1] = val;
                        }
                    }
                }
            }
            for slp0 in 0..=max_s0 {
                for slp1 in 0..=max_s1 {
                    if slp0 + slp1 == s_sum {
                        v_work[slp0][slp1] = next_level[slp0][slp1];
                    }
                }
            }
        }

        // 2. Awake stage game `(slp0, slp1) = (0, 0)` using the Schur-eliminated sleeping values
        let mut next_awake = v_work[0][0];
        let mut max_diff = 0.0f32;
        let mut hm0 = [0u32; B16 + 1];
        let mut hm1 = [0u32; B16 + 1];

        for b0 in 1..=B16 {
            let n0 = if ops[0].has_heal && b0 < B16 { 2 } else { 1 };
            for b1 in 1..=B16 {
                let n1 = if ops[1].has_heal && b1 < B16 { 2 } else { 1 };
                let mut q = [[0.5f32; 2]; 2];
                for a0 in 0..n0 {
                    for a1 in 0..n1 {
                        q[a0][a1] = eval_cell(b0, b1, 0, 0, a0 as u8, a1 as u8, &v_work);
                    }
                }
                let (val, h0, h1) = solve_stage_game_2x2(&q, n0, n1);
                if h0 {
                    hm0[b0] |= 1u32 << b1;
                }
                if h1 {
                    hm1[b1] |= 1u32 << b0;
                }
                let d = (val - v_work[0][0][b0][b1]).abs();
                if d > max_diff {
                    max_diff = d;
                }
                next_awake[b0][b1] = val;
            }
        }
        v_work[0][0] = next_awake;
        heal_mask0 = hm0;
        heal_mask1 = hm1;
        if _iter >= 4 && max_diff < 2e-4 {
            break;
        }
    }

    let (sigmas, u_vecs, v_vecs, rank3_energy) = compute_svd_rank3_16x16(&v_work[0][0]);
    let solved = SolvedBellmanSvd {
        key,
        sigmas,
        u_vecs,
        v_vecs,
        rank3_energy,
        v_table: v_work,
        heal_mask0,
        heal_mask1,
    };

    BELLMAN_CACHE.with(|c| {
        let mut borrow = c.borrow_mut();
        let w = (borrow.hand[set_idx] as usize) & (CACHE_WAYS - 1);
        borrow.sets[set_idx][w] = solved;
        borrow.hand[set_idx] = (w as u8 + 1) & (CACHE_WAYS as u8 - 1);
        f(&borrow.sets[set_idx][w])
    })
}

#[inline]
fn side_sleep_rem(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> usize {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let spec = &table.mons[idx];
    let berry = mon.item != 0 && (spec.is_mintberry || spec.is_miracleberry);
    if mon.status_raw() == Status::Slp as u8 && !berry {
        (mon.status_counter().saturating_sub(1) as usize).min(2)
    } else {
        0
    }
}

#[inline]
fn side_has_usable_heal(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> bool {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    if (mon.vol_flags & vol_bit::PERISH_SONG) != 0 {
        return false;
    }
    let dis = (mon.vol_flags & vol_bit::DISABLE) != 0;
    let dis_slot = ((mon.slot_refs >> 4) & 0x03) as usize;
    for m_i in 0..table.mons[idx].num_moves as usize {
        if mon.pp[m_i] > 0 && !(dis && m_i == dis_slot) {
            let ms = &table.move_specs[idx][m_i];
            if ms.is_half_heal || ms.is_rest {
                return true;
            }
        }
    }
    false
}

/// Evaluate the 1v1 race & endurance differential in `[-1.0, 1.0]` between `(s0, slot0)` and `(s1, slot1)`:
/// - If neither side has a usable heal/Rest move, evaluates the exact stopping-time bilinear form
///   $\sum_{k=1}^6 (f_1(k) S_0(k) - f_0(k) S_1(k) + f_0(k) f_1(k) \text{spd\_sign}) + S_0(6) S_1(6) (\bar{h}_0 - \bar{h}_1)$
///   directly from the 1D Toeplitz survival curves $T_0^k \mathbf{1}$ and $T_1^k \mathbf{1}$.
/// - If either side has a usable heal/Rest move, solves the 2D Coupled Schur-Complement Bellman
///   equation $(I - \gamma P_{\pi^*}) V^* = r_{\pi^*}$ and returns $2 V^*(h_0, h_1, s_0, s_1) - 1$.
#[inline]
pub fn toeplitz_coupled_race(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
) -> f64 {
    let idx0 = s0 * 3 + slot0;
    let idx1 = s1 * 3 + slot1;
    let hp0 = cb.mons[idx0].hp();
    let hp1 = cb.mons[idx1].hp();
    if hp0 <= 0 || hp1 <= 0 {
        return 0.0;
    }

    if side_has_usable_heal(cb, table, s0, slot0) || side_has_usable_heal(cb, table, s1, slot1) {
        let v = with_solved_bellman(cb, table, s0, slot0, s1, slot1, |solved| {
            solved.eval_exact_root(cb, table, s0, slot0, s1, slot1)
        });
        return (2.0 * v - 1.0).clamp(-1.0, 1.0);
    }

    // Pure stopping-time race from the 1D Toeplitz survival curves S_0(k), S_1(k)
    let (s_curve_0, rem0) = survival_curve_6_with_mean_hp(cb, table, s1, s0, hp0);
    let (s_curve_1, rem1) = survival_curve_6_with_mean_hp(cb, table, s0, s1, hp1);

    let spd0 = cb.mon_action_speed(table, s0, slot0);
    let spd1 = cb.mon_action_speed(table, s1, slot1);
    let spd_sign = if spd0 > spd1 {
        1.0
    } else if spd1 > spd0 {
        -1.0
    } else {
        0.0
    };

    let mut margin = 0.0f64;
    for k in 1..=6usize {
        let f0 = (s_curve_0[k - 1] - s_curve_0[k]).max(0.0);
        let f1 = (s_curve_1[k - 1] - s_curve_1[k]).max(0.0);
        margin += f1 * s_curve_0[k] - f0 * s_curve_1[k] + f0 * f1 * spd_sign;
    }
    margin += s_curve_0[6] * s_curve_1[6] * (rem0 - rem1);
    margin.clamp(-1.0, 1.0)
}

/// Perish Song / Mean Look countdown survival probability differential in `[-1.0, 1.0]`
/// computed directly from the Toeplitz survival curve $S(k) = (\mathbf{1}^\top T^k \mathbf{e}_{h_0})$.
#[inline]
pub fn compact_perish_margin(cb: &CompactBattle, table: &MatchTable) -> f64 {
    let slot0 = cb.sides[0].active_idx as usize;
    let slot1 = cb.sides[1].active_idx as usize;
    if (cb.sides[0].alive_mask & (1 << slot0)) == 0 || (cb.sides[1].alive_mask & (1 << slot1)) == 0
    {
        return 0.0;
    }
    let idx0 = slot0;
    let idx1 = 3 + slot1;
    let hp0 = cb.mons[idx0].hp();
    let hp1 = cb.mons[idx1].hp();
    if hp0 <= 0 || hp1 <= 0 {
        return 0.0;
    }

    let p0_perish = if (cb.mons[idx0].vol_flags & vol_bit::PERISH_SONG) != 0 {
        cb.mons[idx0].timers[3]
    } else {
        0
    };
    let p1_perish = if (cb.mons[idx1].vol_flags & vol_bit::PERISH_SONG) != 0 {
        cb.mons[idx1].timers[3]
    } else {
        0
    };
    let p0_trapped = (cb.mons[idx0].vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0
        || cb.sides[0].alive_mask.count_ones() <= 1;
    let p1_trapped = (cb.mons[idx1].vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0
        || cb.sides[1].alive_mask.count_ones() <= 1;

    if p0_perish == 0 && p1_perish == 0 && p0_trapped == p1_trapped {
        return 0.0;
    }

    let s0 = survival_curve_4(cb, table, 1, 0, hp0);
    let s1 = survival_curve_4(cb, table, 0, 1, hp1);
    let mut margin = 0.0f64;

    if p1_perish > 0 && p1_trapped && !p0_trapped {
        let hits = p1_perish.saturating_sub(1).min(3) as usize;
        let p0_prot = cb.mons[idx0].timers[2] == 0
            && (0..table.mons[idx0].num_moves as usize)
                .any(|m| cb.mons[idx0].pp[m] > 0 && table.move_specs[idx0][m].is_protect);
        let p1_phaze = (0..table.mons[idx1].num_moves as usize)
            .any(|m| cb.mons[idx1].pp[m] > 0 && table.move_specs[idx1][m].special_kind == 7);
        let eff_hits = if p0_prot && hits > 0 { hits - 1 } else { hits };
        let p_not_phazed = if p1_phaze && p1_perish >= 2 {
            1.0 - s1[1]
        } else {
            1.0
        };
        margin += s0[eff_hits] * p_not_phazed;
    }

    if p0_perish > 0 && p0_trapped && !p1_trapped {
        let hits = p0_perish.saturating_sub(1).min(3) as usize;
        let p1_prot = cb.mons[idx1].timers[2] == 0
            && (0..table.mons[idx1].num_moves as usize)
                .any(|m| cb.mons[idx1].pp[m] > 0 && table.move_specs[idx1][m].is_protect);
        let p0_phaze = (0..table.mons[idx0].num_moves as usize)
            .any(|m| cb.mons[idx0].pp[m] > 0 && table.move_specs[idx0][m].special_kind == 7);
        let eff_hits = if p1_prot && hits > 0 { hits - 1 } else { hits };
        let p_not_phazed = if p0_phaze && p0_perish >= 2 {
            1.0 - s0[1]
        } else {
            1.0
        };
        margin -= s1[eff_hits] * p_not_phazed;
    }

    // Trapped before Perish Song is sung: probability trapper survives 1 extra turn to sing + countdown
    if p1_perish == 0 && p1_trapped && !p0_trapped {
        let p0_has_song = (0..table.mons[idx0].num_moves as usize)
            .any(|m| cb.mons[idx0].pp[m] > 0 && table.move_specs[idx0][m].special_kind == 19);
        if p0_has_song {
            let p0_prot = (0..table.mons[idx0].num_moves as usize)
                .any(|m| cb.mons[idx0].pp[m] > 0 && table.move_specs[idx0][m].is_protect);
            let hits = if p0_prot { 2 } else { 3 };
            margin += s0[hits];
        }
    }
    if p0_perish == 0 && p0_trapped && !p1_trapped {
        let p1_has_song = (0..table.mons[idx1].num_moves as usize)
            .any(|m| cb.mons[idx1].pp[m] > 0 && table.move_specs[idx1][m].special_kind == 19);
        if p1_has_song {
            let p1_prot = (0..table.mons[idx1].num_moves as usize)
                .any(|m| cb.mons[idx1].pp[m] > 0 && table.move_specs[idx1][m].is_protect);
            let hits = if p1_prot { 2 } else { 3 };
            margin -= s1[hits];
        }
    }

    margin.clamp(-1.0, 1.0)
}

/// Combined linear-algebra differential added to `compact_eval01` (scaled by `w.scale`).
/// Strictly antisymmetric: `compact_linear_diff(cb) == -compact_linear_diff(swapped(cb))`.
#[inline]
pub fn compact_linear_diff(
    cb: &CompactBattle,
    table: &MatchTable,
    base_diff: f64,
    old_race_margin: f64,
    race_weight: f64,
    scale: f64,
) -> f64 {
    let mut delta = compact_perish_margin(cb, table);

    let n0 = cb.sides[0].alive_mask.count_ones();
    let n1 = cb.sides[1].alive_mask.count_ones();
    if n0 == 0 || n1 == 0 {
        return delta;
    }
    let s0 = cb.sides[0].active_idx as usize;
    let s1 = cb.sides[1].active_idx as usize;
    if (cb.sides[0].alive_mask & (1 << s0)) == 0
        || (cb.sides[1].alive_mask & (1 << s1)) == 0
        || race_weight <= 0.0
    {
        return delta;
    }

    if n0 == 1 && n1 == 1 {
        let tr = toeplitz_coupled_race(cb, table, 0, s0, 1, s1);
        let m0 = &cb.mons[s0];
        let m1 = &cb.mons[3 + s1];
        let spd0 = cb.mon_action_speed(table, 0, s0);
        let spd1 = cb.mon_action_speed(table, 1, s1);
        let is_frz = m0.status_raw() == Status::Frz as u8 || m1.status_raw() == Status::Frz as u8;
        let is_rech = ((m0.slot_refs | m1.slot_refs) & 0x80) != 0;
        let exact_mechanics_override = spd0 == spd1
            || m0.status_raw() == Status::Slp as u8
            || m1.status_raw() == Status::Slp as u8
            || is_frz
            || is_rech
            || (m0.vol_flags & vol_bit::CONFUSION) != 0
            || (m1.vol_flags & vol_bit::CONFUSION) != 0
            || old_race_margin == 0.0
            || (tr * old_race_margin > 0.0);

        if exact_mechanics_override {
            if (tr.abs() < 0.99 || is_rech) && !is_frz && scale > 0.0 {
                let p_lin = (0.5 * (1.0 + tr)).clamp(0.005, 0.995);
                let logit_tr =
                    ((p_lin / (1.0 - p_lin)).ln() / scale).clamp(-race_weight, race_weight);
                delta += logit_tr - base_diff;
            } else {
                delta += race_weight * (tr - old_race_margin);
            }
        }
    } else if n0 + n1 == 3
        && old_race_margin == 0.0
        && ((cb.mons[s0].vol_flags | cb.mons[3 + s1].vol_flags) & vol_bit::PERISH_SONG) == 0
    {
        // Monotone zero-sum saddle-point value V_mat(A_0, A_1) over alive pairs (a, d):
        // Guarantees V_mat(A_0', A_1) >= V_mat(A_0, A_1) whenever A_0' is a superset of A_0.
        let mut mine = [0usize; 3];
        let mut k0 = 0usize;
        let mut theirs = [0usize; 3];
        let mut k1 = 0usize;
        for slot in 0..3usize {
            if (cb.sides[0].alive_mask & (1 << slot)) != 0 && cb.mons[slot].hp() > 0 {
                mine[k0] = slot;
                k0 += 1;
            }
            if (cb.sides[1].alive_mask & (1 << slot)) != 0 && cb.mons[3 + slot].hp() > 0 {
                theirs[k1] = slot;
                k1 += 1;
            }
        }
        if k0 > 0 && k1 > 0 {
            let mut mat = [[0.0f64; 3]; 3];
            for (i, &a) in mine[..k0].iter().enumerate() {
                for (j, &d) in theirs[..k1].iter().enumerate() {
                    mat[i][j] = toeplitz_coupled_race(cb, table, 0, a, 1, d);
                }
            }
            let mut maximin = f64::NEG_INFINITY;
            for row in &mat[..k0] {
                let mut rmin = f64::INFINITY;
                for &v in &row[..k1] {
                    if v < rmin {
                        rmin = v;
                    }
                }
                if rmin > maximin {
                    maximin = rmin;
                }
            }
            let mut minimax = f64::INFINITY;
            for j in 0..k1 {
                let mut cmax = f64::NEG_INFINITY;
                for row in &mat[..k0] {
                    if row[j] > cmax {
                        cmax = row[j];
                    }
                }
                if cmax < minimax {
                    minimax = cmax;
                }
            }
            let v_mat = 0.5 * (maximin + minimax);
            delta += 0.5 * race_weight * v_mat;
        }
    }

    delta
}

/// Bellman-Markov rollout action selector (zero magic thresholds):
/// 1. When guaranteed asleep this turn (`status_counter >= 2` and no Mint/Miracle Berry),
///    returns the canonical sleeping Markov action (`Sleep Talk` / `Snore` if available).
/// 2. When awake and holding a heal/Rest move, queries the solved 2D Schur-Bellman fixed point
///    at `(b0, b1)` and selects `Heal/Rest` if and only if $Q_{\text{Heal}}(b_0, b_1) > Q_{\text{Attack}}(b_0, b_1)$.
#[inline]
pub fn compact_rollout_linear_pick(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    cs: &[u8],
) -> Option<u8> {
    let att_slot = cb.sides[side].active_idx as usize;
    let att_idx = side * 3 + att_slot;
    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;
    let def_idx = def_side * 3 + def_slot;

    if cb.sides[side].request != 1
        || (cb.sides[side].alive_mask & (1 << att_slot)) == 0
        || (cb.sides[def_side].alive_mask & (1 << def_slot)) == 0
    {
        return None;
    }

    let me = &cb.mons[att_idx];
    let foe = &cb.mons[def_idx];
    let maxhp = table.mons[att_idx].maxhp.max(1);

    // 1. Canonical sleeping Markov action when guaranteed asleep this turn (`status_counter >= 2`)
    let is_asleep_this_turn = me.status_raw() == Status::Slp as u8
        && me.status_counter() >= 2
        && !(me.item != 0
            && (table.mons[att_idx].is_mintberry || table.mons[att_idx].is_miracleberry));
    if is_asleep_this_turn {
        for &c in cs {
            if c < 4 && table.move_specs[att_idx][c as usize].special_kind == 10 {
                return Some(c);
            }
        }
        for &c in cs {
            if c < 4 && table.move_specs[att_idx][c as usize].is_sleep_usable {
                return Some(c);
            }
        }
        return None;
    }

    // 2. Protect during active Perish countdown (`timers[3] >= 2`) when trapping
    let me_perish = if (me.vol_flags & vol_bit::PERISH_SONG) != 0 {
        me.timers[3]
    } else {
        0
    };
    let foe_perish = if (foe.vol_flags & vol_bit::PERISH_SONG) != 0 {
        foe.timers[3]
    } else {
        0
    };
    let me_trapped = (me.vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0
        || cb.sides[side].alive_mask.count_ones() <= 1;
    let foe_trapped = (foe.vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0
        || cb.sides[def_side].alive_mask.count_ones() <= 1;

    if foe_perish >= 2 && foe_trapped && !me_trapped && me.timers[2] == 0 {
        for &c in cs {
            if c < 4 && table.move_specs[att_idx][c as usize].is_protect {
                return Some(c);
            }
        }
    }

    // 3. Bellman stage-game policy lookup for Heal / Rest when awake and strictly below full HP
    if me_perish == 0 && me.hp() < maxhp {
        let foe_has_perish = (0..table.mons[def_idx].num_moves as usize)
            .any(|m| foe.pp[m] > 0 && table.move_specs[def_idx][m].special_kind == 19);
        if !foe_has_perish {
            let mut heal_act: Option<u8> = None;
            for &c in cs {
                if c < 4 {
                    let ms = &table.move_specs[att_idx][c as usize];
                    if ms.is_half_heal {
                        heal_act = Some(c);
                        break;
                    } else if ms.is_rest && heal_act.is_none() {
                        heal_act = Some(c);
                    }
                }
            }
            if let Some(h_act) = heal_act {
                let s0_slot = cb.sides[0].active_idx as usize;
                let s1_slot = cb.sides[1].active_idx as usize;
                let b0 = Side1DOps::to_b16(cb.mons[s0_slot].hp(), table.mons[s0_slot].maxhp.max(1))
                    as usize;
                let b1 = Side1DOps::to_b16(
                    cb.mons[3 + s1_slot].hp(),
                    table.mons[3 + s1_slot].maxhp.max(1),
                ) as usize;
                if (side == 0 && b0 < B16) || (side == 1 && b1 < B16) {
                    let bellman_wants_heal =
                        with_solved_bellman(cb, table, 0, s0_slot, 1, s1_slot, |solved| {
                            if side == 0 {
                                ((solved.heal_mask0[b0] >> b1) & 1) != 0
                            } else {
                                ((solved.heal_mask1[b1] >> b0) & 1) != 0
                            }
                        });
                    if bellman_wants_heal {
                        return Some(h_act);
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;
    use nc2000_engine::state::Battle;

    fn make_set(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
        serde_json::from_value(serde_json::json!({
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

    #[test]
    fn schur_bellman_svd_is_antisymmetric_and_low_rank() {
        let dex = conformance::load_dex();
        let lax = make_set(
            "Snorlax",
            55,
            "Leftovers",
            &["Body Slam", "Earthquake", "Curse", "Self-Destruct"],
        );
        let skarm = make_set(
            "Skarmory",
            55,
            "Leftovers",
            &["Drill Peck", "Whirlwind", "Rest", "Toxic"],
        );

        let mut b01 =
            Battle::from_fixture(&dex, "1,2,3,4", &[lax.clone()], &[skarm.clone()]).unwrap();
        b01.choose(&dex, 0, "team 1").unwrap();
        b01.choose(&dex, 1, "team 1").unwrap();
        let (t01, cb01) = CompactBattle::from_battle(&b01, &dex);

        let mut b10 = Battle::from_fixture(&dex, "1,2,3,4", &[skarm], &[lax]).unwrap();
        b10.choose(&dex, 0, "team 1").unwrap();
        b10.choose(&dex, 1, "team 1").unwrap();
        let (t10, cb10) = CompactBattle::from_battle(&b10, &dex);

        let svd01 = solve_schur_bellman_2d(&cb01, &t01, 0, 0, 1, 0);
        let svd10 = solve_schur_bellman_2d(&cb10, &t10, 0, 0, 1, 0);

        // Rank-3 SVD must capture > 98% of Frobenius energy
        assert!(
            svd01.rank3_energy > 0.98,
            "expected Rank-3 SVD energy > 98%, got {}",
            svd01.rank3_energy
        );

        // Check exact antisymmetry across the entire 16x16x3x3 Bellman table
        for s0 in 0..3 {
            for s1 in 0..3 {
                for b0 in 1..=B16 {
                    for b1 in 1..=B16 {
                        let v01 = svd01.v_table[s0][s1][b0][b1];
                        let v10 = svd10.v_table[s1][s0][b1][b0];
                        assert!(
                            (v01 + v10 - 1.0).abs() < 1e-5,
                            "antisymmetry violated at ({b0},{b1},{s0},{s1}): {v01} + {v10} != 1.0"
                        );
                    }
                }
            }
        }

        let r01 = toeplitz_coupled_race(&cb01, &t01, 0, 0, 1, 0);
        let r10 = toeplitz_coupled_race(&cb10, &t10, 0, 0, 1, 0);
        assert!((r01 + r10).abs() < 1e-5, "antisymmetry violated: {r01} vs {r10}");
        assert!(
            r01 < -0.25,
            "expected Skarmory Rest to be favored over Rest-less Snorlax, got {r01}"
        );
    }

    #[test]
    fn rollout_linear_pick_uses_bellman_policy_and_sleep_talk() {
        let dex = conformance::load_dex();
        let suicune = make_set(
            "Suicune",
            55,
            "Leftovers",
            &["Surf", "Ice Beam", "Rest", "Sleep Talk"],
        );
        let raikou = make_set(
            "Raikou",
            55,
            "Leftovers",
            &["Thunderbolt", "Crunch", "Rest", "Sleep Talk"],
        );
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &[suicune], &[raikou]).unwrap();
        b.choose(&dex, 0, "team 1").unwrap();
        b.choose(&dex, 1, "team 1").unwrap();
        let (table, mut cb) = CompactBattle::from_battle(&b, &dex);

        // At 100% HP awake, Bellman policy must NEVER choose Rest (`None`)
        let cs = [0u8, 1, 2, 3];
        assert_eq!(compact_rollout_linear_pick(&cb, &table, 0, &cs), None);

        // Put Suicune to sleep with status_counter == 2 (guaranteed asleep this turn) -> Sleep Talk (slot 3)
        cb.mons[0].set_status(Status::Slp as u8, 2);
        assert_eq!(compact_rollout_linear_pick(&cb, &table, 0, &cs), Some(3));

        // At status_counter == 1, Suicune wakes up this turn at 100% HP -> `None` (Attack)
        cb.mons[0].set_status(Status::Slp as u8, 1);
        assert_eq!(compact_rollout_linear_pick(&cb, &table, 0, &cs), None);
    }
}
