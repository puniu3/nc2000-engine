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
use std::cmp::Ordering;

use nc2000_engine::compact::{
    role_bit, vol_bit, CompactBattle, CompactItemSpec, CompactMoveSpec, CompactWeather, MatchTable,
    VarBpKind,
};
use nc2000_engine::rules::Ruleset;
use nc2000_engine::state::Status;

/// Canonical Gen 2 HP grid resolution ($1/16 = 6.25\%$, matching Leftovers $1/16$,
/// Burn/Poison/Leech Seed $2/16$, Substitute $4/16$, and Recover/Soft-Boiled $8/16$).
pub const B16: usize = 16;

/// Sleep-state table dimension: `[0..=3]` covers awake (`0`), PS 2–4T (`0..=2`), and Castle
/// uniform 1–4T non-Rest sleep (`status_counter == 5` -> `slp == 4` -> next state `3`).
const MAX_SLP_STATES: usize = 4;

type BellmanVTable = [[[[f32; B16 + 1]; B16 + 1]; MAX_SLP_STATES]; MAX_SLP_STATES];

/// Fine 1D grid resolution for pure stopping-time convolutions without recovery.
const B32: usize = 32;

/// Discount factor $\gamma$ pulling infinite stall cycles toward the tie value $0.5$.
const GAMMA: f32 = 0.985;

#[inline]
fn instant_heal_amount(
    cb: &CompactBattle,
    ruleset: Ruleset,
    ms: &CompactMoveSpec,
    maxhp: i16,
) -> Option<i16> {
    if ms.is_half_heal {
        // PS `moveexec.rs:2311` rounds odd maxhp up (`round(maxhp / 2.0)`);
        // Castle PBS (`Probe 9A`, 113/113 logs) strictly floors (`floor(maxhp / 2)`).
        if ruleset.is_castle() {
            Some(maxhp / 2)
        } else {
            Some((maxhp + 1) / 2)
        }
    } else if ms.special_kind == nc2000_engine::compact::SpecialMoveKind::WeatherHeal {
        let weather = if cb.weather_turns > 0 {
            cb.weather_kind
        } else {
            CompactWeather::None
        };
        Some(weather.weather_heal_amount(maxhp))
    } else {
        None
    }
}

#[inline]
fn heal_berry_amount(ispec: &CompactItemSpec) -> i16 {
    if ispec.is_goldberry {
        30
    } else if ispec.is_berry_juice {
        20
    } else if ispec.is_berry {
        10
    } else {
        0
    }
}

#[inline]
fn heal_berry_triggers(ruleset: Ruleset, hp: i16, maxhp: i16) -> bool {
    if hp <= 0 {
        return false;
    }
    if ruleset.is_castle() {
        // Castle PBS (`E_20261028`, 275/275 logs): strictly `hp < floor(maxhp / 2)`.
        hp < maxhp / 2
    } else {
        hp <= maxhp / 2
    }
}

#[inline]
fn is_toxic_residual(ruleset: Ruleset, status_raw: u8, cnt_packed: u8) -> bool {
    (cnt_packed & 0x10) != 0
        && (status_raw == Status::Tox as u8
            || status_raw == Status::Psn as u8
            || (!ruleset.is_castle() && status_raw == Status::Brn as u8))
}

#[inline]
fn speed_ord_u8(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
) -> u8 {
    match crate::eval::compare_action_speed(cb, table, s0, slot0, s1, slot1) {
        Ordering::Less => 0,
        Ordering::Equal => 1,
        Ordering::Greater => 2,
    }
}

#[inline]
fn has_after_move_self_slip(m: &nc2000_engine::compact::CompactMon) -> bool {
    let st = m.status_raw();
    st == Status::Brn as u8
        || st == Status::Psn as u8
        || st == Status::Tox as u8
        || (m.vol_flags & (vol_bit::LEECH_SEED | vol_bit::CURSE)) != 0
        || (m.partially_trapped()
            && m.ext_timers[nc2000_engine::compact::ext_timer_idx::PARTIAL_TRAP_TURNS] > 1)
        || ((m.vol_flags & vol_bit::NIGHTMARE) != 0
            && (st == Status::Slp as u8
                || (m.ext_flags & nc2000_engine::compact::ext_vol_bit::MIRACLE_NIGHTMARE) != 0))
}

/// Side-`s0` payoff on a 1v1 mutual residual/EOT KO:
/// - `PsStadium2`: faster side wins (`turn.rs:341`), `0.5` on speed tie.
/// - `CastlePbs`:
///   - `AfterMoveSelf` slip damage (Burn/Poison/Toxic/Leech Seed/Curse/PartialTrap/Nightmare)
///     ticks during the move phase BEFORE EOT Sandstorm:
///     - If only one side has `AfterMoveSelf` slip, that side faints first (loses).
///     - If both sides have `AfterMoveSelf` slip, the faster mover ticks and faints first
///       (`0.0` if `s0` faster, `1.0` if `s1` faster, `0.5` on tie).
///   - Only when NEITHER side has `AfterMoveSelf` slip and both take Sandstorm does mutual
///     Sandstorm KO resolve to `0.5` (Draw, `Probe 2B`).
#[inline]
fn mutual_eot_payoff(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
    spd_ord: u8,
) -> f32 {
    if table.ruleset.is_ps() {
        return match spd_ord {
            2 => 1.0,
            0 => 0.0,
            _ => 0.5,
        };
    }
    let idx0 = s0 * 3 + slot0;
    let idx1 = s1 * 3 + slot1;
    let ams0 = has_after_move_self_slip(&cb.mons[idx0]);
    let ams1 = has_after_move_self_slip(&cb.mons[idx1]);
    if ams0 && !ams1 {
        return 0.0;
    }
    if !ams0 && ams1 {
        return 1.0;
    }
    if ams0 && ams1 {
        return match spd_ord {
            2 => 0.0,
            0 => 1.0,
            _ => 0.5,
        };
    }
    let sand_active = cb.weather_turns > 0 && cb.weather_kind == CompactWeather::Sandstorm;
    let both_sand = sand_active
        && (table.mons[idx0].type_mask & table.sandstorm_immune_mask) == 0
        && (table.mons[idx1].type_mask & table.sandstorm_immune_mask) == 0;
    if both_sand {
        0.5
    } else {
        match spd_ord {
            2 => 0.0,
            0 => 1.0,
            _ => 0.5,
        }
    }
}

/// Stack-allocated 1D damage distribution kernel (up to 16 `(dmg, prob)` support points).
#[derive(Clone, Copy, Debug)]
pub struct DmgKernel1D {
    pub pts: [(i16, f32); 16],
    pub len: usize,
    pub max_dmg: i16,
    pub mean_dmg: f32,
    pub recoil_quarter: bool,
}

impl DmgKernel1D {
    #[inline]
    pub const fn zero() -> Self {
        Self {
            pts: [(0, 1.0); 16],
            len: 1,
            max_dmg: 0,
            mean_dmg: 0.0,
            recoil_quarter: false,
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
        let last_mon = cb.sides[att_side].alive_mask.count_ones() <= 1;

        let mut best_kern = Self::zero();
        let mut best_score = 0.0f32;
        for m_i in 0..num_moves {
            if att_mon.pp[m_i] == 0 || att_mon.is_slot_disabled(m_i) {
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
        let def_mon = &cb.mons[def_idx];
        let def_maxhp = table.mons[def_idx].maxhp.max(1);
        let num_moves = table.mons[att_idx].num_moves as usize;
        let last_mon = cb.sides[att_side].alive_mask.count_ones() <= 1;
        let def_has_sub = (def_mon.vol_flags & vol_bit::SUBSTITUTE) != 0 && def_mon.sub_hp() > 0;

        let mut best_kern = Self::zero();
        let mut best_score = 0.0f32;
        let mut has_sub_breaker = !def_has_sub;
        for m_i in 0..num_moves {
            if att_mon.pp[m_i] == 0 || att_mon.is_slot_disabled(m_i) {
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
            if !mspec.drain_half || mspec.bypasses_sub {
                has_sub_breaker = true;
            }
            let score = kern.mean_dmg / (def_maxhp as f32);
            if score > best_score {
                best_score = score;
                best_kern = kern;
            }
        }
        if !has_sub_breaker {
            return Self::zero();
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

        let mut st_slot = None;
        for m_i in 0..num_moves {
            if att_mon.pp[m_i] > 0
                && !att_mon.is_slot_disabled(m_i)
                && table.move_specs[att_idx][m_i].special_kind
                    == nc2000_engine::compact::SpecialMoveKind::SleepTalk
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
            recoil_quarter: false,
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

        // Gen 2: HP-draining moves (Giga Drain, Mega Drain, Absorb, Leech Life, Dream Eater)
        // fail completely against an active Substitute.
        if !grid_mode
            && mspec.drain_half
            && !mspec.bypasses_sub
            && (def_mon.vol_flags & vol_bit::SUBSTITUTE) != 0
            && def_mon.sub_hp() > 0
        {
            return Self::zero();
        }

        let base_power = if mspec.var_bp_kind == VarBpKind::FlailReversal {
            let att_hp = if grid_mode {
                att_spec.maxhp / 2
            } else {
                att_mon.hp()
            };
            crate::eval::ruleset_flail_bp(table.ruleset, att_hp as i32, att_spec.maxhp as i32)
        } else if table.ruleset.is_castle()
            && matches!(mspec.var_bp_kind, VarBpKind::Return | VarBpKind::Frustration)
        {
            102
        } else {
            mspec.base_power as i32
        };

        let is_super_fang =
            mspec.special_kind == nc2000_engine::compact::SpecialMoveKind::SuperFang;
        if mspec.fixed_damage == 0 && !is_super_fang && (mspec.is_status || base_power <= 0) {
            return Self::zero();
        }
        let def_foresight = (def_mon.vol_flags & vol_bit::FORESIGHT) != 0;
        let (type_mod, is_immune) =
            table.effectiveness_ext(mspec.move_type, def_spec, def_foresight);
        if is_immune {
            return Self::zero();
        }

        let p_hit = crate::eval::ruleset_hit_probability(
            table.ruleset,
            mspec.base_acc_pct,
            mspec.always_hit,
            att_mon.boosts[5],
            def_mon.boosts[6],
            mspec.ignore_accuracy,
            mspec.ignore_evasion,
            mspec.ignore_positive_evasion,
            def_foresight,
            table.item_spec(def_mon.item).is_brightpowder,
        ) as f32;
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
                recoil_quarter: mspec.recoil_quarter,
            };
            if p_hit < 1.0 {
                k.push_mass(0, 1.0 - p_hit);
            }
            k.push_mass(fdmg, p_hit);
            return k;
        }

        let hits_mult = (mspec.multihit_mean_x2 as f32 * 0.5).max(1.0);
        const CRIT_DEN: [u32; 6] = [0, 16, 8, 4, 3, 2];
        let scope_bonus = table.item_spec(att_mon.item).is_scope_lens as usize;
        let c_stage = (mspec.crit_ratio as usize + scope_bonus).min(5);
        let p_crit = if mspec.special_kind
            != nc2000_engine::compact::SpecialMoveKind::FlailReversal
            && c_stage > 0
        {
            if table.ruleset.is_castle() && c_stage == 1 {
                17.0 / 256.0
            } else {
                1.0 / (CRIT_DEN[c_stage] as f32)
            }
        } else {
            0.0
        };

        let is_castle = table.ruleset.is_castle();
        let compute_base = |is_crit: bool| -> i32 {
            let atk_stat = if mspec.is_physical { 0 } else { 2 };
            let def_stat = if mspec.is_physical { 1 } else { 3 };
            let ignore_boosts = is_crit & (att_mon.boosts[atk_stat] <= def_mon.boosts[def_stat]);
            let mut atk = crate::eval::ruleset_mon_stat(
                cb,
                table,
                att_side,
                att_slot,
                atk_stat,
                None,
                ignore_boosts,
                ignore_boosts,
            );
            let mut def = crate::eval::ruleset_mon_stat(
                cb,
                table,
                def_side,
                def_slot,
                def_stat,
                None,
                ignore_boosts,
                false,
            );
            if is_castle {
                // Castle PBS (`Probe 12E2`, #19): Self-Destruct / Explosion halves Defense BEFORE `>= 256 / 4`.
                if mspec.is_selfdestruct && def_stat == 1 {
                    def = (def / 2).max(1);
                }
                if (atk | def) >= 256 {
                    atk = (atk >> 2).max(1);
                    def = (def >> 2).max(1);
                }
            } else {
                if (atk | def) >= 256 {
                    atk = (atk >> 2).max(1);
                    def = (def >> 2).max(1);
                }
                if mspec.is_selfdestruct && def_stat == 1 {
                    def = (def / 2).max(1);
                }
            }
            let level = att_spec.level as i32;
            let mut d = ((level * 2 / 5 + 2) * base_power * atk / def.max(1)) / 50;
            if is_crit {
                d *= 2;
            }
            let item_boost_type = if is_castle {
                table.item_spec(att_mon.item).castle_boost_type
            } else {
                table.item_spec(att_mon.item).boost_type
            };
            if mspec.move_type != nc2000_engine::compact::NO_TYPE_ID
                && item_boost_type == mspec.move_type
            {
                d = d * 110 / 100;
            }
            d = d.min(997) + 2;
            if cb.weather_turns > 0 {
                if cb.weather_kind == CompactWeather::Rain {
                    if mspec.is_water {
                        d = d * 3 / 2;
                    } else if mspec.is_fire || mspec.is_solarbeam {
                        d /= 2;
                    }
                } else if cb.weather_kind == CompactWeather::Sun {
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
            if is_castle {
                d.max(0)
            } else {
                d.max(1)
            }
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
            recoil_quarter: mspec.recoil_quarter,
        };
        if p_hit < 1.0 {
            k.push_mass(0, 1.0 - p_hit);
        }
        let min_roll_dmg = if is_castle { 0 } else { 1 };
        let p_norm_roll = p_hit * (1.0 - p_crit) * 0.25;
        for &r in &ROLL_NUMS {
            let dmg = (((base_norm * r / 255).max(min_roll_dmg) as f32) * hits_mult).round() as i16;
            k.push_mass(dmg, p_norm_roll);
        }
        if p_crit > 0.0 {
            let p_crit_roll = p_hit * p_crit * 0.25;
            for &r in &ROLL_NUMS {
                let dmg =
                    (((base_crit * r / 255).max(min_roll_dmg) as f32) * hits_mult).round() as i16;
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
    let (s, _) = survival_curve_k_with_mean_hp(cb, table, att_side, def_side, start_hp, 3, false);
    [s[0], s[1], s[2], s[3]]
}

#[inline]
fn survival_curve_k_with_mean_hp(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    def_side: usize,
    start_hp: i16,
    max_steps: usize,
    need_mean_hp: bool,
) -> ([f64; 7], f64) {
    let att_slot = cb.sides[att_side].active_idx as usize;
    let def_slot = cb.sides[def_side].active_idx as usize;
    survival_curve_k_slots_with_mean_hp(
        cb,
        table,
        att_side,
        att_slot,
        def_side,
        def_slot,
        start_hp,
        max_steps,
        need_mean_hp,
    )
}

struct SurvivalCurveDetail {
    surv: [f64; 7],
    eot_ko: [f64; 7],
    mean_hp_frac: f64,
}

/// Compute `([S(0)..=S(k)], mean_hp_frac_at_k)` plus per-step EOT KO mass
/// via exact 1D Toeplitz convolution $T^k \mathbf{e}_{h_0}$.
#[allow(clippy::too_many_arguments)]
#[inline]
fn survival_curve_detail(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
    start_hp: i16,
    max_steps: usize,
    need_mean_hp: bool,
) -> SurvivalCurveDetail {
    if start_hp <= 0 {
        return SurvivalCurveDetail {
            surv: [0.0; 7],
            eot_ko: [0.0; 7],
            mean_hp_frac: 0.0,
        };
    }
    let max_steps = max_steps.min(6);
    if max_steps == 0 && !need_mean_hp {
        return SurvivalCurveDetail {
            surv: [1.0; 7],
            eot_ko: [0.0; 7],
            mean_hp_frac: 0.0,
        };
    }
    let att_idx = att_side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    let maxhp = table.mons[def_idx].maxhp.max(1);

    let def_mon = &cb.mons[def_idx];
    let att_mon = &cb.mons[att_idx];
    let def_spec = &table.mons[def_idx];
    let att_spec = &table.mons[att_idx];

    // Attacker sleep, freeze, paralysis, confusion, recharge
    let att_ispec = table.item_spec(att_mon.item);
    let def_ispec = table.item_spec(def_mon.item);
    let att_berry = att_ispec.is_mintberry || att_ispec.is_miracleberry;
    let att_miracle = att_ispec.is_miracleberry;
    let mut att_slp = if att_mon.status_raw() == Status::Slp as u8 && !att_berry {
        att_mon.status_counter().saturating_sub(1)
    } else {
        0
    };

    let kern_awake = DmgKernel1D::best_attack(cb, table, att_side, att_slot, def_side, def_slot);
    let kern_slp = if att_slp > 0 {
        DmgKernel1D::sleep_talk(cb, table, att_side, att_slot, def_side, def_slot)
    } else {
        DmgKernel1D::zero()
    };

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
    let mut min_b = b0;
    let mut max_b = b0;

    let def_heal_berry = heal_berry_amount(def_ispec);
    let is_castle = table.ruleset.is_castle();

    let lefties: i16 = if def_ispec.is_leftovers {
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
    let is_tox = is_toxic_residual(table.ruleset, def_st, cnt_packed);
    let mut tox_counter = (cnt_packed & 0x0F) as i16;
    let psn_brn_dmg: i16 =
        if def_st == Status::Brn as u8 || def_st == Status::Psn as u8 || def_st == Status::Tox as u8
        {
            (maxhp / 8).max(1)
        } else {
            0
        };
    let curse_dmg: i16 = if (def_mon.vol_flags & vol_bit::CURSE) != 0 {
        (maxhp / 4).max(1)
    } else {
        0
    };
    let mut weather_rem = cb.weather_turns;
    let sand_dmg: i16 = if cb.weather_kind == CompactWeather::Sandstorm
        && (def_spec.type_mask & table.sandstorm_immune_mask) == 0
    {
        (maxhp / 8).max(1)
    } else {
        0
    };

    // Defender confusion self-hit
    let def_miracle = def_ispec.is_miracleberry;
    let mut def_conf = if (def_mon.vol_flags & vol_bit::CONFUSION) != 0 && !def_miracle {
        def_mon.timers[nc2000_engine::compact::timer_idx::CONFUSION].saturating_sub(1)
    } else {
        0
    };
    let def_conf_dmg = if def_conf > 0 {
        let def_atk =
            crate::eval::ruleset_mon_stat(cb, table, def_side, def_slot, 0, None, false, false);
        let def_def =
            crate::eval::ruleset_mon_stat(cb, table, def_side, def_slot, 1, None, false, false);
        (((def_spec.level as i32 * 2 / 5 + 2) * 40 * def_atk / def_def.max(1)) / 50 + 2).max(1)
            as i16
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
        att_mon.timers[nc2000_engine::compact::timer_idx::CONFUSION].saturating_sub(1)
    } else {
        0
    };
    let mut att_rech = att_mon.must_recharge();
    let mut p_sub_up = if (def_mon.vol_flags & vol_bit::SUBSTITUTE) != 0 && def_mon.hp_pool[1] > 0 {
        1.0f32
    } else {
        0.0f32
    };
    let mut rem_sub_hp = if p_sub_up > 0.0 {
        def_mon.sub_hp().max(1) as f32
    } else {
        0.0f32
    };

    let apply_eot_and_berry = |hp_post: i16, res: i16, step: usize| -> i16 {
        if hp_post <= 0 {
            return 0;
        }
        let mut h = hp_post;
        let mut berry_used = false;
        if step == 1 && !is_castle && def_heal_berry > 0 && heal_berry_triggers(table.ruleset, h, maxhp) {
            h = (h + def_heal_berry).min(maxhp);
            berry_used = true;
        }
        h = (h + res).clamp(0, maxhp);
        if step == 1
            && !berry_used
            && def_heal_berry > 0
            && heal_berry_triggers(table.ruleset, h, maxhp)
        {
            h = (h + def_heal_berry).min(maxhp);
        }
        h
    };

    let mut out = [1.0f64; 7];
    let mut eot_ko = [0.0f64; 7];
    for step in 1..=max_steps {
        if min_b > max_b {
            for st in step..=max_steps {
                out[st] = 0.0;
            }
            break;
        }
        let mut next = [0.0f32; B32 + 1];
        next[0] = dist[0];
        let mut next_min = B32 + 1;
        let mut next_max = 0usize;
        let mut step_eot_ko = 0.0f32;

        let status_dot = if is_tox {
            tox_counter = (tox_counter + 1).min(15);
            ((maxhp / 16) * tox_counter).max(1)
        } else {
            psn_brn_dmg
        };
        let sand_this_turn = if weather_rem > 0 {
            weather_rem -= 1;
            sand_dmg
        } else {
            0
        };
        let res_this_turn =
            lefties + leech_gain - leech_drain - status_dot - curse_dmg - sand_this_turn;

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
            p_still_frz *= 231.0 / 256.0;
            p_att *= 1.0 - p_still_frz;
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

        let p_sub_now = p_sub_up;
        if p_sub_now > 0.0 && p_att > 0.0 && kern.max_dmg > 0 {
            let mut p_break = 0.0f32;
            let mut exp_chip = 0.0f32;
            for i in 0..kern.len {
                let (dmg, prob) = kern.pts[i];
                if dmg as f32 >= rem_sub_hp {
                    p_break += prob;
                } else if dmg > 0 {
                    exp_chip += prob * (dmg as f32);
                }
            }
            rem_sub_hp = (rem_sub_hp - p_att * exp_chip).max(1.0);
            p_sub_up = (p_sub_up * (1.0 - p_att * p_break)).clamp(0.0, 1.0);
        }

        for b in min_b..=max_b {
            let p_state = dist[b];
            if p_state <= 1e-7 {
                continue;
            }
            let raw_hp = b_to_hp(b, step);

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
                    let final_hp = apply_eot_and_berry(hp_after_self, res_this_turn, step);
                    let tb = to_b(final_hp);
                    next[tb] += p_no_act;
                    if tb == 0 {
                        step_eot_ko += p_no_act;
                    } else if p_no_act > 0.0 {
                        if tb < next_min {
                            next_min = tb;
                        }
                        if tb > next_max {
                            next_max = tb;
                        }
                    }
                }
                if p_att > 0.0 {
                    let pa = pm * p_att;
                    for i in 0..kern.len {
                        let (dmg, prob) = kern.pts[i];
                        if dmg > 0 && p_sub_now > 0.0 {
                            let final_sub_hp =
                                apply_eot_and_berry(hp_after_self, res_this_turn, step);
                            let tb_sub = to_b(final_sub_hp);
                            let add_sub = pa * prob * p_sub_now;
                            next[tb_sub] += add_sub;
                            if tb_sub == 0 {
                                step_eot_ko += add_sub;
                            } else if add_sub > 0.0 {
                                if tb_sub < next_min {
                                    next_min = tb_sub;
                                }
                                if tb_sub > next_max {
                                    next_max = tb_sub;
                                }
                            }
                            let p_unsub = 1.0 - p_sub_now;
                            if p_unsub > 1e-6 {
                                let post_hit = (hp_after_self - dmg).max(0);
                                let final_hp = apply_eot_and_berry(post_hit, res_this_turn, step);
                                let tb = to_b(final_hp);
                                let add = pa * prob * p_unsub;
                                next[tb] += add;
                                if tb == 0 && post_hit > 0 {
                                    step_eot_ko += add;
                                } else if tb > 0 && add > 0.0 {
                                    if tb < next_min {
                                        next_min = tb;
                                    }
                                    if tb > next_max {
                                        next_max = tb;
                                    }
                                }
                            }
                        } else {
                            let post_hit = (hp_after_self - dmg).max(0);
                            let final_hp = apply_eot_and_berry(post_hit, res_this_turn, step);
                            let tb = to_b(final_hp);
                            let add = pa * prob;
                            next[tb] += add;
                            if tb == 0 && post_hit > 0 {
                                step_eot_ko += add;
                            } else if tb > 0 && add > 0.0 {
                                if tb < next_min {
                                    next_min = tb;
                                }
                                if tb > next_max {
                                    next_max = tb;
                                }
                            }
                        }
                    }
                }
            }
        }
        dist = next;
        min_b = next_min;
        max_b = next_max;
        let mut surv = 0.0f32;
        if min_b <= max_b {
            for b in min_b..=max_b {
                surv += dist[b];
            }
        }
        out[step] = (surv as f64).clamp(0.0, 1.0);
        eot_ko[step] = (step_eot_ko as f64).clamp(0.0, 1.0);
    }

    let mean_hp_frac = if !need_mean_hp || min_b > max_b {
        0.0
    } else {
        let mut alive_mass = 0.0f32;
        let mut weighted_b = 0.0f32;
        for b in min_b..=max_b {
            alive_mass += dist[b];
            weighted_b += dist[b] * (b as f32 / B32 as f32);
        }
        if alive_mass > 1e-6 {
            (weighted_b / alive_mass) as f64
        } else {
            0.0
        }
    };
    SurvivalCurveDetail {
        surv: out,
        eot_ko,
        mean_hp_frac,
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn survival_curve_k_slots_with_mean_hp(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
    start_hp: i16,
    max_steps: usize,
    need_mean_hp: bool,
) -> ([f64; 7], f64) {
    let d = survival_curve_detail(
        cb,
        table,
        att_side,
        att_slot,
        def_side,
        def_slot,
        start_hp,
        max_steps,
        need_mean_hp,
    );
    (d.surv, d.mean_hp_frac)
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
    /// Minimum and maximum non-zero surviving HP bin across `b in 1..=16` for `kind in 0..3`
    min_surv_bin: [u8; 3],
    max_surv_bin: [u8; 3],
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

    #[allow(clippy::too_many_arguments)]
    fn build_one(
        cb: &CompactBattle,
        table: &MatchTable,
        side: usize,
        slot: usize,
        foe_side: usize,
        foe_slot: usize,
        inst_heal_amt: i16,
        has_rest: bool,
        has_st: bool,
        max_s_me: usize,
        foe_k_att: &DmgKernel1D,
        foe_k_st: Option<&DmgKernel1D>,
    ) -> Self {
        let idx = side * 3 + slot;
        let foe_idx = foe_side * 3 + foe_slot;
        let mon = &cb.mons[idx];
        let foe_mon = &cb.mons[foe_idx];
        let spec = &table.mons[idx];
        let foe_spec = &table.mons[foe_idx];
        let maxhp = spec.maxhp.max(1);

        let has_half_heal = inst_heal_amt > 0;
        let perish_active = (mon.vol_flags & vol_bit::PERISH_SONG) != 0;
        let ispec = table.item_spec(mon.item);
        let foe_ispec = table.item_spec(foe_mon.item);
        let has_berry = ispec.is_mintberry || ispec.is_miracleberry;
        let is_frz = mon.status_raw() == Status::Frz as u8 && !ispec.is_miracleberry;
        let has_heal = !perish_active && !is_frz && (has_half_heal || has_rest);
        let heal_is_rest = !has_half_heal && has_rest;

        let mut half_heal_bin = [0u8; B16 + 1];
        if has_heal && !heal_is_rest {
            for b in 1..=B16 {
                let hp = Self::b16_to_hp(b, maxhp);
                let healed = (hp + inst_heal_amt).min(maxhp);
                half_heal_bin[b] = Self::to_b16(healed, maxhp);
            }
        }

        let lefties = if ispec.is_leftovers {
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
        let is_tox = is_toxic_residual(table.ruleset, st, cnt_packed);
        // Stationary future-turn Bellman DoT: exact next-turn rate `(c + 1)` for `c >= 1`,
        // floored at `2/16` when `c == 0` (exact Turn-2 rate / 3-turn average `(1+2+3)/3`)
        // so infinite-horizon `v_table` never treats Toxic + Leftovers as 0 net DoT forever.
        let tox_c = ((cnt_packed & 0x0F) as i16 + 1).clamp(2, 15);
        let status_dot = if is_tox {
            ((maxhp / 16) * tox_c).max(1)
        } else if st == Status::Brn as u8 || st == Status::Psn as u8 || st == Status::Tox as u8 {
            (maxhp / 8).max(1)
        } else {
            0
        };
        let curse_dot = if (mon.vol_flags & vol_bit::CURSE) != 0 {
            (maxhp / 4).max(1)
        } else {
            0
        };
        let sand_dot = if cb.weather_turns > 0
            && cb.weather_kind == CompactWeather::Sandstorm
            && (spec.type_mask & table.sandstorm_immune_mask) == 0
        {
            (maxhp / 8).max(1)
        } else {
            0
        };
        let net_uncured = lefties + leech_gain - leech_drain - status_dot - curse_dot - sand_dot;
        let net_cured = lefties + leech_gain - leech_drain - curse_dot - sand_dot;
        let need_cured_eot = (max_s_me > 0 && has_rest) || (has_heal && heal_is_rest);

        let mut eot_bin = [[0u8; B16 + 1]; 2];
        for b in 1..=B16 {
            let hp = Self::b16_to_hp(b, maxhp);
            eot_bin[0][b] = Self::to_b16((hp + net_uncured).clamp(0, maxhp), maxhp);
            if need_cured_eot {
                eot_bin[1][b] = Self::to_b16((hp + net_cured).clamp(0, maxhp), maxhp);
            }
        }

        let foe_miracle = foe_ispec.is_miracleberry;
        let foe_frz = foe_mon.status_raw() == Status::Frz as u8 && !foe_miracle;
        let foe_par = foe_mon.status_raw() == Status::Par as u8 && !foe_miracle;
        let foe_conf = (foe_mon.vol_flags & vol_bit::CONFUSION) != 0 && !foe_miracle;
        let p_foe_act = (if foe_frz { 25.0 / 256.0 } else { 1.0f32 })
            * (if foe_par { 0.75f32 } else { 1.0f32 })
            * (if foe_conf { 0.50f32 } else { 1.0f32 });

        let mut incoming = [[BinDist16::identity(0); B16 + 1]; 3];
        let mut min_surv_bin = [1u8, B16 as u8 + 1, B16 as u8 + 1];
        let mut max_surv_bin = [B16 as u8, 0u8, 0u8];
        for b in 1..=B16 {
            let hp = Self::b16_to_hp(b, maxhp);

            let mut d1 = BinDist16 {
                bins: [(0, 0.0); 8],
                len: 0,
                ko_prob: 0.0,
            };
            if foe_k_att.max_dmg <= 0 {
                d1.push(b as u8, 1.0);
            } else {
                if p_foe_act < 1.0 {
                    d1.push(b as u8, 1.0 - p_foe_act);
                }
                for i in 0..foe_k_att.len {
                    let (dmg, prob) = foe_k_att.pts[i];
                    let post_hp = (hp - dmg).max(0);
                    d1.push(Self::to_b16(post_hp, maxhp), p_foe_act * prob);
                }
            }
            incoming[1][b] = d1;

            if let Some(kern_st) = foe_k_st {
                let mut d2 = BinDist16 {
                    bins: [(0, 0.0); 8],
                    len: 0,
                    ko_prob: 0.0,
                };
                if kern_st.max_dmg <= 0 {
                    d2.push(b as u8, 1.0);
                } else {
                    if p_foe_act < 1.0 {
                        d2.push(b as u8, 1.0 - p_foe_act);
                    }
                    for i in 0..kern_st.len {
                        let (dmg, prob) = kern_st.pts[i];
                        let post_hp = (hp - dmg).max(0);
                        d2.push(Self::to_b16(post_hp, maxhp), p_foe_act * prob);
                    }
                }
                for i in 0..d2.len {
                    let mb = d2.bins[i].0;
                    if mb < min_surv_bin[2] {
                        min_surv_bin[2] = mb;
                    }
                    if mb > max_surv_bin[2] {
                        max_surv_bin[2] = mb;
                    }
                }
                incoming[2][b] = d2;
            }
            for i in 0..d1.len {
                let mb = d1.bins[i].0;
                if mb < min_surv_bin[1] {
                    min_surv_bin[1] = mb;
                }
                if mb > max_surv_bin[1] {
                    max_surv_bin[1] = mb;
                }
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
            min_surv_bin,
            max_surv_bin,
            incoming,
        }
    }

    fn build_pair(
        cb: &CompactBattle,
        table: &MatchTable,
        s0: usize,
        slot0: usize,
        s1: usize,
        slot1: usize,
    ) -> ([Self; 2], usize, usize, u8) {
        let scan_moves = |side: usize, slot: usize| -> (i16, bool, bool) {
            let idx = side * 3 + slot;
            let mon = &cb.mons[idx];
            let spec = &table.mons[idx];
            let maxhp = spec.maxhp.max(1);
            let dis = (mon.vol_flags & vol_bit::DISABLE) != 0;
            let dis_slot = ((mon.slot_refs >> 4) & 0x03) as usize;
            let mut hh_amt = 0i16;
            let mut hr = false;
            let mut hst = false;
            for m_i in 0..spec.num_moves as usize {
                if mon.pp[m_i] > 0 && !(dis && m_i == dis_slot) {
                    let ms = &table.move_specs[idx][m_i];
                    if let Some(amt) = instant_heal_amount(cb, table.ruleset, ms, maxhp) {
                        if amt > hh_amt {
                            hh_amt = amt;
                        }
                    } else if ms.is_rest {
                        hr = true;
                    }
                    if ms.special_kind == nc2000_engine::compact::SpecialMoveKind::SleepTalk {
                        hst = true;
                    }
                }
            }
            (hh_amt, hr, hst)
        };

        let (hh_amt0, hr0, hst0) = scan_moves(s0, slot0);
        let (hh_amt1, hr1, hst1) = scan_moves(s1, slot1);
        let hh0 = hh_amt0 > 0;
        let hh1 = hh_amt1 > 0;

        let idx0 = s0 * 3 + slot0;
        let idx1 = s1 * 3 + slot1;
        let m0 = &cb.mons[idx0];
        let m1 = &cb.mons[idx1];
        let is0 = table.item_spec(m0.item);
        let is1 = table.item_spec(m1.item);
        let berry0 = is0.is_mintberry || is0.is_miracleberry;
        let berry1 = is1.is_mintberry || is1.is_miracleberry;
        let perish0 = (m0.vol_flags & vol_bit::PERISH_SONG) != 0;
        let perish1 = (m1.vol_flags & vol_bit::PERISH_SONG) != 0;
        let frz0 = m0.status_raw() == Status::Frz as u8 && !is0.is_miracleberry;
        let frz1 = m1.status_raw() == Status::Frz as u8 && !is1.is_miracleberry;
        let max_slp_cap = if table.ruleset.is_castle() {
            MAX_SLP_STATES
        } else {
            2
        };
        let init_ns = |m: &nc2000_engine::compact::CompactMon, berry: bool| -> usize {
            if m.status_raw() == Status::Slp as u8 && !berry {
                let slp = (m.status_counter().saturating_sub(1) as usize).min(max_slp_cap);
                slp.saturating_sub(1)
            } else {
                0
            }
        };
        let rest_s0 = if !perish0 && !frz0 && !hh0 && hr0 && !berry0 {
            2usize
        } else {
            0usize
        };
        let rest_s1 = if !perish1 && !frz1 && !hh1 && hr1 && !berry1 {
            2usize
        } else {
            0usize
        };
        let max_s0 = rest_s0.max(init_ns(m0, berry0)).min(MAX_SLP_STATES - 1);
        let max_s1 = rest_s1.max(init_ns(m1, berry1)).min(MAX_SLP_STATES - 1);

        let k01_att = DmgKernel1D::best_attack_grid(cb, table, s0, slot0, s1, slot1);
        let k10_att = DmgKernel1D::best_attack_grid(cb, table, s1, slot1, s0, slot0);
        let k01_st = if max_s0 > 0 && hst0 {
            Some(DmgKernel1D::sleep_talk_grid(cb, table, s0, slot0, s1, slot1))
        } else {
            None
        };
        let k10_st = if max_s1 > 0 && hst1 {
            Some(DmgKernel1D::sleep_talk_grid(cb, table, s1, slot1, s0, slot0))
        } else {
            None
        };

        let ops0 = Self::build_one(
            cb,
            table,
            s0,
            slot0,
            s1,
            slot1,
            hh_amt0,
            hr0,
            hst0,
            max_s0,
            &k10_att,
            k10_st.as_ref(),
        );
        let ops1 = Self::build_one(
            cb,
            table,
            s1,
            slot1,
            s0,
            slot0,
            hh_amt1,
            hr1,
            hst1,
            max_s1,
            &k01_att,
            k01_st.as_ref(),
        );

        let spd_ord = speed_ord_u8(cb, table, s0, slot0, s1, slot1);

        ([ops0, ops1], max_s0, max_s1, spd_ord)
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_exact_root_table(
    v_table: &BellmanVTable,
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
    let is_castle = table.ruleset.is_castle();
    let recoil_att_wins = table
        .ruleset
        .castle_overlay()
        .map_or(true, |cfg| cfg.recoil_1v1_attacker_wins);
    let spd_ord = speed_ord_u8(cb, table, s0, slot0, s1, slot1);
    let mutual_eot_val = mutual_eot_payoff(cb, table, s0, slot0, s1, slot1, spd_ord);

    let slp = [
        side_sleep_rem(cb, table, s0, slot0),
        side_sleep_rem(cb, table, s1, slot1),
    ];

    let mut has_heal = [false; 2];
    let mut inst_heal_amt = [0i16; 2];
    let mut heal_is_rest = [false; 2];
    let mut has_rest = [false; 2];
    let mut has_st = [false; 2];
    let mut has_berry = [false; 2];
    let mut heal_berry = [0i16; 2];
    let mut p_act = [1.0f32; 2];
    let mut eot_net = [[0i16; 2]; 2];
    let mut sub_hp_init = [0i16; 2];

    for i in 0..2 {
        let m = &cb.mons[idx[i]];
        let fm = &cb.mons[idx[1 - i]];
        let sp = &table.mons[idx[i]];
        let fsp = &table.mons[idx[1 - i]];
        if (m.vol_flags & vol_bit::SUBSTITUTE) != 0 {
            sub_hp_init[i] = m.sub_hp().max(1);
        }
        let mut hh_amt = 0i16;
        let mut hr = false;
        let mut hst = false;
        for m_i in 0..sp.num_moves as usize {
            if m.pp[m_i] > 0 && !m.is_slot_disabled(m_i) {
                let ms = &table.move_specs[idx[i]][m_i];
                if let Some(amt) = instant_heal_amount(cb, table.ruleset, ms, maxhp[i]) {
                    if amt > hh_amt {
                        hh_amt = amt;
                    }
                } else if ms.is_rest {
                    hr = true;
                }
                if ms.special_kind == nc2000_engine::compact::SpecialMoveKind::SleepTalk {
                    hst = true;
                }
            }
        }
        let hh = hh_amt > 0;
        inst_heal_amt[i] = hh_amt;
        let perish = (m.vol_flags & vol_bit::PERISH_SONG) != 0;
        let ispec = table.item_spec(m.item);
        let miracle = ispec.is_miracleberry;
        let berry = ispec.is_mintberry || ispec.is_miracleberry;
        heal_berry[i] = heal_berry_amount(ispec);
        let st_raw = m.status_raw();
        let is_frz = st_raw == Status::Frz as u8 && !miracle;
        let is_par = st_raw == Status::Par as u8 && !miracle;
        let is_conf = (m.vol_flags & vol_bit::CONFUSION) != 0
            && m.timers[nc2000_engine::compact::timer_idx::CONFUSION] > 1
            && !miracle;
        let is_rech = m.must_recharge();

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

        let lefties = if ispec.is_leftovers {
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
        let is_tox = is_toxic_residual(table.ruleset, st_raw, cnt_packed);
        let tox_c = ((cnt_packed & 0x0F) as i16 + 1).min(15);
        let status_dot = if is_tox {
            ((maxhp[i] / 16) * tox_c).max(1)
        } else if st_raw == Status::Brn as u8
            || st_raw == Status::Psn as u8
            || st_raw == Status::Tox as u8
        {
            (maxhp[i] / 8).max(1)
        } else {
            0
        };
        let curse_dot = if (m.vol_flags & vol_bit::CURSE) != 0 {
            (maxhp[i] / 4).max(1)
        } else {
            0
        };
        let sand_dot = if cb.weather_turns > 0
            && cb.weather_kind == CompactWeather::Sandstorm
            && (sp.type_mask & table.sandstorm_immune_mask) == 0
        {
            (maxhp[i] / 8).max(1)
        } else {
            0
        };
        eot_net[i][0] = lefties + leech_gain - leech_drain - status_dot - curse_dot - sand_dot;
        eot_net[i][1] = lefties + leech_gain - leech_drain - curse_dot - sand_dot;
    }

    let k_att = [
        if slp[0] == 0 && p_act[0] > 0.0 {
            DmgKernel1D::best_attack(cb, table, s0, slot0, s1, slot1)
        } else {
            DmgKernel1D::zero()
        },
        if slp[1] == 0 && p_act[1] > 0.0 {
            DmgKernel1D::best_attack(cb, table, s1, slot1, s0, slot0)
        } else {
            DmgKernel1D::zero()
        },
    ];
    let k_st = [
        if slp[0] > 0 && has_st[0] && p_act[0] > 0.0 {
            DmgKernel1D::sleep_talk(cb, table, s0, slot0, s1, slot1)
        } else {
            DmgKernel1D::zero()
        },
        if slp[1] > 0 && has_st[1] && p_act[1] > 0.0 {
            DmgKernel1D::sleep_talk(cb, table, s1, slot1, s0, slot0)
        } else {
            DmgKernel1D::zero()
        },
    ];

    let opp_hit_buffer = [0, 1].map(|i| {
        let foe_k = &k_att[1 - i];
        let mut mean_dmg = 0.0f32;
        for p_i in 0..foe_k.len {
            mean_dmg += (foe_k.pts[p_i].0 as f32) * foe_k.pts[p_i].1;
        }
        (mean_dmg.round() as i16).clamp(maxhp[i] / 4, maxhp[i] / 2)
    });

    let interp_coords = |h: i16, mx: i16| -> (usize, usize, f32) {
        let x = ((h as f32) * (B16 as f32) / (mx.max(1) as f32)).clamp(0.0, B16 as f32);
        let lo = (x.floor() as usize).clamp(1, B16);
        let hi = (x.ceil() as usize).clamp(1, B16);
        let w = (x - (lo as f32)).clamp(0.0, 1.0);
        (lo, hi, w)
    };

    let apply_end_berry = |side_i: usize, hp_post: i16, eot_val: i16, used_early: bool| -> i16 {
        if hp_post <= 0 {
            return 0;
        }
        let mut h = (hp_post + eot_val).clamp(0, maxhp[side_i]);
        if !used_early
            && heal_berry[side_i] > 0
            && heal_berry_triggers(table.ruleset, h, maxhp[side_i])
        {
            h = (h + heal_berry[side_i]).min(maxhp[side_i]);
        }
        h
    };

    let eval_order = |first: usize, a0: u8, a1: u8| -> f32 {
        let second = 1 - first;
        let act = [a0, a1];
        let win_first = if first == 0 { 1.0f32 } else { 0.0f32 };
        let win_second = 1.0f32 - win_first;
        let sub_fir = sub_hp_init[first];
        let sub_sec = sub_hp_init[second];

        let (mut hp_fir_mid, ns_fir, cured_fir, k_fir_opt) = if slp[first] > 0 {
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
                    (hp[first] + inst_heal_amt[first]).min(maxhp[first]),
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

        let mut sec_pts = [(0i16, 0i16, false, 0.0f32); 17];
        let mut sec_len = 0usize;
        let mut ko_sec_val = 0.0f32;
        let push_pt = |pts: &mut [(i16, i16, bool, f32); 17],
                       len: &mut usize,
                       h: i16,
                       sub_h: i16,
                       used_b: bool,
                       p: f32| {
            if p <= 0.0 || h <= 0 {
                return;
            }
            for idx_p in 0..*len {
                if pts[idx_p].0 == h && pts[idx_p].1 == sub_h && pts[idx_p].2 == used_b {
                    pts[idx_p].3 += p;
                    return;
                }
            }
            if *len < 17 {
                pts[*len] = (h, sub_h, used_b, p);
                *len += 1;
            } else {
                pts[*len - 1].3 += p;
            }
        };

        let pa_fir = p_act[first];
        let mut fir_surv_recoil_sum = 0.0f32;
        let mut fir_surv_prob_sum = 0.0f32;
        if let Some(kern) = k_fir_opt {
            if kern.max_dmg <= 0 || pa_fir <= 0.0 {
                push_pt(&mut sec_pts, &mut sec_len, hp[second], sub_sec, false, 1.0);
            } else {
                if pa_fir < 1.0 {
                    push_pt(
                        &mut sec_pts,
                        &mut sec_len,
                        hp[second],
                        sub_sec,
                        false,
                        1.0 - pa_fir,
                    );
                }
                for i in 0..kern.len {
                    let (dmg, prob) = kern.pts[i];
                    let p_branch = pa_fir * prob;
                    if sub_sec > 0 && dmg > 0 {
                        if kern.recoil_quarter {
                            // PS: 1 HP recoil vs Sub; Castle: 0 HP recoil vs Sub (`Probe 2B`).
                            let rec = if is_castle { 0.0 } else { 1.0 };
                            fir_surv_recoil_sum += p_branch * rec;
                            fir_surv_prob_sum += p_branch;
                        }
                        push_pt(
                            &mut sec_pts,
                            &mut sec_len,
                            hp[second],
                            (sub_sec - dmg).max(0),
                            false,
                            p_branch,
                        );
                    } else {
                        let mut h_after = (hp[second] - dmg).max(0);
                        if h_after <= 0 {
                            let ko_payoff = if kern.recoil_quarter && is_castle {
                                let rec = (hp[second] / 4).max(1);
                                if hp_fir_mid <= rec && !recoil_att_wins {
                                    0.5f32
                                } else {
                                    win_first
                                }
                            } else {
                                win_first
                            };
                            ko_sec_val += p_branch * ko_payoff;
                        } else {
                            let mut used_b = false;
                            if !is_castle
                                && heal_berry[second] > 0
                                && heal_berry_triggers(table.ruleset, h_after, maxhp[second])
                            {
                                h_after = (h_after + heal_berry[second]).min(maxhp[second]);
                                used_b = true;
                            }
                            if kern.recoil_quarter && dmg > 0 {
                                let rec = (dmg / 4).max(1) as f32;
                                fir_surv_recoil_sum += p_branch * rec;
                                fir_surv_prob_sum += p_branch;
                            }
                            push_pt(&mut sec_pts, &mut sec_len, h_after, sub_sec, used_b, p_branch);
                        }
                    }
                }
            }
        } else {
            push_pt(&mut sec_pts, &mut sec_len, hp[second], sub_sec, false, 1.0);
        }

        let mut total_val = ko_sec_val;
        if sec_len == 0 {
            return total_val.clamp(0.0, 1.0);
        }

        let mut fir_used_berry_early = false;
        if fir_surv_prob_sum > 0.0 && fir_surv_recoil_sum > 0.0 {
            let rec_avg = (fir_surv_recoil_sum / fir_surv_prob_sum).round() as i16;
            hp_fir_mid = (hp_fir_mid - rec_avg).max(0);
            if hp_fir_mid <= 0 {
                let surv_mass: f32 = sec_pts[..sec_len].iter().map(|pt| pt.3).sum();
                total_val += surv_mass * win_second;
                return total_val.clamp(0.0, 1.0);
            }
            if !is_castle
                && heal_berry[first] > 0
                && heal_berry_triggers(table.ruleset, hp_fir_mid, maxhp[first])
            {
                hp_fir_mid = (hp_fir_mid + heal_berry[first]).min(maxhp[first]);
                fir_used_berry_early = true;
            }
        }

        let (sec_mode, ns_sec, cured_sec, k_sec_opt) = if slp[second] > 0 {
            (
                0u8,
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
                (1u8, if has_berry[second] { 0 } else { 2 }, true, None)
            } else {
                (2u8, 0, false, None)
            }
        } else if act[second] == 2 {
            (0u8, 0, false, None)
        } else {
            (0u8, 0, false, Some(&k_att[second]))
        };

        let mut fir_pts = [(0i16, 0i16, false, 0.0f32); 17];
        let mut fir_len = 0usize;
        let mut ko_fir_prob = 0.0f32;
        let mut sec_ko_recoil_sum = 0.0f32;
        let mut sec_surv_recoil_sum = 0.0f32;
        let mut sec_surv_prob_sum = 0.0f32;
        let pa_sec = p_act[second];
        if let Some(kern) = k_sec_opt {
            if kern.max_dmg <= 0 || pa_sec <= 0.0 {
                push_pt(
                    &mut fir_pts,
                    &mut fir_len,
                    hp_fir_mid,
                    sub_fir,
                    fir_used_berry_early,
                    1.0,
                );
            } else {
                if pa_sec < 1.0 {
                    push_pt(
                        &mut fir_pts,
                        &mut fir_len,
                        hp_fir_mid,
                        sub_fir,
                        fir_used_berry_early,
                        1.0 - pa_sec,
                    );
                }
                for i in 0..kern.len {
                    let (dmg, prob) = kern.pts[i];
                    let p_branch = pa_sec * prob;
                    if sub_fir > 0 && dmg > 0 {
                        if kern.recoil_quarter {
                            let rec = if is_castle { 0.0 } else { 1.0 };
                            sec_surv_recoil_sum += p_branch * rec;
                            sec_surv_prob_sum += p_branch;
                        }
                        push_pt(
                            &mut fir_pts,
                            &mut fir_len,
                            hp_fir_mid,
                            (sub_fir - dmg).max(0),
                            fir_used_berry_early,
                            p_branch,
                        );
                    } else {
                        let mut h_after = (hp_fir_mid - dmg).max(0);
                        if h_after <= 0 {
                            ko_fir_prob += p_branch;
                            if kern.recoil_quarter && is_castle {
                                sec_ko_recoil_sum += p_branch * ((hp_fir_mid / 4).max(1) as f32);
                            }
                        } else {
                            let mut used_b = fir_used_berry_early;
                            if !is_castle
                                && !used_b
                                && heal_berry[first] > 0
                                && heal_berry_triggers(table.ruleset, h_after, maxhp[first])
                            {
                                h_after = (h_after + heal_berry[first]).min(maxhp[first]);
                                used_b = true;
                            }
                            if kern.recoil_quarter && dmg > 0 {
                                sec_surv_recoil_sum += p_branch * ((dmg / 4).max(1) as f32);
                                sec_surv_prob_sum += p_branch;
                            }
                            push_pt(&mut fir_pts, &mut fir_len, h_after, sub_fir, used_b, p_branch);
                        }
                    }
                }
            }
        } else {
            push_pt(
                &mut fir_pts,
                &mut fir_len,
                hp_fir_mid,
                sub_fir,
                fir_used_berry_early,
                1.0,
            );
        }

        let sec_surv_rec = if sec_surv_prob_sum > 0.0 {
            (sec_surv_recoil_sum / sec_surv_prob_sum).round() as i16
        } else {
            0
        };
        let sec_ko_rec = if ko_fir_prob > 0.0 && sec_ko_recoil_sum > 0.0 {
            (sec_ko_recoil_sum / ko_fir_prob).round() as i16
        } else {
            0
        };

        let mut fir_end = [(0i16, 0usize, 0usize, 0.0f32, 0.0f32); 17];
        let eot_fir_val = eot_net[first][cured_fir as usize];
        for (idx_f, &(hp_fir_post, sub_fir_after, used_b, p_fir)) in
            fir_pts[..fir_len].iter().enumerate()
        {
            let h_end = apply_end_berry(first, hp_fir_post, eot_fir_val, used_b);
            let (lo, hi, w) = if h_end > 0 {
                let h_eff = if sub_fir_after > 0 {
                    let buf = sub_fir_after.max(opp_hit_buffer[first]);
                    (h_end + buf).min(maxhp[first])
                } else {
                    h_end
                };
                interp_coords(h_eff, maxhp[first])
            } else {
                (1, 1, 0.0)
            };
            fir_end[idx_f] = (h_end, lo, hi, w, p_fir);
        }

        let (ns0, ns1) = if first == 0 {
            (ns_fir.min(MAX_SLP_STATES - 1), ns_sec.min(MAX_SLP_STATES - 1))
        } else {
            (ns_sec.min(MAX_SLP_STATES - 1), ns_fir.min(MAX_SLP_STATES - 1))
        };
        let tbl = &v_table[ns0][ns1];

        let eval_branch = |hp_sec_mid: i16, hp_sec_end: i16, sub_sec_after: i16| -> f32 {
            let ko_fir_payoff = if sec_ko_rec > 0 && hp_sec_mid <= sec_ko_rec && !recoil_att_wins {
                0.5f32
            } else {
                win_second
            };
            let mut branch_val = ko_fir_prob * ko_fir_payoff;
            if hp_sec_end <= 0 {
                for &(hp_fir_end, _, _, _, p_fir) in &fir_end[..fir_len] {
                    let cont = if hp_fir_end <= 0 {
                        mutual_eot_val
                    } else {
                        win_first
                    };
                    branch_val += p_fir * cont;
                }
            } else {
                let h_sec_eff = if sub_sec_after > 0 {
                    let buf = sub_sec_after.max(opp_hit_buffer[second]);
                    (hp_sec_end + buf).min(maxhp[second])
                } else {
                    hp_sec_end
                };
                let (s_lo, s_hi, s_w) = interp_coords(h_sec_eff, maxhp[second]);
                for &(hp_fir_end, f_lo, f_hi, f_w, p_fir) in &fir_end[..fir_len] {
                    let cont = if hp_fir_end <= 0 {
                        win_second
                    } else {
                        let (b0_lo, b0_hi, w0, b1_lo, b1_hi, w1) = if first == 0 {
                            (f_lo, f_hi, f_w, s_lo, s_hi, s_w)
                        } else {
                            (s_lo, s_hi, s_w, f_lo, f_hi, f_w)
                        };
                        let v00 = tbl[b0_lo][b1_lo];
                        let v01 = tbl[b0_lo][b1_hi];
                        let v10 = tbl[b0_hi][b1_lo];
                        let v11 = tbl[b0_hi][b1_hi];
                        let v0 = v00 + w1 * (v01 - v00);
                        let v1 = v10 + w1 * (v11 - v10);
                        let v_cont = (v0 + w0 * (v1 - v0)).clamp(0.0, 1.0);
                        GAMMA * v_cont + (1.0 - GAMMA) * 0.5
                    };
                    branch_val += p_fir * cont;
                }
            }
            branch_val
        };

        let eot_sec_val = eot_net[second][cured_sec as usize];
        if sec_mode == 1 {
            for &(hp_sec_mid, sub_sec_after, used_b, p_sec) in &sec_pts[..sec_len] {
                let hp_sec_end = apply_end_berry(second, maxhp[second], eot_sec_val, used_b);
                total_val += p_sec * eval_branch(hp_sec_mid, hp_sec_end, sub_sec_after);
            }
        } else {
            for &(hp_sec_mid, sub_sec_after, used_b, p_sec) in &sec_pts[..sec_len] {
                let hp_sec_post = if sec_mode == 2 {
                    (hp_sec_mid + inst_heal_amt[second]).min(maxhp[second])
                } else {
                    (hp_sec_mid - sec_surv_rec).max(0)
                };
                let hp_sec_end = apply_end_berry(second, hp_sec_post, eot_sec_val, used_b);
                total_val += p_sec * eval_branch(hp_sec_mid, hp_sec_end, sub_sec_after);
            }
        }
        total_val.clamp(0.0, 1.0)
    };

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
        if spd_ord == 2 {
            eval_order(0, a0, a1)
        } else if spd_ord == 0 {
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

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
struct Key128(u64, u64);

struct Hasher128 {
    h0: u64,
    h1: u64,
}

impl Hasher128 {
    #[inline(always)]
    fn new() -> Self {
        Self {
            h0: 0x9e3779b97f4a7c15,
            h1: 0xbf58476d1ce4e5b9,
        }
    }

    #[inline(always)]
    fn write_u64(&mut self, v: u64) {
        let mut a = self.h0 ^ v;
        a = (a ^ (a >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        let mut b = self.h1 ^ v.rotate_left(29) ^ a;
        b = (b ^ (b >> 27)).wrapping_mul(0x94d049bb133111eb);
        self.h0 = a ^ (b >> 31);
        self.h1 = b ^ (a >> 29);
    }

    #[inline(always)]
    fn finish(self) -> Key128 {
        let mut a = self.h0 ^ (self.h1 >> 32);
        a = (a ^ (a >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        a ^= a >> 31;
        let mut b = self.h1 ^ (self.h0 >> 32);
        b = (b ^ (b >> 27)).wrapping_mul(0x94d049bb133111eb);
        b ^= b >> 31;
        Key128(a | 1, b | 1)
    }
}

#[derive(Clone, Copy)]
struct CachedBellman {
    op_key: Key128,
    v_table: BellmanVTable,
    heal_mask0: [u32; B16 + 1],
    heal_mask1: [u32; B16 + 1],
}

impl CachedBellman {
    const fn empty() -> Self {
        Self {
            op_key: Key128(0, 0),
            v_table: [[[[0.5; B16 + 1]; B16 + 1]; MAX_SLP_STATES]; MAX_SLP_STATES],
            heal_mask0: [0; B16 + 1],
            heal_mask1: [0; B16 + 1],
        }
    }
}

#[cfg(target_arch = "wasm32")]
const L1_SETS: usize = 4096;
#[cfg(not(target_arch = "wasm32"))]
const L1_SETS: usize = 8192;
const L1_WAYS: usize = 4;
#[cfg(target_arch = "wasm32")]
const L2_SETS: usize = 512;
#[cfg(not(target_arch = "wasm32"))]
const L2_SETS: usize = 1024;
const L2_WAYS: usize = 4;
const MAX_CACHE_SLOTS: usize = L2_SETS * L2_WAYS;

#[derive(Clone, Copy, Default)]
struct L1Way {
    key: Key128,
    gen: u32,
    slot: u16,
}

#[derive(Clone, Copy, Default)]
struct L2Set {
    keys: [Key128; L2_WAYS],
    slots: [u16; L2_WAYS],
}

struct BellmanCache {
    l1: Vec<[L1Way; L1_WAYS]>,
    l2: Vec<L2Set>,
    slot_gen: Vec<u32>,
    pool: Vec<CachedBellman>,
}

impl BellmanCache {
    fn new() -> Self {
        Self {
            l1: vec![[L1Way::default(); L1_WAYS]; L1_SETS],
            l2: vec![L2Set::default(); L2_SETS],
            slot_gen: vec![0u32; MAX_CACHE_SLOTS],
            pool: Vec::with_capacity(64),
        }
    }

    #[inline]
    fn record_l1(&mut self, l1_key: Key128, slot: u16, gen: u32) {
        let l1_idx = (l1_key.0 as usize) & (L1_SETS - 1);
        let set = &mut self.l1[l1_idx];
        for way in 0..L1_WAYS {
            if set[way].key == l1_key {
                for w in (1..=way).rev() {
                    set[w] = set[w - 1];
                }
                set[0] = L1Way {
                    key: l1_key,
                    gen,
                    slot,
                };
                return;
            }
        }
        for w in (1..L1_WAYS).rev() {
            set[w] = set[w - 1];
        }
        set[0] = L1Way {
            key: l1_key,
            gen,
            slot,
        };
    }
}

thread_local! {
    static BELLMAN_CACHE: RefCell<BellmanCache> = RefCell::new(BellmanCache::new());
}

#[inline]
fn l1_state_key(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
) -> Key128 {
    let idx0 = s0 * 3 + slot0;
    let idx1 = s1 * 3 + slot1;
    let m0 = &cb.mons[idx0];
    let m1 = &cb.mons[idx1];
    let sp0 = &table.mons[idx0];
    let sp1 = &table.mons[idx1];

    let mut h = Hasher128::new();
    h.write_u64(sp0.static_fp);
    h.write_u64(sp1.static_fp);

    let avail_mask = |m: &nc2000_engine::compact::CompactMon, num_moves: u8| -> u8 {
        let mut mask = 0u8;
        for m_i in 0..num_moves {
            if m.pp[m_i as usize] > 0 && !m.is_slot_disabled(m_i as usize) {
                mask |= 1u8 << m_i;
            }
        }
        mask
    };
    let av0 = avail_mask(m0, sp0.num_moves);
    let av1 = avail_mask(m1, sp1.num_moves);

    let spd_ord = speed_ord_u8(cb, table, s0, slot0, s1, slot1);

    // Hash raw `mon.item` directly so dynamic Thief steals (including type boosters and
    // species-stat items) never collide in L1 cache, with zero `table.item_spec` overhead.
    let w_b0 = u64::from_le_bytes([
        m0.boosts[0] as u8,
        m0.boosts[1] as u8,
        m0.boosts[2] as u8,
        m0.boosts[3] as u8,
        m0.boosts[5] as u8,
        m0.boosts[6] as u8,
        av0 | (spd_ord << 4),
        m0.item,
    ]);
    let w_b1 = u64::from_le_bytes([
        m1.boosts[0] as u8,
        m1.boosts[1] as u8,
        m1.boosts[2] as u8,
        m1.boosts[3] as u8,
        m1.boosts[5] as u8,
        m1.boosts[6] as u8,
        av1,
        m1.item,
    ]);
    h.write_u64(w_b0);
    h.write_u64(w_b1);

    const VOL_MASK: u16 = vol_bit::SUBSTITUTE
        | vol_bit::LEECH_SEED
        | vol_bit::CONFUSION
        | vol_bit::CURSE
        | vol_bit::FORESIGHT
        | vol_bit::PERISH_SONG
        | vol_bit::BRN_ATK_DROP;
    let vol0 = m0.vol_flags & VOL_MASK;
    let vol1 = m1.vol_flags & VOL_MASK;

    let canon_st = |m: &nc2000_engine::compact::CompactMon| -> u8 {
        let raw = m.status_raw();
        let cnt = m.status_counter();
        if is_toxic_residual(table.ruleset, raw, cnt) {
            raw | 0x80 | ((cnt & 0x0F) << 3)
        } else if raw == Status::Slp as u8 {
            let slp_bucket = cnt.saturating_sub(2).min(3);
            raw | (slp_bucket << 3)
        } else {
            raw
        }
    };
    let st0_packed = canon_st(m0);
    let st1_packed = canon_st(m1);

    let sub_bucket = |m: &nc2000_engine::compact::CompactMon,
                      sp: &nc2000_engine::compact::StaticMonSpec|
     -> u64 {
        if (m.vol_flags & vol_bit::SUBSTITUTE) == 0 {
            0
        } else {
            let max_sub = (sp.maxhp / 4).max(1) as i32;
            let sh = m.sub_hp().max(1) as i32;
            (((sh * 4 + max_sub - 1) / max_sub) as u64).clamp(1, 4)
        }
    };

    let weather_active = if cb.weather_turns > 0 {
        (cb.weather_kind as u64) + 1
    } else {
        0
    };
    let w_field = ((cb.sides[s0].alive_mask.count_ones() <= 1) as u64)
        | (((cb.sides[s1].alive_mask.count_ones() <= 1) as u64) << 1)
        | (((cb.sides[s0].reflect_turns > 0) as u64) << 2)
        | (((cb.sides[s0].lightscreen_turns > 0) as u64) << 3)
        | (((cb.sides[s1].reflect_turns > 0) as u64) << 4)
        | (((cb.sides[s1].lightscreen_turns > 0) as u64) << 5)
        | (weather_active << 6)
        | (sub_bucket(m0, sp0) << 9)
        | (sub_bucket(m1, sp1) << 12);

    let w_dyn = (st0_packed as u64)
        | ((st1_packed as u64) << 8)
        | ((vol0 as u64) << 16)
        | ((vol1 as u64) << 32)
        | (w_field << 48);
    h.write_u64(w_dyn);

    let ruleset_tag = table.ruleset.fingerprint_tag();
    if ruleset_tag != 0 {
        h.write_u64(ruleset_tag);
        let seat_bit = u64::from(!table.ruleset.is_seat_antisymmetric() && s0 == 0);
        h.write_u64(seat_bit);
    }

    h.finish()
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn l2_operator_key(
    ruleset_tag: u64,
    mutual_eot_val: f32,
    max_s0: usize,
    max_s1: usize,
    spd_ord: u8,
    ops: &[Side1DOps; 2],
) -> Key128 {
    let mut h = Hasher128::new();
    let max_s = [max_s0, max_s1];
    let side_hdr = |i: usize| -> u64 {
        let op = &ops[i];
        let ms = max_s[i];
        (op.has_heal as u64)
            | (((op.has_heal && op.heal_is_rest) as u64) << 1)
            | (((ms > 0 && op.has_rest) as u64) << 2)
            | (((ms > 0 && op.has_st) as u64) << 3)
            | (((op.has_heal && op.heal_is_rest && op.has_berry) as u64) << 4)
    };
    let hdr = (max_s0 as u64)
        | ((max_s1 as u64) << 2)
        | ((spd_ord as u64) << 4)
        | (side_hdr(0) << 8)
        | (side_hdr(1) << 16);
    h.write_u64(hdr);
    if ruleset_tag != 0 {
        h.write_u64(ruleset_tag);
        h.write_u64(mutual_eot_val.to_bits() as u64);
    }

    let pack_16 = |arr: &[u8; B16 + 1]| -> (u64, u64) {
        let lo = u64::from_le_bytes([
            arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8],
        ]);
        let hi = u64::from_le_bytes([
            arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15], arr[16],
        ]);
        (lo, hi)
    };

    for i in 0..2 {
        let op = &ops[i];
        if op.has_heal && !op.heal_is_rest {
            let (lo, hi) = pack_16(&op.half_heal_bin);
            h.write_u64(lo);
            h.write_u64(hi);
        }
        let (e0_lo, e0_hi) = pack_16(&op.eot_bin[0]);
        h.write_u64(e0_lo);
        h.write_u64(e0_hi);
        if (max_s[i] > 0 && op.has_rest) || (op.has_heal && op.heal_is_rest) {
            let (e1_lo, e1_hi) = pack_16(&op.eot_bin[1]);
            h.write_u64(e1_lo);
            h.write_u64(e1_hi);
        }
        for b in 1..=B16 {
            let d = &op.incoming[1][b];
            h.write_u64((d.ko_prob.to_bits() as u64) | ((d.len as u64) << 32));
            for p_i in 0..d.len {
                let (nb, pr) = d.bins[p_i];
                h.write_u64((nb as u64) | ((pr.to_bits() as u64) << 8));
            }
        }
        if max_s[1 - i] > 0 && ops[1 - i].has_st {
            for b in 1..=B16 {
                let d = &op.incoming[2][b];
                h.write_u64((d.ko_prob.to_bits() as u64) | ((d.len as u64) << 32));
                for p_i in 0..d.len {
                    let (nb, pr) = d.bins[p_i];
                    h.write_u64((nb as u64) | ((pr.to_bits() as u64) << 8));
                }
            }
        }
    }

    h.finish()
}

#[allow(clippy::too_many_arguments)]
fn fill_ordered_q_plane(
    first: usize,
    slp0: usize,
    slp1: usize,
    ops: &[Side1DOps; 2],
    v_table: &BellmanVTable,
    mutual_eot_val: f32,
    gamma: f32,
    second_pass_avg: bool,
    q_plane: &mut [[[[f32; 2]; 2]; B16 + 1]; B16 + 1],
) {
    let second = 1 - first;
    let s = [slp0, slp1];
    let win_first = if first == 0 { 1.0f32 } else { 0.0f32 };
    let win_second = 1.0f32 - win_first;

    let max_a_fir = if s[first] == 0 && ops[first].has_heal {
        2usize
    } else {
        1usize
    };
    let max_a_sec = if s[second] == 0 && ops[second].has_heal {
        2usize
    } else {
        1usize
    };

    for a_fir in 0..max_a_fir {
        let (ns_fir, cured_fir, k_fir) = if s[first] > 0 {
            (
                s[first] - 1,
                ops[first].has_rest,
                if ops[first].has_st { 2usize } else { 0usize },
            )
        } else if a_fir == 1 {
            if ops[first].heal_is_rest {
                (if ops[first].has_berry { 0 } else { 2 }, true, 0usize)
            } else {
                (0, false, 0usize)
            }
        } else {
            (0, false, 1usize)
        };
        let b_fir_max = if a_fir == 1 { B16 - 1 } else { B16 };
        let eot_fir = &ops[first].eot_bin[cured_fir as usize];

        for a_sec in 0..max_a_sec {
            let (ns_sec, cured_sec, k_sec, sec_mode) = if s[second] > 0 {
                (
                    s[second] - 1,
                    ops[second].has_rest,
                    if ops[second].has_st { 2usize } else { 0usize },
                    0u8,
                )
            } else if a_sec == 1 {
                if ops[second].heal_is_rest {
                    (if ops[second].has_berry { 0 } else { 2 }, true, 0usize, 1u8)
                } else {
                    (0, false, 0usize, 2u8)
                }
            } else {
                (0, false, 1usize, 0u8)
            };
            if second_pass_avg && k_fir == 0 && k_sec == 0 {
                continue;
            }
            let b_sec_max = if a_sec == 1 { B16 - 1 } else { B16 };
            let eot_sec = &ops[second].eot_bin[cured_sec as usize];
            let (ns0, ns1) = if first == 0 {
                (ns_fir, ns_sec)
            } else {
                (ns_sec, ns_fir)
            };
            let v_plane = &v_table[ns0][ns1];

            let mut end_of_m = [0u8; B16 + 1];
            if sec_mode == 2 {
                for m in 1..=B16 {
                    end_of_m[m] = eot_sec[ops[second].half_heal_bin[m] as usize];
                }
            } else if sec_mode == 0 {
                end_of_m = *eot_sec;
            }
            let (min_end, max_end) = if sec_mode == 1 {
                let e = eot_sec[B16] as usize;
                (e, e)
            } else if k_fir == 0 {
                (end_of_m[1] as usize, end_of_m[b_sec_max] as usize)
            } else {
                let min_m = ops[second].min_surv_bin[k_fir] as usize;
                let max_m = ops[second].max_surv_bin[k_fir] as usize;
                if min_m <= max_m {
                    (end_of_m[min_m] as usize, end_of_m[max_m] as usize)
                } else {
                    (1usize, 0usize)
                }
            };

            let mut prev_row_key = usize::MAX;
            let mut last_row = [0.0f32; B16 + 1];

            for b_fir in 1..=b_fir_max {
                let b_fir_mid = if s[first] == 0 && a_fir == 1 {
                    if ops[first].heal_is_rest {
                        B16
                    } else {
                        ops[first].half_heal_bin[b_fir] as usize
                    }
                } else {
                    b_fir
                };
                let row_key = if k_sec == 0 {
                    eot_fir[b_fir_mid] as usize
                } else if ops[first].incoming[k_sec][b_fir_mid].len == 0 {
                    0usize
                } else {
                    b_fir_mid
                };

                if row_key != prev_row_key {
                    prev_row_key = row_key;
                    let mut bv_end = [0.0f32; B16 + 1];
                    if min_end <= max_end {
                        if k_sec == 0 {
                            let b_fir_end = row_key;
                            if min_end == 0 {
                                bv_end[0] = if b_fir_end == 0 {
                                    mutual_eot_val
                                } else {
                                    win_first
                                };
                            }
                            let e_lo = min_end.max(1);
                            if e_lo <= max_end {
                                if b_fir_end == 0 {
                                    for cell in &mut bv_end[e_lo..=max_end] {
                                        *cell = win_second;
                                    }
                                } else if first == 0 {
                                    let v_row = &v_plane[b_fir_end];
                                    for end in e_lo..=max_end {
                                        bv_end[end] = gamma * v_row[end] + (1.0 - gamma) * 0.5;
                                    }
                                } else {
                                    for end in e_lo..=max_end {
                                        bv_end[end] =
                                            gamma * v_plane[end][b_fir_end] + (1.0 - gamma) * 0.5;
                                    }
                                }
                            }
                        } else {
                            let dist_fir = &ops[first].incoming[k_sec][b_fir_mid];
                            let base_ko = dist_fir.ko_prob * win_second;
                            for cell in &mut bv_end[min_end..=max_end] {
                                *cell = base_ko;
                            }
                            let e_lo = min_end.max(1);
                            for i_fir in 0..dist_fir.len {
                                let (b_fir_post_u8, p_fir) = dist_fir.bins[i_fir];
                                let b_fir_end = eot_fir[b_fir_post_u8 as usize] as usize;
                                if min_end == 0 {
                                    let cont0 = if b_fir_end == 0 {
                                        mutual_eot_val
                                    } else {
                                        win_first
                                    };
                                    bv_end[0] += p_fir * cont0;
                                }
                                if e_lo <= max_end {
                                    if b_fir_end == 0 {
                                        let add = p_fir * win_second;
                                        for cell in &mut bv_end[e_lo..=max_end] {
                                            *cell += add;
                                        }
                                    } else if first == 0 {
                                        let v_row = &v_plane[b_fir_end];
                                        for end in e_lo..=max_end {
                                            let cont = gamma * v_row[end] + (1.0 - gamma) * 0.5;
                                            bv_end[end] += p_fir * cont;
                                        }
                                    } else {
                                        for end in e_lo..=max_end {
                                            let cont = gamma * v_plane[end][b_fir_end]
                                                + (1.0 - gamma) * 0.5;
                                            bv_end[end] += p_fir * cont;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if sec_mode == 1 {
                        let bv = bv_end[min_end];
                        if k_fir == 0 {
                            let clamped = bv.clamp(0.0, 1.0);
                            for cell in &mut last_row[1..=b_sec_max] {
                                *cell = clamped;
                            }
                        } else {
                            for b_sec in 1..=b_sec_max {
                                let dist_sec = &ops[second].incoming[k_fir][b_sec];
                                let mut total_val = dist_sec.ko_prob * win_first;
                                for i_sec in 0..dist_sec.len {
                                    total_val += dist_sec.bins[i_sec].1 * bv;
                                }
                                last_row[b_sec] = total_val.clamp(0.0, 1.0);
                            }
                        }
                    } else if k_fir == 0 {
                        for b_sec in 1..=b_sec_max {
                            let end = end_of_m[b_sec] as usize;
                            last_row[b_sec] = bv_end[end].clamp(0.0, 1.0);
                        }
                    } else {
                        let min_m = ops[second].min_surv_bin[k_fir] as usize;
                        let max_m = ops[second].max_surv_bin[k_fir] as usize;
                        let mut bv_m = [0.0f32; B16 + 1];
                        if min_m <= max_m {
                            for m in min_m..=max_m {
                                bv_m[m] = bv_end[end_of_m[m] as usize];
                            }
                        }
                        for b_sec in 1..=b_sec_max {
                            let dist_sec = &ops[second].incoming[k_fir][b_sec];
                            let mut total_val = dist_sec.ko_prob * win_first;
                            for i_sec in 0..dist_sec.len {
                                let (m_u8, p_sec) = dist_sec.bins[i_sec];
                                total_val += p_sec * bv_m[m_u8 as usize];
                            }
                            last_row[b_sec] = total_val.clamp(0.0, 1.0);
                        }
                    }
                }

                if first == 0 {
                    if second_pass_avg {
                        for b_sec in 1..=b_sec_max {
                            let cell = &mut q_plane[b_fir][b_sec][a_fir][a_sec];
                            *cell = 0.5 * (*cell + last_row[b_sec]);
                        }
                    } else {
                        for b_sec in 1..=b_sec_max {
                            q_plane[b_fir][b_sec][a_fir][a_sec] = last_row[b_sec];
                        }
                    }
                } else if second_pass_avg {
                    for b_sec in 1..=b_sec_max {
                        let cell = &mut q_plane[b_sec][b_fir][a_sec][a_fir];
                        *cell = 0.5 * (*cell + last_row[b_sec]);
                    }
                } else {
                    for b_sec in 1..=b_sec_max {
                        q_plane[b_sec][b_fir][a_sec][a_fir] = last_row[b_sec];
                    }
                }
            }
        }
    }
}

fn solve_bellman_into(
    max_s0: usize,
    max_s1: usize,
    spd_ord: u8,
    mutual_eot_val: f32,
    ops: &[Side1DOps; 2],
    out: &mut CachedBellman,
) {
    let max_s_sum = max_s0 + max_s1;
    out.v_table = [[[[0.5f32; B16 + 1]; B16 + 1]; MAX_SLP_STATES]; MAX_SLP_STATES];
    for slp0 in 0..=max_s0 {
        for slp1 in 0..=max_s1 {
            for b in 1..=B16 {
                out.v_table[slp0][slp1][0][b] = 0.0;
                out.v_table[slp0][slp1][b][0] = 1.0;
            }
        }
    }

    let mut q_plane = [[[[0.5f32; 2]; 2]; B16 + 1]; B16 + 1];
    let eval_plane_q = |slp0: usize,
                        slp1: usize,
                        vt: &BellmanVTable,
                        gamma: f32,
                        qp: &mut [[[[f32; 2]; 2]; B16 + 1]; B16 + 1]| {
        if slp1 > 0 && !ops[1].has_st {
            fill_ordered_q_plane(0, slp0, slp1, ops, vt, mutual_eot_val, gamma, false, qp);
        } else if slp0 > 0 && !ops[0].has_st {
            fill_ordered_q_plane(1, slp0, slp1, ops, vt, mutual_eot_val, gamma, false, qp);
        } else if spd_ord == 2 {
            fill_ordered_q_plane(0, slp0, slp1, ops, vt, mutual_eot_val, gamma, false, qp);
        } else if spd_ord == 0 {
            fill_ordered_q_plane(1, slp0, slp1, ops, vt, mutual_eot_val, gamma, false, qp);
        } else {
            fill_ordered_q_plane(0, slp0, slp1, ops, vt, mutual_eot_val, gamma, false, qp);
            fill_ordered_q_plane(1, slp0, slp1, ops, vt, mutual_eot_val, gamma, true, qp);
        }
    };

    let mut heal_mask0 = [0u32; B16 + 1];
    let mut heal_mask1 = [0u32; B16 + 1];

    for _iter in 0..18 {
        // 1. Bottom-up Schur-complement elimination over the sleeping DAG `s_sum = 1..=max_s_sum`.
        for s_sum in 1..=max_s_sum {
            for slp0 in 0..=max_s0 {
                for slp1 in 0..=max_s1 {
                    if slp0 + slp1 != s_sum {
                        continue;
                    }
                    if slp0 > 0 && slp1 > 0 && !ops[0].has_st && !ops[1].has_st {
                        let eot0 = &ops[0].eot_bin[ops[0].has_rest as usize];
                        let eot1 = &ops[1].eot_bin[ops[1].has_rest as usize];
                        let prev = out.v_table[slp0 - 1][slp1 - 1];
                        let dst = &mut out.v_table[slp0][slp1];
                        for b0 in 1..=B16 {
                            let e0 = eot0[b0] as usize;
                            if e0 == 0 {
                                for b1 in 1..=B16 {
                                    let e1 = eot1[b1] as usize;
                                    dst[b0][b1] = if e1 == 0 { mutual_eot_val } else { 0.0 };
                                }
                            } else {
                                for b1 in 1..=B16 {
                                    let e1 = eot1[b1] as usize;
                                    dst[b0][b1] = if e1 == 0 {
                                        1.0
                                    } else {
                                        (GAMMA * prev[e0][e1] + (1.0 - GAMMA) * 0.5).clamp(0.0, 1.0)
                                    };
                                }
                            }
                        }
                        continue;
                    }
                    eval_plane_q(slp0, slp1, &out.v_table, GAMMA, &mut q_plane);
                    let c0 = slp0 == 0 && ops[0].has_heal;
                    let c1 = slp1 == 0 && ops[1].has_heal;
                    if !c0 && !c1 {
                        for b0 in 1..=B16 {
                            for b1 in 1..=B16 {
                                out.v_table[slp0][slp1][b0][b1] = q_plane[b0][b1][0][0];
                            }
                        }
                    } else {
                        for b0 in 1..=B16 {
                            let n0 = if c0 && b0 < B16 { 2 } else { 1 };
                            for b1 in 1..=B16 {
                                let n1 = if c1 && b1 < B16 { 2 } else { 1 };
                                let (val, _, _) = solve_stage_game_2x2(&q_plane[b0][b1], n0, n1);
                                out.v_table[slp0][slp1][b0][b1] = val;
                            }
                        }
                    }
                }
            }
        }

        // 2. Awake stage game `(slp0, slp1) = (0, 0)` using the Schur-eliminated sleeping values
        eval_plane_q(0, 0, &out.v_table, GAMMA, &mut q_plane);
        let mut next_awake = out.v_table[0][0];
        let mut max_diff = 0.0f32;
        let mut hm0 = [0u32; B16 + 1];
        let mut hm1 = [0u32; B16 + 1];

        for b0 in 1..=B16 {
            let n0 = if ops[0].has_heal && b0 < B16 { 2 } else { 1 };
            for b1 in 1..=B16 {
                let n1 = if ops[1].has_heal && b1 < B16 { 2 } else { 1 };
                let (val, h0, h1) = solve_stage_game_2x2(&q_plane[b0][b1], n0, n1);
                if h0 {
                    hm0[b0] |= 1u32 << b1;
                }
                if h1 {
                    hm1[b1] |= 1u32 << b0;
                }
                let d = (val - out.v_table[0][0][b0][b1]).abs();
                if d > max_diff {
                    max_diff = d;
                }
                next_awake[b0][b1] = val;
            }
        }
        out.v_table[0][0] = next_awake;
        heal_mask0 = hm0;
        heal_mask1 = hm1;
        if _iter >= 4 && max_diff < 2e-4 {
            break;
        }
    }

    out.heal_mask0 = heal_mask0;
    out.heal_mask1 = heal_mask1;
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

#[inline]
fn with_solved_bellman<R>(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
    f: impl FnOnce(&CachedBellman) -> R,
) -> R {
    let l1_key = l1_state_key(cb, table, s0, slot0, s1, slot1);
    let l1_idx = (l1_key.0 as usize) & (L1_SETS - 1);
    let mut f_opt = Some(f);

    if let Some(res) = BELLMAN_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let set = cache.l1[l1_idx];
        for way in 0..L1_WAYS {
            let w_entry = set[way];
            if w_entry.key == l1_key {
                let slot = w_entry.slot as usize;
                if slot < cache.pool.len() && cache.slot_gen[slot] == w_entry.gen {
                    if way > 0 {
                        for w in (1..=way).rev() {
                            cache.l1[l1_idx][w] = set[w - 1];
                        }
                        cache.l1[l1_idx][0] = w_entry;
                    }
                    return Some((f_opt.take().unwrap())(&cache.pool[slot]));
                }
            }
        }
        None
    }) {
        return res;
    }

    let (ops, max_s0, max_s1, spd_ord) = Side1DOps::build_pair(cb, table, s0, slot0, s1, slot1);
    let ruleset_tag = table.ruleset.fingerprint_tag();
    let mutual_eot_val = mutual_eot_payoff(cb, table, s0, slot0, s1, slot1, spd_ord);
    let op_key = l2_operator_key(
        ruleset_tag,
        mutual_eot_val,
        max_s0,
        max_s1,
        spd_ord,
        &ops,
    );
    let l2_idx = (op_key.0 as usize) & (L2_SETS - 1);

    if let Some(res) = BELLMAN_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let set = &mut cache.l2[l2_idx];
        for way in 0..L2_WAYS {
            if set.keys[way] == op_key {
                let slot = set.slots[way];
                if way > 0 {
                    for w in (1..=way).rev() {
                        set.keys[w] = set.keys[w - 1];
                        set.slots[w] = set.slots[w - 1];
                    }
                    set.keys[0] = op_key;
                    set.slots[0] = slot;
                }
                let gen = cache.slot_gen[slot as usize];
                cache.record_l1(l1_key, slot, gen);
                return Some((f_opt.take().unwrap())(&cache.pool[slot as usize]));
            }
        }
        None
    }) {
        return res;
    }

    let f = f_opt.unwrap();
    BELLMAN_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let last_empty = cache.l2[l2_idx].keys[L2_WAYS - 1] == Key128::default();
        let slot_idx = if last_empty && cache.pool.len() < MAX_CACHE_SLOTS {
            let idx = cache.pool.len();
            cache.pool.push(CachedBellman::empty());
            idx
        } else {
            cache.l2[l2_idx].slots[L2_WAYS - 1] as usize
        };
        let gen = cache.slot_gen[slot_idx].wrapping_add(1).max(1);
        cache.slot_gen[slot_idx] = gen;

        {
            let set = &mut cache.l2[l2_idx];
            for w in (1..L2_WAYS).rev() {
                set.keys[w] = set.keys[w - 1];
                set.slots[w] = set.slots[w - 1];
            }
            set.keys[0] = op_key;
            set.slots[0] = slot_idx as u16;
        }

        cache.pool[slot_idx].op_key = op_key;
        solve_bellman_into(
            max_s0,
            max_s1,
            spd_ord,
            mutual_eot_val,
            &ops,
            &mut cache.pool[slot_idx],
        );
        cache.record_l1(l1_key, slot_idx as u16, gen);
        f(&cache.pool[slot_idx])
    })
}

#[inline]
fn side_sleep_rem(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> usize {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let ispec = table.item_spec(mon.item);
    let berry = ispec.is_mintberry || ispec.is_miracleberry;
    if mon.status_raw() == Status::Slp as u8 && !berry {
        let max_slp = if table.ruleset.is_castle() {
            MAX_SLP_STATES
        } else {
            2
        };
        (mon.status_counter().saturating_sub(1) as usize).min(max_slp)
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
    for m_i in 0..table.mons[idx].num_moves as usize {
        if mon.pp[m_i] > 0 && !mon.is_slot_disabled(m_i) {
            let ms = &table.move_specs[idx][m_i];
            if ms.is_half_heal
                || ms.is_rest
                || ms.special_kind == nc2000_engine::compact::SpecialMoveKind::WeatherHeal
            {
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
        let v = with_solved_bellman(cb, table, s0, slot0, s1, slot1, |cached| {
            eval_exact_root_table(&cached.v_table, cb, table, s0, slot0, s1, slot1)
        });
        return (2.0 * v - 1.0).clamp(-1.0, 1.0);
    }

    // Pure stopping-time race from the 1D Toeplitz survival curves S_0(k), S_1(k)
    let steps = 6usize;
    let d0 = survival_curve_detail(cb, table, s1, slot1, s0, slot0, hp0, steps, true);
    let d1 = survival_curve_detail(cb, table, s0, slot0, s1, slot1, hp1, steps, true);

    let spd_ord = speed_ord_u8(cb, table, s0, slot0, s1, slot1);
    let spd_sign = match spd_ord {
        2 => 1.0,
        0 => -1.0,
        _ => 0.0,
    };
    let eot_sign = (2.0 * (mutual_eot_payoff(cb, table, s0, slot0, s1, slot1, spd_ord) as f64)
        - 1.0)
        .clamp(-1.0, 1.0);

    let mut margin = 0.0f64;
    for k in 1..=steps {
        let f0 = (d0.surv[k - 1] - d0.surv[k]).max(0.0);
        let f1 = (d1.surv[k - 1] - d1.surv[k]).max(0.0);
        let sim_ko = f0 * f1;
        let sim_eot = (d0.eot_ko[k].min(f0) * d1.eot_ko[k].min(f1)).min(sim_ko);
        let sim_atk = (sim_ko - sim_eot).max(0.0);
        margin += f1 * d0.surv[k] - f0 * d1.surv[k] + sim_atk * spd_sign + sim_eot * eot_sign;
    }
    margin += d0.surv[steps] * d1.surv[steps] * (d0.mean_hp_frac - d1.mean_hp_frac);
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
        cb.mons[idx0].perish_turns()
    } else {
        0
    };
    let p1_perish = if (cb.mons[idx1].vol_flags & vol_bit::PERISH_SONG) != 0 {
        cb.mons[idx1].perish_turns()
    } else {
        0
    };
    let p0_trapped = cb.mons[idx0].is_trapped()
        || cb.sides[0].alive_mask.count_ones() <= 1;
    let p1_trapped = cb.mons[idx1].is_trapped()
        || cb.sides[1].alive_mask.count_ones() <= 1;

    if p0_perish == 0 && p1_perish == 0 && p0_trapped == p1_trapped {
        return 0.0;
    }

    let mut need_s0 = 0usize;
    let mut need_s1 = 0usize;
    let mut any_term = false;

    let mut b1_info: Option<(usize, bool)> = None;
    if p1_perish > 0 && p1_trapped && !p0_trapped {
        let hits = p1_perish.saturating_sub(1).min(3) as usize;
        let p0_prot = cb.mons[idx0].stall_counter() == 0
            && (cb.mons[idx0].sub_hp() == 0 || hits >= 2)
            && (0..table.mons[idx0].num_moves as usize)
                .any(|m| cb.mons[idx0].pp[m] > 0 && table.move_specs[idx0][m].is_protect);
        let p1_phaze = (0..table.mons[idx1].num_moves as usize).any(|m| {
            cb.mons[idx1].pp[m] > 0
                && table.move_specs[idx1][m].special_kind
                    == nc2000_engine::compact::SpecialMoveKind::Phaze
        });
        let eff_hits = if p0_prot && hits > 0 { hits - 1 } else { hits };
        let use_phaze = p1_phaze && p1_perish >= 2;
        need_s0 = need_s0.max(eff_hits);
        if use_phaze {
            need_s1 = need_s1.max(1);
        }
        b1_info = Some((eff_hits, use_phaze));
        any_term = true;
    }

    let mut b2_info: Option<(usize, bool)> = None;
    if p0_perish > 0 && p0_trapped && !p1_trapped {
        let hits = p0_perish.saturating_sub(1).min(3) as usize;
        let p1_prot = cb.mons[idx1].stall_counter() == 0
            && (cb.mons[idx1].sub_hp() == 0 || hits >= 2)
            && (0..table.mons[idx1].num_moves as usize)
                .any(|m| cb.mons[idx1].pp[m] > 0 && table.move_specs[idx1][m].is_protect);
        let p0_phaze = (0..table.mons[idx0].num_moves as usize).any(|m| {
            cb.mons[idx0].pp[m] > 0
                && table.move_specs[idx0][m].special_kind
                    == nc2000_engine::compact::SpecialMoveKind::Phaze
        });
        let eff_hits = if p1_prot && hits > 0 { hits - 1 } else { hits };
        let use_phaze = p0_phaze && p0_perish >= 2;
        need_s1 = need_s1.max(eff_hits);
        if use_phaze {
            need_s0 = need_s0.max(1);
        }
        b2_info = Some((eff_hits, use_phaze));
        any_term = true;
    }

    let mut b3_hits: Option<usize> = None;
    if p1_perish == 0 && p1_trapped && !p0_trapped {
        let p0_has_song = (0..table.mons[idx0].num_moves as usize).any(|m| {
            cb.mons[idx0].pp[m] > 0
                && table.move_specs[idx0][m].special_kind
                    == nc2000_engine::compact::SpecialMoveKind::PerishSong
        });
        if p0_has_song {
            let p0_prot = (0..table.mons[idx0].num_moves as usize)
                .any(|m| cb.mons[idx0].pp[m] > 0 && table.move_specs[idx0][m].is_protect);
            let hits = if p0_prot { 2 } else { 3 };
            need_s0 = need_s0.max(hits);
            b3_hits = Some(hits);
            any_term = true;
        }
    }

    let mut b4_hits: Option<usize> = None;
    if p0_perish == 0 && p0_trapped && !p1_trapped {
        let p1_has_song = (0..table.mons[idx1].num_moves as usize).any(|m| {
            cb.mons[idx1].pp[m] > 0
                && table.move_specs[idx1][m].special_kind
                    == nc2000_engine::compact::SpecialMoveKind::PerishSong
        });
        if p1_has_song {
            let p1_prot = (0..table.mons[idx1].num_moves as usize)
                .any(|m| cb.mons[idx1].pp[m] > 0 && table.move_specs[idx1][m].is_protect);
            let hits = if p1_prot { 2 } else { 3 };
            need_s1 = need_s1.max(hits);
            b4_hits = Some(hits);
            any_term = true;
        }
    }

    let mut both_trapped_equal_perish: Option<usize> = None;
    if p0_perish > 0 && p0_perish == p1_perish && p0_trapped && p1_trapped {
        let hits = p0_perish.saturating_sub(1).min(3) as usize;
        need_s0 = need_s0.max(hits);
        need_s1 = need_s1.max(hits);
        both_trapped_equal_perish = Some(hits);
        any_term = true;
    }

    if !any_term {
        return 0.0;
    }

    let (s0, _) = survival_curve_k_with_mean_hp(cb, table, 1, 0, hp0, need_s0, false);
    let (s1, _) = survival_curve_k_with_mean_hp(cb, table, 0, 1, hp1, need_s1, false);
    let mut margin = 0.0f64;

    if let Some((eff_hits, use_phaze)) = b1_info {
        let p_not_phazed = if use_phaze { 1.0 - s1[1] } else { 1.0 };
        margin += s0[eff_hits] * p_not_phazed;
    }
    if let Some((eff_hits, use_phaze)) = b2_info {
        let p_not_phazed = if use_phaze { 1.0 - s0[1] } else { 1.0 };
        margin -= s1[eff_hits] * p_not_phazed;
    }
    // Trapped before Perish Song is sung: probability trapper survives 1 extra turn to sing + countdown
    if let Some(hits) = b3_hits {
        margin += s0[hits];
    }
    if let Some(hits) = b4_hits {
        margin -= s1[hits];
    }
    if let Some(hits) = both_trapped_equal_perish {
        let both_surv = s0[hits] * s1[hits];
        if table.ruleset.is_castle() {
            // Castle PBS (`Probe 12I`): 1v1 simultaneous Perish faint resolves by
            // `perish_dbond_lastmon_forfeit` (caster forfeits) when enabled, else draw (`0.0`);
            // non-1v1 simultaneous Perish has no seat bias (`0.0`).
            let n0 = cb.sides[0].alive_mask.count_ones();
            let n1 = cb.sides[1].alive_mask.count_ones();
            let sign = if n0 == 1 && n1 == 1 {
                let forfeit = table
                    .ruleset
                    .castle_overlay()
                    .is_some_and(|c| c.perish_dbond_lastmon_forfeit);
                let p1_cast = (cb.turn_extra_flags & (1 << 2)) != 0;
                let p2_cast = (cb.turn_extra_flags & (1 << 3)) != 0;
                if forfeit && (p1_cast ^ p2_cast) {
                    if p1_cast { -1.0 } else { 1.0 }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            margin += both_surv * sign;
        } else {
            // PS Stadium 2 (`turn.rs:341`): faster side faints first in queue and wins 1v1 tie-break.
            let spd_sign = match crate::eval::compare_action_speed(cb, table, 0, slot0, 1, slot1) {
                Ordering::Greater => 1.0,
                Ordering::Less => -1.0,
                Ordering::Equal => 0.0,
            };
            margin += both_surv * spd_sign;
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
        let spd_eq =
            crate::eval::compare_action_speed(cb, table, 0, s0, 1, s1) == Ordering::Equal;
        let is_frz = m0.status_raw() == Status::Frz as u8 || m1.status_raw() == Status::Frz as u8;
        let is_rech = m0.must_recharge() || m1.must_recharge();
        let exact_mechanics_override = spd_eq
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
    let me_ispec = table.item_spec(me.item);
    let is_asleep_this_turn = me.status_raw() == Status::Slp as u8
        && me.status_counter() >= 2
        && !(me_ispec.is_mintberry || me_ispec.is_miracleberry);
    if is_asleep_this_turn {
        if (table.mons[att_idx].role_flags & role_bit::SLEEP_USABLE) == 0 {
            return None;
        }
        for &c in cs {
            if c < 4
                && table.move_specs[att_idx][c as usize].special_kind
                    == nc2000_engine::compact::SpecialMoveKind::SleepTalk
            {
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

    // 2. Protect during active Perish countdown (`perish_turns >= 2`) when trapping
    let me_perish = if (me.vol_flags & vol_bit::PERISH_SONG) != 0 {
        me.perish_turns()
    } else {
        0
    };
    let foe_perish = if (foe.vol_flags & vol_bit::PERISH_SONG) != 0 {
        foe.perish_turns()
    } else {
        0
    };
    if foe_perish >= 2
        && (table.mons[att_idx].role_flags & role_bit::PROTECT) != 0
        && me.stall_counter() == 0
        && me.sub_hp() == 0
    {
        let me_trapped = me.is_trapped()
            || cb.sides[side].alive_mask.count_ones() <= 1;
        let foe_trapped = foe.is_trapped()
            || cb.sides[def_side].alive_mask.count_ones() <= 1;
        if foe_trapped && !me_trapped {
            for &c in cs {
                if c < 4 && table.move_specs[att_idx][c as usize].is_protect {
                    return Some(c);
                }
            }
        }
    }

    // 3. Bellman stage-game policy lookup for Heal / Rest when awake and strictly below full HP
    let me_hp = me.hp();
    let me_sub_up = (me.vol_flags & vol_bit::SUBSTITUTE) != 0;
    if (table.mons[att_idx].role_flags & (role_bit::HALF_HEAL | role_bit::REST)) != 0
        && me_perish == 0
        && me_hp < maxhp
        && !(me_sub_up && me_hp * 4 >= maxhp)
    {
        let b_me = Side1DOps::to_b16(me_hp, maxhp) as usize;
        let is_frz = me.status_raw() == Status::Frz as u8
            && !me_ispec.is_miracleberry;
        if b_me < B16 && !is_frz {
            let mut heal_act: Option<u8> = None;
            for &c in cs {
                if c < 4 {
                    let ms = &table.move_specs[att_idx][c as usize];
                    if ms.is_half_heal
                        || ms.special_kind
                            == nc2000_engine::compact::SpecialMoveKind::WeatherHeal
                    {
                        heal_act = Some(c);
                        break;
                    } else if ms.is_rest && heal_act.is_none() {
                        heal_act = Some(c);
                    }
                }
            }
            if let Some(h_act) = heal_act {
                let foe_has_perish = (table.mons[def_idx].role_flags
                    & role_bit::PERISH_PHAZE_TRAP)
                    != 0
                    && (0..table.mons[def_idx].num_moves as usize).any(|m| {
                        foe.pp[m] > 0
                            && table.move_specs[def_idx][m].special_kind
                                == nc2000_engine::compact::SpecialMoveKind::PerishSong
                    });
                if !foe_has_perish {
                    let me_faster = crate::eval::compare_action_speed(
                        cb, table, side, att_slot, def_side, def_slot,
                    ) == Ordering::Greater;
                    if me_faster
                        && cb.best_hit_fraction(table, side, att_slot, def_side, def_slot)
                            >= 1.0
                    {
                        return None;
                    }
                    let b_foe =
                        Side1DOps::to_b16(foe.hp(), table.mons[def_idx].maxhp.max(1)) as usize;
                    let (s0_slot, s1_slot, b0, b1) = if side == 0 {
                        (att_slot, def_slot, b_me, b_foe)
                    } else {
                        (def_slot, att_slot, b_foe, b_me)
                    };
                    let bellman_wants_heal =
                        with_solved_bellman(cb, table, 0, s0_slot, 1, s1_slot, |cached| {
                            if side == 0 {
                                ((cached.heal_mask0[b0] >> b1) & 1) != 0
                            } else {
                                ((cached.heal_mask1[b1] >> b0) & 1) != 0
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

        let v01_tbl = with_solved_bellman(&cb01, &t01, 0, 0, 1, 0, |c| c.v_table);
        let v10_tbl = with_solved_bellman(&cb10, &t10, 0, 0, 1, 0, |c| c.v_table);

        // Check exact antisymmetry across the entire 16x16x3x3 Bellman table
        for s0 in 0..3 {
            for s1 in 0..3 {
                for b0 in 1..=B16 {
                    for b1 in 1..=B16 {
                        let v01 = v01_tbl[s0][s1][b0][b1];
                        let v10 = v10_tbl[s1][s0][b1][b0];
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

    #[test]
    fn canonical_operator_cache_and_toxic_stack_invariants() {
        let dex = conformance::load_dex();
        let suicune_a = make_set(
            "Suicune",
            55,
            "Leftovers",
            &["Surf", "Roar", "Rest", "Toxic"],
        );
        // Permuted move order + different unused status move -> different `static_fp` (L1 miss),
        // but identical 1D Bellman operator (`Surf` + `Rest`, no Sleep Talk) -> L2 hit!
        let suicune_b = make_set(
            "Suicune",
            55,
            "Leftovers",
            &["Rest", "Reflect", "Surf", "Growl"],
        );
        let snorlax = make_set(
            "Snorlax",
            55,
            "Leftovers",
            &["Body Slam", "Earthquake", "Curse", "Rest"],
        );

        let mut ba =
            Battle::from_fixture(&dex, "1,2,3,4", &[suicune_a], &[snorlax.clone()]).unwrap();
        ba.choose(&dex, 0, "team 1").unwrap();
        ba.choose(&dex, 1, "team 1").unwrap();
        let (ta, mut cba) = CompactBattle::from_battle(&ba, &dex);

        let mut bb = Battle::from_fixture(&dex, "1,2,3,4", &[suicune_b], &[snorlax]).unwrap();
        bb.choose(&dex, 0, "team 1").unwrap();
        bb.choose(&dex, 1, "team 1").unwrap();
        let (tb, cbb) = CompactBattle::from_battle(&bb, &dex);

        assert_ne!(ta.mons[0].static_fp, tb.mons[0].static_fp);
        assert_ne!(
            l1_state_key(&cba, &ta, 0, 0, 1, 0),
            l1_state_key(&cbb, &tb, 0, 0, 1, 0)
        );

        let (key_a, hm0_a, hm1_a, vt_a) =
            with_solved_bellman(&cba, &ta, 0, 0, 1, 0, |c| (c.op_key, c.heal_mask0, c.heal_mask1, c.v_table));
        let (key_b, hm0_b, hm1_b) =
            with_solved_bellman(&cbb, &tb, 0, 0, 1, 0, |c| (c.op_key, c.heal_mask0, c.heal_mask1));
        assert_eq!(key_a, key_b);
        assert_eq!(hm0_a, hm0_b);
        assert_eq!(hm1_a, hm1_b);

        // Verify bit-exact 0.0-ULP equivalence against a fresh uncached solve
        let (ops, ms0, ms1, spd_ord) = Side1DOps::build_pair(&cba, &ta, 0, 0, 1, 0);
        let mutual_eot_val = mutual_eot_payoff(&cba, &ta, 0, 0, 1, 0, spd_ord);
        let mut fresh = CachedBellman::empty();
        solve_bellman_into(ms0, ms1, spd_ord, mutual_eot_val, &ops, &mut fresh);
        for s0 in 0..3 {
            for s1 in 0..3 {
                for b0 in 0..=B16 {
                    for b1 in 0..=B16 {
                        assert_eq!(
                            vt_a[s0][s1][b0][b1].to_bits(),
                            fresh.v_table[s0][s1][b0][b1].to_bits()
                        );
                    }
                }
            }
        }

        // Gen 2 Toxic-stacking flag (`0x10` in status_counter on Burn/Poison) must produce
        // distinct L1 keys and distinct L2 Bellman operators when counter >= 3.
        cba.mons[0].set_status(Status::Brn as u8, 4);
        let k_brn_plain = l1_state_key(&cba, &ta, 0, 0, 1, 0);
        let op_brn_plain = with_solved_bellman(&cba, &ta, 0, 0, 1, 0, |c| c.op_key);

        cba.mons[0].set_status(Status::Brn as u8, 0x10 | 4);
        let k_brn_tox = l1_state_key(&cba, &ta, 0, 0, 1, 0);
        let op_brn_tox = with_solved_bellman(&cba, &ta, 0, 0, 1, 0, |c| c.op_key);

        assert_ne!(k_brn_plain, k_brn_tox);
        assert_ne!(op_brn_plain, op_brn_tox);
    }

    #[test]
    fn toeplitz_coupled_race_uses_bench_slot_stats_and_moves_not_active() {
        let dex = conformance::load_dex();
        let unown = make_set("Unown", 50, "No Item", &["Hidden Power"]);
        let marowak = make_set(
            "Marowak",
            55,
            "Thick Club",
            &["Earthquake", "Rock Slide", "Fire Blast", "Swords Dance"],
        );
        let jolteon = make_set(
            "Jolteon",
            50,
            "Magnet",
            &["Thunderbolt", "Hidden Power", "Agility", "Baton Pass"],
        );

        let mut b = Battle::from_fixture(
            &dex,
            "1,2,3,4",
            &[unown, marowak],
            &[jolteon],
        )
        .unwrap();
        b.choose(&dex, 0, "team 1,2").unwrap();
        b.choose(&dex, 1, "team 1").unwrap();
        let (table, cb) = CompactBattle::from_battle(&b, &dex);

        // Active is slot 0 (Unown), Bench is slot 1 (Marowak).
        // Race for bench Marowak (side 0, slot 1) vs Jolteon (side 1, slot 0):
        // Marowak is immune to Thunderbolt and OHKOs Jolteon with Earthquake, so Marowak must strongly win!
        let r_unown = toeplitz_coupled_race(&cb, &table, 0, 0, 1, 0);
        let r_marowak_bench = toeplitz_coupled_race(&cb, &table, 0, 1, 1, 0);
        assert!(
            r_unown < 0.0,
            "Active Unown (slot 0) loses to Jolteon, got {r_unown}"
        );
        assert!(
            r_marowak_bench > 0.70,
            "Bench Marowak (slot 1) must use slot 1's own stats/moves/types and crush Jolteon, got {r_marowak_bench}"
        );
        // Verify exact side-swap antisymmetry: race(0,1 vs 1,0) == -race(1,0 vs 0,1).
        let r_swapped = toeplitz_coupled_race(&cb, &table, 1, 0, 0, 1);
        assert!(
            (r_marowak_bench + r_swapped).abs() < 1e-12,
            "Antisymmetry violated: {r_marowak_bench} vs {r_swapped}"
        );
    }

    #[test]
    fn weather_heal_moonlight_is_priced_across_weathers_and_antisymmetric() {
        let dex = conformance::load_dex();
        let umbreon_heal = make_set(
            "Umbreon",
            50,
            "No Item",
            &["Body Slam", "Moonlight"],
        );
        let umbreon_noheal = make_set(
            "Umbreon",
            50,
            "No Item",
            &["Body Slam", "Sand-Attack"],
        );

        let mut b =
            Battle::from_fixture(&dex, "1,2,3,4", &[umbreon_heal.clone()], &[umbreon_noheal.clone()])
                .unwrap();
        b.choose(&dex, 0, "team 1").unwrap();
        b.choose(&dex, 1, "team 1").unwrap();
        let (table, mut cb) = CompactBattle::from_battle(&b, &dex);

        let mut b_rev =
            Battle::from_fixture(&dex, "1,2,3,4", &[umbreon_noheal], &[umbreon_heal]).unwrap();
        b_rev.choose(&dex, 0, "team 1").unwrap();
        b_rev.choose(&dex, 1, "team 1").unwrap();
        let (table_rev, mut cb_rev) = CompactBattle::from_battle(&b_rev, &dex);

        let r_clear = toeplitz_coupled_race(&cb, &table, 0, 0, 1, 0);
        let r_clear_rev = toeplitz_coupled_race(&cb_rev, &table_rev, 0, 0, 1, 0);
        assert!(r_clear > 0.25, "Moonlight Umbreon must beat heal-less mirror, got {r_clear}");
        assert!((r_clear + r_clear_rev).abs() < 1e-5, "antisymmetry clear: {r_clear} vs {r_clear_rev}");

        // Sunny Day: full-HP Moonlight >= clear Moonlight >= Rain (1/4 heal)
        cb.weather_kind = CompactWeather::Sun;
        cb.weather_turns = 5;
        cb_rev.weather_kind = CompactWeather::Sun;
        cb_rev.weather_turns = 5;
        let r_sun = toeplitz_coupled_race(&cb, &table, 0, 0, 1, 0);
        let r_sun_rev = toeplitz_coupled_race(&cb_rev, &table_rev, 0, 0, 1, 0);
        assert!((r_sun + r_sun_rev).abs() < 1e-5, "antisymmetry sun: {r_sun} vs {r_sun_rev}");

        cb.weather_kind = CompactWeather::Rain;
        cb.weather_turns = 5;
        cb_rev.weather_kind = CompactWeather::Rain;
        cb_rev.weather_turns = 5;
        let r_rain = toeplitz_coupled_race(&cb, &table, 0, 0, 1, 0);
        let r_rain_rev = toeplitz_coupled_race(&cb_rev, &table_rev, 0, 0, 1, 0);
        assert!((r_rain + r_rain_rev).abs() < 1e-5, "antisymmetry rain: {r_rain} vs {r_rain_rev}");
        assert!(r_sun >= r_clear - 1e-5 && r_clear > r_rain + 0.02);

        // Rollout picks Moonlight when Umbreon is low HP
        cb.weather_turns = 0;
        cb.mons[0].hp_pool[0] = table.mons[0].maxhp / 4;
        assert_eq!(compact_rollout_linear_pick(&cb, &table, 0, &[0, 1]), Some(1));
    }

    #[test]
    fn l1_cache_keys_track_dynamic_thief_items_and_protect_fails_behind_sub() {
        let dex = conformance::load_dex();
        let wak = make_set("Marowak", 50, "No Item", &["Earthquake", "Rock Slide", "Thief", "Protect"]);
        let bench = make_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake"]);
        let foe_tc = make_set("Cubone", 50, "Thick Club", &["Body Slam", "Protect"]);
        let foe_ch = make_set("Typhlosion", 50, "Charcoal", &["Fire Blast", "Protect"]);
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &[wak, bench], &[foe_tc, foe_ch]).unwrap();
        b.choose(&dex, 0, "team 1,2").unwrap();
        b.choose(&dex, 1, "team 1,2").unwrap();
        let (table, mut cb) = CompactBattle::from_battle(&b, &dex);

        let k_none = l1_state_key(&cb, &table, 0, 0, 1, 0);
        let tc_item = cb.mons[3].item;
        let ch_item = cb.mons[4].item;
        assert_ne!(tc_item, 0);
        assert_ne!(ch_item, 0);

        cb.mons[0].item = tc_item;
        let k_tc = l1_state_key(&cb, &table, 0, 0, 1, 0);
        cb.mons[0].item = ch_item;
        let k_ch = l1_state_key(&cb, &table, 0, 0, 1, 0);
        assert_ne!(k_none, k_tc, "L1 key must distinguish stolen Thick Club from itemless");
        assert_ne!(k_tc, k_ch, "L1 key must distinguish stolen Thick Club from stolen Charcoal");

        // Positive control: untrapped side 0 vs trapped Perish(2) foe WITHOUT Sub MUST pick Protect (slot 3).
        cb.mons[0].item = 0;
        cb.mons[0].vol_flags |= vol_bit::PERISH_SONG;
        cb.mons[0].set_perish_turns(2);
        cb.mons[3].vol_flags |= vol_bit::PERISH_SONG | vol_bit::TRAPPED_OR_LOCKED;
        cb.mons[3].set_perish_turns(2);
        assert_eq!(
            compact_rollout_linear_pick(&cb, &table, 0, &[0, 1, 2, 3]),
            Some(3),
            "Positive control: rollout must pick Protect against trapped Perish foe when not behind Sub"
        );
        let margin_no_sub = compact_perish_margin(&cb, &table);

        // Negative control: adding own Substitute makes Protect fail in Gen 2 -> must NOT pick Protect!
        cb.mons[0].vol_flags |= vol_bit::SUBSTITUTE;
        cb.mons[0].hp_pool[1] = 30;
        assert_ne!(
            compact_rollout_linear_pick(&cb, &table, 0, &[0, 1, 2, 3]),
            Some(3),
            "Rollout must not pick Protect when behind own Substitute"
        );
        let margin_with_sub = compact_perish_margin(&cb, &table);
        assert!(
            margin_with_sub <= margin_no_sub,
            "Perish margin must not credit impossible Protect behind own Substitute"
        );
    }
}
