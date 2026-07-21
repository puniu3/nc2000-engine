//! Bit-exact port of Pokemon Showdown's Gen5RNG + PRNG wrapper (sim/prng.ts).
//!
//! The LCG is `x_{n+1} = x_n * 0x5D588B656C078965 + 0x269EC3 (mod 2^64)`;
//! each draw returns the upper 32 bits. PS stores the seed as four 16-bit
//! big-endian limbs and serializes it as `"l0,l1,l2,l3"` (decimal) — we must
//! emit the identical string for snapshot parity.

const MULT: u64 = 0x5D58_8B65_6C07_8965;
const ADD: u64 = 0x26_9EC3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prng {
    seed: u64,
}

impl Prng {
    pub fn new(seed: u64) -> Self {
        Prng { seed }
    }

    /// Parses both PS seed spellings:
    /// - `"gen5,0123456789abcdef"` (16 hex chars)
    /// - `"4660,22136,4660,22136"` (legacy: four decimal 16-bit limbs)
    pub fn from_seed_str(s: &str) -> Option<Self> {
        if let Some(hex) = s.strip_prefix("gen5,") {
            if hex.len() != 16 {
                return None;
            }
            return u64::from_str_radix(hex, 16).ok().map(Prng::new);
        }
        let limbs: Vec<u64> = s.split(',').map(|p| p.trim().parse().ok()).collect::<Option<_>>()?;
        if limbs.len() != 4 || limbs.iter().any(|&l| l > 0xFFFF) {
            return None;
        }
        Some(Prng::new(limbs[0] << 48 | limbs[1] << 32 | limbs[2] << 16 | limbs[3]))
    }

    /// PS `Gen5RNG.getSeed()` format: decimal limbs joined by commas.
    pub fn seed_str(&self) -> String {
        format!(
            "{},{},{},{}",
            (self.seed >> 48) & 0xFFFF,
            (self.seed >> 32) & 0xFFFF,
            (self.seed >> 16) & 0xFFFF,
            self.seed & 0xFFFF
        )
    }

    pub fn next_u32(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(MULT).wrapping_add(ADD);
        (self.seed >> 32) as u32
    }

    /// PS `random(n)`: `Math.floor(next * n / 2^32)`.
    ///
    /// Exactly equal to `(next * n) >> 32` in u64 arithmetic as long as
    /// `next * n < 2^53` (JS f64 mantissa) — guaranteed for `n < 2^21`,
    /// far above anything the battle engine passes.
    pub fn random(&mut self, n: u32) -> u32 {
        debug_assert!(n < (1 << 21), "random(n) parity with JS float math requires small n");
        ((self.next_u32() as u64 * n as u64) >> 32) as u32
    }

    /// PS `random(from, to)`: integer in `[from, to)`.
    pub fn random_range(&mut self, from: u32, to: u32) -> u32 {
        from + self.random(to - from)
    }

    /// PS `randomChance(num, den)`: true with probability num/den.
    pub fn random_chance(&mut self, num: u32, den: u32) -> bool {
        self.random(den) < num
    }

    /// PS `sample(items)`: uniform index into a slice.
    pub fn sample_index(&mut self, len: usize) -> usize {
        assert!(len > 0, "cannot sample an empty slice");
        self.random(len as u32) as usize
    }

    /// PS `shuffle(items, start, end)` — Fisher-Yates, exactly as PS does it
    /// (this is how speed ties are resolved; consumption order matters).
    pub fn shuffle<T>(&mut self, items: &mut [T], mut start: usize, end: usize) {
        while start + 1 < end {
            let next = self.random_range(start as u32, end as u32) as usize;
            if start != next {
                items.swap(start, next);
            }
            start += 1;
        }
    }
}

// ---------------------------------------------------------------- BattleRng

/// One chance-node consumption recorded in Oracle mode: the exact partition
/// of the underlying u32 draw into outcome classes (`counts[c]` = how many
/// of the 2^32 raw draws land in class `c`; the counts always sum to 2^32,
/// so `counts[c] / 2^32` is the class's exact probability), plus which class
/// this run took.
#[derive(Clone, Debug)]
pub struct Draw {
    pub label: &'static str,
    pub counts: Vec<u64>,
    pub chosen: usize,
}

/// Experimental damage-roll quotient used only by exhaustive enumeration.
/// Seeded play and `Exact` enumeration retain the bit-exact production path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DamageRollMode {
    #[default]
    Exact,
    /// Replace the complete damage distribution by its nearest attainable
    /// probability-weighted mean.
    Mean,
    /// Split only when a roll crosses an immediate HP semantic threshold.
    Threshold1,
    /// `Threshold1` plus next-hit and residual-damage death clocks.
    Threshold2,
    /// KO plus thresholds backed by an actually-held item or usable move;
    /// omits unconditional 1/4, 1/3, and 1/2 buckets.
    ThresholdLean,
    /// `ThresholdLean` ablations which remove one conservative exact escape.
    ThresholdLeanNoCounter,
    ThresholdLeanNoDrainRecoil,
    ThresholdLeanNoMultiHit,
    ThresholdLeanNoSubstitute,
    /// No conservative exact escapes; retains only semantic HP thresholds.
    ThresholdLeanMinimal,
    /// `ThresholdLean` plus only the next equal-hit death clock.
    ThresholdLeanNext,
    /// `ThresholdLean` plus only the residual-damage death clock.
    ThresholdLeanResidual,
    /// `ThresholdLean` plus both death clocks.
    ThresholdLeanClock,
    /// Two representatives for damage which can enter a usable-heal region.
    ThresholdHealSplit,
    /// `ThresholdLeanClock`, with low-HP heal-capable targets retaining
    /// exact damage rolls (heal/stall policy thresholds are contextual).
    ThresholdHeal,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DamageExactReason {
    #[default]
    None,
    DrainRecoil,
    MultiHit,
    Substitute,
    CounterBide,
    Heal,
}

/// State-local semantics visible after a damage roll. Thresholds are HP
/// values at which a downstream predicate changes (KO, HP fractions, etc.).
#[derive(Clone, Debug, Default)]
pub struct DamageRollContext {
    pub hp: i32,
    pub thresholds: Vec<i32>,
    pub residual_damage: i32,
    pub heal_threshold: i32,
    /// Damage bookkeeping can be observed numerically by mechanics such as
    /// Counter, Bide, drain, recoil, multi-hit, and Substitute. Conservative
    /// modes retain exact rolls; ablation modes can clear individual reasons.
    pub exact_reason: DamageExactReason,
}

impl Draw {
    pub const TOTAL: u64 = 1 << 32;

    pub fn prob(&self) -> f64 {
        self.counts[self.chosen] as f64 / Self::TOTAL as f64
    }
}

/// Oracle state: a script of prescribed class choices (positions past the
/// script's end take the first non-empty class) and the trace of every
/// consumption. `trace.len()` is the cursor.
#[derive(Clone, Debug, Default)]
pub struct Oracle {
    pub script: Vec<usize>,
    pub trace: Vec<Draw>,
    pub damage_mode: DamageRollMode,
}

/// The battle's RNG: either the bit-exact seeded LCG (normal play, search,
/// fixtures) or an enumeration Oracle that records the exact outcome-class
/// partition of every draw (M17e chance-node enumeration). All battle code
/// consumes randomness through this type; in Seeded mode every method's LCG
/// consumption is bit-identical to the pre-BattleRng `Prng` calls.
#[derive(Clone, Debug)]
pub struct BattleRng {
    pub lcg: Prng,
    pub oracle: Option<Box<Oracle>>,
}

/// # of raw u32 draws `x` with `floor(x * n / 2^32) == v` (PS `random(n)`).
fn uniform_count(n: u32, v: u32) -> u64 {
    let ceil = |a: u64, b: u64| a.div_ceil(b);
    ceil((v as u64 + 1) << 32, n as u64) - ceil((v as u64) << 32, n as u64)
}

/// # of raw u32 draws mapping to a `random(n)` value in `[lo, hi)`.
fn range_count(n: u32, lo: u32, hi: u32) -> u64 {
    let ceil = |a: u64, b: u64| a.div_ceil(b);
    ceil((hi as u64) << 32, n as u64) - ceil((lo as u64) << 32, n as u64)
}

impl BattleRng {
    pub fn seeded(lcg: Prng) -> Self {
        BattleRng { lcg, oracle: None }
    }

    /// Oracle mode following `script`; draws past the script take the first
    /// non-empty class. The `lcg` is untouched dead weight in this mode.
    pub fn enumerating(script: Vec<usize>) -> Self {
        Self::enumerating_with_damage_mode(script, DamageRollMode::Exact)
    }

    pub fn enumerating_with_damage_mode(script: Vec<usize>, damage_mode: DamageRollMode) -> Self {
        BattleRng {
            lcg: Prng::new(0),
            oracle: Some(Box::new(Oracle { script, trace: Vec::new(), damage_mode })),
        }
    }

    /// Seeded play reports `Exact`: abstraction is an Oracle-only experiment.
    pub fn damage_roll_mode(&self) -> DamageRollMode {
        self.oracle.as_ref().map_or(DamageRollMode::Exact, |o| o.damage_mode)
    }

    pub fn seed_str(&self) -> String {
        self.lcg.seed_str()
    }

    /// Oracle-mode class pick: scripted if in range, else first non-empty.
    fn pick(o: &mut Oracle, label: &'static str, counts: Vec<u64>) -> usize {
        assert_eq!(counts.iter().sum::<u64>(), Draw::TOTAL, "{label}: partition must cover u32");
        let pos = o.trace.len();
        let chosen = match o.script.get(pos) {
            Some(&c) => c,
            None => counts.iter().position(|&c| c > 0).expect("some class has mass"),
        };
        o.trace.push(Draw { label, counts, chosen });
        chosen
    }

    pub fn random(&mut self, n: u32) -> u32 {
        match &mut self.oracle {
            None => self.lcg.random(n),
            Some(o) => {
                let counts = (0..n).map(|v| uniform_count(n, v)).collect();
                Self::pick(o, "random", counts) as u32
            }
        }
    }

    pub fn random_range(&mut self, from: u32, to: u32) -> u32 {
        from + self.random(to - from)
    }

    pub fn random_chance(&mut self, num: u32, den: u32) -> bool {
        match &mut self.oracle {
            None => self.lcg.random_chance(num, den),
            Some(o) => {
                let t = range_count(den, 0, num.min(den));
                Self::pick(o, "chance", vec![t, Draw::TOTAL - t]) == 0
            }
        }
    }

    pub fn sample_index(&mut self, len: usize) -> usize {
        assert!(len > 0, "cannot sample an empty slice");
        self.random(len as u32) as usize
    }

    pub fn shuffle<T>(&mut self, items: &mut [T], mut start: usize, end: usize) {
        while start + 1 < end {
            let next = self.random_range(start as u32, end as u32) as usize;
            if start != next {
                items.swap(start, next);
            }
            start += 1;
        }
    }

    /// A draw whose value is discarded (in-game RNG burn kept for LCG
    /// parity). Oracle mode records nothing: one class, probability 1.
    pub fn burn(&mut self, n: u32) {
        if self.oracle.is_none() {
            self.lcg.random(n);
        }
    }

    /// `random(n)` consumed only through the bucket index it falls in.
    /// `cuts` are ascending exclusive upper bounds per bucket; the last cut
    /// must equal `n`. Seeded mode consumes exactly like `random(n)`; Oracle
    /// mode branches over buckets instead of all n values.
    pub fn random_bucketed(&mut self, n: u32, cuts: &'static [u32]) -> usize {
        debug_assert_eq!(*cuts.last().unwrap(), n);
        match &mut self.oracle {
            None => {
                let v = self.lcg.random(n);
                cuts.iter().position(|&c| v < c).unwrap()
            }
            Some(o) => {
                let mut lo = 0;
                let counts = cuts
                    .iter()
                    .map(|&c| {
                        let cnt = range_count(n, lo, c);
                        lo = c;
                        cnt
                    })
                    .collect();
                Self::pick(o, "bucketed", counts)
            }
        }
    }

    /// The gen-2 damage random factor applied at its observable boundary:
    /// `floor(damage * roll / 255)` for a roll uniform in [217, 256).
    /// Seeded mode consumes exactly one `random(39)` draw (bit-parity with
    /// PS `random_range(217, 256)`) and computes the same expression.
    /// Oracle mode branches over DISTINCT final damage values (Phase A
    /// integer-damage quotienting): raw roll classes with equal outcomes
    /// are merged arithmetically before any downstream execution — sound
    /// by construction, since the rest of the engine only ever observes
    /// the final value. The label "dmgvar" lets the enumeration driver
    /// apply its second-layer endpoint range merge on top (saturation
    /// ranges where DIFFERENT damages still coincide).
    pub fn apply_damage_variance(&mut self, damage: f64) -> f64 {
        self.apply_damage_variance_with_context(damage, None)
    }

    /// Damage variance with state-local semantic quotienting. Approximate
    /// groups retain exact probability mass; their representative is the
    /// attainable damage nearest that group's conditional mean. The new
    /// `dmgabs` label intentionally bypasses the exact enumerator's endpoint
    /// proof, which applies only to unmodified `dmgvar` classes.
    pub fn apply_damage_variance_with_context(
        &mut self,
        damage: f64,
        context: Option<&DamageRollContext>,
    ) -> f64 {
        let out = |c: u32| (damage * (217 + c) as f64 / 255.0).floor();
        match &mut self.oracle {
            None => out(self.lcg.random(39)),
            Some(o) => {
                // Group contiguous raw classes by exact final damage
                // (monotone in the roll, so contiguous grouping is total).
                let mut counts: Vec<u64> = Vec::new();
                let mut vals: Vec<f64> = Vec::new();
                for c in 0..39u32 {
                    let v = out(c);
                    if vals.last() == Some(&v) {
                        *counts.last_mut().unwrap() += uniform_count(39, c);
                    } else {
                        vals.push(v);
                        counts.push(uniform_count(39, c));
                    }
                }

                let exact_reason = context.map_or(DamageExactReason::None, |c| c.exact_reason);
                let mode = if exact_reason != DamageExactReason::None {
                    DamageRollMode::Exact
                } else {
                    o.damage_mode
                };
                if mode == DamageRollMode::Exact {
                    let label = match exact_reason {
                        DamageExactReason::None => "dmgvar",
                        DamageExactReason::DrainRecoil => "dmgvar-drain-recoil",
                        DamageExactReason::MultiHit => "dmgvar-multihit",
                        DamageExactReason::Substitute => "dmgvar-substitute",
                        DamageExactReason::CounterBide => "dmgvar-counter-bide",
                        DamageExactReason::Heal => "dmgvar-heal",
                    };
                    return vals[Self::pick(o, label, counts)];
                }

                #[derive(Clone, Copy, PartialEq, Eq)]
                struct Signature {
                    threshold_band: usize,
                    next_hit_clock: u8,
                    residual_clock: u8,
                    heal_band: u8,
                }

                let ctx = context.cloned().unwrap_or_default();
                let min_damage = vals.first().copied().unwrap_or(0.0) as i32;
                let max_damage = vals.last().copied().unwrap_or(0.0) as i32;
                let signature = |v: f64| {
                    if mode == DamageRollMode::Mean {
                        return Signature {
                            threshold_band: 0,
                            next_hit_clock: 0,
                            residual_clock: 0,
                            heal_band: 0,
                        };
                    }
                    let post_hp = (ctx.hp - v as i32).max(0);
                    let threshold_band =
                        ctx.thresholds.iter().filter(|&&threshold| post_hp <= threshold).count();
                    let next_clock = matches!(
                        mode,
                        DamageRollMode::Threshold2
                            | DamageRollMode::ThresholdLeanNext
                            | DamageRollMode::ThresholdLeanClock
                            | DamageRollMode::ThresholdHeal
                    );
                    let residual_clock_enabled = matches!(
                        mode,
                        DamageRollMode::Threshold2
                            | DamageRollMode::ThresholdLeanResidual
                            | DamageRollMode::ThresholdLeanClock
                            | DamageRollMode::ThresholdHeal
                    );
                    let next_hit_clock = if !next_clock || post_hp == 0 {
                        0
                    } else if post_hp <= min_damage {
                        1 // every roll of the same hit KOs next time
                    } else if post_hp <= max_damage {
                        2 // only some rolls KO next time
                    } else {
                        3 // cannot KO on the next equal hit
                    };
                    let residual_clock = if residual_clock_enabled
                        && ctx.residual_damage > 0
                        && post_hp > 0
                    {
                        ((post_hp + ctx.residual_damage - 1) / ctx.residual_damage).min(4) as u8
                    } else {
                        0
                    };
                    let heal_band = if mode == DamageRollMode::ThresholdHealSplit
                        && ctx.heal_threshold > 0
                        && (ctx.hp - min_damage).max(0) <= ctx.heal_threshold
                    {
                        1 + u8::from(v as i32 > (min_damage + max_damage) / 2)
                    } else {
                        0
                    };
                    Signature { threshold_band, next_hit_clock, residual_clock, heal_band }
                };

                let mut group_counts: Vec<u64> = Vec::new();
                let mut group_values: Vec<Vec<(f64, u64)>> = Vec::new();
                let mut last_signature: Option<Signature> = None;
                for (&value, &count) in vals.iter().zip(&counts) {
                    let sig = signature(value);
                    if last_signature != Some(sig) {
                        group_counts.push(0);
                        group_values.push(Vec::new());
                        last_signature = Some(sig);
                    }
                    *group_counts.last_mut().unwrap() += count;
                    group_values.last_mut().unwrap().push((value, count));
                }
                let representatives: Vec<f64> = group_values
                    .iter()
                    .map(|group| {
                        let mass: u64 = group.iter().map(|(_, count)| count).sum();
                        let mean = group.iter().map(|(value, count)| value * *count as f64).sum::<f64>()
                            / mass as f64;
                        group
                            .iter()
                            .min_by(|(a, _), (b, _)| {
                                (a - mean)
                                    .abs()
                                    .partial_cmp(&(b - mean).abs())
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap()
                            .0
                    })
                    .collect();
                representatives[Self::pick(o, "dmgabs", group_counts)]
            }
        }
    }

    /// PS `this.random(100) < accuracy` where accuracy is an f64 percent.
    /// Seeded mode reproduces the float comparison bit-exactly; Oracle mode
    /// merges to hit/miss.
    pub fn chance_percent(&mut self, a: f64) -> bool {
        match &mut self.oracle {
            None => (self.lcg.random(100) as f64) < a,
            Some(o) => {
                let hit_vals = (0..100u32).filter(|&v| (v as f64) < a).count() as u32;
                let t = range_count(100, 0, hit_vals);
                Self::pick(o, "percent", vec![t, Draw::TOTAL - t]) == 0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BattleRng, DamageExactReason, DamageRollContext, DamageRollMode, Draw, Prng};

    fn abstract_roll(
        mode: DamageRollMode,
        script: Vec<usize>,
        context: &DamageRollContext,
    ) -> (f64, super::Draw) {
        let mut rng = BattleRng::enumerating_with_damage_mode(script, mode);
        let value = rng.apply_damage_variance_with_context(100.0, Some(context));
        let draw = rng.oracle.unwrap().trace.into_iter().next().unwrap();
        (value, draw)
    }

    #[test]
    fn mean_damage_is_one_probability_preserving_class() {
        let context = DamageRollContext { hp: 90, thresholds: vec![0], ..Default::default() };
        let (value, draw) = abstract_roll(DamageRollMode::Mean, vec![], &context);
        assert_eq!(draw.label, "dmgabs");
        assert_eq!(draw.counts, vec![Draw::TOTAL]);
        assert!((85.0..=100.0).contains(&value));
    }

    #[test]
    fn threshold_damage_splits_ko_from_survival() {
        let context = DamageRollContext { hp: 90, thresholds: vec![0], ..Default::default() };
        let (survive, first) = abstract_roll(DamageRollMode::Threshold1, vec![0], &context);
        let (ko, second) = abstract_roll(DamageRollMode::Threshold1, vec![1], &context);
        assert_eq!(first.counts.len(), 2);
        assert_eq!(first.counts, second.counts);
        assert_eq!(first.counts.iter().sum::<u64>(), Draw::TOTAL);
        assert!(survive < 90.0, "representative {survive}");
        assert!(ko >= 90.0, "representative {ko}");
    }

    #[test]
    fn sensitive_damage_falls_back_to_exact_classes() {
        let context = DamageRollContext {
            hp: 90,
            thresholds: vec![0],
            exact_reason: DamageExactReason::CounterBide,
            ..Default::default()
        };
        let (_, draw) = abstract_roll(DamageRollMode::Threshold2, vec![], &context);
        assert_eq!(draw.label, "dmgvar-counter-bide");
        assert!(draw.counts.len() > 2);
        assert_eq!(draw.counts.iter().sum::<u64>(), Draw::TOTAL);
    }

    #[test]
    fn lean_clock_components_can_be_ablated_independently() {
        let context = DamageRollContext {
            hp: 180,
            thresholds: vec![0],
            residual_damage: 30,
            ..Default::default()
        };
        let (_, lean) = abstract_roll(DamageRollMode::ThresholdLean, vec![], &context);
        let (_, next) = abstract_roll(DamageRollMode::ThresholdLeanNext, vec![], &context);
        let (_, residual) =
            abstract_roll(DamageRollMode::ThresholdLeanResidual, vec![], &context);
        let (_, both) = abstract_roll(DamageRollMode::ThresholdLeanClock, vec![], &context);
        assert_eq!(lean.counts.len(), 1);
        assert!(next.counts.len() > lean.counts.len());
        assert!(residual.counts.len() > lean.counts.len());
        assert!(both.counts.len() >= next.counts.len());
        assert!(both.counts.len() >= residual.counts.len());
    }

    #[test]
    fn heal_split_adds_only_two_representatives() {
        let context = DamageRollContext {
            hp: 180,
            thresholds: vec![0],
            heal_threshold: 100,
            ..Default::default()
        };
        let (_, draw) = abstract_roll(DamageRollMode::ThresholdHealSplit, vec![], &context);
        assert_eq!(draw.counts.len(), 2);
    }

    #[test]
    fn seeded_context_path_is_bit_identical() {
        let mut ordinary = BattleRng::seeded(Prng::new(0x1234_5678_9abc_def0));
        let mut contextual = ordinary.clone();
        let context = DamageRollContext { hp: 1, thresholds: vec![0], ..Default::default() };
        let a = ordinary.apply_damage_variance(237.0);
        let b = contextual.apply_damage_variance_with_context(237.0, Some(&context));
        assert_eq!(a, b);
        assert_eq!(ordinary.seed_str(), contextual.seed_str());
    }
}
