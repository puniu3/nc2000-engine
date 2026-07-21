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
        BattleRng { lcg: Prng::new(0), oracle: Some(Box::new(Oracle { script, trace: Vec::new() })) }
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

    /// The gen-2 damage roll: uniform in [217, 256) (PS
    /// `random_range(217, 256)`). Seeded consumption is bit-identical to
    /// that call; Oracle mode labels the draw "droll" so the enumeration
    /// driver can range-merge rolls whose entire subtrees coincide
    /// (sound: damage is monotone in the roll, so identical endpoint
    /// subtrees pin every interior roll — owner-approved 2026-07-21).
    pub fn damage_roll(&mut self) -> u32 {
        match &mut self.oracle {
            None => 217 + self.lcg.random(39),
            Some(o) => {
                let counts = (0..39).map(|v| uniform_count(39, v)).collect();
                217 + Self::pick(o, "droll", counts) as u32
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
