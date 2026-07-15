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
