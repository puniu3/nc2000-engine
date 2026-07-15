//! SplitMix64 — the bot-side RNG (agent decisions, arena scheduling).
//! Distinct from the battle PRNG on purpose: bot randomness must never
//! consume battle PRNG state.

#[derive(Clone, Debug)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (n > 0).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A battle seed string in PS `Gen5RNG.getSeed()` format
    /// (four decimal 16-bit limbs).
    pub fn battle_seed(&mut self) -> String {
        let v = self.next();
        format!(
            "{},{},{},{}",
            (v >> 48) & 0xffff,
            (v >> 32) & 0xffff,
            (v >> 16) & 0xffff,
            v & 0xffff
        )
    }
}
