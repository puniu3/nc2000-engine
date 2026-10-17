//! FxHash-style word-folding hasher (rustc-hash algorithm): deterministic,
//! no dependencies, ~5x faster than SipHash on struct-walk hashing, quality
//! plenty for 64-bit search keys and search-internal maps. Not DoS-resistant
//! — never use for attacker-controlled map keys.
//!
//! Lives in its own module so search-tree maps (`crates/bot`) can share the
//! hasher `Battle::state_key` already uses instead of paying SipHash on every
//! node lookup.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct WordHasher<const POLYNOMIAL: bool>(u64);

pub type FxHasher = WordHasher<false>;
/// Fixed-width polynomial state hash; persistent fingerprints use `FxHasher`.
pub type SearchHasher = WordHasher<true>;

#[derive(Default)]
pub(crate) struct FingerprintHasher {
    fast: FxHasher,
    sip: std::collections::hash_map::DefaultHasher,
}

impl FingerprintHasher {
    pub(crate) fn finish128(&self) -> u128 {
        ((self.fast.finish() as u128) << 64) | self.sip.finish() as u128
    }
}

macro_rules! forward_fingerprint_words {
    ($($method:ident: $ty:ty),* $(,)?) => { $(
        #[inline]
        fn $method(&mut self, value: $ty) {
            self.fast.$method(value);
            self.sip.$method(value);
        }
    )* };
}

impl Hasher for FingerprintHasher {
    #[inline]
    fn finish(&self) -> u64 { self.fast.finish() }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.fast.write(bytes);
        self.sip.write(bytes);
    }

    forward_fingerprint_words! {
        write_u8:u8, write_u16:u16, write_u32:u32, write_u64:u64, write_u128:u128, write_usize:usize,
        write_i8:i8, write_i16:i16, write_i32:i32, write_i64:i64, write_i128:i128, write_isize:isize,
    }
}

impl<const POLYNOMIAL: bool> WordHasher<POLYNOMIAL> {
    const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;

    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = if POLYNOMIAL {
            self.0.wrapping_add(word).wrapping_mul(0xf1357aea2e62a9c5)
        } else {
            (self.0.rotate_left(5) ^ word).wrapping_mul(Self::K)
        };
    }
}

impl<const POLYNOMIAL: bool> std::hash::Hasher for WordHasher<POLYNOMIAL> {
    #[inline]
    fn finish(&self) -> u64 {
        if POLYNOMIAL {
            self.0.rotate_left(26)
        } else {
            self.0
        }
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rem.len()].copy_from_slice(rem);
            self.add(u64::from_le_bytes(buf) ^ rem.len() as u64);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.add(i as u64);
        self.add((i >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
pub type FxHashSet<T> = HashSet<T, FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn compare<T: Hash + ?Sized>(value: &T) {
        let mut paired = FingerprintHasher::default();
        let mut fast = FxHasher::default();
        let mut sip = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut paired); value.hash(&mut fast); value.hash(&mut sip);
        assert_eq!(paired.finish128(),((fast.finish() as u128)<<64)|sip.finish() as u128);
    }

    #[test]
    fn paired_fingerprint_preserves_primitive_writes_and_composite_hashes() {
        let mut paired = FingerprintHasher::default();
        let mut fast = FxHasher::default();
        let mut sip = std::collections::hash_map::DefaultHasher::new();
        macro_rules! feed {
            ($method:ident,$value:expr) => {{
                paired.$method($value);fast.$method($value);sip.$method($value);
                assert_eq!(paired.finish128(),((fast.finish() as u128)<<64)|sip.finish() as u128);
            }};
        }
        for n in 0..65 {
            let bytes: Vec<_> = (0..n).map(|i|(i*43+n) as u8).collect();
            feed!(write,&bytes);
            feed!(write_u8,n as u8); feed!(write_i8,-(n as i8));
            feed!(write_u16,u16::MAX-n as u16); feed!(write_i16,i16::MIN+n as i16);
            feed!(write_u32,u32::MAX-n as u32); feed!(write_i32,i32::MIN+n as i32);
            feed!(write_u64,u64::MAX-n as u64); feed!(write_i64,i64::MIN+n as i64);
            feed!(write_u128,u128::MAX-n as u128); feed!(write_i128,i128::MIN+n as i128);
            feed!(write_usize,usize::MAX-n); feed!(write_isize,isize::MIN+n as isize);
            compare(&bytes);
            compare(&(bytes.clone(),Some(n as u128),[-1i64,0,i64::MAX]));
        }
        for text in ["","a","counter\0rest","日本語","abcdefghijklmno"] { compare(text); }
        compare(&vec![None,Some(0u128),Some(u128::MAX)]);
    }
}
