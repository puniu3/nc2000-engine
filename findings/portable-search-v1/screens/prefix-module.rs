//! Exact, bounded memoization of the expensive Pokemon hash prefix.
//!
//! A hit compares every input field, including unused inline move slots.
//! It needs no mutation hooks and remains correct after direct public-field
//! writes, determinization, Transform, and switching between battles/dexes.
use std::cell::RefCell;
use std::hash::{Hash, Hasher};

use crate::dex::{SpeciesId, TypeId};
use crate::fxhash::SearchHasher;
use crate::state::{Gender, MoveSlots, PokeName};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PokemonHashPrefix {
    pub species: SpeciesId,
    pub base_species: SpeciesId,
    pub name: PokeName,
    pub level: u8,
    pub gender: Gender,
    pub happiness: u8,
    pub set_ivs: [i32; 6],
    pub set_evs: [i32; 6],
    pub base_move_slots: MoveSlots,
    pub hp_type: TypeId,
    pub hp_power: i32,
    pub base_hp_type: TypeId,
    pub base_hp_power: i32,
    pub base_stored_stats: [i32; 5],
    pub stored_stats: [i32; 5],
    pub base_maxhp: i32,
    pub maxhp: i32,
}

#[derive(Default)]
struct CountedHasher {
    inner: SearchHasher,
    words: u32,
}

macro_rules! counted_words {
    ($($method:ident: $ty:ty => $count:expr),* $(,)?) => { $(
        #[inline]
        fn $method(&mut self, value: $ty) {
            self.inner.$method(value);
            self.words += $count;
        }
    )* };
}

impl Hasher for CountedHasher {
    fn finish(&self) -> u64 { self.inner.finish() }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.inner.write(bytes);
        self.words += bytes.len().div_ceil(8) as u32;
    }
    // Delegate signed methods too: WordHasher's default signed encoding
    // is not interchangeable with a cast followed by an unsigned write.
    counted_words! {
        write_u8:u8=>1, write_u16:u16=>1, write_u32:u32=>1,
        write_u64:u64=>1, write_u128:u128=>2, write_usize:usize=>1,
        write_i8:i8=>1, write_i16:i16=>1, write_i32:i32=>1,
        write_i64:i64=>1, write_i128:i128=>2, write_isize:isize=>1,
    }
}

#[derive(Clone, Copy)]
struct Entry {
    prefix: PokemonHashPrefix,
    multiplier: u64,
    offset: u64,
}

pub(crate) struct PrefixCache {
    entries: [Option<Entry>; 12],
    cursor: usize,
}

impl PrefixCache {
    const fn new() -> Self { Self { entries: [None; 12], cursor: 0 } }

    pub(crate) fn hash(&mut self, prefix: &PokemonHashPrefix, h: &mut SearchHasher) {
        let slot = self.cursor % self.entries.len();
        self.cursor += 1;
        let entry = &mut self.entries[slot];
        if !entry.as_ref().is_some_and(|old| old.prefix == *prefix) {
            let mut segment = CountedHasher::default();
            prefix.hash(&mut segment);
            *entry = Some(Entry {
                prefix: *prefix,
                multiplier: 0xf1357aea2e62a9c5u64.wrapping_pow(segment.words),
                offset: segment.finish().rotate_right(26),
            });
        }
        let entry = entry.as_ref().unwrap();
        h.append_segment(entry.multiplier, entry.offset);
    }
}

thread_local! {
    static CACHE: RefCell<PrefixCache> = const { RefCell::new(PrefixCache::new()) };
}

pub(crate) fn with_cache<T>(f: impl FnOnce(Option<&mut PrefixCache>) -> T) -> T {
    CACHE.with(|cell| match cell.try_borrow_mut() {
        Ok(mut cache) => {
            cache.cursor = 0;
            f(Some(&mut cache))
        }
        Err(_) => f(None),
    })
}
