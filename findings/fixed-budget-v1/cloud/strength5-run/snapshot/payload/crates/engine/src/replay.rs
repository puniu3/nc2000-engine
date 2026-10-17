use std::collections::BTreeMap;

use miniz_oxide::{deflate::compress_to_vec, inflate::decompress_to_vec_with_limit};
use sha2::{Digest, Sha256};

use crate::{
    battle::{EngineError, PokemonSet, SearchChoice},
    choice::Choice,
    dex::{toid, Dex},
    prng::Prng,
    state::Battle,
};

include!(concat!(env!("OUT_DIR"), "/replay_version.rs"));

pub const PREFIX: &str = "NC2-";
const HEADER: usize = 48;
const MAX_BYTES: usize = 4096;
const MAX_ROUNDS: usize = 3010;
const STATS: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];
const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

pub fn fingerprint(json: &str) -> [u8; 16] {
    let mut h = Sha256::new();
    h.update(ENGINE_FINGERPRINT);
    h.update(json.as_bytes());
    h.finalize()[..16].try_into().unwrap()
}

#[derive(Clone)]
pub struct Replay {
    pub teams: [Vec<PokemonSet>; 2],
    pub seed: String,
    pub caps: [Option<u32>; 2],
    pub open: bool,
    pub bot_side: usize,
    pub rounds: Vec<[Option<SearchChoice>; 2]>,
}

pub struct Recorder {
    pub replay: Replay,
    pending: [Option<SearchChoice>; 2],
}

impl Recorder {
    pub fn new(dex: &Dex, battle: &Battle, teams: [Vec<PokemonSet>; 2], seed: &str) -> Self {
        let teams = std::array::from_fn(|side| {
            teams[side]
                .iter()
                .enumerate()
                .map(|(i, set)| {
                    let p = &battle.sides[side].roster[i];
                    PokemonSet {
                        name: p.name.as_str().into(),
                        species: dex.species.key(p.base_species).into(),
                        item: p.item.map(|id| dex.items.key(id)).unwrap_or("").into(),
                        ability: String::new(),
                        moves: p
                            .base_move_slots
                            .iter()
                            .map(|m| dex.moves.key(m.id).into())
                            .collect(),
                        level: p.level as u8,
                        happiness: Some(p.happiness),
                        gender: set
                            .gender
                            .clone()
                            .filter(|g| matches!(g.as_str(), "M" | "F" | "N")),
                        ivs: Some(
                            STATS
                                .iter()
                                .enumerate()
                                .map(|(i, k)| (k.to_string(), p.set_ivs[i] as u8))
                                .collect(),
                        ),
                        evs: Some(
                            STATS
                                .iter()
                                .enumerate()
                                .map(|(i, k)| (k.to_string(), p.set_evs[i] as u16))
                                .collect(),
                        ),
                    }
                })
                .collect()
        });
        Self {
            replay: Replay {
                teams,
                seed: seed.into(),
                caps: battle.preview_level_caps,
                open: true,
                bot_side: 1,
                rounds: Vec::new(),
            },
            pending: [None; 2],
        }
    }

    pub fn choose(
        &mut self,
        dex: &Dex,
        battle: &mut Battle,
        side: usize,
        input: &str,
    ) -> Result<(), EngineError> {
        if side > 1 {
            return Err(EngineError::InvalidChoice("invalid side".into()));
        }
        let needs = battle.needs_choice();
        let legal = battle.legal_choices(dex, side);
        let chosen = resolve(dex, battle, side, &legal, input)
            .ok_or_else(|| EngineError::InvalidChoice(format!("cannot record {input}")))?;
        if let Err(error) = battle.choose(dex, side, input) {
            self.pending[side] = None;
            return Err(error);
        }
        self.pending[side] = Some(chosen);
        if (0..2).all(|s| !needs[s] || self.pending[s].is_some()) {
            self.replay.rounds.push(self.pending);
            self.pending = [None; 2];
        }
        Ok(())
    }
}

fn resolve(
    dex: &Dex,
    battle: &Battle,
    side: usize,
    legal: &[SearchChoice],
    input: &str,
) -> Option<SearchChoice> {
    let parsed = Choice::parse(input).ok()?;
    if parsed.len() != 1 {
        return None;
    }
    let mut parsed = parsed[0].clone();
    if let Choice::MoveBySlot(slot) = parsed {
        let p = battle.poke(battle.active_id(side)?);
        parsed = Choice::MoveById(
            dex.moves
                .key(p.move_slots.iter().nth(slot.checked_sub(1)? as usize)?.id)
                .into(),
        );
    }
    if let Choice::Team(slots) = &mut parsed {
        let size = battle.sides[side].party.len().min(3);
        slots.truncate(size);
        for i in 1..=size as u8 {
            if slots.len() < size && !slots.contains(&i) {
                slots.push(i);
            }
        }
    }
    legal.iter().copied().find(|c| match (&parsed, c) {
        (Choice::Team(slots), SearchChoice::Team(s)) => slots
            .iter()
            .copied()
            .eq(s.iter().copied().filter(|x| *x != 0)),
        (Choice::Switch(a), SearchChoice::Switch(b)) => a == b,
        (Choice::Pass, SearchChoice::Pass) => true,
        (Choice::MoveById(m), SearchChoice::Move(id)) => {
            let name = toid(m);
            dex.moves.key(*id) == name
                || name == "hiddenpower" && dex.moves.key(*id).starts_with("hiddenpower")
                || dex.moves.key(*id) == "struggle"
                || legal.len() == 1
        }
        _ => false,
    })
}

impl Replay {
    pub fn initial(&self, dex: &Dex) -> Result<Battle, String> {
        let mut b = Battle::from_fixture(dex, &self.seed, &self.teams[0], &self.teams[1])
            .map_err(|e| format!("{e:?}"))?;
        b.preview_level_caps = self.caps;
        Ok(b)
    }

    pub fn apply(&self, dex: &Dex, battle: &mut Battle, round: usize) -> Result<(), String> {
        let picks = self.rounds.get(round).ok_or("replay round out of range")?;
        battle
            .apply_choices(dex, *picks)
            .map_err(|e| format!("{e:?}"))
    }

    pub fn at(&self, dex: &Dex, round: usize) -> Result<Battle, String> {
        if round > self.rounds.len() {
            return Err("replay round out of range".into());
        }
        let mut b = self.initial(dex)?;
        for i in 0..round {
            self.apply(dex, &mut b, i)?;
        }
        Ok(b)
    }

    pub fn encode(&self, dex: &Dex) -> Result<String, String> {
        if self.bot_side > 1 || self.rounds.len() > MAX_ROUNDS {
            return Err("replay limits exceeded".into());
        }
        let mut teams = Bits::default();
        for team in &self.teams {
            if team.is_empty() || team.len() > 6 {
                return Err("team size out of range".into());
            }
            for p in team {
                let species = dex.species.id(&toid(&p.species)).ok_or("unknown species")?;
                teams.put(u16::from(species) as u32, 8)?;
                let item = if p.item.is_empty() {
                    0
                } else {
                    u16::from(dex.items.id(&toid(&p.item)).ok_or("unknown item")?) as u32 + 1
                };
                teams.put(item, 6)?;
                teams.put(p.level as u32, 8)?;
                if p.moves.is_empty() || p.moves.len() > 4 {
                    return Err("move count out of range".into());
                }
                teams.put(p.moves.len() as u32, 3)?;
                for i in 0..4 {
                    let id = match p.moves.get(i) {
                        Some(m) => u16::from(dex.moves.id(&toid(m)).ok_or("unknown move")?) as u32,
                        None => 0,
                    };
                    teams.put(id, 9)?;
                }
                for k in STATS {
                    teams.put(
                        (p.ivs
                            .as_ref()
                            .and_then(|m| m.get(k))
                            .copied()
                            .unwrap_or(31)
                            .min(31)
                            >> 1) as u32,
                        4,
                    )?;
                }
                for k in STATS {
                    teams.put(
                        p.evs
                            .as_ref()
                            .and_then(|m| m.get(k))
                            .copied()
                            .unwrap_or(0)
                            .min(255) as u32,
                        8,
                    )?;
                }
                teams.put(p.happiness.unwrap_or(255) as u32, 8)?;
                teams.put(
                    match p.gender.as_deref() {
                        Some("M") => 1,
                        Some("F") => 2,
                        Some("N") => 3,
                        _ => 0,
                    },
                    2,
                )?;
                let name = if p.name == dex.species.get(species).name {
                    ""
                } else {
                    &p.name
                };
                if name.len() > 24 {
                    return Err("nickname too long".into());
                }
                teams.put(name.len() as u32, 6)?;
                for byte in name.bytes() {
                    teams.put(byte as u32, 8)?;
                }
            }
        }
        let mut actions = Bits::default();
        let mut battle = self.initial(dex)?;
        for (round, picks) in self.rounds.iter().enumerate() {
            let needs = battle.needs_choice();
            if !needs.iter().any(|n| *n) {
                return Err("input after battle end".into());
            }
            for side in 0..2 {
                if needs[side] != picks[side].is_some() {
                    return Err("incomplete request round".into());
                }
                if let Some(pick) = picks[side] {
                    let legal = battle.legal_choices(dex, side);
                    let i = legal
                        .iter()
                        .position(|p| *p == pick)
                        .ok_or("illegal replay choice")?;
                    actions.put(i as u32, width(legal.len()))?;
                }
            }
            self.apply(dex, &mut battle, round)?;
        }
        let mut raw = vec![0; HEADER];
        raw[0] = 1;
        raw[1..17].copy_from_slice(&dex.replay_fingerprint);
        let seed = Prng::from_seed_str(&self.seed)
            .ok_or("bad seed")?
            .seed_str();
        for (i, limb) in seed.split(',').enumerate() {
            raw[17 + i * 2..19 + i * 2]
                .copy_from_slice(&limb.parse::<u16>().unwrap().to_be_bytes());
        }
        for s in 0..2 {
            raw[25 + 5 * s] = self.caps[s].is_some() as u8;
            raw[26 + 5 * s..30 + 5 * s].copy_from_slice(&self.caps[s].unwrap_or(0).to_be_bytes());
            raw[35 + s] = self.teams[s].len() as u8;
        }
        for (offset, n) in [(37, self.rounds.len()), (39, teams.len), (41, actions.len)] {
            let n = u16::try_from(n).map_err(|_| "replay too large")?;
            raw[offset..offset + 2].copy_from_slice(&n.to_be_bytes());
        }
        raw[43] = self.open as u8 | (self.bot_side as u8) << 1;
        raw.extend(teams.bytes);
        raw.extend(actions.bytes);
        let checksum = checksum(&raw);
        raw[44..48].copy_from_slice(&checksum);
        if raw.len() > MAX_BYTES {
            return Err("replay too large".into());
        }
        let compressed = compress_to_vec(&raw, 9);
        let mut data = vec![(compressed.len() < raw.len()) as u8];
        data.extend(if compressed.len() < raw.len() {
            compressed
        } else {
            raw
        });
        Ok(format!("{PREFIX}{}", base64_encode(&data)))
    }

    pub fn decode(dex: &Dex, code: &str) -> Result<Self, String> {
        let data = base64_decode(
            code.trim()
                .strip_prefix(PREFIX)
                .ok_or("not a replay code")?,
        )?;
        let (&mode, payload) = data.split_first().ok_or("empty replay")?;
        let raw = match mode {
            0 => payload.to_vec(),
            1 => decompress_to_vec_with_limit(payload, MAX_BYTES).map_err(|_| "damaged replay")?,
            _ => return Err("unsupported replay encoding".into()),
        };
        if raw.len() < HEADER || raw.len() > MAX_BYTES || raw[0] != 1 {
            return Err("unsupported replay encoding".into());
        }
        if raw[44..48] != checksum(&raw) {
            return Err("damaged replay".into());
        }
        if raw[1..17] != dex.replay_fingerprint {
            return Err("incompatible replay version".into());
        }
        if raw[43] > 3 {
            return Err("invalid replay settings".into());
        }
        let read16 = |i| u16::from_be_bytes(raw[i..i + 2].try_into().unwrap()) as usize;
        let count = read16(37);
        if count > MAX_ROUNDS {
            return Err("too many replay rounds".into());
        }
        let team_bits = read16(39);
        let action_bits = read16(41);
        let split = HEADER + team_bits.div_ceil(8);
        if split + action_bits.div_ceil(8) != raw.len() {
            return Err("damaged replay length".into());
        }
        let mut input = Reader::new(&raw[HEADER..split], team_bits);
        let mut teams: [Vec<PokemonSet>; 2] = [Vec::new(), Vec::new()];
        for side in 0..2 {
            if !(1..=6).contains(&raw[35 + side]) {
                return Err("invalid team count".into());
            }
            for _ in 0..raw[35 + side] {
                let species = dex
                    .species
                    .keys
                    .get(input.get(8)? as usize)
                    .ok_or("unknown species")?
                    .clone();
                let item = input.get(6)?;
                let item = if item == 0 {
                    String::new()
                } else {
                    dex.items
                        .keys
                        .get(item as usize - 1)
                        .ok_or("unknown item")?
                        .clone()
                };
                let level = input.get(8)? as u8;
                if level == 0 {
                    return Err("invalid level".into());
                }
                let n = input.get(3)? as usize;
                if !(1..=4).contains(&n) {
                    return Err("invalid move count".into());
                }
                let mut moves = Vec::new();
                for i in 0..4 {
                    let m = input.get(9)? as usize;
                    if i < n {
                        moves.push(dex.moves.keys.get(m).ok_or("unknown move")?.clone());
                    }
                }
                let mut ivs = BTreeMap::new();
                for k in STATS {
                    ivs.insert(k.into(), input.get(4)? as u8 * 2);
                }
                let mut evs = BTreeMap::new();
                for k in STATS {
                    evs.insert(k.into(), input.get(8)? as u16);
                }
                let happiness = Some(input.get(8)? as u8);
                let gender = match input.get(2)? {
                    0 => None,
                    1 => Some("M".into()),
                    2 => Some("F".into()),
                    _ => Some("N".into()),
                };
                let n = input.get(6)? as usize;
                if n > 24 {
                    return Err("nickname too long".into());
                }
                let name = String::from_utf8(
                    (0..n)
                        .map(|_| input.get(8).map(|b| b as u8))
                        .collect::<Result<_, _>>()?,
                )
                .map_err(|_| "invalid nickname")?;
                teams[side].push(PokemonSet {
                    name,
                    species,
                    item,
                    ability: String::new(),
                    moves,
                    level,
                    evs: Some(evs),
                    ivs: Some(ivs),
                    happiness,
                    gender,
                });
            }
        }
        input.finish()?;
        let seed = (0..4)
            .map(|i| read16(17 + i * 2).to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut caps = [None; 2];
        for s in 0..2 {
            if raw[25 + 5 * s] > 1 {
                return Err("invalid cap".into());
            }
            if raw[25 + 5 * s] == 1 {
                caps[s] = Some(u32::from_be_bytes(
                    raw[26 + 5 * s..30 + 5 * s].try_into().unwrap(),
                ));
            }
        }
        let mut replay = Self {
            teams,
            seed,
            caps,
            open: raw[43] & 1 != 0,
            bot_side: (raw[43] >> 1) as usize,
            rounds: Vec::new(),
        };
        let mut battle = replay.initial(dex)?;
        let mut input = Reader::new(&raw[split..], action_bits);
        for _ in 0..count {
            let needs = battle.needs_choice();
            if !needs.iter().any(|n| *n) {
                return Err("input after battle end".into());
            }
            let mut picks = [None; 2];
            for side in 0..2 {
                if needs[side] {
                    let legal = battle.legal_choices(dex, side);
                    let index = input.get(width(legal.len()))? as usize;
                    picks[side] = Some(*legal.get(index).ok_or("illegal replay choice")?);
                }
            }
            battle
                .apply_choices(dex, picks)
                .map_err(|e| format!("{e:?}"))?;
            replay.rounds.push(picks);
        }
        input.finish()?;
        Ok(replay)
    }
}

fn checksum(raw: &[u8]) -> [u8; 4] {
    let mut hash = Sha256::new();
    hash.update(&raw[..44]);
    hash.update(&raw[48..]);
    hash.finalize()[..4].try_into().unwrap()
}

fn width(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    len: usize,
}
impl Bits {
    fn put(&mut self, value: u32, n: usize) -> Result<(), String> {
        if n < 32 && value as u64 >= 1u64 << n {
            return Err("value exceeds replay field".into());
        }
        for i in (0..n).rev() {
            if self.len % 8 == 0 {
                self.bytes.push(0);
            }
            self.bytes[self.len / 8] |= (((value >> i) & 1) as u8) << (7 - self.len % 8);
            self.len += 1;
        }
        Ok(())
    }
}
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], len: usize) -> Self {
        Self { bytes, pos: 0, len }
    }
    fn get(&mut self, n: usize) -> Result<u32, String> {
        if self.pos + n > self.len {
            return Err("truncated replay".into());
        }
        let mut v = 0;
        for _ in 0..n {
            v = v * 2 + ((self.bytes[self.pos / 8] >> (7 - self.pos % 8)) & 1) as u32;
            self.pos += 1;
        }
        Ok(v)
    }
    fn finish(&self) -> Result<(), String> {
        if self.pos != self.len
            || self.len % 8 != 0
                && self.bytes.last().copied().unwrap_or(0) & ((1 << (8 - self.len % 8)) - 1) != 0
        {
            return Err("trailing replay bits".into());
        }
        Ok(())
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::new();
    for c in bytes.chunks(3) {
        let v = (c[0] as u32) << 16
            | (c.get(1).copied().unwrap_or(0) as u32) << 8
            | c.get(2).copied().unwrap_or(0) as u32;
        for i in 0..c.len() + 1 {
            out.push(ALPHABET[((v >> (18 - i * 6)) & 63) as usize] as char);
        }
    }
    out
}
fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    if text.is_empty() || text.len() > (MAX_BYTES + 1) * 4 / 3 + 4 || text.len() % 4 == 1 {
        return Err("invalid replay length".into());
    }
    let mut out = Vec::new();
    let mut value = 0u32;
    let mut bits = 0;
    for c in text.bytes() {
        let n = ALPHABET
            .iter()
            .position(|b| *b == c)
            .ok_or("invalid replay alphabet")? as u32;
        value = (value << 6 | n) & 0xffffff;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((value >> bits) as u8);
        }
    }
    if bits != 0 && value & ((1 << bits) - 1) != 0 {
        return Err("invalid replay padding".into());
    }
    Ok(out)
}
