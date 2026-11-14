//! Trusted local-player adapter. Search receives a reconstruction, never a
//! clone of the real game's hidden runtime state.
use crate::import::{ProtocolTracker, Request};
use crate::player::PlayerChannel;
use crate::{Belief, Observer, SplitMix64};
use nc2000_engine::{
    battle::PokemonSet,
    dex::{toid, Dex},
    state::{Battle, Pokemon},
};

/// Hidden runtime condition timers that cannot be read from the protocol and
/// must be resampled symmetrically per determinization iteration instead of
/// pinned to a single synthetic roll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HiddenTimerKind {
    /// Non-Rest sleep: initial duration `D` in `{2, 3, 4}`, decremented by `slept`.
    Sleep { slept: u8 },
    /// Confusion: initial duration `D` in `{2, 3, 4, 5}`, decremented by `ticks`.
    Confusion { ticks: u8 },
    /// Encore: initial duration `D` in `{3, 4, 5, 6}`, decremented per upkeep.
    Encore { upkeeps: u8 },
    /// Disable: initial base `B` in `{2, 3, 4, 5}` (+1 if `acted_on_start`), decremented per upkeep.
    Disable { upkeeps: u8, acted_on_start: bool },
    /// Clamp / Bind / Wrap / Fire Spin / Whirlpool: initial `D` in `{3, 4, 5}`, decremented per upkeep.
    PartiallyTrapped { upkeeps: u8 },
    /// Bide: initial `D` in `{3, 4}`, decremented per upkeep.
    Bide { upkeeps: u8 },
    /// Thrash / Outrage / Petal Dance lock: initial `D` in `{2, 3}`, decremented per upkeep.
    LockedMove { upkeeps: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HiddenTimerSpec {
    pub side: u8,
    pub slot: u8,
    pub kind: HiddenTimerKind,
}

/// A root synthesized from a player's observations. There is deliberately no
/// public constructor taking a real Battle.
/// ```compile_fail
/// use nc2000_bot::player_state::ObservedPosition;
/// fn smuggle(b: nc2000_engine::state::Battle) -> ObservedPosition {
///     ObservedPosition { battle: b }
/// }
/// ```
pub struct ObservedPosition {
    battle: Battle,
    hidden_timers: Vec<HiddenTimerSpec>,
}
impl ObservedPosition {
    pub(crate) fn reconstructed(battle: Battle) -> Self {
        Self {
            battle,
            hidden_timers: Vec::new(),
        }
    }
    pub(crate) fn reconstructed_with_timers(
        battle: Battle,
        hidden_timers: Vec<HiddenTimerSpec>,
    ) -> Self {
        Self {
            battle,
            hidden_timers,
        }
    }
    pub(crate) fn hidden_timers(&self) -> &[HiddenTimerSpec] {
        &self.hidden_timers
    }
    /// Read-only diagnostics of the synthesized world, not the real battle.
    pub fn model(&self) -> &Battle {
        &self.battle
    }

    /// Counterfactual world for policy-response research. The source is itself
    /// a player reconstruction, so no real hidden state can enter this path.
    pub fn sample(
        &self,
        dex: &Dex,
        belief: &Belief,
        observer: &Observer,
        rng: &mut SplitMix64,
    ) -> Self {
        Self::reconstructed(belief.determinize(dex, &self.battle, observer, rng))
    }
}

/// Trusted adapter for native callers and the legacy WASM facade. Live Web
/// workers instead use the same protocol directly, without receiving truth.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn hash_log_lines(mut hash: u64, lines: &[String]) -> u64 {
    for line in lines {
        hash ^= line.len() as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        for &b in line.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

pub struct LocalPlayer {
    channel: PlayerChannel,
    tracker: ProtocolTracker,
    own_sets: Vec<PokemonSet>,
    request: Request,
    consumed_hash: u64,
}

impl LocalPlayer {
    pub fn new(battle: &Battle, dex: &Dex, side: usize) -> Result<(Self, Observer), String> {
        if !battle.log_enabled {
            return Err("blind play requires player-visible protocol history".into());
        }
        let own_sets = submitted_team(battle, dex, side);
        Self::new_with_own_sets(battle, dex, side, own_sets)
    }

    pub fn new_with_own_sets(
        battle: &Battle,
        dex: &Dex,
        side: usize,
        own_sets: Vec<PokemonSet>,
    ) -> Result<(Self, Observer), String> {
        if !battle.log_enabled {
            return Err("blind play requires player-visible protocol history".into());
        }
        let mut channel = PlayerChannel::new(side);
        let frame = channel.frame(&mut battle.clone(), dex)?;
        let consumed_hash = hash_log_lines(FNV_OFFSET, &battle.log[..channel.log_cursor()]);
        let mut tracker = ProtocolTracker::new(side);
        for line in &frame.lines {
            tracker.push_line(dex, line);
        }
        let mut observer = Observer::from_mons(side, tracker.observer_mons());
        for (slot, name) in tracker.names(1 - side).iter().enumerate() {
            if !name.is_empty() {
                observer.set_name(slot, name);
            }
        }
        for line in &frame.lines {
            observer.ingest_line(line, dex);
        }
        let request = Request::parse(dex, &frame.request.to_string())?;
        Ok((
            Self {
                channel,
                tracker,
                own_sets,
                request,
                consumed_hash,
            },
            observer,
        ))
    }

    pub fn is_compatible(&self, battle: &Battle) -> bool {
        let cursor = self.channel.log_cursor();
        battle.log_enabled
            && battle.log.len() >= cursor
            && hash_log_lines(FNV_OFFSET, &battle.log[..cursor]) == self.consumed_hash
    }

    pub fn own_sets(&self) -> &[PokemonSet] {
        &self.own_sets
    }

    pub fn observe(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        observer: &mut Observer,
    ) -> Result<(), String> {
        let prev_cursor = self.channel.log_cursor();
        let frame = self.channel.frame(&mut battle.clone(), dex)?;
        self.consumed_hash = hash_log_lines(
            self.consumed_hash,
            &battle.log[prev_cursor..self.channel.log_cursor()],
        );
        for line in &frame.lines {
            self.tracker.push_line(dex, line);
            for (slot, name) in self.tracker.names(observer.opp()).iter().enumerate() {
                if !name.is_empty() {
                    observer.set_name(slot, name);
                }
            }
            observer.ingest_line(line, dex);
        }
        self.request = Request::parse(dex, &frame.request.to_string())?;
        Ok(())
    }

    pub fn position(
        &self,
        dex: &Dex,
        belief: &mut Belief,
        observer: &Observer,
        rng: &mut SplitMix64,
    ) -> Result<ObservedPosition, String> {
        self.condition_belief(belief, observer);
        let pick = belief.alive().first().copied();
        let battle = self.tracker.synthesize(
            dex,
            &self.own_sets,
            belief.refs(pick),
            observer,
            &self.request,
            rng,
        )?;
        let timers = self.tracker.hidden_timer_specs(dex, &battle);
        Ok(ObservedPosition::reconstructed_with_timers(battle, timers))
    }

    /// Install only the observed opponent HP buckets; no own hidden set or
    /// runtime state is exported to an opponent-policy model.
    pub fn condition_belief(&self, belief: &mut Belief, observer: &Observer) {
        self.tracker.set_hp_observations(belief, observer.opp());
    }
}

/// Convert a clean Turn-0 roster (as captured at team preview) into `PokemonSet`s
/// with immutable submitted fields.
pub fn clean_sets_from_roster(roster: &[Pokemon], dex: &Dex) -> Vec<PokemonSet> {
    roster
        .iter()
        .map(|p| PokemonSet {
            name: p.name.as_str().to_string(),
            species: dex.species.key(p.base_species).into(),
            item: p.item.map(|i| dex.items.key(i)).unwrap_or("").into(),
            ability: "No Ability".into(),
            moves: p
                .base_move_slots
                .iter()
                .map(|m| dex.moves.key(m.id).to_string())
                .collect(),
            level: p.level,
            happiness: Some(p.happiness),
            gender: Some(p.gender.as_str().into()),
            ivs: Some(
                ["hp", "atk", "def", "spa", "spd", "spe"]
                    .iter()
                    .zip(p.set_ivs)
                    .map(|(k, v)| ((*k).into(), v as u8))
                    .collect(),
            ),
            evs: Some(
                ["hp", "atk", "def", "spa", "spd", "spe"]
                    .iter()
                    .zip(p.set_evs)
                    .map(|(k, v)| ((*k).into(), v as u16))
                    .collect(),
            ),
        })
        .collect()
}

fn parse_log_subject(s: &str) -> Option<(usize, &str)> {
    let b = s.as_bytes();
    if b.len() < 4 || b[0] != b'p' || (b[1] != b'1' && b[1] != b'2') {
        return None;
    }
    let side = (b[1] - b'1') as usize;
    let rest = &s[2..];
    let rest = rest.strip_prefix(|c: char| c.is_ascii_lowercase()).unwrap_or(rest);
    let name = rest.strip_prefix(": ")?;
    Some((side, name))
}

fn find_roster_slot(
    roster: &[Pokemon],
    dex: &Dex,
    name: &str,
    species_hint: &str,
    active_slot: Option<usize>,
) -> Option<usize> {
    if let Some(a) = active_slot {
        if roster.get(a).is_some_and(|p| p.name.as_str() == name) {
            return Some(a);
        }
    }
    let hint_id = (!species_hint.is_empty()).then(|| dex.species.id(&toid(species_hint))).flatten();
    if let Some(sp) = hint_id {
        if let Some(slot) = roster
            .iter()
            .position(|p| p.name.as_str() == name && p.base_species == sp)
        {
            return Some(slot);
        }
    }
    roster.iter().position(|p| p.name.as_str() == name)
}

fn resolve_item_key(dex: &Dex, item_name: &str) -> String {
    if item_name.is_empty() {
        return String::new();
    }
    dex.items
        .id(&toid(item_name))
        .map(|id| dex.items.key(id).to_string())
        .unwrap_or_default()
}

/// Reconstruct Turn-0 initial held items for `target_side` by scanning public
/// item-mutation events (`-enditem` consumption such as Berserk Gene / Berries
/// and `-item` from Thief) when the log starts from battle creation.
fn initial_items_from_log(battle: &Battle, dex: &Dex, target_side: usize) -> Vec<Option<String>> {
    let mut orig: Vec<Option<String>> = vec![None; battle.sides[target_side].roster.len()];
    if !battle.log_enabled {
        return orig;
    }
    let mut active: [Option<usize>; 2] = [None, None];
    let mut split: Option<(usize, bool)> = None;
    for line in &battle.log {
        if let Some((owner, secret)) = split {
            split = secret.then_some((owner, false));
            if secret {
                continue;
            }
        } else if let Some(owner) = line.strip_prefix("|split|") {
            let owner = match owner {
                "p1" => 0,
                "p2" => 1,
                _ => continue,
            };
            split = Some((owner, true));
            continue;
        }
        let Some(body) = line.strip_prefix('|') else { continue };
        let parts: Vec<&str> = body.split('|').collect();
        let cmd = *parts.first().unwrap_or(&"");
        let arg = |i: usize| parts.get(i).copied().unwrap_or("");
        match cmd {
            "switch" | "drag" | "replace" => {
                let Some((s, name)) = parse_log_subject(arg(1)) else { continue };
                let species_hint = arg(2).split(',').next().unwrap_or("").trim();
                active[s] = find_roster_slot(
                    &battle.sides[s].roster,
                    dex,
                    name,
                    species_hint,
                    None,
                );
            }
            "-enditem" => {
                let Some((s, name)) = parse_log_subject(arg(1)) else { continue };
                if s == target_side {
                    if let Some(slot) =
                        find_roster_slot(&battle.sides[s].roster, dex, name, "", active[s])
                    {
                        orig[slot].get_or_insert_with(|| resolve_item_key(dex, arg(2)));
                    }
                }
            }
            "-item" => {
                let item_name = arg(2);
                if let Some((s, name)) = parse_log_subject(arg(1)) {
                    if s == target_side {
                        if let Some(slot) =
                            find_roster_slot(&battle.sides[s].roster, dex, name, "", active[s])
                        {
                            // In Gen 2 only Thief bestows an item on a mon, which
                            // requires the thief started this mutation sequence itemless.
                            orig[slot].get_or_insert_with(String::new);
                        }
                    }
                }
                for tag in parts.iter().skip(3) {
                    if let Some(victim) = tag.strip_prefix("[of] ") {
                        if let Some((vs, vname)) = parse_log_subject(victim) {
                            if vs == target_side {
                                if let Some(vslot) = find_roster_slot(
                                    &battle.sides[vs].roster,
                                    dex,
                                    vname,
                                    "",
                                    active[vs],
                                ) {
                                    orig[vslot]
                                        .get_or_insert_with(|| resolve_item_key(dex, item_name));
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    orig
}

/// Export `target_side`'s Turn-0 submitted sets from `battle`, restoring initial
/// held items from `battle.log` if Berserk Gene / Berries / Thief mutated them
/// prior to Turn-N construction.
pub fn submitted_team(battle: &Battle, dex: &Dex, side: usize) -> Vec<PokemonSet> {
    let mut sets = clean_sets_from_roster(&battle.sides[side].roster, dex);
    for (set, orig_item) in sets
        .iter_mut()
        .zip(initial_items_from_log(battle, dex, side))
    {
        if let Some(item) = orig_item {
            set.item = item;
        }
    }
    sets
}

/// Export only the observing player's canonical submitted set. In particular,
/// do not copy current stats/Transform overlays or hidden condition counters.
pub fn own_team(battle: &Battle, dex: &Dex, side: usize) -> Vec<PokemonSet> {
    submitted_team(battle, dex, side)
}
