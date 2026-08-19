//! What the player's 正着 claims (V1-V4, `docs/PLAYER-VERDICTS-4069-4070.md`)
//! are actually worth, measured instead of argued.
//!
//! Same shape as `perish_counterfactual_4040.rs`: reconstruct ONE recorded
//! decision through the corpus pipeline, then roll it forward N times under
//! fixed policies and count p2 (= the bot's) wins. Three cases:
//!
//!   `--case 4070-endgame`  Suicune alone vs Umbreon + Marowak (V3/V4).
//!                          Arms: as-played / surf-max / no-dead / ice-max /
//!                          search. `no-dead` isolates the *mask* fix (only
//!                          the provably dead picks are replaced) from the
//!                          whole policy change; `ice-max` is the mirror
//!                          control — if surf-max does not beat it, V3's
//!                          reasoning is not what moves the number.
//!   `--case 4069-lead`     turn 1, Miltank vs the Ghost Misdreavus (V1).
//!                          Arms: as-played / switch-zapdos /
//!                          switch-zapdos-noww / switch-snorlax /
//!                          curse-through / return / doubleteam / search.
//!                          `switch-zapdos-noww` is `switch-zapdos` on a
//!                          board where Zapdos has no Whirlwind at all, so
//!                          the pair separates "the phaze escape wins it"
//!                          from "any switch off the doomed mon wins it".
//!                          Takes `--turn N` -- turn 26 (Snorlax, still able
//!                          to switch) is the same experiment on the
//!                          position round 1 left open.
//!   `--case 4069-trapped`  turn 2, Miltank already Mean Looked (V2).
//!
//! Standard counterfactual semantics: the arm script overrides the first
//! `--arm-depth` decisions and the shipped search plays on from there
//! (`--tail`). Single-move arms default to depth 1, scripted arms to their
//! full length.
//!
//! Opponent models (`--foe`): `script` is the state-driven model of the human
//! and is the one to use for counterfactuals; `replay` re-submits the human's
//! RECORDED action per turn (falling back to `script` when the recording
//! cannot supply the turn) and is a validation control — paired with
//! `--arm as-played` it reproduces the actual game, and it degrades to
//! nonsense once an arm makes the line diverge. `max-damage` and `search` are
//! the two strength brackets.
//!
//! Two axes of the SCRIPTED 4069 opponent, both defaulting to what the
//! recorded human actually did, so every earlier number reproduces:
//!   * `--foe-perish-stay` -- the Misdreavus does NOT leave at perish1; it
//!     stays and dies on its own song with the mon it trapped. This is the
//!     opponent V2's addendum names, and V5 says the answer is
//!     board-dependent, so it is the axis on which that would show.
//!   * `--foe-replacement <species|healthiest|first>` -- which bench mon it
//!     sends when it does bail out (and when it is replacing a faint).
//!     `healthiest` (default) always sends Skarmory in 4069; `blissey` is
//!     the branch V2's +4 Return could plausibly KO.
//!
//! Two deliberate research-only leaks, both switchable and both printed:
//!   * `--oracle-moves` (DEFAULT ON) runs `complete_active_moves_from_future`
//!     so the ACTIVE pair carries the moves the full log proves it had. Without
//!     it the reconstruction hands Umbreon `meanlook/batonpass` instead of
//!     `rest` and the whole Rest-loop question stops existing. Only the active
//!     pair is repaired; the opponent's BENCH stays fabricated.
//!   * `--case 4069-*` defaults to `tmp/corpus-4069-truepicks/battle-4069t.raw.log`,
//!     the party-order-corrected copy — the plain log imputes Starmie into the
//!     bench and `switch 2` is then not Zapdos at all (see
//!     `tmp/verdicts-4069-4070/4069-forensics.md`).
//!
//! Usage:
//!   cargo run --release -p nc2000-bot --example verdict_counterfactual -- \
//!     --case 4070-endgame [--turn 50] [--trials 20000] [--arm all] \
//!     [--foe script|replay|max-damage|search] [--rest-at 0.5] [--tail search|...] \
//!     [--iters 30000] [--threads 14] [--no-wake-rule] [--csv FILE] \
//!     [--foe-perish-stay] [--foe-replacement healthiest|first|<species>]
//!
//! Two round-3 (adversarial-verification) additions, both inert by default:
//!   * `--trace-lo N --trace-hi M` -- print, for trial indices in [N, M), every
//!     turn's joint action and the engine log slice it produced. This is the
//!     only way to answer "what is the losing arm actually DOING" without
//!     inferring it from aggregate diagnostics; run it at `--threads 1` or the
//!     lines interleave.
//!   * arms `earthquake-through` (Earthquake on every decision) and
//!     `eq-then-search` (Earthquake for four decisions, then the tail). These
//!     exist to separate "the search plays a provably dead move" from "the
//!     decision under test is wrong": they force the kill the type-immunity
//!     blind spot refuses. NOTE `--arm-depth` OVERRIDES an arm's own depth, so
//!     `--arm-depth 1` collapses both of them back onto `as-played`.
//!
//! Trials are sharded over `--threads`; every seed is a pure function of the
//! trial index, so the thread count cannot move a number (checked: identical
//! output at --threads 1 and 14).

use std::sync::Arc;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::agent::{Agent, MaxDamageAgent};
use nc2000_bot::blind::BlindAgent;
use nc2000_bot::corpus::{
    cfg, complete_active_moves_from_future, load_battle, load_sources, reconstruct, HumanAction,
};
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::smmcts::RmConfig;
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::prng::{BattleRng, Prng};
use nc2000_engine::state::{Battle, MoveSlots, PokeId, Status, DK};

// ------------------------------------------------------------------ args

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

// -------------------------------------------------------------- choosing

fn pick_move(dex: &Dex, choices: &[SearchChoice], key: &str) -> Option<SearchChoice> {
    choices.iter().copied().find(|c| match c {
        SearchChoice::Move(id) => dex.moves.key(*id) == key,
        _ => false,
    })
}

/// Resolve a switch by SPECIES, never by display position: the party order a
/// reconstruction produces is not the order the live client saw.
fn pick_switch(
    dex: &Dex,
    b: &Battle,
    side: usize,
    choices: &[SearchChoice],
    species: &str,
) -> Option<SearchChoice> {
    choices.iter().copied().find(|c| match c {
        SearchChoice::Switch(pos) => switch_species(dex, b, side, *pos) == Some(species.to_string()),
        _ => false,
    })
}

fn switch_species(dex: &Dex, b: &Battle, side: usize, pos: u8) -> Option<String> {
    let s = &b.sides[side];
    let slot = *s.party.get(pos as usize - 1)?;
    Some(dex.species.key(s.roster[slot as usize].species).to_string())
}

/// A forced replacement request: nothing but switches (or a pass) on offer.
fn is_forced_switch(choices: &[SearchChoice]) -> bool {
    choices
        .iter()
        .all(|c| matches!(c, SearchChoice::Switch(_) | SearchChoice::Pass))
}

fn healthiest_switch(b: &Battle, side: usize, choices: &[SearchChoice]) -> SearchChoice {
    let mut best = choices[0];
    let mut best_frac = -1.0f64;
    for c in choices {
        if let SearchChoice::Switch(pos) = c {
            let s = &b.sides[side];
            let Some(&slot) = s.party.get(*pos as usize - 1) else { continue };
            let p = &s.roster[slot as usize];
            if p.fainted {
                continue;
            }
            let frac = p.hp as f64 / p.maxhp.max(1) as f64;
            if frac > best_frac {
                best_frac = frac;
                best = *c;
            }
        }
    }
    best
}

/// Which bench mon the SCRIPTED opponent sends in -- both when it bails out
/// voluntarily (Misdreavus leaving at perish1) and when it is replacing a
/// faint. `Healthiest` is what the harness has always done, and is the
/// default, so every number measured before this flag existed reproduces.
///
/// It is a free parameter of the harness, not of the position: `healthiest`
/// always sends Skarmory in 4069, so the Blissey branch of V2 -- the one
/// replacement a +4 Return could plausibly KO -- was never on the board.
#[derive(Clone, Debug, PartialEq, Eq)]
enum FoeReplacement {
    Healthiest,
    First,
    Species(String),
}

fn parse_replacement(s: &str) -> FoeReplacement {
    match s {
        "healthiest" => FoeReplacement::Healthiest,
        "first" => FoeReplacement::First,
        sp => FoeReplacement::Species(sp.to_lowercase()),
    }
}

/// `Species` falls back to `Healthiest` when that mon is not (or is no
/// longer) a legal switch -- it has fainted, or it is already in. The
/// fallback is silent by design: the run is a distribution over lines in
/// which the named mon is sometimes gone, and panicking there would only
/// select for the lines where it survived.
fn foe_switch(
    b: &Battle,
    dex: &Dex,
    side: usize,
    choices: &[SearchChoice],
    rep: &FoeReplacement,
) -> SearchChoice {
    match rep {
        FoeReplacement::Healthiest => healthiest_switch(b, side, choices),
        FoeReplacement::First => choices
            .iter()
            .copied()
            .find(|c| matches!(c, SearchChoice::Switch(_)))
            .unwrap_or(choices[0]),
        FoeReplacement::Species(sp) => pick_switch(dex, b, side, choices, sp)
            .unwrap_or_else(|| healthiest_switch(b, side, choices)),
    }
}

/// Take a move away from one species for the WHOLE rollout, in the state
/// rather than in the agent. `switch-zapdos-noww` has to be a board on which
/// the phaze does not exist -- if it were only filtered out of the agent's
/// choices, our own search would still plan around a Whirlwind it is never
/// allowed to play, and the arm would measure an irrational policy instead of
/// a Whirlwind-less board.
///
/// Both lists are stripped: `clear_volatile` restores `move_slots` from
/// `base_move_slots` on every switch-in (`pokemon.rs:792`), so stripping only
/// the live list would hand Whirlwind straight back when Zapdos came in.
fn strip_move(dex: &Dex, b: &mut Battle, side: usize, species: &str, key: &str) -> usize {
    let Some(mid) = dex.moves.id(key) else { return 0 };
    let party: Vec<u8> = b.sides[side].party.iter().copied().collect();
    let mut dropped = 0usize;
    for slot in party {
        let p = &mut b.sides[side].roster[slot as usize];
        if dex.species.key(p.species) != species {
            continue;
        }
        for list in [&mut p.move_slots, &mut p.base_move_slots] {
            let mut kept = MoveSlots::default();
            for m in list.iter() {
                if m.id == mid {
                    dropped += 1;
                } else {
                    kept.push(*m);
                }
            }
            *list = kept;
        }
    }
    dropped
}

/// Sleep counter of `side`'s active. `("slp","onBeforeMove")`
/// (`conditions.rs:159`) decrements FIRST and cures at `<= 0`, so the mon is
/// still asleep when its move resolves iff this is >= 2.
fn stays_asleep(b: &Battle, side: usize) -> bool {
    b.active_id(side).is_some_and(|id| {
        let p = b.poke(id);
        p.status == Status::Slp && p.status_state.get_int(DK::Time) >= 2
    })
}

/// The provable no-ops this study is about. Deliberately narrow, and every
/// arm is decidable from OWN-side information only:
///   * Sleep Talk while awake — fails at `moveexec.rs:521`, PP already spent.
///   * Sleep Talk on the wake-up turn (`Time <= 1`) — the sleeper is cured
///     inside its own onBeforeMove and Sleep Talk then sees an awake user,
///     so it fails exactly the same way. `wake_rule = false` drops this arm,
///     which is how the two categories get separated.
///   * Rest at full HP.
fn is_dead(b: &Battle, dex: &Dex, side: usize, choice: SearchChoice, wake_rule: bool) -> bool {
    let SearchChoice::Move(id) = choice else { return false };
    let Some(active) = b.active_id(side) else { return false };
    let p = b.poke(active);
    match dex.moves.key(id) {
        "sleeptalk" => {
            p.status != Status::Slp || (wake_rule && p.status_state.get_int(DK::Time) <= 1)
        }
        "rest" => p.hp >= p.maxhp,
        _ => false,
    }
}

// ------------------------------------------------------------------ arms

#[derive(Clone, Debug)]
enum Act {
    Move(&'static str),
    SwitchTo(&'static str),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tail {
    Search,
    MaxDamage,
    FirstLegal,
    SurfMax,
    IceMax,
    ZapEscape,
}

fn parse_tail(s: &str) -> Tail {
    match s {
        "search" => Tail::Search,
        "max-damage" => Tail::MaxDamage,
        "first-legal" => Tail::FirstLegal,
        "surf-max" => Tail::SurfMax,
        "ice-max" => Tail::IceMax,
        "zap-escape" => Tail::ZapEscape,
        other => panic!("unknown --tail {other}"),
    }
}

/// Water-first / Ice-first greedy for the 4070 endgame. Sleep Talk only
/// while ASLEEP (which is the only state it does anything in), Rest only
/// below max HP.
fn heur_suicune(
    b: &Battle,
    dex: &Dex,
    side: usize,
    choices: &[SearchChoice],
    prefer_ice: bool,
) -> SearchChoice {
    let Some(active) = b.active_id(side) else { return choices[0] };
    let p = b.poke(active);
    // Sleep Talk only while the mon is still asleep WHEN ITS MOVE RESOLVES.
    if stays_asleep(b, side) {
        if let Some(c) = pick_move(dex, choices, "sleeptalk") {
            return c;
        }
    }
    let order: [&str; 2] = if prefer_ice { ["icebeam", "surf"] } else { ["surf", "icebeam"] };
    for key in order {
        if let Some(c) = pick_move(dex, choices, key) {
            return c;
        }
    }
    if p.hp < p.maxhp {
        if let Some(c) = pick_move(dex, choices, "rest") {
            return c;
        }
    }
    choices[0]
}

/// V1's claimed escape route, made available as a TAIL so that it can be
/// measured instead of hoped for. Under `--tail max-damage` a trapped Zapdos
/// can never pick Whirlwind (0 damage), so the escape cannot occur at all and
/// V1's mechanism is untestable; this tail plays it deliberately:
///   * trapped Zapdos -> Whirlwind. Blowing the trapper off the field clears
///     its volatiles, and Mean Look's linked `trapped` goes with it
///     (`pokemon.rs:551 remove_linked_volatiles`).
///   * free Zapdos carrying a Perish Song counter -> switch out, which is the
///     only way to drop the counter.
///   * anything else -> max damage.
/// It is a BEST CASE for V1, not the bot's policy.
fn heur_zap_escape(
    b: &Battle,
    dex: &Dex,
    side: usize,
    choices: &[SearchChoice],
    md: &mut dyn Agent,
) -> SearchChoice {
    if let Some(id) = b.active_id(side) {
        let p = b.poke(id);
        if dex.species.key(p.species) == "zapdos" && !p.fainted {
            if p.trapped {
                if let Some(c) = pick_move(dex, choices, "whirlwind") {
                    return c;
                }
            } else {
                let perished = dex
                    .conds_id("perishsong")
                    .is_some_and(|cid| p.volatile(cid).is_some());
                if perished && choices.iter().any(|c| matches!(c, SearchChoice::Switch(_))) {
                    return healthiest_switch(b, side, choices);
                }
            }
        }
    }
    md.choose(b, dex, side, choices)
}

struct ArmAgent {
    label: String,
    script: Vec<Act>,
    depth: usize,
    tail: Tail,
    inner: Option<Box<dyn Agent>>,
    at: usize,
    /// `no-dead`: scripted picks that were provably dead and got replaced.
    replaced: usize,
    scripted: usize,
    dead_filter: bool,
    wake_rule: bool,
    /// Which mon comes in after ours faints. This is a free parameter of the
    /// harness, not of the position, and in 4069 it is worth more than the
    /// decision under test: with Zapdos and Snorlax both at 100 % HP,
    /// `MaxDamageAgent` takes the LAST tie (`agent.rs:142`) = Snorlax, and
    /// `healthiest_switch` the FIRST = Zapdos. `true` = first-max = Zapdos.
    replace_first_max: bool,
}

impl ArmAgent {
    fn tail_choice(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        match self.tail {
            Tail::FirstLegal => choices[0],
            Tail::SurfMax => heur_suicune(b, dex, side, choices, false),
            Tail::IceMax => heur_suicune(b, dex, side, choices, true),
            Tail::ZapEscape => {
                let inner = self.inner.as_mut().expect("tail agent built");
                heur_zap_escape(b, dex, side, choices, inner.as_mut())
            }
            Tail::Search | Tail::MaxDamage => {
                let inner = self.inner.as_mut().expect("tail agent built");
                inner.choose(b, dex, side, choices)
            }
        }
    }
}

impl Agent for ArmAgent {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if choices.len() == 1 {
            return choices[0];
        }
        // A forced replacement is not a scripted turn: it does not consume
        // the script, or every faint would shift the whole line by one.
        if is_forced_switch(choices) {
            if self.replace_first_max {
                return healthiest_switch(b, side, choices);
            }
            return match self.tail {
                // ZapEscape must route here too: `healthiest_switch` breaks
                // ties on the FIRST maximum and `MaxDamageAgent` on the LAST
                // (`agent.rs:142`), so leaving it on the other branch made the
                // two tails differ in replacement ORDER as well as in the
                // escape -- which is not a controlled comparison.
                Tail::Search | Tail::MaxDamage | Tail::ZapEscape => {
                    self.tail_choice(b, dex, side, choices)
                }
                _ => healthiest_switch(b, side, choices),
            };
        }
        let n = self.at;
        self.at += 1;
        if n < self.depth {
            if let Some(act) = self.script.get(n).cloned() {
                let resolved = match act {
                    Act::Move(k) => pick_move(dex, choices, k),
                    Act::SwitchTo(sp) => pick_switch(dex, b, side, choices, sp),
                };
                if let Some(c) = resolved {
                    if self.dead_filter && is_dead(b, dex, side, c, self.wake_rule) {
                        self.replaced += 1;
                        return heur_suicune(b, dex, side, choices, false);
                    }
                    self.scripted += 1;
                    return c;
                }
            }
        }
        self.tail_choice(b, dex, side, choices)
    }
}

// ------------------------------------------------------------ opponents

/// 4070's Umbreon as it actually played: Rest on waking below `rest_at`,
/// Toxic an unpoisoned AWAKE target, Charm otherwise, never switch while it
/// can still Rest. It bails to the bench when Rest is gone and it is low, or
/// once the foe is out of PP entirely (which is exactly turn 100 in the log).
struct FoeUmbreon {
    rest_at: f64,
    toxic_on_sleeper: bool,
    inner: Box<dyn Agent>,
}

fn foe_is_struggling(b: &Battle, dex: &Dex, foe_side: usize) -> bool {
    let Some(active) = b.active_id(foe_side) else { return false };
    let _ = dex;
    b.poke(active).move_slots.iter().all(|m| m.pp == 0)
}

impl Agent for FoeUmbreon {
    fn name(&self) -> String {
        "script-umbreon".into()
    }

    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if choices.len() == 1 {
            return choices[0];
        }
        if is_forced_switch(choices) {
            return healthiest_switch(b, side, choices);
        }
        let Some(active) = b.active_id(side) else { return choices[0] };
        let p = b.poke(active);
        if dex.species.key(p.species) != "umbreon" {
            return self.inner.choose(b, dex, side, choices);
        }
        let frac = p.hp as f64 / p.maxhp.max(1) as f64;
        let rest = pick_move(dex, choices, "rest");
        // bail to the bench: no Rest left and hurt, or the foe can only Struggle
        if (rest.is_none() && frac <= self.rest_at) || foe_is_struggling(b, dex, 1 - side) {
            if choices.iter().any(|c| matches!(c, SearchChoice::Switch(_))) {
                return healthiest_switch(b, side, choices);
            }
        }
        if stays_asleep(b, side) {
            // still asleep when the move resolves: the pick is invisible
            return pick_move(dex, choices, "charm").unwrap_or(choices[0]);
        }
        if frac <= self.rest_at {
            if let Some(c) = rest {
                return c;
            }
        }
        let foe_ok = b
            .active_id(1 - side)
            .map(|f| {
                let fp = b.poke(f);
                fp.status == Status::None && (self.toxic_on_sleeper || fp.status != Status::Slp)
            })
            .unwrap_or(false);
        if foe_ok {
            if let Some(c) = pick_move(dex, choices, "toxic") {
                return c;
            }
        }
        pick_move(dex, choices, "charm").unwrap_or(choices[0])
    }
}

/// 4069's Misdreavus as it actually played: Mean Look an untrapped foe,
/// Perish Song, then Destiny Bond, and leave at perish1 for the healthiest
/// bench mon (which is what saved it from its own song on turn 5).
struct FoeMisdreavus {
    /// `--foe-perish-stay`: do NOT leave at perish1 -- stay in and die with
    /// the mon it trapped. This is the opponent the player himself names in
    /// V2's addendum; the recorded human did the opposite.
    perish_stay: bool,
    /// `--foe-replacement`: which bench mon it sends, both on the perish1
    /// bail-out and on a forced replacement.
    replacement: FoeReplacement,
    inner: Box<dyn Agent>,
}

impl Agent for FoeMisdreavus {
    fn name(&self) -> String {
        "script-misdreavus".into()
    }

    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if choices.len() == 1 {
            return choices[0];
        }
        if is_forced_switch(choices) {
            return foe_switch(b, dex, side, choices, &self.replacement);
        }
        let Some(active) = b.active_id(side) else { return choices[0] };
        if dex.species.key(b.poke(active).species) != "misdreavus" {
            return self.inner.choose(b, dex, side, choices);
        }
        let perish = dex
            .conds_id("perishsong")
            .and_then(|id| b.poke(active).volatile(id).and_then(|s| s.duration));
        // perish1 = faints at this turn's residual; the human left instead.
        // `--foe-perish-stay` keeps it in: it then falls through to the move
        // ladder below (Destiny Bond, at that point) and dies on the song
        // with whatever it trapped.
        if !self.perish_stay
            && perish.is_some_and(|d| d <= 1)
            && choices.iter().any(|c| matches!(c, SearchChoice::Switch(_)))
        {
            return foe_switch(b, dex, side, choices, &self.replacement);
        }
        let foe_trapped = b.active_id(1 - side).map(|f| b.poke(f).trapped).unwrap_or(true);
        if !foe_trapped {
            if let Some(c) = pick_move(dex, choices, "meanlook") {
                return c;
            }
        }
        if perish.is_none() {
            if let Some(c) = pick_move(dex, choices, "perishsong") {
                return c;
            }
        }
        if let Some(c) = pick_move(dex, choices, "destinybond") {
            return c;
        }
        let p = b.poke(active);
        if p.hp * 2 < p.maxhp {
            if let Some(c) = pick_move(dex, choices, "rest") {
                return c;
            }
        }
        choices[0]
    }
}

// ------------------------------------------------------------ recorded lines

/// p2's submitted action per turn from turn 39 (Suicune's first turn as the
/// last mon) to turn 102, read off `tmp/corpus-4070/battle-4070.raw.log`.
/// Sleep-Talk-called moves are not decisions and are not in here.
const AS_PLAYED_4070: [&str; 64] = [
    "sleeptalk", "icebeam", "surf", "surf", "surf", "surf", "surf", "rest", "sleeptalk",
    "sleeptalk", "icebeam", "sleeptalk", "sleeptalk", "rest", "sleeptalk", "sleeptalk", "surf",
    "surf", "surf", "icebeam", "surf", "surf", "surf", "surf", "surf", "icebeam", "surf",
    "icebeam", "surf", "surf", "icebeam", "surf", "surf", "surf", "icebeam", "surf", "icebeam",
    "sleeptalk", "icebeam", "icebeam", "sleeptalk", "icebeam", "sleeptalk", "sleeptalk", "surf",
    "icebeam", "surf", "icebeam", "icebeam", "icebeam", "rest", "rest", "rest", "rest", "rest",
    "rest", "rest", "rest", "rest", "rest", "rest", "struggle", "struggle", "struggle",
];
const AS_PLAYED_4070_FROM: u16 = 39;

/// p2's submitted action per turn 1..=35 of battle 4069. Forced replacements
/// (turns 5 and 30) are not decisions and are not in here.
const AS_PLAYED_4069: [&str; 35] = [
    "curse", "curse", "return", "return", "return", "thunderbolt", "thunderbolt", "!snorlax",
    "earthquake", "!zapdos", "!snorlax", "earthquake", "!zapdos", "thunderbolt", "!snorlax",
    "bodyslam", "!zapdos", "thunderbolt", "!snorlax", "bodyslam", "rest", "!zapdos",
    "thunderbolt", "!snorlax", "bodyslam", "earthquake", "earthquake", "bodyslam", "bodyslam",
    "bodyslam", "whirlwind", "thunderwave", "whirlwind", "whirlwind", "whirlwind",
];
const AS_PLAYED_4069_FROM: u16 = 1;

/// p1 (the human) as recorded, same indexing as the p2 arrays above. An
/// empty string = the human owed no visible action that turn (asleep /
/// frozen); the `replay` foe then falls through to its scripted policy.
const FOE_PLAYED_4070: [&str; 64] = [
    "doubleedge", "", "", "", "", "", "", "!umbreon", "!snorlax", "bonemerang", "!umbreon",
    "toxic", "rest", "charm", "charm", "charm", "rest", "", "", "rest", "", "", "rest", "", "",
    "rest", "", "", "rest", "", "", "rest", "", "", "rest", "", "", "rest", "", "", "rest", "",
    "", "rest", "", "", "rest", "", "", "rest", "", "", "charm", "charm", "charm", "charm",
    "charm", "charm", "charm", "charm", "toxic", "!marowak", "bonemerang", "bonemerang",
];
const FOE_PLAYED_4069: [&str; 35] = [
    "meanlook", "perishsong", "destinybond", "destinybond", "!skarmory", "!blissey", "toxic",
    "softboiled", "!skarmory", "!blissey", "toxic", "!skarmory", "drillpeck", "!blissey",
    "toxic", "!skarmory", "rest", "!blissey", "healbell", "!skarmory", "drillpeck", "drillpeck",
    "!blissey", "toxic", "!misdreavus", "meanlook", "perishsong", "rest", "destinybond",
    "!skarmory", "!blissey", "perishsong", "rest", "healbell", "destinybond",
];

/// The human's recorded line, with a scripted policy behind it for every
/// turn the recording cannot supply (it ran out, the mon is gone, the move
/// is illegal or out of PP). This is the most faithful opponent available:
/// the state-driven `script` foe is a MODEL of the human and plays Toxic
/// far more often than he did.
struct ReplayFoe {
    script: Vec<Act>,
    at: usize,
    inner: Box<dyn Agent>,
}

impl Agent for ReplayFoe {
    fn name(&self) -> String {
        "replay-foe".into()
    }

    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if choices.len() == 1 {
            return choices[0];
        }
        if is_forced_switch(choices) {
            return healthiest_switch(b, side, choices);
        }
        let n = self.at;
        self.at += 1;
        if let Some(act) = self.script.get(n).cloned() {
            let resolved = match act {
                Act::Move(k) => pick_move(dex, choices, k),
                Act::SwitchTo(sp) => pick_switch(dex, b, side, choices, sp),
            };
            if let Some(c) = resolved {
                return c;
            }
        }
        self.inner.choose(b, dex, side, choices)
    }
}

fn to_acts(keys: &[&'static str]) -> Vec<Act> {
    keys.iter()
        .map(|k| match k.strip_prefix('!') {
            Some(species) => Act::SwitchTo(species),
            // "" never resolves, so the caller's fallback takes the turn
            None => Act::Move(k),
        })
        .collect()
}

// ------------------------------------------------------------ diagnostics

#[derive(Default, Clone, Copy)]
struct Diag {
    crit_on_p1: bool,
    surf_uses: u32,
    surf_hits: u32,
    p1_faints: u32,
    p2_faints: u32,
    first_p2_faint_at: Option<u16>,
    first_p1_faint_at: Option<u16>,
    trapped_p2: bool,
    // --- V1/V2 conditional diagnostics. Read from BATTLE STATE, not the log:
    // `trapped` is recomputed every turn by the TrapPokemon event
    // (`turn.rs:450`) and the trap's release when the trapper leaves is
    // SILENT (no `-end` line), so the log cannot answer these.
    zap_in: bool,
    zap_trapped: bool,
    zap_freed: bool,
    zap_ww_trapped: bool,
    zap_escaped: bool,
    zap_fainted: bool,
    zap_faint_trapped: bool,
    incoming_seen: bool,
    incoming_ko: bool,
}

fn scan_log(diag: &mut Diag, log: &[String], turn: u16) {
    let mut pending_surf = false;
    for l in log {
        if l.starts_with("|move|p2a") {
            pending_surf = l.contains("|Surf|");
            if pending_surf {
                diag.surf_uses += 1;
            }
            continue;
        }
        if l.starts_with("|-damage|p1a") && pending_surf {
            diag.surf_hits += 1;
            pending_surf = false;
        }
        if l.starts_with("|-crit|p1a") {
            diag.crit_on_p1 = true;
        }
        if l.starts_with("|-activate|p2a") && l.ends_with("trapped") {
            diag.trapped_p2 = true;
        }
        if l.starts_with("|faint|p1a") {
            diag.p1_faints += 1;
            diag.first_p1_faint_at.get_or_insert(turn);
        }
        if l.starts_with("|faint|p2a") {
            diag.p2_faints += 1;
            diag.first_p2_faint_at.get_or_insert(turn);
        }
    }
}

fn species_alive(b: &Battle, dex: &Dex, side: usize, species: &str) -> Option<bool> {
    let s = &b.sides[side];
    s.party
        .iter()
        .map(|&slot| &s.roster[slot as usize])
        .find(|p| dex.species.key(p.species) == species)
        .map(|p| !p.fainted && p.hp > 0)
}

// ------------------------------------------------------------------ main

struct ArmSpec {
    name: &'static str,
    script: Vec<Act>,
    depth: usize,
    tail: Tail,
    dead_filter: bool,
    /// Remove Whirlwind from side-1 Zapdos for the whole rollout (see
    /// `strip_move`). The control that separates "the phaze escape wins it"
    /// from "any switch off the doomed mon wins it".
    strip_whirlwind: bool,
}

// ------------------------------------------------------------- trial loop

/// Everything a trial needs that is not the position, the arm, or the seed.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    case: &'a str,
    foe_kind: &'a str,
    turn: u16,
    depth: usize,
    rest_at: f64,
    toxic_on_sleeper: bool,
    foe_perish_stay: bool,
    foe_replacement: &'a FoeReplacement,
    wake_rule: bool,
    replace_first_max: bool,
    iters: u32,
    base_seed: u64,
    max_turns: u16,
    trace_lo: u64,
    trace_hi: u64,
}

#[derive(Default, Clone, Copy)]
struct Acc {
    wins: u64,
    losses: u64,
    ties: u64,
    turns_survived: u64,
    n_crit: u64,
    n_surf_hits: u64,
    n_p1_ko: u64,
    n_traded: u64,
    n_zapdos_alive: u64,
    n_replaced: u64,
    n_zap_in: u64,
    n_zap_trapped: u64,
    n_zap_freed: u64,
    n_zap_ww: u64,
    n_zap_escaped: u64,
    n_zap_fainted: u64,
    n_zap_faint_trapped: u64,
    n_incoming: u64,
    n_incoming_ko: u64,
}

impl Acc {
    fn merge(&mut self, o: &Acc) {
        self.wins += o.wins;
        self.losses += o.losses;
        self.ties += o.ties;
        self.turns_survived += o.turns_survived;
        self.n_crit += o.n_crit;
        self.n_surf_hits += o.n_surf_hits;
        self.n_p1_ko += o.n_p1_ko;
        self.n_traded += o.n_traded;
        self.n_zapdos_alive += o.n_zapdos_alive;
        self.n_replaced += o.n_replaced;
        self.n_zap_in += o.n_zap_in;
        self.n_zap_trapped += o.n_zap_trapped;
        self.n_zap_freed += o.n_zap_freed;
        self.n_zap_ww += o.n_zap_ww;
        self.n_zap_escaped += o.n_zap_escaped;
        self.n_zap_fainted += o.n_zap_fainted;
        self.n_zap_faint_trapped += o.n_zap_faint_trapped;
        self.n_incoming += o.n_incoming;
        self.n_incoming_ko += o.n_incoming_ko;
    }
}

fn build_foe(
    dex_pool: &Arc<MetaPool>,
    spec_seed: u64,
    ctx: &Ctx,
) -> Box<dyn Agent> {
    let script_foe = || -> Box<dyn Agent> {
        if ctx.case == "4070-endgame" {
            Box::new(FoeUmbreon {
                rest_at: ctx.rest_at,
                toxic_on_sleeper: ctx.toxic_on_sleeper,
                inner: Box::new(MaxDamageAgent::new()),
            })
        } else {
            Box::new(FoeMisdreavus {
                perish_stay: ctx.foe_perish_stay,
                replacement: ctx.foe_replacement.clone(),
                inner: Box::new(MaxDamageAgent::new()),
            })
        }
    };
    match ctx.foe_kind {
        "max-damage" => Box::new(MaxDamageAgent::new()),
        "search" => Box::new(BlindAgent::new(
            RmConfig { iterations: ctx.iters, ..cfg() },
            dex_pool.clone(),
            None,
            spec_seed ^ 0xabcd,
        )),
        "script" => script_foe(),
        "replay" => {
            let (all, from): (&[&'static str], u16) = if ctx.case == "4070-endgame" {
                (&FOE_PLAYED_4070, AS_PLAYED_4070_FROM)
            } else {
                (&FOE_PLAYED_4069, AS_PLAYED_4069_FROM)
            };
            let off = ctx.turn.saturating_sub(from) as usize;
            let tail: Vec<&'static str> = all.get(off..).map(<[&str]>::to_vec).unwrap_or_default();
            Box::new(ReplayFoe { script: to_acts(&tail), at: 0, inner: script_foe() })
        }
        other => panic!("unknown --foe {other}"),
    }
}

fn run_range(
    dex: &Dex,
    base: &Battle,
    pool: &Arc<MetaPool>,
    spec: &ArmSpec,
    ctx: &Ctx,
    lo: u64,
    hi: u64,
) -> Acc {
    let mut acc = Acc::default();
    for t in lo..hi {
        let mut b = base.clone();
        b.set_log_enabled(true);
        // The battle seed is a pure function of the trial index, so the
        // thread count cannot move a number. It must NOT go through
        // `Prng::from_seed_str`: that parses four 16-bit limbs and rejects
        // anything above 0xFFFF (`prng.rs:32`), so the string form panics on
        // every trial index past 65506 and silently caps this harness at ~65k
        // trials -- below the sample size these questions need. splitmix64
        // instead, which fills all 64 bits.
        let mut z = (ctx.base_seed ^ 0x243f_6a88_85a3_08d3)
            .wrapping_add((t + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        b.prng = BattleRng::seeded(Prng::new(z));
        let start = b.turn;
        let seed = ctx
            .base_seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(t)
            .wrapping_add(spec.name.len() as u64);

        let tracing = t >= ctx.trace_lo && t < ctx.trace_hi;
        if tracing {
            println!("\n######## TRACE arm={} trial={} ########", spec.name, t);
        }

        let inner: Option<Box<dyn Agent>> = match spec.tail {
            Tail::Search => Some(Box::new(BlindAgent::new(
                RmConfig { iterations: ctx.iters, ..cfg() },
                pool.clone(),
                None,
                seed,
            ))),
            Tail::MaxDamage | Tail::ZapEscape => Some(Box::new(MaxDamageAgent::new())),
            _ => None,
        };
        let mut me = ArmAgent {
            label: spec.name.to_string(),
            script: spec.script.clone(),
            depth: ctx.depth,
            tail: spec.tail,
            inner,
            at: 0,
            replaced: 0,
            scripted: 0,
            dead_filter: spec.dead_filter,
            wake_rule: ctx.wake_rule,
            replace_first_max: ctx.replace_first_max,
        };
        let mut foe = build_foe(pool, seed, ctx);

        let mut diag = Diag::default();
        // V1/V2 anchors: our mon at the decision (Miltank / Suicune) and the
        // opponent's (Misdreavus / Umbreon), so "the mon that comes in after
        // the trapper leaves" is identifiable.
        let start_p2: Option<PokeId> = b.active_id(1);
        let name_of = |id: Option<PokeId>| -> String {
            id.map(|i| dex.species.key(b.poke(i).species).to_string()).unwrap_or_default()
        };
        let start_p1_name = name_of(b.active_id(0));
        // V2's own mechanism. The replacement arrives and can die INSIDE one
        // turn (the trapper leaves at perish1, the new mon eats our attack,
        // our mon then perishes at the same turn's residual), so this cannot
        // be read from the between-turn state -- it is read off the turn's
        // own log slice.
        let mut incoming_name: Option<String> = None;
        // `Battle::log` is NEVER truncated per turn (only `set_log_enabled(false)`
        // drops it), so the diagnostics must read a moving cursor — scanning
        // `b.log` whole after every apply would multiply-count every event.
        let mut cursor = b.log.len();
        let mut result: Option<Outcome> = None;
        loop {
            if let Some(o) = b.outcome() {
                result = Some(o);
                break;
            }
            if b.turn > ctx.max_turns {
                break;
            }
            let turn_now = b.turn;
            let mut picks = [None, None];
            {
                let agents: [&mut dyn Agent; 2] = [foe.as_mut(), &mut me];
                for s in 0..2 {
                    let cs = b.legal_choices(dex, s);
                    if !cs.is_empty() {
                        picks[s] = Some(agents[s].choose(&b, dex, s, &cs));
                    }
                }
            }
            if picks == [None, None] {
                break;
            }
            // ---- V1: what actually happens to Zapdos (state, pre-apply)
            let zap = b
                .active_id(1)
                .filter(|id| dex.species.key(b.poke(*id).species) == "zapdos");
            let zap_trapped_now = zap.is_some_and(|id| b.poke(id).trapped);
            if let Some(id) = zap {
                if !b.poke(id).fainted {
                    diag.zap_in = true;
                    if zap_trapped_now {
                        diag.zap_trapped = true;
                        if matches!(picks[1], Some(SearchChoice::Move(m))
                            if dex.moves.key(m) == "whirlwind")
                        {
                            diag.zap_ww_trapped = true;
                        }
                    } else if diag.zap_trapped {
                        // the trap released (the trapper left the field)
                        diag.zap_freed = true;
                        if matches!(picks[1], Some(SearchChoice::Switch(_))) {
                            diag.zap_escaped = true;
                        }
                    }
                }
            }
            // ---- V2: is our starter still the one on the field this turn?
            let p2_still_start = b.active_id(1) == start_p2
                && start_p2.is_some_and(|id| !b.poke(id).fainted);

            let turn_lo = b.log.len();
            if tracing {
                let lbl = |c: Option<SearchChoice>| -> String {
                    match c {
                        Some(SearchChoice::Move(m)) => format!("move {}", dex.moves.key(m)),
                        Some(SearchChoice::Switch(p)) => format!(
                            "switch {p} ({})",
                            switch_species(dex, &b, 0, p).unwrap_or_default()
                        ),
                        Some(SearchChoice::Pass) => "pass".into(),
                        Some(SearchChoice::Team(t)) => format!("team {t:?}"),
                        None => "-".into(),
                    }
                };
                let lbl1 = |c: Option<SearchChoice>| -> String {
                    match c {
                        Some(SearchChoice::Switch(p)) => format!(
                            "switch {p} ({})",
                            switch_species(dex, &b, 1, p).unwrap_or_default()
                        ),
                        other => lbl(other),
                    }
                };
                let hp = |s: usize| -> String {
                    match b.active_id(s) {
                        Some(id) => {
                            let p = b.poke(id);
                            format!(
                                "{} {}/{}{}",
                                dex.species.key(p.species),
                                p.hp,
                                p.maxhp,
                                if p.status == Status::None {
                                    String::new()
                                } else {
                                    format!(" {:?}", p.status)
                                }
                            )
                        }
                        None => "-".into(),
                    }
                };
                println!(
                    "T{turn_now}  [p1 {}]  [p2 {}]   p1={}  p2={}",
                    hp(0),
                    hp(1),
                    lbl(picks[0]),
                    lbl1(picks[1])
                );
            }
            b.apply_choices(dex, picks).expect("apply");
            if tracing {
                for l in &b.log[turn_lo..] {
                    if l.is_empty() || l.starts_with("|t:|") || l.starts_with("|upkeep") {
                        continue;
                    }
                    println!("      {l}");
                }
            }
            scan_log(&mut diag, &b.log[cursor..], turn_now);
            cursor = b.log.len();

            if p2_still_start {
                for l in &b.log[turn_lo..] {
                    if incoming_name.is_none() {
                        if let Some(rest) = l
                            .strip_prefix("|switch|p1a: ")
                            .or_else(|| l.strip_prefix("|drag|p1a: "))
                        {
                            let who = rest.split('|').next().unwrap_or("").trim().to_lowercase();
                            if !who.is_empty() && who != start_p1_name {
                                incoming_name = Some(who);
                                diag.incoming_seen = true;
                            }
                        }
                    }
                    if let (Some(inc), Some(rest)) =
                        (incoming_name.as_deref(), l.strip_prefix("|faint|p1a: "))
                    {
                        if rest.split('|').next().unwrap_or("").trim().to_lowercase() == inc {
                            diag.incoming_ko = true;
                        }
                    }
                }
            }
            if let Some(id) = zap {
                if b.poke(id).fainted && !diag.zap_fainted {
                    diag.zap_fainted = true;
                    diag.zap_faint_trapped = zap_trapped_now;
                }
            }
        }

        if tracing {
            println!("######## TRACE end arm={} trial={} outcome={:?} turn={} ########", spec.name, t, result, b.turn);
        }
        acc.turns_survived += b.turn.saturating_sub(start) as u64;
        match result {
            Some(Outcome::P2Win) => acc.wins += 1,
            Some(Outcome::P1Win) => acc.losses += 1,
            _ => acc.ties += 1,
        }
        if diag.crit_on_p1 {
            acc.n_crit += 1;
        }
        acc.n_surf_hits += diag.surf_hits as u64;
        acc.n_replaced += me.replaced as u64;
        if ctx.case == "4070-endgame" {
            if species_alive(&b, dex, 0, "umbreon") == Some(false) {
                acc.n_p1_ko += 1;
            }
        } else {
            if species_alive(&b, dex, 1, "zapdos") == Some(true) {
                acc.n_zapdos_alive += 1;
            }
            let traded = match (diag.first_p1_faint_at, diag.first_p2_faint_at) {
                (Some(a), Some(bb)) => a <= bb,
                (Some(_), None) => true,
                _ => false,
            };
            if traded {
                acc.n_traded += 1;
            }
        }
        for (hit, slot) in [
            (diag.zap_in, &mut acc.n_zap_in),
            (diag.zap_trapped, &mut acc.n_zap_trapped),
            (diag.zap_freed, &mut acc.n_zap_freed),
            (diag.zap_ww_trapped, &mut acc.n_zap_ww),
            (diag.zap_escaped, &mut acc.n_zap_escaped),
            (diag.zap_fainted, &mut acc.n_zap_fainted),
            (diag.zap_faint_trapped, &mut acc.n_zap_faint_trapped),
            (diag.incoming_seen, &mut acc.n_incoming),
            (diag.incoming_ko, &mut acc.n_incoming_ko),
        ] {
            if hit {
                *slot += 1;
            }
        }
    }
    acc
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let case = arg(&args, "--case").unwrap_or("4070-endgame").to_string();
    let trials: u64 = arg(&args, "--trials").unwrap_or("2000").parse().unwrap();
    let iters: u32 = arg(&args, "--iters").unwrap_or("30000").parse().unwrap();
    let rest_at: f64 = arg(&args, "--rest-at").unwrap_or("0.5").parse().unwrap();
    let foe_kind = arg(&args, "--foe").unwrap_or("script").to_string();
    let max_turns: u16 = arg(&args, "--max-turns").unwrap_or("1000").parse().unwrap();
    let base_seed: u64 = arg(&args, "--seed").unwrap_or("1").parse().unwrap();
    let recon_seed: u64 = arg(&args, "--recon-seed").unwrap_or("1").parse().unwrap();
    let oracle_moves = !flag(&args, "--no-oracle-moves");
    let toxic_on_sleeper = flag(&args, "--toxic-on-sleeper");
    // `no-dead` counts a wake-up-turn Sleep Talk as dead too; --no-wake-rule
    // restricts it to the strictly-awake case so the two can be separated.
    let wake_rule = !flag(&args, "--no-wake-rule");
    let replace_first_max = flag(&args, "--replace-first-max");
    // The two opponent axes V5 says the answer depends on. Both default to
    // exactly what the harness did before they existed.
    let foe_perish_stay = flag(&args, "--foe-perish-stay");
    let dump_recon = flag(&args, "--dump-recon");
    let foe_replacement = parse_replacement(arg(&args, "--foe-replacement").unwrap_or("healthiest"));
    let trace_lo: u64 = arg(&args, "--trace-lo").unwrap_or("0").parse().unwrap();
    let trace_hi: u64 = arg(&args, "--trace-hi").unwrap_or("0").parse().unwrap();
    let csv_path = arg(&args, "--csv").map(str::to_string);
    let threads: usize = arg(&args, "--threads")
        .map(|s| s.parse().expect("--threads"))
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
        .max(1);
    let arm_depth_override: Option<usize> =
        arg(&args, "--arm-depth").map(|s| s.parse().expect("--arm-depth"));

    let root = repo_root();
    let default_log = match case.as_str() {
        "4070-endgame" => "tmp/corpus-4070/battle-4070.raw.log",
        "4069-lead" | "4069-trapped" => "tmp/corpus-4069-truepicks/battle-4069t.raw.log",
        other => panic!("unknown --case {other}"),
    };
    let log = root.join(arg(&args, "--log").unwrap_or(default_log));

    let dex = load_dex();
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let pool: Arc<MetaPool> = Arc::new(load_meta_pool(&pool_path));
    let battle_log = load_battle(&log);

    // ---- pick the decision
    let turn: u16 = match arg(&args, "--turn") {
        Some(t) => t.parse().unwrap(),
        None => match case.as_str() {
            // first p2 MOVE decision with Suicune already the last mon
            "4070-endgame" => {
                let mut found = None;
                for d in &battle_log.decisions {
                    if d.side != 1 || !matches!(d.action, HumanAction::Move(_)) {
                        continue;
                    }
                    let faints = battle_log.lines[..=d.cut]
                        .iter()
                        .filter(|l| l.starts_with("|faint|p2a"))
                        .count();
                    if faints >= 2 {
                        found = Some(d.turn);
                        break;
                    }
                }
                found.expect("no last-mon p2 decision found")
            }
            "4069-lead" => 1,
            _ => 2,
        },
    };

    let d = battle_log
        .decisions
        .iter()
        .find(|d| d.side == 1 && d.turn == turn)
        .unwrap_or_else(|| panic!("no p2 decision at turn {turn}"));
    let mut base = reconstruct(
        &dex,
        &src,
        &pool_path,
        &battle_log.lines,
        &battle_log.evidence,
        d,
        recon_seed,
    )
    .expect("reconstruction");
    let oracled = if oracle_moves {
        complete_active_moves_from_future(&dex, &mut base, &battle_log.lines)
    } else {
        [Vec::new(), Vec::new()]
    };

    println!("case {case}  log {}", log.display());
    println!(
        "decision: p2 turn {turn}, played {:?}; oracle-moves={oracle_moves} {:?}",
        d.action, oracled
    );
    for s in 0..2 {
        let mons: Vec<String> = base.sides[s]
            .party
            .iter()
            .map(|&slot| {
                let p = &base.sides[s].roster[slot as usize];
                format!(
                    "{} {:.0}%{}",
                    dex.species.key(p.species),
                    100.0 * p.hp as f64 / p.maxhp.max(1) as f64,
                    if p.status == Status::None {
                        String::new()
                    } else {
                        format!(" {:?}", p.status)
                    }
                )
            })
            .collect();
        println!("  side {s} (left {}): {}", base.sides[s].pokemon_left, mons.join(", "));
        if let Some(id) = base.active_id(s) {
            let pp: Vec<String> = base
                .poke(id)
                .move_slots
                .iter()
                .map(|m| format!("{}({})", dex.moves.key(m.id), m.pp))
                .collect();
            println!("    active {}: {}", dex.species.key(base.poke(id).species), pp.join(" "));
        }
    }

    // `--dump-recon`: the FULL reconstructed board for both sides -- every
    // party member's set as the rollout will actually use it, plus which of
    // its moves the protocol had actually revealed by this decision's cut.
    // Everything not in the revealed list is fabricated (`corpus::fabricate_set`
    // for our own side, the belief's candidate team for the opponent's) or,
    // for the ACTIVE pair only, leaked from the future by `--oracle-moves`.
    if dump_recon {
        let mut revealed: [std::collections::BTreeMap<String, Vec<String>>; 2] =
            [Default::default(), Default::default()];
        let mut t: u16 = 0;
        for ln in &battle_log.lines {
            if let Some(rest) = ln.strip_prefix("|turn|") {
                t = rest.trim().parse().unwrap_or(t);
            }
            if t >= turn {
                break;
            }
            if let Some(rest) = ln.strip_prefix("|move|") {
                let mut it = rest.split('|');
                let who = it.next().unwrap_or("");
                let mv = it.next().unwrap_or("");
                let side = if who.starts_with("p1") { 0 } else { 1 };
                let nick = who.split_once(": ").map(|x| x.1).unwrap_or(who).to_string();
                let e = revealed[side].entry(nick).or_default();
                let key = mv.to_lowercase().replace([' ', '-', '\''], "");
                if !e.contains(&key) {
                    e.push(key);
                }
            }
        }
        println!("\n--- reconstruction dump (cut = start of turn {turn}) ---");
        for s in 0..2 {
            let active = base.active_id(s);
            for (i, &slot) in base.sides[s].party.iter().enumerate() {
                let p = &base.sides[s].roster[slot as usize];
                let nick = p.name.to_string();
                let empty: Vec<String> = Vec::new();
                let rv = revealed[s].get(&nick).unwrap_or(&empty);
                let mvs: Vec<String> = p
                    .base_move_slots
                    .iter()
                    .map(|m| {
                        let k = dex.moves.key(m.id).to_string();
                        let tag = if rv.iter().any(|r| *r == k) { "REVEALED" } else { "fabricated" };
                        format!("{k}[{tag}]")
                    })
                    .collect();
                let mark = if active.map(|a| a.slot) == Some(slot) { " <ACTIVE>" } else { "" };
                let item = p.item.map(|i| dex.items.key(i).to_string()).unwrap_or_else(|| "-".into());
                println!(
                    "  side {s} slot{i} {}{mark} L{} hp {}/{} {:?} item {item}\n      moves: {}\n      revealed-by-cut: {rv:?}",
                    dex.species.key(p.species),
                    p.level,
                    p.hp,
                    p.maxhp,
                    p.status,
                    mvs.join(" ")
                );
            }
        }
        println!("--- end dump ---\n");
    }

    if let FoeReplacement::Species(sp) = &foe_replacement {
        let on_team = base.sides[0]
            .party
            .iter()
            .any(|&slot| dex.species.key(base.sides[0].roster[slot as usize].species) == sp);
        assert!(on_team, "--foe-replacement {sp}: side 0 has no such mon in this reconstruction");
    }

    // ---- arms
    let default_tail = match case.as_str() {
        "4070-endgame" => Tail::FirstLegal,
        _ => Tail::Search,
    };
    let tail = arg(&args, "--tail").map(parse_tail).unwrap_or(default_tail);

    let all_arms: Vec<ArmSpec> = match case.as_str() {
        "4070-endgame" => {
            let from = turn.saturating_sub(AS_PLAYED_4070_FROM) as usize;
            let played: Vec<&'static str> =
                AS_PLAYED_4070.get(from..).map(<[&str]>::to_vec).unwrap_or_default();
            let script = to_acts(&played);
            vec![
                ArmSpec {
                    name: "as-played",
                    depth: script.len(),
                    script: script.clone(),
                    tail,
                    dead_filter: false,
                    strip_whirlwind: false,
                },
                ArmSpec {
                    name: "no-dead",
                    depth: script.len(),
                    script,
                    tail: Tail::SurfMax,
                    dead_filter: true,
                    strip_whirlwind: false,
                },
                ArmSpec {
                    name: "surf-max",
                    script: Vec::new(),
                    depth: 0,
                    tail: Tail::SurfMax,
                    dead_filter: false,
                    strip_whirlwind: false,
                },
                ArmSpec {
                    name: "ice-max",
                    script: Vec::new(),
                    depth: 0,
                    tail: Tail::IceMax,
                    dead_filter: false,
                    strip_whirlwind: false,
                },
                ArmSpec {
                    name: "search",
                    script: Vec::new(),
                    depth: 0,
                    tail: Tail::Search,
                    dead_filter: false,
                    strip_whirlwind: false,
                },
            ]
        }
        "4069-lead" | "4069-trapped" => {
            let from = turn.saturating_sub(AS_PLAYED_4069_FROM) as usize;
            let played: Vec<&'static str> =
                AS_PLAYED_4069.get(from..).map(<[&str]>::to_vec).unwrap_or_default();
            let script = to_acts(&played);
            let rep = |k: &'static str| vec![Act::Move(k); 64];
            let mut v = vec![ArmSpec {
                name: "as-played",
                depth: script.len(),
                script,
                tail,
                dead_filter: false,
                strip_whirlwind: false,
            }];
            if case == "4069-lead" {
                v.push(ArmSpec {
                    name: "switch-zapdos",
                    script: vec![Act::SwitchTo("zapdos")],
                    depth: 1,
                    tail,
                    dead_filter: false,
                    strip_whirlwind: false,
                });
                // Same first action, on a board where Zapdos has no
                // Whirlwind at all. `switch-zapdos` minus this arm is the
                // part of the gain that the phaze escape is responsible for;
                // this arm alone is the part that is just "get off the mon
                // that is about to die".
                v.push(ArmSpec {
                    name: "switch-zapdos-noww",
                    script: vec![Act::SwitchTo("zapdos")],
                    depth: 1,
                    tail,
                    dead_filter: false,
                    strip_whirlwind: true,
                });
                v.push(ArmSpec {
                    name: "switch-snorlax",
                    script: vec![Act::SwitchTo("snorlax")],
                    depth: 1,
                    tail,
                    dead_filter: false,
                    strip_whirlwind: false,
                });
                v.push(ArmSpec {
                    name: "curse-through",
                    script: rep("curse"),
                    depth: 64,
                    tail,
                    dead_filter: false,
                    strip_whirlwind: false,
                });
                v.push(ArmSpec {
                    name: "earthquake-through",
                    script: rep("earthquake"),
                    depth: 64,
                    tail,
                    dead_filter: false,
                    strip_whirlwind: false,
                });
                v.push(ArmSpec {
                    name: "eq-then-search",
                    script: rep("earthquake"),
                    depth: 4,
                    tail,
                    dead_filter: false,
                    strip_whirlwind: false,
                });
                v.push(ArmSpec {
                    name: "return",
                    script: vec![Act::Move("return")],
                    depth: 1,
                    tail,
                    dead_filter: false,
                    strip_whirlwind: false,
                });
                v.push(ArmSpec {
                    name: "doubleteam",
                    script: vec![Act::Move("doubleteam")],
                    depth: 1,
                    tail,
                    dead_filter: false,
                    strip_whirlwind: false,
                });
            } else {
                for k in ["curse", "doubleteam", "milkdrink", "return"] {
                    let name: &'static str = match k {
                        "curse" => "curse-through",
                        "doubleteam" => "doubleteam-through",
                        "milkdrink" => "milkdrink-through",
                        _ => "return-through",
                    };
                    v.push(ArmSpec {
                        name,
                        script: rep(k),
                        depth: 64,
                        tail,
                        dead_filter: false,
                        strip_whirlwind: false,
                    });
                }
            }
            v.push(ArmSpec {
                name: "search",
                script: Vec::new(),
                depth: 0,
                tail: Tail::Search,
                dead_filter: false,
                strip_whirlwind: false,
            });
            v
        }
        other => panic!("unknown --case {other}"),
    };

    let want = arg(&args, "--arm").unwrap_or("default");
    let arms: Vec<ArmSpec> = all_arms
        .into_iter()
        .filter(|a| match want {
            "all" => true,
            // the search arm costs `--iters` playouts per decision per trial
            "default" => a.name != "search",
            list => list.split(',').any(|w| w == a.name),
        })
        .collect();
    assert!(!arms.is_empty(), "--arm selected nothing");

    println!(
        "\nfoe={foe_kind} rest-at={rest_at:.2} tail={tail:?} iters={iters} trials={trials} \
         max-turns={max_turns} wake-rule={wake_rule} replace-first-max={replace_first_max} \
         threads={threads}\n\
         foe-perish-stay={foe_perish_stay} foe-replacement={foe_replacement:?}\n"
    );

    let mut csv = String::from(
        "case,turn,arm,foe,tail,iters,trials,wins,losses,ties,winrate,ci95,mean_turns,\
         p1_ko,crit,surf_hits,replaced,zapdos_alive,traded,\
         zap_in,zap_meanlooked,zap_ww_trapped,zap_freed,zap_escaped,zap_fainted,\
         zap_died_trapped,incoming_seen,incoming_ko\n",
    );

    for spec in &arms {
        let depth = arm_depth_override.unwrap_or(spec.depth);
        let started = std::time::Instant::now();
        // Per-arm board. Only `strip_whirlwind` arms differ from `base`, and
        // they differ in the STATE, so the search, the mask and PP accounting
        // all see the same Zapdos.
        let arm_base: Battle = {
            let mut nb = base.clone();
            if spec.strip_whirlwind {
                let n = strip_move(&dex, &mut nb, 1, "zapdos", "whirlwind");
                assert!(n > 0, "arm {}: side-1 Zapdos carries no Whirlwind to strip", spec.name);
                println!("[{}] stripped {n} Whirlwind slot(s) from side-1 Zapdos", spec.name);
            }
            nb
        };
        let ctx = Ctx {
            case: &case,
            foe_kind: &foe_kind,
            turn,
            depth,
            rest_at,
            toxic_on_sleeper,
            foe_perish_stay,
            foe_replacement: &foe_replacement,
            wake_rule,
            replace_first_max,
            iters,
            base_seed,
            max_turns,
            trace_lo,
            trace_hi,
        };
        // Trials are independent; shard them. Every seed is a pure function
        // of the trial index, so the thread count cannot move a number.
        let bounds_of = |k: usize| -> (u64, u64) {
            let per = trials / threads as u64;
            let rem = trials % threads as u64;
            let k = k as u64;
            let lo = k * per + k.min(rem);
            let hi = lo + per + u64::from(k < rem);
            (lo, hi)
        };
        let acc = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|k| {
                    let (lo, hi) = bounds_of(k);
                    let dex = &dex;
                    let base = &arm_base;
                    let pool = pool.clone();
                    let ctx = ctx;
                    scope.spawn(move || run_range(dex, base, &pool, spec, &ctx, lo, hi))
                })
                .collect();
            handles.into_iter().fold(Acc::default(), |mut a, h| {
                a.merge(&h.join().expect("trial thread"));
                a
            })
        });
        let Acc {
            wins,
            losses,
            ties,
            turns_survived,
            n_crit,
            n_surf_hits,
            n_p1_ko,
            n_traded,
            n_zapdos_alive,
            n_replaced,
            n_zap_in,
            n_zap_trapped,
            n_zap_freed,
            n_zap_ww,
            n_zap_escaped,
            n_zap_fainted,
            n_zap_faint_trapped,
            n_incoming,
            n_incoming_ko,
        } = acc;


        let p = wins as f64 / trials as f64;
        let ci = 1.96 * (p * (1.0 - p) / trials as f64).sqrt();
        println!(
            "{:<18} win {wins:>6}/{trials} = {p:.4} ± {ci:.4}  (loss {losses}, tie/cap {ties}, \
             mean turns {:.1}, {:.0}s)",
            spec.name,
            turns_survived as f64 / trials as f64,
            started.elapsed().as_secs_f64()
        );
        if case == "4070-endgame" {
            println!(
                "                   P(Umbreon KO) {:.4}  P(>=1 crit on p1) {:.4}  \
                 mean Surf hits {:.2}  dead picks replaced {:.2}",
                n_p1_ko as f64 / trials as f64,
                n_crit as f64 / trials as f64,
                n_surf_hits as f64 / trials as f64,
                n_replaced as f64 / trials as f64
            );
        } else {
            let f = |n: u64| n as f64 / trials as f64;
            println!(
                "                   P(foe faints no later than our first) {:.4}  \
                 P(Zapdos alive at end) {:.4}  P(>=1 crit on p1) {:.4}",
                f(n_traded),
                f(n_zapdos_alive),
                f(n_crit)
            );
            println!(
                "                   zapdos: reached {:.4}  MeanLooked {:.4}  \
                 WW-while-trapped {:.4}  freed {:.4}  escaped-by-switch {:.4}  \
                 fainted {:.4}  died-trapped {:.4}",
                f(n_zap_in),
                f(n_zap_trapped),
                f(n_zap_ww),
                f(n_zap_freed),
                f(n_zap_escaped),
                f(n_zap_fainted),
                f(n_zap_faint_trapped)
            );
            let cond = if n_incoming > 0 {
                n_incoming_ko as f64 / n_incoming as f64
            } else {
                f64::NAN
            };
            println!(
                "                   incoming (trapper left, our starter still in): \
                 seen {:.4}  KO'd {:.4}  P(KO | seen) {:.4}",
                f(n_incoming),
                f(n_incoming_ko),
                cond
            );
        }
        let arm_tail = if spec.name == "search" { Tail::Search } else { spec.tail };
        csv.push_str(&format!(
            "{case},{turn},{},{foe_kind},{arm_tail:?},{iters},{trials},{wins},{losses},{ties},\
             {p:.6},{ci:.6},{:.3},\
             {:.6},{:.6},{:.4},{:.4},{:.6},{:.6},\
             {:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            spec.name,
            turns_survived as f64 / trials as f64,
            n_p1_ko as f64 / trials as f64,
            n_crit as f64 / trials as f64,
            n_surf_hits as f64 / trials as f64,
            n_replaced as f64 / trials as f64,
            n_zapdos_alive as f64 / trials as f64,
            n_traded as f64 / trials as f64,
            n_zap_in as f64 / trials as f64,
            n_zap_trapped as f64 / trials as f64,
            n_zap_ww as f64 / trials as f64,
            n_zap_freed as f64 / trials as f64,
            n_zap_escaped as f64 / trials as f64,
            n_zap_fainted as f64 / trials as f64,
            n_zap_faint_trapped as f64 / trials as f64,
            n_incoming as f64 / trials as f64,
            n_incoming_ko as f64 / trials as f64,
        ));
    }

    if let Some(path) = csv_path {
        std::fs::write(&path, csv).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("\ncsv -> {path}");
    }
}
