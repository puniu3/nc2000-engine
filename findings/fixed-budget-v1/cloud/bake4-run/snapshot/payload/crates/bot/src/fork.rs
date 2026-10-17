//! Forked positions: one recorded decision point restarted as a live battle
//! in which the bot's first action is fixed to one of several arms and both
//! sides play freely afterwards.
//!
//! An `exact_replay` reconstructs the true battle from its initial seed and
//! committed choices. Legacy positions without a replay synthesize hidden
//! quantities from `position` and the supplied opponent team.
//! The bot plays through [`ProtocolSeat`]: the ladder `ProtocolAgent`,
//! installed from `position` and fed that side's player stream from the fork
//! onwards, so its information set continues the recorded one.

use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::state::Battle;
use nc2000_engine::replay::Replay;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::agent::Agent;
use crate::import::{apply_party, ProtocolAgent};
use crate::player::{action_input, PlayerChannel};
use crate::position::{synthesize_spec, PositionSpec};
use crate::preview::MetaPool;
use crate::rng::SplitMix64;
use crate::smmcts::{RmAgent, RmConfig, SelRule};

pub const SCHEMA: &str = "nc2000-fork-v1";

const PICKS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Info {
    /// The bot sees only public information, as on the ladder.
    Blind,
    /// The bot's belief is pinned to `opponent_team`.
    Open,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    /// PS choice string for the bot's first decision.
    pub input: String,
    #[serde(default)]
    pub label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForkSpec {
    pub schema: String,
    #[serde(default)]
    pub label: String,
    pub info: Info,
    /// The bot's decision point; `position.side` is the bot.
    pub position: PositionSpec,
    /// The opponent's true sets in team-preview order.
    pub opponent_team: Vec<PokemonSet>,
    /// Species of the opponent's three picks. Required while fewer than
    /// three have appeared at the fork.
    #[serde(default)]
    pub opponent_picks: Vec<String>,
    /// The same decision point from the opponent's information set, for an
    /// opponent played by a [`ProtocolSeat`].
    #[serde(default)]
    pub opponent_position: Option<PositionSpec>,
    pub arms: Vec<Arm>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_replay: Option<ReplayOrigin>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayOrigin {
    pub code: String,
    pub round: usize,
}

impl ReplayOrigin {
    fn agent(&self, dex: &Dex, pool: MetaPool, cfg: RmConfig, seed: u64, side: usize) -> Result<ProtocolAgent, String> {
        let replay = Replay::decode(dex, &self.code)?;
        if self.round >= replay.rounds.len() || side > 1 { return Err("invalid replay fork point".into()); }
        let mut agent = ProtocolAgent::new(dex, side, pool, cfg, seed);
        agent.set_own_team(replay.teams[side].clone());
        if replay.open { agent.pin_opponent(replay.teams[1-side].clone()); }
        let mut channel = PlayerChannel::new(side);
        let mut battle = replay.initial(dex)?;
        for round in 0..=self.round {
            let frame = channel.frame(&mut battle, dex)?;
            for line in &frame.lines { agent.push_line(dex, line); }
            agent.on_request(dex, &frame.request.to_string())?;
            if round < self.round { replay.apply(dex, &mut battle, round)?; }
        }
        Ok(agent)
    }
}

impl ForkSpec {
    pub fn from_replay(dex: &Dex, code: &str, round: usize, alternative: &str, pool: &MetaPool) -> Result<Self, String> {
        let replay = Replay::decode(dex, code)?;
        let bot = replay.bot_side;
        let played = replay.rounds.get(round).and_then(|p| p[bot]).ok_or("bot has no recorded action here")?;
        let mut battle = replay.at(dex, round)?;
        if battle.turn == 0 { return Err("choose a battle turn".into()); }
        let legal = battle.legal_choices(dex, bot);
        let other = find_choice(dex, &legal, alternative).ok_or("alternative is not legal")?;
        let origin = ReplayOrigin { code: code.into(), round };
        let cfg = RmConfig::default();
        let agent = origin.agent(dex, pool.clone(), cfg.clone(), 0, bot)?;
        let position = agent.to_position_spec(dex).ok_or("missing bot position")?;
        let foe = origin.agent(dex, pool.clone(), cfg, 0, 1-bot)?;
        let mut arms = vec![Arm { input: played.to_input(dex), label: "元の手".into() }];
        if other != played { arms.push(Arm { input: other.to_input(dex), label: "試す手".into() }); }
        Ok(Self {
            schema: SCHEMA.into(), label: format!("{}ターン目", battle.turn),
            info: if replay.open { Info::Open } else { Info::Blind },
            position, opponent_team: replay.teams[1-bot].clone(),
            opponent_picks: battle.sides[1-bot].party.iter().map(|&s| dex.species.key(battle.sides[1-bot].roster[s as usize].base_species).into()).collect(),
            opponent_position: foe.to_position_spec(dex), arms, exact_replay: Some(origin),
        })
    }
    pub fn parse(json: &str) -> Result<ForkSpec, String> {
        let spec: ForkSpec = serde_json::from_str(json).map_err(|e| format!("fork: {e}"))?;
        if spec.schema != SCHEMA {
            return Err(format!("fork: schema `{}` is not `{SCHEMA}`", spec.schema));
        }
        Ok(spec)
    }

    pub fn bot_side(&self) -> usize {
        self.position.side
    }

    pub fn opponent_side(&self) -> usize {
        1 - self.position.side
    }

    /// `seed` controls future chance; an exact replay preserves every other
    /// battle field, including hidden durations and queued mid-turn actions.
    pub fn battle(&self, dex: &Dex, seed: u64) -> Result<Battle, String> {
        if let Some(origin) = &self.exact_replay {
            let replay = Replay::decode(dex, &origin.code)?;
            if origin.round >= replay.rounds.len() || replay.bot_side != self.bot_side()
                || replay.open != (self.info == Info::Open) {
                return Err("replay metadata differs from fork".into());
            }
            let mut battle = replay.at(dex, origin.round)?;
            if battle.turn != self.position.turn || battle.turn == 0 { return Err("replay turn differs from fork".into()); }
            battle.reseed(seed);
            battle.set_log_enabled(false);
            battle.set_log_enabled(true);
            return Ok(battle);
        }
        self.check_static()?;
        let party = self.opponent_party()?;
        let pool = MetaPool { teams: Vec::new() };
        let mut b = synthesize_spec(dex, &self.position, &pool, Some(&self.opponent_team), seed)?;
        apply_party(&mut b, self.opponent_side(), &party);
        b.battle_mask = b.recompute_battle_mask(dex);
        b.set_log_enabled(true);
        Ok(b)
    }

    /// Each arm's choice at the fork, in `arms` order.
    pub fn arm_choices(&self, dex: &Dex, battle: &mut Battle) -> Result<Vec<SearchChoice>, String> {
        if self.arms.is_empty() {
            return Err("fork: no arms".into());
        }
        let legal = battle.legal_choices(dex, self.bot_side());
        let mut out = Vec::new();
        for arm in &self.arms {
            let c = find_choice(dex, &legal, &arm.input).ok_or_else(|| {
                let names: Vec<String> = legal.iter().map(|&c| action_input(dex, c)).collect();
                format!("fork: arm `{}` is not legal for the bot; legal: {}", arm.input, names.join(" / "))
            })?;
            if out.contains(&c) {
                return Err(format!("fork: arm `{}` repeats an earlier arm", arm.input));
            }
            out.push(c);
        }
        Ok(out)
    }

    /// Builds the battle once and resolves every arm: the whole validation a
    /// consumer needs before offering the fork.
    pub fn check(&self, dex: &Dex) -> Result<Vec<SearchChoice>, String> {
        let mut b = self.battle(dex, 0)?;
        self.arm_choices(dex, &mut b)
    }

    pub fn bot_agent(
        &self,
        dex: &Dex,
        pool: MetaPool,
        cfg: RmConfig,
        seed: u64,
    ) -> Result<ProtocolAgent, String> {
        if let Some(origin) = &self.exact_replay {
            return origin.agent(dex, pool, cfg, seed, self.bot_side());
        }
        let mut agent = ProtocolAgent::new(dex, self.bot_side(), pool, cfg, seed);
        if self.info == Info::Open {
            agent.pin_opponent(self.opponent_team.clone());
        }
        agent.set_position(dex, &self.position)?;
        Ok(agent)
    }

    /// The opponent as a ladder agent, under the same information policy.
    pub fn opponent_agent(
        &self,
        dex: &Dex,
        pool: MetaPool,
        cfg: RmConfig,
        seed: u64,
    ) -> Result<ProtocolAgent, String> {
        if let Some(origin) = &self.exact_replay {
            return origin.agent(dex, pool, cfg, seed, self.opponent_side());
        }
        let view = self
            .opponent_position
            .as_ref()
            .ok_or("fork: no opponent_position for a protocol opponent")?;
        let opp = self.opponent_side();
        if view.side != opp {
            return Err(format!("fork: opponent_position is for side {}, not {opp}", view.side));
        }
        let species = |sets: &[PokemonSet]| sets.iter().map(|s| toid(&s.species)).collect::<Vec<_>>();
        if species(&view.own_sets) != species(&self.opponent_team) {
            return Err("fork: opponent_position.own_sets differ from opponent_team".into());
        }
        let mut stated = view.own_party()?;
        let mut truth = self.opponent_party()?;
        stated.sort_unstable();
        truth.sort_unstable();
        if stated.iter().map(|&s| s as u8).collect::<Vec<_>>() != truth {
            return Err("fork: opponent_position's party differs from the opponent's picks".into());
        }
        let mut agent = ProtocolAgent::new(dex, opp, pool, cfg, seed);
        if self.info == Info::Open {
            agent.pin_opponent(self.position.own_sets.clone());
        }
        agent.set_position(dex, view)?;
        Ok(agent)
    }

    fn check_static(&self) -> Result<(), String> {
        self.position.check()?;
        if self.position.team_preview {
            return Err("fork: team preview cannot be forked".into());
        }
        let mons = &self.position.sides[self.opponent_side()].mons;
        if self.opponent_team.len() != mons.len() {
            return Err(format!(
                "fork: opponent_team has {} sets but the opponent previewed {}",
                self.opponent_team.len(),
                mons.len()
            ));
        }
        for (i, (set, mon)) in self.opponent_team.iter().zip(mons).enumerate() {
            if toid(&set.species) != toid(&mon.species) || set.level != mon.level {
                return Err(format!(
                    "fork: opponent_team[{i}] is {} L{}, preview slot {i} is {} L{}",
                    set.species, set.level, mon.species, mon.level
                ));
            }
        }
        Ok(())
    }

    /// Roster slots of the opponent's picks, active first then the other
    /// appeared picks, as `synthesize` orders them.
    fn opponent_party(&self) -> Result<Vec<u8>, String> {
        let side = &self.position.sides[self.opponent_side()];
        let mut party: Vec<u8> = Vec::new();
        if let Some(a) = side.active {
            party.push(a as u8);
        }
        for (i, m) in side.mons.iter().enumerate() {
            if m.appeared && side.active != Some(i) {
                party.push(i as u8);
            }
        }
        if self.opponent_picks.is_empty() {
            if party.len() < PICKS {
                return Err(format!(
                    "fork: only {} of the opponent's {PICKS} picks have appeared; list all of them in opponent_picks",
                    party.len()
                ));
            }
            return Ok(party);
        }
        if self.opponent_picks.len() != PICKS {
            return Err(format!("fork: opponent_picks must name {PICKS} species"));
        }
        let mut picks = Vec::new();
        for species in &self.opponent_picks {
            let slot = side
                .mons
                .iter()
                .position(|m| toid(&m.species) == toid(species))
                .ok_or_else(|| format!("fork: opponent pick `{species}` is not in the opponent's preview"))?;
            if picks.contains(&(slot as u8)) {
                return Err(format!("fork: opponent pick `{species}` is listed twice"));
            }
            picks.push(slot as u8);
        }
        if let Some(&missing) = party.iter().find(|s| !picks.contains(s)) {
            return Err(format!(
                "fork: `{}` has appeared but is missing from opponent_picks",
                side.mons[missing as usize].species
            ));
        }
        for slot in picks {
            if !party.contains(&slot) {
                party.push(slot);
            }
        }
        Ok(party)
    }
}

/// `input` as either rendering of a legal choice (`move hiddenpowerbug` and
/// the protocol's plain `move hiddenpower` both name the same move).
pub fn find_choice(dex: &Dex, legal: &[SearchChoice], input: &str) -> Option<SearchChoice> {
    let input = input.trim();
    legal
        .iter()
        .copied()
        .find(|&c| c.to_input(dex) == input || action_input(dex, c) == input)
}

/// A `ProtocolAgent` seated on one side of a forked battle.
pub struct ProtocolSeat {
    pub agent: ProtocolAgent,
    channel: PlayerChannel,
    iterations: u32,
    positioned: bool,
}

impl ProtocolSeat {
    /// `agent` must have been installed at the fork with `set_position`.
    pub fn new(agent: ProtocolAgent, iterations: u32) -> ProtocolSeat {
        let side = agent.side();
        ProtocolSeat { agent, channel: PlayerChannel::new(side), iterations, positioned: true }
    }

    pub fn choose(&mut self, battle: &mut Battle, dex: &Dex) -> Result<SearchChoice, String> {
        let frame = self.channel.frame(battle, dex)?;
        let at_fork = std::mem::take(&mut self.positioned) && frame.lines.is_empty();
        if !at_fork {
            for line in &frame.lines {
                self.agent.push_line(dex, line);
            }
            if !self.agent.on_request(dex, &frame.request.to_string())? {
                return Err("seat asked to choose on a wait request".into());
            }
        }
        self.agent.step(dex, self.iterations)?;
        let input = self.agent.best(dex).ok_or("seat produced no decision")?;
        let legal = battle.legal_choices(dex, self.agent.side());
        find_choice(dex, &legal, &input)
            .ok_or_else(|| format!("seat chose `{input}`, not legal at turn {}", battle.turn))
    }

    /// Consume the fork point without deciding it (an arm decides it).
    pub fn skip(&mut self, battle: &mut Battle, dex: &Dex) -> Result<(), String> {
        let frame = self.channel.frame(battle, dex)?;
        if !std::mem::take(&mut self.positioned) || !frame.lines.is_empty() {
            return Err("seat skipped a decision that is not the fork point".into());
        }
        Ok(())
    }
}

pub enum Seat {
    Protocol(ProtocolSeat),
    /// A full-information agent reading the battle directly.
    Engine(Box<dyn Agent>),
}

impl Seat {
    fn choose(
        &mut self,
        battle: &mut Battle,
        dex: &Dex,
        side: usize,
        legal: &[SearchChoice],
    ) -> Result<SearchChoice, String> {
        match self {
            Seat::Protocol(seat) => seat.choose(battle, dex),
            Seat::Engine(agent) => Ok(agent.choose(battle, dex, side, legal)),
        }
    }

    fn skip(&mut self, battle: &mut Battle, dex: &Dex) -> Result<(), String> {
        match self {
            Seat::Protocol(seat) => seat.skip(battle, dex),
            Seat::Engine(_) => Ok(()),
        }
    }
}

pub struct Playout {
    /// `None` when the step cap ended the game.
    pub outcome: Option<Outcome>,
    pub steps: u32,
}

/// Plays a forked battle to the end. The bot's first decision is `arm`; the
/// opponent's first decision is its seat's, made on the same state whatever
/// the arm, so trials sharing a seed share it.
pub fn play_out(
    battle: &mut Battle,
    dex: &Dex,
    bot_side: usize,
    arm: SearchChoice,
    seats: &mut [Seat; 2],
    max_steps: u32,
) -> Result<Playout, String> {
    let mut steps = 0;
    loop {
        if let Some(outcome) = battle.outcome() {
            return Ok(Playout { outcome: Some(outcome), steps });
        }
        if steps >= max_steps {
            return Ok(Playout { outcome: None, steps });
        }
        let mut choices = [None, None];
        for side in 0..2 {
            let legal = battle.legal_choices(dex, side);
            if legal.is_empty() {
                continue;
            }
            let pick = if steps == 0 && side == bot_side {
                seats[side].skip(battle, dex)?;
                arm
            } else {
                seats[side].choose(battle, dex, side, &legal)?
            };
            if !legal.contains(&pick) {
                return Err(format!("side {side} picked an illegal choice at turn {}", battle.turn));
            }
            choices[side] = Some(pick);
        }
        if steps == 0 && choices[bot_side] != Some(arm) {
            return Err("the bot owes no decision at the fork".into());
        }
        battle.apply_choices(dex, choices).map_err(|e| format!("apply: {e:?}"))?;
        steps += 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    /// The ladder agent from that side's recorded information set.
    Protocol,
    /// Full-information search over the true battle.
    Skuct,
}

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Policy::Protocol => "protocol",
            Policy::Skuct => "skuct",
        }
    }
}

/// A bot-vs-bot measurement of a fork. Trial `k` is fully determined by
/// `seed` and `k`, and every arm of a trial shares its seeds.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arena {
    pub bot: Policy,
    pub foe: Policy,
    pub iters: u32,
    pub foe_iters: u32,
    /// Exploration constant of protocol seats.
    pub c: f64,
    pub seed: u64,
    pub max_steps: u32,
}

impl Arena {
    /// Battle, bot and opponent seeds of `trial`.
    pub fn trial_seeds(&self, trial: usize) -> [u64; 3] {
        let mut rng = SplitMix64::new(self.seed);
        for _ in 0..trial * 3 {
            rng.next();
        }
        [rng.next(), rng.next(), rng.next()]
    }

    /// Arm `arm` of `trial` as one `tools/summarize-counterfactual.py` row;
    /// `score` is the bot's, `null` when the step cap ended the game.
    pub fn play(
        &self,
        dex: &Dex,
        fork: &ForkSpec,
        pool: &MetaPool,
        trial: usize,
        arm: usize,
    ) -> Result<Value, String> {
        let [battle_seed, bot_seed, foe_seed] = self.trial_seeds(trial);
        let bot_side = fork.bot_side();
        let mut battle = fork.battle(dex, battle_seed)?;
        let choice = *fork
            .arm_choices(dex, &mut battle)?
            .get(arm)
            .ok_or_else(|| format!("fork: no arm {arm}"))?;
        let bot = self.seat(dex, fork, pool, self.bot, true, self.iters, bot_seed)?;
        let foe = self.seat(dex, fork, pool, self.foe, false, self.foe_iters, foe_seed)?;
        let mut seats = if bot_side == 0 { [bot, foe] } else { [foe, bot] };
        let result = play_out(&mut battle, dex, bot_side, choice, &mut seats, self.max_steps)
            .map_err(|e| format!("trial {trial} arm {}: {e}", fork.arms[arm].input))?;
        let (score, outcome) = match (result.outcome, bot_side) {
            (None, _) => (None, "cap"),
            (Some(Outcome::Tie), _) => (Some(0.5), "tie"),
            (Some(Outcome::P1Win), 0) | (Some(Outcome::P2Win), 1) => (Some(1.0), "win"),
            _ => (Some(0.0), "loss"),
        };
        let (drift, projections) = match &seats[bot_side] {
            Seat::Protocol(seat) => (seat.agent.legality_drift, seat.agent.projections),
            Seat::Engine(_) => (0, 0),
        };
        let info = match fork.info {
            Info::Blind => "blind",
            Info::Open => "open",
        };
        Ok(json!({
            "fork": fork.label, "turn": fork.position.turn,
            "trial": trial, "seed": self.seed, "battle_seed": battle_seed,
            "action": fork.arms[arm].input, "arm_label": fork.arms[arm].label,
            "reply": "search", "tail": self.bot.name(),
            "policy": format!("fork/{info}/{}-vs-{}", self.bot.name(), self.foe.name()),
            "iters": self.iters, "foe_iters": self.foe_iters,
            "score": score, "outcome": outcome, "final_turn": battle.turn, "steps": result.steps,
            "legality_drift": drift, "projections": projections,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn seat(
        &self,
        dex: &Dex,
        fork: &ForkSpec,
        pool: &MetaPool,
        policy: Policy,
        bot: bool,
        iters: u32,
        seed: u64,
    ) -> Result<Seat, String> {
        Ok(match policy {
            Policy::Skuct => Seat::Engine(Box::new(RmAgent::new(
                RmConfig { iterations: iters, rule: SelRule::Ucb, ..RmConfig::default() },
                seed,
            ))),
            Policy::Protocol => {
                let cfg = RmConfig { rule: SelRule::Ucb, c: self.c, hp_buckets: 16, ..RmConfig::default() };
                let agent = if bot {
                    fork.bot_agent(dex, pool.clone(), cfg, seed)?
                } else {
                    fork.opponent_agent(dex, pool.clone(), cfg, seed)?
                };
                Seat::Protocol(ProtocolSeat::new(agent, iters))
            }
        })
    }
}
