use std::sync::Arc;

use nc2000_bot::mcts::{perish_combo, perish_escape};
use nc2000_bot::preview::MetaPool;
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{Agent, BlindAgent, OpenAgent, RmConfig};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::json;

#[derive(Clone)]
pub struct Spec {
    blind: bool,
    iterations: u32,
    c: f64,
    perish: bool,
    combo: bool,
    residual_scale: f64,
    rollout_residual: bool,
}

impl Spec {
    pub fn parse(s: &str) -> Spec {
        let parts: Vec<&str> = s.split(':').collect();
        let blind = match parts[0] {
            "open" => false,
            "blind" => true,
            other => panic!("unsupported agent {other}"),
        };
        let mut nums = Vec::new();
        let shipped = RmConfig::default();
        let (mut perish, mut combo) = (shipped.rollout_perish, shipped.rollout_combo);
        let mut residual_scale = 1.0;
        let mut rollout_residual = shipped.rollout_residual;
        for p in &parts[1..] {
            match *p {
                "perish_escape" => (perish, combo) = (true, false),
                "perish_combo" => (perish, combo) = (false, true),
                "-perish_combo" | "-perish_escape" => (perish, combo) = (false, false),
                "res" => residual_scale = 1.0,
                "-res" => residual_scale = 0.0,
                "rres" => rollout_residual = true,
                "-rres" => rollout_residual = false,
                n => nums.push(n.parse::<f64>().unwrap_or_else(|_| panic!("bad field {n}"))),
            }
        }
        Spec {
            blind,
            iterations: nums.first().map_or(1000, |&n| n as u32),
            c: nums.get(1).copied().unwrap_or(1.0),
            perish,
            combo,
            residual_scale,
            rollout_residual,
        }
    }

    pub fn build(&self, seed: u64, pool: &Arc<MetaPool>) -> Box<dyn Agent> {
        let mut cfg = RmConfig {
            iterations: self.iterations,
            rule: SelRule::Ucb,
            c: self.c,
            rollout_perish: self.perish,
            rollout_combo: self.combo,
            rollout_residual: self.rollout_residual,
            ..Default::default()
        };
        if let nc2000_bot::mcts::Playout::Heavy { ref mut weights, .. } = cfg.playout {
            weights.residual_scale = self.residual_scale;
        }
        if self.blind {
            Box::new(BlindAgent::new(cfg, pool.clone(), None, seed))
        } else {
            Box::new(OpenAgent::new(cfg, None, seed))
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct Tally {
    pub sing: u32,
    pub sing_foe_stuck: u32,
    pub trap: u32,
    pub c1_escape: u32,
    pub c1_stay: u32,
    pub c1_trapped: u32,
    pub perish_faint: u32,
    /// Decisions with a Mean Look + Perish Song user active against a foe
    /// that is not yet trapped, and what was chosen there. `armed` and
    /// `loaded` both exclude the user's last mon (the song fails there) and
    /// a benchless foe.
    pub armed: u32,
    pub armed_trap: u32,
    pub armed_sing: u32,
    pub armed_switch: u32,
    /// Decisions with the foe already trapped and not yet counting down.
    pub loaded: u32,
    pub loaded_sing: u32,
    /// Games in which a Mean Look + Perish Song user took the field.
    pub fielded: u32,
}

#[allow(dead_code)]
impl Tally {
    pub fn add(&mut self, o: &Tally) {
        self.sing += o.sing;
        self.sing_foe_stuck += o.sing_foe_stuck;
        self.trap += o.trap;
        self.c1_escape += o.c1_escape;
        self.c1_stay += o.c1_stay;
        self.c1_trapped += o.c1_trapped;
        self.perish_faint += o.perish_faint;
        self.armed += o.armed;
        self.armed_trap += o.armed_trap;
        self.armed_sing += o.armed_sing;
        self.armed_switch += o.armed_switch;
        self.loaded += o.loaded;
        self.loaded_sing += o.loaded_sing;
        self.fielded += o.fielded;
    }

    pub fn json(&self) -> serde_json::Value {
        json!({"sing":self.sing,"sing_foe_stuck":self.sing_foe_stuck,"trap":self.trap,
            "c1_escape":self.c1_escape,"c1_stay":self.c1_stay,"c1_trapped":self.c1_trapped,
            "perish_faint":self.perish_faint,
            "armed":self.armed,"armed_trap":self.armed_trap,"armed_sing":self.armed_sing,
            "armed_switch":self.armed_switch,"loaded":self.loaded,"loaded_sing":self.loaded_sing,
            "fielded":self.fielded})
    }
}

fn perish_count(b: &Battle, dex: &Dex, s: usize) -> Option<u8> {
    let id = b.active_id(s)?;
    let cond = dex.conds_id("perishsong")?;
    b.poke(id).volatile(cond)?.duration.map(|d| d as u8)
}

pub fn norm(m: &str) -> String {
    m.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Plays one game, returning p1's score, turns and each side's tally.
pub fn play(
    dex: &Dex,
    b: &mut Battle,
    agents: [&mut dyn Agent; 2],
    max_turns: u16,
) -> (f64, u16, [Tally; 2]) {
    let mut t = [Tally::default(); 2];
    let mut fielded = [false; 2];
    loop {
        if let Some(o) = b.outcome() {
            let p1 = score_of(o);
            for s in 0..2 {
                t[s].fielded = u32::from(fielded[s]);
            }
            return (p1, b.turn, t);
        }
        if b.turn > max_turns {
            for s in 0..2 {
                t[s].fielded = u32::from(fielded[s]);
            }
            return (0.5, b.turn, t);
        }
        let legal: [Vec<SearchChoice>; 2] = std::array::from_fn(|s| b.legal_choices(dex, s));
        let mut picks = [None, None];
        for s in 0..2 {
            if !legal[s].is_empty() {
                picks[s] = Some(agents[s].choose(b, dex, s, &legal[s]));
            }
        }
        let mut dying = [None, None];
        for s in 0..2 {
            let Some(pick) = picks[s] else { continue };
            let moving = legal[s].iter().any(|c| matches!(c, SearchChoice::Move(_)));
            let has = |key: &str| {
                legal[s]
                    .iter()
                    .any(|c| matches!(c, SearchChoice::Move(id) if dex.moves.key(*id) == key))
            };
            if moving && has("perishsong") && (has("meanlook") || has("spiderweb")) {
                fielded[s] = true;
                let foe = b.active_id(1 - s).map(|id| b.poke(id));
                let foe_counting = perish_count(b, dex, 1 - s).is_some();
                let foe_trapped = foe.is_some_and(|p| p.trapped);
                let chosen = match pick {
                    SearchChoice::Move(id) => dex.moves.key(id).to_string(),
                    _ => String::new(),
                };
                let song_works = b.sides[s].pokemon_left > 1;
                if foe.is_some() && !foe_counting && song_works && b.sides[1 - s].pokemon_left > 1 {
                    if !foe_trapped {
                        t[s].armed += 1;
                        t[s].armed_trap += u32::from(chosen == "meanlook" || chosen == "spiderweb");
                        t[s].armed_sing += u32::from(chosen == "perishsong");
                        t[s].armed_switch += u32::from(matches!(pick, SearchChoice::Switch(_)));
                    } else {
                        t[s].loaded += 1;
                        t[s].loaded_sing += u32::from(chosen == "perishsong");
                    }
                }
            }
            let can_switch = |side: usize| {
                legal[side]
                    .iter()
                    .any(|c| matches!(c, SearchChoice::Switch(_)))
            };
            if let SearchChoice::Move(id) = pick {
                match norm(dex.moves.key(id)).as_str() {
                    "perishsong" => {
                        t[s].sing += 1;
                        let foe_moving = legal[1 - s]
                            .iter()
                            .any(|c| matches!(c, SearchChoice::Move(_)));
                        if foe_moving && !can_switch(1 - s) {
                            t[s].sing_foe_stuck += 1;
                        }
                    }
                    "meanlook" | "spiderweb" => t[s].trap += 1,
                    _ => {}
                }
            }
            if moving && perish_count(b, dex, s) == Some(1) {
                dying[s] = b.active_id(s);
                if !can_switch(s) {
                    t[s].c1_trapped += 1;
                } else if matches!(pick, SearchChoice::Switch(_)) {
                    t[s].c1_escape += 1;
                } else {
                    t[s].c1_stay += 1;
                }
            }
        }
        if std::env::var_os("COMBO_TRACE").is_some() {
            let side_str = |s: usize| {
                let Some(id) = b.active_id(s) else {
                    return "-".to_string();
                };
                let p = b.poke(id);
                let pick = picks[s].map_or("-".to_string(), |c| c.to_input(dex));
                format!(
                    "{} {}/{} perish={:?} trapped={} left={} -> {pick}",
                    dex.species.key(p.species),
                    p.hp,
                    p.maxhp,
                    perish_count(b, dex, s),
                    p.trapped,
                    b.sides[s].pokemon_left
                )
            };
            eprintln!("T{:>3} | {} || {}", b.turn, side_str(0), side_str(1));
        }
        b.apply_choices(dex, picks).unwrap();
        for s in 0..2 {
            if let Some(id) = dying[s] {
                if b.poke(id).hp <= 0 {
                    t[s].perish_faint += 1;
                }
            }
        }
    }
}

/// Whether a set carries the whole Perish trap.
#[allow(dead_code)]
pub fn is_trapper(set: &PokemonSet) -> bool {
    let moves: Vec<String> = set.moves.iter().map(|m| norm(m)).collect();
    moves.iter().any(|m| m == "perishsong")
        && moves.iter().any(|m| m == "meanlook" || m == "spiderweb")
}

/// Wraps an agent for the gauntlet roles. `lead` = the team's first trapper
/// (1-based display position): the team preview pick is rewritten to lead
/// with it, keeping the inner agent's other two picks in order.
/// `specialist` additionally replaces every in-battle choice with the Perish
/// rollout rules' choice wherever they have one: a fixed opponent that
/// traps, sings, escapes at count 1 and phazes out of a trap, like the
/// human in battle 4069.
pub struct Forced {
    pub inner: Box<dyn Agent>,
    pub lead: Option<u8>,
    pub specialist: bool,
}

impl Agent for Forced {
    fn name(&self) -> String {
        format!("forced:{}", self.inner.name())
    }

    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let pick = self.inner.choose(b, dex, side, choices);
        if let (SearchChoice::Team(t), Some(lead)) = (pick, self.lead) {
            // Legal picks are level-capped and canonically ordered, so take
            // the legal lead-`lead` pick sharing most mons with the inner one.
            let shared = |c: &SearchChoice| match c {
                SearchChoice::Team(u) => u.iter().filter(|p| t.contains(p)).count(),
                _ => 0,
            };
            return choices
                .iter()
                .copied()
                .filter(|c| matches!(c, SearchChoice::Team(u) if u[0] == lead))
                .max_by_key(|c| shared(c))
                .expect("no legal pick leads with the trapper");
        }
        if self.specialist {
            if let Some(c) =
                perish_escape(b, dex, side, choices).or_else(|| perish_combo(b, dex, side, choices))
            {
                return c;
            }
        }
        pick
    }
}

pub fn score_of(o: Outcome) -> f64 {
    match o {
        Outcome::P1Win => 1.0,
        Outcome::P2Win => 0.0,
        Outcome::Tie => 0.5,
    }
}
