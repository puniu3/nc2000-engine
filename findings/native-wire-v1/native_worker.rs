use std::io::{BufRead, Write};

use nc2000_bot::{Belief, Observer, RmConfig, SplitMix64};
use nc2000_bot::blind::BlindSearch;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::SelRule;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Roundtrip { battle: Battle },
    New { battle: Battle, side: usize, seed: u32 },
    Choose { battle: Battle },
}

pub struct NativeSession {
    side: usize,
    observer: Observer,
    belief: Belief,
    rng: SplitMix64,
}

impl NativeSession {
    pub fn new(dex: &Dex, pool: &nc2000_bot::preview::MetaPool, battle: &Battle, side: usize, seed: u32) -> Self {
        assert!(side < 2);
        let observer = Observer::new(battle, side);
        let belief = Belief::new(dex, pool, &observer);
        Self { side, observer, belief, rng: SplitMix64::new(seed as u64) }
    }

    pub fn choose(&mut self, dex: &Dex, battle: &Battle, iterations: u32) -> Result<Value, String> {
        let cfg = RmConfig { rule: SelRule::Ucb, c: 0.4, hp_buckets: 16, ..RmConfig::default() };
        self.observer.observe(battle, dex);
        self.belief.sync_checked(dex, &self.observer)?;
        let seed = self.rng.next();
        let mut search = BlindSearch::new(battle, dex, cfg, self.side, seed);
        search.step(dex, &self.belief, &self.observer, iterations);
        let action = search.best().ok_or("no action")?;
        let actions = search.actions().iter().zip(search.visits()).zip(search.means())
            .map(|((a, visits), mean)| json!({"input":a.to_input(dex), "visits":visits, "mean":mean}))
            .collect::<Vec<_>>();
        Ok(json!({"action": action.to_input(dex), "actions": actions,
            "iterations": search.iterations(), "search_seed": seed, "key": battle.state_key()}))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 { return Err("native_worker DEX POOL ITERATIONS".into()); }
    let dex = Dex::from_json(&std::fs::read_to_string(&args[1])?)?;
    let pool = load_meta_pool(std::path::Path::new(&args[2]));
    let iterations: u32 = args[3].parse()?;
    if iterations == 0 { return Err("positive iterations required".into()); }
    let mut game: Option<NativeSession> = None;
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let command: Command = serde_json::from_str(&line?)?;
        let response: Value = match command {
            Command::Roundtrip { battle } => json!({
                "battle": battle, "key": battle.state_key(),
                "debug": format!("{:?}", battle.clone()),
            }),
            Command::New { battle, side, seed } => {
                if side > 1 { return Err("invalid side".into()); }
                game = Some(NativeSession::new(&dex, &pool, &battle, side, seed));
                json!({"ready": true})
            }
            Command::Choose { battle } => {
                game.as_mut().ok_or("choose before new")?.choose(&dex, &battle, iterations)?
            }
        };
        serde_json::to_writer(&mut output, &response)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}
