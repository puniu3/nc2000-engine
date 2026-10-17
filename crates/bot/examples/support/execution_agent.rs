//! Research adapter for the live WASM decision lifecycle, without preview
//! tables. Unlike BlindAgent, each observe draws a fresh search seed.
use std::{cell::RefCell, rc::Rc, sync::Arc};
use nc2000_bot::{Agent, Belief, BlindSearch, Observer, RmConfig, SplitMix64};
use nc2000_bot::{decision::{DecisionPolicy, DecisionSampler, PolicyMode}, preview::MetaPool, smmcts::SelRule};
use nc2000_engine::{battle::SearchChoice, dex::Dex, state::Battle};

#[derive(Clone, Default, serde::Serialize)]
pub struct Counts {
    pub decisions: u64,
    pub mixed: u64,
    pub fallback: u64,
    pub differs_from_legacy: u64,
    pub entropy_sum: f64,
    pub iterations: u64,
}
#[derive(Clone, Default, serde::Serialize)]
pub struct Stats { pub preview: Counts, pub battle: Counts }

pub struct ExecutionAgent {
    pool: Arc<MetaPool>,
    mode: PolicyMode,
    budget: u32,
    rng: SplitMix64,
    sampler: DecisionSampler<SearchChoice>,
    state: Option<(Observer, Belief)>,
    pub stats: Rc<RefCell<Stats>>,
    pub last_policy: Option<DecisionPolicy<SearchChoice>>,
    pub last_visits: Vec<(SearchChoice, u32, f64)>,
}
impl ExecutionAgent {
    pub fn new(pool: Arc<MetaPool>, mode: PolicyMode, budget: u32, seed: u32) -> Self {
        Self { pool, mode, budget, rng: SplitMix64::new(u64::from(seed)),
            sampler: DecisionSampler::new(u64::from(seed)), state: None,
            stats: Rc::new(RefCell::new(Stats::default())), last_policy: None, last_visits: vec![] }
    }
    pub fn policy(&mut self, battle: &Battle, dex: &Dex, side: usize) -> (DecisionPolicy<SearchChoice>, SearchChoice) {
        let (obs, belief) = self.state.get_or_insert_with(|| {
            let obs = Observer::new(battle, side);
            let belief = Belief::new(dex, &self.pool, &obs);
            (obs, belief)
        });
        obs.observe(battle, dex);
        belief.sync_checked(dex, obs).unwrap();
        let mut search = BlindSearch::new(battle, dex, RmConfig {
            rule: SelRule::Ucb, c: 0.4, hp_buckets: 16, ..Default::default()
        }, side, self.rng.next());
        search.step(dex, belief, obs, self.budget);
        let policy = search.decision_policy(self.mode).unwrap();
        self.last_policy = Some(policy.clone());
        self.last_visits = search.actions().iter().copied().zip(search.visits().iter().copied()).zip(search.means())
            .map(|((a,n),m)|(a,n,m)).collect();
        let mut stats = self.stats.borrow_mut();
        let stats = if search.is_preview() { &mut stats.preview } else { &mut stats.battle };
        stats.decisions += 1;
        stats.iterations += u64::from(search.iterations());
        stats.mixed += u64::from(policy.actions().len() > 1);
        stats.fallback += u64::from(policy.source().ends_with("-fallback"));
        stats.entropy_sum += policy.actions().iter().map(|r| -r.probability * r.probability.ln()).sum::<f64>();
        (policy, search.best().unwrap())
    }
}
impl Agent for ExecutionAgent {
    fn name(&self) -> String { format!("execution:{:?}:{}", self.mode, self.budget) }
    fn choose(&mut self, battle: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> SearchChoice {
        let (policy, legacy) = self.policy(battle, dex, side);
        self.sampler.reset();
        let action = self.sampler.commit(policy).input;
        assert!(choices.contains(&action));
        let mut stats = self.stats.borrow_mut();
        let counts = if matches!(action, SearchChoice::Team(_)) { &mut stats.preview } else { &mut stats.battle };
        counts.differs_from_legacy += u64::from(action != legacy);
        action
    }
    fn root_policy(&mut self, battle: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> Vec<f64> {
        let (policy, _) = self.policy(battle, dex, side);
        let p: Vec<f64> = choices.iter().map(|a| policy.actions().iter()
            .find(|r| r.input == *a).map_or(0., |r| r.probability)).collect();
        assert!((p.iter().sum::<f64>() - 1.).abs() < 1e-10);
        p
    }
}

/// Independent persistent policy replicas; their seeds are never the live
/// defender's seed. Oracle queries expose probabilities, not sampled actions.
pub struct Ensemble(pub Vec<ExecutionAgent>);
impl Agent for Ensemble {
    fn name(&self) -> String { "independent-policy-ensemble".into() }
    fn choose(&mut self, _: &Battle, _: &Dex, _: usize, _: &[SearchChoice]) -> SearchChoice { panic!("probability model only") }
    fn root_policy(&mut self, battle: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> Vec<f64> {
        let mut p = vec![0.; choices.len()];
        let n = self.0.len() as f64;
        for model in &mut self.0 {
            for (a, b) in p.iter_mut().zip(model.root_policy(battle, dex, side, choices)) { *a += b / n; }
        }
        p
    }
}
