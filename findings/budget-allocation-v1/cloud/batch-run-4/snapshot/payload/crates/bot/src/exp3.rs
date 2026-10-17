use crate::rng::SplitMix64;

pub(crate) struct Exp3Ix {
    losses: Vec<f64>,
    rounds: u32,
}

pub(crate) struct Draw {
    pub action: usize,
    probability: f64,
    gamma: f64,
}

impl Exp3Ix {
    pub fn new(actions: usize) -> Self {
        assert!(actions > 1);
        Self { losses: vec![0.; actions], rounds: 0 }
    }

    pub fn sample(&mut self, rng: &mut SplitMix64) -> Draw {
        self.rounds += 1;
        let k = self.losses.len() as f64;
        let eta = (k.ln() / (k * f64::from(self.rounds))).sqrt();
        let minimum = self.losses.iter().copied().fold(f64::INFINITY, f64::min);
        let weights: Vec<_> = self.losses.iter().map(|loss| (-eta*(loss-minimum)).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut remaining = rng.next_f64()*total;
        let mut action = weights.iter().rposition(|&w| w > 0.).unwrap();
        for (i, &weight) in weights.iter().enumerate() {
            remaining -= weight;
            if remaining < 0. { action = i; break; }
        }
        Draw { action, probability: weights[action]/total, gamma: eta/2. }
    }

    pub fn observe(&mut self, draw: Draw, loss: f64) {
        assert!((0. ..=1.).contains(&loss));
        self.losses[draw.action] += loss/(draw.probability+draw.gamma);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simultaneous_bandits_approach_asymmetric_and_cyclic_game_values() {
        for (matrix, rows, cols) in [
            (vec![0.91461181640625,0.,0.91461181640625,0.,0.13065883091517858,0.91461181640625],2,3),
            (vec![0.5,0.,1.,1.,0.5,0.,0.,1.,0.5],3,3),
        ] {
            for seed in 0..4 {
                let mut rng = SplitMix64::new(seed);
                let mut a = Exp3Ix::new(rows);
                let mut b = Exp3Ix::new(cols);
                let mut visits = [vec![0u32;rows],vec![0u32;cols]];
                for _ in 0..200000 {
                    let x = a.sample(&mut rng);
                    let y = b.sample(&mut rng);
                    visits[0][x.action] += 1;
                    visits[1][y.action] += 1;
                    let value = matrix[x.action*cols+y.action];
                    a.observe(x,1.-value);
                    b.observe(y,value);
                }
                let lower = (0..cols).map(|j| (0..rows).map(|i|
                    f64::from(visits[0][i])/200000.*matrix[i*cols+j]).sum::<f64>()).fold(1.,f64::min);
                let upper = (0..rows).map(|i| (0..cols).map(|j|
                    f64::from(visits[1][j])/200000.*matrix[i*cols+j]).sum::<f64>()).fold(0.,f64::max);
                assert!(upper-lower < 0.035, "seed={seed} gap={}",upper-lower);
            }
        }
    }
}
