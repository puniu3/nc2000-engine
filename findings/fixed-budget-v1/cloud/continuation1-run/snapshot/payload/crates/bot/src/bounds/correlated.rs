use super::{node_strategies, normalized_policy, BoundSolver};
use nc2000_engine::state::Battle;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AffineCell {
    pub fixed: [f64; 2],
    pub pending_mass: f64,
    pub successors: Vec<(usize, f64)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootCorrelation {
    pub dimensions: [usize; 2],
    pub cell_bounds: Vec<[f64; 2]>,
    pub children: Vec<[f64; 2]>,
    pub cells: Vec<AffineCell>,
    pub policies: [Vec<f64>; 2],
    pub independent_loss: [f64; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrelatedLoss {
    pub cell_gap: f64,
    pub successor_gap: f64,
    pub max_loss: [f64; 2],
    pub shared_children: usize,
}

impl RootCorrelation {
    pub fn loss_bounds(&self) -> CorrelatedLoss {
        let [rows,cols]=self.dimensions;
        assert_eq!(self.cells.len(),rows*cols);
        assert_eq!(self.cell_bounds.len(),self.cells.len());
        for side in 0..2 {
            assert_eq!(self.policies[side].len(),self.dimensions[side]);
        }
        let policies=self.policies.each_ref().map(|p|normalized_policy(p).expect("valid policy weights"));
        for bounds in self.children.iter().chain(&self.cell_bounds) {
            assert!(bounds.iter().all(|v|v.is_finite()) && bounds[0]>=-1e-9 && bounds[1]<=1.+1e-9 && bounds[0]<=bounds[1]);
        }
        assert!(self.independent_loss.iter().all(|v|v.is_finite() && (0.0..=1.0).contains(v)));
        for cell in &self.cells {
            assert!(cell.fixed.iter().all(|v|v.is_finite()) && cell.fixed[0]>=0. && cell.fixed[0]<=cell.fixed[1]);
            assert!(cell.pending_mass.is_finite() && cell.pending_mass>=0.);
            assert!(cell.successors.iter().all(|&(key,mass)|key<self.children.len() && mass.is_finite() && mass>=0.));
            assert!(cell.fixed[1]+cell.pending_mass+cell.successors.iter().map(|(_,mass)|mass).sum::<f64>()<=1.+1e-8);
        }
        let edges: usize=self.cells.iter().map(|c|c.successors.len()).sum();
        let cell_pad=1e-12*(self.cells.len()+1) as f64;
        let successor_pad=1e-12*(edges+self.children.len()+self.cells.len()+1) as f64;
        let mut cell_gap=0.0f64;
        let mut successor_gap=0.0f64;
        for row in 0..rows {
            for col in 0..cols {
                let mut coefficients=vec![0.;rows*cols];
                for j in 0..cols { coefficients[row*cols+j]+=policies[1][j]; }
                for i in 0..rows { coefficients[i*cols+col]-=policies[0][i]; }
                let mut children=vec![0.;self.children.len()];
                let mut independent=0.;
                let mut combined=0.;
                for (index,&coefficient) in coefficients.iter().enumerate() {
                    if coefficient==0. { continue; }
                    independent+=coefficient*self.cell_bounds[index][usize::from(coefficient>0.)];
                    let cell=&self.cells[index];
                    combined+=coefficient*(cell.fixed[usize::from(coefficient>0.)]
                        +if coefficient>0. {cell.pending_mass} else {0.});
                    for &(child,mass) in &cell.successors { children[child]+=coefficient*mass; }
                }
                for (&coefficient,bounds) in children.iter().zip(&self.children) {
                    combined+=coefficient*bounds[usize::from(coefficient>0.)];
                }
                cell_gap=cell_gap.max(independent+cell_pad);
                successor_gap=successor_gap.max(combined+successor_pad);
            }
        }
        let mut incidence=vec![0;self.children.len()];
        for cell in &self.cells {
            let keys:BTreeSet<_>=cell.successors.iter().filter(|(_,mass)|*mass>0.).map(|(key,_)|*key).collect();
            for key in keys { incidence[key]+=1; }
        }
        let gap=cell_gap.min(successor_gap).min(1.);
        CorrelatedLoss {cell_gap:cell_gap.min(1.),successor_gap:successor_gap.min(1.),
            max_loss:self.independent_loss.map(|v|v.min(gap)),
            shared_children:incidence.into_iter().filter(|&n|n>1).count()}
    }
}

impl BoundSolver<'_> {
    pub fn root_correlation(&self, battle: &Battle) -> Option<RootCorrelation> {
        let node=self.nodes.get(&self.node_key(battle))?;
        let policies=node_strategies(node)?;
        Some(self.correlation_model(node,
            policies.clone().map(|p|p.actions.into_iter().map(|(_,mass)|mass).collect()),
            policies.map(|p|p.max_loss)))
    }

    pub fn root_correlation_for_policies(&self, battle: &Battle, policies: [Vec<f64>;2]) -> Option<RootCorrelation> {
        let node=self.nodes.get(&self.node_key(battle))?;
        let payoffs=self.root_payoffs(battle)?;
        let policies=[normalized_policy(&policies[0])?,normalized_policy(&policies[1])?];
        if (0..2).any(|side|policies[side].len()!=payoffs.actions[side].len()) {return None;}
        let independent_loss=[0,1].map(|side|payoffs.policy_loss(side,&policies[side]).hi);
        Some(self.correlation_model(node,policies,independent_loss))
    }

    fn correlation_model(&self, node: &super::Node, policies: [Vec<f64>;2], independent_loss: [f64;2]) -> RootCorrelation {
        let keys:BTreeSet<_>=node.cells.iter().flat_map(|c|c.resolved.iter().map(|(_,key)|*key)).collect();
        let index:HashMap<_,_>=keys.iter().enumerate().map(|(i,key)|(*key,i)).collect();
        RootCorrelation {
            dimensions:[node.acts0.len(),node.acts1.len()],
            cell_bounds:node.cells.iter().map(|c|[c.lo,c.hi]).collect(),
            children:keys.iter().map(|key|{let n=&self.nodes[key];[n.lo,n.hi]}).collect(),
            cells:node.cells.iter().map(|c|AffineCell {fixed:[c.fixed_lo,c.fixed_hi],pending_mass:c.pending_mass,
                successors:c.resolved.iter().map(|&(mass,key)|(index[&key],mass)).collect()}).collect(),
            policies, independent_loss,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::solve_matrix;

    #[test]
    fn shared_unknown_successor_cancels_without_value_closure() {
        let mut model=RootCorrelation {dimensions:[2,2],cell_bounds:vec![[0.,1.];4],children:vec![[0.,1.]],
            cells:vec![AffineCell {fixed:[0.,0.],pending_mass:0.,successors:vec![(0,1.)]};4],
            policies:[vec![0.2,0.8],vec![0.7,0.3]],independent_loss:[1.,1.]};
        let result=model.loss_bounds();
        assert!(result.cell_gap>0.5);
        assert!(result.successor_gap<1e-9);
        assert_eq!(result.shared_children,1);
        model.policies[0].iter_mut().for_each(|p|*p*=3.);
        assert!(model.loss_bounds().successor_gap<1e-9);
        model.cells[3].successors=vec![(1,1.)];
        model.children.push([0.,1.]);
        assert!(model.loss_bounds().successor_gap>0.1);
    }

    #[test]
    fn correlated_gap_encloses_every_vertex_with_pending_and_duplicate_edges() {
        for variant in 0..8 {
            let mut model=RootCorrelation {dimensions:[2,2],cell_bounds:Vec::new(),children:vec![[0.,1.];3],
                cells:(0..4).map(|i|AffineCell {fixed:[0.05*((i+variant)%3) as f64;2],pending_mass:0.1,
                    successors:vec![(0,0.2),(0,0.3),(1+(i+variant)%2,0.25)]}).collect(),
                policies:[vec![0.125*(variant+1) as f64,1.-0.125*(variant+1) as f64],vec![0.7,0.3]],
                independent_loss:[1.,1.]};
            model.cell_bounds=model.cells.iter().map(|c|[c.fixed[0],c.fixed[1]+0.85]).collect();
            let bounds=model.loss_bounds();
            for corner in 0..128 {
                let values:Vec<_>=model.cells.iter().enumerate().map(|(i,c)| {
                    c.fixed[0]+c.pending_mass*((corner>>(3+i))&1) as f64
                        +c.successors.iter().map(|&(key,mass)|mass*((corner>>key)&1) as f64).sum::<f64>()
                }).collect();
                let value=solve_matrix(&values,2,2).0;
                let floor=(0..2).map(|j|(0..2).map(|i|model.policies[0][i]*values[i*2+j]).sum::<f64>()).fold(1.,f64::min);
                let ceiling=(0..2).map(|i|(0..2).map(|j|model.policies[1][j]*values[i*2+j]).sum::<f64>()).fold(0.,f64::max);
                assert!(ceiling-floor<=bounds.successor_gap+1e-10);
                assert!(value-floor<=bounds.max_loss[0]+1e-10);
                assert!(ceiling-value<=bounds.max_loss[1]+1e-10);
            }
        }
    }
}
