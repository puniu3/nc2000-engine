//! Embedded pure-Rust weights for `compact_az` (`P_net` policy prior + Siamese `V_net` value head).
//! Generated offline by `train_compact_az.py` (strictly mechanical features, zero species/team IDs).

pub const POLICY_WEIGHTS_LOADED: bool = false;
pub const VALUE_WEIGHTS_LOADED: bool = false;

pub const POLICY_CTX_DIM: usize = 28;
pub const POLICY_ACT_DIM: usize = 32;
pub const POLICY_H1: usize = 48;
pub const POLICY_H2: usize = 32;

pub const VALUE_SIDE_DIM: usize = 100;
pub const VALUE_H1: usize = 64;
pub const VALUE_H2: usize = 32;

// Policy head weights (Layer 1: ctx -> H1, act -> H1; Layer 2: [h1, h1_mean] -> H2 -> 1)
pub static P_W_CTX: [[f32; POLICY_CTX_DIM]; POLICY_H1] = [[0.0; POLICY_CTX_DIM]; POLICY_H1];
pub static P_W_ACT: [[f32; POLICY_ACT_DIM]; POLICY_H1] = [[0.0; POLICY_ACT_DIM]; POLICY_H1];
pub static P_B1: [f32; POLICY_H1] = [0.0; POLICY_H1];
pub static P_W2_SELF: [[f32; POLICY_H1]; POLICY_H2] = [[0.0; POLICY_H1]; POLICY_H2];
pub static P_W2_MEAN: [[f32; POLICY_H1]; POLICY_H2] = [[0.0; POLICY_H1]; POLICY_H2];
pub static P_B2: [f32; POLICY_H2] = [0.0; POLICY_H2];
pub static P_W_OUT: [f32; POLICY_H2] = [0.0; POLICY_H2];
pub const P_TAU: f32 = 1.0;

// Siamese value head weights (exact 0-ULP antisymmetric: z = w_out·(h0 - h1) + w_fg*l_fg + w_bs*l_bs)
pub static V_W1_SELF: [[f32; VALUE_SIDE_DIM]; VALUE_H1] = [[0.0; VALUE_SIDE_DIM]; VALUE_H1];
pub static V_W1_OPP: [[f32; VALUE_SIDE_DIM]; VALUE_H1] = [[0.0; VALUE_SIDE_DIM]; VALUE_H1];
pub static V_B1: [f32; VALUE_H1] = [0.0; VALUE_H1];
pub static V_W2_SELF: [[f32; VALUE_H1]; VALUE_H2] = [[0.0; VALUE_H1]; VALUE_H2];
pub static V_W2_OPP: [[f32; VALUE_H1]; VALUE_H2] = [[0.0; VALUE_H1]; VALUE_H2];
pub static V_B2: [f32; VALUE_H2] = [0.0; VALUE_H2];
pub static V_W_OUT: [f32; VALUE_H2] = [0.0; VALUE_H2];
pub const V_SKIP_FG: f32 = 0.55;
pub const V_SKIP_BS: f32 = 0.45;
