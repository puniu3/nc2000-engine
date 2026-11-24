//! Leaf settlement modes for MCTS playout cutoffs and leaf evaluation.

/// How MCTS tree leaves settle rollout playouts and cutoff board evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeafSettlementMode {
    /// Legacy 8-turn `eps = 0.2` rollout + `eval::compact_eval_leaf` (used at TeamPreview & M6 callers).
    #[default]
    LegacyRollout,
    /// Shipped in-battle default: Setup-Relay simultaneous-move Bellman leaf lift +
    /// equilibrium-gated rollout continuation on top of `eval::compact_eval_leaf_blindspot`.
    BlindSpotLeaf,
    /// Ablation baseline without the Setup-Relay Bellman solver (`-relay`): 8-turn
    /// `eps = 0.2` rollout + static `eval::compact_eval_leaf_blindspot_norelay`.
    BlindSpotLeafNoRelay,
    /// Certified production default: 144-dim Siamese `CF-V_net` v2 0-rollout value leaf
    /// + `P_net` PUCT interior selection + pure MC (`lam = 0`) backup + `n0+n1 <= 3`
    /// `8r` `BlindSpotLeaf` Hybrid Endgame Guard (plus `SteppedCfPreview` at Team Preview).
    CfValuePuct,
}

impl LeafSettlementMode {
    #[inline]
    pub const fn uses_blindspot_leaf(self) -> bool {
        matches!(
            self,
            Self::BlindSpotLeaf | Self::BlindSpotLeafNoRelay | Self::CfValuePuct
        )
    }

    #[inline]
    pub const fn uses_relay_bellman(self) -> bool {
        matches!(self, Self::BlindSpotLeaf | Self::CfValuePuct)
    }

    #[inline]
    pub const fn uses_cf_value_puct(self) -> bool {
        matches!(self, Self::CfValuePuct)
    }
}
