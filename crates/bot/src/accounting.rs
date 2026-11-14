//! Leaf settlement modes for MCTS playout cutoffs and leaf evaluation.

/// How MCTS tree leaves settle rollout playouts and cutoff board evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeafSettlementMode {
    /// Legacy 8-turn `eps = 0.2` rollout + `eval::compact_eval_leaf`.
    #[default]
    LegacyRollout,
    /// Legacy 8-turn `eps = 0.2` rollout + `eval::compact_eval_leaf_blindspot` (balance-sheet
    /// setup/boost liability + frozen asset impairment + healer poison liability).
    BlindSpotLeaf,
    /// Surgical sign-error-guarded 8-turn `eps = 0.2` rollout + `eval::compact_eval_leaf`.
    BlindSpotRollout,
    /// Best-of-both-worlds ("いいとこ取り"): surgical sign-error-guarded 8-turn `CompactBattle`
    /// rollout + `eval::compact_eval_leaf_blindspot`.
    BlindSpotBoth,
}

impl LeafSettlementMode {
    #[inline]
    pub const fn uses_blindspot_leaf(self) -> bool {
        matches!(self, Self::BlindSpotLeaf | Self::BlindSpotBoth)
    }

    #[inline]
    pub const fn uses_blindspot_rollout(self) -> bool {
        matches!(self, Self::BlindSpotRollout | Self::BlindSpotBoth)
    }
}
