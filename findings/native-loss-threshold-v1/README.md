# Reproduce the loss-threshold pilot

Implementation is commit 2a63a63. Prepare a clean isolated checkout with tools/native-wire-prepare, then copy native_loss_threshold.rs into its bot examples and build without additional features. The engine adapter changes serialization only; ordinary main engine state serialization remains unchanged.

Generate each arm protocol by taking protocol.json without arms, overlaying its named arm settings, and setting arm to the name. Invoke the helper with PROTOCOL ROOTS DEX OUTPUT. Inputs are the frozen native-endgame-census-v1 snapshots and native-deployed-duel-v1 dex. Outputs are created exclusively and retain all unresolved roots.

The three positive controls override only the analysis target, preserving the archived bot-selected action. Negative controls and actual_decisions use the actual selected action. Run `python3 tools/native-loss-threshold-audit tmp/native-loss-threshold-v1 tmp/native-endgame-census-v1/roots.jsonl` to check all arms. The native module tests are run separately with cargo test --release -p nc2000-bot --lib bounds:: -- --test-threads=4. All archives use gzip mtime zero.
