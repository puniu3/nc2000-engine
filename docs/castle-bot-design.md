# Gym Leader's Castle (`PS.cgi`) Bot — Design Decisions

> **Status**: Adopted & implemented (`2026-11-25`). Keep this document aligned with [`crates/engine/src/rules.rs`](../crates/engine/src/rules.rs), [`crates/bot/src/`](../crates/bot/src/), and [`README.md`](../README.md).

---

## Decision 1: Ownership Boundary (Upstream vs. Operational Fork)

- **Downstream operational fork (`hm0615g/nc2000-engine`)**: Owns live Gym Leader's Castle (`PS.cgi`) transport, room management, Discord bot integration, and operational log conversion.
- **Upstream (`nc2000-engine`)**: Owns Castle ruleset engine correctness (`Battle`, `CompactBattle`, `Validator`, compact replay), the Core Engine SDK / WASM API, and **playing-strength optimization of the bot itself under Castle rules** (search, evaluation, belief/observation, endgame solvers, NN fine-tuning, and team/mixture tuning).

---

## Decision 2: Unlimited Turn Limit Unification (`L_TURN = 0`, Abolish 80T Bot Adaptation)

- **Default rule**: `CastleOverlayConfig::default()` and unspecified Castle ruleset aliases (`"castle"`, `"castle-pbs"`, `"pbs"`, `"pbs-nc2000"`, `"castle-free"`, `"pbs-free"`) resolve to **`CastleOverlayConfig::free()` (`turn_limit: None`, canonical ID `"castle-free"`)**.
- **Semantics of `None` (`L_TURN = 0`)**: **`0` is truly unlimited** — neither the engine nor the bot synthesizes any hidden turn limit (no 80-turn judge, no PS 1000-turn auto-tie, no 200/500-turn adjudication rule). Any self-play `max_turns` cap (`runner::effective_turn_cap` -> `GameResult::TurnCapped`) or search rollout depth is strictly an evaluation/search horizon, never a battle rule.
- **Bot 80-turn strategic adaptation abolished**:
  - Removed pre-limit 80-turn heuristic/DP adaptation across bot evaluation and subgame solvers:
    - [`crates/bot/src/eval.rs`](../crates/bot/src/eval.rs): removed `castle_turn_limit_proximity_diff` (the 16-turn proximity judge steering).
    - [`crates/bot/src/linear.rs`](../crates/bot/src/linear.rs): removed finite-horizon Castle `L_TURN` backward induction (`HorizonJudge` / `finite_turn_horizon`).
    - [`crates/bot/src/relay.rs`](../crates/bot/src/relay.rs): removed Castle `L_TURN` horizon cache-keying, leaf damping (`turn_scale`), and turn-limit rollout gate cutoff.
  - Do NOT add `L_TURN` proximity shaping to `CF-V_net` or `P_net`.
- **Engine-level `turn_limit: Option<u16>` retained (Option A)**:
  - [`crates/engine/src/rules.rs`](../crates/engine/src/rules.rs), [`crates/engine/src/battle/turn.rs`](../crates/engine/src/battle/turn.rs), and [`crates/engine/src/compact.rs`](../crates/engine/src/compact.rs) retain optional `turn_limit: Option<u16>` and exact `judge_castle_turn_limit` (accessible explicitly via `"castle-tournament"` or `"castle-custom-..."`) solely for faithful `PS.cgi` specification parity, Probe 12A conformance tests, and historical tournament log replay.
  - Exact terminal adjudication checks (`turn_limit_outcome_value` / `compact_turn_limit_outcome`) remain no-ops (`None`) whenever `turn_limit` is `None`.
- **Unlimited-turn safety invariants**:
  - `Battle::end_turn` saturates `turn` at `u16::MAX` (matching `CompactBattle`).
  - `ExactSolver::ival` ([`crates/bot/src/exact.rs`](../crates/bot/src/exact.rs)) relies on iterative-deepening horizons (`cfg.horizons`) for acyclicity and explicitly guards `b.turn == u16::MAX` under `turn_limit: None` so infinite cycles (switching loops, Ghost-vs-Struggle 0-recoil, Leftovers-vs-Struggle) always return safe `[0.0, 1.0]` brackets without hanging or fabricating a terminal value.
  - `BoundSolver` ([`crates/bot/src/bounds.rs`](../crates/bot/src/bounds.rs)) + [`crates/bot/src/stall.rs`](../crates/bot/src/stall.rs) treat `OneSidedHeal`/`TwoSidedHeal` strictly as scheduling priorities and terminate cleanly via budgets / `idle` limits without fabricating a turn-cap adjudication.

---

## Decision 3: Single-Spine (PS-Primary) + Thin Castle Delta Optimization

To avoid maintaining two full research lines, Castle playing-strength optimization piggybacks on the PS Stadium 2 primary line via minimal deltas:

| Layer | Primary Line (`PsStadium2`) | Castle (`castle-free`) Delta Policy |
| :--- | :--- | :--- |
| **Search & Solvers** | `CompactBattle` + `CfValuePuct` + `SteppedCfPreview` + Bayesian Root LP + `linear.rs` / `relay.rs` / `bounds.rs` | **100% shared architecture.** Rule mechanics (EOT `1P -> 2P`, Quick Claw, exact speed, items, masks, exact HP observation) flow automatically through `Ruleset` / `MatchTable` / `Observer` / `Belief`. No Castle-only search algorithm line. |
| **Neural Weights (`CF-V_net` / `P_net`)** | Architecture, features, and full training pipeline live on PS. | Benchmark PS weights directly under `castle-free` first. Only if measurable gap exists, apply a lightweight fine-tune from PS checkpoint (or single mixed PS+Castle checkpoint) without creating Castle-only architectures or asymmetric networks. |
| **Team Pool & Mixture** | Full evolutionary search & Nash certification on PS (currently `team-pool-v4`, already JP cartridge & Castle legal). | No separate Castle PSRO / evolution loop. Seed from the latest PS certified pool, re-measure top candidates under `castle-free` to re-solve Castle mixture weights, and make only targeted small-slot adjustments where Castle mechanics clearly shift value. |
| **Opponent Prior (`Belief`)** | `data/belief-pool-v3/belief-pool.json` (Castle-illegal sets filtered automatically via `data/castle-validation.json`). | Reuse `belief-pool-v3` as base; layer Castle-specific empirical stats only as thin opt-in priors. |
| **Verification** | Large-scale PS Elo & Nash gates. | Castle Gate A/B/C/D correctness suites + lightweight smoke duels on `policy-vN` promotions. |

---

## Empirical Rationale for `L_TURN = 0` (`2026-11-23` Investigation)

1. **`PS.cgi` software default**: All 13 live PBS rooms (`C`–`O` on `psense.lib.net`) present `<input name="L_TURN" size="2" value="0">(０で制限無し）` and `<input name="TLIMIT" size="2" value="0">(０で制限無し）`. No room enforces a server-side turn limit; players enter `L_TURN` manually per match.
2. **Community practice** (verified across `1,234` real PBS logs + official Castle manuals and tournament records):
   - **Casual / free play (`野試合`, Discord rooms `C`–`H`)**: Default is `L_TURN = 0` (*Gym Leaders' Castle Manual* §2.1). Across `711` unique logs in [`data/pbs-castle-logs/`](../data/pbs-castle-logs/README.md), `9` of `10` games reaching turn 80 continued past turn 80 (up to `134` turns) and finished by normal KO.
   - **Tournaments & weekly meetups (`翼門祭`, `突発対戦会`, BO3 practice)**: Manually set `L_TURN = 80` (`14 / 523` archived tournament games = `2.68%` hit turn-80 adjudication), which remains representable via explicit `"castle-tournament"` for historical replay / conformance verification.
3. **Live empirical verification of `L_TURN = 0` on `psense.lib.net` Room K (`2026-11-23`)**:
   - Directly probed `http://psense.lib.net/PBS/_/K/PS.cgi` with `L_TURN = 0, TLIMIT = 0` through Turn `1,050` alternating switches and normal KO cleanup at **Turn `1,076`** (`LLOG.html` = `351,043` bytes, `1,077` `<hr>` separators).
   - Passed `80`, `255`, `256`, `500`, `511`, `512`, `513`, `999`, `1000`, `1024`, and `1050` turns with **zero forced `DRAW`, zero hidden judge, and zero server timeout**.
4. **Mechanics admit true infinite games under `L_TURN = 0`**: Gen 2 / PBS has no 50-move rule or Endless Battle Clause — switching costs 0 PP, PBS Struggle against Ghost-types deals 0 damage and 0 recoil, and Leftovers can out-heal Struggle recoil.
