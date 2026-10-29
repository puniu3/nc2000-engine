# Residual Value & Rollout Calibration NN Handoff

Status: **READY FOR NEXT SESSION** (Prerequisites — Bayesian soft-likelihood fallback `FallbackPolicy::Bayesian`, supplemental prior `belief_supplement_v1.json`, and `BlindSearch::det_cache` fallback key caching — shipped and deployed on 2026-10-29 in commits `aa794f85` / `d8096e0d` / `dc2824ab`).

## 1. Context & Why Previous NN/Rollout Attempts Failed

Before implementing any neural network or rollout modification, read the prior experimental records (`docs/GAME-SOLVING-PLAN.md`, `memory/nn-learning-history.md`, and `memory/m16c-rollout-null.md`):

1. **Why standalone Value NNs plateaued (`docs/GAME-SOLVING-PLAN.md`, 2026-10-02 S1 CUT)**:
   - S1 trained a full-replacement value network (`bot::value`, `15.0 µs/eval` even after mon-encoding caching) to replace the 8-turn rollout + `eval01`. It reached `0.495 ± 0.042` at matched wall-clock time against `skuct:27000` and plateaued.
   - **Root cause**: `CompactBattle` (`192B`, 3 cache lines) + `eval_leaf_compact` (`crates/bot/src/eval.rs`) + `linear.rs` already compute exact Gen 2 damage rolls, KO-race hazards, and 1v1 Bellman values in sub-microsecond time, and 70% of 8-turn rollouts reach real terminal or KO/status transitions. Asking a standalone embedding MLP (`15 µs`) to re-learn Gen 2 damage mechanics from raw species/move IDs spent 56% of search time on float matrix multiplies while losing exact KO-threshold sharpness.
2. **Why hand-written Rollout heuristics were null (`m16c-rollout-null.md`, 2026-09-28)**:
   - Hand-tuned status/switch rollout rules (`0.505 ± 0.019` combined, `0.466 ± 0.032` switch-only) failed because static `if-then` thresholds (`ROLLOUT_SWITCH_TRIGGER`, fixed `compact_status_pseudo_score` constants in `crates/bot/src/mcts.rs`) ignore board context (boosts, bench receivers, incoming KO risk), while calling full leaf evaluation inside rollout steps is too slow.

## 2. Core Design Principles (The `~/mtg_ai` Residual Calibration Approach)

To avoid repeating the S1 and M16c failure modes, the next session must adhere to four hard invariants:

1. **Never replace `CompactBattle` rollouts or `eval_leaf_compact` / `linear.rs`**:
   - The analytical evaluator `eval_leaf_compact` (`crates/bot/src/eval.rs`) remains the backbone. The NN only predicts the **logit residual** $\Delta_\theta(s)$ where the analytical evaluator's linear-additive assumptions break down (non-linear interactions between active stat stages, volatiles like `Substitute`, and bench synergy).
2. **Zero hand-written per-combo / per-move `if-then` rules**:
   - Do **not** add move-name branches (`if move == "batonpass"` or `if move == "agility"`) to `eval.rs` or `mcts.rs`.
   - All features fed into the residual network must be **continuous physical quantities** already computed by `eval_leaf_compact` / `CompactBattle` or **generic numerical move/effect properties** read directly from `Dex` (`boosts`, `self_switch`, `force_switch`, `volatile_status`, `status`, `heal`, `priority`).
3. **Zero-init identity & L1-cache-resident execution (`< 50 ns`)**:
   - With output weights initialized to `0.0`, the residual network is bit-identical to the shipped bot ($\Delta_\theta(s) \equiv 0$).
   - The network must operate on stack-allocated `[f32; N]` arrays inside `eval_leaf_compact` / `compact_greedy_pick` with zero heap allocations and `< 3 KB` total parameter footprint so `perf_probe` wall-clock time does not regress.
4. **Never fragment `BlindSearch::table` or `det_cache`**:
   - Keep `state_key` and `det_cache` invariants intact (lesson from 2026-10-29: splitting root determinizations across multiple uncached loadouts doubles `det_60k` time and dilutes MCTS tree depth).

---

## 3. Architecture Part A: Leaf Evaluation Residual NN (`eval_residual`)

### 3.1 Formulation

In `crates/bot/src/eval.rs` (`eval_leaf_compact`, around line 880):
$$L_{\text{base}}(s) = \sum_i w_i \phi_i(s), \qquad V_{\text{base}}(s) = \sigma(L_{\text{base}}(s))$$
connect a tiny 2-layer residual MLP $\Delta_\theta(x(s))$:
$$V(s) = \sigma\!\Big(L_{\text{base}}(s) + \Delta_\theta\big(x(s)\big)\Big), \qquad \Delta_\theta(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2$$
 Enforce **exact side-antisymmetry** by evaluating the perspective difference or antisymmetric pair:
$$\Delta_{\text{sym}}(s) = \Delta_\theta(x(s, \text{side}=0)) - \Delta_\theta(x(s, \text{side}=1))$$
so that $V(\text{swap}(s)) = 1 - V(s)$ holds by construction with zero bias drift.

### 3.2 Feature Vector $x(s, \text{side}) \in \mathbb{R}^{32}$ (Zero Extra Engine Calls)

Extract $x(s, \text{side})$ exclusively from quantities **already in registers** inside `eval_leaf_compact` (`crates/bot/src/eval.rs`) plus a precomputed per-`MoveId` static descriptor table in `CompactTable`:

- **Active matchup & tempo (10 floats, already computed in `eval_leaf_compact`)**:
  - Own/foe active HP fraction (`hp / maxhp`).
  - `best_eff[0]`, `best_eff[1]` (best expected single-turn damage fraction each active deals to the other).
  - `ko_hazard[0]`, `ko_hazard[1]` (`race` term inputs: probability/rate of 1HKO/2HKO).
  - Speed advantage indicator (`+1.0` faster, `0.0` tie, `-1.0` slower) and active item indicators (Leftovers/Thick Club/Miracle Berry via existing item table).
- **Active state & volatiles (10 floats)**:
  - Own/foe normalized boost stages: `atk / 6.0`, `def / 6.0`, `spa / 6.0`, `spd / 6.0`, `spe / 6.0`.
  - Own/foe `Substitute` HP fraction, status flags (sleep, paralysis, poison/burn), and `Perish` / `LeechSeed` / `Encore` volatile flags.
- **Generic move-effect capability scalars on active + alive bench (12 floats, from `Dex::move_static` numerical fields — no move-name `if-then`)**:
  - Precompute a tiny `MoveCapability` struct per `MoveId` at `CompactTable` init directly from `Dex` fields:
    - `self_switch: f32` (`1.0` if move has `self_switch` effect, i.e. passes state/switches out)
    - `force_switch: f32` (`1.0` if move forces opponent switch)
    - `atk_boost: f32`, `spe_boost: f32`, `spa_boost: f32`, `def_boost: f32` (from `move_static.boosts`)
    - `sleep_acc: f32`, `sub_flag: f32`, `heal_flag: f32`
  - Aggregate max/sum over active mon's usable moves and alive bench mons' `best_eff` / physical vs special offense ceiling (`max_atk_stat`, `max_spa_stat`, `alive_count`, `bench_hp_sum`).

Because the network sees `(active_spe_boost, active_has_sub, active_has_self_switch_move, bench_max_atk_stat, foe_has_force_switch_move)` as continuous numbers, its hidden layer (`32 -> 16 -> 1`, **545 floats = 2.1 KB**) naturally learns the non-linear value of setup + `self_switch` + high-attack bench receiver (and its counter by `force_switch`) without a single combo-specific `if-then`.

### 3.3 Offline Calibration Pipeline (Residual Aggregation)

1. **Data collection**:
   - Sample states $s_k$ from self-play, `combo_gauntlet`, and human/arena corpus battles at decision points and rollout leaves.
   - Compute high-confidence teacher values $V_{\text{teacher}}(s_k) \in [0, 1]$ using deep search (`blind:27000` / `skuct:27000` or multi-rollout backup) and game outcomes ($z_k \in \{0, 0.5, 1\}$).
2. **Residual aggregation diagnostics (before training)**:
   - Bucket the signed error $e_k = V_{\text{teacher}}(s_k) - V_{\text{base}}(s_k)$ by generic state coordinates (`sum_boosts > 0`, `has_self_switch`, `has_substitute`, `alive_diff`, `ko_hazard`) to verify where `eval_leaf_compact` systematically under/over-estimates.
3. **Training objective**:
   - Fit $\theta$ by minimizing penalized binary cross-entropy / Huber loss on the residual logit:
     $$\mathcal{L}(\theta) = \frac{1}{N} \sum_{k=1}^N \ell\!\Big(V_{\text{teacher}}(s_k),\; \sigma\big(L_{\text{base}}(s_k) + \Delta_{\text{sym}}(s_k; \theta)\big)\Big) + \lambda \|\theta\|_2^2$$
   - Export quantized/raw `f32` weights into a static Rust array or embedded JSON artifact validated at build time.

---

## 4. Architecture Part B: Rollout Action-Residual (`rollout_residual`)

### 4.1 Why Connect a Residual to Rollout?

In `crates/bot/src/mcts.rs` (`compact_greedy_pick`, lines 595–660), the rollout policy currently picks:
$$\arg\max_{c} \; \text{score}(s, c)$$
where $\text{score}(s, c)$ is `cb.expected_hit_fraction(...)` for damaging moves (`c < 4`), `compact_status_pseudo_score(...)` only when `score <= 0.0`, and voluntary switches (`c >= 4`) only when `stay < ROLLOUT_SWITCH_TRIGGER (0.15)`.
Consequently, rollouts **almost never execute multi-step setup transfers or proactive phazing** unless forced at the tree root:
- A mon with `+2 Atk` or `+2 Spe` and a `self_switch` move sees `expected_hit_fraction == 0.0` on `self_switch`, so rollout attacks with the weak passer instead of passing to the bench sweeper.
- A defender facing a boosted setup mon attacks for `0.20` chip damage instead of clicking a `force_switch` move (`expected_hit_fraction == 0.0`).

### 4.2 Ultra-Fast Action-Residual Scoring (`~20 ns/action`)

Do **not** run a slow neural network per rollout ply. Instead, in `compact_greedy_pick`, add a **rank-4 bilinear / linear action-residual term** $\Delta Q_\psi(s, c)$ to the existing base score:
$$\text{score}_{\text{total}}(s, c) = \text{score}_{\text{base}}(s, c) + \Delta Q_\psi(s, c)$$
$$\Delta Q_\psi(s, c) = \langle u_\psi(\phi_{\text{state}}(s)),\; v_\psi(\phi_{\text{act}}(s, c)) \rangle$$
where $u_\psi(\phi_{\text{state}}(s)) \in \mathbb{R}^4$ is computed **once per rollout ply** (`4` dot products of length `12`), and $v_\psi(\phi_{\text{act}}(s, c)) \in \mathbb{R}^4$ is a precomputed or 6-feature descriptor of choice $c$:
- For move choices (`c < 4`): $\phi_{\text{act}}$ is looked up from `CompactTable`'s per-`MoveId` capability vector (`self_switch`, `force_switch`, `boost_value`, `status_value`, `heal_value`, `priority`) modulated by current state scalars (e.g., own total positive boosts, foe total positive boosts, own HP fraction, whether an unboosted/healthy bench receiver exists).
- For switch choices (`c >= 4`): $\phi_{\text{act}}$ uses the already-available incoming/outgoing `expected_hit_fraction` delta on the target bench mon.

### 4.3 Training & Gate Criteria for `rollout_residual`

1. **Distillation target**:
   - Train $\psi$ via cross-entropy / margin ranking loss to match the **MCTS root visit distribution (`my_n` / `root_matrix`)** of `skuct:27000` / `blind:27000` on states where the search root disagrees with `compact_greedy_pick`.
2. **Strict Two-Stage Gate**:
   - **Gate 1 (`eval_residual` alone)**: Must not regress `perf_probe` wall-clock time by more than `2%`, must reduce held-out TD residual RMS and M17e anchor violations, and must score `> 0.50` in CRN-paired duels (`eval_ab_duel` + `combo_gauntlet` + Baton Pass gauntlet) at equal wall-clock time.
   - **Gate 2 (`eval_residual` + `rollout_residual`)**: Must beat Gate 1 in CRN-paired duels at **matched wall-clock time** (not just matched iteration count). If `rollout_residual` is null at matched wall-clock time, ship `eval_residual` alone.

---

## 5. Recommended Step-by-Step Plan for Next Session

1. **Step 1 — Instrumentation & Dataset Generation (`examples/residual_dataset.rs`)**:
   - Add a read-only feature extractor `extract_eval_features(cb: &CompactBattle, table: &CompactTable, side: usize) -> [f32; 32]` in `crates/bot/src/eval.rs` (reusing the local variables of `eval_leaf_compact`).
   - Generate ~50,000–100,000 labeled states `(x(s, 0), x(s, 1), L_base(s), V_teacher(s), mcts_root_policy)` across meta self-play, perturbed off-pool teams, Perish trap corpus, and Baton Pass / setup corpus.
2. **Step 2 — Residual Diagnostic & Training (`tools/train-residual-nn.py`)**:
   - Report residual RMS by state slice (unboosted vs boosted vs `Substitute` vs `self_switch` on field).
   - Train the antisymmetric `32 -> 16 -> 1` `eval_residual` MLP (and optionally the rank-4 `rollout_residual` head) with PyTorch (`tmp/venv`), verifying bit-accurate forward parity (`< 1e-6`) against Rust.
3. **Step 3 — Integration & Non-Degradation Gates**:
   - Wire `eval_residual` into `eval_leaf_compact` (`crates/bot/src/eval.rs`) and `eval_leaf`.
   - Run `perf_probe` (`blind 27000 8`) to verify wall-clock non-degradation.
   - Run CRN-paired duel gates (`meta-pool`, `combo_gauntlet`, and high-speed Baton Pass gauntlet) at matched budget and matched wall-clock time before deploying.
