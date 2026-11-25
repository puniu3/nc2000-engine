# Bot Strength, Search & Belief Reference

Authoritative snapshot of `crates/bot` playing-strength mechanisms, measured-null dead ends, and open structural frontiers on `HEAD`. Consult live code under [`crates/bot/src/`](../crates/bot/src/) rather than gitignored scratch archives (`findings/`, `tmp/`, `.git/sota/`).

---

## Opponent switch coverage on turn 35 (2026-11-18)

The ordinary allocator remains the default after the [equal-budget strength comparison](#equal-budget-strength-comparison).

In the wake-corrected search at `641900f9`, Zapdos's switch received 545 visits by 54k and still had 545 at 729k. All 486,000 visits after 243k went to Zapdos's Rest. Raikou executed pure Rest at both budgets. Replaying from Zapdos's own observation gave five root types and selected the switch at 9k, 54k, 243k and 729k. At 9k that search faced Raikou's Rest on 3,210/9,000 iterations. The two players' searches valued replies under different information and action distributions.

**Why visits stalled.** The opponent's [root UCB selector](../crates/bot/src/smmcts.rs) pools each reply's reward over the Raikou actions it happened to face. At 243k, Zapdos's switch faced Rest on 507/545 visits (93.0%), while Zapdos's Rest faced it on 237,642/238,435 (99.7%). Using each column's own history, the switch's UCB was 0.31445 against 0.31697 for Rest. Giving every column the same Raikou policy (100% Rest) and keeping the visit counts fixed reversed the ranking to 0.32166 against 0.31684. The switch ranked first among all four replies. The same reversal held at 729k. With 545 switch visits its exploration bonus grew only from 0.06034 to 0.06295 over those 486,000 further iterations.

The table gives **Zapdos's mean reward** at 243k. Rest was its most visited stay-in reply. The fresh columns use only new returns from 4,096 forced visits per eligible action pair, shuffled each round on a copy replayed to the same checkpoint (schedule seed `20261123`).

| Raikou action | Ordinary vs Zapdos Rest | Ordinary vs switch (visits) | Fresh vs Zapdos Rest | Fresh vs switch |
| --- | ---: | ---: | ---: | ---: |
| Rest | 0.314 | 0.261 (507) | 0.317 | 0.311 |
| Thunder | 0.354 | 0.143 (35) | 0.350 | 0.239 |
| Hidden Power Ice | 0.364 | 0.500 (2) | 0.409 | 0.340 |
| Switch | 0.000 | 0.000 (1) | 0.083 | 0.232 |

**Effect on the move played.** In a second replay of the original tree, we replaced only the four opponent root reward sums at 243k with the corresponding mean against Raikou's Rest times the original visit count. The tree then searched normally to 729k. Using ordinary cells made Zapdos switch on the very next iteration, but it ended at 580 switch visits and Raikou still played pure Rest.

Using fresh cells from four shuffle seeds (`20261123` through `20261126`) produced 1,416 / 28,619 / 25,071 / 2,834 switch visits at 729k. Seeds 23 and 26 ranked Thunder first for Zapdos immediately after the reset and left Raikou on pure Rest. Seeds 24 and 25 ranked the switch first and changed Raikou to a Rest/Thunder mixture with 43.96% and 48.46% Thunder. Both draws selected Thunder. All four runs began from the same ordinary tree. These are schedule replicates on one position.

In 96 fresh paired continuations of that position (9k per player, seed `2026112701`), Raikou scored 69/96 with Thunder and 48/96 with Rest. Of the discordant pairs, 35 favored Thunder and 14 favored Rest. Zapdos switched in 93/96 pairs. Weighting those scores by the two mixed policies gives gains of 9.62 and 10.60 percentage points over pure Rest on this roster. The other two resets give zero gain. The ordinary controls reproduced turn 33's pure switch and turn 6's 74.14% → 74.49% Self-Destruct mixture from 243k to 729k.

**Reproduction.** Use the recorded games in the directory named by `/tmp/nc2000-current-reply-audit`. If they expire, regenerate them at `8c4ca02e` with `budget_probe collect-positions --pairs 3 --iters 3000 --preview-iters 27000 --seed 20261017003 --out GAMES.json`. On the local diagnostic branch `experiment/root-reply-reweight-audit` (`3a8e4fc6`), the following replays the seed-24 intervention. For the ordinary control, read `.checkpoints[-1].joint_cells[]` in place of `.joint_resample.cells[]`. To reproduce the unmodified 729k matrix, extend the first command's checkpoints to `54000,243000,729000`. The turn 33 and turn 6 control slots are `1,35,1` and `0,6,1`.

```sh
audit=$(cat /tmp/nc2000-current-reply-audit)
out=$(mktemp -d)
probe() { cargo run --quiet --release -p nc2000-bot --example budget_probe -- "$@"; }
probe convergence --games "$audit/focal.json" --slot 1,37,0 --checkpoints 54000,243000 \
  --joint-trials 4096 --seed 20261124 --out "$out/cells.jsonl"
reset=$(jq -c '{after:243000,means:([.joint_resample.cells[] | select(.action=="move rest") | {key:.reply,value:(1-.mean)}] | from_entries)}' "$out/cells.jsonl")
probe convergence --games "$audit/focal.json" --slot 1,37,0 --checkpoints 243000,243001,729000 \
  --reply-reset "$reset" --out "$out/reweighted.jsonl"
probe counterfactual --games "$audit/focal.json" --conv "$out/reweighted.jsonl" \
  --pair 243000,729000 --trials 96 --cont-iters 9000 --seed 2026112701 --out "$out/paired.jsonl"
```

### Equal-budget strength comparison

On the experimental branch `experiment/budgeted-reply-reweight` (`95293242`), the candidate with probes scored 60–68 in the primary 81k comparison. Its 95% seat-pair bootstrap interval was 39.06–54.69%. The adoption criteria were a lower bound above 50% and a median time ratio of at most 1.05.

Both variants froze the current execution probabilities every 256 iterations starting at 512. For each hidden root type, they reweighted the cell rewards for each legal reply to that common own-action distribution, shrinking sparse cells toward their reply's historical marginal with eight pseudo-visits. The resulting offset was added to the live opponent UCB score. The probe variant also spent every 32nd type visit after warm-up on a uniformly sampled legal reply and an independent sample from the frozen own policy. Forced probes counted within the budget. With the experiment disabled, all non-timing checkpoint fields matched the baseline across 48 positions and ten checkpoints.

Applying the candidate’s sparse-cell correction to the ordinary 243k turn-35 checkpoint raised switch UCB from 0.31445 to 0.32154, ahead of Rest at 0.31684. On the original turn-35 seed, the 729k switch count was 545 for ordinary search, 565 for reweighting alone and 170,826 with probes. Probes changed Raikou's executed choice from Rest to Thunder and extended adaptive search from 54k to 243k. In a new 48-pair 9k continuation sample, Thunder scored 27/48 and Rest 25/48. The same change in allocation cost 25 percentage points on the turn-33 control at 729k, where switching scored 34/48 and Thunder 22/48. The turn-6 control gained 3.19 points in policy-weighted score. Across the three search seeds (XOR 0, 131, 1597), the probe variant's turn-33 differences at 729k were −25.00, +27.41 and 0 points.

On 48 fresh games we sampled one decision per game and scored every changed execution distribution, including changes in mixing probabilities that preserved the chosen action. There were 20 distinct action contrasts, each evaluated with 32 paired 9k continuations. The entries below are mean score changes in percentage points over all 48 positions, with 95% game-bootstrap intervals. The continuation estimates condition on the recorded rosters.

| Battle budget | Reweight only | Reweight with probes |
| --- | ---: | ---: |
| 81k | −0.49 [−1.84, +0.61] | +0.21 [−0.36, +0.91] |
| 243k | −0.90 [−2.14, +0.01] | −0.15 [−0.98, +0.38] |
| 729k | −1.04 [−2.54, +0.26] | −0.44 [−1.02, 0.00] |
| Adaptive, 243k cap | −0.90 [−2.14, +0.01] | −0.15 [−0.98, +0.38] |

At 81k, reweighting changed the policy on 8/48 decisions and probes changed it on 10/48. Median search times on these positions over two passes in reverse arm order were 824 ms (ordinary), 849 ms (reweight) and 833 ms (probes). Mean adaptive iterations on the sampled decisions were 172,125, 188,438 and 185,625, respectively. Only the probe variant had a positive 81k point estimate and advanced to the independent seat-swapped comparison.

| Equal battle budget | Candidate W–L | Score (95% paired CI) | Median ms, candidate / ordinary |
| --- | ---: | ---: | ---: |
| 81k | 60–68 | 46.88% [39.06, 54.69] | 891 / 891 |
| 243k | 11–13 | 45.83% [25.00, 66.67] | 2,798 / 2,744 |
| 729k | 8–8 | 50.00% [25.00, 75.00] | 9,200 / 9,414 |
| Adaptive, 243k cap | 14–10 | 58.33% [41.67, 75.00] | 2,604 / 2,316 |

Across the adaptive games the candidate used 191,900 iterations per searched battle decision and the baseline used 176,045. All games finished before the turn cap. The three secondary comparisons have only 8–12 seat pairs each.

To reproduce on `95293242`, build the release `budget_probe` example. Collect 48 games with `collect-positions --pairs 48 --iters 3000 --preview-iters 27000 --seed 2026112801 --out GAMES.json`, then run `python3 tools/bench-reply-policies.py GAMES.json NEW_OUTPUT_DIR --probe target/release/examples/budget_probe`. For the independent full-game comparison use `budget_probe duel --reply-mode probe --iters-a 81000 --iters-b 81000 --pairs 64 --preview-iters 27000 --seed 2026112806 --out duel-81k.jsonl`. Use 12 pairs at 243k, 8 pairs at 729k and 12 pairs with `--adaptive` at the 243k cap, with a separate output file for each. Focal continuations used seed `2026112804`, 48 paired trials and 9k search. The independent depth-17 backup experiment remains at `2d29f0b3`.

### Root-allocation vs. depth cross-check & backup/bench ablations (2026-11-18)

Cross-checking schedule-vs-depth replay audits on `8d0180fe` against paired full-game `BlindAgent` (`c = 0.4`, `team-pool-v2` `drawWeight`, `belief-pool-v3`, shared `27k` preview, seed `20261017003`) duels separated three independent failure links and ruled out naive depth caps and unconditional terminal backup shrinkage as standalone `81k` fixes:

1. **Three independent root bottlenecks on turn 35 (`1,37,0`):**
   - Paired `9k` continuations (`128` trials per fixed root reply) favor Raikou's `Thunder` over `Rest` against all four Zapdos replies (`96/128 = 75.0%` vs `66/128 = 51.6%` against `switch`, `+23.44` pp, 95% CI `[+11.72, +35.16]`).
   - Replaying the `243k` tree through `729k` with `103,949` forced/shallow-schedule `Rest × switch` visits shows **ordinary full-depth descent accurately prices post-switch danger** (`0.480` raw / `0.482` backed vs `0.516` continuation score), whereas handing the identical 4-decision prefix to the `rollout_switch = false` 8-turn `CompactBattle` playout inflates `Rest × switch` to **`0.727` (`0.709` backed)** (`+24.7` pp).
   - Even with `103,949` `Rest × switch` visits at full depth, own UCB pairs all new switch visits against `Rest` (`Thunder × switch` stays at `35` visits) and `min_equilibrium_visits` (`max_v / 8 = 90,956` in [`root_policy.rs`](../crates/bot/src/root_policy.rs)) drops `Thunder` (`1,218` visits) from the root LP. Relaxing only `min_own_v` to `1` at final readout on that deep tree (`reopen-deep`) lifts `Thunder` to `55.13%` execution weight (`pure_guarantees`: `0.635` vs `0.483` for `Rest`).
   - On a 24-game low-budget panel (`9k -> 27k`), changing only traversal depth moved execution policies by `> 5` pp on `3/24` roots (mean TV `0.029`), whereas changing only the root joint-visit schedule moved `9/24` roots (mean TV `0.244`).
2. **Terminal-backup guard & bench-canonicalization ablations (full-game duels vs. `HEAD`):**
   - **Unconditional internal backup (`uncond`, removing `val0 > 1e-6 && val0 < 1 - 1e-6` in [`smmcts.rs`](../crates/bot/src/smmcts.rs))**: Spontaneously lifts turn-35 opponent `switch` visits (`545 -> 30,090` at `243k`, `82,626` at `729k`; `Thunder × switch` `35 -> 22,422`) and scores **`139W–117L` (`54.30%`, `+29.9` Elo)** at `9k` (`128` pairs), **but** stacking `TAU = 1.5, lambda = 0.85` `max`-shrinkage across deep `0/1`-terminal plies (`97.1%` terminal rate on turn 35) inflates `Rest × switch` from `0.482` to `0.653` and flattens strength to **`64W–64L` (`50.00%`)** at `27k` (`64` pairs). Softer backup (`TAU = 8, lambda = 0.50`: `123W–133L` at `9k`), plain MC backup (`127W–129L` at `9k`), disabling `bayesian_row_q` (`129W–127L` at `9k`; `61W–67L` with `uncond` at `27k`), and internal sleep-alias collapse (`123W–133L` at `9k`) also failed to improve strength.
   - **Unseen bench slot canonicalization + species-keyed `Switch` prior pooling (`canonbench,sempool` in [`blind.rs`](../crates/bot/src/blind.rs))**: Collapses redundant `[lead, A, B]` / `[lead, B, A]` unseen bench permutations (`root_types` roughly halved on single-candidate Turn-1 roots) and stops pooling `Switch(2)`/`Switch(3)` priors across unrelated species; improves `uncond` at `9k` to **`141W–115L` (`55.08%`, `+35.4` Elo)** (`65W–63L` at `27k`; `125W–131L` at `9k` under `HEAD` backup alone).
   - **4-decision tree cutoff (`depth4` + `uncond,canonbench,sempool`)**: Regresses at `9k` (`121W–135L`, `47.27%`, `-19.0` Elo; `68W–60L` at `27k`, `35W–29L` pilot at `81k`, `0.825x` median time) because cutting off tactical/post-switch lines substitutes optimistic `rollout_switch = false` evaluations (`0.727` vs `0.482`), confirming `depth4` is not a sound fix.
3. **Next-session priority:** Keep ordinary full-depth tree evaluation (do not cap depth at 4) and combine unseen-bench canonicalization / semantic switch pooling with a root-only repair that jointly (a) scores opponent root UCB replies against the current own policy rather than stale marginal collision history, (b) cross-samples rival own actions specifically against active opponent threat replies (`y* > 0`), and (c) admits rival own actions with sufficient threat-cell evidence past the `max_v / 8` LP row cutoff at readout.

## 1. Shipped Search, Root, Endgame & Belief Stack

### 1.1 Budgets & Profiles
- **Live Web budgets**: Battle decisions use the adaptive cap in `battle.adaptive`. Team Preview searches for `unknownPreview` iterations on a compiled table miss. [Live search configuration](../data/live-search.json)
- **Profile baseline ([`data/search-profiles.json`](../data/search-profiles.json))**: `blind` UCB `c = 0.4` (`27,000` profile baseline used by `?evaluate`, `?fork`, and offline probes; `open` is retired except historical kifu/fork replay).

**2026-11-19 adaptive budget saturation.** In 4,608 full games across three schedule sets, ordinary-width adaptive 135k scored 194–190 against adaptive 189k. From 189k, increasing the cap to 405k or 729k had one-sided 95% upper score gains of 2.9 and 3.9 percentage points. Fixed 243k and fixed 729k scored slightly higher against adaptive 135k and 189k, with intervals that still allow gains above five points. The practical early-stop profile uses UCB `c = 0.4`, checks every 13.5k and a 0.01 reward gap. On 48 matched WASM decisions the 135k cap took 0.30 / 1.58 s at P50 / P95, compared with 0.74 / 0.98 s for fixed 81k. On a fresh 384-game comparison fixed 81k scored 189–195 against fixed 243k. [Full results, latency curve and reproduction](search-budget-saturation.md).

**2026-11-18 native budget sweep (`cb6d485d`).** Over `256` games per comparison, `81k` scored `133W–123L` against `27k` (`51.95%`, 95% CI `46.48–57.42%`) and `243k` scored `139W–117L` against `81k` (`54.30%`, 95% CI `48.44–60.16%`). Median time per searched battle decision rose `3.33x` and `3.32x`, respectively, in the Linux batch. Both intervals include `50%`, so the saturation point remains unresolved.

The comparisons shared `128` seat-swapped schedules and fixed seed `20261017003`. Teams were sampled by `drawWeight` from `team-pool-v2` with `belief-pool-v3` used for inference. Both arms used `BlindAgent` (`c = 0.4`) and `243k` live preview, with identical preview choices per seat across swaps and comparisons. The intervals bootstrap schedule pairs, and no games reached the turn cap. The earlier `7fa23a1d` sweep scored `50.39%` for `243k` against `81k` (95% CI `44.53–56.25%`) with a different catalog and belief pool.

**2026-11-18 extended budget ladder (`4abbe162`, [`budget_probe.rs`](../crates/bot/examples/budget_probe.rs)).** Extending the same `20261017003` protocol (`team-pool-v2` `drawWeight`, `belief-pool-v3`, `BlindAgent c = 0.4`, shared `243k` live preview, seat-paired independent RNG seeds `(seat_seed_p1, seat_seed_p2)`, `0` turn caps, `0/8` null split pairs at `1k vs 1k`) across `2,816` full games resolved `243k > 81k` cleanly above `50%` and left the `729k` comparison against `81k` inconclusive:

| Matchup | Games (Pairs) | W–L | Score | 95% Bootstrap CI | Elo | Median Searched Decision (A / B, Ratio) |
| :--- | ---: | :--- | ---: | :--- | ---: | :--- |
| `9k vs 3k` | 512 (256) | `297W–215L` | `58.01%` | `54.10–61.91%` | `+56.1` | `92.74 ms / 27.27 ms` (`3.401x`) |
| `27k vs 3k` | 512 (256) | `299W–213L` | `58.40%` | `54.49–62.30%` | `+58.9` | `311.13 ms / 27.42 ms` (`11.347x`) |
| `27k vs 9k` | 512 (256) | `253W–259L` | `49.41%` | `45.90–52.93%` | `-4.1` | `306.66 ms / 91.66 ms` (`3.346x`) |
| `81k vs 27k` | 512 (256) | `250W–262L` | `48.83%` | `44.73–52.73%` | `-8.1` | `1044.90 ms / 311.73 ms` (`3.352x`) |
| `243k vs 81k` | 512 (256) | `279W–233L` | `54.49%` | `50.78–58.40%` | `+31.3` | `3505.26 ms / 1041.52 ms` (`3.366x`) |
| `729k vs 81k` | 256 (128) | `131W–125L` | `51.17%` | `45.70–56.64%` | `+8.1` | `11659.64 ms / 1036.24 ms` (`11.252x`) |

On the shared first `128` pairs (`256` games), `243k vs 81k` scored `146W–110L` (`57.03%`, 95% CI `51.95–62.50%`, `+49.2` Elo; second `128` pairs `133W–123L` / `51.95%`), whereas `729k vs 81k` scored `-5.86%` lower (`95%` paired bootstrap CI `-12.50%` to `+0.39%`; `24` pairs better, `38` worse, `66` equal). Over `200` searched positions sampled from `24` `81k`-vs-`81k` games (`3k`–`729k` checkpoints), `mean_depth` climbed linearly (`1.790 -> 2.290 -> 2.831 -> 3.396 -> 4.017 -> 4.637`), `bayesian_equilibrium()` probe overhead remained `< 0.2%` (`~6–10 μs` per call), and `best()` flip rates remained `13.0%` (`27k->81k`), `9.5%` (`81k->243k`), and `13.0%` (`243k->729k`) with `legacy` UCB argmax flips plateauing at `8.0–9.0%`. The original probe flagged the equilibrium margin on `42–47%` of flips. That count included endgame roots that bypass the equilibrium policy and pure policies where the draw cannot change the action. Counterfactual `27k` continuations (`24` paired trials each) on the `19` `81k` vs `243k` disagreements scored `-1.54%` (`95%` CI `-6.14%` to `+3.95%`; `4` better, `9` worse, `6` tied).

**Position audit of the 243k to 729k transition (2026-11-18, search revision `fcc2ec6a`).** In three native self-play games at 3k battle / 27k preview, two position samples (32 with seed 1 and 40 with seed 178) covered 67 distinct decisions. Nine changed action when the same tree continued from 243k to 729k. Sampling another 60 distinct decisions from the remaining 105 produced four flips, for 13/127 overall. All positions come from the same three games.

| Acting side and position | Action at 243k → 729k | Paired continuation results |
| --- | --- | --- |
| Game 1, step 35, P2, turn 33. Zapdos 136/215 vs Raikou 179/215 | Switch → Hidden Power Ice. Its probability increased from 25.67% to 30.60% as mean depth rose from 3.365 to 4.093. | At 9k, switch scored 44/64 vs 21/64 for Hidden Power Ice. Fresh seeds gave 82/128 vs 46/128 (50 pairs favored switching, 14 favored attacking). |
| Game 1, step 37, P1, turn 35. Raikou 129/215 vs Zapdos 65/215 | Rest → Sleep Talk. Mean tree depth 6.795 → 7.318. Both picks led in visits. | At 3k, Rest scored 28/40 and Sleep Talk 22/40. At 9k with fresh seeds, 68/128 vs 67/128 (34 pairs favored Rest, 33 favored Sleep Talk). |
| Game 0, step 6, P2, turn 6. Snorlax vs Heracross, both at full HP | Self-Destruct → Earthquake. Self-Destruct retained 74.14% → 74.49% of the executed policy. The changed draw at `decision_seed ^ done` selected Earthquake. | At 3k, Self-Destruct scored 15/40 and Earthquake 6/40. At 9k with fresh seeds, 39/128 vs 16/128 (31 pairs favored Self-Destruct, 8 favored Earthquake). |

For turn 33, weighting the two measured action scores (82/128 and 46/128) by their execution probabilities gives 56.84% at 243k and 55.46% at 729k. Holding 729k and the search seed fixed, starting the usual rollout after at most four joint decisions shifted the attack probability to 18.56% and the estimated policy score to 58.84%. Thus deeper descent also lowered conditional expected score at equal cost in iterations on this seed. Two other search seeds produced different policies. These estimates condition on the actual hidden roster and a 9k continuation on both sides.

Raikou used Rest on turn 32 and slept through turns 33 and 34. Both the true state and the player's reconstruction have sleep time 1. On turn 35 Raikou woke before executing Sleep Talk, which called zero moves over all four legal opposing replies and 64 battle seeds per reply. Hidden Power Ice knocked out Zapdos on all 192 stay-in trials. The opponent could switch. Forcing Hidden Power Ice at the same 128 paired 9k seeds scored 74/128 vs 67/128 for Sleep Talk. The end-to-end advantage of Rest over Sleep Talk did not replicate at 9k.

At 729k the ordinary root gave Sleep Talk 390,392 visits, Rest 337,360, and Hidden Power Ice 871. The empirical mean for Rest was 0.6866 and for Sleep Talk 0.6857. At both checkpoints the normal selector played the visit leader. There was one root type. All 486,000 additional root replies were Zapdos's Hidden Power. Its legal switch stayed at 280 visits, yet the independent 9k continuation agent switched in 126/128 paired trials. This identifies a mismatch between the replies used to value Raikou's actions and the replies it faced.

Two controls held the total at 729k. In a diagnostic build, limiting descent to four joint decisions before starting the usual rollout picked Rest at 729k for all three search seeds (original seed XOR 0, 131, 1597). Unrestricted descent picked Sleep Talk on the first two. In the same unrestricted trees, spending one in every 32 iterations after 243k on the existing `step_typed_joint_probes` raised switch coverage to 6,464–6,971 visits and all three seeds selected Rest. The probes change allocations for both players. This is position-level evidence for testing reply coverage. The conditional wins in the table apply to individual actions forced against the recorded roster and continuation agent. The expected value of Snorlax's mixed policy barely changes when its sampling probabilities move by 0.35 percentage points.

To reproduce the two rows with `budget_probe`, collect with `--pairs 3 --iters 3000 --preview-iters 27000 --seed 20261017003`, then run `convergence --samples 40 --seed 178 --checkpoints 243000,729000`. Its position IDs 27 and 2 are the Raikou and Snorlax rows. Use `counterfactual --pair 243000,729000` with `--trials 40 --cont-iters 3000 --seed 20261118` or `--trials 128 --cont-iters 9000 --seed 20261120`. Turn 33 was position ID 41 in the additional 60-position sample (sampling seed 188 after clearing the `searched` flags for the earlier two samples). Its continuation seeds were 20261121 and 20261122. The probe records per-action visits, means and equilibrium probabilities, whether an executed override had multiple positive weights, discordant continuation pairs, and the frequencies of root replies in search and in continuations.

**Root reply selection and Rest wake turns (2026-11-18).** On the turn-35 tree above, the opponent's switch reached 280 visits by 54k and stayed there through 729k. The switch met the equilibrium solver's 96-visit column cutoff. Allocation had already stalled in the opponent's UCB selection. At 243k its estimated reward plus exploration bonus was 0.2268 + 0.0842 for switching and 0.3100 + 0.0030 for Hidden Power. Extending the same tree gave every additional iteration to Hidden Power.

At revision `8c4ca02e`, we warmed the original tree to 243k, then shuffled all five own actions against all four legal replies once per round for 4,096 rounds. Each cell below shows Raikou's mean backed-up search reward over the *new* visits. Under Zapdos's Hidden Power, Sleep Talk ranked above Rest and Hidden Power Ice. Under its switch, Sleep Talk ranked below both. The ordinary tree had estimated Sleep Talk against the switch at 0.751 from 147 visits. These scores depend on the continuation tree and its sampling schedule.

| Raikou's action | Zapdos uses Hidden Power | Zapdos switches |
| --- | ---: | ---: |
| Sleep Talk | 0.688 | 0.579 |
| Rest | 0.663 | 0.707 |
| Hidden Power Ice | 0.636 | 0.664 |
| Thunder | 0.634 | 0.765 |
| Switch | 0.217 | 0.772 |

Reproduce from the three recorded games above with `budget_probe convergence --games GAMES.json --slot 1,37,0 --checkpoints 54000,243000 --joint-trials 4096 --seed 20261123 --out cells.jsonl` at `8c4ca02e`. Coordinates are zero-based. Ordinary checkpoint cells and fresh resampled cells go to separate JSON fields. A depth control on turn 33 kept Zapdos's switch ahead of Hidden Power Ice against the most frequent reply (Sleep Talk) both at a four-decision cutoff (0.484 vs 0.417) and at unrestricted depth (0.487 vs 0.403). The executed root mixture was sensitive to depth even with that conditional ordering intact.

**Battle policy fix (`dde5bb91`).** Rest's public timer identifies the turn on which the engine wakes the user before executing its move. The root now excludes Sleep Talk and Snore on that turn when a safe alternative exists and the existing strategic-wait guards permit exclusion. On turn 35 all three search seeds selected Rest at both 243k and 729k. On turn 33 the original seed selected the switch with probability 1 at both budgets, up from 0.743 and 0.694. The two additional seeds respectively picked Hidden Power Ice and Thunder at both budgets. The turn-6 mixed policy reproduced exactly on the original seed. Adaptive stopping on turn 35 moved from 243k to 54k. Switch coverage there still stalled at 545 through 729k. In 48 fresh paired 9k continuations using the corrected policy (seed `2026112403`), the opponent switched 48/48 times on turn 35 and Rest scored 27/48 against Sleep Talk’s 26/48. On turn 33 switching scored 30/48 against Hidden Power Ice’s 19/48.

In 48 independent draw-weighted games (3k battle, 27k preview, seed `2026112401`), all 66 publicly reconstructed wake decisions with legal Sleep Talk matched the actual Rest timer. We sampled one such decision per qualifying game (21 games, seed `2026112402`). The fix excluded Sleep Talk in 17 of 21. Compared with the prior policy, execution probabilities changed on 2, 3 and 3 decisions at 81k, 243k and 729k. Weighting the changed actions by 32 paired 9k continuations per contrast gave mean score changes of −0.45, +0.11 and −0.03 percentage points across all 21 decisions (+0.11 for the adaptive 243k cap). All game-bootstrap intervals included zero. A separate unfiltered sample of one decision from each of 16 new games produced identical policies with the wake fix.

**Depth-dependent backup defect.** The search records both players' selected actions in a 32-entry inline buffer. Beyond that boundary, the old fallback credited the raw leaf reward to overflow actions and skipped those nodes when propagating corrected child values toward the root. With two branching players this can occur at the 17th decision. A regression at depth 17 returned 0.250000 where the same child statistics require 0.608594. Branch `experiment/deep-backup-path` (`2d29f0b3`) processes the full path in reverse order and passes the regression at depths 2, 16, 17 and 33.

Instrumenting the original 729k runs on turns 6, 33 and 35 found one overflow in total, on turn 35 with a terminal-valued return. The depth fix reproduced all three wake-corrected policies. On the separate 16-game sample the old path overflowed 2,518 times in game 9, step 49, P2, including 2,278 returns strictly between the terminal thresholds. The fix moved the 729k visit leader's mean from 0.9911 to 0.9904 and preserved its choice of Self-Destruct. All 16 sampled choices were preserved at all three budgets. On eight Team Preview roots from the same schedules, all joint visit counts and mean rewards also matched at 81k, 243k and 729k. The branch preserves the correction for a broader high-depth battle and preview test.

**Reply-allocation experiment.** Branch `experiment/root-reply-coverage` (`10b2f395`) periodically tests a sparsely sampled reply for the ordinary selected own action within the sampled root type. On turn 35 the 729k switch counts rose from 280/153/70 to 1,099/1,217/1,249 on the three search seeds. On the original seed most new visits went to Thunder and turn 33 selected Rest. Fresh paired 9k continuations there scored 61/96 for switching, 40/96 for Hidden Power Ice and 36/96 for Rest. On the separate 16-game sample (3k collection, seed `2026112301`, one searched decision per game), policy-score changes averaged −1.13, −0.56 and −0.06 points at the three fixed budgets and −0.56 under adaptive stopping (40 paired 9k continuations per changed contrast, seed `2026112303`). Early stops fell from 8 to 6 of 16, raising mean iterations from 151,875 to 178,875. The default allocator retains its existing selection rule. A further candidate should estimate replies on independently sampled own-action distributions and test whether an updated reply estimate changes the executed policy and paired continuation score.

The full-game pilot used draw-weighted teams, paired seat swaps, independent seat RNG streams, shared 27k preview, and schedule seed `2026112302`. Each arm used the same battle budget or the same adaptive controller. Each comparison used the original battle policy as its opponent. Ratios compare median time per searched battle decision within each run.

| Equal battle budgets | Wake fix W–L | Wake time ratio | Reply allocator W–L | Allocator time ratio |
| --- | ---: | ---: | ---: | ---: |
| 81k | 2–2 | 1.002 | 12–12 | 0.999 |
| 243k | 2–2 | 1.004 | 5–11 | 1.010 |
| 729k | 2–2 | 1.000 | 7–5 | 0.997 |
| Adaptive 243k cap | 2–2 | 0.960 | 5–3 | 0.762 |

All games ended before the 500-turn cap. The wake comparison has only two pairs per budget. The allocator's estimates vary across these small schedules. On the experiment branch use `budget_probe duel --reply-coverage --iters-a 729000 --iters-b 729000 --pairs 6 --preview-iters 27000 --seed 2026112302 --out duel.jsonl`. The adaptive run used `--adaptive`, both iteration limits set to 243000 and four pairs.

**Adaptive battle budget experiment (2026-11-18).** The original opt-in profile used a 243,000 iteration battle cap. The subsequent live default keeps that cap and uses the 13.5k checkpoints and 0.01 gap from the saturation study above. The following describes the original profile. Public pruning down to one eligible action ends the search immediately. Otherwise, every 27,000 iterations the controller checks whether the same action wins two consecutive checkpoints. At both checkpoints every eligible rival must have samples, fall below the existing Bayesian root LP visit cutoff (one eighth of the leader's visits, rounded down), and trail the leader's mean reward by at least 0.02. The endgame-aware final selector must also pick the leader. The earliest empirical stop is 54,000. A disagreement between the reconstructed and request-legal action sets keeps the full battle budget. These checks are deterministic for a given search and independent of worker slice timing. Team Preview uses its usual lookup and budget.

One decision was sampled uniformly from each of 570 spectator games (seed `20261118`). There were 567 reconstructed positions with a searched fixed-budget decision. Continuing each tree to 243k gave the following comparison.

| Battle budget | Mean iterations | Same action as fixed 243k |
| --- | ---: | ---: |
| Fixed 81k | 81,000 | 513/567 (90.48%) |
| Adaptive, 243k cap | 182,762 | 565/567 (99.65%) |
| Fixed 243k | 243,000 | 567/567 |

The controller stopped early on 196 positions (34.57%), including 146 at 54k. Two early picks changed by 243k (2/196). The mean saving was 24.79% of fixed 243k, and the adaptive search used 2.26 times the iterations of the default 81k. Agreement measures sensitivity to further search on the same seed.

In a native pilot against fixed 243k, the adaptive agent scored 7 wins and 9 losses across eight seed-paired, seat-swapped schedules with no turn caps. Seven pairs split, and the fixed agent won both seats in one pair. Both agents searched preview at 243k. Teams were sampled uniformly from `team-pool-v2`, with `belief-pool-v3` used for inference. The mean observed time per choice across both battle and preview was 2.35 s for adaptive and 3.70 s for fixed in this concurrent local run. Eight pairs leave playing-strength parity unresolved. Reproduce with `cargo run --release -p nc2000-bot --example adaptive_budget_probe -- duel 8 4 20261118`.

Reproduce the position screen with `cargo run --release -p nc2000-bot --example adaptive_budget_probe -- <spectator-dir> 570 6 20261118` after extracting the `spectator/*.raw.log` entries from `data/corpus-spectator-logs.zip`. The probe uses the corpus reconstruction of the acting player's team and prefix-only opponent observations with `belief-pool-v3`.

### 1.2 Imperfect-Information Reconstruction & Belief ([`belief.rs`](../crates/bot/src/belief.rs), [`observe.rs`](../crates/bot/src/observe.rs), [`hp_evidence.rs`](../crates/bot/src/hp_evidence.rs))
- **Player observation boundary**: Web workers, `ProtocolAgent`, `BlindAgent`, and `BlindSearcher` reconstruct roots strictly through [`ObservedPosition`](../crates/bot/src/position.rs) / [`LocalPlayer`](../crates/bot/src/player.rs) with symmetric hidden runtime timers ([`player_state.rs`](../crates/bot/src/player_state.rs)), `|teamsize|` opponent pick-count tracking ([`import.rs`](../crates/bot/src/import.rs)), live Hidden Power effectiveness narrowing, and deductive passive-item exclusions.
- **Strict pool trust until contradiction**: While `Belief::alive` is non-empty over [`data/belief-pool-v3/belief-pool.json`](../data/belief-pool-v3/belief-pool.json) (`522` entries), candidate filtering and Turn 0 Team Preview remain exact.
- **Post-contradiction off-pool synthesis**: Once the known pool is contradicted, `Belief` activates near-pool soft shell repair, quantile-coupled `FallbackCandPool` synthesis, bundled zero-prior Gen 2 species coverage ([`belief_supplement_v1.json`](../crates/bot/src/belief_supplement_v1.json)), and low-Speed / special low-Attack EV/IV spread synthesis validated against all revealed moves, items, and public HP buckets (preventing impossible-HP leaks), using an independent party-sampling RNG stream.
- **Opt-in priors ([`prior.rs`](../crates/bot/src/prior.rs), [`pick_prior.rs`](../crates/bot/src/pick_prior.rs))**: `BeliefPrior` (carry marginals for moves/items/leads, applied once in `FallbackCandPool` without `LoadoutDraw` double-counting) and `PickPrior` (3-of-6 team preview frequencies) are opt-in and inert by default; see [`docs/community-belief-prior-design.md`](community-belief-prior-design.md).
- **World subsampling & canonical Quick Claw keys**: `BlindSearch` defaults to `MAX_SUBSAMPLED_WORLDS = 24` with bounded warmup ([`blind.rs`](../crates/bot/src/blind.rs)) and ships `RmConfig::canonical_claw_key = true` + `RmConfig::unseen_claw_root = true` ([`smmcts.rs`](../crates/bot/src/smmcts.rs)):
  - Dead-claw search-key quotienting (`quick_claw_observable_fast` in `smmcts.rs` + `BattleSearchCache::any_quick_claw`) omits `quick_claw_roll` whenever no surviving picked Pokemon holds `Quick Claw` (`+0.34` ply mean tree depth at `27k`/`81k`) and strips stale previous-turn `Residual` `speed == QUICK_CLAW_ACTION_SPEED` (`65535`) on active `Quick Claw` holders at idle move requests.
  - Idle roots share a single root `Node`, `det_cache` entry, and canonical-speed `RootTypeStats` (`type_key`) across sampled `quick_claw_roll` draws (and force `quick_claw_roll = false` at Turn-0 Team Preview), preventing hidden-coin root/LP splits and eliminating per-iteration `update_all_speeds` calls (`638W–562L`, `0.532`, `+22` Elo over `1,200` paired CRN games on `belief-pool-v3 --claw-min 1`; `0.507 ± 0.031` over `800` CRN games on full `belief-pool-v3`).

### 1.3 Public Root Masks & Root Decision Policy ([`mask.rs`](../crates/bot/src/mask.rs), [`root_policy.rs`](../crates/bot/src/root_policy.rs), [`endgame_root.rs`](../crates/bot/src/endgame_root.rs))
- **Turn-order-independent public root pruning**: `BlindSearch::use_public_exclusions` masks only actions proven useless or self-forfeiting from public and own-certain state (never from guessed opponent Speed, Quick Claw, bench sets, or HP rolls): public Self-KO and operational Sleep-Clause forfeits, awake `Sleep Talk`/`Snore` (vs sleep-less foe learnset or statused/Safeguard/Sub/curing-berry self), frozen `Sleep Talk`/`Snore`, safe full-HP `Rest`/heals, weak/already-up `Substitute`, weak/capped `Belly Drum`, capped self-boosts, already-up `Spikes`, and public/full-switch-in type immunity checks (`mask.rs`). Intentional stall/wait turns (`destiny_bond`, `perishsong == 1`, residual/recovery waits, `encore`) are preserved via `wait_exempt`.
- **Root selection & Bayesian equilibrium override**:
  - Forced switches (`Request::Switch`) use UCB1 visit allocation and mean tie-breaking; flat `(visits, mean)` ties break deterministically toward `Move` before `Switch` across all entrypoints.
  - Exact column-generation Bayesian root LP ([`root_policy.rs`](../crates/bot/src/root_policy.rs)) runs over eligible actions (`min_own_v = (max_v / 8).max(4).min(max_v)`) with conservative cell shrinkage, `override_value` LCB gating (`MIN_MIX_MARGIN = 0.02`), and `SafeWin` / `FastWin` / `ComebackLoss` + Spikes/status-aware `tactical_desperation_bonus` ([`endgame_root.rs`](../crates/bot/src/endgame_root.rs)).

### 1.4 Rollout, Setup-Relay Subgame, Stage-Game Valuation & Certified Solver ([`mcts.rs`](../crates/bot/src/mcts.rs), [`relay.rs`](../crates/bot/src/relay.rs), [`linear.rs`](../crates/bot/src/linear.rs), [`eval.rs`](../crates/bot/src/eval.rs), [`bounds.rs`](../crates/bot/src/bounds.rs))
- **Continuation (`mcts.rs`, `eval.rs`)**: 8-turn `CompactBattle` heavy playout with `RmConfig::rollout_combo = true` (Perish trap/escape/phaze rules) and `LeafSettlementMode::BlindSpotLeaf`.
- **Setup-Relay simultaneous-move Bellman subgame (`relay.rs`)**: Prices multi-turn `Substitute` + boost + `Baton Pass` chains (`n0 + n1 >= 4`, `0-ULP` no-op on `1v1`) via a cached 2-axis simultaneous-move Bellman subgame solver (`RelayCache`, `512` sets $\times$ `4` ways) and tree-initiated rollout continuation (executing behind-`Sub` boost stacking, directed `Baton Pass` receiver encoding, and on-field `Roar`/`Whirlwind`/`Haze`/`Encore` disruption without uninitiated rollout switches; `+1.4%`, `0.514 ± 0.009`, `z = +3.05` over `1,600` CRN games on `belief-pool-v3`).
- **Low-piece stage-game valuation (`linear.rs`)**: When `n0 + n1 <= 3`, `compact_linear_diff` / `compact_rollout_linear_pick` apply Schur-Bellman / Toeplitz stage-game valuation (including weather-dependent `Moonlight`/`Morning Sun`/`Synthesis` heals, eliminating full-HP recovery passivity in 1v1–2v1 endgames).
- **Certified exact/interval solver (`exact.rs`, `bounds.rs`, `stall.rs`)**: Integer-damage-quotiented lazy interval solver (`BoundSolver`) with pointwise matrix dominance pruning and `TwoSidedHeal` resource-DAG scheduling (`?solver` door).

---

## 2. Measured-Null & Rejected Directions (Do Not Repeat)

| Candidate / Lever | Measured Result | Why It Fails / Stays Off |
| :--- | :--- | :--- |
| **`RmConfig::rollout_status` / `rollout_switch` (`m16c`)** | `0.493 ± 0.023` (status, `1,600` CRN) / `0.466 ± 0.032` (switch, `800` CRN); combined `0.505 ± 0.019` (`2,400` CRN, [`data/m16c-rollout-v2/README.md`](../data/m16c-rollout-v2/README.md)) | Leaf `eval.rs` already prices boosts/status so tree lookahead recovers setup value; heuristic rollout switches surrender tempo. |
| **`LeafSettlementMode::BlindSpotBoth`** (`R1/R2a/R2b/R3b` rollout overrides) | **`-10.0%`** (`0.450` vs `0.550` on `setup_gauntlet`, `160` paired games) vs `BlindSpotLeaf` | Hard deterministic `if-then` overrides inside fixed 8-turn rollouts over-fire and corrupt rollout expectations. |
| **`O(0.1)` static `eval.rs` balance-sheet tweaks & Spikes-zombie leaf reduction** (`exchange_scheme`, `spikes_no_arbitrage`, `phaze_boost_writedown`, `spikes_zombie`) | `1.00x` single-seed flip floor; `spikes_no_arbitrage` `0.5071 ± 0.030` (`840` CRN @ `3k`) & `0.4875 ± 0.0537` (`240` CRN @ `9k`); `phaze_boost_writedown` `0.5000` (`240W–240L`, `480` CRN); `spikes_zombie` (`95%` bench write-down + $\tilde{n}_0 + \tilde{n}_1 \le 3$ `linear.rs` reduction + SD tie-breaker) `0.498 ± 0.017` (`1,195W–1,205L`, `2,400` CRN @ `3k`+`9k`), midgame forks on `163` real `HP <= 1/8` active roots `0.506 ± 0.027` (`3k`) / `0.497 ± 0.025` (`9k`) and `22` bench-zombie roots (`0.73%` game rate, `21/22` identical picks @ `9k`) | At `3k`–`81k` (`c = 0.4`), tree + 8-turn rollout + `linear.rs` + root Bayesian LP already reject voluntary switch-outs on `98.2%` (`160/163`) of real `HP <= 1/8` Spikes roots and resolve `95.5%` (`21/22`) of bench-zombie roots at `9k`; leaf shifts mostly perturb early rollout branches at seed noise. |
| **Locked-pair finite-PP endurance Bellman & `< 3`-PP leaf shadow pricing** (`FinitePpBellman`, `side_pp_scarcity_lambda`, `compact_pp_shadow_diff`) | `0.507 ± 0.021` (`belief-pool-v3 --heal-min 6`), `0.499 ± 0.021` (`team-pool-v2 --heal-min 4`), `0.501 ± 0.020` (`--heal-max 2`), `0.498 ± 0.015` (`3,200` CRN on `team-pool-v2`) | Gating on scarce remaining PP (`pp < 3` / `Rest <= 4 PP`) stays inert during early/midgame stall decisions while 8-turn `rollout_switch = false` neither reaches `n0 + n1 <= 3` in wall mirrors nor preserves scarce PP across branches; real Gen 2 stall is a `n0 + n1 >= 4` multi-mon switching/progress cycle. |
| **Blanket switch-loop penalties** | Breaks `0.71–0.89` Q-value zero-Spikes double-counter pivots | Prediction-bleed switch loops at `10k` were already resolved by `81k` depth + Bayesian root LP (`root_policy.rs`) catching worst-case prediction vulnerability. |
| **Raising `MAX_SUBSAMPLED_WORLDS > 24`** (`world_budget_probe`) | `K = 4..192` flip rates `<=` seed noise (`0.23–0.33`) at `27k`/`81k`; `K = 64` vs `24` off-pool duel `0.5208 ± 0.0712` (`96` games) | `24` quantile-coupled worlds already saturate root action resolution under current `FallbackCandPool` diversity. |
| **Merging internal `is_idle_move_request()` nodes across active `Quick Claw` rolls** (`unseen_claw_idle`) | `383W–417L` (`0.479 ± 0.031`, `800` CRN on `belief-pool-v3 --claw-min 1`) vs `canonical_claw_key + unseen_claw_root` | Pooling `0/1` turn-order flips into low-visit internal decoupled-UCB stats injects high selection variance; keep `quick_claw_roll` separated at observable internal continuations while sharing idle roots and quotienting dead/stale claw state. |
| **Retired search/value architectures** | Residual NN / TPU (`tpu/`, `bot::value`, `fvi_*`), 30-tree independent ensemble, frozen-tree root re-eval | Dual-engine drift / per-iteration latency overhead, or loss of deep single-tree continuation resolution. |
| **Rejected micro-optimizations** | WASM `simd128` / manual `u64x2` `WordHasher`, `BellmanCache` inner SIMD, tail-batched `hash_state_impl`, per-field `Battle` dirty bitmasks | Failed the 5% median/p95 gate in [`web/tests/portable-perf.mjs`](../web/tests/portable-perf.mjs) (`BellmanCache` already hits on most leaves) or risked replay/determinism desync. |
| **External Team Preview grounding (`Turn1-V_net` & online `CF-P_net` rollouts on `SteppedCfPreview`)** | `4,632` live `243k` CRN games across 4 zero-overlap campaigns (`P1` full `12×12` `CF-P_net` `K=24..64`: `+0.63 ± 1.45 pp`, `1,032`g; `P2A` zero-leak 0-ply `Turn1-V_net v3` + Stage-2 blend: `-2.83 ± 1.96 pp`, `S2b_div -13.04 ± 3.45 pp` [`z=-3.78`], `600`g; `P2B` `v3` + Stage-3 `K=192` `CF-P_net`: `+1.50 ± 1.93 pp`, `600`g; `P2c` shipped `policy-v2` Stage-1/2 + Stage-3 `<=4×4`/`<=3×3` `K=192` `CF-P_net`: `+0.79 ± 0.96 pp` overall / `-0.10 ± 1.05 pp` disclosed, `2,400`g). Pooled `CF-P_net` (`P1+P2B+P2c`, `4,032`g): `+0.86 ± 0.74 pp` (`z=+1.16`) overall; **`+0.25 ± 0.80 pp` (`z=+0.31`, 95% CI `[-1.32, +1.82]` pp)** over `3,400` disclosed games. | Although `CF-V_net v2` was trained only on `cb.turn >= 2` ([`tempo_poc.rs:3243`](../crates/bot/examples/tempo_poc.rs#L3243)) and shows large offline Turn-1 cardinal/between-matchup biases (`+0.00476` LL, `z=+4.22` for `blend(S2, cfp64)` across `11,311` independent `243k` games), ~75–80% of that error is a matchup-constant offset that cancels inside `argmax_i min_j V[i,j]`, where shipped `policy-v2` already ranks viable picks tightly. Static 0-ply `Turn1-V_net v3` mis-prunes unseen defensive/relay shells (`Blissey`, `Jolteon` lead, `Exeggutor/Miltank`), while online `CF-P_net` rollouts inject `CF-P_net != 243k MCTS` policy mismatch + `~3.8 pp` cell MC noise (`entropy 1.60 -> 1.79` bits). Separate unconfirmed subgame note: 2-world `S3_camo2w` Stage-3 `max_dim=3` hedging scored `+5.67 ± 2.12 pp` (`z=+2.68`, `476`g pooled across `P2B+P2c`, `p_bonf4=0.067` in `P2c`, absent in `P1` `-0.64 pp`). |

---

## 3. Open Structural Frontiers (at `81,000` Iterations)

1. **Long-horizon recovery-PP exhaustion & multi-mon stall cycles**:
   - [`eval::compact_side_score`](../crates/bot/src/eval.rs) pools all 12 move slots across 3 Pokémon into `w.pp * (sum_pp / sum_max_pp)` (`w.pp = 0.2`, `~0.001` per recovery PP), and `solve_bellman_into` in [`linear.rs`](../crates/bot/src/linear.rs) assumes infinite PP (`GAMMA = 0.985`) on `n0 + n1 <= 3`. Any stall/endurance lever must gate on low mutual break hazard (stagnation) rather than already-depleted PP and must operate on `n0 + n1 >= 4` multi-mon switching/progress cycles rather than locked `1v1` Struggles.
2. **Off-pool cold-start belief calibration**:
   - Improving early-turn move/item/spread joint priors after known-pool contradiction before all 4 moves are revealed. Always evaluate against `OpenAgent` oracle controls and held-out human rosters via [`crates/bot/examples/belief_probe.rs`](../crates/bot/examples/belief_probe.rs) — never rely on `BlindAgent`-vs-`BlindAgent` self-play alone (`offpool_fallback_gauntlet`) or raw spectator-reveal counts (which under-count passive/reactive moves) or hard Gen 2 damage-roll / speed-tie exclusions.
3. **`n0 + n1 >= 4` non-relay leaf-frontier switch lock**:
   - `linear.rs` activates only when `n0 + n1 <= 3` and `relay.rs` prices `Substitute` + boost + `Baton Pass` chains, while `rollout_switch = false` otherwise locks both actives for 8 rollout turns after the tree leaf (and `eval.rs` prices `w.spikes = 1.5` on grounded bench slots only). Any candidate targeting general `4+`-alive horizon valuation must clear the concrete-position seed-noise screen before CRN duels; naive static leaf adjustments already measured `0.500`.
4. **In-tree `CF-V_net` / `CF-P_net` retraining on `243k` MCTS self-play (including `cb.turn == 1`)**:
   - External preview add-ons (`Turn1-V_net`, online `CF-P_net` rollouts) measured null on disclosed play (`+0.25 ± 0.80 pp`, `3,400`g) because `CF-V_net v2` omits `cb.turn == 1` ([`tempo_poc.rs:3243`](../crates/bot/examples/tempo_poc.rs#L3243)) and `0`-ply nets/rollouts diverge from `243k` MCTS continuation values. Future Team Preview grounding should retrain the shared in-tree `CF-V_net`/`CF-P_net` directly on `243k` self-play (including Turn 1), improving Preview and Battle trees together with zero MC noise; separately, 2-world camo subgame hedging (`S3_camo2w`, `+5.67 ± 2.12 pp`, `476`g) can be ablated without `CF-P_net` if hidden-set camo play is prioritized.

---

## 4. Screening, Profiling & Epoch Discipline

- **Strength screening order**:
  1. Inspect concrete boards and measure top-1 flip rate vs the single-seed noise floor at `81k` (`node tools/diagnose-position.js`, `cargo run --release -p nc2000-bot --example setup_gauntlet`, `combo_gauntlet`, `belief_probe`).
  2. Run paired CRN duels (`cargo run --release -p nc2000-bot --example arena -- ... --crn-seeds`) on `team-pool-v2` / `belief-pool-v3` (`>= 3,000–9,000` iters).
- **Performance profiling (always measure current `HEAD`)**:
  - Native throughput: `cargo run --release -p nc2000-bot --example perf_probe`
  - Multi-turn live hotspot flamegraph: `cargo run --release -p nc2000-bot --example profile_blind -- <RECORDED_GAMES> tmp/blind.svg` (avoid single-turn `profile_mcts`, which measures cold allocator setup)
  - Portable WASM/browser gate: `node web/tests/portable-perf.mjs`
- **Epoch & cache discipline**: Bump `replay::ENGINE_EPOCH` on breaking mechanics/`NC2-…` replay changes and `compiled_preview::POLICY_EPOCH` on search/policy changes (governs server-side preview cache identity and evaluator build IDs); follow [`AGENTS.md`](../AGENTS.md) §2.
