# Headroom probe v1 — how far is the shipped bot from its ceilings?

Status: **complete 2026-10-16 — playing-strength headroom demonstrated by extra search.**
The same-information comparison `blind:81000:0.4` vs `blind:27000:0.4`
finished with **427 wins / 329 losses in 756 confirmation games** (score 0.564815).
Its registered alpha=0.025 sequential interval, **[0.504114, 0.625394]**, first
cleared 0.5 at block 6. The experiment stopped after 896 executed comparison games;
128 discovery games and six first pairs are excluded under the pre-confirmation amendment.
The native mean choose-call cost was 3.19x, not equal latency. This answers whether
an implementable same-information improvement remains on the tested catalog distribution;
it does not quantify distance to optimality or gains against human custom teams.
Details and final audit: [`headroom-scale-v1`](../headroom-scale-v1/README.md).
Product configuration unchanged. Shared preview entry
[`nc2000-headroom`](https://mini-usb.local:18080/nc2000-headroom/index.html).
Owner question: the bot's strength looks saturated; how much room is left before the
theoretical limit, and where? This probe splits "the limit" into three measurable
comparisons instead of asking A-vs-B duels between near-identical bots. These are
empirical probes, not certified distances to optimal play.

Repo HEAD at measurement: `da407a5` (plus pre-existing untracked files from other
work, untouched). Binary: `cargo build --release -p nc2000-bot --example arena`.
All duels: `--teams data/team-pool-v2/team-pool.json` (24-party catalog) unless noted,
`--belief-pool data/belief-pool-v3/belief-pool.json`, `--threads 10`, max-turns 500,
seed-paired side swaps; ± = arena 95% CI over side-swap pairs. Raw rows in `duels/`
(each passes `arena --validate-jsonl`). No baked preview tables apply to this catalog
(arena logged 0 loaded), so preview uses live search at each arm's iteration budget.
The product uses 27k plus pondering; this arena has no pondering. Catalog teams are
drawn uniformly by `run_duel`, unlike the product's `drawWeight` distribution.

Null control first: `blind:3000:0.4` vs itself, `--crn-seeds`, 8 games → 0.500, 0 split pairs.

## Results

### 1. Budget scaling of the shipped algorithm (NOT time-matched)

| A vs B | games | score A | ≈Elo | seed |
|---|---|---|---|---|
| blind:3000 vs blind:1000 | 200 | 0.620 ± 0.066 | +85 | 2026101601 |
| blind:9000 vs blind:3000 | 200 | 0.550 ± 0.065 | +35 | 2026101602 |
| blind:27000 vs blind:9000 | 160 | 0.556 ± 0.070 | +39 | 2026101603 |

Reading: point estimates for the two higher-budget comparisons are positive
(~+35–40 Elo), but both 95% intervals include 0.5. They establish neither
saturation nor a reliable gain at the product budget. The earlier "budget doubling
n.s." observation remains unresolved. The subsequent registered 81k-vs-27k confirmation demonstrates a gain; see the result above.
These are fixed iteration budgets, with A doing 3x the iterations; they do not
measure strength at equal wall-clock cost.

### 2. Root noise at 27k, read from existing records (no new games)

`analyze_root_regret.py` over `findings/release-strength-extension-v1/records.jsonl.gz`
(1,536 games of blind:27000:0.4). Each action ~ N(mean, mean(1−mean)/visits):

| decisions | n | P(fresh argmax ≠ played) | simple regret / decision |
|---|---|---|---|
| in-battle, ≥2 actions | 50,118 | 18.8% | 0.11% win prob |
| — top-1 visit share < 0.50 | 11,677 | 43.9% | 0.23% |
| — share ≥ 0.90 | 13,920 | 1.7% | 0.03% |
| team preview (60 actions) | 2,304 | 72.9% | 1.30% win prob |

Also: 60.7% of in-battle roots have |Q1−Q2| < 2·SE; argmax(visits) ≠ argmax(mean) on 3.2%.
Caveat: UCB means are biased, normal approximation; this is an indicator, not a certificate.
Reading: the approximation assigns much greater uncertainty to preview than to
in-battle decisions. The 72.9% is a Gaussian surrogate draw's disagreement with the
recorded action, not a measured fresh-search flip rate. Per-decision quantities
cannot be summed into a certified whole-game loss.

### 3. Hidden-information ceilings (CRN seeds)

| A vs B | pool | games | score A | ≈Elo | seed |
|---|---|---|---|---|---|
| open:9000 vs blind:9000 | catalog (24, in prior) | 160 | 0.537 ± 0.065 | +26 | 2026101605 |
| skuct:9000 vs open:9000 | catalog | 160 | 0.525 ± 0.074 | +17 | 2026101607 |
| open:9000 vs blind:9000 | fixtures (120, **off-prior**) | 160 | 0.656 ± 0.054 | +112 | 2026101606 |
| open:27000 vs blind:27000 | fixtures (off-prior) | 80 | 0.738 ± 0.078 | +181 | 2026101608 |

`open` = blind machinery with the opponent's TRUE sets pinned (picks still hidden);
`skuct` = full information including picks. Reading:
- In-catalog, knowing sets/picks is worth little (both n.s.): the belief prior already
  identifies catalog teams from preview.
- **Off-prior, the observed exact-set advantage was about +110 to +180 Elo.** The larger
  point estimate at 27k is not an independently established budget interaction. This is the
  largest gap found. The Aug-2026 ladder corpus (190 bot-vs-human games) was off-pool on
  188/190 (project memory `~/.claude/projects/-Users-puniu-nc2000-engine/memory/ladder-corpus-2026-08.md`),
  suggesting that off-prior handling is relevant to custom teams. This does not
  make the fixture result representative of human play. In-app play against the
  catalog parties is the in-prior regime above.
- Exact-set access is an informational benchmark for better inference, not a formal
  upper bound on gains from changing a bounded search algorithm. Fixture teams are
  generated, not human; the share of the gap recoverable by a better fallback/set
  prior is unmeasured. The fixture pool mixes 60 `puredata` teams (all 360
  Pokémon without items) and 60 `full` teams; every member is level 50–51.
  `fixture-scope-audit.json` records this composition and current input digests.
  This audit checks sampling scope, not a new independent legality certificate.

### 4. Best-response probe (matched oracle controls complete)

| A vs B | games | score A | seed |
|---|---|---|---|
| exploit:skuct:3000 vs skuct:3000 | 100 | 0.650 ± 0.094 | 2026101609 |
| exploit:blind:3000 vs blind:3000 | 100 | 0.710 ± 0.089 | 2026101610 |

Confounds, in order: the exploiter (`BrAgent`, `crates/bot/src/exploit.rs`) runs 3x the
target's budget (9k, M6 heavy DUCT, c=1.0) — §1 says 3x alone is worth ~0.55–0.62; it
sees the full battle state — against `blind` that adds the §3 information edge;
CRN agent seeds were not requested, though a later code audit found a same-seed
collision in each block's first game (see the headroom-scale confirmation amendment).
**These raw scores are not isolated
exploitability measurements.**

The original `mcts:9000` control proposal failed the code audit: `BrAgent`
enables rollout `status` and `switch`, while M6 disables both. The new
`exploit-control:<inner>` uses `BrAgent` itself, disabling only model queries
and selecting root replies with UCB. Both arms have 9k own iterations,
c=1.0, heavy eps=0.2, rollout turns=8 and horizon=100. Targets explicitly
retain **c=0.4**; omitting it as the old proposed commands did would use c=1.
Oracle queries add up to 3×3k target iterations, so this matches own search
work, not total cost. Both A arms see full state.

| Target | oracle on | oracle off | paired increment (95% CI) |
|---|---|---|---|
| skuct:3000:0.4 | 0.650 | 0.520 | +0.130 [−0.012, +0.272] |
| blind:3000:0.4 | 0.710 | 0.630 | +0.080 [−0.043, +0.203] |

Each arm has 100 games / 50 side-swap pairs. Across arms, schedules and seeds
are identical to the original target row. Intervals use the paired differences,
not the difference of separate CI endpoints; normal approximation, descriptive,
no multiple-comparison correction or product-acceptance claim. Both intervals
include zero: the point estimates suggest an oracle benefit, but neither resolves
it at this sample size. This does not establish absence of exploitable behavior.
The oracle arm takes about twice the control's mean thinking time here.

Validation: 408 new games total (200 old-oracle replays, 200 controls, 8 null).
Every replayed game score and total turn count matches the old executable's
artifact. Null `exploit-control:blind:30:0.4` vs itself, CRN, seed 2026101611:
score 0.500, zero split pairs. No caps or invalid games in any batch.
The model-query test and three arena tests pass. Product defaults are unchanged.

Reproduce or validate completed batches:

```sh
cargo build --release -p nc2000-bot --example arena
python3 findings/headroom-probe-v1/run_br_controls.py
python3 findings/headroom-probe-v1/render_report.py
```

The frozen protocol binds the binary and inputs; resume rejects changed hashes.
A rebuild with a different binary hash requires a separately registered run,
not replacement of the existing protocol or results.
The post-run resume check found a tuple/list mismatch after JSON serialization.
`run_br_controls.py` now normalizes that comparison. The measured runner is
preserved as `run_br_controls_frozen.py`; `br-control-resume-fix.json` binds both
runner hashes and the untouched original protocol. Only this recorded runner
correction is allowed on resume. Reusing all five completed batches passed.

The preview serves the HTML report and aggregate/audit JSON from `report/` through
`~/.cache/nc2000-headroom-preview/`; launching the hub backend directly from this
external-volume checkout returned `kill EPERM`. After rendering an updated report,
copy those report files to that preview directory. Only the report and its aggregate
JSON are served. The route and hub index entry were verified over HTTPS.

## Optional follow-up research

The active question is answered by the registered 81k-vs-27k result. Further
experiments are not needed to establish this observed extra-compute headroom.
Useful separate questions are:

1. How much of the gain can be obtained at equal response time, including a
   preview-only budget increase or a more efficient search implementation?
2. How much of the off-prior informational benchmark can a feasible set model
   recover on a representative, separately frozen distribution of custom teams?
3. How does the gain change with product draw weights and pondering?

Throughput on this Mac (10 threads): blind 9k ≈ 1.1 games/s vs 3k, blind 27k vs 9k ≈ 0.32 games/s.

## Files
- `duels/*.jsonl` — nine original artifacts plus five BR replay/control/null artifacts.
- `analyze_root_regret.py` — §2, deterministic (seed 20261016), ~10 s.
- `run_br_controls.py`, `br-control-protocol.json` — fixed control experiment and input hashes.
- `br-control-summary.json`, `br-control-progress.json`, `br-*.log` — paired results and execution record.
- `validation.json` — all 14 artifacts checked for completeness, scores, paired intervals and zero caps.
- `render_report.py`, `report/` — shareable Japanese report, scoped to the preview directory.
