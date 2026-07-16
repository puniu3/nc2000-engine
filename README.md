# nc2000-engine

Rust port of Pokemon Showdown's **`[Gen 2] NC 2000`** format (mod: `gen2stadium2`), built to raise bot-research search throughput by orders of magnitude.

**The source of truth is PS as actually implemented.** Divergence from cartridge GSC or real Stadium 2 hardware is out of scope. Correctness is defined as **bit-exact parity** (state + PRNG seed, at every snapshot point) against golden fixtures generated from PS by `tools/gen-fixtures.js`.

## Layout

```
tools/            Node scripts run against the reference PS build (needs PS_ROOT=PS repo, `node build` done)
  export-dex.js            dump the flattened gen2stadium2 dex into data/
  gen-prng-vectors.js      PRNG parity vectors
  gen-fixtures.js          golden fixture generator (live battle -> inputLog replay -> snapshot extraction)
  gen-porting-checklist.js regenerate PORTING.md
data/gen2stadium2.json     reference data (functions replaced by callback-name lists; meta.psCommit records origin)
data/meta-pool-v0/         curated meta team pool (M8): 34 tournament/expert teams, provenance in its README
data/preview-tables-v0/    baked team-preview equilibria per matchup (M8), format in its README
fixtures/prng-vectors.json PRNG vectors
fixtures/corpus-v1/        60 battles (30 puredata + 30 full; 2,268 turns / 2,585 snapshots)
crates/engine/             the engine (prng / dex / state / choice / battle; battle/search.rs = M3 search API)
crates/conformance/        conformance harness (fixture schema, divergence reporter, replay tests)
crates/bot/                bots: random / max-damage / open-loop DUCT MCTS (M5) + heavy playouts, static
                           eval, duel harness, SPSA tuner (M6) + state-keyed tree, RM-solved mixed root,
                           best-response exploitability probe (M7) + baked preview tables, baked/counter
                           agents (M8); examples: arena / play / tune / profile_mcts / bake_preview
PORTING.md                 porting checklist (377 callbacks, generated)
```

## Verification model (the snapshot contract)

- Fixture `choices` are PS's canonicalized inputLog choice lines (e.g. `team 5, 6, 1` / `move surf` / `switch 2`).
- Snapshot points = **immediately after every input line that grew the battle log**. Each snapshot records `turn / requestState / prngSeed / field / sides` (every mon's HP, status, boosts, PP, volatiles) plus the log lines produced since the previous snapshot.
- `prngSeed` uses PS `Gen5RNG.getSeed()` format (four decimal 16-bit limbs, comma-joined). **Seed equality = RNG-consumption-order equality** — a drift in consumption order is caught immediately even when outcomes happen to match.
- Nondeterministic `|t:|` wall-clock lines are stripped at generation time.

## Workflow

```bash
# all tests (green: PRNG parity, dex load, fixture schema, both corpus replays)
cargo test
# adversarial soak: generate fresh fixtures with any seed and sweep them
node tools/gen-fixtures.js --n 100 --pool full --out /tmp/soak --seed 12345
cargo run -p conformance --example sweep -- /tmp/soak
# drill into one diverging fixture (per-choice log diff + seed check)
cargo run -p conformance --example debug -- /tmp/soak/battle-042.json [from_snapshot]
# throughput benchmark (turns/s, playouts/s, ns/clone, allocs)
cargo run --release -p conformance --example bench
# bot arena; deterministic per --seed; prints per-agent think ms/move
# agents: random | maxdamage | mcts[:iters[:c[:eps[:turns]]]] (M6 heavy) | mcts5[:iters[:c]] (M5 baseline)
#         rm[:iters[:probe[:threshold[:buckets]]]] (M7 mixed) | skuct[:iters[:c[:buckets]]] (M7 ablation)
#         exploit:<inner> (best-response probe with a frozen <inner> policy oracle)
#         baked:<inner> | bakedarg:<inner> (M8 table preview, mixed/argmax) | counter[arg]:<inner>
#         (M8 counter-picking probe); teams via --pool fixtures|meta[:LO-HI], tables via --tables DIR
cargo run --release -p nc2000-bot --example arena -- mcts:1000 maxdamage --games 100
cargo run --release -p nc2000-bot --example arena -- rm:1000 mcts:1000 --games 200
cargo run --release -p nc2000-bot --example arena -- baked:skuct:300 skuct:300 --pool meta:0-3 --games 400
# M8: bake preview tables (resumable per pair; deterministic at any thread count)
cargo run --release -p nc2000-bot --example bake_preview -- --teams 0-33
cargo run --release -p nc2000-bot --example bake_preview -- --summarize   # gate report over baked pairs
# play the bot yourself in the terminal (or spectate: bot vs bot)
cargo run --release -p nc2000-bot --example play -- human mcts:1000
# SPSA self-play tuning of the eval weights (weights file rewritten every iteration)
cargo run --release -p nc2000-bot --example tune -- --iters 120 --games 48 --mcts-iters 300 --out w.txt
# profile the heavy-playout MCTS workload (writes target/flamegraph-mcts.svg + self-time table)
cargo run --release -p nc2000-bot --example profile_mcts -- 1000
# regenerate artifacts (e.g. after a PS update)
node tools/export-dex.js && node tools/gen-porting-checklist.js && node tools/dump-callbacks.js
node tools/gen-fixtures.js --n 30 --pool puredata --out fixtures/corpus-v1/puredata --seed 100
node tools/gen-fixtures.js --n 30 --pool full     --out fixtures/corpus-v1/full     --seed 200
```

Porting loop: port one callback → tick it off in `PORTING.md` → keep the replay test green as the legal pool grows. On divergence, `compare::Divergence` auto-localizes to the first differing snapshot + JSON path + that turn's log lines.

### Search API (M3)

`Battle` is a plain deep-clonable value; DUCT/MCTS drives it like this:

```rust
let mut b = Battle::from_fixture(&dex, seed, &p1, &p2)?;
b.set_log_enabled(false);              // search mode: skip the protocol log
while b.outcome().is_none() {          // Some(P1Win|P2Win|Tie) when over
    let picks = [0, 1].map(|s| {
        let cs = b.legal_choices(&dex, s);   // empty ⇔ side owes nothing
        cs.first().copied()                  // SearchChoice is Copy
    });
    b.apply_choices(&dex, picks)?;     // advances to the next request point
}
```

Determinized playouts: clone the battle, `clone.reseed(sample)` — the PRNG is
part of the state. `SearchChoice::to_input` renders the PS-canonical choice
string (`move surf` / `switch 3` / `team 5, 6, 1`) for interop with fixtures
and the debug tools.

### Milestones

1. **M1 — puredata corpus green: DONE (2026-07)**. Team init, team preview, switching, the full turn engine (queue/speed ties/PRNG parity), gen2stadium2 damage pipeline, residuals, and the core conditions (statuses, confusion, flinch, partiallytrapped, mustrecharge, weathers, sleep/freeze clauses) replay all 30 puredata battles bit-exact (state + PRNG seed + protocol log at every snapshot).
2. **M2 — full corpus green: DONE (2026-07)**. All 88 callback moves, all 38 callback items, every condition reachable in this format, and the runtime rules. Verified beyond the golden 30: 350 additional fresh-seed battles replay bit-exact (see the soak workflow above); the last 100 diverged nowhere. Every reachable `PORTING.md` entry is ticked; unreachable entries are documented there.
3. **M3 — search API: DONE (2026-07)**. `crates/engine/src/battle/search.rs`: `SearchChoice` (compact/`Copy`), `Battle::legal_choices` / `apply_choice(s)` / `needs_choice` / `outcome` / `reseed`, plus `set_log_enabled(false)` for protocol-log-free stepping. Enumeration mirrors the `choices.rs` validation rules and application funnels through `Battle::choose` with PS-canonical strings — one code path shared with fixture replay. Verified by `tests/search_api.rs`: at every decision point of all 60 golden fixtures the PS choice is inside the enumerated set and (sampled) every enumerated choice is accepted; log-off replay is state+seed-identical; random playouts terminate. A 100-battle fresh-seed soak stayed bit-exact after the M3 perf work.
4. **M4 — throughput: DONE (2026-07)**. Both prescribed structural fixes landed end-to-end: (a) integer event/callback identity — `Cb` ids + `CbMask` bitsets, per-event `Ev` statics with precomputed prefix callbacks, precomputed handler Order/Priority/SubOrder, aggregate handler masks per Pokemon/Side plus one battle-level union mask that lets zero-handler runEvents return without any machinery; (b) flat state — `TypeId`/`TypeList` with effectiveness matrices, Copy `EffectState` (`EffId`/`DK`/compact `Scalar`), inline move slots/name/gender, interned move targets + flag bitmasks, log-only formatters gated off in search mode. Measured (same machine): **54k turns/s replay, 103k turns/s clone-based playouts (the MCTS workload), 4.7 µs / 4-alloc clones** — 5.3x / 4.7x / 38x-fewer-allocs vs M3. Verified: full corpus bit-exact + 420 fresh-seed battles (3 soak batches), which also caught and fixed two real M2 gaps (frz thaw on Tri Attack's rolled burn; substitute's onHit sub-cost, reached for the first time in ~600 soak battles). Remaining ideas toward 1e6 (fn-pointer dispatch tables keyed by (CondId, Cb), direct SearchChoice application without the string round-trip, construction diet) are deferred until bot-side search actually wants them.
5. **M5 — baseline bots + arena: DONE (2026-07)**. `crates/bot`: `Agent` trait, `RandomAgent`, `MaxDamageAgent` (static base power x STAB x type effectiveness, no voluntary switches), and `MctsAgent` — **open-loop DUCT** (decoupled UCB1 per side; statistics keyed by `SearchChoice` maps because chance can change even a node's *request kind* between iterations; every iteration re-simulates from a fresh root clone with `reseed` = the determinized-playout pattern; uniform-random rollouts; HP-fraction eval squashed into [0.25, 0.75] at the horizon). `examples/arena.rs`: seed-paired side-swapped duels over the 120 fixture teams, deterministic for a given `--seed` at any thread count. Measured (95% CI): maxdamage > random **0.770±0.058** (200 games); mcts:1000 > maxdamage **0.830±0.074** (100 games); mcts:100 *loses* to maxdamage at 0.450±0.098 — search strength scales with budget (the 120-action team-preview root is undersampled at 100 iterations). Budget-scaling ladder (adjacent-tier duels, 60–100 games): 300v100 **0.775**±0.082, 1000v300 **0.710**±0.089, 3000v1000 **0.580**±0.097, 10000v3000 **0.567**±0.126 — per-3x-budget gain decays ~215 → 156 → 56 → 47 Elo, i.e. the curve enters its saturation band around 3k–10k iterations with uniform rollouts. Further strength must come from rollout/eval quality, not budget. Throughput: mcts:1000 plays ~0.6 games/s over 12 threads. Known levers when more strength is wanted: heuristic (ε-greedy max-damage) rollouts, team-preview candidate pruning, tree reuse across decisions, root parallelization, and the deferred M4 perf ideas.
6. **M6 — strength core (rollout/eval quality): DONE (2026-07)**. `crates/bot/src/eval.rs`: weighted static eval — per living mon HP fraction + alive bonus + status penalties, per-stage boost values on the active, PP fraction, and a **threat feature** (best expected hit fraction vs the opposing active: gen-2 damage core on *effective* stats via `get_stat`, so boosts/burn/screens/boost items feed in; STAB x effectiveness x mean roll x mean multi-hit x accuracy; hidden power uses the real DV-derived type/power) — linear in tunable `EvalWeights`, sigmoid on the side differential, leaf squashed into (0.25, 0.75). `Playout::Heavy` in the MCTS: **ε-greedy max-damage rollouts** (ε=0.2, sharing the damage estimate) **truncated 8 turns** past the rollout start, eval leaf; the M5 agent survives bit-identical as `Playout::Uniform` (arena spec `mcts5`), verified game-for-game: mcts5:1000 vs maxdamage (seed 1) = 82W/18L, exactly what the unmodified M5 commit produces in a worktree (the M5 entry's 0.830 was recorded from a pre-commit state; the committed code's figure is 0.820±0.076). Hyperparameters settled head-to-head (truncation 8 > 4 at 0.683, > 16; ε 0.2 ≈ 0.1 — equal-iteration duels vs a third opponent had pointed the other way, a lesson: compare variants directly). Weight tuning: two SPSA self-play runs over the shared duel harness (`examples/tune.rs` + `duel.rs`; 120 iters x 48 games at mcts:300 and 160 x 96 at mcts:100 with larger perturbations) both held out at no gain vs the hand-written weights (0.490±0.069 / 0.483±0.057) and drifted every weight <15% — the hand values sit on a plateau at arena noise levels; they ship as `EvalWeights::default()`. **Gate passed: mcts:1800 vs mcts5:1000 = 0.665±0.066 (200 games) at equal-or-cheaper wall-clock — 399 vs 484 ms/move measured in-duel** (the duel harness now reports per-agent think time; equal-iteration mcts:300 vs mcts5:300 is 0.650±0.122, and heavy is ~4x cheaper per iteration). Ladder re-measured (heavy adjacent tiers): 300v100 0.670±0.093, 1000v300 0.690±0.091, **3000v1000 0.717±0.115** (M5: 0.580), 10000v3000 0.567±0.126 — the gain per 3x budget now holds ~120–160 Elo through 3k and the saturation knee moved a full tier (~1k–3k → ~3k–10k). Post-truncation profile (`examples/profile_mcts.rs`) is flat — top item `run_event` 9.3%, the whole eval+policy adds ~1.7% — so the deferred M4 perf ideas stay deferred.

7. **M7 — mixed strategies: DONE (2026-07). Parity gate passed; exploitability gate measured null.**
   Engine: `Battle::state_key()` / `state_key_bucketed(b)` — a drift-proof (totally-destructured,
   so adding a state field breaks the build until placed) hash of every decision-relevant field,
   PRNG excluded, with HP and roll-magnitude bookkeeping optionally bucketed; inline FxHash-style
   hasher (SipHash cost 1.5x'd think time before). Bot (`crates/bot/src/smmcts.rs`): state-keyed
   transposition-table tree over bucketed keys — chance splits on anything discrete (KOs, status
   procs, request kinds, durations) so nodes have stable request kinds and cached legal-action
   sets, while HP pools into 16 buckets (exact keys measured weaker: every damage roll becomes
   its own node and the tree starves of depth) — decoupled UCB1, and a two-phase budget: 75%
   tree, 25% probing the top-3×top-3 root joint cells (seeded with the late tree half's on-policy
   samples), whose payoff matrix is solved offline by full-width RM+ with linear averaging;
   play samples the thresholded average strategy and defers to argmax-visits at solver-pure
   spots. Rejected by measurement en route (scores vs maxdamage where mcts:300 = 0.82):
   online outcome-sampling RM at every node 0.30–0.43 (IS spikes up to |A|/γ + flat exploration
   tax), online root-only RM 0.50–0.58, RM+ under sampling (regret-clamp ping-pong), EMA cell
   estimates over sparse probes, RM team previews (120 actions is outside the sampled-equilibrium
   regime; preview stays UCB+argmax — M8 bakes it offline anyway). **Parity gate (200 games each,
   seed 1, equal iterations AND equal wall-clock — e.g. 230.6 vs 228.4 ms/move):** rm:1000 vs
   mcts:1000 = 0.475±0.069, rm:3000 vs mcts:3000 = 0.480±0.069, argmax ablation skuct:1000 =
   0.460±0.069, skuct:3000 = 0.525±0.069 — every CI straddles 0.5: the state-keyed tree replaces
   open-loop aliasing at zero strength cost, and the mixed layer costs nothing measurable on top.
   M5/M6 agents preserved bit-identical (mcts5:1000 vs maxdamage seed 1 reproduces 82W/18L
   exactly). **Exploitability gate: null.** Probe = `exploit.rs` best response with a
   seed-marginal (3-sample) policy oracle at 3x budget — a single-sample point-mass oracle is
   neutralized by MCTS seed noise (0.480, no exploitation; an "argmax" policy is only
   deterministic per rng seed). At 200-game resolution the probe gains +0.118±0.067 vs frozen
   mcts argmax, +0.110 vs skuct argmax, +0.100 vs rm mixed — indistinguishable; heavier mixing
   (threshold 0.15) made the target MORE exploitable (+0.135), and the argmax-tuned BR beat the
   mixed agent hardest (0.680). Reading: mixing from an estimated matrix is not equilibrium
   mixing — against non-adapting opponents it only leaks play quality, and instance seed noise
   already provides free mixing. The RM layer and the `Agent::root_policy` oracle API stay for
   when an adapting opponent exists (M8 human play, M10 hidden info); the strength
   recommendation after M7 is the state-keyed argmax agent (`skuct`), which is also the natural
   substrate for M10's determinized belief states.

### Roadmap (M8+)

Scope decisions (2026-07, settled with the owner):

- **Mixed strategies / per-decision GTO approximation: IN.** Exploiting weak opponents: OUT — no opponent modeling beyond the equilibrium baseline.
  - Post-M7 refinement (2026-07, settled with the owner): in-battle online mixing measured null (see M7) and is parked; **GTO ships at the team-preview layer in M8** (offline-solved equilibrium over the meta matchup matrix — exact cells, real equilibrium, and the adapting adversary — humans studying the tables — actually exists there) and returns in-battle at M10 where mixing has a structural job (hidden-information leakage). In-battle flagship until then: `skuct` (state-keyed argmax).
- **Imperfect-information play is a gradient goal**: pursued after the strength core lands. The perfect-info bot is never discarded — it stays as dev harness, teacher, and upper-bound benchmark.
- **Ship target: Rust→wasm, client-side, no GPU anywhere.** No hard latency cap (the player trades wait time for strength; ponder hides most of it). Dev-time compute baked into shipped tables is allowed and encouraged.
- **Mainline meta battles: IN. Minor/casual party robustness: OUT** — priors, tables, and evaluation specialize on a curated meta pool.
- **Auto team building: stretch** — big completeness win, deferred unless it turns out cheap.

Milestones:

8. **M8 — meta pool + baked tables: DONE (2026-07-16). Both gates passed.** Plan as approved: curate the NC2000 meta team/set pool (replaces the random fixture pool as the product distribution), then bake dev-time compute into shipped tables: **team-preview policy per matchup as an offline-solved (mixed) equilibrium** over the meta matchup matrix, solved with M7's full-width RM+ solver — this is where mixed strategies ship, because the estimates are offline-exact and the adapting adversary (humans studying the tables) exists. In-battle search: `skuct`.
   **Meta pool v0: DONE (2026-07-16, curation delegated to mechanical tournament-pedigree criteria after the owner opted out of meta judgment).** `data/meta-pool-v0/`: 34 PS-validator-clean teams — 8 tournament-proven (Historia Cup 7.5 top-8, the last live tournament on NC2000 rules proper; later Historia Cups use an incompatible special ruleset) + 26 expert-built samples (Smogon NC2000 Resource Hub, intl+JP authors) — ranked by placement → Chio-VR mean → HC7.5 usage mean, all components in the JSON for downstream re-weighting. `tools/build-meta-pool.js` regenerates from `raw/`; conformance test `meta_pool` proves every team plays to completion in the engine. Provenance, exclusions, and expansion sources (Gym Leader's Castle tournament archive etc.): `data/meta-pool-v0/README.md`. Notable: the shipped PS format bans neither evasion nor OHKO nor Bright Powder, so JP tournament sets are legal verbatim.
   **Baked preview tables: DONE (2026-07-16).** `crates/bot/src/preview.rs` + `examples/bake_preview.rs` + `data/preview-tables-v0/`. Preview space quotiented 120 → **60 canonical actions** (20 subsets × 3 leads; bench order only picks which display slot a random drag-in lands in — uniform either way, so win probability is invariant — 4x fewer matrix cells). Per pair: full-width 60×60 **screen** with ε-greedy max-damage self-play (16 games/cell ≈ 3 s — ranking signal only, the policy never switches voluntarily), **support** = top screen-equilibrium best responses ∪ `skuct:2000` advisor picks at the real preview root (8/side), **refine** support×support with `skuct:200` self-play (24 games/cell, P1/P2 alternated per game, fresh seeds; ~5 min/pair on 12 threads), **solve** with full-width RM+ → mixed + argmax policies + exact counter-picking guarantees, one resumable JSON per pair, byte-deterministic at any thread count (verified). Consumption: `TableSet` resolves live rosters to pool teams by full-set signature; `baked[arg]:<inner>` plays table preview (mixed sample / argmax) and delegates the battle to `<inner>`, falling back to `<inner>`'s own preview search on unknown matchups; `counter[arg]:<inner>` is the probe that best-responds at preview to a known table policy. **Gate "mixed beats counter-picking" passed (offline, exact):** over the 10 pairs of the top-4 tournament teams, mixed-vs-argmax counter-pick margin mean **+0.077**, max **+0.245**, 0 violations; genuine mixing in 8/10 pairs (mirror equilibria mix up to 6 preview actions; 1st-vs-3rd holds 0.370 mixed where argmax is counter-picked to 0.125). **Gate "baked ≥ online" passed (runtime, equal in-battle budget):** baked:skuct:300 vs skuct:300 = **0.630±0.047** (400 games; argmax tables alone 0.600±0.048, and the baked side is also cheaper per move — preview costs a lookup); at skuct:1000, 0.555±0.069 — the online 120-action preview root improves with budget but does not catch up. **End-to-end counter-pick probe (the optional certification): null at 400-game resolution** — counterarg vs bakedarg 0.460±0.049, counter vs baked 0.495±0.049; the matrix-exact exploitation edge (~+0.08 predicted) does not survive 24-game/cell estimation noise + BR winner's curse + instance seed noise at product budgets. Same reading as M7: mixing's measurable value lives at the matrix level (where the gate is exact and the table-studying adversary operates), and it costs nothing in play (0.630 vs 0.600 vs the same online opponent). Full-pool bake (595 pairs, ≈2 days) left running resumably (`--teams 0-33`, tournament teams first; `tmp/bake-full.log`); the pilot 10 pairs ship with this commit and the directory extends in place. Eval weights stay at the M6 plateau conclusion; "opening lines" are dropped — baked previews + `skuct` in battle cover the opening. Preview-time team identification vs humans (species+levels only; one species-set collision in the pool) is deferred to M10's belief machinery — the wasm client falls back to online preview until then.
9. **M9 — wasm ship**: wasm32 build (engine already compiles for `wasm32-unknown-unknown`), ponder, JS bridge + table loading, browser demo. Gate: ≥ native mcts:3000-equivalent strength at 2–3 s/move on the target device, degrading gracefully into longer opt-in thinks.
10. **M10 — imperfect info (gradient)**: belief = distribution over the M8 meta sets filtered by observations; per-iteration determinization imputes sampled hidden movesets/items onto the public state. The meta-pool restriction is exactly what makes this cheap. DV-level purity and hidden-counter purity are non-goals. Gate: blind-vs-perfect-info gap measured; no psychic tells at parity budget.
11. **M11 — auto team building (stretch)**: self-play round-robin + mutation hill-climbing over the set space; only if the manual meta pool proves limiting.

Non-goals: exploitation/opponent modeling, large NNs / GPU inference, minor-party coverage, whole-game equilibrium solving, PS ladder client (revisit after M9/M10). Longer-term verification ideas stay live: coverage-forcing corpora, expert scenario fixtures, predicted-vs-actual diffing during live play.

## Baseline measurements (this machine, WSL)

- PS (TS): 6.5 battles/s, 570 turns/s, 5.5 ms per clone → tree search is infeasible on the TS engine
- Target: 1e5–1e6 turns/s, sub-microsecond clones (prior art pkmn/engine claims >1000x over PS)
- Rust engine after M4 (`cargo run --release -p conformance --example bench`):
  54k turns/s log-off replay (35k log-on), 52k turns/s full-game random playouts,
  **103k turns/s clone-based playouts** (clone + reseed + play to the end — the MCTS
  determinization workload), 4.7 µs / 4 allocs per mid-battle clone (the 4 are the two
  roster Vecs + occasional volatile spill). That is ~95x PS on turns and ~1100x on clones.
  M4 history: 10k → 20k (integer event identity) → 24k (lookup hoisting + aggregate
  masks) → 31k (TypeId + log gating) → 41k (battle-level mask gate) → 52k (zero-handler
  fast path + ActiveMove diet); clones 22 µs/153 allocs → 4.7 µs/4 allocs (compact
  Copy state). Profile is flat now (top item ~9%); see `examples/profile.rs`
  (flamegraph) and `examples/clone_anatomy.rs` (per-field clone costs).

## Porting landmines (facts measured in this repo)

1. **The sim runs whatever ability it is handed, even in Gen 2.** The validator's canonical form is `ability: 'No Ability'`. A blank ability round-tripped through pack/unpack gets default-filled with the species ability, and e.g. Shed Skin then fires and changes battle outcomes (proven on battle-005). All fixtures are TeamValidator-clean.
2. **Gen 2 DV consistency**: SpA DV must equal SpD DV; the HP DV is derived from the low bits of Atk/Def/Spe/Spc. The validator rejects teams that violate this.
3. **After team preview, PS truncates `side.pokemon` to the 3 picked mons** — snapshot party size goes 6 (preview) → 3 (battle).
4. Replay must reconstruct players from the inputLog's `>player` lines (packed teams) or it will not match the live battle (the generator does this).
5. The PRNG is the Gen 5 64-bit LCG. `random(n)` is `(next * n) >> 32`, exactly matching JS float math for n < 2^21 (debug_assert enforced).

## References

- Reference implementation: `PS_ROOT` (default `~/pokemon-showdown`, `node build` required). Data provenance commit: `meta.psCommit` in `data/gen2stadium2.json`.
- Measured scope: 267 moves (88 with callbacks) / 62 items (38) / 37 conditions / 0 abilities / 251 species (246 after the 5 Ubers), 377 callbacks total, 76 distinct event hooks in use.
