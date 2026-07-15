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
fixtures/prng-vectors.json PRNG vectors
fixtures/corpus-v1/        60 battles (30 puredata + 30 full; 2,268 turns / 2,585 snapshots)
crates/engine/             the engine (prng / dex / state / choice / battle; battle/search.rs = M3 search API)
crates/conformance/        conformance harness (fixture schema, divergence reporter, replay tests)
crates/bot/                M5 bots (random / max-damage / open-loop DUCT MCTS) + arena evaluator
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
# bot arena (agents: random | maxdamage | mcts[:iters[:c]]); deterministic per --seed
cargo run --release -p nc2000-bot --example arena -- mcts:1000 maxdamage --games 100
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
5. **M5 — baseline bots + arena: DONE (2026-07)**. `crates/bot`: `Agent` trait, `RandomAgent`, `MaxDamageAgent` (static base power x STAB x type effectiveness, no voluntary switches), and `MctsAgent` — **open-loop DUCT** (decoupled UCB1 per side; statistics keyed by `SearchChoice` maps because chance can change even a node's *request kind* between iterations; every iteration re-simulates from a fresh root clone with `reseed` = the determinized-playout pattern; uniform-random rollouts; HP-fraction eval squashed into [0.25, 0.75] at the horizon). `examples/arena.rs`: seed-paired side-swapped duels over the 120 fixture teams, deterministic for a given `--seed` at any thread count. Measured (95% CI): maxdamage > random **0.770±0.058** (200 games); mcts:1000 > maxdamage **0.830±0.074** (100 games); mcts:100 *loses* to maxdamage at 0.450±0.098 — search strength scales with budget (the 120-action team-preview root is undersampled at 100 iterations). Throughput: mcts:1000 plays ~0.6 games/s over 12 threads. Known levers when more strength is wanted: heuristic (ε-greedy max-damage) rollouts, team-preview candidate pruning, tree reuse across decisions, root parallelization, and the deferred M4 perf ideas.
6. Beyond: hidden-information play (opponent-set inference feeding determinization), PS client protocol adapter for live play, exhaustive-runner-style coverage-forcing corpora, expert scenario fixtures, automatic predicted-vs-actual diffing during live bot play.

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
