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
crates/engine/             the engine (prng / dex / state / choice / battle: M1 turn engine complete)
crates/conformance/        conformance harness (fixture schema, divergence reporter, replay tests)
PORTING.md                 porting checklist (377 callbacks, generated)
```

## Verification model (the snapshot contract)

- Fixture `choices` are PS's canonicalized inputLog choice lines (e.g. `team 5, 6, 1` / `move surf` / `switch 2`).
- Snapshot points = **immediately after every input line that grew the battle log**. Each snapshot records `turn / requestState / prngSeed / field / sides` (every mon's HP, status, boosts, PP, volatiles) plus the log lines produced since the previous snapshot.
- `prngSeed` uses PS `Gen5RNG.getSeed()` format (four decimal 16-bit limbs, comma-joined). **Seed equality = RNG-consumption-order equality** — a drift in consumption order is caught immediately even when outcomes happen to match.
- Nondeterministic `|t:|` wall-clock lines are stripped at generation time.

## Workflow

```bash
# all tests (green: PRNG parity, dex load, fixture schema, puredata replay)
cargo test
# the M2 conformance gate (full corpus; un-ignore as callback moves/items land)
cargo test -p conformance --test replay -- --include-ignored
# regenerate artifacts (e.g. after a PS update)
node tools/export-dex.js && node tools/gen-porting-checklist.js
node tools/gen-fixtures.js --n 30 --pool puredata --out fixtures/corpus-v1/puredata --seed 100
node tools/gen-fixtures.js --n 30 --pool full     --out fixtures/corpus-v1/full     --seed 200
```

Porting loop: port one callback → tick it off in `PORTING.md` → keep the replay test green as the legal pool grows. On divergence, `compare::Divergence` auto-localizes to the first differing snapshot + JSON path + that turn's log lines.

### Milestones

1. **M1 — puredata corpus green: DONE (2026-07)**. Team init, team preview, switching, the full turn engine (queue/speed ties/PRNG parity), gen2stadium2 damage pipeline, residuals, and the core conditions (statuses, confusion, flinch, partiallytrapped, mustrecharge, weathers, sleep/freeze clauses) replay all 30 puredata battles bit-exact (state + PRNG seed + protocol log at every snapshot).
2. **M2 — full corpus green**: the 88 callback moves + 38 callback items + runtime format rules (Stadium Sleep Clause, Freeze Clause, ...).
3. **M3 — search API**: `Battle` is already flat/Copy-friendly; add clone- or apply/undo-based enumeration for DUCT/MCTS.
4. Beyond: exhaustive-runner-style coverage-forcing corpora, expert scenario fixtures, automatic predicted-vs-actual diffing during live bot play.

## Baseline measurements (this machine, WSL)

- PS (TS): 6.5 battles/s, 570 turns/s, 5.5 ms per clone → tree search is infeasible on the TS engine
- Target: 1e5–1e6 turns/s, sub-microsecond clones (prior art pkmn/engine claims >1000x over PS)

## Porting landmines (facts measured in this repo)

1. **The sim runs whatever ability it is handed, even in Gen 2.** The validator's canonical form is `ability: 'No Ability'`. A blank ability round-tripped through pack/unpack gets default-filled with the species ability, and e.g. Shed Skin then fires and changes battle outcomes (proven on battle-005). All fixtures are TeamValidator-clean.
2. **Gen 2 DV consistency**: SpA DV must equal SpD DV; the HP DV is derived from the low bits of Atk/Def/Spe/Spc. The validator rejects teams that violate this.
3. **After team preview, PS truncates `side.pokemon` to the 3 picked mons** — snapshot party size goes 6 (preview) → 3 (battle).
4. Replay must reconstruct players from the inputLog's `>player` lines (packed teams) or it will not match the live battle (the generator does this).
5. The PRNG is the Gen 5 64-bit LCG. `random(n)` is `(next * n) >> 32`, exactly matching JS float math for n < 2^21 (debug_assert enforced).

## References

- Reference implementation: `PS_ROOT` (default `~/pokemon-showdown`, `node build` required). Data provenance commit: `meta.psCommit` in `data/gen2stadium2.json`.
- Measured scope: 267 moves (88 with callbacks) / 62 items (38) / 37 conditions / 0 abilities / 251 species (246 after the 5 Ubers), 377 callbacks total, 76 distinct event hooks in use.
