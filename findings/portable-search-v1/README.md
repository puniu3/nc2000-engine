# Portable search optimization, 2026-10-24

This release is based on deployed commit
`5131c32ff52f16665e5aa2e960f301b79c9558b1`. Unpublished research on the local
master branch is deliberately excluded. No search policy, prior, budget,
RNG sequence, target CPU option, PGO profile or fast-math setting changed.

## Changes and stopping rule

The retained changes are exact prefix memoization for the polynomial search
hash and constant folding of empty effect payloads. The prefix cache has 12
entries per thread. A hit compares every hashed prefix field, including the
backing arrays, before appending the polynomial segment. It does not depend
on mutation hooks. The segment's multiplier and offset reproduce the old
word stream for any incoming hash state. Persistent hashes still walk the
same fields in the same order. Empty effects hash their original id and
order around a constant payload with exactly the old writes.

Tests cover every typed hasher write, nonzero starting state, all prefix
fields, eviction, nested calls, payload options, signed zero/NaN, unused
linked slots and emptied scalar bags. The multithread comparison checks
independent thread-local caches. The replay compatibility alias applies to
one exact engine-source digest; later source edits still invalidate it.

The screening threshold was at least 5% lower total median time with no
category p95 regression above 5%. Direct tuple-hash expansion failed
(native -0.2%, Node WASM -2.2% time reduction) and was reverted. Prefix
memoization passed (7.5% / 7.0%); empty-effect folding passed incrementally
(6.5% / 6.8%). These are separate exploratory runs, not additive final gains.
Removing UCB's temporary allocation looked promising in the short native
screen, but provided only 2.5% in the 27,000-iteration Node WASM confirmation
and was reverted. We stopped after these two useful changes and the small,
variable next result. This does not prove that all further optimization
opportunities are exhausted. `screens/` preserves the trials and patches.

## Final measurements

The release decision uses the actual deployment-toolchain artifacts: the
old public bundle versus the candidate GitHub Actions build, both Rust 1.94.0
and wasm-pack 0.13.1. In Chromium, 27,000-iteration search time fell **7.7%**
(median paired reduction; individual runs 7.5–8.3%). The paired corpus median
was 26.331 -> 24.310 seconds across 35 decisions; p95 was 817.8 -> 763.9 ms.
That is approximately 8.3% higher throughput at this fixed workload; the
shipped budgets are unchanged. Every root policy, choice, battle state, RNG
and log matched. See `production-chromium-27000/`, `ci-candidate.json` and
`deployed-baseline.json`. These binaries were measured on the local host,
not timed on a GitHub runner. The candidate CI build passed all checks and
intentionally skipped deployment until this comparison completed.

The following development measurements used Rust 1.98.1 and remain separate
from the production-toolchain result above.

| Runtime | Iterations | Decisions | Time reduction |
|---|---:|---:|---:|
| Native | 27,000 | 49 | 16.3% |
| Native | 81,000 | 49 | 16.2% |
| Node WASM (adjacent pairs) | 27,000 | 35 | 12.1% |
| Chromium WASM (adjacent pairs) | 27,000 | 35 | 11.1% |

Each row uses three repetitions. Native reports the ratio of corpus medians;
WASM adjacent-pair rows report the median of paired corpus reductions.

The native final comparisons use three alternating baseline/candidate pairs
at 27,000 and 81,000 iterations. The first 81,000 pair was slower, concentrated
in a contiguous middle portion of the candidate run. It is retained in full:
`native-81000-confirmation/` reuses that pair and adds two pairs in alternating
order. Median reduction is 16.3% at 27,000 and 16.2% at 81,000. All category
p95 ratios are below 1.05; the largest is 1.013 for recovery at 81,000.
`native-81000-followup.json` records the anomaly and the follow-up protocol.

The initial separate-process Node WASM 27,000 runs varied substantially
(27.6–33.7 seconds baseline; 24.4–28.5 candidate), giving only 1.2% reduction
when dividing the two independent medians. These original results remain
in `wasm-27000/`. Before further measurements, the follow-up switched to
adjacent baseline/candidate execution at every decision, alternating which
binary runs first and warming up each equally. Three repetitions in Node
(`wasm-paired-27000/`) and Chromium (`chromium-27000/`) report the median of
paired reductions, with a 5% improvement threshold and p95 ratio <=1.05.
This reduces long host-load drift between arms; it does not erase the first
measurement. The Node 81,000 one-pair spot check improved 16.0%, with exact
parity, but is not a stable performance estimate.

The 49 native positions cover preview, ordinary play, forced replacement,
Perish Song and recovery. The WASM probe runs persistent agents through three
games; its positions depend on the budget, with all decisions and full root
policies compared exactly within each budget. The adjacent probe additionally
compares battle state, RNG and logs after each decision. The four-thread
native check preserves all 49 results.

`environment.json` records the host and toolchains. Browser measurements
run the identical frozen WASM binaries and generated binding code under
Chromium, with a minimal CommonJS byte-loading shim. The candidate Node and web WASM bytes
are identical. The real browser worker
and UI are tested separately. Timings include the search call, excluding
fixture generation, startup and compilation. They are local measurements,
not promises for every browser, server or CPU. The local-toolchain WASM payload grows by
32,064 bytes (6,734 bytes with gzip), recorded in `binary-size.json`. The browser and native probes
use different corpora; do not divide their times to claim native/WASM gain.

## Behavior and artifact provenance

The real web worker, evaluator worker, all 11 Node WASM suites and 19
Playwright tests passed. Browser checks verified all 576 catalog pairs (504
unique policies) and the exact 27,000/81,000/729,000 production budgets.
`local-validation.json` binds the commands and logs. Handoff HTML, Markdown,
navigation and the bundled ZIP were checked against the authoritative source.

All 335 native tests passed, including the frozen compact replay and full
conformance suites. `hash-audit.json` records byte-identical outputs for 60
fixture battles and 4,178 request states: stable 64/128-bit keys, all tested
transient hash modes, final state and full logs. The old output is preserved
as `baseline-hash-audit.json.gz`. Fixed-budget comparisons assert exact state,
selected choice and search statistics, not merely the same winner.

The 504 precomputed preview policies are retained after this exact-behavior
change. They were **not recomputed** at 729,000 iterations. The original
artifact is archived as `source-preview-artifact.json`, including its prior
catalog migration provenance, budgets, seeds and jobs hash. The dedicated
`tools/reuse-portable-preview.cjs` checks the pinned source/evidence hashes,
measured WASM binaries, and all 576 catalog pairs in both builds before
updating build/input metadata. Policies, keys and legal choices must match.
`preview-reuse-audit.json` binds this one migration to the measured sources;
normal stale-input rejection remains enabled.

Equal-budget behavior is preserved. The release does not increase the
27,000/81,000 battle budgets or 729,000 unknown-preview budget, and does not
claim a measured win-rate increase. A future equal-latency strength experiment
can spend the measured throughput gain on more iterations.

## Reproduction and downstream work

Build the deployed baseline and release in separate checkouts with identical
toolchains and flags. Copy the release's `perf_probe.rs` and `hash_audit.rs`
examples into the baseline checkout before compiling; neither is part of the
algorithm or the solver build identity. Preserve both binaries/binding
folders, then use `tools/perf-check.py`, `tools/perf-wasm-check.py`, and
`web/tests/portable-perf.mjs` with the budgets/repeats in their manifests.
Run one benchmark at a time, without builds or other tests competing for CPU.
The screening WASM probe before budget parameterization is archived as
`screens/perf-wasm-3000.cjs`; the native probe is unchanged between screens.

The authoritative native integration handoff is
[docs/NATIVE-FORK-HANDOFF.md](../../docs/NATIVE-FORK-HANDOFF.md). It describes
runtime discovery, the existing Rust protocol adapter, request lifecycle,
cache identity, native/WASM parity gates, concurrency measurement, and later
server-specific CPU/PGO experiments. The downstream deployment hardware has
not been verified here.

## Published verification

Deployment [40086597749](https://github.com/puniu3/nc2000-engine/actions/runs/40086597749)
published commit `4dd9d748f81472c30739afcef0bcfec6dc528456` successfully.
All 16 checked public assets matched the pre-deployment CI artifact exactly,
including the WASM, workers, data and handoff HTML/Markdown. The public ZIP's
inner WASM and handoff also matched. A fresh browser on the public URL used
the compiled preview (zero search iterations), completed the standard
27,000-iteration battle search, and reported no page errors. See
`public-deployment.json`. This final receipt changes documentation only.
