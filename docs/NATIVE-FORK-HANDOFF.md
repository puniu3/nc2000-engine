# Native search handoff for downstream forks

This document is for the maintainer or AI implementing native search in a
fork. Upstream's browser still runs portable WebAssembly. A server-native
adapter and server-specific compiler tuning belong in the fork; they are
not prerequisites for using upstream's portable optimizations.

## Portable changes available upstream

The portable release memoizes an exact Pokemon hash prefix in a bounded
12-entry thread-local cache and folds empty effect payloads without changing
the original hash stream. Every prefix field is compared before reuse;
there are no mutation hooks for a native adapter to maintain. Search rules,
seeds, budgets and the prior are unchanged.

The [measurement and parity record](https://github.com/puniu3/nc2000-engine/tree/master/findings/portable-search-v1)
contains frozen baseline provenance, accepted and rejected experiments,
fixed-budget comparisons, and the explicit compiled-preview reuse audit.
Start from the deployed release on upstream master, not an unpublished
research branch. Benchmark native integration against this portable release.

## Establish the actual execution environment

Check the running command, revision, runtime, CPU architecture/features,
CPU quota, memory limit and expected concurrent games. Hosting a static
browser app on a rental server does not execute its search there. In the
public `hm0615g/nc2000-engine` fork, the inspected `review-local-changes`
revision `6de518e6a330ef4ce237daeb95f6115348b2cf53` loads
`crates/wasm/pkg-node/nc2000_wasm.js` from `tools/ps-client.js`. Its real
deployment and hardware were not verified upstream.

Preserve the fork's transport, authentication, request cancellation, team
selection, logging and mode-specific behavior. Merge the upstream performance
changes before comparing runtimes, so the comparison uses one algorithm.

## Reuse the existing Rust implementation

The implementation behind `ProtocolSearcher` is
`nc2000_bot::import::ProtocolAgent`. Treat
`crates/wasm/src/lib.rs::WasmProtocolSearcher` as the authoritative adapter
specification. Do not translate the search algorithm into another language.

The ordinary lifecycle is:

1. Load one `Dex` and the actual configured opponent `MetaPool`. Construct
   `ProtocolAgent::new(&dex, side, pool, cfg, seed)` once per battle, then
   `set_own_team`. Apply any explicitly enabled open-sheet or custom-prior
   settings at the same point as the existing WASM adapter.
2. Feed each newly visible protocol line through `push_line` in order.
   Pass the corresponding request JSON to `on_request`.
3. If `on_request` returns true, run `step(&dex, n)` in bounded chunks until
   the exact requested iteration budget is reached. Its return value is
   the cumulative iteration count. Forced or baked decisions may make no
   progress; handle them as the current adapter does instead of looping.
4. Read `best(&dex)` and return the canonical Showdown input. Bind it to
   the battle ID, request ID and a cancellation generation. Submit it only
   if that request is still current.
5. Keep the agent alive between requests. Reconnection rebuilds it from
   the same visible history, own team, configuration and seed, following
   the fork's existing resume checks.

For the ordinary blind configuration, mirror `skuct_config`: use UCB,
`c` from `data/search-profiles.json` (currently 0.4), and 16 HP buckets,
with the other `RmConfig` defaults unchanged. Plain `RmConfig::default()`
is not that configuration. Mirror JavaScript's unsigned 32-bit seed
conversion before passing the value as Rust `u64`.

Budget policy is separate from the runtime. `data/live-search.json` currently
contains 27,000/81,000 battle iterations and 729,000 for an unknown preview.
The CLI's selected flags may specify a different existing policy. Preserve
that policy during migration; do not make human input timing, chunk size or
thread scheduling change the iteration count.

A persistent Rust subprocess with a versioned JSON-lines protocol is a
small integration option: initialize, feed lines/request, step or search,
cancel, inspect, and close a battle. Include request identity in every
response, reserve stdout for protocol messages, and keep diagnostics on
stderr. A native Node module is also possible. Keep the process/module
boundary outside the individual simulation iterations. Use bounded worker
capacity and preserve the current reconnect/cancellation guarantees.

The event callback lookup assumes one consistent dex per process. Do not
mix independently reindexed dex definitions inside the same worker.

## Preserve preview artifacts and cache identity

The compiled preview policy is not a generic battle-result cache. Use
`compiled_preview::key` and its build fingerprint, and preserve the existing
selection seed mapping. Feed the observer/request lifecycle even on a
preview cache hit. Consult upstream's `preview-cache.html` and README API
section before reusing the Node cache helper.

The existing server cache binds to the actual WASM binary. A native adapter
must use a separate, versioned identity bound to its native binary and all
the same rules, dex, prior, search configuration and information-set inputs.
Do not put a native hash into a field claiming to be a WASM hash, share
entries solely because the source revision matches, or relax stale-build
checks. Keep provenance when migrating a compiled artifact.

## Demonstrate parity before tuning

Replay identical visible requests through the old WASM adapter and the native
adapter with the same data, seeds and exact budgets. Include preview cache
hits/misses, ordinary moves, forced switches, custom-team fallback, recovery,
Perish Song, reconnect and cancellation. Compare reconstructed state,
belief, completed iterations, root visits/rewards and selected input.
Require zero unexplained legality drift or projected choices.

Run the existing native conformance tests and the WASM Node suites. The
portable performance probe and hash audit are useful additional controls,
but they do not replace real protocol-adapter tests. Test multiple games and
worker counts; one game must retain deterministic search order. Parallel
independent games measure server capacity, not the speedup of one decision.

Measure end-to-end response latency (including queueing and IPC), search
throughput, p95 latency and peak memory at the configured production budgets.
Alternate baseline/candidate order and repeat on the actual server under
both isolated and expected concurrent load. Report runtime/compiler/CPU,
revision, data hashes, budget, seeds and failures with the timings.

## Server-specific follow-up experiments

After generic native execution passes parity, compare CPU targeting and PGO
as separate changes. `-C target-cpu=native` targets the build host, so it
must not accidentally target a developer Mac for a Linux server. On a VM,
use the features actually exposed to that VM and account for host migration.

Train PGO on representative searches and evaluate on held-out cases; keep
the training and optimized build flags consistent. See the official Rust
[PGO workflow](https://doc.rust-lang.org/rustc/profile-guided-optimization.html)
and [CPU option](https://doc.rust-lang.org/rustc/codegen-options/index.html#target-cpu).
Upstream's fat LTO, one codegen unit and native mimalloc are already enabled.
Do not count those existing settings as new gains.

Do not introduce fast-math, approximate simulation, a changed prior,
different masks, root ensembles or shared-tree scheduling into the runtime
parity comparison. Those can change decisions and need separate playing
strength evaluation. Any later iteration-budget increase also needs its
own equal-latency strength check. No speedup or win-rate improvement is
promised for hardware that upstream has not measured.
