// Bot search worker. Owns its own wasm instance and a mirror Battle kept in
// lockstep with the main thread's (same teams + seed + choice sequence =
// identical state; the engine PRNG is part of the state). Searches run in
// step() slices with a macrotask yield in between so progress posts flush
// and cancel/flush messages can interleave — the UI thread never blocks.
//
// Ponder (M9c): a search launched with `ponder: true` (a simultaneous
// request point — the human also owes a pick) does not stop at its budget.
// Budget iterations are the required think; past them it keeps stepping —
// free bonus strength while the human deliberates — up to PONDER_CAP x
// budget (bounds memory). A `flush` message (human committed) makes it
// return `best()` at the next slice boundary; if the flush arrives before
// the budget is met, the required think still completes first. Bot-only
// points (`ponder: false`) stop exactly at budget, as before.
//
// Open team sheet (M12, the single information policy): every game gets a
// per-game BlindSearcher whose belief is PINNED to the opponent's true sets
// (`pinOpponent`) — the bot knows the human's team exactly (as the human
// knows the bot's, from the team list), while the human's SELECTION (which
// 3 of 6 + lead, until revealed) stays hidden: the searcher determinizes
// unseen pick identities per iteration. The mirror battle runs log-ON (the
// observer's trace-free reveal channel reads it). Per search: observe()
// feeds the mirror, then either the baked preview answers instantly (src
// "table" — the pair is resolved by public signature, no identification
// condition) or the stepped search ponders (src "search").

import init, { Dex, Battle, Searcher, BlindSearcher } from "../../crates/wasm/pkg-web/nc2000_wasm";

export type WorkerRequest =
  | {
      t: "battle";
      p1: string;
      p2: string;
      seed: string;
      /** Open-team-sheet searcher config (always present: single policy). */
      open: { poolJson: string; side: number; seed: number };
    }
  | { t: "pair"; json: string }
  | { t: "apply"; picks: [number, string][] }
  | {
      t: "search";
      id: number;
      side: number;
      budget: number;
      seed: number;
      ponder: boolean;
    }
  | { t: "flush" }
  | { t: "cancel" }
  | {
      t: "bench";
      id: number;
      p1: string;
      p2: string;
      seed: string;
      searchSeed: number;
      iters: number;
    };

export type WorkerResponse =
  | { t: "ready" }
  | { t: "progress"; id: number; done: number; budget: number }
  | {
      t: "result";
      id: number;
      best: string | null;
      policy: string;
      ms: number;
      /** Where the pick came from (preview: table/search). */
      src?: "table" | "search";
    }
  | { t: "benchProgress"; id: number; done: number; total: number; ms: number }
  | { t: "benchResult"; id: number; iters: number; ms: number }
  | { t: "error"; message: string };

const post = (m: WorkerResponse) => self.postMessage(m);

let dex: Dex;
const ready = init().then(() => {
  dex = new Dex();
  post({ t: "ready" });
});

let battle: Battle | null = null;
let searcher: BlindSearcher | null = null;
let gen = 0; // bumped whenever the battle state moves on -> running searches abort
let flushed = false; // human committed: stop pondering at the next slice

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  void handle(e.data).catch((err) =>
    post({ t: "error", message: String(err) }),
  );
};

async function handle(m: WorkerRequest): Promise<void> {
  await ready;
  switch (m.t) {
    case "battle": {
      gen += 1;
      searcher?.free();
      searcher = null;
      battle?.free();
      battle = new Battle(dex, m.p1, m.p2, m.seed);
      // Keep the protocol log ON — the observer's trace-free reveal
      // channel (Leftovers / Focus Band / Sleep Talk) reads it.
      battle.setLogEnabled(true);
      searcher = new BlindSearcher(
        battle,
        m.open.side,
        m.open.poolJson,
        m.open.seed >>> 0,
      );
      // Open team sheet: pin the belief to the opponent's true sets.
      searcher.pinOpponent(m.open.side === 0 ? m.p2 : m.p1);
      break;
    }
    case "pair":
      try {
        searcher?.addPair(m.json);
      } catch (e) {
        console.warn("pair table rejected:", e);
      }
      break;
    case "apply":
      gen += 1;
      for (const [side, input] of m.picks) battle!.applyChoice(side, input);
      break;
    case "cancel":
      gen += 1;
      break;
    case "flush":
      flushed = true;
      break;
    case "search":
      await runSearch(m);
      break;
    case "bench":
      await runBench(m);
      break;
  }
}

const PONDER_CAP = 10; // max total think = cap x budget

// Adaptive slice size: target ~125 ms per step() call so progress ticks
// ~8x/s on any device (>=4x/s even when a slice overshoots 2x), and cancel
// latency stays bounded. Shared across searches — device speed is stable.
const SLICE_TARGET_MS = 125;
let slice = 250;

function stepAdaptive(s: { step(n: number): number }, n: number): number {
  const t0 = performance.now();
  const done = s.step(n);
  const dt = performance.now() - t0;
  if (n >= slice) {
    // only full slices inform the estimate
    const factor = Math.max(0.5, Math.min(2, SLICE_TARGET_MS / Math.max(dt, 1)));
    slice = Math.round(Math.max(50, Math.min(4000, slice * factor)));
  }
  return done;
}

interface SearchMsg {
  id: number;
  side: number;
  budget: number;
  seed: number;
  ponder: boolean;
}

// One decision point on the persistent per-game searcher: observe()
// snapshots the mirror's state (updating the pinned belief's observations),
// then either the baked preview answers instantly or the stepped search
// runs the ponder loop. The searcher is NOT freed per decision — it carries
// the game's accumulated observations.
async function runSearch(m: SearchMsg): Promise<void> {
  const myGen = gen;
  flushed = false;
  const cap = m.budget * PONDER_CAP;
  const s = searcher!;
  s.observe(battle!);
  const t0 = performance.now();
  const baked = s.bakedPreview();
  if (baked !== undefined) {
    post({
      t: "result",
      id: m.id,
      best: baked,
      policy: s.rootPolicy(),
      ms: performance.now() - t0,
      src: "table",
    });
    return;
  }
  let done = 0;
  for (;;) {
    if (gen !== myGen) return; // superseded: next observe() resets the search
    // Required think first; then ponder until flushed or capped.
    const target = !m.ponder || flushed ? m.budget : cap;
    if (done >= target) break;
    done = stepAdaptive(s, Math.min(slice, target - done));
    post({ t: "progress", id: m.id, done, budget: m.budget });
    // Macrotask yield: flush the progress post, let cancel/flush interleave.
    await new Promise((r) => setTimeout(r, 0));
  }
  if (gen !== myGen) return;
  post({
    t: "result",
    id: m.id,
    best: s.best() ?? null,
    policy: s.rootPolicy(),
    ms: performance.now() - t0,
    src: "search",
  });
}

// Device benchmark: a fixed, deterministic workload (fixed teams + battle
// seed + searcher seed + iteration count => every device runs the identical
// iteration stream), independent of the mirror battle. Measures in-battle
// root throughput — the M9 gate quantity — on the perfect-info Searcher
// (the historical workload: numbers stay comparable across devices and
// releases).
async function runBench(m: {
  id: number;
  p1: string;
  p2: string;
  seed: string;
  searchSeed: number;
  iters: number;
}): Promise<void> {
  const b = new Battle(dex, m.p1, m.p2, m.seed);
  let s: Searcher | null = null;
  try {
    b.setLogEnabled(false);
    // Fixed preview picks: land on the first in-battle decision point.
    // Max-Total-Level-legal for the fixed bench teams (pool 0: 52/52/52/51…,
    // pool 1: 55/55/50/50…) — changed 2026-07-17 with the 155-cap fix, so
    // benchmark numbers are comparable only within the post-fix workload.
    b.applyChoice(0, "team 1, 2, 4");
    b.applyChoice(1, "team 1, 3, 4");
    s = new Searcher(b, 1, m.searchSeed >>> 0);
    let done = 0;
    let ms = 0;
    while (done < m.iters) {
      // Timing excludes the yields (postMessage/setTimeout overhead is not
      // search throughput; the in-game loop pays it, the gate number no).
      const t1 = performance.now();
      done = s.step(Math.min(500, m.iters - done));
      ms += performance.now() - t1;
      post({ t: "benchProgress", id: m.id, done, total: m.iters, ms });
      await new Promise((r) => setTimeout(r, 0));
    }
    post({ t: "benchResult", id: m.id, iters: done, ms });
  } finally {
    s?.free();
    b.free();
  }
}
