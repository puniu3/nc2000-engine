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

import init, {
  Dex,
  Battle,
  Searcher,
} from "../../crates/wasm/pkg-web/nc2000_wasm";

export type WorkerRequest =
  | { t: "battle"; p1: string; p2: string; seed: string }
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
  | { t: "result"; id: number; best: string | null; policy: string; ms: number }
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
    case "battle":
      gen += 1;
      battle?.free();
      battle = new Battle(dex, m.p1, m.p2, m.seed);
      battle.setLogEnabled(false);
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

function stepAdaptive(s: Searcher, n: number): number {
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

async function runSearch(m: {
  id: number;
  side: number;
  budget: number;
  seed: number;
  ponder: boolean;
}): Promise<void> {
  const myGen = gen;
  flushed = false;
  const cap = m.budget * PONDER_CAP;
  const s = new Searcher(battle!, m.side, m.seed >>> 0);
  const t0 = performance.now();
  try {
    let done = 0;
    for (;;) {
      if (gen !== myGen) return; // superseded: drop silently
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
    });
  } finally {
    s.free();
  }
}

// Device benchmark: a fixed, deterministic workload (fixed teams + battle
// seed + searcher seed + iteration count => every device runs the identical
// iteration stream), independent of the mirror battle. Measures in-battle
// root throughput — the M9 gate quantity.
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
    b.applyChoice(0, "team 1, 2, 3");
    b.applyChoice(1, "team 1, 2, 3");
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
