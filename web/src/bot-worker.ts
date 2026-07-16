// Bot search worker. Owns its own wasm instance and a mirror Battle kept in
// lockstep with the main thread's (same teams + seed + choice sequence =
// identical state; the engine PRNG is part of the state). Searches run in
// step() slices with a macrotask yield in between so progress posts flush
// and cancel messages can interleave — the UI thread never blocks.

import init, {
  Dex,
  Battle,
  Searcher,
} from "../../crates/wasm/pkg-web/nc2000_wasm";

export type WorkerRequest =
  | { t: "battle"; p1: string; p2: string; seed: string }
  | { t: "apply"; picks: [number, string][] }
  | { t: "search"; id: number; side: number; budget: number; seed: number }
  | { t: "cancel" };

export type WorkerResponse =
  | { t: "ready" }
  | { t: "progress"; id: number; done: number; budget: number }
  | { t: "result"; id: number; best: string | null; policy: string; ms: number }
  | { t: "error"; message: string };

const post = (m: WorkerResponse) => self.postMessage(m);

let dex: Dex;
const ready = init().then(() => {
  dex = new Dex();
  post({ t: "ready" });
});

let battle: Battle | null = null;
let gen = 0; // bumped whenever the battle state moves on -> running searches abort

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
    case "search":
      await runSearch(m);
      break;
  }
}

const SLICE = 250;

async function runSearch(m: {
  id: number;
  side: number;
  budget: number;
  seed: number;
}): Promise<void> {
  const myGen = gen;
  const s = new Searcher(battle!, m.side, m.seed >>> 0);
  const t0 = performance.now();
  try {
    let done = 0;
    while (done < m.budget) {
      if (gen !== myGen) return; // superseded: drop silently
      done = s.step(Math.min(SLICE, m.budget - done));
      post({ t: "progress", id: m.id, done, budget: m.budget });
      // Macrotask yield: flush the progress post, let cancels interleave.
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
