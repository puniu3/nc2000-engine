// Solver worker: one hand-entered position at a time, searched off the UI
// thread.
//
// Deliberately not a second life for `bot-worker.ts`. That worker owns a
// mirror battle that has to stay in lockstep with a live game for its whole
// length; this one owns nothing across positions — every `solve` builds a
// fresh `ProtocolSearcher`, hands it the position, and searches. The two
// lifecycles have no overlap worth sharing, and merging them would put a
// game invariant ("the mirror equals the real battle") next to code that has
// no real battle at all.
//
// The search is stepped in slices so progress ticks and `extend` can
// interleave: the user watches the estimate settle, and asks for more when
// it has not. `extend` keeps the SAME searcher — a deeper answer to the same
// question is more search, not a restart.

import init, { Dex, ProtocolSearcher } from "../../crates/wasm/pkg-web/nc2000_wasm";

export type SolverRequest =
  | {
      t: "solve";
      id: number;
      /** A `nc2000-position-v1` document. */
      spec: string;
      /** Belief candidate pool (the same file the blind bot uses). */
      poolJson: string;
      /** Optional community belief prior (M18), governing the fallback
       * imputation when no pool team fits. */
      priorJson?: string;
      seed: number;
      budget: number;
      /** Plies of searched line to report (0 = none). */
      plies: number;
    }
  | { t: "extend"; id: number; budget: number; plies: number }
  /** Answer now with whatever has been searched — the user pressing "stop"
   * wants the current estimate, not a discarded one. */
  | { t: "stop" }
  | { t: "cancel" };

export type SolverResponse =
  | { t: "ready" }
  | { t: "progress"; id: number; done: number; budget: number }
  | {
      t: "report";
      id: number;
      json: string;
      /** `stateView()` of the synthesized battle: the board the solver
       * actually read, which the screen shows back so a mistyped position is
       * visible rather than silently analyzed. */
      state: string;
      ms: number;
    }
  /** The position was refused — `message` names the field. */
  | { t: "rejected"; id: number; message: string }
  | { t: "error"; message: string };

const post = (m: SolverResponse) => self.postMessage(m);

let dex: Dex;
const ready = init().then(() => {
  dex = new Dex();
  post({ t: "ready" });
});

let searcher: ProtocolSearcher | null = null;
/** Bumped by cancel and by every new position: a running slice loop that no
 * longer matches stops posting. */
let gen = 0;
let stopping = false;
let plies = 0;
let seed = 1;

self.onmessage = (e: MessageEvent<SolverRequest>) => {
  void handle(e.data).catch((err) => post({ t: "error", message: String(err) }));
};

async function handle(m: SolverRequest): Promise<void> {
  await ready;
  switch (m.t) {
    case "cancel":
      gen += 1;
      break;
    case "stop":
      stopping = true;
      break;
    case "solve": {
      gen += 1;
      stopping = false;
      searcher?.free();
      searcher = null;
      seed = m.seed >>> 0;
      plies = m.plies;
      const s = new ProtocolSearcher(dex, JSON.parse(m.spec).side ?? 0, m.poolJson, seed);
      if (m.priorJson) s.setBeliefPrior(m.priorJson);
      try {
        s.setPosition(m.spec);
      } catch (err) {
        s.free();
        post({ t: "rejected", id: m.id, message: cause(err) });
        return;
      }
      searcher = s;
      await run(m.id, m.budget, gen);
      break;
    }
    case "extend": {
      if (!searcher) return;
      stopping = false;
      plies = m.plies;
      await run(m.id, m.budget, gen);
      break;
    }
  }
}

/** Adaptive slice, same rule as the game worker: aim for ~125 ms of work per
 * call so progress ticks several times a second on any device and a cancel
 * is never more than one slice away. */
const SLICE_TARGET_MS = 125;
let slice = 250;

async function run(id: number, budget: number, myGen: number): Promise<void> {
  const s = searcher;
  if (!s) return;
  const t0 = performance.now();
  let done = s.iterations();
  while (done < budget && gen === myGen && !stopping) {
    const n = Math.min(slice, budget - done);
    const t = performance.now();
    done = s.step(n);
    const dt = performance.now() - t;
    if (n >= slice) {
      const factor = Math.max(0.5, Math.min(2, SLICE_TARGET_MS / Math.max(dt, 1)));
      slice = Math.round(Math.max(50, Math.min(4000, slice * factor)));
    }
    post({ t: "progress", id, done, budget });
    await new Promise((r) => setTimeout(r, 0));
  }
  if (gen !== myGen) return;
  post({
    t: "report",
    id,
    json: s.report(plies, seed),
    state: s.stateView(),
    ms: performance.now() - t0,
  });
}

/** wasm-bindgen throws `Error`s whose message is the Rust string; anything
 * else is stringified rather than swallowed. */
function cause(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
