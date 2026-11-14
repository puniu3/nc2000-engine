// One bot-vs-bot trial at a time over `ForkArena` (the native
// fork_counterfactual trial, compiled to wasm). The main thread deals trial
// indices; rows come back per arm, then `done` once every arm has played.

import init, { Dex, ForkArena } from "../../crates/wasm/pkg-web/nc2000_wasm";

export type ArenaRequest =
  | { t: "start"; fork: string; pool: string; arena: string; arms: number }
  | { t: "trial"; trial: number };

export type ArenaResponse =
  | { t: "ready" }
  | { t: "row"; trial: number; arm: number; row: string; ms: number }
  | { t: "done"; trial: number }
  | { t: "error"; message: string };

const post = (m: ArenaResponse) => self.postMessage(m);
let arena: ForkArena | null = null;
let arms = 0;
const ready = init();

self.onmessage = (e: MessageEvent<ArenaRequest>) => {
  void handle(e.data).catch((err) => post({ t: "error", message: String(err) }));
};

async function handle(m: ArenaRequest) {
  await ready;
  if (m.t === "start") {
    arena?.free();
    arena = null;
    const dex = new Dex();
    try {
      arena = new ForkArena(dex, m.fork, m.pool, m.arena);
    } finally {
      dex.free();
    }
    arms = m.arms;
    post({ t: "ready" });
    return;
  }
  for (let arm = 0; arm < arms; arm++) {
    const t0 = performance.now();
    const row = arena!.play(m.trial, arm);
    post({ t: "row", trial: m.trial, arm, row, ms: performance.now() - t0 });
  }
  post({ t: "done", trial: m.trial });
}
