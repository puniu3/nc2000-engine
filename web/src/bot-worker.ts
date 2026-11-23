// The live bot worker receives only its own submitted team and PlayerChannel
// frames. No opponent true team, game RNG or submitted hidden choices enter
// this worker. Historical fork reconstruction retains its recorded policy.
// Search runs in fixed-iteration slices; timing never changes the budget.

import init, {
  Dex,
  Battle,
  PlayerChannel,
  ProtocolSearcher,
} from "../../crates/wasm/pkg-web/nc2000_wasm";
import { searchProfile } from "./search-profile";
import type { ActionPolicyMode, PlayPolicy } from "./types";
import type { EndgameSettings } from "./endgame-settings";
import { runEndgameSearch } from "./endgame-search";

export type WorkerRequest =
  | {
      t: "battle";
      ownTeam: string;
      /** Per-game searcher config (always present). `priorJson` is the raw
       * text of a belief-prior table. */
      searcher: {
        poolJson: string;
        side: number;
        seed: number;
        ruleset?: string;
        priorJson?: string;
        pickPriorJson?: string;
        adaptiveBudget?: boolean;
        endgame?: EndgameSettings;
        actionPolicy?: ActionPolicyMode;
      };
    }
  /** A forked battle (`?fork`): the mirror is `Battle.fromFork(fork, seed)`
   * and the bot is the ladder's ProtocolSearcher installed at the fork's
   * position, fed its own player stream from there on. */
  | {
      t: "fork";
      fork: string;
      seed: number;
      searcher: {
        poolJson: string;
        side: number;
        seed: number;
        mode: "open" | "blind";
        endgame?: EndgameSettings;
        actionPolicy?: ActionPolicyMode;
      };
    }
  | { t: "apply"; picks: [number, string][] }
  | {
      t: "search";
      id: number;
      side: number;
      budget: number;
      seed: number;
      ponder: boolean;
      frame?: PlayerFrame;
    }
  | { t: "flush" }
  | { t: "cancel" };

export interface PlayerFrame {
  lines: string[];
  request: unknown;
  legal_actions: string[];
}

export type WorkerResponse =
  | { t: "ready" }
  | { t: "progress"; id: number; done: number; budget: number }
  | {
      t: "result";
      id: number;
      best: string | null;
      policy: string;
      playPolicy: PlayPolicy | null;
      ms: number;
      wasmMemoryBytes: number;
      endgame?: string;
      /** Live fixed-budget search. */
      src?: "search";
    }
  /** Raw `setBeliefPrior` report JSON, posted once per battle that carried a
   * `priorJson` — applied or refused, the caller gets the verdict rather
   * than the bot silently playing without the table the user chose. */
  | { t: "prior"; report: string }
  | { t: "pickPrior"; report: string }
  /** Live battles: the bot's read, posted right after each observe(). `info` =
   * `beliefInfo()`, `prior` = `beliefPriorInfo()`, both raw JSON. */
  | { t: "belief"; info: string; prior: string }
  | { t: "error"; message: string };

const post = (m: WorkerResponse) => self.postMessage(m);

let dex: Dex;
let wasmMemory: WebAssembly.Memory;
const ready = init().then(exports => {
  wasmMemory = exports.memory;
  dex = new Dex();
  post({ t: "ready" });
});

let battle: Battle | null = null;
let actionPolicy: ActionPolicyMode = "legacy";
let forked: {
  channel: PlayerChannel | null;
  searcher: ProtocolSearcher;
  positioned: boolean;
  endgame?: EndgameSettings;
  live?: boolean;
  adaptiveBudget?: boolean;
} | null = null;

function dropForked() {
  forked?.channel?.free();
  forked?.searcher.free();
  forked = null;
}
let gen = 0; // bumped whenever the battle state moves on -> running searches abort
let flushed = false; // only the opt-in timed experiment observes this
let forkMode: "open" | "blind" = "blind"; // a forked battle's recorded policy

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  void handle(e.data).catch((err) =>
    post({ t: "error", message: String(err) }),
  );
};

async function handle(m: WorkerRequest): Promise<void> {
  await ready;
  switch (m.t) {
    case "battle": {
      actionPolicy = m.searcher.actionPolicy ?? "legacy";
      gen += 1;
      dropForked();
      battle?.free();
      battle = null;
      const ps = m.searcher.ruleset
        ? ProtocolSearcher.withRuleset(dex, m.searcher.side, m.searcher.poolJson, m.searcher.seed >>> 0, searchProfile("blind").c, undefined, m.searcher.ruleset)
        : new ProtocolSearcher(dex, m.searcher.side, m.searcher.poolJson, m.searcher.seed >>> 0, searchProfile("blind").c);
      ps.setOwnTeam(m.ownTeam);
      if (m.searcher.priorJson) post({ t: "prior", report: ps.setBeliefPrior(m.searcher.priorJson) });
      if (m.searcher.pickPriorJson) post({ t: "pickPrior", report: ps.setPickPrior(m.searcher.pickPriorJson) });
      forked = { channel: null, searcher: ps, positioned: false, live: true, adaptiveBudget: m.searcher.adaptiveBudget, endgame: m.searcher.endgame };
      break;
    }
    case "fork": {
      actionPolicy = m.searcher.actionPolicy ?? "legacy";
      gen += 1;
      forkMode = m.searcher.mode;
      dropForked();
      battle?.free();
      battle = Battle.fromFork(dex, m.fork, m.seed >>> 0);
      const ps = ProtocolSearcher.fromFork(
        dex,
        m.fork,
        m.searcher.poolJson,
        m.searcher.seed >>> 0,
        searchProfile(forkMode).c,
        m.searcher.side,
      );
      forked = {
        channel: new PlayerChannel(m.searcher.side),
        searcher: ps,
        positioned: true,
        endgame: m.searcher.endgame,
      };
      break;
    }
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
      await runForkSearch(m);
      break;
  }
}


// Adaptive slice size: target ~125 ms per step() call so progress ticks
// ~8x/s on any device (>=4x/s even when a slice overshoots 2x), and cancel
// latency stays bounded. Shared across searches — device speed is stable.
const SLICE_TARGET_MS = 125;
let slice = 250;

function stepSlice(step: (n: number) => number, n: number): number {
  const t0 = performance.now();
  const done = step(n);
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
  frame?: PlayerFrame;
}

// Commit only after the search finishes. Polls/progress never redraw. The
// protocol path maps the entire distribution to engine input syntax so the
// reported probabilities describe exactly the submitted action space.
function finishDecision(s: { commitDecision(mode: string): string }, protocolSide?: number): {
  best: string | null; playPolicy: PlayPolicy | null;
} {
  const decision = JSON.parse(s.commitDecision(actionPolicy)) as { input: string; policy: PlayPolicy } | null;
  if (!decision) return { best: null, playPolicy: null };
  const resolve = (input: string) => {
    if (protocolSide === undefined) return input;
    const resolved = battle!.resolveChoice(protocolSide, input);
    if (!resolved) throw new Error(`Cannot resolve policy action: ${input}`);
    return resolved;
  };
  const probabilities = new Map<string, number>();
  for (const row of decision.policy.actions) {
    const input = resolve(row.input);
    probabilities.set(input, (probabilities.get(input) ?? 0) + row.probability);
  }
  return { best: resolve(decision.input), playPolicy: { source: decision.policy.source,
    actions: [...probabilities].map(([input, probability]) => ({ input, probability })) } };
}

// The forked bot's decision: catch the searcher up on the player stream
// since its last decision, then the same fixed-budget loop. At the fork itself the
// searcher is already installed from the position and no line has arrived.
async function runForkSearch(m: SearchMsg): Promise<void> {
  const t0 = performance.now();
  const myGen = gen;
  flushed = false;
  const f = forked!;
  const frame: PlayerFrame = m.frame ?? JSON.parse(f.channel!.frame(battle!));
  const atFork = f.positioned && frame.lines.length === 0;
  f.positioned = false;
  if (!atFork) {
    f.searcher.pushLines(JSON.stringify(frame.lines));
    if (!f.searcher.onRequest(JSON.stringify(frame.request)))
      throw new Error("fork: search requested on a wait request");
  }
  if (f.live) post({ t: "belief", info: f.searcher.beliefInfo(), prior: f.searcher.beliefPriorInfo() });
  if (f.endgame) {
    const current = await runEndgameSearch(f.searcher, f.endgame, {
      cancelled: () => gen !== myGen,
      stop: () => flushed,
      ponder: m.ponder,
      progress: done => post({ t: "progress", id: m.id, done, budget: m.budget }),
    }, t0);
    if (!current) return;
    post({ t: "result", id: m.id, ...finishDecision(f.searcher, f.live ? undefined : m.side),
      policy: f.searcher.rootPolicy(), ms: performance.now() - t0, wasmMemoryBytes: wasmMemory.buffer.byteLength, src: "search", endgame: f.searcher.endgameStats() });
    return;
  }
  const step = f.adaptiveBudget && actionPolicy === "legacy"
    ? (n: number) => f.searcher.stepAdaptiveBattle(n)
    : (n: number) => f.searcher.step(n);
  let done = 0;
  for (;;) {
    if (gen !== myGen) return;
    const target = m.budget;
    if (done >= target) break;
    const next = stepSlice(step, Math.min(slice, target - done));
    if (next === done) break; // forced or adaptively settled
    done = next;
    post({ t: "progress", id: m.id, done, budget: m.budget });
    await new Promise((r) => setTimeout(r, 0));
  }
  if (gen !== myGen) return;
  post({
    t: "result",
    id: m.id,
    ...finishDecision(f.searcher, f.live ? undefined : m.side),
    policy: f.searcher.rootPolicy(),
    ms: performance.now() - t0,
    wasmMemoryBytes: wasmMemory.buffer.byteLength,
    src: "search",
  });
}
