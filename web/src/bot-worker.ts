// Bot search worker. Owns its own wasm instance and a mirror Battle kept in
// lockstep with the main thread's (same teams + seed + choice sequence =
// identical state; the engine PRNG is part of the state). Searches run in
// step() slices with a macrotask yield in between so progress posts flush
// and cancel/flush messages can interleave — the UI thread never blocks.
//
// Live searches start during the human's selection and stop at a fixed
// iteration count. Input timing never adds or removes iterations.
//
// Information policy: every live game is blind. The per-game BlindSearcher
// sees what the human also sees — the opponent's six species/levels/types
// and the public log. The belief identifies the opponent within the shipped
// prior and, when no prior team is consistent (a custom team), falls back
// to imputation on a synthesized roster. An optional community belief prior
// (`searcher.priorJson`, M18) governs that fallback; it is installed in the
// same synchronous block that constructs the searcher, which is what makes
// the "after construction, before the first observe()" window
// `setBeliefPrior` demands structural rather than incidental. The search
// uses the blind profile (data/search-profiles.json).
//
// A forked battle (`?fork`) keeps the information policy its record was
// made under, including the retired open-sheet policy, and that policy's
// profile.
//
// The mirror battle runs log-ON (the observer's trace-free reveal channel
// reads it). Per search: observe() feeds the mirror, then the stepped
// search runs to its fixed count.

import init, {
  Dex,
  Battle,
  BlindSearcher,
  PlayerChannel,
  ProtocolSearcher,
  embeddedDexJson,
} from "../../crates/wasm/pkg-web/nc2000_wasm";
import { searchProfile } from "./search-profile";
import { compiledPreviewMatches, compiledPreviewPick } from "./compiled-preview";
import type { ActionPolicyMode, PlayPolicy } from "./types";
import type { EndgameSettings } from "./endgame-settings";
import { runEndgameSearch } from "./endgame-search";

export type WorkerRequest =
  | {
      t: "battle";
      p1: string;
      p2: string;
      seed: string;
      /** Per-game searcher config (always present). `priorJson` is the raw
       * text of a belief-prior table. */
      searcher: {
        poolJson: string;
        side: number;
        seed: number;
        priorJson?: string;
        compiledPreview?: boolean;
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
    }
  | { t: "flush" }
  | { t: "cancel" };

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
      /** Known preview lookup or a live fixed-budget search. */
      src?: "table" | "search";
    }
  /** Raw `setBeliefPrior` report JSON, posted once per battle that carried a
   * `priorJson` — applied or refused, the caller gets the verdict rather
   * than the bot silently playing without the table the user chose. */
  | { t: "prior"; report: string }
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
let searcher: BlindSearcher | null = null;
let compiledAllowed: Promise<boolean> = Promise.resolve(false);
let previewSeed = 0;
let actionPolicy: ActionPolicyMode = "legacy";
let forked: {
  channel: PlayerChannel;
  searcher: ProtocolSearcher;
  positioned: boolean;
  endgame?: EndgameSettings;
  live?: boolean;
} | null = null;

function dropForked() {
  forked?.channel.free();
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
      previewSeed = m.searcher.seed;
      compiledAllowed = actionPolicy !== "legacy" || m.searcher.compiledPreview === false || m.searcher.priorJson
        ? Promise.resolve(false) : compiledPreviewMatches(m.searcher.poolJson, embeddedDexJson()).catch(() => false);
      searcher?.free();
      searcher = null;
      dropForked();
      battle?.free();
      battle = new Battle(dex, m.p1, m.p2, m.seed);
      // Keep the protocol log ON — the observer's trace-free reveal
      // channel (Leftovers / Focus Band / Sleep Talk) reads it.
      battle.setLogEnabled(true);
      if (m.searcher.endgame) {
        const ps = new ProtocolSearcher(dex, m.searcher.side, m.searcher.poolJson, m.searcher.seed >>> 0, searchProfile("blind").c);
        ps.setOwnTeam(m.searcher.side === 0 ? m.p1 : m.p2);
        if (m.searcher.priorJson) post({ t: "prior", report: ps.setBeliefPrior(m.searcher.priorJson) });
        forked = { channel: new PlayerChannel(m.searcher.side), searcher: ps, positioned: false, live: true, endgame: m.searcher.endgame };
        break;
      }
      searcher = new BlindSearcher(
        battle,
        m.searcher.side,
        m.searcher.poolJson,
        m.searcher.seed >>> 0,
        searchProfile("blind").c,
      );
      if (m.searcher.priorJson) {
        // The community prior governs the fallback imputation (the
        // hidden-custom-team branch of the determinizer). setBeliefPrior only
        // accepts the window "after construction, before the first
        // observe()" — and this runs in the same synchronous block as the
        // constructor above, ahead of any observe(), which happens solely in
        // runSearch. A later `search` message cannot overtake it either:
        // handle() awaits nothing but the shared `ready`, so once that is
        // settled the queued handlers resume in message-arrival order. The
        // call reports instead of throwing; forward the verdict verbatim.
        post({
          t: "prior",
          report: searcher.setBeliefPrior(m.searcher.priorJson),
        });
      }
      break;
    }
    case "fork": {
      actionPolicy = m.searcher.actionPolicy ?? "legacy";
      gen += 1;
      forkMode = m.searcher.mode;
      searcher?.free();
      searcher = null;
      dropForked();
      battle?.free();
      battle = Battle.fromFork(dex, m.fork, m.seed >>> 0);
      const ps = ProtocolSearcher.fromFork(
        dex,
        m.fork,
        m.searcher.poolJson,
        m.searcher.seed >>> 0,
        searchProfile(forkMode).c,
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
      if (forked) await runForkSearch(m);
      else await runSearch(m);
      break;
  }
}


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

// One decision point on the persistent per-game searcher: observe()
// snapshots the mirror's state (updating the belief's observations), then
// the stepped search runs to its fixed count. The searcher is NOT freed per
// decision — it carries the game's accumulated observations.
async function runSearch(m: SearchMsg): Promise<void> {
  const myGen = gen;
  flushed = false;
  const s = searcher!;
  const allowCompiled = await compiledAllowed;
  if (gen !== myGen) return;
  s.observe(battle!);
  // That observe() is what re-filtered the belief and rebuilt the fallback
  // roster the prior drives, so this is the first instant either read means
  // anything at this decision point. Posted before t0 so the reported search
  // ms stays the search's.
  post({ t: "belief", info: s.beliefInfo(), prior: s.beliefPriorInfo() });
  const t0 = performance.now();
  const compiled = allowCompiled
    ? compiledPreviewPick(s.compiledPreviewBuild(), s.compiledPreviewKey(battle!), previewSeed, searchProfile("blind").c, s.previewCacheInfo())
    : undefined;
  // Validate against this engine's actual ordered action space, even though
  // the artifact is built and versioned with the same sources.
  if (compiled && (JSON.parse(battle!.legalChoices(m.side)) as { input: string }[]).some(c => c.input === compiled)) {
    post({ t: "result", id: m.id, best: compiled,
      playPolicy: { source: "preview-sampled", actions: [{ input: compiled, probability: 1 }] },
      policy: JSON.stringify({ iterations: 0, preview: true, baked: true,
        actions: [{ input: compiled, visits: 0, mean: 0.5, frac: 1 }] }),
      ms: performance.now() - t0, wasmMemoryBytes: wasmMemory.buffer.byteLength, src: "table" });
    return;
  }
  let done = 0;
  for (;;) {
    if (gen !== myGen) return; // superseded: next observe() resets the search
    // Always finish the requested count, including after an early flush.
    const target = m.budget;
    if (done >= target) break;
    const next = stepAdaptive(s, Math.min(slice, target - done));
    if (next === done) break;
    done = next;
    post({ t: "progress", id: m.id, done, budget: m.budget });
    // Macrotask yield: flush the progress post, let cancel/flush interleave.
    await new Promise((r) => setTimeout(r, 0));
  }
  if (gen !== myGen) return;
  post({
    t: "result",
    id: m.id,
    ...finishDecision(s),
    policy: s.rootPolicy(),
    ms: performance.now() - t0,
    wasmMemoryBytes: wasmMemory.buffer.byteLength,
    src: "search",
  });
}

// The forked bot's decision: catch the searcher up on the player stream
// since its last decision, then the same fixed-budget loop. At the fork itself the
// searcher is already installed from the position and no line has arrived.
async function runForkSearch(m: SearchMsg): Promise<void> {
  const t0 = performance.now();
  const myGen = gen;
  flushed = false;
  const f = forked!;
  const frame = JSON.parse(f.channel.frame(battle!)) as {
    lines: string[];
    request: unknown;
  };
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
    post({ t: "result", id: m.id, ...finishDecision(f.searcher, m.side),
      policy: f.searcher.rootPolicy(), ms: performance.now() - t0, wasmMemoryBytes: wasmMemory.buffer.byteLength, src: "search", endgame: f.searcher.endgameStats() });
    return;
  }
  let done = 0;
  for (;;) {
    if (gen !== myGen) return;
    const target = m.budget;
    if (done >= target) break;
    const next = stepAdaptive(f.searcher, Math.min(slice, target - done));
    if (next === done) break; // a forced choice: the searcher does not step
    done = next;
    post({ t: "progress", id: m.id, done, budget: m.budget });
    await new Promise((r) => setTimeout(r, 0));
  }
  if (gen !== myGen) return;
  post({
    t: "result",
    id: m.id,
    ...finishDecision(f.searcher, m.side),
    policy: f.searcher.rootPolicy(),
    ms: performance.now() - t0,
    wasmMemoryBytes: wasmMemory.buffer.byteLength,
    src: "search",
  });
}
