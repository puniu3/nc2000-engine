import { Battle, loadEngine } from "./engine";
import { PlayerChannel } from "../../crates/wasm/pkg-web/nc2000_wasm";
// Main-thread client for the bot search worker.

import type { WorkerRequest, WorkerResponse } from "./bot-worker";
import type { BeliefInfo, PriorInfo, PriorReport, RootPolicy, PlayPolicy, ActionPolicyMode } from "./types";
import { endgameSettings, type EndgameSettings } from "./endgame-settings";

export interface SearchOutcome {
  best: string | null;
  policy: RootPolicy;
  playPolicy: PlayPolicy | null;
  ms: number;
  wasmMemoryBytes: number;
  endgame?: string;
  /** Where the pick came from (preview: "table" | "search"). */
  src?: "table" | "search";
}

export class BotWorker {
  private mirror: Battle | null = null;
  private channel: PlayerChannel | null = null;
  private side = 0;
  private worker: Worker;
  private ready: Promise<void>;
  private nextId = 1;
  /** Blind mode: the bot's read of the opponent, refreshed at every search
   * (the worker posts it right after observe(), so it is the read the search
   * about to run is actually using). Never fires in open mode. */
  onBelief?: (info: BeliefInfo, prior: PriorInfo) => void;
  /** The verdict on the belief-prior table handed to `newBattle` — fires
   * once per battle that carried one, whether it was applied or refused. */
  onPriorReport?: (report: PriorReport) => void;
  private pending = new Map<
    number,
    {
      resolve: (r: SearchOutcome) => void;
      reject: (e: Error) => void;
      onProgress?: (done: number, budget: number) => void;
      resolveInput: (input: string) => string;
    }
  >();

  constructor() {
    this.worker = new Worker(new URL("./bot-worker.ts", import.meta.url), {
      type: "module",
    });
    let onReady!: () => void;
    this.ready = new Promise<void>((r) => (onReady = r));
    const rejectAll = (message: string) => {
      for (const [id, p] of this.pending) {
        this.pending.delete(id);
        p.reject(new Error(message));
      }
    };
    this.worker.onerror = (ev: ErrorEvent) => {
      console.error("bot worker error:", ev.message);
      rejectAll(ev.message || "Bot worker error");
    };
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const m = e.data;
      switch (m.t) {
        case "ready":
          onReady();
          break;
        case "progress":
          this.pending.get(m.id)?.onProgress?.(m.done, m.budget);
          break;
        case "result": {
          const p = this.pending.get(m.id);
          if (p) {
            this.pending.delete(m.id);
            const resolve = p.resolveInput;
            const policy = JSON.parse(m.policy) as RootPolicy;
            for (const row of policy.actions) row.input = resolve(row.input);
            if (m.playPolicy) for (const row of m.playPolicy.actions) row.input = resolve(row.input);
            p.resolve({
              best: m.best === null ? null : resolve(m.best),
              policy,
              playPolicy: m.playPolicy,
              ms: m.ms,
              wasmMemoryBytes: m.wasmMemoryBytes,
              endgame: m.endgame,
              src: m.src,
            });
          }
          break;
        }
        case "belief":
          this.onBelief?.(
            JSON.parse(m.info) as BeliefInfo,
            JSON.parse(m.prior) as PriorInfo,
          );
          break;
        case "prior":
          this.onPriorReport?.(JSON.parse(m.report) as PriorReport);
          break;
        case "error":
          console.error("bot worker:", m.message);
          rejectAll(m.message);
          break;
      }
    };
  }

  private send(m: WorkerRequest): void {
    this.worker.postMessage(m);
  }

  /** Start a blind game: the bot sees what the human also sees, and
   * `searcher.priorJson` (raw table text) governs its fallback imputation.
   * The prior verdict comes back on `onPriorReport`; the belief itself
   * streams on `onBelief`. */
  async newBattle(
    p1: string,
    p2: string,
    seed: string,
    searcher: {
      poolJson: string;
      side: number;
      seed: number;
      priorJson?: string;
      compiledPreview?: boolean;
      endgame?: EndgameSettings;
      actionPolicy?: ActionPolicyMode;
    },
  ): Promise<void> {
    await this.ready;
    this.channel?.free();
    this.mirror?.free();
    this.side = searcher.side;
    this.mirror = new Battle(await loadEngine(), p1, p2, seed);
    this.channel = new PlayerChannel(this.side);
    this.send({ t: "battle", ownTeam: this.side === 0 ? p1 : p2,
      searcher: { ...searcher, endgame: searcher.endgame ?? endgameSettings() } });
  }

  /** Start a forked battle (`?fork`): the worker's mirror is
   * `Battle.fromFork(fork, seed)` and the bot plays `searcher.side` from the
   * fork's recorded information set under `searcher.mode`. */
  async newFork(
    fork: string,
    seed: number,
    searcher: {
      poolJson: string;
      side: number;
      seed: number;
      mode: "open" | "blind";
      endgame?: EndgameSettings;
      actionPolicy?: ActionPolicyMode;
    },
  ): Promise<void> {
    await this.ready;
    this.channel?.free(); this.channel = null;
    this.mirror?.free(); this.mirror = null;
    this.send({ t: "fork", fork, seed, searcher: { ...searcher, endgame: searcher.endgame ?? endgameSettings() } });
  }

  /** Keep the mirror battle in lockstep (same picks, same order). */
  apply(picks: [number, string][]): void {
    if (this.mirror) {
      for (const [side, input] of picks) this.mirror.applyChoice(side, input);
      this.send({ t: "cancel" });
    } else this.send({ t: "apply", picks });
  }

  /** Search the mirror battle's current decision point for `side`.
   * Live searches stop at exactly budget, even when launched while the
   * human is choosing. `ponder` is retained for the opt-in solver experiment.
   * A later apply/newBattle/cancelAll supersedes the search: its promise
   * then never settles (callers holding a stale promise are per-game and
   * torn down with the game). */
  search(
    side: number,
    budget: number,
    seed: number,
    ponder: boolean,
    onProgress?: (done: number, budget: number) => void,
  ): Promise<SearchOutcome> {
    const id = this.nextId++;
    return new Promise<SearchOutcome>((resolve, reject) => {
      const inputs: string[] = this.mirror ? JSON.parse(this.mirror.legalChoices(side)).map((c: {input:string})=>c.input) : [];
      const key = (input: string) => input.replace(/\s/g, "").replace(/^movehiddenpower.*/, "movehiddenpower");
      const resolveInput = (input: string) => {
        if (!inputs.length) return input;
        const canonical = inputs.find(candidate => key(candidate) === key(input));
        if (!canonical) throw new Error(`Cannot resolve player action: ${input}`);
        return canonical;
      };
      this.pending.set(id, { resolve, reject, onProgress, resolveInput });
      const frame = this.mirror ? JSON.parse(this.channel!.frame(this.mirror)) : undefined;
      this.send({ t: "search", id, side, budget, seed, ponder, frame });
    });
  }

  /** Human committed. Fixed-iteration searches ignore this notification;
   * the opt-in timed solver experiment still uses it. */
  flush(): void {
    this.send({ t: "flush" });
  }

  cancelAll(): void {
    this.pending.clear();
    this.send({ t: "cancel" });
  }

  terminate(): void {
    this.pending.clear();
    this.worker.terminate();
    this.channel?.free(); this.channel = null;
    this.mirror?.free(); this.mirror = null;
  }
}
