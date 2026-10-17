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
      onProgress?: (done: number, budget: number) => void;
    }
  >();

  constructor() {
    this.worker = new Worker(new URL("./bot-worker.ts", import.meta.url), {
      type: "module",
    });
    let onReady!: () => void;
    this.ready = new Promise<void>((r) => (onReady = r));
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
            p.resolve({
              best: m.best,
              policy: JSON.parse(m.policy) as RootPolicy,
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
    this.send({ t: "battle", p1, p2, seed, searcher: { ...searcher, endgame: searcher.endgame ?? endgameSettings() } });
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
    this.send({ t: "fork", fork, seed, searcher: { ...searcher, endgame: searcher.endgame ?? endgameSettings() } });
  }

  /** Keep the mirror battle in lockstep (same picks, same order). */
  apply(picks: [number, string][]): void {
    this.send({ t: "apply", picks });
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
    return new Promise<SearchOutcome>((resolve) => {
      this.pending.set(id, { resolve, onProgress });
      this.send({ t: "search", id, side, budget, seed, ponder });
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
  }
}
