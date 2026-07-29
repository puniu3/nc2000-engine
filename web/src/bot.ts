// Main-thread client for the bot search worker.

import type { WorkerRequest, WorkerResponse } from "./bot-worker";
import type { BeliefInfo, PriorInfo, PriorReport, RootPolicy } from "./types";

export interface SearchOutcome {
  best: string | null;
  policy: RootPolicy;
  ms: number;
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
              ms: m.ms,
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

  /** Start a game. `searcher.mode` fixes the worker's information policy for
   * the whole battle: "open" pins the opponent's true sets as the belief —
   * only the opponent's picks (which 3 + lead) stay hidden to it — while
   * "blind" leaves it with what the human also sees, and lets
   * `searcher.priorJson` (raw table text, blind only) govern its fallback
   * imputation. The prior verdict comes back on `onPriorReport`; the belief
   * itself streams on `onBelief`. */
  async newBattle(
    p1: string,
    p2: string,
    seed: string,
    searcher: {
      poolJson: string;
      side: number;
      seed: number;
      mode: "open" | "blind";
      priorJson?: string;
    },
  ): Promise<void> {
    await this.ready;
    this.send({ t: "battle", p1, p2, seed, searcher });
  }

  /** Feed one baked pair table for the table preview (call before the
   * preview search; messages are ordered). */
  addPair(json: string): void {
    this.send({ t: "pair", json });
  }

  /** Keep the mirror battle in lockstep (same picks, same order). */
  apply(picks: [number, string][]): void {
    this.send({ t: "apply", picks });
  }

  /** Search the mirror battle's current decision point for `side`.
   * `ponder: true` (simultaneous point) keeps thinking past the budget —
   * bonus iterations while the human deliberates — until `flush()` or the
   * ponder cap; `false` (bot-only point) stops exactly at budget.
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

  /** The human committed: a pondering search returns its best at the next
   * slice boundary (a search still inside its budget finishes it first). */
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
