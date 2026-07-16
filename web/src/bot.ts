// Main-thread client for the bot search worker.

import type { WorkerRequest, WorkerResponse } from "./bot-worker";
import type { RootPolicy } from "./types";

export interface SearchOutcome {
  best: string | null;
  policy: RootPolicy;
  ms: number;
}

export class BotWorker {
  private worker: Worker;
  private ready: Promise<void>;
  private nextId = 1;
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
            });
          }
          break;
        }
        case "error":
          console.error("bot worker:", m.message);
          break;
      }
    };
  }

  private send(m: WorkerRequest): void {
    this.worker.postMessage(m);
  }

  async newBattle(p1: string, p2: string, seed: string): Promise<void> {
    await this.ready;
    this.send({ t: "battle", p1, p2, seed });
  }

  /** Keep the mirror battle in lockstep (same picks, same order). */
  apply(picks: [number, string][]): void {
    this.send({ t: "apply", picks });
  }

  /** Search the mirror battle's current decision point for `side`.
   * A later apply/newBattle/cancelAll supersedes the search: its promise
   * then never settles (callers holding a stale promise are per-game and
   * torn down with the game). */
  search(
    side: number,
    budget: number,
    seed: number,
    onProgress?: (done: number, budget: number) => void,
  ): Promise<SearchOutcome> {
    const id = this.nextId++;
    return new Promise<SearchOutcome>((resolve) => {
      this.pending.set(id, { resolve, onProgress });
      this.send({ t: "search", id, side, budget, seed });
    });
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
