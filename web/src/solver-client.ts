// Main-thread client for the solver worker. Mirrors `bot.ts`'s shape: one
// worker, ids to match replies to requests, promises out.

import type { SolverRequest, SolverResponse } from "./solver-worker";
import type { AnalysisReport } from "./position";
import type { StateView } from "./types";

export interface SolveResult {
  report: AnalysisReport;
  /** The board the solver read — shown back to the user. */
  state: StateView;
  ms: number;
}

export interface SolveOptions {
  spec: string;
  poolJson: string;
  priorJson?: string;
  seed: number;
  budget: number;
  plies: number;
  onProgress?: (done: number, budget: number) => void;
}

/** Thrown when the position itself was refused — the message names the
 * offending field and is meant to be shown, not logged. */
export class PositionRejected extends Error {}

export class SolverWorker {
  private worker: Worker;
  private ready: Promise<void>;
  private nextId = 1;
  private pending = new Map<
    number,
    {
      resolve: (r: SolveResult) => void;
      reject: (e: Error) => void;
      onProgress?: (done: number, budget: number) => void;
    }
  >();

  constructor() {
    this.worker = new Worker(new URL("./solver-worker.ts", import.meta.url), {
      type: "module",
    });
    let onReady!: () => void;
    this.ready = new Promise<void>((r) => (onReady = r));
    this.worker.onmessage = (e: MessageEvent<SolverResponse>) => {
      const m = e.data;
      switch (m.t) {
        case "ready":
          onReady();
          break;
        case "progress":
          this.pending.get(m.id)?.onProgress?.(m.done, m.budget);
          break;
        case "report": {
          const p = this.pending.get(m.id);
          if (p) {
            this.pending.delete(m.id);
            p.resolve({
              report: JSON.parse(m.json) as AnalysisReport,
              state: JSON.parse(m.state) as StateView,
              ms: m.ms,
            });
          }
          break;
        }
        case "rejected": {
          const p = this.pending.get(m.id);
          if (p) {
            this.pending.delete(m.id);
            p.reject(new PositionRejected(m.message));
          }
          break;
        }
        case "error":
          for (const [id, p] of this.pending) {
            this.pending.delete(id);
            p.reject(new Error(m.message));
          }
          break;
      }
    };
  }

  /** Search a position from scratch. */
  async solve(o: SolveOptions): Promise<SolveResult> {
    await this.ready;
    const id = this.nextId++;
    const done = new Promise<SolveResult>((resolve, reject) =>
      this.pending.set(id, { resolve, reject, onProgress: o.onProgress }),
    );
    const req: SolverRequest = {
      t: "solve",
      id,
      spec: o.spec,
      poolJson: o.poolJson,
      priorJson: o.priorJson,
      seed: o.seed,
      budget: o.budget,
      plies: o.plies,
    };
    this.worker.postMessage(req);
    return done;
  }

  /** Keep searching the position already loaded, up to a larger total. */
  async extend(
    budget: number,
    plies: number,
    onProgress?: (done: number, budget: number) => void,
  ): Promise<SolveResult> {
    await this.ready;
    const id = this.nextId++;
    const done = new Promise<SolveResult>((resolve, reject) =>
      this.pending.set(id, { resolve, reject, onProgress }),
    );
    const req: SolverRequest = { t: "extend", id, budget, plies };
    this.worker.postMessage(req);
    return done;
  }

  /** Answer now with what has been searched so far. */
  stop(): void {
    const req: SolverRequest = { t: "stop" };
    this.worker.postMessage(req);
  }

  /** Throw away the running search (a new position is coming). */
  cancel(): void {
    const req: SolverRequest = { t: "cancel" };
    this.worker.postMessage(req);
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}
