import type { EndgameSettings } from "./endgame-settings";

interface TimedSearcher {
  beginEndgame(config: string | undefined, milliseconds: number): void;
  extendEndgame(milliseconds: number): void;
  endgameStats(): string;
  iterations(): number;
  step(n: number): number;
}

export async function runEndgameSearch(
  searcher: TimedSearcher,
  settings: EndgameSettings,
  control: {
    cancelled(): boolean;
    stop(): boolean;
    ponder?: boolean;
    progress(done: number): void;
  },
  started = performance.now(),
): Promise<boolean> {
  const begun = performance.now();
  const setup = begun - started;
  let allowance = settings.milliseconds;
  const cap = Math.min(60_000, allowance * (control.ponder ? 10 : 1));
  searcher.beginEndgame(JSON.stringify({ solver_fraction: settings.solverFraction, max_nodes: settings.maxNodes ?? 12000 }), Math.max(0, allowance - setup));
  let done = searcher.iterations();
  while (!control.cancelled()) {
    const elapsed = performance.now() - started;
    if (elapsed >= cap || (control.stop() && (!control.ponder || elapsed >= settings.milliseconds))) break;
    if (elapsed >= allowance) {
      if (!control.ponder || control.stop()) break;
      allowance = Math.min(cap, allowance + settings.milliseconds);
      searcher.extendEndgame(Math.max(0, allowance - setup));
    }
    const slice = performance.now();
    let advanced = false;
    do {
      const next = searcher.step(1);
      if (next === done) break;
      advanced = true;
      done = next;
    } while (performance.now() - slice < 8 && performance.now() - started < allowance);
    control.progress(done);
    if (!advanced && !JSON.parse(searcher.endgameStats()).expired) break;
    await new Promise<void>(resolve => setTimeout(resolve, 0));
  }
  return !control.cancelled();
}
