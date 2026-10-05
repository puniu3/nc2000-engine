import { useToolMessage, type ToolMessage } from "./tool-message";
import { toolText } from "./tool-strings";
import { useEffect, useRef, useState } from "preact/hooks";
import { randomSeed32 } from "./engine";
import { searchProfile } from "./search-profile";
import { download, mcnemarP, pct, wilson } from "./fork-stats";
import type { ArenaRequest, ArenaResponse } from "./fork-arena-worker";
import type { ForkInfo } from "./types";

type Foe = "protocol" | "skuct";

interface ArenaConfig {
  bot: "protocol";
  foe: Foe;
  iters: number;
  foe_iters: number;
  c: number;
  seed: number;
  max_steps: number;
}

interface Row {
  trial: number;
  action: string;
  score: number | null;
  outcome: "win" | "loss" | "tie" | "cap";
  [key: string]: unknown;
}

interface Run {
  config: ArenaConfig;
  target: number;
  next: number;
  retry: number[];
  complete: Map<number, Row[]>;
  partial: Map<number, Row[]>;
  armsDone: number;
  elapsedMs: number;
  since: number | null;
}

const MAX_STEPS = 3300;

function defaultWorkers(): number {
  const n = typeof navigator === "undefined" ? 2 : navigator.hardwareConcurrency || 2;
  return Math.max(1, Math.min(n - 1, 8));
}

function clock(ms: number): string {
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function ArenaPanel(props: {
  info: ForkInfo;
  json: string;
  poolJson: string;
  armNames: string[];
  fileKey: string;
  simple?: boolean;
  autoStart?: boolean;
}) {
  const { info } = props;
  const profile = searchProfile(info.info);
  const [trials, setTrials] = useState(32);
  const [iters, setIters] = useState(props.simple ? profile.iterations : 3000);
  const [foe, setFoe] = useState<Foe>(info.opponentView ? "protocol" : "skuct");
  const [seed, setSeed] = useState(() => randomSeed32());
  const [workers, setWorkers] = useState(defaultWorkers);
  const [running, setRunning] = useState(false);
  const [error, setError] = useToolMessage(null);
  const [, setTick] = useState(0);
  const runRef = useRef<Run | null>(null);
  const poolRef = useRef<Worker[]>([]);
  const rerender = () => setTick((t) => t + 1);

  useEffect(() => () => poolRef.current.forEach((w) => w.terminate()), []);
  useEffect(() => { if (props.autoStart) start(); }, []);
  useEffect(() => {
    if (!running) return;
    const id = setInterval(rerender, 1000);
    return () => clearInterval(id);
  }, [running]);

  const config: ArenaConfig = {
    bot: "protocol",
    foe,
    iters,
    foe_iters: iters,
    c: profile.c,
    seed,
    max_steps: MAX_STEPS,
  };
  const run = runRef.current;
  const sameConfig = run !== null && JSON.stringify(run.config) === JSON.stringify(config);

  function halt(message?: ToolMessage) {
    poolRef.current.forEach((w) => w.terminate());
    poolRef.current = [];
    const r = runRef.current;
    if (r) {
      for (const trial of r.partial.keys()) r.retry.push(trial);
      r.retry.sort((a, b) => a - b);
      r.armsDone -= [...r.partial.values()].reduce((a, rows) => a + rows.filter(Boolean).length, 0);
      r.partial.clear();
      if (r.since !== null) r.elapsedMs += performance.now() - r.since;
      r.since = null;
    }
    if (message) setError(message);
    setRunning(false);
    rerender();
  }

  function deal(w: Worker) {
    const r = runRef.current!;
    const trial = r.retry.length > 0 ? r.retry.shift()! : r.next < r.target ? r.next++ : null;
    if (trial === null) {
      w.terminate();
      poolRef.current = poolRef.current.filter((x) => x !== w);
      if (poolRef.current.length === 0) halt();
      return;
    }
    r.partial.set(trial, []);
    w.postMessage({ t: "trial", trial } satisfies ArenaRequest);
  }

  function start() {
    setError(null);
    if (!sameConfig) {
      runRef.current = {
        config,
        target: trials,
        next: 0,
        retry: [],
        complete: new Map(),
        partial: new Map(),
        armsDone: 0,
        elapsedMs: 0,
        since: null,
      };
    } else if (run!.complete.size >= run!.target) {
      run!.target += trials;
    }
    const r = runRef.current!;
    r.since = performance.now();
    const pool: Worker[] = [];
    const count = Math.min(workers, r.target - r.complete.size);
    for (let i = 0; i < count; i++) {
      const w = new Worker(new URL("./fork-arena-worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<ArenaResponse>) => {
        const m = e.data;
        if (runRef.current !== r) return;
        if (m.t === "ready") deal(w);
        else if (m.t === "row") {
          const row = JSON.parse(m.row) as Row;
          row.elapsed_ms = Math.round(m.ms);
          r.partial.get(m.trial)![m.arm] = row;
          r.armsDone += 1;
          rerender();
        } else if (m.t === "done") {
          r.complete.set(m.trial, r.partial.get(m.trial)!);
          r.partial.delete(m.trial);
          rerender();
          deal(w);
        } else halt(() => toolText("battleError", m.message));
      };
      w.onerror = (e) => halt(() => toolText("workerError", e.message));
      w.postMessage({
        t: "start",
        fork: props.json,
        pool: props.poolJson,
        arena: JSON.stringify(config),
        arms: info.arms.length,
      } satisfies ArenaRequest);
      pool.push(w);
    }
    poolRef.current = pool;
    setRunning(true);
  }

  function clear() {
    runRef.current = null;
    setError(null);
    rerender();
  }

  const rows = run
    ? [...run.complete.entries()].sort((a, b) => a[0] - b[0]).flatMap(([, r]) => r)
    : [];
  const done = run?.complete.size ?? 0;
  const elapsed = run ? run.elapsedMs + (run.since !== null ? performance.now() - run.since : 0) : 0;
  const armsTotal = run ? run.target * info.arms.length : 0;
  const armsDone = run?.armsDone ?? 0;
  const eta = armsDone > 0 ? (elapsed / armsDone) * (armsTotal - armsDone) : null;

  function exportRows() {
    download(
      `fork-arena-${props.fileKey.slice(0, 12)}-seed${run!.config.seed}.jsonl`,
      rows.map((r) => JSON.stringify(r)).join("\n") + "\n",
    );
  }

  const cli = run
    ? `fork_counterfactual --fork FORK.json --trials ${run.target} --seed ${run.config.seed} ` +
      `--bot protocol --foe ${run.config.foe} --iters ${run.config.iters} ` +
      `--foe-iters ${run.config.foe_iters} --c ${run.config.c}`
    : null;

  return (
    <section class="fork-panel" data-testid="arena-panel">
      <h2>{props.simple ? toolText("botComparison") : toolText("botTrials")}</h2>
      <p class="fork-muted">{toolText("arenaHelp")}</p>
      <div class="arena-settings">
        <label>
          {toolText("trialCount")}<input
            type="number"
            min={1}
            max={10000}
            value={trials}
            disabled={running}
            onInput={(e) => setTrials(Math.max(1, Number((e.currentTarget as HTMLInputElement).value) || 1))}
          />
        </label>
        {!props.simple && <><label>
          {toolText("actionBudget")}<select
            value={iters}
            disabled={running}
            onChange={(e) => setIters(Number((e.currentTarget as HTMLSelectElement).value))}
          >
            {[...new Set([1000, 3000, 10000, profile.iterations])].map((n) => (
              <option key={n} value={n}>{toolText("arenaBudget", n.toLocaleString(), n === profile.iterations ? toolText("ladderBudgetSuffix") : "")}</option>
            ))}
          </select>
        </label>
        <label>
          {toolText("opponent")}<select
            value={foe}
            disabled={running}
            onChange={(e) => setFoe((e.currentTarget as HTMLSelectElement).value as Foe)}
          >
            {info.opponentView && <option value="protocol">{toolText("protocolBot")}</option>}
            <option value="skuct">{toolText("omniscientBot")}</option>
          </select>
        </label>
        <label>
          {toolText("seed")}<input
            type="number"
            min={0}
            value={seed}
            disabled={running}
            onInput={(e) => setSeed(Math.max(0, Math.floor(Number((e.currentTarget as HTMLInputElement).value) || 0)))}
          />
        </label>
        <label>
          {toolText("parallelism")}<input
            type="number"
            min={1}
            max={32}
            value={workers}
            disabled={running}
            onInput={(e) => setWorkers(Math.max(1, Number((e.currentTarget as HTMLInputElement).value) || 1))}
          />
        </label></>}
      </div>
      <div class="fork-actions">
        {running ? (
          <button onClick={() => halt()}>{toolText("stop")}</button>
        ) : (
          <button class="primary" onClick={start}>
            {sameConfig && done > 0
              ? done >= run!.target
                ? toolText("moreTrials", trials)
                : toolText("resume")
              : toolText("startTrials")}
          </button>
        )}
        {!props.simple && <button onClick={exportRows} disabled={running || rows.length === 0}>{toolText("exportJsonl")}</button>}
        <button class="ghost" onClick={clear} disabled={running || run === null}>{toolText("clearResults")}</button>
      </div>
      {error && <p class="fork-error" role="alert">{error}</p>}
      {run && (
        <>
          <p class="arena-progress" data-testid="arena-progress">{toolText("trialProgress", done, run.target, clock(elapsed), running && eta !== null && toolText("estimatedRemaining", clock(eta)), !sameConfig && toolText("trialsConfigChanged"))}</p>
          {running && (
            <div class="think-progress" aria-hidden="true">
              <div
                class="think-fill"
                style={{ width: `${Math.min(100, (armsDone / Math.max(1, armsTotal)) * 100)}%` }}
              />
            </div>
          )}
          {props.simple ? <div class="kp-arena-results" aria-live="polite">{info.arms.map((arm, i) => {
            const mine = rows.filter(row => row.action === arm.input);
            return <div class="kp-card" key={arm.input}><h3>{arm.label} · {props.armNames[i]}</h3>
              <p>{toolText("winLossTie", mine.filter(row => row.outcome === "win").length, mine.filter(row => row.outcome === "loss").length, mine.filter(row => row.outcome === "tie").length)}</p>
              <p class="kp-muted">{toolText("arenaCompleted", mine.length, mine.some(row => row.outcome === "cap") && toolText("cappedGames", mine.filter(row => row.outcome === "cap").length))}</p>
            </div>;
          })}</div> : <ArenaTable info={info} armNames={props.armNames} rows={rows} />}
          {!props.simple && cli && <p class="fork-muted arena-cli">{toolText("cliEquivalent")}<code>{cli}</code></p>}
        </>
      )}
      <p class="fork-muted">{toolText("arenaResultsHelp", !props.simple && toolText("exportToKeep"))}</p>
    </section>
  );
}

function ArenaTable(props: { info: ForkInfo; armNames: string[]; rows: Row[] }) {
  const { info, rows } = props;
  const byTrial = new Map<number, Map<string, Row>>();
  for (const r of rows) {
    if (!byTrial.has(r.trial)) byTrial.set(r.trial, new Map());
    byTrial.get(r.trial)!.set(r.action, r);
  }
  const base = info.arms[0].input;
  return (
    <div class="fork-table-wrap">
      <table class="fork-table" data-testid="arena-table">
        <thead>
          <tr>
            <th>{toolText("botFirstAction")}</th>
            <th>{toolText("trials")}</th>
            <th>{toolText("botWins")}</th>
            <th>{toolText("botLosses")}</th>
            <th>{toolText("draws")}</th>
            <th>{toolText("caps")}</th>
            <th>{toolText("botWinInterval")}</th>
            <th>{toolText("differenceInterval", info.arms[0].label || toolText("firstCandidate"))}</th>
            <th>McNemar p</th>
          </tr>
        </thead>
        <tbody>
          {info.arms.map((arm, i) => {
            const mine = rows.filter((r) => r.action === arm.input);
            const scored = mine.filter((r) => r.score !== null);
            const score = scored.reduce((a, r) => a + r.score!, 0);
            const rate = scored.length ? score / scored.length : null;
            const ci = wilson(score, scored.length);
            let diff: string = "—";
            let p: string = "—";
            if (i > 0) {
              const pairs: [number, number][] = [];
              for (const trial of byTrial.values()) {
                const a = trial.get(arm.input);
                const b = trial.get(base);
                if (a && b && a.score !== null && b.score !== null) pairs.push([a.score, b.score]);
              }
              if (pairs.length > 0) {
                const d = pairs.map(([a, b]) => a - b);
                const mean = d.reduce((x, y) => x + y, 0) / d.length;
                const sd =
                  d.length > 1
                    ? Math.sqrt(d.reduce((x, y) => x + (y - mean) ** 2, 0) / (d.length - 1))
                    : null;
                const half = sd !== null ? (1.96 * sd) / Math.sqrt(d.length) : null;
                const sign = (v: number) => `${v >= 0 ? "+" : ""}${pct(v)}`;
                diff = half !== null ? `${sign(mean)} (${sign(mean - half)}〜${sign(mean + half)})` : sign(mean);
                if (pairs.every(([a, b]) => (a === 0 || a === 1) && (b === 0 || b === 1)))
                  p = mcnemarP(d.filter((x) => x === 1).length, d.filter((x) => x === -1).length).toPrecision(2);
              }
            }
            return (
              <tr key={i}>
                <td>
                  {props.armNames[i]}
                  {arm.label && <span class="fork-arm-label">{arm.label}</span>}
                </td>
                <td>{mine.length}</td>
                <td>{mine.filter((r) => r.outcome === "win").length}</td>
                <td>{mine.filter((r) => r.outcome === "loss").length}</td>
                <td>{mine.filter((r) => r.outcome === "tie").length}</td>
                <td>{mine.filter((r) => r.outcome === "cap").length}</td>
                <td>{rate === null ? "—" : `${pct(rate)} (${pct(ci[0])}–${pct(ci[1])})`}</td>
                <td>{diff}</td>
                <td>{p}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
