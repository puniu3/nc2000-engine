import { useToolMessage, errorText, type ToolMessage } from "./tool-message";
import { toolText } from "./tool-strings";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { loadEngine, randomSeed32 } from "./engine";
import {
  fetchBeliefPool,
  fetchI18nJa,
  fetchNashArtifact,
  fetchDexJson,
} from "./data";
import {
  loadJaNames,
  locale,
  speciesName,
  itemName,
  moveName,
} from "./i18n";
import {
  appendPair,
  resultsCsv,
  sameConfig,
  summarize,
  type EvaluationConfig,
  type EvaluationRun,
  type EvaluationTeam,
  type WorkerResponse,
} from "./evaluate-core";
import {
  exportDistribution,
  importDistribution,
  readOpponents,
  readTeam,
  sha256,
  type OpponentDraft,
} from "./evaluate-input";
import { PRODUCT_ITERATIONS, searchProfile } from "./search-profile";
import { TeamEditor, type EditorDex } from "./evaluate-team-editor";
import "./evaluate.css";

function download(name: string, text: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const defaultLabels = new Map<string, number>();
const opponentLabel = (id: string) => {
  const index = defaultLabels.get(id);
  return index === undefined ? id : toolText("defaultOpponent", index);
};
const percent = (v: number | null) =>
  v === null ? "—" : `${(v * 100).toFixed(1)}%`;
function Findings({ team }: { team: EvaluationTeam }) {
  const warnings = useMemo(() => readTeam(JSON.stringify(team.sets)).warnings, [team, locale()]);
  return (
    <>
      {team.relaxed && (
        <p class="eval-warning">{toolText("relaxedLevels")}</p>
      )}
      {warnings.length > 0 ? (
        <ul class="eval-findings eval-warning">
          {warnings.map((w, i) => (
            <li key={i}>{toolText("warning", w)}</li>
          ))}
        </ul>
      ) : (
        <p class="eval-muted">{toolText("legalTeam")}</p>
      )}
    </>
  );
}
function ResultTable({ run }: { run: EvaluationRun }) {
  return (
    <div class="eval-table-wrap">
      <table>
        <thead>
          <tr>
            {[
              toolText("opponents"),
              toolText("games"),
              toolText("wins"),
              toolText("losses"),
              toolText("ties"),
              toolText("caps"),
              toolText("winRate"),
              toolText("score"),
              toolText("interval"),
            ].map((t) => (
              <th key={t}>{t}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[null, ...run.config.opponents.map((e) => e.id)].map((id) => {
            const s = summarize(
              id === null
                ? run.pairs
                : run.pairs.filter((p) => p.opponent === id),
            );
            return (
              <tr key={id ?? "overall"}>
                <th>{id === null ? toolText("overall") : opponentLabel(id)}</th>
                <td>{s.games}</td>
                <td>{s.win}</td>
                <td>{s.loss}</td>
                <td>{s.tie}</td>
                <td>{s.cap}</td>
                <td>{percent(s.winRate)}</td>
                <td>{percent(s.mean)}</td>
                <td>{s.interval ? s.interval.map(percent).join("–") : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** A budget as the page shows it: the product's own budget is named as
 * such, anything else is marked as a quick run. */
function budgetLabel(n: number): string {
  return n === PRODUCT_ITERATIONS
    ? toolText("productBudget", n.toLocaleString())
    : toolText("quickBudget", n.toLocaleString());
}

export function Evaluate() {
  const [ready, setReady] = useState(false);
  const [editorDex, setEditorDex] = useState<EditorDex | null>(null);
  const [entries, setEntries] = useState<OpponentDraft[]>([]);
  const [party, setParty] = useState("");
  const [iterations, setIterations] = useState(PRODUCT_ITERATIONS);
  const [games, setGames] = useState("32");
  const [belief, setBelief] = useState({ json: "", hash: "" });
  const [error, setError] = useToolMessage("");
  const [run, setRun] = useState<EvaluationRun | null>(null);
  const [history, setHistory] = useState<EvaluationRun[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useToolMessage("");
  const [notice, setNotice] = useToolMessage("");
  const worker = useRef<Worker | null>(null);
  const currentRun = useRef<EvaluationRun | null>(null);
  const alive = useRef(true);

  function selectRun(saved: EvaluationRun) {
    currentRun.current = saved;
    setRun(saved);
    setParty(JSON.stringify(saved.config.player.sets, null, 2));
    setEntries(
      saved.config.opponents.map((e) => ({
        id: e.id,
        weight: String(e.weight),
        text: JSON.stringify(e.sets, null, 2),
      })),
    );
    setIterations(saved.config.iterations);
    setError("");
    setProgress("");
    setNotice("");
  }
  useEffect(() => {
    void (async () => {
      try {
        const [, nash, pool, , dex] = await Promise.all([
          loadEngine(),
          fetchNashArtifact(),
          fetchBeliefPool(),
          loadJaNames(fetchI18nJa),
          fetchDexJson(),
        ]);
        const draft = importDistribution(nash);
        draft.forEach((d, i) => defaultLabels.set(d.id, i + 1));
        const hash = await sha256(pool.poolJson);
        if (!alive.current) return;
        setEditorDex(dex as EditorDex);
        setEntries(draft);
        setBelief({ json: pool.poolJson, hash });
        setReady(true);
      } catch (e) {
        setError(() => errorText(e));
      }
    })();
    return () => {
      alive.current = false;
      worker.current?.terminate();
    };
  }, []);

  const validation = useMemo(() => {
    if (!ready)
      return {
        player: null,
        opponents: null,
        playerError: "",
        opponentError: "",
      };
    let player: EvaluationTeam | null = null;
    let opponents: ReturnType<typeof readOpponents> | null = null;
    let playerError = "",
      opponentError = "";
    try {
      player = readTeam(party);
    } catch (e) {
      playerError = errorText(e);
    }
    try {
      opponents = readOpponents(entries);
    } catch (e) {
      opponentError = errorText(e);
    }
    return { player, opponents, playerError, opponentError };
  }, [ready, party, entries, locale()]);
  const testBudget =
    import.meta.env.MODE === "test"
      ? Number(import.meta.env.VITE_NC2000_TEST_BUDGET)
      : NaN;
  const budget =
    Number.isSafeInteger(testBudget) && testBudget > 0
      ? testBudget
      : iterations;
  const config: EvaluationConfig | null =
    validation.player && validation.opponents
      ? {
          build: __EVALUATOR_BUILD__,
          beliefHash: belief.hash,
          player: validation.player,
          opponents: validation.opponents,
          iterations: budget,
          c: searchProfile("blind").c,
          turnLimit: 500,
        }
      : null;
  const count = Number(games);
  const validCount =
    Number.isSafeInteger(count) && count > 0 && count % 2 === 0;
  const compatible = !!(config && run && sameConfig(config, run.config));

  function retain(next: EvaluationRun) {
    currentRun.current = next;
    setRun(next);
    setHistory((list) =>
      [next, ...list.filter((r) => r.id !== next.id)].sort((a, b) =>
        b.createdAt.localeCompare(a.createdAt),
      ),
    );
  }
  function stop(
    message: ToolMessage = () => toolText("evaluationStopped"),
  ) {
    worker.current?.terminate();
    worker.current = null;
    setRunning(false);
    setProgress("");
    setNotice(message);
  }
  function launch(next: EvaluationRun) {
    if (worker.current) return;
    setRunning(true);
    setError("");
    setNotice("");
    setProgress(() => toolText("preparingBattle"));
    retain(next);
    try {
      const w = new Worker(new URL("./evaluate-worker.ts", import.meta.url), {
        type: "module",
      });
      worker.current = w;
      w.onerror = (event) => {
        setError(() => toolText("battleStartError", event.message));
        stop(() => toolText("evaluationError"));
      };
      w.onmessage = (event: MessageEvent<WorkerResponse>) => {
        if (worker.current !== w) return;
        const msg = event.data;
        if (msg.type === "progress")
          setProgress(
            () => toolText("evaluationProgress", msg.pair * 2 + msg.game, next.targetPairs * 2, msg.turn === 0 ? toolText("selectingPokemon") : toolText("thinkingTurn", msg.turn)),
          );
        if (msg.type === "pair") {
          try {
            const updated = appendPair(currentRun.current!, msg.pair);
            retain(updated);
            w.postMessage({ type: "ack" });
          } catch (e) {
            setError(() => errorText(e));
            stop(() => toolText("aggregationStopped"));
          }
        }
        if (msg.type === "done") stop(() => toolText("evaluationDone"));
        if (msg.type === "error") {
          setError(() => msg.issue ? toolText(msg.issue.key, ...msg.issue.args) : errorText(msg.message));
          stop(() => toolText("evaluationError"));
        }
      };
      w.postMessage({ type: "start", run: next, beliefJson: belief.json });
    } catch (e) {
      setError(() => errorText(e));
      stop(() => toolText("evaluationStartError"));
    }
  }
  function newRun() {
    if (!config || !validCount || running) return;
    launch({
      version: 1,
      id:
        crypto.randomUUID?.() ??
        `${Date.now().toString(36)}-${randomSeed32().toString(16)}-${randomSeed32().toString(16)}`,
      createdAt: new Date().toISOString(),
      config,
      seed: randomSeed32(),
      targetPairs: count / 2,
      pairs: [],
    });
  }
  function resume(add: boolean) {
    if (!run || !compatible || running || (add && !validCount)) return;
    launch({
      ...run,
      targetPairs: add ? run.pairs.length + count / 2 : run.targetPairs,
    });
  }
  async function importFile(
    file: File | undefined,
    apply: (text: string) => void,
  ) {
    if (!file) return;
    try {
      apply(await file.text());
      setError("");
    } catch (e) {
      setError(() => errorText(e));
    }
  }

  return (
    <main class="evaluate">
      <header>
        <p class="eval-eyebrow">{toolText("evaluateEyebrow")}</p>
        <h1>{toolText("evaluateHeading")}</h1>
        <p>{toolText("evaluateIntro")}</p>
      </header>
      {error && (
        <p class="eval-error" role="alert">
          {error}
        </p>
      )}
      {!ready && !error && <p role="status">{toolText("loadingBattleData")}</p>}
      {history.length > 0 && (
        <section class="eval-panel">
          <label>
            {toolText("tabRuns")}{" "}
            <select
              aria-label={toolText("tabRuns")}
              disabled={running}
              value={run?.id}
              onChange={(e) => {
                const saved = history.find(
                  (r) => r.id === e.currentTarget.value,
                );
                if (saved) selectRun(saved);
              }}
            >
              {history.map((r) => (
                <option key={r.id} value={r.id}>{toolText("runSummary", r.createdAt.slice(0, 16).replace("T", " "), r.pairs.length * 2, budgetLabel(r.config.iterations))}</option>
              ))}
            </select>
          </label>
        </section>
      )}
      <fieldset disabled={!ready || running}>
        <div class="eval-columns">
          <section class="eval-panel">
            <h2>{toolText("yourTeam")}</h2>
            <p>{toolText("teamHelp")}</p>
            {editorDex && (
              <TeamEditor
                name={toolText("yourTeam")}
                text={party}
                onChange={setParty}
                dex={editorDex}
              />
            )}
            <details class="eval-import">
              <summary>{toolText("importText")}</summary>
              <p>{toolText("importTextHelp")}</p>
              <textarea
                aria-label={toolText("yourTeam")}
                value={party}
                onInput={(e) => setParty(e.currentTarget.value)}
                placeholder={
                  "Snorlax @ Leftovers\nLevel: 55\n- Double-Edge\n- Curse\n- Rest\n- Sleep Talk"
                }
              />
              <label class="eval-file">
                {toolText("importTeam")}<input
                  aria-label={toolText("teamFile")}
                  type="file"
                  accept=".txt,.json"
                  onChange={(e) => {
                    void importFile(e.currentTarget.files?.[0], setParty);
                    e.currentTarget.value = "";
                  }}
                />
              </label>
            </details>
            <p class="eval-muted">{toolText("ruleWarningHelp")}</p>
            {party.trim() && validation.playerError && (
              <p class="eval-error" role="alert">
                {validation.playerError}
              </p>
            )}
            {validation.player && <Findings team={validation.player} />}
          </section>
          <section
            class="eval-panel eval-opponents"
            aria-label={toolText("opponentSettings")}
          >
            <h2>{toolText("opponents")}</h2>
            <p>{toolText("opponentHelp")}</p>
            {validation.opponents?.map((t) => (
              <div class="eval-entry" key={t.id}>
                <div class="eval-opponent">
                  <strong>{opponentLabel(t.id)}</strong>
                  <span>{percent(t.weight)}</span>
                </div>
                <ul class="eval-opponent-roster">
                  {(t.sets as { species: string; level: number }[]).map(
                    (mon, i) => (
                      <li key={i}>
                        {speciesName(mon.species)} <small>Lv.{mon.level}</small>
                      </li>
                    ),
                  )}
                </ul>
                <details>
                  <summary>{toolText("showSets")}</summary>
                  <div class="eval-opponent-sets">
                    {(
                      t.sets as {
                        species: string;
                        level: number;
                        item: string;
                        moves: string[];
                      }[]
                    ).map((mon, i) => (
                      <div key={i}>
                        <strong>
                          {speciesName(mon.species)} Lv.{mon.level}
                        </strong>
                        <p>
                          {mon.item ? itemName(mon.item) : toolText("noItem")}
                          <br />
                          {mon.moves.map(moveName).join(" / ") || toolText("noMoves")}
                        </p>
                      </div>
                    ))}
                  </div>
                </details>
                {t.warnings.length > 0 && <Findings team={t} />}
              </div>
            ))}
            {validation.opponentError && (
              <p class="eval-error" role="alert">
                {validation.opponentError}
              </p>
            )}
            <p class="eval-muted eval-opponent-help">{toolText("opponentImportHelp")}</p>
            <button
              onClick={() =>
                download(
                  "evaluate-opponents.json",
                  exportDistribution(entries),
                )
              }
            >{toolText("saveOpponents")}</button>
            <label class="eval-file">
              {toolText("importOpponents")}<input
                aria-label={toolText("opponentFile")}
                type="file"
                accept=".json"
                onChange={(e) => {
                  void importFile(e.currentTarget.files?.[0], (text) =>
                    setEntries(importDistribution(text)),
                  );
                  e.currentTarget.value = "";
                }}
              />
            </label>
          </section>
        </div>
        <section class="eval-panel">
          <h2>{toolText("evaluationSettings")}</h2>
          <div class="eval-actions">
            <label>
              {toolText("gameCount")}{" "}
              <input
                aria-label={toolText("gameCount")}
                type="number"
                value={games}
                min={2}
                step={2}
                onInput={(e) => setGames(e.currentTarget.value)}
              />
            </label>
            <label>
              {toolText("iterations")}{" "}
              <select
                aria-label={toolText("iterations")}
                value={iterations}
                onChange={(e) => setIterations(Number(e.currentTarget.value))}
              >
                {[3000, 10000, PRODUCT_ITERATIONS].map((n) => (
                  <option key={n} value={n}>
                    {budgetLabel(n)}
                  </option>
                ))}
                {![3000, 10000, PRODUCT_ITERATIONS].includes(iterations) && (
                  <option value={iterations}>{budgetLabel(iterations)}</option>
                )}
              </select>
            </label>
          </div>
          {!validCount && (
            <p class="eval-error">{toolText("evenGames")}</p>
          )}
          {Number.isFinite(testBudget) && (
            <p class="eval-warning">{toolText("testBudget", budget)}</p>
          )}
          <p class="eval-muted">{toolText("budgetHelp", PRODUCT_ITERATIONS.toLocaleString())}</p>
        </section>
      </fieldset>
      <section class="eval-panel">
        <h2>{toolText("evaluationResults")}</h2>
        <div class="eval-actions">
          <button
            class="eval-start"
            disabled={!config || !validCount || running}
            onClick={newRun}
          >
            {run ? toolText("newEvaluation") : toolText("startEvaluation")}
          </button>
          {run && (
            <>
              <button
                disabled={
                  !compatible || running || run.pairs.length >= run.targetPairs
                }
                onClick={() => resume(false)}
              >{toolText("resume")}</button>
              <button
                disabled={!compatible || running || !validCount}
                onClick={() => resume(true)}
              >{toolText("extendEvaluation")}</button>
            </>
          )}
          {running && <button onClick={() => stop()}>{toolText("stop")}</button>}
        </div>
        {run && !compatible && ready && (
          <p class="eval-warning">{toolText("configurationChanged")}</p>
        )}
        {run && (
          <p>{toolText("completedGames", run.pairs.length * 2, run.targetPairs * 2, budgetLabel(run.config.iterations))}</p>
        )}
        {run && (
          <progress
            aria-label={toolText("completedGamesLabel")}
            max={run.targetPairs}
            value={run.pairs.length}
          />
        )}
        <p class="eval-progress" role="status">
          {progress || notice}
        </p>
        {run ? (
          <>
            <ResultTable run={run} />
            <div class="eval-actions">
              <button
                onClick={() =>
                  download(
                    `evaluation-${run.id}.json`,
                    JSON.stringify(run, null, 2),
                  )
                }
              >{toolText("saveResults")}</button>
              <button
                onClick={() =>
                  download(
                    `evaluation-${run.id}.csv`,
                    resultsCsv(run),
                    "text/csv;charset=utf-8",
                  )
                }
              >{toolText("saveCsv")}</button>
            </div>
            <details>
              <summary>{toolText("resultConditions")}</summary>
              <p>{toolText("startedAt", run.createdAt.slice(0, 16).replace("T", " "))}</p>
              <h3>{toolText("yourTeam")}</h3>
              <Findings team={run.config.player} />
              {run.config.opponents.map((t) => (
                <div key={t.id}>
                  <h3>
                    {opponentLabel(t.id)} · {percent(t.weight)}
                  </h3>
                  <Findings team={t} />
                </div>
              ))}
            </details>
          </>
        ) : (
          <p class="eval-empty">{toolText("emptyResults")}</p>
        )}
        <p class="eval-muted">{toolText("statisticsHelp")}</p>
        <p class="eval-muted">{toolText("resultsHelp")}</p>
      </section>
    </main>
  );
}
