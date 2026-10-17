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
  setLocale,
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
/** The shipped mixture's teams are shown as 基本の相手N in file order
 * rather than by id; an uploaded distribution keeps its own ids. Filled
 * once, when the default mixture loads. */
const defaultLabels = new Map<string, string>();
const opponentLabel = (id: string) => defaultLabels.get(id) ?? id;
const percent = (v: number | null) =>
  v === null ? "—" : `${(v * 100).toFixed(1)}%`;
function Findings({ team }: { team: EvaluationTeam }) {
  return (
    <>
      {team.relaxed && (
        <p class="eval-warning">
          このパーティは、選ぶ3匹のレベル合計が155を超えていても対戦できます。
        </p>
      )}
      {team.warnings.length > 0 ? (
        <ul class="eval-findings eval-warning">
          {team.warnings.map((w, i) => (
            <li key={i}>警告: {w}</li>
          ))}
        </ul>
      ) : (
        <p class="eval-muted">大会ルールに合っています。</p>
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
              "対戦相手",
              "試合",
              "勝",
              "負",
              "引分",
              "打切",
              "勝率",
              "引き分けを含む成績",
              "推定の幅",
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
                <th>{id === null ? "総合" : opponentLabel(id)}</th>
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
    ? `${n.toLocaleString()}回(「強い」の対戦中と同じ回数)`
    : `${n.toLocaleString()}回(簡易計測)`;
}

export function Evaluate() {
  const [ready, setReady] = useState(false);
  const [editorDex, setEditorDex] = useState<EditorDex | null>(null);
  const [entries, setEntries] = useState<OpponentDraft[]>([]);
  const [party, setParty] = useState("");
  const [iterations, setIterations] = useState(PRODUCT_ITERATIONS);
  const [games, setGames] = useState("32");
  const [belief, setBelief] = useState({ json: "", hash: "" });
  const [error, setError] = useState("");
  const [run, setRun] = useState<EvaluationRun | null>(null);
  const [history, setHistory] = useState<EvaluationRun[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [notice, setNotice] = useState("");
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
    setLocale("ja", false);
    document.title = "NC2000 — パーティ強度測定";
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
        draft.forEach((d, i) => defaultLabels.set(d.id, `基本の相手${i + 1}`));
        const hash = await sha256(pool.poolJson);
        if (!alive.current) return;
        setEditorDex(dex as EditorDex);
        setEntries(draft);
        setBelief({ json: pool.poolJson, hash });
        setReady(true);
      } catch (e) {
        setError(String(e));
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
      playerError = String(e).replace(/^Error: /, "");
    }
    try {
      opponents = readOpponents(entries);
    } catch (e) {
      opponentError = String(e).replace(/^Error: /, "");
    }
    return { player, opponents, playerError, opponentError };
  }, [ready, party, entries]);
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
    message = "停止しました。終わっていない2戦は、再開したときにやり直します。",
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
    setProgress("対戦を準備しています…");
    retain(next);
    try {
      const w = new Worker(new URL("./evaluate-worker.ts", import.meta.url), {
        type: "module",
      });
      worker.current = w;
      w.onerror = (event) => {
        setError(`対戦を開始できません: ${event.message}`);
        stop("エラーで停止しました。終わった対戦の結果は残っています。");
      };
      w.onmessage = (event: MessageEvent<WorkerResponse>) => {
        if (worker.current !== w) return;
        const msg = event.data;
        if (msg.type === "progress")
          setProgress(
            `対戦 ${msg.pair * 2 + msg.game} / ${next.targetPairs * 2} · ${msg.turn === 0 ? "ポケモンを選んでいます" : `${msg.turn}ターン目・次の手を考えています`}`,
          );
        if (msg.type === "pair") {
          try {
            const updated = appendPair(currentRun.current!, msg.pair);
            retain(updated);
            w.postMessage({ type: "ack" });
          } catch (e) {
            setError(String(e));
            stop("結果の集計中に停止しました。");
          }
        }
        if (msg.type === "done") stop("指定した試合数の計測が完了しました。");
        if (msg.type === "error") {
          setError(msg.message);
          stop("エラーで停止しました。終わった対戦の結果は残っています。");
        }
      };
      w.postMessage({ type: "start", run: next, beliefJson: belief.json });
    } catch (e) {
      setError(String(e));
      stop("計測を開始できませんでした。");
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
      setError(String(e));
    }
  }

  return (
    <main class="evaluate">
      <header>
        <p class="eval-eyebrow">NC2000 パーティ診断</p>
        <h1>パーティの強さを調べる</h1>
        <p>
          ポケモンを登録すると、コンピューター同士で対戦して勝率を調べます。相手の技や持ち物は、対戦で判明するまで見えないルールです。
        </p>
      </header>
      {error && (
        <p class="eval-error" role="alert">
          {error}
        </p>
      )}
      {!ready && !error && <p role="status">対戦データを読み込んでいます…</p>}
      {history.length > 0 && (
        <section class="eval-panel">
          <label>
            このタブの計測{" "}
            <select
              aria-label="このタブの計測"
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
                <option key={r.id} value={r.id}>
                  {r.createdAt.slice(0, 16).replace("T", " ")} UTC ·{" "}
                  {r.pairs.length * 2}戦 ·{" "}
                  {budgetLabel(r.config.iterations)}
                </option>
              ))}
            </select>
          </label>
        </section>
      )}
      <fieldset disabled={!ready || running}>
        <div class="eval-columns">
          <section class="eval-panel">
            <h2>自分のパーティ</h2>
            <p>
              ポケモンを選び、レベル・持ち物・技を入力してください。6匹まで登録できます。
            </p>
            {editorDex && (
              <TeamEditor
                name="自分のパーティ"
                text={party}
                onChange={setParty}
                dex={editorDex}
              />
            )}
            <details class="eval-import">
              <summary>テキストから読み込む</summary>
              <p>他のツールからコピーしたパーティ情報を貼り付けられます。</p>
              <textarea
                aria-label="自分のパーティ"
                value={party}
                onInput={(e) => setParty(e.currentTarget.value)}
                placeholder={
                  "Snorlax @ Leftovers\nLevel: 55\n- Double-Edge\n- Curse\n- Rest\n- Sleep Talk"
                }
              />
              <label class="eval-file">
                パーティを読み込む
                <input
                  aria-label="パーティファイル"
                  type="file"
                  accept=".txt,.json"
                  onChange={(e) => {
                    void importFile(e.currentTarget.files?.[0], setParty);
                    e.currentTarget.value = "";
                  }}
                />
              </label>
            </details>
            <p class="eval-muted">
              大会ルール違反は警告を表示し、対戦に参加できます。
            </p>
            {party.trim() && validation.playerError && (
              <p class="eval-error" role="alert">
                {validation.playerError}
              </p>
            )}
            {validation.player && <Findings team={validation.player} />}
          </section>
          <section
            class="eval-panel eval-opponents"
            aria-label="対戦相手の設定"
          >
            <h2>対戦相手</h2>
            <p>この中から、表示した確率で対戦相手を選びます。</p>
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
                  <summary>技・持ち物を見る</summary>
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
                          {mon.item ? itemName(mon.item) : "持ち物なし"}
                          <br />
                          {mon.moves.map(moveName).join(" / ") || "技なし"}
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
            <p class="eval-muted eval-opponent-help">
              今の設定をファイルに保存し、書き換えてから読み込むと相手を変更できます。
            </p>
            <button
              onClick={() =>
                download(
                  "evaluate-opponents.json",
                  exportDistribution(entries),
                )
              }
            >
              相手の設定をファイルに保存
            </button>
            <label class="eval-file">
              相手の設定を読み込む（JSON）
              <input
                aria-label="相手の設定ファイル"
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
          <h2>計測設定</h2>
          <div class="eval-actions">
            <label>
              試合数{" "}
              <input
                aria-label="試合数"
                type="number"
                value={games}
                min={2}
                step={2}
                onInput={(e) => setGames(e.currentTarget.value)}
              />
            </label>
            <label>
              考える回数{" "}
              <select
                aria-label="考える回数"
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
            <p class="eval-error">試合数は2以上の偶数にしてください。</p>
          )}
          {Number.isFinite(testBudget) && (
            <p class="eval-warning">
              テストビルド：実際に考える回数は{budget}回です。
            </p>
          )}
          <p class="eval-muted">
            1手を決めるまでに試す回数です。「強い」の対戦中は
            {PRODUCT_ITERATIONS.toLocaleString()}
            回です。ここでは選出にも同じ回数を使うため、通常対戦の事前計算済みの選出とは異なります。両方に同じ回数を使い、先後を入れ替えて2戦ずつ対戦します。
          </p>
        </section>
      </fieldset>
      <section class="eval-panel">
        <h2>計測と結果</h2>
        <div class="eval-actions">
          <button
            class="eval-start"
            disabled={!config || !validCount || running}
            onClick={newRun}
          >
            {run ? "別の計測を開始" : "計測を開始"}
          </button>
          {run && (
            <>
              <button
                disabled={
                  !compatible || running || run.pairs.length >= run.targetPairs
                }
                onClick={() => resume(false)}
              >
                再開
              </button>
              <button
                disabled={!compatible || running || !validCount}
                onClick={() => resume(true)}
              >
                追加計測
              </button>
            </>
          )}
          {running && <button onClick={() => stop()}>停止</button>}
        </div>
        {run && !compatible && ready && (
          <p class="eval-warning">
            パーティ・対戦設定・アプリのバージョンが計測開始時と異なります。この結果に追加せず、新しい計測を始めてください。
          </p>
        )}
        {run && (
          <p>
            {run.pairs.length * 2} / {run.targetPairs * 2}戦完了 ·{" "}
            {budgetLabel(run.config.iterations)}
          </p>
        )}
        {run && (
          <progress
            aria-label="完了した試合"
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
              >
                結果をファイルに保存
              </button>
              <button
                onClick={() =>
                  download(
                    `evaluation-${run.id}.csv`,
                    resultsCsv(run),
                    "text/csv;charset=utf-8",
                  )
                }
              >
                表計算ソフト用に保存
              </button>
            </div>
            <details>
              <summary>この結果の条件と警告</summary>
              <p>開始: {run.createdAt.slice(0, 16).replace("T", " ")} UTC</p>
              <h3>自分のパーティ</h3>
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
          <p class="eval-empty">
            パーティを入力して計測を開始すると、総合成績と相手別成績がここに表示されます。
          </p>
        )}
        <p class="eval-muted">
          勝率は勝った試合の割合です。「引き分けを含む成績」では、引き分けと500ターンでの打ち切りを半勝として数えます。「推定の幅」は偶然によるばらつきの目安です。試合が少ない間は、結果も大きく変わります。
        </p>
        <p class="eval-muted">
          このコンピューターが、指定した相手と対戦したときの成績です。結果は自動保存されません。再読み込みやタブを閉じる操作でリセットされます。残したい結果はファイルに保存してください。
        </p>
      </section>
    </main>
  );
}
