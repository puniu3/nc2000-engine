import { useToolMessage } from "./tool-message";
import { toolText } from "./tool-strings";
import { useMemo, useEffect, useRef, useState } from "preact/hooks";
import { Kifu, getDex, loadEngine, randomSeed32, readFork } from "./engine";
import { fetchBeliefPool, fetchDexJson, fetchI18nJa } from "./data";
import { loadJaNames, moveName, locale, speciesName, statusLongName } from "./i18n";
import { loadSetDex } from "./set-info";
import { extractKifu } from "./kifu-code";
import { armNames, ForkGame, type LoadedFork, type GameRecord } from "./fork";
import { ArenaPanel } from "./fork-arena";
import { BotWorker } from "./bot";
import { searchProfile } from "./search-profile";
import { Narrator } from "./narrate";
import type { Choice, RootPolicy, StateView } from "./types";
import type { InfoMode } from "./info-mode";
import "./kifu.css";

interface SceneIndex { round: number; turn: number; active: (string | null)[]; played: Choice }
interface RecordInfo { scenes: SceneIndex[]; turns: number; outcome: string | null; botSide: number; side: number; info: InfoMode }
interface Scene { view: StateView; choices: Choice[]; played: Choice; log: string[] }
interface Run { fork: LoadedFork; method: "human" | "bot"; record: GameRecord }

export function choiceName(choice: Choice): string {
  if (choice.kind === "move") return moveName(choice.name);
  if (choice.kind === "switch") return toolText("switchTo", speciesName(choice.species));
  return choice.kind === "pass" ? toolText("pass") : toolText("selectPokemon");
}

function recordError(error: unknown): string {
  const text = String(error);
  if (text.includes("incompatible replay version")) return toolText("replayVersionError");
  if (text.includes("not a replay code")) return toolText("missingReplay");
  return toolText("invalidReplay");
}

export function KifuTool() {
  const [screen, setScreen] = useState<"import" | "scene">("import");
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [paste, setPaste] = useState("");
  const [notice, setNotice] = useToolMessage("");
  const [metadata, setMetadata] = useState<RecordInfo | null>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [selected, setSelected] = useState(0);
  const [alternative, setAlternative] = useState("");
  const [method, setMethod] = useState<"human" | "bot">("human");
  const [storedRun, setRun] = useState<Run | null>(null);
  const run = useMemo(() => storedRun && ({ ...storedRun, fork: { ...storedRun.fork, armNames: armNames(storedRun.fork.json, storedRun.fork.info) } }), [storedRun, locale()]);
  const [poolJson, setPoolJson] = useState("");
  const [hintPolicy, setHintPolicy] = useState<RootPolicy | null>(null);
  const [hintBest, setHintBest] = useState<string | null>(null);
  const [hintLoading, setHintLoading] = useState(false);
  const replay = useRef<Kifu | null>(null);
  const hintWorker = useRef<BotWorker | null>(null);
  const hintGen = useRef(0);
  const alive = useRef(true);
  const heading = useRef<HTMLHeadingElement>(null);
  const pasteField = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    alive.current = true;
    void (async () => {
      try {
        const [, pool] = await Promise.all([loadEngine(), fetchBeliefPool(), loadJaNames(fetchI18nJa), loadSetDex(fetchDexJson)]);
        if (!alive.current) return;
        setPoolJson(pool.poolJson);
        setReady(true);
        const hashParams = new URLSearchParams(location.hash.slice(1));
        const searchParams = new URLSearchParams(location.search);
        const code = hashParams.get("kifu") ?? searchParams.get("kifu");
        const sideParam = hashParams.get("side") ?? searchParams.get("side");
        const roundParam = hashParams.get("round") ?? searchParams.get("round");
        const turnParam = hashParams.get("turn") ?? searchParams.get("turn");
        const sideOpt = sideParam === "0" ? 0 : sideParam === "1" ? 1 : undefined;
        const roundOpt = roundParam !== null ? Number(roundParam) : undefined;
        const turnOpt = turnParam !== null ? Number(turnParam) : undefined;
        if (code) { setPaste(code); await load(code, sideOpt, roundOpt, turnOpt); }
      } catch (error) {
        console.error("kifu setup", error);
        if (alive.current) setNotice(() => toolText("battleDataError"));
      }
    })();
    return () => {
      alive.current = false;
      hintWorker.current?.terminate();
      hintWorker.current = null;
      replay.current?.free();
      replay.current = null;
    };
  }, []);

  useEffect(() => {
    const round = metadata?.scenes[selected]?.round;
    const side = metadata?.side;
    const infoMode = metadata?.info;
    if (screen !== "scene" || run !== null || !replay.current || !poolJson || !scene || round === undefined || side === undefined || !infoMode || scene.choices.length <= 1) {
      setHintPolicy(null);
      setHintBest(null);
      setHintLoading(false);
      return;
    }
    const myGen = ++hintGen.current;
    setHintPolicy(null);
    setHintBest(null);
    setHintLoading(true);
    void (async () => {
      try {
        const forkJson = replay.current!.fork(round, scene.played.input, poolJson, side);
        if (!alive.current || hintGen.current !== myGen) return;
        if (!hintWorker.current) hintWorker.current = new BotWorker();
        const worker = hintWorker.current;
        worker.cancelAll();
        await worker.newFork(forkJson, 1, {
          poolJson,
          side,
          seed: 1,
          mode: infoMode,
        });
        if (!alive.current || hintGen.current !== myGen) return;
        const budget = Math.min(3000, searchProfile(infoMode).iterations);
        const res = await worker.search(side, budget, 1, false);
        if (!alive.current || hintGen.current !== myGen) return;
        setHintPolicy(res.policy);
        setHintBest(res.best);
      } catch (err) {
        console.warn("kifu hint", err);
      } finally {
        if (alive.current && hintGen.current === myGen) setHintLoading(false);
      }
    })();
  }, [screen, run !== null, metadata?.side, metadata?.info, selected, scene, poolJson]);

  useEffect(() => { heading.current?.focus(); }, [screen, run !== null]);

  async function load(text: string, initialSide?: number, initialRound?: number, initialTurn?: number) {
    setNotice("");
    setBusy(true);
    await new Promise(resolve => setTimeout(resolve, 0));
    if (!alive.current) return;
    let next: Kifu | null = null;
    try {
      const code = extractKifu(text);
      if (!code) throw new Error("not a replay code");
      next = new Kifu(getDex(), code);
      const info = JSON.parse(next.scenes(initialSide)) as RecordInfo;
      if (info.scenes.length === 0) {
        setNotice(() => toolText("noReplayScenes"));
        return;
      }
      let idx = 0;
      if (initialRound !== undefined && Number.isFinite(initialRound)) {
        const found = info.scenes.findIndex(s => s.round === initialRound);
        if (found >= 0) idx = found;
      } else if (initialTurn !== undefined && Number.isFinite(initialTurn)) {
        const found = info.scenes.findIndex(s => s.turn === initialTurn);
        if (found >= 0) idx = found;
      }
      const first = JSON.parse(next.scene(info.scenes[idx].round, info.side)) as Scene;
      replay.current?.free();
      replay.current = next;
      next = null;
      setMetadata(info);
      setScene(first);
      setSelected(idx);
      setAlternative("");
      setScreen("scene");
    } catch (error) {
      console.error("kifu import", error);
      setNotice(() => text.trim() ? recordError(error) : toolText("pasteReplayError"));
      pasteField.current?.focus();
    } finally {
      next?.free();
      setBusy(false);
    }
  }

  function changeSide(nextSide: number) {
    if (!metadata || !replay.current || nextSide === metadata.side) return;
    const currentTurn = metadata.scenes[selected]?.turn;
    const info = JSON.parse(replay.current.scenes(nextSide)) as RecordInfo;
    if (info.scenes.length === 0) return;
    let idx = 0;
    if (currentTurn !== undefined) {
      const found = info.scenes.findIndex(s => s.turn === currentTurn);
      if (found >= 0) idx = found;
    }
    setMetadata(info);
    setScene(JSON.parse(replay.current.scene(info.scenes[idx].round, nextSide)) as Scene);
    setSelected(idx);
    setAlternative("");
    setNotice("");
  }

  function chooseScene(index: number) {
    if (!metadata || !replay.current || !metadata.scenes[index]) return;
    setScene(JSON.parse(replay.current.scene(metadata.scenes[index].round, metadata.side)) as Scene);
    setSelected(index);
    setAlternative("");
    setNotice("");
  }

  function newGame(fork: LoadedFork, previous = 0): GameRecord {
    const arm = Math.max(0, fork.info.arms.length - 1);
    return { game: previous, arm, input: fork.info.arms[arm].input, label: fork.info.arms[arm].label,
      battleSeed: randomSeed32(), botSeed: randomSeed32(), outcome: null, forfeit: false,
      finalTurn: null, started: new Date().toISOString(), finished: null };
  }

  async function start() {
    if (!replay.current || !metadata || !scene) return;
    setBusy(true);
    setNotice("");
    await new Promise(resolve => setTimeout(resolve, 0));
    if (!alive.current) return;
    try {
      const json = replay.current.fork(metadata.scenes[selected].round, alternative, poolJson, metadata.side);
      const info = readFork(json);
      const doc = JSON.parse(json) as { opponent_team: unknown[]; position?: { own_sets?: unknown[] } };
      const humanSide = 1 - metadata.botSide;
      const fork: LoadedFork = { json, info, key: `kifu-${metadata.side}-${metadata.scenes[selected].round}-${randomSeed32()}`,
        humanSets: humanSide === info.botSide ? (doc.position?.own_sets ?? []) : doc.opponent_team,
        armNames: info.arms.map(a => choiceName(scene.choices.find(c => c.input === a.input)!)) };
      setRun({ fork, method, record: newGame(fork) });
    } catch (error) {
      console.error("kifu fork", error);
      setNotice(() => toolText("resumeSceneError"));
    } finally { setBusy(false); }
  }

  if (run?.method === "human") return <ForkGame
    key={run.record.game} fork={run.fork} record={run.record} poolJson={poolJson}
    budget={searchProfile(run.fork.info.info).iterations}
    onFinish={() => {}}
    onForfeit={() => setRun(null)} onBack={() => setRun(null)} backLabel={toolText("chooseSceneAgain")}
    onNext={() => setRun({ ...run, record: newGame(run.fork, run.record.game + 1) })}
  />;

  const bot = metadata?.botSide ?? 1;
  const targetSide = metadata?.side ?? bot;
  const isYourSide = targetSide !== bot;
  const active = scene?.view.sides.map(side => side.active === null ? null : side.party[side.active]);
  const sameComparison = method === "bot" && alternative === scene?.played.input;
  const alternativeChoice = scene?.choices.find(c => c.input === alternative);
  const journal = scene ? new Narrator(1 - bot).render(scene.log).slice(-30) : [];

  function actionStat(choice: Choice) {
    if (!hintPolicy) return null;
    return hintPolicy.actions.find(
      a => a.input === choice.input || (choice.input === "move hiddenpower" && a.input.startsWith("move hiddenpower"))
    ) ?? null;
  }

  const recommendedChoice = (() => {
    if (!scene || !hintPolicy) return null;
    if (hintBest) {
      const matched = scene.choices.find(
        c => c.input === hintBest || (c.input === "move hiddenpower" && hintBest.startsWith("move hiddenpower"))
      );
      if (matched) return matched;
    }
    let best: Choice | null = null;
    let bestVisits = -1;
    let bestMean = -1;
    for (const c of scene.choices) {
      const stat = actionStat(c);
      if (!stat) continue;
      if (stat.visits > bestVisits || (stat.visits === bestVisits && stat.mean > bestMean)) {
        best = c;
        bestVisits = stat.visits;
        bestMean = stat.mean;
      }
    }
    return best;
  })();
  const recommendedStat = recommendedChoice ? actionStat(recommendedChoice) : null;

  function choiceOption(choice: Choice) {
    const stat = actionStat(choice);
    const isRecommended = recommendedChoice?.input === choice.input;
    return <label class={`kp-option ${alternative === choice.input ? "is-selected" : ""} ${isRecommended ? "is-recommended" : ""}`} key={choice.input}>
      <input type="radio" name="alternative" value={choice.input} checked={alternative === choice.input} onChange={() => setAlternative(choice.input)} disabled={busy} />
      <span>{choiceName(choice)}{scene?.played.input === choice.input && <small>{toolText("originalMove")}</small>}
        {isRecommended && stat && <small class="kp-rec-badge">{toolText("recommendedMoveBadge", Math.round(stat.mean * 100))}</small>}
        {!isRecommended && stat && stat.visits > 0 && <small>{toolText("estimatedWinRate", Math.round(stat.mean * 100))}</small>}
        {choice.kind === "move" && <small>PP {choice.pp} / {choice.maxpp}</small>}
      </span>
    </label>;
  }

  return <div class="kifu-tool kifu-page"><main class="kp-main">
    {run?.method === "bot" ? <>
      <button class="kp-back" onClick={() => setRun(null)}>{toolText("backToScene")}</button>
      <header class="kp-heading"><h1 ref={heading} tabIndex={-1}>{toolText("compareMoves")}</h1><p class="kp-muted">{toolText("turnWithLabel", run.fork.info.turn, run.fork.armNames.join(" ／ "))}</p></header>
      <ArenaPanel info={run.fork.info} json={run.fork.json} poolJson={poolJson} armNames={run.fork.armNames} fileKey={run.fork.key} simple autoStart />
    </> : screen === "import" ? <>
      <header class="kp-heading"><p class="kp-eyebrow">{toolText("kifuEyebrow")}</p><h1 ref={heading} tabIndex={-1}>{toolText("tryAnotherMove")}</h1><p class="kp-muted">{toolText("kifuIntro")}</p></header>
      <form class="kp-card" onSubmit={e => { e.preventDefault(); if (ready && !busy) void load(paste); }}>
        <label class="kp-label" for="kp-paste">{toolText("pasteReplay")}</label>
        <p id="kp-paste-help" class="kp-muted">{toolText("pasteReplayHelp")}</p>
        <textarea ref={pasteField} id="kp-paste" rows={6} value={paste} onInput={e => setPaste(e.currentTarget.value)} aria-describedby="kp-paste-help kp-import-status" placeholder={toolText("pasteReplayPlaceholder")} spellcheck={false} autoCapitalize="off" />
        <button class="primary kp-full" type="submit" disabled={!ready || busy}>{busy ? toolText("loadingReplay") : ready ? toolText("loadReplay") : toolText("preparingData")}</button>
        <p id="kp-import-status" class="kp-status" role="status">{notice}</p>
      </form>
      <nav class="kp-tools-nav"><a href={import.meta.env.BASE_URL}>{toolText("playBot")}</a></nav>
    </> : metadata && scene && <>
      <button class="kp-back" disabled={busy} onClick={() => { setScreen("import"); setNotice(""); }}>{toolText("loadAnotherReplay")}</button>
      <header class="kp-heading"><p class="kp-eyebrow">{toolText("recordSummary", metadata.turns, metadata.outcome ? toolText("recordEnded") : toolText("recordPartial"))}</p><h1 ref={heading} tabIndex={-1}>{toolText("choosePosition")}</h1></header>
      <section class="kp-card" aria-labelledby="kp-scene-title">
        <h2 id="kp-scene-title">{toolText("chooseScene")}</h2>
        <div class="kp-side-tabs" role="radiogroup" aria-label={toolText("chooseSide")}>
          <label class={`kp-option ${isYourSide ? "is-selected" : ""}`}>
            <input type="radio" name="target-side" checked={isYourSide} onChange={() => changeSide(1 - bot)} disabled={busy} />
            <span>{toolText("yourMoves")}</span>
          </label>
          <label class={`kp-option ${!isYourSide ? "is-selected" : ""}`}>
            <input type="radio" name="target-side" checked={!isYourSide} onChange={() => changeSide(bot)} disabled={busy} />
            <span>{toolText("botMoves")}</span>
          </label>
        </div>
        <label class="kp-sr-only" for="kp-scene">{toolText("resumePosition")}</label>
        <select id="kp-scene" value={selected} disabled={busy} onChange={e => chooseScene(Number(e.currentTarget.value))}>
          {metadata.scenes.map((s, i) => <option value={i} key={s.round}>{toolText("sceneOption", s.turn, s.active[targetSide] ? speciesName(s.active[targetSide]!) : (isYourSide ? toolText("you") : "bot"), choiceName(s.played))}</option>)}
        </select>
        <div class="kp-scene-nav"><button disabled={busy || selected === 0} onClick={() => chooseScene(selected - 1)}>{toolText("previousScene")}</button><button disabled={busy || selected === metadata.scenes.length - 1} onClick={() => chooseScene(selected + 1)}>{toolText("nextScene")}</button></div>
        <div class="kp-scene-state" aria-live="polite" aria-atomic="true">
          <p class="kp-turn">{toolText("beforeTurn", scene.view.turn)}</p>
          <dl class="kp-matchup">{[1 - bot, bot].map(side => <div key={side}><dt>{side === bot ? "bot" : toolText("you")}</dt><dd><strong>{active?.[side] ? speciesName(active[side]!.species) : toolText("awaitingSwitch")}</strong>{active?.[side] && <><span>HP {active[side]!.hp} / {active[side]!.maxhp}</span>{active?.[side]!.status && <span>{statusLongName(active[side]!.status)}</span>}</>}</dd></div>)}</dl>
          <p class="kp-played">{toolText(isYourSide ? "originalYourChoicePrefix" : "originalChoicePrefix")}<strong>{choiceName(scene.played)}</strong>{toolText("originalChoiceSuffix")}</p>
        </div>
        <details class="kp-details"><summary>{toolText("showJournal")}</summary><div class="kp-journal">{journal.map((entry, i) => <p key={i}>{entry.text}</p>)}</div></details>
      </section>
      <section class="kp-card" aria-labelledby="kp-action-title">
        <h2 id="kp-action-title">{toolText(isYourSide ? "chooseYourMove" : "chooseBotMove")}</h2>
        <p class="kp-rec-banner" role="status" aria-live="polite">
          {hintLoading ? toolText("searchingRecommendation") : recommendedChoice && recommendedStat ? (
            <button type="button" class="kp-rec-pick" disabled={busy} onClick={() => setAlternative(recommendedChoice.input)}>
              {toolText("recommendedMoveBanner", choiceName(recommendedChoice), Math.round(recommendedStat.mean * 100))}
            </button>
          ) : null}
        </p>
        {scene.choices.some(c => c.kind === "move") && <fieldset class="kp-choice-field"><legend>{toolText("useMove")}</legend><div class="kp-options">{scene.choices.filter(c => c.kind === "move").map(choiceOption)}</div></fieldset>}
        {scene.choices.some(c => c.kind === "switch") && <fieldset class="kp-choice-field"><legend>{toolText("switchPokemon")}</legend><div class="kp-options">{scene.choices.filter(c => c.kind === "switch").map(choiceOption)}</div></fieldset>}
        {scene.choices.filter(c => c.kind === "pass").map(choiceOption)}
      </section>
      <section class="kp-card" aria-labelledby="kp-method-title">
        <h2 id="kp-method-title">{toolText("chooseMethod")}</h2>
        <div class="kp-methods" role="radiogroup" aria-labelledby="kp-method-title">
          <label class={`kp-option ${method === "human" ? "is-selected" : ""}`}><input type="radio" name="method" checked={method === "human"} onChange={() => setMethod("human")} disabled={busy} /><span>{toolText("playYourself")}<small>{toolText("playYourselfHelp")}</small></span></label>
          <label class={`kp-option ${method === "bot" ? "is-selected" : ""}`}><input type="radio" name="method" checked={method === "bot"} onChange={() => setMethod("bot")} disabled={busy} /><span>{toolText("compareBots")}<small>{toolText("compareBotsHelp")}</small></span></label>
        </div>
        <p class="kp-summary" aria-live="polite">{sameComparison ? toolText("sameMoveWarning") : alternativeChoice ? toolText(isYourSide ? "yourContinuationSummary" : "continuationSummary", scene.view.turn, choiceName(alternativeChoice)) : toolText("chooseMoveHelp")}</p>
        <button class="primary kp-full" disabled={busy || !alternative || sameComparison} onClick={() => void start()}>{busy ? toolText("preparingPosition") : method === "human" ? toolText("playThisMove") : toolText("compareTwoMoves")}</button>
        <p class="kp-status" role="status">{notice}</p>
      </section>
    </>}
  </main></div>;
}
