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
import { searchProfile } from "./search-profile";
import { Narrator } from "./narrate";
import type { Choice, StateView } from "./types";
import type { InfoMode } from "./info-mode";
import "./kifu.css";

interface SceneIndex { round: number; turn: number; active: (string | null)[]; played: Choice }
interface RecordInfo { scenes: SceneIndex[]; turns: number; outcome: string | null; botSide: number; info: InfoMode }
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
  const replay = useRef<Kifu | null>(null);
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
        const code = new URLSearchParams(location.hash.slice(1)).get("kifu") ?? new URLSearchParams(location.search).get("kifu");
        if (code) { setPaste(code); await load(code); }
      } catch (error) {
        console.error("kifu setup", error);
        if (alive.current) setNotice(() => toolText("battleDataError"));
      }
    })();
    return () => { alive.current = false; replay.current?.free(); replay.current = null; };
  }, []);

  useEffect(() => { heading.current?.focus(); }, [screen, run !== null]);

  async function load(text: string) {
    setNotice("");
    setBusy(true);
    await new Promise(resolve => setTimeout(resolve, 0));
    if (!alive.current) return;
    let next: Kifu | null = null;
    try {
      const code = extractKifu(text);
      if (!code) throw new Error("not a replay code");
      next = new Kifu(getDex(), code);
      const info = JSON.parse(next.scenes()) as RecordInfo;
      if (info.scenes.length === 0) {
        setNotice(() => toolText("noReplayScenes"));
        return;
      }
      const first = JSON.parse(next.scene(info.scenes[0].round)) as Scene;
      replay.current?.free();
      replay.current = next;
      next = null;
      setMetadata(info);
      setScene(first);
      setSelected(0);
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

  function chooseScene(index: number) {
    if (!metadata || !replay.current || !metadata.scenes[index]) return;
    setScene(JSON.parse(replay.current.scene(metadata.scenes[index].round)) as Scene);
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
      const json = replay.current.fork(metadata.scenes[selected].round, alternative, poolJson);
      const info = readFork(json);
      const doc = JSON.parse(json) as { opponent_team: unknown[] };
      const fork: LoadedFork = { json, info, key: `kifu-${metadata.scenes[selected].round}-${randomSeed32()}`,
        humanSets: doc.opponent_team, armNames: info.arms.map(a => choiceName(scene.choices.find(c => c.input === a.input)!)) };
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
  const active = scene?.view.sides.map(side => side.active === null ? null : side.party[side.active]);
  const sameComparison = method === "bot" && alternative === scene?.played.input;
  const alternativeChoice = scene?.choices.find(c => c.input === alternative);
  const journal = scene ? new Narrator(1 - bot).render(scene.log).slice(-30) : [];

  function choiceOption(choice: Choice) {
    return <label class={`kp-option ${alternative === choice.input ? "is-selected" : ""}`} key={choice.input}>
      <input type="radio" name="alternative" value={choice.input} checked={alternative === choice.input} onChange={() => setAlternative(choice.input)} disabled={busy} />
      <span>{choiceName(choice)}{scene?.played.input === choice.input && <small>{toolText("originalMove")}</small>}
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
        <label class="kp-sr-only" for="kp-scene">{toolText("resumePosition")}</label>
        <select id="kp-scene" value={selected} disabled={busy} onChange={e => chooseScene(Number(e.currentTarget.value))}>
          {metadata.scenes.map((s, i) => <option value={i} key={s.round}>{toolText("sceneOption", s.turn, s.active[bot] ? speciesName(s.active[bot]!) : "bot", choiceName(s.played))}</option>)}
        </select>
        <div class="kp-scene-nav"><button disabled={busy || selected === 0} onClick={() => chooseScene(selected - 1)}>{toolText("previousScene")}</button><button disabled={busy || selected === metadata.scenes.length - 1} onClick={() => chooseScene(selected + 1)}>{toolText("nextScene")}</button></div>
        <div class="kp-scene-state" aria-live="polite" aria-atomic="true">
          <p class="kp-turn">{toolText("beforeTurn", scene.view.turn)}</p>
          <dl class="kp-matchup">{[1 - bot, bot].map(side => <div key={side}><dt>{side === bot ? "bot" : toolText("you")}</dt><dd><strong>{active?.[side] ? speciesName(active[side]!.species) : toolText("awaitingSwitch")}</strong>{active?.[side] && <><span>HP {active[side]!.hp} / {active[side]!.maxhp}</span>{active[side]!.status && <span>{statusLongName(active[side]!.status)}</span>}</>}</dd></div>)}</dl>
          <p class="kp-played">{toolText("originalChoicePrefix")}<strong>{choiceName(scene.played)}</strong>{toolText("originalChoiceSuffix")}</p>
        </div>
        <details class="kp-details"><summary>{toolText("showJournal")}</summary><div class="kp-journal">{journal.map((entry, i) => <p key={i}>{entry.text}</p>)}</div></details>
      </section>
      <section class="kp-card" aria-labelledby="kp-action-title">
        <h2 id="kp-action-title">{toolText("chooseBotMove")}</h2>
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
        <p class="kp-summary" aria-live="polite">{sameComparison ? toolText("sameMoveWarning") : alternativeChoice ? toolText("continuationSummary", scene.view.turn, choiceName(alternativeChoice)) : toolText("chooseMoveHelp")}</p>
        <button class="primary kp-full" disabled={busy || !alternative || sameComparison} onClick={() => void start()}>{busy ? toolText("preparingPosition") : method === "human" ? toolText("playThisMove") : toolText("compareTwoMoves")}</button>
        <p class="kp-status" role="status">{notice}</p>
      </section>
    </>}
  </main></div>;
}
