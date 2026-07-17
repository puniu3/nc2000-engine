// One game: team preview -> battle -> end. The human is always side 0
// (p1); the bot is side 1 and thinks in a Web Worker holding a lockstep
// mirror of the battle. Per request point: collect the picks each owing
// side must make (forced single choices auto-apply), then commit both picks
// in side order to the main battle AND the mirror.
//
// Information policy (M12): OPEN TEAM SHEET — both sides' sets are public
// (the bot's belief is pinned to the human's true sets in the worker), only
// selection (which 3 of 6 + lead, until revealed) is hidden: the worker's
// searcher determinizes the human's unseen picks per iteration. Bot preview
// comes from the M8 baked table whenever the matchup is baked (the worker
// reports "table"), else the live search at the preview root ("search").
// Strength is fixed at max (BUDGET) — ponder hides the wait.

import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import {
  Battle,
  getDex,
  legalChoices,
  needsChoice,
  newBattleSeed,
  randomSeed32,
  stateView,
  takeNewLog,
} from "./engine";
import { fetchPairJson } from "./data";
import { BotWorker } from "./bot";
import { Narrator } from "./narrate";
import type {
  Choice,
  LogEntry,
  MetaPool,
  MoveChoice,
  StateView,
  SwitchChoice,
  TeamChoice,
} from "./types";
import {
  ActiveCard,
  FieldStrip,
  HpBar,
  Lvl,
  StatusBadge,
  TypeBadge,
} from "./battle-ui";
import {
  moveName,
  speciesName,
  statusLongName,
  toId,
  typeName,
  ui,
} from "./i18n";
import { announce, announceAssertive } from "./announcer";
import { moveNote } from "./behavior-notes";
import { noteRef } from "./tooltip";
import { BUDGET, type SelectedTeam } from "./app";
import { Modal } from "./modal";
import { sheetMon } from "./set-info";
import { MonSheet, SetDetail, TeamSheets } from "./team-sheet";

const HUMAN = 0;
const BOT = 1;

interface Request {
  needs: [boolean, boolean];
  picks: [string | null, string | null];
  committed: boolean;
}

interface Thinking {
  done: number;
  budget: number;
}

const fmtK = (n: number) =>
  n >= 10000
    ? `${Math.round(n / 1000)}k`
    : n >= 1000
      ? `${(n / 1000).toFixed(1)}k`
      : String(n);

/** Whether keyboard focus has nowhere meaningful to be — the previously
 * focused control unmounted and focus fell back to <body>. UI-4 moves
 * focus ONLY in this state: an orphaned focus is relocated to the new
 * screen/decision heading, a live one (reading the team-sheets modal,
 * resting on Quit) is never yanked. */
function focusIsOrphaned(): boolean {
  const ae = document.activeElement;
  return !ae || ae === document.body || !ae.isConnected;
}

/** Subtle bot-think status while the human deliberates: "thinking" =
 * required budget still running, "pondering" = budget done, bonus
 * iterations accumulating until the human commits. Hidden from screen
 * readers entirely — the ticking counter must not reach the SR (the
 * polite announcer speaks the meaningful transitions instead). */
function ThinkChip(props: { thinking: Thinking | null }) {
  const t = props.thinking;
  if (!t) return null;
  const pondering = t.done >= t.budget;
  return (
    <span
      class={`think-chip ${pondering ? "pondering" : ""}`}
      aria-hidden="true"
      data-done={t.done}
      data-budget={t.budget}
      data-pondering={pondering ? "1" : "0"}
    >
      <span class="think-chip-dot" />
      {pondering
        ? ui().ponderChip(fmtK(t.done - t.budget))
        : ui().thinkChip(fmtK(t.done), fmtK(t.budget))}
    </span>
  );
}

export function Game(props: {
  pool: MetaPool;
  poolJson: string;
  humanTeam: SelectedTeam;
  botIdx: number;
  onRematch: () => void;
  onNewTeams: () => void;
}) {
  const [phase, setPhase] = useState<"init" | "preview" | "battle" | "end">(
    "init",
  );
  const [view, setView] = useState<StateView | null>(null);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [humanChoices, setHumanChoices] = useState<Choice[] | null>(null);
  const [humanWaiting, setHumanWaiting] = useState(false); // picked, bot still thinking
  const [thinking, setThinking] = useState<Thinking | null>(null);
  const [previewSel, setPreviewSel] = useState<number[]>([]); // 1-based display positions, lead first
  const [botPreviewSrc, setBotPreviewSrc] = useState<
    "table" | "search" | null
  >(null);
  const [sheetOpen, setSheetOpen] = useState(false); // battle: team-sheets modal

  const battleRef = useRef<Battle | null>(null);
  const botRef = useRef<BotWorker | null>(null);
  const reqRef = useRef<Request | null>(null);
  const aliveRef = useRef(true);
  // Foe species publicly revealed so far (ever seen active at a request
  // point — the same state the battle screen renders; faints are handled
  // at render time). Drives the team-sheet marks WITHOUT reading which 3
  // the foe picked from engine internals.
  const revealedFoeRef = useRef<Set<string>>(new Set());
  const narrator = useMemo(() => new Narrator(HUMAN), []);
  const pairPromiseRef = useRef<Promise<string | null> | null>(null);
  // UI-4 focus targets (tabindex=-1 headings).
  const previewHeadRef = useRef<HTMLHeadingElement>(null);
  const battleHeadRef = useRef<HTMLHeadingElement>(null);
  const choiceHeadRef = useRef<HTMLHeadingElement>(null);
  const endHeadRef = useRef<HTMLHeadingElement>(null);

  const humanTeam = props.humanTeam;
  const botTeam = props.pool.teams[props.botIdx];

  // UI-4: a new decision request — announce it politely and, if focus is
  // orphaned (the picked button unmounted), land on the decision heading
  // so one Tab reaches the first choice. Declared before the phase effect
  // so on the first battle request the decision heading wins.
  useEffect(() => {
    if (!humanChoices || phase !== "battle") return;
    const hasMove = humanChoices.some((c) => c.kind === "move");
    announce(hasMove ? ui().srYourTurn : ui().srChooseSwitch);
    // Relocate an orphaned focus — or our own battle-h1 transition focus
    // (first decision arrives moments after the preview->battle focus).
    if (focusIsOrphaned() || document.activeElement === battleHeadRef.current)
      choiceHeadRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [humanChoices, phase]);

  // UI-4: screen transitions — focus the new screen's heading; the
  // outcome is additionally announced assertively.
  useEffect(() => {
    if (phase === "preview") {
      if (focusIsOrphaned()) previewHeadRef.current?.focus();
    } else if (phase === "battle") {
      if (focusIsOrphaned()) battleHeadRef.current?.focus();
    } else if (phase === "end") {
      const o = view?.outcome;
      announceAssertive(
        o === "p1" ? ui().youWin : o === "p2" ? ui().botWins : ui().tie,
      );
      if (focusIsOrphaned()) endHeadRef.current?.focus();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  // ------------------------------------------------------------ lifecycle

  useEffect(() => {
    aliveRef.current = true;
    const bot = new BotWorker();
    botRef.current = bot;
    const battle = new Battle(
      getDex(),
      JSON.stringify(humanTeam.sets),
      JSON.stringify(botTeam.sets),
      newBattleSeed(),
    );
    battleRef.current = battle;
    // Baked pair tables exist only between pool teams — a custom human
    // team (poolIdx null) always sends the bot preview to live search.
    pairPromiseRef.current =
      humanTeam.poolIdx === null
        ? Promise.resolve(null)
        : fetchPairJson(humanTeam.poolIdx, props.botIdx);
    void bot
      .newBattle(JSON.stringify(humanTeam.sets), JSON.stringify(botTeam.sets), battle.seed(), {
        poolJson: props.poolJson,
        side: BOT,
        seed: randomSeed32(),
      })
      .then(() => {
        if (!aliveRef.current) return;
        drain();
        startRequest();
      });
    return () => {
      aliveRef.current = false;
      bot.terminate();
      battle.free();
      battleRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ------------------------------------------------------- decision loop

  /** Pull new protocol lines + fresh state view into the UI. */
  function drain() {
    const battle = battleRef.current!;
    const lines = takeNewLog(battle);
    if (lines.length > 0) {
      const entries = narrator.render(lines);
      if (entries.length > 0) {
        setLog((prev) => [...prev, ...entries]);
        // One batched polite announcement per drain. Result lines are
        // excluded — the outcome banner announces assertively instead.
        const speak = entries
          .filter((e) => e.kind !== "result")
          .map((e) => e.text)
          .join(" ");
        if (speak) announce(speak);
      }
    }
    const v = stateView(battle);
    // Track the foe's public reveals from what the screen shows anyway:
    // whichever foe mon is active at a request point has appeared.
    const foeSide = v.sides[BOT];
    if (foeSide.active !== null)
      revealedFoeRef.current.add(toId(foeSide.party[foeSide.active].species));
    setView(v);
  }

  function startRequest() {
    if (!aliveRef.current) return;
    const battle = battleRef.current!;
    const needs = needsChoice(battle);
    if (!needs[HUMAN] && !needs[BOT]) {
      setPhase("end");
      return;
    }
    const req: Request = { needs, picks: [null, null], committed: false };
    reqRef.current = req;
    setHumanWaiting(false);

    // Human side first: forced picks land before the bot search launches,
    // so decideBot sees whether the human still owes (=> ponder or not).
    if (needs[HUMAN]) {
      const legal = legalChoices(battle, HUMAN);
      if (legal.length === 1) {
        req.picks[HUMAN] = legal[0].input;
        setHumanChoices(null);
      } else if (legal[0].kind === "team") {
        setPhase("preview");
        setPreviewSel([]);
        setHumanChoices(legal);
      } else {
        setPhase("battle");
        setHumanChoices(legal);
      }
    } else {
      setHumanChoices(null);
      if (phase === "init") setPhase("battle");
    }
    if (needs[BOT]) decideBot(req);
    maybeCommit(req);
  }

  function decideBot(req: Request) {
    const battle = battleRef.current!;
    const legal = legalChoices(battle, BOT);
    if (legal.length === 1) {
      req.picks[BOT] = legal[0].input;
      return;
    }
    if (legal[0].kind === "team") {
      void decideBotPreview(req, legal);
      return;
    }
    void searchBot(req, legal);
  }

  /** Bot team preview: feed the baked pair table (when it exists) to the
   * worker, which answers from the M8 equilibrium ("table") or falls back
   * to the live preview search ("search" — matchup not baked yet). */
  async function decideBotPreview(req: Request, legal: Choice[]) {
    const pairJson = await pairPromiseRef.current;
    if (!aliveRef.current || req !== reqRef.current) return;
    if (pairJson) botRef.current!.addPair(pairJson);
    await searchBot(req, legal);
  }

  async function searchBot(req: Request, legal: Choice[]) {
    const bot = botRef.current!;
    // Ponder iff the human still owes a pick at launch: the search then
    // keeps running past its budget (bonus strength) until the human
    // commits (humanPick -> flush) or the ponder cap.
    const ponder = req.needs[HUMAN] && req.picks[HUMAN] === null;
    // A non-ponder search is a genuine wait for the human — say so once.
    // Ponder searches run behind the human's own deliberation: silent.
    if (!ponder) announce(ui().srBotThinking);
    setThinking({ done: 0, budget: BUDGET });
    const r = await bot.search(BOT, BUDGET, randomSeed32(), ponder, (done, b) => {
      if (aliveRef.current && req === reqRef.current)
        setThinking({ done, budget: b });
    });
    if (!aliveRef.current || req !== reqRef.current) return;
    setThinking(null);
    // Previews report their source (baked table vs live search).
    if (r.src && legal[0].kind === "team") setBotPreviewSrc(r.src);
    req.picks[BOT] = r.best ?? legal[0].input;
    maybeCommit(req);
  }

  function humanPick(input: string) {
    const req = reqRef.current;
    if (!req || req.committed || req.picks[HUMAN] !== null) return;
    req.picks[HUMAN] = input;
    setHumanChoices(null);
    setHumanWaiting(true);
    // A pondering bot search returns its best immediately.
    if (req.needs[BOT] && req.picks[BOT] === null) botRef.current!.flush();
    maybeCommit(req);
  }

  function maybeCommit(req: Request) {
    if (req.committed) return;
    for (const side of [HUMAN, BOT]) {
      if (req.needs[side] && req.picks[side] === null) return;
    }
    req.committed = true;
    const battle = battleRef.current!;
    const picks: [number, string][] = [];
    for (const side of [HUMAN, BOT]) {
      if (req.needs[side]) picks.push([side, req.picks[side]!]);
    }
    try {
      for (const [side, input] of picks) battle.applyChoice(side, input);
    } catch (e) {
      console.error("applyChoice failed:", e, picks);
      return;
    }
    botRef.current!.apply(picks);
    setHumanWaiting(false);
    setThinking(null);
    if (phase !== "battle") setPhase("battle");
    drain();
    // Yield a frame before enumerating the next request point.
    setTimeout(() => {
      if (aliveRef.current) startRequest();
    }, 0);
  }

  // -------------------------------------------------------------- preview

  /** NC2000 `Max Total Level = 155`: the 3 picked mons' level sum may not
   * exceed 155. The engine enforces it (legalChoices omits overweight
   * picks; applyChoice rejects them), so the picker must not offer dead
   * ends: a mon is selectable only while some legal completion exists. */
  const MAX_TOTAL_LEVEL = 155;

  /** Can the current selection, extended with display position `pos`, still
   * be completed to a legal 3-pick? (Fill the remaining slots with the
   * lightest unselected mons — if even that overshoots, `pos` is a dead
   * end.) */
  function fitsLevelCap(levels: number[], sel: number[], pos: number): boolean {
    const picked = [...sel, pos];
    let sum = picked.reduce((a, p) => a + levels[p - 1], 0);
    const rest = levels
      .map((_, i) => i + 1)
      .filter((p) => !picked.includes(p))
      .map((p) => levels[p - 1])
      .sort((a, b) => a - b);
    for (let i = 0; i < 3 - picked.length; i++) sum += rest[i];
    return sum <= MAX_TOTAL_LEVEL;
  }

  function togglePreview(pos: number) {
    const levels = view!.sides[HUMAN].party.map((p) => p.level);
    const sel = previewSel;
    const next = sel.includes(pos)
      ? sel.filter((x) => x !== pos)
      : sel.length < 3 && fitsLevelCap(levels, sel, pos)
        ? [...sel, pos]
        : sel;
    if (next === sel) return; // over-cap / already-3: aria-disabled no-op
    setPreviewSel(next);
    // Keep the running total audible: sighted players watch the sum tick.
    const sum = next.reduce((a, p) => a + levels[p - 1], 0);
    announce(ui().levelSum(sum, MAX_TOTAL_LEVEL));
  }

  function confirmPreview() {
    if (previewSel.length !== 3 || !humanChoices) return;
    const [a, b, c] = previewSel;
    const match = humanChoices.find(
      (ch): ch is TeamChoice =>
        ch.kind === "team" &&
        ch.slots[0] === a &&
        ((ch.slots[1] === b && ch.slots[2] === c) ||
          (ch.slots[1] === c && ch.slots[2] === b)),
    );
    if (match) humanPick(match.input);
    else console.error("no legal team choice for", previewSel);
  }

  // --------------------------------------------------------------- render

  if (!view) {
    return (
      <div class="center-screen">
        <div class="loading-pulse">{ui().settingUp}</div>
      </div>
    );
  }

  const mine = view.sides[HUMAN];
  const foe = view.sides[BOT];

  /** The static set JSON behind display position i (open team sheet: the
   * detail panels render from the client-side team JSON, not the engine).
   * Sets order matches the preview display order; the species check guards
   * against drift, falling back to a species lookup. */
  function ownSet(i: number, species: string): unknown {
    const sets = humanTeam.sets as { species?: string }[];
    if (sets[i] && toId(sets[i].species ?? "") === toId(species)) return sets[i];
    return sets.find((s) => toId(s.species ?? "") === toId(species)) ?? sets[i];
  }

  if (phase === "preview") {
    return (
      <main class="screen preview-screen">
        <h1 class="screen-title" tabIndex={-1} ref={previewHeadRef}>
          {ui().teamPreview}
        </h1>
        <p class="sheet-hint">{ui().previewTapHint}</p>
        <section>
          <h2 class="sub-h">{ui().foeTeam(botTeam.id)}</h2>
          <ul class="sheet-list">
            {botTeam.sets.map((s, i) => (
              <li key={i}>
                <MonSheet mon={sheetMon(s)} />
              </li>
            ))}
          </ul>
        </section>
        <section>
          <h2 class="sub-h">{ui().yourTeamPick}</h2>
          <div class="preview-grid">
            {mine.party.map((p, i) => {
              const pos = i + 1;
              const order = previewSel.indexOf(pos);
              const levels = mine.party.map((q) => q.level);
              // Max Total Level: gray out mons with no legal completion left
              // (only while a slot is open — a full selection already just
              // ignores further clicks)
              const overCap =
                order < 0 &&
                previewSel.length < 3 &&
                !fitsLevelCap(levels, previewSel, pos);
              // Pick-control name: species, level, types, pick state. The
              // set body (item / moves + notes) is no longer folded into
              // the label — it sits permanently visible right below, as
              // browseable content (UI-5), so repeating it here would be
              // duplication spam on every tab stop.
              const pickLabel = [
                speciesName(p.species),
                ui().srLevel(p.level),
                p.types.map(typeName).join("/"),
                ...(order >= 0 ? [ui().srPicked(order)] : []),
              ].join(", ");
              return (
                // UI-5: the card is a plain container; the pick control is
                // its header button and the full set detail sits below as a
                // sibling — the detail's noted rows are tabindex=0 toggletip
                // anchors (UI-3) and must never nest inside the button.
                <div
                  class={`preview-cell ${order >= 0 ? "selected" : ""} ${overCap ? "over-cap" : ""}`}
                  key={i}
                >
                  <button
                    class="preview-mon"
                    // aria-disabled (not disabled): the button stays
                    // focusable so the cap reason stays reachable; the
                    // click is a no-op via togglePreview's cap guard.
                    aria-disabled={overCap}
                    aria-pressed={order >= 0}
                    aria-label={pickLabel}
                    aria-describedby={overCap ? `cap-note-${pos}` : undefined}
                    data-mon={toId(p.species)}
                    onClick={() => togglePreview(pos)}
                  >
                    {order >= 0 && (
                      <span class="pick-badge" aria-hidden="true">
                        {order === 0 ? ui().lead : order + 1}
                      </span>
                    )}
                    {overCap && (
                      <span class="cap-badge" id={`cap-note-${pos}`}>
                        {ui().overLevelCap(MAX_TOTAL_LEVEL)}
                      </span>
                    )}
                    <div class="preview-mon-head">
                      <span class="mon-name">{speciesName(p.species)}</span>
                      <span class="mon-level">
                        <Lvl n={p.level} />
                      </span>
                      {p.types.map((t) => (
                        <TypeBadge type={t} key={t} />
                      ))}
                    </div>
                  </button>
                  <SetDetail mon={sheetMon(ownSet(i, p.species))} />
                </div>
              );
            })}
          </div>
        </section>
        <div class="preview-actions">
          <button
            class="primary"
            disabled={previewSel.length !== 3}
            onClick={confirmPreview}
          >
            {previewSel.length === 3
              ? ui().confirmPicks
              : ui().pickMore(3 - previewSel.length)}
          </button>
          <span class="level-sum">
            {ui().levelSum(
              previewSel.reduce((a, p) => a + mine.party[p - 1].level, 0),
              MAX_TOTAL_LEVEL,
            )}
          </span>
          <span class="bot-preview-note">
            {thinking ? (
              <ThinkChip thinking={thinking} />
            ) : botPreviewSrc === "table" ? (
              ui().previewFromTable
            ) : botPreviewSrc === "search" ? (
              ui().previewFromSearch
            ) : (
              ""
            )}
          </span>
          <button class="ghost quit-btn" onClick={props.onNewTeams}>
            {ui().quit}
          </button>
        </div>
      </main>
    );
  }

  const activeMine = mine.active !== null ? mine.party[mine.active] : null;
  const activeFoe = foe.active !== null ? foe.party[foe.active] : null;
  const bench = mine.party.filter((_, i) => i !== mine.active);

  return (
    <main class="screen battle-screen">
      <h1 class="sr-only" tabIndex={-1} ref={battleHeadRef}>
        {ui().srBattleHeading}
      </h1>
      <header class="battle-header">
        <span class="turn-label">{ui().turnLabel(view.turn)}</span>
        {humanChoices && <ThinkChip thinking={thinking} />}
        <button
          class="ghost sheets-btn"
          onClick={() => setSheetOpen(true)}
        >
          {ui().teamSheets}
        </button>
        <button class="ghost quit-btn" onClick={props.onNewTeams}>
          {ui().quit}
        </button>
      </header>

      {sheetOpen && (
        <Modal title={ui().teamSheets} onClose={() => setSheetOpen(false)}>
          <TeamSheets
            mineId={humanTeam.id}
            mineSets={humanTeam.sets}
            foeId={botTeam.id}
            foeSets={botTeam.sets}
            view={view}
            revealedFoe={revealedFoeRef.current}
          />
        </Modal>
      )}

      <div class="arena">
        {activeFoe && (
          <ActiveCard
            poke={activeFoe}
            mine={false}
            extra={ui().nLeft(foe.pokemonLeft)}
          />
        )}
        <FieldStrip
          weather={view.field.weather}
          pseudo={view.field.pseudoWeather}
          mineConds={mine.sideConditions}
          foeConds={foe.sideConditions}
        />
        {activeMine && <ActiveCard poke={activeMine} mine={true} />}
        {bench.length > 0 && (
          <div class="bench-row" role="group" aria-label={ui().srBench}>
            {bench.map((p, i) => (
              <span
                class={`bench-chip ${p.fainted ? "fainted" : ""}`}
                key={i}
              >
                {speciesName(p.species)}{" "}
                {p.fainted ? (
                  <small>
                    <span aria-hidden="true">{ui().fnt}</span>
                    <span class="sr-only">{statusLongName("fnt")}</span>
                  </small>
                ) : (
                  <small>
                    <span class="sr-only">HP </span>
                    {p.hp}/{p.maxhp}
                  </small>
                )}
                <StatusBadge status={p.status} />
              </span>
            ))}
          </div>
        )}
      </div>

      <LogPane log={log} />

      <section class="choice-panel" aria-label={ui().srYourAction}>
        {phase !== "end" && (
          <h2 class="sr-only" tabIndex={-1} ref={choiceHeadRef}>
            {ui().srYourAction}
          </h2>
        )}
        {phase === "end" ? (
          <EndBanner
            outcome={view.outcome}
            onRematch={props.onRematch}
            onNewTeams={props.onNewTeams}
            headingRef={endHeadRef}
          />
        ) : humanChoices ? (
          <ChoiceButtons choices={humanChoices} onPick={humanPick} />
        ) : (
          <ThinkingBar thinking={thinking} waiting={humanWaiting} />
        )}
      </section>
    </main>
  );
}

// ------------------------------------------------------------- components

/** Visible battle log. Deliberately NOT a live region (no role="log"):
 * new lines are announced once, batched, by the off-screen announcer —
 * a VDOM-diffed live region here would double-announce on re-render.
 * Labelled + focusable so it stays reachable for browsing/scrolling. */
function LogPane(props: { log: LogEntry[] }) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [props.log]);
  return (
    <section
      class="log-pane"
      ref={ref}
      aria-label={ui().srBattleLog}
      tabIndex={0}
    >
      {props.log.map((e, i) => (
        <div class={`log-line log-${e.kind}`} key={i}>
          {e.text}
        </div>
      ))}
    </section>
  );
}

/** Complete spoken move name: name, type, category, power, PP. The
 * behavior note (when present) rides separately via aria-describedby.
 * Typeless moves (Curse's "???") skip the unpronounceable type. */
function moveAria(m: MoveChoice): string {
  const type = m.type === "???" ? "" : `${typeName(m.type)} `;
  const bits = [moveName(m.name), `${type}${ui().moveCat(m.category)}`];
  if (m.basePower > 0) bits.push(ui().bp(m.basePower));
  if (m.pp >= 0) bits.push(`PP ${m.pp}/${m.maxpp}`);
  return bits.join(", ");
}

/** Complete spoken switch name: species, HP percent, status. */
function switchAria(s: SwitchChoice): string {
  const pct =
    s.maxhp > 0
      ? s.hp <= 0
        ? 0
        : Math.max(1, Math.round((s.hp / s.maxhp) * 100))
      : 0;
  const base = ui().srSwitchTo(speciesName(s.species), pct);
  return s.status ? `${base}, ${statusLongName(s.status)}` : base;
}

function ChoiceButtons(props: {
  choices: Choice[];
  onPick: (input: string) => void;
}) {
  const moves = props.choices.filter(
    (c): c is MoveChoice => c.kind === "move",
  );
  const switches = props.choices.filter(
    (c): c is SwitchChoice => c.kind === "switch",
  );
  const others = props.choices.filter(
    (c) => c.kind !== "move" && c.kind !== "switch",
  );
  return (
    <div class="choices">
      {moves.length > 0 && (
        <div class="move-grid">
          {moves.map((m) => {
            const note = moveNote(m.name);
            return (
              <button
                class={`move-btn${note ? " has-note" : ""}`}
                key={m.input}
                aria-label={moveAria(m)}
                data-move={toId(m.name)}
                onClick={() => props.onPick(m.input)}
              >
                <span class="move-name">{moveName(m.name)}</span>
                <span class="move-meta">
                  <TypeBadge type={m.type} />
                  <span class="move-cat">{ui().moveCat(m.category)}</span>
                  {m.basePower > 0 && (
                    <span class="move-bp">{ui().bp(m.basePower)}</span>
                  )}
                  {m.pp >= 0 && (
                    <span class="move-pp">
                      PP {m.pp}/{m.maxpp}
                    </span>
                  )}
                </span>
                {note && (
                  <span class="bn-note" ref={noteRef}>
                    {note}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
      {switches.length > 0 && (
        <div class="switch-row">
          {switches.map((s) => (
            <button
              class="switch-btn"
              key={s.input}
              aria-label={switchAria(s)}
              onClick={() => props.onPick(s.input)}
            >
              <span class="switch-label">{ui().switchLabel}</span>
              <span class="mon-name">{speciesName(s.species)}</span>
              <span class="switch-hp">
                {s.hp}/{s.maxhp}
              </span>
              <StatusBadge status={s.status} />
              <HpBar pct={s.maxhp > 0 ? (s.hp / s.maxhp) * 100 : 0} />
            </button>
          ))}
        </div>
      )}
      {others.map((c) => (
        <button
          class="switch-btn"
          key={c.input}
          onClick={() => props.onPick(c.input)}
        >
          {c.input}
        </button>
      ))}
    </div>
  );
}

function ThinkingBar(props: { thinking: Thinking | null; waiting: boolean }) {
  const t = props.thinking;
  return (
    <div class="thinking-bar">
      <span class="thinking-dot" />
      {/* The ticking counter is visual-only; screen readers get one
          static line (the transition itself was announced politely). */}
      <span aria-hidden="true">
        {t
          ? t.done >= t.budget
            ? ui().botFinishing // flush in flight: answer is imminent
            : ui().botThinking(t.done, t.budget)
          : props.waiting
            ? ui().waitingBot
            : "…"}
      </span>
      <span class="sr-only">{ui().srBotThinking}</span>
      {t && (
        <div class="think-progress" aria-hidden="true">
          <div
            class="think-fill"
            style={{ width: `${Math.min(100, (t.done / t.budget) * 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}

function EndBanner(props: {
  outcome: "p1" | "p2" | "tie" | null;
  onRematch: () => void;
  onNewTeams: () => void;
  headingRef: { current: HTMLHeadingElement | null };
}) {
  const text =
    props.outcome === "p1"
      ? ui().youWin
      : props.outcome === "p2"
        ? ui().botWins
        : ui().tie;
  return (
    <div class="end-banner">
      <h2
        class={`end-text ${props.outcome === "p1" ? "win" : "lose"}`}
        tabIndex={-1}
        ref={props.headingRef}
      >
        {text}
      </h2>
      <div class="end-actions">
        <button class="primary" onClick={props.onRematch}>
          {ui().rematch}
        </button>
        <button class="ghost" onClick={props.onNewTeams}>
          {ui().newTeams}
        </button>
      </div>
    </div>
  );
}
