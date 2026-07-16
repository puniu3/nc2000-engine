// One game: team preview -> battle -> end. The human is always side 0
// (p1); the bot is side 1 and thinks in a Web Worker holding a lockstep
// mirror of the battle. Per request point: collect the picks each owing
// side must make (forced single choices auto-apply; bot preview comes from
// the baked tables when the matchup is baked, else from live search), then
// commit both picks in side order to the main battle AND the mirror.

import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import {
  Battle,
  PreviewTables,
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
  StatusBadge,
  TypeBadge,
} from "./battle-ui";
import { STRENGTHS } from "./app";

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

export function Game(props: {
  pool: MetaPool;
  tables: PreviewTables;
  addedPairs: Set<string>;
  humanIdx: number;
  botIdx: number;
  strength: number;
  onStrength: (n: number) => void;
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

  const battleRef = useRef<Battle | null>(null);
  const botRef = useRef<BotWorker | null>(null);
  const reqRef = useRef<Request | null>(null);
  const aliveRef = useRef(true);
  const strengthRef = useRef(props.strength);
  strengthRef.current = props.strength;
  const narrator = useMemo(() => new Narrator(HUMAN), []);
  const pairPromiseRef = useRef<Promise<string | null> | null>(null);

  const humanTeam = props.pool.teams[props.humanIdx];
  const botTeam = props.pool.teams[props.botIdx];

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
    pairPromiseRef.current = fetchPairJson(props.humanIdx, props.botIdx);
    void bot
      .newBattle(
        JSON.stringify(humanTeam.sets),
        JSON.stringify(botTeam.sets),
        battle.seed(),
      )
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
      if (entries.length > 0) setLog((prev) => [...prev, ...entries]);
    }
    setView(stateView(battle));
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

    if (needs[BOT]) decideBot(req);
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

  async function decideBotPreview(req: Request, legal: Choice[]) {
    const battle = battleRef.current!;
    let picked: string | null = null;
    const pairJson = await pairPromiseRef.current;
    if (!aliveRef.current || req !== reqRef.current) return;
    if (pairJson) {
      const lo = Math.min(props.humanIdx, props.botIdx);
      const hi = Math.max(props.humanIdx, props.botIdx);
      const key = `${lo}-${hi}`;
      try {
        if (!props.addedPairs.has(key)) {
          props.tables.addPair(pairJson);
          props.addedPairs.add(key);
        }
        const res = JSON.parse(props.tables.resolve(battle, BOT)) as {
          found: boolean;
        };
        if (res.found) {
          picked = props.tables.sample(battle, BOT, randomSeed32()) ?? null;
        }
      } catch (e) {
        console.warn("baked table unusable, falling back to search:", e);
        picked = null;
      }
    }
    if (picked) {
      setBotPreviewSrc("table");
      req.picks[BOT] = picked;
      maybeCommit(req);
    } else {
      setBotPreviewSrc("search");
      await searchBot(req, legal);
    }
  }

  async function searchBot(req: Request, legal: Choice[]) {
    const bot = botRef.current!;
    const budget = strengthRef.current;
    setThinking({ done: 0, budget });
    const r = await bot.search(BOT, budget, randomSeed32(), (done, b) => {
      if (aliveRef.current && req === reqRef.current)
        setThinking({ done, budget: b });
    });
    if (!aliveRef.current || req !== reqRef.current) return;
    setThinking(null);
    req.picks[BOT] = r.best ?? legal[0].input;
    maybeCommit(req);
  }

  function humanPick(input: string) {
    const req = reqRef.current;
    if (!req || req.committed || req.picks[HUMAN] !== null) return;
    req.picks[HUMAN] = input;
    setHumanChoices(null);
    setHumanWaiting(true);
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

  function togglePreview(pos: number) {
    setPreviewSel((sel) =>
      sel.includes(pos)
        ? sel.filter((x) => x !== pos)
        : sel.length < 3
          ? [...sel, pos]
          : sel,
    );
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
        <div class="loading-pulse">Setting up battle…</div>
      </div>
    );
  }

  const mine = view.sides[HUMAN];
  const foe = view.sides[BOT];

  if (phase === "preview") {
    return (
      <div class="screen preview-screen">
        <h2>Team preview</h2>
        <section>
          <h3>Foe team ({botTeam.id})</h3>
          <div class="preview-foe">
            {foe.party.map((p, i) => (
              <span class="species-chip" key={i}>
                {p.species} <small>L{p.level}</small>
              </span>
            ))}
          </div>
        </section>
        <section>
          <h3>Your team — pick 3, lead first</h3>
          <div class="preview-grid">
            {mine.party.map((p, i) => {
              const pos = i + 1;
              const order = previewSel.indexOf(pos);
              return (
                <button
                  class={`preview-mon ${order >= 0 ? "selected" : ""}`}
                  key={i}
                  onClick={() => togglePreview(pos)}
                >
                  {order >= 0 && (
                    <span class="pick-badge">
                      {order === 0 ? "Lead" : order + 1}
                    </span>
                  )}
                  <div class="preview-mon-head">
                    <span class="mon-name">{p.species}</span>
                    <span class="mon-level">L{p.level}</span>
                    {p.types.map((t) => (
                      <TypeBadge type={t} key={t} />
                    ))}
                  </div>
                  {p.item && <div class="preview-item">@ {p.item}</div>}
                  <div class="preview-moves">
                    {p.moves.map((m) => m.name).join(" · ")}
                  </div>
                </button>
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
              ? "Confirm picks"
              : `Pick ${3 - previewSel.length} more`}
          </button>
          <span class="bot-preview-note">
            {thinking
              ? `Opponent thinking… ${thinking.done}/${thinking.budget}`
              : botPreviewSrc === "table"
                ? "Opponent picks from the baked equilibrium table"
                : botPreviewSrc === "search"
                  ? "Opponent picks by live search (matchup not baked yet)"
                  : ""}
          </span>
          <button class="ghost quit-btn" onClick={props.onNewTeams}>
            Quit
          </button>
        </div>
      </div>
    );
  }

  const activeMine = mine.active !== null ? mine.party[mine.active] : null;
  const activeFoe = foe.active !== null ? foe.party[foe.active] : null;
  const bench = mine.party.filter((_, i) => i !== mine.active);

  return (
    <div class="screen battle-screen">
      <header class="battle-header">
        <span class="turn-label">Turn {view.turn}</span>
        <label class="strength-label compact">
          <select
            value={props.strength}
            onChange={(e) =>
              props.onStrength(Number((e.target as HTMLSelectElement).value))
            }
          >
            {STRENGTHS.map((s) => (
              <option value={s.iters} key={s.iters}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <button class="ghost quit-btn" onClick={props.onNewTeams}>
          Quit
        </button>
      </header>

      <div class="arena">
        {activeFoe && (
          <ActiveCard
            poke={activeFoe}
            mine={false}
            extra={`${foe.pokemonLeft} left`}
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
          <div class="bench-row">
            {bench.map((p, i) => (
              <span
                class={`bench-chip ${p.fainted ? "fainted" : ""}`}
                key={i}
              >
                {p.species}{" "}
                {p.fainted ? (
                  <small>fnt</small>
                ) : (
                  <small>
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

      <div class="choice-panel">
        {phase === "end" ? (
          <EndBanner
            outcome={view.outcome}
            onRematch={props.onRematch}
            onNewTeams={props.onNewTeams}
          />
        ) : humanChoices ? (
          <ChoiceButtons choices={humanChoices} onPick={humanPick} />
        ) : (
          <ThinkingBar thinking={thinking} waiting={humanWaiting} />
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------- components

function LogPane(props: { log: LogEntry[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [props.log]);
  return (
    <div class="log-pane" ref={ref}>
      {props.log.map((e, i) => (
        <div class={`log-line log-${e.kind}`} key={i}>
          {e.text}
        </div>
      ))}
    </div>
  );
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
          {moves.map((m) => (
            <button
              class="move-btn"
              key={m.input}
              onClick={() => props.onPick(m.input)}
            >
              <span class="move-name">{m.name}</span>
              <span class="move-meta">
                <TypeBadge type={m.type} />
                <span class="move-cat">{m.category}</span>
                {m.basePower > 0 && (
                  <span class="move-bp">{m.basePower} BP</span>
                )}
                {m.pp >= 0 && (
                  <span class="move-pp">
                    PP {m.pp}/{m.maxpp}
                  </span>
                )}
              </span>
            </button>
          ))}
        </div>
      )}
      {switches.length > 0 && (
        <div class="switch-row">
          {switches.map((s) => (
            <button
              class="switch-btn"
              key={s.input}
              onClick={() => props.onPick(s.input)}
            >
              <span class="switch-label">switch</span>
              <span class="mon-name">{s.species}</span>
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
      <span>
        {t
          ? `Bot is thinking… ${t.done} / ${t.budget}`
          : props.waiting
            ? "Waiting for the bot…"
            : "…"}
      </span>
      {t && (
        <div class="think-progress">
          <div
            class="think-fill"
            style={{ width: `${(t.done / t.budget) * 100}%` }}
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
}) {
  const text =
    props.outcome === "p1"
      ? "You win!"
      : props.outcome === "p2"
        ? "The bot wins!"
        : "Tie";
  return (
    <div class="end-banner">
      <div class={`end-text ${props.outcome === "p1" ? "win" : "lose"}`}>
        {text}
      </div>
      <div class="end-actions">
        <button class="primary" onClick={props.onRematch}>
          Rematch
        </button>
        <button class="ghost" onClick={props.onNewTeams}>
          New teams
        </button>
      </div>
    </div>
  );
}
