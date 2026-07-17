// Team select: pick your team from the meta pool — or a saved custom team
// (M14) — then the opponent's (specific pool team or random-from-pool; the
// bot's own team stays pool-only). Also hosts the custom-team import flow
// (paste PS export -> parse -> canonicalize -> save to localStorage) and
// the device benchmark — the M9 gate ("skuct:3000 within 2-3 s/move") is
// certified per device by tapping it.
//
// Open team sheet (M12): the bot's sets are readable right here in the
// team list, and the bot receives the human's exact sets — a single
// information policy for pool and custom teams alike. Only picks stay
// hidden.

import { useEffect, useRef, useState } from "preact/hooks";
import type { MetaPool, PoolTeam } from "./types";
import { BUDGET, type SelectedTeam } from "./app";
import { BotWorker } from "./bot";
import { getValidator } from "./engine";
import { parsePsExport } from "./ps-import";
import { findingAnchor, findingText, type Finding } from "./findings";
import {
  deleteCustomTeam,
  loadCustomTeams,
  saveCustomTeam,
  type CustomTeam,
} from "./custom-teams";
import { speciesName, ui, type Locale } from "./i18n";

function provenanceLine(t: PoolTeam): string {
  const p = t.provenance;
  const bits: string[] = [];
  if (p.player) bits.push(p.player);
  if (p.placement) bits.push(p.placement);
  if (p.event) bits.push(p.event);
  return bits.join(" · ") || p.source || "";
}

function TeamCard(props: {
  team: PoolTeam;
  index: number;
  selected: boolean;
  onTap: () => void;
}) {
  const { team, index, selected } = props;
  return (
    <button
      class={`team-card ${selected ? "selected" : ""}`}
      onClick={props.onTap}
      data-team={index}
    >
      <div class="team-card-head">
        <span class="team-rank">#{index + 1}</span>
        <span class="team-id">{team.id}</span>
        <span class="team-tier">{team.tier}</span>
      </div>
      <div class="team-prov">{provenanceLine(team)}</div>
      <div class="team-species">
        {team.species.map((sp, i) => (
          <span class="species-chip" key={i}>
            {speciesName(sp)} <small>L{team.levels[i]}</small>
          </span>
        ))}
      </div>
    </button>
  );
}

// Fixed workload: identical battle + searcher seeds + iteration count on
// every device, so the numbers are directly comparable. 10k iterations at
// an in-battle root ~= 4-7 s on 2025 hardware. Reported: iters/s (the
// cross-device number), s/move at the fixed 30k strength (what you wait,
// mostly hidden by ponder), and the historical 3k reference gate (the M9
// envelope — kept so old and new device numbers stay comparable).
const BENCH_ITERS = 10000;
const BENCH_BATTLE_SEED = "1,2,3,4";
const BENCH_SEARCH_SEED = 123456789;
const GATE_ITERS = 3000; // the M9 gate strength (skuct:3000 == mcts:3000)
const GATE_SECONDS = 3;

type BenchState =
  | { s: "idle" }
  | { s: "running"; done: number; total: number }
  | { s: "done"; itersPerSec: number; secPerMove: number; secAtFull: number }
  | { s: "error"; msg: string };

function DeviceBench(props: { pool: MetaPool }) {
  const [state, setState] = useState<BenchState>({ s: "idle" });
  const aliveRef = useRef(true);
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  async function run() {
    if (state.s === "running") return;
    setState({ s: "running", done: 0, total: BENCH_ITERS });
    const bot = new BotWorker();
    try {
      const r = await bot.bench(
        JSON.stringify(props.pool.teams[0].sets),
        JSON.stringify(props.pool.teams[1].sets),
        BENCH_BATTLE_SEED,
        BENCH_SEARCH_SEED,
        BENCH_ITERS,
        (done, total) => {
          if (aliveRef.current) setState({ s: "running", done, total });
        },
      );
      const itersPerSec = r.iters / (r.ms / 1000);
      if (aliveRef.current)
        setState({
          s: "done",
          itersPerSec,
          secPerMove: GATE_ITERS / itersPerSec,
          secAtFull: BUDGET / itersPerSec,
        });
    } catch (e) {
      if (aliveRef.current) setState({ s: "error", msg: String(e) });
    } finally {
      bot.terminate();
    }
  }

  return (
    <div class="bench-box">
      <div class="bench-head">
        <span class="bench-title">{ui().benchTitle}</span>
        <button
          class="ghost bench-btn"
          disabled={state.s === "running"}
          onClick={() => void run()}
        >
          {state.s === "running"
            ? ui().benchRunning(
                Math.round((state.done / state.total) * 100),
              )
            : state.s === "idle"
              ? ui().benchRun
              : ui().benchAgain}
        </button>
      </div>
      {state.s === "done" && (
        <div
          class={`bench-result ${state.secPerMove <= GATE_SECONDS ? "pass" : "fail"}`}
          data-iters-per-sec={Math.round(state.itersPerSec)}
          data-sec-per-move={state.secPerMove.toFixed(2)}
          data-sec-at-full={state.secAtFull.toFixed(2)}
        >
          {ui().benchResult({
            ips: Math.round(state.itersPerSec),
            fullK: BUDGET / 1000,
            fullSec: state.secAtFull.toFixed(2),
            gateK: GATE_ITERS / 1000,
            gateSec: GATE_SECONDS,
            pass: state.secPerMove <= GATE_SECONDS,
            sec: state.secPerMove.toFixed(2),
          })}
        </div>
      )}
      {state.s === "error" && <div class="bench-result fail">{state.msg}</div>}
      <div class="bench-note">{ui().benchNote(BENCH_ITERS / 1000)}</div>
    </div>
  );
}

// ------------------------------------------------- custom teams (M14)

function FindingRows(props: { findings: Finding[]; kind: "error" | "fix" }) {
  return (
    <ul class={`finding-list ${props.kind}`}>
      {props.findings.map((f, i) => {
        const anchor = findingAnchor(f);
        return (
          <li class="finding-row" key={i}>
            {anchor && <span class="finding-anchor">{anchor}</span>}
            <span class="finding-text">{findingText(f)}</span>
          </li>
        );
      })}
    </ul>
  );
}

type ImportResult =
  | { ok: true; savedName: string; applied: Finding[] }
  | { ok: false; errors: Finding[]; applied: Finding[] };

/** Paste -> parse (PS export) -> canonicalize ("fix it for me") -> save.
 * Applied fixes are informational; remaining errors are localized and
 * anchored to the mon (validator) or paste line (parser). */
function CustomImport(props: {
  onSaved: (t: CustomTeam) => void;
  onClose: () => void;
}) {
  const [text, setText] = useState("");
  const [name, setName] = useState("");
  const [result, setResult] = useState<ImportResult | null>(null);

  function doImport() {
    const parsed = parsePsExport(text);
    if (parsed.findings.length > 0) {
      setResult({ ok: false, errors: parsed.findings as Finding[], applied: [] });
      return;
    }
    const res = JSON.parse(
      getValidator().canonicalizeTeam(JSON.stringify(parsed.sets)),
    ) as {
      ok: boolean;
      team: unknown[];
      applied: Finding[];
      errors: Finding[];
    };
    if (!res.ok) {
      setResult({ ok: false, errors: res.errors, applied: res.applied });
      return;
    }
    const saved = saveCustomTeam(name || parsed.teamName || "", res.team);
    setResult({ ok: true, savedName: saved.name, applied: res.applied });
    props.onSaved(saved);
  }

  return (
    <div class="import-panel">
      <div class="import-head">
        <h3>{ui().importTitle}</h3>
        <button class="ghost" onClick={props.onClose}>
          {ui().importCancel}
        </button>
      </div>
      <p class="import-help">{ui().importHelp}</p>
      <textarea
        class="import-text"
        placeholder={ui().importPlaceholder}
        value={text}
        onInput={(e) => setText((e.target as HTMLTextAreaElement).value)}
        rows={10}
        spellcheck={false}
        autocorrect="off"
        autocapitalize="off"
      />
      <div class="import-row">
        <label class="import-name-label">
          {ui().importNameLabel}
          <input
            class="import-name"
            type="text"
            placeholder={ui().importNamePlaceholder}
            value={name}
            onInput={(e) => setName((e.target as HTMLInputElement).value)}
          />
        </label>
        <button
          class="primary import-btn"
          disabled={text.trim() === ""}
          onClick={doImport}
        >
          {ui().importButton}
        </button>
      </div>
      {result && (
        <div class={`import-result ${result.ok ? "ok" : "bad"}`}>
          {result.ok ? (
            <div class="import-ok-note">{ui().importedOk(result.savedName)}</div>
          ) : (
            <div class="import-err-note">
              {ui().importErrors(result.errors.length)}
            </div>
          )}
          {!result.ok && <FindingRows findings={result.errors} kind="error" />}
          {result.applied.length > 0 && (
            <details class="applied-fixes" open={result.ok}>
              <summary>{ui().appliedFixes(result.applied.length)}</summary>
              <FindingRows findings={result.applied} kind="fix" />
            </details>
          )}
        </div>
      )}
    </div>
  );
}

function CustomTeamCard(props: {
  team: CustomTeam;
  selected: boolean;
  onTap: () => void;
  onDelete: () => void;
}) {
  const { team, selected } = props;
  const [confirming, setConfirming] = useState(false);
  return (
    <div class={`team-card custom-card ${selected ? "selected" : ""}`}>
      <button class="custom-card-body" onClick={props.onTap} data-custom={team.id}>
        <div class="team-card-head">
          <span class="team-id">{team.name}</span>
          <span class="team-tier custom-tier">{ui().customBadge}</span>
        </div>
        <div class="team-species">
          {team.species.map((sp, i) => (
            <span class="species-chip" key={i}>
              {speciesName(sp)} <small>L{team.levels[i]}</small>
            </span>
          ))}
        </div>
      </button>
      <button
        class={`ghost delete-btn ${confirming ? "confirming" : ""}`}
        onClick={() => {
          if (confirming) props.onDelete();
          else {
            setConfirming(true);
            setTimeout(() => setConfirming(false), 3000);
          }
        }}
      >
        {confirming ? ui().deleteConfirm : ui().deleteTeam}
      </button>
    </div>
  );
}

// --------------------------------------------------------- select screen

type HumanPick = { kind: "pool"; idx: number } | { kind: "custom"; id: string };

export function TeamSelect(props: {
  pool: MetaPool;
  locale: Locale;
  onLocale: (l: Locale) => void;
  onStart: (human: SelectedTeam, botIdx: number | "random") => void;
}) {
  const [humanPick, setHumanPick] = useState<HumanPick | null>(null);
  const [botIdx, setBotIdx] = useState<number | "random">("random");
  const [step, setStep] = useState<0 | 1>(0);
  const [customs, setCustoms] = useState<CustomTeam[]>(loadCustomTeams);
  const [importing, setImporting] = useState(false);
  const teams = props.pool.teams;

  const pickedCustom =
    humanPick?.kind === "custom"
      ? customs.find((t) => t.id === humanPick.id) ?? null
      : null;
  const humanLabel =
    humanPick === null
      ? "—"
      : humanPick.kind === "pool"
        ? teams[humanPick.idx].id
        : (pickedCustom?.name ?? "—");
  const humanTeam: SelectedTeam | null =
    humanPick === null
      ? null
      : humanPick.kind === "pool"
        ? {
            id: teams[humanPick.idx].id,
            sets: teams[humanPick.idx].sets,
            poolIdx: humanPick.idx,
          }
        : pickedCustom && {
            id: pickedCustom.name,
            sets: pickedCustom.sets,
            poolIdx: null,
          };

  return (
    <div class="screen select-screen">
      <header class="app-header">
        <h1>NC2000</h1>
        <span class="subtitle">{ui().subtitle}</span>
        <span class="locale-toggle">
          {(["en", "ja"] as const).map((l) => (
            <button
              key={l}
              class={`locale-btn ${props.locale === l ? "on" : ""}`}
              onClick={() => props.onLocale(l)}
            >
              {l === "en" ? "EN" : "日本語"}
            </button>
          ))}
        </span>
      </header>
      <div class="botinfo-note">{ui().openSheetNote}</div>

      <div class="select-bar">
        {/* slots are tappable: switch which side the list below picks
            (also the way back to your own team / custom management) */}
        <button
          class={`select-slot ${step === 0 ? "current" : ""}`}
          onClick={() => setStep(0)}
        >
          <span class="slot-label">{ui().you}</span>
          <span class="slot-value">{humanLabel}</span>
        </button>
        <span class="vs">vs</span>
        <button
          class={`select-slot ${step === 1 ? "current" : ""}`}
          onClick={() => setStep(1)}
        >
          <span class="slot-label">{ui().bot}</span>
          <span class="slot-value">
            {botIdx === "random" ? ui().randomFromPool : teams[botIdx].id}
          </span>
        </button>
        <button
          class="primary start-btn"
          disabled={!humanTeam}
          onClick={() => props.onStart(humanTeam!, botIdx)}
        >
          {ui().startBattle}
        </button>
      </div>

      <h2>{step === 0 ? ui().chooseYours : ui().chooseOpp}</h2>

      {step === 0 && (
        <section class="custom-section">
          <h3>{ui().customSection}</h3>
          <div class="team-list">
            {customs.map((t) => (
              <CustomTeamCard
                key={t.id}
                team={t}
                selected={humanPick?.kind === "custom" && humanPick.id === t.id}
                onTap={() => {
                  setHumanPick({ kind: "custom", id: t.id });
                  setStep(1);
                  window.scrollTo({ top: 0 });
                }}
                onDelete={() => {
                  setCustoms(deleteCustomTeam(t.id));
                  if (humanPick?.kind === "custom" && humanPick.id === t.id) {
                    setHumanPick(null);
                  }
                }}
              />
            ))}
            {!importing && (
              <button
                class="team-card add-custom-card"
                onClick={() => setImporting(true)}
              >
                {ui().addCustom}
              </button>
            )}
          </div>
          {importing && (
            <CustomImport
              onSaved={(t) => {
                setCustoms(loadCustomTeams());
                setHumanPick({ kind: "custom", id: t.id });
              }}
              onClose={() => setImporting(false)}
            />
          )}
        </section>
      )}

      {step === 1 && (
        <button
          class={`team-card random-card ${botIdx === "random" ? "selected" : ""}`}
          onClick={() => setBotIdx("random")}
        >
          {ui().randomCard(teams.length)}
        </button>
      )}
      <div class="team-list">
        {teams.map((t, i) => (
          <TeamCard
            key={t.id}
            team={t}
            index={i}
            selected={
              step === 0
                ? humanPick?.kind === "pool" && humanPick.idx === i
                : botIdx === i
            }
            onTap={() => {
              if (step === 0) {
                setHumanPick({ kind: "pool", idx: i });
                setStep(1);
                window.scrollTo({ top: 0 });
              } else {
                setBotIdx(i);
              }
            }}
          />
        ))}
      </div>

      <DeviceBench pool={props.pool} />
    </div>
  );
}
