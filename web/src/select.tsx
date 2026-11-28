// Start screen: a minimal centered column — Start battle, a one-line
// banner naming the rules in force, Your party, and the setup button. Your
// party defaults to Random, so one tap on Start begins a game; the party
// button opens a modal with the catalog (the built-in parties, the same ids
// and exact sets the bot draws from) and the shared M14 custom-team
// import/pick. A pinned choice persists in localStorage by team id. The
// language selector is an unobtrusive corner dropdown.
//
// Every game is blind (info-mode.ts): neither side sees the other's sets,
// and the opponent is drawn at start and redrawn on rematch — there is
// nothing to choose, and the banner says so, so there is no opponent row.
// Both Random draws come from app.tsx, which owns the rule: the catalog's
// draw weights, or uniform over a pool file the user loaded.
//
// Setup is a single modal, not two buttons: the team pool and the belief
// prior are the same question asked twice — which teams the bot may be
// facing, and what it assumes about a team it cannot identify. Inside, each
// half keeps its own heading, controls and report box: someone who just
// loaded a pool file must not read the prior's verdict as a verdict on
// their file.

import { useEffect, useRef, useState } from "preact/hooks";
import type { MetaPool, PickPriorReport, PoolTeam, PriorReport } from "./types";
import type { SelectedTeam } from "./app";
import {
  clearStoredPool,
  parsePoolText,
  storePool,
  type LoadedPool,
} from "./team-pool";
import {
  clearStoredPickPrior,
  clearStoredPrior,
  storePickPrior,
  storePrior,
  type StoredPrior,
} from "./belief-prior";
import { Modal } from "./modal";
import { getValidator, probePickPriorTable, probePrior } from "./engine";
import { parsePsExport } from "./ps-import";
import { findingAnchor, findingText, type Finding } from "./findings";
import {
  deleteCustomTeam,
  loadCustomTeams,
  saveCustomTeam,
  type CustomTeam,
} from "./custom-teams";
import { itemName, moveName, speciesName, ui, type Locale } from "./i18n";
import { Lvl } from "./battle-ui";
import { teamDisplayName } from "./team-names";

interface MonSetLike {
  species?: string;
  level?: number;
  item?: string;
  moves?: string[];
}

function TeamCard(props: {
  team: PoolTeam;
  index: number;
  selected: boolean;
  onTap: () => void;
}) {
  const { team, index, selected } = props;
  const name = teamDisplayName(team.id);
  const sets = Array.isArray(team.sets) ? (team.sets as MonSetLike[]) : [];

  return (
    <div
      class={`team-card ${selected ? "selected" : ""}`}
      data-team={index}
    >
      <button
        class="team-card-select"
        onClick={props.onTap}
        aria-pressed={selected}
      >
        <div class="team-card-head">
          <span class="team-rank">#{index + 1}</span>
          {name ? (
            <>
              <span class="team-name">{name}</span>
              <span class="team-id">{team.id}</span>
            </>
          ) : (
            <span class="team-id">{team.id}</span>
          )}
        </div>
        <div class="team-species">
          {team.species.map((sp, i) => (
            <span class="species-chip" key={i}>
              {speciesName(sp)}{" "}
              <small>
                <Lvl n={team.levels[i]} />
              </small>
            </span>
          ))}
        </div>
      </button>
      {sets.length > 0 && (
        <details class="team-sets-details">
          <summary class="team-sets-summary">{ui().viewSets}</summary>
          <div class="team-sets-content">
            {sets.map((s, i) => (
              <div class="team-set-row" key={i}>
                <div class="team-set-header">
                  <span class="team-set-mon">
                    <strong>{speciesName(s.species ?? team.species[i])}</strong>{" "}
                    <small>
                      <Lvl n={s.level ?? team.levels[i]} />
                    </small>
                  </span>
                  {s.item && (
                    <span class="team-set-item">@{itemName(s.item)}</span>
                  )}
                </div>
                {Array.isArray(s.moves) && s.moves.length > 0 && (
                  <div class="team-set-moves">
                    {s.moves.map((mv) => moveName(mv)).join(" · ")}
                  </div>
                )}
              </div>
            ))}
          </div>
        </details>
      )}
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
        aria-label={ui().importTitle}
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
      <button
        class="custom-card-body"
        onClick={props.onTap}
        aria-pressed={selected}
        data-custom={team.id}
      >
        <div class="team-card-head">
          <span class="team-id">{team.name}</span>
          <span class="team-tier custom-tier">{ui().customBadge}</span>
        </div>
        <div class="team-species">
          {team.species.map((sp, i) => (
            <span class="species-chip" key={i}>
              {speciesName(sp)}{" "}
              <small>
                <Lvl n={team.levels[i]} />
              </small>
            </span>
          ))}
        </div>
      </button>
      <button
        class={`ghost delete-btn ${confirming ? "confirming" : ""}`}
        aria-label={`${ui().srDeleteFor(team.name)}${confirming ? ` — ${ui().deleteConfirm}` : ""}`}
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

// ------------------------------------------------- pinned party choices

type PartyChoice =
  | { kind: "random" }
  | { kind: "pool"; id: string }
  | { kind: "custom"; id: string };
interface Picks {
  human: PartyChoice;
}

const PICKS_KEY = "nc2000-start-picks";
const RANDOM = { kind: "random" } as const;

/** Load the pinned party choices; anything stale (pool id gone after a
 * pool update, custom deleted elsewhere) falls back to random. */
function loadPicks(pool: MetaPool, customs: CustomTeam[]): Picks {
  const picks: Picks = { human: RANDOM };
  try {
    const raw = localStorage.getItem(PICKS_KEY);
    if (!raw) return picks;
    const p = JSON.parse(raw) as Partial<Picks>;
    const h = p.human;
    if (
      (h?.kind === "pool" && pool.teams.some((t) => t.id === h.id)) ||
      (h?.kind === "custom" && customs.some((t) => t.id === h.id))
    ) {
      picks.human = h;
    }
  } catch {
    /* storage unavailable / corrupt: defaults stand */
  }
  return picks;
}

function storePicks(picks: Picks): void {
  try {
    localStorage.setItem(PICKS_KEY, JSON.stringify(picks));
  } catch {
    /* storage unavailable: the choice still holds this session */
  }
}

function CustomTeamSection(props: {
  choice: PartyChoice;
  onPick: (c: PartyChoice) => void;
  customs: CustomTeam[];
  onCustomsChange: (list: CustomTeam[], picked?: CustomTeam) => void;
}) {
  const [importing, setImporting] = useState(false);
  return (
    <section class="custom-section">
      <h3>{ui().customSection}</h3>
      <div class="team-list">
        {props.customs.map((t) => (
          <CustomTeamCard
            key={t.id}
            team={t}
            selected={props.choice.kind === "custom" && props.choice.id === t.id}
            onTap={() => props.onPick({ kind: "custom", id: t.id })}
            onDelete={() => props.onCustomsChange(deleteCustomTeam(t.id))}
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
            const stored = loadCustomTeams();
            // localStorage may be unavailable/full. Keep a successfully
            // canonicalized import playable for this session regardless.
            const list = stored.some((x) => x.id === t.id)
              ? stored
              : [...props.customs, t];
            props.onCustomsChange(list, t);
          }}
          onClose={() => setImporting(false)}
        />
      )}
    </section>
  );
}

// ------------------------------------------------- party picker modals

/** Human party picker (modal body): random card, saved customs + import
 * flow, then the pool list. Picking closes the modal (onPick); managing
 * customs (import/delete) keeps it open. */
function HumanPicker(props: {
  teams: PoolTeam[];
  drawCount: number;
  choice: PartyChoice;
  onPick: (c: PartyChoice) => void;
  customs: CustomTeam[];
  onCustomsChange: (list: CustomTeam[], picked?: CustomTeam) => void;
}) {
  const { teams, choice, customs } = props;

  return (
    <>
      <p class="modal-note">{ui().blindSheetNote}</p>
      <button
        class={`team-card random-card ${choice.kind === "random" ? "selected" : ""}`}
        aria-pressed={choice.kind === "random"}
        onClick={() => props.onPick(RANDOM)}
      >
        {ui().randomCard(props.drawCount)}
      </button>
      <CustomTeamSection
        choice={choice}
        onPick={props.onPick}
        customs={customs}
        onCustomsChange={props.onCustomsChange}
      />
      <h3>{ui().poolSection}</h3>
      <div class="team-list">
        {teams.map((t, i) => (
          <TeamCard
            key={t.id}
            team={t}
            index={i}
            selected={choice.kind === "pool" && choice.id === t.id}
            onTap={() => props.onPick({ kind: "pool", id: t.id })}
          />
        ))}
      </div>
    </>
  );
}

// ------------------------------------------------- belief prior (M18)

/** The one table shipped with the app, served like every other data file
 * (repo `data/`, mapped under the deploy base — see vite.config.ts). It is
 * fetched only when the user presses the button: nothing loads a prior on
 * its own, by policy (docs/community-belief-prior-design.md). */
const SAMPLE_PRIOR = "belief-prior-v0.sample.json";

/** Interpret a table without installing it. The wasm interpreter is total —
 * a malformed file degrades into `warnings` — so a throw here means the
 * boundary itself failed, which is reported as a load failure rather than
 * as a verdict about the table. */
function probeQuiet<R>(
  probeFn: (json: string) => R,
  json: string,
): { report: R | null; error: string | null } {
  try {
    return { report: probeFn(json), error: null };
  } catch (e) {
    return { report: null, error: String(e) };
  }
}

interface PriorReportShape {
  applied: boolean;
  warnings: string[];
}

function PriorFilePanel<R extends PriorReportShape>(props: {
  prior: StoredPrior | null;
  onPrior: (p: StoredPrior | null) => void;
  probe: (json: string) => R;
  store: (name: string, json: string) => string | null;
  clear: () => void;
  help: string;
  pickLabel: string;
  sampleLabel?: string;
  onLoadSample?: (adopt: (name: string, json: string) => void) => Promise<void>;
  summarize: (report: R) => string;
  fileTestId: string;
  sampleTestId?: string;
  clearTestId: string;
  reportTestId: string;
}) {
  // Mounting IS "the modal opened" (the parent mounts it on open), so a
  // table kept from an earlier session gets its report back before the user
  // touches anything.
  const [report, setReport] = useState<R | null>(() =>
    props.prior ? probeQuiet(props.probe, props.prior.json).report : null,
  );
  const [failure, setFailure] = useState<string | null>(null);

  function adopt(name: string, json: string) {
    const probed = probeQuiet(props.probe, json);
    setReport(probed.report);
    if (probed.error !== null) {
      setFailure(ui().priorLoadFailed(probed.error));
      return;
    }
    const why = props.store(name, json);
    if (why !== null) {
      setFailure(ui().priorLoadFailed(why));
      return;
    }
    setFailure(null);
    props.onPrior({ name, json });
  }

  return (
    <div class="prior-panel">
      <p class="modal-note">{props.help}</p>
      <div class="prior-actions">
        {/* A real <label> wrapping the input: the native file button has no
         * accessible name of its own, and the visible text must be the one
         * a screen reader announces. */}
        <label class="prior-pick">
          <span class="prior-pick-label">{props.pickLabel}</span>
          <input
            type="file"
            accept=".json,application/json"
            data-testid={props.fileTestId}
            onChange={(e) => {
              const input = e.currentTarget as HTMLInputElement;
              const file = input.files?.[0];
              // Clearing the value lets the same file be re-picked (the
              // change event would not fire again otherwise).
              input.value = "";
              if (!file) return;
              void file
                .text()
                .then((text) => adopt(file.name, text))
                .catch((err: unknown) => {
                  setReport(null);
                  setFailure(ui().priorLoadFailed(String(err)));
                });
            }}
          />
        </label>
        {props.onLoadSample && props.sampleLabel && (
          <button
            data-testid={props.sampleTestId}
            onClick={() => {
              void props.onLoadSample!(adopt).catch((e: unknown) => {
                setReport(null);
                setFailure(ui().priorLoadFailed(String(e)));
              });
            }}
          >
            {props.sampleLabel}
          </button>
        )}
        <button
          class="ghost"
          data-testid={props.clearTestId}
          disabled={props.prior === null}
          onClick={() => {
            props.clear();
            props.onPrior(null);
            setReport(null);
            setFailure(null);
          }}
        >
          {ui().priorClear}
        </button>
      </div>
      {failure !== null && <p class="prior-failure">{failure}</p>}
      {report && (
        // Reading order is the order of the claims: does it apply, what does
        // it contain, what did the interpreter object to.
        <div class="prior-report" data-testid={props.reportTestId}>
          {/* The verdict is the one line that must not be missed, so it
              carries colour as well as words (never colour alone). */}
          <div class={`prior-verdict ${report.applied ? "ok" : "no"}`}>
            {report.applied ? ui().priorApplied : ui().priorNotApplied}
          </div>
          <div class="prior-summary">{props.summarize(report)}</div>
          {report.warnings.length > 0 && (
            <>
              <h3>{ui().priorWarnings}</h3>
              <ul class="prior-warnings">
                {report.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/** Belief-prior panel: pick a table file or load the shipped sample. */
function PriorPanel(props: {
  prior: StoredPrior | null;
  onPrior: (p: StoredPrior | null) => void;
}) {
  return (
    <PriorFilePanel<PriorReport>
      prior={props.prior}
      onPrior={props.onPrior}
      probe={probePrior}
      store={storePrior}
      clear={clearStoredPrior}
      help={ui().priorHelp}
      pickLabel={ui().priorPick}
      sampleLabel={ui().priorSample}
      sampleTestId="prior-sample"
      fileTestId="prior-file"
      clearTestId="prior-clear"
      reportTestId="prior-report"
      onLoadSample={async (adopt) => {
        const res = await fetch(
          `${import.meta.env.BASE_URL}data/${SAMPLE_PRIOR}`,
        );
        if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
        adopt(SAMPLE_PRIOR, await res.text());
      }}
      summarize={(report) =>
        ui().priorSummary(report.species, report.meanMoveSum, report.skipped)
      }
    />
  );
}

/** Team Preview pick-prior panel: validate and store a `nc2000-pick-prior-v1` file. */
function PickPriorPanel(props: {
  prior: StoredPrior | null;
  onPrior: (p: StoredPrior | null) => void;
}) {
  return (
    <PriorFilePanel<PickPriorReport>
      prior={props.prior}
      onPrior={props.onPrior}
      probe={probePickPriorTable}
      store={storePickPrior}
      clear={clearStoredPickPrior}
      help={ui().pickPriorHelp}
      pickLabel={ui().pickPriorPick}
      fileTestId="pick-prior-file"
      clearTestId="pick-prior-clear"
      reportTestId="pick-prior-report"
      summarize={(report) =>
        ui().pickPriorSummary(
          report.rows,
          report.teams,
          report.smoothing,
          report.backoffStrength,
        )
      }
    />
  );
}

// ------------------------------------------------- team pool (swappable)

/** What the last attempt in the pool panel did. One value, because the
 * cases are exclusive: either a file is in play (possibly unsaved), or it
 * was refused and nothing moved. */
type PoolResult =
  | { ok: true; teams: number; notStored: string | null }
  | { ok: false; errors: string[] };

/** Team-pool panel (the first half of the Blind setup modal): load a pool
 * file, or go back to the catalog.
 *
 * Adoption is all-or-nothing — parsePoolText validates every team before
 * anything is installed, so a refused file leaves the running pool exactly
 * where it was and the report says so in as many words. Persistence is the
 * one step allowed to fail on its own: unlike a belief prior, an accepted
 * pool is already proven playable, so a browser that cannot store it still
 * gets to play it this session and is told it will not survive a reload. */
function PoolPanel(props: {
  loaded: LoadedPool;
  catalog: LoadedPool;
  onPool: (p: LoadedPool) => void;
}) {
  const [result, setResult] = useState<PoolResult | null>(null);

  function adopt(name: string, text: string) {
    const parsed = parsePoolText(text);
    if (!parsed.ok) {
      setResult({ ok: false, errors: parsed.errors });
      return;
    }
    // What gets stored is the normalized text, not the file's own bytes:
    // it is what the searcher is handed, so it is what has to come back
    // unchanged on the next visit.
    const notStored = storePool(name, parsed.poolJson);
    setResult({ ok: true, teams: parsed.teams, notStored });
    props.onPool({ name, pool: parsed.pool, poolJson: parsed.poolJson });
  }

  return (
    <div class="pool-panel">
      <p class="modal-note">{ui().poolHelp}</p>
      <div class="pool-actions">
        {/* Same construction as the prior panel's picker: a real <label>
         * around the input, because the native file button has no
         * accessible name of its own. */}
        <label class="pool-pick">
          <span>{ui().poolPick}</span>
          <input
            type="file"
            accept=".json,application/json"
            data-testid="pool-file"
            onChange={(e) => {
              const input = e.currentTarget as HTMLInputElement;
              const file = input.files?.[0];
              // Clearing the value lets the same file be re-picked after a
              // hand edit (no change event otherwise).
              input.value = "";
              if (!file) return;
              void file
                .text()
                .then((text) => adopt(file.name, text))
                // A file the browser could not read at all: reported in the
                // rejection slot, since that is what happened to the pool.
                .catch((err: unknown) =>
                  setResult({ ok: false, errors: [String(err)] }),
                );
            }}
          />
        </label>
        <button
          class="ghost"
          data-testid="pool-reset"
          disabled={props.loaded.name === null}
          onClick={() => {
            clearStoredPool();
            props.onPool(props.catalog);
            setResult(null);
          }}
        >
          {ui().poolReset}
        </button>
      </div>
      {result && (
        <div class="pool-report" data-testid="pool-report">
          <div class={`pool-verdict ${result.ok ? "ok" : "no"}`}>
            {result.ok ? ui().poolAccepted(result.teams) : ui().poolRejected}
          </div>
          {result.ok
            ? result.notStored !== null && (
                <p class="pool-not-stored">
                  {ui().poolNotStored(result.notStored)}
                </p>
              )
            : result.errors.length > 0 && (
                <ul class="pool-errors">
                  {result.errors.map((line, i) => (
                    <li key={i}>{line}</li>
                  ))}
                </ul>
              )}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------- blind setup (one modal)

/** Body of the Blind setup modal: the pool panel and the prior panel, in
 * that order (the pool decides what the bot can be; the prior only matters
 * for a team no pool explains).
 *
 * They stay two <section>s with two headings rather than one flowing panel
 * because each ends in a verdict box, and the two verdicts are about
 * different files. Loading a pool and then glancing down at "NOT applied"
 * — the prior's line about a table that was never loaded — would read as
 * the pool having been refused. The separator rule and the headings are
 * what keep that misreading off the screen. */
function SetupPanel(props: {
  loaded: LoadedPool;
  catalog: LoadedPool;
  onPool: (p: LoadedPool) => void;
  prior: StoredPrior | null;
  onPrior: (p: StoredPrior | null) => void;
  pickPrior: StoredPrior | null;
  onPickPrior: (p: StoredPrior | null) => void;
}) {
  return (
    <>
      <section class="setup-section">
        <h3>{ui().poolLabel}</h3>
        <PoolPanel
          loaded={props.loaded}
          catalog={props.catalog}
          onPool={props.onPool}
        />
      </section>
      <section class="setup-section">
        <h3>{ui().priorLabel}</h3>
        <PriorPanel prior={props.prior} onPrior={props.onPrior} />
      </section>
      <section class="setup-section">
        <h3>{ui().pickPriorLabel}</h3>
        <PickPriorPanel
          prior={props.pickPrior}
          onPrior={props.onPickPrior}
        />
      </section>
    </>
  );
}

// ---------------------------------------------------------- start screen

export function StartScreen(props: {
  /** The lists in play: the catalog, or a pool file the user loaded. */
  loadedPool: LoadedPool;
  /** The catalog, held by app.tsx so the reset button has something to
   * install without going back to the network. */
  catalogPool: LoadedPool;
  onPool: (p: LoadedPool) => void;
  locale: Locale;
  onLocale: (l: Locale) => void;
  /** The bot's draw, owned by app.tsx so that pressing Start and taking a
   * rematch roll by the same rule. */
  drawOpponent: () => SelectedTeam;
  /** The human's "Random", drawn by the same rule from the same lists. */
  drawHuman: () => SelectedTeam;
  /** How many teams the lists hold. */
  botDrawCount: number;
  prior: StoredPrior | null;
  onPrior: (p: StoredPrior | null) => void;
  pickPrior: StoredPrior | null;
  onPickPrior: (p: StoredPrior | null) => void;
  onStart: (human: SelectedTeam, bot: SelectedTeam) => void;
}) {
  const pool = props.loadedPool.pool;
  const teams = pool.teams;
  const [customs, setCustoms] = useState<CustomTeam[]>(loadCustomTeams);
  const [picks, setPicks] = useState<Picks>(() =>
    loadPicks(pool, loadCustomTeams()),
  );
  const [modal, setModal] = useState<null | "human" | "settings">(null);

  // A pool swap invalidates the pinned pick: an id from the old lists is
  // either absent from the new ones — the button would name a team that no
  // longer exists, and start() would quietly draw a random one instead — or
  // it names a different team. Re-running loadPicks is the same
  // reconciliation the first mount does, so a pin survives exactly when the
  // new lists have that id. (Remounting the whole screen would do it too,
  // but it would tear down the modal that is showing the load report.)
  // Identity, not name: two files can share a name and hold different
  // teams, and app.tsx only ever hands over a new object when the pool
  // actually changed.
  const poolRef = useRef(props.loadedPool);
  useEffect(() => {
    if (poolRef.current === props.loadedPool) return;
    poolRef.current = props.loadedPool;
    setPicks(loadPicks(props.loadedPool.pool, customs));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.loadedPool]);

  function update(next: Picks) {
    setPicks(next);
    storePicks(next);
  }

  const humanChoice = picks.human;
  const pickedCustom =
    humanChoice.kind === "custom"
      ? customs.find((t) => t.id === humanChoice.id) ?? null
      : null;

  const humanValue =
    humanChoice.kind === "random"
      ? ui().randomLabel
      : humanChoice.kind === "pool"
        ? humanChoice.id
        : (pickedCustom?.name ?? ui().randomLabel);

  function humanTeam(): SelectedTeam {
    if (humanChoice.kind === "custom" && pickedCustom)
      return { id: pickedCustom.name, sets: pickedCustom.sets };
    const pinned =
      humanChoice.kind === "pool"
        ? teams.find((t) => t.id === humanChoice.id)
        : undefined;
    // Random is resolved here, at start: a fresh roll every game unless
    // the user pinned a team.
    if (!pinned) return props.drawHuman();
    return { id: pinned.id, sets: pinned.sets };
  }

  function customsChanged(list: CustomTeam[], picked?: CustomTeam) {
    setCustoms(list);
    let human = picks.human;
    if (picked) human = { kind: "custom", id: picked.id };
    const customId = human.kind === "custom" ? human.id : null;
    if (customId && !list.some((t) => t.id === customId)) human = RANDOM;
    if (human !== picks.human) update({ human });
  }

  return (
    <div class="start-screen">
      <select
        class="lang-select"
        aria-label={ui().languageLabel}
        value={props.locale}
        onChange={(e) =>
          props.onLocale((e.target as HTMLSelectElement).value as Locale)
        }
      >
        <option value="en">English</option>
        <option value="ja">日本語</option>
      </select>

      <main class="start-col">
        <h1 class="start-title">NC2000</h1>
        <div class="start-subtitle">{ui().subtitle}</div>
        <button
          class="primary start-main-btn"
          onClick={() => props.onStart(humanTeam(), props.drawOpponent())}
        >
          {ui().startBattle}
        </button>
        {/* A readout, not a control. One line, because it has exactly two
            things to say — sets are hidden both ways, and the opponent is
            redrawn each battle — and the second of them is why no opponent
            picker follows. */}
        <p class="mode-banner" data-testid="mode-banner">
          {ui().blindBanner}
        </p>
        <button
          class="party-btn"
          data-party="human"
          onClick={() => setModal("human")}
        >
          <span class="party-label">{ui().yourParty}</span>
          <span class="party-value">{humanValue}</span>
        </button>
        <button
          class="party-btn"
          data-party="settings"
          onClick={() => setModal("settings")}
        >
          <span class="party-label">{ui().settingsLabel}</span>
          <span class="party-value">
            {(() => {
              const poolStr =
                props.loadedPool.name !== null
                  ? ui().poolLoaded(props.loadedPool.name, teams.length)
                  : ui().poolBundled(teams.length);
              const priorStr = props.prior ? props.prior.name : ui().priorNone;
              return props.pickPrior !== null
                ? ui().settingsValueWithPickPrior(poolStr, priorStr, props.pickPrior.name)
                : ui().settingsValue(poolStr, priorStr);
            })()}
          </span>
        </button>
        <a class="start-link" href={`${import.meta.env.BASE_URL}?evaluate`}>
          {ui().evaluateLink}
        </a>
        <a class="start-link" href={`${import.meta.env.BASE_URL}?pdinput`} data-testid="pdinput-link">
          {ui().pdinputLink}
        </a>
      </main>

      {modal === "human" && (
        <Modal title={ui().chooseYours} onClose={() => setModal(null)}>
          <HumanPicker
            teams={teams}
            drawCount={props.botDrawCount}
            choice={picks.human}
            onPick={(c) => {
              update({ human: c });
              setModal(null);
            }}
            customs={customs}
            // A fresh import is pinned; the modal stays open so applied
            // fixes remain visible.
            onCustomsChange={customsChanged}
          />
        </Modal>
      )}
      {modal === "settings" && (
        <Modal title={ui().settingsTitle} onClose={() => setModal(null)}>
          <SetupPanel
            loaded={props.loadedPool}
            catalog={props.catalogPool}
            onPool={props.onPool}
            prior={props.prior}
            onPrior={props.onPrior}
            pickPrior={props.pickPrior}
            onPickPrior={props.onPickPrior}
          />
        </Modal>
      )}
    </div>
  );
}
