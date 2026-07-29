// Start screen (UI-1): a minimal centered column — Start battle / Your
// party / Opponent's party. Both parties default to random-from-pool, so
// one tap on Start begins a game; a party button opens a modal with the
// full selection content (pool team list with rank/provenance/species,
// the shared M14 custom-team import/pick for either side). Pinned choices
// persist in localStorage by team id. The language selector is an
// unobtrusive corner dropdown. (The device benchmark — a dev instrument
// for the M9 think-time gate — was removed from the product UI in UI-2.)
//
// Open team sheet (M12): the bot's sets are readable in the party modal,
// and the bot receives the human's exact sets — a single information
// policy for pool and custom teams alike. Only picks stay hidden.
//
// Information mode (M18): the start screen also owns the open/blind toggle
// and the belief-prior table, both per-browser preferences (app.tsx holds
// the state; a Game freezes the mode at start). Blind changes this screen
// in three places — the opponent is no longer choosable (it is drawn from
// the pool at start and redrawn on rematch, so the human never knows the
// foe's sets), the party-modal note describes the blind policy instead of
// the open one, and the belief-prior button appears (a prior is only
// consulted when the bot cannot identify the opponent, which cannot happen
// in open mode).

import { useState } from "preact/hooks";
import type { MetaPool, PoolTeam, PriorReport } from "./types";
import type { SelectedTeam } from "./app";
import { randomPoolTeam } from "./pool-pick";
import type { InfoMode } from "./info-mode";
import {
  clearStoredPrior,
  storePrior,
  type StoredPrior,
} from "./belief-prior";
import { Modal } from "./modal";
import { getValidator, probePrior } from "./engine";
import { parsePsExport } from "./ps-import";
import { findingAnchor, findingText, type Finding } from "./findings";
import {
  deleteCustomTeam,
  loadCustomTeams,
  saveCustomTeam,
  type CustomTeam,
} from "./custom-teams";
import { speciesName, ui, type Locale } from "./i18n";
import { Lvl } from "./battle-ui";

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
      aria-pressed={selected}
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
            {speciesName(sp)}{" "}
            <small>
              <Lvl n={team.levels[i]} />
            </small>
          </span>
        ))}
      </div>
    </button>
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
  bot: PartyChoice;
}

const PICKS_KEY = "nc2000-start-picks";
const RANDOM = { kind: "random" } as const;

/** Load the pinned party choices; anything stale (pool id gone after a
 * pool update, custom deleted elsewhere) falls back to random. */
function loadPicks(pool: MetaPool, customs: CustomTeam[]): Picks {
  const picks: Picks = { human: RANDOM, bot: RANDOM };
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
    const b = p.bot;
    if (
      (b?.kind === "pool" && pool.teams.some((t) => t.id === b.id)) ||
      (b?.kind === "custom" && customs.some((t) => t.id === b.id))
    ) {
      picks.bot = b;
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
  choice: PartyChoice;
  onPick: (c: PartyChoice) => void;
  customs: CustomTeam[];
  onCustomsChange: (list: CustomTeam[], picked?: CustomTeam) => void;
  mode: InfoMode;
}) {
  const { teams, choice, customs } = props;
  return (
    <>
      {/* The note states what the choice costs in information: in open mode
       * the bot reads the sets you pick here, in blind mode it does not. */}
      <p class="modal-note">
        {props.mode === "blind" ? ui().blindSheetNote : ui().openSheetNote}
      </p>
      <button
        class={`team-card random-card ${choice.kind === "random" ? "selected" : ""}`}
        aria-pressed={choice.kind === "random"}
        onClick={() => props.onPick(RANDOM)}
      >
        {ui().randomCard(teams.length)}
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

/** Opponent party picker: the same saved custom parties are available as
 * for the human side. Import/delete stays shared through localStorage. */
function BotPicker(props: {
  teams: PoolTeam[];
  choice: PartyChoice;
  onPick: (c: PartyChoice) => void;
  customs: CustomTeam[];
  onCustomsChange: (list: CustomTeam[], picked?: CustomTeam) => void;
  mode: InfoMode;
}) {
  const { teams, choice } = props;
  return (
    <>
      {/* Blind never opens this modal (the opponent button is not rendered),
       * but the branch is kept so the note can never contradict the mode. */}
      <p class="modal-note">
        {props.mode === "blind" ? ui().blindSheetNote : ui().openSheetNote}
      </p>
      <button
        class={`team-card random-card ${choice.kind === "random" ? "selected" : ""}`}
        aria-pressed={choice.kind === "random"}
        onClick={() => props.onPick(RANDOM)}
      >
        {ui().randomCard(teams.length)}
      </button>
      <CustomTeamSection
        choice={choice}
        onPick={props.onPick}
        customs={props.customs}
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
 * its own, by policy (crates/bot/src/prior.rs:491). */
const SAMPLE_PRIOR = "belief-prior-v0.sample.json";

/** Interpret a table without installing it. The wasm interpreter is total —
 * a malformed file degrades into `warnings` — so a throw here means the
 * boundary itself failed, which is reported as a load failure rather than
 * as a verdict about the table. */
function probeQuiet(json: string): {
  report: PriorReport | null;
  error: string | null;
} {
  try {
    return { report: probePrior(json), error: null };
  } catch (e) {
    return { report: null, error: String(e) };
  }
}

/** Belief-prior panel (modal body): pick a table file or load the shipped
 * sample, read back what the engine makes of it, or clear it. Both sources
 * run the same adopt() path — probe first (so the user sees the verdict
 * even when the table is useless), then persist, and only a table the
 * browser can keep is adopted: one that cannot be stored would silently
 * vanish on reload, which is worse than refusing it now. */
function PriorPanel(props: {
  prior: StoredPrior | null;
  onPrior: (p: StoredPrior | null) => void;
}) {
  // Mounting IS "the modal opened" (the parent mounts it on open), so a
  // table kept from an earlier session gets its report back before the user
  // touches anything.
  const [report, setReport] = useState<PriorReport | null>(() =>
    props.prior ? probeQuiet(props.prior.json).report : null,
  );
  const [failure, setFailure] = useState<string | null>(null);

  function adopt(name: string, json: string) {
    const probed = probeQuiet(json);
    setReport(probed.report);
    if (probed.error !== null) {
      setFailure(ui().priorLoadFailed(probed.error));
      return;
    }
    // OPEN QUESTION: a table the engine reports as NOT applied (unparseable
    // or empty) is still adopted here — the contract gates adoption on the
    // storage result only, and the report says plainly that it will not
    // apply. Refusing it outright may read better; owner call.
    const why = storePrior(name, json);
    if (why !== null) {
      setFailure(ui().priorLoadFailed(why));
      return;
    }
    setFailure(null);
    props.onPrior({ name, json });
  }

  async function loadSample() {
    try {
      const res = await fetch(
        `${import.meta.env.BASE_URL}data/${SAMPLE_PRIOR}`,
      );
      if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
      adopt(SAMPLE_PRIOR, await res.text());
    } catch (e) {
      setReport(null);
      setFailure(ui().priorLoadFailed(String(e)));
    }
  }

  return (
    <div class="prior-panel">
      <p class="modal-note">{ui().priorHelp}</p>
      <div class="prior-actions">
        {/* A real <label> wrapping the input: the native file button has no
         * accessible name of its own, and the visible text must be the one
         * a screen reader announces. */}
        <label class="prior-pick">
          <span class="prior-pick-label">{ui().priorPick}</span>
          <input
            type="file"
            accept=".json,application/json"
            data-testid="prior-file"
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
        <button data-testid="prior-sample" onClick={() => void loadSample()}>
          {ui().priorSample}
        </button>
        <button
          class="ghost"
          data-testid="prior-clear"
          disabled={props.prior === null}
          onClick={() => {
            clearStoredPrior();
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
        <div class="prior-report" data-testid="prior-report">
          {/* The verdict is the one line that must not be missed, so it
              carries colour as well as words (never colour alone). */}
          <div class={`prior-verdict ${report.applied ? "ok" : "no"}`}>
            {report.applied ? ui().priorApplied : ui().priorNotApplied}
          </div>
          <div class="prior-summary">
            {ui().priorSummary(
              report.species,
              report.meanMoveSum,
              report.skipped,
            )}
          </div>
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

// ---------------------------------------------------------- start screen

export function StartScreen(props: {
  pool: MetaPool;
  locale: Locale;
  onLocale: (l: Locale) => void;
  mode: InfoMode;
  onMode: (m: InfoMode) => void;
  prior: StoredPrior | null;
  onPrior: (p: StoredPrior | null) => void;
  onStart: (human: SelectedTeam, bot: SelectedTeam) => void;
}) {
  const teams = props.pool.teams;
  const blind = props.mode === "blind";
  const [customs, setCustoms] = useState<CustomTeam[]>(loadCustomTeams);
  const [picks, setPicks] = useState<Picks>(() =>
    loadPicks(props.pool, loadCustomTeams()),
  );
  const [modal, setModal] = useState<null | "human" | "bot" | "prior">(null);

  function update(next: Picks) {
    setPicks(next);
    storePicks(next);
  }

  const poolIdx = (id: string) => teams.findIndex((t) => t.id === id);
  const humanChoice = picks.human;
  const botChoice = picks.bot;
  const pickedCustom =
    humanChoice.kind === "custom"
      ? customs.find((t) => t.id === humanChoice.id) ?? null
      : null;
  const pickedBotCustom =
    botChoice.kind === "custom"
      ? customs.find((t) => t.id === botChoice.id) ?? null
      : null;

  const humanValue =
    picks.human.kind === "random"
      ? ui().randomLabel
      : picks.human.kind === "pool"
        ? picks.human.id
        : (pickedCustom?.name ?? ui().randomLabel);
  const botValue =
    botChoice.kind === "random"
      ? ui().randomLabel
      : botChoice.kind === "pool"
        ? botChoice.id
        : (pickedBotCustom?.name ?? ui().randomLabel);

  function selectedTeam(choice: PartyChoice): SelectedTeam {
    if (choice.kind === "custom") {
      const custom = customs.find((t) => t.id === choice.id);
      if (custom)
        return { id: custom.name, sets: custom.sets, poolIdx: null };
    }
    // Random is resolved here, at start: a fresh roll every game unless
    // the user pinned a pool team. The roll is pool-pick.ts's, shared with
    // the blind rematch redraw so both draw by exactly the same rule.
    const pinned = choice.kind === "pool" ? poolIdx(choice.id) : -1;
    if (pinned < 0) return randomPoolTeam(props.pool);
    return { id: teams[pinned].id, sets: teams[pinned].sets, poolIdx: pinned };
  }

  function customsChanged(
    side: "human" | "bot",
    list: CustomTeam[],
    picked?: CustomTeam,
  ) {
    setCustoms(list);
    let human = picks.human;
    let bot = picks.bot;
    if (picked) {
      const choice = { kind: "custom", id: picked.id } as const;
      if (side === "human") human = choice;
      else bot = choice;
    }
    // One saved team may be pinned on both sides. Deleting it invalidates
    // both choices atomically; an in-progress Game already owns snapshots.
    const humanCustomId = human.kind === "custom" ? human.id : null;
    const botCustomId = bot.kind === "custom" ? bot.id : null;
    if (humanCustomId && !list.some((t) => t.id === humanCustomId))
      human = RANDOM;
    if (botCustomId && !list.some((t) => t.id === botCustomId))
      bot = RANDOM;
    if (human !== picks.human || bot !== picks.bot) update({ human, bot });
  }

  function start() {
    // Blind ignores the pinned opponent entirely: a foe you chose is a foe
    // whose sets you know, which is precisely the information the mode
    // withholds. The pin is kept in storage, unread, for the way back to
    // open mode.
    props.onStart(
      selectedTeam(picks.human),
      blind ? randomPoolTeam(props.pool) : selectedTeam(picks.bot),
    );
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
        <button class="primary start-main-btn" onClick={start}>
          {ui().startBattle}
        </button>
        {/* Information mode: a two-option radio built from buttons, so the
         * state is carried by aria-pressed (announced as pressed/not) and
         * the selected one also reads as `primary` for sighted users — the
         * page must not depend on a stylesheet rule to show which is on. */}
        <div class="mode-row" data-testid="mode-row">
          <span class="mode-label">{ui().modeLabel}</span>
          <button
            class={`mode-opt ${blind ? "ghost" : "primary"}`}
            data-mode="open"
            aria-pressed={!blind}
            onClick={() => props.onMode("open")}
          >
            {ui().modeOpen}
          </button>
          <button
            class={`mode-opt ${blind ? "primary" : "ghost"}`}
            data-mode="blind"
            aria-pressed={blind}
            onClick={() => props.onMode("blind")}
          >
            {ui().modeBlind}
          </button>
        </div>
        <p class="mode-note">
          {blind ? ui().modeNoteBlind : ui().modeNoteOpen}
        </p>
        <button
          class="party-btn"
          data-party="human"
          onClick={() => setModal("human")}
        >
          <span class="party-label">{ui().yourParty}</span>
          <span class="party-value">{humanValue}</span>
        </button>
        {blind ? (
          // Not a control: in blind mode the opponent is drawn at start and
          // redrawn on every rematch, so there is nothing to open. The slot
          // stays, in the party buttons' shape, to say what will happen.
          <div class="party-btn party-static" data-party="bot-random">
            <span class="party-label">{ui().oppParty}</span>
            <span class="party-value">{ui().oppRandomBlind}</span>
          </div>
        ) : (
          <button
            class="party-btn"
            data-party="bot"
            onClick={() => setModal("bot")}
          >
            <span class="party-label">{ui().oppParty}</span>
            <span class="party-value">{botValue}</span>
          </button>
        )}
        {blind && (
          // Only blind can consult a prior: open mode pins the opponent's
          // real sets, and a pinned searcher refuses the table outright.
          <button
            class="party-btn"
            data-party="prior"
            onClick={() => setModal("prior")}
          >
            <span class="party-label">{ui().priorLabel}</span>
            <span class="party-value">
              {props.prior ? props.prior.name : ui().priorNone}
            </span>
          </button>
        )}
      </main>

      {modal === "human" && (
        <Modal title={ui().chooseYours} onClose={() => setModal(null)}>
          <HumanPicker
            teams={teams}
            choice={picks.human}
            onPick={(c) => {
              update({ ...picks, human: c });
              setModal(null);
            }}
            customs={customs}
            onCustomsChange={(list, picked) => {
              // Fresh import is pinned to the side whose modal owns the
              // panel; the modal stays open so applied fixes remain visible.
              customsChanged("human", list, picked);
            }}
            mode={props.mode}
          />
        </Modal>
      )}
      {modal === "bot" && (
        <Modal title={ui().chooseOpp} onClose={() => setModal(null)}>
          <BotPicker
            teams={teams}
            choice={picks.bot}
            onPick={(c) => {
              update({ ...picks, bot: c });
              setModal(null);
            }}
            customs={customs}
            onCustomsChange={(list, picked) =>
              customsChanged("bot", list, picked)
            }
            mode={props.mode}
          />
        </Modal>
      )}
      {modal === "prior" && (
        <Modal title={ui().priorTitle} onClose={() => setModal(null)}>
          <PriorPanel prior={props.prior} onPrior={props.onPrior} />
        </Modal>
      )}
    </div>
  );
}
