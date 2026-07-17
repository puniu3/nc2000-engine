// Start screen (UI-1): a minimal centered column — Start battle / Your
// party / Opponent's party. Both parties default to random-from-pool, so
// one tap on Start begins a game; a party button opens a modal with the
// full selection content (pool team list with rank/provenance/species,
// the M14 custom-team import/pick for the human side). Pinned choices
// persist in localStorage by team id. The language selector is an
// unobtrusive corner dropdown. (The device benchmark — a dev instrument
// for the M9 think-time gate — was removed from the product UI in UI-2.)
//
// Open team sheet (M12): the bot's sets are readable in the party modal,
// and the bot receives the human's exact sets — a single information
// policy for pool and custom teams alike. Only picks stay hidden.

import { useState } from "preact/hooks";
import type { MetaPool, PoolTeam } from "./types";
import type { SelectedTeam } from "./app";
import { Modal } from "./modal";
import { getValidator, randomSeed32 } from "./engine";
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

type HumanChoice =
  | { kind: "random" }
  | { kind: "pool"; id: string }
  | { kind: "custom"; id: string };
type BotChoice = { kind: "random" } | { kind: "pool"; id: string };
interface Picks {
  human: HumanChoice;
  bot: BotChoice;
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
    if (b?.kind === "pool" && pool.teams.some((t) => t.id === b.id)) {
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

// ------------------------------------------------- party picker modals

/** Human party picker (modal body): random card, saved customs + import
 * flow, then the pool list. Picking closes the modal (onPick); managing
 * customs (import/delete) keeps it open. */
function HumanPicker(props: {
  teams: PoolTeam[];
  choice: HumanChoice;
  onPick: (c: HumanChoice) => void;
  customs: CustomTeam[];
  onCustomsChange: (list: CustomTeam[], picked?: CustomTeam) => void;
}) {
  const { teams, choice, customs } = props;
  const [importing, setImporting] = useState(false);
  return (
    <>
      <p class="modal-note">{ui().openSheetNote}</p>
      <button
        class={`team-card random-card ${choice.kind === "random" ? "selected" : ""}`}
        aria-pressed={choice.kind === "random"}
        onClick={() => props.onPick(RANDOM)}
      >
        {ui().randomCard(teams.length)}
      </button>
      <section class="custom-section">
        <h3>{ui().customSection}</h3>
        <div class="team-list">
          {customs.map((t) => (
            <CustomTeamCard
              key={t.id}
              team={t}
              selected={choice.kind === "custom" && choice.id === t.id}
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
            onSaved={(t) => props.onCustomsChange(loadCustomTeams(), t)}
            onClose={() => setImporting(false)}
          />
        )}
      </section>
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

/** Opponent party picker (modal body): random card + pool list. */
function BotPicker(props: {
  teams: PoolTeam[];
  choice: BotChoice;
  onPick: (c: BotChoice) => void;
}) {
  const { teams, choice } = props;
  return (
    <>
      <button
        class={`team-card random-card ${choice.kind === "random" ? "selected" : ""}`}
        aria-pressed={choice.kind === "random"}
        onClick={() => props.onPick(RANDOM)}
      >
        {ui().randomCard(teams.length)}
      </button>
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

// ---------------------------------------------------------- start screen

export function StartScreen(props: {
  pool: MetaPool;
  locale: Locale;
  onLocale: (l: Locale) => void;
  onStart: (human: SelectedTeam, botIdx: number | "random") => void;
}) {
  const teams = props.pool.teams;
  const [customs, setCustoms] = useState<CustomTeam[]>(loadCustomTeams);
  const [picks, setPicks] = useState<Picks>(() =>
    loadPicks(props.pool, loadCustomTeams()),
  );
  const [modal, setModal] = useState<null | "human" | "bot">(null);

  function update(next: Picks) {
    setPicks(next);
    storePicks(next);
  }

  const poolIdx = (id: string) => teams.findIndex((t) => t.id === id);
  const humanChoice = picks.human;
  const pickedCustom =
    humanChoice.kind === "custom"
      ? customs.find((t) => t.id === humanChoice.id) ?? null
      : null;

  const humanValue =
    picks.human.kind === "random"
      ? ui().randomLabel
      : picks.human.kind === "pool"
        ? picks.human.id
        : (pickedCustom?.name ?? ui().randomLabel);
  const botValue =
    picks.bot.kind === "random" ? ui().randomLabel : picks.bot.id;

  function start() {
    let human: SelectedTeam;
    if (picks.human.kind === "custom" && pickedCustom) {
      human = { id: pickedCustom.name, sets: pickedCustom.sets, poolIdx: null };
    } else {
      // Random is resolved here, at start: a fresh roll every game unless
      // the user pinned a team.
      const idx =
        picks.human.kind === "pool"
          ? poolIdx(picks.human.id)
          : randomSeed32() % teams.length;
      human = { id: teams[idx].id, sets: teams[idx].sets, poolIdx: idx };
    }
    props.onStart(
      human,
      picks.bot.kind === "pool" ? poolIdx(picks.bot.id) : "random",
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
          data-party="bot"
          onClick={() => setModal("bot")}
        >
          <span class="party-label">{ui().oppParty}</span>
          <span class="party-value">{botValue}</span>
        </button>
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
              setCustoms(list);
              if (picked) {
                // Freshly imported: pin it (modal stays open so the
                // import result / applied fixes remain readable).
                update({ ...picks, human: { kind: "custom", id: picked.id } });
              } else if (
                humanChoice.kind === "custom" &&
                !list.some((t) => t.id === humanChoice.id)
              ) {
                update({ ...picks, human: RANDOM }); // pinned custom deleted
              }
            }}
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
          />
        </Modal>
      )}
    </div>
  );
}
