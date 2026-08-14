// The study board (`?solver`): type a position in, get every option scored.
//
// Not a battle screen with the playing removed — a different object. A game
// owns a real `Battle` and advances it; this owns a DESCRIPTION of one, which
// the user edits freely and which may be re-solved as often as they like. So
// there is no engine instance on this thread at all: the description goes to
// the worker as JSON, and what comes back is a report plus the board the
// solver read, which is shown next to the answer. A position typed one way
// and analyzed another is the only failure that would make this tool worse
// than useless, and the readback is what makes it visible.
//
// The information structure is the ladder bot's, not the demo's: the user's
// own sets are exact, the opponent is public facts only. That is why the
// opponent editor asks for species, HP, status and *shown* moves rather than
// a team — and why every derived number that leans on an imputed set says so
// where it is printed.

import { useEffect, useRef, useState } from "preact/hooks";
import { Modal } from "./modal";
import { HpBar, StatusBadge, TypeBadge } from "./battle-ui";
import { getValidator } from "./engine";
import { findingText, type Finding } from "./findings";
import { parsePsExport } from "./ps-import";
import { Narrator } from "./narrate";
import {
  loadCustomTeams,
  type CustomTeam,
} from "./custom-teams";
import {
  fixedGender,
  itemList,
  maxPp,
  moveList,
  speciesDisplay,
  speciesList,
  speciesTypes,
} from "./set-info";
import {
  boostLabel,
  condName,
  itemName,
  moveName,
  setLocale,
  speciesName,
  statusName,
  toId,
  ui,
  type Locale,
} from "./i18n";
import {
  blankPosition,
  clearStoredPosition,
  hasCondition,
  hasVolatile,
  loadStoredPosition,
  poolRoster,
  positionProblems,
  setActive,
  setCondition,
  setMon,
  setVolatile,
  storePosition,
  toSolverSpec,
  withFoeRoster,
  withOwnTeam,
  type ActionRef,
  type AnalysisReport,
  type MonSpec,
  type PositionSpec,
} from "./position";
import { PositionRejected, SolverWorker, type SolveResult } from "./solver-client";
import type { MetaPool, StateView } from "./types";

/** Search budgets the button offers. The default matches the shipped bot
 * (app.tsx BUDGET), so "what would the bot do here" is answerable exactly;
 * the larger ones are the study case, where waiting is the point. Browser
 * E2E builds use Vite's explicit `test` mode to make a whole analysis cheap
 * — the same override app.tsx applies to the game, restated here rather than
 * imported, because importing it would close a cycle (app renders this). */
const testBudget =
  import.meta.env.MODE === "test"
    ? Number(import.meta.env.VITE_NC2000_TEST_BUDGET)
    : Number.NaN;
const DEFAULT_BUDGET =
  Number.isSafeInteger(testBudget) && testBudget > 0 ? testBudget : 30_000;
const BUDGETS = [...new Set([DEFAULT_BUDGET, 3_000, 30_000, 100_000, 300_000])].sort(
  (a, b) => a - b,
);
/** Plies of searched line to report. Six is about where a gen-2 line stops
 * being a claim about this turn and starts being a story. */
const LINE_PLIES = 6;

const STATUSES = ["", "brn", "par", "slp", "frz", "psn", "tox"];
const BOOST_STATS = ["atk", "def", "spa", "spd", "spe"];
/** Announced volatiles a human actually tracks. The spec accepts every
 * condition the engine has; this list is the subset worth a checkbox. */
const VOLATILES = [
  "substitute",
  "confusion",
  "leechseed",
  "curse",
  "attract",
  "nightmare",
  "focusenergy",
  "destinybond",
];
const SIDE_CONDS = ["spikes", "reflect", "lightscreen", "safeguard", "mist"];
const WEATHERS = ["", "raindance", "sunnyday", "sandstorm", "hail"];

export function Solver(props: {
  /** The bundled meta pool — the team lists the pickers offer. */
  pool: MetaPool;
  /** The belief candidate pool the searcher reasons with. */
  poolJson: string;
  locale: Locale;
  onLocale: (l: Locale) => void;
}) {
  const [spec, setSpec] = useState<PositionSpec>(
    () => loadStoredPosition() ?? blankPosition(),
  );
  const [result, setResult] = useState<SolveResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<[number, number] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [budget, setBudget] = useState(DEFAULT_BUDGET);
  const [picker, setPicker] = useState<null | "own" | "foe">(null);
  const worker = useRef<SolverWorker | null>(null);

  useEffect(() => {
    worker.current = new SolverWorker();
    return () => worker.current?.dispose();
  }, []);
  useEffect(() => storePosition(spec), [spec]);

  const s = ui().solver;
  const me = spec.side;
  const foe = 1 - me;
  const problems = positionProblems(spec);

  /** Editing the position invalidates the answer: a report next to a board
   * it was not computed from is the same lie as a mistyped position. */
  const edit = (next: PositionSpec) => {
    setSpec(next);
    setResult(null);
    setError(null);
  };

  const run = async (extend: boolean) => {
    const w = worker.current;
    if (!w) return;
    setBusy(true);
    setError(null);
    const onProgress = (done: number, total: number) => setProgress([done, total]);
    try {
      const total = extend ? (result?.report.iterations ?? 0) + budget : budget;
      const r = extend
        ? await w.extend(total, LINE_PLIES, onProgress)
        : await w.solve({
            spec: JSON.stringify(toSolverSpec(spec)),
            poolJson: props.poolJson,
            seed: 1,
            budget: total,
            plies: LINE_PLIES,
            onProgress,
          });
      setResult(r);
    } catch (e) {
      setError(e instanceof PositionRejected ? s.rejected(e.message) : String(e));
      setResult(null);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  return (
    <main class="screen solver-screen">
      <header class="solver-header">
        <h1 class="solver-title">{s.title}</h1>
        <div class="solver-header-actions">
          <button
            class="ghost"
            onClick={() => {
              const l: Locale = props.locale === "ja" ? "en" : "ja";
              setLocale(l);
              props.onLocale(l);
            }}
          >
            {props.locale === "ja" ? "English" : "日本語"}
          </button>
          <button
            class="ghost"
            data-testid="solver-reset"
            onClick={() => {
              clearStoredPosition();
              edit(blankPosition());
            }}
          >
            {s.reset}
          </button>
        </div>
      </header>
      <p class="solver-intro">{s.intro}</p>

      <section class="solver-panel" data-testid="solver-own">
        <div class="solver-panel-head">
          <h2>{s.yourSide}</h2>
          <button class="ghost" onClick={() => setPicker("own")}>
            {s.fromPool} / {s.fromPaste}
          </button>
        </div>
        {spec.sides[me].mons.length === 0 ? (
          <p class="solver-empty">{s.noTeamYet}</p>
        ) : (
          <div class="solver-mons">
            {spec.sides[me].mons.map((m, i) => (
              <MonEditor
                key={`me-${i}-${m.species}`}
                mon={m}
                mine
                slot={i}
                spec={spec}
                side={me}
                onEdit={edit}
              />
            ))}
          </div>
        )}
      </section>

      <section class="solver-panel" data-testid="solver-foe">
        <div class="solver-panel-head">
          <h2>{s.foeSide}</h2>
          <button class="ghost" onClick={() => setPicker("foe")}>
            {s.fromPool}
          </button>
        </div>
        {spec.sides[foe].mons.length === 0 ? (
          <p class="solver-empty">{s.noFoeYet}</p>
        ) : (
          <div class="solver-mons">
            {spec.sides[foe].mons.map((m, i) => (
              <MonEditor
                key={`foe-${i}-${m.species}`}
                mon={m}
                mine={false}
                slot={i}
                spec={spec}
                side={foe}
                onEdit={edit}
              />
            ))}
          </div>
        )}
      </section>

      <FieldEditor spec={spec} onEdit={edit} />

      <section class="solver-run">
        <label class="solver-budget">
          {s.budgetLabel}
          <select
            value={String(budget)}
            onChange={(e) =>
              setBudget(Number((e.target as HTMLSelectElement).value))
            }
          >
            {BUDGETS.map((b) => (
              <option key={b} value={String(b)}>
                {b.toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <button
          class="primary solver-solve"
          data-testid="solver-solve"
          disabled={busy || problems.length > 0}
          onClick={() => void run(false)}
        >
          {busy ? s.solving : s.solve}
        </button>
        {busy && (
          <button class="ghost" onClick={() => worker.current?.stop()}>
            {s.stop}
          </button>
        )}
        {!busy && result && (
          <button class="ghost" onClick={() => void run(true)}>
            {s.deeper}
          </button>
        )}
        {progress && (
          <div class="solver-progress">
            <div
              class="solver-progress-fill"
              style={{ width: `${(100 * progress[0]) / Math.max(1, progress[1])}%` }}
            />
          </div>
        )}
      </section>

      {problems.length > 0 && (
        <p class="solver-problems" data-testid="solver-problems">
          {s.problems}{" "}
          {problems
            .map((p) =>
              p === "own-team"
                ? s.problemOwnTeam
                : p === "foe-team"
                  ? s.problemFoeTeam
                  : p === "own-active"
                    ? s.problemOwnActive
                    : p === "foe-active"
                      ? s.problemFoeActive
                      : p === "hp"
                        ? s.problemHp
                        : s.problemFainted,
            )
            .join(" / ")}
        </p>
      )}
      {error && (
        <p class="solver-error" data-testid="solver-error">
          {error}
        </p>
      )}

      {result && <Results result={result} side={me} />}

      {/* One datalist per vocabulary, at the root: the move and item inputs
        * repeat per Pokémon, and a list repeated with them would be six
        * copies of the same 267 options under six copies of one id. */}
      <datalist id="solver-move-list">
        {moveList().map((id) => (
          <option key={id} value={id}>
            {moveName(id)}
          </option>
        ))}
      </datalist>
      <datalist id="solver-item-list">
        {itemList().map((id) => (
          <option key={id} value={id}>
            {itemName(id)}
          </option>
        ))}
      </datalist>

      {picker === "own" && (
        <Modal title={s.yourSide} onClose={() => setPicker(null)}>
          <OwnTeamPicker
            pool={props.pool}
            onPick={(sets) => {
              edit(withOwnTeam(spec, sets as never[]));
              setPicker(null);
            }}
          />
        </Modal>
      )}
      {picker === "foe" && (
        <Modal title={s.foeSide} onClose={() => setPicker(null)}>
          <FoeRosterPicker
            pool={props.pool}
            current={spec.sides[foe].mons}
            onPick={(mons) => {
              edit(withFoeRoster(spec, mons));
              setPicker(null);
            }}
          />
        </Modal>
      )}
    </main>
  );
}

// ------------------------------------------------------------- mon editor

function MonEditor(props: {
  mon: MonSpec;
  mine: boolean;
  slot: number;
  side: number;
  spec: PositionSpec;
  onEdit: (s: PositionSpec) => void;
}) {
  const { mon, slot, side, spec } = props;
  const s = ui().solver;
  const patch = (p: Partial<MonSpec>) => props.onEdit(setMon(spec, side, slot, p));
  const den = mon.hp_den ?? 100;
  const pct = Math.round((100 * (mon.hp_num ?? den)) / Math.max(1, den));
  const types = speciesTypes(mon.species) ?? [];

  return (
    <div class={`solver-mon${mon.active ? " is-active" : ""}`} data-species={toId(mon.species)}>
      <div class="solver-mon-head">
        <button
          class={`solver-active-btn${mon.active ? " on" : ""}`}
          title={s.setActive}
          aria-pressed={!!mon.active}
          disabled={!!mon.fainted}
          onClick={() => props.onEdit(setActive(spec, side, mon.active ? null : slot))}
        >
          {s.activeLabel}
        </button>
        <span class="solver-mon-name">{speciesName(speciesDisplay(mon.species))}</span>
        <span class="solver-lvl">L{mon.level}</span>
        {/* Gender is public at preview and the belief matches teams on it, so
          * for a species that can be either it is a question, not a default.
          * Ours comes from the sets and is never asked. */}
        {!props.mine && fixedGender(mon.species) === null && (
          <select
            class="solver-gender"
            value={mon.gender ?? "M"}
            onChange={(e) => patch({ gender: (e.target as HTMLSelectElement).value })}
          >
            <option value="M">♂</option>
            <option value="F">♀</option>
          </select>
        )}
        {types.map((t) => (
          <TypeBadge key={t} type={t} />
        ))}
        <label class="solver-check">
          <input
            type="checkbox"
            checked={!!mon.appeared}
            onChange={(e) =>
              patch({ appeared: (e.target as HTMLInputElement).checked })
            }
          />
          {/* Our own three are a choice we made and know; theirs is a fact
            * the battle revealed. Same field, and deliberately not the same
            * word. */}
          {props.mine ? s.pickedLabel : s.seenLabel}
        </label>
        <label class="solver-check">
          <input
            type="checkbox"
            checked={!!mon.fainted}
            onChange={(e) => {
              const on = (e.target as HTMLInputElement).checked;
              patch({
                fainted: on,
                hp_num: on ? 0 : den,
                status: on ? "fnt" : "",
                active: on ? false : mon.active,
                appeared: on ? true : mon.appeared,
              });
              if (on && spec.sides[side].active === slot) {
                props.onEdit(setActive(setMon(spec, side, slot, {
                  fainted: true, hp_num: 0, status: "fnt", appeared: true,
                }), side, null));
              }
            }}
          />
          {s.faintedLabel}
        </label>
      </div>

      {!mon.fainted && (
        <div class="solver-mon-row">
          <label class="solver-field">
            {s.hpLabel}
            <input
              type="number"
              min={1}
              max={den}
              value={String(mon.hp_num ?? den)}
              onInput={(e) => {
                const v = Number((e.target as HTMLInputElement).value);
                patch({ hp_num: Math.max(0, Math.min(den, v || 0)) });
              }}
            />
          </label>
          <HpBar pct={pct} />
          <label class="solver-field">
            {s.statusLabel}
            <select
              value={mon.status ?? ""}
              onChange={(e) =>
                patch({ status: (e.target as HTMLSelectElement).value })
              }
            >
              {STATUSES.map((st) => (
                <option key={st} value={st}>
                  {st === "" ? "—" : statusName(st)}
                </option>
              ))}
            </select>
          </label>
          {mon.status === "slp" && (
            <label class="solver-check">
              <input
                type="checkbox"
                checked={!!mon.rest}
                onChange={(e) =>
                  patch({ rest: (e.target as HTMLInputElement).checked })
                }
              />
              {s.restLabel}
            </label>
          )}
        </div>
      )}

      {mon.active && !mon.fainted && (
        <ActiveDetail
          mon={mon}
          mine={props.mine}
          slot={slot}
          side={side}
          spec={spec}
          onEdit={props.onEdit}
        />
      )}

      {!props.mine && (
        <>
          <RevealedMoves mon={mon} slot={slot} side={side} spec={spec} onEdit={props.onEdit} />
          <FoeItem mon={mon} slot={slot} side={side} spec={spec} onEdit={props.onEdit} />
        </>
      )}
    </div>
  );
}

/** Boosts, field volatiles and (ours) PP — the things that only matter for
 * the two Pokémon actually out. Asking for them on all twelve would bury the
 * position under fields nobody fills in. */
function ActiveDetail(props: {
  mon: MonSpec;
  mine: boolean;
  slot: number;
  side: number;
  spec: PositionSpec;
  onEdit: (s: PositionSpec) => void;
}) {
  const { mon, slot, side, spec } = props;
  const s = ui().solver;
  const boosts = mon.boosts ?? [0, 0, 0, 0, 0, 0, 0];
  const setBoost = (i: number, v: number) => {
    const next = boosts.slice() as NonNullable<MonSpec["boosts"]>;
    next[i] = Math.max(-6, Math.min(6, v));
    props.onEdit(setMon(spec, side, slot, { boosts: next }));
  };
  const ownMoves = props.mine ? ownSetMoves(spec, slot) : [];

  return (
    <div class="solver-active-detail">
      <div class="solver-mon-row">
        <span class="solver-label">{s.boostsLabel}</span>
        {BOOST_STATS.map((stat, i) => (
          <label key={stat} class="solver-boost">
            {boostLabel(stat)}
            <input
              type="number"
              min={-6}
              max={6}
              value={String(boosts[i])}
              onInput={(e) => setBoost(i, Number((e.target as HTMLInputElement).value) || 0)}
            />
          </label>
        ))}
      </div>
      <div class="solver-mon-row solver-vols">
        <span class="solver-label">{s.volatilesLabel}</span>
        {VOLATILES.map((key) => (
          <label key={key} class="solver-check">
            <input
              type="checkbox"
              checked={hasVolatile(mon, key)}
              onChange={(e) =>
                props.onEdit(
                  setVolatile(spec, side, slot, key, (e.target as HTMLInputElement).checked),
                )
              }
            />
            {condName(key)}
          </label>
        ))}
      </div>
      {props.mine && ownMoves.length > 0 && (
        <div class="solver-mon-row">
          <span class="solver-label">{s.ppLabel}</span>
          {ownMoves.map((move) => {
            const mx = maxPp(move) ?? 0;
            const used = (mon.uses ?? []).find((u) => toId(u.move) === toId(move))?.n ?? 0;
            return (
              <label key={move} class="solver-pp">
                {moveName(toId(move))}
                <input
                  type="number"
                  min={0}
                  max={mx}
                  value={String(Math.max(0, mx - used))}
                  onInput={(e) => {
                    const left = Number((e.target as HTMLInputElement).value);
                    const uses = (mon.uses ?? []).filter(
                      (u) => toId(u.move) !== toId(move),
                    );
                    const n = Math.max(0, Math.min(mx, mx - (left || 0)));
                    if (n > 0) uses.push({ move: toId(move), n });
                    props.onEdit(setMon(spec, side, slot, { uses }));
                  }}
                />
                <span class="solver-pp-max">/{mx}</span>
              </label>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** The opponent's reveal channel: only what has actually been shown. This is
 * the single most valuable field on the screen — it is what narrows the
 * belief from "any team" to "this one" — so it is a list the user builds,
 * never a set they fill in. */
function RevealedMoves(props: {
  mon: MonSpec;
  slot: number;
  side: number;
  spec: PositionSpec;
  onEdit: (s: PositionSpec) => void;
}) {
  const { mon, slot, side, spec } = props;
  const s = ui().solver;
  const [draft, setDraft] = useState("");
  const shown = mon.revealed_moves ?? [];

  const add = () => {
    const id = toId(draft);
    if (!id || shown.includes(id) || !moveList().includes(id)) return;
    props.onEdit(
      setMon(spec, side, slot, {
        revealed_moves: [...shown, id],
        uses: [...(mon.uses ?? []), { move: id, n: 1 }].filter(
          (u, i, all) => all.findIndex((x) => x.move === u.move) === i,
        ),
      }),
    );
    setDraft("");
  };

  return (
    <div class="solver-mon-row solver-revealed">
      <span class="solver-label" title={s.revealedHelp}>
        {s.revealedLabel}
      </span>
      {shown.map((id) => (
        <button
          key={id}
          class="solver-chip"
          title={ui().close}
          onClick={() =>
            props.onEdit(
              setMon(spec, side, slot, {
                revealed_moves: shown.filter((x) => x !== id),
                uses: (mon.uses ?? []).filter((u) => u.move !== id),
              }),
            )
          }
        >
          {moveName(id)} ×
        </button>
      ))}
      <input
        class="solver-move-input"
        list="solver-move-list"
        placeholder={s.addMove}
        value={draft}
        onInput={(e) => setDraft((e.target as HTMLInputElement).value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            add();
          }
        }}
        onBlur={add}
      />
    </div>
  );
}

/** What is known about an opponent's item. Two separate facts, because the
 * belief uses them differently: whether it brought one at all (the `|poke|`
 * preview flag, which every candidate team is filtered on) and which one, if
 * it has been shown. Getting the first wrong is not cosmetic — a roster that
 * claims items nobody holds matches no team at all. */
function FoeItem(props: {
  mon: MonSpec;
  slot: number;
  side: number;
  spec: PositionSpec;
  onEdit: (s: PositionSpec) => void;
}) {
  const { mon, slot, side, spec } = props;
  const s = ui().solver;
  const known = mon.item_original?.known ? (mon.item_original.item ?? "") : "";
  return (
    <div class="solver-mon-row">
      <label class="solver-check">
        <input
          type="checkbox"
          checked={mon.item_flag !== false}
          onChange={(e) =>
            props.onEdit(
              setMon(spec, side, slot, {
                item_flag: (e.target as HTMLInputElement).checked,
              }),
            )
          }
        />
        {s.itemHeldLabel}
      </label>
      <label class="solver-field">
        {s.itemLabel}
        <input
          list="solver-item-list"
          placeholder={s.itemUnknown}
          value={known}
          onInput={(e) => {
            const raw = (e.target as HTMLInputElement).value;
            const id = toId(raw);
            const k = id ? { known: true, item: id } : { known: false, item: null };
            props.onEdit(
              setMon(spec, side, slot, {
                item_original: k,
                item_current: k,
                item_flag: id ? true : mon.item_flag,
              }),
            );
          }}
        />
      </label>
    </div>
  );
}

function ownSetMoves(spec: PositionSpec, slot: number): string[] {
  const set = spec.own_sets[slot] as { moves?: string[] } | undefined;
  return set?.moves ?? [];
}

// ----------------------------------------------------------------- field

function FieldEditor(props: {
  spec: PositionSpec;
  onEdit: (s: PositionSpec) => void;
}) {
  const { spec } = props;
  const s = ui().solver;
  const me = spec.side;
  const foe = 1 - me;
  return (
    <section class="solver-panel solver-field-panel">
      <div class="solver-panel-head">
        <h2>{s.fieldSection}</h2>
      </div>
      <div class="solver-mon-row">
        <label class="solver-field">
          {s.turnLabel}
          <input
            type="number"
            min={1}
            value={String(spec.turn)}
            onInput={(e) =>
              props.onEdit({
                ...spec,
                turn: Math.max(1, Number((e.target as HTMLInputElement).value) || 1),
              })
            }
          />
        </label>
        <label class="solver-field">
          {s.weatherLabel}
          <select
            value={spec.weather?.key ?? ""}
            onChange={(e) => {
              const key = (e.target as HTMLSelectElement).value;
              props.onEdit({ ...spec, weather: key ? { key, upkeeps: 0 } : null });
            }}
          >
            {WEATHERS.map((w) => (
              <option key={w} value={w}>
                {w === "" ? s.weatherNone : condName(w)}
              </option>
            ))}
          </select>
        </label>
        <label class="solver-check">
          <input
            type="checkbox"
            checked={!!spec.force_switch}
            onChange={(e) =>
              props.onEdit({
                ...spec,
                force_switch: (e.target as HTMLInputElement).checked,
              })
            }
          />
          {s.forceSwitchLabel}
        </label>
        <label class="solver-check">
          <input
            type="checkbox"
            checked={!!spec.trapped}
            onChange={(e) =>
              props.onEdit({ ...spec, trapped: (e.target as HTMLInputElement).checked })
            }
          />
          {s.trappedLabel}
        </label>
      </div>
      {[me, foe].map((side) => (
        <div key={side} class="solver-mon-row">
          <span class="solver-label">
            {side === me ? s.yourSide : s.foeSide}
          </span>
          {SIDE_CONDS.map((key) => (
            <label key={key} class="solver-check">
              <input
                type="checkbox"
                checked={hasCondition(spec, side, key)}
                onChange={(e) =>
                  props.onEdit(
                    setCondition(spec, side, key, (e.target as HTMLInputElement).checked),
                  )
                }
              />
              {condName(key)}
            </label>
          ))}
        </div>
      ))}
    </section>
  );
}

// ---------------------------------------------------------------- pickers

interface SetLike {
  species: string;
  level?: number;
  gender?: string;
  item?: string;
  moves?: string[];
}

function OwnTeamPicker(props: { pool: MetaPool; onPick: (sets: SetLike[]) => void }) {
  const s = ui().solver;
  const [tab, setTab] = useState<"pool" | "custom" | "paste">("pool");
  const [customs] = useState<CustomTeam[]>(loadCustomTeams);
  const [text, setText] = useState("");
  const [errors, setErrors] = useState<string[]>([]);

  const importPaste = () => {
    const parsed = parsePsExport(text);
    if (parsed.findings.length > 0 || parsed.sets.length === 0) {
      setErrors(
        parsed.findings.length > 0
          ? parsed.findings.map((f) => findingText(f as unknown as Finding))
          : [ui().importErrors(1)],
      );
      return;
    }
    const verdict = JSON.parse(
      getValidator().canonicalizeTeam(JSON.stringify(parsed.sets)),
    ) as { ok: boolean; team?: SetLike[]; errors?: unknown[] };
    if (!verdict.ok || !verdict.team) {
      setErrors((verdict.errors ?? []).map((f) => findingText(f as Finding)));
      return;
    }
    props.onPick(verdict.team);
  };

  return (
    <div class="solver-picker">
      <div class="solver-tabs">
        {(["pool", "custom", "paste"] as const).map((t) => (
          <button
            key={t}
            class={`ghost${tab === t ? " on" : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "pool" ? s.fromPool : t === "custom" ? s.fromCustom : s.fromPaste}
          </button>
        ))}
      </div>
      {tab === "pool" && (
        <ul class="solver-team-list">
          {props.pool.teams.map((t) => (
            <li key={t.id}>
              <button class="solver-team-btn" onClick={() => props.onPick(t.sets as SetLike[])}>
                <strong>{t.id}</strong>
                <span>{t.species.map((sp) => speciesName(sp)).join(" / ")}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {tab === "custom" && (
        <ul class="solver-team-list">
          {customs.length === 0 && <li class="solver-empty">{s.noTeamYet}</li>}
          {customs.map((t) => (
            <li key={t.id}>
              <button class="solver-team-btn" onClick={() => props.onPick(t.sets as SetLike[])}>
                <strong>{t.name}</strong>
                <span>{t.species.map((sp) => speciesName(sp)).join(" / ")}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {tab === "paste" && (
        <div class="solver-paste">
          <textarea
            rows={12}
            value={text}
            placeholder={ui().importPlaceholder}
            onInput={(e) => setText((e.target as HTMLTextAreaElement).value)}
          />
          <button class="primary" onClick={importPaste}>
            {ui().importButton}
          </button>
          {errors.length > 0 && (
            <ul class="solver-errors">
              {errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

/** The opponent as the game shows them: six species and levels, nothing
 * else. Picking a pool team here copies only that — never its sets, which
 * are exactly what the solver must not be told. */
function FoeRosterPicker(props: {
  pool: MetaPool;
  current: MonSpec[];
  onPick: (mons: SetLike[]) => void;
}) {
  const s = ui().solver;
  const [rows, setRows] = useState<SetLike[]>(() =>
    props.current.length === 6
      ? props.current.map((m) => ({ species: m.species, level: m.level }))
      : Array.from({ length: 6 }, () => ({ species: "", level: 55 })),
  );
  const set = (i: number, patch: Partial<SetLike>) =>
    setRows(rows.map((r, j) => (i === j ? { ...r, ...patch } : r)));
  const ready = rows.every((r) => speciesList().includes(toId(r.species)));

  return (
    <div class="solver-picker">
      <ul class="solver-team-list">
        {props.pool.teams.map((t) => (
          <li key={t.id}>
            <button
              class="solver-team-btn"
              onClick={() => props.onPick(poolRoster(t))}
            >
              <strong>{t.id}</strong>
              <span>{t.species.map((sp) => speciesName(sp)).join(" / ")}</span>
            </button>
          </li>
        ))}
      </ul>
      <div class="solver-foe-manual">
        {rows.map((r, i) => (
          <div key={i} class="solver-mon-row">
            <input
              list="solver-species-list"
              value={r.species}
              placeholder="Pokémon"
              onInput={(e) => set(i, { species: (e.target as HTMLInputElement).value })}
            />
            <input
              type="number"
              min={50}
              max={55}
              value={String(r.level ?? 55)}
              onInput={(e) =>
                set(i, { level: Number((e.target as HTMLInputElement).value) || 55 })
              }
            />
          </div>
        ))}
        <datalist id="solver-species-list">
          {speciesList().map((id) => (
            <option key={id} value={id}>
              {speciesName(speciesDisplay(id))}
            </option>
          ))}
        </datalist>
        <button
          class="primary"
          disabled={!ready}
          onClick={() =>
            props.onPick(rows.map((r) => ({ species: toId(r.species), level: r.level })))
          }
        >
          {s.solve}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- results

function Results(props: { result: SolveResult; side: number }) {
  const s = ui().solver;
  const { report, state, ms } = props.result;
  return (
    <section class="solver-results" data-testid="solver-results">
      <div class="solver-results-head">
        <h2>{s.resultTitle}</h2>
        <span class="solver-meta">{s.iterationsDone(report.iterations, ms)}</span>
        {report.belief && (
          <span class="solver-meta" data-testid="solver-belief">
            {report.belief.fallback
              ? s.beliefOffPool
              : s.beliefCount(report.belief.count)}
          </span>
        )}
      </div>
      <p class="solver-note">{s.estimateNote}</p>
      <p class="solver-value" data-testid="solver-value">
        {s.positionValue(pct(report.equilibrium.value))}
      </p>
      <p class="solver-note">{s.valueHelp}</p>

      <table class="solver-actions" data-testid="solver-actions">
        <thead>
          <tr>
            <th>{s.colOption}</th>
            <th>{s.colWinRate}</th>
            <th>{s.colWorst}</th>
            <th title={s.mixHelp}>{s.colMix}</th>
            <th>{s.colShare}</th>
            <th>{s.colVisits}</th>
          </tr>
        </thead>
        <tbody>
          {report.actions.map((a) => (
            <tr key={a.input} class={a.dominated ? "is-dominated" : ""}>
              <td>
                {actionLabel(a)}
                {a.dominated && (
                  <span class="solver-tag" title={a.reason ?? ""}>
                    {s.dominatedTag}
                  </span>
                )}
              </td>
              <td class="num">{pct(a.equity)}</td>
              {/* The floor. Blank when no reply to this action was sampled
                * often enough to be evidence about a worst case. */}
              <td class="num solver-dim">{a.worst === null ? "—" : pct(a.worst)}</td>
              <td class="num">{a.mix > 0.005 ? pct(a.mix) : ""}</td>
              <td>
                <div class="solver-share">
                  <div class="solver-share-fill" style={{ width: `${a.frac * 100}%` }} />
                  <span>{pct(a.frac)}</span>
                </div>
              </td>
              <td class="num">{a.visits.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {report.matrix.cols.length > 0 && (
        <div class="solver-block">
          <h3>{s.matrixTitle}</h3>
          <p class="solver-note">{s.matrixHelp}</p>
          <div class="solver-scroll">
            <table class="solver-matrix" data-testid="solver-matrix">
              <thead>
                <tr>
                  <th />
                  {report.matrix.cols.map((c) => (
                    <th key={c.input}>
                      {actionLabel(c)}
                      {/* A reply that only some candidate sets carry is a
                        * statement about those sets, not about the opponent. */}
                      {c.available < 0.995 && (
                        <span class="solver-avail">{s.availableIn(pct(c.available))}</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {report.actions.map((a, i) => (
                  <tr key={a.input}>
                    <th>{actionLabel(a)}</th>
                    {report.matrix.cells[i]?.map((cell, j) => (
                      <td
                        key={j}
                        class="solver-cell"
                        style={cell ? cellStyle(cell.mean, cell.n) : undefined}
                        title={cell ? `${cell.n}` : s.matrixEmptyCell}
                      >
                        {cell ? pct(cell.mean) : "·"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div class="solver-block">
        <h3>{s.damageTitle}</h3>
        <div class="solver-damage">
          <DamageList title={s.damageMine} rows={report.damage.mine} />
          <DamageList title={s.damageTheirs} rows={report.damage.theirs} />
        </div>
      </div>

      {report.line && report.line.steps.length > 0 && (
        <div class="solver-block">
          <h3>{s.lineTitle}</h3>
          <p class="solver-note">{s.lineHelp}</p>
          <Line line={report.line} side={props.side} />
        </div>
      )}

      <div class="solver-block">
        <h3>{s.understood}</h3>
        <p class="solver-note">{s.understoodHelp}</p>
        <BoardReadback state={state} side={props.side} />
      </div>
    </section>
  );
}

function DamageList(props: { title: string; rows: AnalysisReport["damage"]["mine"] }) {
  const s = ui().solver;
  return (
    <div>
      <h4>{props.title}</h4>
      <ul class="solver-damage-list">
        {props.rows.map((d) => {
          const pctOf = (v: number) => Math.round((100 * v) / Math.max(1, d.maxhp));
          return (
            <li key={d.move}>
              <span class="solver-move">{moveName(d.move)}</span>
              {d.max <= 0 ? (
                <span class="solver-dim">{s.koNone}</span>
              ) : (
                <>
                  <span>
                    {d.min}–{d.max} ({pctOf(d.min)}–{pctOf(d.max)}%)
                  </span>
                  <span class="solver-dim">
                    {s.critLabel} {d.crit}
                  </span>
                  <span class="solver-ko">
                    {d.ko === "always"
                      ? s.koAlways
                      : d.ko === "possible"
                        ? s.koPossible
                        : d.hitsGuaranteed
                          ? s.koNever(d.hitsGuaranteed)
                          : ""}
                  </span>
                </>
              )}
              {!d.revealed && <span class="solver-dim">({s.damageAssumed})</span>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Line(props: { line: NonNullable<AnalysisReport["line"]>; side: number }) {
  const s = ui().solver;
  const narrator = new Narrator(props.side);
  const hidden = props.line.assumed.filter((a) => !a.appeared);
  return (
    <div class="solver-line">
      {hidden.length > 0 && (
        <p class="solver-note">
          {s.lineAssumed}:{" "}
          {hidden
            .map(
              (a) =>
                `${speciesName(speciesDisplay(a.species))} (${a.moves
                  .map((m) => moveName(m))
                  .join(", ")})`,
            )
            .join(" / ")}
        </p>
      )}
      <ol class="solver-line-steps">
        {props.line.steps.map((step, i) => (
          <li key={i}>
            <div class="solver-line-picks">
              <span>
                {s.lineUs}: {step.mine ? inputLabel(step.mine) : "—"}
              </span>
              <span>
                {s.lineThem}: {step.theirs ? inputLabel(step.theirs) : "—"}
              </span>
            </div>
            <ul class="solver-line-log">
              {narrator.render(step.log).map((e, j) => (
                <li key={j} class={`log-${e.kind}`}>
                  {e.text}
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** The synthesized board, rendered from the same `StateView` the battle
 * screen uses. Only the two actives and the HP figures matter here: this is
 * a check on the input, not a second battle view. */
function BoardReadback(props: { state: StateView; side: number }) {
  const me = props.state.sides[props.side];
  const foe = props.state.sides[1 - props.side];
  return (
    <div class="solver-readback">
      {[foe, me].map((sd, i) => {
        const p = sd.active != null ? sd.party[sd.active] : null;
        return (
          <div key={i} class="solver-readback-row">
            <span class="solver-readback-who">
              {i === 0 ? ui().solver.foeSide : ui().solver.yourSide}
            </span>
            {p ? (
              <>
                <span>{speciesName(p.species)}</span>
                <span class="num">
                  {p.hp}/{p.maxhp}
                </span>
                <HpBar pct={Math.round((100 * p.hp) / Math.max(1, p.maxhp))} />
                {p.status && <StatusBadge status={p.status} />}
                {p.item && <span class="solver-dim">{itemName(p.item)}</span>}
              </>
            ) : (
              <span class="solver-dim">—</span>
            )}
            <span class="solver-dim">
              {sd.sideConditions.map((c) => condName(c)).join(" ")}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// --------------------------------------------------------------- helpers

const pct = (x: number) => `${(100 * x).toFixed(1)}%`;

/** A win rate painted onto a red-to-green ramp, faded by how little the cell
 * was sampled: a bright colour on four playouts would be a lie told in the
 * most persuasive channel the table has. */
function cellStyle(mean: number, n: number): Record<string, string> {
  const hue = Math.round(120 * Math.max(0, Math.min(1, mean)));
  const trust = Math.min(1, n / 200);
  return {
    background: `hsl(${hue} 45% ${18 + 14 * trust}%)`,
    opacity: String(0.45 + 0.55 * trust),
  };
}

function actionLabel(a: ActionRef): string {
  if (a.kind === "move" && a.move) return moveName(a.move);
  if (a.kind === "switch") {
    return a.species
      ? `→ ${speciesName(speciesDisplay(a.species))}`
      : `→ ${ui().solver.unknownMon(a.pos ?? 0)}`;
  }
  if (a.kind === "team" && a.slots) return a.slots.filter(Boolean).join("-");
  return a.input;
}

/** A PS choice string ("move surf" / "switch 2") for the line, where no
 * structured action is available. */
function inputLabel(input: string): string {
  const m = input.match(/^move (.+)$/);
  return m ? moveName(toId(m[1])) : input;
}
