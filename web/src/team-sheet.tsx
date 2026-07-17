// UI-2 open-team-sheet surfaces: per-mon expandable set panels and the
// in-battle "Team sheets" modal body. Everything renders from the STATIC
// team JSON the client already holds (human team = the selected sets, bot
// team = its pool entry); battle overlays are limited to protocol-public
// facts — revealed / active / fainted for the foe (pick membership itself
// stays hidden until a mon appears; the reveal set is tracked from what
// the battle screen already shows), own picks for the player's own side.
//
// Structure is deliberately semantic (UI-3 tooltips / UI-4 screen readers
// attach here): each mon is a heading button (aria-expanded) over <ul>
// rows; moves are addressable via data-move, mons via data-mon.

import { useState } from "preact/hooks";
import type { PokeView, SideView, StateView } from "./types";
import { TypeBadge } from "./battle-ui";
import { itemName, moveName, speciesName, toId, ui } from "./i18n";
import { itemNote, moveNote } from "./behavior-notes";
import { noteRef } from "./tooltip";
import {
  hiddenPowerType,
  moveMeta,
  sheetMon,
  speciesTypes,
  type SheetMon,
} from "./set-info";

/** Battle-public marks for one sheet row. */
export interface MonMarks {
  picked?: boolean; // own side only — the player knows their picks
  revealed?: boolean; // foe side only — has publicly appeared
  active?: boolean;
  fainted?: boolean;
}

function genderMark(g: string): string {
  return g === "M" ? "♂" : g === "F" ? "♀" : "—";
}

/** Full set detail: item / gender / Hidden Power type rows, then the
 * moves with type, category and base power from the dex data. */
export function SetDetail(props: { mon: SheetMon }) {
  const m = props.mon;
  const hasHp = m.moves.some((mv) => toId(mv).startsWith("hiddenpower"));
  const iNote = m.item ? itemNote(m.item) : null;
  return (
    <div class="set-detail">
      <ul class="set-fields">
        <li class="set-field">
          <span class="set-field-k">{ui().sheetItem}</span>
          <span
            class={`set-field-v${iNote ? " has-note" : ""}`}
            data-item={m.item ? toId(m.item) : undefined}
            tabIndex={iNote ? 0 : undefined}
          >
            {m.item ? itemName(m.item) : ui().sheetNoItem}
            {iNote && (
              <span class="bn-note" ref={noteRef}>
                {iNote}
              </span>
            )}
          </span>
        </li>
        <li class="set-field">
          <span class="set-field-k">{ui().sheetGender}</span>
          <span class="set-field-v">{genderMark(m.gender)}</span>
        </li>
        {hasHp && m.ivs && (
          <li class="set-field">
            <span class="set-field-k">{ui().sheetHp}</span>
            <span class="set-field-v">
              <TypeBadge type={hiddenPowerType(m.ivs)} />
            </span>
          </li>
        )}
      </ul>
      <ul class="set-moves">
        {m.moves.map((mv) => {
          const meta = moveMeta(mv, m.ivs);
          const note = moveNote(mv);
          return (
            <li
              class={`set-move${note ? " has-note" : ""}`}
              key={mv}
              data-move={toId(mv)}
              tabIndex={note ? 0 : undefined}
            >
              <span class="move-name">{moveName(mv)}</span>
              {meta && (
                <span class="move-meta">
                  <TypeBadge type={meta.type} />
                  <span class="move-cat">{ui().moveCat(meta.category)}</span>
                  {meta.basePower > 0 && (
                    <span class="move-bp">{ui().bp(meta.basePower)}</span>
                  )}
                </span>
              )}
              {note && (
                <span class="bn-note" ref={noteRef}>
                  {note}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** One expandable sheet row: species head (tap to open the full set) plus
 * optional battle-public marks. Several rows may be open at once. */
export function MonSheet(props: { mon: SheetMon; marks?: MonMarks }) {
  const [open, setOpen] = useState(false);
  const m = props.mon;
  const types = speciesTypes(m.species);
  const marks = props.marks ?? {};
  return (
    <div
      class={`mon-sheet ${open ? "open" : ""} ${marks.fainted ? "is-fainted" : ""}`}
    >
      <button
        class="mon-sheet-head"
        aria-expanded={open}
        data-mon={toId(m.species)}
        onClick={() => setOpen(!open)}
      >
        <span class="mon-name">{speciesName(m.species)}</span>
        <span class="mon-level">L{m.level}</span>
        {types?.map((t) => (
          <TypeBadge type={t} key={t} />
        ))}
        <span class="sheet-marks">
          {marks.picked && (
            <span class="sheet-mark picked">{ui().markPicked}</span>
          )}
          {marks.revealed && (
            <span class="sheet-mark revealed">{ui().markRevealed}</span>
          )}
          {marks.active && (
            <span class="sheet-mark active">{ui().markActive}</span>
          )}
          {marks.fainted && (
            <span class="sheet-mark fainted">{ui().markFainted}</span>
          )}
        </span>
        <span class="sheet-chevron" aria-hidden="true">
          &#9656;
        </span>
      </button>
      {open && <SetDetail mon={m} />}
    </div>
  );
}

function findInParty(side: SideView, mon: SheetMon): PokeView | null {
  return side.party.find((p) => toId(p.species) === toId(mon.species)) ?? null;
}

function isActive(side: SideView, entry: PokeView | null): boolean {
  // A fainted mon can still be the "active" slot while its replacement is
  // being chosen — show it as fainted only.
  return (
    entry !== null && !entry.fainted && side.party.indexOf(entry) === side.active
  );
}

/** "Team sheets" modal body: both sides' full 6-mon open sheets with
 * battle-public marks overlaid. */
export function TeamSheets(props: {
  mineId: string;
  mineSets: unknown[];
  foeId: string;
  foeSets: unknown[];
  view: StateView;
  /** Foe species ids that have publicly appeared (tracked from the same
   * active-mon state the battle screen renders). */
  revealedFoe: ReadonlySet<string>;
}) {
  const [mineView, foeView] = props.view.sides;
  return (
    <div class="team-sheets">
      <p class="modal-note">{ui().sheetNote}</p>
      <section>
        <h3>{ui().yourTeam(props.mineId)}</h3>
        <ul class="sheet-list">
          {props.mineSets.map((s, i) => {
            const mon = sheetMon(s);
            const entry = findInParty(mineView, mon);
            return (
              <li key={i}>
                <MonSheet
                  mon={mon}
                  marks={{
                    picked: entry !== null,
                    active: isActive(mineView, entry),
                    fainted: entry?.fainted ?? false,
                  }}
                />
              </li>
            );
          })}
        </ul>
      </section>
      <section>
        <h3>{ui().foeTeam(props.foeId)}</h3>
        <ul class="sheet-list">
          {props.foeSets.map((s, i) => {
            const mon = sheetMon(s);
            const entry = findInParty(foeView, mon);
            // Public information only: a foe mon carries battle marks
            // solely once it has appeared (or publicly fainted) — mere
            // pick membership (entry existence) must not leak.
            const revealed =
              props.revealedFoe.has(toId(mon.species)) ||
              (entry?.fainted ?? false);
            return (
              <li key={i}>
                <MonSheet
                  mon={mon}
                  marks={
                    revealed
                      ? {
                          revealed: true,
                          active: isActive(foeView, entry),
                          fainted: entry?.fainted ?? false,
                        }
                      : {}
                  }
                />
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}
