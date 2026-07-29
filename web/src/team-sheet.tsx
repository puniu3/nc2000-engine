// UI-2 open-team-sheet surfaces: per-mon expandable set panels and the
// in-battle "Team sheets" modal body. Everything renders from the STATIC
// team JSON the client already holds (human team = the selected sets, bot
// team = its pool entry); battle overlays are limited to protocol-public
// facts — revealed / active / fainted for the foe (pick membership itself
// stays hidden until a mon appears; the reveal set is tracked from what
// the battle screen already shows), own picks for the player's own side.
//
// M18 blind mode: the same components render the foe side with the set
// body CUT OFF — `MonSheet hideDetail` makes the row inert (no toggle, no
// SetDetail branch at all, so item / gender / DVs / moves have no path
// into the DOM) and `TeamSheets blind` swaps the note and the foe heading
// (which would otherwise print the pool team's id — an identification
// leak by itself). Only species / level / types and the battle-public
// marks survive, which is exactly what the bot gets from the human.
//
// Structure is deliberately semantic (UI-3 tooltips / UI-4 screen readers
// attach here): each mon is a heading button (aria-expanded) over <ul>
// rows; moves are addressable via data-move, mons via data-mon.

import { useState } from "preact/hooks";
import type { PokeView, SideView, StateView } from "./types";
import { Lvl, TypeBadge } from "./battle-ui";
import { itemName, moveName, speciesName, toId, typeName, ui } from "./i18n";
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
          <span class="set-field-v">
            <span aria-hidden="true">{genderMark(m.gender)}</span>
            <span class="sr-only">{ui().srGender(m.gender)}</span>
          </span>
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

/** Own-side preview pick affordance (UI-7): with `pick` set the row
 * renders TWO sibling buttons — the main pick button (species / level /
 * types + pick-state chips, aria-pressed, aria-disabled at the level cap)
 * and a compact expand toggle (aria-expanded chevron) opening the same
 * SetDetail the foe rows use. Strings and handlers come from the caller;
 * this component stays pure presentation. */
export interface PickProps {
  /** Complete pick-control name: species, level, types, pick state. */
  label: string;
  pressed: boolean;
  /** Pick-order chip text ("Lead" / "2" / "3"); null while unpicked. */
  badge: string | null;
  /** No legal completion under the level cap: aria-disabled + cap chip. */
  overCap: boolean;
  /** Compact visible chip ("Over 155") — must fit the row's one line so
   * the cap state never resizes the row. */
  capChip: string;
  /** Full cap sentence, screen readers only (the describedby target). */
  capNote: string;
  capNoteId: string;
  /** Expand-toggle name ("Snorlax — details"). */
  detailsLabel: string;
  onPick: () => void;
}

/** One expandable sheet row: species head (tap to open the full set) plus
 * optional battle-public marks, or — with `pick` — the own-preview
 * pick+expand sibling pair. Several rows may be open at once.
 *
 * `hideDetail` (blind mode, foe side only) drops the whole set body: the
 * head becomes a non-interactive div and SetDetail is never instantiated.
 * It is never combined with `pick` — the player's own side always keeps
 * its detail. */
export function MonSheet(props: {
  mon: SheetMon;
  marks?: MonMarks;
  pick?: PickProps;
  hideDetail?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const m = props.mon;
  const types = speciesTypes(m.species);
  const marks = props.marks ?? {};
  const pick = props.pick;
  const hideDetail = props.hideDetail === true;
  // Screen-reader row name: species, spoken level, types, then the
  // battle-public marks (選出/出場中/判明/ひんし) as plain words.
  const label = [
    speciesName(m.species),
    ui().srLevel(m.level),
    ...(types ? [types.map(typeName).join("/")] : []),
    ...(marks.picked ? [ui().markPicked] : []),
    ...(marks.revealed ? [ui().markRevealed] : []),
    ...(marks.active ? [ui().markActive] : []),
    ...(marks.fainted ? [ui().markFainted] : []),
  ].join(", ");
  const headContent = (
    <>
      <span class="mon-name">{speciesName(m.species)}</span>
      <span class="mon-level">L{m.level}</span>
      {types?.map((t) => (
        <TypeBadge type={t} key={t} />
      ))}
    </>
  );
  // Battle-public marks — identical in every head variant (the pick head
  // carries its own pick-order/cap chips instead).
  const markChips = (
    <span class="sheet-marks">
      {marks.picked && <span class="sheet-mark picked">{ui().markPicked}</span>}
      {marks.revealed && (
        <span class="sheet-mark revealed">{ui().markRevealed}</span>
      )}
      {marks.active && <span class="sheet-mark active">{ui().markActive}</span>}
      {marks.fainted && (
        <span class="sheet-mark fainted">{ui().markFainted}</span>
      )}
    </span>
  );
  return (
    <div
      class={`mon-sheet ${open ? "open" : ""} ${marks.fainted ? "is-fainted" : ""}${
        pick?.pressed ? " is-picked" : ""
      }${pick?.overCap ? " over-cap" : ""}`}
    >
      {pick ? (
        // Pick + expand are SIBLINGS in one row — the expand toggle (and
        // the focusable toggletip anchors inside SetDetail) must never
        // nest inside the pick button.
        <div class="mon-sheet-row">
          <button
            class="mon-sheet-head pick-head"
            // aria-disabled (not disabled): the button stays focusable so
            // the cap reason stays reachable; the click is a no-op via
            // the caller's cap guard.
            aria-disabled={pick.overCap}
            aria-pressed={pick.pressed}
            aria-label={pick.label}
            aria-describedby={pick.overCap ? pick.capNoteId : undefined}
            data-mon={toId(m.species)}
            onClick={pick.onPick}
          >
            {headContent}
            <span class="sheet-marks">
              {pick.badge !== null && (
                <span class="sheet-mark pick-order">{pick.badge}</span>
              )}
              {pick.overCap && (
                <span class="sheet-mark cap">
                  <span aria-hidden="true">{pick.capChip}</span>
                  <span class="sr-only" id={pick.capNoteId}>
                    {pick.capNote}
                  </span>
                </span>
              )}
            </span>
          </button>
          <button
            class="sheet-expand"
            aria-expanded={open}
            aria-label={pick.detailsLabel}
            data-expand={toId(m.species)}
            onClick={() => setOpen(!open)}
          >
            <span class="sheet-chevron" aria-hidden="true">
              &#9656;
            </span>
          </button>
        </div>
      ) : hideDetail ? (
        // Blind foe row: an INERT head — not a button, no aria-expanded,
        // no chevron, nothing to open. The row's spoken form is its own
        // content (aria-label has no effect on a generic <div>), so the
        // level uses <Lvl> to stay speakable where the button variants
        // rely on their composed aria-label.
        <div class="mon-sheet-head static" data-mon={toId(m.species)}>
          <span class="mon-name">{speciesName(m.species)}</span>
          <span class="mon-level">
            <Lvl n={m.level} />
          </span>
          {types?.map((t) => (
            <TypeBadge type={t} key={t} />
          ))}
          {markChips}
        </div>
      ) : (
        <button
          class="mon-sheet-head"
          aria-expanded={open}
          aria-label={label}
          data-mon={toId(m.species)}
          onClick={() => setOpen(!open)}
        >
          {headContent}
          {markChips}
          <span class="sheet-chevron" aria-hidden="true">
            &#9656;
          </span>
        </button>
      )}
      {/* Structural, not merely visual: with hideDetail there is no code
          path that instantiates SetDetail, so the hidden set cannot reach
          the DOM even if `open` were somehow set. */}
      {open && !hideDetail && <SetDetail mon={m} />}
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
    entry !== null &&
    !entry.fainted &&
    side.party.indexOf(entry) === side.active
  );
}

/** "Team sheets" modal body: both sides' full 6-mon open sheets with
 * battle-public marks overlaid. With `blind` the foe half degrades to
 * species / level / types + marks and its heading drops the team id; the
 * player's own half is untouched in both modes. The caller turns `blind`
 * off once the game is decided — the post-game reveal is the same modal
 * with nothing withheld. */
export function TeamSheets(props: {
  mineId: string;
  mineSets: unknown[];
  foeId: string;
  foeSets: unknown[];
  view: StateView;
  /** Foe species ids that have publicly appeared (tracked from the same
   * active-mon state the battle screen renders). */
  revealedFoe: ReadonlySet<string>;
  blind?: boolean;
}) {
  const [mineView, foeView] = props.view.sides;
  const blind = props.blind === true;
  return (
    <div class="team-sheets">
      <p class="modal-note">{blind ? ui().sheetNoteBlind : ui().sheetNote}</p>
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
        {/* The team id names the pool entry — under blind it identifies
            the whole set list, so the heading loses it too. */}
        <h3>{blind ? ui().foeTeamBlind : ui().foeTeam(props.foeId)}</h3>
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
                  hideDetail={blind}
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
