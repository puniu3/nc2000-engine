// Presentational pieces for the battle screen. Display semantics mirror
// the CLI panel in crates/bot/examples/play.rs: foe HP as %, own HP exact,
// boosts as signed chips, plus field conditions. (UI-6: the own-bench row
// is gone — switch buttons already carry the same mons.)
//
// M13: all display names route through i18n (species/types via the JP
// tables, statuses/conditions/boost labels via the hand tables); status
// badge classes stay keyed by the raw status code.

import type { PokeView } from "./types";
import {
  boostLabel,
  condName,
  itemName,
  speciesName,
  statLongName,
  statusLongName,
  statusName,
  toId,
  typeName,
  ui,
} from "./i18n";
import { itemNote } from "./behavior-notes";
import { noteRef } from "./tooltip";

export const TYPE_COLORS: Record<string, string> = {
  Normal: "#9fa19f",
  Fire: "#e35d34",
  Water: "#4a7fd4",
  Electric: "#d8b430",
  Grass: "#5fa93f",
  Ice: "#5fc0c9",
  Fighting: "#b5544a",
  Poison: "#9c56a5",
  Ground: "#ba9c58",
  Flying: "#7e9ad4",
  Psychic: "#d4608f",
  Bug: "#95a234",
  Rock: "#a9924d",
  Ghost: "#6a5aa0",
  Dragon: "#6f61c2",
  Dark: "#5e5449",
  Steel: "#8f8fa3",
};

export function hpPct(p: PokeView): number {
  return p.maxhp > 0 ? Math.max(0, (p.hp / p.maxhp) * 100) : 0;
}

/** Rounded % with the CLI's floor-at-1 rule for a living mon. */
export function hpPctLabel(p: PokeView): string {
  if (p.hp <= 0) return ui().fnt;
  return `${Math.max(1, Math.round(hpPct(p)))}%`;
}

/** "L52" glyph with a spoken "Level 52" equivalent — drop into any level
 * wrapper (span.mon-level, small, …); visuals unchanged. */
export function Lvl(props: { n: number }) {
  return (
    <>
      <span aria-hidden="true">L{props.n}</span>
      <span class="sr-only">{ui().srLevel(props.n)}</span>
    </>
  );
}

/** Purely decorative — the owning control/summary carries the HP as text. */
export function HpBar(props: { pct: number }) {
  const cls =
    props.pct > 50 ? "hp-high" : props.pct > 20 ? "hp-mid" : "hp-low";
  return (
    <div class="hp-bar" aria-hidden="true">
      <div class={`hp-fill ${cls}`} style={{ width: `${props.pct}%` }} />
    </div>
  );
}

export function StatusBadge(props: { status: string }) {
  if (!props.status) return null;
  // The compact badge code ("par") stays visual; screen readers get the
  // expanded word ("paralyzed" / まひ).
  return (
    <span class={`status-badge st-${props.status}`}>
      <span aria-hidden="true">{statusName(props.status)}</span>
      <span class="sr-only">{statusLongName(props.status)}</span>
    </span>
  );
}

export function TypeBadge(props: { type: string }) {
  return (
    <span
      class="type-badge"
      style={{ background: TYPE_COLORS[props.type] ?? "#777" }}
    >
      {typeName(props.type)}
    </span>
  );
}

/** Current held item — public for both sides under the open team sheet
 * (initial sets are public; every item change is protocol-public: berry
 * consumption, Thief, …). `item` is the CURRENT item from stateView;
 * `initial` is the set's starting item, distinguishing "had one, now
 * gone" (struck-through name) from "never held" (quiet em-dash). A UI-3
 * behavior-note toggletip attaches when the held item has one; `note:
 * false` renders a plain chip (inside switch buttons, where the composed
 * aria-label carries the item and a nested tab stop would be noise). */
export function ItemChip(props: {
  item: string | null;
  initial?: string | null;
  note?: boolean;
}) {
  const it = props.item;
  const note = props.note !== false && it ? itemNote(it) : null;
  return (
    <span
      class={`item-chip${it ? "" : " none"}${note ? " has-note" : ""}`}
      data-item={it ? toId(it) : undefined}
      tabIndex={note ? 0 : undefined}
    >
      {it ? (
        <>
          <span class="sr-only">{ui().sheetItem} </span>
          {itemName(it)}
        </>
      ) : props.initial ? (
        <>
          <span class="item-gone" aria-hidden="true">
            {itemName(props.initial)}
          </span>
          <span class="sr-only">{ui().srItemGone(itemName(props.initial))}</span>
        </>
      ) : (
        <>
          <span aria-hidden="true">—</span>
          <span class="sr-only">{ui().srNoItem}</span>
        </>
      )}
      {note && (
        <span class="bn-note" ref={noteRef}>
          {note}
        </span>
      )}
    </span>
  );
}

function BoostChips(props: { boosts: Record<string, number> }) {
  const chips = Object.entries(props.boosts).filter(([, v]) => v !== 0);
  if (chips.length === 0) return null;
  return (
    <span class="boost-chips">
      {chips.map(([k, v]) => (
        <span class={`boost-chip ${v > 0 ? "up" : "down"}`} key={k}>
          <span aria-hidden="true">
            {boostLabel(k)}
            {v > 0 ? `+${v}` : v}
          </span>
          <span class="sr-only">
            {statLongName(k)} {v > 0 ? `+${v}` : v}
          </span>
        </span>
      ))}
    </span>
  );
}

/** Active mon card. `mine`: exact HP; foe: % only (play.rs semantics).
 * `initialItem`: the set's starting item (see ItemChip). */
export function ActiveCard(props: {
  poke: PokeView;
  mine: boolean;
  extra?: string;
  initialItem?: string | null;
}) {
  const p = props.poke;
  const pct = hpPct(p);
  return (
    <div
      class={`active-card ${props.mine ? "mine" : "foe"}`}
      role="group"
      aria-label={props.mine ? ui().srYourActive : ui().srFoeActive}
    >
      <div class="active-head">
        <span class="mon-name">
          {props.mine ? "" : ui().foePrefix}
          {speciesName(p.name)}
        </span>
        <span class="mon-level">
          <Lvl n={p.level} />
        </span>
        {p.types.map((t) => (
          <TypeBadge type={t} key={t} />
        ))}
        <ItemChip item={p.item} initial={props.initialItem} />
        {props.extra && <span class="active-extra">{props.extra}</span>}
      </div>
      <div class="active-hp">
        <HpBar pct={pct} />
        <span class="hp-num">
          <span class="sr-only">HP </span>
          {props.mine ? `${p.hp}/${p.maxhp}` : hpPctLabel(p)}
        </span>
        <StatusBadge status={p.status} />
      </div>
      <div class="active-tags">
        <BoostChips boosts={p.boosts} />
        {p.volatiles.map((v) => (
          <span class="volatile-chip" key={v}>
            {condName(v)}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Format-rule pseudo-weathers: always on, zero information — hide them. */
const RULE_CONDS = new Set([
  "maxtotallevel",
  "stadiumsleepclause",
  "freezeclausemod",
  "sleepclausemod",
  "endlessbattleclause",
]);

/** Field strip: weather + per-side conditions. */
export function FieldStrip(props: {
  weather: string | null;
  pseudo: string[];
  mineConds: string[];
  foeConds: string[];
}) {
  const bits: { cls: string; text: string }[] = [];
  if (props.weather)
    bits.push({ cls: "weather", text: condName(props.weather) });
  for (const p of props.pseudo)
    if (!RULE_CONDS.has(p)) bits.push({ cls: "pseudo", text: condName(p) });
  for (const c of props.foeConds)
    bits.push({ cls: "foe-cond", text: ui().fieldFoe(condName(c)) });
  for (const c of props.mineConds)
    bits.push({ cls: "mine-cond", text: ui().fieldYou(condName(c)) });
  if (bits.length === 0) return null;
  return (
    <div class="field-strip">
      {bits.map((b, i) => (
        <span class={`field-chip ${b.cls}`} key={i}>
          {b.text}
        </span>
      ))}
    </div>
  );
}
