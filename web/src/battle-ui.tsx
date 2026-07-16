// Presentational pieces for the battle screen. Display semantics mirror
// the CLI panel in crates/bot/examples/play.rs: foe HP as %, own HP exact,
// boosts as signed chips, bench with exact HP, plus field conditions.

import type { PokeView } from "./types";

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

const COND_NAMES: Record<string, string> = {
  raindance: "Rain",
  sunnyday: "Sun",
  sandstorm: "Sandstorm",
  reflect: "Reflect",
  lightscreen: "Light Screen",
  safeguard: "Safeguard",
  spikes: "Spikes",
  mist: "Mist",
  confusion: "Confused",
  substitute: "Substitute",
  leechseed: "Leech Seed",
  curse: "Curse",
  encore: "Encore",
  attract: "Attract",
  nightmare: "Nightmare",
  partiallytrapped: "Trapped",
  meanlook: "Mean Look",
  focusenergy: "Focus Energy",
  lockedmove: "Locked",
  mustrecharge: "Recharging",
  perishsong: "Perish Song",
  disable: "Disable",
  foresight: "Foresight",
  destinybond: "Destiny Bond",
};

export function condName(key: string): string {
  return COND_NAMES[key] ?? key;
}

const BOOST_LABELS: Record<string, string> = {
  atk: "Atk",
  def: "Def",
  spa: "SpA",
  spd: "SpD",
  spe: "Spe",
  accuracy: "Acc",
  evasion: "Eva",
};

export function hpPct(p: PokeView): number {
  return p.maxhp > 0 ? Math.max(0, (p.hp / p.maxhp) * 100) : 0;
}

/** Rounded % with the CLI's floor-at-1 rule for a living mon. */
export function hpPctLabel(p: PokeView): string {
  if (p.hp <= 0) return "fnt";
  return `${Math.max(1, Math.round(hpPct(p)))}%`;
}

export function HpBar(props: { pct: number }) {
  const cls =
    props.pct > 50 ? "hp-high" : props.pct > 20 ? "hp-mid" : "hp-low";
  return (
    <div class="hp-bar">
      <div class={`hp-fill ${cls}`} style={{ width: `${props.pct}%` }} />
    </div>
  );
}

export function StatusBadge(props: { status: string }) {
  if (!props.status) return null;
  return <span class={`status-badge st-${props.status}`}>{props.status}</span>;
}

export function TypeBadge(props: { type: string }) {
  return (
    <span
      class="type-badge"
      style={{ background: TYPE_COLORS[props.type] ?? "#777" }}
    >
      {props.type}
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
          {BOOST_LABELS[k] ?? k}
          {v > 0 ? `+${v}` : v}
        </span>
      ))}
    </span>
  );
}

/** Active mon card. `mine`: exact HP; foe: % only (play.rs semantics). */
export function ActiveCard(props: {
  poke: PokeView;
  mine: boolean;
  extra?: string;
}) {
  const p = props.poke;
  const pct = hpPct(p);
  return (
    <div class={`active-card ${props.mine ? "mine" : "foe"}`}>
      <div class="active-head">
        <span class="mon-name">
          {props.mine ? "" : "Foe "}
          {p.name}
        </span>
        <span class="mon-level">L{p.level}</span>
        {p.types.map((t) => (
          <TypeBadge type={t} key={t} />
        ))}
        {props.extra && <span class="active-extra">{props.extra}</span>}
      </div>
      <div class="active-hp">
        <HpBar pct={pct} />
        <span class="hp-num">
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
    bits.push({ cls: "foe-cond", text: `Foe: ${condName(c)}` });
  for (const c of props.mineConds)
    bits.push({ cls: "mine-cond", text: `You: ${condName(c)}` });
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
