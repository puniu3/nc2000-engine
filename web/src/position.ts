// The solver's position document — the JSON contract with
// `crates/bot/src/position.rs`, which is the source of truth for every field
// name and default here.
//
// The screen edits a `PositionSpec` directly rather than keeping a second
// "form model" beside it. A form model would need a mapping in both
// directions, and every field that drifted between the two would be a
// position analyzed as something other than what was typed — the one bug
// this feature cannot ship. The cost is that a few fields are wider than the
// UI ever sets (`transformed_into`, `mimic_overlay`, the flow flags); they
// default to "start of turn, nothing pending" and are simply never touched.

import { fixedGender } from "./set-info";
import type { PoolTeam } from "./types";

export const POSITION_SCHEMA = "nc2000-position-v1";
/** NC2000 brings three of six. */
export const PICKS = 3;

export interface PpSpec {
  move: string;
  pp: number;
  maxpp: number;
  disabled?: boolean;
}

export interface UseSpec {
  move: string;
  n: number;
}

export interface VolSpec {
  key: string;
  start_turn?: number;
  move?: string | null;
  source?: [number, number] | null;
  counter?: number | null;
}

export interface ItemKnowledge {
  known: boolean;
  item?: string | null;
}

export interface MonSpec {
  species: string;
  level: number;
  gender?: string;
  name?: string;
  item_flag?: boolean;
  appeared?: boolean;
  appear_count?: number;
  switch_in_turn?: number;
  active?: boolean;
  fainted?: boolean;
  /** Announced HP as `hp_num` / `hp_den`; 100 = the percentage stream. */
  hp_num?: number;
  hp_den?: number;
  /** Own side: exact current HP (the request carries this). */
  hp_exact?: number | null;
  status?: string;
  rest?: boolean;
  slept?: number;
  tox_counter?: number | null;
  comp_brn?: boolean;
  comp_par?: boolean;
  /** atk, def, spa, spd, spe, accuracy, evasion. */
  boosts?: [number, number, number, number, number, number, number];
  volatiles?: VolSpec[];
  uses?: UseSpec[];
  pp?: PpSpec[];
  item_now?: string | null;
  locked?: UseSpec | null;
  charging?: string | null;
  must_recharge?: boolean;
  transformed_into?: [number, number] | null;
  mimic_overlay?: string | null;
  last_move?: string | null;
  stall_streak?: number;
  last_protect_turn?: number;
  protected_this_turn?: boolean;
  // opponent knowledge
  revealed_moves?: string[];
  item_original?: ItemKnowledge;
  item_current?: ItemKnowledge;
  item_gained?: boolean;
}

export interface CondSpec {
  key: string;
  start_turn?: number;
}

export interface SideSpec {
  mons: MonSpec[];
  active?: number | null;
  party?: number[];
  conditions?: CondSpec[];
  pending_bp?: boolean;
  acted_this_turn?: boolean;
  fainted_this_turn?: number | null;
  fainted_last_turn?: number | null;
  last_move?: string | null;
}

export interface PositionSpec {
  schema: string;
  side: number;
  turn: number;
  upkeep_this_turn?: boolean;
  own_sets: unknown[];
  sides: [SideSpec, SideSpec];
  weather?: { key: string; upkeeps?: number } | null;
  team_preview?: boolean;
  force_switch?: boolean;
  trapped?: boolean;
}

// ------------------------------------------------------- analysis report

/** `crates/bot/src/analysis.rs` — one report per solved position. */
export interface AnalysisReport {
  schema: string;
  side: number;
  turn: number;
  iterations: number;
  preview: boolean;
  belief: { count: number; fallback: boolean; candidates: string[] } | null;
  actions: ActionRow[];
  matrix: { cols: ActionRef[]; cells: (MatrixCell | null)[][] };
  damage: { mine: DamageRow[]; theirs: DamageRow[] };
  line: PrincipalLine | null;
}

export interface ActionRef {
  input: string;
  kind?: "move" | "switch" | "team" | "pass";
  move?: string;
  pos?: number;
  /** `null` when the target is a mon the opponent has never shown — its
   * identity is imputed, and naming it would present a guess as a fact. */
  species?: string | null;
  slots?: number[];
}

export interface ActionRow extends ActionRef {
  visits: number;
  mean: number;
  frac: number;
  dominated: boolean;
  /** Why the search proved this action pointless, when it did. */
  reason: string | null;
}

export interface MatrixCell {
  n: number;
  mean: number;
}

export interface DamageRow {
  move: string;
  /** The move was publicly revealed (theirs) or is simply ours. */
  revealed: boolean;
  min: number;
  max: number;
  crit: number;
  /** `null` = never misses, which is not the same as 100. */
  accuracy: number | null;
  hp: number;
  maxhp: number;
  hitsGuaranteed: number | null;
  hitsBest: number | null;
  ko: "always" | "possible" | "never";
}

export interface PrincipalLine {
  assumed: { slot: number; species: string; moves: string[]; appeared: boolean }[];
  steps: {
    mine: string | null;
    theirs: string | null;
    log: string[];
    outcome: "p1" | "p2" | "tie" | null;
  }[];
}

// ------------------------------------------------------------- authoring

/** A set as the pool and the PS importer hand it over. */
interface SetLike {
  species: string;
  level?: number;
  gender?: string;
  name?: string;
  item?: string;
}

const FULL: MonSpec = { species: "", level: 50, hp_num: 100, hp_den: 100 };

export function blankMon(species = "", level = 50, gender = ""): MonSpec {
  // A species with only one possible gender needs no asking; one that admits
  // both keeps whatever the caller knew (the sets carry it; a hand-typed
  // roster gets a control).
  return { ...FULL, species, level, gender: gender || fixedGender(species) || "M" };
}

function blankSide(): SideSpec {
  return { mons: [], active: null, conditions: [] };
}

/** An empty position: nobody entered, turn 1, us on p1. */
export function blankPosition(): PositionSpec {
  return {
    schema: POSITION_SCHEMA,
    side: 0,
    turn: 1,
    own_sets: [],
    sides: [blankSide(), blankSide()],
    weather: null,
    team_preview: false,
    force_switch: false,
    trapped: false,
  };
}

/** Board state that survives a team change, keyed by species so swapping one
 * slot does not silently move another mon's HP onto a different Pokémon. */
function carryOver(prev: MonSpec[], species: string, level: number): Partial<MonSpec> {
  const hit = prev.find((m) => m.species === species && m.level === level);
  if (!hit) return {};
  const {
    species: _s,
    level: _l,
    gender: _g,
    ...rest
  } = hit;
  return rest;
}

/** Install our own six sets: they define both the private truth (`own_sets`)
 * and the public half of our roster. */
export function withOwnTeam(spec: PositionSpec, sets: SetLike[]): PositionSpec {
  const me = spec.side;
  const prev = spec.sides[me].mons;
  const mons = sets.map((s) =>
    Object.assign(
      blankMon(s.species, s.level ?? 50, s.gender ?? ""),
      carryOver(prev, s.species, s.level ?? 50),
      { item_flag: !!s.item && s.item !== "" },
    ),
  );
  const sides: [SideSpec, SideSpec] = [spec.sides[0], spec.sides[1]];
  sides[me] = { ...sides[me], mons, active: reactive(mons, sides[me].active) };
  return { ...spec, own_sets: sets as unknown[], sides };
}

/** Install the opponent's six public identities (species / level / item
 * flag). Their sets are exactly what we do NOT get to know. */
export function withFoeRoster(spec: PositionSpec, mons: SetLike[]): PositionSpec {
  const foe = 1 - spec.side;
  const prev = spec.sides[foe].mons;
  const next = mons.map((s) =>
    Object.assign(
      blankMon(s.species, s.level ?? 50, s.gender ?? ""),
      carryOver(prev, s.species, s.level ?? 50),
      { item_flag: s.item === undefined ? true : !!s.item },
    ),
  );
  const sides: [SideSpec, SideSpec] = [spec.sides[0], spec.sides[1]];
  sides[foe] = { ...sides[foe], mons: next, active: reactive(next, sides[foe].active) };
  return { ...spec, sides };
}

/** Keep the active pointer and the per-mon flag in agreement — the Rust side
 * refuses a position where they disagree, and rightly so. */
function reactive(mons: MonSpec[], active: number | null | undefined): number | null {
  const flagged = mons.findIndex((m) => m.active);
  if (flagged >= 0) return flagged;
  if (active != null && active < mons.length && !mons[active].fainted) {
    mons[active] = { ...mons[active], active: true };
    return active;
  }
  return null;
}

/** The PUBLIC half of a pool team, for the "they brought this rental"
 * shortcut: species, level, and whether each mon holds an item — exactly
 * what the `|poke|` preview lines say, and no more. Copying the sets here
 * would hand the solver the one thing it is supposed to be guessing.
 *
 * The item flag is not decoration: the belief filters candidates on it, so a
 * roster that claims an item every mon does not hold matches no known team
 * and drops the solver into set-by-set imputation. */
export function poolRoster(team: PoolTeam): SetLike[] {
  const sets = team.sets as { item?: string; gender?: string }[];
  return team.species.map((species, i) => ({
    species,
    level: team.levels[i] ?? 50,
    item: sets?.[i]?.item ?? "",
    // Public at preview, and matched on by the belief — a roster that gets
    // it wrong identifies nothing.
    gender: sets?.[i]?.gender ?? "",
  }));
}

export function setMon(
  spec: PositionSpec,
  side: number,
  slot: number,
  patch: Partial<MonSpec>,
): PositionSpec {
  const sides: [SideSpec, SideSpec] = [spec.sides[0], spec.sides[1]];
  const mons = sides[side].mons.slice();
  mons[slot] = { ...mons[slot], ...patch };
  sides[side] = { ...sides[side], mons };
  return { ...spec, sides };
}

/** Move the active pointer, clearing the previous holder's flag. Exactly one
 * mon per side may be active. */
export function setActive(spec: PositionSpec, side: number, slot: number | null): PositionSpec {
  const sides: [SideSpec, SideSpec] = [spec.sides[0], spec.sides[1]];
  const mons = sides[side].mons.map((m, i) => ({ ...m, active: i === slot }));
  if (slot != null) mons[slot] = { ...mons[slot], appeared: true };
  sides[side] = { ...sides[side], mons, active: slot };
  return { ...spec, sides };
}

export function setCondition(
  spec: PositionSpec,
  side: number,
  key: string,
  on: boolean,
): PositionSpec {
  const sides: [SideSpec, SideSpec] = [spec.sides[0], spec.sides[1]];
  const conds = (sides[side].conditions ?? []).filter((c) => c.key !== key);
  if (on) conds.push({ key, start_turn: Math.max(1, spec.turn) });
  sides[side] = { ...sides[side], conditions: conds };
  return { ...spec, sides };
}

export function hasCondition(spec: PositionSpec, side: number, key: string): boolean {
  return (spec.sides[side].conditions ?? []).some((c) => c.key === key);
}

export function setVolatile(
  spec: PositionSpec,
  side: number,
  slot: number,
  key: string,
  on: boolean,
): PositionSpec {
  const mon = spec.sides[side].mons[slot];
  const vols = (mon.volatiles ?? []).filter((v) => v.key !== key);
  if (on) vols.push({ key, start_turn: Math.max(1, spec.turn) });
  return setMon(spec, side, slot, { volatiles: vols });
}

export function hasVolatile(mon: MonSpec, key: string): boolean {
  return (mon.volatiles ?? []).some((v) => v.key === key);
}

/** Everything the Rust validator would reject, checked here so the screen
 * can point at the field instead of showing a thrown message. Returns [] when
 * the position is ready to solve. */
export function positionProblems(spec: PositionSpec): string[] {
  const out: string[] = [];
  const me = spec.side;
  const foe = 1 - me;
  if (spec.own_sets.length === 0) out.push("own-team");
  if (spec.sides[me].mons.length === 0) out.push("own-team");
  if (spec.sides[foe].mons.length === 0) out.push("foe-team");
  if (!spec.team_preview) {
    if (spec.sides[me].active == null && !spec.force_switch) out.push("own-active");
    if (spec.sides[foe].active == null) out.push("foe-active");
  }
  for (const side of [0, 1]) {
    for (const m of spec.sides[side].mons) {
      const den = m.hp_den ?? 100;
      const num = m.hp_num ?? den;
      if (num < 0 || num > den) out.push("hp");
      if (!!m.fainted !== (num === 0)) out.push("fainted");
    }
  }
  return [...new Set(out)];
}

/** The document to hand the solver: party order stated (active first), and
 * the picks the user has marked as the party. */
export function toSolverSpec(spec: PositionSpec): PositionSpec {
  const me = spec.side;
  const mons = spec.sides[me].mons;
  const active = spec.sides[me].active;
  const party: number[] = [];
  if (active != null) party.push(active);
  for (let i = 0; i < mons.length; i++) {
    if (i !== active && mons[i].appeared) party.push(i);
  }
  for (let i = 0; i < mons.length && party.length < PICKS; i++) {
    if (!party.includes(i)) party.push(i);
  }
  const sides: [SideSpec, SideSpec] = [spec.sides[0], spec.sides[1]];
  sides[me] = { ...sides[me], party: spec.team_preview ? [] : party.slice(0, PICKS) };
  return { ...spec, sides };
}

// --------------------------------------------------------------- storage

const LS_KEY = "nc2000-solver-position";

export function storePosition(spec: PositionSpec): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(spec));
  } catch {
    /* storage unavailable: the position is still in memory */
  }
}

/** The last position this browser was editing. Re-validated by the solver
 * itself, so a document an older build wrote is dropped on its first refusal
 * rather than trusted. */
export function loadStoredPosition(): PositionSpec | null {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return null;
    const v = JSON.parse(raw) as PositionSpec;
    if (v && v.schema === POSITION_SCHEMA && Array.isArray(v.sides)) return v;
    localStorage.removeItem(LS_KEY);
  } catch {
    /* unreadable: treat as absent */
  }
  return null;
}

export function clearStoredPosition(): void {
  try {
    localStorage.removeItem(LS_KEY);
  } catch {
    /* storage unavailable */
  }
}
