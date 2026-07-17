// UI-2 open-team-sheet display data. The dex JSON (data/gen2stadium2.json,
// the same file the wasm engine embeds) is fetched once at startup and
// consulted client-side for move meta (type/category/base power) and
// species types; a fetch failure just hides the meta — names still render.
// Hidden Power's type and power are derived from the set's DVs with the
// gen-2 formula (DV = floor(iv/2); PS stores gen-2 DVs as ivs) — every
// pool spread cross-checks against its typed move name.

import { toId } from "./i18n";

export type MoveCategory = "Physical" | "Special" | "Status";

export interface MoveMeta {
  type: string;
  category: MoveCategory;
  basePower: number;
}

interface DexMove {
  type: string;
  category: MoveCategory;
  basePower: number;
}

interface DexData {
  moves: Record<string, DexMove>;
  species: Record<string, { types: string[] }>;
}

let dexData: DexData | null = null;

/** Load the dex tables (non-fatal: on failure set sheets render names
 * without type/category/BP meta). Called once at app startup. */
export async function loadSetDex(
  fetchJson: () => Promise<unknown>,
): Promise<void> {
  try {
    const d = (await fetchJson()) as Partial<DexData> | null;
    if (d && d.moves && d.species) dexData = d as DexData;
  } catch {
    dexData = null;
  }
}

// Gen-2 Hidden Power type table, indexed by 4*(atkDV%4) + (defDV%4).
const HP_TYPES = [
  "Fighting",
  "Flying",
  "Poison",
  "Ground",
  "Rock",
  "Bug",
  "Ghost",
  "Steel",
  "Fire",
  "Water",
  "Grass",
  "Electric",
  "Psychic",
  "Ice",
  "Dragon",
  "Dark",
];

const dv = (ivs: Record<string, number>, k: string) =>
  Math.floor((ivs[k] ?? 31) / 2);

/** Gen-2 Hidden Power type from the set's DVs. */
export function hiddenPowerType(ivs: Record<string, number>): string {
  return HP_TYPES[4 * (dv(ivs, "atk") % 4) + (dv(ivs, "def") % 4)];
}

/** Gen-2 Hidden Power base power (31..70) from the set's DVs. */
export function hiddenPowerPower(ivs: Record<string, number>): number {
  const msb = (k: string) => (dv(ivs, k) >= 8 ? 1 : 0);
  const x = msb("atk") + 2 * msb("def") + 4 * msb("spe") + 8 * msb("spa");
  return Math.floor((5 * x + (dv(ivs, "spa") % 4)) / 2) + 31;
}

/** Move meta for a set's move (display name or id). Hidden Power resolves
 * its type through the set's DVs (an untyped "Hidden Power" is typed via
 * the DV formula; a typed name is looked up directly). Null when the dex
 * table is unavailable or the move is unknown. */
export function moveMeta(
  move: string,
  ivs?: Record<string, number>,
): MoveMeta | null {
  if (!dexData) return null;
  let id = toId(move);
  if (id === "hiddenpower" && ivs)
    id = `hiddenpower${hiddenPowerType(ivs).toLowerCase()}`;
  const e = dexData.moves[id];
  if (!e) return null;
  const basePower =
    id.startsWith("hiddenpower") && id !== "hiddenpower" && ivs
      ? hiddenPowerPower(ivs)
      : e.basePower;
  return { type: e.type, category: e.category, basePower };
}

/** Species types from the dex ("Exeggutor" -> Grass/Psychic); null when
 * the dex table is unavailable. */
export function speciesTypes(species: string): string[] | null {
  return dexData?.species[toId(species)]?.types ?? null;
}

// ------------------------------------------------ set JSON normalization

/** One mon of a static team JSON (pool entry / selected human team /
 * saved custom), normalized for sheet display. */
export interface SheetMon {
  species: string;
  level: number;
  gender: string; // "M" | "F" | "N" | ""
  item: string | null;
  moves: string[];
  ivs?: Record<string, number>;
}

export function sheetMon(set: unknown): SheetMon {
  const s = set as {
    species?: string;
    name?: string;
    level?: number;
    gender?: string;
    item?: string;
    moves?: string[];
    ivs?: Record<string, number>;
  };
  return {
    species: s.species ?? s.name ?? "?",
    level: s.level ?? 100,
    gender: s.gender ?? "",
    item: s.item ? s.item : null,
    moves: Array.isArray(s.moves) ? s.moves : [],
    ivs: s.ivs,
  };
}
