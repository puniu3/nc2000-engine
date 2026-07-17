// M14b: saved custom teams — localStorage-backed, canonical team JSON only
// (the import flow runs parse -> canonicalizeTeam before saving, so a
// stored team is always Battle-constructor-ready). Storage failures degrade
// to an empty list / silent no-op: customs are a convenience layer.

export interface CustomTeam {
  /** Unique, stable ("custom-<epoch-ms>[-n]"). Never a pool id. */
  id: string;
  name: string;
  /** Canonicalized sets (Battle/Validator JSON shape). */
  sets: unknown[];
  /** Display metadata derived from the sets at save time. */
  species: string[];
  levels: number[];
  savedAt: number;
}

const LS_KEY = "nc2000-custom-teams";

export function loadCustomTeams(): CustomTeam[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return [];
    const list = JSON.parse(raw) as CustomTeam[];
    if (!Array.isArray(list)) return [];
    return list.filter(
      (t) => t && typeof t.id === "string" && Array.isArray(t.sets),
    );
  } catch {
    return [];
  }
}

function store(list: CustomTeam[]): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(list));
  } catch {
    /* storage unavailable/full: the team still plays this session */
  }
}

/** Save a canonicalized team; returns the stored record. */
export function saveCustomTeam(name: string, sets: unknown[]): CustomTeam {
  const list = loadCustomTeams();
  let id = `custom-${Date.now()}`;
  let n = 1;
  while (list.some((t) => t.id === id)) id = `custom-${Date.now()}-${n++}`;
  const mons = sets as { species?: string; level?: number }[];
  const team: CustomTeam = {
    id,
    name: name.trim() || defaultName(list),
    sets,
    species: mons.map((m) => m.species ?? "?"),
    levels: mons.map((m) => m.level ?? 55),
    savedAt: Date.now(),
  };
  store([...list, team]);
  return team;
}

export function deleteCustomTeam(id: string): CustomTeam[] {
  const list = loadCustomTeams().filter((t) => t.id !== id);
  store(list);
  return list;
}

function defaultName(list: CustomTeam[]): string {
  let n = list.length + 1;
  while (list.some((t) => t.name === `Custom ${n}`)) n++;
  return `Custom ${n}`;
}
