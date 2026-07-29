// Shapes of the JSON crossing the wasm boundary (crates/wasm/src/lib.rs is
// the source of truth) plus the meta-pool file format.

export interface MoveChoice {
  kind: "move";
  input: string;
  id: string;
  name: string;
  type: string;
  category: "Physical" | "Special" | "Status";
  basePower: number;
  pp: number;
  maxpp: number;
  target: string;
}

export interface SwitchChoice {
  kind: "switch";
  input: string;
  pos: number;
  species: string;
  name: string;
  level: number;
  hp: number;
  maxhp: number;
  status: string;
}

export interface TeamChoice {
  kind: "team";
  input: string;
  slots: [number, number, number];
}

export interface PassChoice {
  kind: "pass";
  input: string;
}

export type Choice = MoveChoice | SwitchChoice | TeamChoice | PassChoice;

export interface MoveSlotView {
  id: string;
  name: string;
  pp: number;
  maxpp: number;
  disabled: boolean;
}

export interface PokeView {
  species: string;
  name: string;
  level: number;
  gender: string;
  hp: number;
  maxhp: number;
  fainted: boolean;
  status: string;
  boosts: Record<string, number>;
  moves: MoveSlotView[];
  item: string | null;
  types: string[];
  volatiles: string[];
  trapped: boolean;
}

export interface SideView {
  name: string;
  active: number | null;
  party: PokeView[];
  pokemonLeft: number;
  sideConditions: string[];
  request: string;
}

export interface StateView {
  turn: number;
  sides: [SideView, SideView];
  field: { weather: string | null; pseudoWeather: string[] };
  outcome: "p1" | "p2" | "tie" | null;
}

export interface RootPolicy {
  iterations: number;
  preview: boolean;
  /** The pick came from the baked preview table. */
  baked?: boolean;
  actions: { input: string; visits: number; mean: number; frac: number }[];
}

// --------------------------------------------------- belief (blind mode)

/** `BlindSearcher.beliefInfo()` — the bot's current read of the opponent's
 * team. `count` 1 = publicly identified; `fallback` true = no pool team is
 * consistent with everything observed (a custom team), so imputation runs on
 * a synthesized roster instead of on a pool set. */
export interface BeliefInfo {
  count: number;
  fallback: boolean;
  /** Pool team ids still alive in the belief. */
  candidates: string[];
}

/** `BlindSearcher.beliefPriorInfo()` — whether a non-empty community prior
 * (M18) actually reached this game's belief, and which opponent roster slots
 * it currently drives (observer roster order). `governed` is empty before the
 * first observe(), on a pool-identified or pinned opponent, and whenever the
 * table says nothing usable about this roster — installed but governing
 * nothing is a real and reportable state. */
export interface PriorInfo {
  installed: boolean;
  governed: boolean[];
}

/** `BlindSearcher.setBeliefPrior()` / `probeBeliefPrior()` — the table
 * interpreter is total, so a malformed or half-usable file degrades into
 * `warnings` rather than throwing. `applied` false means nothing was
 * installed (from a probe: nothing would be); `skipped` counts entries the
 * interpreter dropped, so `species` + `skipped` is what the file offered. */
export interface PriorReport {
  applied: boolean;
  species: number;
  meanMoveSum: number;
  skipped: number;
  warnings: string[];
}

// ------------------------------------------------------------- meta pool

export interface PoolTeam {
  id: string;
  tier: string;
  rank: number;
  species: string[];
  levels: number[];
  provenance: {
    source?: string;
    event?: string;
    player?: string;
    placement?: string;
    notes?: string;
  };
  sets: unknown[];
}

export interface MetaPool {
  meta: { teams: number };
  teams: PoolTeam[];
}

// --------------------------------------------------------------- log view

export interface LogEntry {
  kind: "turn" | "major" | "minor" | "result";
  text: string;
}
