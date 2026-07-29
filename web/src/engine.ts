// Main-thread wasm instance: Dex + Battle live here (the searcher lives in
// the bot worker's own instance — wasm memory is not shared across
// threads; preview tables are consumed by the worker too).

import init, {
  Dex,
  Battle,
  Validator,
  deriveBattleSeed,
  probeBeliefPrior,
} from "../../crates/wasm/pkg-web/nc2000_wasm";
import type { Choice, PriorReport, StateView } from "./types";

export { Battle };

let dex: Dex | null = null;
let validator: Validator | null = null;

export async function loadEngine(): Promise<Dex> {
  if (!dex) {
    await init();
    dex = new Dex();
  }
  return dex;
}

export function getDex(): Dex {
  if (!dex) throw new Error("engine not loaded");
  return dex;
}

/** M14a team validator (embedded learnsets), lazily constructed — only the
 * custom-team import flow needs it. */
export function getValidator(): Validator {
  if (!validator) validator = new Validator(getDex());
  return validator;
}

export function randomSeed32(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

export function newBattleSeed(): string {
  return deriveBattleSeed(randomSeed32());
}

// Typed wrappers over the JSON-string API.

export function legalChoices(battle: Battle, side: number): Choice[] {
  return JSON.parse(battle.legalChoices(side)) as Choice[];
}

export function needsChoice(battle: Battle): [boolean, boolean] {
  return JSON.parse(battle.needsChoice()) as [boolean, boolean];
}

export function stateView(battle: Battle): StateView {
  return JSON.parse(battle.stateView()) as StateView;
}

export function takeNewLog(battle: Battle): string[] {
  return JSON.parse(battle.takeNewLog()) as string[];
}

/** M18: read a belief-prior table WITHOUT installing it — the start screen
 * reports what a picked file says long before the game's searcher (which
 * lives in the worker's own wasm instance) exists. `applied` here means
 * "this table would apply if installed": the per-searcher refusals (pinned
 * opponent, already observed) are not knowable from the main thread.
 * The interpreter is total, so a malformed file comes back as warnings. */
export function probePrior(json: string): PriorReport {
  return JSON.parse(probeBeliefPrior(json)) as PriorReport;
}
