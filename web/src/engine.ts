// Main-thread wasm instance: Dex + Battle live here (the searcher lives in
// the bot worker's own instance — wasm memory is not shared across
// threads; preview tables are consumed by the worker too).

import init, {
  Dex,
  Battle,
  deriveBattleSeed,
} from "../../crates/wasm/pkg-web/nc2000_wasm";
import type { Choice, StateView } from "./types";

export { Battle };

let dex: Dex | null = null;

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
