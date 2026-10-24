import live from "../../data/live-search.json";
import profilesJson from "../../data/search-profiles.json?raw";
import type { InfoMode } from "./info-mode";

const profiles: Record<InfoMode, { c: number; iterations: number }> =
  JSON.parse(profilesJson);
const testBudget =
  import.meta.env.MODE === "test"
    ? Number(import.meta.env.VITE_NC2000_TEST_BUDGET)
    : Number.NaN;

/** Default live battle budget, never the test override. Selection has a
 * separate policy; this number alone does not reproduce the whole product. */
export type Strength = keyof typeof live.battle;
const isStrength = (value: unknown): value is Strength =>
  typeof value === "string" && Object.prototype.hasOwnProperty.call(live.battle, value);
export const DEFAULT_STRENGTH: Strength = isStrength(live.defaultStrength)
  ? live.defaultStrength : "strong";
export const PRODUCT_ITERATIONS = live.battle[DEFAULT_STRENGTH];
export const PREVIEW_ITERATIONS = live.unknownPreview;
export function liveBudget(strength: Strength, preview = false): number {
  return Number.isSafeInteger(testBudget) && testBudget > 0
    ? testBudget : preview ? live.unknownPreview : live.battle[strength];
}
export function loadStrength(): Strength {
  try {
    const saved = localStorage.getItem("nc2000-strength");
    return isStrength(saved) ? saved : DEFAULT_STRENGTH;
  }
  catch { return DEFAULT_STRENGTH; }
}
export function storeStrength(value: Strength): void {
  try { localStorage.setItem("nc2000-strength", value); } catch { /* Private browsing. */ }
}

/** Live play is "blind"; "open" is the retired profile, only for replaying
 * records made under it. */
export function searchProfile(mode: InfoMode) {
  return {
    c: profiles[mode].c,
    iterations:
      Number.isSafeInteger(testBudget) && testBudget > 0
        ? testBudget
        : profiles[mode].iterations,
  };
}
