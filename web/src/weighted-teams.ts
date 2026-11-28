// A shipped file of teams, each with a draw probability — the shape the
// catalog (team-pool-v4, `drawWeight`) takes — validated into something
// that can play, and the draw that samples it.
//
// Strict both ways: a team that does not validate, or a weight that is not
// a non-negative number, refuses the whole file. These files are build
// output, not user input, so a refusal is a deploy fault and belongs in the
// boot error box in the developer's words (terse English, not i18n).

import { getValidator, randomSeed32 } from "./engine";
import { findingAnchor, findingText, type Finding } from "./findings";
import type { SelectedTeam } from "./pool-pick";

export interface WeightedTeam {
  id: string;
  /** Draw probability, renormalized so the file sums to exactly 1. */
  weight: number;
  species: string[];
  levels: number[];
  sets: unknown[];
}

export type WeightedParse =
  | { ok: true; teams: WeightedTeam[]; raw: Record<string, unknown> }
  | { ok: false; errors: string[] };

/** `canonicalizeTeam`'s JSON (crates/engine/src/validate.rs). */
interface CanonResult {
  ok: boolean;
  team: unknown[];
  errors: Finding[];
}

const TEAM_SIZE = 6;

export function parseWeightedTeams(
  text: string,
  weightKey: string,
  what: string,
): WeightedParse {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { ok: false, errors: [`${what} is not JSON: ${String(e)}`] };
  }
  const obj = (raw ?? {}) as Record<string, unknown>;
  const entries = Array.isArray(obj.teams)
    ? (obj.teams as Record<string, unknown>[])
    : null;
  if (!entries || entries.length === 0)
    return { ok: false, errors: [`${what} has no teams`] };

  const validator = getValidator();
  const teams: WeightedTeam[] = [];
  const errors: string[] = [];
  entries.forEach((entry, i) => {
    const src = entry ?? {};
    const id =
      typeof src.id === "string" && src.id.trim() ? src.id.trim() : `${what}-${i + 1}`;
    const w = src[weightKey];
    const weight = typeof w === "number" ? w : Number.NaN;
    if (!Number.isFinite(weight) || weight < 0) {
      errors.push(`${id}: ${weightKey} is not a non-negative number`);
      return;
    }
    if (!Array.isArray(src.sets) || src.sets.length !== TEAM_SIZE) {
      errors.push(
        `${id}: expected ${TEAM_SIZE} sets, got ${
          Array.isArray(src.sets) ? src.sets.length : "none"
        }`,
      );
      return;
    }
    let res: CanonResult;
    try {
      res = JSON.parse(
        validator.canonicalizeTeam(JSON.stringify(src.sets)),
      ) as CanonResult;
    } catch (e) {
      errors.push(`${id}: ${String(e)}`);
      return;
    }
    if (!res.ok || !Array.isArray(res.team)) {
      errors.push(`${id}: ${firstProblem(res.errors)}`);
      return;
    }
    // Display metadata from the canonicalized sets, never the file's own
    // listing: a level the validator rewrote must not be mislabelled.
    const mons = res.team as { species?: string; level?: number }[];
    teams.push({
      id,
      weight,
      species: mons.map((m) => m.species ?? "?"),
      levels: mons.map((m) => m.level ?? 55),
      sets: res.team,
    });
  });
  if (errors.length > 0) return { ok: false, errors };

  // Weights are rounded solver/allocation output; renormalizing here lands
  // the rounding residue proportionally instead of on whichever team the
  // cumulative walk happens to end on.
  const total = teams.reduce((a, t) => a + t.weight, 0);
  if (!(total > 0)) return { ok: false, errors: [`${what} weights sum to zero`] };
  for (const t of teams) t.weight /= total;
  return { ok: true, teams, raw: obj };
}

/** One battle, one draw: a fresh 32-bit CSPRNG roll scaled into [0, 1). The
 * last team catches whatever float drift leaves over. */
export function drawWeighted(teams: WeightedTeam[]): SelectedTeam {
  const r = randomSeed32() / 2 ** 32;
  let acc = 0;
  for (let i = 0; i < teams.length - 1; i++) {
    acc += teams[i].weight;
    if (r < acc) return selected(teams[i]);
  }
  return selected(teams[teams.length - 1]);
}

function selected(t: WeightedTeam): SelectedTeam {
  return { id: t.id, sets: t.sets };
}

function firstProblem(errors: Finding[]): string {
  const f = Array.isArray(errors) ? errors[0] : undefined;
  if (!f) return "?";
  const anchor = findingAnchor(f);
  return anchor ? `${anchor}: ${findingText(f)}` : findingText(f);
}
