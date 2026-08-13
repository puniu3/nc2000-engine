// META-NASH v1's conclusion, as the browser plays it: a fixed probability
// distribution over three teams, and the draw that samples it.
//
// `data/meta-nash-v1/pool-artifact.json` is the shipped end of the study
// (docs/META-NASH-V1.md). Its claim is about the MIXTURE, not about any one
// team: the three are exploitable individually and the weighted blend is
// what no best-response search could beat. So the draw is the product — a
// nash game samples it once per battle, independently, and a rematch draws
// again. Playing the top-weighted team every time would be a different
// strategy with a different (worse) guarantee, and would also be the one
// reading of "the conclusion" a visitor could mistake for the whole of it.
//
// Two things this module deliberately does NOT do.
//
// It does not resolve the artifact's teams against the meta pool. All three
// happen to be in it (sample-07 / 08 / 10 are pool entries 14 / 5 / 12, sets
// identical), which is a fact worth knowing — the bot's own team is one the
// pool describes, so blind play against it looks like blind play against any
// pool team — but depending on it would couple the mode to the pool file's
// ids and ordering for no gain. The artifact carries its own sets; those are
// what gets played. Pool indices stay null, exactly like a custom team, which
// costs nothing: blind play skips the baked pair tables anyway (game.tsx).
//
// And it does not touch the belief. The mixture governs which team the BOT
// brings; what the bot assumes about the team it is FACING is unchanged —
// still the bundled pool as the candidate set, still the same fallback
// imputation (docs/META-NASH-V1.md §4: bot self-team choice only, opponent
// belief unmoved). Wiring the mixture into the belief would be a different
// experiment, and not one that has been run.
//
// The loader is strict: a team that does not validate refuses the whole
// artifact, and a refusal fails the page rather than degrading. `?nash`
// without its mixture is not a weaker nash mode, it is blind mode wearing
// its name — and the one thing the mode exists to show would be the thing
// silently missing.

import { getValidator, randomSeed32 } from "./engine";
import { findingAnchor, findingText, type Finding } from "./findings";
import type { SelectedTeam } from "./pool-pick";

/** One team of the mixture, validated, with the probability it is drawn
 * with. `species` / `levels` are for display — the start screen shows the
 * mixture's composition, never its sets. */
export interface NashTeam {
  id: string;
  /** Draw probability, renormalized so the three sum to exactly 1. */
  weight: number;
  species: string[];
  levels: number[];
  sets: unknown[];
}

export interface NashMix {
  teams: NashTeam[];
  /** The solution file the weights came from, for the credit line. */
  source: string;
}

/** Same shape as `parsePoolText`'s answer, for the same reason: either a
 * mixture that can play, or the lines explaining why not. Never a partial
 * mixture. */
export type NashParse =
  | { ok: true; mix: NashMix }
  | { ok: false; errors: string[] };

/** `canonicalizeTeam`'s JSON (crates/engine/src/validate.rs). */
interface CanonResult {
  ok: boolean;
  team: unknown[];
  errors: Finding[];
}

interface ArtifactTeam {
  id?: unknown;
  weight?: unknown;
  sets?: unknown;
}

/** Exactly six, as everywhere else: the format picks 3 of 6 under a 155
 * total-level cap. */
const TEAM_SIZE = 6;

/**
 * Validate `pool-artifact.json`'s text into a playable mixture.
 *
 * Errors are terse English rather than i18n lines. Every other refusal in
 * this app answers a file the *user* just picked; this one answers a file
 * the deploy shipped, so it is a build fault with no user-facing repair —
 * it belongs in the boot error box in the words the developer needs.
 */
export function parseNashArtifact(text: string): NashParse {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { ok: false, errors: [`nash artifact is not JSON: ${String(e)}`] };
  }
  const obj = (raw ?? {}) as { teams?: unknown; source_solution?: unknown };
  const entries = Array.isArray(obj.teams) ? (obj.teams as ArtifactTeam[]) : null;
  if (!entries || entries.length === 0)
    return { ok: false, errors: ["nash artifact has no teams"] };

  const validator = getValidator();
  const teams: NashTeam[] = [];
  const errors: string[] = [];

  entries.forEach((entry, i) => {
    const src = entry ?? {};
    const id = typeof src.id === "string" && src.id.trim() ? src.id.trim() : `nash-${i + 1}`;
    // A weight that is absent, negative or not finite makes the whole
    // distribution meaningless — there is no sane repair, and a silently
    // dropped arm would still look like a mixture on screen.
    const weight = typeof src.weight === "number" ? src.weight : Number.NaN;
    if (!Number.isFinite(weight) || weight < 0) {
      errors.push(`${id}: weight is not a non-negative number`);
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
    // listing — the same rule team-pool.ts follows, and for the same
    // reason: a level the validator rewrote must not be mislabelled on the
    // card that names the mixture.
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

  // The shipped weights are the solver's rounded output (0.575 / 0.222 /
  // 0.201) and sum to 0.998, not 1 — three decimal places is the precision
  // the study itself claims, and README says the exact weights are budget-
  // dependent anyway. Renormalizing here is what makes the residue land
  // proportionally instead of on whichever arm the walk happens to end on.
  const total = teams.reduce((a, t) => a + t.weight, 0);
  if (!(total > 0)) return { ok: false, errors: ["nash artifact weights sum to zero"] };
  for (const t of teams) t.weight /= total;

  const source =
    typeof obj.source_solution === "string" ? obj.source_solution : "";
  return { ok: true, mix: { teams, source } };
}

/**
 * Sample the mixture. One battle, one draw — this is the mixed strategy
 * being played, not a shuffle of a fixed list.
 *
 * The roll is `randomSeed32()`, the same CSPRNG source as the uniform pool
 * draw in pool-pick.ts, scaled into [0, 1). The final arm catches whatever
 * the cumulative walk leaves over, so float drift can never fall off the
 * end and return nothing.
 */
export function drawNashTeam(mix: NashMix): SelectedTeam {
  const teams = mix.teams;
  const r = randomSeed32() / 2 ** 32;
  let acc = 0;
  for (let i = 0; i < teams.length - 1; i++) {
    acc += teams[i].weight;
    if (r < acc) return selected(teams[i]);
  }
  return selected(teams[teams.length - 1]);
}

/** Pool index null: the sets come from the artifact, not from a pool slot,
 * so nothing downstream may index a baked table by them. */
function selected(t: NashTeam): SelectedTeam {
  return { id: t.id, sets: t.sets, poolIdx: null };
}

/** The first validator error, anchored to the mon it points at — same
 * formatting as team-pool.ts, since both quote the same validator. */
function firstProblem(errors: Finding[]): string {
  const f = Array.isArray(errors) ? errors[0] : undefined;
  if (!f) return "?";
  const anchor = findingAnchor(f);
  return anchor ? `${anchor}: ${findingText(f)}` : findingText(f);
}
