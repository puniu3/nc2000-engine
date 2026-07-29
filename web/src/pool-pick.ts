// One side's selected team, and the pool draw that produces one.
//
// The draw lives here (rather than inside the select screen) because two
// callers need the same rule: the start screen resolving a "Random" pick,
// and blind-mode rematch, which redraws the opponent every game so a lost
// battle cannot be re-run against a now-known team. Both must roll exactly
// the same way — one rule, one place.

import { randomSeed32 } from "./engine";
import type { MetaPool } from "./types";

/** One side's selected team: a pool team (poolIdx set — baked pair tables
 * may apply) or a saved custom team (poolIdx null — preview is always live
 * search). Sets are captured at start, so deleting a saved custom during
 * the game cannot alter the current battle or its rematches. */
export interface SelectedTeam {
  id: string;
  sets: unknown[];
  poolIdx: number | null;
}

/** Draw a uniformly random pool team. The rule is the shipped one from the
 * select screen's random branch: a fresh 32-bit CSPRNG roll reduced modulo
 * the pool size (the modulo bias over 32 bits is far below anything a
 * 32-team pool could express). */
export function randomPoolTeam(pool: MetaPool): SelectedTeam {
  const teams = pool.teams;
  const idx = randomSeed32() % teams.length;
  return { id: teams[idx].id, sets: teams[idx].sets, poolIdx: idx };
}
