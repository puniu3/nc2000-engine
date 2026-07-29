// The meta pool as something the user can replace — one file, the whole
// pool.
//
// Everything downstream of "the pool" moves together, because they are all
// reads of this one object: the bot's random draw, the candidate set its
// blind-mode belief is built over, and both team lists on the start screen.
// There is no way to swap one and not the others, and the UI copy says so.
//
// The loader is generous going in and strict coming out.
//
// Generous, because a pool file is hand-made: `{teams:[…]}` (the bundled
// file's shape) or a bare array, ids optional, tier/rank/provenance
// optional. The engine side asks for almost nothing — crates/bot/src/
// preview.rs reads a pool as `{teams:[{id, sets}]}` and ignores the rest —
// so everything else is display metadata this module can derive rather than
// demand.
//
// Strict, because a half-applied pool is worse than no pool. Every team goes
// through the M14a validator (`canonicalizeTeam`, the same call the custom
// team import runs), and one unplayable team refuses the whole file: a pool
// that loaded is a pool that can play, in either information mode.
//
// What reaches wasm is the re-serialized *normalized* pool, never the file's
// own bytes. The ids this module filled in and the sets the validator
// rewrote have to be the ones the searcher sees — otherwise the bot draws
// from a pool that differs from the one the screen is showing.

import { getValidator } from "./engine";
import { findingAnchor, findingText, type Finding } from "./findings";
import { ui } from "./i18n";
import type { MetaPool, PoolTeam } from "./types";

/** The pool a session is actually playing with. `name` is the file it came
 * from, or null for the bundled pool — which is also the flag the rest of
 * the app tests to know that pool indices (baked pair tables, ranks) mean
 * what they historically meant. */
export interface LoadedPool {
  name: string | null;
  pool: MetaPool;
  /** Normalized JSON text; what the worker and wasm are handed. */
  poolJson: string;
}

export interface StoredPool {
  name: string;
  json: string;
}

/** Either a pool ready to install, or the lines to show the human who wrote
 * the file. Never a partial pool. */
export type PoolParse =
  | { ok: true; pool: MetaPool; poolJson: string; teams: number }
  | { ok: false; errors: string[] };

const LS_KEY = "nc2000-team-pool";

/** Exactly six, always: the format picks 3 of 6 under a 155 total-level cap,
 * and both rules read the party as a fixed-size thing. */
const TEAM_SIZE = 6;

/** Enough lines to see a pattern, few enough to read at a glance; the rest
 * collapses into `poolMore`. Re-loading after a fix shows the next batch. */
const MAX_ERROR_LINES = 5;

/** Refuse to persist pools that cannot plausibly fit localStorage (~5 MB per
 * origin, shared with saved teams, picks and the belief prior). The bundled
 * 32-team pool is ~180 KB and a 135-team pool ~760 KB, so this is roughly
 * 350 teams — far past anything hand-assembled. Over the cap the pool still
 * plays; only the persistence is refused. */
export const POOL_MAX_BYTES = 2_000_000;

/** `canonicalizeTeam`'s JSON (crates/engine/src/validate.rs). `applied` (the
 * fixes it made) is deliberately not surfaced here: for a pool file, a fix
 * is not something the user is asked to act on — the normalized set is what
 * gets played and what gets saved. */
interface CanonResult {
  ok: boolean;
  team: unknown[];
  errors: Finding[];
}

/**
 * Validate and normalize a pool file's text. On success the caller gets a
 * pool it can install as-is; on failure, nothing — the current pool must
 * stay untouched.
 */
export function parsePoolText(text: string): PoolParse {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { ok: false, errors: [ui().poolErrJson(String(e))] };
  }

  const entries = poolEntries(raw);
  if (!entries || entries.length === 0)
    return { ok: false, errors: [ui().poolErrNoTeams] };

  const validator = getValidator();
  const teams: PoolTeam[] = [];
  const errors: string[] = [];
  const seenIds = new Set<string>();

  entries.forEach((entry, i) => {
    const src = (entry ?? {}) as Partial<PoolTeam>;
    // The id is the handle the error lines use, so it is resolved before
    // anything can fail. A file with no ids gets positional ones.
    const given = typeof src.id === "string" ? src.id.trim() : "";
    const id = given || `pool-${i + 1}`;
    if (seenIds.has(id)) {
      // Pinned start-screen picks are stored by team id, so two teams under
      // one id would restore the wrong team on the next visit.
      errors.push(ui().poolErrDupId(id));
      return;
    }
    seenIds.add(id);

    // Size is checked before the validator: `canonicalizeTeam` does report
    // team-size, but buried under one finding per malformed mon, and a
    // `sets` that is not an array is not a team to stringify at all.
    if (!Array.isArray(src.sets)) {
      errors.push(ui().poolErrSets(id));
      return;
    }
    if (src.sets.length !== TEAM_SIZE) {
      errors.push(ui().poolErrTeamSize(id, src.sets.length));
      return;
    }

    let res: CanonResult;
    try {
      res = JSON.parse(
        validator.canonicalizeTeam(JSON.stringify(src.sets)),
      ) as CanonResult;
    } catch (e) {
      errors.push(ui().poolErrTeam(id, String(e)));
      return;
    }
    if (!res.ok || !Array.isArray(res.team)) {
      errors.push(ui().poolErrTeam(id, firstProblem(res.errors)));
      return;
    }

    // Display metadata comes from the canonicalized sets, never from the
    // file's own `species` / `levels`: canonicalization can rewrite a level
    // (0/missing → 55) or a species spelling, and a list that disagrees with
    // the sets would mislabel the team cards for the whole session.
    const mons = res.team as { species?: string; level?: number }[];
    teams.push({
      id,
      tier: typeof src.tier === "string" ? src.tier : "",
      rank: typeof src.rank === "number" ? src.rank : i + 1,
      species: mons.map((m) => m.species ?? "?"),
      levels: mons.map((m) => m.level ?? 55),
      provenance:
        src.provenance && typeof src.provenance === "object"
          ? src.provenance
          : {},
      sets: res.team,
    });
  });

  if (errors.length > 0) return { ok: false, errors: trim(errors) };

  // Same shape as the bundled file, so a saved pool re-reads through this
  // very function on the next load.
  const pool: MetaPool = { meta: { teams: teams.length }, teams };
  return {
    ok: true,
    pool,
    poolJson: JSON.stringify(pool),
    teams: teams.length,
  };
}

export function loadStoredPool(): StoredPool | null {
  try {
    const rawItem = localStorage.getItem(LS_KEY);
    if (!rawItem) return null;
    const p = JSON.parse(rawItem) as Partial<StoredPool>;
    if (!p || typeof p.name !== "string" || typeof p.json !== "string")
      return null;
    return { name: p.name, json: p.json };
  } catch {
    return null;
  }
}

/**
 * Persist an accepted pool. Returns `null` on success or a short reason
 * string on failure.
 *
 * Unlike `storePrior`, a failure here must NOT block adoption: the pool has
 * already been validated and the session can play it perfectly well. All the
 * caller reports is that it will be gone after a reload (`poolNotStored`).
 *
 * Reasons are terse English and name a storage fault, not a pool defect —
 * pool defects never get this far.
 */
export function storePool(name: string, json: string): string | null {
  const bytes = byteLength(json);
  if (bytes > POOL_MAX_BYTES)
    return `pool too large (${bytes} bytes, limit ${POOL_MAX_BYTES})`;
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({ name, json }));
    return null;
  } catch (e) {
    return `could not be saved (${String(e)})`;
  }
}

export function clearStoredPool(): void {
  try {
    localStorage.removeItem(LS_KEY);
  } catch {
    /* storage unavailable: nothing was persisted to begin with */
  }
}

/** `{teams:[…]}` or a bare array — both are shapes people actually hand
 * around, and telling them apart is cheaper than making anyone re-wrap a
 * file. */
function poolEntries(raw: unknown): unknown[] | null {
  if (Array.isArray(raw)) return raw;
  if (raw && typeof raw === "object") {
    const t = (raw as { teams?: unknown }).teams;
    if (Array.isArray(t)) return t;
  }
  return null;
}

/** The first validator error, anchored to the mon it points at ("#2 Snorlax:
 * Can't learn Recover") — one line per team, because a hand-edited file is
 * fixed one problem at a time and five teams' worth of every finding is not
 * readable. `canonicalizeTeam` never answers ok:false with an empty error
 * list, so the fallback is type narrowing only. */
function firstProblem(errors: Finding[]): string {
  const f = Array.isArray(errors) ? errors[0] : undefined;
  if (!f) return "?";
  const anchor = findingAnchor(f);
  return anchor ? `${anchor}: ${findingText(f)}` : findingText(f);
}

function trim(errors: string[]): string[] {
  if (errors.length <= MAX_ERROR_LINES) return errors;
  return [
    ...errors.slice(0, MAX_ERROR_LINES),
    ui().poolMore(errors.length - MAX_ERROR_LINES),
  ];
}

/** UTF-8 size (mirrors belief-prior.ts — the same cap needs the same
 * measure, and the fallback keeps it meaningful in a runtime without
 * TextEncoder: UTF-16 length under-counts multi-byte text, never over). */
function byteLength(s: string): number {
  try {
    return new TextEncoder().encode(s).length;
  } catch {
    return s.length;
  }
}
