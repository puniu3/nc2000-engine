// Runtime data fetching, served read-only from the repo data/ dir (see
// vite.config.ts).

import type { MetaPool } from "./types";

/** All data fetches go under the deploy base (`/` locally,
 * `/nc2000-engine/` on GH Pages — see vite.config.ts). BASE_URL always ends
 * with a slash. */
const dataUrl = (rel: string) => `${import.meta.env.BASE_URL}data/${rel}`;

export interface PoolData {
  pool: MetaPool;
  poolJson: string;
}

/** JP name tables (M13). Throws on failure — the caller (i18n loadJaNames)
 * treats any failure as "no tables" and falls back to English names. */
export async function fetchI18nJa(): Promise<unknown> {
  const res = await fetch(dataUrl("i18n-ja.json"));
  if (!res.ok) throw new Error(`i18n-ja fetch failed: ${res.status}`);
  return res.json();
}

/** Dex JSON (data/gen2stadium2.json — the same data the wasm engine
 * embeds). Consulted client-side for set-sheet move meta (type/category/
 * BP) and species types. Throws on failure — the caller treats any
 * failure as "no meta available". */
export async function fetchDexJson(): Promise<unknown> {
  const res = await fetch(dataUrl("gen2stadium2.json"));
  if (!res.ok) throw new Error(`dex fetch failed: ${res.status}`);
  return res.json();
}

/** A hosted `nc2000-fork-v1` document, `data/forks/<name>.json`. */
export async function fetchFork(name: string): Promise<string> {
  if (!/^[A-Za-z0-9_-]+$/.test(name)) throw new Error(`invalid fork name: ${name}`);
  const res = await fetch(dataUrl(`forks/${name}.json`));
  if (!res.ok) throw new Error(`fork ${name}: ${res.status}`);
  return res.text();
}

/** The retired bundled 32 (meta-pool-v0): the belief the `?fork` arena
 * continues its hosted research records with, as they were recorded. Not a
 * list any live door offers. */
export async function fetchRecordPool(): Promise<PoolData> {
  const res = await fetch(dataUrl("meta-pool-v0/meta-pool.json"));
  if (!res.ok) throw new Error(`meta pool fetch failed: ${res.status}`);
  const poolJson = await res.text();
  return { pool: JSON.parse(poolJson) as MetaPool, poolJson };
}

/** The shipped mixture: the `?nash` draw and the play door's Nash draw
 * choice. Returned as text because nash-mix.ts is what decides what the file
 * means — and because a draw that names the mixture cannot quietly become a
 * plainer one, this throws: the caller lets it reach the boot error box. */
export async function fetchNashArtifact(): Promise<string> {
  const res = await fetch(dataUrl("kodoku-r3/conclusion-equilibrium-deme.json"));
  if (!res.ok) throw new Error(`nash artifact fetch failed: ${res.status}`);
  return res.text();
}

/** The previous v3 Nash mixture (meta-nash-v3), kept alongside the new
 * Full-Combination Bayesian Preview Nash until non-regression is confirmed. */
export async function fetchOldNashArtifact(): Promise<string> {
  const res = await fetch(dataUrl("meta-nash-v3/pool-artifact.json"));
  if (!res.ok) throw new Error(`old nash artifact fetch failed: ${res.status}`);
  return res.text();
}

/** The shipped catalog (own-pool.ts decides what it means): the built-in
 * parties on the start screen and the bot's ordinary draw. A page that
 * cannot load it fails rather than quietly drawing from some other list. */
export async function fetchCatalog(): Promise<string> {
  const res = await fetch(dataUrl("team-pool-v2/team-pool.json"));
  if (!res.ok) throw new Error(`team catalog fetch failed: ${res.status}`);
  return res.text();
}

/** The shipped opponent prior (data/belief-pool-v3): the candidate set, with
 * weights, that every searcher narrows its opponent down to — play, nash,
 * the solver, the evaluator and kifu continuations alike, whatever pool file
 * the user loaded. Belief-only: the catalog and every draw never read it.
 * Strict: a page that cannot load it fails rather than quietly playing
 * under a plainer prior. */
export async function fetchBeliefPool(): Promise<PoolData> {
  const res = await fetch(dataUrl("belief-pool-v3/belief-pool.json"));
  if (!res.ok) throw new Error(`belief pool fetch failed: ${res.status}`);
  const poolJson = await res.text();
  const pool = JSON.parse(poolJson) as MetaPool & { version?: number };
  // Sanity, not validation: a stale or wrong file must not stand in for the
  // frozen prior, whose every team carries a weight.
  const teams = Array.isArray(pool.teams) ? pool.teams : [];
  if (
    pool.version !== 3 ||
    teams.length === 0 ||
    teams.some((t) => typeof (t as { weight?: unknown }).weight !== "number")
  ) {
    throw new Error("belief pool: not the weighted v3 prior");
  }
  return { pool, poolJson };
}
