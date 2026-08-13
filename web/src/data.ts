// Runtime data fetching. The meta pool and the baked preview tables are
// served read-only from the repo data/ dir (see vite.config.ts) — pair
// files are still being baked in the background, so a missing or
// half-written file is an expected condition, answered with null (the
// caller falls back to live Searcher preview).

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

export async function fetchPool(): Promise<PoolData> {
  const res = await fetch(dataUrl("meta-pool-v0/meta-pool.json"));
  if (!res.ok) throw new Error(`meta pool fetch failed: ${res.status}`);
  const poolJson = await res.text();
  return { pool: JSON.parse(poolJson) as MetaPool, poolJson };
}

/** META-NASH v1's shipped mixture (`?nash` only, so it is fetched only on
 * that door). Returned as text because nash-mix.ts is what decides what the
 * file means — and because a nash page with no mixture is not a mode, this
 * one throws: the caller lets it reach the boot error box rather than
 * quietly starting a plainer game under the mode's name. */
export async function fetchNashArtifact(): Promise<string> {
  const res = await fetch(dataUrl("meta-nash-v1/pool-artifact.json"));
  if (!res.ok) throw new Error(`nash artifact fetch failed: ${res.status}`);
  return res.text();
}

/** Pair table for pool indices (i, j); canonical file is lo-hi. Returns the
 * raw JSON text, or null when the pair is not baked yet (404) or the file
 * is mid-write (parse failure). */
export async function fetchPairJson(
  i: number,
  j: number,
): Promise<string | null> {
  const lo = Math.min(i, j);
  const hi = Math.max(i, j);
  const pad = (n: number) => String(n).padStart(2, "0");
  const url = dataUrl(`preview-tables-v0/pair-${pad(lo)}-${pad(hi)}.json`);
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const text = await res.text();
    JSON.parse(text); // reject half-written files
    return text;
  } catch {
    return null;
  }
}
