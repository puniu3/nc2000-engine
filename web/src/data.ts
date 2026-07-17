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

export async function fetchPool(): Promise<PoolData> {
  const res = await fetch(dataUrl("meta-pool-v0/meta-pool.json"));
  if (!res.ok) throw new Error(`meta pool fetch failed: ${res.status}`);
  const poolJson = await res.text();
  return { pool: JSON.parse(poolJson) as MetaPool, poolJson };
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
