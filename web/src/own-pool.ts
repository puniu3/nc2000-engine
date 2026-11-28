// The shipped catalog (data/team-pool-v4/team-pool.json): the curated
// 10-shell (15-team) equal-weight strong generalist pool. One file answers
// two questions with identical ids and exact sets — which built-in parties
// a human can pick on the start screen, and which party the bot brings when
// nobody pinned one — and both Random paths (the human's and the bot's) draw
// it by its `drawWeight`.
//
// It is NOT the bot's picture of its opponent (the opponent prior,
// belief-pool-v3).

import type { SelectedTeam } from "./pool-pick";
import type { LoadedPool } from "./team-pool";
import type { MetaPool, PoolTeam } from "./types";
import {
  drawWeighted,
  parseWeightedTeams,
  type WeightedTeam,
} from "./weighted-teams";

export interface Catalog {
  /** The catalog in the lists' shape; `name` null marks it as the shipped
   * catalog rather than a file the user loaded. */
  pool: LoadedPool;
  /** The same teams in the same order, with draw probabilities. */
  teams: WeightedTeam[];
}

export type CatalogParse =
  | { ok: true; catalog: Catalog }
  | { ok: false; errors: string[] };

interface CatalogEntry {
  id: string;
  label?: string;
  evolved?: boolean;
  provenance?: {
    source?: string;
    version?: string | null;
    family?: string;
    origin?: string;
  };
}

export function parseCatalog(text: string): CatalogParse {
  const parsed = parseWeightedTeams(text, "drawWeight", "team catalog");
  if (!parsed.ok) return parsed;
  const entries = (parsed.raw.teams ?? []) as CatalogEntry[];
  const teams: PoolTeam[] = parsed.teams.map((t, i) => {
    const entry = entries[i];
    const p = entry?.provenance ?? {};
    return {
      id: t.id,
      tier: "",
      rank: i + 1,
      species: t.species,
      levels: t.levels,
      evolved: Boolean(entry?.evolved || p.origin === "evolved"),
      provenance: { source: [p.source, p.version].filter(Boolean).join(" · ") },
      sets: t.sets,
    };
  });
  const pool: MetaPool = { meta: { teams: teams.length }, teams };
  return {
    ok: true,
    catalog: {
      pool: { name: null, pool, poolJson: JSON.stringify(pool) },
      teams: parsed.teams,
    },
  };
}

export function drawCatalogTeam(catalog: Catalog): SelectedTeam {
  return drawWeighted(catalog.teams);
}
