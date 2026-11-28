// App shell: engine + team files loading and the select -> game screen
// switch. A Game instance is keyed by game number so rematch / new-teams
// remount it cleanly.
//
// Product policy: every game is blind (info-mode.ts) and searches with the
// blind search policy. The default adaptive battle budget is configured in
// data/live-search.json. `/` and its alias `?blind` are the same door.
//
// Two team files, two questions (docs/TEAM-POOL-REBUILD-PLAN.md):
// - the catalog (team-pool-v4): the built-in parties a human can pick, and
//   the party the bot brings — both Random draws use its draw weights;
// - the opponent prior (belief-pool-v3): what the bot assumes about the
//   party it faces, on every door.
// Neither is read in the other's place. A pool file the user loads in
// the setup panel is an explicit override of the lists and the bot's draw
// (drawn uniformly, because that file is what they chose to face); it never
// replaces the prior. The bot's draw lives in one place, `drawOpponent`,
// because the start screen and the rematch must roll identically.

import { useEffect, useState } from "preact/hooks";
import { loadEngine } from "./engine";
import {
  fetchBeliefPool,
  fetchCatalog,
  fetchDexJson,
  fetchI18nJa,
} from "./data";
import { loadSetDex } from "./set-info";
import { randomPoolTeam, type SelectedTeam } from "./pool-pick";
import { drawCatalogTeam, parseCatalog, type Catalog } from "./own-pool";
import { readDoor, type Door } from "./info-mode";
import {
  clearStoredPool,
  loadStoredPool,
  parsePoolText,
  type LoadedPool,
} from "./team-pool";
import {
  loadStoredPickPrior,
  loadStoredPrior,
  type StoredPrior,
} from "./belief-prior";
import { StartScreen } from "./select";
import { Game } from "./game";
import { Solver } from "./solver";
import { loadJaNames, locale, setLocale, ui, type Locale } from "./i18n";

/** `SelectedTeam` moved to pool-pick.ts, next to the pool draw that builds
 * one; re-exported here so the existing `from "./app"` imports keep
 * working. */
export type { SelectedTeam } from "./pool-pick";

/** This page load's door. Read at module scope because that is the truth
 * about it: the query string cannot change without a navigation. */
const DOOR: Door = readDoor();
/** The study board is not a way to play: it replaces the whole screen, so it
 * is checked before any of the game state below is consulted. */
const SOLVER = DOOR === "solver";

interface GameSpec {
  human: SelectedTeam;
  bot: SelectedTeam;
  n: number;
}

/** The pool file this browser was handed in an earlier session —
 * re-validated, never trusted: it is text the user picked by hand, saved by
 * an older build, against a validator that may since have moved. A file
 * that no longer parses is dropped and deleted without a word: boot must not
 * hang on a stale preference, and a record that cannot be adopted would
 * otherwise cost a full validator pass on every load. */
function restoreStoredPool(): LoadedPool | null {
  try {
    const stored = loadStoredPool();
    if (!stored) return null;
    const parsed = parsePoolText(stored.json);
    if (!parsed.ok) {
      clearStoredPool();
      return null;
    }
    return { name: stored.name, pool: parsed.pool, poolJson: parsed.poolJson };
  } catch {
    clearStoredPool();
    return null;
  }
}

export function App() {
  const [status, setStatus] = useState<"loading" | "error" | "ready">(
    "loading",
  );
  const [error, setError] = useState("");
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  // The lists in play: the catalog, or a pool file the user loaded.
  const [loadedPool, setLoadedPool] = useState<LoadedPool | null>(null);
  const [game, setGame] = useState<GameSpec | null>(null);
  // The shipped opponent prior: the blind searcher's candidate set only.
  const [beliefJson, setBeliefJson] = useState<string | null>(null);
  const [loc, setLoc] = useState<Locale>(locale());
  // A table the user once picked by hand; nothing here ever fetches one on
  // its own (docs/community-belief-prior-design.md).
  const [prior, setPrior] = useState<StoredPrior | null>(loadStoredPrior);
  const [pickPrior, setPickPrior] = useState<StoredPrior | null>(
    loadStoredPickPrior,
  );

  useEffect(() => {
    void (async () => {
      try {
        // JP name tables and the set-sheet dex load alongside the engine;
        // both swallow failures (missing tables just mean English names /
        // sheets without move meta).
        const [, catalogText, , , beliefPd] = await Promise.all([
          loadEngine(),
          fetchCatalog(),
          loadJaNames(fetchI18nJa),
          loadSetDex(fetchDexJson),
          fetchBeliefPool(),
        ]);
        // Parsed after the engine is up, because validating a team file is
        // a wasm call. Strict: a file that cannot play takes the page down
        // rather than leaving a door running under a name it cannot honour.
        const parsed = parseCatalog(catalogText);
        if (!parsed.ok) throw new Error(parsed.errors.join("; "));
        setCatalog(parsed.catalog);
        setBeliefJson(beliefPd.poolJson);
        setLoadedPool(restoreStoredPool() ?? parsed.catalog.pool);
        setStatus("ready");
      } catch (e) {
        setError(String(e));
        setStatus("error");
      }
    })();
  }, []);

  if (status === "loading") {
    return (
      <div class="center-screen">
        <div class="loading-pulse">{ui().loadingEngine}</div>
      </div>
    );
  }
  if (status === "error" || !loadedPool || !catalog || !beliefJson) {
    return (
      <div class="center-screen">
        <div class="error-box">
          <strong>{ui().failedLoad}</strong>
          <div>{error}</div>
        </div>
      </div>
    );
  }

  const activePool = SOLVER ? catalog.pool : loadedPool;
  const custom = activePool.name !== null;

  if (SOLVER) {
    return (
      <Solver
        pool={catalog.pool.pool}
        poolJson={beliefJson}
        priorJson={prior?.json}
        pickPriorJson={pickPrior?.json}
        locale={loc}
        onLocale={(l) => {
          setLocale(l);
          setLoc(l);
        }}
      />
    );
  }

  /** The bot's party when nobody pinned one, in one place because the start
   * screen and every rematch must roll the same way. */
  const drawOpponent = (): SelectedTeam =>
    custom ? randomPoolTeam(activePool.pool) : drawCatalogTeam(catalog);
  /** The human's "Random": the catalog's draw weights, or uniform over a
   * loaded file — the same rule as the bot's draw from the same lists. */
  const drawHuman = (): SelectedTeam =>
    custom ? randomPoolTeam(activePool.pool) : drawCatalogTeam(catalog);

  if (!game) {
    return (
      <StartScreen
        loadedPool={activePool}
        catalogPool={catalog.pool}
        onPool={setLoadedPool}
        locale={loc}
        onLocale={(l) => {
          setLocale(l);
          setLoc(l);
        }}
        drawOpponent={drawOpponent}
        drawHuman={drawHuman}
        botDrawCount={activePool.pool.teams.length}
        prior={prior}
        onPrior={setPrior}
        pickPrior={pickPrior}
        onPickPrior={setPickPrior}
        onStart={(human, bot) => setGame({ human, bot, n: 1 })}
      />
    );
  }

  return (
    <Game
      key={game.n}
      poolJson={beliefJson}
      humanTeam={game.human}
      botTeam={game.bot}
      priorJson={prior?.json}
      pickPriorJson={pickPrior?.json}
      // A rematch redraws the opponent: replaying a lost battle against the
      // party you just watched play would hand the human the very
      // information blind play withholds.
      onRematch={() =>
        setGame((g) =>
          g === null ? g : { ...g, n: g.n + 1, bot: drawOpponent() },
        )
      }
      onNewTeams={() => setGame(null)}
    />
  );
}
