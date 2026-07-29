// App shell: engine + meta pool loading and the select -> game screen
// switch. A Game instance is keyed by game number so rematch / new-teams
// remount it cleanly.
//
// M12 product policy: strength is fixed at max (30k iterations — ponder
// hides the wait) and the information policy is OPEN TEAM SHEET — both
// sides' sets are public, only selection (which 3 of 6 + lead, until
// revealed) is hidden. No settings.
//
// M18 adds two things, neither of them a setting on the screen. The
// information mode (open / blind, see info-mode.ts) is read once from the
// URL at module load — `?blind` is the only door, and nothing the user can
// press moves it; a GameSpec still carries the mode its game started under,
// which is the guarantee game.tsx is written against. The team pool is
// state: one file replaces the pool everywhere it is read (team-pool.ts).
// The bundled pool is held next to the active one so going back to it is a
// state change rather than a second trip to the network.

import { useEffect, useState } from "preact/hooks";
import { loadEngine } from "./engine";
import { fetchDexJson, fetchI18nJa, fetchPool } from "./data";
import { loadSetDex } from "./set-info";
import { randomPoolTeam, type SelectedTeam } from "./pool-pick";
import { readInfoMode, type InfoMode } from "./info-mode";
import { loadStoredPool, parsePoolText, type LoadedPool } from "./team-pool";
import { loadStoredPrior, type StoredPrior } from "./belief-prior";
import { StartScreen } from "./select";
import { Game } from "./game";
import { loadJaNames, locale, setLocale, ui, type Locale } from "./i18n";

/** The fixed bot strength: the former "Max" tier, always on. Browser E2E
 * builds use Vite's explicit `test` mode to exercise whole games cheaply;
 * production mode cannot observe or honor that override. */
const testBudget =
  import.meta.env.MODE === "test"
    ? Number(import.meta.env.VITE_NC2000_TEST_BUDGET)
    : Number.NaN;
export const BUDGET =
  Number.isSafeInteger(testBudget) && testBudget > 0 ? testBudget : 30000;

/** `SelectedTeam` moved to pool-pick.ts, next to the pool draw that builds
 * one; re-exported here so the existing `from "./app"` imports keep
 * working. */
export type { SelectedTeam } from "./pool-pick";

/** This page load's information mode. Read at module scope because that is
 * the truth about it: the query string cannot change without a navigation,
 * and holding it in state would suggest something here could flip it. */
const MODE: InfoMode = readInfoMode();

interface GameSpec {
  human: SelectedTeam;
  bot: SelectedTeam;
  n: number;
  /** The information mode this game runs under, frozen at start. */
  mode: InfoMode;
}

/** The pool this browser was handed in an earlier session — re-validated,
 * never trusted: it is text the user picked by hand, saved by an older
 * build, against a validator that may since have moved. A file that no
 * longer parses is dropped without a word: boot must not hang on a stale
 * preference, and there is nowhere honest to report a file the user is not
 * loading right now. The bundled pool then stands, as it did before. */
function restoreStoredPool(): LoadedPool | null {
  try {
    const stored = loadStoredPool();
    if (!stored) return null;
    const parsed = parsePoolText(stored.json);
    if (!parsed.ok) return null;
    return { name: stored.name, pool: parsed.pool, poolJson: parsed.poolJson };
  } catch {
    return null;
  }
}

export function App() {
  const [status, setStatus] = useState<"loading" | "error" | "ready">(
    "loading",
  );
  const [error, setError] = useState("");
  // The pool in play, and the bundled one it can always fall back to. The
  // same object until a file is loaded, but two references: "use the
  // bundled pool" has to work after a swap, and the bundled pool is already
  // in memory — refetching it to get it back would be the one path that can
  // fail offline.
  const [bundled, setBundled] = useState<LoadedPool | null>(null);
  const [loadedPool, setLoadedPool] = useState<LoadedPool | null>(null);
  const [game, setGame] = useState<GameSpec | null>(null);
  const [loc, setLoc] = useState<Locale>(locale());
  // A table the user once picked by hand; nothing here ever fetches one on
  // its own (crates/bot/src/prior.rs:491).
  const [prior, setPrior] = useState<StoredPrior | null>(loadStoredPrior);

  useEffect(() => {
    void (async () => {
      try {
        // JP name tables and the set-sheet dex load alongside the engine;
        // both swallow failures (missing tables just mean English names /
        // sheets without move meta).
        const [, pd] = await Promise.all([
          loadEngine(),
          fetchPool(),
          loadJaNames(fetchI18nJa),
          loadSetDex(fetchDexJson),
        ]);
        const bundledPool: LoadedPool = {
          name: null,
          pool: pd.pool,
          poolJson: pd.poolJson,
        };
        setBundled(bundledPool);
        // Only now: re-validating a stored pool runs the wasm validator,
        // which the engine load above is what makes available.
        setLoadedPool(restoreStoredPool() ?? bundledPool);
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
  if (status === "error" || !loadedPool || !bundled) {
    return (
      <div class="center-screen">
        <div class="error-box">
          <strong>{ui().failedLoad}</strong>
          <div>{error}</div>
        </div>
      </div>
    );
  }

  if (!game) {
    return (
      <StartScreen
        loadedPool={loadedPool}
        bundledPool={bundled}
        onPool={setLoadedPool}
        locale={loc}
        onLocale={(l) => {
          setLocale(l);
          setLoc(l);
        }}
        mode={MODE}
        prior={prior}
        onPrior={setPrior}
        onStart={(human, bot) => setGame({ human, bot, n: 1, mode: MODE })}
      />
    );
  }

  return (
    <Game
      key={game.n}
      poolJson={loadedPool.poolJson}
      // Baked artifacts are indexed by the bundled pool's rank order, so a
      // swapped pool's indices name different teams entirely.
      poolIsCustom={loadedPool.name !== null}
      humanTeam={game.human}
      botTeam={game.bot}
      mode={game.mode}
      // The prior only ever reaches a blind game: in open mode the searcher
      // pins the human's real team and refuses the table outright.
      priorJson={game.mode === "blind" ? prior?.json : undefined}
      // Blind rematch redraws the opponent: replaying a lost battle against
      // the team you just watched play would hand the human the very
      // information blind mode withholds. Open mode keeps the same foe.
      onRematch={() =>
        setGame((g) =>
          g === null
            ? g
            : {
                ...g,
                n: g.n + 1,
                bot:
                  g.mode === "blind" ? randomPoolTeam(loadedPool.pool) : g.bot,
              },
        )
      }
      onNewTeams={() => setGame(null)}
    />
  );
}
