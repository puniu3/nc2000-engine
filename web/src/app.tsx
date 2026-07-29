// App shell: engine + meta pool loading and the select -> game screen
// switch. A Game instance is keyed by game number so rematch / new-teams
// remount it cleanly.
//
// M12 product policy: strength is fixed at max (30k iterations — ponder
// hides the wait) and the information policy is OPEN TEAM SHEET — both
// sides' sets are public, only selection (which 3 of 6 + lead, until
// revealed) is hidden. No settings.
//
// M18 adds one setting after all: the information mode (open / blind, see
// info-mode.ts), still defaulting to open. It is a start-screen preference,
// not a battle parameter — a GameSpec captures the mode in force at start,
// so toggling it later cannot change the information structure of a running
// game, and a rematch of an open game stays open even if the toggle moved.

import { useEffect, useRef, useState } from "preact/hooks";
import { loadEngine } from "./engine";
import { fetchDexJson, fetchI18nJa, fetchPool } from "./data";
import { loadSetDex } from "./set-info";
import type { MetaPool } from "./types";
import { randomPoolTeam, type SelectedTeam } from "./pool-pick";
import { loadInfoMode, storeInfoMode, type InfoMode } from "./info-mode";
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

interface GameSpec {
  human: SelectedTeam;
  bot: SelectedTeam;
  n: number;
  /** The information mode this game runs under, frozen at start. */
  mode: InfoMode;
}

export function App() {
  const [status, setStatus] = useState<"loading" | "error" | "ready">(
    "loading",
  );
  const [error, setError] = useState("");
  const [pool, setPool] = useState<MetaPool | null>(null);
  const poolJsonRef = useRef("");
  const [game, setGame] = useState<GameSpec | null>(null);
  const [loc, setLoc] = useState<Locale>(locale());
  // Per-browser preferences, restored on load. `prior` is a table the user
  // once picked by hand; nothing here ever fetches one on its own
  // (crates/bot/src/prior.rs:491).
  const [mode, setMode] = useState<InfoMode>(loadInfoMode);
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
        poolJsonRef.current = pd.poolJson;
        setPool(pd.pool);
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
  if (status === "error" || !pool) {
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
        pool={pool}
        locale={loc}
        onLocale={(l) => {
          setLocale(l);
          setLoc(l);
        }}
        mode={mode}
        onMode={(m) => {
          setMode(m);
          storeInfoMode(m);
        }}
        prior={prior}
        onPrior={setPrior}
        onStart={(human, bot) => setGame({ human, bot, n: 1, mode })}
      />
    );
  }

  return (
    <Game
      key={game.n}
      poolJson={poolJsonRef.current}
      humanTeam={game.human}
      botTeam={game.bot}
      mode={game.mode}
      // The prior only ever reaches a blind game: in open mode the searcher
      // pins the human's real team and refuses the table outright. `game.mode`
      // (not the live toggle) decides, for the same reason the mode is frozen.
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
                bot: g.mode === "blind" ? randomPoolTeam(pool) : g.bot,
              },
        )
      }
      onNewTeams={() => setGame(null)}
    />
  );
}
