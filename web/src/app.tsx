// App shell: engine + meta pool loading and the select -> game screen
// switch. A Game instance is keyed by game number so rematch / new-teams
// remount it cleanly.
//
// M12 product policy: strength is fixed at max (30k iterations — ponder
// hides the wait) and the information policy is OPEN TEAM SHEET — both
// sides' sets are public, only selection (which 3 of 6 + lead, until
// revealed) is hidden. No settings.

import { useEffect, useRef, useState } from "preact/hooks";
import { loadEngine } from "./engine";
import { fetchDexJson, fetchI18nJa, fetchPool } from "./data";
import { loadSetDex } from "./set-info";
import type { MetaPool } from "./types";
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

/** One side's selected team: a pool team (poolIdx set — baked pair tables
 * may apply) or a saved custom team (poolIdx null — preview is always live
 * search). Sets are captured at start, so deleting a saved custom during
 * the game cannot alter the current battle or its rematches. */
export interface SelectedTeam {
  id: string;
  sets: unknown[];
  poolIdx: number | null;
}

interface GameSpec {
  human: SelectedTeam;
  bot: SelectedTeam;
  n: number;
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
        onStart={(human, bot) => setGame({ human, bot, n: 1 })}
      />
    );
  }

  return (
    <Game
      key={game.n}
      poolJson={poolJsonRef.current}
      humanTeam={game.human}
      botTeam={game.bot}
      onRematch={() => setGame({ ...game, n: game.n + 1 })}
      onNewTeams={() => setGame(null)}
    />
  );
}
