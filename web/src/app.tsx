// App shell: engine + meta pool loading and the select -> game screen
// switch. A Game instance is keyed by game number so rematch / new-teams
// remount it cleanly.
//
// M12 product policy: strength is fixed at max (30k iterations — ponder
// hides the wait) and the information policy is OPEN TEAM SHEET — both
// sides' sets are public, only selection (which 3 of 6 + lead, until
// revealed) is hidden. No settings.

import { useEffect, useRef, useState } from "preact/hooks";
import { loadEngine, randomSeed32 } from "./engine";
import { fetchPool } from "./data";
import type { MetaPool } from "./types";
import { TeamSelect } from "./select";
import { Game } from "./game";

/** The fixed bot strength: the former "Max" tier, always on. */
export const BUDGET = 30000;

interface GameSpec {
  humanIdx: number;
  botIdx: number;
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

  useEffect(() => {
    void (async () => {
      try {
        const [, pd] = await Promise.all([loadEngine(), fetchPool()]);
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
        <div class="loading-pulse">Loading engine…</div>
      </div>
    );
  }
  if (status === "error" || !pool) {
    return (
      <div class="center-screen">
        <div class="error-box">
          <strong>Failed to load</strong>
          <div>{error}</div>
        </div>
      </div>
    );
  }

  if (!game) {
    return (
      <TeamSelect
        pool={pool}
        onStart={(humanIdx, botIdx) => {
          const bot =
            botIdx === "random"
              ? randomSeed32() % pool.teams.length
              : botIdx;
          setGame({ humanIdx, botIdx: bot, n: 1 });
        }}
      />
    );
  }

  return (
    <Game
      key={game.n}
      pool={pool}
      poolJson={poolJsonRef.current}
      humanIdx={game.humanIdx}
      botIdx={game.botIdx}
      onRematch={() => setGame({ ...game, n: game.n + 1 })}
      onNewTeams={() => setGame(null)}
    />
  );
}
