// App shell: engine + meta pool loading, strength + bot-information
// settings, and the select -> game screen switch. A Game instance is keyed
// by game number so rematch / new-teams remount it cleanly.

import { useEffect, useRef, useState } from "preact/hooks";
import { loadEngine, PreviewTables, randomSeed32 } from "./engine";
import { fetchPool } from "./data";
import type { BotInfo, MetaPool } from "./types";
import { TeamSelect } from "./select";
import { Game } from "./game";

export const STRENGTHS = [
  { iters: 1000, label: "Quick (1k iterations)" },
  { iters: 3000, label: "Normal (3k iterations)" },
  { iters: 10000, label: "Strong (10k iterations)" },
  { iters: 30000, label: "Max (30k iterations)" },
];

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
  const tablesRef = useRef<PreviewTables | null>(null);
  const addedPairsRef = useRef(new Set<string>());
  const [strength, setStrength] = useState(() => {
    const v = Number(localStorage.getItem("nc2000-strength"));
    return STRENGTHS.some((s) => s.iters === v) ? v : 3000;
  });
  const [botInfo, setBotInfo] = useState<BotInfo>(() =>
    localStorage.getItem("nc2000-botinfo") === "xray" ? "xray" : "fair",
  );
  const [game, setGame] = useState<GameSpec | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const [dex, pd] = await Promise.all([loadEngine(), fetchPool()]);
        tablesRef.current = new PreviewTables(dex, pd.poolJson);
        poolJsonRef.current = pd.poolJson;
        setPool(pd.pool);
        setStatus("ready");
      } catch (e) {
        setError(String(e));
        setStatus("error");
      }
    })();
  }, []);

  const pickStrength = (iters: number) => {
    setStrength(iters);
    localStorage.setItem("nc2000-strength", String(iters));
  };

  const pickBotInfo = (v: BotInfo) => {
    setBotInfo(v);
    localStorage.setItem("nc2000-botinfo", v);
  };

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
        strength={strength}
        onStrength={pickStrength}
        botInfo={botInfo}
        onBotInfo={pickBotInfo}
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
      tables={tablesRef.current!}
      addedPairs={addedPairsRef.current}
      humanIdx={game.humanIdx}
      botIdx={game.botIdx}
      strength={strength}
      onStrength={pickStrength}
      botInfo={botInfo}
      onRematch={() => setGame({ ...game, n: game.n + 1 })}
      onNewTeams={() => setGame(null)}
    />
  );
}
