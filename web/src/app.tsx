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
// URL at module load — the query string is the only door, and nothing the
// user can press moves it; a GameSpec still carries the mode its game
// started under, which is the guarantee game.tsx is written against. The
// team pool is state: one file replaces the pool everywhere it is read
// (team-pool.ts). The bundled pool is held next to the active one so going
// back to it is a state change rather than a second trip to the network —
// and, since the swap is now blind-only, so that open mode has the
// untouched pool to play no matter what the user loaded (see `activePool`).
//
// `?nash` (META-NASH v1's conclusion) is blind play with one substitution
// and one subtraction. The substitution: the opponent is drawn from the
// solved three-team mixture instead of uniformly from the pool — the draw
// rule lives in one place here, `drawOpponent`, because the start screen and
// the rematch must roll identically or the mixed strategy is not the thing
// being played. The subtraction: nothing is configurable, so nash pins the
// bundled pool exactly as open mode does and never passes a belief prior.
// What the bot ASSUMES about its opponent is untouched by all of this
// (docs/META-NASH-V1.md §4) — only which team it brings.

import { useEffect, useState } from "preact/hooks";
import { loadEngine } from "./engine";
import {
  fetchBeliefPool,
  fetchDexJson,
  fetchI18nJa,
  fetchNashArtifact,
  fetchPool,
} from "./data";
import { loadSetDex } from "./set-info";
import { randomPoolTeam, type SelectedTeam } from "./pool-pick";
import { drawNashTeam, parseNashArtifact, type NashMix } from "./nash-mix";
import { infoModeOf, readDoor, type Door, type InfoMode } from "./info-mode";
import {
  clearStoredPool,
  loadStoredPool,
  parsePoolText,
  type LoadedPool,
} from "./team-pool";
import { loadStoredPrior, type StoredPrior } from "./belief-prior";
import { StartScreen } from "./select";
import { Game } from "./game";
import { Solver } from "./solver";
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

/** This page load's door and the information mode it implies. Read at
 * module scope because that is the truth about them: the query string cannot
 * change without a navigation, and holding either in state would suggest
 * something here could flip it. */
const DOOR: Door = readDoor();
const MODE: InfoMode = infoModeOf(DOOR);
const NASH = DOOR === "nash";
/** The study board is not a way to play: it replaces the whole screen, so it
 * is checked before any of the game state below is consulted. */
const SOLVER = DOOR === "solver";

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
 * loading right now. The bundled pool then stands, as it did before.
 *
 * Dropped *and deleted*, though. This runs after the engine is up, so a
 * failure here is a verdict on the record, not on the browser: it will fail
 * the same way on every future load, costing a full validator pass each
 * time, while the only control that could remove it — the panel's reset
 * button — is disabled exactly when the bundled pool is in play. A record
 * that cannot be adopted and cannot be cleared is unreachable forever, so
 * the read is what clears it. */
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
  // The pool the user chose, and the bundled one it can always fall back to.
  // The same object until a file is loaded, but two references: "use the
  // bundled pool" has to work after a swap, and the bundled pool is already
  // in memory — refetching it to get it back would be the one path that can
  // fail offline. Which of the two is actually played is `activePool`, below.
  const [bundled, setBundled] = useState<LoadedPool | null>(null);
  const [loadedPool, setLoadedPool] = useState<LoadedPool | null>(null);
  const [game, setGame] = useState<GameSpec | null>(null);
  // The solved mixture, on the `?nash` door only. Null everywhere else, and
  // never null on a nash page that got past boot: a failure to load or
  // validate it fails the page (see the boot effect).
  const [nashMix, setNashMix] = useState<NashMix | null>(null);
  // The nash door's belief candidate pool (EXP-PRIOR-EXPLOIT v1): the
  // searcher's candidate set only. Own-team draw, screen lists and baked
  // tables keep reading the bundled pool — this is the one documented
  // exception to "everything downstream of the pool moves together", and
  // it exists because identification is the prior's entire measured value
  // while the pool's other roles must not inherit the study's exploiter
  // teams (drawing them as the bot's own team would be a strength
  // regression, Route A's non-regression FAIL).
  const [nashBeliefJson, setNashBeliefJson] = useState<string | null>(null);
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
        // The nash artifact rides along in the same wave — one door's
        // 11 KB, fetched only on that door, never on the product page.
        const [, pd, , , nashText, beliefPd] = await Promise.all([
          loadEngine(),
          fetchPool(),
          loadJaNames(fetchI18nJa),
          loadSetDex(fetchDexJson),
          NASH ? fetchNashArtifact() : Promise.resolve(""),
          // The solver reasons with the same shipped belief pool the nash
          // door hands its bot: identification is the whole of what the
          // opponent panel's "moves shown" field buys, and the wider table
          // is what it is bought against.
          NASH || SOLVER ? fetchBeliefPool() : Promise.resolve(null),
        ]);
        const bundledPool: LoadedPool = {
          name: null,
          pool: pd.pool,
          poolJson: pd.poolJson,
        };
        setBundled(bundledPool);
        // Parsed after the engine is up, because validating the mixture is
        // a wasm call. Strict: a mixture that cannot play takes the page
        // down with it rather than leaving `?nash` running as plain blind
        // under a name that promises otherwise.
        if (NASH) {
          const parsed = parseNashArtifact(nashText);
          if (!parsed.ok) throw new Error(parsed.errors.join("; "));
          setNashMix(parsed.mix);
        }
        if (NASH || SOLVER) {
          setNashBeliefJson(beliefPd ? beliefPd.poolJson : null);
        }
        // Only now: re-validating a stored pool runs the wasm validator,
        // which the engine load above is what makes available. Restored in
        // either mode — the record belongs to the user, not to the mode, so
        // coming back to `?blind` finds the file still loaded, and a record
        // that has gone bad gets swept whichever door they came in by.
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
  if (
    status === "error" ||
    !loadedPool ||
    !bundled ||
    (NASH && (!nashMix || !nashBeliefJson)) ||
    (SOLVER && !nashBeliefJson)
  ) {
    return (
      <div class="center-screen">
        <div class="error-box">
          <strong>{ui().failedLoad}</strong>
          <div>{error}</div>
        </div>
      </div>
    );
  }

  // The one pool everything below reads. A loaded file stays in state — it
  // is the user's, and `?blind` will find it again — but only blind mode
  // plays it: whoever opens `/` gets the bundled pool, every time. Keeping
  // the file live in open mode while hiding the control that loaded it would
  // leave a stored pool quietly rewriting the public team lists, with no sign
  // of why and nothing on screen to undo it. This is the same line the belief
  // prior already sits behind, where an open game refuses the table outright
  // rather than half-using it.
  // Nash sits on the bundled side of this line with open mode: the mode is
  // a fixed configuration or it is not the conclusion, so a pool file the
  // user once loaded through `?blind` must not quietly redefine the bot's
  // candidate set here. `?blind` still finds that file, untouched.
  const activePool = MODE === "blind" && !NASH && !SOLVER ? loadedPool : bundled;

  // The study board takes over the page. Nothing below it runs: there is no
  // game, no opponent draw, and no team selection — a position is typed,
  // not played into.
  if (SOLVER) {
    return (
      <Solver
        pool={bundled.pool}
        poolJson={nashBeliefJson!}
        locale={loc}
        onLocale={(l) => {
          setLocale(l);
          setLoc(l);
        }}
      />
    );
  }

  /** The opponent draw, in one place because two callers must roll the same
   * way: pressing Start, and every rematch. Nash samples the solved mixture;
   * plain blind draws uniformly from the pool. (Open mode never comes here —
   * its opponent is whatever the human picked on the screen.) */
  const drawOpponent = (): SelectedTeam =>
    NASH && nashMix ? drawNashTeam(nashMix) : randomPoolTeam(activePool.pool);

  if (!game) {
    return (
      <StartScreen
        loadedPool={activePool}
        bundledPool={bundled}
        onPool={setLoadedPool}
        locale={loc}
        onLocale={(l) => {
          setLocale(l);
          setLoc(l);
        }}
        mode={MODE}
        nash={NASH}
        nashMix={nashMix}
        drawOpponent={drawOpponent}
        prior={prior}
        onPrior={setPrior}
        onStart={(human, bot) => setGame({ human, bot, n: 1, mode: MODE })}
      />
    );
  }

  return (
    <Game
      key={game.n}
      // The searcher's candidate pool. On the nash door this is the
      // belief-pool-v1 artifact (curated 32 + known strong candidates) —
      // reaching ONLY the belief: nash is blind, and a blind game never
      // fetches pair tables (game.tsx), so no baked artifact is ever read
      // against these indices.
      poolJson={NASH && nashBeliefJson ? nashBeliefJson : activePool.poolJson}
      // Baked artifacts are indexed by the bundled pool's rank order, so a
      // swapped pool's indices name different teams entirely. Open mode is
      // never custom by construction, which is what gives the public build
      // its pair tables back.
      poolIsCustom={activePool.name !== null}
      humanTeam={game.human}
      botTeam={game.bot}
      mode={game.mode}
      // The prior only ever reaches a blind game: in open mode the searcher
      // pins the human's real team and refuses the table outright. Nash is
      // blind and still gets none — the mode ships one configuration, and a
      // table left in storage by an earlier `?blind` visit is exactly the
      // kind of invisible state it must not inherit.
      priorJson={
        game.mode === "blind" && !NASH ? prior?.json : undefined
      }
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
                bot: g.mode === "blind" ? drawOpponent() : g.bot,
              },
        )
      }
      onNewTeams={() => setGame(null)}
    />
  );
}
