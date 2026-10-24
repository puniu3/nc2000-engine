// Real WASM/Worker checks for live selection, deterministic resume and replay.
import assert from "node:assert/strict";
import { createServer as http } from "node:http";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = http();
let vite, browser;
try {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  vite = await createServer({ root, server: { middlewareMode: true, hmr: { server } } });
  server.on("request", vite.middlewares);
  browser = await chromium.launch({ args: ["--mute-audio"],
    ...(process.env.NC2000_CHROMIUM_EXE ? { executablePath: process.env.NC2000_CHROMIUM_EXE } : {}) });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/?evaluate`, { waitUntil: "networkidle" });
  const result = await page.evaluate(async wasmUrl => {
    const wasm = await import(wasmUrl);
    await wasm.default();
    const { default: Evaluator } = await import("/src/evaluate-worker.ts?worker");
    const { appendGame, completedGames, pairSchedule } = await import("/src/evaluate-core.ts");
    const { liveBudget, DEFAULT_STRENGTH } = await import("/src/search-profile.ts");
    const { BotWorker } = await import("/src/bot.ts");
    const { loadEngine } = await import("/src/engine.ts");
    const { readTeam } = await import("/src/evaluate-input.ts");
    await loadEngine();
    const dex = new wasm.Dex();
    const beliefJson = await (await fetch("/data/belief-pool-v3/belief-pool.json")).text();
    const catalog = await (await fetch("/data/team-pool-v2/team-pool.json")).json();
    const team = sets => readTeam(JSON.stringify(sets));
    const makeRun = (player, opponent, previewIterations, iterations, turnLimit) => ({
      version: 2, id: "worker-check", createdAt: "2026-10-24", seed: 9182,
      targetPairs: 1, pairs: [], config: { build: "test", beliefHash: "test",
        player: team(player), opponents: [{ ...team(opponent), id: "opponent", weight: 1 }],
        previewIterations, iterations, turnLimit, c: 0.4 },
    });
    const execute = (initial, stopAfter = Infinity) => new Promise((resolve, reject) => {
      const worker = new Evaluator();
      let current = initial, received = 0;
      const timer = setTimeout(() => finish(new Error("evaluation worker timeout")), 300000);
      const finish = error => {
        clearTimeout(timer); worker.terminate();
        if (error) reject(error); else resolve({ run: current, received });
      };
      worker.onerror = e => finish(new Error(e.message));
      worker.onmessage = ({ data }) => {
        try {
          if (data.type === "game") {
            current = appendGame(current, data.record);
            if (++received === stopAfter) { finish(); return; }
            worker.postMessage({ type: "ack" });
          }
          if (data.type === "done") finish();
          if (data.type === "error") finish(new Error(data.message));
        } catch (e) { finish(e); }
      };
      worker.postMessage({ type: "start", run: initial, beliefJson });
    });
    const check = (condition, message) => { if (!condition) throw new Error(message); };
    const verifyReplays = run => {
      for (const { game } of completedGames(run)) {
        const replay = new wasm.Kifu(dex, game.kifu);
        try {
          const meta = JSON.parse(replay.scenes());
          check(meta.turns === game.turns && meta.botSide === game.playerSide && meta.info === "blind", "replay perspective/turns changed");
          const expected = game.outcome === "cap" ? null : game.outcome === "tie" ? "Tie"
            : (game.outcome === "win" ? game.playerSide : 1 - game.playerSide) === 0 ? "P1Win" : "P2Win";
          check(meta.outcome === expected, "replay outcome changed");
          for (const { round } of meta.scenes) {
            const scene = JSON.parse(replay.scene(round));
            check(JSON.stringify(scene.log) === JSON.stringify(game.log.slice(0, scene.log.length)), "replay log is not the recorded prefix");
          }
        } finally { replay.free(); }
      }
    };
    try {
      // Catalog matchups that the normal bot can answer from the artifact
      // must still search both selections here, including a non-32 budget.
      const initial = makeRun(catalog.teams[0].sets, catalog.teams[1].sets, 65, 7, 3);
      const full = await execute(initial);
      verifyReplays(full.run);
      for (const { game } of completedGames(full.run))
        check(game.selections.length === 2 && game.selections.every(s => s.iterations === 65), "selection skipped live search or used battle budget");
      const half = await execute(initial, 1);
      check(half.run.pending && half.run.pairs.length === 0, "first game was not retained independently");
      const resumed = await execute(half.run);
      check(resumed.received === 1 && JSON.stringify(resumed.run) === JSON.stringify(full.run), "resume replayed or changed a completed game");
      const scheduled = pairSchedule(initial.seed, 0, initial.config.opponents);
      const parity = [];
      for (const side of [0, 1]) {
        const bot = new BotWorker();
        try {
          await bot.newBattle(JSON.stringify(initial.config.player.sets), JSON.stringify(initial.config.opponents[0].sets),
            wasm.deriveBattleSeed(scheduled.battleSeed), { poolJson: beliefJson, side, seed: scheduled.agentSeeds[side], compiledPreview: false });
          const normal = await bot.search(side, 65, 99, false);
          const selected = full.run.pairs[0].games[0].selections[side];
          check(normal.src === "search" && normal.best === selected.input && normal.policy.iterations === selected.iterations, "selection differs from normal live bot");
          parity.push(normal.best);
        } finally { bot.terminate(); }
      }
      const production = makeRun([{ species: "Mewtwo", level: 100, moves: ["Psychic"] }],
        [{ species: "Magikarp", level: 1, moves: ["Splash"] }], liveBudget(DEFAULT_STRENGTH, true), liveBudget(DEFAULT_STRENGTH), 500);
      const actual = await execute(production);
      verifyReplays(actual.run);
      check(actual.run.pairs[0].games.every(g => g.outcome === "win" && g.selections.every(s => s.iterations === production.config.previewIterations)), "production budgets/results wrong");
      const stall = [{ species: "Chansey", level: 100, item: "Leftovers",
        moves: ["Splash", "Growl", "Harden", "Defense Curl"],
        ivs: { atk: 0 }, evs: { atk: 0 } }];
      const capped = await execute(makeRun(stall, stall, 1, 1, 500));
      verifyReplays(capped.run);
      check(capped.run.pairs[0].games.every(g => g.turns === 500 && g.outcome === "cap"), "500-turn logs were truncated or assigned a winner");
      return { parity, resumedGames: resumed.received, capped: full.run.pairs[0].games.map(g => g.outcome),
        longLogs: capped.run.pairs[0].games.map(g => ({ turns: g.turns, lines: g.log.length, codeLength: g.kifu.length })),
        production: { preview: production.config.previewIterations, battle: production.config.iterations,
          games: actual.run.pairs[0].games.map(g => ({ turns: g.turns, outcome: g.outcome, selections: g.selections })) } };
    } finally { dex.free(); }
  }, `/@fs${fileURLToPath(new URL("../../crates/wasm/pkg-web/nc2000_wasm.js", import.meta.url))}`);
  assert.equal(result.production.preview, 729000);
  assert.equal(result.production.battle, 81000);
  assert.deepEqual(result.capped, ["cap", "cap"]);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser?.close();
  await vite?.close();
  server.close();
}
