import assert from "node:assert/strict";
import { createServer as createHttpServer } from "node:http";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = createHttpServer();
let browser, vite;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  vite = await createServer({
    root,
    server: { middlewareMode: true, hmr: { server, clientPort: server.address().port } },
  });
  server.on("request", vite.middlewares);
  browser = await chromium.launch({
    ...(process.env.NC2000_CHROMIUM_EXE
      ? { executablePath: process.env.NC2000_CHROMIUM_EXE }
      : {}),
  });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: "networkidle" });
  const results = await page.evaluate(async (wasmUrl) => {
    const { BotWorker } = await import("/src/bot.ts");
    const { searchProfile } = await import("/src/search-profile.ts");
    const wasm = await import(wasmUrl);
    await wasm.default();
    const dex = new wasm.Dex();
    const poolJson = await (await fetch("/data/belief-pool-v3/belief-pool.json")).text();
    const catalog = await (await fetch("/data/team-pool-v2/team-pool.json")).json();
    const teams = catalog.teams.slice(0, 2).map((team) => JSON.stringify(team.sets));
    const seed = 7;
    const expected = (iterations, c) => {
      const battle = new wasm.Battle(dex, teams[0], teams[1], "1,2,3,4");
      battle.setLogEnabled(true);
      const search = new wasm.BlindSearcher(battle, 1, poolJson, seed, c);
      search.observe(battle);
      search.step(iterations);
      const policy = JSON.parse(search.rootPolicy());
      search.free();
      battle.free();
      return policy;
    };
    const results = [];
    // Live battles are blind-only: the worker must search with the blind
    // profile's c, and a c = 1.0 search must not reproduce its policy.
    {
      for (const scenario of ["fixed", "early-flush", "late-flush", "background-complete"]) {
        const worker = new BotWorker();
        try {
          await worker.newBattle(teams[0], teams[1], "1,2,3,4", { poolJson, side: 1, seed, compiledPreview: false });
          const budget = 512;
          const promise = worker.search(1, budget, seed, scenario !== "fixed");
          if (scenario === "early-flush") worker.flush();
          if (scenario === "late-flush") setTimeout(() => worker.flush(), 125);
          const result = await Promise.race([
            promise,
            new Promise((_, reject) => setTimeout(() => reject(new Error("worker search timed out")), 60000)),
          ]);
          const iterations = budget;
          if (result.wasmMemoryBytes <= 0 || result.wasmMemoryBytes % 65536 !== 0) throw new Error("missing Wasm memory measurement");
          const profile = searchProfile("blind");
          results.push({ scenario, c: profile.c, iterations: result.policy.iterations,
            expectedIterations: iterations,
            policyMatches: JSON.stringify(result.policy) === JSON.stringify(expected(iterations, profile.c)),
            wrongCoefficientMatches: JSON.stringify(result.policy) === JSON.stringify(expected(iterations, 1)),
          });
        } finally {
          worker.terminate();
        }
      }
    }
    // Input timing must also preserve battle decisions and the persistent RNG.
    {
      const trajectories = [];
      for (const timing of ["early", "background"]) {
        const worker = new BotWorker();
        const truth = new wasm.Battle(dex, teams[0], teams[1], "1,2,3,4");
        try {
          await worker.newBattle(teams[0], teams[1], "1,2,3,4", { poolJson, side: 1, seed, compiledPreview: false });
          const policies = [];
          for (let turn = 0; turn < 3; turn++) {
            const legal = JSON.parse(truth.legalChoices(1));
            if (!legal.length) break;
            const promise = worker.search(1, 1024, seed, true);
            if (timing === "early") worker.flush();
            const result = await promise;
            if (timing === "background") worker.flush();
            if (result.policy.iterations !== 1024) throw new Error("human timing changed battle budget");
            policies.push(result.policy);
            const picks = [];
            const own = JSON.parse(truth.legalChoices(0));
            if (own.length) picks.push([0, own[0].input]);
            picks.push([1, result.best]);
            for (const [side, input] of picks) truth.applyChoice(side, input);
            worker.apply(picks);
          }
          trajectories.push(policies);
        } finally { worker.terminate(); truth.free(); }
      }
      if (JSON.stringify(trajectories[0]) !== JSON.stringify(trajectories[1])) throw new Error("input timing changed the persistent search trajectory");
    }
    for (const scenario of ["fixed", "early-flush", "late-flush", "ponder-cap", "cancel"]) {
      const worker = new BotWorker();
      const truth = new wasm.Battle(dex, teams[0], teams[1], "1,2,3,4");
      try {
        await worker.newBattle(teams[0], teams[1], "1,2,3,4", {
          poolJson, side: 1, seed, endgame: { solverFraction: 0.1, milliseconds: 50 },
        });
        const budget = 512;
        const promise = worker.search(1, budget, seed, scenario !== "fixed");
        if (scenario === "early-flush") worker.flush();
        if (scenario === "late-flush") setTimeout(() => worker.flush(), 125);
        if (scenario === "cancel") {
          worker.cancelAll();
          const result = await Promise.race([promise.then(() => "stale-result"), new Promise(resolve => setTimeout(() => resolve("cancelled"), 200))]);
          if (result !== "cancelled") throw new Error("cancelled endgame search posted a stale result");
          continue;
        }
        const result = await Promise.race([promise, new Promise((_, reject) => setTimeout(() => reject(new Error("timed worker hung")), 10000))]);
        const stats = JSON.parse(result.endgame);
        if (result.wasmMemoryBytes <= 0 || result.wasmMemoryBytes % 65536 !== 0) throw new Error("missing timed Wasm memory measurement");
        if (stats.scope !== "conditional-determinized" || stats.stats.root_overrides !== 0) throw new Error("blind worker reported a full-information policy");
        if (result.ms > 2000 || result.ms < 45) throw new Error(`timed worker duration ${result.ms}`);
        if (scenario === "late-flush" && result.ms < 100) throw new Error("ponder stopped before flush");
        if (scenario === "ponder-cap" && result.ms < 490) throw new Error("ponder failed to extend its budget");
        const own = JSON.parse(truth.legalChoices(0))[0].input;
        truth.applyChoice(0, own);
        truth.applyChoice(1, result.best);
        worker.apply([[0, own], [1, result.best]]);
        const next = await worker.search(1, budget, seed + 1, false);
        if (!JSON.parse(truth.legalChoices(1)).some(choice => choice.input === next.best)) throw new Error(`illegal endgame move ${next.best}`);
      } finally {
        truth.free();
        worker.terminate();
      }
    }
    {
      const fork = await (await fetch("/data/forks/4296-t11.json")).text();
      const record = JSON.parse(fork);
      const worker = new BotWorker();
      const truth = wasm.Battle.fromFork(dex, fork, 11);
      try {
        await worker.newFork(fork, 11, { poolJson, side: record.position.side, seed, mode: "blind", endgame: { solverFraction: 0.1, milliseconds: 100 } });
        const result = await worker.search(record.position.side, 512, seed, false);
        if (!JSON.parse(truth.legalChoices(record.position.side)).some(choice => choice.input === result.best)) throw new Error("illegal timed fork move");
        if (JSON.parse(result.endgame).scope !== "conditional-determinized") throw new Error("fork bypassed the conditional solver");
      } finally {
        truth.free();
        worker.terminate();
      }
    }
    {
      const { SolverWorker } = await import("/src/solver-client.ts");
      const previous = location.search;
      history.replaceState({}, "", "?solver&endgame=10&endgame-ms=100&endgame-nodes=16");
      const worker = new SolverWorker();
      try {
        const own = { name: "Pikachu", species: "Pikachu", level: 50, gender: "M", item: "", ability: "", moves: ["Splash", "Seismic Toss"] };
        const foe = { ...own, moves: ["Splash"] };
        const mon = { species: "Pikachu", level: 50, gender: "M", item_flag: false, hp_num: 100, hp_den: 100, appeared: true, active: true };
        const spec = JSON.stringify({ schema: "nc2000-position-v1", side: 0, turn: 999, own_sets: [own], sides: [0, 1].map(side => ({ mons: [{ ...mon, ...(side === 0 ? { hp_num: 1, hp_den: 125, hp_exact: 1 } : {}) }], active: 0, party: [0] })) });
        const result = await worker.solve({ spec, poolJson: JSON.stringify({ teams: [{ id: "foe", sets: [foe] }] }), seed: 73, budget: 512, plies: 0 });
        if (JSON.parse(result.endgame).stats.value_replacements < 1) throw new Error("analysis worker did not use solved values");
        if (JSON.parse(result.endgame).stats.peak_entries > 16) throw new Error("analysis worker exceeded its configured cache cap");
        const extended = await worker.extend(1024, 0);
        if (extended.report.iterations <= result.report.iterations) throw new Error("timed analysis did not extend");
      } finally {
        worker.dispose();
        history.replaceState({}, "", previous || "/");
      }
    }
    dex.free();
    return results;
  }, "/@fs" + fileURLToPath(new URL("../../crates/wasm/pkg-web/nc2000_wasm.js", import.meta.url)));
  for (const result of results) {
    assert.equal(result.iterations, result.expectedIterations, JSON.stringify(result));
    assert.equal(result.policyMatches, true, JSON.stringify(result));
    assert.equal(result.wrongCoefficientMatches, false, JSON.stringify(result));
  }
  console.log(JSON.stringify(results, null, 2));
} finally {
  await browser?.close();
  await vite?.close();
  await new Promise((resolve) => server.close(resolve));
}
