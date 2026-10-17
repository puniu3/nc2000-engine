// Exercise the rebuilt WASM through the real worker, including a protocol
// fork. This tests execution semantics, not playing strength.
import assert from "node:assert/strict";
import { createServer as createHttpServer } from "node:http";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const server = createHttpServer();
let browser, vite;
try {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  vite = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)),
    server: { middlewareMode: true, hmr: { server, clientPort: server.address().port } } });
  server.on("request", vite.middlewares);
  browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async ({ wasmUrl, compiledDisabled }) => {
    const { BotWorker } = await import("/src/bot.ts");
    const { searchProfile } = await import("/src/search-profile.ts");
    const wasm = await import(wasmUrl);
    await wasm.default();
    const dex = new wasm.Dex();
    const poolJson = await (await fetch("/data/belief-pool-v3/belief-pool.json")).text();
    const catalog = await (await fetch("/data/team-pool-v2/team-pool.json")).json();
    const { compiledPreviewMatches } = await import("/src/compiled-preview.ts");
    if (await compiledPreviewMatches(poolJson, wasm.embeddedDexJson()) !== !compiledDisabled)
      throw new Error("compiled-preview build setting was not enforced at runtime");
    const teams = catalog.teams.slice(0, 2).map(t => JSON.stringify(t.sets));
    const seed = 7, budget = 512, rows = [];
    const equal = (a, b, label) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(label); };
    const check = (result, legal) => {
      const actions = result.playPolicy?.actions;
      if (!actions?.length || Math.abs(actions.reduce((s, a) => s + a.probability, 0) - 1) > 1e-12)
        throw new Error("execution probabilities do not sum to one");
      if (actions.some(a => !Number.isFinite(a.probability) || a.probability <= 0 || !legal.includes(a.input)))
        throw new Error("invalid execution support");
      if (!actions.some(a => a.input === result.best)) throw new Error("pick outside execution support");
    };
    for (const mode of ["legacy", "visit-sample", "estimated-mix"]) {
      const trajectories = [];
      for (const timing of ["early-flush", "complete"]) {
        const worker = new BotWorker();
        const battle = new wasm.Battle(dex, teams[0], teams[1], "1,2,3,4");
        const reference = new wasm.BlindSearcher(battle, 1, poolJson, seed, searchProfile("blind").c);
        const trajectory = [];
        try {
          await worker.newBattle(teams[0], teams[1], "1,2,3,4", {
            poolJson, side: 1, seed, compiledPreview: false, actionPolicy: mode,
          });
          for (let turn = 0; turn < 3; turn++) {
            reference.observe(battle); reference.step(budget);
            const prior = JSON.parse(reference.playPolicy(mode));
            const expected = JSON.parse(reference.commitDecision(mode));
            equal(prior, expected.policy, "inspection changed the policy");
            equal(JSON.parse(reference.commitDecision("legacy")), expected, "commit redrew");
            const pending = worker.search(1, budget, seed, true);
            if (timing === "early-flush") worker.flush();
            const result = await pending;
            if (timing === "complete") worker.flush();
            equal(result.policy, JSON.parse(reference.rootPolicy()), "worker search diverged");
            equal(result.playPolicy, expected.policy, "worker dropped/altered distribution");
            equal(result.best, expected.input, "worker did not use committed sample");
            check(result, JSON.parse(battle.legalChoices(1)).map(a => a.input));
            if (mode === "legacy") equal(result.best, reference.best(), "legacy action changed");
            if (mode === "visit-sample" && turn === 0 && result.playPolicy.actions.length < 2)
              throw new Error("visit sampling collapsed to one action");
            trajectory.push({ best: result.best, policy: result.playPolicy });
            const picks = [[0, JSON.parse(battle.legalChoices(0))[0].input], [1, result.best]];
            for (const [side, input] of picks) battle.applyChoice(side, input);
            worker.apply(picks);
          }
        } finally { worker.terminate(); reference.free(); battle.free(); }
        trajectories.push(trajectory);
      }
      equal(trajectories[0], trajectories[1], "human timing changed execution policy/draws");
      rows.push({ path: "live", mode, decisions: 6, timingInvariant: true,
        supportSizes: trajectories[0].map(r => r.policy.actions.length) });
    }
    const fork = await (await fetch("/data/forks/4296-t11.json")).text();
    const record = JSON.parse(fork), side = record.position.side;
    for (const mode of ["legacy", "visit-sample", "estimated-mix"]) {
      const worker = new BotWorker();
      const battle = wasm.Battle.fromFork(dex, fork, 11);
      const reference = wasm.ProtocolSearcher.fromFork(dex, fork, poolJson, seed, searchProfile("blind").c);
      try {
        await worker.newFork(fork, 11, { poolJson, side, seed, mode: "blind", actionPolicy: mode });
        reference.step(budget);
        const expected = JSON.parse(reference.commitDecision(mode));
        const result = await worker.search(side, budget, seed, false);
        check(result, JSON.parse(battle.legalChoices(side)).map(a => a.input));
        equal(result.best, battle.resolveChoice(side, expected.input), "fork did not sample committed policy");
        equal(result.policy, JSON.parse(reference.rootPolicy()), "fork search diverged");
        const mapped = { ...expected.policy, actions: expected.policy.actions.map(a => ({ ...a, input: battle.resolveChoice(side, a.input) })) };
        equal(result.playPolicy, mapped, "fork distribution projection mismatch");
        equal(JSON.parse(reference.commitDecision("legacy")), expected, "protocol commit redrew");
        const position = reference.positionSpec();
        reference.setPosition(position);
        equal(JSON.parse(reference.commitDecision("legacy")).policy.source, "legacy", "setPosition retained stale policy");
        reference.onRequest('{"wait":true}');
        equal(JSON.parse(reference.commitDecision(mode)), null, "wait retained stale decision");
        rows.push({ path: "fork", mode, decisions: 1, support: result.playPolicy.actions.length });
      } finally { worker.terminate(); reference.free(); battle.free(); }
    }
    dex.free();
    return rows;
  }, { wasmUrl: "/@fs" + fileURLToPath(new URL("../../crates/wasm/pkg-web/nc2000_wasm.js", import.meta.url)),
    compiledDisabled: process.env.NC2000_DISABLE_COMPILED_PREVIEW === "1" });
  assert.equal(result.length, 6);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser?.close();
  await vite?.close();
  await new Promise(resolve => server.close(resolve));
}
