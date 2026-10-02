import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer as createHttpServer } from "node:http";
import { readFileSync, mkdirSync, writeFileSync, appendFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const root = fileURLToPath(new URL("../..", import.meta.url));
const protocolPath = resolve(process.argv[2] ?? join(root, "findings/endgame-integration-v1/web-timing-protocol.json"));
const destination = resolve(process.argv[3] ?? join(root, "tmp/endgame-web-timing-v1"));
const protocolBytes = readFileSync(protocolPath);
const protocol = JSON.parse(protocolBytes);
assert.equal(protocol.schema, "nc2000-web-timing-v1");
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
for (const name of ["catalog", "prior"]) assert.equal(sha(readFileSync(join(root, protocol[name]))), protocol[`${name}_sha256`]);
mkdirSync(destination);
writeFileSync(join(destination, "protocol.json"), protocolBytes);
const output = join(destination, "timing.jsonl");
const server = createHttpServer();
let browser, vite;
try {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  vite = await createServer({ root: join(root, "web"), server: { middlewareMode: true, hmr: { server, clientPort: server.address().port } } });
  server.on("request", vite.middlewares);
  browser = await chromium.launch({ ...(process.env.NC2000_CHROMIUM_EXE ? { executablePath: process.env.NC2000_CHROMIUM_EXE } : {}) });
  const page = await browser.newPage();
  await page.exposeFunction("recordTiming", row => appendFileSync(output, JSON.stringify(row) + "\n"));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  writeFileSync(join(destination, "manifest.json"), JSON.stringify({
    protocol_sha256: sha(protocolBytes), wasm_sha256: sha(readFileSync(join(root, "crates/wasm/pkg-web/nc2000_wasm_bg.wasm"))),
    revision: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
    dirty: execFileSync("git", ["status", "--porcelain"], { cwd: root, encoding: "utf8" }).trim(),
    browser: browser.version(), platform: process.platform, arch: process.arch,
  }, null, 2) + "\n");
  await page.evaluate(async ({ protocol, wasmUrl }) => {
    const { BotWorker } = await import("/src/bot.ts");
    const { PRODUCT_ITERATIONS, searchProfile } = await import("/src/search-profile.ts");
    if (PRODUCT_ITERATIONS !== protocol.iterations || searchProfile("blind").c !== protocol.c) throw new Error("timing protocol differs from shipped profile");
    const wasm = await import(wasmUrl);
    await wasm.default();
    const dex = new wasm.Dex();
    const catalog = await (await fetch(`/${protocol.catalog}`)).json();
    const prior = await (await fetch(`/${protocol.prior}`)).text();
    const weights = catalog.teams.map(team => team.drawWeight);
    if (!weights.every(w => Number.isFinite(w) && w >= 0)) throw new Error("invalid drawWeight");
    const weight = weights.reduce((a, b) => a + b, 0);
    let state = BigInt(protocol.seed);
    const mask = (1n << 64n) - 1n;
    function next() {
      state = (state + 0x9e3779b97f4a7c15n) & mask;
      let value = state;
      value = ((value ^ (value >> 30n)) * 0xbf58476d1ce4e5b9n) & mask;
      value = ((value ^ (value >> 27n)) * 0x94d049bb133111ebn) & mask;
      return (value ^ (value >> 31n)) & mask;
    }
    function draw() {
      let sample = Number(next() >> 11n) / 2 ** 53 * weight;
      return catalog.teams[weights.findIndex(w => (sample -= w) < 0)];
    }
    async function bounded(promise) {
      let timeout;
      try { return await Promise.race([promise, new Promise((_, reject) => { timeout = setTimeout(() => reject(new Error("worker exceeded 60-second timing guard")), 60000); })]); }
      finally { clearTimeout(timeout); }
    }
    await window.recordTiming({ type: "environment", userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, visibility: document.visibilityState });
    try {
      for (let game = 0; game < protocol.games; game++) {
        const teams = [draw(), draw()];
        const encoded = teams.map(team => JSON.stringify(team.sets));
        const value = next();
        const seed = [48n, 32n, 16n, 0n].map(shift => String((value >> shift) & 65535n)).join(",");
        const agentSeeds = [Number(next() & 0xffffffffn), Number(next() & 0xffffffffn)];
        const battle = new wasm.Battle(dex, encoded[0], encoded[1], seed);
        const workers = [new BotWorker(), new BotWorker()];
        await window.recordTiming({ type: "game", game, teams: teams.map(t => t.id), seed, agentSeeds });
        try {
          for (let side = 0; side < 2; side++) await workers[side].newBattle(encoded[0], encoded[1], seed, { poolJson: prior, side, seed: agentSeeds[side] });
          let decision = 0;
          while (!battle.outcome()) {
            const needs = JSON.parse(battle.needsChoice());
            const choices = [];
            const turn = battle.turn();
            const view = JSON.parse(battle.stateView());
            const alive = view.sides.map(s => s.party.filter(p => !p.fainted && p.hp > 0).length);
            for (const side of [decision % 2, 1 - decision % 2]) {
              if (!needs[side]) continue;
              const legal = JSON.parse(battle.legalChoices(side));
              const started = performance.now();
              const result = await bounded(workers[side].search(side, protocol.iterations, agentSeeds[side], false));
              const elapsed = performance.now() - started;
              const choice = battle.resolveChoice(side, result.best);
              if (!choice || !legal.some(a => a.input === choice)) throw new Error(`illegal baseline choice ${result.best}`);
              choices[side] = choice;
              await window.recordTiming({ type: "decision", game, decision, side, turn, alive,
                phase: turn === 0 ? "preview" : legal.length === 1 ? "forced" : alive.every(n => n <= 2) ? "small-endgame" : "normal",
                iterations: result.policy.iterations, request_ms: elapsed, search_ms: result.ms, visibility: document.visibilityState, choice });
            }
            const picks = choices.flatMap((choice, side) => choice ? [[side, choice]] : []);
            for (const [side, choice] of picks) battle.applyChoice(side, choice);
            for (const worker of workers) worker.apply(picks);
            decision++;
            if (decision > 4000) throw new Error("timing game exceeded emergency guard");
          }
          await window.recordTiming({ type: "result", game, outcome: battle.outcome(), turn: battle.turn() });
        } finally {
          workers.forEach(w => w.terminate());
          battle.free();
        }
      }
    } finally { dex.free(); }
  }, { protocol, wasmUrl: "/@fs" + join(root, "crates/wasm/pkg-web/nc2000_wasm.js") });
  console.log(`Completed ${protocol.games} baseline Web timing games.`);
} finally {
  await browser?.close();
  await vite?.close();
  await new Promise(resolve => server.close(resolve));
}
