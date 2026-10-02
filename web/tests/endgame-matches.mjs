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
assert.equal(process.argv.length, 4, "usage: node tests/endgame-matches.mjs PROTOCOL.json NEW-DIRECTORY");
const protocolBytes = readFileSync(resolve(process.argv[2]));
const protocol = JSON.parse(protocolBytes);
assert.equal(protocol.schema, "nc2000-web-matches-v1");
assert(protocol.pairs > 0 && protocol.threads > 0);
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
for (const name of ["catalog", "prior"]) assert.equal(sha(readFileSync(join(root, protocol[name]))), protocol[`${name}_sha256`]);
for (const arm of [protocol.candidate, protocol.baseline]) {
  assert(["raw", "public", "empty"].includes(arm.information));
  assert(arm.solver_fraction >= 0 && arm.solver_fraction <= .2);
  assert(arm.information === "public" || arm.solver_fraction === 0);
}
if (!protocol.calibration) {
  const selection = JSON.parse(readFileSync(resolve(root, protocol.candidate_selection)));
  assert.equal(selection.complete, true);
  const chosen = selection.arms[selection.selected_for_confirmation];
  assert.equal(protocol.candidate.solver_fraction, chosen.protocol.candidate.solver_fraction);
  execFileSync("git", ["diff", "--quiet", chosen.revision, "--", "crates/bot/src", "crates/engine/src", "data/gen2stadium2.json"], { cwd: root });
}
const destination = resolve(process.argv[3]);
mkdirSync(destination);
writeFileSync(join(destination, "protocol.json"), protocolBytes);
const catalog = JSON.parse(readFileSync(join(root, protocol.catalog)));
const weight = catalog.teams.reduce((sum, team) => sum + team.drawWeight, 0);
assert(weight > 0 && catalog.teams.every(team => Number.isFinite(team.drawWeight) && team.drawWeight >= 0));
const mask = (1n << 64n) - 1n;
let state = BigInt(protocol.seed);
function next() {
  state = (state + 0x9e3779b97f4a7c15n) & mask;
  let value = state;
  value = ((value ^ (value >> 30n)) * 0xbf58476d1ce4e5b9n) & mask;
  value = ((value ^ (value >> 27n)) * 0x94d049bb133111ebn) & mask;
  return (value ^ (value >> 31n)) & mask;
}
function draw() {
  let sample = Number(next() >> 11n) / 2 ** 53 * weight;
  return catalog.teams.findIndex(team => (sample -= team.drawWeight) < 0);
}
const schedule = Array.from({ length: protocol.pairs }, (_, pair) => {
  const teams = [draw(), draw()];
  const value = next();
  const seed = [48n, 32n, 16n, 0n].map(shift => String((value >> shift) & 65535n)).join(",");
  return { pair, teams, seed, agentSeed: Number(next() & 0xffffffffn) };
});
const server = createHttpServer();
let browser, vite;
try {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  vite = await createServer({ root: join(root, "web"), server: { middlewareMode: true, hmr: { server, clientPort: server.address().port } } });
  server.on("request", vite.middlewares);
  browser = await chromium.launch({ ...(process.env.NC2000_CHROMIUM_EXE ? { executablePath: process.env.NC2000_CHROMIUM_EXE } : {}) });
  writeFileSync(join(destination, "manifest.json"), JSON.stringify({ protocol, protocol_sha256: sha(protocolBytes),
    wasm_sha256: sha(readFileSync(join(root, "crates/wasm/pkg-web/nc2000_wasm_bg.wasm"))),
    revision: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
    dirty: execFileSync("git", ["status", "--porcelain"], { cwd: root, encoding: "utf8" }).trim(),
    browser: browser.version(), platform: process.platform, arch: process.arch, schedule,
  }, null, 2) + "\n");
  let cursor = 0;
  await Promise.all(Array.from({ length: protocol.threads }, async () => {
    const page = await browser.newPage();
    await page.exposeFunction("takeMatch", () => schedule[cursor++] ?? null);
    await page.exposeFunction("recordMatch", row => {
      assert(Number.isInteger(row.pair) && row.pair >= 0 && row.pair < protocol.pairs && [0, 1].includes(row.leg));
      const path = join(destination, `pair-${String(row.pair).padStart(4, "0")}-${row.leg}.jsonl`);
      if (row.type === "game") writeFileSync(path, JSON.stringify(row) + "\n", { flag: "wx" });
      else appendFileSync(path, JSON.stringify(row) + "\n");
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async ({ protocol, catalog, wasmUrl }) => {
      const { BotWorker } = await import("/src/bot.ts");
      const { PRODUCT_ITERATIONS, searchProfile } = await import("/src/search-profile.ts");
      if (!protocol.calibration && PRODUCT_ITERATIONS !== protocol.iterations) throw new Error("confirmation must use the shipped baseline iterations");
      if (searchProfile("blind").c !== protocol.c) throw new Error("wrong exploration constant");
      const wasm = await import(wasmUrl);
      await wasm.default();
      const dex = new wasm.Dex();
      const prior = await (await fetch(`/${protocol.prior}`)).text();
      async function bounded(promise) {
        let timeout;
        try { return await Promise.race([promise, new Promise((_, reject) => { timeout = setTimeout(() => reject(new Error("Web decision exceeded 60 seconds")), 60000); })]); }
        finally { clearTimeout(timeout); }
      }
      try {
        for (;;) {
          const job = await window.takeMatch();
          if (!job) break;
          const teams = job.teams.map(index => catalog.teams[index]);
          const encoded = teams.map(team => JSON.stringify(team.sets));
          for (const leg of [job.pair % 2, 1 - job.pair % 2]) {
            const arms = [0, 1].map(side => side === leg ? protocol.candidate : protocol.baseline);
            const workers = arms.map(arm => arm.information === "empty" ? null : new BotWorker());
            const battle = new wasm.Battle(dex, encoded[0], encoded[1], job.seed);
            const rng = [job.agentSeed || 1, job.agentSeed || 1];
            const random = side => { let x = rng[side]; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return rng[side] = x >>> 0; };
            await window.recordMatch({ type: "game", pair: job.pair, leg, candidate_side: leg, teams: teams.map(t => t.id), seed: job.seed, agent_seed: job.agentSeed, user_agent: navigator.userAgent });
            try {
              for (let side = 0; side < 2; side++) if (workers[side]) await workers[side].newBattle(encoded[0], encoded[1], job.seed,
                { poolJson: prior, side, seed: job.agentSeed,
                  ...(arms[side].information === "public" ? { endgame: { solverFraction: arms[side].solver_fraction, milliseconds: protocol.milliseconds } } : {}) });
              let decision = 0;
              while (!battle.outcome()) {
                const needs = JSON.parse(battle.needsChoice());
                const choices = [];
                const reports = [{ wait: true }, { wait: true }];
                const turn = battle.turn();
                for (const side of [decision % 2, 1 - decision % 2]) {
                  if (!needs[side]) continue;
                  const legal = JSON.parse(battle.legalChoices(side));
                  const started = performance.now();
                  const result = workers[side] ? await bounded(workers[side].search(side, protocol.iterations, job.agentSeed, false))
                    : { best: legal[random(side) % legal.length].input, ms: 0, policy: { iterations: 0 } };
                  const elapsed = performance.now() - started;
                  const choice = battle.resolveChoice(side, result.best);
                  if (!choice || !legal.some(action => action.input === choice)) throw new Error(`illegal Web move ${result.best}`);
                  const endgame = result.endgame ? JSON.parse(result.endgame) : null;
                  if (arms[side].information === "public" && (endgame?.scope !== "conditional-determinized" || endgame.stats.root_overrides !== 0)) throw new Error("wrong public solver semantics");
                  choices[side] = choice;
                  reports[side] = { choice, request_ms: elapsed, search_ms: result.ms, iterations: result.policy.iterations,
                    endgame, wasm_memory_bytes: result.wasmMemoryBytes ?? null, forced: legal.length === 1, legal: true, visibility: document.visibilityState };
                }
                const picks = choices.flatMap((choice, side) => choice ? [[side, choice]] : []);
                for (const [side, choice] of picks) battle.applyChoice(side, choice);
                for (const worker of workers) worker?.apply(picks);
                await window.recordMatch({ type: "decision", pair: job.pair, leg, decision, turn, sides: reports });
                if (++decision > 4000) throw new Error("Web game exceeded emergency guard");
              }
              const outcome = battle.outcome();
              await window.recordMatch({ type: "result", pair: job.pair, leg, candidate_score: outcome === "tie" ? .5 : outcome === `p${leg + 1}` ? 1 : 0, outcome, turns: battle.turn(), engine_cap: battle.turn() >= 1000 });
            } finally { workers.forEach(w => w?.terminate()); battle.free(); }
          }
        }
      } finally { dex.free(); }
    }, { protocol, catalog, wasmUrl: "/@fs" + join(root, "crates/wasm/pkg-web/nc2000_wasm.js") });
    await page.close();
  }));
  console.log(`Completed ${protocol.pairs * 2} Web games.`);
} finally {
  await browser?.close();
  await vite?.close();
  await new Promise(resolve => server.close(resolve));
}
