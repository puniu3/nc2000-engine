const fs = require("node:fs");
const path = require("node:path");
const wasm = require(path.resolve(process.argv[2], "nc2000_wasm.js"));
const root = path.resolve(__dirname, "..");
const poolText = fs.readFileSync(
  path.join(root, "data/belief-pool-v3/belief-pool.json"), "utf8",
);
const pool = JSON.parse(poolText);
const dex = new wasm.Dex();
const rows = [];
for (let game = 0; game < 3; game++) {
  const battle = new wasm.Battle(
    dex,
    JSON.stringify(pool.teams[game * 2].sets),
    JSON.stringify(pool.teams[game * 2 + 1].sets),
    wasm.deriveBattleSeed(700 + game),
  );
  const blind = new wasm.BlindSearcher(battle, 1, poolText, 4242 + game);
  for (let decision = 0; decision < 12 && battle.outcome() === undefined; decision++) {
    const needs = JSON.parse(battle.needsChoice());
    blind.observe(battle);
    const picks = [];
    if (needs[1]) {
      const start = performance.now();
      blind.step(3000);
      rows.push({
        game, decision, turn: battle.turn(), ms: performance.now() - start,
        signature: blind.rootPolicy(), best: blind.best(),
      });
      picks.push([1, blind.best()]);
    }
    if (needs[0]) {
      const picker = new wasm.Searcher(battle, 0, 9000 + decision);
      picker.step(120);
      picks.push([0, picker.best()]);
      picker.free();
    }
    for (const [side, input] of picks) battle.applyChoice(side, input);
  }
  blind.free();
  battle.free();
}
dex.free();
console.log(JSON.stringify({ rows, total_ms: rows.reduce((sum, row) => sum + row.ms, 0) }));
