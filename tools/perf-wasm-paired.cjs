// Compare the frozen binaries adjacent to each other at every decision, so
// long host-load drift does not confound separate full-corpus process runs.
const fs = require('node:fs');
const path = require('node:path');
const modules = ['baseline', 'candidate'];
const wasm = Object.fromEntries(modules.map((arm, i) => [arm,
  require(path.resolve(process.argv[i + 2], 'nc2000_wasm.js'))]));
const iterations = Number(process.argv[4] ?? 27000);
const repetition = Number(process.argv[5] ?? 0);
if (!Number.isSafeInteger(iterations) || iterations < 1) throw Error('invalid iterations');
const poolText = fs.readFileSync(path.resolve(__dirname, '../data/belief-pool-v3/belief-pool.json'), 'utf8');
const pool = JSON.parse(poolText);
const dex = Object.fromEntries(modules.map(arm => [arm, new wasm[arm].Dex()]));
const rows = [];
function equal(values, label) {
  if (values[0] !== values[1]) throw Error(`${label} diverged`);
  return values[0];
}
try {
  // One unmeasured warm-up per binary, with identical inputs and work.
  for (const arm of modules) {
    const b = new wasm[arm].Battle(dex[arm], JSON.stringify(pool.teams[0].sets), JSON.stringify(pool.teams[1].sets), '1,2,3,4');
    const s = new wasm[arm].BlindSearcher(b, 1, poolText, 999);
    s.observe(b); s.step(1000); s.free(); b.free();
  }
  for (let game = 0; game < 3; game++) {
    const battles = Object.fromEntries(modules.map(arm => [arm, new wasm[arm].Battle(dex[arm],
      JSON.stringify(pool.teams[game * 2].sets), JSON.stringify(pool.teams[game * 2 + 1].sets),
      wasm[arm].deriveBattleSeed(700 + game))]));
    const agents = Object.fromEntries(modules.map(arm => [arm, new wasm[arm].BlindSearcher(battles[arm], 1, poolText, 4242 + game)]));
    try {
      for (let decision = 0; decision < 12; decision++) {
        const outcome = equal(modules.map(arm => battles[arm].outcome()), 'outcome');
        if (outcome !== undefined) break;
        const needs = JSON.parse(equal(modules.map(arm => battles[arm].needsChoice()), 'request'));
        for (const arm of modules) agents[arm].observe(battles[arm]);
        const order = (game + decision + repetition) % 2 ? modules.toReversed() : modules;
        const choices = [];
        if (needs[1]) {
          const ms = {};
          for (const arm of order) {
            const start = performance.now(); agents[arm].step(iterations);
            ms[arm] = performance.now() - start;
          }
          const signature = equal(modules.map(arm => agents[arm].rootPolicy()), 'root policy');
          const best = equal(modules.map(arm => agents[arm].best()), 'choice');
          rows.push({ game, decision, turn: battles.baseline.turn(), signature, best, ms });
          choices.push([1, best]);
        }
        if (needs[0]) {
          const best = modules.map(arm => {
            const picker = new wasm[arm].Searcher(battles[arm], 0, 9000 + decision);
            try { picker.step(120); return picker.best(); } finally { picker.free(); }
          });
          choices.push([0, equal(best, 'opponent choice')]);
        }
        for (const [side, input] of choices) for (const arm of modules) battles[arm].applyChoice(side, input);
        equal(modules.map(arm => battles[arm].stateView()), 'battle state');
        equal(modules.map(arm => battles[arm].seed()), 'battle RNG');
        equal(modules.map(arm => battles[arm].takeNewLog()), 'battle log');
      }
    } finally { for (const arm of modules) { agents[arm].free(); battles[arm].free(); } }
  }
} finally { for (const arm of modules) dex[arm].free(); }
console.log(JSON.stringify({ rows, total_ms: Object.fromEntries(modules.map(arm => [arm, rows.reduce((sum, row) => sum + row.ms[arm], 0)])) }));
