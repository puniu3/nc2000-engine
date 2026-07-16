// Wasm twin of examples/native_bench.rs: identical battle, identical
// searcher seeds, identical iteration counts — the wasm/native throughput
// ratio is pure per-iteration engine cost.
//
//   node crates/wasm/tests-node/bench.js
"use strict";

const { wasm, loadFixture } = require("./common");

const BENCH_ITERS = 1000;
const PICK_ITERS = 100;
const MAX_DECISIONS = 200;

const dex = new wasm.Dex();
const fx = loadFixture("full/battle-001.json");
const battle = new wasm.Battle(
  dex,
  JSON.stringify(fx.p1team),
  JSON.stringify(fx.p2team),
  wasm.deriveBattleSeed(5)
);

let prevNs = 0n, prevIters = 0; // preview decisions
let batNs = 0n, batIters = 0; // in-battle decisions
let decisions = 0;
while (battle.outcome() === undefined && decisions < MAX_DECISIONS) {
  const needs = JSON.parse(battle.needsChoice());
  const picks = [];
  for (let side = 0; side < 2; side++) {
    if (!needs[side]) continue;
    const bench = new wasm.Searcher(battle, side, 100000 + decisions * 2 + side);
    const preview = battle.turn() === 0;
    const t = process.hrtime.bigint();
    bench.step(BENCH_ITERS);
    const ns = process.hrtime.bigint() - t;
    if (preview) {
      prevNs += ns;
      prevIters += BENCH_ITERS;
    } else {
      batNs += ns;
      batIters += BENCH_ITERS;
    }
    bench.free();
    const picker = new wasm.Searcher(battle, side, 42 + decisions * 2 + side);
    picker.step(PICK_ITERS);
    picks.push([side, picker.best()]);
    picker.free();
  }
  for (const [side, input] of picks) battle.applyChoice(side, input);
  decisions += 1;
}

const rate = (iters, ns) => iters / (Number(ns) / 1e9);
console.log(`wasm skuct throughput (BENCH_ITERS=${BENCH_ITERS}/decision):`);
console.log(
  `  preview roots: ${prevIters} iters, ${(Number(prevNs) / 1e9).toFixed(2)} s, ` +
    `${rate(prevIters, prevNs).toFixed(0)} iters/s`
);
console.log(
  `  battle roots:  ${batIters} iters, ${(Number(batNs) / 1e9).toFixed(2)} s, ` +
    `${rate(batIters, batNs).toFixed(0)} iters/s`
);
console.log(
  `  game: outcome ${battle.outcome()} in ${battle.turn()} turns, ${decisions} decisions`
);
