// Determinism: the same battle seed + same searcher seeds → identical
// trajectory (outcome, turn count, full protocol log, final PRNG seed).
"use strict";

const { wasm, loadFixture, checkEq, finish } = require("./common");

const dex = new wasm.Dex();
const fx = loadFixture("full/battle-001.json");

function playOnce() {
  const battle = new wasm.Battle(
    dex,
    JSON.stringify(fx.p1team),
    JSON.stringify(fx.p2team),
    wasm.deriveBattleSeed(5)
  );
  const log = [];
  let decisions = 0;
  while (battle.outcome() === undefined && decisions < 2000) {
    const needs = JSON.parse(battle.needsChoice());
    const picks = [];
    for (let side = 0; side < 2; side++) {
      if (!needs[side]) continue;
      const s = new wasm.Searcher(battle, side, 42 + decisions * 2 + side);
      s.step(100);
      picks.push([side, s.best()]);
      s.free();
    }
    for (const [side, input] of picks) battle.applyChoice(side, input);
    decisions += 1;
    log.push(...JSON.parse(battle.takeNewLog()));
  }
  return {
    outcome: battle.outcome(),
    turn: battle.turn(),
    seed: battle.seed(),
    log,
  };
}

const a = playOnce();
const b = playOnce();
checkEq(a.outcome, b.outcome, "outcome identical");
checkEq(a.turn, b.turn, "turn count identical");
checkEq(a.seed, b.seed, "final PRNG seed identical");
checkEq(a.log.length, b.log.length, "log length identical");
checkEq(a.log, b.log, "full protocol log identical");
console.log(
  `  two runs: outcome ${a.outcome}, ${a.turn} turns, ` +
    `${a.log.length} log lines — identical`
);
finish("determinism");
