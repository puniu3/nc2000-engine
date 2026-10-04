"use strict";

const assert = require("node:assert/strict");
const { wasm, readData } = require("./common");
const set = (species, moves, item = "") => ({
  species, name: species, moves, item, level: 50, happiness: 255,
  evs: { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 },
  ivs: { hp: 30, atk: 30, def: 30, spa: 30, spd: 30, spe: 30 },
});
const dex = new wasm.Dex();
const results = [];
try {
  for (const seed of [82001, 82002, 82003, 82004]) {
    for (const reply of ["counter", "mirrorcoat", "safeguard", "destinybond"]) {
      const battle = new wasm.Battle(dex,
        JSON.stringify([set("Jolteon", ["Return", "Thunder Shock", "Toxic", "Thunder Wave"])]),
        JSON.stringify([set("Wobbuffet", ["Counter", "Mirror Coat", "Safeguard", "Destiny Bond"], "Leftovers")]),
        "1,2,3,1");
      const search = new wasm.BlindSearcher(battle, 0, readData("belief-pool-v3/belief-pool.json"), seed, 0.4);
      const step = (a, b) => { battle.applyChoice(0, a); battle.applyChoice(1, b); };
      try {
        step("team 1", "team 1");
        step("move return", "move counter");
        step("move thundershock", "move mirrorcoat");
        step("move toxic", "move mirrorcoat");
        for (let i = 0; i < 4; i++) step("move thunderwave", "move mirrorcoat");
        search.observe(battle);
        search.step(27000);
        const best = search.best();
        assert.ok(["move toxic", "move thunderwave"].includes(best), best);
        step(best, `move ${reply}`);
        assert.equal(battle.outcome(), "p1");
        results.push({ seed, reply, best, outcome: battle.outcome() });
      } finally {
        search.free();
        battle.free();
      }
    }
  }
} finally {
  dex.free();
}
console.log(JSON.stringify(results));
console.log("residual-wait: OK");
