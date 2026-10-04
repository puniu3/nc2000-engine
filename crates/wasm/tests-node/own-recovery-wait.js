"use strict";
const assert = require("node:assert/strict");
const { wasm, readData } = require("./common");
const set = (species, moves, item = "") => ({
  species, name: species, moves, item, level: 50, happiness: 255,
  evs: { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 },
  ivs: { hp: 30, atk: 30, def: 30, spa: 30, spd: 30, spe: 30 },
});
const dex = new wasm.Dex();
const rows = [];
try {
  for (const seed of [94001, 94002, 94003, 94004]) {
    const battle = new wasm.Battle(dex,
      JSON.stringify([set("Jolteon", ["Return", "Body Slam", "Toxic", "Thunder Wave"], "Leftovers")]),
      JSON.stringify([set("Wobbuffet", ["Counter", "Mirror Coat", "Safeguard", "Destiny Bond"])]),
      "1,2,3,1");
    const search = new wasm.BlindSearcher(battle, 0, readData("belief-pool-v3/belief-pool.json"), seed, 0.4);
    const step = (a, b) => { battle.applyChoice(0, a); battle.applyChoice(1, b); };
    const choices = [];
    try {
      step("team 1", "team 1");
      for (const [a, b] of [
        ["return", "counter"], ["return", "mirrorcoat"], ["return", "destinybond"],
        ["return", "counter"], ["thunderwave", "safeguard"], ["return", "mirrorcoat"],
      ]) step(`move ${a}`, `move ${b}`);
      for (let turn = 0; turn < 100 && !battle.outcome(); turn++) {
        search.observe(battle);
        search.step(27000);
        const best = search.best();
        if (turn === 0) assert.ok(["move toxic", "move thunderwave"].includes(best), best);
        choices.push(best);
        const legal = JSON.parse(battle.legalChoices(1)).map(c => c.input);
        step(best, legal.includes("move counter") ? "move counter" : legal[0]);
      }
      assert.equal(battle.outcome(), "p1");
      rows.push({ seed, choices, outcome: battle.outcome() });
    } finally {
      search.free(); battle.free();
    }
  }
} finally { dex.free(); }
console.log(JSON.stringify(rows));
console.log("own-recovery-wait: OK");
