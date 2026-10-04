"use strict";
const assert = require("node:assert/strict");
const { wasm, readData } = require("./common");
const set = (species, moves) => ({
  species, name: species, moves, item: "", level: 50, happiness: 255,
  evs: { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 },
  ivs: { hp: 30, atk: 30, def: 30, spa: 30, spd: 30, spe: 30 },
});
const dex = new wasm.Dex();
const rows = [];
try {
  for (const side of [0, 1]) {
    for (const [species, reply, priority] of [
      ["Scizor", "quickattack", true], ["Hitmonchan", "machpunch", true],
      ["Arcanine", "extremespeed", true], ["Snorlax", "return", false],
    ]) {
      const own = [set("Alakazam", ["Psychic", "Recover"])];
      const foe = [set(species, [reply])];
      const teams = side === 0 ? [own, foe] : [foe, own];
      const battle = new wasm.Battle(dex, ...teams.map(team => JSON.stringify(team)), "1,2,3,4");
      const search = new wasm.BlindSearcher(battle, side, readData("belief-pool-v3/belief-pool.json"), 98001, 0.4);
      try {
        battle.applyChoice(0, "team 1"); battle.applyChoice(1, "team 1");
        search.observe(battle);
        assert.equal(search.best(), priority ? "move recover" : "move psychic");
        battle.takeNewLog();
        battle.applyChoice(side, "move recover");
        battle.applyChoice(1 - side, `move ${reply}`);
        const log = JSON.parse(battle.takeNewLog());
        const healed = log.some(line => line.startsWith(`|-heal|p${side + 1}a:`));
        assert.equal(healed, priority);
        rows.push({ side, species, reply, healed, scope: "zero-iteration mask integration probe; no strength claim" });
      } finally { search.free(); battle.free(); }
    }
  }
} finally { dex.free(); }
console.log(JSON.stringify(rows));
console.log("priority-heal: OK");
