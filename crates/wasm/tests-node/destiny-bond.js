"use strict";

function verifyDestinyBond(wasm, poolJson) {
  const dex = new wasm.Dex();
  const set = (species, moves, level = 50, item = "") => ({
    species, name: species, moves, level, item, ability: "No Ability", happiness: 255,
    evs: { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 },
    ivs: { hp: 30, atk: 30, def: 30, spa: 30, spd: 30, spe: 30 },
  });
  const results = [];
  try {
    for (const plain of [false, true]) {
      for (const seed of [71001, 71002]) {
        const own = [set("Snorlax", ["Self-Destruct"]), set("Gengar",
          ["Mean Look", plain ? "Psychic" : "Curse", "Shadow Ball", "Return"], 55, "Leftovers")];
        const foe = [set("Gengar", ["Destiny Bond", "Haze"]), set("Magikarp", ["Splash"])];
        const battle = new wasm.Battle(dex, JSON.stringify(own), JSON.stringify(foe), "1,2,3,4");
        const search = new wasm.BlindSearcher(battle, 0, poolJson, seed, 0.4);
        const step = (a, b) => { battle.applyChoice(0, a); battle.applyChoice(1, b); };
        try {
          step("team 1,2", "team 1,2");
          step("move selfdestruct", "move haze");
          battle.applyChoice(0, "switch 2");
          if (plain) {
            step("move psychic", "move haze");
            for (let i = 0; i < 8; i++) step("move meanlook", "move destinybond");
          } else {
            for (let i = 0; i < 7; i++) step("move meanlook", "move destinybond");
            step("move meanlook", "move haze");
            step("move curse", "move haze");
            step("move return", "move haze");
            step("move return", "move destinybond");
          }
          search.observe(battle);
          search.step(27000);
          const best = search.best();
          if (best !== "move return" && (plain || best !== "move curse")) {
            throw new Error(`Destiny Bond waiting excluded: plain=${plain}, seed=${seed}, best=${best}`);
          }
          results.push({ plain, seed, best });
        } finally {
          search.free();
          battle.free();
        }
      }
    }
  } finally {
    dex.free();
  }
  return results;
}

module.exports = verifyDestinyBond;
if (require.main === module) {
  const { wasm, readData } = require("./common");
  console.log(JSON.stringify(verifyDestinyBond(wasm, readData("belief-pool-v3/belief-pool.json"))));
  console.log("destiny-bond: OK");
}
