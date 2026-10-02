"use strict";

const { wasm, check, checkEq, finish } = require("./common");

const dex = new wasm.Dex();
const set = moves => ({ species: "Pikachu", name: "Pikachu", level: 50, gender: "M", item: "", ability: "", moves });
const own = set(["Splash", "Seismic Toss"]);
const foe = set(["Splash"]);
const pool = JSON.stringify({ teams: [{ id: "foe", sets: [foe] }] });
const publicMon = { species: "Pikachu", level: 50, gender: "M", item_flag: false, hp_num: 100, hp_den: 100, appeared: true, active: true };
const spec = JSON.stringify({
  schema: "nc2000-position-v1", side: 0, turn: 999, own_sets: [own],
  sides: [0, 1].map(side => ({ mons: [{ ...publicMon, ...(side === 0 ? { hp_num: 1, hp_den: 125, hp_exact: 1 } : {}) }], active: 0, party: [0] })),
});

function search() {
  const s = new wasm.ProtocolSearcher(dex, 0, pool, 73, 0.4);
  s.setPosition(spec);
  return s;
}

function pump(s, iterations) {
  for (let i = 0; i < iterations && s.iterations() < iterations; i++) {
    const previous = s.iterations();
    s.step(1000);
    checkEq(s.iterations() - previous, 1, "enabled stepping yields after one iteration");
  }
  checkEq(s.iterations(), iterations, "requested iterations completed within the budget");
}

const baseline = search();
const off = search();
off.beginEndgame(JSON.stringify({ solver_fraction: 0 }), 10000);
baseline.step(96);
pump(off, 96);
checkEq(off.rootPolicy(), baseline.rootPolicy(), "OFF preserves search statistics");
checkEq(off.best(), baseline.best(), "OFF preserves choice");
checkEq(JSON.parse(off.endgameStats()).stats.calls, 0, "OFF never calls solver");

const on = search();
on.beginEndgame("{}", 10000);
pump(on, 96);
const report = JSON.parse(on.endgameStats());
checkEq(report.scope, "conditional-determinized", "conditional values are labelled");
check(report.stats.calls > 0, "solver invoked");
check(report.stats.value_replacements > 0, `solved values actually backpropagated: ${JSON.stringify(report.stats)}`);
checkEq(report.stats.root_overrides, 0, "blind root never takes a full-information strategy");
check(report.stats.max_value_error <= 0.01, "replacement error bounded");
const pick = on.best();
for (let i = 0; i < 10; i++) checkEq(on.best(), pick, "reads do not resample");
checkEq(JSON.parse(on.endgameStats()).stats, report.stats, "reads do not search");

on.setPosition(spec);
checkEq(on.endgameStats(), "null", "new position clears the decision budget");
on.beginEndgame("{}", 0);
checkEq(on.step(1000), 0, "expired budget stops before the first iteration");
check(on.best() !== undefined, "expired search retains a legal fallback");
checkEq(JSON.parse(on.endgameStats()).stats.runs, 0, "expired budget consumes no solver work");
on.beginEndgame(undefined, 0);
checkEq(on.endgameStats(), "null", "explicit OFF removes session");

baseline.free(); off.free(); on.free(); dex.free();
finish("endgame");
