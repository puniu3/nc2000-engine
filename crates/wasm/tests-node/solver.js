// Solver mode through the wasm bridge: a hand-written position in, a scored
// report out — and, when the native twin is available, the same report.
//
// The parity half is the point. `analysis::report` runs in one place and is
// reached two ways (this bridge, and `examples/solve_position`), so a number
// the browser prints must be reproducible on the command line. What is
// asserted is everything discrete — the same actions in the same order, the
// same visit counts, the same matrix cells with the same sample counts, the
// same integer damage — because those are what a divergence in the SEARCH
// would move.
//
// The averaged values are compared to 1e-9 rather than exactly, and that is
// a real limit rather than a hedge: the eval's sigmoid and UCB's logarithm
// are libm calls, and libm is not bit-identical across targets (the wasm
// build carries Rust's own implementation; the native one calls the
// system's). A last-digit difference in one leaf evaluation is expected and
// means nothing; a changed visit count would mean the two builds searched
// different trees, and that is what this checks.
//
//   node crates/wasm/tests-node/solver.js                  # structure only
//   NC2000_NATIVE_PARITY=1 node crates/wasm/tests-node/solver.js  # + twin
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");
const { wasm, REPO, readData, check, checkEq, finish } = require("./common");

const dex = new wasm.Dex();
const poolJson = readData("meta-pool-v0/meta-pool.json");
const pool = JSON.parse(poolJson);

const ITERS = 2000;
const SEED = 7;

/** An action row without its averaged value — everything the search decides
 * discretely, which must match the native twin exactly. */
function strip(a) {
  const { mean, ...rest } = a;
  void mean;
  return rest;
}

function checkClose(a, b, what) {
  check(Math.abs(a - b) < 1e-9, `${what}: ${a} vs ${b} (beyond libm tolerance)`);
}

/** The public half of a team: species, level, gender and whether it holds an
 * item — the four things a `|poke|` preview line states, and the four the
 * belief filters candidate teams on. Its SETS are exactly what a blind
 * position must not carry. */
function publicRoster(sets, active) {
  return sets.map((s, i) => ({
    species: s.species,
    level: s.level,
    gender: s.gender ?? "",
    item_flag: !!s.item,
    hp_num: 100,
    hp_den: 100,
    appeared: i === active,
    active: i === active,
  }));
}

const mine = pool.teams[0];
const foe = pool.teams[1];
const spec = {
  schema: "nc2000-position-v1",
  side: 0,
  turn: 1,
  own_sets: mine.sets,
  sides: [
    { mons: publicRoster(mine.sets, 0), active: 0, party: [0, 1, 2] },
    { mons: publicRoster(foe.sets, 0), active: 0 },
  ],
};
const specJson = JSON.stringify(spec);

const searcher = new wasm.ProtocolSearcher(dex, 0, poolJson, SEED);
searcher.setPosition(specJson);
searcher.step(ITERS);
const report = JSON.parse(searcher.report(0, SEED));

checkEq(report.schema, "nc2000-analysis-v1", "report schema");
checkEq(report.iterations, ITERS, "iterations run");
checkEq(report.preview, false, "a board position is not team preview");
check(report.actions.length > 0, "the position has scored actions");
checkEq(
  report.belief.count,
  1,
  "a roster carrying the public facts identifies its pool team"
);
checkEq(report.belief.fallback, false, "identified, so no fallback imputation");

const shares = report.actions.reduce((a, r) => a + r.frac, 0);
check(Math.abs(shares - 1) < 1e-9, `visit shares sum to 1 (got ${shares})`);
const visits = report.actions.map((a) => a.visits);
checkEq(
  visits,
  [...visits].sort((a, b) => b - a),
  "actions are ordered by playouts"
);
for (const a of report.actions) {
  for (const k of ["mean", "equity"]) {
    check(a[k] >= 0 && a[k] <= 1, `${k} in [0,1] for ${a.input}`);
  }
  check(
    a.worst === null || a.worst <= a.equity + 1e-9,
    `${a.input}: the worst reply cannot beat the equilibrium answer`
  );
}
const mixMine = report.actions.reduce((x, a) => x + a.mix, 0);
check(Math.abs(mixMine - 1) < 1e-6, `our equilibrium mixture sums to 1 (${mixMine})`);
const mixTheirs = report.equilibrium.theirs.reduce((x, p) => x + p, 0);
check(Math.abs(mixTheirs - 1) < 1e-6, `their mixture sums to 1 (${mixTheirs})`);
check(
  report.equilibrium.value >= 0 && report.equilibrium.value <= 1,
  "the position value is a probability"
);

// the joint the marginals hide
check(report.matrix.cols.length > 0, "the root matrix has opponent columns");
for (const c of report.matrix.cols) {
  check(
    c.available > 0 && c.available <= 1,
    `${c.input}: availability is a share of playouts (${c.available})`
  );
}
checkEq(
  report.matrix.cells.length,
  report.actions.length,
  "one matrix row per action"
);
report.matrix.cells.forEach((row, i) => {
  checkEq(row.length, report.matrix.cols.length, `row ${i} spans the columns`);
  const n = row.reduce((a, c) => a + (c ? c.n : 0), 0);
  check(
    n <= report.actions[i].visits,
    `row ${i}: ${n} joint samples cannot exceed ${report.actions[i].visits} visits`
  );
});

// engine-truth damage, both directions
check(report.damage.mine.length > 0, "our moves have damage rows");
check(report.damage.theirs.length > 0, "their moves have damage rows");
for (const d of [...report.damage.mine, ...report.damage.theirs]) {
  check(d.min <= d.max, `${d.move}: min roll <= max roll`);
  check(d.crit >= d.max, `${d.move}: a crit is never smaller than a top roll`);
}
// nothing of theirs is claimed as revealed here: the position showed nothing
check(
  report.damage.theirs.every((d) => d.revealed === false),
  "unshown opponent moves are marked assumed"
);

// the searched line, when asked for, is searched rather than sampled
{
  const withLine = new wasm.ProtocolSearcher(dex, 0, poolJson, SEED);
  withLine.setPosition(specJson);
  withLine.step(ITERS);
  const line = JSON.parse(withLine.report(3, SEED)).line;
  check(line !== null && line.steps.length > 0, "a line was produced");
  for (const st of line.steps) {
    check(st.prob > 0 && st.prob <= 1, `step probability in (0,1]: ${st.prob}`);
    check(st.iterations > 0, "every step of the line was searched");
  }
  withLine.free();
}

// the synthesized board is readable back
const view = JSON.parse(searcher.stateView());
checkEq(view.turn, 1, "state view turn");
check(view.sides[0].active !== null, "our side has an active mon");

// a position that cannot happen is refused, by name
let refused = "";
try {
  const bad = JSON.parse(specJson);
  bad.sides[0].mons[0].hp_num = 0; // alive at 0 HP
  new wasm.ProtocolSearcher(dex, 0, poolJson, SEED).setPosition(
    JSON.stringify(bad)
  );
} catch (e) {
  refused = String(e.message ?? e);
}
check(/fainted/.test(refused), `refusal names the defect (got "${refused}")`);

console.log(
  `  solver: ${report.actions.length} actions, ${report.matrix.cols.length} replies, ` +
    `best "${report.actions[0].input}" ${(report.actions[0].mean * 100).toFixed(1)}%`
);

// ------------------------------------------------------- native twin
if (process.env.NC2000_NATIVE_PARITY === "1") {
  const file = path.join(os.tmpdir(), `nc2000-solver-parity-${process.pid}.json`);
  fs.writeFileSync(file, specJson);
  try {
    const out = execFileSync(
      "cargo",
      [
        "run", "--release", "-q", "-p", "nc2000-bot", "--example", "solve_position",
        "--", file, "--iters", String(ITERS), "--seed", String(SEED),
        "--plies", "0", "--json",
      ],
      { cwd: REPO, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 }
    );
    const native = JSON.parse(out);
    checkEq(
      native.actions.map(strip),
      report.actions.map(strip),
      "native ≡ wasm: scored actions (order, visits, shares)"
    );
    native.actions.forEach((a, i) =>
      checkClose(a.mean, report.actions[i].mean, `action ${a.input} win rate`)
    );
    checkEq(
      native.matrix.cols.map((c) => ({ ...c, available: undefined })),
      report.matrix.cols.map((c) => ({ ...c, available: undefined })),
      "native ≡ wasm: matrix columns"
    );
    native.matrix.cols.forEach((c, i) =>
      checkClose(c.available, report.matrix.cols[i].available, `col ${c.input} availability`)
    );
    checkEq(
      native.matrix.cells.map((row) => row.map((c) => (c ? c.n : null))),
      report.matrix.cells.map((row) => row.map((c) => (c ? c.n : null))),
      "native ≡ wasm: matrix sample counts"
    );
    native.matrix.cells.forEach((row, i) =>
      row.forEach((c, j) => {
        if (c) checkClose(c.mean, report.matrix.cells[i][j].mean, `cell ${i},${j}`);
      })
    );
    checkEq(native.damage, report.damage, "native ≡ wasm: damage table");
    console.log("  solver: native twin agrees");
  } finally {
    fs.unlinkSync(file);
  }
}

searcher.free();
finish("solver");
