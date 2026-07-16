// Smoke: a full bot-vs-bot game driven end-to-end through the wasm API
// (stepped Searcher on both sides), plus the baked-preview-table path
// (pool + pair JSON fed as strings, resolve/sample at a real preview).
"use strict";

const fs = require("fs");
const path = require("path");
const { wasm, REPO, loadFixture, readData, check, checkEq, finish } =
  require("./common");

const dex = new wasm.Dex();

// ---------------------------------------------------------- full game
{
  const fx = loadFixture("full/battle-001.json");
  const battle = new wasm.Battle(
    dex,
    JSON.stringify(fx.p1team),
    JSON.stringify(fx.p2team),
    wasm.deriveBattleSeed(99)
  );
  let logLines = JSON.parse(battle.takeNewLog()).length;
  check(logLines > 0, "construction produced log lines");

  let decisions = 0;
  const MAX_DECISIONS = 2000;
  while (battle.outcome() === undefined && decisions < MAX_DECISIONS) {
    const needs = JSON.parse(battle.needsChoice());
    const picks = [];
    for (let side = 0; side < 2; side++) {
      if (!needs[side]) continue;
      const legal = JSON.parse(battle.legalChoices(side));
      check(legal.length > 0, `side ${side} owes a choice but has none`);
      // choice metadata sanity on the first move decision
      if (decisions === 2 && legal[0].kind === "move") {
        check(typeof legal[0].name === "string", "move choice carries a name");
        check(legal[0].pp >= 0, "move choice carries pp");
      }
      const s = new wasm.Searcher(battle, side, 1000 + decisions * 2 + side);
      const total = s.step(150);
      checkEq(total, 150, "step() returns total iterations");
      const best = s.best();
      check(
        legal.some((c) => c.input === best),
        `searcher best "${best}" is legal`
      );
      const pol = JSON.parse(s.rootPolicy());
      checkEq(pol.iterations, 150, "rootPolicy iterations");
      check(pol.actions.length === legal.length, "rootPolicy covers all actions");
      // forced spots (one legal action) bypass visit stats by design
      const vsum = pol.actions.reduce((a, r) => a + r.visits, 0);
      check(vsum > 0 || legal.length === 1, "rootPolicy has visits");
      const fsum = pol.actions.reduce((a, r) => a + r.frac, 0);
      check(Math.abs(fsum - 1) < 1e-9, "rootPolicy fracs sum to 1");
      picks.push([side, best]);
      s.free();
    }
    for (const [side, input] of picks) battle.applyChoice(side, input);
    decisions += 1;
    logLines += JSON.parse(battle.takeNewLog()).length;
  }
  check(battle.outcome() !== undefined, "game reached an outcome");
  check(logLines > 50, `protocol log accumulated (${logLines} lines)`);
  const view = JSON.parse(battle.stateView());
  checkEq(view.outcome, battle.outcome(), "state view outcome agrees");
  console.log(
    `  full game: outcome ${battle.outcome()} in ${battle.turn()} turns, ` +
      `${decisions} decisions, ${logLines} log lines`
  );
}

// ------------------------------------------------------ preview tables
{
  const poolJson = readData("meta-pool-v0/meta-pool.json");
  const pool = JSON.parse(poolJson);
  const tables = new wasm.PreviewTables(dex, poolJson);
  // pair-00-01: committed pilot pair (teams ranked 0 and 1)
  const pairJson = fs.readFileSync(
    path.join(REPO, "data", "preview-tables-v0", "pair-00-01.json"),
    "utf8"
  );
  tables.addPair(pairJson);
  checkEq(tables.pairCount(), 1, "pair count");

  const pair = JSON.parse(pairJson);
  const idx = (id) => pool.teams.findIndex((t) => t.id === id);
  const teamA = pool.teams[idx(pair.team_a)];
  const teamB = pool.teams[idx(pair.team_b)];
  check(teamA && teamB, "pair teams exist in pool");

  const battle = new wasm.Battle(
    dex,
    JSON.stringify(teamA.sets),
    JSON.stringify(teamB.sets),
    wasm.deriveBattleSeed(7)
  );
  const res = JSON.parse(tables.resolve(battle, 0));
  check(res.found, "matchup resolves");
  checkEq(res.iAmA, true, "orientation: side 0 is team_a");
  check(res.mixed.length >= 1, "mixed policy non-empty");
  const psum = res.mixed.reduce((a, r) => a + r.p, 0);
  check(Math.abs(psum - 1) < 1e-9, `mixed policy sums to 1 (${psum})`);
  check(
    typeof res.argmax.input === "string" && res.argmax.input.startsWith("team "),
    "argmax pick is a team choice"
  );

  // sampled + argmax picks are legal and apply cleanly for both sides
  for (let side = 0; side < 2; side++) {
    const legal = JSON.parse(battle.legalChoices(side));
    const sampled = tables.sample(battle, side, 1234 + side);
    check(
      legal.some((c) => c.input === sampled),
      `sampled preview "${sampled}" is legal (side ${side})`
    );
    const am = tables.argmax(battle, side);
    check(
      legal.some((c) => c.input === am),
      `argmax preview "${am}" is legal (side ${side})`
    );
    battle.applyChoice(side, sampled);
  }
  check(battle.turn() === 1, "preview applied, battle started");

  // unknown matchup falls through
  const fx = loadFixture("full/battle-001.json");
  const other = new wasm.Battle(
    dex,
    JSON.stringify(fx.p1team),
    JSON.stringify(fx.p2team),
    "1,2,3,4"
  );
  checkEq(
    JSON.parse(tables.resolve(other, 0)).found,
    false,
    "unknown matchup: found=false"
  );
  checkEq(tables.sample(other, 0, 1), undefined, "unknown matchup: no sample");
  console.log(
    `  tables: resolved ${pair.team_a} vs ${pair.team_b}, ` +
      `mixed support ${res.mixed.length}, value ${res.value.toFixed(3)}`
  );
}

finish("smoke");
