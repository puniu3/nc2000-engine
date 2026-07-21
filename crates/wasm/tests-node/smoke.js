// Smoke: a full bot-vs-bot game driven end-to-end through the wasm API
// (stepped Searcher on both sides), the baked-preview-table path
// (pool + pair JSON fed as strings, resolve/sample at a real preview), a
// full blind game (M10c BlindSearcher: observe/step/best + belief surface,
// baked preview via the belief, fallback belief vs an off-pool team), and
// a full open-team-sheet game (M12 pinOpponent: pinned singleton belief,
// preview by direct signature lookup, custom opponents never fallback).
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
    path.join(REPO, "fixtures", "preview-tables-test", "pair-00-01.json"),
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

// ---------------------------------------------------------- blind game
{
  const poolJson = readData("meta-pool-v0/meta-pool.json");
  const pool = JSON.parse(poolJson);
  // Human stand-in (side 0, plain Searcher) plays pool team 0; the blind
  // bot (side 1) plays pool team 1 — the pair-00-01 matchup is committed,
  // so the bot's preview must come from the baked table via the belief.
  const battle = new wasm.Battle(
    dex,
    JSON.stringify(pool.teams[0].sets),
    JSON.stringify(pool.teams[1].sets),
    wasm.deriveBattleSeed(42)
  );
  // the blind observer's log channel wants the observed battle log-ON
  // (it is by default)
  const blind = new wasm.BlindSearcher(battle, 1, poolJson, 4242);
  blind.addPair(
    fs.readFileSync(
      path.join(REPO, "fixtures", "preview-tables-test", "pair-00-01.json"),
      "utf8"
    )
  );

  // preview: belief collapses to the true team, baked table plays
  blind.observe(battle);
  const info0 = JSON.parse(blind.beliefInfo());
  checkEq(info0.count, 1, "belief is a singleton at preview");
  checkEq(info0.fallback, false, "pool opponent is not fallback");
  checkEq(info0.candidates[0], pool.teams[0].id, "belief holds the true team");
  const baked = blind.bakedPreview();
  check(typeof baked === "string", "baked preview pick exists");
  checkEq(blind.best(), baked, "best() returns the baked pick");
  checkEq(blind.step(50), 0, "step is a no-op at a baked preview");
  const bpol = JSON.parse(blind.rootPolicy());
  checkEq(bpol.baked, true, "rootPolicy reports baked");
  checkEq(bpol.actions.length, 1, "baked policy is a point mass");

  let decisions = 0;
  const MAX_DECISIONS = 2000;
  while (battle.outcome() === undefined && decisions < MAX_DECISIONS) {
    const needs = JSON.parse(battle.needsChoice());
    const picks = [];
    if (needs[0]) {
      const legal = JSON.parse(battle.legalChoices(0));
      const s = new wasm.Searcher(battle, 0, 9000 + decisions);
      s.step(120);
      const best = s.best();
      check(
        legal.some((c) => c.input === best),
        `searcher best "${best}" is legal (blind game, side 0)`
      );
      picks.push([0, best]);
      s.free();
    }
    if (needs[1]) {
      const legal = JSON.parse(battle.legalChoices(1));
      blind.observe(battle);
      let best = blind.bakedPreview();
      if (best === undefined) {
        const total = blind.step(120);
        check(total >= 120, "blind step() returns total iterations");
        best = blind.best();
      }
      check(
        legal.some((c) => c.input === best),
        `blind best "${best}" is legal`
      );
      const info = JSON.parse(blind.beliefInfo());
      check(!info.fallback, "pool opponent never goes fallback");
      check(
        info.candidates.includes(pool.teams[0].id),
        "true team stays in the belief"
      );
      picks.push([1, best]);
    }
    for (const [side, input] of picks) battle.applyChoice(side, input);
    decisions += 1;
  }
  check(battle.outcome() !== undefined, "blind game reached an outcome");
  console.log(
    `  blind game: outcome ${battle.outcome()} in ${battle.turn()} turns, ` +
      `${decisions} decisions, preview via baked table`
  );
  blind.free();
  battle.free();

  // fallback: an off-pool opponent (fixture team) flips the belief to
  // fallback and the blind search still returns legal picks
  const fx = loadFixture("full/battle-001.json");
  const fb = new wasm.Battle(
    dex,
    JSON.stringify(fx.p1team),
    JSON.stringify(pool.teams[1].sets),
    wasm.deriveBattleSeed(43)
  );
  const bfs = new wasm.BlindSearcher(fb, 1, poolJson, 4343);
  bfs.observe(fb);
  const finfo = JSON.parse(bfs.beliefInfo());
  checkEq(finfo.fallback, true, "off-pool opponent goes fallback");
  checkEq(finfo.count, 0, "fallback has zero candidates");
  checkEq(bfs.bakedPreview(), undefined, "no baked preview in fallback");
  bfs.step(60);
  const fbest = bfs.best();
  const flegal = JSON.parse(fb.legalChoices(1));
  check(
    flegal.some((c) => c.input === fbest),
    `fallback blind best "${fbest}" is legal`
  );
  console.log(
    `  blind fallback: belief fallback vs off-pool team, best "${fbest}"`
  );
  bfs.free();
  fb.free();
}

// -------------------------------------- open team sheet (M12 pinOpponent)
{
  const poolJson = readData("meta-pool-v0/meta-pool.json");
  const pool = JSON.parse(poolJson);
  // Human stand-in (side 0, plain Searcher) plays pool team 1; the bot
  // (side 1) plays pool team 0 — pair-00-01 is committed, so the pinned
  // preview must come from the baked table via the signature lookup.
  const p1 = JSON.stringify(pool.teams[1].sets);
  const p2 = JSON.stringify(pool.teams[0].sets);
  const battle = new wasm.Battle(dex, p1, p2, wasm.deriveBattleSeed(1212));
  const bot = new wasm.BlindSearcher(battle, 1, poolJson, 2024);
  bot.pinOpponent(p1);
  bot.addPair(
    fs.readFileSync(
      path.join(REPO, "fixtures", "preview-tables-test", "pair-00-01.json"),
      "utf8"
    )
  );

  bot.observe(battle);
  const info = JSON.parse(bot.beliefInfo());
  checkEq(info.count, 1, "pinned belief is a singleton");
  checkEq(info.fallback, false, "pinned belief is not fallback");
  const baked = bot.bakedPreview();
  check(typeof baked === "string", "open-sheet preview comes from the table");
  checkEq(bot.best(), baked, "best() returns the table pick");

  let decisions = 0;
  const MAX_DECISIONS = 2000;
  while (battle.outcome() === undefined && decisions < MAX_DECISIONS) {
    const needs = JSON.parse(battle.needsChoice());
    const picks = [];
    if (needs[0]) {
      const legal = JSON.parse(battle.legalChoices(0));
      const s = new wasm.Searcher(battle, 0, 7000 + decisions);
      s.step(120);
      const best = s.best();
      check(
        legal.some((c) => c.input === best),
        `searcher best "${best}" is legal (open-sheet game, side 0)`
      );
      picks.push([0, best]);
      s.free();
    }
    if (needs[1]) {
      const legal = JSON.parse(battle.legalChoices(1));
      bot.observe(battle);
      let best = bot.bakedPreview();
      if (best === undefined) {
        bot.step(120);
        best = bot.best();
      }
      check(
        legal.some((c) => c.input === best),
        `open-sheet best "${best}" is legal`
      );
      const inf = JSON.parse(bot.beliefInfo());
      checkEq(inf.count, 1, "pinned belief stays a singleton in battle");
      checkEq(inf.fallback, false, "pinned belief never goes fallback");
      picks.push([1, best]);
    }
    for (const [side, input] of picks) battle.applyChoice(side, input);
    decisions += 1;
  }
  check(battle.outcome() !== undefined, "open-sheet game reached an outcome");
  console.log(
    `  open-sheet game: outcome ${battle.outcome()} in ${battle.turn()} turns, ` +
      `${decisions} decisions, preview via signature-resolved table`
  );
  bot.free();
  battle.free();

  // custom (off-pool) opponent pinned: a real singleton — never fallback —
  // with live-search preview (no table for the matchup)
  const fx = loadFixture("full/battle-001.json");
  const cp1 = JSON.stringify(fx.p1team);
  const cb = new wasm.Battle(dex, cp1, p2, wasm.deriveBattleSeed(1313));
  const cbot = new wasm.BlindSearcher(cb, 1, poolJson, 2025);
  cbot.pinOpponent(cp1);
  cbot.observe(cb);
  const cinfo = JSON.parse(cbot.beliefInfo());
  checkEq(cinfo.count, 1, "pinned custom opponent is a singleton");
  checkEq(cinfo.fallback, false, "pinned custom opponent is not fallback");
  checkEq(cbot.bakedPreview(), undefined, "custom matchup has no table");
  cbot.step(60);
  const cbest = cbot.best();
  const clegal = JSON.parse(cb.legalChoices(1));
  check(
    clegal.some((c) => c.input === cbest),
    `pinned custom best "${cbest}" is legal`
  );
  console.log(`  open-sheet custom: pinned singleton, live preview "${cbest}"`);
  cbot.free();
  cb.free();
}

finish("smoke");
