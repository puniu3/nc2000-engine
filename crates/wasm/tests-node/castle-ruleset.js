// Node smoke test for Gym Leader's Castle PBS WASM bridge APIs:
// Battle.withRuleset, Battle.ruleset, Kifu.ruleset, Validator.validateTeamWithRuleset,
// and Validator.canonicalizeTeamWithRuleset.
"use strict";

const { wasm, REPO, loadFixture, readData, check, checkEq, finish } = require("./common");

const dex = new wasm.Dex();
const validator = new wasm.Validator(dex);

function baseMon(species, level, moves, item = "") {
  return {
    name: species,
    species,
    item,
    ability: "No Ability",
    moves,
    nature: "Serious",
    evs: { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 },
    gender: ["Zapdos", "Suicune", "Starmie", "Raikou", "Porygon"].includes(species)
      ? "N"
      : "M",
    ivs: { hp: 30, atk: 30, def: 30, spa: 30, spd: 30, spe: 30 },
    level,
  };
}

function padTeam(sets) {
  const fillers = [
    baseMon("Snorlax", 50, ["Body Slam", "Rest"]),
    baseMon("Zapdos", 50, ["Thunderbolt", "Rest"]),
    baseMon("Suicune", 50, ["Surf", "Rest"]),
    baseMon("Cloyster", 50, ["Surf", "Spikes"]),
    baseMon("Marowak", 50, ["Earthquake", "Rest"]),
    baseMon("Umbreon", 50, ["Toxic", "Rest"]),
  ];
  const out = [...sets];
  for (const f of fillers) {
    if (out.length >= 6) break;
    if (out.some((s) => s.species === f.species)) continue;
    out.push(f);
  }
  return JSON.stringify(out);
}

// 1. Default Battle remains PS Stadium 2
{
  const fx = loadFixture("full/battle-001.json");
  const psBattle = new wasm.Battle(
    dex,
    JSON.stringify(fx.p1team),
    JSON.stringify(fx.p2team),
    fx.seed,
  );
  checkEq(psBattle.ruleset(), "gen2stadium2", "default Battle uses gen2stadium2");
  psBattle.free();
}

// 2. Battle.withRuleset + Castle Replay / Kifu roundtrip
{
  const fx = loadFixture("full/battle-001.json");
  const battle = wasm.Battle.withRuleset(
    dex,
    JSON.stringify(fx.p1team),
    JSON.stringify(fx.p2team),
    fx.seed,
    "castle",
  );
  checkEq(battle.ruleset(), "castle-tournament", "castle alias resolves to castle-tournament");

  let lastView = null;
  let lastRound = -1;
  for (let round = 0; round < 6 && battle.outcome() === undefined; round++) {
    const needs = JSON.parse(battle.needsChoice());
    const picks = needs.map((n, side) =>
      n ? JSON.parse(battle.legalChoices(side))[0].input : null,
    );
    const view = JSON.parse(battle.stateView());
    if (view.turn > 0 && needs[1]) {
      lastView = view;
      lastRound = round;
    }
    for (let side = 0; side < 2; side++) {
      if (picks[side]) battle.applyChoice(side, picks[side]);
    }
  }

  const code = battle.exportKifu(false, 1);
  check(/^NC2-[A-Za-z0-9_-]+$/.test(code), "Castle replay emits NC2- code");
  const kifu = new wasm.Kifu(dex, code);
  checkEq(kifu.ruleset(), "castle-tournament", "Kifu preserves castle-tournament ruleset");
  check(lastRound >= 0, "recorded at least one Castle decision");
  const scene = JSON.parse(kifu.scene(lastRound));
  checkEq(scene.view, lastView, "Castle Kifu reproduces exact board state");

  kifu.free();
  battle.free();
}

// 3. Unknown ruleset throws
{
  const fx = loadFixture("full/battle-001.json");
  let threw = false;
  try {
    wasm.Battle.withRuleset(
      dex,
      JSON.stringify(fx.p1team),
      JSON.stringify(fx.p2team),
      fx.seed,
      "not-a-real-ruleset",
    );
  } catch (_) {
    threw = true;
  }
  check(threw, "unknown ruleset id throws cleanly");
}

// 4. Validator.validateTeamWithRuleset & canonicalizeTeamWithRuleset
{
  // NYPC Snorlax Lovely Kiss: PS accepts, Castle rejects
  const nypcTeam = padTeam([baseMon("Snorlax", 55, ["Lovely Kiss", "Body Slam"])]);
  const psRes = JSON.parse(validator.validateTeamWithRuleset(nypcTeam, "gen2stadium2"));
  const castleRes = JSON.parse(validator.validateTeamWithRuleset(nypcTeam, "castle"));
  checkEq(psRes.ok, true, "PS accepts Snorlax Lovely Kiss");
  checkEq(castleRes.ok, false, "Castle rejects NYPC Snorlax Lovely Kiss");

  // JP domestic event Donphan Encore: PS rejects, Castle accepts
  const jpEventTeam = padTeam([baseMon("Donphan", 50, ["Encore", "Earthquake"])]);
  checkEq(
    JSON.parse(validator.validateTeamWithRuleset(jpEventTeam, "gen2stadium2")).ok,
    false,
    "PS rejects JP event Donphan Encore",
  );
  checkEq(
    JSON.parse(validator.validateTeamWithRuleset(jpEventTeam, "castle")).ok,
    true,
    "Castle accepts JP event Donphan Encore",
  );

  // Inert item warning without rewrite or Item Clause collision
  const inertTeam = JSON.stringify([
    baseMon("Snorlax", 55, ["Body Slam", "Rest"], "Bitter Berry"),
    baseMon("Zapdos", 50, ["Thunderbolt", "Rest"], "Ice Berry"),
    baseMon("Dragonite", 55, ["Dragon Breath", "Rest"], "Dragon Fang"),
    baseMon("Kingdra", 50, ["Surf", "Dragon Breath"], "Dragon Scale"),
    baseMon("Cloyster", 50, ["Surf", "Spikes"], "Leftovers"),
    baseMon("Umbreon", 50, ["Toxic", "Rest"], "Gold Berry"),
  ]);
  const canon = JSON.parse(validator.canonicalizeTeamWithRuleset(inertTeam, "castle"));
  checkEq(canon.ok, true, "inert items stay legal in Castle canonicalize");
  checkEq(canon.team[0].item, "Bitter Berry", "Bitter Berry is never silently rewritten");
  checkEq(canon.team[2].item, "Dragon Fang", "Dragon Fang is never silently rewritten");
  checkEq(canon.warnings.length, 2, "both inert items emit warnings");

  // Impossible EV spread (63-63-10-63-63) fails under Castle, passes under PS
  const badEvMon = baseMon("Snorlax", 55, ["Body Slam", "Rest"]);
  badEvMon.evs = { hp: 255, atk: 255, def: 40, spa: 255, spd: 255, spe: 255 };
  const badEvTeam = padTeam([badEvMon]);
  checkEq(
    JSON.parse(validator.validateTeamWithRuleset(badEvTeam, "gen2stadium2")).ok,
    true,
    "PS accepts Def effort 10",
  );
  checkEq(
    JSON.parse(validator.validateTeamWithRuleset(badEvTeam, "castle")).ok,
    false,
    "Castle rejects impossible Def effort 10",
  );
}

// 5. ProtocolSearcher.withRuleset / setRuleset / ruleset / compiledPreviewBuild & PreviewTables Castle soft-miss
{
  const fs = require("fs");
  const path = require("path");
  const poolJson = readData("meta-pool-v0/meta-pool.json");
  const pool = JSON.parse(poolJson);
  const pairJson = fs.readFileSync(
    path.join(REPO, "fixtures", "preview-tables-test", "pair-00-01.json"),
    "utf8",
  );
  const pairObj = JSON.parse(pairJson);
  const teamA = pool.teams.find((t) => t.id === pairObj.team_a);
  const teamB = pool.teams.find((t) => t.id === pairObj.team_b);
  check(Boolean(teamA && teamB), "found pool teams matching pair table");

  const psSearcher = new wasm.ProtocolSearcher(dex, 0, poolJson, 1, 1.4);
  checkEq(psSearcher.ruleset(), "gen2stadium2", "default ProtocolSearcher uses gen2stadium2");
  checkEq(psSearcher.compiledPreviewBuild(), "policy-v2", "PS compiledPreviewBuild is policy-v2");
  psSearcher.setRuleset("castle");
  checkEq(psSearcher.ruleset(), "castle-tournament", "setRuleset switches to castle-tournament");
  checkEq(
    psSearcher.compiledPreviewBuild(),
    "castle-policy-v2",
    "Castle compiledPreviewBuild is castle-policy-v2",
  );
  psSearcher.free();

  const castleSearcher = wasm.ProtocolSearcher.withRuleset(
    dex,
    0,
    poolJson,
    1,
    1.4,
    undefined,
    "castle",
  );
  checkEq(castleSearcher.ruleset(), "castle-tournament", "ProtocolSearcher.withRuleset sets Castle");
  checkEq(
    castleSearcher.compiledPreviewBuild(),
    "castle-policy-v2",
    "ProtocolSearcher.withRuleset reports castle-policy-v2",
  );
  castleSearcher.free();

  // PreviewTables resolves PS pair match, but soft-misses on Castle battle with identical teams
  const tables = new wasm.PreviewTables(dex, poolJson);
  tables.addPair(pairJson);

  const bPs = new wasm.Battle(
    dex,
    JSON.stringify(teamA.sets),
    JSON.stringify(teamB.sets),
    "1,2,3,4",
  );
  const bCastle = wasm.Battle.withRuleset(
    dex,
    JSON.stringify(teamA.sets),
    JSON.stringify(teamB.sets),
    "1,2,3,4",
    "castle",
  );
  checkEq(JSON.parse(tables.resolve(bPs, 0)).found, true, "PreviewTables hits on PS battle");
  check(tables.argmax(bPs, 0) !== undefined, "PreviewTables.argmax returns pick on PS battle");
  check(tables.sample(bPs, 0, 7) !== undefined, "PreviewTables.sample returns pick on PS battle");

  checkEq(
    JSON.parse(tables.resolve(bCastle, 0)).found,
    false,
    "PreviewTables soft-misses on Castle battle",
  );
  checkEq(
    tables.argmax(bCastle, 0),
    undefined,
    "PreviewTables.argmax returns undefined on Castle battle",
  );
  checkEq(
    tables.sample(bCastle, 0, 7),
    undefined,
    "PreviewTables.sample returns undefined on Castle battle",
  );

  bPs.free();
  bCastle.free();
  tables.free();
}

validator.free();
dex.free();
finish("castle-ruleset");
