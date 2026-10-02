"use strict";

const { writeFileSync } = require("node:fs");
const { wasm, loadFixture, readData, check, checkEq, finish } = require("./common");

const dex = new wasm.Dex();
const legacy = require("../../../fixtures/replay/transition-compatible.json");
const legacyKifu = new wasm.Kifu(dex, legacy.code);
check(JSON.parse(legacyKifu.scenes()).scenes.length > 0, "frozen replay remains compatible");
legacyKifu.free();
const fixture = loadFixture("full/battle-001.json");
const pool = readData("meta-pool-v0/meta-pool.json");
for (const team of [fixture.p1team, fixture.p2team]) for (const mon of team) delete mon.gender;
const battle = new wasm.Battle(dex, JSON.stringify(fixture.p1team), JSON.stringify(fixture.p2team), fixture.seed);
const scenes = new Map();
let lines = JSON.parse(battle.takeNewLog());
let round = 0;
while (battle.outcome() === undefined && round < 100) {
  const needs = JSON.parse(battle.needsChoice());
  const picks = needs.map((need, side) => need ? JSON.parse(battle.legalChoices(side))[0].input : null);
  const view = JSON.parse(battle.stateView());
  if (view.turn > 0 && needs[1]) scenes.set(round, { view, log: [...lines] });
  for (const side of [1, 0]) if (picks[side]) battle.applyChoice(side, picks[side]);
  lines.push(...JSON.parse(battle.takeNewLog()));
  round++;
}
const code = battle.exportKifu(false, 1);
check(/^NC2-[A-Za-z0-9_-]+$/.test(code), "copy payload contains only a replay code");
check(code.length <= 2090, "replay fits the envelope");
const kifu = new wasm.Kifu(dex, code);
const index = JSON.parse(kifu.scenes());
checkEq(index.scenes.map(s => s.round), [...scenes.keys()], "all recorded bot decisions are listed");
for (const entry of index.scenes) {
  const scene = JSON.parse(kifu.scene(entry.round));
  checkEq(scene.view, scenes.get(entry.round).view, `exact view at ${entry.round}`);
  checkEq(scene.log, scenes.get(entry.round).log, `exact log at ${entry.round}`);
}
const point = index.scenes.at(-1).round;
const scene = JSON.parse(kifu.scene(point));
const other = scene.choices.find(c => c.input !== scene.played.input) || scene.played;
const forkJson = kifu.fork(point, other.input, pool);
const info = JSON.parse(wasm.forkInfo(dex, forkJson));
check(info.opponentView, "both information sets can be reconstructed");
const restored = wasm.Battle.fromFork(dex, forkJson, 17);
checkEq(JSON.parse(restored.stateView()), scene.view, "fork preserves the exact board");
const searcher = wasm.ProtocolSearcher.fromFork(dex, forkJson, pool, 23, 0.4);
searcher.step(100);
check(scene.choices.some(c => c.input === restored.resolveChoice(1, searcher.best())), "reconstructed bot selects a legal action");
const arena = new wasm.ForkArena(dex, forkJson, pool, JSON.stringify({bot: "protocol", foe: "protocol", iters: 100, foe_iters: 100, c: 0.4, seed: 11, max_steps: 3300}));
for (let arm = 0; arm < info.arms.length; arm++) {
  const result = JSON.parse(arena.play(0, arm));
  check(["win", "loss", "tie"].includes(result.outcome), "exact fork arena completes");
  checkEq([result.legality_drift, result.projections], [0, 0], "exact fork agent legality");
}
if (process.env.KIFU_FIXTURE) writeFileSync(process.env.KIFU_FIXTURE, JSON.stringify({ code, point, other: other.input }));
arena.free(); searcher.free(); restored.free(); kifu.free(); battle.free(); dex.free();
finish("kifu");
