// Native≡wasm parity: replay golden fixtures through the wasm build and
// assert bit-exactness at every snapshot point — PRNG seed (RNG-consumption
// order), turn, and the protocol log chunk — plus the final state view
// (hp/status/fainted per mon) and outcome against the fixture's record.
"use strict";

const { wasm, loadFixture, check, checkEq, finish } = require("./common");

const FIXTURES = [
  "puredata/battle-001.json",
  "full/battle-001.json",
  "full/battle-015.json",
];

const dex = new wasm.Dex();

for (const rel of FIXTURES) {
  const fx = loadFixture(rel);
  const battle = new wasm.Battle(
    dex,
    JSON.stringify(fx.p1team),
    JSON.stringify(fx.p2team),
    fx.seed
  );

  // snapshots[0] = right after construction (afterLine -1)
  const snaps = fx.snapshots.slice();
  let snapIdx = 0;
  const compareSnap = (snap, where) => {
    checkEq(battle.seed(), snap.prngSeed, `${rel} ${where}: prng seed`);
    checkEq(battle.turn(), snap.turn, `${rel} ${where}: turn`);
    checkEq(
      JSON.parse(battle.takeNewLog()),
      snap.log,
      `${rel} ${where}: log chunk`
    );
  };
  check(snaps[0].afterLine === -1, `${rel}: snapshots[0].afterLine === -1`);
  compareSnap(snaps[0], "snap0 (construction)");
  snapIdx = 1;

  for (const line of fx.choices) {
    const side = line.side === "p1" ? 0 : 1;
    battle.applyChoice(side, line.choice);
    if (snapIdx < snaps.length && snaps[snapIdx].afterLine === line.index) {
      compareSnap(snaps[snapIdx], `snap${snapIdx} (after line ${line.index})`);
      snapIdx += 1;
    }
  }
  checkEq(snapIdx, snaps.length, `${rel}: all snapshots visited`);

  // final outcome vs the fixture's record
  const want =
    fx.result.winner === "P1" ? "p1" : fx.result.winner === "P2" ? "p2" : "tie";
  checkEq(battle.outcome(), want, `${rel}: outcome`);
  checkEq(battle.turn(), fx.result.turns, `${rel}: final turn`);

  // final state view vs the last snapshot's essence (hp/status/fainted in
  // display order; snapshot pokemon are in PS side.pokemon order = party)
  const view = JSON.parse(battle.stateView());
  const last = snaps[snaps.length - 1];
  for (let s = 0; s < 2; s++) {
    const vparty = view.sides[s].party;
    const fparty = last.sides[s].pokemon;
    checkEq(vparty.length, fparty.length, `${rel}: side ${s} party size`);
    for (let i = 0; i < fparty.length; i++) {
      checkEq(
        { hp: vparty[i].hp, maxhp: vparty[i].maxhp, fainted: vparty[i].fainted,
          status: vparty[i].status },
        { hp: fparty[i].hp, maxhp: fparty[i].maxhp, fainted: fparty[i].fainted,
          status: fparty[i].status },
        `${rel}: side ${s} mon ${i} (${fparty[i].species}) final state`
      );
    }
    checkEq(
      view.sides[s].pokemonLeft,
      last.sides[s].pokemonLeft,
      `${rel}: side ${s} pokemonLeft`
    );
  }
  console.log(
    `  ${rel}: ${fx.choices.length} choices, ${snaps.length} snapshots bit-exact`
  );
}

finish("parity");
