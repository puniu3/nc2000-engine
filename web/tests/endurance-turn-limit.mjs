// Engine contract check only: this synthetic switching sequence is not a bot trial.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {dir, replay} from './endurance-runtime.mjs';
const {battle, bots} = replay('g4', 12);
const startTurn = battle.turn();
let rounds = 0, lastLog = [];
try {
  battle.takeNewLog();
  while (!battle.outcome() && rounds < 1100) {
    const choices = [0, 1].map(side => {
      const choice = JSON.parse(battle.legalChoices(side)).find(c => c.input.startsWith('switch '));
      assert(choice, 'synthetic switch sequence became illegal');
      return choice.input;
    });
    for (let side = 0; side < 2; side++) battle.applyChoice(side, choices[side]);
    lastLog = JSON.parse(battle.takeNewLog());
    rounds++;
  }
  assert.equal(battle.outcome(), 'tie');
  assert(lastLog.some(line => line.includes('You have hit the turn limit')));
  const result = {scope: 'Synthetic engine contract; excluded from every bot outcome comparison',
    startTurn, endTurn: battle.turn(), rounds, outcome: battle.outcome(), terminalLog: lastLog};
  fs.writeFileSync(`${dir}/turn-limit-audit.json`, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
} finally {
  bots.forEach(bot => bot.free());
  battle.free();
}
