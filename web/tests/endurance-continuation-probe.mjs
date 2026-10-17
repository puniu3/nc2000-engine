// Post-hoc diagnostic of a completed continuation; does not add a battle trial.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {dir, fork} from './endurance-runtime.mjs';
const [file, requestedTurn] = process.argv.slice(2);
const game = JSON.parse(fs.readFileSync(`${dir}/continuations/${file}`, 'utf8'));
assert(game.outcome, 'diagnose only terminal records');
const target = game.trace.findIndex(row => row.turn === Number(requestedTurn));
assert(target >= 0, 'requested turn missing');
const {battle, bots} = fork(game.id, game.decision, game.chanceSeed, game.searchSeeds);
const rows = [];
try {
  for (let index = 0; index <= target; index++) {
    const saved = game.trace[index];
    const state = JSON.parse(battle.stateView());
    assert.deepEqual(state, saved.state);
    for (const [side, expected] of saved.picks) {
      bots[side].observe(battle);
      if (index !== target) continue;
      for (const budget of [81000, 243000, 729000]) {
        bots[side].step(budget - bots[side].iterations());
        const policy = JSON.parse(bots[side].rootPolicy());
        const best = bots[side].best();
        let played = best;
        if (side === game.side && game.action === 'conserve-heal' && best === 'move softboiled') {
          const own = state.sides[side].party[state.sides[side].active];
          const foe = state.sides[1-side].party[state.sides[1-side].active];
          if (own.hp > .6 * own.maxhp && own.status !== 'frz' && foe.species === 'Blissey' && foe.status !== 'frz'
              && JSON.parse(battle.legalChoices(side)).some(c => c.input === 'move icebeam')) played = 'move icebeam';
        }
        assert.equal(policy.iterations, budget);
        if (budget === 81000) assert.equal(played, expected, 'recorded production choice did not reproduce');
        rows.push({side, budget, best, played, policy});
      }
    }
    if (index === target) break;
    for (const [side, choice] of saved.picks) battle.applyChoice(side, choice);
  }
  const result = {scope: 'Post-hoc budget diagnostic; excluded from win-rate comparisons',
    file, decision: target, turn: battle.turn(), state: JSON.parse(battle.stateView()), rows};
  fs.mkdirSync(`${dir}/continuation-probes`, {recursive: true});
  fs.writeFileSync(`${dir}/continuation-probes/${file.replace('.json', '')}-t${requestedTurn}.json`, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify({file, turn: result.turn, choices: rows.map(r => [r.side, r.budget, r.best])}));
} finally {
  bots.forEach(bot => bot.free());
  battle.free();
}
