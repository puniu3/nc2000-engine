// Build the Node WASM target first. Run from any directory:
// node web/tests/nash-refresh.mjs GAMES.jsonl NEW_OUTPUT_DIR [workers]
// Each game specifies the two team IDs, their search seeds and the battle seed.
// Optional picks force a preview action in a paired counterfactual.
// Optional mixtures name each side's disclosed team distribution. Both players
// then use live preview and receive only the opposing pool and player frames.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import wasm from '../../crates/wasm/pkg-node/nc2000_wasm.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
const data = name => JSON.parse(read('data/' + name + '.json'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const live = data('live-search');
const { c } = data('search-profiles').blind;

const candidateFile = 'nash-refresh-v1/candidate';
const incumbentFile = 'kodoku-r3/conclusion-equilibrium-deme';
const teamFiles = [
  'kodoku-beq-v1/novel_evolved_teams',
  'team-pool-v2/team-pool',
  incumbentFile,
];

async function start() {
  const [, , taskFile, output, n = '8'] = process.argv;
  const source = fs.readFileSync(taskFile, 'utf8').trim();
  const records = source.startsWith('[')
    ? JSON.parse(source)
    : source.split('\n').map(line => JSON.parse(line));
  const tasks = records.map(({ block, cell, rowSeat, ids, seeds, battleSeed, picks, mixtures, policies }) =>
    ({ block, cell, rowSeat, ids, seeds, battleSeed, ...(picks ? { picks } : {}), ...(mixtures ? { mixtures } : {}), ...(policies ? { policies } : {}) }));
  assert(tasks.length && Number.isSafeInteger(+n) && +n > 0);
  fs.mkdirSync(output); // refuse to mix builds or overwrite an earlier run
  const logDir = process.env.NASH_SAVE_LOGS === '1' ? path.join(output, 'logs') : null;
  if (logDir) fs.mkdirSync(logDir);
  const needsDefaultPool = tasks.some(task => !task.mixtures);
  const inputs = [...teamFiles, 'live-search', 'search-profiles'];
  if (needsDefaultPool) inputs.push('belief-pool-v3/belief-pool');
  const hasDisclosure = tasks.some(task => task.mixtures);
  if (hasDisclosure) inputs.push(candidateFile);
  const poolsText = process.env.NASH_POOLS ? fs.readFileSync(process.env.NASH_POOLS, 'utf8') : null;
  const needsOld = tasks.some(t => t.policies?.includes('old'));
  const oldPath = needsOld ? path.resolve(process.env.NASH_OLD_WASM) : null;
  const stamp = {
    ...(poolsText ? { poolsSha256: hash(poolsText) } : {}),
    ...(needsOld ? { oldWasm: hash(fs.readFileSync(path.join(oldPath, 'nc2000_wasm_bg.wasm'))) } : {}),
    tasks: hash(fs.readFileSync(taskFile)),
    wasm: hash(fs.readFileSync(path.join(root, 'crates/wasm/pkg-node/nc2000_wasm_bg.wasm'))),
    data: Object.fromEntries(inputs.map(k => [k, hash(read('data/' + k + '.json'))])),
  };
  fs.writeFileSync(path.join(output, 'manifest.json'), JSON.stringify(stamp));
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(+n, tasks.length) }, () => new Promise((resolve, reject) => {
    const thread = new Worker(new URL(import.meta.url), { workerData: { needsDefaultPool, hasDisclosure, oldPath, poolsText, logDir } });
    thread.on('error', reject);
    thread.on('exit', code => code ? reject(Error('worker exit ' + code)) : resolve());
    thread.on('message', result => {
      if (result) fs.writeFileSync(path.join(output, result.index + '.json'), JSON.stringify(result));
      if (next === tasks.length) thread.postMessage(null);
      else {
        const index = next++;
        thread.postMessage({ ...tasks[index], index });
      }
    });
  })));
  console.log(JSON.stringify({ games: tasks.length, output }));
}

async function worker() {
  const pool = workerData.needsDefaultPool ? read('data/belief-pool-v3/belief-pool.json') : null;
  const dex = new wasm.Dex();
  const old = workerData.oldPath ? (await import(pathToFileURL(path.join(workerData.oldPath, 'nc2000_wasm.js')))).default : null;
  const oldDex = old ? new old.Dex() : null;
  const validator = new wasm.Validator(dex);
  const teams = new Map();
  for (const name of teamFiles) {
    for (const t of data(name).teams) teams.set(t.id, JSON.stringify(t.sets));
  }
  const disclosed = workerData.hasDisclosure ? {
    candidate: data(candidateFile).teams,
    incumbent: data(incumbentFile).teams,
    deme: [{ id: 'deme-r1-00-s0030', weight: 1, sets: JSON.parse(teams.get('deme-r1-00-s0030')) }],
  } : {};
  if (workerData.poolsText) {
    const extra = JSON.parse(workerData.poolsText);
    for (const t of extra.teams) {
      assert(!teams.has(t.id), 'extra team ID already exists: ' + t.id);
      assert(JSON.parse(validator.validateTeam(JSON.stringify(t.sets))).ok, 'invalid team: ' + t.id);
      teams.set(t.id, JSON.stringify(t.sets));
    }
    for (const [name, members] of Object.entries(extra.mixtures)) {
      assert(!Object.hasOwn(disclosed, name), 'duplicate mixture: ' + name);
      assert(new Set(members.map(t => t.id)).size === members.length);
      disclosed[name] = members.map(({ id, weight }) => ({ id, weight, sets: JSON.parse(teams.get(id)) }));
    }
  }
  for (const members of Object.values(disclosed)) {
    assert(Math.abs(members.reduce((sum, t) => sum + t.weight, 0) - 1) < 1e-10);
    for (const t of members) {
      assert(t.weight > 0);
      assert.deepEqual(t.sets, JSON.parse(teams.get(t.id)));
    }
  }
  parentPort.on('message', task => {
    if (!task) { parentPort.close(); return; }
    const { ids, seeds, battleSeed, picks = [], mixtures, policies = ['current', 'current'] } = task;
    assert(policies.length === 2 && policies.every(p => ['current', 'old', 'current81'].includes(p)));
    if (mixtures) {
      assert.equal(mixtures.length, 2);
      assert(picks.length === 0 || picks.length === 2);
      for (const [side, mix] of mixtures.entries()) assert(disclosed[mix].some(t => t.id === ids[side]));
    }
    const sets = ids.map(id => {
      const team = teams.get(id);
      assert(team && JSON.parse(validator.validateTeam(team)).ok, 'invalid ' + id);
      return team;
    });
    const battle = new wasm.Battle(dex, ...sets, wasm.deriveBattleSeed(battleSeed));
    const channels = [new wasm.PlayerChannel(0), new wasm.PlayerChannel(1)];
    const agents = sets.map((team, side) => {
      const prior = mixtures ? JSON.stringify({ teams: disclosed[mixtures[1-side]] }) : pool;
      const historical = policies[side] === 'old';
      const api = historical ? old : wasm;
      const agent = new api.ProtocolSearcher(historical ? oldDex : dex, side, prior, seeds[side], c);
      agent.setOwnTeam(team);
      return agent;
    });
    const preview = [];
    const began = performance.now();
    const previewBeliefs = [];
    const transcript = createHash('sha256');
    let first = true;
    while (!battle.outcome() && battle.turn() < 500) {
      const inputs = JSON.parse(battle.needsChoice()).map((needs, side) => {
        if (!needs) return null;
        const frame = JSON.parse(channels[side].frame(battle));
        const agent = agents[side];
        agent.pushLines(JSON.stringify(frame.lines));
        assert(agent.onRequest(JSON.stringify(frame.request)));
        let input;
        let source = 'live';
        if (mixtures) {
          // Audit externally. Neither agent receives the opposing realized ID.
          const belief = JSON.parse(agent.beliefInfo());
          assert(!belief.fallback && belief.candidates.includes(ids[1-side]), 'disclosure lost the played team');
          if (first) previewBeliefs[side] = belief.candidates;
        }
        if (first && !mixtures && picks[side]) {
          const legal = pick => frame.legal_actions.some(a => a.replaceAll(' ', '') === pick.replaceAll(' ', ''));
          assert(legal(picks[side]), 'illegal forced selection');
          source = 'forced';
          input = picks[side];
        }
        if (!input) {
          if (policies[side] === 'current') agent.stepAdaptiveBattle(first ? live.unknownPreview : live.battle.adaptive);
          else agent.step(first ? live.unknownPreview : 81000);
          input = JSON.parse(agent.commitDecision('legacy')).input;
        }
        // Preserve preview search and decision RNG before applying the paired override.
        if (first && mixtures && picks[side]) {
          assert(frame.legal_actions.some(a => a.replaceAll(' ', '') === picks[side].replaceAll(' ', '')));
          input = picks[side];
          source = 'forced';
        }
        if (first) preview[side] = { source, input };
        return input;
      });
      transcript.update(JSON.stringify(inputs) + '\n');
      for (const [side, input] of inputs.entries()) if (input) battle.applyChoice(side, input);
      first = false;
    }
    const result = { ...task, preview, outcome: battle.outcome() ?? 'cap', turns: battle.turn(),
      actionSha256: transcript.digest('hex'), ms: Math.round(performance.now() - began), ...(mixtures ? { previewBeliefs } : {}) };
    if (workerData.logDir) fs.writeFileSync(path.join(workerData.logDir, task.index + '.json'), battle.takeNewLog());
    for (const object of [...agents, ...channels, battle]) object.free();
    parentPort.postMessage(result);
  });
  parentPort.postMessage(null);
}

if (isMainThread) await start(); else await worker();
