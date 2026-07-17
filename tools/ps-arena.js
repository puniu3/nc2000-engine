// M15a PS sim-stream harness: our bot plays complete games against a
// PS-hosted battle through PS's own BattleStream, from PLAYER-VISIBLE
// information only (its player stream: battle lines + |request| JSON).
// The bot side runs in-process via the wasm nodejs build's
// ProtocolSearcher (protocol→state importer + M10 belief + blind search).
//
// Gates (see README M15a):
//  a. protocol soundness — zero |error| rejections, zero desyncs/timeouts;
//  b. state tracking — at every decision point the synthesized battle's
//     public fields are asserted against PS's true (omniscient) state;
//  c. strength — W/L vs RandomPlayerAI.
//
// Usage:
//   node tools/ps-arena.js --games 50 --mode blind|open --opp random|maxbp \
//     --oppteams pool|fixtures|mixed --iters 300 --seed 1 [--quiet] \
//     [--affected]   # both sides from the Max-Total-Level-affected teams
'use strict';
const fs = require('fs');
const path = require('path');
const { sim, prng, rpai, FORMAT } = require('./ps');
const { BattleStream, getPlayerStreams, Teams, TeamValidator, Dex } = sim;
const { PRNG } = prng;
const { RandomPlayerAI } = rpai;

const REPO = path.join(__dirname, '..');
const wasm = require(path.join(REPO, 'crates/wasm/pkg-node/nc2000_wasm.js'));

const args = {};
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (!a.startsWith('--')) continue;
  const key = a.slice(2);
  if (i + 1 < process.argv.length && !process.argv[i + 1].startsWith('--')) {
    args[key] = process.argv[++i];
  } else {
    args[key] = true;
  }
}
const GAMES = parseInt(args.games || '10', 10);
const MODE = args.mode || 'blind'; // blind | open
const OPP = args.opp || 'random'; // random | maxbp
const OPPTEAMS = args.oppteams || 'mixed'; // pool | fixtures | mixed
const ITERS = parseInt(args.iters || '300', 10);
const SEED = parseInt(args.seed || '1', 10);
const QUIET = !!args.quiet;
const VERIFY = !args.noverify;
// --affected: restrict BOTH sides to pool teams with >155 triples (the
// Max-Total-Level-affected list) — the preview-fix certification population.
const AFFECTED_ONLY = !!args.affected;

const psDex = Dex.mod('gen2stadium2');
const validator = new TeamValidator(FORMAT);
const pool = JSON.parse(fs.readFileSync(path.join(REPO, 'data/meta-pool-v0/meta-pool.json'), 'utf8'));
const poolJson = JSON.stringify(pool);

/// Pool indices with at least one >155 triple (illegal picks existed pre-fix).
const affectedIdx = pool.teams
  .map((t, i) => {
    const lv = t.sets.map(s => s.level);
    for (let a = 0; a < 6; a++) {
      for (let b = a + 1; b < 6; b++) {
        for (let c = b + 1; c < 6; c++) {
          if (lv[a] + lv[b] + lv[c] > 155) return i;
        }
      }
    }
    return -1;
  })
  .filter(i => i >= 0);
const poolPick = master => {
  if (!AFFECTED_ONLY) return pool.teams[master.random(pool.teams.length)].sets;
  return pool.teams[affectedIdx[master.random(affectedIdx.length)]].sets;
};

// fixture opponent teams (validator-clean random teams — off-pool)
const fixtureTeams = [];
for (const dir of ['full', 'puredata']) {
  const d = path.join(REPO, 'fixtures/corpus-v1', dir);
  if (!fs.existsSync(d)) continue;
  for (const f of fs.readdirSync(d).sort()) {
    if (!f.endsWith('.json') || f.includes('DIVERGED')) continue;
    const fx = JSON.parse(fs.readFileSync(path.join(d, f), 'utf8'));
    fixtureTeams.push(fx.p1team, fx.p2team);
    if (fixtureTeams.length >= 40) break;
  }
  if (fixtureTeams.length >= 40) break;
}

// baked pair tables (fed to the searcher for table-answered previews)
const pairDir = path.join(REPO, 'data/preview-tables-v0');
const pairFiles = fs.existsSync(pairDir)
  ? fs.readdirSync(pairDir).filter(f => f.startsWith('pair-') && f.endsWith('.json'))
  : [];

const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
// PS normalizes typed hidden powers to the plain id; our engine keeps
// typed ids — compare canonically.
const nMove = id => (id || '').startsWith('hiddenpower') ? 'hiddenpower' : id;

// ------------------------------------------------------------- opponents
function parseLevel(details) {
  const m = /, L(\d+)/.exec(details);
  return m ? parseInt(m[1], 10) : 100;
}

/// Random 3-of-6 preview pick respecting Max Total Level = 155.
class LevelCapPickAI extends RandomPlayerAI {
  chooseTeamPreview(team) {
    const levels = team.map(p => parseLevel(p.details));
    for (let tries = 0; tries < 100; tries++) {
      const order = [1, 2, 3, 4, 5, 6];
      this.prng.shuffle(order);
      const pick = order.slice(0, 3);
      const total = pick.reduce((a, i) => a + levels[i - 1], 0);
      if (total <= 155) return `team ${pick.join('')}`;
    }
    return 'team 123';
  }
}

/// Max-base-power scripted player: highest-BP usable move, else random.
class MaxBPAI extends LevelCapPickAI {
  chooseMove(active, moves) {
    let best = null;
    let bestBp = -1;
    for (const m of moves) {
      const bp = psDex.moves.get(m.move.id || m.move.move).basePower || 0;
      if (bp > bestBp) {
        bestBp = bp;
        best = m;
      }
    }
    return best ? best.choice : super.chooseMove(active, moves);
  }
}

// ------------------------------------------------------------ gate b check
function px(hp, maxhp) {
  if (hp <= 0) return 0;
  const p = Math.floor((48 * hp) / maxhp);
  return p === 0 ? 1 : p;
}

/// Compare the synthesized battle's public fields against PS's true state.
function verifyState(view, battle, ourSide, mismatches, ctx) {
  const miss = what => mismatches.push(`${ctx}: ${what}`);
  if (view.turn !== battle.turn) miss(`turn ${view.turn} vs ${battle.turn}`);
  const tw = battle.field.weather || '';
  const vw = view.field.weather || '';
  if (toID(vw) !== toID(tw)) miss(`weather ${vw} vs ${tw}`);
  for (let s = 0; s < 2; s++) {
    const own = s === ourSide;
    const vSide = view.sides[s];
    const tSide = battle.sides[s];
    // side conditions (key sets)
    const vc = [...vSide.sideConditions].sort();
    const tc = Object.keys(tSide.sideConditions).sort();
    if (JSON.stringify(vc) !== JSON.stringify(tc)) {
      miss(`side ${s} conditions ${JSON.stringify(vc)} vs ${JSON.stringify(tc)}`);
    }
    if (own) {
      // display order must match exactly
      const vOrder = vSide.party.map(p => toID(p.species));
      const tOrder = tSide.pokemon.map(p => p.species.id);
      if (JSON.stringify(vOrder) !== JSON.stringify(tOrder)) {
        miss(`own order ${vOrder} vs ${tOrder}`);
      }
    }
    for (const tp of tSide.pokemon) {
      const vp = vSide.party.find(p => toID(p.species) === tp.species.id)
        || vSide.party.find(p => toID(p.species) === tp.baseSpecies?.id);
      if (!vp) {
        // benched hidden picks: the imputation may hold other roster mons —
        // only appeared/true-picked mons are asserted
        if (tp.previouslySwitchedIn > 0 || tp.isActive) miss(`missing ${tp.species.id}`);
        continue;
      }
      if (vp.level !== tp.level) miss(`${tp.species.id} level ${vp.level} vs ${tp.level}`);
      const tFnt = !!tp.fainted;
      if (vp.fainted !== tFnt) {
        miss(`${tp.species.id} fainted ${vp.fainted} vs ${tFnt}`);
        continue;
      }
      // boosts exact
      for (const b of ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']) {
        const tb = tp.boosts[b] || 0;
        if ((vp.boosts[b] || 0) !== tb) miss(`${tp.species.id} boost ${b} ${vp.boosts[b]} vs ${tb}`);
      }
      if (!tFnt) {
        const tStatus = tp.status === 'fnt' ? '' : tp.status || '';
        const vStatus = vp.status === 'fnt' ? '' : vp.status || '';
        if (vStatus !== tStatus) miss(`${tp.species.id} status ${vStatus} vs ${tStatus}`);
      }
      if (own) {
        if (vp.hp !== tp.hp || vp.maxhp !== tp.maxhp) {
          miss(`${tp.species.id} hp ${vp.hp}/${vp.maxhp} vs ${tp.hp}/${tp.maxhp}`);
        }
        if (toID(vp.item || '') !== toID(tp.item || '')) {
          miss(`${tp.species.id} item ${vp.item} vs ${tp.item}`);
        }
        if (!tp.transformed) {
          for (const ts of tp.moveSlots) {
            const vm = vp.moves.find(m => nMove(m.id) === nMove(ts.id));
            if (!vm) miss(`${tp.species.id} missing own move ${ts.id}`);
            else if (vm.pp !== ts.pp) miss(`${tp.species.id} ${ts.id} pp ${vm.pp} vs ${ts.pp}`);
          }
        }
      } else if (!tFnt) {
        // HP within the announced 1/48 bucket
        const tpx = px(tp.hp, tp.maxhp);
        const vpx = px(vp.hp, vp.maxhp);
        if (tpx !== vpx) miss(`${tp.species.id} hp bucket ${vpx} vs ${tpx}`);
        // PP marks: any truth usage must be a tracked reveal with the exact
        // count; skip transformed / overlaid movesets
        const overlaid = tp.transformed || tp.volatiles['transform'] || tp.volatiles['mimic'];
        if (!overlaid) {
          for (const ts of tp.moveSlots) {
            const used = ts.maxpp - ts.pp;
            const vm = vp.moves.find(m => nMove(m.id) === nMove(ts.id));
            if (!vm) {
              if (used > 0) miss(`${tp.species.id} used ${ts.id} not in imputation`);
              continue;
            }
            const vUsed = vm.maxpp - vm.pp;
            if (vUsed !== used) miss(`${tp.species.id} ${ts.id} used ${vUsed} vs ${used}`);
          }
        }
      }
    }
  }
}

// --------------------------------------------------------------- one game
async function playGame(gameIdx, master, counters) {
  const ourSide = gameIdx % 2; // alternate p1/p2
  const ourSlot = `p${ourSide + 1}`;
  const oppSlot = `p${2 - ourSide}`;

  // teams (--affected: both sides from the Max-Total-Level-affected list)
  const ourTeam = poolPick(master);
  let oppTeam;
  const src = AFFECTED_ONLY ? 'pool'
    : OPPTEAMS === 'mixed' ? (master.random(2) === 0 ? 'pool' : 'fixtures') : OPPTEAMS;
  if (src === 'pool') oppTeam = poolPick(master);
  else oppTeam = fixtureTeams[master.random(fixtureTeams.length)];

  for (const [name, team] of [['ours', ourTeam], ['opp', oppTeam]]) {
    const errs = validator.validateTeam(Teams.unpack(Teams.pack(team)));
    if (errs) throw new Error(`${name} team invalid: ${errs.join('; ')}`);
  }

  const hex = () => master.random(0x10000).toString(16).padStart(4, '0');
  const battleSeed = `gen5,${hex()}${hex()}${hex()}${hex()}`;

  const battleStream = new BattleStream();
  const streams = getPlayerStreams(battleStream);
  const oppStream = streams[oppSlot];
  const AI = OPP === 'maxbp' ? MaxBPAI : LevelCapPickAI;
  const oppAI = new AI(oppStream, { seed: `gen5,${hex()}${hex()}${hex()}0002` });
  void oppAI.start().catch(err => {
    counters.oppErrors.push(`game ${gameIdx}: opp AI error: ${err.message}`);
  });

  // our searcher
  const dex = new wasm.Dex();
  const searcher = new wasm.ProtocolSearcher(dex, ourSide, poolJson, SEED * 1000 + gameIdx);
  searcher.setOwnTeam(JSON.stringify(ourTeam));
  if (MODE === 'open') searcher.pinOpponent(JSON.stringify(oppTeam));
  for (const f of pairFiles) {
    try {
      searcher.addPair(fs.readFileSync(path.join(pairDir, f), 'utf8'));
    } catch (e) {
      // stale (pre-Max-Total-Level) or malformed table: treat as missing
      if (gameIdx === 0) console.warn(`pair ${f} rejected: ${e.message || e}`);
    }
  }

  const p1team = ourSide === 0 ? ourTeam : oppTeam;
  const p2team = ourSide === 0 ? oppTeam : ourTeam;
  await streams.omniscient.write(
    `>start ${JSON.stringify({ formatid: FORMAT, seed: battleSeed })}\n` +
    `>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}\n` +
    `>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`
  );
  // guard against stall wars
  const guard = setInterval(() => {
    if (battleStream.battle && battleStream.battle.turn > 300) void battleStream.write('>forcetie');
  }, 2000);

  const ourStream = streams[ourSlot];
  const idle = () => new Promise(res => setImmediate(() => setImmediate(() => res(null))));
  const sleep = ms => new Promise(res => setTimeout(res, ms));

  let nextP = null;
  let pendingRequest = null;
  let lineBuffer = [];
  let decisions = 0;
  const mismatches = [];
  let rejected = 0;
  let desync = false;
  const t0 = Date.now();

  const actOnRequest = () => {
    const req = pendingRequest;
    pendingRequest = null;
    if (lineBuffer.length) {
      searcher.pushLines(JSON.stringify(lineBuffer));
      lineBuffer = [];
    }
    const owes = searcher.onRequest(req);
    if (!owes) return;
    decisions++;
    if (VERIFY) {
      const view = JSON.parse(searcher.stateView());
      verifyState(view, battleStream.battle, ourSide, mismatches, `game ${gameIdx} d${decisions}`);
    }
    let choice = searcher.bakedPreview();
    if (!choice) {
      searcher.step(ITERS);
      choice = searcher.best();
    }
    if (!choice) throw new Error('searcher returned no choice');
    void battleStream.write(`>${ourSlot} ${choice}`);
  };

  while (true) {
    if (!nextP) nextP = ourStream.read().then(v => ({ v }));
    const r = await Promise.race([nextP, idle()]);
    if (r === null) {
      // stream quiescent: everything PS had to say is in
      if (pendingRequest) {
        actOnRequest();
        continue;
      }
      if (battleStream.battle && battleStream.battle.ended) break;
      if (Date.now() - t0 > 300000) {
        desync = true;
        break;
      }
      await sleep(2);
      continue;
    }
    nextP = null;
    if (r.v === null || r.v === undefined) break; // stream end
    for (const line of String(r.v).split('\n')) {
      if (!line) continue;
      if (line.startsWith('|request|')) {
        pendingRequest = line.slice(9);
      } else if (line.startsWith('|error|')) {
        rejected++;
        counters.rejections.push(`game ${gameIdx}: ${line}`);
      } else if (line.startsWith('|')) {
        lineBuffer.push(line);
      }
    }
  }
  clearInterval(guard);

  const battle = battleStream.battle;
  const winner = battle && battle.winner ? battle.winner : '';
  const ourName = ourSide === 0 ? 'P1' : 'P2';
  const result = winner === '' ? 'T' : winner === ourName ? 'W' : 'L';
  const metrics = JSON.parse(searcher.metrics());
  searcher.free();
  dex.free();
  try {
    void battleStream.destroy();
  } catch { /* stream already ended */ }

  counters.games++;
  counters[result]++;
  counters.decisions += decisions;
  counters.turns += battle ? battle.turn : 0;
  counters.mismatches.push(...mismatches);
  counters.rejectedTotal += rejected;
  counters.desyncs += desync ? 1 : 0;
  counters.legalityDrift += metrics.legalityDrift;
  counters.projections += metrics.projections;
  if (!QUIET) {
    console.log(
      `game ${gameIdx}: ${result} as ${ourSlot} vs ${src}/${OPP} in ${battle ? battle.turn : '?'} turns, ` +
      `${decisions} decisions, ${mismatches.length} mismatches, ${rejected} rejections` +
      (metrics.legalityDrift ? `, drift ${metrics.legalityDrift}` : '') +
      (metrics.projections ? `, proj ${metrics.projections}` : '') +
      (desync ? ', DESYNC' : '')
    );
  }
}

// ------------------------------------------------------------------- main
(async () => {
  const master = new PRNG(`gen5,${SEED.toString(16).padStart(16, '0')}`);
  const counters = {
    games: 0, W: 0, L: 0, T: 0, decisions: 0, turns: 0,
    mismatches: [], rejections: [], oppErrors: [],
    rejectedTotal: 0, desyncs: 0, legalityDrift: 0, projections: 0,
  };
  for (let i = 0; i < GAMES; i++) {
    await playGame(i, master, counters);
  }
  console.log('----------------------------------------------------------');
  console.log(
    `${MODE} vs ${OPP} (${OPPTEAMS} teams, iters ${ITERS}, seed ${SEED}): ` +
    `${counters.W}W ${counters.L}L ${counters.T}T over ${counters.games} games`
  );
  console.log(
    `decisions ${counters.decisions}, avg turns ${(counters.turns / counters.games).toFixed(1)}, ` +
    `rejections ${counters.rejectedTotal}, desyncs ${counters.desyncs}, ` +
    `state mismatches ${counters.mismatches.length}, ` +
    `legality drift ${counters.legalityDrift}, projections ${counters.projections}`
  );
  for (const m of counters.mismatches.slice(0, 30)) console.log('  MISMATCH', m);
  for (const m of counters.rejections.slice(0, 10)) console.log('  REJECTED', m);
  for (const m of counters.oppErrors.slice(0, 5)) console.log('  OPP', m);
  const bad = counters.rejectedTotal + counters.desyncs + counters.mismatches.length;
  process.exit(bad > 0 ? 1 : 0);
})().catch(err => { console.error(err); process.exit(2); });
