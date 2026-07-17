// Certification probe for PS's Max Total Level enforcement at team preview
// (the rule the engine's preview choice space must mirror — see
// crates/engine/src/battle/choices.rs and the README M8 correction note).
//
// Certifies, against the live PS clone (BattleStream, format gen2nc2000):
//  (a) the rule binds on the PICKED mons' level sum, threshold 155 INCLUSIVE
//      (sum 155 accepted, sum 156 rejected);
//  (b) rejection is a CHOICE error on the player stream
//      (`|error|[Invalid choice] Can't choose for Team Preview: ...`), the
//      request stays open and a corrected choice is accepted — not a team
//      validation failure;
//  (c) lead ordering is irrelevant (the rule sums the picked set);
//  (d) a team whose EVERY triple exceeds 155 cannot exist: TeamValidator
//      rejects any team whose 3 lowest levels sum over 155 (rulesets.ts
//      maxtotallevel.onValidateTeam check 1), so a legal team always has a
//      legal triple.
//
// Usage: node tools/probe-max-total-level.js   (PS_ROOT built)
'use strict';
const fs = require('fs');
const path = require('path');
const { sim, FORMAT } = require('./ps');
const { BattleStream, getPlayerStreams, Teams, TeamValidator } = sim;

const REPO = path.join(__dirname, '..');
const pool = JSON.parse(
  fs.readFileSync(path.join(REPO, 'data/meta-pool-v0/meta-pool.json'), 'utf8'),
);

const idle = () => new Promise(res => setImmediate(() => setImmediate(res)));
const settle = async () => { for (let i = 0; i < 20; i++) await idle(); };

let failures = 0;
function check(cond, label) {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${label}`);
  if (!cond) failures++;
}

async function newBattle(team) {
  const bs = new BattleStream();
  const streams = getPlayerStreams(bs);
  const lines = { p1: [], p2: [] };
  for (const p of ['p1', 'p2']) {
    void (async () => {
      for await (const chunk of streams[p]) lines[p].push(...String(chunk).split('\n'));
    })().catch(() => {});
  }
  await streams.omniscient.write(
    `>start ${JSON.stringify({ formatid: FORMAT, seed: 'gen5,1,2,3,4' })}\n` +
    `>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(team) })}\n` +
    `>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(team) })}`,
  );
  await settle();
  return { bs, streams, lines };
}

async function tryChoice(b, slot, choice) {
  const before = b.lines[slot].length;
  await b.bs.write(`>${slot} ${choice}`);
  await settle();
  const fresh = b.lines[slot].slice(before);
  const err = fresh.find(l => l.startsWith('|error|'));
  const side = b.bs.battle.sides[slot === 'p1' ? 0 : 1];
  return { err, actions: side.choice.actions.length };
}

(async () => {
  // Team 0 (hc75-1st-show): levels 52,52,52,51,51,51 — the exact boundary.
  const t0 = pool.teams[0].sets;
  console.log(`team 0 levels: ${t0.map(s => s.level).join(',')}`);
  let b = await newBattle(t0);

  // 156 rejected (52+52+52), as a choice error; request stays open.
  let r = await tryChoice(b, 'p1', 'team 1, 2, 3');
  check(!!r.err && r.actions === 0, `sum 156 rejected as choice error: ${r.err}`);
  check(
    /Can't choose for Team Preview: .*total level of 156.*can't be above 155/.test(r.err || ''),
    'error text names the sum and the 155 cap',
  );
  check(b.bs.battle.turn === 0 || !b.bs.battle.started === false, 'battle has not advanced');

  // 155 accepted after the rejection (52+52+51) — the request stayed open.
  r = await tryChoice(b, 'p1', 'team 1, 2, 4');
  check(!r.err && r.actions === 3, 'sum 155 accepted (inclusive threshold) after a rejection');

  // Lead ordering irrelevant: same 155 set, different lead, for p2. This
  // was the last owed choice, so acceptance = the battle commits + advances
  // (side.choice is reset by commitChoices; can't count actions here).
  r = await tryChoice(b, 'p2', 'team 4, 2, 1');
  check(!r.err, 'sum 155 accepted with a different lead order');
  await settle();
  check(b.bs.battle.turn === 1, 'both previews in: battle advanced to turn 1');
  void b.bs.destroy();

  // Team 1 (hc75-2nd-toriyama): 55,55,50,50,50,50. An illegal SET stays
  // illegal under any lead ordering (the rule sums the picked set).
  const t1 = pool.teams[1].sets;
  console.log(`team 1 levels: ${t1.map(s => s.level).join(',')}`);
  b = await newBattle(t1);
  r = await tryChoice(b, 'p1', 'team 3, 1, 2'); // 50+55+55 = 160, lead 50
  check(!!r.err && r.actions === 0, `sum 160 rejected regardless of lead order: ${r.err}`);
  r = await tryChoice(b, 'p1', 'team 1, 3, 4'); // 55+50+50 = 155
  check(!r.err && r.actions === 3, 'sum 155 accepted (two-55 team keeps one 55 max)');
  void b.bs.destroy();

  // (d) A team whose every triple exceeds 155 is validator-rejected:
  // 3 lowest levels 52+52+52 = 156 > 155 (onValidateTeam check 1).
  const allBad = t0.map(s => ({ ...s, level: 52 }));
  const validator = new TeamValidator(FORMAT);
  const errs = validator.validateTeam(Teams.unpack(Teams.pack(allBad)));
  check(
    !!errs && errs.some(e => /combined levels .* is 156, above the format's total level limit of 155/.test(e)),
    `all-triples-illegal team rejected by TeamValidator: ${errs && errs[0]}`,
  );

  console.log(failures === 0 ? '\nCERTIFIED: all checks passed' : `\n${failures} FAILURES`);
  process.exit(failures === 0 ? 0 : 1);
})().catch(err => { console.error(err); process.exit(2); });
