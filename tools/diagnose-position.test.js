'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const { computeEntropyNorm, normalizeAction } = require('./diagnose-position');

const REPO = path.resolve(__dirname, '..');

test('action normalization handles varied formats', () => {
  assert.equal(normalizeAction('move bodyslam'), 'move bodyslam');
  assert.equal(normalizeAction('move:bodyslam'), 'move bodyslam');
  assert.equal(normalizeAction('move_bodyslam'), 'move bodyslam');
  assert.equal(normalizeAction('  Switch:2  '), 'switch 2');
});

test('entropy norm computes correctly', () => {
  // Single action -> 0 entropy
  assert.equal(computeEntropyNorm([{ visits: 100, frac: 1.0 }]), 0);

  // Equal 2 actions -> 1.0 entropy
  const equalTwo = [
    { visits: 50, frac: 0.5 },
    { visits: 50, frac: 0.5 },
  ];
  assert(Math.abs(computeEntropyNorm(equalTwo) - 1.0) < 1e-4);

  // Skewed actions -> 0 < entropy < 1
  const skewed = [
    { visits: 90, frac: 0.9 },
    { visits: 10, frac: 0.1 },
  ];
  const ent = computeEntropyNorm(skewed);
  assert(ent > 0 && ent < 1.0);
});

test('diagnose-position CLI --self-check outputs valid JSON', () => {
  const stdout = execFileSync('node', ['tools/diagnose-position.js', '--self-check'], {
    cwd: REPO,
    encoding: 'utf8',
  }).trim();
  const res = JSON.parse(stdout);
  assert.equal(res.status, 'OK');
  assert.equal(res.wasm_loaded, true);
  assert.equal(res.manifest_present, true);
});

test('bundle-core-sdk generates valid manifest and bundle', (t) => {
  const distDir = path.join(REPO, 'dist');
  const hadDist = fs.existsSync(distDir);
  t.after(() => {
    if (!hadDist) fs.rmSync(distDir, { recursive: true, force: true });
  });
  const stdout = execFileSync('node', ['tools/bundle-core-sdk.js'], {
    cwd: REPO,
    encoding: 'utf8',
  }).trim();
  const res = JSON.parse(stdout);
  assert.equal(res.status, 'ok');
  assert.equal(typeof res.git_sha, 'string');
  assert(res.files_count >= 11);

  const manifestPath = path.join(REPO, 'dist/nc2000-core-sdk/engine-manifest.json');
  assert(fs.existsSync(manifestPath));
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  assert.equal(manifest.schema_version, 'nc2000-sdk-v1');
  assert.equal(manifest.position_schema, 'nc2000-position-v1');
  assert(typeof manifest.wasm_sha256 === 'string');
});

test('diagnose-position --actual-foe-spec computes oracle hindsight without setPosition order error', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nc2000-diag-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));
  const pool = JSON.parse(
    fs.readFileSync(path.join(REPO, 'data/belief-pool-v3/belief-pool.json'), 'utf8'),
  );
  const mine = pool.teams[0];
  const foe = pool.teams[1];
  const pub = (sets, active) =>
    sets.map((s, i) => ({
      species: s.species,
      level: s.level,
      gender: s.gender ?? '',
      item_flag: !!s.item,
      hp_num: 100,
      hp_den: 100,
      appeared: i === active,
      active: i === active,
    }));
  const specPath = path.join(tmpDir, 'spec.json');
  const foePath = path.join(tmpDir, 'foe.json');
  fs.writeFileSync(
    specPath,
    JSON.stringify({
      schema: 'nc2000-position-v1',
      side: 0,
      turn: 1,
      own_sets: mine.sets,
      sides: [
        { mons: pub(mine.sets, 0), active: 0, party: [0, 1, 2] },
        { mons: pub(foe.sets, 0), active: 0 },
      ],
    }),
  );
  fs.writeFileSync(foePath, JSON.stringify(foe.sets));
  const stdout = execFileSync(
    'node',
    [
      'tools/diagnose-position.js',
      '--spec',
      specPath,
      '--actual-foe-spec',
      foePath,
      '--iters',
      '20,20',
      '--seeds',
      '1',
    ],
    { cwd: REPO, encoding: 'utf8' },
  ).trim();
  const report = JSON.parse(stdout);
  assert.equal(report.schema, 'nc2000-triage-report-v1');
  assert.equal(report.spec_valid, true);
  assert.equal(typeof report.oracle_hindsight_q_bot, 'number');
});
