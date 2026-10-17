'use strict';

// Reuse the existing choices for the explicitly approved low-Attack catalog
// adjustment. This changes identity keys, not search results or search budgets.
// node tools/rekey-low-attack-preview.js BASELINE_POOL OUTPUT_ARTIFACT
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { optimizeLowAttack } = require('./optimize-low-attack');
const { attractGenderCensus } = require('./attract-gender-census');
const wasm = require('../crates/wasm/pkg-node/nc2000_wasm.js');

const root = path.resolve(__dirname, '..');
const read = rel => fs.readFileSync(path.resolve(root, rel));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const [baselinePath, outputPath] = process.argv.slice(2);
assert(baselinePath && outputPath && process.argv.length === 4,
  'usage: rekey-low-attack-preview.js BASELINE_POOL OUTPUT_ARTIFACT');
const catalogPath = 'data/team-pool-v2/team-pool.json';
const artifactBytes = read('data/compiled-preview-v1.json');
const artifact = JSON.parse(artifactBytes);
const baselineBytes = read(baselinePath);
const baseline = JSON.parse(baselineBytes);
const catalogBytes = read(catalogPath);
const catalog = JSON.parse(catalogBytes);
const poolJson = read('data/belief-pool-v3/belief-pool.json').toString();
const dexJson = read('data/gen2stadium2.json').toString();
assert.equal(artifact.format, 'nc2000-compiled-preview-v1');
assert.equal(artifact.inputs[catalogPath], sha(baselineBytes), 'baseline is not the cached catalog');
assert.equal(artifact.priorSha256, sha(Buffer.from(poolJson)));
assert.equal(artifact.dexSha256, sha(Buffer.from(dexJson)));
assert.equal(artifact.c, JSON.parse(read('data/search-profiles.json')).blind.c);
for (const [rel, expected] of Object.entries(artifact.inputs)) {
  if (rel !== catalogPath) assert.equal(sha(read(rel)), expected, `unrelated stale input: ${rel}`);
}

// Require precisely the audited optimization, including unchanged team order,
// moves, items, levels and probabilities. No general cache-key relaxation.
const expected = structuredClone(baseline);
const preferFemale = attractGenderCensus(JSON.parse(poolJson)).preferFemale;
for (const team of expected.teams) {
  team.sets = optimizeLowAttack(team.sets, JSON.parse(dexJson), { preferFemale }).sets;
}
expected.setOptimization = catalog.setOptimization;
assert.deepEqual(catalog, expected, 'catalog changes exceed the approved optimization');

const dex = new wasm.Dex();
assert.deepEqual(JSON.parse(wasm.embeddedDexJson()), JSON.parse(dexJson), 'stale WASM dex');
const policies = {};
const consumed = new Set();
function identity(own, foe) {
  const battle = new wasm.Battle(dex, JSON.stringify(own), JSON.stringify(foe), '1,2,3,4');
  let searcher;
  try {
    searcher = new wasm.BlindSearcher(battle, 0, poolJson, 7, artifact.c);
    assert.equal(searcher.compiledPreviewBuild(), artifact.build, 'stale WASM build');
    return { key: searcher.compiledPreviewKey(battle), legal: JSON.parse(battle.legalChoices(0)).map(c => c.input) };
  } finally {
    searcher?.free();
    battle.free();
  }
}
try {
  for (let i = 0; i < catalog.teams.length; i++) {
    for (let j = 0; j < catalog.teams.length; j++) {
      const before = identity(baseline.teams[i].sets, baseline.teams[j].sets);
      const after = identity(catalog.teams[i].sets, catalog.teams[j].sets);
      const choices = artifact.policies[before.key];
      assert(choices?.length, `missing cached policy: ${i}/${j}`);
      assert.deepEqual(after.legal, before.legal, 'optimization changed the selection space');
      assert(choices.every(c => after.legal.includes(c)), `illegal cached choice: ${i}/${j}`);
      if (Object.hasOwn(policies, after.key)) assert.deepEqual(policies[after.key], choices, 'conflicting rekey');
      policies[after.key] = choices;
      consumed.add(before.key);
    }
  }
} finally {
  dex.free();
}
assert.equal(consumed.size, Object.keys(artifact.policies).length, 'unmapped old policies');
assert.equal(Object.keys(policies).length, artifact.matchups, 'incomplete new grid');
const result = {
  ...artifact,
  method: artifact.method + ' Choices reused for the approved low-attack-v2 catalog optimization; not remeasured.',
  catalogReuse: {
    policy: 'low-attack-v2',
    sourceArtifactSha256: sha(artifactBytes),
    sourceCatalogSha256: sha(baselineBytes),
    targetCatalogSha256: sha(catalogBytes),
    reason: 'User approved reusing the existing selection conclusions after Attack and gender optimization.',
    remeasured: false,
  },
  inputs: { ...artifact.inputs, [catalogPath]: sha(catalogBytes) },
  policies,
};
// Read everything before replacing the artifact, and retain the computation's
// original jobs hash, build, budgets and seeds as its provenance.
assert.equal(sha(read(catalogPath)), sha(catalogBytes), 'catalog changed during rekey');
assert.equal(sha(read('data/compiled-preview-v1.json')), sha(artifactBytes), 'artifact changed during rekey');
const output = path.resolve(root, outputPath);
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output + '.tmp', JSON.stringify(result) + '\n', { flag: 'wx' });
fs.renameSync(output + '.tmp', output);
console.log(`${Object.keys(policies).length} policies reused across ${catalog.teams.length ** 2} matchups; no search rerun`);
