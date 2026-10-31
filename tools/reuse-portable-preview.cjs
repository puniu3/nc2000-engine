"use strict";

// One audited, exact-behavior migration. This is not a general stale-cache bypass.
// node tools/reuse-portable-preview.cjs BASELINE_WASM_DIR CANDIDATE_WASM_DIR
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");
const root = path.resolve(__dirname, "..");
const read = p => fs.readFileSync(path.resolve(root, p));
const sha = b => createHash("sha256").update(b).digest("hex");
const auditPath = "findings/portable-search-v1/preview-reuse-audit.json";
const audit = JSON.parse(read(auditPath));
const originalBytes = read("findings/portable-search-v1/source-preview-artifact.json");
const original = JSON.parse(originalBytes);
assert.equal(sha(originalBytes), audit.sourceArtifactSha256);
assert.equal(original.build, audit.sourceBuild);
assert.deepEqual(original.inputs, audit.sourceInputs);
for (const [file, digest] of Object.entries(audit.targetInputs)) {
  assert.equal(sha(read(file)), digest, `unaudited source change: ${file}`);
}
for (const [file, digest] of Object.entries(audit.evidence)) {
  assert.equal(sha(read(file)), digest, `changed parity evidence: ${file}`);
}
assert.equal(audit.hashAudit.baselineSha256, audit.hashAudit.candidateSha256);
assert.equal(audit.hashAudit.states, 4178);
for (const file of audit.searchComparisons) {
  const result = JSON.parse(read(file));
  assert.equal(result.exact_match, true, `search parity failed: ${file}`);
  assert(result.time_reduction > 0, `search regression: ${file}`);
}
const modules = process.argv.slice(2).map(p => path.resolve(p));
assert.equal(modules.length, 2, "provide baseline and candidate Node WASM directories");
for (let i = 0; i < 2; i++) {
  assert.equal(sha(fs.readFileSync(path.join(modules[i], "nc2000_wasm_bg.wasm"))),
    audit.wasmSha256[i], "WASM differs from the measured build");
}
const catalog = JSON.parse(read("data/team-pool-v2/team-pool.json"));
const pool = read("data/belief-pool-v3/belief-pool.json").toString();
const expectedDex = JSON.parse(read("data/gen2stadium2.json"));
const identities = modules.map((module, arm) => {
  const wasm = require(path.join(module, "nc2000_wasm.js"));
  assert.deepEqual(JSON.parse(wasm.embeddedDexJson()), expectedDex);
  const dex = new wasm.Dex();
  const rows = [];
  try {
    for (const own of catalog.teams) for (const foe of catalog.teams) {
      const battle = new wasm.Battle(dex, JSON.stringify(own.sets), JSON.stringify(foe.sets), "1,2,3,4");
      let search;
      try {
        search = new wasm.BlindSearcher(battle, 0, pool, 7, original.c);
        assert.equal(search.compiledPreviewBuild(), arm ? audit.targetBuild : original.build);
        const key = search.compiledPreviewKey(battle);
        const legal = JSON.parse(battle.legalChoices(0)).map(c => c.input);
        assert(original.policies[key]?.every(choice => legal.includes(choice)), "missing/illegal policy");
        rows.push({ key, legal });
      } finally { search?.free(); battle.free(); }
    }
  } finally { dex.free(); }
  return rows;
});
assert.deepEqual(identities[0], identities[1], "preview keys or legal choices changed");
assert.equal(new Set(identities[0].map(r => r.key)).size, original.matchups);
const result = {
  ...original,
  build: audit.targetBuild,
  inputs: audit.targetInputs,
  method: original.method + " Choices reused after an audited exact-behavior portable optimization; not remeasured.",
  performanceReuse: {
    policy: "portable-search-v1",
    sourceArtifactSha256: audit.sourceArtifactSha256,
    sourceBuild: original.build,
    targetBuild: audit.targetBuild,
    auditPath,
    auditSha256: sha(read(auditPath)),
    remeasured: false,
  },
};
const output = path.join(root, "data/compiled-preview-v1.json");
const current = fs.readFileSync(output);
const serialized = JSON.stringify(result) + "\n";
assert(sha(current) === sha(originalBytes) || current.toString() === serialized,
  "refusing to replace an unrelated preview artifact");
fs.writeFileSync(output + ".tmp", serialized, { flag: "wx" });
fs.renameSync(output + ".tmp", output);
console.log(`${original.matchups} policies unchanged; ${identities[0].length} catalog pairs verified in both WASM builds`);
