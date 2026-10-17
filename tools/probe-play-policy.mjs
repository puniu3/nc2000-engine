// Inspect (not strength-test) all execution providers on the previously
// reconstructed counterexample, using the newly built local WASM.
// node tools/probe-play-policy.mjs > findings/mixed-policy-architecture-v1/probe.json
import fs from "node:fs";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { initSync, Dex, Battle, BlindSearcher } from "../crates/wasm/pkg-web/nc2000_wasm.js";

const read = path => fs.readFileSync(new URL("../" + path, import.meta.url));
const bytes = read("crates/wasm/pkg-web/nc2000_wasm_bg.wasm");
initSync({ module: bytes });
const fixture = JSON.parse(read("findings/policy-class-v1/fixtures.json")).find(c => c.id === "wait-7");
const dex = new Dex();
const battle = new Battle(dex, ...fixture.teams.map(JSON.stringify), fixture.seed);
const search = new BlindSearcher(battle, 0, read("data/belief-pool-v3/belief-pool.json").toString(), 813001, 0.4);
try {
  for (const d of fixture.decisions.slice(0, -1)) {
    const choices = d.choices.map((i, side) => i === null ? null : JSON.parse(battle.legalChoices(side))[i].input);
    if (choices[0] !== null) search.observe(battle);
    choices.forEach((input, side) => { if (input !== null) battle.applyChoice(side, input); });
  }
  search.observe(battle);
  search.step(81000);
  const policies = Object.fromEntries(["legacy", "visit-sample", "estimated-mix"]
    .map(mode => [mode, JSON.parse(search.playPolicy(mode))]));
  const legal = JSON.parse(battle.legalChoices(0)).map(a => a.input);
  for (const policy of Object.values(policies)) {
    assert(Math.abs(policy.actions.reduce((s, r) => s + r.probability, 0) - 1) < 1e-12);
    assert(policy.actions.every(r => r.probability > 0 && legal.includes(r.input)));
  }
  const decision = JSON.parse(search.commitDecision("visit-sample"));
  assert.deepEqual(decision.policy, policies["visit-sample"]);
  assert.deepEqual(JSON.parse(search.commitDecision("legacy")), decision);
  console.log(JSON.stringify({ fixture: fixture.id, seed: 813001, iterations: 81000,
    runtime: "locally-rebuilt-wasm", wasmSha256: createHash("sha256").update(bytes).digest("hex"),
    best: search.best(), policies, decision, diagnostics: JSON.parse(search.rootPolicy()),
    claim: "API and sampling demonstration only; no new win-rate or equilibrium certificate" }, null, 2));
} finally { search.free(); battle.free(); dex.free(); }
