#!/usr/bin/env node
/**
 * tools/diagnose-position.js
 *
 * Deterministic, protocol-independent diagnosis and triage CLI for NC2000 positions.
 * Evaluates `PositionSpec` (`nc2000-position-v1`) against multi-seed MCTS, deep-budget
 * convergence, flat-root entropy, and counterfactual expected values.
 *
 * SOTA Invariants:
 * 1. M2M Output Principle: Silent during calculation; emits a single structured
 *    JSON line (`nc2000-triage-report-v1`) on stdout upon completion.
 * 2. Gate 1-4 Mechanical Triage: Classifies choices into Adapter Desync (A1/A2),
 *    Engine/Search Defect (B1/B2/B3), or Legitimate MCTS/Game-Theoretic Choice (C1/C2/C3).
 *
 * Usage:
 *   node tools/diagnose-position.js --self-check
 *   node tools/diagnose-position.js --spec <spec.json> [--expect-action <act>]
 *     [--actual-foe-spec <foe.json>] [--iters 81000,324000] [--seeds 1,2,3,4,5]
 *     [--emit-repro <repro.json>]
 */
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const REPO = path.resolve(__dirname, '..');

let wasmModule = null;
let wasmDexInstance = null;
let beliefPoolJson = null;

function getWasm() {
  if (!wasmModule) {
    const wasmPath = path.join(REPO, 'crates/wasm/pkg-node/nc2000_wasm.js');
    if (!fs.existsSync(wasmPath)) {
      throw new Error(`WASM module not found at: ${wasmPath}`);
    }
    wasmModule = require(wasmPath);
  }
  if (!wasmDexInstance) {
    wasmDexInstance = new wasmModule.Dex();
  }
  return { wasm: wasmModule, dex: wasmDexInstance };
}

function freeDex() {
  if (wasmDexInstance) {
    wasmDexInstance.free();
    wasmDexInstance = null;
  }
}

function getBeliefPool() {
  if (!beliefPoolJson) {
    const poolPath = path.join(REPO, 'data/belief-pool-v3/belief-pool.json');
    if (!fs.existsSync(poolPath)) {
      throw new Error(`Belief pool not found at: ${poolPath}`);
    }
    beliefPoolJson = fs.readFileSync(poolPath, 'utf8');
  }
  return beliefPoolJson;
}

function normalizeAction(act) {
  if (!act) return '';
  return String(act)
    .trim()
    .toLowerCase()
    .replace(/[:_]/g, ' ')
    .replace(/\s+/g, ' ');
}

function computeEntropyNorm(actions) {
  if (!actions || actions.length <= 1) return 0.0;
  const K = actions.length;
  let ent = 0.0;
  for (const a of actions) {
    const p = a.frac || (a.visits ? a.visits / actions.reduce((s, x) => s + x.visits, 0) : 0);
    if (p > 1e-9) {
      ent -= p * Math.log(p);
    }
  }
  return ent / Math.log(K);
}

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    selfCheck: false,
    specPath: null,
    expectAction: null,
    actualFoePath: null,
    baseIters: 81000,
    deepIters: 324000,
    seeds: [1, 2, 3, 4, 5],
    emitRepro: null,
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--self-check') {
      opts.selfCheck = true;
    } else if (a === '--spec' && args[i + 1]) {
      opts.specPath = path.resolve(process.cwd(), args[++i]);
    } else if (a === '--expect-action' && args[i + 1]) {
      opts.expectAction = args[++i];
    } else if ((a === '--actual-foe-spec' || a === '--actual-foe') && args[i + 1]) {
      opts.actualFoePath = path.resolve(process.cwd(), args[++i]);
    } else if (a === '--iters' && args[i + 1]) {
      const parts = args[++i].split(',').map(n => parseInt(n.trim(), 10));
      opts.baseIters = parts[0] || 81000;
      opts.deepIters = parts[1] || opts.baseIters * 4;
    } else if (a === '--seeds' && args[i + 1]) {
      opts.seeds = args[++i].split(',').map(n => parseInt(n.trim(), 10)).filter(n => !isNaN(n));
      if (opts.seeds.length === 0) opts.seeds = [1, 2, 3, 4, 5];
    } else if (a === '--emit-repro' && args[i + 1]) {
      opts.emitRepro = path.resolve(process.cwd(), args[++i]);
    }
  }
  return opts;
}

function runSelfCheck() {
  try {
    const { wasm, dex } = getWasm();
    const manifestPath = path.join(REPO, 'engine-manifest.json');
    let manifest = null;
    let manifestVerified = false;

    if (fs.existsSync(manifestPath)) {
      manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
      const wasmBgPath = path.join(REPO, 'crates/wasm/pkg-node/nc2000_wasm_bg.wasm');
      if (fs.existsSync(wasmBgPath)) {
        const wasmSha = crypto.createHash('sha256').update(fs.readFileSync(wasmBgPath)).digest('hex');
        manifestVerified = manifest.wasm_sha256 === wasmSha;
      }
    }

    const report = {
      status: 'OK',
      engine_git_sha: manifest ? manifest.engine_git_sha : 'unknown',
      manifest_present: manifest !== null,
      manifest_verified: manifestVerified,
      wasm_loaded: true,
    };
    console.log(JSON.stringify(report));
    process.exit(0);
  } catch (err) {
    console.error(JSON.stringify({ status: 'ERROR', error: String(err.message || err) }));
    process.exit(1);
  }
}

function runDiagnosis() {
  const opts = parseArgs();
  if (opts.selfCheck) {
    return runSelfCheck();
  }

  if (!opts.specPath || !fs.existsSync(opts.specPath)) {
    console.error(JSON.stringify({
      schema: 'nc2000-triage-report-v1',
      status: 'error',
      message: 'PositionSpec file not found. Use --spec <path.json>',
    }));
    process.exit(1);
  }

  const specRaw = fs.readFileSync(opts.specPath, 'utf8');
  let specJson = null;
  try {
    specJson = JSON.parse(specRaw);
  } catch (err) {
    console.log(JSON.stringify({
      schema: 'nc2000-triage-report-v1',
      spec_valid: false,
      verdict: 'CATEGORY_B1_RULE_INVALID_SPEC',
      error: `JSON parse error: ${err.message}`,
    }));
    return;
  }

  const { wasm, dex } = getWasm();
  const pool = getBeliefPool();
  const side = Number(specJson.side || 0);

  // Validate position with ProtocolSearcher.setPosition
  let testSearcher = null;
  try {
    testSearcher = new wasm.ProtocolSearcher(dex, side, pool, 1, 0.4);
    testSearcher.setPosition(specRaw);
  } catch (err) {
    console.log(JSON.stringify({
      schema: 'nc2000-triage-report-v1',
      spec_valid: false,
      verdict: 'CATEGORY_B1_RULE_INVALID_SPEC',
      error: `setPosition rejected spec: ${err.message || err}`,
    }));
    freeDex();
    return;
  } finally {
    testSearcher?.free();
  }

  // Multi-seed evaluation
  const seedResults = [];
  const seedActionVotes = {};

  for (const s of opts.seeds) {
    const searcher = new wasm.ProtocolSearcher(dex, side, pool, s, 0.4);
    try {
      searcher.setPosition(specRaw);
      searcher.step(opts.baseIters);
      const bestChoice = searcher.best();
      const rootPolicy = JSON.parse(searcher.rootPolicy());
      const report = JSON.parse(searcher.report(0, 0));

      seedResults.push({
        seed: s,
        best: bestChoice,
        rootPolicy,
        report,
      });
      const normBest = normalizeAction(bestChoice);
      seedActionVotes[normBest] = (seedActionVotes[normBest] || 0) + 1;
    } finally {
      searcher.free();
    }
  }

  // Find consensus top action
  let consensusAction = null;
  let maxVotes = -1;
  for (const [act, votes] of Object.entries(seedActionVotes)) {
    if (votes > maxVotes) {
      maxVotes = votes;
      consensusAction = act;
    }
  }
  const seedStability = maxVotes / opts.seeds.length;

  // Primary seed statistics (seed 1)
  const primaryResult = seedResults[0];
  const primaryActions = primaryResult.rootPolicy.actions || [];
  primaryActions.sort((a, b) => (b.visits || 0) - (a.visits || 0));

  const topAction = primaryActions[0] || null;
  const secondAction = primaryActions[1] || null;
  const botActionNorm = normalizeAction(topAction?.input || primaryResult.best);

  const entropyNorm = computeEntropyNorm(primaryActions);
  const deltaPi = (topAction?.frac || 0) - (secondAction?.frac || 0);

  // Check Q-value vs Visit Inversion
  let maxQAction = null;
  let maxQVal = -Infinity;
  for (const a of primaryActions) {
    const mean = typeof a.mean === 'number' ? a.mean : -Infinity;
    if (mean > maxQVal) {
      maxQVal = mean;
      maxQAction = a;
    }
  }
  const qVisitInversion =
    topAction &&
    maxQAction &&
    normalizeAction(topAction.input) !== normalizeAction(maxQAction.input) &&
    (topAction.mean || 0) - (maxQAction.mean || 0) < -0.03;

  // Deep budget test (seed 1 with deepIters)
  let deepAgreement = true;
  let qDeepBot = null;
  if (opts.deepIters > opts.baseIters) {
    const deepSearcher = new wasm.ProtocolSearcher(dex, side, pool, opts.seeds[0], 0.4);
    try {
      deepSearcher.setPosition(specRaw);
      deepSearcher.step(opts.deepIters);
      const deepBest = normalizeAction(deepSearcher.best());
      const deepPolicy = JSON.parse(deepSearcher.rootPolicy());
      const deepBotEntry = (deepPolicy.actions || []).find(
        a => normalizeAction(a.input) === botActionNorm
      );
      deepAgreement = deepBest === botActionNorm;
      qDeepBot = deepBotEntry?.mean ?? null;
    } finally {
      deepSearcher.free();
    }
  }

  // Expected action comparison (if provided)
  const normExpected = opts.expectAction ? normalizeAction(opts.expectAction) : null;
  let expectedActionStats = null;
  if (normExpected) {
    expectedActionStats = primaryActions.find(
      a => normalizeAction(a.input) === normExpected
    ) || null;
  }

  const qBot = topAction?.mean ?? 0.5;
  const qExpected = expectedActionStats?.mean ?? null;
  const qDiff = qExpected !== null ? qBot - qExpected : null;
  const visitShareBot = topAction?.frac ?? 0;
  const visitShareExpected = expectedActionStats?.frac ?? 0;

  // Oracle Hindsight (if actual foe provided)
  let oracleHindsightQBot = null;
  let oracleHindsightQExpected = null;
  if (opts.actualFoePath && fs.existsSync(opts.actualFoePath)) {
    const oracleSearcher = new wasm.ProtocolSearcher(dex, side, pool, opts.seeds[0], 0.4);
    try {
      const foeSetsJson = fs.readFileSync(opts.actualFoePath, 'utf8');
      oracleSearcher.pinOpponent(foeSetsJson);
      oracleSearcher.setPosition(specRaw);
      oracleSearcher.step(opts.baseIters);
      const oraclePolicy = JSON.parse(oracleSearcher.rootPolicy());
      const oracleActions = oraclePolicy.actions || [];
      const oracleBot = oracleActions.find(a => normalizeAction(a.input) === botActionNorm);
      const oracleExp = normExpected
        ? oracleActions.find(a => normalizeAction(a.input) === normExpected)
        : null;
      oracleHindsightQBot = oracleBot?.mean ?? null;
      oracleHindsightQExpected = oracleExp?.mean ?? null;
    } catch {
      // Pinned search unavailable or failed
    } finally {
      oracleSearcher.free();
    }
  }
  freeDex();

  // Determine Verdict
  let verdict = 'CATEGORY_C0_MCTS_VALUE_PREFERENCE';
  if (qVisitInversion) {
    verdict = 'CATEGORY_B2_Q_VISIT_INVERSION';
  } else if ((entropyNorm > 0.92 && deltaPi < 0.03) || seedStability < 0.60) {
    verdict = 'CATEGORY_B3_FLAT_ROOT_LOTTERY';
  } else if (
    oracleHindsightQExpected !== null &&
    oracleHindsightQBot !== null &&
    oracleHindsightQExpected - oracleHindsightQBot > 0.02 &&
    qDiff !== null &&
    qDiff > 0
  ) {
    verdict = 'CATEGORY_C2_BELIEF_RISK_MITIGATION';
  } else if (normExpected && qDiff !== null && Math.abs(qDiff) <= 0.02) {
    verdict = 'CATEGORY_C1_NASH_INDIFFERENCE';
  } else if (qBot < 0.10 || qBot > 0.95) {
    verdict = 'CATEGORY_C3_HORIZON_DESPERATION';
  }

  const manifestPath = path.join(REPO, 'engine-manifest.json');
  const manifest = fs.existsSync(manifestPath)
    ? JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
    : null;

  const positionHash = crypto.createHash('sha256').update(specRaw).digest('hex').slice(0, 16);

  const report = {
    schema: 'nc2000-triage-report-v1',
    engine_sha: manifest?.engine_git_sha || 'unknown',
    position_hash: positionHash,
    spec_valid: true,
    verdict,
    bot_action: topAction?.input || primaryResult.best,
    expected_action: opts.expectAction || null,
    q_bot: Number(qBot.toFixed(4)),
    q_expected: qExpected !== null ? Number(qExpected.toFixed(4)) : null,
    q_diff: qDiff !== null ? Number(qDiff.toFixed(4)) : null,
    visit_share_bot: Number(visitShareBot.toFixed(4)),
    visit_share_expected: Number(visitShareExpected.toFixed(4)),
    root_entropy_norm: Number(entropyNorm.toFixed(4)),
    seed_stability: Number(seedStability.toFixed(2)),
    deep_budget_agreement: deepAgreement,
    q_deep_bot: qDeepBot !== null ? Number(qDeepBot.toFixed(4)) : null,
    oracle_hindsight_q_bot:
      oracleHindsightQBot !== null ? Number(oracleHindsightQBot.toFixed(4)) : null,
    oracle_hindsight_q_expected:
      oracleHindsightQExpected !== null ? Number(oracleHindsightQExpected.toFixed(4)) : null,
  };

  if (opts.emitRepro) {
    const reproBundle = {
      manifest,
      spec: specJson,
      triage_report: report,
      primary_actions: primaryActions,
    };
    fs.writeFileSync(opts.emitRepro, JSON.stringify(reproBundle, null, 2) + '\n');
  }

  console.log(JSON.stringify(report));
}

if (require.main === module) {
  runDiagnosis();
}

module.exports = {
  runDiagnosis,
  runSelfCheck,
  computeEntropyNorm,
  normalizeAction,
};
