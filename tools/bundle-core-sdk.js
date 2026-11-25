#!/usr/bin/env node
/**
 * Assemble the self-contained NC2000 Core Engine SDK under `dist/nc2000-core-sdk/`
 * and generate `engine-manifest.json` with cryptographic SHA-256 signatures.
 *
 * Usage:
 *   node tools/bundle-core-sdk.js [--out dist/nc2000-core-sdk] [--zip]
 */
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execSync } = require('child_process');

const REPO = path.resolve(__dirname, '..');
const outArgIdx = process.argv.indexOf('--out');
const OUT_DIR = path.resolve(
  REPO,
  outArgIdx >= 0 && process.argv[outArgIdx + 1]
    ? process.argv[outArgIdx + 1]
    : 'dist/nc2000-core-sdk'
);
const DO_ZIP = process.argv.includes('--zip');

const FILES_TO_COPY = [
  'crates/wasm/pkg-node/nc2000_wasm.js',
  'crates/wasm/pkg-node/nc2000_wasm_bg.wasm',
  'crates/wasm/pkg-node/package.json',
  'data/gen2stadium2.json',
  'data/i18n-ja.json',
  'data/search-profiles.json',
  'data/live-search.json',
  'data/team-pool-v3/team-pool.json',
  'data/team-pool-v3/flagship-mixture.json',
  'data/team-pool-v4/team-pool.json',
  'data/team-pool-v4/flagship-mixture.json',
  'data/belief-pool-v3/belief-pool.json',
  'tools/diagnose-position.js',
];

function sha256File(filePath) {
  const buf = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buf).digest('hex');
}

function getGitSha() {
  try {
    return execSync('git rev-parse HEAD', { cwd: REPO, encoding: 'utf8' }).trim();
  } catch {
    return 'unknown';
  }
}

function copyRelFile(relPath) {
  const src = path.join(REPO, relPath);
  if (!fs.existsSync(src)) {
    throw new Error(`Missing required source file: ${src}`);
  }
  const dst = path.join(OUT_DIR, relPath);
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
}

function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const gitSha = getGitSha();
  const fileHashes = {};

  for (const rel of FILES_TO_COPY) {
    if (rel === 'tools/diagnose-position.js' && !fs.existsSync(path.join(REPO, rel))) {
      // Allow diagnose-position.js to be written before or during copy
      continue;
    }
    copyRelFile(rel);
    fileHashes[rel] = sha256File(path.join(REPO, rel));
  }

  // Generate engine-manifest.json
  const manifest = {
    schema_version: 'nc2000-sdk-v1',
    position_schema: 'nc2000-position-v1',
    created_at: new Date().toISOString(),
    engine_git_sha: gitSha,
    wasm_sha256: fileHashes['crates/wasm/pkg-node/nc2000_wasm_bg.wasm'] || null,
    dex_sha256: fileHashes['data/gen2stadium2.json'] || null,
    default_belief_sha256: fileHashes['data/belief-pool-v3/belief-pool.json'] || null,
    file_hashes: fileHashes,
  };

  const manifestJson = JSON.stringify(manifest, null, 2) + '\n';
  fs.writeFileSync(path.join(OUT_DIR, 'engine-manifest.json'), manifestJson);

  let zipPath = null;
  if (DO_ZIP) {
    zipPath = path.join(REPO, 'dist/nc2000-core-sdk.zip');
    try {
      execSync(`zip -r -q "${zipPath}" .`, { cwd: OUT_DIR });
    } catch (err) {
      zipPath = null;
    }
  }

  // Single-line structured telemetry
  const summary = {
    status: 'ok',
    bundle_dir: path.relative(REPO, OUT_DIR),
    git_sha: gitSha,
    files_count: Object.keys(fileHashes).length,
    wasm_sha256: manifest.wasm_sha256,
    zip: zipPath ? path.relative(REPO, zipPath) : null,
  };
  console.log(JSON.stringify(summary));
}

if (require.main === module) {
  try {
    main();
  } catch (err) {
    console.error(JSON.stringify({ status: 'error', error: String(err.message || err) }));
    process.exit(1);
  }
}

module.exports = {
  FILES_TO_COPY,
  sha256File,
  getGitSha,
};
