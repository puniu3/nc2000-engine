// Shared helpers for the Node-side wasm tests.
// Run any test from the repo root or anywhere: paths resolve from __dirname.
"use strict";

const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "..", "..", "..");
const wasm = require(path.join(__dirname, "..", "pkg-node", "nc2000_wasm.js"));

function loadFixture(rel) {
  const p = path.join(REPO, "fixtures", "corpus-v1", rel);
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function readData(rel) {
  return fs.readFileSync(path.join(REPO, "data", rel), "utf8");
}

let failures = 0;
function check(cond, msg) {
  if (!cond) {
    failures += 1;
    console.error(`FAIL: ${msg}`);
  }
}

function checkEq(a, b, msg) {
  const ja = JSON.stringify(a);
  const jb = JSON.stringify(b);
  if (ja !== jb) {
    failures += 1;
    console.error(`FAIL: ${msg}\n  got:      ${ja}\n  expected: ${jb}`);
  }
}

function finish(name) {
  if (failures > 0) {
    console.error(`${name}: ${failures} failure(s)`);
    process.exit(1);
  }
  console.log(`${name}: OK`);
}

module.exports = { wasm, REPO, loadFixture, readData, check, checkEq, finish };
