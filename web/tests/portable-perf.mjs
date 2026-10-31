// Paired per-decision WASM measurements in Node or Chromium. Chromium runs the
// unmodified generated Node bindings with a byte-loading CommonJS shim. Real
// web bindings, worker and UI are checked separately by the production tests.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { chromium } from '@playwright/test';

const [before, after, output, count = '27000', repetitions = '3', runtime = 'chromium'] = process.argv.slice(2);
assert(before && after && output && ['node', 'chromium'].includes(runtime));
const iterations = Number(count), repeats = Number(repetitions);
assert(Number.isSafeInteger(iterations) && iterations > 0 && repeats > 0);
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const root = path.resolve(import.meta.dirname, '../..');
const files = new Map();
for (const [arm, directory] of [['baseline', before], ['candidate', after]]) {
  for (const name of ['nc2000_wasm.js', 'nc2000_wasm_bg.wasm'])
    files.set(`/${arm}/${name}`, fs.readFileSync(path.join(directory, name)));
}
files.set('/pool', fs.readFileSync(path.join(root, 'data/belief-pool-v3/belief-pool.json')));
files.set('/probe', fs.readFileSync(path.join(root, 'tools/perf-wasm-paired.cjs')));
const server = createServer((req, res) => {
  if (req.url === '/') { res.end('<!doctype html><title>Internal WASM benchmark</title>'); return; }
  const bytes = files.get(req.url);
  if (!bytes) { res.writeHead(404); res.end(); return; }
  res.end(bytes);
});
let browser;
try {
  if (runtime === 'chromium') {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch();
  }
  const version = runtime === 'node' ? process.version : browser.version();
  fs.mkdirSync(output, { recursive: true });
  const manifest = { iterations, repeats, runtime, version, method: 'alternating adjacent decisions; 1000-iteration warm-up',
    sha256: Object.fromEntries([...files].map(([name, bytes]) => [name, sha(bytes)])) };
  fs.writeFileSync(path.join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  const runs = [];
  let reference;
  for (let repeat = 0; repeat < repeats; repeat++) {
    let result;
    if (runtime === 'node') {
      const child = spawnSync(process.execPath, [path.join(root, 'tools/perf-wasm-paired.cjs'),
        path.resolve(before), path.resolve(after), String(iterations), String(repeat)], { encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
      assert.equal(child.status, 0, child.stderr); result = JSON.parse(child.stdout);
    } else {
      const page = await browser.newPage();
      await page.goto(`http://127.0.0.1:${server.address().port}/`);
      result = await page.evaluate(async ({ iterations, repeat }) => {
        const wasm = {};
        for (const arm of ['baseline', 'candidate']) {
          const [js, bytes] = await Promise.all([
            fetch(`/${arm}/nc2000_wasm.js`).then(r => r.text()),
            fetch(`/${arm}/nc2000_wasm_bg.wasm`).then(r => r.arrayBuffer()),
          ]);
          wasm[arm] = {};
          new Function('exports', 'require', '__dirname', js)(wasm[arm],
            name => { if (name !== 'fs') throw Error(name); return { readFileSync: () => new Uint8Array(bytes) }; }, '/wasm');
        }
        const [pool, probe] = await Promise.all([fetch('/pool').then(r => r.text()), fetch('/probe').then(r => r.text())]);
        let result;
        const require = name => {
          if (name === 'node:fs') return { readFileSync: () => pool };
          if (name === 'node:path') return { resolve: (...parts) => parts.join('/') };
          for (const arm of ['baseline', 'candidate']) if (name === `/${arm}/nc2000_wasm.js`) return wasm[arm];
          throw Error(name);
        };
        new Function('require', '__dirname', 'process', 'console', probe)(require, '/tools',
          { argv: ['', '', '/baseline', '/candidate', String(iterations), String(repeat)] },
          { log: text => { result = JSON.parse(text); } });
        return result;
      }, { iterations, repeat });
      await page.close();
    }
    const signature = result.rows.map(({ ms, ...row }) => row);
    reference ??= signature;
    assert.deepEqual(signature, reference, `repeat ${repeat} exploration diverged`);
    fs.writeFileSync(path.join(output, `pair-${repeat}.json`), JSON.stringify(result, null, 2) + '\n');
    runs.push(result);
    console.log(repeat, result.total_ms);
  }
  const median = values => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)];
  const median_ms = Object.fromEntries(['baseline', 'candidate'].map(arm => [arm, median(runs.map(r => r.total_ms[arm]))]));
  const p95_ms = Object.fromEntries(['baseline', 'candidate'].map(arm => [arm,
    median(runs.map(r => r.rows.map(x => x.ms[arm]).sort((a, b) => a - b)[Math.ceil(r.rows.length * .95) - 1]))]));
  const time_reduction = median(runs.map(r => 1 - r.total_ms.candidate / r.total_ms.baseline));
  const result = { median_ms, p95_ms, time_reduction, metric: 'median of adjacent-work paired time reductions',
    pair_reductions: runs.map(r => 1 - r.total_ms.candidate / r.total_ms.baseline),
    exact_match: true, positions: reference.length, runtime: `${runtime} ${version}`,
    accepted: time_reduction >= .05 && p95_ms.candidate / p95_ms.baseline <= 1.05 };
  fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(result);
} finally {
  await browser?.close();
  if (server.listening) await new Promise(resolve => server.close(resolve));
}
