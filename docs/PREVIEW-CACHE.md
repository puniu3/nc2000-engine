`tools/preview-cache.js` supplies a Node.js interface for caching completed
blind team-preview decisions. It performs no I/O and is not enabled in the
GitHub Pages app or the websocket client. A fork owns persistence, sharing
between users, concurrent-request deduplication, eviction, worker scheduling,
and submitting the result to the still-current battle request.

Build the bindings with `crates/wasm/build.sh nodejs`. Both `BlindSearcher`
and `ProtocolSearcher` expose `previewCacheInfo()` after `observe(battle)` or
`onRequest(requestJson)`, respectively. Call these normal initialization
methods even on a cache hit: they advance the per-game observer and RNG.
The returned identity is opaque; use the helper instead of parsing it.

```js
const fs = require('node:fs');
const {createHash} = require('node:crypto');
const wasm = require('./crates/wasm/pkg-node/nc2000_wasm.js');
const {createPreviewCache} = require('./tools/preview-cache');

const cache = createPreviewCache({
  // Hash the actual binary loaded by this server process.
  wasmSha256: createHash('sha256').update(fs.readFileSync(
    './crates/wasm/pkg-node/nc2000_wasm_bg.wasm'
  )).digest('hex'),
  dexJson: wasm.embeddedDexJson(), // for new Dex(); pass actual JSON for Dex.fromJson
  iterations: 720000,
  formatId: 'gen2nintendocup2000noohkostadium2strict',
});

// After constructing the searcher, setting its own team/prior and feeding
// all player-visible lines, require onRequest(requestJson) === true.
// A BlindSearcher instead requires observe(battle).
async function choosePreview(searcher, store) {
  const key = cache.key(searcher);
  if (key === null) return null; // caller's ordinary non-cache path
  const hit = cache.readRecord(searcher, await store.get(key));
  if (hit) return hit.choice;

  // The fork should coalesce concurrent misses for this key, and put CPU
  // work in its own worker. This loop only illustrates the exact budget.
  while (searcher.iterations() < cache.iterations) {
    const before = searcher.iterations();
    const after = searcher.step(Math.min(1000, cache.iterations - before));
    if (after <= before) throw new Error('Preview search did not advance');
    await new Promise(resolve => setImmediate(resolve));
  }
  const recordJson = cache.createRecord(searcher);
  await store.put(key, recordJson); // atomic persistence is the fork's responsibility
  return searcher.best();
}
```

The three operations are:

| Operation | Result |
| --- | --- |
| `cache.key(searcher)` | SHA-256 storage key, or `null` outside a supported live blind preview |
| `cache.createRecord(searcher)` | JSON string; throws unless the live search completed **exactly** the configured iterations and selected a legal team |
| `cache.readRecord(searcher, json)` | `{choice, source: 'cache', iterations: 0, sourceIterations}` on a hit; otherwise `null` |

An entry uses the versioned `nc2000-preview-cache-v1` format with `key`,
`iterations`, `choice`, and an integrity checksum. Treat the JSON as opaque
server-generated data; the checksum detects corruption and is not an
authentication signature. Loading an entry does not change `best()`, root
statistics, or the searcher: submit **`hit.choice`**. Lookups after any search
iterations have already run are rejected. Truncated JSON, different budgets,
stale binaries/data/settings, invalid choices and unsupported versions miss.
Incomplete, overshot, baked and forced decisions cannot be exported.

Identity includes the actual WASM binary hash, dex contents, format, complete
search configuration, opponent pool, applied belief prior, own canonical
sets, ordered public opponent facts (including gender and item presence),
and the legal action set. Hidden opponent moves and exact held items are
excluded. Seeds, user names and battle IDs are excluded: the first completed
selection is reused across users and seeds. This intentionally reuses one
sampled decision rather than preserving fresh-search seed diversity. Different
search drivers do not share entries. Binary updates conservatively invalidate
entries even if the policy happens to be unchanged.

The current interface supports blind selection only. Pinned/open sheets,
`fromFork`, hand-entered positions, endgame sessions and in-battle decisions
return no usable key. It caches final choices, not search trees/checkpoints;
it cannot resume a partial search. The 240k default matches `data/live-search.json`
(`SteppedCfPreview` live preview search) and is independent of the adaptive battle cap.
The same interface can generate a record offline or on the first live miss;
storage and distribution remain entirely with the fork.

Verification: `node crates/wasm/tests-node/preview-cache.js` exercises real
WASM generation/loading, both drivers, input/settings invalidation, hidden
information exclusion, and unchanged subsequent RNG/belief behavior after a
hit. No server or browser persistence is installed by this interface.
