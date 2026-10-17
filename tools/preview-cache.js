'use strict';

// Server-side, storage-independent cache of completed blind preview choices.
// Persistence, locking and request scheduling belong to the caller.
const {createHash} = require('node:crypto');
const FORMAT = 'nc2000-preview-cache-v1';
const hash = value => createHash('sha256').update(value).digest('hex');
const canonical = value => {
	if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
	if (value && typeof value === 'object') {
		return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
	}
	return JSON.stringify(value);
};

/** wasmSha256 must hash the actual loaded .wasm; dexJson is the exact data
 * used to construct Dex (embeddedDexJson() when using new Dex()). */
function createPreviewCache({wasmSha256, dexJson, iterations = 729000, formatId = 'gen2nintendocup2000noohkostadium2strict'}) {
	if (!/^[a-f0-9]{64}$/.test(wasmSha256 || '')) throw new TypeError('wasmSha256 must be a SHA-256 hex digest');
	if (!Number.isInteger(iterations) || iterations <= 0 || iterations > 0xffffffff) throw new TypeError('iterations must be a positive u32');
	if (typeof formatId !== 'string' || !formatId) throw new TypeError('formatId is required');
	const scope = {format: FORMAT, wasmSha256, dexSha256: hash(canonical(JSON.parse(dexJson))), iterations, formatId};

	function state(searcher) {
		const text = searcher.previewCacheInfo();
		if (!text) return null;
		const info = JSON.parse(text);
		if (info.format !== 'nc2000-preview-cache-info-v1') return null;
		const policy = JSON.parse(searcher.rootPolicy());
		if (policy.baked || policy.forced || !Array.isArray(policy.actions) || !policy.actions.length) return null;
		const legal = policy.actions.map(a => a.input).sort();
		if (!legal.every(a => typeof a === 'string' && /^team \d+, \d+, \d+$/.test(a))) return null;
		return {key: hash(canonical({scope, info, legal})), legal, policy};
	}

	return Object.freeze({
		iterations,
		/** Call after observe/onRequest, before searching. null means unsupported. */
		key(searcher) { return state(searcher)?.key ?? null; },
		/** Serialize only a completed live search at exactly the configured budget. */
		createRecord(searcher) {
			const current = state(searcher);
			if (!current) throw new Error('A live blind team preview is required');
			if (searcher.iterations() !== iterations || current.policy.iterations !== iterations) {
				throw new Error(`Preview must complete exactly ${iterations} iterations before caching`);
			}
			const choice = searcher.best();
			if (!current.legal.includes(choice)) throw new Error('Selected preview is not legal');
			const record = {format: FORMAT, key: current.key, iterations, choice};
			return JSON.stringify({...record, checksum: hash(canonical(record))});
		},
		/** A hit does not mutate the searcher or pretend it ran any iterations.
		 * Submit hit.choice instead of searcher.best(). Malformed/stale entries miss. */
		readRecord(searcher, text) {
			if (searcher.iterations() !== 0 || typeof text !== 'string') return null;
			const current = state(searcher);
			if (!current) return null;
			let record;
			try { record = JSON.parse(text); } catch { return null; }
			if (!record || record.format !== FORMAT || record.key !== current.key || record.iterations !== iterations || !current.legal.includes(record.choice)) return null;
			const {format, key, choice} = record;
			if (record.checksum !== hash(canonical({format, key, iterations, choice}))) return null;
			return {choice, source: 'cache', iterations: 0, sourceIterations: iterations};
		},
	});
}

module.exports = {createPreviewCache};
