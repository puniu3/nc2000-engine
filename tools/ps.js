// Shared access to the reference Pokemon Showdown build.
'use strict';
const path = require('path');
const os = require('os');
const PS_ROOT = process.env.PS_ROOT || path.join(os.homedir(), 'pokemon-showdown');
const sim = require(path.join(PS_ROOT, 'dist/sim'));
const prng = require(path.join(PS_ROOT, 'dist/sim/prng'));
const rpai = require(path.join(PS_ROOT, 'dist/sim/tools/random-player-ai'));

const FORMAT = 'gen2nc2000';
const MOD = 'gen2stadium2';

// Own enumerable function-valued property names, recursing one level into
// plain sub-objects (condition/secondary/self blocks).
function fnNames(obj, prefix = '', depth = 0) {
	const names = [];
	for (const k in obj) {
		const v = obj[k];
		if (typeof v === 'function') names.push(prefix + k);
		else if (v && typeof v === 'object' && !Array.isArray(v) && depth < 2) {
			names.push(...fnNames(v, `${prefix}${k}.`, depth + 1));
		}
	}
	return names;
}

// Deep-copy dropping functions; arrays preserved.
function stripFns(v, depth = 0) {
	if (typeof v === 'function') return undefined;
	if (Array.isArray(v)) return v.map(x => stripFns(x, depth + 1)).filter(x => x !== undefined);
	if (v && typeof v === 'object') {
		if (depth > 6) return undefined;
		const out = {};
		for (const k in v) {
			const s = stripFns(v[k], depth + 1);
			if (s !== undefined) out[k] = s;
		}
		return out;
	}
	return v;
}

const legal = x => x.exists && x.gen <= 2 && !x.isNonstandard;

module.exports = { PS_ROOT, sim, prng, rpai, FORMAT, MOD, fnNames, stripFns, legal };
