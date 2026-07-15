// Dump merged (mod-resolved) callback sources for moves and items into
// reference/merged-moves.txt and reference/merged-items.txt, in the same
// style as reference/merged-conditions.txt. Run: node tools/dump-callbacks.js
'use strict';
const fs = require('fs');
const path = require('path');
const { sim, MOD } = require('./ps');

const dex = sim.Dex.mod(MOD);
const SEP = '='.repeat(70);

function dumpEntry(out, obj, prefix = '') {
	const keys = Object.keys(obj);
	for (const k of keys) {
		const v = obj[k];
		if (typeof v === 'function') {
			out.push(`--- ${prefix}${k}:`);
			out.push(String(v));
		} else if (/Priority$|Order$|^order$/.test(k) && typeof v === 'number') {
			out.push(`--- ${prefix}${k} = ${v}`);
		} else if (v && typeof v === 'object' && !Array.isArray(v) &&
			['condition', 'secondary', 'self', 'secondaries'].includes(k)) {
			out.push(`--- ${prefix}${k}: ${JSON.stringify(v, (kk, vv) => typeof vv === 'function' ? `<fn>` : vv)}`);
			dumpEntry(out, v, `${prefix}${k}.`);
		}
	}
}

function hasFns(obj, depth = 0) {
	for (const k in obj) {
		const v = obj[k];
		if (typeof v === 'function') return true;
		if (v && typeof v === 'object' && !Array.isArray(v) && depth < 2 && hasFns(v, depth + 1)) return true;
	}
	return false;
}

function legal(e) {
	return !e.isNonstandard && !e.isZ && (e.exists !== false);
}

for (const [table, file] of [[dex.moves.all(), 'merged-moves.txt'], [dex.items.all(), 'merged-items.txt']]) {
	const out = [];
	for (const e of table.filter(legal)) {
		if (!hasFns(e)) continue;
		out.push(SEP);
		out.push(`## ${e.id}  (name: ${e.name})`);
		// non-function fields that matter for semantics
		const meta = {};
		for (const k of ['basePower', 'accuracy', 'pp', 'type', 'category', 'priority', 'target',
			'flags', 'secondary', 'self', 'volatileStatus', 'status', 'weather', 'sideCondition',
			'critRatio', 'willCrit', 'multihit', 'drain', 'recoil', 'heal', 'boosts', 'ignoreImmunity',
			'selfSwitch', 'forceSwitch', 'selfdestruct', 'sleepUsable', 'noDamageVariance', 'damage',
			'ohko', 'fling', 'onHit', 'naturalGift', 'isBerry', 'ignoreKlutz']) {
			if (e[k] !== undefined && typeof e[k] !== 'function') meta[k] = e[k];
		}
		out.push(`meta: ${JSON.stringify(meta, (kk, vv) => typeof vv === 'function' ? '<fn>' : vv)}`);
		dumpEntry(out, e);
		if (e.condition) {
			out.push(`--- condition (${JSON.stringify(Object.keys(e.condition))}):`);
			dumpEntry(out, e.condition, 'condition.');
		}
	}
	const dest = path.join(__dirname, '..', 'reference', file);
	fs.writeFileSync(dest, out.join('\n') + '\n');
	console.log(`wrote ${dest} (${out.filter(l => l === SEP).length} entries)`);
}
