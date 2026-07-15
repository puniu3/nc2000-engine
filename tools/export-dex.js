// Export the flattened gen2stadium2 dex (functions stripped, callback names
// recorded) to data/gen2stadium2.json for the Rust engine.
'use strict';
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { PS_ROOT, sim, MOD, FORMAT, fnNames, stripFns, legal } = require('./ps');

const dex = sim.Dex.mod(MOD);

function exportTable(list) {
	const out = {};
	for (const e of list) {
		const data = stripFns(e);
		delete data.desc;
		delete data.shortDesc;
		data.callbacks = fnNames(e);
		// Key by name (e.g. hiddenpowerfire); typed Hidden Power variants share id 'hiddenpower'.
		out[sim.toID(e.name)] = data;
	}
	return out;
}

const species = {};
for (const s of dex.species.all().filter(legal)) {
	species[s.id] = {
		num: s.num, name: s.name, types: s.types, baseStats: s.baseStats,
		genderRatio: s.genderRatio, gender: s.gender || null, weightkg: s.weightkg,
		tier: s.tier,
	};
}

const conditions = {};
for (const id of Object.keys(dex.data.Conditions)) {
	const raw = dex.data.Conditions[id];
	const data = stripFns(raw);
	data.callbacks = fnNames(raw);
	conditions[id] = data;
}

const typechart = {};
for (const t of dex.types.all()) {
	if (!t.exists || t.gen > 2 || t.isNonstandard) continue;
	typechart[t.id] = { name: t.name, damageTaken: t.damageTaken };
}

let psCommit = 'unknown';
try { psCommit = execSync('git rev-parse HEAD', { cwd: PS_ROOT }).toString().trim(); } catch {}

const out = {
	meta: { format: FORMAT, mod: MOD, psCommit, generator: 'tools/export-dex.js' },
	typechart,
	species,
	moves: exportTable(dex.moves.all().filter(legal)),
	items: exportTable(dex.items.all().filter(legal)),
	conditions,
};

const dest = path.join(__dirname, '..', 'data', 'gen2stadium2.json');
fs.writeFileSync(dest, JSON.stringify(out, null, 1));
const c = k => Object.keys(out[k]).length;
console.log(`wrote ${dest}: ${c('species')} species, ${c('moves')} moves, ${c('items')} items, ${c('conditions')} conditions, ${c('typechart')} types`);
