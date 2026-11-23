// Rebuild training-v3 products without relabelling historical measurements.
'use strict';
const fs = require('fs');
const cp = require('child_process');
const path = require('path');
const root = path.join(__dirname, '..');
process.chdir(root);
const { normalizeTraining } = require('./normalize-training');
const read = p => JSON.parse(fs.readFileSync(p, 'utf8'));
const write = (p, d) => fs.writeFileSync(p, JSON.stringify(d, null,
	p.includes('conclusion-equilibrium-r1') ? 2 : 1) + '\n');
const dex = read('data/gen2stadium2.json');
fs.mkdirSync('data/training-v3', { recursive: true });
for (const [source, output] of [
	['crates/bot/src/belief_supplement_v1.json', 'data/training-v3/supplemental.json'],
	['data/community-rentals-v0/teams.json', 'data/training-v3/rentals.json'],
]) {
	const data = read(source);
	const full = data.teams.map(t => ({ ...t, id: t.id || `rental-cban-${t.cban}`,
		sets: t.sets.map(s => ({ ...s, name: s.name || s.species, ability: s.ability || 'No Ability',
			nature: s.nature || 'Serious', happiness: s.happiness ?? 255,
			gender: s.gender || (() => {
				const sp=dex.species[s.species.toLowerCase().replace(/[^a-z0-9]/g,'')];
				return sp.gender || (Math.floor((s.ivs?.atk ?? 31)/2)<(sp.genderRatio?.F ?? .5)*16?'F':'M');
			})(),
			ivs: s.ivs || Object.fromEntries(['hp','atk','def','spa','spd','spe'].map(k=>[k,31])),
			evs: s.evs || Object.fromEntries(['hp','atk','def','spa','spd','spe'].map(k=>[k,255])),
		})),
	}));
	write(output, { source, trainingPolicy: 'training-v3',
		teams: full.map(t => ({ ...t, sets: normalizeTraining(t.sets, dex) })) });
}
cp.execFileSync(process.execPath, ['tools/build-team-products-v3.js'], { stdio: 'inherit' });
cp.execFileSync('python3', ['tools/belief-pool-v3.py', 'build'], { stdio: 'inherit' });
cp.execFileSync(process.execPath, ['tools/build-team-products-v2.js'], { stdio: 'inherit' });
const catalog = read('data/team-pool-v2/team-pool.json');
const byId = new Map(catalog.teams.map(t => [t.id, t]));
for (const p of ['data/kodoku-r3/conclusion-equilibrium-deme.json', 'data/kodoku-r3/conclusion-equilibrium-r1.json']) {
	const data = read(p);
	data.trainingPolicy = 'training-v3';
	data.measurementStatus = 'Historical mixture weights/payoffs; normalized training and opponent prior have not been remeasured.';
	for (const t of data.teams) t.sets = byId.get(t.id)?.sets || normalizeTraining(t.sets, dex);
	write(p, data);
}
