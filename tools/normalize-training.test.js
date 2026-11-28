'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const crypto = require('crypto');
const dex = require('../data/gen2stadium2.json');
const { normalizeTraining } = require('./normalize-training');
const set = (species, moves, gender = 'N') => ({ species, moves, gender, level: 50,
	ivs: {hp:31,atk:31,def:31,spa:31,spd:31,spe:31},
	evs: {hp:192,atk:128,def:64,spa:0,spd:0,spe:192} });
const norm = s => normalizeTraining([s], dex)[0];
test('repairs physical low Attack and all non-Attack training together', () => {
	const s=set('Starmie',['Rapid Spin']);s.ivs.atk=2;
	const r=norm(s);
	assert.equal(r.ivs.atk,31);assert(Object.values(r.evs).every(v=>v===255));
	assert(Object.values(r.ivs).every(v=>Math.floor(v/2)===15));
});
test('special Hidden Power retains maximum type-constrained power while minimizing Attack', () => {
	const s=set('Zapdos',['Thunderbolt','Hidden Power Ice']);s.ivs.def=26;
	const r=norm(s);
	assert.equal(r.ivs.atk,22);assert.equal(r.evs.atk,0);assert.equal(r.ivs.def,26);
	assert.equal(r.ivs.hp,31);assert.equal(r.ivs.spe,31);assert.equal(r.ivs.spa,31);
});
test('physical Hidden Power maximizes Attack within its type', () => {
	const s=set('Marowak',['Hidden Power Bug'],'M');s.ivs.atk=2;s.ivs.def=2;s.gender='F';
	const r=norm(s); // female 50/50 cap: largest DV congruent to 1 is 5
	assert.equal(r.ivs.atk,10);assert.equal(r.ivs.def,26);assert.equal(r.gender,'F');
	assert.equal(r.evs.atk,255);
});
test('maximum HP takes precedence over an unnecessarily even Attack DV', () => {
	const s=set('Starmie',['Surf']);s.ivs.atk=0;s.ivs.hp=14;
	const r=norm(s);assert.equal(r.ivs.hp,31);assert.equal(r.ivs.atk,2);assert.equal(r.evs.atk,0);
});
test('male camouflage and female physical caps constrain Attack', () => {
	const c=norm(set('Cloyster',['Ice Beam','Spikes'],'M'));
	assert.equal(c.ivs.atk,18);assert.equal(c.gender,'M');assert.equal(c.evs.atk,0);
	const s=set('Snorlax',['Double-Edge'],'F');s.ivs.atk=2;
	assert.equal(norm(s).ivs.atk,2);assert.equal(norm(s).evs.atk,255);
});
test('fixed damage and Transform do not require base Attack; called physical moves do', () => {
	for(const [species,moves,gender] of [['Blissey',['Counter'],'F'],['Ditto',['Transform'],'N']]) {
		const r=norm(set(species,moves,gender));assert.equal(r.evs.atk,0);assert.equal(r.ivs.atk,2);
	}
	const r=norm(set('Clefable',['Metronome'],'M'));assert.equal(r.evs.atk,255);assert.equal(r.ivs.atk,31);
});
test('shiny state is preserved while training is maximized', () => {
	const s=set('Gengar',['Psychic'],'M');Object.assign(s.ivs,{hp:16,atk:22,def:20,spa:20,spd:20,spe:20});s.shiny=true;
	const r=norm(s);assert.deepEqual(r.ivs,s.ivs);assert.equal(r.evs.atk,0);
	assert.equal(r.evs.def,255);assert.equal(r.shiny,true);
});
test('normalization is pure, idempotent and rejects incomplete inputs', () => {
	const s=set('Starmie',['Surf']), original=structuredClone(s), r=norm(s);
	assert.deepEqual(s,original);assert.deepEqual(norm(r),r);assert.deepEqual(r.moves,s.moves);
	delete s.evs.hp;assert.throws(()=>norm(s),/complete validated/);
});
test('Return and Frustration have maximum power rather than default happiness', () => {
	const s=set('Porygon2',['Frustration']);s.happiness=255;
	assert.equal(norm(s).happiness,0);
	s.moves=['Return'];s.happiness=0;assert.equal(norm(s).happiness,255);
	s.moves=['Return','Frustration'];assert.throws(()=>norm(s),/conflicting happiness/);
});
test('native exhaustive adversarial certificate covers current products', () => {
	const report=require('../data/training-v3/verification.json');
	assert.deepEqual(report.issues,[]);assert.equal(report.catalog_teams_with_exact_prior_support,15);
	assert.equal(report.dv_tuples_per_class,65536);
	let sets=0;
	for(const [p,hash] of Object.entries(report.inputs)) {
		const bytes=fs.readFileSync(p);
		assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),hash,`stale native verification: ${p}`);
		sets+=JSON.parse(bytes).teams.reduce((n,t)=>n+t.sets.length,0);
	}
	for(const [p,hash] of Object.entries(report.code_and_rules)) {
		assert.equal(crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),hash,`stale verification rules: ${p}`);
	}
	assert.equal(report.checked_sets,sets-report.excluded_sets);
	assert.equal(report.excluded_fallback_rentals.length,9);
});
