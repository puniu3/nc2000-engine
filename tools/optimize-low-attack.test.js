'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const dex = require('../data/gen2stadium2.json');
const { optimizeLowAttack } = require('./optimize-low-attack');

function set(species, moves, atk = 31) {
	return { species, moves,
		ivs: { hp: 31, atk, def: 31, spa: 31, spd: 31, spe: 31 },
		evs: { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 } };
}
const optimize = mon => optimizeLowAttack([mon], dex).sets[0];

test('preserves male gender threshold and maximum HP parity', () => {
	for (const species of ['Venusaur', 'Jolteon', 'Vaporeon', 'Umbreon']) {
		const mon = optimize(set(species, ['Rest']));
		assert.equal(mon.ivs.atk, 6, species); // male threshold DV 2, first odd DV 3
		assert.equal(mon.ivs.hp, 31);
		assert.equal(mon.evs.atk, 0);
	}
	assert.equal(optimize(set('Gengar', ['Perish Song'])).ivs.atk, 18); // male, 1:1 ratio
});

test('female, fixed-gender and genderless sets can reach odd DV 1', () => {
	for (const mon of [set('Gengar', ['Perish Song'], 2), set('Jynx', ['Psychic']), set('Starmie', ['Surf'])]) {
		const result = optimize(mon);
		assert.equal(result.ivs.atk, 2);
		assert.equal(result.ivs.hp, mon.ivs.hp);
	}
});

test('preserves an even Attack DV parity rather than silently increasing HP', () => {
	const mon = set('Starmie', ['Surf'], 8);
	mon.ivs.hp = 14;
	const result = optimize(mon);
	assert.equal(result.ivs.atk, 0);
	assert.equal(result.ivs.hp, 14);
});

test('preserves the shiny DV predicate together with male gender and HP', () => {
	const mon = set('Gengar', ['Psychic']);
	Object.assign(mon.ivs, { hp: 16, def: 20, spe: 20, spa: 20, spd: 20 });
	mon.gender = 'M';
	mon.shiny = true;
	const result = optimize(mon);
	assert.equal(result.ivs.atk, 22); // DV 11: male, odd and shiny
	assert.equal(result.shiny, true);
	assert.equal(result.gender, 'M');
	assert.equal(result.ivs.hp, 16);
});

test('excludes Physical moves, every Hidden Power spelling and external move access', () => {
	for (const move of ['Explosion', 'Counter', 'Hidden Power', 'Hidden Power Ice',
		'Metronome', 'Mimic', 'Mirror Move', 'Transform']) {
		const mon = set('Gengar', [move]);
		assert.deepEqual(optimize(mon), mon, move);
	}
});

test('uses Gen 2 categories and allows Sleep Talk on qualifying own moves', () => {
	for (const mon of [set('Jynx', ['Thief']), set('Umbreon', ['Pursuit']),
		set('Suicune', ['Surf', 'Ice Beam', 'Rest', 'Sleep Talk'])]) {
		assert.equal(optimize(mon).evs.atk, 0);
	}
});

test('does not mutate inputs, is idempotent, and fails closed on unknown data', () => {
	const original = [set('Starmie', ['Surf'])];
	const frozen = structuredClone(original);
	const once = optimizeLowAttack(original, dex);
	assert.deepEqual(original, frozen);
	assert.equal(once.changedSets, 1);
	const twice = optimizeLowAttack(once.sets, dex);
	assert.deepEqual(twice.sets, once.sets);
	assert.equal(twice.changedSets, 0);
	assert.throws(() => optimize(set('Starmie', ['Nonexistent Move'])), /unknown move/);
	const incomplete = set('Starmie', ['Surf']);
	delete incomplete.evs;
	assert.throws(() => optimize(incomplete), /complete validated/);
});

test('generated catalog matches the independently audited 27 adjustments exactly', () => {
	const catalog = require('../data/team-pool-v2/team-pool.json');
	const inventory = require('../data/team-inventory-v1/inventory.json');
	const audit = require('../findings/low-attack-catalog-v1/audit.json');
	const original = new Map(inventory.teams.map(team => [team.id, team.sets]));
	assert.equal(catalog.teams.length, 24);
	assert.equal(catalog.setOptimization.changedTeams, 14);
	assert.equal(catalog.setOptimization.changedSets, 27);
	assert.equal(audit.rows.length, 27);
	for (const team of catalog.teams) {
		const expected = structuredClone(original.get(team.id));
		for (const row of audit.rows.filter(row => row.baseId === team.id)) {
			const mon = expected[row.slot - 1];
			assert.equal(mon.species, row.species);
			assert.equal(Math.floor(mon.ivs.atk / 2), row.attackDvBefore);
			assert.equal(mon.evs.atk, row.attackEvBefore);
			mon.ivs.atk = row.attackDvAfter * 2;
			mon.evs.atk = 0;
		}
		assert.deepEqual(team.sets, expected, team.id);
	}
});
