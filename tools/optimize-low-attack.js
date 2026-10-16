'use strict';

const toid = name => String(name).toLowerCase().replace(/[^a-z0-9]/g, '');
const CALLERS = new Set(['metronome', 'mimic', 'mirrormove', 'transform']);
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];

/** Optimize validated Gen 2 catalog sets without changing HP, gender or shiny
 * status. Use the generation-specific dex: e.g. Pursuit and Thief are Special.
 * Hidden Power and external move access are deliberately outside this policy.
 * The inventory, measured Nash mixture and frozen opponent prior stay original.
 */
function optimizeLowAttack(sets, dex) {
	const optimized = structuredClone(sets);
	let changedSets = 0;
	for (const set of optimized) {
		const ids = set.moves.map(toid);
		if (ids.some(id => id.startsWith('hiddenpower'))) continue;
		const moves = ids.map(id => {
			if (!dex.moves[id]) throw new Error(`unknown move: ${id}`);
			return dex.moves[id];
		});
		if (moves.some(move => move.category === 'Physical') || ids.some(id => CALLERS.has(id))) continue;
		const species = dex.species[toid(set.species)];
		if (!species) throw new Error(`unknown species: ${set.species}`);
		// This is a build step on validated inventory records, not an importer.
		for (const stat of STATS) {
			if (!Number.isInteger(set.ivs?.[stat]) || set.ivs[stat] < 0 || set.ivs[stat] > 31 ||
				!Number.isInteger(set.evs?.[stat]) || set.evs[stat] < 0 || set.evs[stat] > 255) {
				throw new Error(`${set.species}: complete validated IV/EV maps required`);
			}
		}
		const dv = stat => Math.floor(set.ivs[stat] / 2);
		const old = dv('atk');
		const gender = atk => species.gender || (atk < (species.genderRatio?.F ?? 0.5) * 16 ? 'F' : 'M');
		const shiny = atk => ['def', 'spe', 'spa'].every(stat => dv(stat) === 10) && atk % 4 >= 2;
		let minimum = old;
		for (let candidate = 0; candidate <= old; candidate++) {
			if (candidate % 2 === old % 2 && gender(candidate) === gender(old) && shiny(candidate) === shiny(old)) {
				minimum = candidate;
				break;
			}
		}
		if (minimum !== old || set.evs.atk !== 0) {
			set.ivs.atk = minimum * 2;
			set.evs.atk = 0;
			changedSets++;
		}
	}
	return { sets: optimized, changedSets };
}

module.exports = { optimizeLowAttack };
