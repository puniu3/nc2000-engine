'use strict';
const toid = name => String(name).toLowerCase().replace(/[^a-z0-9]/g, '');

/** Weighted direct-Attract source-set mass, not an encounter/win probability.
 * Team selection, matchup and move choice can all change actual exposure.
 */
function attractGenderCensus(pool, weightKey = 'weight') {
	const weightedSources = { M: 0, F: 0, N: 0 };
	const sources = [];
	let totalTeamWeight = 0;
	for (const team of pool.teams) {
		const weight = team[weightKey];
		if (!Number.isFinite(weight) || weight < 0) throw new Error(`invalid ${weightKey}: ${team.id}`);
		totalTeamWeight += weight;
		team.sets.forEach((set, index) => {
			if (!set.moves.some(move => toid(move) === 'attract')) return;
			if (!Object.hasOwn(weightedSources, set.gender)) throw new Error(`unknown Attract source gender: ${team.id}`);
			weightedSources[set.gender] += weight;
			sources.push({ team: team.id, slot: index + 1, species: set.species, gender: set.gender, weight });
		});
	}
	return { totalTeamWeight, weightedSources, sources,
		preferFemale: weightedSources.F > weightedSources.M,
		basis: 'female targets resist female sources and are susceptible to male sources; weighted direct-source-set mass is a proxy, not encounter or win probability' };
}

module.exports = { attractGenderCensus };
