// Export per-species move-legality sets for gen2nc2000 to
// data/learnsets-gen2.json, probed through PS's REAL TeamValidator so the
// acceptance set mirrors the format's full lens exactly: Obtainable Moves
// (level-up/TM/egg/tradeback gen-1 sources, prevos, Smeargle Sketch), Event
// Moves Clause (event-only moves banned), the Uber species ban, and the
// min/max level rules (50-55, so move legality is probed per level).
//
// Contract: FLAT per-species sets — each (species, move) is validated as a
// single-move set, so cross-move compatibility constraints (e.g. egg moves
// whose gen-2 parents are mutually exclusive, event-only combinations) are
// NOT encoded. A client validating against this table accepts a small
// superset of true PS legality; it never rejects a PS-legal move.
//
// Output schema:
//   meta: { format, mod, psCommit, generator, generated, note, counts }
//   hpDvs: { <typeid>: {atk?, def?} }   PS typechart HPdvs (DV units) — the
//          canonical DV spreads the validator's canonicalizer applies for
//          typed Hidden Power sets
//   species: {
//     <speciesid>: {
//       minLevel?: N          smallest legal level in 50..55 (omitted = 50;
//                             e.g. dragonite/tyranitar = 55, evolve at 55)
//       moves: [moveids...]   legal at level 55 (single-move probe)
//       moveMinLevel?: { <moveid>: N }  moves only legal from level N > minLevel
//                             (level-up moves learned inside the 51..55 window)
//     }
//   }
'use strict';
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { PS_ROOT, sim, MOD, FORMAT, legal } = require('./ps');

const dex = sim.Dex.mod(MOD);
const validator = new sim.TeamValidator(FORMAT);

// The format's move universe (typed Hidden Power variants collapse onto the
// base hiddenpower learnset entry — probe the base id only; the mod's move
// overrides make dex.moves.all() repeat some ids — dedupe).
const seenIds = new Set();
const allMoves = dex.moves.all().filter(m => {
	if (!legal(m) || m.realMove || seenIds.has(m.id)) return false;
	seenIds.add(m.id);
	return true;
});

// Probe set: default EVs/DVs chosen to be unconditionally clean under the
// gen-2 stat checks (DV 15s: HP DV consistent, not shiny, SpA==SpD; gender
// left blank so the validator derives it from the Atk DV). Unown's letter
// forme is DV-derived (Obtainable Formes: base Unown = forme A), so it
// probes with the max forme-A spread instead (Atk/Def/Spe DV 9, Spc DV 15,
// HP DV consistent).
const IVS_DEFAULT = { hp: 30, atk: 30, def: 30, spa: 30, spd: 30, spe: 30 };
const IVS_UNOWN_A = { hp: 30, atk: 18, def: 18, spa: 30, spd: 30, spe: 18 };
function probe(speciesName, moveName, level) {
	const set = {
		name: '', species: speciesName, item: '', ability: '', nature: '', gender: '',
		moves: [moveName],
		evs: { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 },
		ivs: { ...(speciesName === 'Unown' ? IVS_UNOWN_A : IVS_DEFAULT) },
		level,
	};
	return validator.validateSet(set, {}) === null;
}

// Candidate moves per species: any move with a gen<=2 learnset source
// (prevos included via getFullLearnset). Smeargle's Sketch reaches moves
// with no learnset entry at all, so it probes the full universe. This is
// only a prefilter — acceptance is always the real validator's verdict.
function candidateMoves(species) {
	if (species.id === 'smeargle') return allMoves;
	const ids = new Set();
	for (const { learnset } of dex.species.getFullLearnset(species.id)) {
		for (const moveid in learnset) {
			if (learnset[moveid].some(source => source.startsWith('1') || source.startsWith('2'))) {
				ids.add(moveid);
			}
		}
	}
	return allMoves.filter(m => ids.has(m.id));
}

const speciesOut = {};
let banned = [];
const t0 = Date.now();
const speciesList = dex.species.all().filter(legal);
for (const species of speciesList) {
	const candidates = candidateMoves(species);
	const legalAt55 = candidates.filter(m => probe(species.name, m.name, 55));
	if (!legalAt55.length) {
		banned.push(species.id); // Uber-tagged: every single-move probe fails
		continue;
	}
	// Species floor: smallest level in 50..55 at which ANY move validates
	// (captures "must be at least level N to be evolved").
	let minLevel = 55;
	outer: for (let level = 50; level < 55; level++) {
		for (const m of legalAt55) {
			if (probe(species.name, m.name, level)) { minLevel = level; break outer; }
		}
	}
	// Per-move floor above the species floor (level-up moves learned in the
	// 51..55 window). Only moves failing at minLevel need refinement.
	const moveMinLevel = {};
	for (const m of legalAt55) {
		if (minLevel === 55 || probe(species.name, m.name, minLevel)) continue;
		let lo = minLevel + 1;
		while (lo < 55 && !probe(species.name, m.name, lo)) lo++;
		moveMinLevel[m.id] = lo;
	}
	const entry = { moves: legalAt55.map(m => m.id).sort() };
	if (minLevel > 50) entry.minLevel = minLevel;
	if (Object.keys(moveMinLevel).length) entry.moveMinLevel = moveMinLevel;
	speciesOut[species.id] = entry;
	process.stdout.write(`\r${Object.keys(speciesOut).length + banned.length}/${speciesList.length} ${species.id}                `);
}
console.log();

// Sanity: the excluded species must be exactly the Uber-tagged ones.
const ubers = speciesList.filter(s => s.tier === 'Uber').map(s => s.id).sort();
banned = banned.sort();
if (JSON.stringify(banned) !== JSON.stringify(ubers)) {
	throw new Error(`probe-banned ${JSON.stringify(banned)} != Uber tier ${JSON.stringify(ubers)}`);
}

// PS typechart HPdvs (DV units): the canonical spreads the validator applies
// when a typed Hidden Power set has maxed DVs (and the canonicalizer's fix).
const hpDvs = {};
for (const t of dex.types.all()) {
	if (!t.exists || t.gen > 2 || t.isNonstandard) continue;
	if (t.HPdvs && Object.keys(t.HPdvs).length) hpDvs[t.id] = t.HPdvs;
}

let psCommit = 'unknown';
try { psCommit = execSync('git rev-parse HEAD', { cwd: PS_ROOT }).toString().trim(); } catch {}

const counts = {
	species: Object.keys(speciesOut).length,
	bannedUbers: banned.length,
	movePairs: Object.values(speciesOut).reduce((n, e) => n + e.moves.length, 0),
	speciesWithMinLevel: Object.values(speciesOut).filter(e => e.minLevel).length,
	moveMinLevelEntries: Object.values(speciesOut).reduce((n, e) => n + Object.keys(e.moveMinLevel || {}).length, 0),
};
const out = {
	meta: {
		format: FORMAT, mod: MOD, psCommit,
		generator: 'tools/export-learnsets.js',
		generated: new Date().toISOString().slice(0, 10),
		note: 'Flat per-species acceptance sets probed one move at a time through PS TeamValidator.validateSet under the full gen2nc2000 lens (Obtainable + Event Moves Clause + level rules). Cross-move compatibility constraints (incompatible egg-move parents, event-only combinations) are NOT encoded: clients accept a small superset of true legality, never rejecting a PS-legal move.',
		counts,
	},
	hpDvs,
	species: speciesOut,
};

const dest = path.join(__dirname, '..', 'data', 'learnsets-gen2.json');
fs.writeFileSync(dest, JSON.stringify(out));
console.log(`wrote ${dest} (${(fs.statSync(dest).size / 1024).toFixed(0)} KB) in ${((Date.now() - t0) / 1000).toFixed(0)}s:`, JSON.stringify(counts));
