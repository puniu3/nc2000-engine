// Build the NC2000 meta team pool (data/meta-pool-v0) from curated sources.
//
// Sources (see data/meta-pool-v0/README.md for provenance):
//   raw/hc75-top8.txt    Historia Cup 7.5 top-8 teams (live NC2000 tournament, 2022-05-03)
//   raw/samples-27.txt   Smogon NC2000 Resource Hub sample teams (expert/JP-community built)
//   raw/vr.json          Chio's viability ranking (species -> tier)
//   raw/usage-hc75.json  HC7.5 species usage over all 27 entrant teams
//
// Every team is validated with PS's own TeamValidator for gen2nc2000 — the same
// oracle the fixture corpus uses. Output: meta-pool.json (validated, packed,
// provenance-annotated, pedigree-ranked).
//
// Usage: node tools/build-meta-pool.js
'use strict';
const fs = require('fs');
const path = require('path');
const { sim, FORMAT, MOD } = require('./ps');
const { Teams, TeamValidator, Dex } = sim;

const DIR = path.join(__dirname, '..', 'data', 'meta-pool-v0');
const RAW = path.join(DIR, 'raw');
const validator = new TeamValidator(FORMAT);
const dex = Dex.mod(MOD);

// ---------------------------------------------------------------- sources
const HC75_URL = 'https://gold.hatenadiary.jp/entry/2022/05/05/151859';
const HUB_URL = 'https://www.smogon.com/forums/threads/nintendo-cup-2000-resource-hub.3682691/';

const HC75_PLACEMENT = { '1st': 100, '2nd': 80, '3rd': 70, '4th': 60, top8: 50 };

const SAMPLE_AUTHORS = {
	1: 'Beelzemon 2003',
	2: 'Beelzemon 2003, Chio, the Parrot 99',
	3: 'Beelzemon 2003, Chio, Friend of Mr. Golem 120, MissingNo.',
	4: 'Beelzemon 2003, Chio, Friend of Mr. Golem 120, Stealth Croc',
	5: 'Beelzemon 2003, Chio, Friend of Mr. Golem 120',
	6: 'Kitty',
	7: 'Japanese Poké Cup community',
	8: 'Japanese Poké Cup community',
	9: 'Beelzemon 2003, Japanese Poké Cup community',
	10: 'Japanese Poké Cup community',
	11: 'Beelzemon 2003, Japanese Poké Cup community',
	12: 'Kitty',
	13: 'Japanese Poké Cup community',
	14: 'Beelzemon 2003, Japanese Poké Cup community',
	15: 'Beelzemon 2003, Japanese Poké Cup community',
	16: 'Beelzemon 2003, Japanese Poké Cup community',
	17: 'Beelzemon 2003, Japanese Poké Cup community',
	18: 'Beelzemon 2003, Japanese Poké Cup community',
	19: 'Beelzemon 2003, Chio, Friend of Mr. Golem 120, Japanese Poké Cup community',
	20: 'Japanese Poké Cup community',
	21: 'Japanese Poké Cup community',
	22: 'Beelzemon 2003, Japanese Poké Cup community',
	23: 'Japanese Poké Cup community',
	24: 'Japanese Poké Cup community',
	25: 'Beelzemon 2003, Japanese Poké Cup community',
	26: 'Japanese Poké Cup community',
	27: 'Chio, Friend of Mr. Golem 120, Japanese Poké Cup community',
};

// Thread-documented deviations from the original JP teams (the thread swapped
// Bright Powder out assuming a ban that the shipped PS format does not have).
const SAMPLE_NOTES = {
	9: 'Smeargle had Bright Powder in the original team.',
	11: 'Suicune had Bright Powder in the original team.',
	14: 'Starmie had Bright Powder in the original team.',
	15: 'Starmie had Bright Powder in the original team.',
	16: 'Porygon2 had Bright Powder in the original team.',
	17: 'Suicune had Bright Powder in the original team.',
	18: 'Thread lists a Mean Look + Baton Pass Umbreon variant.',
	19: 'Thread lists a Double Team Miltank variant and a Mean Look + Baton Pass Umbreon variant.',
	22: 'Snorlax had Bright Powder in the original team.',
	25: 'Starmie had Bright Powder in the original team.',
	27: 'Gligar Hidden Power Rock can be Wing Attack; on cartridge GSC, Blissey Thunder becomes Present.',
};

const VR_POINTS = { S: 8, 'S-': 7, 'A+': 6, A: 5, 'A-': 4, 'B+': 3, B: 2, 'B-': 1 };

// ---------------------------------------------------------------- helpers
function splitTeams(text) {
	const marks = [];
	const re = /^===\s*(.*?)\s*===\s*$/gm;
	let m;
	while ((m = re.exec(text))) marks.push([m.index, re.lastIndex, m[1]]);
	return marks.map(([, bodyStart, name], i) => ({
		name,
		body: text.slice(bodyStart, i + 1 < marks.length ? marks[i + 1][0] : text.length).trim(),
	}));
}

function importTeam(body, label) {
	const team = Teams.import(body);
	if (!team || team.length !== 6) throw new Error(`${label}: imported ${team ? team.length : 0} sets, want 6`);
	for (const set of team) {
		set.ability = 'No Ability';
		// Universal convention in this format: fully trained (max stat exp).
		set.evs = { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 };
		set.happiness = set.moves.some(mv => dex.moves.get(mv).id === 'frustration') ? 0 : 255;
	}
	return team;
}

// ---------------------------------------------------------------- build
const vr = JSON.parse(fs.readFileSync(path.join(RAW, 'vr.json'), 'utf8'));
const usage = JSON.parse(fs.readFileSync(path.join(RAW, 'usage-hc75.json'), 'utf8'));

const pool = [];
let failures = 0;

function addTeam(id, tier, provenance, body, label) {
	let team;
	try {
		team = importTeam(body, label);
	} catch (err) {
		console.error(`IMPORT FAIL ${id}: ${err.message}`);
		failures++;
		return;
	}
	const errors = validator.validateTeam(team);
	if (errors) {
		console.error(`VALIDATE FAIL ${id}:`);
		for (const e of errors) console.error(`  - ${e}`);
		failures++;
		return;
	}
	const species = team.map(s => dex.species.get(s.species).name);
	const vrScore = species.reduce((a, sp) => a + (VR_POINTS[vr[sp]] || 0), 0) / 6;
	const usageScore = species.reduce((a, sp) => a + (usage[sp] ? usage[sp].pct : 0), 0) / 6;
	// Canonical set shape = pack→unpack round-trip, identical to fixture
	// p1team/p2team (the engine's PokemonSet deserializes it directly).
	const packed = Teams.pack(team);
	pool.push({
		id, tier, provenance,
		species,
		levels: team.map(s => s.level),
		pedigree: {
			tournamentPoints: provenance.placementPoints || 0,
			vrMean: +vrScore.toFixed(2),
			hc75UsageMean: +usageScore.toFixed(1),
		},
		export: Teams.export(team),
		packed,
		sets: Teams.unpack(packed),
	});
}

// T1: Historia Cup 7.5 top 8 (tournament-proven, NC2000 proper)
for (const { name, body } of splitTeams(fs.readFileSync(path.join(RAW, 'hc75-top8.txt'), 'utf8'))) {
	const m = /^HC7\.5 (\S+) (\S+)$/.exec(name);
	const [, placement, player] = m;
	addTeam(`hc75-${placement}-${player.toLowerCase()}`, 'T1', {
		source: 'Historia Cup 7.5 (live NC2000 tournament, 27 players, 2022-05-03, hosted by Gold)',
		event: 'Historia Cup 7.5',
		player,
		placement,
		placementPoints: HC75_PLACEMENT[placement],
		url: HC75_URL,
		notes: 'Transcribed from the Smogon NC2000 Resource Hub tournament report; Hidden Power DV spreads use the canonical spreads from the hub sample teams.',
	}, body, name);
}

// Thread teams that are illegal under the shipped PS format (validator-verified).
const SAMPLE_SKIP = {
	27: "Gligar's Earthquake is event-only on PS → Event Moves Clause violation; no documented substitute, so the team is excluded.",
};

// T2: Smogon NC2000 Resource Hub sample teams (expert-curated)
for (const { name, body } of splitTeams(fs.readFileSync(path.join(RAW, 'samples-27.txt'), 'utf8'))) {
	const n = parseInt(/Sample (\d+)/.exec(name)[1], 10);
	if (SAMPLE_SKIP[n]) {
		console.log(`skip sample-${n}: ${SAMPLE_SKIP[n]}`);
		continue;
	}
	addTeam(`sample-${String(n).padStart(2, '0')}`, 'T2', {
		source: 'Smogon NC2000 Resource Hub sample teams (built by international + Japanese experts)',
		authors: SAMPLE_AUTHORS[n],
		url: HUB_URL,
		notes: SAMPLE_NOTES[n] || undefined,
	}, body, name);
}

if (failures) {
	console.error(`\n${failures} team(s) failed — pool NOT written.`);
	process.exit(1);
}

// Rank: tournament pedigree first, then VR support, then tournament usage support.
pool.sort((a, b) =>
	b.pedigree.tournamentPoints - a.pedigree.tournamentPoints ||
	b.pedigree.vrMean - a.pedigree.vrMean ||
	b.pedigree.hc75UsageMean - a.pedigree.hc75UsageMean);
pool.forEach((t, i) => (t.rank = i + 1));

const out = {
	meta: {
		format: FORMAT,
		mod: MOD,
		generated: '2026-07-16',
		teams: pool.length,
		rankCriteria: 'tournamentPoints (HC7.5 placement) desc, then vrMean (Chio VR) desc, then hc75UsageMean desc',
		sources: [
			{ name: 'Smogon NC2000 Resource Hub', url: HUB_URL },
			{ name: 'Historia Cup 7.5 report', url: HC75_URL },
			{ name: "Chio's viability ranking", url: 'https://seesaawiki.jp/pbs-thread/d/Tear%20list%20in%20Nintendo%20Cup%202000' },
		],
		excluded: [
			'Historia Cup 10/11 (2024-25) — played under the incompatible Historia Cup 2024 special ruleset (Kanto species capped at L52 etc.), not NC2000.',
			...Object.entries(SAMPLE_SKIP).map(([n, why]) => `sample-${n} — ${why}`),
		],
	},
	teams: pool,
};
fs.writeFileSync(path.join(DIR, 'meta-pool.json'), JSON.stringify(out, null, 1) + '\n');
console.log(`meta-pool.json written: ${pool.length} teams (${pool.filter(t => t.tier === 'T1').length} T1 + ${pool.filter(t => t.tier === 'T2').length} T2)`);
for (const t of pool) {
	console.log(
		`${String(t.rank).padStart(2)} ${t.tier} ${t.id.padEnd(22)} tp=${String(t.pedigree.tournamentPoints).padStart(3)}` +
		` vr=${t.pedigree.vrMean.toFixed(2).padStart(5)} use=${t.pedigree.hc75UsageMean.toFixed(1).padStart(5)}  ${t.species.join('/')}`);
}
