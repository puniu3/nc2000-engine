// M14a oracle cross-check: the wasm validator vs PS's real TeamValidator.
//
//   node tools/validate-oracle.js        (needs crates/wasm/build.sh nodejs)
//
// Phases:
//   A. all 34 meta-pool teams + all 120 fixture teams -> both must accept.
//   B. ~200 seeded mutations (illegal moves, legal-but-random move stuffing,
//      level/DV/gender/shiny violations, clause dupes, illegal species,
//      nickname faults, ...) -> PS verdict vs ours per case.
//   C. canonicalizeTeam on every mutant: whenever it claims ok, PS must
//      accept the canonicalized team.
//
// Agreement contract: "superset" disagreements (we accept, PS rejects — the
// documented flat-learnset caveat: cross-move compatibility is not encoded)
// are allowed and counted; the reverse direction (we reject a PS-legal
// team) must be ZERO.
'use strict';
const fs = require('fs');
const path = require('path');
const { sim, FORMAT } = require('./ps');

const REPO = path.join(__dirname, '..');
const wasm = require(path.join(REPO, 'crates/wasm/pkg-node/nc2000_wasm.js'));

const psValidator = new sim.TeamValidator(FORMAT);
const dex = new wasm.Dex();
const ours = new wasm.Validator(dex);
const learnsets = JSON.parse(fs.readFileSync(path.join(REPO, 'data/learnsets-gen2.json'), 'utf8'));
const dexJson = JSON.parse(fs.readFileSync(path.join(REPO, 'data/gen2stadium2.json'), 'utf8'));
const psDex = sim.Dex.mod('gen2stadium2');

const clone = t => JSON.parse(JSON.stringify(t));
const psVerdict = team => psValidator.validateTeam(clone(team)); // null = accept (PS mutates its input)
const ourVerdict = team => JSON.parse(ours.validateTeam(JSON.stringify(team)));

// ---------------------------------------------------------------- phase A

const pool = JSON.parse(fs.readFileSync(path.join(REPO, 'data/meta-pool-v0/meta-pool.json'), 'utf8'));
const baseTeams = []; // {id, team}
for (const t of pool.teams) baseTeams.push({ id: `pool:${t.id}`, team: t.sets });
for (const dir of ['puredata', 'full']) {
	const d = path.join(REPO, 'fixtures/corpus-v1', dir);
	for (const f of fs.readdirSync(d).sort()) {
		if (!f.endsWith('.json')) continue;
		const fx = JSON.parse(fs.readFileSync(path.join(d, f), 'utf8'));
		baseTeams.push({ id: `${dir}/${f}:p1`, team: fx.p1team });
		baseTeams.push({ id: `${dir}/${f}:p2`, team: fx.p2team });
	}
}

let aFail = 0, aFixFindings = 0;
for (const { id, team } of baseTeams) {
	const ps = psVerdict(team);
	const us = ourVerdict(team);
	aFixFindings += us.findings.length;
	if (ps !== null) { console.log(`A REJECT(PS) ${id}: ${ps.join(' | ')}`); aFail++; }
	if (!us.ok || us.findings.length) { console.log(`A FINDINGS(ours) ${id}: ${JSON.stringify(us.findings)}`); if (!us.ok) aFail++; }
}
const poolCount = pool.teams.length;
console.log(`\nPhase A: ${baseTeams.length} teams (${poolCount} pool + ${baseTeams.length - poolCount} fixture): ` +
	`${aFail === 0 ? 'both validators accept all' : aFail + ' FAILURES'}; stray findings: ${aFixFindings}`);
if (aFail) process.exit(1);

// ---------------------------------------------------------------- phase B

// deterministic RNG (mulberry32)
let rngState = 42 >>> 0;
function rand() {
	rngState = (rngState + 0x6D2B79F5) >>> 0;
	let t = rngState;
	t = Math.imul(t ^ (t >>> 15), t | 1);
	t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
	return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const ri = n => Math.floor(rand() * n);
const pick = a => a[ri(a.length)];

const allMoveIds = Object.keys(dexJson.moves).filter(m => !m.startsWith('hiddenpower') || m === 'hiddenpower');
const learnsetOf = s => learnsets.species[sim.toID(s)];
const ubers = ['Mewtwo', 'Mew', 'Lugia', 'Ho-Oh', 'Celebi'];

function randomBase() {
	const b = pick(baseTeams);
	return { id: b.id, team: clone(b.team) };
}

// materialize a set's ivs before a DV mutation (missing = all 31, PS default)
function ivsOf(set) {
	set.ivs ??= { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
	return set.ivs;
}

// Mutation operators: name -> (team) -> description (or null to resample).
const ops = {
	'illegal-move'(team) {
		const i = ri(team.length);
		const ls = learnsetOf(team[i].species);
		if (!ls) return null;
		const illegal = allMoveIds.filter(m => !ls.moves.includes(m));
		const mv = pick(illegal);
		team[i].moves[ri(team[i].moves.length)] = mv;
		return `${team[i].species} gets ${mv}`;
	},
	// deliberate superset probe: pairs of individually-legal moves PS
	// rejects as a COMBINATION (incompatible egg-move fathers; a gen-1
	// exclusive next to a gen-2 egg move). The flat learnset cannot see
	// these — expected outcome: we accept, PS rejects (counted superset).
	'incompatible-combo'(team) {
		const combos = [
			['Marowak', ['Ancient Power', 'Belly Drum']],
			['Charizard', ['Ancient Power', 'Beat Up']],
			['Cloyster', ['Bide', 'Rapid Spin']],
			['Pikachu', ['Body Slam', 'Encore']],
			['Umbreon', ['Bide', 'Charm']],
		];
		const [sp, moves] = pick(combos);
		const i = ri(team.length);
		team[i].species = sp;
		team[i].name = sp;
		team[i].moves = moves;
		team[i].gender = undefined;
		return `${sp} with ${moves.join(' + ')}`;
	},
	// superset probe: stuff a mon with 4 random moves from its own flat
	// learnset — PS may reject cross-move-incompatible combos we accept
	'learnset-stuffing'(team) {
		const i = ri(team.length);
		const ls = learnsetOf(team[i].species);
		if (!ls || ls.moves.length < 4) return null;
		const pool = ls.moves.filter(m => !(ls.moveMinLevel?.[m] > team[i].level));
		const moves = new Set();
		while (moves.size < 4) moves.add(pick(pool));
		team[i].moves = [...moves];
		return `${team[i].species} = ${[...moves].join('/')}`;
	},
	'level-low'(team) {
		const i = ri(team.length);
		team[i].level = 49;
		return `${team[i].species} level 49`;
	},
	'level-high'(team) {
		const i = ri(team.length);
		team[i].level = 56;
		return `${team[i].species} level 56`;
	},
	'level-sum'(team) {
		for (const s of team) s.level = 52;
		return `all levels 52 (3 lowest sum 156)`;
	},
	'dv-spc'(team) {
		const i = ri(team.length);
		const ivs = ivsOf(team[i]);
		ivs.spd = (ivs.spa + 4) % 32;
		return `${team[i].species} spa iv ${ivs.spa} vs spd iv ${ivs.spd}`;
	},
	'dv-hp'(team) {
		const i = ri(team.length);
		const ivs = ivsOf(team[i]);
		ivs.hp = (ivs.hp + 4) % 32;
		return `${team[i].species} hp iv skewed to ${ivs.hp}`;
	},
	'dv-gender'(team) {
		for (let tries = 0; tries < 12; tries++) {
			const i = ri(team.length);
			const sp = psDex.species.get(team[i].species);
			if (sp.gender) continue; // fixed-gender: PS silently overrides
			const atkDV = Math.floor(ivsOf(team[i]).atk / 2);
			const expected = atkDV >= sp.genderRatio.F * 16 ? 'M' : 'F';
			team[i].gender = expected === 'M' ? 'F' : 'M';
			return `${team[i].species} gender ${team[i].gender} vs atk DV ${atkDV}`;
		}
		return null;
	},
	'dv-shiny'(team) {
		const i = ri(team.length);
		team[i].ivs = { hp: 0, atk: 28, def: 20, spa: 20, spd: 20, spe: 20 };
		team[i].gender = undefined;
		return `${team[i].species} shiny DVs without shiny flag`;
	},
	'hp-type-mismatch'(team) {
		const i = ri(team.length);
		const ivs = ivsOf(team[i]);
		const derived = ['Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel',
			'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark'][
			4 * (Math.floor(ivs.atk / 2) % 4) + (Math.floor(ivs.def / 2) % 4)];
		const want = derived === 'Ice' ? 'Fire' : 'Ice';
		team[i].moves[0] = `Hidden Power ${want}`;
		return `${team[i].species} Hidden Power ${want} on ${derived} DVs`;
	},
	'dupe-species'(team) {
		if (team.length < 2) return null;
		team[1].species = team[0].species;
		team[1].name = team[0].species;
		team[1].moves = clone(team[0].moves);
		if (team[0].ivs) team[1].ivs = clone(team[0].ivs); else delete team[1].ivs;
		team[1].gender = team[0].gender;
		return `two ${team[0].species}`;
	},
	'dupe-item'(team) {
		const withItem = team.filter(s => s.item);
		if (withItem.length < 1 || team.length < 2) return null;
		const src = withItem[0];
		const dst = team.find(s => s !== src);
		dst.item = src.item;
		return `two ${src.item}`;
	},
	'uber-species'(team) {
		const i = ri(team.length);
		const u = pick(ubers);
		team[i].species = u;
		team[i].name = u;
		team[i].moves = ['Rest'];
		return `${u} injected`;
	},
	'unknown-species'(team) {
		const i = ri(team.length);
		team[i].species = 'Blaziken';
		team[i].name = 'Blaziken';
		team[i].moves = ['Ember'];
		return `Blaziken injected`;
	},
	'underleveled-evo'(team) {
		const i = ri(team.length);
		const sp = pick(['Dragonite', 'Tyranitar']);
		const ls = learnsetOf(sp);
		team[i].species = sp;
		team[i].name = sp;
		team[i].level = Math.min(team[i].level, 54);
		team[i].moves = [pick(ls.moves.filter(m => !ls.moveMinLevel?.[m]))];
		return `${sp} at level ${team[i].level}`;
	},
	'event-only-move'(team) {
		const i = ri(team.length);
		team[i].species = 'Gligar';
		team[i].name = 'Gligar';
		team[i].moves = ['Earthquake'];
		team[i].gender = undefined;
		return `Gligar with event-only Earthquake`;
	},
	'move-none'(team) {
		const i = ri(team.length);
		team[i].moves = [];
		return `${team[i].species} no moves`;
	},
	'move-five'(team) {
		const i = ri(team.length);
		const ls = learnsetOf(team[i].species);
		if (!ls || ls.moves.length < 5) return null;
		const pool = ls.moves.filter(m => !(ls.moveMinLevel?.[m] > team[i].level));
		const moves = new Set();
		while (moves.size < 5) moves.add(pick(pool));
		team[i].moves = [...moves];
		return `${team[i].species} 5 moves`;
	},
	'move-dupe'(team) {
		const i = ri(team.length);
		if (team[i].moves.length < 2) return null;
		team[i].moves[1] = team[i].moves[0];
		return `${team[i].species} duplicate ${team[i].moves[0]}`;
	},
	'unknown-item'(team) {
		const i = ri(team.length);
		team[i].item = 'Choice Band';
		return `${team[i].species} holds gen-3 Choice Band`;
	},
	'nickname-long'(team) {
		const i = ri(team.length);
		team[i].name = 'A'.repeat(19);
		return `${team[i].species} 19-char nickname`;
	},
	'nickname-species'(team) {
		const i = ri(team.length);
		const other = team[i].species === 'Pikachu' ? 'Snorlax' : 'Pikachu';
		team[i].name = other;
		return `${team[i].species} nicknamed ${other}`;
	},
	'nickname-dupe'(team) {
		if (team.length < 2) return null;
		team[0].name = 'Blob';
		team[1].name = 'Blob';
		return `two mons named Blob`;
	},
	'unown-forme'(team) {
		const i = ri(team.length);
		team[i].species = 'Unown';
		team[i].name = 'Unown';
		team[i].moves = ['Hidden Power'];
		team[i].gender = undefined;
		// keep the base mon's DVs: almost never the forme-A spread
		return `Unown with forme ${'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[Math.floor(parseInt(
			['atk', 'def', 'spe', 'spa'].map(k => ivsOf(team[i])[k].toString(2).padStart(5, '0').slice(1, 3)).join(''), 2) / 10)]} DVs`;
	},
};

// case plan: heavier weight on the interesting classes, ~200 total
const plan = [];
const weight = {
	'illegal-move': 40, 'learnset-stuffing': 30, 'incompatible-combo': 10,
	'level-low': 7, 'level-high': 7, 'level-sum': 6,
	'dv-spc': 8, 'dv-hp': 8, 'dv-gender': 8, 'dv-shiny': 6, 'hp-type-mismatch': 8,
	'dupe-species': 10, 'dupe-item': 10,
	'uber-species': 8, 'unknown-species': 4, 'underleveled-evo': 6, 'event-only-move': 4,
	'move-none': 5, 'move-five': 5, 'move-dupe': 6,
	'unknown-item': 4, 'nickname-long': 3, 'nickname-species': 3, 'nickname-dupe': 3, 'unown-forme': 4,
};
for (const [op, n] of Object.entries(weight)) for (let k = 0; k < n; k++) plan.push(op);

let agreeReject = 0, agreeAccept = 0, superset = 0, reverse = 0, done = 0, canonOk = 0, canonSuperset = 0;
const supersetList = [], reverseList = [];
for (const op of plan) {
	let base, desc;
	for (let tries = 0; tries < 20 && !desc; tries++) {
		base = randomBase();
		desc = ops[op](base.team);
	}
	if (!desc) { console.log(`SKIP ${op}: no applicable base team`); continue; }
	done++;
	const ps = psVerdict(base.team);
	const us = ourVerdict(base.team);
	const label = `${op} [${base.id}] ${desc}`;
	let isSuperset = false;
	if (ps === null && us.ok) agreeAccept++;
	else if (ps !== null && !us.ok) agreeReject++;
	else if (ps !== null && us.ok) { superset++; isSuperset = true; supersetList.push(`${label}\n    PS: ${ps.join(' | ')}`); }
	else { reverse++; reverseList.push(`${label}\n    ours: ${JSON.stringify(us.findings.filter(f => f.severity === 'error'))}`); }

	// phase C: whenever canonicalize claims ok, PS must accept the fixed
	// team — except on superset cases, where the flat learnset can't see
	// PS's move-compatibility objection any better after fixing (the same
	// documented caveat, counted separately).
	const c = JSON.parse(ours.canonicalizeTeam(JSON.stringify(base.team)));
	if (c.ok) {
		const psFixed = psVerdict(c.team);
		if (psFixed === null) canonOk++;
		else if (isSuperset) canonSuperset++;
		else {
			reverse++; // canonicalizer broke a team PS otherwise accepts
			reverseList.push(`CANON ${label}\n    PS on canonicalized: ${psFixed.join(' | ')}`);
		}
	}
}

console.log(`\nPhase B: ${done} mutations`);
console.log(`  agree-reject: ${agreeReject}`);
console.log(`  agree-accept: ${agreeAccept}`);
console.log(`  superset (we accept, PS rejects — documented flat-learnset caveat): ${superset}`);
console.log(`  REVERSE (we reject / canonicalize-break a PS-legal team): ${reverse}`);
console.log(`  agreement rate: ${(100 * (agreeReject + agreeAccept) / done).toFixed(1)}%`);
console.log(`Phase C: canonicalize ok + PS accepts the fixed team: ${canonOk}; ` +
	`ok but PS still objects to the (unencodable) move combination: ${canonSuperset}`);
if (supersetList.length) {
	console.log(`\nsuperset cases:`);
	for (const s of supersetList) console.log(`  ${s}`);
}
if (reverseList.length) {
	console.log(`\nREVERSE cases (must be zero):`);
	for (const s of reverseList) console.log(`  ${s}`);
	process.exit(1);
}
console.log('\nOK: zero reverse disagreements.');
