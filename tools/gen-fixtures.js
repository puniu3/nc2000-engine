// Generate golden battle fixtures for the Rust conformance harness.
//
// Phase 1: play a battle with RandomPlayerAI over battle streams (choice
//          diversity, legality handled by the AI).
// Phase 2: replay the recorded inputLog on a fresh synchronous Battle and
//          snapshot an "essence" of the state after every input line that
//          produced log output. The Rust replayer must reproduce every
//          snapshot bit-for-bit (see conformance crate).
//
// Usage: node tools/gen-fixtures.js --n 30 --pool full|puredata --out fixtures/corpus-v1 --seed 42
'use strict';
const fs = require('fs');
const path = require('path');
const { sim, prng, rpai, FORMAT, MOD, fnNames, legal } = require('./ps');
const { Battle, BattleStream, getPlayerStreams, Dex, Teams, TeamValidator } = sim;
const { PRNG } = prng;
const { RandomPlayerAI } = rpai;

const validator = new TeamValidator(FORMAT);

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
const N = parseInt(args.n || '10', 10);
const POOL = args.pool || 'full';
const OUT = path.join(__dirname, '..', args.out || `fixtures/corpus-v1/${POOL}`);
const MASTER_SEED = parseInt(args.seed || '1', 10);

const dex = Dex.mod(MOD);

// ---------------------------------------------------------------- team pool
const uber = new Set(['mewtwo', 'mew', 'lugia', 'hooh', 'celebi']);
const speciesPool = dex.species.all().filter(s => legal(s) && !uber.has(s.id));

const pureDataMove = m => fnNames(m).length === 0;
function legalMoves(speciesId) {
	const lsData = dex.species.getLearnsetData(speciesId);
	if (!lsData?.learnset) return [];
	const ids = [];
	for (const [moveId, sources] of Object.entries(lsData.learnset)) {
		// Gen 2 sources only (no tradebacks, no 'S' event moves — Event Moves Clause).
		if (!sources.some(s => s.startsWith('2') && !s.includes('S'))) continue;
		const move = dex.moves.get(moveId);
		if (!legal(move)) continue;
		if (POOL === 'puredata' && !pureDataMove(move)) continue;
		ids.push(moveId);
	}
	return ids;
}

const itemPool = POOL === 'puredata' ? [] :
	dex.items.all().filter(legal).filter(i => !i.isPokeball).map(i => i.id);

function makeTeam(rng) {
	// Retry until the team passes the real NC2000 validator — fixtures must be
	// legal by PS's own definition, not by our learnset approximation.
	for (let attempt = 0; attempt < 20; attempt++) {
		const team = makeTeamOnce(rng);
		const errors = validator.validateTeam(team);
		if (!errors) return team;
	}
	throw new Error('could not generate a validator-clean team in 20 attempts');
}

function makeTeamOnce(rng) {
	const team = [];
	const usedSpecies = new Set();
	const usedItems = new Set(); // Item Clause = 1
	while (team.length < 6) {
		const s = rng.sample(speciesPool);
		if (usedSpecies.has(s.id)) continue;
		const moves = legalMoves(s.id);
		if (moves.length === 0) continue;
		usedSpecies.add(s.id);
		const picked = [];
		while (picked.length < Math.min(4, moves.length)) {
			const m = rng.sample(moves);
			if (!picked.includes(m)) picked.push(m);
		}
		let item = '';
		if (itemPool.length && rng.randomChance(3, 4)) {
			for (let tries = 0; tries < 10 && !item; tries++) {
				const cand = rng.sample(itemPool);
				if (!usedItems.has(cand)) { usedItems.add(cand); item = cand; }
			}
		}
		// Gen 2 DV rules: one shared Special DV (SpA==SpD) and HP DV derived
		// from the low bits of Atk/Def/Spe/Spc. ivs = DV*2 in PS encoding.
		const dv = () => rng.random(16);
		const [atk, def, spe, spc] = [dv(), dv(), dv(), dv()];
		const hpDv = (atk % 2) * 8 + (def % 2) * 4 + (spe % 2) * 2 + (spc % 2);
		const ivs = { hp: hpDv * 2, atk: atk * 2, def: def * 2, spa: spc * 2, spd: spc * 2, spe: spe * 2 };
		// Levels 50-51 keep any 3 picks under Max Total Level = 155.
		team.push({
			// Gen 2 canonical form per TeamValidator: ability is 'No Ability'.
			// (The sim runs whatever ability it is handed, even in gen2 — an
			// unvalidated default-filled ability makes e.g. Shed Skin fire.)
			name: s.name, species: s.name, item, ability: 'No Ability', gender: '',
			moves: picked, nature: '', evs: { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 },
			ivs, level: 50 + rng.random(2), happiness: 255,
		});
	}
	return team;
}

// ------------------------------------------------------- essence extraction
function scal(state) {
	const out = {};
	for (const k in state) {
		if (k === 'effectOrder') continue;
		const v = state[k];
		const t = typeof v;
		if (t === 'number' || t === 'string' || t === 'boolean') out[k] = v;
	}
	return out;
}
const mapScal = states => Object.fromEntries(Object.entries(states).map(([id, st]) => [id, scal(st)]));

function essence(battle) {
	return {
		turn: battle.turn,
		prngSeed: battle.prng.getSeed(),
		requestState: battle.requestState,
		field: {
			weather: battle.field.weather,
			weatherState: scal(battle.field.weatherState),
			pseudoWeather: mapScal(battle.field.pseudoWeather),
		},
		sides: battle.sides.map(side => ({
			pokemonLeft: side.pokemonLeft,
			sideConditions: mapScal(side.sideConditions),
			active: side.active.map(p => (p ? p.fullname : null)),
			pokemon: side.pokemon.map(p => ({
				ident: p.fullname, species: p.species.id,
				hp: p.hp, maxhp: p.maxhp, fainted: p.fainted, status: p.status,
				statusState: scal(p.statusState),
				boosts: { ...p.boosts },
				item: p.item, lastItem: p.lastItem, itemState: scal(p.itemState),
				moves: p.moveSlots.map(m => ({ id: m.id, pp: m.pp, disabled: !!m.disabled })),
				volatiles: mapScal(p.volatiles),
				types: p.types, transformed: p.transformed,
				active: p.isActive, position: p.position,
			})),
		})),
	};
}

// --------------------------------------------------------- phase 1: play
class RandomPickAI extends RandomPlayerAI {
	chooseTeamPreview(team) {
		// Random 3-of-6 pick (NC2000: Picked Team Size = 3).
		const order = [1, 2, 3, 4, 5, 6];
		this.prng.shuffle(order);
		return `team ${order.slice(0, 3).join('')}`;
	}
}

async function playBattle(battleSeed, p1team, p2team, aiSeedBase) {
	const battleStream = new BattleStream();
	const streams = getPlayerStreams(battleStream);
	const p1 = new RandomPickAI(streams.p1, { seed: `gen5,${aiSeedBase}0001` });
	const p2 = new RandomPickAI(streams.p2, { seed: `gen5,${aiSeedBase}0002` });
	void p1.start();
	void p2.start();
	const done = (async () => { for await (const _ of streams.omniscient) void _; })();
	await streams.omniscient.write(
		`>start ${JSON.stringify({ formatid: FORMAT, seed: battleSeed })}\n` +
		`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}\n` +
		`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`
	);
	// Guard against pathological stall wars.
	const guard = setInterval(() => {
		if (battleStream.battle && battleStream.battle.turn > 400) void battleStream.write('>forcetie');
	}, 2000);
	await done;
	clearInterval(guard);
	return battleStream.battle;
}

// --------------------------------------------------------- phase 2: replay
function replayAndExtract(inputLog) {
	// Reconstruct the battle from the inputLog verbatim (packed teams included)
	// so live and replay are identical by construction.
	const startLine = inputLog.find(l => l.startsWith('>start '));
	const { formatid, seed } = JSON.parse(startLine.slice(7));
	const battle = new Battle({ formatid, seed, strictChoices: true });
	for (const line of inputLog) {
		const m = /^>player (p[12]) (.*)$/.exec(line);
		if (m) battle.setPlayer(m[1], JSON.parse(m[2]));
	}

	// |t:| carries wall-clock time — the only nondeterministic log line; drop it.
	const cleanLog = lines => lines.filter(l => !l.startsWith('|t:|'));
	const snapshots = [{ afterLine: -1, ...essence(battle) }];
	const logs = [cleanLog(battle.log)];
	let logPos = battle.log.length;

	const choiceLines = [];
	inputLog.forEach((line, i) => {
		if (line.startsWith('>player ')) return;
		const m = /^>(p[12]) (.*)$/.exec(line);
		if (!m) return;
		choiceLines.push({ index: choiceLines.length, side: m[1], choice: m[2] });
		if (m[2] === 'undo') battle.undoChoice(m[1]);
		else battle.choose(m[1], m[2]);
		if (battle.log.length > logPos) {
			snapshots.push({ afterLine: choiceLines.length - 1, ...essence(battle) });
			logs.push(cleanLog(battle.log.slice(logPos)));
			logPos = battle.log.length;
		}
	});
	return { battle, choiceLines, snapshots, logs };
}

// ------------------------------------------------------------------- main
(async () => {
	fs.mkdirSync(OUT, { recursive: true });
	const master = new PRNG(`gen5,${MASTER_SEED.toString(16).padStart(16, '0')}`);
	let written = 0;
	for (let i = 0; i < N; i++) {
		const hex = () => master.random(0x10000).toString(16).padStart(4, '0');
		const battleSeed = `gen5,${hex()}${hex()}${hex()}${hex()}`;
		const aiSeedBase = `${hex()}${hex()}${hex()}`;
		const p1team = makeTeam(master);
		const p2team = makeTeam(master);

		const played = await playBattle(battleSeed, p1team, p2team, aiSeedBase);
		if (!played.ended) { console.error(`battle ${i}: did not end, skipping`); continue; }

		let replayed;
		try {
			replayed = replayAndExtract(played.inputLog);
		} catch (err) {
			const dbg = path.join(OUT, `battle-${String(i).padStart(3, '0')}.DIVERGED.json`);
			fs.writeFileSync(dbg, JSON.stringify({ error: String(err), inputLog: played.inputLog, log: played.log }, null, 1));
			console.error(`battle ${i}: replay error, dumped ${dbg}`);
			continue;
		}
		const { battle, choiceLines, snapshots, logs } = replayed;
		if (battle.winner !== played.winner || !battle.ended) {
			throw new Error(`battle ${i}: replay diverged from live battle (winner ${battle.winner} vs ${played.winner})`);
		}
		// Canonical teams = what the battle actually saw (packed in the inputLog).
		const packed = {};
		for (const line of played.inputLog) {
			const m = /^>player (p[12]) (.*)$/.exec(line);
			if (m) packed[m[1]] = JSON.parse(m[2]).team;
		}
		const fixture = {
			meta: { format: FORMAT, mod: MOD, pool: POOL, index: i },
			seed: battleSeed,
			p1team: Teams.unpack(packed.p1), p2team: Teams.unpack(packed.p2),
			p1packed: packed.p1, p2packed: packed.p2,
			choices: choiceLines,
			snapshots: snapshots.map((s, j) => ({ ...s, log: logs[j] })),
			result: { winner: battle.winner || '', turns: battle.turn },
		};
		const file = path.join(OUT, `battle-${String(i).padStart(3, '0')}.json`);
		fs.writeFileSync(file, JSON.stringify(fixture));
		written++;
		if ((i + 1) % 10 === 0) console.log(`${i + 1}/${N}...`);
	}
	console.log(`wrote ${written}/${N} fixtures to ${OUT}`);
})().catch(err => { console.error(err); process.exit(1); });
