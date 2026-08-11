// M15b PS websocket client: plays the no-OHKO NC2000 ladder format on a Pokemon Showdown
// server over the real network protocol (SockJS raw-websocket endpoint
// /showdown/websocket). Battles are driven by the M15a protocol->state
// importer (wasm ProtocolSearcher: player-visible lines + request JSON in,
// PS-canonical choice strings out) — the transport layer added here is
// login (challstr), challenges (/utm + /challenge + /accept), rqid-guarded
// choices, |error| recovery, the battle timer, and reconnect/resume (on a
// socket drop the client reconnects, rejoins the battle room, and rebuilds
// the searcher from scratch off the server's replayed room log + the
// re-sent |request| — the importer is stateless-rebuildable, and the
// rebuild is PROVEN by comparing the rebuilt stateView bit-for-bit against
// the pre-drop one when the rqid matches).
//
// POLICY: botting on the main ladder (play.pokemonshowdown.com) requires
// permission from PS staff. This client is for the owner's LOCAL server
// (the clone at ~/pokemon-showdown:
// `node pokemon-showdown start --skip-build --no-security 8123`)
// and, by explicit config, any self-hosted server. No public server is a
// default target.
//
// Usage:
//   node tools/ps-client.js --server ws://127.0.0.1:8123 --name BOTNAME \
//     --team pool:0|pool:random|FILE.json [--challenge USER | --accept any|U1,U2] \
//     [--games N] [--iters 30000] [--seed 1] [--mode blind|open] \
//     [--opp-team-file FILE.json] [--random] [--timer] [--no-tables] \
//     [--decision-log FILE.jsonl] [--belief-prior FILE.json|auto] \
//     [--password PW] [--loginserver URL] [--format gen2nintendocup2000noohkostadium2strict] \
//     [--lobby lobby] [--pool FILE.json | --team-dir DIR] \
//     [--unknown-log FILE.jsonl] [--drop SPEC] [--quiet]
//
// --random turns the client into the second driver: choices are drawn
// uniformly from the request-legal set (level-cap-aware at team preview)
// instead of running the searcher — the simplest correct opponent for
// gate runs.
//
// --drop SPEC (verification hook): comma list, one entry per battle in
// order, each `PHASE:pre|post` — kill the socket at that decision point,
// before answering (pre) or right after (post). PHASE = `preview` (the
// team-preview request), `fs` (our first forced-switch request), `moveN`
// (our Nth move request). Untriggered entries are reported.
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const WebSocket = require('ws');
const { sim, FORMAT, MOD } = require('./ps');
const { Teams, TeamValidator, Dex } = sim;

const REPO = path.join(__dirname, '..');

// ------------------------------------------------------------------ args
const args = {};
for (let i = 2; i < process.argv.length; i++) {
	const a = process.argv[i];
	if (!a.startsWith('--')) continue;
	const key = a.slice(2);
	if (i + 1 < process.argv.length && !process.argv[i + 1].startsWith('--')) {
		args[key] = process.argv[++i];
	} else {
		args[key] = true;
	}
}

if (args.help || args.h) {
	console.log(`node tools/ps-client.js — nc2000 bot over a PS server websocket

  --server URL      ws:// or wss:// server URL, or host:port (required;
                    e.g. ws://127.0.0.1:8123). NOTE: botting on the main
                    ladder (play.pokemonshowdown.com) requires permission
                    from PS staff — point this at your own server.
  --name NAME       login name (required)
  --password PW     registered-account password (optional; without it the
                    client logs in as an unregistered guest, which needs
                    either a noguestsecurity server — start the local clone
                    with --no-security — or a reachable login server)
  --loginserver URL login API base for assertions (default
                    https://play.pokemonshowdown.com; only contacted when a
                    bare guest /trn is refused or --password is given)
  --format ID       format id (default ${FORMAT})
  --team SPEC       pool:IDX | pool:random | FILE.json (required)
  --pool FILE       opponent/team pool JSON (default data/meta-pool-v0/meta-pool.json)
  --team-dir DIR    rebuild the pool from the latest Showdown .txt teams in DIR
                    before each new battle; useful for teams-nc2000/
  --no-validate-pool
                    skip TeamValidator when building --team-dir pool
  --challenge USER  challenge USER repeatedly until --games are done
  --accept WHO      accept challenges: 'any' or comma list of names
  --games N         number of complete battles to play (default 1; 0 = unlimited)
  --mode M          blind (default; pool-prior belief) | open (pin the
                    opponent's true sets — needs --opp-team-file, only
                    meaningful where sheets are genuinely open)
  --opp-team-file F opponent sets JSON for --mode open
  --iters N         search iterations per decision (default 30000 — the
                    shipped Web budget; see the note by ITERS)
  --seed N          searcher / random-mode seed (default 1)
  --random          random driver mode (no searcher; uniform legal choice)
  --timer           turn the battle timer on in every game
  --lobby ROOM      join ROOM after login so the bot is visible (default lobby)
  --no-lobby        do not join a lobby room
  --no-tables       skip loading baked preview tables
  --belief-prior F  M18 community belief prior (a table in the
                    data/belief-prior-v0.sample.json shape), or 'auto' to
                    generate one at startup from --team-dir/--pool. Read once
                    at startup and handed to each game's searcher as JSON text;
                    it governs ONLY the hidden-team fallback imputation, so
                    it needs --mode blind. Without the flag the fallback
                    imputation is exactly today's. A malformed table warns
                    and degrades rather than failing the run
  --unknown-log F   JSONL log for opponent observations outside the pool
                    (default logs/opponent-observations.jsonl)
  --unknown-log-all log every completed battle, not only unknown observations
  --no-unknown-log  disable opponent observation logging
  --drop SPEC       verification hook: socket kills at chosen decision
                    points (see header comment)
  --decision-log F  append private (mode 0600) JSONL for regret replay:
                    request, incremental visible protocol, exact own team,
                    pinned opponent team in open mode, submitted action,
                    diagnostic state, root policy/config
  --quiet           per-game lines only`);
	process.exit(0);
}

const need = k => {
	if (!args[k] || args[k] === true) {
		console.error(`--${k} is required (see --help)`);
		process.exit(2);
	}
	return String(args[k]);
};
const SERVER_RAW = need('server');
const NAME = need('name');
const TEAMSPEC = need('team');
const FORMATID = String(args.format || FORMAT);
function resolveUserPath(p) {
	return path.isAbsolute(p) ? p : path.resolve(process.cwd(), p);
}

const PASSWORD = args.password && args.password !== true ? String(args.password) : '';
const LOGINSERVER = String(args.loginserver || 'https://play.pokemonshowdown.com').replace(/\/$/, '');
const CHALLENGE = args.challenge && args.challenge !== true ? String(args.challenge) : '';
const ACCEPT = args.accept && args.accept !== true ? String(args.accept) : '';
const GAMES = parseInt(args.games || '1', 10);
const MODE = String(args.mode || 'blind');
// Aligned to the shipped Web budget (M12b: open sheet, 30k + ponder) so that
// ladder/postmortem evidence is about the configuration that actually ships.
// It used to default to 10000, which was only ever a seed-stability FLOOR, not
// an operating point: 1000 iters left flat roots as a visit-count lottery
// (battle-3623 T6: argmax split 23/16/7/4 over 50 seeds; unanimous at 10000 —
// replay_postmortem_3623). That floor still holds; 30000 sits above it.
// Latency: the searcher is single-threaded wasm-in-node, so the .wslconfig core
// cap (processors=12 of 16, set for host responsiveness) does not touch per-move
// time. The host's processor-performance state does, and this box runs capped
// (~70-80% of base clock). The figures below were measured under that cap and
// already include it: blind:1000 was 367-395 ms avg / 603-653 ms max (M15b gates
// a+c), so 30k lands around 11-20 s/move, still ~8x inside PS's 150 s per turn.
const ITERS = parseInt(args.iters || '30000', 10);
const SEED = parseInt(args.seed || '1', 10);
const RANDOM = !!args.random;
const TIMER = !!args.timer;
const QUIET = !!args.quiet;
const LOBBY_ROOM = args['no-lobby'] ? '' : String(args.lobby || 'lobby');
const VALIDATE_POOL = !args['no-validate-pool'];
const TEAM_DIR = args['team-dir'] && args['team-dir'] !== true ? resolveUserPath(String(args['team-dir'])) : '';
const POOL_FILE = args.pool && args.pool !== true ? resolveUserPath(String(args.pool)) : path.join(REPO, 'data/meta-pool-v0/meta-pool.json');
const UNKNOWN_LOG = args['no-unknown-log'] ? '' : resolveUserPath(String(args['unknown-log'] && args['unknown-log'] !== true ? args['unknown-log'] : path.join(REPO, 'logs/opponent-observations.jsonl')));
const UNKNOWN_LOG_ALL = !!args['unknown-log-all'];
const DECISION_LOG = args['decision-log'] && args['decision-log'] !== true ?
	path.resolve(String(args['decision-log'])) : '';
const BELIEF_PRIOR_ARG = args['belief-prior'] && args['belief-prior'] !== true ? String(args['belief-prior']) : '';
const BELIEF_PRIOR_AUTO = BELIEF_PRIOR_ARG.toLowerCase() === 'auto';
const BELIEF_PRIOR_FILE = BELIEF_PRIOR_ARG && !BELIEF_PRIOR_AUTO ? resolveUserPath(BELIEF_PRIOR_ARG) : '';
const DEBOUNCE_MS = 100; // network analogue of the M15a stream-quiescence wait
const RECONNECT_MS = 500;

if (DECISION_LOG) {
	fs.mkdirSync(path.dirname(DECISION_LOG), { recursive: true });
	if (!fs.existsSync(DECISION_LOG)) fs.closeSync(fs.openSync(DECISION_LOG, 'a', 0o600));
	fs.chmodSync(DECISION_LOG, 0o600);
}

function appendDecision(row) {
	if (!DECISION_LOG) return;
	fs.appendFileSync(DECISION_LOG, `${JSON.stringify(row)}\n`, { mode: 0o600 });
}

function safeServer(raw) {
	try {
		const url = new URL(/^wss?:\/\//.test(raw) ? raw : `ws://${raw}`);
		url.username = '';
		url.password = '';
		url.search = '';
		url.hash = '';
		return url.toString();
	} catch {
		return 'invalid-server-url';
	}
}

const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');

const wsUrl = (() => {
	let u = SERVER_RAW;
	if (!/^wss?:\/\//.test(u)) u = `ws://${u}`;
	if (!/\/showdown\/websocket$/.test(u)) u = u.replace(/\/$/, '') + '/showdown/websocket';
	return u;
})();

// ------------------------------------------------------------------ teams
function listTxtFilesRecursive(dir) {
	const out = [];
	for (const name of fs.readdirSync(dir).sort()) {
		const full = path.join(dir, name);
		const st = fs.statSync(full);
		if (st.isDirectory()) out.push(...listTxtFilesRecursive(full));
		else if (st.isFile() && name.toLowerCase().endsWith('.txt')) out.push(full);
	}
	return out;
}

function normalizeImportedTeam(team, label) {
	if (!team || team.length !== 6) throw new Error(`${label}: imported ${team ? team.length : 0} sets, want 6`);
	const dexMod = Dex.mod(MOD);
	for (const set of team) {
		set.ability = 'No Ability';
		if (!set.evs) set.evs = { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 };
		for (const stat of ['hp', 'atk', 'def', 'spa', 'spd', 'spe']) {
			if (set.evs[stat] === undefined) set.evs[stat] = 255;
		}
		if (set.happiness === undefined) {
			set.happiness = set.moves && set.moves.some(mv => dexMod.moves.get(mv).id === 'frustration') ? 0 : 255;
		}
	}
	return team;
}

function teamToPoolEntry(team, id, sourceFile) {
	const packed = Teams.pack(team);
	const sets = Teams.unpack(packed);
	return {
		id,
		tier: 'local',
		provenance: { source: 'local team directory', file: sourceFile },
		species: sets.map(set => set.species || set.name).filter(Boolean),
		levels: sets.map(set => set.level || 100),
		pedigree: { tournamentPoints: 0, vrMean: 0, hc75UsageMean: 0 },
		export: Teams.export(team),
		packed,
		sets,
	};
}

function makeTeamId(file, root, i) {
	const rel = path.relative(root, file).replace(/\\/g, '/').replace(/\.txt$/i, '');
	const base = rel.replace(/[^A-Za-z0-9_.\/-]+/g, '_').replace(/[\/]+/g, '__');
	return base || `team-${i + 1}`;
}

function buildPoolFromTeamDir(dir) {
	const files = listTxtFilesRecursive(dir);
	if (!files.length) throw new Error(`no .txt team files found in --team-dir ${dir}`);
	const validator = VALIDATE_POOL ? new TeamValidator(FORMATID) : null;
	const teams = [];
	const failures = [];
	for (let i = 0; i < files.length; i++) {
		const file = files[i];
		const label = path.relative(dir, file) || file;
		try {
			const imported = normalizeImportedTeam(Teams.import(fs.readFileSync(file, 'utf8')), label);
			if (validator) {
				const errors = validator.validateTeam(imported);
				if (errors) throw new Error(errors.join('; '));
			}
			teams.push(teamToPoolEntry(imported, makeTeamId(file, dir, i), file));
		} catch (err) {
			failures.push(`${label}: ${err && err.message || err}`);
		}
	}
	if (!teams.length) {
		throw new Error(`all teams failed while building --team-dir pool ${dir}: ${failures.slice(0, 5).join(' | ')}`);
	}
	if (failures.length) {
		console.error(`[${NAME}] WARN_POOL_TEAM_FAILURES ${failures.length}/${files.length}; first=${failures.slice(0, 3).join(' | ')}`);
	}
	teams.forEach((team, i) => (team.rank = i + 1));
	return {
		meta: {
			format: FORMATID,
			mod: MOD,
			generated: new Date().toISOString(),
			teams: teams.length,
			source: 'team-dir',
			dir,
			files: files.length,
			failures: failures.length,
		},
		teams,
	};
}

function loadPoolSnapshot() {
	const pool = TEAM_DIR
		? buildPoolFromTeamDir(TEAM_DIR)
		: JSON.parse(fs.readFileSync(POOL_FILE, 'utf8'));
	if (!pool || !Array.isArray(pool.teams) || !pool.teams.length) {
		throw new Error(`pool has no teams (${TEAM_DIR ? TEAM_DIR : POOL_FILE})`);
	}
	return {
		pool,
		poolJson: JSON.stringify(pool),
		label: TEAM_DIR ? `team-dir:${TEAM_DIR}` : `pool:${POOL_FILE}`,
	};
}

function prob(n, d) {
	return Number((n / d).toFixed(6));
}

function buildBeliefPriorFromPool(pool, label) {
	const speciesCounts = new Map();
	for (const team of pool && pool.teams || []) {
		for (const set of team.sets || []) {
			const sid = toID(set.species || set.name || '');
			if (!sid) continue;
			let entry = speciesCounts.get(sid);
			if (!entry) {
				entry = { n: 0, moves: new Map(), items: new Map() };
				speciesCounts.set(sid, entry);
			}
			entry.n++;
			for (const mv of new Set(set.moves || [])) {
				const mid = toID(mv);
				if (mid) entry.moves.set(mid, (entry.moves.get(mid) || 0) + 1);
			}
			const iid = toID(set.item || '');
			if (iid) entry.items.set(iid, (entry.items.get(iid) || 0) + 1);
		}
	}
	const species = {};
	for (const sid of Array.from(speciesCounts.keys()).sort()) {
		const entry = speciesCounts.get(sid);
		const moves = {};
		for (const mid of Array.from(entry.moves.keys()).sort()) moves[mid] = prob(entry.moves.get(mid), entry.n);
		const items = {};
		for (const iid of Array.from(entry.items.keys()).sort()) items[iid] = prob(entry.items.get(iid), entry.n);
		species[sid] = { moves, items, n: entry.n };
	}
	return JSON.stringify({
		format: 'nc2000-belief-prior',
		version: 1,
		note: `auto-generated at ps-client startup from ${label}; per-species move/item carry marginals`,
		species,
	}, null, 2);
}

let rngState = (SEED ^ 0x9e3779b9) >>> 0;
const rng = () => { // mulberry32 (random-mode choices; crypto.randomInt is used for pool:random picks)
	rngState = (rngState + 0x6d2b79f5) >>> 0;
	let t = rngState;
	t = Math.imul(t ^ (t >>> 15), t | 1);
	t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
	return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const rngInt = n => Math.floor(rng() * n);

function pickTeam() {
	const snapshot = loadPoolSnapshot();
	if (TEAMSPEC.startsWith('pool:')) {
		const which = TEAMSPEC.slice(5);
		const idx = which === 'random' ? crypto.randomInt(snapshot.pool.teams.length) : parseInt(which, 10);
		if (!(idx >= 0 && idx < snapshot.pool.teams.length)) throw new Error(`bad pool index ${which}`);
		return {
			sets: snapshot.pool.teams[idx].sets,
			label: `${snapshot.label}:pool:${idx}`,
			pool: snapshot.pool,
			poolJson: snapshot.poolJson,
			poolLabel: snapshot.label,
		};
	}
	const raw = JSON.parse(fs.readFileSync(TEAMSPEC, 'utf8'));
	const sets = Array.isArray(raw) ? raw : raw.sets;
	if (!Array.isArray(sets)) throw new Error(`${TEAMSPEC}: expected a JSON array of sets (or {sets:[...]})`);
	return { sets, label: TEAMSPEC, pool: snapshot.pool, poolJson: snapshot.poolJson, poolLabel: snapshot.label };
}

// ------------------------------------------------------------------ wasm
let wasm = null;
let dex = null;
let pairJsons = [];
if (!RANDOM) {
	wasm = require(path.join(REPO, 'crates/wasm/pkg-node/nc2000_wasm.js'));
	dex = new wasm.Dex();
	if (!args['no-tables']) {
		const pairDir = path.join(REPO, 'data/preview-tables-v0');
		if (fs.existsSync(pairDir)) {
			for (const f of fs.readdirSync(pairDir).sort()) {
				if (f.startsWith('pair-') && f.endsWith('.json')) {
					try {
						pairJsons.push(fs.readFileSync(path.join(pairDir, f), 'utf8'));
					} catch { /* mid-write during a bake: treat as missing */ }
				}
			}
		}
	}
}
let oppTeamJson = '';
let oppTeamSets = null;
if (MODE === 'open') {
	const f = args['opp-team-file'];
	if (!f || f === true) {
		console.error('--mode open needs --opp-team-file (the opponent\'s true sets)');
		process.exit(2);
	}
	const raw = JSON.parse(fs.readFileSync(String(f), 'utf8'));
	oppTeamSets = Array.isArray(raw) ? raw : raw.sets;
	if (!Array.isArray(oppTeamSets) || !oppTeamSets.length) {
		console.error('--opp-team-file must contain a non-empty sets array');
		process.exit(2);
	}
	oppTeamJson = JSON.stringify(oppTeamSets);
}

// ------------------------------------------------------- M18 belief prior
// wasm has no filesystem, so the table crosses the boundary as JSON TEXT and
// the reading happens here, once. The interpreter is total: a malformed table
// degrades into warnings and play continues exactly as it does with no prior
// at all — so the only hard failure is a file we cannot read, which is a typo
// in the flag rather than a statement about the table.
function reportPrior(reportJson, log) {
	let r = null;
	try {
		r = JSON.parse(reportJson);
	} catch { /* fall through to the unreadable-report line */ }
	if (!r) {
		log('belief prior: setBeliefPrior returned no readable report');
		return;
	}
	for (const w of r.warnings || []) log(`belief prior: ${w}`);
	log(`belief prior: ${r.species} species, mean move-probability sum ` +
		`${Number(r.meanMoveSum).toFixed(2)}, ${r.skipped} entries skipped — ` +
		`${r.applied ? 'applied' : 'NOT applied'}`);
}

let priorText = '';
let priorSource = '';
if (BELIEF_PRIOR_ARG) {
	if (RANDOM) {
		console.error('--belief-prior has nothing to act on in --random mode (no searcher)');
		process.exit(2);
	}
	if (MODE !== 'blind') {
		console.error(`--belief-prior needs --mode blind: --mode ${MODE} pins the opponent's ` +
			'true sets, and the prior must never reach the open-sheet path');
		process.exit(2);
	}
	let probeSnapshot;
	try {
		probeSnapshot = loadPoolSnapshot();
	} catch (e) {
		console.error(`--belief-prior: cannot load pool for probe/auto prior: ${e.message}`);
		process.exit(2);
	}
	if (BELIEF_PRIOR_AUTO) {
		priorText = buildBeliefPriorFromPool(probeSnapshot.pool, probeSnapshot.label);
		priorSource = `auto:${probeSnapshot.label}`;
	} else {
		try {
			priorText = fs.readFileSync(BELIEF_PRIOR_FILE, 'utf8');
			priorSource = BELIEF_PRIOR_FILE;
		} catch (e) {
			console.error(`--belief-prior: cannot read ${BELIEF_PRIOR_FILE}: ${e.message}`);
			process.exit(2);
		}
	}
	// Probe once up front so a typo in the TABLE is visible before the first
	// challenge instead of one line per game; each game's searcher gets its
	// own install below (the prior lives on the searcher, one per battle).
	const probe = new wasm.ProtocolSearcher(dex, 0, probeSnapshot.poolJson, 0);
	reportPrior(probe.setBeliefPrior(priorText), m => console.log(`[${NAME}] ${m}`));
	console.log(`[${NAME}] belief prior source: ${priorSource}`);
	if (typeof probe.free === 'function') probe.free();
}

// ------------------------------------------------------------- drop specs
// one entry per battle index; `null` = no drop for that battle
const dropSpecs = String(args.drop && args.drop !== true ? args.drop : '')
	.split(',')
	.filter(Boolean)
	.map(tok => {
		const m = /^(preview|fs|move(\d+)):(pre|post)$/.exec(tok.trim());
		if (!m) {
			console.error(`bad --drop token: ${tok}`);
			process.exit(2);
		}
		return { phase: m[2] ? 'move' : m[1], nth: m[2] ? parseInt(m[2], 10) : 1, when: m[3], triggered: false };
	});

// -------------------------------------------------------- random choices
function parseLevel(details) {
	const m = /, L(\d+)/.exec(details);
	return m ? parseInt(m[1], 10) : 100;
}

function randomChoice(req) {
	if (req.teamPreview) {
		const mons = req.side.pokemon;
		const size = req.maxChosenTeamSize || 3;
		const levels = mons.map(p => parseLevel(p.details));
		for (let tries = 0; tries < 200; tries++) {
			const order = mons.map((_, i) => i + 1);
			for (let i = order.length - 1; i > 0; i--) {
				const j = rngInt(i + 1);
				[order[i], order[j]] = [order[j], order[i]];
			}
			const pick = order.slice(0, size);
			if (pick.reduce((a, s) => a + levels[s - 1], 0) <= 155) return `team ${pick.join('')}`;
		}
		return `team ${mons.map((_, i) => i + 1).slice(0, size).join('')}`;
	}
	if (req.forceSwitch) {
		const mons = req.side.pokemon;
		const can = [];
		for (let i = 0; i < mons.length; i++) {
			if (!mons[i].active && !mons[i].condition.endsWith(' fnt')) can.push(i + 1);
		}
		return can.length ? `switch ${can[rngInt(can.length)]}` : 'default';
	}
	if (req.active) {
		const moves = req.active[0].moves;
		const can = [];
		for (let i = 0; i < moves.length; i++) {
			if (!moves[i].disabled) can.push(i + 1);
		}
		return can.length ? `move ${can[rngInt(can.length)]}` : 'move 1';
	}
	return 'default';
}

// lines that are room/global noise, not battle protocol (the importer
// ignores unknown line types anyway; this keeps its input at the M15a
// player-stream vocabulary)
const NOISE = new Set([
	'', 'init', 'title', 'j', 'J', 'l', 'L', 'n', 'N', 'join', 'leave', 'name',
	'c', 'c:', 'chat', ':', 'raw', 'html', 'uhtml', 'uhtmlchange', 'inactive',
	'inactiveoff', 'bigerror', 'debug', 'seed', 'askreg', 'deinit', 'expire',
	'pm', 'usercount', 'formats', 'updatesearch', 'updatechallenges',
	'updateuser', 'queryresponse', 'popup', 'nametaken', 'challstr', 'rated',
	'notify', 'tempnotify', 'tempnotifyoff', 'hidelines', 'unlink', 'b', 'battle',
]);

function ensureParentDir(file) {
	if (!file) return;
	fs.mkdirSync(path.dirname(file), { recursive: true });
}

function appendJsonl(file, obj) {
	if (!file) return;
	ensureParentDir(file);
	fs.appendFileSync(file, JSON.stringify(obj) + '\n');
}

function identSide(ident) {
	const m = /^p([12])/.exec(String(ident || ''));
	return m ? `p${m[1]}` : '';
}

function identSlot(ident) {
	const m = /^(p[12][a-z]?):/.exec(String(ident || ''));
	return m ? m[1] : '';
}

function speciesFromIdent(ident) {
	const s = String(ident || '');
	const i = s.indexOf(':');
	return i >= 0 ? s.slice(i + 1).trim() : '';
}

function parseDetails(details) {
	const raw = String(details || '');
	const parts = raw.split(',').map(x => x.trim()).filter(Boolean);
	const species = parts[0] || '';
	let level = 100;
	let gender = '';
	for (const part of parts.slice(1)) {
		const lm = /^L(\d+)$/.exec(part);
		if (lm) level = parseInt(lm[1], 10);
		else if (part === 'M' || part === 'F' || part === 'N') gender = part;
	}
	return { species, level, gender, details: raw };
}

function poolIndex(pool) {
	const bySpecies = new Map();
	const teamKeys = new Set();
	const teamIdsByKey = new Map();
	for (const team of pool && pool.teams || []) {
		const keyParts = [];
		for (const set of team.sets || []) {
			const species = set.species || set.name || '';
			const sid = toID(species);
			if (!sid) continue;
			let entry = bySpecies.get(sid);
			if (!entry) {
				entry = { species, moves: new Map(), items: new Map(), setCount: 0 };
				bySpecies.set(sid, entry);
			}
			entry.setCount++;
			for (const mv of set.moves || []) entry.moves.set(toID(mv), mv);
			if (set.item) entry.items.set(toID(set.item), set.item);
			keyParts.push(`${sid}:L${set.level || 100}`);
		}
		const key = keyParts.sort().join('|');
		teamKeys.add(key);
		if (!teamIdsByKey.has(key)) teamIdsByKey.set(key, []);
		teamIdsByKey.get(key).push(team.id || String(team.rank || ''));
	}
	return { bySpecies, teamKeys, teamIdsByKey };
}

function previewKey(preview) {
	return preview.map(p => `${toID(p.species)}:L${p.level || 100}`).sort().join('|');
}


// ----------------------------------------------------------------- stats
const stats = {
	games: 0, W: 0, L: 0, T: 0, turns: 0, decisions: 0,
	rejections: [], desyncs: 0, drops: 0, resumes: 0,
	proofsOk: 0, proofsBad: [], proofsSkipped: 0,
	maxThinkMs: 0, thinkMsSum: 0, thinkN: 0,
	legalityDrift: 0, projections: 0, reconnects: 0,
	untriggeredDrops: 0,
};

// ---------------------------------------------------------- battle driver
class BattleDriver {
	constructor(client, room, battleIdx) {
		this.client = client;
		this.room = room;
		this.battleIdx = battleIdx;
		this.searcher = null;
		this.side = -1;
		this.lineBuffer = [];
		this.visibleLines = [];
		this.loggedLineCount = 0;
		this.protocolReset = false;
		this.loggedRqids = new Set();
		this.pendingReq = null;
		this.lastReq = null; // last non-wait request seen (|error| recovery)
		this.errRecoveries = 0;
		this.sentchoice = null;
		this.sawPreviewLine = false;
		this.actTimer = null;
		this.retries = 0;
		this.ended = false;
		this.turn = 0;
		this.decisions = 0;
		this.moveReqs = 0;
		this.result = '';
		this.timerSent = false;
		this.initialized = false;
		this.awaitingReplay = false; // set when we /join after a reconnect
		this.drop = dropSpecs[battleIdx] || null;
		this.preDropView = null; // { rqid, view } for the resume proof
		this.players = {};
		this.selfSide = '';
		this.oppSide = '';
		this.opponentName = '';
		this.activeSpeciesBySlot = new Map();
		this.oppPreview = [];
		this.poolIdx = null;
		this.poolLabel = '';
		this.unknownTeam = false;
		this.unknownTeamChecked = false;
		this.unknownMoves = [];
		this.unknownItems = [];
		this.unknownSpecies = [];
		this.revealedMoves = new Map();
		this.revealedItems = new Map();
		this.log = m => console.log(`[${this.room}] ${m}`);
	}

	freeSearcher() {
		if (this.searcher) {
			try {
				const m = JSON.parse(this.searcher.metrics());
				stats.legalityDrift += m.legalityDrift;
				stats.projections += m.projections;
			} catch { /* ignore */ }
			this.searcher.free();
			this.searcher = null;
		}
	}

	// a replayed |init|battle after /join: rebuild from scratch (stats kept)
	resetForRejoin() {
		this.freeSearcher();
		this.lineBuffer = [];
		this.visibleLines = [];
		this.loggedLineCount = 0;
		this.protocolReset = true;
		this.pendingReq = null;
		this.lastReq = null;
		this.errRecoveries = 0;
		this.sentchoice = null;
		this.sawPreviewLine = false;
		this.retries = 0;
		if (this.actTimer) clearTimeout(this.actTimer);
		this.actTimer = null;
		this.awaitingReplay = false;
		stats.resumes++;
		this.log(`rejoined; rebuilding from the replayed room log`);
	}


	setSelfSide(side) {
		if (!side || (side !== 'p1' && side !== 'p2')) return;
		this.selfSide = side;
		this.oppSide = side === 'p1' ? 'p2' : 'p1';
		this.opponentName = this.players[this.oppSide] || this.opponentName;
	}

	getPoolIndex() {
		if (this.poolIdx) return this.poolIdx;
		const source = this.client.currentTeam || {};
		let pool = source.pool;
		this.poolLabel = source.poolLabel || '';
		if (!pool) {
			try {
				const snap = loadPoolSnapshot();
				pool = snap.pool;
				this.poolLabel = snap.label;
			} catch { /* unknown logging can proceed without pool */ }
		}
		this.poolIdx = poolIndex(pool || { teams: [] });
		return this.poolIdx;
	}

	noteSpecies(species, line) {
		const sid = toID(species);
		if (!sid) return;
		const idx = this.getPoolIndex();
		if (!idx.bySpecies.has(sid) && !this.unknownSpecies.some(x => toID(x.species) === sid)) {
			this.unknownSpecies.push({ species, line });
		}
	}

	noteMove(species, move, line) {
		const sid = toID(species);
		const mid = toID(move);
		if (!sid || !mid || mid === 'struggle') return;
		this.noteSpecies(species, line);
		if (!this.revealedMoves.has(sid)) this.revealedMoves.set(sid, new Map());
		this.revealedMoves.get(sid).set(mid, move);
		const idx = this.getPoolIndex();
		const known = idx.bySpecies.get(sid);
		if ((!known || !known.moves.has(mid)) && !this.unknownMoves.some(x => toID(x.species) === sid && toID(x.move) === mid)) {
			this.unknownMoves.push({ species, move, line });
		}
	}

	noteItem(species, item, line) {
		const sid = toID(species);
		const iid = toID(item);
		if (!sid || !iid) return;
		this.noteSpecies(species, line);
		if (!this.revealedItems.has(sid)) this.revealedItems.set(sid, new Map());
		this.revealedItems.get(sid).set(iid, item);
		const idx = this.getPoolIndex();
		const known = idx.bySpecies.get(sid);
		if ((!known || !known.items.has(iid)) && !this.unknownItems.some(x => toID(x.species) === sid && toID(x.item) === iid)) {
			this.unknownItems.push({ species, item, line });
		}
	}

	speciesForIdent(ident, details) {
		const slot = identSlot(ident);
		if (details) return parseDetails(details).species;
		if (slot && this.activeSpeciesBySlot.has(slot)) return this.activeSpeciesBySlot.get(slot);
		return speciesFromIdent(ident);
	}

	maybeCheckUnknownTeam() {
		if (this.unknownTeamChecked) return;
		if (!this.oppSide || this.oppPreview.length < 6) return;
		this.unknownTeamChecked = true;
		const key = previewKey(this.oppPreview);
		const idx = this.getPoolIndex();
		this.unknownTeam = !idx.teamKeys.has(key);
	}

	observeOpponentLine(line) {
		if (!UNKNOWN_LOG && !UNKNOWN_LOG_ALL) return;
		if (!line.startsWith('|')) return;
		const parts = line.split('|');
		const cmd = parts[1] || '';
		if (cmd === 'player') {
			const side = parts[2] || '';
			const name = (parts[3] || '').trim();
			if (side === 'p1' || side === 'p2') this.players[side] = name;
			if (toID(name) === toID(NAME)) this.setSelfSide(side);
			else if (this.selfSide && side === this.oppSide) this.opponentName = name;
			return;
		}
		if (cmd === 'poke') {
			const side = parts[2] || '';
			const info = parseDetails(parts[3] || '');
			if (side && side === this.oppSide && info.species) {
				this.oppPreview.push({ species: info.species, level: info.level, gender: info.gender, details: info.details, hasItem: !!parts[4] });
				this.noteSpecies(info.species, line);
			}
			return;
		}
		if (cmd === 'teampreview') {
			this.maybeCheckUnknownTeam();
			return;
		}
		if (cmd === 'switch' || cmd === 'drag' || cmd === 'replace') {
			const ident = parts[2] || '';
			const side = identSide(ident);
			const info = parseDetails(parts[3] || '');
			const slot = identSlot(ident);
			if (slot && info.species) this.activeSpeciesBySlot.set(slot, info.species);
			if (side === this.oppSide && info.species) this.noteSpecies(info.species, line);
			return;
		}
		if (cmd === 'move') {
			const ident = parts[2] || '';
			if (identSide(ident) !== this.oppSide) return;
			const move = parts[3] || '';
			if (parts.slice(4).some(p => /\[from\]\s*(move:\s*)?(metronome|mirror move|mimic)/i.test(p))) return;
			const species = this.speciesForIdent(ident);
			this.noteMove(species, move, line);
			return;
		}
		if (cmd === '-enditem' || cmd === '-item') {
			const ident = parts[2] || '';
			if (identSide(ident) !== this.oppSide) return;
			this.noteItem(this.speciesForIdent(ident), parts[3] || '', line);
			return;
		}
		for (const token of parts.slice(3)) {
			const m = /\[from\]\s*item:\s*(.+)$/i.exec(token);
			if (!m) continue;
			const ident = parts[2] || '';
			if (identSide(ident) === this.oppSide) this.noteItem(this.speciesForIdent(ident), m[1], line);
		}
	}

	writeUnknownObservation() {
		if (!UNKNOWN_LOG) return;
		this.maybeCheckUnknownTeam();
		const hasUnknown = this.unknownTeam || this.unknownMoves.length || this.unknownItems.length || this.unknownSpecies.length;
		if (!hasUnknown && !UNKNOWN_LOG_ALL) return;
		const mapToObject = map => {
			const obj = {};
			for (const [sid, inner] of map) obj[sid] = Array.from(inner.values()).sort();
			return obj;
		};
		appendJsonl(UNKNOWN_LOG, {
			timestamp: new Date().toISOString(),
			room: this.room,
			format: FORMATID,
			bot: NAME,
			opponent: this.opponentName,
			pool: this.poolLabel,
			poolTeams: this.client.currentTeam && this.client.currentTeam.pool ? this.client.currentTeam.pool.teams.length : undefined,
			unknownTeam: this.unknownTeam,
			preview: this.oppPreview,
			revealedMoves: mapToObject(this.revealedMoves),
			revealedItems: mapToObject(this.revealedItems),
			unknownSpecies: this.unknownSpecies,
			unknownMoves: this.unknownMoves,
			unknownItems: this.unknownItems,
			result: this.result || 'T',
			turns: this.turn,
		});
	}

	onFrame(lines) {
		if (this.ended) return;
		if (lines[0] === '|init|battle' && this.initialized) this.resetForRejoin();
		this.initialized = true;
		for (const line of lines) this.onLine(line);
		if (TIMER && !this.timerSent && !RANDOM) {
			this.timerSent = true;
			this.client.send(`${this.room}|/timer on`);
		}
		this.scheduleAct();
	}

	onLine(line) {
		if (!line.startsWith('|')) return;
		this.observeOpponentLine(line);
		const cmd = line.split('|')[2] !== undefined ? line.split('|')[1] : line.slice(1);
		if (cmd === 'request') {
			const j = line.slice('|request|'.length);
			if (j && j !== 'null') {
				this.pendingReq = j;
				this.lastReq = j;
				this.errRecoveries = 0;
				this.sentchoice = null;
				this.retries = 0;
			}
			return;
		}
		if (cmd === 'sentchoice') {
			this.sentchoice = line.slice('|sentchoice|'.length);
			return;
		}
		if (cmd === 'error') {
			this.onError(line);
			return;
		}
		if (cmd === 'inactive' && /timer is (now )?ON/i.test(line)) {
			this.log(`timer confirmed: ${line.slice('|inactive|'.length)}`);
		}
		if (cmd === 'turn') this.turn = parseInt(line.split('|')[2], 10) || this.turn;
		if (cmd === 'teampreview') this.sawPreviewLine = true;
		if (cmd === 'win' || cmd === 'tie') {
			const winner = cmd === 'win' ? line.split('|')[2] : '';
			this.result = cmd === 'tie' ? 'T' : toID(winner) === toID(NAME) ? 'W' : 'L';
		}
		if (NOISE.has(cmd)) return;
		this.lineBuffer.push(line);
		this.visibleLines.push(line);
		if (cmd === 'win' || cmd === 'tie') this.finalize();
	}

	onError(line) {
		// every |error| in a battle room is a choice rejection — target 0.
		stats.rejections.push(`${this.room} d${this.decisions}: ${line}`);
		this.log(`REJECTED: ${line}`);
		// recovery: after [Unavailable choice] PS re-sends an updated
		// |request| and the normal pendingReq flow re-chooses. Any other
		// rejection leaves the request open with no re-send — re-pose the
		// last request ourselves (the searcher re-syncs and re-chooses);
		// if that keeps getting rejected, fall back to the always-legal
		// `default` choice rather than stall out the battle.
		if (line.includes('[Unavailable choice]')) return;
		if (!this.pendingReq && this.lastReq) {
			this.errRecoveries++;
			if (this.errRecoveries > 2) {
				let rqid;
				try {
					rqid = JSON.parse(this.lastReq).rqid;
				} catch { /* leave undefined */ }
				this.client.send(`${this.room}|/choose default${rqid !== undefined ? `|${rqid}` : ''}`);
				return;
			}
			this.pendingReq = this.lastReq;
			this.scheduleAct();
		}
	}

	scheduleAct() {
		if (this.ended || !this.pendingReq) return;
		if (this.actTimer) clearTimeout(this.actTimer);
		this.actTimer = setTimeout(() => this.act(), DEBOUNCE_MS);
	}

	holdForMoreLines(why) {
		this.retries++;
		if (this.retries > 100) { // ~10s: the update never arrived
			stats.desyncs++;
			this.log(`DESYNC: gave up waiting (${why})`);
			this.pendingReq = null;
			return;
		}
		if (this.actTimer) clearTimeout(this.actTimer);
		this.actTimer = setTimeout(() => this.act(), DEBOUNCE_MS);
	}

	act() {
		if (this.ended || !this.pendingReq) return;
		const reqStr = this.pendingReq;
		let req;
		try {
			req = JSON.parse(reqStr);
		} catch {
			this.pendingReq = null;
			return;
		}
		const rqid = req.rqid;

		// |request| reaches the socket before the update lines that led to it
		// (same ordering as the M15a sim stream); the debounce absorbs that,
		// but the team-preview request additionally needs the |poke| lines.
		if (!RANDOM && req.teamPreview && !this.sawPreviewLine) {
			return this.holdForMoreLines('teampreview request before |poke| lines');
		}

		if (RANDOM) {
			this.pendingReq = null;
			if (req.wait) return;
			if (this.sentchoice) return; // rejoin replay: already answered
			this.decisions++;
			stats.decisions++;
			if (this.maybeDrop(req, 'pre')) return;
			const choice = randomChoice(req);
			this.client.send(`${this.room}|/choose ${choice}${rqid !== undefined ? `|${rqid}` : ''}`);
			this.recordDecision(req, choice, null, null, null, 'random');
			this.maybeDrop(req, 'post');
			return;
		}

		if (!this.searcher) {
			this.side = req.side && req.side.id === 'p2' ? 1 : 0;
			const activePoolJson = this.client.currentTeam && this.client.currentTeam.poolJson ?
				this.client.currentTeam.poolJson : loadPoolSnapshot().poolJson;
			this.searcher = new wasm.ProtocolSearcher(dex, this.side, activePoolJson, SEED * 1000 + this.battleIdx);
			this.searcher.setOwnTeam(JSON.stringify(this.client.currentTeam.sets));
			if (MODE === 'open') this.searcher.pinOpponent(oppTeamJson);
			// after pinOpponent on purpose: if the two ever coexist the binding
			// refuses out loud rather than contaminating the open-sheet belief
			if (priorText) reportPrior(this.searcher.setBeliefPrior(priorText), m => this.log(m));
			for (const pj of pairJsons) {
				try {
					this.searcher.addPair(pj);
				} catch { /* stale table: fall back to live preview search */ }
			}
		}
		if (this.lineBuffer.length) {
			this.searcher.pushLines(JSON.stringify(this.lineBuffer));
			this.lineBuffer = [];
		}
		let owes;
		try {
			owes = this.searcher.onRequest(reqStr);
		} catch (e) {
			// synthesis needs lines that haven't arrived yet — keep waiting
			return this.holdForMoreLines(`onRequest: ${e.message || e}`);
		}
		this.pendingReq = null;
		this.retries = 0;

		// resume proof: same request as before the drop -> the rebuilt state
		// must be bit-identical to the pre-drop synthesis
		if (this.preDropView && owes) {
			if (this.preDropView.rqid === rqid) {
				const view = this.searcher.stateView();
				if (view === this.preDropView.view) {
					stats.proofsOk++;
					this.log(`resume proof: rebuilt stateView identical (rqid ${rqid})`);
				} else {
					stats.proofsBad.push(`${this.room} rqid ${rqid}`);
					this.log(`RESUME PROOF FAILED: rebuilt stateView differs (rqid ${rqid})`);
				}
			} else {
				stats.proofsSkipped++;
				this.log(`resume proof skipped: turn advanced during the drop (rqid ${this.preDropView.rqid} -> ${rqid})`);
			}
			this.preDropView = null;
		}
		if (!owes) return;
		if (this.sentchoice) {
			// rejoin replay of a request we had already answered: the server
			// kept our choice (|sentchoice|); a re-send would be refused
			this.log(`already answered rqid ${rqid} (sentchoice=${this.sentchoice}); not re-sending`);
			return;
		}
		this.decisions++;
		stats.decisions++;
		if (this.maybeDrop(req, 'pre')) return;

		const t0 = Date.now();
		let choice = this.searcher.bakedPreview();
		if (!choice) {
			const searchIters = req.teamPreview ? ITERS * 5 : ITERS;
			this.searcher.step(searchIters);
			choice = this.searcher.best();
		}
		const think = Date.now() - t0;
		stats.maxThinkMs = Math.max(stats.maxThinkMs, think);
		stats.thinkMsSum += think;
		stats.thinkN++;
		if (!choice) throw new Error('searcher returned no choice');
		this.client.send(`${this.room}|/choose ${choice}${rqid !== undefined ? `|${rqid}` : ''}`);
		let policy = null;
		try { policy = JSON.parse(this.searcher.rootPolicy()); } catch { /* diagnostic only */ }
		let state = null;
		try { state = JSON.parse(this.searcher.stateView()); } catch { /* diagnostic only */ }
		let beliefInfo = null;
		try { beliefInfo = JSON.parse(this.searcher.beliefInfo()); } catch { /* diagnostic only */ }
		this.recordDecision(req, choice, policy, state, beliefInfo, 'search');
		this.maybeDrop(req, 'post');
	}

	recordDecision(req, choice, rootPolicy, stateView, beliefInfo, driver) {
		if (!DECISION_LOG) return;
		const rqidKey = String(req.rqid ?? `decision-${this.decisions}`);
		if (this.loggedRqids.has(rqidKey)) return;
		this.loggedRqids.add(rqidKey);
		const protocolDelta = this.visibleLines.slice(this.loggedLineCount);
		this.loggedLineCount = this.visibleLines.length;
		appendDecision({
			version: 3,
			type: 'decision',
			room: this.room,
			battle: this.battleIdx,
			decision: this.decisions,
			rqid: req.rqid,
			side: this.side >= 0 ? this.side : (req.side && req.side.id === 'p2' ? 1 : 0),
			turn: this.turn,
			server: safeServer(SERVER_RAW),
			format: FORMATID,
			mode: MODE,
			driver,
			iterations: driver === 'search' ? ITERS : 0,
			seed: SEED * 1000 + this.battleIdx,
			teamLabel: this.client.currentTeam.label.startsWith('pool:') ?
				this.client.currentTeam.label : path.basename(this.client.currentTeam.label),
			ownTeam: this.client.currentTeam.sets,
			opponentTeam: MODE === 'open' ? oppTeamSets : null,
			request: req,
			protocolReset: this.protocolReset,
			protocolDelta,
			submitted: choice,
			rootPolicy,
			beliefInfo,
			stateViewKind: 'diagnostic-imputed',
			stateView,
		});
		this.protocolReset = false;
	}

	maybeDrop(req, when) {
		const d = this.drop;
		if (!d || d.triggered || d.when !== when) return false;
		const phase = req.teamPreview ? 'preview' : req.forceSwitch ? 'fs' : 'move';
		if (phase !== d.phase) return false;
		if (d.phase === 'move' && ++this.moveReqs !== d.nth) return false;
		d.triggered = true;
		if (!RANDOM && this.searcher) {
			this.preDropView = { rqid: req.rqid, view: this.searcher.stateView() };
		}
		this.log(`CHAOS: dropping socket (${d.phase}:${d.when}, decision ${this.decisions}, rqid ${req.rqid})`);
		stats.drops++;
		if (when === 'pre') {
			// leave the request unanswered; the rejoin replay re-poses it
			this.pendingReq = null;
			this.client.chaosDrop();
			return true;
		}
		setTimeout(() => this.client.chaosDrop(), 30); // let the choice flush
		return false;
	}

	finalize() {
		if (this.ended) return;
		this.ended = true;
		if (this.actTimer) clearTimeout(this.actTimer);
		stats.games++;
		stats[this.result || 'T']++;
		stats.turns += this.turn;
		this.writeUnknownObservation();
		console.log(
			`game ${stats.games}/${hasGameLimit() ? GAMES : 'unlimited'}: ${this.result} in ${this.turn} turns, ` +
			`${this.decisions} decisions (${this.room})`
		);
		this.freeSearcher();
		this.client.send(`${this.room}|/leave`);
		this.client.onBattleEnd(this.room);
	}
}

// ----------------------------------------------------------------- client
class PSClient {
	constructor() {
		this.ws = null;
		this.challstr = '';
		this.loggedIn = false;
		this.triedAssertion = false;
		this.drivers = new Map();
		this.battleIdx = 0;
		this.currentTeam = null;
		this.lastChallengeAt = 0;
		this.pendingChallenges = {}; // incoming: challenger id -> format id
		this.outgoingTo = ''; // outstanding outgoing challenge target id
		this.shuttingDown = false;
		this.tick = setInterval(() => this.onTick(), 2000);
	}

	connect() {
		this.log(`connecting to ${wsUrl}`);
		this.ws = new WebSocket(wsUrl);
		this.ws.on('open', () => this.log('socket open'));
		this.ws.on('message', data => this.onMessage(String(data)));
		this.ws.on('error', err => this.log(`socket error: ${err.message}`));
		this.ws.on('close', () => {
			this.loggedIn = false;
			if (this.shuttingDown) return;
			this.log(`socket closed; reconnecting in ${RECONNECT_MS}ms`);
			stats.reconnects++;
			setTimeout(() => this.connect(), RECONNECT_MS);
		});
	}

	chaosDrop() {
		this.ws.terminate();
	}

	send(msg) {
		if (this.ws && this.ws.readyState === WebSocket.OPEN) {
			this.ws.send(msg);
		} else {
			this.log(`SEND DROPPED (socket not open): ${msg.slice(0, 80)}`);
		}
	}

	log(m) {
		if (!QUIET) console.log(`[${NAME}] ${m}`);
	}

	onMessage(msg) {
		let room = '';
		let lines = msg.split('\n');
		if (msg.startsWith('>')) {
			room = lines[0].slice(1);
			lines = lines.slice(1);
		}
		if (room.startsWith('battle-')) {
			let driver = this.drivers.get(room);
			if (!driver) {
				if (!lines.some(l => l.startsWith('|init|battle'))) return; // trailing frames of a left room
				driver = new BattleDriver(this, room, this.battleIdx++);
				this.drivers.set(room, driver);
			}
			driver.onFrame(lines);
			return;
		}
		for (const line of lines) this.onGlobalLine(line);
	}

	onGlobalLine(line) {
		if (!line.startsWith('|')) return;
		const idx = line.indexOf('|', 1);
		const cmd = idx < 0 ? line.slice(1) : line.slice(1, idx);
		const rest = idx < 0 ? '' : line.slice(idx + 1);
		switch (cmd) {
			case 'challstr':
				this.challstr = rest;
				void this.login();
				break;
			case 'updateuser': {
				const [rawName, named] = rest.split('|');
				if (named === '1' && toID(rawName) === toID(NAME) && !this.loggedIn) {
					this.loggedIn = true;
					this.log(`logged in as ${rawName.trim()}`);
					this.onLoggedIn();
				}
				break;
			}
			case 'nametaken': {
				const [, message] = rest.split('|');
				if (!PASSWORD && !this.triedAssertion) {
					this.log(`bare guest login refused (${message}); trying the login server`);
					this.triedAssertion = true;
					void this.guestAssertionLogin();
				} else {
					console.error(`login failed: ${message}`);
					process.exit(2);
				}
				break;
			}
			case 'updatechallenges':
				// legacy servers only; current PS announces challenges via |pm|
				try {
					this.pendingChallenges = JSON.parse(rest).challengesFrom || {};
				} catch { /* ignore */ }
				break;
			case 'popup':
				this.log(`popup: ${rest.replace(/\|\|/g, ' / ')}`);
				break;
			case 'pm': {
				// |pm|FROM|TO|/challenge FORMAT|TEAMBUILDER|MSG|BTN|BTN
				// (an empty /challenge = that challenge was cancelled/resolved)
				const parts = rest.split('|');
				const [u1, u2] = [toID(parts[0]), toID(parts[1])];
				const msg = parts.slice(2).join('|');
				if (!msg.startsWith('/challenge')) break;
				const fmt = msg.slice('/challenge'.length).trim().split('|')[0];
				if (!fmt) { // cleared (cancelled / rejected / accepted)
					if (u1 !== toID(NAME)) delete this.pendingChallenges[u1];
					if (u2 !== toID(NAME)) delete this.pendingChallenges[u2];
					if (this.outgoingTo === u1 || this.outgoingTo === u2) this.outgoingTo = '';
				} else if (u1 === toID(NAME)) { // our outgoing challenge
					this.outgoingTo = u2;
				} else if (u2 === toID(NAME)) { // incoming
					this.pendingChallenges[u1] = toID(fmt);
				}
				break;
			}
			default:
				break;
		}
	}

	async login() {
		if (PASSWORD) {
			try {
				const res = await fetch(`${LOGINSERVER}/api/login`, {
					method: 'POST',
					headers: { 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8' },
					body: `name=${encodeURIComponent(NAME)}&pass=${encodeURIComponent(PASSWORD)}&challstr=${encodeURIComponent(this.challstr)}`,
				});
				const text = await res.text();
				const data = JSON.parse(text.slice(1));
				if (!data.assertion || data.assertion.startsWith(';')) {
					console.error(`login failed: ${data.assertion || 'no assertion'}`);
					process.exit(2);
				}
				this.send(`|/trn ${NAME},0,${data.assertion}`);
			} catch (e) {
				console.error(`login server unreachable: ${e.message}`);
				process.exit(2);
			}
		} else {
			// guest: a noguestsecurity server accepts a bare /trn; otherwise
			// |nametaken| triggers the assertion fallback
			this.send(`|/trn ${NAME}`);
		}
	}

	async guestAssertionLogin() {
		try {
			const res = await fetch(
				`${LOGINSERVER}/api/getassertion?userid=${toID(NAME)}&challstr=${encodeURIComponent(this.challstr)}`
			);
			const assertion = await res.text();
			if (assertion.startsWith(';')) {
				console.error(`guest assertion refused (${assertion}); registered name? use --password`);
				process.exit(2);
			}
			this.send(`|/trn ${NAME},0,${assertion}`);
		} catch (e) {
			console.error(`login server unreachable: ${e.message}`);
			process.exit(2);
		}
	}

	onLoggedIn() {
		if (LOBBY_ROOM) {
			this.send(`|/join ${LOBBY_ROOM}`);
			this.log(`joining ${LOBBY_ROOM}`);
		}
		// resume: rejoin every battle that was live when the socket dropped
		for (const [room, driver] of this.drivers) {
			if (!driver.ended) {
				driver.awaitingReplay = true;
				this.send(`|/join ${room}`);
				this.log(`rejoining ${room}`);
			}
		}
	}

	activeBattles() {
		let n = 0;
		for (const d of this.drivers.values()) if (!d.ended) n++;
		return n;
	}

	onTick() {
		if (!this.loggedIn || this.shuttingDown) return;
		if (reachedGameLimit()) return this.shutdown();
		if (this.activeBattles() > 0) return;
		if (CHALLENGE) {
			if (this.outgoingTo) {
				// challenge outstanding; if it's gone stale (opponent hung),
				// cancel and let the next tick re-issue
				if (Date.now() - this.lastChallengeAt > 30000) {
					this.send(`|/cancelchallenge ${this.outgoingTo}`);
					this.outgoingTo = '';
				}
				return;
			}
			if (Date.now() - this.lastChallengeAt < 8000) return;
			this.lastChallengeAt = Date.now();
			this.currentTeam = pickTeam();
			this.send(`|/utm ${Teams.pack(this.currentTeam.sets)}`);
			this.send(`|/challenge ${CHALLENGE}, ${FORMATID}`);
			this.log(`challenging ${CHALLENGE} (${FORMATID}, team ${this.currentTeam.label})`);
		} else if (ACCEPT) {
			const allowed = ACCEPT === 'any' ? null : ACCEPT.split(',').map(toID);
			for (const [from, fmt] of Object.entries(this.pendingChallenges)) {
				if (toID(fmt) !== toID(FORMATID)) continue;
				if (allowed && !allowed.includes(toID(from))) continue;
				this.currentTeam = pickTeam();
				this.send(`|/utm ${Teams.pack(this.currentTeam.sets)}`);
				this.send(`|/accept ${from}`);
				this.log(`accepting ${from} (team ${this.currentTeam.label})`);
				delete this.pendingChallenges[from];
				break;
			}
		}
	}

	onBattleEnd(room) {
		setTimeout(() => this.drivers.delete(room), 5000); // let deinit frames drain
		if (reachedGameLimit()) this.shutdown();
	}

	shutdown() {
		if (this.shuttingDown) return;
		this.shuttingDown = true;
		clearInterval(this.tick);
		summarize();
		try {
			this.ws.close();
		} catch { /* already closed */ }
		const bad = stats.rejections.length + stats.desyncs + stats.proofsBad.length;
		process.exit(bad > 0 ? 1 : 0);
	}
}

function hasGameLimit() {
	return GAMES > 0;
}

function reachedGameLimit() {
	return hasGameLimit() && stats.games >= GAMES;
}

function summarize() {
	console.log('----------------------------------------------------------');
	console.log(
		`${NAME} (${RANDOM ? 'random' : `${MODE}:${ITERS}`}, seed ${SEED}): ` +
		`${stats.W}W ${stats.L}L ${stats.T}T over ${stats.games} games`
	);
	console.log(
		`decisions ${stats.decisions}, avg turns ${(stats.turns / Math.max(1, stats.games)).toFixed(1)}, ` +
		`rejections ${stats.rejections.length}, desyncs ${stats.desyncs}` +
		(RANDOM ? '' : `, legality drift ${stats.legalityDrift}, projections ${stats.projections}`)
	);
	if (!RANDOM && stats.thinkN) {
		console.log(
			`think latency: max ${stats.maxThinkMs}ms, avg ${(stats.thinkMsSum / stats.thinkN).toFixed(0)}ms ` +
			`over ${stats.thinkN} searched decisions`
		);
	}
	if (stats.drops || stats.reconnects) {
		const untriggered = dropSpecs.filter(d => !d.triggered).length;
		console.log(
			`reconnect: ${stats.drops} chaos drops, ${stats.reconnects} socket closures, ${stats.resumes} battle resumes, ` +
			`resume proofs ${stats.proofsOk} ok / ${stats.proofsBad.length} failed / ${stats.proofsSkipped} skipped (turn advanced)` +
			(untriggered ? `, ${untriggered} drop specs untriggered` : '')
		);
	}
	for (const r of stats.rejections.slice(0, 20)) console.log('  REJECTED', r);
	for (const p of stats.proofsBad.slice(0, 10)) console.log('  PROOF FAILED', p);
}

process.on('SIGINT', () => {
	summarize();
	process.exit(130);
});

const client = new PSClient();
client.connect();
