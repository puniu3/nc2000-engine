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
//     [--games N] [--iters 10000] [--seed 1] [--mode blind|open] \
//     [--opp-team-file FILE.json] [--random] [--timer] [--no-tables] \
//     [--decision-log FILE.jsonl] \
//     [--password PW] [--loginserver URL] [--format gen2nintendocup2000noohkostadium2strict] \
//     [--drop SPEC] [--quiet]
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
const WebSocket = require('ws');
const { sim, FORMAT } = require('./ps');
const { Teams } = sim;

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
  --challenge USER  challenge USER repeatedly until --games are done
  --accept WHO      accept challenges: 'any' or comma list of names
  --games N         number of complete battles to play (default 1)
  --mode M          blind (default; pool-prior belief) | open (pin the
                    opponent's true sets — needs --opp-team-file, only
                    meaningful where sheets are genuinely open)
  --opp-team-file F opponent sets JSON for --mode open
  --iters N         blind search iterations per decision (default 10000)
  --seed N          searcher / random-mode seed (default 1)
  --random          random driver mode (no searcher; uniform legal choice)
  --timer           turn the battle timer on in every game
  --no-tables       skip loading baked preview tables
  --drop SPEC       verification hook: socket kills at chosen decision
                    points (see header comment)
  --decision-log F  append private (mode 0600) JSONL for regret replay:
                    request, incremental visible protocol, exact own team,
                    submitted action, diagnostic state, root policy/config
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
const PASSWORD = args.password && args.password !== true ? String(args.password) : '';
const LOGINSERVER = String(args.loginserver || 'https://play.pokemonshowdown.com').replace(/\/$/, '');
const CHALLENGE = args.challenge && args.challenge !== true ? String(args.challenge) : '';
const ACCEPT = args.accept && args.accept !== true ? String(args.accept) : '';
const GAMES = parseInt(args.games || '1', 10);
const MODE = String(args.mode || 'blind');
// 1000 iters left flat roots as a visit-count lottery (battle-3623 T6: argmax
// split 23/16/7/4 over 50 seeds; unanimous at 10000 — replay_postmortem_3623).
// ~3 s/move native at 10k vs the 150 s battle timer.
const ITERS = parseInt(args.iters || '10000', 10);
const SEED = parseInt(args.seed || '1', 10);
const RANDOM = !!args.random;
const TIMER = !!args.timer;
const QUIET = !!args.quiet;
const DECISION_LOG = args['decision-log'] && args['decision-log'] !== true ?
	path.resolve(String(args['decision-log'])) : '';
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
const pool = JSON.parse(fs.readFileSync(path.join(REPO, 'data/meta-pool-v0/meta-pool.json'), 'utf8'));
const poolJson = JSON.stringify(pool);

let rngState = (SEED ^ 0x9e3779b9) >>> 0;
const rng = () => { // mulberry32 (random-mode choices + pool:random picks)
	rngState = (rngState + 0x6d2b79f5) >>> 0;
	let t = rngState;
	t = Math.imul(t ^ (t >>> 15), t | 1);
	t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
	return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const rngInt = n => Math.floor(rng() * n);

function pickTeam() {
	if (TEAMSPEC.startsWith('pool:')) {
		const which = TEAMSPEC.slice(5);
		const idx = which === 'random' ? rngInt(pool.teams.length) : parseInt(which, 10);
		if (!(idx >= 0 && idx < pool.teams.length)) throw new Error(`bad pool index ${which}`);
		return { sets: pool.teams[idx].sets, label: `pool:${idx}` };
	}
	const raw = JSON.parse(fs.readFileSync(TEAMSPEC, 'utf8'));
	const sets = Array.isArray(raw) ? raw : raw.sets;
	if (!Array.isArray(sets)) throw new Error(`${TEAMSPEC}: expected a JSON array of sets (or {sets:[...]})`);
	return { sets, label: TEAMSPEC };
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
if (MODE === 'open') {
	const f = args['opp-team-file'];
	if (!f || f === true) {
		console.error('--mode open needs --opp-team-file (the opponent\'s true sets)');
		process.exit(2);
	}
	const raw = JSON.parse(fs.readFileSync(String(f), 'utf8'));
	oppTeamJson = JSON.stringify(Array.isArray(raw) ? raw : raw.sets);
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
			this.recordDecision(req, choice, null, null, 'random');
			this.maybeDrop(req, 'post');
			return;
		}

		if (!this.searcher) {
			this.side = req.side && req.side.id === 'p2' ? 1 : 0;
			this.searcher = new wasm.ProtocolSearcher(dex, this.side, poolJson, SEED * 1000 + this.battleIdx);
			this.searcher.setOwnTeam(JSON.stringify(this.client.currentTeam.sets));
			if (MODE === 'open') this.searcher.pinOpponent(oppTeamJson);
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
			this.searcher.step(ITERS);
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
		this.recordDecision(req, choice, policy, state, 'search');
		this.maybeDrop(req, 'post');
	}

	recordDecision(req, choice, rootPolicy, stateView, driver) {
		if (!DECISION_LOG) return;
		const rqidKey = String(req.rqid ?? `decision-${this.decisions}`);
		if (this.loggedRqids.has(rqidKey)) return;
		this.loggedRqids.add(rqidKey);
		const protocolDelta = this.visibleLines.slice(this.loggedLineCount);
		this.loggedLineCount = this.visibleLines.length;
		appendDecision({
			version: 2,
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
			request: req,
			protocolReset: this.protocolReset,
			protocolDelta,
			submitted: choice,
			rootPolicy,
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
		console.log(
			`game ${stats.games}/${GAMES}: ${this.result} in ${this.turn} turns, ` +
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
		if (stats.games >= GAMES) return this.shutdown();
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
		if (stats.games >= GAMES) this.shutdown();
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
