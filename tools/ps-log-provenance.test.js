'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const { provenance } = require('./ps-log-provenance');

test('identity follows binary and actual configuration, independent of artifact insertion order', () => {
	const config = { driver: 'search', iterations: 27000, searchC: 0.4 };
	const first = provenance({ wasm: Buffer.from('compiled A'), pool: 'pool' }, config);
	assert.deepEqual(first, provenance({ pool: 'pool', wasm: Buffer.from('compiled A') }, config));
	assert.notEqual(first.id, provenance({ wasm: Buffer.from('compiled B'), pool: 'pool' }, config).id);
	assert.notEqual(first.id, provenance({ wasm: Buffer.from('compiled A'), pool: 'changed pool' }, config).id);
	assert.notEqual(first.id, provenance({ wasm: Buffer.from('compiled A'), pool: 'pool' }, { ...config, iterations: 300 }).id);
	assert.equal(provenance({ wasm: null }, { driver: 'random' }).artifacts.wasm, null);
	config.iterations = 300;
	assert.equal(first.configuration.iterations, 27000);
});

for (const random of [true, false]) test(`client records private decisions and terminal tails (${random ? 'random' : 'search'})`, () => {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nc2000-provenance-'));
	try {
		const log = path.join(dir, 'decisions.jsonl');
		const matchLog = `${log}.matches.jsonl`;
		const source = fs.readFileSync(path.join(__dirname, 'ps-client.js'), 'utf8');
		fs.writeFileSync(log, '', { mode: 0o644 });
		fs.writeFileSync(matchLog, '', { mode: 0o644 });
		const fakeFs = Object.create(fs);
		fakeFs.readFileSync = (file, ...args) => /pkg-node\/nc2000_wasm/.test(String(file))
			? Buffer.from(String(file).endsWith('.wasm') ? 'test wasm' : 'test glue') : fs.readFileSync(file, ...args);
		class Socket { on() {} }
		const context = {
			__dirname, __filename: path.join(__dirname, 'ps-client.js'),
			console: { log() {}, error() {} }, URL,
			setInterval: () => 1, clearInterval() {}, setTimeout() {}, clearTimeout() {},
			process: { version: process.version, on() {}, exit(code) { throw new Error(`unexpected exit ${code}`); },
				argv: ['node', 'client', '--server', 'ws://user:server-secret@fixture.invalid/?token=query-secret', '--name', 'audit-bot', '--team', 'pool:0',
					'--decision-log', log, '--password', 'do-not-record-this', ...(random ? ['--random'] : [])] },
			require(name) {
				if (name === 'fs') return fakeFs;
				if (name === 'ws') return Socket;
				if (name === './ps') return { sim: { Teams: {} }, FORMAT: 'gen2nintendocup2000noohkostadium2strict' };
				if (name.includes('/pkg-node/')) return { Dex: class {} };
				return require(name);
			},
		};
		vm.createContext(context);
		vm.runInContext(`${source}\nglobalThis.Driver = BattleDriver;`, context);
		const client = { currentTeam: { label: 'pool:fixture', sets: [] }, send() {}, onBattleEnd() {} };
		const driver = new context.Driver(client, 'battle-fixture', 0);
		driver.side = 0;
		driver.turn = 1;
		driver.decisions = 1;
		driver.visibleLines.push('|turn|1');
		const req = { rqid: 1, side: { id: 'p1' } };
		driver.recordDecision(req, 'move rest', null, null, random ? 'random' : 'search');
		driver.recordDecision(req, 'move rest', null, null, random ? 'random' : 'search');
		assert.equal(fs.readFileSync(matchLog, 'utf8'), '');
		assert.throws(() => driver.finalize(), /without a terminal result/);
		driver.onLine('|win|audit-bot');
		driver.finalize();
		const decisions = fs.readFileSync(log, 'utf8').trim().split('\n').map(JSON.parse);
		const matches = fs.readFileSync(matchLog, 'utf8').trim().split('\n').map(JSON.parse);
		assert.equal(decisions.length, 1);
		assert.equal(matches.length, 1);
		assert.equal(decisions[0].version, 3);
		assert.deepEqual(decisions[0].protocolDelta, ['|turn|1']);
		assert.deepEqual(matches[0].protocolTail, ['|win|audit-bot']);
		assert.deepEqual(matches[0].provenance, decisions[0].provenance);
		assert.equal(matches[0].result, 'W');
		assert.equal(matches[0].terminal, true);
		assert.equal(matches[0].server, 'ws://fixture.invalid/');
		assert.equal(matches[0].provenance.configuration.driver, random ? 'random' : 'search');
		assert.equal(matches[0].provenance.configuration.iterations, random ? 0 : 27000);
		assert.equal(matches[0].provenance.artifacts.wasm === null, random);
		assert.ok(!fs.readFileSync(log, 'utf8').includes('do-not-record-this'));
		assert.ok(!fs.readFileSync(matchLog, 'utf8').includes('do-not-record-this'));
		for (const secret of ['server-secret', 'query-secret']) {
			assert.ok(!fs.readFileSync(log, 'utf8').includes(secret));
			assert.ok(!fs.readFileSync(matchLog, 'utf8').includes(secret));
		}
		for (const [index, terminal] of ['|win|opponent', '|tie|'].entries()) {
			const next = new context.Driver(client, `battle-fixture-${index}`, index + 1);
			next.side = 0;
			next.protocolReset = true;
			next.onLine(terminal);
		}
		const all = fs.readFileSync(matchLog, 'utf8').trim().split('\n').map(JSON.parse);
		assert.deepEqual(all.map(r => r.result), ['W', 'L', 'T']);
		assert.deepEqual(all[2].protocolTail, ['|tie|']);
		assert.equal(all[2].protocolReset, true);
		for (const file of [log, matchLog]) assert.equal(fs.statSync(file).mode & 0o777, 0o600);
	} finally {
		fs.rmSync(dir, { recursive: true, force: true });
	}
});
