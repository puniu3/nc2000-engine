// Parse + map + validate the psense.lib.net community rental-team DB
// (「一撃無し２０００ルール」 = no-OHKO NC2000 variant) as a candidate
// team-data source for nc2000-engine.
//
//   node tools/parse-community-rentals.js
//
// Input : data/community-rentals-v0/raw/team-NN.html  (EUC-JP, as fetched)
//         data/community-rentals-v0/raw/index.html
// Uses  : data/i18n-ja.json      (PS-id -> JP; inverted here for JP -> PS-id)
//         data/gen2stadium2.json (dex: national-dex num, Uber tier)
//         crates/wasm/pkg-node    (Validator: canonicalizeTeam + validateTeam)
// Output: data/community-rentals-v0/raw.json    (parsed + mapped teams)
//         data/community-rentals-v0/teams.json  (canonicalized engine sets + verdicts)
//         data/community-rentals-v0/stats.json  (delta / novelty / mapping stats)
//
// Everything fetched is untrusted DATA; this only reads page text, never
// executes anything from it.
'use strict';
const fs = require('fs');
const path = require('path');

const REPO = path.join(__dirname, '..');
const DIR = path.join(REPO, 'data', 'community-rentals-v0');
const RAW = path.join(DIR, 'raw');
const wasm = require(path.join(REPO, 'crates/wasm/pkg-node/nc2000_wasm.js'));

const i18n = JSON.parse(fs.readFileSync(path.join(REPO, 'data/i18n-ja.json'), 'utf8'));
const dexJson = JSON.parse(fs.readFileSync(path.join(REPO, 'data/gen2stadium2.json'), 'utf8'));
const pool = JSON.parse(fs.readFileSync(path.join(REPO, 'data/meta-pool-v0/meta-pool.json'), 'utf8'));

// ---------------------------------------------------------------- normalize
// The DB writes U+2212 (MINUS SIGN) and other dashes where JP names use the
// U+30FC prolonged-sound mark; cells are padded with full/half-width spaces.
function norm(s) {
	return s
		.replace(/[−―—–‐‑－ｰ-]/g, 'ー')
		.replace(/[\s　]+/g, '')
		.normalize('NFC');
}

// ---------------------------------------------------------------- inverted i18n
function invert(table, kind) {
	const map = new Map();
	const collisions = [];
	for (const [id, jp] of Object.entries(table)) {
		const k = norm(jp);
		if (map.has(k) && map.get(k) !== id) collisions.push([k, map.get(k), id]);
		map.set(k, id);
	}
	if (collisions.length) console.error(`WARN ${kind} normalized collisions:`, collisions);
	return map;
}
const speciesByJp = invert(i18n.species, 'species');
const movesByJp = invert(i18n.moves, 'moves');
const itemsByJp = invert(i18n.items, 'items');

// national-dex num -> PS id, and PS id -> {num, tier, name}
const numToId = new Map();
const idInfo = new Map();
for (const [id, s] of Object.entries(dexJson.species)) {
	numToId.set(s.num, id);
	idInfo.set(id, { num: s.num, tier: s.tier, name: s.name });
}

// ------------------------------------------------------------ HP-type decode
// Gen-2 Hidden Power type from Atk/Def DVs (same order as validate.rs).
const HP_TYPES = ['fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost', 'steel',
	'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark'];
function hpTypeFromDvs(atkDv, defDv) {
	return HP_TYPES[(4 * (atkDv % 4) + (defDv % 4)) % 16];
}

// ---------------------------------------------------------------- HTML helpers
const decode = buf => new TextDecoder('euc-jp').decode(buf);
function ent(s) { // decode the few HTML entities the DB emits
	return s.replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(+n))
		.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
}
const stripTags = s => ent(s.replace(/<br\s*\/?>/gi, '\n').replace(/<[^>]+>/g, '')).trim();

// ---------------------------------------------------------------- parse one detail page
function parseTeam(cban, html) {
	const h1 = /<h1[^>]*>([\s\S]*?)<\/h1>/i.exec(html);
	let archetype = h1 ? ent(h1[1]).replace(/^レンタルパーティ名：/, '').trim() : '';
	// team 1's h1 carries the global ruleset note; keep the name, drop the note
	archetype = archetype.replace(/\s*（※[\s\S]*?）\s*$/, '').trim();

	// PD hidden field: national-dex + DVs (Hidden Power type) per mon
	const pdm = /name=PD\s+VALUE=([^>\s]*)/i.exec(html);
	const pd = pdm ? pdm[1] : '';
	const pdMons = pd ? pd.split('_x_').map(rec => {
		const t = rec.split('_');
		const atk = parseInt(t[2], 16), def = parseInt(t[3], 16);
		return {
			num: +t[0],
			dvsHex: [t[2], t[3], t[4], t[5]],
			hpType: Number.isFinite(atk) && Number.isFinite(def) ? hpTypeFromDvs(atk, def) : null,
		};
	}) : [];

	// data rows: submit button + LV + species + 4 moves + item
	const rows = [];
	const rowRe = /<TR><TD><input[^>]*value=\d+><\/TD><TD[^>]*>([\s\S]*?)<\/TD><TD>([\s\S]*?)<\/TD><TD>([\s\S]*?)<\/TD><TD>([\s\S]*?)<\/TD><TD>([\s\S]*?)<\/TD><TD>([\s\S]*?)<\/TD><TD>([\s\S]*?)<\/TD><\/tr>/gi;
	let m;
	while ((m = rowRe.exec(html))) {
		const cell = i => ent(m[i]).replace(/[\s　]+$/g, '').replace(/^[\s　]+/g, '');
		rows.push({
			level: parseInt(cell(1), 10),
			species: cell(2),
			// drop empty / dash-placeholder ("－") move slots
			moves: [cell(3), cell(4), cell(5), cell(6)].filter(x => x !== '' && !/^ー+$/.test(norm(x))),
			item: cell(7),
		});
	}

	// comment + provenance (second table)
	let comment = '', sourceUrl = null, sourceName = null, maker = null;
	const cm = /コメント<td>([\s\S]*?)<\/table>/i.exec(html);
	if (cm) {
		const block = cm[1];
		const srcm = /出典：<a href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/i.exec(block);
		if (srcm) { sourceUrl = srcm[1]; sourceName = stripTags(srcm[2]); }
		const mk = /制作(?:者)?[：]?\s*(?:<[^>]*>)*([\s\S]*?)(?:<\/|<P|$)/i.exec(block);
		if (mk) maker = stripTags(mk[1]).replace(/^[:：]/, '').trim() || null;
		// comment = the block up to the 出典/制作 provenance line
		comment = stripTags(block.split(/出典：|制作/)[0]).replace(/\n+/g, ' ').trim();
	}

	return { cban, archetype, comment, provenance: { sourceUrl, sourceName, maker }, rows, pdMons };
}

// ---------------------------------------------------------------- map + build set
function mapTeam(team, unmapped) {
	const mons = team.rows.map((row, i) => {
		const spKey = norm(row.species);
		const speciesId = speciesByJp.get(spKey) || null;
		if (!speciesId) unmapped.push({ cban: team.cban, kind: 'species', jp: row.species });

		const moves = row.moves.map(jp => {
			const id = movesByJp.get(norm(jp));
			if (!id) unmapped.push({ cban: team.cban, kind: 'move', jp });
			return { jp, id: id || null };
		});

		let itemId = null;
		if (row.item) {
			itemId = itemsByJp.get(norm(row.item)) || null;
			if (!itemId) unmapped.push({ cban: team.cban, kind: 'item', jp: row.item });
		}

		// national-dex cross-check against PD
		const pd = team.pdMons[i] || {};
		let dexCheck = 'n/a';
		if (speciesId && pd.num != null) {
			dexCheck = idInfo.get(speciesId).num === pd.num ? 'ok'
				: `MISMATCH name->${idInfo.get(speciesId).num} pd->${pd.num}`;
		}

		return {
			jpSpecies: row.species, speciesId, level: row.level,
			moves, jpItem: row.item || null, itemId,
			nationalDex: pd.num ?? null, hpTypeFromPd: pd.hpType ?? null, dexCheck,
		};
	});
	return mons;
}

// engine team-JSON (Battle/validator shape). Generic Hidden Power is upgraded
// to the PD-decoded typed variant so canonicalize applies the real DV spread;
// DVs/EVs/gender left absent -> canonicalizeTeam fills them per M14a.
function engineTeam(mons) {
	return mons.map(mon => {
		const name = mon.speciesId ? idInfo.get(mon.speciesId).name : mon.jpSpecies;
		const moves = mon.moves.map(mv => {
			if (mv.id === 'hiddenpower' && mon.hpTypeFromPd) return 'hiddenpower' + mon.hpTypeFromPd;
			return mv.id || mv.jp; // unmapped -> pass JP through so the validator flags it
		});
		const set = { name, species: name, item: '', ability: 'No Ability', moves, level: mon.level };
		if (mon.itemId) set.item = mon.itemId;      // PS item id; validator toid()s it, canonicalize -> display name
		else if (mon.jpItem) set.item = mon.jpItem; // unmapped -> pass JP through so the validator flags it
		return set;
	});
}

// ---------------------------------------------------------------- run
const dex = new wasm.Dex();
const validator = new wasm.Validator(dex);

const files = fs.readdirSync(RAW).filter(f => /^team-\d+\.html$/.test(f)).sort();
const unmapped = [];
const parsed = [];
const engineTeams = [];
const results = [];

for (const f of files) {
	const cban = parseInt(f.match(/\d+/)[0], 10);
	const team = parseTeam(cban, decode(fs.readFileSync(path.join(RAW, f))));
	const mons = mapTeam(team, unmapped);
	const et = engineTeam(mons);

	// canonicalize (fills DVs/EVs/gender legally) then validate the canonical form
	const canon = JSON.parse(validator.canonicalizeTeam(JSON.stringify(et)));
	let verdict;
	if (canon.ok) {
		verdict = JSON.parse(validator.validateTeam(JSON.stringify(canon.team)));
		verdict.canonOk = true;
	} else {
		verdict = { ok: false, canonOk: false, errors: canon.errors, findings: canon.errors };
	}
	const errorCodes = (verdict.errors && Array.isArray(verdict.errors))
		? verdict.errors.map(e => e.code)
		: (verdict.findings || []).filter(f => f.severity === 'error').map(f => f.code);

	parsed.push({ ...team, mons });
	engineTeams.push({ cban, archetype: team.archetype, sets: canon.ok ? canon.team : et });
	results.push({
		cban, archetype: team.archetype, size: mons.length,
		ok: !!verdict.ok, errorCodes,
		errors: verdict.errors && Array.isArray(verdict.errors) ? verdict.errors
			: (verdict.findings || []).filter(f => f.severity === 'error'),
	});
}

// ---------------------------------------------------------------- ruleset-delta + novelty
const OHKO = new Set(['horndrill', 'guillotine', 'fissure']);
const EVASION = new Set(['doubleteam', 'minimize']);
const deltaMons = [];
let ohkoCount = 0, evasionCount = 0, brightCount = 0, uberCount = 0;
for (const t of parsed) {
	for (const mon of t.mons) {
		for (const mv of mon.moves) {
			if (OHKO.has(mv.id)) { ohkoCount++; deltaMons.push({ cban: t.cban, species: mon.speciesId, kind: 'OHKO', move: mv.id }); }
			if (EVASION.has(mv.id)) { evasionCount++; deltaMons.push({ cban: t.cban, species: mon.speciesId, kind: 'evasion', move: mv.id }); }
		}
		if (mon.itemId === 'brightpowder') { brightCount++; deltaMons.push({ cban: t.cban, species: mon.speciesId, kind: 'brightpowder' }); }
		if (mon.speciesId && idInfo.get(mon.speciesId).tier === 'Uber') { uberCount++; deltaMons.push({ cban: t.cban, species: mon.speciesId, kind: 'Uber' }); }
	}
}

// novelty vs the 34 meta-pool teams (species-set signature)
const sig = sp => [...sp].map(x => x.toLowerCase()).sort().join(',');
const poolSigs = new Set(pool.teams.map(t => sig(t.species)));
const poolSpeciesSets = pool.teams.map(t => new Set(t.species.map(x => x.toLowerCase())));
const novelty = parsed.map(t => {
	const species = t.mons.map(m => m.speciesId).filter(Boolean);
	const s = sig(species);
	const dup = poolSigs.has(s);
	// nearest pool team by species overlap (Jaccard on species set)
	const mine = new Set(species);
	let best = 0, bestId = null;
	pool.teams.forEach((pt, k) => {
		const ps = poolSpeciesSets[k];
		const inter = [...mine].filter(x => ps.has(x)).length;
		const uni = new Set([...mine, ...ps]).size;
		const j = uni ? inter / uni : 0;
		if (j > best) { best = j; bestId = pt.id; }
	});
	return { cban: t.cban, archetype: t.archetype, species, exactDup: dup, nearestPool: bestId, nearestJaccard: +best.toFixed(2) };
});

// species-frequency in the rental DB vs pool
const rentalSpeciesFreq = {};
for (const t of parsed) for (const m of t.mons) if (m.speciesId) rentalSpeciesFreq[m.speciesId] = (rentalSpeciesFreq[m.speciesId] || 0) + 1;
const poolSpeciesFreq = {};
for (const pt of pool.teams) for (const sp of pt.species) { const id = sp.toLowerCase(); poolSpeciesFreq[id] = (poolSpeciesFreq[id] || 0) + 1; }

const stats = {
	teams: parsed.length,
	validClean: results.filter(r => r.ok).length,
	failures: results.filter(r => !r.ok).map(r => ({ cban: r.cban, archetype: r.archetype, size: r.size, errorCodes: r.errorCodes, errors: r.errors })),
	unmapped,
	unmappedUnique: [...new Set(unmapped.map(u => `${u.kind}:${u.jp}`))],
	rulesetDelta: { ohkoCount, evasionCount, brightPowderCount: brightCount, uberCount, detail: deltaMons },
	novelty: { exactDupes: novelty.filter(n => n.exactDup).length, novel: novelty.filter(n => !n.exactDup).length, teams: novelty },
	dexMismatches: parsed.flatMap(t => t.mons.filter(m => m.dexCheck && m.dexCheck.startsWith('MISMATCH')).map(m => ({ cban: t.cban, species: m.jpSpecies, detail: m.dexCheck }))),
	rentalSpeciesFreq, poolSpeciesFreq,
};

// ---------------------------------------------------------------- write
fs.writeFileSync(path.join(DIR, 'raw.json'), JSON.stringify({
	meta: {
		source: 'http://psense.lib.net/_/PDINPUT2.cgi (community rental-team DB, 「一撃無し２０００ルール」 = no-OHKO NC2000)',
		fetched: new Date().toISOString().slice(0, 10),
		ruleset: 'no-OHKO NC2000 variant (differs from shipped gen2nc2000 which bans neither OHKO nor evasion nor Bright Powder)',
		teams: parsed.length,
		note: 'DVs/EVs are not published in the human-readable view; per-mon nationalDex + hpTypeFromPd are decoded from the page PD hidden field.',
	},
	teams: parsed,
}, null, 1) + '\n');

fs.writeFileSync(path.join(DIR, 'teams.json'), JSON.stringify({
	meta: { note: 'canonicalized engine-shape sets (M14a canonicalizeTeam fills DVs/EVs/gender); generic Hidden Power upgraded to PD-decoded typed variant', teams: engineTeams.length },
	teams: engineTeams,
}, null, 1) + '\n');

fs.writeFileSync(path.join(DIR, 'stats.json'), JSON.stringify(stats, null, 1) + '\n');

// ---------------------------------------------------------------- report
console.log(`teams parsed: ${parsed.length}`);
console.log(`unmapped names: ${unmapped.length} (unique ${stats.unmappedUnique.length}) -> ${JSON.stringify(stats.unmappedUnique)}`);
console.log(`dex-num mismatches: ${stats.dexMismatches.length} ${JSON.stringify(stats.dexMismatches)}`);
console.log(`validate clean: ${stats.validClean}/${parsed.length}`);
for (const r of results.filter(r => !r.ok)) console.log(`  FAIL No.${r.cban} [${r.archetype}] size=${r.size} :: ${r.errorCodes.join(', ')}`);
console.log(`ruleset-delta: OHKO=${ohkoCount} evasion=${evasionCount} brightPowder=${brightCount} Uber=${uberCount}`);
console.log(`novelty: exact-dupes=${stats.novelty.exactDupes} novel=${stats.novelty.novel}`);
for (const n of novelty) console.log(`  No.${n.cban} ${n.exactDup ? 'DUP ' : 'NEW '} J=${n.nearestJaccard} ~${n.nearestPool} [${n.archetype}] ${n.species.join('/')}`);
