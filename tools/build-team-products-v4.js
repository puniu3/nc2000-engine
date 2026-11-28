// Generate the curated `team-pool-v4` 10-shell equal-weight Strong Generalist Bot Pool
// (with targeted low-poison 6/6-same-item camouflage and CRN-verified singletons),
// matching `flagship-mixture.json`, AND the preserved full 32-shell environment catalog
// (`environment-pool-32.json`) from committed repo sources with `training-v3` normalization
// and strict WASM (`gen2stadium2` + `castle`) + JP-NC2000 legality checks.
//
// Outputs:
//   data/team-pool-v4/team-pool.json            10-shell (15-team) equal-weight Strong Generalist Bot Pool
//   data/team-pool-v4/flagship-mixture.json     10-shell (15-team) equal-weight Strong Generalist Bot Mixture
//   data/team-pool-v4/environment-pool-32.json  32-shell (33-team) preserved full environment knowledge catalog
//
// Usage:
//   node tools/build-team-products-v4.js [--out-root DIR]
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { normalizeCatalog } = require('./normalize-training');
const { attractGenderCensus } = require('./attract-gender-census');
const wasm = require('../crates/wasm/pkg-node/nc2000_wasm.js');

const REPO = path.join(__dirname, '..');
const arg = name => {
	const i = process.argv.indexOf(name);
	return i > 0 ? process.argv[i + 1] : null;
};
const OUT = arg('--out-root') ? path.resolve(arg('--out-root')) : REPO;
const read = rel => JSON.parse(fs.readFileSync(path.resolve(REPO, rel), 'utf8'));
const write = (rel, obj) => {
	const full = path.join(OUT, rel);
	fs.mkdirSync(path.dirname(full), { recursive: true });
	fs.writeFileSync(full, JSON.stringify(obj, null, 1) + '\n');
};
const fnv1a64 = rel => {
	let h = 0xcbf29ce484222325n;
	for (const b of fs.readFileSync(path.resolve(REPO, rel))) {
		h = ((h ^ BigInt(b)) * 0x100000001b3n) & 0xffffffffffffffffn;
	}
	return `fnv1a64:${h.toString(16).padStart(16, '0')}`;
};
const sha256File = rel =>
	crypto.createHash('sha256').update(fs.readFileSync(path.resolve(REPO, rel))).digest('hex');

// Full 32-shell (33-team) baseline environment catalog preserved as opponent/metagame knowledge
// (`data/team-pool-v4/environment-pool-32.json`). The 10 curated Tier-1 generalist shells form
// the trimmed equal-weight bot pool; all 22 remaining shells (weaker lower-tier archetypes and
// narrow specialist assassins like `rental-cban-13`) are preserved intact here alongside
// untouched `data/belief-pool-v3/`.
const ENV32_ROSTER_SPEC = [
	// 10 Curated Strong Generalist Shells (promoted into the trimmed bot pool)
	{ id: 'pb-131', tier: 'S', inBotPool: true, v4PlainScore: 0.5026 },
	{ id: 'pb-6', tier: 'S', inBotPool: true, v4PlainScore: 0.4902 },
	{ id: 'sample-07', tier: 'S', inBotPool: true, v4PlainScore: 0.4995 },
	{ id: 'apex-c04-bug-3atkstar-boomgen', tier: 'S', inBotPool: true, evolved: true, v4PlainScore: 0.4947 },
	{ id: 'ayagen-m-zap50', tier: 'S', inBotPool: true, evolved: true, v4PlainScore: 0.4939, fromUpgrade: true },
	{ id: 'pb63-m-gar55', tier: 'S', inBotPool: true, evolved: true, v4PlainScore: 0.5019, fromUpgrade: true },
	{ id: 'rraikou-m-cune50', tier: 'S', inBotPool: true, evolved: true, v4PlainScore: 0.4808, fromUpgrade: true },
	{ id: 'pb85-m-cune50', tier: 'S', inBotPool: true, evolved: true, v4PlainScore: 0.5023, fromUpgrade: true },
	{ id: 'pb-87', tier: 'S', inBotPool: true, v4PlainScore: 0.4971 },
	{ id: 'sample-12', tier: 'S', inBotPool: true, v4PlainScore: 0.4668 },

	// 22 Preserved Environment Knowledge Shells (narrow specialists & lower-tier archetypes; 23 teams)
	{ id: 'rental-cban-13', tier: '1', inBotPool: false, envRole: 'specialist-assassin', v4PlainScore: 0.4758 },
	{ id: 'pb84-cln-skarm50', tier: '1', inBotPool: false, evolved: true, v4PlainScore: 0.4714, fromUpgrade: true },
	{ id: 'pb66-m-wak55', tier: '1', inBotPool: false, evolved: true, v4PlainScore: 0.4725, fromUpgrade: true },
	{ id: 'pb-89', tier: '1', inBotPool: false, v4PlainScore: 0.4917 },
	{ id: 'offantom-m-wak50', tier: '1', inBotPool: false, evolved: true, v4PlainScore: 0.4736, fromUpgrade: true },
	{ id: 'pb-96', tier: '1', inBotPool: false, v4PlainScore: 0.4653 },
	{ id: 'sample-06', tier: '1', inBotPool: false, v4PlainScore: 0.4533 },
	{ id: 'mjj-2014-thdrcomb', tier: '1', inBotPool: false, v4PlainScore: 0.4492 },
	{ id: 'sample-22-orig', tier: '1', inBotPool: false, v4PlainScore: 0.4418 },
	{ id: 'pb122-m-lax50', tier: '1', inBotPool: false, evolved: true, v4PlainScore: 0.4320, fromUpgrade: true },
	{ id: 'rental-cban-8', tier: '1', inBotPool: false, v4PlainScore: 0.4280 },
	{ id: 's0053-lefties-hpi', tier: '1', inBotPool: false, variantCluster: 'vc-s0053-lefties', evolved: true, v4PlainScore: 0.4274 },
	{ id: 's0053-lefties-dp', tier: '1', inBotPool: false, variantCluster: 'vc-s0053-lefties', evolved: true, v4PlainScore: 0.4274 },
	{ id: 'sample-17-orig', tier: '1', inBotPool: false, v4PlainScore: 0.4096 },
	{ id: 'mjj-2010-garap-a', tier: '1', inBotPool: false, v4PlainScore: 0.4007 },
	{ id: 'sample-08', tier: '1', inBotPool: false, v4PlainScore: 0.3662 },
	{ id: 'mjj-type-QA-moudoku', tier: '1', inBotPool: false, v4PlainScore: 0.3638 },
	{ id: 'hc75-top8-tsuru', tier: '1', inBotPool: false, v4PlainScore: 0.3592 },
	{ id: 'sample-09', tier: '1', inBotPool: false, v4PlainScore: 0.3491 },
	{ id: 'sample-14', tier: '1', inBotPool: false, v4PlainScore: 0.3458 },
	{ id: 's02-m-lax-curse', tier: '1', inBotPool: false, evolved: true, v4PlainScore: 0.3438, fromUpgrade: true },
	{ id: 'sample-11', tier: '1', inBotPool: false, v4PlainScore: 0.3425 },
	{ id: 'deme-r1-00-s0030', tier: '1', inBotPool: false, evolved: true, v4PlainScore: 0.2912 },
];

// Curated 10-shell (15-team) Equal-Weight Strong Generalist Bot Pool (`P_strong10`, 1/10 = 0.100 per shell):
// Ordered with distinct shells at indices 0 and 1 (`pb-131` and `pb-6`) and by Stage 6d frozen dual-axis score.
//   - 4 low-poison 6/6-same-item camouflaged shells (`apex` 50/50, `pb63` 50/50, `rraikou` 80/20, `pb85` 50/35/15)
//   - 3 CRN-upgraded singletons (`pb131-c3r`, `s07-c8`, `pb87-c5`) where camouflage proved null or harmful
//   - 3 verified clean base singletons (`pb-6`, `ayagen-m-zap50`, `sample-12`)
const BOT_POOL_SPEC = [
	{
		id: 'pb131-c3r-fbhera-magzap-refcloy',
		variantCluster: 'pb-131',
		mode: 'upgrade-singleton',
		intraShellWeight: 1.00,
		score: 0.5906,
		scoreVsPclean7: 0.6063,
		scoreVsPlain10: 0.5750,
		fromStrongVariant: true,
	},
	{
		id: 'pb-6',
		variantCluster: 'pb-6',
		mode: 'base-singleton',
		intraShellWeight: 1.00,
		score: 0.5493,
		scoreVsPclean7: 0.5069,
		scoreVsPlain10: 0.5917,
	},
	{
		id: 's07-c8-mcoatrode-fbwak-delax-thunzap',
		variantCluster: 'sample-07',
		mode: 'upgrade-singleton',
		intraShellWeight: 1.00,
		score: 0.5277,
		scoreVsPclean7: 0.5222,
		scoreVsPlain10: 0.5333,
		fromStrongVariant: true,
	},
	{
		id: 'apex-c3r-fbhera-twstar-refcloy',
		variantCluster: 'apex-c04-bug-3atkstar-boomgen',
		mode: 'camo-2way-50-50',
		intraShellWeight: 0.50,
		score: 0.5228,
		scoreVsPclean7: 0.5123,
		scoreVsPlain10: 0.5333,
		fromStrongVariant: true,
	},
	{
		id: 'apex-c5-perishgar-3atkstar-fbwak-curcloy',
		variantCluster: 'apex-c04-bug-3atkstar-boomgen',
		mode: 'camo-2way-50-50',
		intraShellWeight: 0.50,
		score: 0.5228,
		scoreVsPclean7: 0.5123,
		scoreVsPlain10: 0.5333,
		fromStrongVariant: true,
	},
	{
		id: 'ayagen-m-zap50',
		variantCluster: 'ayagen-m-zap50',
		mode: 'base-singleton',
		intraShellWeight: 1.00,
		score: 0.5155,
		scoreVsPclean7: 0.4893,
		scoreVsPlain10: 0.5417,
		fromUpgrade: true,
	},
	{
		id: 'pb63-c3-restlax-4atkgar-wwzap',
		variantCluster: 'pb63-m-gar55',
		mode: 'camo-2way-50-50',
		intraShellWeight: 0.50,
		score: 0.5101,
		scoreVsPclean7: 0.5119,
		scoreVsPlain10: 0.5083,
		fromStrongVariant: true,
	},
	{
		id: 'pb63-c3b-sdlax-4atkgar-dpzap',
		variantCluster: 'pb63-m-gar55',
		mode: 'camo-2way-50-50',
		intraShellWeight: 0.50,
		score: 0.5101,
		scoreVsPclean7: 0.5119,
		scoreVsPlain10: 0.5083,
		fromStrongVariant: true,
	},
	{
		id: 'rraikou-m-cune50',
		variantCluster: 'rraikou-m-cune50',
		mode: 'camo-2way-80-20',
		intraShellWeight: 0.80,
		score: 0.4904,
		scoreVsPclean7: 0.4224,
		scoreVsPlain10: 0.5583,
		fromUpgrade: true,
	},
	{
		id: 'rraikou-c1-roarkou-roarcune-fblax',
		variantCluster: 'rraikou-m-cune50',
		mode: 'camo-2way-80-20',
		intraShellWeight: 0.20,
		score: 0.4904,
		scoreVsPclean7: 0.4224,
		scoreVsPlain10: 0.5583,
		fromStrongVariant: true,
	},
	{
		id: 'pb85-m-cune50',
		variantCluster: 'pb85-m-cune50',
		mode: 'camo-3way-50-35-15',
		intraShellWeight: 0.50,
		score: 0.4870,
		scoreVsPclean7: 0.4823,
		scoreVsPlain10: 0.4917,
		fromUpgrade: true,
	},
	{
		id: 'pb85-c5-roartalkcune-wwzap',
		variantCluster: 'pb85-m-cune50',
		mode: 'camo-3way-50-35-15',
		intraShellWeight: 0.35,
		score: 0.4870,
		scoreVsPclean7: 0.4823,
		scoreVsPlain10: 0.4917,
		fromStrongVariant: true,
	},
	{
		id: 'pb85-c3r-sdlax-refcloy',
		variantCluster: 'pb85-m-cune50',
		mode: 'camo-3way-50-35-15',
		intraShellWeight: 0.15,
		score: 0.4870,
		scoreVsPclean7: 0.4823,
		scoreVsPlain10: 0.4917,
		fromStrongVariant: true,
	},
	{
		id: 'pb87-c5-fblax-dpzap-crunchttar',
		variantCluster: 'pb-87',
		mode: 'upgrade-singleton',
		intraShellWeight: 1.00,
		score: 0.4648,
		scoreVsPclean7: 0.4545,
		scoreVsPlain10: 0.4750,
		fromStrongVariant: true,
	},
	{
		id: 'sample-12',
		variantCluster: 'sample-12',
		mode: 'base-singleton',
		intraShellWeight: 1.00,
		score: 0.4338,
		scoreVsPclean7: 0.4258,
		scoreVsPlain10: 0.4417,
	},
];

function checkJpNc2000Legality(id, sets) {
	if (!Array.isArray(sets) || sets.length !== 6) {
		throw new Error(`${id}: must have exactly 6 sets`);
	}
	const items = new Set();
	const species = new Set();
	for (const s of sets) {
		if (species.has(s.species)) throw new Error(`${id}: duplicate species ${s.species}`);
		species.add(s.species);
		if (s.level < 50 || s.level > 55) throw new Error(`${id}: illegal level ${s.species} L${s.level}`);
		const moves = new Set(s.moves);
		if (s.species === 'Snorlax' && moves.has('Lovely Kiss')) {
			throw new Error(`${id}: NYPC USA event move Lovely Kiss on Snorlax is forbidden`);
		}
		if (s.species === 'Marowak' && moves.has('Belly Drum') && moves.has('Rock Slide')) {
			throw new Error(`${id}: incompatible Gen 2 egg combo Belly Drum + Rock Slide on Marowak`);
		}
		if (s.species === 'Zapdos' && s.level < 61 && moves.has('Light Screen')) {
			throw new Error(`${id}: Zapdos cannot legally know Light Screen below Lv 61`);
		}
		if (s.species === 'Blissey' && moves.has('Heal Bell') && moves.has('Counter')) {
			throw new Error(`${id}: incompatible Gen 1 TM Counter + Gen 2 Egg Heal Bell on Blissey`);
		}
		if (s.species === 'Espeon' && moves.has('Growth')) {
			throw new Error(`${id}: NYPC USA event move Growth on Espeon is forbidden`);
		}
		if (s.species === 'Slowking' && moves.has('Belly Drum')) {
			throw new Error(`${id}: NYPC USA event move Belly Drum on Slowking is forbidden`);
		}
		if (s.item) {
			if (items.has(s.item)) throw new Error(`${id}: duplicate item ${s.item}`);
			items.add(s.item);
		}
	}
	const min3Sum = sets.map(s => s.level).sort((a, b) => a - b).slice(0, 3).reduce((a, b) => a + b, 0);
	if (min3Sum > 155) {
		throw new Error(`${id}: lowest 3 levels sum to ${min3Sum} > 155 (no legal 3-mon NC2000 pick)`);
	}
}

function validateAndNormalize(items, dex, attractPreferFemale, engineDex, validator) {
	for (const t of items) {
		const canon = JSON.parse(validator.canonicalizeTeam(JSON.stringify(t.sets)));
		if (!canon.ok) {
			throw new Error(`WASM canonicalizeTeam rejected ${t.id}: ${JSON.stringify(canon.errors)}`);
		}
		t.sets = canon.team;
	}
	const optimized = normalizeCatalog(items, dex, attractPreferFemale);
	for (const t of items) {
		const sets = optimized.get(t.id).sets;
		checkJpNc2000Legality(t.id, sets);
		const resPs = JSON.parse(validator.validateTeamWithRuleset(JSON.stringify(sets), 'gen2stadium2'));
		if (!resPs.ok) {
			throw new Error(`WASM PsStadium2 Validator rejected ${t.id}: ${JSON.stringify(resPs.findings || resPs.errors)}`);
		}
		const resCastle = JSON.parse(validator.validateTeamWithRuleset(JSON.stringify(sets), 'castle'));
		if (!resCastle.ok) {
			throw new Error(`WASM Castle Validator rejected ${t.id}: ${JSON.stringify(resCastle.findings || resCastle.errors)}`);
		}
		const b = new wasm.Battle(engineDex, JSON.stringify(sets), JSON.stringify(sets), '1,2,3,4');
		b.free();
	}
	return optimized;
}

function main() {
	const dex = read('data/gen2stadium2.json');
	const v3 = read('data/team-pool-v3/team-pool.json');
	const up2 = read('docs/reports/okimono-upgrades-v2.json');
	const strongVars = read('docs/reports/policy-v4-strong-pool-variants.json');
	const bakePriorPath = 'data/belief-pool-v3/bake-prior.json';
	const bakePrior = read(bakePriorPath);

	const v3ById = new Map(v3.teams.map(t => [t.id, t]));
	const up2ById = new Map(up2.teams.map(t => [t.id, t]));
	const svById = new Map(strongVars.teams.map(t => [t.id, t]));

	function resolveBaseEntry(spec) {
		if (spec.fromUpgrade) {
			const u = up2ById.get(spec.id);
			if (!u) throw new Error(`Upgrade team not found in okimono-upgrades-v2.json: ${spec.id}`);
			const parent = v3ById.get(u.parent);
			return {
				provenance: {
					family: 'policy-v2-okimono-refresh',
					origin: 'evolved',
					parent: u.parent,
					category: u.category,
					source: `NC2000 policy-v2 Refresh (${u.parent} -> ${spec.id}: ${u.change})`,
					url: 'docs/reports/okimono-upgrades-v2.json',
					version: `v4-${spec.id}`,
					...(parent?.provenance ? { parentSource: parent.provenance.source } : {}),
				},
				evolved: true,
				sets: structuredClone(u.sets),
			};
		}
		const src = v3ById.get(spec.id);
		if (!src) throw new Error(`Preserved team not found in team-pool-v3: ${spec.id}`);
		return {
			provenance: src.provenance,
			evolved: Boolean(spec.evolved || src.evolved),
			sets: structuredClone(src.sets),
		};
	}

	// 1. Build preserved 32-shell (33-team) environment catalog
	const env32Items = ENV32_ROSTER_SPEC.map(spec => {
		const base = resolveBaseEntry(spec);
		return {
			id: spec.id,
			tier: spec.tier,
			inBotPool: Boolean(spec.inBotPool),
			...(spec.envRole ? { envRole: spec.envRole } : {}),
			variantCluster: spec.variantCluster || spec.id,
			evolved: base.evolved,
			score: spec.v4PlainScore,
			provenance: base.provenance,
			sets: base.sets,
		};
	});

	// 2. Build curated 10-shell (15-team) Equal-Weight Strong Generalist Bot Pool
	const botItems = BOT_POOL_SPEC.map(spec => {
		if (spec.fromStrongVariant) {
			const sv = svById.get(spec.id);
			if (!sv) throw new Error(`Strong variant not found in policy-v4-strong-pool-variants.json: ${spec.id}`);
			return {
				id: spec.id,
				tier: 'S',
				variantCluster: spec.variantCluster,
				mode: spec.mode,
				intraShellWeight: spec.intraShellWeight,
				evolved: true,
				score: spec.score,
				scoreVsPclean7: spec.scoreVsPclean7,
				scoreVsPlain10: spec.scoreVsPlain10,
				provenance: {
					family: 'policy-v4-strong-pool-camo',
					origin: 'evolved',
					parent: sv.parent,
					mode: sv.mode,
					stages: sv.stages,
					source: `NC2000 policy-v4 Strong Pool (${sv.parent} -> ${spec.id}: ${sv.change})`,
					url: 'docs/reports/policy-v4-strong-pool-variants.json',
					version: `v4-strong10-${spec.id}`,
				},
				sets: structuredClone(sv.sets),
			};
		}
		const base = resolveBaseEntry(spec);
		return {
			id: spec.id,
			tier: 'S',
			variantCluster: spec.variantCluster,
			mode: spec.mode,
			intraShellWeight: spec.intraShellWeight,
			evolved: base.evolved,
			score: spec.score,
			scoreVsPclean7: spec.scoreVsPclean7,
			scoreVsPlain10: spec.scoreVsPlain10,
			provenance: base.provenance,
			sets: base.sets,
		};
	});

	// Canonicalize, training-v3 normalize, and validate EVERY team under BOTH PsStadium2 & Castle
	const attract = attractGenderCensus(bakePrior);
	const engineDex = new wasm.Dex();
	const validator = new wasm.Validator(engineDex);
	let envOptimized;
	let botOptimized;
	try {
		envOptimized = validateAndNormalize(env32Items, dex, attract.preferFemale, engineDex, validator);
		botOptimized = validateAndNormalize(botItems, dex, attract.preferFemale, engineDex, validator);
	} finally {
		validator.free();
		engineDex.free();
	}

	// Verify 32 distinct shells in environment-pool-32
	const envClusterSizes = new Map();
	const envClusterSigs = new Map();
	const envShellMultisets = new Map();
	const envParentSigs = new Map();
	for (const t of env32Items) {
		const sets = envOptimized.get(t.id).sets;
		const ck = t.variantCluster;
		envClusterSizes.set(ck, (envClusterSizes.get(ck) || 0) + 1);
		const sig = JSON.stringify(sets.map(s => [s.species, s.level, s.gender, Boolean(s.item)]));
		const hpSig = JSON.stringify(sets.map(s => s.ivs.hp));
		if (envClusterSigs.has(ck)) {
			if (envClusterSigs.get(ck) !== sig) {
				throw new Error(`Environment cluster ${ck} has mismatched Turn-0 signature on ${t.id}`);
			}
		} else {
			envClusterSigs.set(ck, sig);
			envParentSigs.set(ck, { t0: sig, hp: hpSig });
			const multiset = JSON.stringify(sets.map(s => `${s.species}:${s.level}`).sort());
			if (envShellMultisets.has(multiset)) {
				throw new Error(`Duplicate shell in env32 between ${envShellMultisets.get(multiset)} and ${t.id}`);
			}
			envShellMultisets.set(multiset, t.id);
		}
	}
	if (envClusterSizes.size !== 32) {
		throw new Error(`Expected 32 distinct shells in environment-pool-32, found ${envClusterSizes.size}`);
	}

	// Verify 10 distinct shells in bot pool + 6/6 identical Turn-0 sig, 6/6 identical HP_IVs,
	// 6/6 identical intra-cluster items, and exact intra-shell weight sums = 1.0
	const botClusters = new Map();
	const botShellMultisets = new Map();
	for (const t of botItems) {
		const sets = botOptimized.get(t.id).sets;
		const ck = t.variantCluster;
		const sig = JSON.stringify(sets.map(s => [s.species, s.level, s.gender, Boolean(s.item)]));
		const hpSig = JSON.stringify(sets.map(s => s.ivs.hp));
		const itemSig = JSON.stringify(sets.map(s => s.item));
		const parentRef = envParentSigs.get(ck);
		if (!parentRef) throw new Error(`Bot pool shell ${ck} not found in environment-pool-32`);
		if (sig !== parentRef.t0) throw new Error(`Bot pool team ${t.id} Turn-0 signature diverges from parent ${ck}`);
		if (hpSig !== parentRef.hp) throw new Error(`Bot pool team ${t.id} HP_IV signature diverges from parent ${ck} (Castle stealth break)`);

		if (!botClusters.has(ck)) {
			const multiset = JSON.stringify(sets.map(s => `${s.species}:${s.level}`).sort());
			if (botShellMultisets.has(multiset)) {
				throw new Error(`Duplicate shell in bot pool between ${botShellMultisets.get(multiset)} and ${t.id}`);
			}
			botShellMultisets.set(multiset, t.id);
			botClusters.set(ck, { sig, hpSig, itemSig, weightSum: 0, members: [] });
		} else {
			const cur = botClusters.get(ck);
			if (cur.sig !== sig) throw new Error(`Bot cluster ${ck} Turn-0 signature mismatch on ${t.id}`);
			if (cur.hpSig !== hpSig) throw new Error(`Bot cluster ${ck} HP_IV mismatch on ${t.id}`);
			if (cur.itemSig !== itemSig) throw new Error(`Bot cluster ${ck} intra-cluster item mismatch on ${t.id}`);
		}
		const cur = botClusters.get(ck);
		cur.weightSum += t.intraShellWeight;
		cur.members.push(t.id);
	}
	if (botClusters.size !== 10) {
		throw new Error(`Expected 10 distinct shells in bot pool, found ${botClusters.size}`);
	}
	for (const [ck, info] of botClusters.entries()) {
		if (Math.abs(info.weightSum - 1.0) > 1e-9) {
			throw new Error(`Bot cluster ${ck} intraShellWeight sum = ${info.weightSum} != 1.0`);
		}
	}

	const round = x => Math.round(x * 1e6) / 1e6;
	const numBotShells = botClusters.size;
	const shellWeight = 1.0 / numBotShells;

	const botTeamsOut = botItems.map((t, idx) => {
		const sets = botOptimized.get(t.id).sets;
		const w = round(shellWeight * t.intraShellWeight);
		return {
			id: t.id,
			rank: idx + 1,
			tier: t.tier,
			label: 'selected',
			variantCluster: t.variantCluster,
			mode: t.mode,
			shellWeight: round(shellWeight),
			intraShellWeight: round(t.intraShellWeight),
			nash: false,
			evolved: t.evolved,
			species: sets.map(s => s.species),
			levels: sets.map(s => s.level),
			drawWeight: w,
			score: round(t.score),
			scoreVsPclean7: round(t.scoreVsPclean7),
			scoreVsPlain10: round(t.scoreVsPlain10),
			provenance: t.provenance,
			sets,
		};
	});
	const botWeightSum = botTeamsOut.reduce((acc, t) => acc + t.drawWeight, 0);
	if (Math.abs(botWeightSum - 1.0) > 1e-9) {
		throw new Error(`bot pool drawWeight sum mismatch: ${botWeightSum}`);
	}

	const flagshipTeams = botTeamsOut.map(t => ({
		id: t.id,
		variantCluster: t.variantCluster,
		mode: t.mode,
		origin: t.provenance.family,
		weight: t.drawWeight,
		sets: t.sets,
	}));

	const envTeamsOut = env32Items.map((t, idx) => {
		const sets = envOptimized.get(t.id).sets;
		const w = round(1.0 / (envClusterSizes.size * envClusterSizes.get(t.variantCluster)));
		return {
			id: t.id,
			rank: idx + 1,
			tier: t.tier,
			label: t.inBotPool ? 'selected-bot-shell' : 'environment-only',
			inBotPool: t.inBotPool,
			...(t.envRole ? { envRole: t.envRole } : {}),
			variantCluster: t.variantCluster,
			evolved: t.evolved,
			species: sets.map(s => s.species),
			levels: sets.map(s => s.level),
			drawWeight: w,
			score: round(t.score),
			provenance: t.provenance,
			sets,
		};
	});
	const envWeightSum = envTeamsOut.reduce((acc, t) => acc + t.drawWeight, 0);
	if (Math.abs(envWeightSum - 1.0) > 1e-5) {
		throw new Error(`env32 drawWeight sum mismatch: ${envWeightSum}`);
	}

	const sharedCondition = {
		policyEpoch: 'policy-v4',
		previewIters: 720000,
		battleIters: 243000,
		battleMode: 'adaptive',
		c: 0.4,
		maxTurns: 500,
		previewPolicy: 'SteppedCfPreview (720k: 65% 128i/128t1 Present19 TrajCalib + 35% Turn-1 Double-Oracle LCB-gated Mixed Nash)',
		battlePolicy: 'CfValuePuct (P_net PUCT + 192-dim Siamese CF-V_net v3 + 8r Hybrid Endgame Guard)',
		previewMode: 'strict-live-no-cache',
		dexSha256: sha256File('data/gen2stadium2.json'),
	};

	const certificationSummary = {
		stage6dDomain: 'nc2000-policy-v4-strong10-v2-20261128::stage6d-final-cert',
		stage6dGames: 2400,
		stage6dLiveSides: 4800,
		stage6dBeliefErrors: 0,
		stage6dCapsTies: 0,
		vsPlain10Raw: { games: 1200, U: 0.5250, se: 0.0137, deltaPp: 2.50, z: 1.82, p1s: 0.03425, note: 'includes 108g identical base3xbase3 mirror noise (+8.33 pp)' },
		vsPlain10Denoised: { nonMirrorGames: 1092, U: 0.5175, se: 0.0133, deltaPp: 1.75, z: 1.32, p1s: 0.0937, note: 'replaces exact identical base3xbase3 block (true EV=50.00%) with exact 0.50' },
		vsPlain10Blocks: {
			mod7VsMod7: { games: 588, U: 0.5340, se: 0.0205, deltaPp: 3.40, z: 1.66, p1s: 0.0484 },
			mod7VsBase3: { games: 252, U: 0.5040, se: 0.0345, deltaPp: 0.40, z: 0.12, p1s: 0.4542 },
			base3VsMod7: { games: 252, U: 0.5000, se: 0.0324, deltaPp: 0.00, z: 0.00, p1s: 0.5000 },
			base3VsBase3Mirror: { games: 108, U: 0.5833, se: 0.0453, deltaPp: 8.33, z: 1.84, trueEv: 0.5000 },
		},
		vsCertifiedPlainNash7: { games: 1200, U: 0.4934, se: 0.0145, deltaPp: -0.66, z: -0.46, modified7U: 0.5017 },
		byModeVsPlain10: {
			camouflaged4: 0.5229,
			upgradedSingleton3: 0.5278,
			modified7: 0.5250,
			cleanBase3RawIncludesMirror: 0.5250,
		},
		apexCamoCrnStage6c: {
			cluster: 'apex-mix50-c3r-c5',
			games: 1560,
			u0: 0.4833,
			uMix: 0.5417,
			dcamoPp: 8.75,
			dcamoSePp: 3.87,
			dcamoZ: 2.262,
			dcamoP1s: 0.0119,
			dsuperPp: 7.50,
			fragPp: 3.33,
		},
		campaignTotalStrictLiveGames: 51556,
		castleTournamentSmoke: {
			ruleset: 'castle-tournament',
			policyEpoch: 'castle-policy-v4',
			previewCache: false,
			games: 16,
			teamsCovered: 15,
			liveSides: 32,
			beliefErrors: 0,
			capsTies: 0,
		},
	};

	const catalogDoc = {
		format: 'nc2000-team-pool-v4',
		policyEpoch: 'policy-v4',
		generator: 'tools/build-team-products-v4.js',
		flagship: {
			cluster: 'policy-v4-strong-generalist-10',
			mixture: 'data/team-pool-v4/flagship-mixture.json',
			environmentCatalog: 'data/team-pool-v4/environment-pool-32.json',
			report: 'docs/reports/policy-v4-strong-pool-camo.md',
			priorNashReport: 'docs/reports/policy-v4-nash-certification.md',
			poolPhilosophy: 'Curated 10-shell equal-weight (1/10 = 10% per shell) Strong Generalist Bot Pool. Retreats deliberately from brittle self-play exact-Nash support claims: every shell in this pool is empirically verified broadly strong across the NC2000 policy-v4 metagame, excluding narrow specialist assassins (e.g., rental-cban-13) and lower-tier shells (preserved intact in environment-pool-32.json). Low-poison 6/6-same-item camouflage is applied exclusively to the 4 shells where CRN proved net positive rent without weak-twin drag.',
			verification: '51,556 strict NO-CACHE LIVE policy-v4 games across Stages 1–6d (103,112/103,112 preview.source=live, 0 belief mismatches, 0 caps/ties), culminating in Stage 6d frozen zero-leakage certification (2,400 LIVE games: P_strong10 vs Plain10 de-noised causal lift 51.75% ± 1.33% [+1.75 pp, z = +1.32, p = 0.0937] after fixing exact identical base3xbase3 mirror=50% [raw 1,200g: 52.50% ± 1.37%], with modified-7 vs plain-modified-7 at 53.40% ± 2.05% [+3.40 pp, z = +1.66, p = 0.0484]; holds 49.34% ± 1.45% [modified-7 50.17%] against certified 7-shell plain Nash p_clean7; plus 16-game genuine castle-tournament / castle-policy-v4 live smoke covering all 15/15 teams with 0 errors).',
			provisionalUniformCore: false,
			certifiedNash: false,
			equalWeightStrongPool: true,
			infoRegime: 'I1_disclosed_self_consistent_shell',
			shellCount: 10,
			teamCount: botTeamsOut.length,
			camouflagedShells: ['apex-c04-bug-3atkstar-boomgen', 'pb63-m-gar55', 'rraikou-m-cune50', 'pb85-m-cune50'],
			upgradedSingletonShells: ['pb-131', 'sample-07', 'pb-87'],
			cleanBaseShells: ['pb-6', 'ayagen-m-zap50', 'sample-12'],
			certification: certificationSummary,
		},
		condition: sharedCondition,
		setOptimization: {
			policy: 'training-v3: maximum HP/non-Attack stats and Hidden Power power; maximum usable Attack, otherwise minimum; preserve public gender, shiny status, Hidden Power type, Castle HP_IV stealth, and 6/6 intra-cluster item stealth',
			changedTeams: botItems.filter(t => botOptimized.get(t.id).changedSets > 0).length,
			changedSets: botItems.reduce((sum, t) => sum + botOptimized.get(t.id).changedSets, 0),
			changedGenders: botItems.reduce((sum, t) => sum + botOptimized.get(t.id).changedGenders, 0),
			femalePreference: {
				enabled: attract.preferFemale,
				weightedDirectSources: attract.weightedSources,
				basis: attract.basis,
				priorFingerprint: fnv1a64(bakePriorPath),
			},
		},
		drawRule: 'each of the 10 strong generalist shells (variantCluster) is drawn with equal probability 1/10 = 0.100000, split within camouflaged clusters by empirically verified low-poison weights (apex 50/50 = 0.050 each; pb63 50/50 = 0.050 each; rraikou 80/20 = 0.080/0.020; pb85 50/35/15 = 0.050/0.035/0.015); drawWeight sums to exact 1.000000',
		teams: botTeamsOut,
	};

	const flagshipDoc = {
		name: 'team-pool-v4-policy-v4-strong-generalist-10',
		policyEpoch: 'policy-v4',
		source_solution: 'data/team-pool-v4/team-pool.json',
		environment_catalog: 'data/team-pool-v4/environment-pool-32.json',
		certification_report: 'docs/reports/policy-v4-strong-pool-camo.md',
		infoRegime: 'I1_disclosed_self_consistent_shell',
		certifiedNash: false,
		equalWeightStrongPool: true,
		active_shells: 10,
		support_size: flagshipTeams.length,
		trainingPolicy: 'training-v3',
		measurementStatus: 'Curated 10-shell equal-weight (1/10 per shell, 15 teams) Strong Generalist Bot Pool with targeted low-poison 6/6-same-item camouflage (4 shells) and CRN-verified set upgrades (3 shells). Frozen zero-leakage Stage 6d certification (2,400 strict NO-CACHE LIVE policy-v4 games, 4,800/4,800 live preview sides, 0 belief mismatches, 0 caps/ties): vs uncamouflaged Plain10, modified-7 vs plain-modified-7 scores 53.40% ± 2.05% (+3.40 pp, z = +1.66, p = 0.0484) and de-noised full-pool lift (setting exact identical base3xbase3 mirror=50.00%) is 51.75% ± 1.33% (+1.75 pp, z = +1.32, p = 0.0937; raw 1,200g 52.50% ± 1.37%); vs certified 7-shell plain Nash p_clean7, 49.34% ± 1.45% (modified-7 50.17%). All 22 cut shells preserved intact in data/team-pool-v4/environment-pool-32.json.',
		certification: certificationSummary,
		teams: flagshipTeams,
	};

	const env32Doc = {
		format: 'nc2000-environment-pool-v4',
		policyEpoch: 'policy-v4',
		generator: 'tools/build-team-products-v4.js',
		description: 'Full preserved 32-shell (33-team) NC2000 environment knowledge catalog. Retains all 22 shells trimmed from the 10-shell equal-weight bot pool (narrow specialist assassins such as rental-cban-13 and lower-tier archetypes) as permanent opponent/metagame evaluation knowledge alongside untouched data/belief-pool-v3/.',
		botPool: 'data/team-pool-v4/team-pool.json',
		shells: 32,
		teamsCount: envTeamsOut.length,
		botPoolShells: 10,
		environmentOnlyShells: 22,
		condition: sharedCondition,
		drawRule: 'each variant cluster among the 32 preserved environment shells carries uniform 1/32 = 0.03125 weight (s0053-lefties split 0.015625 + 0.015625); drawWeight sums to 1.000000',
		teams: envTeamsOut,
	};

	write('data/team-pool-v4/team-pool.json', catalogDoc);
	write('data/team-pool-v4/flagship-mixture.json', flagshipDoc);
	write('data/team-pool-v4/environment-pool-32.json', env32Doc);

	const aces = new Set();
	for (const t of botTeamsOut) {
		for (const s of t.sets) {
			if (s.level >= 55) aces.add(s.species);
		}
	}
	console.log(JSON.stringify({
		ok: true,
		format: catalogDoc.format,
		policyEpoch: catalogDoc.policyEpoch,
		bot_shells: numBotShells,
		bot_teams: botTeamsOut.length,
		bot_weight_sum: Number(botWeightSum.toFixed(6)),
		env32_shells: envClusterSizes.size,
		env32_teams: envTeamsOut.length,
		env32_weight_sum: Number(envWeightSum.toFixed(6)),
		distinct_l55_aces_bot_pool: aces.size,
		changed_teams: catalogDoc.setOptimization.changedTeams,
		changed_sets: catalogDoc.setOptimization.changedSets,
		changed_genders: catalogDoc.setOptimization.changedGenders,
	}));
}

main();
