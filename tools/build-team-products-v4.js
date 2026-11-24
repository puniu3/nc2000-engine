// Generate the `team-pool-v4` (`policy-v2`: CF-V_net v2 PUCT + SteppedCfPreview 240k)
// strong party catalog and provisional Tier-S flagship mixture from committed repo
// sources with `training-v3` normalization and strict WASM + JP-NC2000 legality checks.
//
// Outputs:
//   data/team-pool-v4/team-pool.json         33-team (32-shell) policy-v2 Tier-1 catalog
//   data/team-pool-v4/flagship-mixture.json  5-shell provisional Tier-S core mixture (policy-v2)
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

// Curated 33-team / 32-shell `policy-v2` Tier-1 roster:
//   - 9 causally verified `policy-v2` upgrades (docs/reports/okimono-upgrades-v2.json),
//     each showing positive in-battle CRN `InΔ` under strict NO-CACHE LIVE (243k/240k);
//     both earlier minor-gem candidates (`pb84-cln-sking50` and `pb63-m-esp55`) were retracted
//     after the 1,840g Step 4 true shared-seed CRN audit exposed hitchhiking/confounding and
//     replaced by their true-CRN Major winners (`pb84-cln-skarm50` and `pb63-m-gar55`).
//   - 23 preserved clean/unbeaten `team-pool-v3` shells (including 2-variant `s0053-lefties`).
//   - Tier S (`nash: true`) is a provisional uniform 5-shell core (NOT a certified policy-v2
//     Nash equilibrium; >=48g/edge policy-v2 Nash certification including `pb63-m-gar55` and
//     `pb84-cln-skarm50` is pending).
const ROSTER_SPEC = [
	// Tier S: Provisional 5-Shell `policy-v2` Core (uniform 0.200000 each; full >=48g/edge Nash RR pending)
	{ id: 'rraikou-m-cune50', tier: 'S', nash: true, nashWeight: 0.200000, evolved: true, score: 0.6019, fromUpgrade: true },
	{ id: 'pb122-m-lax50', tier: 'S', nash: true, nashWeight: 0.200000, evolved: true, score: 0.6250, fromUpgrade: true },
	{ id: 'pb63-m-gar55', tier: 'S', nash: true, nashWeight: 0.200000, evolved: true, score: 0.5938, fromUpgrade: true },
	{ id: 'pb66-m-wak55', tier: 'S', nash: true, nashWeight: 0.200000, evolved: true, score: 0.6146, fromUpgrade: true },
	{ id: 'sample-08', tier: 'S', nash: true, nashWeight: 0.200000, score: 0.6129 },

	// Tier 1A: Verified `policy-v2` Top Contenders & Camouflage Core (13 shells / 14 teams)
	{ id: 'ayagen-m-zap50', tier: '1', evolved: true, score: 0.6562, fromUpgrade: true },
	{ id: 'pb84-cln-skarm50', tier: '1', evolved: true, score: 0.5391, fromUpgrade: true },
	{ id: 'pb-6', tier: '1', score: 0.5968 },
	{ id: 's0053-lefties-hpi', tier: '1', variantCluster: 'vc-s0053-lefties', evolved: true, score: 0.5873 },
	{ id: 's0053-lefties-dp', tier: '1', variantCluster: 'vc-s0053-lefties', evolved: true, score: 0.5873 },
	{ id: 'pb-89', tier: '1', score: 0.6290 },
	{ id: 'mjj-2014-thdrcomb', tier: '1', score: 0.5968 },
	{ id: 'sample-06', tier: '1', score: 0.5806 },
	{ id: 's02-m-lax-curse', tier: '1', evolved: true, score: 0.5781, fromUpgrade: true },
	{ id: 'sample-07', tier: '1', score: 0.5645 },
	{ id: 'sample-11', tier: '1', score: 0.5484 },
	{ id: 'offantom-m-wak50', tier: '1', evolved: true, score: 0.5312, fromUpgrade: true },
	{ id: 'sample-17-orig', tier: '1', score: 0.5161 },
	{ id: 'rental-cban-13', tier: '1', score: 0.5161 },

	// Tier 1B: Strong Diverse NC2000 Archetypes (14 distinct shells)
	{ id: 'rental-cban-8', tier: '1', score: 0.5000 },
	{ id: 'deme-r1-00-s0030', tier: '1', evolved: true, score: 0.5000 },
	{ id: 'mjj-type-QA-moudoku', tier: '1', score: 0.4839 },
	{ id: 'sample-14', tier: '1', score: 0.4839 },
	{ id: 'sample-12', tier: '1', score: 0.4839 },
	{ id: 'pb85-m-cune50', tier: '1', evolved: true, score: 0.4688, fromUpgrade: true },
	{ id: 'pb-131', tier: '1', score: 0.4677 },
	{ id: 'pb-87', tier: '1', score: 0.4677 },
	{ id: 'pb-96', tier: '1', score: 0.4677 },
	{ id: 'sample-22-orig', tier: '1', score: 0.4677 },
	{ id: 'apex-c04-bug-3atkstar-boomgen', tier: '1', evolved: true, score: 0.4355 },
	{ id: 'hc75-top8-tsuru', tier: '1', score: 0.4194 },
	{ id: 'sample-09', tier: '1', score: 0.4032 },
	{ id: 'mjj-2010-garap-a', tier: '1', score: 0.3387 },
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

function main() {
	const dex = read('data/gen2stadium2.json');
	const v3 = read('data/team-pool-v3/team-pool.json');
	const up2 = read('docs/reports/okimono-upgrades-v2.json');
	const bakePriorPath = 'data/belief-pool-v3/bake-prior.json';
	const bakePrior = read(bakePriorPath);

	const v3ById = new Map(v3.teams.map(t => [t.id, t]));
	const up2ById = new Map(up2.teams.map(t => [t.id, t]));

	const selected = ROSTER_SPEC.map(spec => {
		if (spec.fromUpgrade) {
			const u = up2ById.get(spec.id);
			if (!u) throw new Error(`Upgrade team not found in okimono-upgrades-v2.json: ${spec.id}`);
			const parent = v3ById.get(u.parent);
			const provenance = {
				family: 'policy-v2-okimono-refresh',
				origin: 'evolved',
				parent: u.parent,
				category: u.category,
				...(spec.minorGem ? { minorGem: spec.minorGem } : {}),
				source: `NC2000 policy-v2 Refresh (${u.parent} -> ${spec.id}: ${u.change})`,
				url: 'docs/reports/okimono-upgrades-v2.json',
				version: `v4-${spec.id}`,
				...(parent?.provenance ? { parentSource: parent.provenance.source } : {}),
			};
			return {
				id: spec.id,
				tier: spec.tier,
				variantCluster: spec.variantCluster || spec.id,
				nash: Boolean(spec.nash),
				nashWeight: spec.nashWeight || 0,
				evolved: true,
				score: spec.score,
				provenance,
				sets: structuredClone(u.sets),
			};
		}
		const src = v3ById.get(spec.id);
		if (!src) throw new Error(`Preserved team not found in team-pool-v3: ${spec.id}`);
		return {
			id: spec.id,
			tier: spec.tier,
			variantCluster: spec.variantCluster || spec.id,
			nash: Boolean(spec.nash),
			nashWeight: spec.nashWeight || 0,
			evolved: Boolean(spec.evolved || src.evolved),
			score: spec.score,
			provenance: src.provenance,
			sets: structuredClone(src.sets),
		};
	});

	// Canonicalize, training-v3 normalize, and strictly validate via WASM Validator + Battle
	const attract = attractGenderCensus(bakePrior);
	const engineDex = new wasm.Dex();
	const validator = new wasm.Validator(engineDex);
	let optimized;
	try {
		for (const t of selected) {
			const canon = JSON.parse(validator.canonicalizeTeam(JSON.stringify(t.sets)));
			if (!canon.ok) {
				throw new Error(`WASM canonicalizeTeam rejected ${t.id}: ${JSON.stringify(canon.errors)}`);
			}
			t.sets = canon.team;
		}
		optimized = normalizeCatalog(selected, dex, attract.preferFemale);
		for (const t of selected) {
			const sets = optimized.get(t.id).sets;
			checkJpNc2000Legality(t.id, sets);
			const res = JSON.parse(validator.validateTeam(JSON.stringify(sets)));
			if (!res.ok) {
				throw new Error(`WASM Validator rejected ${t.id}: ${JSON.stringify(res.findings || res.errors)}`);
			}
			const b = new wasm.Battle(engineDex, JSON.stringify(sets), JSON.stringify(sets), '1,2,3,4');
			b.free();
		}
	} finally {
		validator.free();
		engineDex.free();
	}

	// Verify pairwise Turn-0 shell uniqueness across clusters & byte-identical camouflage within clusters
	const clusterSizes = new Map();
	const clusterPreviewSigs = new Map();
	const seenShellMultisets = new Map();
	for (const t of selected) {
		const sets = optimized.get(t.id).sets;
		const ck = t.variantCluster;
		clusterSizes.set(ck, (clusterSizes.get(ck) || 0) + 1);
		const previewSig = JSON.stringify(sets.map(s => [s.species, s.level, s.gender, Boolean(s.item)]));
		if (clusterPreviewSigs.has(ck)) {
			if (clusterPreviewSigs.get(ck) !== previewSig) {
				throw new Error(`Camouflage cluster ${ck} has mismatched Turn-0 preview signature on ${t.id}`);
			}
		} else {
			clusterPreviewSigs.set(ck, previewSig);
			const multiset = JSON.stringify(sets.map(s => `${s.species}:${s.level}`).sort());
			if (seenShellMultisets.has(multiset)) {
				throw new Error(`Duplicate shell between ${seenShellMultisets.get(multiset)} and ${t.id}: ${multiset}`);
			}
			seenShellMultisets.set(multiset, t.id);
		}
	}

	const numClusters = clusterSizes.size;
	if (numClusters !== 32) {
		throw new Error(`Expected 32 distinct shells, found ${numClusters}`);
	}
	const round = x => Math.round(x * 1e6) / 1e6;
	const teamsOut = selected.map((t, idx) => {
		const sets = optimized.get(t.id).sets;
		const w = 1 / (numClusters * clusterSizes.get(t.variantCluster));
		return {
			id: t.id,
			rank: idx + 1,
			tier: t.tier,
			label: 'selected',
			nash: t.nash,
			evolved: t.evolved,
			species: sets.map(s => s.species),
			levels: sets.map(s => s.level),
			drawWeight: round(w),
			score: round(t.score),
			provenance: t.provenance,
			sets,
		};
	});

	const weightSum = teamsOut.reduce((acc, t) => acc + t.drawWeight, 0);
	if (Math.abs(weightSum - 1.0) > 1e-5) {
		throw new Error(`drawWeight sum mismatch: ${weightSum}`);
	}

	const flagshipTeams = selected
		.filter(t => t.nash)
		.map(t => ({
			id: t.id,
			origin: t.provenance.family,
			weight: round(t.nashWeight),
			sets: optimized.get(t.id).sets,
		}));
	const nashSum = flagshipTeams.reduce((acc, t) => acc + t.weight, 0);
	if (Math.abs(nashSum - 1.0) > 1e-5) {
		throw new Error(`flagship nash weight sum mismatch: ${nashSum}`);
	}

	const catalogDoc = {
		format: 'nc2000-team-pool-v4',
		policyEpoch: 'policy-v2',
		generator: 'tools/build-team-products-v4.js',
		flagship: {
			cluster: 'policy-v2-provisional-core-5',
			mixture: 'data/team-pool-v4/flagship-mixture.json',
			verification: '10,152 strict NO-CACHE LIVE games (243k adaptive CF-V_net v2 PUCT / 240k SteppedCfPreview, preview.source=live, cache_hits=0) + 19,788 exact 240k previews (15,016 unique-key), including the 1,840g Step 4 true shared-seed CRN audit that retracted pb84-cln-sking50 and pb63-m-esp55 in favor of pb84-cln-skarm50 and pb63-m-gar55',
			provisionalUniformCore: true,
			certifiedMinorGems: [],
			retractedMinorCandidates: ['pb84-cln-sking50 (Slowking L50)', 'pb63-m-esp55 (Espeon L55)'],
			v4CensusZeroOkimonos: 0,
			v4CensusLt5Okimonos: 2,
			v4CensusMeanMinSlotPickRate: 0.1539,
		},
		condition: {
			policyEpoch: 'policy-v2',
			previewIters: 240000,
			battleIters: 243000,
			battleMode: 'adaptive',
			c: 0.4,
			maxTurns: 500,
			previewPolicy: 'SteppedCfPreview (CF-V_net v2 + Double-Oracle + LCB-gated Mixed Nash)',
			battlePolicy: 'CfValuePuct (P_net PUCT + 144-dim Siamese CF-V_net v2 + 8r Hybrid Endgame Guard)',
			previewMode: 'strict-live-no-cache',
			dexSha256: sha256File('data/gen2stadium2.json'),
		},
		setOptimization: {
			policy: 'training-v3: maximum HP/non-Attack stats and Hidden Power power; maximum usable Attack, otherwise minimum; preserve public gender, shiny status, Hidden Power type and camouflage',
			changedTeams: selected.filter(t => optimized.get(t.id).changedSets > 0).length,
			changedSets: selected.reduce((sum, t) => sum + optimized.get(t.id).changedSets, 0),
			changedGenders: selected.reduce((sum, t) => sum + optimized.get(t.id).changedGenders, 0),
			femalePreference: {
				enabled: attract.preferFemale,
				weightedDirectSources: attract.weightedSources,
				basis: attract.basis,
				priorFingerprint: fnv1a64(bakePriorPath),
			},
		},
		drawRule: 'each variant cluster among the 32 distinct shells is drawn with equal probability (1/32 = 0.03125), split equally among its members (s0053-lefties 2-variant cluster = 0.015625 each); drawWeight sums to 1.000000',
		teams: teamsOut,
	};

	const flagshipDoc = {
		name: 'team-pool-v4-policy-v2-provisional-core-5',
		policyEpoch: 'policy-v2',
		source_solution: 'data/team-pool-v4/team-pool.json',
		active_shells: flagshipTeams.length,
		support_size: flagshipTeams.length,
		trainingPolicy: 'training-v3',
		measurementStatus: 'Provisional uniform 5-shell Tier-S core (0.200000 each; NOT a certified policy-v2 Nash equilibrium). The earlier 12g/edge 10x10 RR omitted pb63-m-gar55 and pb84-cln-skarm50 and had 90% bootstrap weight CIs starting at 0%; pb84-cln-sking50 and pb63-m-esp55 were retracted after the 1,840g Step 4 true shared-seed CRN audit. Full >=48g/edge policy-v2 Nash certification pending.',
		teams: flagshipTeams,
	};

	write('data/team-pool-v4/team-pool.json', catalogDoc);
	write('data/team-pool-v4/flagship-mixture.json', flagshipDoc);

	const aces = new Set();
	for (const t of teamsOut) {
		for (const s of t.sets) {
			if (s.level >= 55) aces.add(s.species);
		}
	}
	console.log(JSON.stringify({
		ok: true,
		format: catalogDoc.format,
		policyEpoch: catalogDoc.policyEpoch,
		teams: teamsOut.length,
		clusters: numClusters,
		flagship_shells: flagshipTeams.length,
		flagship_weight_sum: Number(nashSum.toFixed(6)),
		draw_weight_sum: Number(weightSum.toFixed(6)),
		distinct_l55_aces: aces.size,
		changed_teams: catalogDoc.setOptimization.changedTeams,
		changed_sets: catalogDoc.setOptimization.changedSets,
		changed_genders: catalogDoc.setOptimization.changedGenders,
	}));
}

main();
