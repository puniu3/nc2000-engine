// Generate the shipped team files of the blind rebuild
// (docs/TEAM-POOL-REBUILD-PLAN.md step 6) from explicit versioned inputs:
//
//   data/team-pool-v2/team-pool.json          the catalog: selected parties —
//                                             human built-in choices and the
//                                             bot's ordinary draw (drawWeight)
//   data/meta-nash-v3/pool-artifact.json      Nash support and weights
//   data/team-selection-v2/reference.json     every inventory record: v1 label,
//                                             selection label and reason,
//                                             memberships and probabilities
//
//   node tools/build-team-products-v2.js [--selection F] [--solution F] [--out-root DIR]
//
// Inputs: data/team-inventory-v1/inventory.json,
// data/team-selection-v2/selection.json, data/meta-nash-v3/solution.json,
// and the frozen prior data/belief-pool-v3/belief-pool.json, which is read
// and checked, never written: prior membership does not follow selection
// labels. The three probability fields stay separate: drawWeight (catalog),
// weight (Nash), priorWeight (reference only).
'use strict';
const fs = require('fs');
const path = require('path');
const { optimizeLowAttack } = require('./optimize-low-attack');

const REPO = path.join(__dirname, '..');
const arg = name => {
	const i = process.argv.indexOf(name);
	return i > 0 ? process.argv[i + 1] : null;
};
const OUT = arg('--out-root') ? path.resolve(arg('--out-root')) : REPO;
const read = rel => JSON.parse(fs.readFileSync(path.resolve(REPO, rel), 'utf8'));
const write = (rel, obj) => {
	fs.mkdirSync(path.dirname(path.join(OUT, rel)), { recursive: true });
	fs.writeFileSync(path.join(OUT, rel), JSON.stringify(obj, null, 1) + '\n');
};
/** The fingerprint team_eval stamps into every record's `cond.belief_*`. */
const fnv1a64 = rel => {
	let h = 0xcbf29ce484222325n;
	for (const b of fs.readFileSync(path.resolve(REPO, rel))) h = ((h ^ BigInt(b)) * 0x100000001b3n) & 0xffffffffffffffffn;
	return `fnv1a64:${h.toString(16).padStart(16, '0')}`;
};
const fail = msg => { throw new Error(msg); };

const inv = read('data/team-inventory-v1/inventory.json');
const v1 = read('data/team-inventory-v1/classification.json');
const sel = read(arg('--selection') || 'data/team-selection-v2/selection.json');
const sol = read(arg('--solution') || 'data/meta-nash-v3/solution.json');
const PRIOR = 'data/belief-pool-v3/belief-pool.json';
const prior = read(PRIOR);
const byId = new Map(inv.teams.map(t => [t.id, t]));

const priorFp = fnv1a64(PRIOR);
if (sel.prior.fingerprint !== priorFp) fail(`selection was measured on another prior (${sel.prior.fingerprint} vs ${priorFp})`);
if (sol.prior.fingerprint !== priorFp) fail(`Nash solution was measured on another prior (${sol.prior.fingerprint} vs ${priorFp})`);

const LABELS = new Set(['selected', 'pending', 'not-selected']);
const measured = inv.teams.filter(t => t.eligibility.status === 'eligible' && t.measuredAs === t.id);
for (const t of measured) {
	const l = sel.labels[t.id];
	if (!l || !LABELS.has(l.label)) fail(`${t.id}: no selection label`);
	if (l.label !== 'selected' && !l.reason) fail(`${t.id}: ${l.label} without a reason`);
}
for (const id of Object.keys(sel.labels)) {
	if (!byId.has(id) || byId.get(id).measuredAs !== id) fail(`selection label for non-measured id ${id}`);
}
const priorIds = new Set(prior.teams.map(t => t.id));
for (const t of measured) if (!priorIds.has(t.id)) fail(`${t.id}: measured but missing from the frozen prior`);

// ------------------------------------------------------------ catalog
const clusterKey = t => t.variantCluster || t.id;
const selected = measured.filter(t => sel.labels[t.id].label === 'selected');
if (selected.length === 0) fail('no selected parties');
selected.sort((a, b) => sel.labels[b.id].score - sel.labels[a.id].score || a.id.localeCompare(b.id));
const size = new Map();
for (const t of selected) size.set(clusterKey(t), (size.get(clusterKey(t)) || 0) + 1);
const clusters = size.size;
const drawWeight = t => 1 / (clusters * size.get(clusterKey(t)));
const round = x => Math.round(x * 1e6) / 1e6;
const provenance = t => ({ family: t.family, origin: t.origin, source: t.source.name, url: t.source.url, version: t.source.version || null });
// Optimize only the playable catalog. All measurements, the Nash solution,
// reference inventory and frozen opponent prior keep their original inputs.
const dex = read('data/gen2stadium2.json');
const optimized = new Map(selected.map(t => [t.id, optimizeLowAttack(t.sets, dex)]));
write('data/team-pool-v2/team-pool.json', {
	format: 'nc2000-team-pool-v2',
	generator: 'tools/build-team-products-v2.js',
	protocol: sel.protocol,
	condition: sel.condition,
	setOptimization: {
		policy: 'low-attack-v1: no Physical move, Hidden Power or external move access; minimum Attack DV preserving HP, gender and shiny status; Attack EV 0',
		changedTeams: selected.filter(t => optimized.get(t.id).changedSets > 0).length,
		changedSets: selected.reduce((sum, t) => sum + optimized.get(t.id).changedSets, 0),
		scoreBasis: 'confirmation scores refer to the original inventory sets, before low-attack-v1; optimized sets have not been remeasured',
	},
	drawRule: 'each variant cluster among the selected parties is drawn with equal probability, split equally among its members (drawWeight sums to 1); the same weights serve the human Random and the bot draw',
	teams: selected.map((t, i) => ({
		id: t.id, rank: i + 1, label: 'selected',
		species: t.species, levels: t.levels,
		drawWeight: round(drawWeight(t)),
		score: round(sel.labels[t.id].score),
		provenance: provenance(t),
		sets: optimized.get(t.id).sets,
	})),
});

// ------------------------------------------------------------ nash
const nash = Object.entries(sol.weights).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1]);
for (const [id] of nash) {
	if (!sel.labels[id] || sel.labels[id].label !== 'selected') fail(`Nash support ${id} is not a selected party`);
}
write('data/meta-nash-v3/pool-artifact.json', {
	format: 'meta-nash-pool-v3',
	source_solution: 'data/meta-nash-v3/solution.json',
	conditions: sol.conditions,
	teams: nash.map(([id, w]) => ({ id, origin: byId.get(id).family, weight: w, sets: byId.get(id).sets })),
});

// ------------------------------------------------------------ reference
const nashW = new Map(nash);
const priorW = new Map(prior.teams.map(t => [t.id, t.weight]));
write('data/team-selection-v2/reference.json', {
	format: 'nc2000-team-reference-v2',
	generator: 'tools/build-team-products-v2.js',
	protocol: sel.protocol,
	counts: {
		records: inv.teams.length,
		eligible: inv.teams.filter(t => t.eligibility.status === 'eligible').length,
		measured: measured.length,
		selected: selected.length,
		pending: measured.filter(t => sel.labels[t.id].label === 'pending').length,
		notSelected: measured.filter(t => sel.labels[t.id].label === 'not-selected').length,
		prior: prior.teams.length,
		nashSupport: nash.length,
	},
	teams: inv.teams.map(t => {
		const rep = t.eligibility.status === 'eligible' ? t.measuredAs : null;
		const l = rep ? sel.labels[rep] : null;
		return {
			id: t.id, family: t.family, origin: t.origin, source: t.source,
			eligibility: t.eligibility,
			measuredAs: t.measuredAs,
			v1Label: rep && v1.labels[rep] ? v1.labels[rep].label : null,
			label: l ? l.label : null,
			reason: l ? l.reason || null : null,
			evidence: l ? { score: l.score ?? null, se: l.se ?? null, stage: l.stage ?? null } : null,
			catalog: !!l && l.label === 'selected' && rep === t.id,
			drawWeight: l && l.label === 'selected' && rep === t.id ? round(drawWeight(t)) : 0,
			priorWeight: priorW.get(t.id) || 0,
			nashWeight: nashW.get(t.id) || 0,
			variantCluster: t.variantCluster,
			species: t.species, levels: t.levels,
		};
	}),
});

console.log(`catalog ${selected.length} (${clusters} clusters) | prior ${prior.teams.length} (frozen) | nash ${nash.length} | reference ${inv.teams.length}`);
