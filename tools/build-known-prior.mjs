// Explicit known-party expansion. Catalog sets/draw weights remain frozen.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const {normalizeTraining} = require('./normalize-training.js');
const wasm = require('../crates/wasm/pkg-node/nc2000_wasm.js');
const id = value => String(value ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const sorted = rows => rows.sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b), 'en'));
const shellOf = rows => JSON.stringify(sorted(rows));
export function teamMaterial(sets, dex) {
 return sets.map(p => {
  const species = id(p.species);
  if (!dex.species[species]) throw Error(`Unknown species: ${species}`);
  const gender = p.gender ?? dex.species[species].gender;
  if (!['M', 'F', 'N'].includes(gender)) throw Error('Explicit gender required');
  const item = id(p.item);
  if (item && !dex.items[item]) throw Error(`Unknown item: ${item}`);
  const moves = p.moves.map(m => {
   const name = id(m);
   if (!dex.moves[name]) throw Error(`Unknown move: ${name}`);
   return name.startsWith('hiddenpower') ? 'hiddenpower' : name;
  });
  return [species, p.level, gender, p.happiness ?? 255,
   stats.map(s => Math.min(31, Math.max(0, p.ivs?.[s] ?? 31)) & 30),
   stats.map(s => Math.min(255, Math.max(0, p.evs?.[s] ?? 0))), item, moves];
 }).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'));
}
const materialShell = material => shellOf(material.map(p => [p[0], p[1], p[2], !!p[6]]));
const read = p => JSON.parse(fs.readFileSync(p));
const digest = s => crypto.createHash('sha256').update(s).digest('hex');
const baselinePath = 'data/belief-pool-v3/bake-prior.json';
const baseline = read(baselinePath), catalog = read('data/team-pool-v3/team-pool.json');
const dex = read('data/gen2stadium2.json'), engineDex = new wasm.Dex(), validator = new wasm.Validator(engineDex);
const identity = sets => JSON.stringify(teamMaterial(sets, dex));
const catalogTypes = new Set(catalog.teams.map(t => identity(t.sets)));
const retained = baseline.teams.filter(t => t.provenance.origin === 'human' || catalogTypes.has(identity(t.sets)));
const removed = baseline.teams.filter(t => !retained.includes(t)).map(t => t.id);
const teams = structuredClone(retained), types = new Set(teams.map(t => identity(t.sets))), ids = new Set(teams.map(t => t.id));
const additions = [], aliases = [], excluded = [];
const sources = [
 ['catalog', 'data/team-pool-v3/team-pool.json'],
 ['inventory', 'data/team-inventory-v1/inventory.json'],
 ['partybox', 'data/partybox-v0/teams.json'],
 ['majinjima', 'data/kodoku-r3/majinjima-assassins.json'],
 ['rentals', 'data/community-rentals-v0/teams.json'],
 ['supplement', 'data/training-v3/supplemental.json'],
];
try {
 for (const [source, path] of sources) for (const record of read(path).teams) {
  if (source === 'inventory' && record.origin !== 'human') continue;
  const originalId = record.id || `rental-cban-${record.cban}`;
  const canonical = JSON.parse(validator.canonicalizeTeam(JSON.stringify(record.sets)));
  if (!canonical.ok) { excluded.push({source: path, id: originalId, findings: JSON.parse(validator.validateTeam(JSON.stringify(record.sets))).findings}); continue; }
  // Catalog and retained entries already carry the chosen training. New source
  // entries use the same normalization without changing loadouts or HP types.
  const sets = source === 'catalog' ? record.sets : normalizeTraining(canonical.team, dex);
  const validation = JSON.parse(validator.validateTeam(JSON.stringify(sets)));
  if (!validation.ok) throw Error(`Normalized party invalid: ${originalId}`);
  const type = identity(sets);
  if (types.has(type)) { aliases.push({source: path, id: originalId}); continue; }
  const partyId = ids.has(originalId) ? `${originalId}-known-${source}` : originalId;
  if (ids.has(partyId)) throw Error(`Duplicate known ID: ${partyId}`);
  const old = baseline.teams.find(t => t.id === originalId);
  const origin = source === 'catalog' ? record.provenance.origin : 'human';
  const sourceWeight = source === 'majinjima' && typeof record.weight === 'number' ? record.weight : null;
  const weight = old?.weight ?? sourceWeight ?? (origin === 'human' ? 1 : .25);
  teams.push({id: partyId, weight, provenance: {origin, source: path, sourceId: originalId}, sets});
  additions.push({id: partyId, source: path, weight, shell: materialShell(teamMaterial(sets, dex))});
  types.add(type); ids.add(partyId);
 }
} finally {validator.free(); engineDex.free();}
for (const t of teams) if (t.provenance.origin !== 'human' && !catalogTypes.has(identity(t.sets))) throw Error(`Off-catalog generated party: ${t.id}`);
for (const t of catalog.teams) if (!types.has(identity(t.sets))) throw Error(`Missing catalog type: ${t.id}`);
const out = {...baseline, generator: 'tools/belief-pool-v3.py build -> tools/build-known-prior.mjs',
 note: 'Known real-world parties including variants; generated parties only when their exact sets occur in the current catalog. Existing admitted weights are retained; new real-world types default to weight 1. Duplicate source aliases do not add mass. Invalid-format parties are listed in known-expansion.json. Catalog and Nash draws are unchanged.',
 membershipPolicy: 'known-real-and-catalog-generated-v1', teams};
const proof = {baseline: baselinePath, inputs: Object.fromEntries([baselinePath, ...sources.map(([,p])=>p)].map(p=>[p,digest(fs.readFileSync(p))])),
 parties: teams.length, human: teams.filter(t=>t.provenance.origin==='human').length,
 generated: teams.filter(t=>t.provenance.origin!=='human').length,
 removedGenerated: removed, additions, duplicateSourceAliases: aliases, excludedInvalidFormat: excluded};
fs.writeFileSync('data/belief-pool-v3/belief-pool.json', JSON.stringify(out,null,1)+'\n');
fs.writeFileSync('data/belief-pool-v3/known-expansion.json', JSON.stringify(proof,null,2)+'\n');
console.log(JSON.stringify({parties:proof.parties,added:additions.length,removed:removed.length,excluded:excluded.length}));
