import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';

const read = p => JSON.parse(fs.readFileSync(new URL(`../${p}`, import.meta.url)));
const id = value => String(value ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
function teamMaterial(sets, dex) {
  return sets.map(p => {
    const species = id(p.species);
    const gender = p.gender ?? dex.species[species].gender;
    const item = id(p.item);
    const moves = p.moves.map(m => {
      const name = id(m);
      return name.startsWith('hiddenpower') ? 'hiddenpower' : name;
    });
    return [species, p.level, gender, p.happiness ?? 255,
      stats.map(s => Math.min(31, Math.max(0, p.ivs?.[s] ?? 31)) & 30),
      stats.map(s => Math.min(255, Math.max(0, p.evs?.[s] ?? 0))), item, moves];
  }).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'));
}

test('deployed expansion keeps only catalog-generated types for team-pool-v3', () => {
  const dex = read('data/gen2stadium2.json');
  const pool = read('data/belief-pool-v3/belief-pool.json');
  const catalog = read('data/team-pool-v3/team-pool.json');
  const types = new Set(catalog.teams.map(t => JSON.stringify(teamMaterial(t.sets, dex))));
  for (const t of pool.teams) {
    if (t.provenance.origin !== 'human') {
      assert.ok(types.has(JSON.stringify(teamMaterial(t.sets, dex))), t.id);
    }
  }
  assert.ok(pool.teams.some(t => t.id === 's0053-lefties-hpi'));
  assert.ok(pool.teams.some(t => t.id === 's0053-lefties-dp'));
  assert.ok(pool.teams.some(t => t.id === 'mjj-2022-nrsk'));
  assert.ok(pool.teams.some(t => t.id === 'mjj-gym-Tokiwa'));
  assert.ok(pool.teams.some(t => t.id === 'mjj-2026-kabegon'));
  assert.ok(pool.teams.some(t => t.id === 'mjj-type-QA-fuhen'));
  assert.ok(pool.teams.some(t => t.id === 'mjj-2022-hapimu-v1' && t.weight < 1));
  assert.equal(pool.teams.length, 520);
});
