// Build data/i18n-ja.json — Japanese display names for everything the web
// UI shows by name: the format's 251 species, 267 moves (251 real + 16
// typed Hidden Power variants), 62 items, and 17 types.
//
// One-time generator (M13): fetches official Japanese names from PokéAPI
// (https://pokeapi.co, `names` arrays, language ja-Hrkt with ja fallback)
// and writes a committed artifact — the app never talks to PokéAPI at
// runtime. Keys are PS ids (the keys of data/gen2stadium2.json, which match
// what the wasm bridge emits after toID normalization).
//
// Gen-2-only items PokéAPI doesn't carry (the old berries, bows, Berserk
// Gene, generic Mail) are filled from MANUAL_ITEMS below — names verified
// against Bulbapedia item infoboxes (jname), 2026-07.
//
// Usage: node tools/build-i18n-ja.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DEX = require(path.join(ROOT, 'data', 'gen2stadium2.json'));
const OUT = path.join(ROOT, 'data', 'i18n-ja.json');
const API = 'https://pokeapi.co/api/v2';

// Gen-2 exclusive items missing from PokéAPI (source: Bulbapedia jname).
const MANUAL_ITEMS = {
  berry: 'きのみ',
  berserkgene: 'はかいのいでんし',
  bitterberry: 'にがいきのみ',
  burntberry: 'やけたきのみ',
  goldberry: 'おうごんのみ',
  iceberry: 'こおったきのみ',
  mintberry: 'はっかのみ',
  miracleberry: 'きせきのみ',
  mysteryberry: 'ふしぎなきのみ',
  przcureberry: 'まひなおしのみ',
  psncureberry: 'どくけしのみ',
  pinkbow: 'ピンクのリボン',
  polkadotbow: 'みずたまリボン',
  mail: 'メール', // PS's generic gen-2 Mail item
};

// PS item id -> PokéAPI slug when the name-derived slug doesn't match.
const ITEM_SLUG_OVERRIDES = {};

function slugFromName(name) {
  return name
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/['’.]/g, '') // King's Rock -> kings-rock
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

async function getJson(url, tries = 4) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url);
      if (res.status === 404) return null;
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      if (i === tries - 1) throw new Error(`${url}: ${e.message}`);
      await new Promise((r) => setTimeout(r, 500 * (i + 1)));
    }
  }
}

function jaName(obj) {
  if (!obj || !obj.names) return null;
  const pick = (lang) => obj.names.find((n) => n.language.name === lang);
  const hit = pick('ja-Hrkt') ?? pick('ja');
  return hit ? hit.name : null;
}

/** Run tasks with bounded concurrency. */
async function pool(items, worker, limit = 6) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (next < items.length) {
        const i = next++;
        out[i] = await worker(items[i], i);
      }
    }),
  );
  return out;
}

async function main() {
  const manualFills = [];

  // ---- types (17, keyed by lowercase id; the UI lowercases its
  //      capitalized type strings before lookup)
  const typeIds = Object.keys(DEX.typechart);
  const types = {};
  await pool(typeIds, async (t) => {
    const j = await getJson(`${API}/type/${t}`);
    const name = jaName(j);
    if (!name) throw new Error(`no ja name for type ${t}`);
    types[t] = name;
  });

  // ---- species (251, by national dex num == PokéAPI id)
  const species = {};
  const speciesEntries = Object.entries(DEX.species);
  await pool(speciesEntries, async ([id, s]) => {
    const j = await getJson(`${API}/pokemon-species/${s.num}`);
    const name = jaName(j);
    if (!name) throw new Error(`no ja name for species ${id} (#${s.num})`);
    species[id] = name;
  });

  // ---- moves (251 real moves by move num == PokéAPI id; the 16 typed
  //      Hidden Power variants share num 237 and are composed locally)
  const moves = {};
  const realMoves = Object.entries(DEX.moves).filter(
    ([id]) => !/^hiddenpower.+/.test(id),
  );
  await pool(realMoves, async ([id, m]) => {
    const j = await getJson(`${API}/move/${m.num}`);
    const name = jaName(j);
    if (!name) throw new Error(`no ja name for move ${id} (#${m.num})`);
    moves[id] = name;
  });
  for (const [id, m] of Object.entries(DEX.moves)) {
    if (!/^hiddenpower.+/.test(id)) continue;
    const t = m.type.toLowerCase();
    moves[id] = `${moves.hiddenpower}(${types[t]})`;
    manualFills.push(`moves.${id} (composed: Hidden Power + type)`);
  }

  // ---- items (62; slug from the PS display name, manual for gen-2-only)
  const items = {};
  const itemEntries = Object.entries(DEX.items);
  await pool(itemEntries, async ([id, it]) => {
    if (MANUAL_ITEMS[id]) {
      items[id] = MANUAL_ITEMS[id];
      manualFills.push(`items.${id} (Bulbapedia)`);
      return;
    }
    const slug = ITEM_SLUG_OVERRIDES[id] ?? slugFromName(it.name);
    const j = await getJson(`${API}/item/${slug}`);
    const name = jaName(j);
    if (!name) throw new Error(`no ja name for item ${id} (slug ${slug})`);
    items[id] = name;
  });

  const out = {
    meta: {
      source: 'PokéAPI v2 (https://pokeapi.co), language ja-Hrkt/ja',
      generator: 'tools/build-i18n-ja.js',
      generated: new Date().toISOString().slice(0, 10),
      counts: {
        species: Object.keys(species).length,
        moves: Object.keys(moves).length,
        items: Object.keys(items).length,
        types: Object.keys(types).length,
      },
      manualFills,
    },
    types,
    species,
    moves,
    items,
  };

  // Sanity: full coverage of the dex.
  for (const [table, keys] of [
    [species, Object.keys(DEX.species)],
    [moves, Object.keys(DEX.moves)],
    [items, Object.keys(DEX.items)],
    [types, typeIds],
  ]) {
    for (const k of keys) {
      if (!table[k]) throw new Error(`missing entry: ${k}`);
    }
  }

  fs.writeFileSync(OUT, JSON.stringify(out, null, 1) + '\n');
  console.log(
    `wrote ${OUT}: ${out.meta.counts.species} species, ` +
      `${out.meta.counts.moves} moves, ${out.meta.counts.items} items, ` +
      `${out.meta.counts.types} types (${manualFills.length} manual/composed)`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
