// Generate PRNG parity vectors from the reference Gen5RNG/PRNG implementation.
'use strict';
const fs = require('fs');
const path = require('path');
const { prng } = require('./ps');
const { PRNG, Gen5RNG } = prng;

const vectors = [];

// Raw LCG output (upper 32 bits) from assorted seeds.
for (const seed of [[0, 0, 0, 1], [0x5d58, 0x8b65, 0x6c07, 0x8965], [1, 2, 3, 4], [0xffff, 0xffff, 0xffff, 0xffff]]) {
	const rng = new Gen5RNG(seed);
	vectors.push({ kind: 'next', seed: seed.join(','), values: Array.from({ length: 32 }, () => rng.next()) });
}

// PRNG.random(n) — the (value * n) >> 32 mapping.
{
	const p = new PRNG('gen5,0001000200030004');
	vectors.push({ kind: 'random_n', seed: 'gen5,0001000200030004', n: 16, values: Array.from({ length: 32 }, () => p.random(16)) });
}
{
	const p = new PRNG('1,2,3,4');
	vectors.push({ kind: 'random_range', seed: '1,2,3,4', from: 85, to: 101, values: Array.from({ length: 32 }, () => p.random(85, 101)) });
}

// randomChance
{
	const p = new PRNG('gen5,00ff00ff00ff00ff');
	vectors.push({ kind: 'random_chance', seed: 'gen5,00ff00ff00ff00ff', num: 63, den: 256, values: Array.from({ length: 32 }, () => p.randomChance(63, 256)) });
}

// shuffle (speed-tie resolution) and sample
{
	const p = new PRNG('gen5,1234123412341234');
	const runs = [];
	for (let i = 0; i < 8; i++) {
		const arr = Array.from({ length: 10 }, (_, j) => j);
		p.shuffle(arr);
		runs.push(arr);
	}
	vectors.push({ kind: 'shuffle', seed: 'gen5,1234123412341234', size: 10, runs });
}
{
	const p = new PRNG('gen5,000000000000002a');
	vectors.push({ kind: 'sample', seed: 'gen5,000000000000002a', size: 7, values: Array.from({ length: 32 }, () => p.sample([0, 1, 2, 3, 4, 5, 6])) });
}

// Seed state after N draws (serialization parity).
{
	const p = new PRNG('gen5,0001000200030004');
	for (let i = 0; i < 100; i++) p.random(16);
	vectors.push({ kind: 'seed_after', seed: 'gen5,0001000200030004', draws: 100, endSeed: p.getSeed() });
}

const dest = path.join(__dirname, '..', 'fixtures', 'prng-vectors.json');
fs.writeFileSync(dest, JSON.stringify(vectors, null, 1));
console.log(`wrote ${dest}: ${vectors.length} vector sets`);
