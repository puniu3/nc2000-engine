'use strict';
const { createHash } = require('node:crypto');

function provenance(artifacts, configuration) {
	const hashes = Object.fromEntries(Object.entries(artifacts).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
		.map(([name, bytes]) => [name, bytes === null ? null : createHash('sha256').update(bytes).digest('hex')]));
	const record = { schema: 'nc2000-client-provenance-v1', artifacts: hashes, configuration: JSON.parse(JSON.stringify(configuration)) };
	return { ...record, id: createHash('sha256').update(JSON.stringify(record)).digest('hex') };
}

module.exports = { provenance };
