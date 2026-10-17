# team-pool-v2 — the catalog

`team-pool.json` is generated (`node tools/build-team-products-v2.js`) from
`data/team-selection-v2/selection.json`: the 24 parties labelled `selected`
under `data/team-selection-v2/PROTOCOL.md`, ranked by their confirmation
score at blind:27000:0.4.

## Who reads it

| Consumer | Use |
|---|---|
| web play door `/` (and `?blind`) | the built-in parties a human picks from, the human's Random, and the bot's party every battle and rematch |
| web `?solver` | the team pickers |
| ladder client `tools/ps-client.js` | `--team pool:random` / `pool:ID` |

Same ids and exact sets on every path. It is **not** the bot's model of its
opponent (`data/belief-pool-v3`) and not the `?nash` draw
(`data/meta-nash-v3`). No baked pair table is keyed to it.

## Draw rule

`drawWeight`: every variant cluster among the selected parties is drawn with
equal probability, split equally among its selected members (24 parties in 16
clusters) — near copies of one design do not multiply its share. Both Random
buttons and the ladder's `pool:random` use these weights; the selection's
pool value `V` was measured under the same rule.

`score` on each entry is the confirmation panel score (fresh seeds, stage C)
of the original inventory sets, before the optimization below. No new win-rate
measurement is claimed for the optimized sets.

## Low-Attack optimization

The generator applies `tools/optimize-low-attack.js` to the playable catalog:
14 parties / 27 sets have Attack EV reduced to 0 and Attack DV minimized while
preserving HP DV and the shiny DV predicate. Other stats, moves,
items, levels, team IDs and draw weights are preserved. Sets with a Physical
move (using this generation's categories), any Hidden Power, Metronome,
Mimic, Mirror Move or Transform are excluded. Sleep Talk is allowed when the
set's own moves otherwise qualify. An odd Attack DV preserves maximum HP.

Version 2 allows female preference in these qualifying sets when the frozen
prior's weighted direct-Attract sources favor it. Attract works only between
opposite genders: female is immune to a female user, but susceptible to a
male user. The prior has Ninetales F (team weight 1) and Snorlax M (0.25).
This favors female on a source-weight proxy; it is not a measured encounter
rate or win-rate improvement. The catalog and Nash support contain no direct
Attract, so a catalog-only mirror has no direct Attract advantage either way.

Six formerly male sets now use female DV 1 instead of male DV 3: Venusaur in
`rental-cban-8`, `sample-10`, `mjj-2014-thdrcomb`; Umbreon in `sample-06`;
Jolteon in `mjj-2010-garap-a`; Vaporeon in `rental-cban-14`. HP and every
non-Attack stat stay identical; Attack falls another 2 points. Fixed gender,
genderless and already-female sets keep their gender. Own Attract users keep
their gender, and shiny constraints can prevent a female change. The pure
optimizer preserves gender unless this preference is explicitly enabled.

The original inventory, frozen belief prior, measured Nash mixture and
selection/reference evidence remain unchanged. Only the catalog applies
this policy, so rebuilding products reproduces it. The `setOptimization`
metadata identifies the policy and the historical basis of `score`.

The baseline audit in `findings/low-attack-catalog-v1/audit.json` records all
27 changes and 18,252 native-engine self-hit comparisons (each move, Attack
and Defense stages -6 through +6, no burn/weather/screens): no self-hit
increase and unchanged HP, gender and non-Attack stats in that first pass.
`findings/attract-gender-v1/mechanics.json` independently verifies the six
version-2 changes: nine exact gender-only Attract cases, 4,056 additional
self-hit comparisons, and unchanged HP/non-Attack stats. All 24 optimized
teams pass the engine validator. Struggle also loses power; this is a
practical configuration optimization, not a universal dominance claim.

Checks: `node --test tools/optimize-low-attack.test.js`. Reproduce the baseline
audit using `crates/engine/examples/low_attack_catalog_audit.rs` and an original
catalog input. The existing compiled preview choices may be reused for this
specific optimization, as approved by the user. Run
`node tools/rekey-low-attack-preview.js BASELINE_POOL OUTPUT_ARTIFACT` with
the original cached catalog to update exact own-set keys using the real
engine. The tool requires all other inputs to match and checks the complete
matchup grid and legal selections. `catalogReuse` records the original
artifact/catalog hashes and that the choices were reused without remeasurement.
The gender audit is reproducible with `attract_gender_audit` using the
`d7cb2fc` catalog as the baseline and the current catalog as the new pool.
