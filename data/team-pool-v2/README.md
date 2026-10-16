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
preserving HP DV, gender, and the shiny DV predicate. Other stats, moves,
items, levels, team IDs and draw weights are preserved. Sets with a Physical
move (using this generation's categories), any Hidden Power, Metronome,
Mimic, Mirror Move or Transform are excluded. Sleep Talk is allowed when the
set's own moves otherwise qualify. An odd Attack DV preserves maximum HP;
the male Venusaur and Eevee evolutions here need DV 3, rather than DV 1.

The original inventory, frozen belief prior, measured Nash mixture and
selection/reference evidence remain unchanged. Only the catalog applies
this policy, so rebuilding products reproduces it. The `setOptimization`
metadata identifies the policy and the historical basis of `score`.

The baseline audit in `findings/low-attack-catalog-v1/audit.json` records all
27 changes and 18,252 native-engine self-hit comparisons (each move, Attack
and Defense stages -6 through +6, no burn/weather/screens): no self-hit
increase and unchanged HP, gender and non-Attack stats. All 24 optimized
teams pass the engine validator. Struggle also loses power; this is a
practical configuration optimization, not a universal dominance claim.

Checks: `node --test tools/optimize-low-attack.test.js`. Reproduce the baseline
audit using `crates/engine/examples/low_attack_catalog_audit.rs` and an original
catalog input. Exact-own-set preview caches need regeneration or their
existing mismatch fallback when these catalog sets are integrated.
