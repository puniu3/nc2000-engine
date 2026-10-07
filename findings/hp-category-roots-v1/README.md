# Common-root impact of the HP category correction

The fixed protocol uses new initial seeds and the existing constructed fixture definitions: 24 stress roots and two on-pool roots immediately after their forced prefixes. Each root gets three paired search seeds at 27,000 iterations, c=0.4. Public HP type inference stays off in both source snapshots. True opponent attributes are recorded only for diagnosis; they are not supplied to search.

The same probe is compiled against baseline a48a40e in an isolated worktree and corrected source 2090bcd (runtime fix fcaa4b8). Source fingerprints must differ only at crates/bot/src/eval.rs. Every paired search must have identical battle history, state key, public observation, legal/eligible actions, search seed, fallback mode, and sampled model attributes/selection counts. Damage estimates and static evaluations may change. The corrected build is repeated with the same inputs; all fields except elapsed time must match.

The smoke covers eight roots at 64 iterations. Before/after input and model checks and corrected-build repeat checks pass. Five mutations (missing result, changed seed, changed model attributes, invalid selected move, changed repeat statistics) are rejected; seed/model/best mutations also alter their repeat copy so agreement with a second wrong record cannot pass them.

This measures whether the correctness fix reaches search decisions, not whether changed choices are better. No outcomes, optimal-action labels, or strength claims are inferred from action differences. Timings are descriptive Mac measurements; budgets are equal iteration counts, not equal time. Historical proof archives and the earlier 104-game trial remain unchanged.


## Result

All 78 source-paired searches and all 78 corrected-build repeats completed. Choices changed in 13 pairs across seven of the 26 roots: Articuno 0/24, Dragonair 12/24 (six roots), Exeggutor 1/24 (one root), on-pool control 0/6. Full visit/mean vectors changed in 56/78 pairs. Corrected-build repeats matched every field except timing. Five negative controls passed.

The independent 64-seed model probes have identical attributes and counts before/after; their damage/static-evaluation outputs may differ. These probes establish unchanged model inputs, not identical determinization sequences throughout the adaptive searches. Both snapshots use the ordinary default with public HP type inference off.

All 24 stress roots still model the active foe's HP as Dark while the constructed truth is Fighting or Bug; all 1,536 probed samples omit at least one actual move. These deliberately constructed fixtures are not a prevalence estimate. At Dragonair trial 0, the modeled HP estimate changes from 76.8157 to 24.0627 damage against Blissey, while applying the same estimator to the actual Fighting set gives 149.9294 in both snapshots. The former corrects the category of the sampled Dark move; it does not repair the wrong sampled type. These are approximate estimator outputs, not native damage certificates. The sampled Dragonair has only HP and no held item; the actual set also has Attract, Thunder Wave and Rest.

The correctness fix reaches decisions, but no better-action labels or whole-game gain follow from that fact. The next diagnostic is public type evidence under the corrected estimator, with remaining loadout uncertainty kept explicit. The earlier type-evidence trial used the old estimator and cannot settle this comparison.

`records.json.gz` includes both source manifests, before/after searches and corrected repeats. `diagnostics.json` is reproduced by `tools/hp-category-roots-diagnostics`; it counts distinct roots without tripling the repeated model samples. The baseline checkout remains isolated under tmp/hp-category-baseline. Existing running previews are untouched.
