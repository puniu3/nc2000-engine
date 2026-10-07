# Common-root impact of the HP category correction

The fixed protocol uses new initial seeds and the existing constructed fixture definitions: 24 stress roots and two on-pool roots immediately after their forced prefixes. Each root gets three paired search seeds at 27,000 iterations, c=0.4. Public HP type inference stays off in both source snapshots. True opponent attributes are recorded only for diagnosis; they are not supplied to search.

The same probe is compiled against baseline a48a40e in an isolated worktree and corrected source 2090bcd (runtime fix fcaa4b8). Source fingerprints must differ only at crates/bot/src/eval.rs. Every paired search must have identical battle history, state key, public observation, legal/eligible actions, search seed, fallback mode, and sampled model attributes/selection counts. Damage estimates and static evaluations may change. The corrected build is repeated with the same inputs; all fields except elapsed time must match.

The smoke covers eight roots at 64 iterations. Before/after input and model checks and corrected-build repeat checks pass. Five mutations (missing result, changed seed, changed model attributes, invalid selected move, changed repeat statistics) are rejected; seed/model/best mutations also alter their repeat copy so agreement with a second wrong record cannot pass them.

This measures whether the correctness fix reaches search decisions, not whether changed choices are better. No outcomes, optimal-action labels, or strength claims are inferred from action differences. Timings are descriptive Mac measurements; budgets are equal iteration counts, not equal time. Historical proof archives and the earlier 104-game trial remain unchanged.
