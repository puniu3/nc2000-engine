# Higher-cap native first-step census

All 360 roots were rerun at 4,096 engine executions per joint action: 5,352 cells,
4,182,554 executions, 4,717 complete cells and 635 incomplete cells. Every one of
the original 3,570 complete cells reproduced exactly. All 21,408 outcome-probability
interval comparisons overlap; 1,147 formerly partial cells are now complete.

The same conservative ranking now nominates 16 alternatives at 15 roots. All
original nominees remain. The largest new difference is game87/frame49: Double
Edge wins immediately with probability at least 0.7288161056732747 against every
opponent first reply, while the selected Rest has no immediate win. This is not
proof that Rest loses long-term value; healing and later play remain relevant.
The next largest new difference is game113/frame37, Hydro Pump versus Ice Beam.

Ten nominated roots were absent from the previous 70-root deeper loss screen.
The follow-up in native-first-step-loss-v1 deepens all ten using the unchanged
frozen loss solver, without selecting by its outcomes. All incomplete cells and
all nominees remain archived. More candidates may still be hidden by unresolved
first-step mass; this is not a complete characterization of endgame mistakes.

The inherited checker passes the coverage, probability and ranking controls.
The follow-up checker also rejects wrong-root, altered-complete-cell and disjoint
interval records. No ordinary bot, rule, or public deployment changed.
