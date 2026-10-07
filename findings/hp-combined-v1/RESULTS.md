# Completed paired comparison

Candidate (category correction plus public type inference) scored 94 wins, 96 losses and 2 ties in all 192 preregistered games: score 0.4947916667. The prespecified pair-based betting95 interval is [0.3949936280, 0.5947503179], so superiority over the pre-category local baseline is not established. This is inconclusive rather than an equivalence result.

All 96 swapped pairs passed the frozen audit. The 13,376 decisions were legal, had zero protocol drift/projections, matched the fixed iteration budget or an identified forced choice, and used the intended feature settings. Public type constraints were active on the candidate in 32 games and 298 decisions; the baseline never used constraints.

Games 54 and 55 reached the engine's turn-limit tie at turn 1001. They are retained at score 0.5. The final outstanding game, 168, ended after 313 turns. No trial was discarded or replaced and no sample extension was made.

Keep type inference disabled in the ordinary bot. The rules-based damage correction remains a correctness fix, but its strength contribution is not isolated by this combined test. This study uses ProtocolAgent and does not establish deployed native bot superiority, human superiority or proximity to the full game's ceiling. The separately registered native current-default versus deployed comparison continues.
