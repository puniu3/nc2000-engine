# Six-root scheduling comparison

The full control reproduces frozen results. All five arms cover the same six roots at the same execution budget and node stopping threshold. Independent rational matrix checks pass, and all root/cell intervals overlap their prior intervals. No rule or ordinary bot behavior changed.

Enabling only the existing guarded Quick Claw quotient lowers the selected-action loss upper bound on all six diagnostic roots. The previously unresolved game 1 frame 37 reaches a loss upper bound of 0.00965006469544133, compared with 0.025426207619589414 in the control. It remains a fixed-world statement, not a certificate for the blind bot.

Retaining exhausted eager prefixes adds only small improvements in some roots and consumes more nodes. Refining only the selected action helps several roots but severely worsens game 0's upper bound (about 0.65 to about 1.0). Neither addition is selected for the next screen. The simplest quotient-only setting is chosen for the 64 roots outside this diagnostic set from the original 70-root refinement sample, preserving all other settings.

Existing release tests passed for quotient domain guards, full successor-kernel preservation including switching/Thief, retained-prefix exhaustive successor distributions, and exhaustive two-turn policy loss bounds. The six-root comparison is exploratory; no runtime default is changed and it establishes no bot-strength gain.
