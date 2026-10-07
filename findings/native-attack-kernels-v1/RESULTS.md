# Complete survivor kernels

The six frozen native graphs reproduce prior loss results exactly. The largest eight retained HP families per graph supply 384 states and 768 representative legal action pairs. Only 12 action pairs in two source games meet the existing HP transition guard, yielding 96 benchmark cases.

Three kernel tests cover complete distribution agreement, caps/storage/evictions and all ten admitted moves crossed with four item/status conditions in both orientations. The implementation is additive and unused by ordinary bots or bound search.

The initial benchmark stopped on a raw-state-key mismatch in the existing script cache. All 350 normalized successor states and their probabilities agree; the only full serialized-state differences are two magnitudes in a stale attacked_by record (42 versus 86). The existing no-Counter/no-Mirror-Coat guard permits omission of this record. Raw failure, diagnostic snapshots and the follow-up guarded-key protocol are preserved. No guard or game rule changed.

All 576 successor distributions in the corrected control/scripts/kernel/kernel/scripts/control run have identical guarded probability-map hashes. Survivor-only kernels have zero hits: every complete source distribution contains at least one KO, so all 96 attempts are rejected. Both kernel passes use the same 35,815 executions as control. Script caching reduces executions to 28,675 but takes about 9% longer in this small measurement.

Do not integrate the survivor-only kernel into bound search based on this result. The measured obstruction is KO branches inside otherwise reusable HP families. The follow-up prototype partitions HP by preserved per-leaf survival/KO pattern, retaining ordinary enumeration outside each certified region. No bot strength or overall solver-speed improvement is established.
