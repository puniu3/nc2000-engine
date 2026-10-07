# Thick Club HP regions: fixed engine-work follow-up

The opt-in extension produced 230 actual cache hits, all at game 10/frame 32.
At 250,000 engine executions, the selected action's conditional loss interval
changed from [0, 0.06826167249688964] to [0, 0.06816043268501848].
The other five roots were unchanged. This is no new small-loss certificate and
does not establish a better bot or a general solver speedup.

Serial ABBA repeats matched exactly apart from elapsed time. The exposed root
took about 7.99% longer on average at equal engine-work budget. Total time across
the six roots rose about 0.77%. Candidate admission was 417 of 8,210 calls;
405 admitted calls were at the exposed root. Peak retained leaves there: 16,380.

The extension retains Thick Club in the state key and in engine execution.
Only the optional item admission changes. The original guard stays unchanged.
The preceding actual-call profile identified 327 newly admissible calls and
103,687 engine executions at this root. Relevant release tests passed (25),
including 72 complete HP-grid distribution comparisons and item-identity checks.

Every one of the 230 reused distributions was subsequently compared with direct
engine enumeration under the existing guarded semantic identity. Maximum
probability difference was zero. This required 52,336 additional verification
executions, excluded from performance. Search outputs matched the saved candidate.
Independent rational matrix checks and corruption rejection checks also passed.

Protocols and helpers were committed before execution. Raw records are archived
as deterministic gzip files; hashes in summaries refer to decompressed bytes.
`tools/native-club-regions-audit` checks the comparison and
`tools/native-club-verification-audit` checks the direct verification.
The equal-time follow-up is in `../native-club-time-v1`.

Scope remains a fixed true hidden state, full-information continuation, and the
solver's independent chance model. Ordinary bot and main bound solver are
unchanged; the optional connection exists only in the isolated research adapter.
