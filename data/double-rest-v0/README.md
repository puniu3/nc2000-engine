# Double-Rest endgame corpus v0

Offline research slice for last-Pokemon 1v1 positions where both current
actives are proven to carry Rest by a move reveal somewhere in the complete
spectator log.

`corpus-oracle.csv` contains 310 deduplicated decision states from 11 battles
and 11 unordered species-pair groups. It includes 51 positions where both
actives carry Sleep Talk, 226 where exactly one does, and 33 where neither
does. At the decision cut, both sides are asleep in 14 positions and exactly
one is asleep in 166.

This is an **offline oracle-set slice**: state, HP, status, observed PP, and
all other fields are reconstructed from the legal protocol prefix, then moves
revealed later in the same log are inserted into unused candidate moveslots.
Future information is never exposed through the live-agent reconstruction
path. A Sleep-Talk-called move counts as a set reveal; other called moves do
not.

Rows are grouped by unordered active species pair. `holdout` is a stable 20%
hash split over that group, so adjacent turns and the same species pair never
cross dev/holdout. This is stricter than a position split but not a full
moveset-pair split: spectator logs do not reveal every slot.

Rebuild:

```bash
cargo run --release -p nc2000-bot --example damage_abstraction -- \
  --corpus tmp/corpus-spectator --battles 0-569 \
  --positions 999 --per-battle 99 --alive-max 1 --hp-cap 2000 \
  --both-rest --both-rest-revealed --oracle-future-moves --collect-only \
  --out data/double-rest-v0/corpus-oracle.csv
```

The source spectator logs are private/raw research input and are deliberately
not redistributed with this manifest.
