# NC2000 meta team pool v0

Curated team/set pool for M8 (the product distribution replacing the random
fixture pool). Built autonomously from public tournament + expert sources with
mechanical filtering criteria (owner-level meta judgment was explicitly
delegated, 2026-07-16); every inclusion/exclusion decision below is traceable
to a rule, not taste.

Regenerate: `node tools/build-meta-pool.js` (needs `PS_ROOT`, `node build` done).
Engine smoke test: `cargo test -p conformance --test meta_pool` — every team
loads as `PokemonSet`s and plays random full games to completion.

## Contents

- `meta-pool.json` — 34 validated teams, ranked. Per team: provenance,
  pedigree scores, PS export text, PS packed string, and `sets` in the exact
  JSON shape of fixture `p1team` (directly deserializable as engine `PokemonSet`s).
- `raw/hc75-top8.txt` — Historia Cup 7.5 top-8 transcription (source of T1).
- `raw/samples-27.txt` — Smogon Resource Hub sample teams (source of T2),
  parsed from the thread post; per-team alternate-moveset variants and the
  Stadium-2 rental Dragonite/Tyranitar are excluded.
- `raw/vr.json` — Chio's viability ranking (94 species, S…C-), parsed from
  [the English tear list](https://seesaawiki.jp/pbs-thread/d/Tear%20list%20in%20Nintendo%20Cup%202000).
- `raw/usage-hc75.json` — species usage over all 27 HC7.5 entrant teams
  (count + %), parsed from the Resource Hub tournament report.

## Sources and tiers

- **T1 (8 teams): Historia Cup 7.5** — live NC2000 tournament (Stadium 2,
  Crystal moves), 27 players, 2022-05-03, hosted by Gold. Top-8 with full sets
  from the [Smogon NC2000 Resource Hub](https://www.smogon.com/forums/threads/nintendo-cup-2000-resource-hub.3682691/)
  tournament report; original JP report: [gold.hatenadiary.jp](https://gold.hatenadiary.jp/entry/2022/05/05/151859).
  These are tournament-proven teams and rank above all samples.
- **T2 (26 teams): Resource Hub sample teams 1–26** — built/curated by
  international + Japanese experts (Beelzemon 2003, Chio, Kitty, the JP Poké
  Cup community; per-team attribution in `provenance.authors`).

## Ranking (mechanical)

`tournamentPoints` (1st=100, 2nd=80, 3rd=70, 4th=60, top8=50, sample=0)
→ `vrMean` (mean Chio-VR points of the 6 species: S=8 … B-=1, else 0)
→ `hc75UsageMean` (mean HC7.5 usage % of the 6 species).
All components ship in the JSON so downstream can re-weight or cut at any N.

## Exclusions (rule-based)

- **Historia Cup 10/11 (2024–25)**: played under the "Historia Cup 2024"
  special ruleset (Kanto species capped at L50–52, different species pool) —
  not NC2000; ingesting would contaminate the distribution.
- **sample-27**: Gligar's Earthquake is event-only on PS → Event Moves Clause
  violation (validator-verified); no documented substitute exists.
- **The hub's 22-team PokePaste**: an older, different sample collection
  superseded by the thread's 27; dropped to avoid near-duplicates.
- **PS ladder replays**: only 49 public `gen2nc2000` replays exist (checked
  2026-07-16), casual quality; no usage stats are published for the format.

## Transcription conventions (HC7.5 teams)

- The report gives species/level/item/moves but no DVs: Hidden Power sets use
  the canonical DV spreads from the sample teams (Ice `26 Def`; Bug
  `26 Atk / 26 Def`; Grass `6 HP / 28 Atk / 28 Def`; Fighting
  `6 HP / 24 Atk / 24 Def`; Flying `14 HP / 24 Atk / 26 Def`); everything else
  is max DVs.
- All sets: `Ability: No Ability`, EVs 255 across (fully trained — fixture
  convention), happiness 255 (0 for Frustration users).
- Every team passes PS's own `TeamValidator` for the target format — the same
  oracle the fixture corpus uses.

## Format target (re-based 2026-07-21)

The pool now validates against **`gen2nintendocup2000noohkostadium2strict`**
(the community server's no-OHKO NC2000 Stadium2 Strict — the regulation the
bot actually plays; definition in `pokemon-showdown.zip`). Consequences:

- **OHKO moves are banned**: 3 HC7.5 teams (`hc75-4th-kg`, `hc75-top8-tako`,
  `hc75-top8-shinobu` — Fissure / Horn Drill carriers) are excluded as
  format-illegal. T1 is now 5 teams. HC7.5 itself was an OHKO-allowed
  tournament, so this is expected source attrition, not data loss.
- **No Event Moves Clause**: event moves are legal, which re-admits
  `sample-27` (Gligar's event-only Earthquake was the old exclusion reason).
- Evasion / Bright Powder remain legal in BOTH regulations. The sample-team
  thread swapped Bright Powder out assuming a ban that never shipped;
  original items are recorded in `provenance.notes`.
- Crystal moves (SD Marowak, Spikes Cloyster, Baton Pass Umbreon…) validate
  fine.

## Expansion sources (not yet ingested)

- Gym Leader's Castle simulator tournament archive (up to 178 entrants):
  http://pokemon.s20.xrea.com/2nd/pbs/historia_01.html and the
  [ジムリーダーの城 wiki](https://seesaawiki.jp/pbs-thread/) named-archetype
  team pages (キチカビ, テンプレWA, 受けンタ, …) — richest untapped trove;
  needs JP-name → PS-set translation.
- Earlier Historia Cups (1–7) and プチドラサマ杯 (2020–21) usage stats.
- [単体考察 for Nintendo Cup 2000](http://pokemon.s20.xrea.com/2nd/2000/) for
  per-species canonical set variants.
