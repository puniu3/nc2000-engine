# Community rental-team DB — data-source assessment (v0)

Evaluation of `http://psense.lib.net/_/PDINPUT2.cgi` (「レンタルパーティUtility」)
as a candidate team-data source for nc2000-engine. Crawled, parsed, mapped to
PS ids, and legality-checked 2026-07-20.

**TL;DR — effective, best as a *community belief prior* (M15) and a *cross-source
metagame validator*; low-yield as a raw meta-pool extender because it overlaps
the existing pool heavily. All 28 entries map cleanly to the engine; 25/28
validate legal under the shipped `gen2nc2000` format as-is.**

## The source

A community-published "rental party" (レンタルパーティ) DB — teams openly shared
for newcomers to borrow — for the **「一撃無し２０００ルール」** (no-OHKO NC2000)
metagame. Old Apache/CGI, EUC-JP, HTTP-only. Index enumerates 28 named
archetypes (`CBAN=1..28`); `CBAN>=28` returns the empty `名称不明` template, so
28 is the full range. Each detail page gives a human-readable table
(Lv / species / 4 moves / item), an archetype name, a free-text comment, and
`出典`(source) / `制作`(author) provenance. **No DVs/EVs in the visible table.**

Raw HTML archived under `raw/` (as fetched, EUC-JP); parsed output in `raw.json`;
canonicalized engine-shape teams in `teams.json`; machine stats in `stats.json`.
Parser: `tools/parse-community-rentals.js`.

## Crawl + parse + mapping (deliverables 1–2)

- **28/28 teams parsed** — archetype, comment, provenance, and the 6-mon table.
- **JP→PS-id mapping: 100%.** Inverting `data/i18n-ja.json` after one
  normalization (the DB substitutes **U+2212 MINUS SIGN** for the `ー`
  prolonged-sound mark in table cells; full/half-width spaces stripped) maps
  **every** species / move / item. **Zero unmapped names.** The only near-miss
  was a `－` placeholder in empty move slots, filtered out (not a real move).
- **Species cross-check: zero mismatches.** Each page's hidden `PD` field carries
  the national-dex number per mon; all 28×6 mapped species agree with it —
  independent proof the name mapping is correct.
- **Bonus — Hidden Power types recovered.** The visible table shows only generic
  「めざめるパワー」, but the `PD` field also encodes per-mon DVs. Decoding them
  (`type = HP_TYPES[4*(atkDV%4)+(defDV%4)]`) recovers the real HP type, verified
  against the free-text comments (No.2 comment states Zapdos=氷/Exeggutor=草/
  Machamp=ゴースト — matching the decode exactly). `teams.json` upgrades generic
  Hidden Power to the typed variant so canonicalize applies the canonical DV
  spread — a faithfulness win the human table alone can't give.

## Legality validation (deliverable 3)

Each team → `Validator.canonicalizeTeam` (fills DVs/EVs/gender legally per M14a)
→ `Validator.validateTeam` (the oracle-certified PS mirror). **25/28 validate
clean.** The 3 failures are **genuine source/format properties, not mapping bugs**
(0 `move-illegal`, 0 `item-unknown`, 0 mapping-induced errors):

| # | Archetype | Failure | Nature |
|---|-----------|---------|--------|
| 16 | タマムシジム | `item-clause` | **Real illegality**: two Miracle Berry (Exeggutor + Umbreon). Both map correctly; the source team simply violates Item Clause. |
| 26 | 【リトルカップ専用】 | `level-min` ×6 | **Different format**: self-labeled *Little Cup* (all L5). Correctly rejected by the L50 floor. |
| 28 | 参考ページ等 リンク集 | `team-size` | **Not a team**: a links/reference page dressed as a 1-mon (Ditto) entry. |

So of the 26 real NC2000 teams, **25 are legal verbatim** and 1 (No.16) carries a
source-side Item Clause violation. No parse or mapping defect produced any failure.

## Ruleset-delta quantification (deliverable 4)

The DB's default rule (`一撃無し`) bans OHKO; the shipped `gen2nc2000` format bans
neither OHKO nor evasion nor Bright Powder (README M8). Measured effect:

- **OHKO moves: 5 total — all confined to the 3 teams the author explicitly tags
  【一撃あり用】 (OHKO-allowed variant): No.23 (Tauros Horn Drill, Machamp Fissure),
  No.24 (Snorlax Fissure), No.25 (Dugtrio Fissure, Tauros Horn Drill).** The 25
  default teams contain **zero** OHKO moves — exactly consistent with the header.
  The ruleset delta is therefore *honestly annotated per team*: default teams
  never run Horn Drill/Fissure/Guillotine, and the 3 variant teams that do are
  directly compatible with the project's (OHKO-permitting) format.
  *(Addendum 2026-07-21: the project re-targeted to the no-OHKO Strict format
  — `gen2nintendocup2000noohkostadium2strict` — so the polarity flipped: the
  25 default teams are the format-matched ones and No.23/24/25 are now
  format-illegal. This DB is the better-matched source overall.)*
- **Evasion: 1** (No.16 Miltank Double Team). Legal in both rulesets — no delta.
- **Bright Powder: 11 mons** (No.4/7/11/12/13/14/16/19/23/25/27). Legal in both —
  no delta, but notably heavier usage than the meta-pool (whose sample-team
  source *removed* Bright Powder assuming a ban that never shipped).
- **Uber-tagged / out-of-format species: 0** in standard teams. All species are
  in the 246-species format. (The Little Cup entry No.26 uses L5 Scyther/Elekid/
  Chansey/Cubone/Porygon/Voltorb — a different format, excluded above.)

Net: the **only** genuine format delta is the no-OHKO default, and it manifests
as *absence* of OHKO moves in 25/28 teams — a strict subset of what the project
format allows, so those teams import without contradiction.

## Novelty vs the 34-team meta pool (deliverable 5)

Signature = sorted species set (the pool's `species` notion; a full-`packed`
match would be stricter). **16/28 teams share an exact species set with a pool
team; 12 do not** (several of those are near-dupes, J≥0.5).

This DB and the meta-pool draw from the **same JP community canon**: No.1's `出典`
is `q9con.net/PartyBox` (キチカビ) and other sources are `majinjima`(魔人島/
ジムリーダーの城), fc2/hatena blogs — exactly the ジム城 wiki / PartyBox lineage
the meta-pool README lists as its richest untapped expansion trove. High overlap
is expected and *corroborates* both corpora.

Genuinely-new **standard-format** archetypes the pool lacks (species core):

- **No.14 しゃわポリWA** — Vaporeon/Porygon2/Snorlax/Zapdos/Exeggutor/Steelix.
  Documented **tournament winner** (2018 Historia Cup《うら》, no-OHKO 2000 rule).
  Introduces **Vaporeon** — the *only* standard-format species absent from all 34
  pool teams.
- **No.2 サンダーバンギWA** — Zapdos/Tyranitar/Snorlax/Cloyster/Exeggutor/Machamp
  (Sandstorm WA core; distinct 6th from the nearest HC7.5 team).
- **No.10 セミフルカビ**, **No.12 カビガラポリ**, **No.21/22 エース追加型** —
  new Snorlax-lead cores over species the pool already covers.
- Plus **No.23/24/25 【一撃あり用】** — new archetypes built for the OHKO-allowed
  variant (a different intent than the pool).

Off-format novelties excluded from the above: No.26 (Little Cup), No.27
(【指振りルール】 Metronome-only gimmick — validates clean but off-meta), No.28
(links page).

## Verdict + recommendation (deliverable 6)

**Yes, this is an effective data source — with the right use.**

1. **Community belief prior for M15 PS interop — STRONGEST fit.** This is
   literally the set of teams this community's newcomers borrow, so an opponent
   drawn from here is disproportionately likely to field one of these 28 (or a
   close variant). As a *pinned belief / opponent-team prior* for playing against
   this specific community it is high-value and directly usable — the
   canonicalized `teams.json` sets plug into the same `PokemonSet` shape the
   open-team-sheet belief already consumes.
2. **Cross-source metagame validation — good fit.** The 16 species-set matches
   independently reproduce the meta-pool's cores via a *different* extraction path
   (this DB vs the Smogon hub / HC7.5 report), a genuine consistency check. The
   PD-decoded Hidden Power types + author comments also supply canonical set
   details (HP typing, item choices) that could tighten the pool's transcription
   conventions.
3. **M11b meta-pool extension — usable but LOW-YIELD.** 16/28 are species-set
   dupes, so net-new is small: ~6 novel standard archetypes + Vaporeon + the
   tournament-proven No.14. If ingested, cherry-pick those, note the no-OHKO
   caveat (default teams simply won't carry OHKO — a subset, harmless), **exclude
   No.16** (Item Clause violation) or fix its duplicate berry, and **exclude the
   off-format No.26/27/28**. Do not bulk-merge.

**Recommendation:** adopt as a **community belief prior (M15)** and a **cross-source
validator** now; defer any meta-pool merge to a deliberate cherry-pick of the ~6
novel archetypes (+ No.14) under the M11b criteria, not a bulk import. *This
assessment does not modify the shipped meta pool or any other artifact — that is
an owner decision.*

## ⚠ External-data-into-public-repo flag (owner decision)

`data/community-rentals-v0/` is **externally-sourced third-party content**
(team lists, free-text comments, pseudonymous author handles, and source URLs
scraped from a community CGI site). This repo's `.github/workflows` deploys to
GitHub Pages, so **pushing this directory = redistributing that content**. The
material is community-published reference data (openly-shared rental teams,
public URLs, pseudonymous handles), but redistribution + attribution is a
publication decision reserved to the owner. It has been committed **locally only
and NOT pushed** — review provenance/attribution (and whether to keep the raw
EUC-JP HTML, `raw/`) before any push. Consider adding these sources to
`THIRD-PARTY-NOTICES.md` if adopted.
