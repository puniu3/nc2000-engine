# Strong-Pool Okimono (Dead-Slot) Exposure, Mutation & Minor-Gem Strict LIVE Verification (2026-11-23)

## Objective & Strict Production Methodology

As a deliberate detour from exact disclosed-pool Nash convergence, we audited every team in the shipped 33-team / 32-shell strong pool ([`data/team-pool-v3/team-pool.json`](../../data/team-pool-v3/team-pool.json)) to:
1. **Expose true dead-weight slots ("置物" / okimono)** under exact production **`729k`** Legacy Team Preview (never `9k/81k`),
2. **Mutate and strengthen** affected teams under strict Japanese NC2000 cartridge legality, WASM `Validator`, and `training-v3` normalization,
3. **Independently verify** all promoted major mutations AND all top minor-Pokémon proposals from [`minor-gem-finder.md`](./minor-gem-finder.md) using **5,915 strictly NO-CACHE LIVE (`243k/729k`, `NASH_PREVIEW_CACHE=0`, `preview.source == "live"` on 100% of sides, `0` cache hits)** paired Common-Random-Numbers (CRN) games across **all 32 distinct shells of `team-pool-v3`**.

- **Driver**: [`tools/okimono-audit.py`](../../tools/okimono-audit.py)
- **Certified Upgraded Rosters**: [`docs/reports/okimono-upgrades-v1.json`](./okimono-upgrades-v1.json)
- **Raw gitignored artifacts**: `target/nash-work/okimono-v1/{census-v3-k4,round1-preview,wave1-live-ab,wave2-live-ab,stage4-holdout-panelB,minor-gem-preview,stage5b-minor-gems,stage6-cert}/`

---

## 1. Stage 1: Exact `729k` Full 33×32 Strong-Pool Okimono Census

Computed `1,089` exact `K=4` (`729k`) preview distributions (`4,356` previews) covering all 33 teams against all 32 shells of `team-pool-v3`.

### Exposed `< 10%` Okimono Slots (14 of 33 Teams)

| Severity | Team (`team-pool-v3` Rank) | Okimono Slot (`729k` Pick%, Nonzero Opps / 32) | Additional Structural Defect |
| :--- | :--- | :--- | :--- |
| **Absolute `0.0%`** | `pb-122` (#5) | `s3: Electrode L50 @ Bright Powder` (**`0.0%`**, `0/32`) | `Self-Destruct` / `HP Water`; `s6 Cloyster` has `Curse` |
| **Absolute `0.0%`** | `pb-85` (#20) | `s6: Umbreon L50 @ Miracle Berry` (**`0.0%`**, `0/32`) | Only 1 L55 (`Snorlax55` forced 100%) |
| **Absolute `0.0%`** | `sample-09` (#29) | `s4: Smeargle L50 @ PRZ Cure Berry` (**`0.0%`**, `0/32`) | Zero Water/Ice resist |
| **Severe `1.9%`** | `mjj-2014-offantom` (#4) | `s2: Charizard L55 @ Miracle Berry` (**`1.9%`**, `2/32`) | Double-Normal L55s (`P2` + `Tauros`) walled by Rock/Steel |
| **Severe `4.3%`** | `sample-12` (#21) | `s1: Gengar L50 @ Berry Juice` (**`4.3%`**, `3/32`) | Perish-trap + `Berry Juice` |
| **Low `6.2%–9.4%`** | `mjj-2010-garap-a` (#27) | `s6: Exeggutor L50 @ Miracle Berry` (`6.2%`, `3/32`) | Only 1 L55 (`Marowak55` 100%) |
| **Low `6.6%`** | `mjj-2020-rraikou` (#26) | `s2: Cloyster L50 @ Berry Juice` (`6.6%`, `5/32`) | `Berry Juice` + `Curse` without `Ice Beam` |
| **Low `7.0%–7.8%`** | `rental-cban-13` (#32), `hc75-top8-tsuru` (#31), `mjj-type-QA-moudoku` (#10) | `Poliwrath55` (`7.0%`), EQ-less `Golem50` (`7.8%`), `Crobat55` (`7.8%`) | |
| **Low `8.6%–9.4%`** | `sample-07` (#15), `pb-87` (#9), `mjj-2015-ayagen` (#23), `pb-66` (#13) | `Tauros55` (`8.6%`), Rollout `Miltank50` (`9.4%`), `Exeggutor50` (`9.4%`), Perish `Gengar50` (`9.4%`) | |

**Key Reversal vs `9k` Audits**: `pb-84` Slot 6 `Umbreon50` is **not** a slot okimono at `729k` across the 32-shell pool (`21.9%`); `pb-84`'s primary structural defect was its `66.4%`-picked **`Cloyster L50 @ Berry Juice` (`Reflect` over `Surf`)**.

---

## 2. Certified Okimono Upgrades (5,915 Strictly NO-CACHE LIVE Games, All 32 Shells)

All mutations passed exact `729k` activation screening (`round1-preview`: 45 candidates; `minor-gem-preview`: 23 candidates), paired CRN LIVE screening on stratified Panel A (16 shells), and **independent out-of-sample LIVE certification on disjoint Panel B (the remaining 16 shells)** and/or fresh independent 32-shell passes + direct Head-to-Head (H2H).

| Rank | Upgraded Team | Parent (`Base WR`) | Exact Mutation | LIVE Games | All-32-Shell WR | Gain vs Parent (`z` / H2H) | `>=50%` Shells | Min Slot Pick% |
|---:|---|---|---|---:|---:|---|---:|---:|
| **1** | **`rraikou-m-cune50`** | `mjj-2020-rraikou` (`56.2%`, 192g) | `s2 Cloyster@BJ` → **`Suicune L50 @ Bright Powder`** (`Surf/Ice Beam/Rest/Talk`) | **192** | **`62.5% ± 3.6`** | **`+6.3 pp`** (H2H `58.3%`; PanA `64.1%` / PanB `64.1%` / Fresh `59.4%`) | **`26/32`** | `7.8% → 18.8%` |
| **2** | **`pb85-m-raikou55`** | `pb-85` (`44.5%`, 128g) | `s6 Umbreon50` → **`Raikou L55 @ Magnet`** (`Thunder/HP Ice/Rest/Talk`) | **128** | **`57.0% ± 4.1`** | **`+12.5 pp` (`z = +2.05`)** (PanA `+10.9` / PanB `+14.1`) | **`26/32`** | `0.0% → 16.4%` |
| **3** | **`pb66-m-wak55`** | `pb-66` (`46.1%`, 128g) | `s6 Perish Gengar` → **`Marowak L55 @ Thick Club`** (`EQ/RS/HP Bug/SD`) | **128** | **`54.7% ± 4.8`** | **`+8.6 pp` (`z = +1.37`)**, **H2H `66.7%`** | `20/32` | `9.4% → 21.1%` |
| *(Ref)* | `s0053-lefties` *(incumbent)* | *(flagship)* | *(unchanged 50/50 HPI/DP camo)* | **320** | **`54.1% ± 2.8`** | *Baseline reference* | `19/32` | `15.9%` |
| **4** | **`offantom-m-cune55`** | `mjj-2014-offantom` (`43.8%`, 128g) | `s2 Charizard55` → **`Suicune L55 @ Mint Berry`** (`Surf/Ice Beam/Rest/Talk`) | **128** | **`51.6% ± 4.6`** | **`+7.8 pp`** (PanA `+20.3` / PanB `-4.7`) | **`24/32`** | `1.6% → 28.1%` |
| **4t** | **`offantom-m-wak55`** | `mjj-2014-offantom` (`43.8%`, 128g) | `s2 Charizard55` → **`Marowak L55 @ Thick Club`** (`EQ/RS/HP Bug/SD`) | **128** | **`51.6% ± 5.0`** | **`+7.8 pp`** (PanA `+21.9` / PanB `-6.2`) | `19/32` | `1.6% → 21.1%` |
| **6** | **`garap-m-cune55`** | `mjj-2010-garap-a` (`36.7%`, 128g) | `s6 Exeggutor50` → **`Suicune L55 @ Bright Powder`** (`Surf/Ice Beam/Rest/Talk`) | **128** | **`50.0% ± 4.9`** | **`+13.3 pp` (`z = +2.24`)** (PanA `+18.8` / PanB `+7.8`) | `18/32` | `8.6% → 32.0%` |
| **7** | **`pb84-cln-sking50`** *(minor gem)* | `pb-84` (`39.1%`, 128g) | `Cloyster @ Gold Berry Surf` + `Gligar @ Powder` + `s6` → **`Slowking L50 @ Leftovers`** (`Surf/Fire Blast/Rest/Talk`) | **192** | **`48.4% ± 3.5`** | **`+9.4 pp`** vs raw `pb-84`; **`+3.6 pp` & H2H `68.8%`** vs `pb84-m-cloysurf` | `21/32` | `15.1%` |
| **7b** | **`pb84-m-cloysurf`** *(same-shell)* | `pb-84` (`39.1%`, 128g) | Same shell: `s4 Cloyster @ Berry Juice Reflect` → `Gold Berry Surf`, `s5 Gligar @ Powder` | **320** | **`46.6% ± 2.6`** | **`+10.2 pp` (`z = +1.75`, 128g CRN)**; causal proof vs `pb84-m-cloyref` (`37.5%`) | `18/32` | `21.6%` |
| **8** | **`tsuru-m-cloy50`** | `hc75-top8-tsuru` (`42.2%`, 128g) | `s3 EQ-less Golem` → **`Cloyster L50 @ Gold Berry`** (`Surf/IB/Spikes/Explosion`) | **128** | **`47.7% ± 4.5`** | **`+5.5 pp`**, **H2H `66.7%`** (PanA `+6.2` / PanB `+4.7`) | `19/32` | `10.2% → 28.1%` |
| **9** | **`s09-m-cloy50`** | `sample-09` (`39.8%`, 128g) | `s4 Smeargle50` → **`Cloyster L50 @ Gold Berry`** (`Surf/IB/Spikes/Explosion`) | **128** | **`44.5% ± 4.9`** | **`+4.7 pp`**, **H2H `75.0%`** | `20/32` | `0.0% → 20.3%` |

### Rejected Mutations (Important Negative Results Preserving Clean Baselines)
- **`sample-07` (`57.8%` Panel A)** and **`pb-87` (`54.7%` Panel A)**: Both original rosters outperformed all tested mutations (`s07-m-ttar55-clean` `-20.3 pp`, `s07-m-cloy-clean` `-6.2 pp`, `pb87-m-miltank-curse` `-7.8 pp`). Keep untouched.
- **`pb84-m-eggy50`** (`54.7%` Panel A → `32.8%` disjoint Panel B) and **`rraikou-m-zap50`** (`56.2%` Panel A → `45.3%` disjoint Panel B): Caught as Panel-A overfitting by disjoint Panel-B holdout.
- **`pb-122` (`47.7%` 128g)** and **`sample-12` (`48.4%` Panel A)**: Neither Snorlax/Machamp (`+0.0 to +1.6 pp`), Zapdos55 (`+0.0 pp`), nor minor gems (`-6.2 to -17.2 pp`) improved LIVE win rate.

---

## 3. Strict `243k/729k` LIVE Verdict on [`minor-gem-finder.md`](./minor-gem-finder.md)

We tested 23 minor-gem splices at exact `729k` preview (`minor-gem-preview`), promoted 12 passing candidates to 32-shell strict NO-CACHE LIVE (`stage5b-minor-gems`, `1,023g`), and ran independent `128g` + `16g` H2H confirmation (`stage6-cert`) on the two Panel-wide breakouts:

| Minor-Gem Proposal | `9k/81k` Report | `243k/729k` LIVE Screen (64g, 32sh) | Independent Confirmation (128g + H2H) | Final Strict LIVE Verdict |
| :--- | ---: | ---: | ---: | :--- |
| **`pb84-cln-sking50`** (`Slowking50 @ Leftovers` + clean Cloyster) | `50.0%` (`+0.0 pp`) | **`53.1%`** (`+6.2 pp` vs `cloysurf`, `26/32 >= 50%`) | **`46.1%`** (`+2.3 pp` vs `cloysurf`, **H2H `68.8%`**; pooled 192g **`48.4%`**, `52.5%` in 6-Champ RR) | **PROMOTED** — sole minor gem surviving strict LIVE holdout (`75%` vs `rraikou-m-cune50`!) |
| `s0053-shuck50-mix` (`Shuckle50 @ Miracle`, 50/50 camo) | `50.0%` (`+9.4 pp`) | `59.4%` (`+7.8 pp`, picked WR `58.1%`) | **`50.0%`** vs `s0053-lefties` `54.7%` (`-4.7 pp`, **H2H `31.2%`**; pooled `53.1%` vs `53.6%`) | **REJECTED** — holdout regression + lost H2H `5–11` |
| `pb84-raw-pilo50` / `pb84-cln-pilo50` (`Piloswine50`) | **`65.6%`** (`+15.6 pp`) | **`37.5%`** / **`43.8%`** (picked WR **`33.3%`** vs unpicked `57.1%`) | — | **REJECTED** — classic `9k` preview over-pick trap at `243k/729k` |
| `pb84-cln-shuck50` (`Shuckle50 @ Leftovers`) | `62.5%` (`+12.5 pp`) | **`40.6%`** (picked WR **`34.5%`**) | — | **REJECTED** — preview over-pick trap |
| `pb84-cln-queen50` (`Nidoqueen50 @ Leftovers`) | `62.5%` (`+12.5 pp`) | **`42.2%`** (picked WR **`26.3%`** vs unpicked `48.9%`) | — | **REJECTED** — severe preview over-pick trap |
| `pb122-gem-{pilo,queen,shuck}50` | — | `42.2%` / `35.9%` (`z=-2.16`) / `46.9%` vs base `53.1%` | — | **REJECTED** — high pick balance (`27–42%`) masked negative LIVE EV |
| `rraikou-gem-pilo50`, `offantom-gem-{muk55,mag55}` | — | `51.6%` / `51.6%` / `54.7%` (`Mag55` pick `1.6%`) | — | **REJECTED** — dominated by `rraikou-m-cune50` (`62.5%`) / `offantom-m-cune55` |

---

## 4. 6×6 Super-Champions Round-Robin (`stage6-cert`, `16g/edge` Strict NO-CACHE LIVE)

Directly pitting our top 6 shells against each other revealed a clean **3-shell RPS core** combining the incumbent flagship with two newly evolved okimono upgrades:

| Team | `rraik-cune` | `s0053-shuk` | `pb85-raik` | `s0053-left` | `pb66-wak` | `pb84-sking` | RR Mean | RM+ Nash Weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`s0053-lefties`** | `62.5%` | `68.8%` | `62.5%` | — | `37.5%` | `81.2%` | **`62.5%`** | **`55.1%`** (`EV 50.0%`) |
| **`pb66-m-wak55`** | `18.8%` | `50.0%` | `37.5%` | `62.5%` | — | `50.0%` | `43.8%` | **`22.3%`** (`EV 50.0%`) |
| **`rraikou-m-cune50`** | — | `56.2%` | `43.8%` | `37.5%` | `81.2%` | `25.0%` | `48.8%` | **`21.7%`** (`EV 50.0%`) |
| `pb85-m-raikou55` | `56.2%` | `50.0%` | — | `37.5%` | `62.5%` | `56.2%` | `52.5%` | `0.9%` (`EV 47.3%`) |
| `pb84-cln-sking50` | `75.0%` | `75.0%` | `43.8%` | `18.8%` | `50.0%` | — | `52.5%` | `0.0%` (`EV 38.1%`) |
| `s0053-shuck50-mix` | `43.8%` | — | `50.0%` | `31.2%` | `50.0%` | `25.0%` | `40.0%` | `0.0%` (`EV 38.3%`) |

**Strategic Takeaway**: Even though exact Nash was deferred, okimono mutation organically produced (1) a new #1 broad-field champion **`rraikou-m-cune50` (`62.5%` across 192g LIVE)**, (2) a second `s0053`-beating all-rounder **`pb85-m-raikou55` (`57.0%`, `+12.5 pp`)**, (3) a clean flagship counter **`pb66-m-wak55` (`54.7%`, `62.5%` vs `s0053-lefties`)** forming a natural 3-shell cycle `{s0053-lefties, pb66-m-wak55, rraikou-m-cune50}`, and (4) a certified minor-gem counter **`pb84-cln-sking50` (`75.0%` vs `rraikou-m-cune50`)**.

---

## 5. Gen-1 Self-Referential Co-Evolution & Gen-2 Snapshot (Paused Pending Engine Bugfix)

> [!WARNING]
> Research paused cleanly on 2026-11-23 (`barn` stopped, `0` active Colab VMs / `0.0 CU/hr`) to prioritize an engine-side bug fix. All Gen-0/Gen-1/Gen-2 measurements in this report reflect pre-fix engine behavior and should be re-verified after the engine fix lands.

Constructing an experimental **Gen-1 32-shell pool** (`target/nash-work/okimono-v1/gen1-pool.json`, replacing the 8 Gen-0 parents with the 8 certified Gen-1 upgrades) and running a full self-referential **`729k` 33×32 census** (`gen1-census-k4`, `4,356` previews) + **Gen-1 Elite-18 `gem_1v1` Stage 1 & 2** (`2,556` played-out cells) + **Gen-2 `729k` activation screen** (`gen2-preview`, 29 candidates) + **Gen-2 LIVE snapshot** (`gen2-live-ab`, `88/107` chunks = **`1,056` strict NO-CACHE LIVE games** against all 32 Gen-1 shells) confirmed self-referential three-body dynamics:

1. **Gen-0 → Gen-1 Metagame & Okimono Shift (`gen1-census-k4`)**:
   - `Suicune` (`+3.7 pp`) and `Raikou` (`+1.5 pp`) share rose across the pool, causing untouched Ground/Rock/physical slots to drop (`pb-122` `Marowak55 -9.4 pp`, `pb-6` `Machamp55 -7.8 pp`, `sample-06` `Skarmory50 -7.0 pp`, `cban-8` `Tyranitar55 -6.3 pp`) while Electric/Water/Grass surged (`pb-122` `Raikou55 +9.4 pp`, `apex-c04` `Starmie50 +8.6 pp`, `s0053` / `pb-6` / `sample-02` `Zapdos +7.8 pp`).
   - **Newly exposed Gen-1 `< 10%` okimonos**: `mjj-type-QA-moudoku` `Crobat55` (`7.8% → 4.7%`), `pb-6` `Gengar50` (`14.8% → 8.6%`), `sample-08` `Cloyster50` (`10.2% → 8.6%`), and our own Gen-1 upgrade `pb84-cln-sking50` `Gligar50` (`9.4%`).
2. **Gen-1 Gem Rediscovery & Pre-Pause `1,056g` Strict LIVE Snapshot (`gen2-live-ab`)**:
   - **Three-body rank shift in Gen-1**: `rraikou-m-cune50` jumped to **`71.9% ± 5.5` (`30/32 >= 50%`)** in Gen-1 (and replacing `Marowak55` with `Zapdos55` collapsed `-34.4 pp`, `z = -3.78`); untouched `pb-6` surged to **`60.0%`**; meanwhile `s0053-lefties` (`50.0%`), `pb85-m-raikou55` (`50.0%`), and `pb66-m-wak55` (`46.9%`) compressed `-4 to -8 pp` against the tougher Gen-1 field.
   - **Consistent Gen-2 meta winners — `Blissey L50` & `Raikou L50` (5/5 positive across all tested shells)**:
     - `pb6-g2-bliss50` (`DE/Curse/Counter/Soft-Boiled` + `Cloyster Surf`): **`70.8% ± 6.7`** (`22/24`, `+8.7 pp` vs `pb-6` `60.0%`, `minPk 8.0% → 10.4%`)
     - `mut:s0053-g2-raik50-mix` (`Electrode50 → Raikou50@Miracle`, 50/50 camo): **`62.5% ± 6.4`** (`27/32`, **`+12.5 pp`** vs `s0053-lefties` `50.0%`, `minPk 17.2% → 29.7%`)
     - `s08-g2-raik50`: **`57.7% ± 7.2`** (`+3.8 pp` vs `sample-08`, `minPk 5.8% → 25.0%`)
     - `pb84g2-bliss50` (`Gligar50 → Blissey50@PRZ`): **`51.7% ± 5.6`** (`25/30`, **`+8.3 pp`** vs `pb84-cln-sking50` `43.3%`, `minPk 8.3% → 18.3%`)
     - `moudoku-g2-bliss50`: **`46.7% ± 6.3`** (`+8.3 pp` vs `moudoku` `38.3%`, `minPk 3.3% → 21.7%`)
   - **1v1-to-6v6 minor-gem mismatches rejected on LIVE**: `Muk L55/L50 @ Leftovers` (despite #1 `65.3%` in played-out 1v1+SW) scored `0.0 pp`, `-3.3 pp`, and `-10.9 pp` across `moudoku`, `pb84`, and `pb-6`; `Espeon55` scored `-10.0 pp`.
3. **Clean Resume Point After Engine Bugfix**:
   - Rebuild native (`tools/build-native-pgo.sh`) and Colab x86_64 (`tools/build-colab-x86.sh`) `nash_refresh` binaries after fixing the engine bug, then re-verify the top Gen-1/Gen-2 discoveries (`rraikou-m-cune50`, `pb85-m-raikou55`, `pb66-m-wak55`, `pb84-cln-sking50`, `pb6-g2-bliss50`, `s0053-g2-raik50-mix`, `pb84g2-bliss50`) under the fixed engine.
4. **Completed under `policy-v2` (2026-11-24)**: Full re-verification (`8,312` strictly NO-CACHE LIVE `243k/240k` games + `10,392` exact `240k` previews), `policy-v2` minor-gem threshold recalibration, disjoint Panel-B certification, 10×10 Super-Champions RR, and [`data/team-pool-v4/`](../../data/team-pool-v4/) generation are documented in [`policy-v2-team-pool-refresh.md`](./policy-v2-team-pool-refresh.md) and [`okimono-upgrades-v2.json`](./okimono-upgrades-v2.json).

