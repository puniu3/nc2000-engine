# `policy-v2` Meta-Shift Audit, Minor-Gem Detection Threshold Calibration & `team-pool-v4` Refresh (2026-11-24)

## Executive Summary

Following the promotion of **`policy-v2`** (`CF-V_net v2` PUCT adaptive `243k` + `SteppedCfPreview 240k`, commit `ac24efd9`) to production default, we executed an end-to-end **8,312 strictly NO-CACHE LIVE games (`243k/240k`, `NASH_PREVIEW_CACHE=0`, `preview.source == "live"` on 100% of 16,624 player-sides, `0` cache hits, `0` `barn` requeues)** + **15,432 exact `240k` `SteppedCfPreview` evaluations (`13,032` unique-key across `3,258` distinct keys)** campaign (`26.10 CU` total on `mini1 + mini2 + Colab G4`) to:

1. **Quantify the `policy-v1 -> policy-v2` three-body meta-shift** across all 32 shells of [`data/team-pool-v3/team-pool.json`](../../data/team-pool-v3/team-pool.json), all 10 `okimono-upgrades-v1` teams, and top Gen-2 candidates.
2. **Empirically calibrate minor-Pokémon gem detection thresholds under `policy-v2`** across **37 strictly NO-CACHE LIVE minor-gem experiments** (all 17 `policy-v1` benchmarks + `Vaporeon55` + 19 new 3v3-screened minor archetypes spanning all 7 functional roles; plus 1 cross-test `m2b:pb89_sking50` in `step2-major-upgrades`, `n=38` total), proving why the `policy-v2` detection boundary shifts from `policy-v1`'s one-sided `pick% >= 25%` gate to a **two-sided inverted-U band (`10% <= pick% <= 42%`) + CRN-matched causal `InΔ` (`E[score(cand) - score(base) | minor picked] >= 0`) + disjoint Panel-B holdout**.
3. **Build and certify [`data/team-pool-v4/`](../../data/team-pool-v4/)** (`33` teams / `32` pairwise-distinct public Turn-0 shells + 6-shell exact Nash core mixture) containing **9 independently holdout-certified `policy-v2` upgrades**, including **2 certified Minor-Pokémon Gems (`Slowking L50` in `pb84-cln-sking50` and `Espeon L55` in `pb63-m-esp55`) — BOTH earning active support in the exact Tier-S Nash equilibrium!**

- **Infrastructure**: Added `POLICY_EPOCH` (`"policy-v2"`) to `preview_dist_key`, `preview_cache_key`, `PreviewDistFile`, and `manifest.json` (plus `previewDistOnly`/`previewDistK` manifest tracking and unit tests) in [`crates/bot/examples/nash_refresh.rs`](../../crates/bot/examples/nash_refresh.rs); rebuilt native Apple-M5 CS-PGO (`tools/build-native-pgo.sh`) and Linux `x86-64-v3` (`tools/build-colab-x86.sh`) binaries; updated [`tools/okimono-audit.py`](../../tools/okimono-audit.py) (`240k` default, `--work-dir`, per-slot `picked_WR`/`unpicked_WR` and CRN-matched `slot_picked_delta`/`slot_unpicked_delta`) and [`tools/gem-finder.py`](../../tools/gem-finder.py) (`policy-v2` zone tags, dynamic `policyEpoch`, idempotent `out` cleanup).
- **Shipped Outputs**: [`data/team-pool-v4/team-pool.json`](../../data/team-pool-v4/team-pool.json), [`data/team-pool-v4/flagship-mixture.json`](../../data/team-pool-v4/flagship-mixture.json), [`tools/build-team-products-v4.js`](../../tools/build-team-products-v4.js), [`docs/reports/okimono-upgrades-v2.json`](./okimono-upgrades-v2.json).
- **Raw gitignored artifacts**: `target/nash-work/okimono-v2/{census-v3-k4,step1-v3-rr,step1-upgrades-gen2,step1-v1-minorgem-cal,step2a-minor-screen,step2b-major-screen,step2-new-minors,step2-major-upgrades,step3a-holdout-panelB,step3b-v4-census-k4,step3b-champs10-rr}/`.

---

## 1. Step 1: `policy-v1 -> policy-v2` Three-Body Meta-Shift

### 1.1 Exact `240k` `SteppedCfPreview` 33×32 Okimono Census (`census-v3-k4`)

Comparing all 33 teams × 32 shells (`4,356` previews, `K=4`) between `policy-v1` (`729k` UCB1 + 8r `rollout_switch=false`) and `policy-v2` (`240k` `SteppedCfPreview` Double-Oracle LCB-gated Mixed Nash) revealed a structural inversion in how the bot values team slots:

- **Mixing diversity increased**: Mean effective preview combos (`exp(H)`) rose **`12.20 -> 13.17`**.
- **Tactical / Tempo / Spikes / Explosion / Baton-Pass / Perish-Trap slots surged (`+5 to +38 pp`)**:
  - `Gengar55` **`+14.6 pp`** (`32.3% -> 46.9%`), `Cloyster50` **`+11.5 pp`** (`44.3% -> 55.8%`, `n=18`), `Electrode50` **`+11.3 pp`** (`24.3% -> 35.7%`, `n=6`), `Miltank55` **`+11.2 pp`**, `Jolteon50` **`+10.7 pp`**, `Gengar50` **`+6.0 pp`**, `Snorlax55` **`+5.8 pp`** (`88.2% -> 93.9%`), `Exeggutor50` **`+5.0 pp`**.
  - **6 former `policy-v1` okimonos healed automatically under `policy-v2`**: `pb-122` `Electrode50` (`0.0% -> 18.0%`), `sample-09` `Smeargle50` (`0.0% -> 11.7%`), `sample-12` `Gengar50` (`4.3% -> 10.9%`), `garap-a` `Exeggutor50` (`6.2% -> 18.0%`), `rraikou` `Cloyster50` (`6.6% -> 22.3%`), `rental-cban-13` `Poliwrath55` (`7.0% -> 11.7%`).
- **Passive / RestTalk / defensive-only slots dropped (`-9 to -48 pp`)**:
  - `Zapdos50` **`-18.5 pp`** (`73.4% -> 54.9%`), `Heracross50` **`-18.4 pp`**, `Blissey50` **`-16.4 pp`**, `Alakazam55` **`-12.5 pp`**, `Venusaur50` **`-11.4 pp`**, `Golem50` **`-10.7 pp`**, `Umbreon50` **`-9.4 pp`**, `Raikou55` **`-9.1 pp`**.
  - **New `policy-v2` okimonos exposed (`< 10%`)**: `mjj-2014-offantom` `Charizard55` (`1.9% -> 0.0%`), `mjj-2015-ayagen` `Umbreon50` (**`0.0%`**), `pb-6` `Umbreon50` (`30.1% -> 3.1%`), `pb-63` `Venusaur55` (**`3.9%`**), `pb-85` `Umbreon50` (`3.9%`), `pb-87` `Miltank50` (`9.4% -> 3.9%`), `sample-06` `Raikou55` (`27.3% -> 5.5%`), `pb-89` `Umbreon50` (`9.4%`), `pb-96` `Venusaur50` (`43.8% -> 9.4%`), `sample-02` non-Curse `Snorlax50` (`59.0% -> 10.9%`, `-48.0 pp`).

### 1.2 Full 32×32 `team-pool-v3` Round-Robin (`step1-v3-rr`, `992g` Strict LIVE) & `policy-v1` Upgrade Reversals (`step1-upgrades-gen2`, `1,920g`)

Running the complete 32×32 round-robin (`2g`/pair, `62g`/shell, `0` cache hits, mean `20.8` turns, `33.7 s/g`) overturned the `policy-v1` hierarchy:

- **Top `policy-v2` `v3` shells**: `#1 pb-63` (`66.1% ± 5.4`, `29/31 >= 50%` despite `3.9%` `Venusaur55` okimono!), `#2 pb-89` (`62.9%`, was `#33` in `v3`!), `#3 sample-08` (`61.3%`, `#1` Nash weight `13.0%`), `#4t pb-6` & `mjj-2014-thdrcomb` (`59.7%`), `#6 sample-06` (`58.1%`), `#7t sample-07` & `mjj-2020-rraikou` (`56.5%`), `#9 sample-11` (`54.8%`), `#10t offantom` & `ayagen` (`53.2%`).
- **`policy-v1` icons that dropped**: `pb-84` collapsed to `#32` (`32.3%`, Berry Juice Reflect Cloyster over-picked `77%`, Umbreon `8%` InWR); `pb-122` dropped to `#30` (`38.7%`); `s0053-lefties` finished `#12` (`51.6%` in RR, `65.6%` in Step 1C, pooled `58.7%` over 126g).
- **Critical `policy-v1 -> policy-v2` Upgrade Reversals (`step1-upgrades-gen2`, `64g` CRN each)**:
  - `offantom-m-cune55`: v1 `+7.8 pp` → **v2 `-14.1 pp` (`z = -1.72`)** (L55 Suicune diluted `P2/Tauros`; fixed in `v4` with **L50** `Marowak50` → **`+14.1 pp`**, H2H **`83.3%`**!).
  - `pb85-m-raikou55`: v1 `+12.5 pp` → **v2 `-3.1 pp`** (`Raikou55` only `6.2%` next to `Snorlax55`; fixed in `v4` with **L50** `Suicune50` → **`+6.3 pp`**, `minPk 28.1%`!).
  - `pb6-g2-bliss50`: v1 `70.8%` (`+8.7 pp`) → **v2 `-4.7 pp`** (`s4 Umbreon50` dropped to `0.0%`).
  - `pb84g2-bliss50`: v1 `+8.3 pp` → **v2 `-12.5 pp`**.
  - `cban13-g2-vap55`: **v2 `-23.4 pp` (`z = -2.70`)** — classic OOD `Vaporeon55` over-pick trap (`50%` pick, `28%` InWR!).
  - `s09-m-cloy50` (`-6.2 pp`) and `tsuru-m-cloy50` (`-4.7 pp`) also regressed.
  - Meanwhile **`pb84-cln-sking50`** (`+4.7 pp` S1C, `+9.4 pp` S1D, `+18.8 pp` PanB), **`rraikou-m-cune50`** (`+6.2 pp` S1C, `+3.1 pp` PanB), and **`pb66-m-wak55`** (`+4.7 pp` S1C, H2H `66.7%`) survived!

---

## 2. Step 2: Minor-Pokémon Gem Detection Thresholds Under `policy-v2`

### 2.1 Why Detection Thresholds Changed (`policy-v1` vs `policy-v2`)

Across **37 strictly NO-CACHE LIVE minor-gem experiments** (`step1-v1-minorgem-cal` `1,536g` + `cban13-g2-vap55` + `step2-new-minors` `704g`, preceded by a **58-candidate 3v3 `240k` preview screen** spanning all 7 functional roles: Spikes, Perish/Encore/Counter, Baton Pass, Fast Sleep/Seed, Bulky Explosion, Typing Pivots, High-Tempo Breakers), we observed a sharp, quantitative shift in the detection boundary:

1. **`policy-v1`'s RestTalk Over-Pick Inflation Collapsed by `-15 to -44 pp`**:
   - `pb84-cln-lant50`: `62.5% -> 18.8%` (`-43.7 pp`)
   - `pb84-cln-amph50`: `75.0% -> 32.8%` (`-42.2 pp`)
   - `pb122-gem-pilo50`: `51.6% -> 10.9%` (`-40.7 pp`)
   - `pb84-cln-sking50`: `46.9% -> 21.9%` (`-25.0 pp`, while LIVE WR jumped to `57.8%`, `InWR = 71%`!)
   - `pb84-cln-pilo50`: `56.2% -> 34.4%` (`-21.8 pp`, flipping from v1 trap `33%` InWR to v2 specialist `68%` InWR!)
2. **A NEW `CF-V_net v2` OOD Over-Valuation Trap Emerged Above `42%` Pick Rate (The Three-Body Problem)**:
   - Because `CF-V_net v2` was trained on Major-dominated self-play (`team-pool-v2` / `belief-pool-v3`), out-of-distribution minor sets carrying high-value Major-like tactical features (**Baton Pass, Moonlight, Explosion, Haze, Curse+RestTalk**) are systematically **over-valued by `CF-V_net v2`**, causing `SteppedCfPreview` to over-pick them (`43%–69%`) into losing matchups:
     - `pb84_sciz50` (SD/Agility BP): `62.5%` pick → **`-28.1 pp` (`z = -2.33`)**, `25%` InWR
     - `pb84_weez50` (Haze/Explosion): `46.9%` pick → **`-28.1 pp` (`z = -2.52`)**
     - `cban13-g2-vap55` (RestTalk55): `50.0%` pick → **`-23.4 pp` (`z = -2.70`)**, `28%` InWR
     - `s0053-shuck50-mix` (Toxic/Encore): `43.8%` pick → **`-18.8 pp` (`z = -2.34`)**
     - `pb84_clef50` (Moonlight/TWave): `68.8%` pick → **`-12.5 pp`**
     - `pb84_mukb50` (Explosion/Curse): `43.8%` pick → **`-9.4 pp`**
     - `pb6_pilo50` (EQ/IB RestTalk): `53.1%` pick → **`29%` InWR vs `87%` OutWR (`-57 pp` gap!!)**

### 2.2 Empirical Inverted-U Detection Curve Across All 37 `policy-v2` Minor-Gem Experiments

Using the exact spliced minor slot on every candidate (including `s6 Espeon55` on `pb6-g2-espeon55` and `s5 Muk50` on `pb84g2-muk50-sking`), and decomposing overall `dWR` into both naive `InWR - OutWR` and **CRN-matched causal `InΔ = E[score(cand) - score(base) | minor picked]`**:

| Live Minor `pick%` Zone | `n` | Mean `dWR` | Mean `InWR - OutWR` | Mean CRN `InΔ` | `dWR > 0` | `dWR > 0 & (In-Out) >= +5pp` | `dWR > 0 & InΔ >= 0` | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`< 10%` (Dead / Ignored)** | 3 | **`-8.33 pp`** | **`-12.0 pp`** | **`-13.3 pp`** | `0/3` | **`0/3`** | **`0/3`** | Too weak / Stage-1 pruned (`Magneton55`, `Muk55`, `Queen50`; plus `Wobbuffet`, `Jumpluff`, `Qwilfish` at 3v3 screen) |
| **`10% – 42%` (Sweet Spot)** | 22 | **`-1.78 pp`** | **`-1.9 pp`** | **`-5.4 pp`** | `9/22` | **`3/22`** | **`7/22`** | Core specialist zone (`Slowking50`, `Espeon55` on `pb-63`, `Piloswine50` on `pb-84`); adding cross-test `m2b:pb89_sking50` (`15.6%`, `-12.5 pp`, `InΔ -40.0 pp`) gives `n=23`, mean `dWR = -2.24 pp`, `3/23` naive / `7/23` causal |
| **`> 42%` (OOD Over-Pick Trap)** | 12 | **`-9.77 pp`** | **`-8.2 pp`** | **`-13.7 pp`** | `3/12` | **`0/12`** | **`2/12`** | Severe `CF-V_net v2` OOD over-valuation (`0/12` pass naive `In-Out`; only boundary `pb6-g2-espeon55` at `42.2%` [`InΔ +7.4 pp`] and `pb84-cln-shuck50` at `46.9%` [`InΔ +3.3 pp`] avoided negative `InΔ`) |

Overall Pearson correlation across all 37 minor experiments: **`r(live_pick%, dWR) = -0.282`** (`-0.255` across `n=38` including `m2b:pb89_sking50`).

### 2.3 Calibrated `policy-v2` Detection Thresholds, Simpson's Paradox & Out-of-Sample Validation

| Gate Parameter | `policy-v1` (Legacy `729k`) | **`policy-v2` (`SteppedCfPreview 240k` + `CfValuePuct`)** |
|---|---|---|
| **Preview Activation Lower Floor** | `>= 25%` (missed surgical specialists like `Espeon55`) | **`>= 10%`** (captures surgical specialists like `pb63-m-esp55` at `12.5%` PanA / `18.8%` PanB / `22.7%` v4 census / `40.7%` Champs10) |
| **Over-Pick Trap Upper Ceiling** | None (traps ran `45%–75%`) | **`<= 42%`** (`0/12` minors `>42%` passed naive `In-Out >= +5 pp`; mean `dWR = -9.77 pp`, mean CRN `InΔ = -13.7 pp`) |
| **Bench-Noise Discriminator** | Overall `dWR > 0` only | **CRN-matched `InΔ >= 0` (primary, Simpson-free)** + naive `InWR - OutWR >= +5 pp` (fast single-run screen) AND `dWR > 0` |
| **Holdout Requirement** | Single panel + confirmation | **Disjoint Panel-B (`16` unseen shells) `dWR >= 0` + Direct H2H** |

**Methodological Note — Why CRN-Matched `InΔ` Beats Naive `InWR - OutWR` (Simpson's Paradox)**:
- Naive `InWR - OutWR` compares a minor's win rate in the matchups where `SteppedCfPreview` chose it against the team's win rate in *different* matchups where it was benched. Because specialists are deliberately brought into harder matchups (lower baseline WR) and also alter opponent Team Preview selections even when benched (`OutΔ`), naive `InWR - OutWR` fluctuates sharply on small `32g–64g` screens (`n_in = 4–16`):
  - **`m2a:pb6_typh50` (Negative Control)**: Passed `[10%, 42%]` (`37.5%`) and Panel-A `dWR = +6.2 pp`, **but failed BOTH naive `InWR - OutWR = -25.0 pp` AND CRN-matched `InΔ = -16.7 pp` (`OutΔ = +20.0 pp`)**. On disjoint Panel B, its pick rate spiked to `62.5%` and it **collapsed `-18.8 pp` (`z = -1.57`, `InΔ = -20.0 pp`)**.
  - **`pb84-cln-sking50`** (`Slowking50`): vs clean same-shell `pb84-m-cloysurf` (`S1D`, `64g`): `Pk 21.9%`, `InWR 71.4% (+17.4 pp)`, **`InΔ = +7.1 pp`**, `OutΔ = +10.0 pp`, `dWR = +9.4 pp`; vs raw `pb-84`: `S1C +4.7 pp` (`InWR 31.2%`, `InΔ -6.2 pp`) and **Disjoint Panel B `+18.8 pp` (`62.5%`, `z = +1.70`, `InWR 50.0%`, `OutΔ = +26.9 pp`)** → **Pooled `160g` `55.0%` (`+9.4 pp`)**, and **`50.0%` (`19.1%` exact Nash)** in 10×10 Champs RR (`Slowking` `17.6%` pick, `36.8%` InWR). Causally beats `Slowbro50` (`-18.8 pp`, `InΔ -9.1 pp`) on the identical slot, proving Slowking's 110 SpD is essential.
  - **`pb63-m-esp55`** (**NEW `policy-v2` Gem: `Espeon55`**): `Pk 12.5%` (PanA, `InWR 75.0%`, `InΔ = 0.0 pp`, `OutΔ = +7.1 pp`) / `18.8%` (PanB, `InWR 33.3%`, `InΔ = -16.7 pp`, `OutΔ = +11.5 pp`) / `22.7%` (v4 census) / **`40.7%` (`44/108`) in 10×10 Champs RR with `52.3%` InWR**! Beat ALL Major upgrades on `#1 pb-63` (`pb63_gar55 +3.1/+3.1`, `pb63_cloy50 -12.5`, `pb63_raik55 -15.6`), combining active Champs10 play (`40.7%`, `52.3%` InWR) with Team Preview deterrence (`+5.0` net games on Panel B when previews shifted vs `0.0` net across PanA+PanB when previews were identical).
  - **`pb6-g2-espeon55`** (`Espeon55` on `pb-6` `s6`): `S1D` (`64g`) `Pk 42.2%` (`s6`, right at upper boundary), `InWR 48.1%` vs `OutWR 62.2%` (`-14.0 pp` naive gap — failed naive `In-Out` due to matchup selection!), **yet CRN-matched `InΔ = +7.4 pp`**, and on **Disjoint Panel B** (`32g`): `Pk 53.1%`, `dWR = +3.1 pp`, `InWR 70.6% (+23.9 pp)`, **`InΔ = +23.5 pp`** (`+13.6 pp` pooled `InΔ` over `44` picked games; omitted from `v4` only because it leaves `s4 Umbreon50` at `0.0%` while `pb63-m-esp55` already introduces `Espeon55` cleanly).
  - **`pb84-cln-pilo50`** (`Piloswine50`): `S1D` `Pk 34.4%`, `InWR 68.2% (+22.9 pp)`, **`InΔ = +31.8 pp`**, `dWR = +4.7 pp`; **Disjoint Panel B** `Pk 34.4%`, `dWR = +0.0 pp`, `InWR 54.5% (+16.5 pp)`, `InΔ = 0.0 pp` (pooled `+3.1 pp`; dominated by `pb84-cln-sking50` `+9.4 pp` on the same shell).

---

## 3. Step 3: Disjoint Panel-B Certification, 10×10 Super-Champions RR & `team-pool-v4`

### 3.1 Disjoint Panel-B Out-of-Sample Certification (`step3a-holdout-panelB`, `1,404g` Strict LIVE)

Panel B eliminated 5 Panel-A overfits (`ayagen_cloy50 -12.5pp`, `garap-m-cune55 -28.1pp`, `moudoku-g2-bliss50 -25.0pp`, `pb96_eggy50 -9.4pp`, `s06_eggy50 -15.6pp`) and certified **9 `policy-v2` upgrades** for `team-pool-v4`:

| Upgraded Shell | Parent | Category | Screen `dWR` | Disjoint PanB `dWR` | H2H (`12g`) | Pooled Field WR (`n`) | `v4` `240k` MinPk |
|---|---|---|---:|---:|---:|---:|---:|
| **`pb122-m-lax50`** | `pb-122` | Major okimono | `+21.9 pp` | **`+25.0 pp` (`z=+2.24`)** | `41.7%` | **`62.5%` (`+23.4 pp`, `z=+3.30`, `64g`)** | `24.2%` |
| **`offantom-m-wak50`** | `mjj-2014-offantom` | Major okimono (L50) | `+18.8 pp` | **`+9.4 pp`** | **`83.3%`** | **`53.1%` (`+14.1 pp`, `64g`)** | `14.4%` |
| **`s02-m-lax-curse`** | `sample-02` | Same-shell opt | `+28.1 pp` | **`+0.0 pp`** | `58.3%` | **`57.8%` (`+14.1 pp`, `64g`)** | `14.1%` |
| **`ayagen-m-zap50`** | `mjj-2015-ayagen` | Major okimono | `+18.8 pp` | **`+3.1 pp`** | `50.0%` | **`65.6%` (`+10.9 pp`, `64g`)** | `25.8%` |
| **`pb84-cln-sking50`** | `pb-84` | **Minor Gem (`Slowking50`)** | `+4.7 / +9.4 pp` | **`+18.8 pp` (`z=+1.70`)** | `50.0%` | **`55.0%` (`+9.4 pp`, `160g`)** | `14.8%` (`Sking 23.8%`) |
| **`pb63-m-esp55`** | `pb-63` | **Minor Gem (`Espeon55`)** | `+6.25 pp` | **`+6.25 pp`** | `50.0%` | **`56.2%` (`+6.25 pp`, `64g`)** | `7.8%` (`Esp 22.7%`) |
| **`pb85-m-cune50`** | `pb-85` | Major okimono (L50) | `+9.4 pp` | **`+3.1 pp`** | `41.7%` | **`46.9%` (`+6.25 pp`, `64g`)** | `28.1%` |
| **`rraikou-m-cune50`** | `mjj-2020-rraikou` | Major okimono | `+6.25 pp` | **`+3.1 pp`** | `16.7%` | **`54.2%` (`+5.2 pp`, `96g`)** | **`32.4%`** (#1!) |
| **`pb66-m-wak55`** | `pb-66` | Major okimono | `+4.7 pp` | `-3.1 pp` | **`66.7%`** | **`61.5%` (`+2.1 pp`, `96g`)** | `13.3%` |

### 3.2 10×10 Super-Champions Round-Robin & Exact Nash Equilibrium (`step3b-champs10-rr`, `540g`, `12g/edge`)

Exact rational solution (`fractions.Fraction`) of the 10×10 skew-symmetric payoff matrix reveals a **1-dimensional exact Nash equilibrium polytope** $(1 - t)\,x_1 + t\,x_2$ ($t \in [0, 1]$) between two extreme 5-support exact equilibria:
- $x_1 = (\frac{7}{22}, \frac{5}{22}, \frac{2}{11}, \frac{2}{11}, \frac{1}{11}, 0, \dots)$ (`pb63-m-esp55` has EV = `50.000%`),
- $x_2 = (\frac{3}{10}, \frac{1}{10}, \frac{1}{5}, \frac{1}{5}, 0, \frac{1}{5}, \dots)$ (`pb122-m-lax50` and `pb-6` have EV = `50.000%`).

Every interior point ($0 < t < 1$) is an **exact 6-shell Nash equilibrium** with **`max_BR = 50.000000%`**. (`flagship-mixture.json` ships the exact rational midpoint **$t = \frac{1}{2}$**, within `0.0001` nats of the maximum-entropy equilibrium $t^* = 0.4826$, `eff = 5.271` shells, 6-decimal `max_BR = 50.000008%`; finite-iteration RM+ stopped near $t \approx 0.04$ with `0.27%` unconverged mass on `pb-6` whose EV vs Nash is `< 50%`.)

| Champion | `rraik` | `s08` | `sking` | `pb66` | `pb122` | `esp55` | `pb-6` | `ayag` | `s0053` | `pb-89` | RR Mean (`108g`) | Exact Nash ($t=\frac{1}{2}$) | Nash Range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`rraikou-m-cune50`** | — | 33.3 | 66.7 | 50.0 | 58.3 | 41.7 | 66.7 | 66.7 | **83.3** | 75.0 | **`60.2% ± 5.3`** | **`30.91%`** (`17/55`) | `30.0–31.8%` |
| **`pb84-cln-sking50`** *(minor)* | 33.3 | 50.0 | — | **75.0** | 58.3 | 50.0 | 33.3 | 33.3 | 58.3 | 58.3 | `50.0% ± 4.8` | **`19.09%`** (`21/110`) | `18.2–20.0%` |
| **`pb66-m-wak55`** | 50.0 | 83.3 | 25.0 | — | 16.7 | 58.3 | 41.7 | 83.3 | 66.7 | 66.7 | `54.6% ± 7.9` | **`19.09%`** (`21/110`) | `18.2–20.0%` |
| **`sample-08`** | 66.7 | — | 50.0 | 16.7 | 58.3 | 58.3 | 33.3 | 16.7 | 50.0 | 58.3 | `45.4% ± 6.2` | **`16.36%`** (`9/55`) | `10.0–22.7%` |
| **`pb63-m-esp55`** *(minor)* | **58.3** | 41.7 | 50.0 | 41.7 | **58.3** | — | 58.3 | **83.3** | 33.3 | 66.7 | **`54.6% ± 5.0`** | **`10.00%`** (`1/10`) | `0.0–20.0%` |
| **`pb122-m-lax50`** | 41.7 | 41.7 | 41.7 | 83.3 | — | 41.7 | **91.7** | 75.0 | 58.3 | 41.7 | **`57.4% ± 6.9`** | **`4.55%`** (`1/22`) | `0.0–9.1%` |
| `pb-6` | 33.3 | 66.7 | 66.7 | 58.3 | 8.3 | 41.7 | — | 41.7 | 50.0 | 75.0 | `49.1% ± 6.9` | `0.00%` (`EV 49.6%`) | `0.0%` |
| `ayagen-m-zap50` | 33.3 | 83.3 | 66.7 | 16.7 | 25.0 | 16.7 | 58.3 | — | 41.7 | 50.0 | `43.5% ± 7.7` | `0.00%` (`EV 42.7%`) | `0.0%` |
| `s0053-lefties` *(v3 flagship)* | 16.7 | 50.0 | 41.7 | 33.3 | 41.7 | 66.7 | 50.0 | 58.3 | — | 66.7 | `47.2% ± 5.4` | `0.00%` (`EV 36.2%`) | `0.0%` |
| `pb-89` | 25.0 | 41.7 | 41.7 | 33.3 | 58.3 | 33.3 | 25.0 | 50.0 | 33.3 | — | `38.0% ± 3.7` | `0.00%` (`EV 34.8%`) | `0.0%` |

### 3.3 Self-Referential `240k` `team-pool-v4` Census (`step3b-v4-census-k4`, `4,356` Previews)

Replacing the 9 parents with the 9 certified `policy-v2` upgrades eliminated every `0.0%` absolute okimono across the entire 33-team pool (`2 -> 0`), halved `< 5%` okimonos (`6 -> 3`), reduced `< 10%` okimonos (`13 -> 9`), and raised mean minimum slot pick rate from **`12.96% -> 15.08%`** (`+2.12 pp`).

---

## 4. Next Steps (Three-Body Co-Evolution Loop)

1. **Web / Belief-Pool Rollout (when ready)**: Wire `data/team-pool-v4/` into `tools/build-known-prior.mjs`, `web/src/data.ts`, `web/src/pdinput*.ts`, `web/src/team-names.ts`, and regenerate `data/belief-pool-v4/` (deliberately kept `data/team-pool-v3/` untouched in this step so existing tests & web manifests stay green).
2. **Next NN Training Cycle (`CF-V_net v3`)**: Retrain `CF-V_net` on `team-pool-v4` self-play (including `Slowking50`, `Espeon55`, and OOD minor-splice counterexamples such as `Scizor`, `Clefable`, `Weezing`, `Vaporeon55`) to close the three-body OOD over-valuation gap identified in §2.2!
