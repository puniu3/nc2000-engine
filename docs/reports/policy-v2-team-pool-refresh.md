# `policy-v2` Meta-Shift Audit, Minor-Gem Detection Threshold Calibration & `team-pool-v4` Refresh (2026-11-24)

## Executive Summary

Following the promotion of **`policy-v2`** (`CF-V_net v2` PUCT adaptive `243k` + `SteppedCfPreview 240k`, commit `ac24efd9`) to production default, we executed a **10,152 strictly NO-CACHE LIVE games (`243k/240k`, `NASH_PREVIEW_CACHE=0`, `preview.source == "live"` on 100% of 20,304 player-sides, `0` cache hits, `0` `barn` requeues)** + **19,788 exact `240k` `SteppedCfPreview` evaluations (`15,016` unique-key across `3,754` distinct keys)** campaign (`32.56 CU` total on `mini1 + mini2 + Colab G4/V5E1`) to:

1. **Quantify the `policy-v1 -> policy-v2` three-body meta-shift** across all 32 shells of [`data/team-pool-v3/team-pool.json`](../../data/team-pool-v3/team-pool.json), all 10 `okimono-upgrades-v1` teams, and top Gen-2 candidates.
2. **Empirically calibrate minor-Pokémon gem detection thresholds under `policy-v2`** across **37 strictly NO-CACHE LIVE minor-gem experiments** (plus `m2b:pb89_sking50`, `n=38` total), proving why the `policy-v2` detection boundary shifts from `policy-v1`'s one-sided `pick% >= 25%` gate to a **two-sided inverted-U band (`10% <= pick% <= 42%`) + true shared-seed CRN causal `InΔ > 0` (`OutΔ ≈ 0`) + same-slot Major controls**.
3. **Conduct a Step 4 true shared-seed CRN forensic audit (`1,840g`, §4)** after sanity-checking the tentative minor-gem Nash claims: this exposed (a) a fake-CRN agent-seed bug in `tools/okimono-audit.py`, (b) genetic hitchhiking on `pb84-cln-sking50` (`Slowking50` Champs10 `InWR 36.8%` vs `OutWR 52.8%`; `39.8%` / `9th of 9` in true CRN, `-14.06 pp` [`z = -2.46`] vs `Skarmory50`), (c) replication failure on `pb63-m-esp55` (`50.8%`, `-4.69 pp` vs parent `pb-63`, `-8.59 pp` vs `Gengar55`), and (d) `12g/edge` Champs10 Nash bootstrap CIs starting at `0.0%` for every shell.
4. **Build and audit [`data/team-pool-v4/`](../../data/team-pool-v4/)** (`33` teams / `32` pairwise-distinct public Turn-0 shells + provisional uniform 5-shell Tier-S core) containing **9 causally verified `policy-v2` upgrades (`9/9` with positive in-battle CRN `InΔ`)**, replacing both retracted minor-gem candidates with their true-CRN Major winners (**`pb84-cln-skarm50`** [`InΔ +25.7 pp`, `OutΔ 0.0 pp`] and **`pb63-m-gar55`** [`InΔ +17.9 pp`, `OutΔ 0.0 pp`]).

- **Infrastructure**: Added `POLICY_EPOCH` (`"policy-v2"`) to `preview_dist_key`, `preview_cache_key`, `PreviewDistFile`, and `manifest.json` (plus `previewDistOnly`/`previewDistK` manifest tracking and unit tests) in [`crates/bot/examples/nash_refresh.rs`](../../crates/bot/examples/nash_refresh.rs); rebuilt native Apple-M5 CS-PGO (`tools/build-native-pgo.sh`) and Linux `x86-64-v3` (`tools/build-colab-x86.sh`) binaries; fixed the fake-CRN agent-seed bug (`seeds` now shared identically across group members alongside `battleSeed`) and added `slot_picked_delta`/`slot_unpicked_delta`/`slot_unpicked_breakdown` in [`tools/okimono-audit.py`](../../tools/okimono-audit.py); updated [`tools/gem-finder.py`](../../tools/gem-finder.py) (`policy-v2` zone tags, dynamic `policyEpoch`, idempotent `out` cleanup).
- **Shipped Outputs**: [`data/team-pool-v4/team-pool.json`](../../data/team-pool-v4/team-pool.json), [`data/team-pool-v4/flagship-mixture.json`](../../data/team-pool-v4/flagship-mixture.json), [`tools/build-team-products-v4.js`](../../tools/build-team-products-v4.js), [`docs/reports/okimono-upgrades-v2.json`](./okimono-upgrades-v2.json).
- **Raw gitignored artifacts**: `target/nash-work/okimono-v2/{census-v3-k4,step1-v3-rr,step1-upgrades-gen2,step1-v1-minorgem-cal,step2a-minor-screen,step2b-major-screen,step2-new-minors,step2-major-upgrades,step3a-holdout-panelB,step3b-v4-census-k4,step3b-champs10-rr,step4-true-crn-audit,step4b-v4-census-k4}/`.

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
  - Meanwhile **`rraikou-m-cune50`** (`+6.2 pp` S1C, `+3.1 pp` PanB) and **`pb66-m-wak55`** (`+4.7 pp` S1C, H2H `66.7%`) survived! (`pb84-cln-sking50` initially scored `+4.7 / +9.4 / +18.8 pp` under Steps 1–3's fake-CRN/uncontrolled setup, but later failed true shared-seed CRN replication in §4.)

---

## 2. Step 2: Minor-Pokémon Gem Detection Thresholds Under `policy-v2`

### 2.1 Why Detection Thresholds Changed (`policy-v1` vs `policy-v2`)

Across **37 strictly NO-CACHE LIVE minor-gem experiments** (`step1-v1-minorgem-cal` `1,536g` + `cban13-g2-vap55` + `step2-new-minors` `704g`, preceded by a **58-candidate 3v3 `240k` preview screen** spanning all 7 functional roles: Spikes, Perish/Encore/Counter, Baton Pass, Fast Sleep/Seed, Bulky Explosion, Typing Pivots, High-Tempo Breakers), we observed a sharp, quantitative shift in the detection boundary:

1. **`policy-v1`'s RestTalk Over-Pick Inflation Collapsed by `-15 to -44 pp`**:
   - `pb84-cln-lant50`: `62.5% -> 18.8%` (`-43.7 pp`)
   - `pb84-cln-amph50`: `75.0% -> 32.8%` (`-42.2 pp`)
   - `pb122-gem-pilo50`: `51.6% -> 10.9%` (`-40.7 pp`)
   - `pb84-cln-sking50`: `46.9% -> 21.9%` (`-25.0 pp`, `S1D` `57.8%`; later retracted in §4 under true CRN at `39.8%`)
   - `pb84-cln-pilo50`: `56.2% -> 34.4%` (`-21.8 pp`, `S1D` `InWR 68%`; Step 4 true-CRN `InΔ -5.6 pp`)
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
| **`10% – 42%` (Sweet Spot)** | 22 | **`-1.78 pp`** | **`-1.9 pp`** | **`-5.4 pp`** | `9/22` | **`3/22`** | **`7/22`** | Least-penalized candidate band (still requires true shared-seed CRN `InΔ > 0` + Major controls — §4 retracted `Slowking50` & `Espeon55`!); adding cross-test `m2b:pb89_sking50` (`15.6%`, `-12.5 pp`, `InΔ -40.0 pp`) gives `n=23`, mean `dWR = -2.24 pp`, `3/23` naive / `7/23` causal |
| **`> 42%` (OOD Over-Pick Trap)** | 12 | **`-9.77 pp`** | **`-8.2 pp`** | **`-13.7 pp`** | `3/12` | **`0/12`** | **`2/12`** | Severe `CF-V_net v2` OOD over-valuation (`0/12` pass naive `In-Out`; only boundary `pb6-g2-espeon55` at `42.2%` [`InΔ +7.4 pp`] and `pb84-cln-shuck50` at `46.9%` [`InΔ +3.3 pp`] avoided negative `InΔ`) |

Overall Pearson correlation across all 37 minor experiments: **`r(live_pick%, dWR) = -0.282`** (`-0.255` across `n=38` including `m2b:pb89_sking50`).

### 2.3 Calibrated `policy-v2` Detection Thresholds, Simpson's Paradox & Out-of-Sample Validation

| Gate Parameter | `policy-v1` (Legacy `729k`) | **`policy-v2` (`SteppedCfPreview 240k` + `CfValuePuct`)** |
|---|---|---|
| **Preview Activation Lower Floor** | `>= 25%` (missed surgical specialists) | **`>= 10%`** (captures surgical specialists like `pb63-m-gar55` at `15.6%–21.9%` and `pb84-cln-skarm50` at `27.3%–28.1%`) |
| **Over-Pick Trap Upper Ceiling** | None (traps ran `45%–75%`) | **`<= 42%`** (`0/12` minors `>42%` passed naive `In-Out >= +5 pp`; mean `dWR = -9.77 pp`, mean CRN `InΔ = -13.7 pp`) |
| **Bench-Noise Discriminator** | Overall `dWR > 0` only | **True shared-seed CRN `InΔ > 0` (`OutΔ ≈ 0`, Zero-Hitchhiker Law)** + `dWR > 0` |
| **Holdout & Control Requirement** | Single panel + confirmation | **Same-slot Major controls + disjoint Panel-B / `128g` true-CRN + Direct H2H** |

**Methodological Note — Why CRN-Matched `InΔ` Beats Naive `InWR - OutWR` (Simpson's Paradox)**:
- Naive `InWR - OutWR` compares a minor's win rate in the matchups where `SteppedCfPreview` chose it against the team's win rate in *different* matchups where it was benched. Because specialists are deliberately brought into harder matchups (lower baseline WR) and also alter opponent Team Preview selections even when benched (`OutΔ`), naive `InWR - OutWR` fluctuates sharply on small `32g–64g` screens (`n_in = 4–16`):
  - **`m2a:pb6_typh50` (Negative Control)**: Passed `[10%, 42%]` (`37.5%`) and Panel-A `dWR = +6.2 pp`, **but failed BOTH naive `InWR - OutWR = -25.0 pp` AND CRN-matched `InΔ = -16.7 pp` (`OutΔ = +20.0 pp`)**. On disjoint Panel B, its pick rate spiked to `62.5%` and it **collapsed `-18.8 pp` (`z = -1.57`, `InΔ = -20.0 pp`)**.
  - **`pb84-cln-sking50`** (`Slowking50`) & **`pb63-m-esp55`** (`Espeon55`) — **RETRACTED in §4**: Although both survived the initial Steps 1–3 screens (`dWR > 0` on Panel A & Panel B), **BOTH** had negative pooled `InΔ` (`-3.3 pp` vs clean `cloysurf` / `-9.1 pp` vs raw `pb-84` for Slowking; `-10.0 pp` for Espeon) carried almost entirely by `OutΔ` while `tools/okimono-audit.py` had a fake-CRN agent-seed bug (`seeds` hashed subject ID `m`, injecting RNG variance into `sameBoth` bench games). In the **1,840-game Step 4 true shared-seed CRN audit (§4)** against same-slot Major controls, **BOTH collapsed** (`pb84-cln-sking50` `39.8%` dead last `9th/9`, `-14.06 pp` [`z = -2.46`] vs `Skarmory50`; `pb63-m-esp55` `50.8%`, `-4.69 pp` vs parent `pb-63`, `-8.59 pp` vs `Gengar55`). This validates the strict **Zero-Hitchhiker Law**: under true shared-seed CRN, genuine upgrades exhibit strongly positive **`InΔ`** (`+10 to +26 pp`) with **`OutΔ ≈ 0`**, whereas `InΔ <= 0` carried by `OutΔ` is a red flag for hitchhiking/confounding!
  - **`pb6-g2-espeon55`** (`Espeon55` on `pb-6` `s6`): `S1D` (`64g`) `Pk 42.2%`, `InΔ = +7.4 pp`; Panel B (`32g`) `Pk 53.1%`, `InΔ = +23.5 pp` (`+13.6 pp` pooled `InΔ`, omitted from `v4` because `s4 Umbreon50` dropped to `0.0%`).
  - **`pb84-cln-pilo50`** (`Piloswine50`): Despite `S1D` `InΔ = +31.8 pp`, Step 4 true-CRN (`128g`) yielded `52.3%` (`+5.47 pp`) with **negative `InΔ = -5.6 pp (-3.0g/54)`** and `OutΔ = +13.5 pp (+10.0g/74)`, failing the Zero-Hitchhiker criterion compared to `pb84-cln-skarm50` (`InΔ = +25.7 pp`, `OutΔ = 0.0 pp`).

---

## 3. Step 3: Disjoint Panel-B Screens, Champs10 RR Audit & Updated `team-pool-v4` Census

### 3.1 Provisional Step-3 Panel-B Screen (`step3a-holdout-panelB`, `1,404g` Strict LIVE)

Panel B eliminated 5 Panel-A overfits (`ayagen_cloy50 -12.5pp`, `garap-m-cune55 -28.1pp`, `moudoku-g2-bliss50 -25.0pp`, `pb96_eggy50 -9.4pp`, `s06_eggy50 -15.6pp`). Subsequent cross-upgrade hitchhiker auditing and the **Step 4 true shared-seed CRN audit (§4)** retracted `pb84-cln-sking50` and `pb63-m-esp55` and promoted their true-CRN Major winners (`pb84-cln-skarm50` and `pb63-m-gar55`), yielding **9 causally verified `policy-v2` upgrades (`9/9` with positive in-battle CRN `InΔ`)**:

| Upgraded Shell | Parent | Category | Field / True-CRN WR (`n`) | `dWR` vs Base | Causal `InΔ` | `OutΔ` | H2H | `v4` `240k` MinPk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **`pb122-m-lax50`** | `pb-122` | Major okimono | **`62.5%`** (`64g`) | **`+23.4 pp`** (`z=+3.30`) | **`+20.8 pp`** (`+10g/48`) | `+31.2 pp` | `41.7%` | `21.9%` |
| **`s02-m-lax-curse`** | `sample-02` | Same-shell opt | **`57.8%`** (`64g`) | **`+14.1 pp`** | **`+23.3 pp`** (`+10g/43`) | `-4.8 pp` | `58.3%` | `14.8%` |
| **`pb84-cln-skarm50`** *(Step 4)* | `pb-84` | Major okimono | **`53.9%`** (`128g` true-CRN) | **`+7.03 pp`** vs clean (`+7.81` raw) | **`+25.7 pp`** (`+9g/35`) | **`+0.0 pp`** | **`62.5%`** (`16g`) | `7.8%` (`Skarm 28.1%`) |
| **`pb63-m-gar55`** *(Step 4)* | `pb-63` | Major okimono | **`59.4%`** (`128g` true-CRN; `57.3%` `192g`) | **`+3.91 pp`** (`+3.65` `192g`) | **`+17.9 pp`** (`+5g/28`) | **`+0.0 pp`** | **`57.1%`** (`28g`) | `10.9%` (`Gar 15.6%`) |
| **`ayagen-m-zap50`** | `mjj-2015-ayagen` | Major okimono | **`65.6%`** (`64g`) | **`+10.9 pp`** | **`+14.3 pp`** (`+4g/28`) | `+8.3 pp` | `50.0%` | `28.1%` |
| **`pb85-m-cune50`** | `pb-85` | Major okimono (L50) | `46.9%` (`64g`) | **`+6.25 pp`** | **`+12.9 pp`** (`+4g/31`) | **`+0.0 pp`** | `41.7%` | `28.1%` |
| **`rraikou-m-cune50`** | `mjj-2020-rraikou` | Major okimono | **`54.2%`** (`96g`; `60.2%` Champs10) | **`+5.2 pp`** | **`+10.4 pp`** (`+5g/48`) | **`+0.0 pp`** | `16.7%` | **`32.4%`** (#1!) |
| **`pb66-m-wak55`** | `pb-66` | Major okimono | **`61.5%`** (`96g`) | `+2.1 pp` | **`+7.1 pp`** (`+1g/14`) | `+1.2 pp` | **`66.7%`** | `14.1%` |
| **`offantom-m-wak50`** | `mjj-2014-offantom` | Major okimono (L50) | **`53.1%`** (`64g`) | **`+14.1 pp`** | `+3.1 pp` (`+1g/32`) | `+25.0 pp` | **`83.3%`** | `15.2%` |

### 3.2 Historical 10×10 Super-Champions RR (`step3b-champs10-rr`, `540g`, `12g/edge`) — Bootstrap Instability & Hitchhiker Autopsy

> [!WARNING]
> **Do not treat the `12g/edge` `step3b-champs10-rr` Nash weights as a certified `policy-v2` equilibrium.** Seat-pair bootstrap (`B = 1,000`) shows `90%` weight CIs starting at **`0.0%` for all 6 active shells** (`rraikou-m-cune50` `[0.0, 55.7%]`, `pb63-m-esp55` `[0.0, 50.4%]`, `pb122-m-lax50` `[0.0, 44.9%]`, `pb84-cln-sking50` `[0.0, 43.0%]`, `sample-08` `[0.0, 35.0%]`, `pb66-m-wak55` `[0.0, 32.6%]`), flipping just 2 games (`~1.1 SE`) on 1 edge (`pb84` vs `pb122` or `sample-08`) collapses `pb84-cln-sking50` from `19.1% -> 0.0%`, and the matrix omitted `pb84-m-cloysurf`, `pb84-cln-skarm50`, parent `pb-63`, and `pb63-m-gar55`. Accordingly, `data/team-pool-v4/flagship-mixture.json` ships an honest **provisional uniform 5-shell Tier-S core** until a `>=48g/edge` `policy-v2` Nash RR is run.

Forensic autopsy of `pb84-cln-sking50` inside `step3b-champs10-rr` (`108g`, `50.00%` overall) proved that **Slowking was a pure genetic hitchhiker riding on `pb-84`'s cleaned 4-mon core (`Snorlax55 / Raikou55 / Machamp50 / Gold Berry Surf Cloyster50`)**:
- **Slowking picked (`19/108 = 17.6%`, `0` leads)**: **`7/19 = 36.8%`**.
- **Slowking benched (`89/108 = 82.4%`)**: **`47/89 = 52.8%`** (`+16.0 pp` higher!).
- On 3 of 4 winning edges (`pb122-m-lax50` `7/12`, `s0053-lefties` `7/12`, `pb-89` `7/12`), Slowking was picked **`0/12` times**; on `pb66-m-wak55` (`9/12`), Slowking went `2/3 (66.7%)` vs benched `7/9 (77.8%)`; on `pb63-m-esp55` (`6/12`), Slowking went `0/3 (0.0%)` vs benched `6/9 (66.7%)`; on `ayagen-m-zap50` (`4/12`), Slowking went `3/10 (30.0%)`.
- Replacing `pb84-cln-sking50`'s payoffs with ONLY its 89 non-Slowking games *raises* its RR mean **`50.00% -> 53.40%`** and raises its Nash weight **`19.1% -> 19.5%`**.

### 3.3 Final Self-Referential `240k` `team-pool-v4` Census (`step4b-v4-census-k4`, `4,356` Previews, `0 CU` Local)

Replacing the 9 parents with the 9 causally verified `policy-v2` upgrades (including `pb84-cln-skarm50` and `pb63-m-gar55`) eliminated every `0.0%` absolute okimono across all 33 teams (`2 -> 0`), cut `< 5%` okimonos from **`6 -> 2`** (only untouched `pb-87` `Miltank50 3.9%` and `hc75-top8-tsuru` `Golem50 4.7%`), cut `< 10%` okimonos (`13 -> 9`, with `pb63-m-gar55` cleanly eliminating `pb63-m-esp55`'s `7.8%` `Blissey50` slot to `10.9%`), and raised mean minimum slot pick rate from **`12.96% -> 15.39%`** (`+2.43 pp`).

---

## 4. Step 4: True Shared-Seed CRN Forensic Audit (`step4-true-crn-audit`, `1,840g` Strict LIVE)

### 4.1 Root Cause of Earlier False Positives (`tools/okimono-audit.py` Fake-CRN Bug)

Auditing the raw `results.jsonl` files from Steps 1–3 uncovered that `tools/okimono-audit.py` shared `battleSeed` across group members but hashed the subject ID `m` into agent search seeds (`s0`, `s1`). Consequently, even when two single-slot variants brought **byte-identical 3-Pokémon teams** (slot 6/2 unpicked on both sides, `sameBoth`), MCTS and preview RNG diverged (`+3.0g / 14` on `pb84-cln-sking50` vs `pb84-m-cloysurf`, `+2.0g / 6` = 100% of `pb63-m-esp55`'s Panel-A lift). We fixed `tools/okimono-audit.py` to share identical `(battleSeed, s0, s1)` across all members of a CRN group (`bEq = 128, sEq = 128`), and ran **`1,840` strictly NO-CACHE LIVE games (`128g`/candidate across all 32 `v3` shells + `16g` direct H2H)** with proper same-slot Major controls.

### 4.2 Group 1: Clean `pb-84` Slot-6 Showdown (`pb84_slot6_audit`, `128g`/arm vs `pb84-m-cloysurf`)

All clean variants share byte-identical Slots 1–5 (`Snorlax55@Miracle / Raikou55@Magnet / Machamp50@Scope / Cloyster50@Gold[Surf/IB/Spikes/Exp] / Gligar50@BrightPowder`), swapping ONLY Slot 6 `@ Leftovers`:

| Rank | Candidate (Slot 6 `@ Leftovers`) | 128g WR | vs `cloysurf` (`z_o`) | s6 Pick% | s6 InWR | **CRN `InΔ`** | **CRN `OutΔ`** | H2H (`16g`) | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| **1** | **`pb84-cln-skarm50`** (**Major: Skarmory50**) | **`53.91%`** | **`+7.03 pp`** (`+1.03`) | `27.3%` | **`65.7%`** | **`+25.7 pp` (`+9.0g`)** | **`+0.0 pp`** | **`62.5%`** | **PROMOTED** (`+14.06 pp`, `z=+2.46` vs Slowking!) |
| 2 | `pb84-cln-pilo50` (Minor: Piloswine50) | `52.34%` | `+5.47 pp` (`+1.07`) | `42.2%` | `46.3%` | `-5.6 pp` (`-3.0g`) | `+13.5 pp` | `43.8%` | Rejected (`InΔ < 0`, bench-carried) |
| 3 | `pb84-cln-cune50` (Major: Suicune50) | `49.22%` | `+2.34 pp` (`+0.41`) | `88.3%` | `47.8%` | `+1.8 pp` (`+2.0g`) | `+6.7 pp` | `37.5%` | Beat Slowking `+9.38 pp` (`z=+1.83`) |
| 4 | `pb84-cln-star50` (Major: Starmie50) | `48.44%` | `+1.56 pp` (`+0.25`) | `42.2%` | `48.1%` | `+1.9 pp` (`+1.0g`) | `+1.4 pp` | `50.0%` | Beat Slowking `+8.59 pp` (`InΔ +22.2 pp`) |
| 5t | `pb84-m-cloysurf` (Clean Base: Umbreon50) | `46.88%` | `BASE` | `15.6%` | — | — | — | — | Baseline |
| 5t | `pb84-cln-eggy50` (Major: Exeggutor50) | `46.88%` | `+0.00 pp` (`0.00`) | `55.5%` | `47.9%` | `+4.2 pp` (`+3.0g`) | `-5.3 pp` | `68.8%` | Beat Slowking `+7.03 pp` |
| 7 | `shell:pb-84` (Raw `v3` parent) | `46.09%` | `-0.78 pp` (`-0.13`) | `18.0%` | — | `s4 -1.0 pp` | `+0.0 pp` | `68.8%` | Clean Cloyster alone was flat (`+0.8 pp`) |
| 8 | `pb84-cln-toed50` (Minor: Politoed50) | `41.41%` | `-5.47 pp` (`-1.23`) | `30.5%` | `38.5%` | `-12.8 pp` (`-5.0g`) | `-2.2 pp` | `31.2%` | Rejected |
| **9** | **`pb84-cln-sking50`** (**Retracted: Slowking50**) | **`39.84%`** | **`-7.03 pp`** (`-1.27`) | `21.1%` | **`40.7%`** | **`+0.0 pp` (`0.0g`)** | **`-8.9 pp`** | **`43.8%`** | **RETRACTED** (dead last `9th/9`; pooled `256g` `+0.39 pp`, `InΔ -1.75 pp`) |

### 4.3 Group 2: `pb-63` Slot-2 Showdown (`pb63_slot2_audit`, `128g`/arm vs `shell:pb-63`)

| Rank | Candidate (Slot 2) | 128g WR | vs `pb-63` (`z_o`) | s2 Pick% | s2 InWR | **CRN `InΔ`** | **CRN `OutΔ`** | H2H (`16g`) | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| **1** | **`pb63-m-gar55`** (**Major: Gengar55@Gold**) | **`59.38%`** | **`+3.91 pp`** (`+0.80`) | `21.9%` | **`67.9%`** | **`+17.9 pp` (`+5.0g`)** | **`+0.0 pp`** (`sameBoth 41:0g`) | **`56.2%`** (`57.1%` `28g`) | **PROMOTED** (`+8.59 pp`, `z=+1.73` vs Espeon; `192g` `3/3` positive `+3.65 pp`) |
| 2t | `shell:pb-63` (Parent: Venusaur55) | `55.47%` | `BASE` | `3.1%` | — | — | — | — | Baseline |
| 2t | `pb63-m-zam55` (Major: Alakazam55@PRZ) | `55.47%` | `+0.00 pp` (`0.00`) | `3.9%` | `40.0%` | `-40.0 pp` (`-2.0g`) | `+1.6 pp` | `56.2%` | Still ignored (`3.9%` pick) |
| **4** | **`pb63-m-esp55`** (**Retracted: Espeon55@PRZ**) | **`50.78%`** | **`-4.69 pp`** (`-1.03`) | `19.5%` | **`44.0%`** | **`-4.0 pp` (`-1.0g`)** | **`-4.9 pp`** | `56.2%` | **RETRACTED** (pooled `192g` `-1.04 pp`, `InΔ -5.71 pp`, `0/3` runs positive `InΔ`) |

---

## 5. Next Steps (Three-Body Co-Evolution Loop)

1. **Properly Powered `policy-v2` Nash Certification (`>=48g/edge`)**: Run an adequately powered `policy-v2` round-robin including `pb63-m-gar55`, `pb84-cln-skarm50`, `rraikou-m-cune50`, `pb122-m-lax50`, `pb66-m-wak55`, `sample-08`, `ayagen-m-zap50`, `s0053-lefties`, and `pb-6` before replacing the provisional uniform 5-shell mixture in `data/team-pool-v4/flagship-mixture.json`.
2. **Web / Belief-Pool Rollout (when ready)**: Wire `data/team-pool-v4/` into `tools/build-known-prior.mjs`, `web/src/data.ts`, `web/src/pdinput*.ts`, `web/src/team-names.ts`, and regenerate `data/belief-pool-v4/` (deliberately kept `data/team-pool-v3/` untouched in this step so existing tests & web manifests stay green).
3. **Next NN Training Cycle (`CF-V_net v3`)**: Retrain `CF-V_net` on `belief-pool-v3` + `team-pool-v4` self-play + the OOD minor-splice over-pick traps (`Scizor`, `Clefable`, `Weezing`, `Vaporeon55`, `Muk`) identified in §2.2 to close the three-body OOD over-valuation gap!
