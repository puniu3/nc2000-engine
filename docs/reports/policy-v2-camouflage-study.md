# Round 2.5: Same-Frame Support & Borderline Camouflage Study (`policy-v2`, `team-pool-v4`)

- **Date**: 2026-11-25
- **Policy Epoch**: `policy-v2` (`CF-V_net v2` adaptive PUCT `243k` + `SteppedCfPreview 240k`, `c = 0.4`, `maxTurns = 500` cap = `0.5`, strict `NASH_PREVIEW_CACHE=0` LIVE)
- **Baseline Equilibrium**: Certified 6-shell Bayes-Minimax equilibrium `p_clean6` (`pb66-m-wak55 24.0%`, `mjj-2014-thdrcomb 21.5%`, `rraikou-m-cune50 21.0%`, `pb63-m-gar55 13.0%`, `sample-07 13.0%`, `pb-6 7.5%`; see [`policy-v2-nash-certification.md`](./policy-v2-nash-certification.md))
- **Experimental Scale**: **`8,016` strict NO-CACHE LIVE games (`16,032 / 16,032` `preview.source == "live"`)** across 4 campaigns (`15` twins screened in 4-arm CRN; `5` finalists confirmed at `K = 144` fresh pairs/arm = `4,896` fresh confirmatory games), `0` `previewBeliefs` errors, `0` focal Turn-0 mismatches across all `2,304` CRN tuples, `0` seed overlaps across all `10` certification + camouflage campaigns (`15,792` grand-total games / `31,584` live sides).
- **Independent Forensic Audit (`af3055d1` & `6d691ce5`)**: **`CERTIFIED PASS (CLOSED)`**.

---

## 1. Executive Summary

We investigated whether **same-frame camouflage clusters** $M_{0.5} = \frac{1}{2}A_0 + \frac{1}{2}T$ — where twin $T$ shares the **exact** Turn-0 public signature `(species, level, gender, has_item)` AND exact `6/6` `HP_IV / base_maxhp` with base shell $A_0$ — generate measurable **pure camouflage rent** ($\Delta_{\text{camo}}$) under production `policy-v2`'s Multi-World Bayesian preview (`SteppedCfPreview::solve_bayesian_rm_plus`) and in-battle belief tracker (`BlindSearch` + `BeliefTracker`), either on certified support shells (**Track A**) or on borderline non-support shells attempting a reverse Nash invasion (**Track B**).

### Bottom-Line Verdicts (Strictly Honoring Pre-Registered Holdout Gates)

1. **Flagship Mixture Unchanged (`0 / 5` pre-registered promotions)**: [`data/team-pool-v4/flagship-mixture.json`](../../data/team-pool-v4/flagship-mixture.json) remains at the certified 6-shell `p_clean6` equilibrium. Neither the 3 support-camouflage finalists (`0/3`) nor the 2 borderline invasion finalists (`0/2`) cleared their pre-registered fresh `K = 144` promotion thresholds.
2. **Pure Camouflage Rent Is Real, Twin-Specific, and Mechanistically Verified (`pb66-camo-a1-talklax-spin`)**:
   - On strictly fresh `K = 144` holdout (`288g/arm`), `pb66-camo-a1` (`6/6` identical items; `RestTalk` Snorlax55 + `Surf` Cloyster + `Rapid Spin` Starmie + `Roar` Steelix) replicated **`D_CAMO = +5.19 ± 2.80 pp`** (`z = +1.86`, `p_1s = 0.0317`, `BH_q3 = 0.095`, seat-symmetric `+5.87 / +4.51 pp`), driven by **`C1` (twin ambush rent) = `+9.36 ± 4.15 pp`** (`z = +2.26`, `p_1s = 0.0120`, survives 3-way Bonferroni **`q3 = 0.0361 < 0.05`**, `q5 = 0.060`).
   - Super-dominance replicated (`U0 = 51.11%`, `U1 = 47.29%`, `Umix = 54.39%`; `Umix − U1 = +7.10 ± 3.69 pp [z = +1.92]`), with concrete Turn-0 counter-pick ambushes against `mjj-2014-thdrcomb` (`+21.0 pp` C1), `pb63-m-gar55` (`+13.2 pp` C1, `+19.4 pp` C0 fog), and `pb66` mirror (`+7.4 pp` C1). Net `D_TOT = +3.28 ± 3.47 pp (z = +0.95)` missed the pre-registered `z >= +1.645` bar because its `-3.82 pp` `RestTalk` set-strength tax absorbed part of the rent.
   - Independently replicated on a second shell in Track B screening: `s02-camo-a1` achieved `D_CAMO = +11.55 ± 5.70 pp (z = +2.03, p_1s = 0.021)`, lifting `s02-m-lax-curse` from `37.3% -> 48.2%` (`5+/0-/10` across `p_clean6` opponents, though still `< 50%`).
3. **Bounded-Rationality "Anti-Camouflage" & `CF-V_net v2` Preview Biases**:
   - Adding camouflage is **not** a free generic lunch under bounded-rational `CF-V_net v2` search: on `mjj-camo-a1/b1`, `mix50` accidentally *rescued* `sample-07` off its losing `Electrode` lead (`34/38 → 21/38`, where `mjj` won `73.7%`) onto a much stronger `Marowak` lead (`0 → 15`, dropping `mjj` to `43–47%`, `C0 = -15.8 pp`), yielding statistically separated fresh `D_CAMO` vs `pb66-a1` (`p_2s = 0.013–0.045`).
   - Across all 6 Stage-1 mono-Normal `RestTalk` Snorlax twins, disclosed `243k` battle `D_SET` averaged **`-7.68 pp`** despite `+2.2 to +6.2 pp` `240k` preview scores — identifying a clear `CF-V_net v2` preview overvaluation bias to address in `policy-v3`.
   - Furthermore, auditing `crates/bot/src/blind.rs:2101-2105` exposed why raw `rootPolicy().mean` is biased when comparing 1-world vs 2-world preview runs (coarse-vs-polish visit-mix dilution), establishing direct `solve_cf_preview` exact Bayesian RM+ equilibrium evaluation (`maxDualityGap <= 0.0002`) as the correct screening metric.
4. **Major Bonus Finding — Multi-Slot Same-Shell Optimization Breaks the `0/30` Single-Slot Plateau (`s08-camo-a1`)**:
   - On strictly fresh `K = 144` holdout (`288g/arm`), **`s08-camo-a1-sleepEgg-eqtank-leftsdlax-mirwwzap`** crushed base `sample-08` by **`+11.47 ± 4.40 pp` (`z = +2.61`, `p_1s = 0.0046`, `Bonf2 = 0.0092`)**, winning strictly on **`6 / 6` certified flagship opponents** (`pb66 +22.1`, `mjj +6.5`, `rraikou +1.7`, `pb63 +13.2`, `sample-07 +10.5`, `pb-6 +18.2 pp`), lifting `sample-08` from `41.65%` to **`53.12 ± 3.11%`** disclosed (`54.80 ± 3.03%` under `mix50`, `C1 = +1.68 ± 3.30 pp`, zero preview-exploit collapse). While `z(U1 > 50%) = +1.00` did not clear the `>= +1.96` immediate Nash-invasion bar, `s08-camo-a1` is a **certified multi-slot same-shell upgrade** ready for `team-pool-v5`.

---

## 2. Exact Indistinguishability Contract & 4-Arm Shared-Seed CRN Decomposition

Code audit (`crates/bot/src/belief.rs:4985-5030`, `observe.rs:539-1152`, `preview_search.rs:208-390`, `nash_refresh.rs:549-891`) established the exact camouflage mechanics in `policy-v2`:

- **Turn-0 Indistinguishability**: `BeliefTracker` matches `(len, species, level, gender, preview_has_item)`. In NC2000 (`PsStadium2`), `observed_maxhp` remains `None` all game (`100`-denominator percentages), whereas `Castle` reveals exact `base_maxhp` on switch-in; all twins in this study enforced **`6/6` exact `HP_IV` / `base_maxhp`** for cross-ruleset stealth.
- **In-Battle World Filtering**: Worlds are eliminated on revealed moves (`|move|`), `Hidden Power` type effectiveness (`hp_type_mask`), direct item reveals, cross-slot Item Clause, and deductive upkeep exclusions (`Leftovers` at first damaged upkeep, `Berserk Gene` on first active update, status/heal berries on failure to trigger). Non-announcing passive items (`Pink Bow`, `Thick Club`, `Magnet`, `Scope Lens`, `Bright Powder`, `Polkadot Bow`) and 3rd/4th-slot surprise moves leak zero information until executed.
- **Exact 4-Arm Shared-Seed CRN (Explicit `previewSeeds`)**: On every mirrored seat-pair `(opp, blk, rowSeat)`, all 4 arms share identical `(battleSeed, seeds, previewSeeds)`:
  - `Arm 0`: $A_0 \mid \{A_0\}$ (disclosed base) — `Arm 0 == Arm 2` focal Turn-0 pick **100% bit-identical**.
  - `Arm 1`: $T \mid \{T\}$ (disclosed twin) — `Arm 1 == Arm 3` focal Turn-0 pick **100% bit-identical**.
  - `Arm 2`: $A_0 \mid M_{0.5}(A_0, T)$ (base under disguise); `Arm 3`: $T \mid M_{0.5}(A_0, T)$ (twin under disguise).
  - Exact causal decomposition:
    $$\Delta_{\text{total}} = \tfrac{1}{2}(U_2 + U_3) - U_0 = \tfrac{1}{2}\underbrace{(U_1 - U_0)}_{\Delta_{\text{set}}} + \underbrace{\tfrac{1}{2}(U_2 - U_0)}_{C_0/2} + \underbrace{\tfrac{1}{2}(U_3 - U_1)}_{C_1/2}$$
  - Empirical CRN variance reduction: `1.26x–1.92x` on `D_CAMO`, `1.21x–2.01x` on `C1`.

---

## 3. Campaign Summary (`8,016` Strict Live Games)

| Campaign | Scope & Stratification (`p_clean6`) | Games | Live Sides | Caps / Ties | T0 Mismatches | Key Outcome |
|---|---|---:|---:|---:|---:|---|
| **Track A S1** (`v4-camo-s1-4arm`) | `9` support twins × `K=24` (`48g/arm`), 4-arm shared-Arm0 | `1,536` | `3,072/3,072` | `1 / 0` | `0 / 432` | Mean `D_CAMO = +1.89 ± 2.41 pp`; strong `C1` heterogeneity (`Q=27.28, p=0.0006`); surfaced `pb66-a1`, `mjj-a1`, `mjj-b1` |
| **Track A S2** (`v4-camo-s2-confirm`) | Top `3` support finalists × `K=144` (`288g/arm`), 4-arm fresh | `3,168` | `6,336/6,336` | `0 / 0` | `0 / 864` | **`pb66-a1` replicated `D_CAMO=+5.19 pp (p=0.032)`, `C1=+9.36 pp (p=0.012, q3=0.036)`**; `mjj-a1/b1` collapsed (`0/3` promoted) |
| **Track B S1** (`v4-camo-borderline-s1`) | `6` borderline twins (`apex`, `pb84`, `s02`, `s08`) × `K=36` (`72g/arm`), 4-arm | `1,584` | `3,168/3,168` | `0 / 0` | `0 / 432` | `s02-a1` `D_CAMO=+11.55 pp (p=0.021)` (`Umix 48.2%`); surfaced standalone invaders `pb84-b1 (65.2%)` & `s08-a1 (62.6%)` |
| **Track B S2** (`v4-camo-borderline-s2`) | `pb84-b1` & `s08-a1` × `K=144` (`288g/arm`), 3-arm (`A0, A1, A3`) fresh | `1,728` | `3,456/3,456` | `0 / 0` | `0 / 576` | `pb84-b1` rejected (`U1=45.5%`); **`s08-a1` certified `+11.47 ± 4.40 pp (z=+2.61, p=0.0046, 6/6 opps+)`** (`U1=53.1% / U3=54.8%`) |
| **Total** | **Round 2.5 Support & Borderline Camouflage** | **`8,016`** | **`16,032/16,032`** | **`1 / 0`** | **`0 / 2,304`** | **`0` seed overlaps across all 10 campaigns (`15,792g`)** |

---

## 4. High-Power `K = 144` Confirmatory Results (`4,896` Fresh Games, Zero Screening Leakage)

| Finalist Twin | Base Shell | `U0` | `U1` (Disc) | `Umix` / `U3` | `D_SET` (`z`) | `D_CAMO` (`z`, `p_1s`) | `C1 = U3 − U1` (`z`, `p_1s`) | Pre-Registered Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`pb66-camo-a1-talklax-spin`** | `pb66-m-wak55` | `51.11%` | `47.29%` | `54.39%` *(mix)* | `-3.82 ± 4.48` (`-0.85`) | **`+5.19 ± 2.80`** (`+1.86`, `0.032`) | **`+9.36 ± 4.15`** (`+2.26`, `0.012`, `q3=0.036`) | `PATH_C` (`D_TOT=+3.28±3.47`, verified camo rent) |
| `mjj-camo-a1-swap-leftlax-mirttar-qclix` | `mjj-2014-thdrcomb` | `50.62%` | `49.60%` | `48.18%` *(mix)* | `-1.02 ± 3.94` (`-0.26`) | `-1.93 ± 2.19` (`-0.88`, `0.811`) | `-3.46 ± 3.16` (`-1.10`, `0.864`) | `PATH_C` (gutted `sample-07` `73.7%→48.7%`) |
| `mjj-camo-b1-pursuit-ttar-curselax-qclix` | `mjj-2014-thdrcomb` | `50.62%` | `42.98%` | `42.60%` *(mix)* | `-7.64 ± 4.17` (`-1.83`) | `-4.20 ± 2.54` (`-1.65`, `0.951`) | `-4.55 ± 3.70` (`-1.23`, `0.891`) | `PATH_C` (Stage-1 winner's curse) |
| `pb84-camo-b1-minteqrestlax-leftraik-keepchamp-curglig` | `pb84-cln-skarm50` | `51.39%` | `45.52%` | `48.96%` *(U3)* | `-5.87 ± 4.14` (`-1.42`) | — *(3-arm)* | `+3.44 ± 3.78` (`+0.91`, `0.182`) | `REJECTED_NULL` (`z_hom(U1)=-2.79`) |
| **`s08-camo-a1-sleepEgg-eqtank-leftsdlax-mirwwzap`** | `sample-08` | `41.65%` | **`53.12%`** | **`54.80%`** *(U3)* | **`+11.47 ± 4.40`** (`+2.61`, **`0.0046`**) | — *(3-arm)* | `+1.68 ± 3.30` (`+0.51`, `0.305`) | **`VERIFIED_SAME_SHELL_UPGRADE`** (`6/6` opps+) |

- **Grand 5-Twin `K = 144` `C1` Meta-Analysis (`5×5` exact covariance)**: Equal-weight `+1.29 ± 1.64 pp (z=+0.79)`, GLS `+0.79 ± 1.62 pp`, heterogeneity `Q = 8.49 (4 df, p = 0.075)`. All `3/3` non-`mjj` independent bases (`pb66`, `pb84`, `s08`) were positive (`+4.83 ± 2.17 pp` eq [`z = +2.22`] / `+4.27 ± 2.13 pp` IVW [`z = +2.00`]), offset by both `mjj` twins where `mix50` disrupted opponents off sub-optimal `CF-V_net` leads.

---

## 5. Preserved Reference Rosters (100% JP NC2000 & `castle-tournament` Legal, `6/6` Exact `HP_IV`)

### A. Certified Multi-Slot Same-Shell Upgrade: `s08-camo-a1-sleepEgg-eqtank-leftsdlax-mirwwzap` (`+11.47 ± 4.40 pp` vs `sample-08`)
- `Exeggutor L50 (M) @Quick Claw` `IV=31/31/31` `[Psychic / Sleep Powder / Leech Seed / Explosion]`
- `Machamp L50 (M) @Scope Lens` `IV=31/31/31` `[Cross Chop / Rock Slide / Rest / Sleep Talk]`
- `Miltank L55 (F) @Gold Berry` `IV=31/31/31` `[Body Slam / Earthquake / Curse / Milk Drink]`
- `Cloyster L50 (M) @PRZ Cure Berry` `IV=31/31/31` `[Surf / Ice Beam / Spikes / Explosion]`
- `Snorlax L50 (M) @Leftovers` `IV=31/31/31` `[Body Slam / Earthquake / Curse / Self-Destruct]`
- `Zapdos L55 (N) @Miracle Berry` `IV=31/22/26` `[Thunderbolt / Hidden Power Ice / Whirlwind / Rest]`

### B. Verified Pure Camouflage Mechanism #1: `pb66-camo-a1-talklax-spin` (`D_CAMO = +5.19 pp`, `C1 = +9.36 pp`, `6/6` Same Items)
- `Snorlax L55 (M) @Leftovers` `IV=31/31/31` `[Double-Edge / Curse / Rest / Sleep Talk]`
- `Cloyster L50 (M) @Gold Berry` `IV=31/31/31` `[Surf / Ice Beam / Spikes / Explosion]`
- `Zapdos L50 (N) @Mint Berry` `IV=31/22/26` `[Thunderbolt / Hidden Power Ice / Thunder Wave / Rest]`
- `Starmie L50 (N) @Miracle Berry` `IV=31/31/31` `[Surf / Thunderbolt / Rapid Spin / Recover]`
- `Steelix L50 (M) @Quick Claw` `IV=31/31/31` `[Earthquake / Rock Slide / Roar / Explosion]`
- `Marowak L55 (M) @Thick Club` `IV=31/26/26` `[Earthquake / Rock Slide / Hidden Power Bug / Swords Dance]`

### C. Verified Pure Camouflage Mechanism #2 (Screen): `s02-camo-a1-tboltcurseP2-sdfbwak-goldsdlax-twstar` (`D_CAMO = +11.55 pp`)
- `Jynx L50 (F) @(none)` `IV=31/2/31` `[Ice Beam / Psychic / Thief / Lovely Kiss]`
- `Starmie L50 (N) @Miracle Berry` `IV=31/2/31` `[Surf / Thunderbolt / Thunder Wave / Recover]`
- `Marowak L50 (M) @Thick Club` `IV=31/31/31` `[Earthquake / Rock Slide / Fire Blast / Swords Dance]`
- `Porygon2 L55 (N) @Leftovers` `IV=31/31/31` `[Double-Edge / Thunderbolt / Curse / Recover]`
- `Snorlax L50 (M) @Gold Berry` `IV=31/31/31` `[Body Slam / Earthquake / Curse / Self-Destruct]`
- `Zapdos L50 (N) @Scope Lens` `IV=31/22/26` `[Thunder / Hidden Power Ice / Rest / Sleep Talk]`
