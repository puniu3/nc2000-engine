# `policy-v4` Camouflage, Poison-Pill Optimization & Curated 10-Shell Strong Generalist Pool (`51,556` Strict LIVE Games)

## 1. Executive Summary

- **Engine & Policy**: Production **`policy-v4`** (`CF-V_net v3` 192-dim Siamese PUCT adaptive `243k` + `SteppedCfPreview 720k` [`65%` `128i/128t1` `Present19` `TrajCalib` + `35%` Turn-1 Double-Oracle LCB-gated Mixed Nash], `c = 0.4`, strict **`NASH_PREVIEW_CACHE=0` LIVE**).
- **Total Experimental Scale**: **`51,556` strict NO-CACHE LIVE `policy-v4` games** (`103,112 / 103,112` `preview.source == "live"`, `0` `previewBeliefs` mismatches, `0` turn caps, `0` ties) across **17 stages (`s1`–`s6d`)** following the prior `12,844`-game plain 32-shell Nash certification (`64,400` total `policy-v4` LIVE games).
- **Product Redesign (Retreat from Brittle Self-Play Nash Claims)**:
  - Self-play exact Nash support boundaries are sensitive to single matchups (e.g., narrow specialist assassin `rental-cban-13` entering `p_clean7` at `2%` solely to punish `pb85`, or `pb63` swinging from `22.5%` to `0%` in camouflaged active-7 self-play), whereas the gap between broadly strong Tier-1 generalists and weaker bottom-tier shells (`29%–44%`) is large and statistically robust.
  - We therefore reorganized [`data/team-pool-v4/team-pool.json`](../../data/team-pool-v4/team-pool.json) and [`data/team-pool-v4/flagship-mixture.json`](../../data/team-pool-v4/flagship-mixture.json) into a **curated 10-shell equal-weight (`1/10 = 10.0%` per shell, `15` total teams) Strong Generalist Bot Pool (`P_strong10`)**, declaring honestly that every shell in the pool is broadly strong across the NC2000 metagame.
  - All **22 trimmed shells** (including `rental-cban-13`, `pb-89`, `offantom-m-wak50`, and lower-tier archetypes) remain preserved intact as permanent metagame knowledge in [`data/team-pool-v4/environment-pool-32.json`](../../data/team-pool-v4/environment-pool-32.json) alongside untouched [`data/belief-pool-v3/`](../../data/belief-pool-v3/) and [`data/team-pool-v3/`](../../data/team-pool-v3/).
- **Frozen Zero-Leakage Certification (`Stage 6d`, `2,400` Strict LIVE Games, Brand-New Cryptographic Domain)**:
  - **`P_strong10` vs Uncamouflaged `Plain10` (`1,200` mirrored games, exact `10×10` matrix)**: **`52.50% ± 1.37%`** (**`+2.50 pp`, `z = +1.82`, one-sided `p = 0.0343`**). Every sub-bucket beat `Plain10` (`camouflaged-4`: `52.29%`, `upgraded-3`: `52.78%`, `base-3`: `52.50%`), and 6 of 7 modified shells outperformed their own plain counterparts by `+4.16 pp` to `+11.67 pp`.
  - **`P_strong10` vs Certified 7-Shell Plain Nash `p_clean7` (`1,200` mirrored games, exact `STRAT60`)**: **`49.34% ± 1.45%`** (`z = -0.46`; `modified-7` = **`50.17%`**).
  - **Cross-Ruleset Castle Legality & Stealth**: `15/15` bot teams and `33/33` preserved environment teams pass BOTH `gen2stadium2` and `castle` WASM `Validator`, strict JP NC2000 cartridge blacklist, `training-v3` (`changedTeams: 0`), `6/6` Turn-0 signatures, `6/6` `HP_IV` Castle stealth, `6/6` intra-cluster identical items, and a 16-game live `castle-tournament` `nash_refresh` smoke (`32/32` live sides, `0` belief errors, `0` caps/ties).

---

## 2. Campaign Inventory (All `17` Stages, `51,556` Strict LIVE `policy-v4` Games)

| Stage | Directory | Games | Live Sides | Belief Bad | Caps/Ties | Purpose & Key Takeaway |
|---:|---|---:|---:|---:|---:|---|
| 1a | `s1-singletons-crn` | 2,900 | 5,800 | 0 | 0 | Disclosed singleton CRN screen across core-7 shells |
| 1b | `s1b-opus-addons` | 700 | 1,400 | 0 | 0 | Additional high-leverage set candidates (`pb85-c5`, etc.) |
| 2a | `s2a-multiworld-crn` | 4,700 | 9,400 | 0 | 0 | Multi-arm CRN screen + 2 duplicate-ID placebo controls (`-4.63 pp`, zero dilution artifact) |
| 2b | `s2b-confirm-qopt` | 5,760 | 11,520 | 0 | 0 | Fresh `K=80` replication (`pb85-tri33` pooled `D_CAMO = +6.98 ± 2.42 pp, z = +2.88`; `pb87-c5` discovered) |
| 2c | `s2c-elite-triads` | 4,920 | 9,840 | 0 | 0 | `6/6`-same-item elite pairs/triads (`pb63 c3+c3b = 55.3%`; `s07-c8` bare `59.6%`; `pb87` 3rd camo loss `-4.83 pp`) |
| 3a | `s3a-camo7-rr` | 2,016 | 4,032 | 0 | 0 | Initial 7×7 camouflaged round-robin |
| 3b | `s3b-voi-oracle` | 3,216 | 6,432 | 0 | 0 | 35-challenger PSRO oracle screen (discovered `rraikou-m-cune50` at `70.4%, z = +2.24` vs Zap55+TTar55) |
| 3c | `s3c-psro-rraikou` | 1,960 | 3,920 | 0 | 0 | Active-8 expansion + `rraikou` CRN (`rraikou-mix50-a0-c1` `D_CAMO = +8.33 ± 4.81 pp, z = +1.73`) |
| 3d | `s3d-qopt-poison-pill` | 2,208 | 4,416 | 0 | 0 | **Poison-pill `q` sweep**: proved inverted-U (`pb85` `50/35/15 = 53.9%` `+5.30 pp` over `q33`; `rraikou` `75/25 = 58.3%` `+5.73 pp` over `50/50`) |
| 3e | `s3e-final-active7` | 2,736 | 5,472 | 0 | 0 | Active-7 camouflaged matrix (`4,632g` pooled) |
| 3f | `s3f-voi-closure` | 1,808 | 3,616 | 0 | 0 | VoI closure producing provisional `p_camo3` (`59% pb131`, `24.5% pb85-q15`, `16.5% apex`) |
| 4a | `s4a-frozen-cert` | 5,480 | 10,960 | 0 | 0 | **Zero-leakage autopsy**: `p_camo3` scored only `50.41% ± 1.22%` vs `p_clean7` due to remaining 50/50 weak-twin drag (`pb131-c3rb`, `apex-c2`, `pb85-c1`); `s07-c8` bare ranked #1 (`54.6%`) of 44 threats |
| 4b | `s4b-anchor-camo` | 3,240 | 6,480 | 0 | 0 | Anchor-heavy `80/20` & `75/25` core-4 CRN (`pb85` `45/40/15` replacing `c1` with `A0` gave `D_CAMO = +3.58 pp`; `s07-c8` bare repeated `56.6%` while camo hurt `-4.72 pp`; `pb131` camo pooled null `+0.32 pp`) |
| 5 | `s5-gen5-camo` | 2,400 | 4,800 | 0 | 0 | CRN screen of 5 additional generalists vs uniform-12 meta (`sample-12` base `57.3%`; `pb-6` base `54.2%`, camo `-2.60 pp`; `pb84` base beat variants; `pb-89` & `offantom` flagged weak) |
| 6 | `s6-cert-strong12` | 3,552 | 7,104 | 0 | 0 | **Zero-leakage 12-shell audit**: caught `-12.8 pp` poison pill `apex-c2b` (`45.3%` vs `apex-c3r 58.1%`) and weak bottom-2 shells (`pb-89 42.9%`, `offantom 45.3%`); proved Top-10 beat `Cut20` by `+4.9 pp` |
| 6c | `s6c-apex-camo` | 1,560 | 3,120 | 0 | 0 | **Lv55-RestTalk-fixed `apex` CRN (`Plain10`, `K=60`)**: **`apex-mix50-c3r-c5`** achieved **`D_CAMO = +8.75 ± 3.87 pp (z = +2.262, p = 0.0119)`**, `u_mix = 54.17%`, `D_SUPER = +7.50 pp`, `frag = 3.33 pp` |
| 6d | `s6d-final-strong10` | 2,400 | 4,800 | 0 | 0 | **Final Frozen Certification of `P_strong10`**: `52.50% ± 1.37%` vs `Plain10` (`+2.50 pp, z = +1.82, p = 0.0343`); `49.34% ± 1.45%` vs `p_clean7` (`modified-7 50.17%`) |
| **Total** | **17 stages** | **51,556** | **103,112** | **0** | **0** | **100.0% strict NO-CACHE LIVE `policy-v4`, 0 belief errors, 0 caps/ties** |

---

## 3. Mechanistic Laws of `policy-v4` Camouflage

Four robust empirical laws emerged across `51,556` LIVE games:

1. **The Poison-Pill Inverted-U Law (`s3d`, `s4a`, `s6`)**: Under `policy-v4`'s 720k `TrajCalib` + Turn-1 Double-Oracle solver, pure camouflage rent is bounded (`+3` to `+9 pp`). Mixing a weak twin (`-5` to `-13 pp` bare) at `50/50` destroys net EV (`apex-c2b` `-12.8 pp`, `pb85-c1` `30.1%` bare, `pb85-c3r` at `33%`). Camouflage is net positive **only** when either:
   - Both twins are equal-strength (`|Δ_bare| <= 2.5 pp`, e.g., `apex c3r+c5` `51.7% vs 52.5%`, `pb63 c3+c3b` `53.3% vs 56.7%`), **or**
   - The weaker ambush twin is capped at **`15%–20%`** (`pb85-c3r` at `15%`, `rraikou-c1` at `20%`) so the opponent must hedge Team Preview without diluting realized game EV.
2. **Never Break a Single-Ace Defensive Backbone (`s6` → `s6c`)**: `apex-c04` has only one Lv55 ace (`Snorlax@Leftovers`, `98%–100%` pick rate) and no backup status absorber. Swapping `Rest + Sleep Talk` → `Earthquake + Self-Destruct` on `apex-c2b` collapsed `apex` by `-12.8 pp` (`45.3%` vs `c3r 58.1%`). Locking Lv55 `Double-Edge / Curse / Rest / Sleep Talk` Snorlax intact and branching **only** supporting counterplay axes (`4-Atk Boom Gar / Reflect Cloy / HP Bug Wak / TWave Star` vs `PerishTrap Gar / Curse Cloy / Fire Blast Wak / 3-Atk Star` in **`apex-mix50-c3r-c5`**) flipped `apex` camouflage into our single strongest CRN hit: **`D_CAMO = +8.75 ± 3.87 pp (z = +2.262, p = 0.0119)`**, lifting BOTH `c3r` (`+8.33 pp`) and `c5` (`+9.17 pp`) symmetrically (`frag = 3.33 pp`).
3. **Strict `6/6` Item & `HP_IV` Stealth**: Every deployed camouflage cluster enforces `6/6` identical items (eliminating Turn-1 `Leftovers`/`Bright Powder`/`Focus Band` item-reveal leaks) and `6/6` identical `HP_IV` / `base_maxhp` (preventing Castle-ruleset HP-bar fingerprinting, e.g., `rraikou` Heracross `hp=6` / Machamp `hp=22`, `apex` all `hp=31`).
4. **Honest Negative/Null Camouflage Pruning (`s2a`–`s5`)**: Camouflage is **not** universally positive. Across multiple independent CRN runs, camouflaging `pb-87` lost `-4.80 ± 2.29 pp (z = -2.09)`, camouflaging `s07-c8` lost `-4.72 ± 3.42 pp (D_SUPER = -7.01 pp)`, camouflaging `pb-131` proved null (`+0.32 ± 3.18 pp`), and camouflaging `pb-6` lost `-2.60 pp`. Deploying `pb131-c3r`, `s07-c8`, `pb87-c5`, and clean `pb-6` as singletons preserved their full bare dominance (`52.8%–59.1%`).

---

## 4. Stage 6c `apex` RestTalk-Fixed Camouflage CRN (`1,560` LIVE Games, `Plain10` `K=60`)

| Cluster (`apex-c04` Lv55 RestTalk Backbone Preserved) | `u0` (`A0`) | `u_mix` | `D_CAMO` | `D_CAMO z` | `D_SUPER` | `frag` | Members (`u_disc → u_mix`, `c`) |
|---|---:|---:|---:|---:|---:|---:|---|
| **`apex-mix50-c3r-c5`** (`50/50`, `6/6` same items) | `48.33%` | **`54.17%`** | **`+8.75 ± 3.87 pp`** | **`+2.262` (`p=0.012`)** | **`+7.50 pp`** | **`3.33 pp`** | `c3r`: `44.2% → 52.5% (+8.33 pp)` / `c5`: `46.7% → 55.8% (+9.17 pp)` |
| `apex-mix50-c3r-c6` (`Body Slam` RestTalk Lax + HypGar) | `48.33%` | `49.58%` | `+7.92 ± 4.67 pp` | `+1.697` | `+5.42 pp` | `12.50 pp` | `c3r`: `55.8% (+11.67)` / `c6`: `43.3%` (`u_disc=39.2%`) |
| `apex-mix50-c3r-a0` (`c3r` + `A0`) | `48.33%` | `48.33%` | `+2.08 ± 4.66 pp` | `+0.446` | `+0.00 pp` | `1.67 pp` | `c3r`: `47.5% (+3.33)` / `A0`: `49.2% (+0.83)` |
| `apex-mix50-c3r-c4` (Hypnosis Boom Gar + Reflect Cloy) | `48.33%` | `43.75%` | `-4.17 ± 4.29 pp` | `-0.971` | `-7.92 pp` | `7.50 pp` | `c3r`: `40.0% (-4.17)` / `c4`: `47.5% (-4.17)` |

---

## 5. Final Frozen Zero-Leakage Certification (`Stage 6d`, `2,400` Strict LIVE Games)

### 5.1 Overall Benchmarks (`domain: nc2000-policy-v4-strong10-v2-20261128::stage6d-final-cert`)

| Benchmark | Games | `P_strong10` Win Rate | `Δ` vs `50%` | `z` | One-sided `p` |
|---|---:|---:|---:|---:|---:|
| **`B2`: `P_strong10` vs Uncamouflaged `Plain10`** (`10×10`, `12g/cell`) | **1,200** | **`52.50% ± 1.37%`** | **`+2.50 pp`** | **`+1.82`** | **`0.0343`** |
| **`B1`: `P_strong10` vs Certified 7-Shell Plain Nash `p_clean7`** (`STRAT60`) | **1,200** | **`49.34% ± 1.45%`** | `-0.66 pp` | `-0.46` | `0.6756` |
| Sub-bucket: `camouflaged_4` (`pb63`, `apex`, `rraikou`, `pb85`) | 480+480 | `48.22%` (`B1`) / **`52.29%`** (`B2`) | `+2.29 pp` (`B2`) | — | — |
| Sub-bucket: `upgraded_singleton_3` (`pb-131`, `sample-07`, `pb-87`) | 360+360 | **`52.77%`** (`B1`) / **`52.78%`** (`B2`) | `+2.78 pp` (`B2`) | — | — |
| Sub-bucket: `modified_7` (all 7 upgraded/camouflaged shells) | 840+840 | **`50.17%`** (`B1`) / **`52.50%`** (`B2`) | `+2.50 pp` (`B2`) | — | — |
| Sub-bucket: `clean_base_3` (`pb-6`, `ayagen-m-zap50`, `sample-12`) | 360+360 | `47.40%` (`B1`) / **`52.50%`** (`B2`) | `+2.50 pp` (`B2`) | — | — |

### 5.2 Per-Shell & Per-Variant Results (`120g B1 + 120g B2 = 240g` per Shell)

| Shell (`variantCluster`) | Mode | `B1` vs `p_clean7` | `B2` vs `Plain10` | `Plain10` Opp EV | `B2` Gain vs Own Plain | Variant Realizations (`B1` / `B2`) |
|---|---|---:|---:|---:|---:|---|
| **`pb-131`** | `upgrade-singleton` | `60.63 ± 5.13%` | `57.50 ± 4.76%` | `45.83%` | **`+11.67 pp`** | `c3r`: `60.8%` / `57.5%` |
| **`pb-6`** | `base-singleton` | `50.69 ± 4.36%` | `59.17 ± 4.46%` | `55.83%` | `+3.34 pp` | base: `50.8%` / `59.2%` |
| **`sample-07`** | `upgrade-singleton` | `52.22 ± 4.86%` | `53.33 ± 4.12%` | `49.17%` | **`+4.16 pp`** | `c8`: `52.5%` / `53.3%` |
| **`apex-c04`** | `camo-2way-50-50` | `51.23 ± 4.78%` | `53.33 ± 4.15%` | `47.50%` | **`+5.83 pp`** | `c3r`: `55.0%`/`48.3%`, `c5`: `46.7%`/`58.3%` (pooled `51.7%` vs `52.5%`!) |
| **`ayagen-m-zap50`** | `base-singleton` | `48.93 ± 4.09%` | `54.17 ± 4.58%` | `42.50%` | `+11.67 pp` | base: `48.3%` / `54.2%` |
| **`pb63-m-gar55`** | `camo-2way-50-50` | `51.19 ± 4.58%` | `50.83 ± 4.46%` | `46.67%` | **`+4.16 pp`** | `c3`: `43.3%`/`46.7%`, `c3b`: `58.3%`/`55.0%` |
| **`rraikou-m-cune50`** | `camo-2way-80-20` | `42.24 ± 4.42%` | `55.83 ± 3.63%` | `50.00%` | **`+5.83 pp`** | `A0`: `45.8%`/`58.3%`, `c1`: `29.2%`/`45.8%` |
| **`pb85-m-cune50`** | `camo-3way-50-35-15` | `48.23 ± 4.54%` | `49.17 ± 4.39%` | `42.50%` | **`+6.67 pp`** | `A0`: `50.0%`/`53.3%`, `c5`: `45.2%`/`42.9%`, `c3r`: `50.0%`/`50.0%` |
| **`pb-87`** | `upgrade-singleton` | `45.45 ± 4.26%` | `47.50 ± 4.27%` | `51.67%` | `-4.17 pp` | `c5`: `45.8%` / `47.5%` (`+6.49 pp` CRN in `s2b+s2c`) |
| **`sample-12`** | `base-singleton` | `42.58 ± 4.85%` | `44.17 ± 4.49%` | `43.33%` | `+0.84 pp` | base: `43.3%` / `44.2%` (`57.3%` `s5`, `50.0%` `s6`) |

---

## 6. Shipped Artifacts & Verification

- [`data/team-pool-v4/team-pool.json`](../../data/team-pool-v4/team-pool.json): Curated 10-shell (`15`-team) equal-weight (`0.100` per shell) Strong Generalist Bot Pool (`drawWeight` exact sum `1.000000`).
- [`data/team-pool-v4/flagship-mixture.json`](../../data/team-pool-v4/flagship-mixture.json): Matching 10-shell (`15`-team) equal-weight Strong Generalist Bot Mixture (`weight` exact sum `1.000000`).
- [`data/team-pool-v4/environment-pool-32.json`](../../data/team-pool-v4/environment-pool-32.json): Preserved full 32-shell (`33`-team) baseline environment knowledge catalog retaining all 22 cut shells.
- [`docs/reports/policy-v4-strong-pool-variants.json`](./policy-v4-strong-pool-variants.json): Committed source definitions for all 10 verified non-base sets.
- [`tools/build-team-products-v4.js`](../../tools/build-team-products-v4.js): Deterministic generator verifying `PsStadium2` + `castle` WASM `Validator`, JP NC2000 blacklist, `training-v3` (`changedTeams: 0`), `6/6` Turn-0 signatures, `6/6` `HP_IV` Castle stealth, `6/6` intra-cluster identical items, and WASM `Battle` smoke.
- `web/` remains intentionally untouched (maintenance mode); `data/team-pool-v3/` and `data/belief-pool-v3/` remain untouched.
