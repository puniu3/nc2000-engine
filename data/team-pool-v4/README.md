# `team-pool-v4` (Curated 10-Shell Equal-Weight Strong Generalist Bot Pool & Preserved 32-Shell Environment Knowledge)

Generated deterministically by [`tools/build-team-products-v4.js`](../../tools/build-team-products-v4.js) from committed repo sources ([`data/team-pool-v3/team-pool.json`](../team-pool-v3/team-pool.json), [`docs/reports/okimono-upgrades-v2.json`](../../docs/reports/okimono-upgrades-v2.json), and [`docs/reports/policy-v4-strong-pool-variants.json`](../../docs/reports/policy-v4-strong-pool-variants.json)).

## Product Architecture

Rather than shipping brittle self-play exact-Nash support weights (where narrow specialist assassins such as `rental-cban-13` can enter at 2% solely to counter one shell, and marginal support weights swing sharply with self-play meta composition), `team-pool-v4` cleanly separates **the bot's playable strong pool** from **metagame environment knowledge**:

- **Bot Playable Pool**: [`data/team-pool-v4/team-pool.json`](./team-pool.json) and matching [`data/team-pool-v4/flagship-mixture.json`](./flagship-mixture.json) — **10 broadly strong NC2000 generalist shells** drawn with **equal probability (`1/10 = 10.0%` per shell, `15` total teams, exact weight sum `1.000000`)**. Every shell in this pool is empirically verified broadly strong under production **`policy-v4`** (`CF-V_net v3` 192-dim PUCT adaptive `243k` + `SteppedCfPreview 720k` [`65%` `128i/128t1` `Present19` `TrajCalib`], `c = 0.4`, strict `NASH_PREVIEW_CACHE=0` LIVE). Targeted **low-poison `6/6`-same-item camouflage** is applied **exclusively** to the 4 shells where shared-seed CRN proved net positive rent without weak-twin drag; 3 shells deploy CRN-verified upgraded singletons where camouflage proved null or negative; 3 shells preserve their verified clean bases.
- **Preserved 32-Shell Environment Knowledge**: [`data/team-pool-v4/environment-pool-32.json`](./environment-pool-32.json) preserves all **32 pairwise-distinct baseline shells (`33` teams)** intact (retaining all 22 cut shells — including narrow specialist `rental-cban-13` and lower-tier archetypes — as permanent metagame evaluation knowledge alongside untouched [`data/belief-pool-v3/`](../belief-pool-v3/) and [`data/team-pool-v3/`](../team-pool-v3/)).

Full experimental evidence:
- **51,556-Game `policy-v4` Camouflage, Poison-Pill Optimization & 10-Shell Strong Pool Certification**: [`docs/reports/policy-v4-strong-pool-camo.md`](../../docs/reports/policy-v4-strong-pool-camo.md)
- **Prior 12,844-Game `policy-v4` Plain 32-Shell Nash Certification (`p_clean7`)**: [`docs/reports/policy-v4-nash-certification.md`](../../docs/reports/policy-v4-nash-certification.md)
- **Historical `policy-v2` Reports**: [`policy-v2-nash-certification.md`](../../docs/reports/policy-v2-nash-certification.md), [`policy-v2-camouflage-study.md`](../../docs/reports/policy-v2-camouflage-study.md), [`cf-party-ab-v4-exploration.md`](../../docs/reports/cf-party-ab-v4-exploration.md), [`policy-v2-team-pool-refresh.md`](../../docs/reports/policy-v2-team-pool-refresh.md)

## Frozen Zero-Leakage Certification (`Stage 6d`, `2,400` Strict LIVE `policy-v4` Games)

Evaluated under brand-new cryptographic domain `nc2000-policy-v4-strong10-v2-20261128::stage6d-final-cert` (`4,800/4,800` `preview.source == "live"`, `0` belief mismatches, `0` caps/ties, exact `I1` disclosed shell mixtures):

- **`P_strong10` vs Uncamouflaged `Plain10` (`1,200` mirrored games, complete `10×10` matrix)**:
  - **Pure modified-shell causal block (`mod7 (Strong)` vs `mod7 (Plain)`, `588` games)**: **`53.40% ± 2.05%`** (**`+3.40 pp`, `z = +1.66`, one-sided `p = 0.0484`**).
  - **De-noised full-pool lift (`1,092` non-mirror games, fixing exact identical `base3 × base3` mirror to true `50.00%`)**: **`51.75% ± 1.33%`** (**`+1.75 pp`, `z = +1.32`, one-sided `p = 0.0937`**). *(Raw `1,200g` headline `52.50% ± 1.37%` includes `+0.75 pp` of random mirror variance from the `108g` identical `base3 × base3` block [`58.33%`].)*
- **`P_strong10` vs Certified 7-Shell Plain Nash `p_clean7` (`1,200` mirrored games, exact `STRAT60`)**: **`49.34% ± 1.45%`** (`z = -0.46`; `modified-7` = **`50.17%`**) — equal-weight 10-shell diversity matches the exact optimized 7-shell plain Nash within noise.
- **Legality & Cross-Ruleset Stealth**: `15/15` bot teams and `33/33` preserved environment teams pass BOTH `gen2stadium2` and `castle` WASM `Validator`, strict JP NC2000 cartridge blacklist, `training-v3` (`changedTeams: 0`), exact `6/6` Turn-0 `(species, level, gender, has_item)` signatures, exact `6/6` `HP_IV` Castle stealth, exact `6/6` intra-cluster identical items, and a genuine 16-game live `castle-tournament` (`policyEpoch: "castle-policy-v4"`, `previewCache: false`) `nash_refresh` smoke (`32/32` live sides, `0` belief errors, `0` caps/ties).

## Curated 10-Shell Equal-Weight Bot Pool (`15` Teams, `1/10` per Shell)

| Rank | Shell (`variantCluster`) | Mode | Members & Exact Weights (`drawWeight`) | `s6d` vs `p_clean7` | `s6d` vs `Plain10` | `Plain10` Opp EV | Key Design & CRN Evidence |
|---:|---|---|---|---:|---:|---:|---|
| 1 | **`pb-131`** | Upgraded Singleton | `pb131-c3r-fbhera-magzap-refcloy` (`0.100`) | `60.63%` | `57.50%` | `45.83%` | FocusBand EndRev Hera + Magnet Zap55 + Reflect Cloy (`+11.7 pp` over plain `pb-131`; camo proved null `+0.32 pp`) |
| 2 | **`pb-6`** | Clean Base Singleton | `pb-6` (`0.100`) | `50.69%` | `59.17%` | `55.83%` | Zap55/Machamp55/SD Lax/TrapPass Umb/Curse Cloy/Perish Gar; camo hurt `-2.60 pp` |
| 3 | **`sample-07`** | Upgraded Singleton | `s07-c8-mcoatrode-fbwak-delax-thunzap` (`0.100`) | `52.22%` | `53.33%` | `49.17%` | Mirror Coat Electrode + Fire Blast Wak55 + DE Lax + Thunder Zap55 (`58.3%` in `s2c+s4b`; camo hurt `-4.72 pp`) |
| 4 | **`apex-c04-bug-3atkstar-boomgen`** | **Camouflaged `50/50`** (`6/6` items) | `apex-c3r` (`0.050`) + `apex-c5` (`0.050`) | `51.23%` | `53.33%` | `47.50%` | **Lv55 RestTalk Snorlax fixed!** 4-Atk Boom Gar/Reflect Cloy/Bug Wak/TWave Star vs Perish Gar/Curse Cloy/FB Wak/3-Atk Star (`s6c` `D_CAMO = +8.75 ± 3.87 pp, z = +2.262, p = 0.012`, `frag = 3.33 pp`; `s6d` `c3r 51.7%` vs `c5 52.5%`) |
| 5 | **`ayagen-m-zap50`** | Clean Base Singleton | `ayagen-m-zap50` (`0.100`) | `48.93%` | `54.17%` | `42.50%` | Dual-Special-Ace Gengar55/Alakazam55/Eggy/Lax/Cune/Zap (`#8` overall vs `p_clean7`, `58.3%` vs `P_strong12`) |
| 6 | **`pb63-m-gar55`** | **Camouflaged `50/50`** (`6/6` items) | `pb63-c3` (`0.050`) + `pb63-c3b` (`0.050`) | `51.19%` | `50.83%` | `46.67%` | Rest/Curse Lax55 + WW Zap vs DE/Curse/SD Lax55 + DP Zap (both 4-Atk Gar55 [Shadow Ball/TBolt/IP/FP]; `+6.7` to `+7.5 pp` over base, `54.4%` in `s6`) |
| 7 | **`rraikou-m-cune50`** | **Camouflaged `80/20`** (`6/6` items & `HP_IV`) | `rraikou-m-cune50` (`0.080`) + `rraikou-c1` (`0.020`) | `42.24%` | `55.83%` | `50.00%` | Low-poison 20% Roar Raikou/Roar Cune/FB Lax ambush twin (`s3c+s3d` pooled `D_CAMO = +6.87 ± 3.18 pp, z = +2.16`) |
| 8 | **`pb85-m-cune50`** | **Camouflaged `50/35/15`** (`6/6` items) | `pb85-m-cune50` (`0.050`) + `pb85-c5` (`0.035`) + `pb85-c3r` (`0.015`) | `48.23%` | `49.17%` | `42.50%` | Weak `c1` eliminated, `A0` promoted to 50% anchor, SD Lax/Reflect Cloy `c3r` capped at 15% (`s2a+s2b` `D_CAMO = +6.63 ± 2.38 pp, z = +2.79`; `s4b` `+3.58 pp`; `s6d` `c3r = 50.0%` both benchmarks) |
| 9 | **`pb-87`** | Upgraded Singleton | `pb87-c5-fblax-dpzap-crunchttar` (`0.100`) | `45.45%` | `47.50%` | `51.67%` | Fire Blast Lax50 + DP Zap55 + Crunch TTar55 (`+6.49 pp` CRN in `s2b+s2c`, `65.0%` vs `Cut20`; camo hurt `-4.80 ± 2.29 pp, z = -2.09`) |
| 10 | **`sample-12`** | Clean Base Singleton | `sample-12` (`0.100`) | `42.58%` | `44.17%` | `43.33%` | Perish/DBond Gar/HPump+TWave Star/Reflect Cloy/Toxic Umb/Flamethrower Wak/RestTalk Lax55 (`57.3%` in `s5`, `50.0%` over `296g` in `s6`) |

## Reproduction

```sh
node tools/build-team-products-v4.js
```
