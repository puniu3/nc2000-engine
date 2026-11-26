# `team-pool-v4` (Preserved 32-Shell Diverse Catalog & Certified `policy-v4` 7-Shell Nash Equilibrium)

Generated deterministically by [`tools/build-team-products-v4.js`](../../tools/build-team-products-v4.js) from committed repo sources ([`data/team-pool-v3/team-pool.json`](../team-pool-v3/team-pool.json) + [`docs/reports/okimono-upgrades-v2.json`](../../docs/reports/okimono-upgrades-v2.json)). All **32 pairwise-distinct shells (`33` teams)** are preserved unchanged as a diverse playable catalog, while the flagship Nash equilibrium mixture is certified under production **`policy-v4`** (`CF-V_net v3` 192-dim PUCT adaptive `243k` + `SteppedCfPreview 720k` [`65%` `128i/128t1` `Present19` `TrajCalib`], `c = 0.4`, strict `NASH_PREVIEW_CACHE=0` LIVE).

Full experimental evidence:
- **9-Round `12,844`-Game `policy-v4` Nash Equilibrium & Full 32-Shell Support Certification**: [`docs/reports/policy-v4-nash-certification.md`](../../docs/reports/policy-v4-nash-certification.md)
- **Historical `policy-v2` 6-Shell Certification (`7,776g`)**: [`docs/reports/policy-v2-nash-certification.md`](../../docs/reports/policy-v2-nash-certification.md)
- **Historical `policy-v2` Same-Frame Camouflage Study (`8,016g`, 4-Arm CRN)**: [`docs/reports/policy-v2-camouflage-study.md`](../../docs/reports/policy-v2-camouflage-study.md)
- **Local Single-Slot Saturation & Decoy Audit (`9,112g`, `cf_party_ab`)**: [`docs/reports/cf-party-ab-v4-exploration.md`](../../docs/reports/cf-party-ab-v4-exploration.md)
- **Initial `v3 -> v4` Catalog Refresh & Step 4 True-CRN Audit (`10,152g`)**: [`docs/reports/policy-v2-team-pool-refresh.md`](../../docs/reports/policy-v2-team-pool-refresh.md)

## Summary

- **Catalog**: [`data/team-pool-v4/team-pool.json`](./team-pool.json) (`format: "nc2000-team-pool-v4"`, `policyEpoch: "policy-v4"`, all 32 shells / 33 teams preserved byte-identically)
- **Certified Flagship Mixture**: [`data/team-pool-v4/flagship-mixture.json`](./flagship-mixture.json) (**7-shell certified `policy-v4` Bayes-Minimax Nash equilibrium**, `I1` Disclosed / Self-Consistent Shell regime, **`12,844` strict NO-CACHE LIVE games / `25,688` live preview sides**, `3,083/3,083` verified `policy-v4` manifests, `0` belief mismatches, `0` caps/ties, `0` cross-round seed overlaps, cross-round homogeneity $\chi^2 = 137.49$ on `143 df`, **`100.0%` direct support coverage across all `32/32` shells**, `0/25` non-support detections [`25/25 < 50.0%`, `max EV = 49.39%`, `max z = -0.15`, `min BH_q = 0.9978`])
- **Support Rigor**: All `7` deployed shells match the **unique exact rational 7-shell point-LP equilibrium 1-to-1** (identical on both zero-leakage and pooled matrices), with `10,000`-draw True Bayesian Leave-One-Out inclusion probabilities **`59.8%–95.1%`** and positive `E[LOO]` (`+0.76 pp` to `+5.14 pp`) for all `7/7`, separated by a **`41.8 pp` cliff** from #8 (`pb66-m-wak55`, `18.0%`, `E[LOO] = -2.27 pp`, `p_LP = 0.00%`).
- **Exploitability Certificate (`50k` held-out posterior draws)**:
  - Active-8 Zero-Leakage (`96g–408g/edge`): Jeffreys Beta **`E[true] = 1.52 pp`**, `p90 = 2.33 pp`, **`p95 = 2.64 pp`**, `CVaR95 = 3.02 pp` (Mirrored-Pair Bootstrap: **`1.56 / 2.38 / 2.68 / 3.10 pp`**; in-sample gap `0.26 pp`, `N_eff = 5.67`)
  - Full 32-Shell Empirical-Bayes Closure (`100%` direct coverage, zero uncovered imputation): **`E[true] = 1.52 pp`**, `p90 = 2.33 pp`, **`p95 = 2.64 pp`**
- **Selected teams**: `33` across **`32` pairwise-distinct public Turn-0 shells** (`9` causally verified upgrades + `23` preserved `v3` shells including 2-variant `vc-s0053-lefties`)
- **Distinct Level-55 Aces**: `22` (`Zapdos`, `Suicune`, `Marowak`, `Snorlax`, `Raikou`, `Porygon2`, `Tauros`, `Tyranitar`, `Miltank`, `Machamp`, `Gengar`, `Victreebel`, `Nidoking`, `Crobat`, `Heracross`, `Electabuzz`, `Alakazam`, `Kingdra`, `Dragonite`, `Entei`, `Poliwrath`, `Rhydon`)
- **Draw Weight Sum**: `1.000000` exact (`1/32 = 0.03125` per shell; `s0053-lefties` split `0.015625 + 0.015625`)
- **Legality & Training**: `33/33` WASM `Validator` PASS, strict Japanese NC2000 cartridge legality, `training-v3` normalized (`changedTeams: 0`).

## Certified 7-Shell `policy-v4` Flagship Equilibrium (`flagship-mixture.json`)

> [!NOTE]
> Under `policy-v4`, the unique exact rational point-LP Nash equilibrium has exact odd support size **7**, matching the deployed Bayes-Minimax support (`p_clean7`) **1-to-1**. Using `p_clean7` (`0.5%`-rounded Bayes-Minimax optimum, `dual_gap = 0.034 pp`) cuts held-out `p95` exploitability from `3.55 pp` (point-LP) down to **`2.64 pp`** at only `0.26 pp` in-sample gap.

| Rank | Shell | Certified Weight (`p_clean7`) | Exact Point-LP | Bayesian `P(in)` | `E[LOO]` | Active-7 RR | EV vs `p_clean7` | Archetype / Role |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **`pb63-m-gar55`** | **`22.50%`** | `19.36%` | **`95.1%`** | **`+5.14 pp`** | `51.62%` | `50.19%` | Lax55/Gengar55/Blissey/Endure-Rev Hera; crushes `pb-131` (`58.0%`) & `rental-cban-13` (`66.7%`) |
| 2 | **`pb-131`** | **`22.00%`** | `15.90%` | **`94.3%`** | **`+4.36 pp`** | **`53.32%`** | `50.26%` | #1 RR; Zapdos55/TTar55/Spikes+Dual-Phaze Steelix/Hera; crushes `apex` (`62.0%`), `cban-13` (`60.4%`), `s07` (`58.0%`) |
| 3 | **`pb85-m-cune50`** | **`17.00%`** | `24.26%` | **`81.8%`** | **`+2.29 pp`** | `44.99%` | `50.23%` | Lax55/Steelix/Machamp/Cloy/Zap/Suicune50; anti-TTar/Spikes wall beating `pb-131` (`56.8%`), `pb-87` (`56.7%`), `pb63` (`52.5%`) |
| 4 | **`sample-07`** | **`17.00%`** | `24.13%` | **`71.1%`** | **`+1.25 pp`** | `51.66%` | `49.95%` | Triple-L55 Boom HO (`BG-Tauros55/Wak55/Zapdos55`); crushes `apex` (`61.7%`), `cban-13` (`59.4%`), beats `pb63` (`52.8%`) & `pb85` (`52.4%`) |
| 5 | **`apex-c04-bug-3atkstar-boomgen`** | **`10.50%`** | `4.95%` | **`71.3%`** | **`+2.03 pp`** | `48.63%` | `49.47%` | RestTalk Lax55/Cloy/4-Atk Gar/Hera/Wak55/Starmie; polices `pb85` (`63.3%`) & `pb63` (`59.4%`) |
| 6 | **`pb-87`** | **`9.00%`** | `8.48%` | **`60.5%`** | **`+0.76 pp`** | `50.81%` | `49.71%` | Zapdos55/TTar55/BP-Miltank/Cloy/Lax/Wak; polices `sample-07` (`58.2%`), beats `cban-13` (`55.2%`) & `apex` (`52.1%`) |
| 7 | **`rental-cban-13`** | **`2.00%`** | `2.92%` | **`59.8%`** | **`+1.40 pp`** | `48.96%` | `47.58%` | P2-55/Drum-Poliwrath55/Gar55/Eggy/Lax/Roar-Golem; nuclear `pb85` policeman (`80.2%`!) locking `pb63` into the equilibrium |

*(Catalog `score` values record each shell's exact `policy-v4` expected win rate against `p_clean7` across `100%` of the certified support.)*

## All 9 Causally Verified Catalog Upgrades (Preserved in `team-pool-v4`)

1. `pb-63` → **`pb63-m-gar55`** (`s2 Venusaur55 -> Gengar55@Gold Berry`) — **Certified `policy-v4` Tier-S (`22.5%`, #1)**
2. `pb-85` → **`pb85-m-cune50`** (`s6 Umbreon50 -> Suicune50@Bright Powder`) — **Certified `policy-v4` Tier-S (`17.0%`, #3)**
3. `mjj-2015-ayagen` → **`ayagen-m-zap50`** (`s5 Umbreon50 -> Zapdos50@Mint Berry`)
4. `mjj-2020-rraikou` → **`rraikou-m-cune50`** (`s2 Cloyster@BJ -> Suicune50@Bright Powder`)
5. `mjj-2014-offantom` → **`offantom-m-wak50`** (`s2 Charizard55 -> Marowak50@Thick Club`)
6. `pb-66` → **`pb66-m-wak55`** (`s6 Gengar50 -> Marowak55@Thick Club`)
7. `pb-84` → **`pb84-cln-skarm50`** (clean `Cloyster@Gold Berry Surf` + `Gligar@Bright Powder` + `s6 Skarmory50@Leftovers`)
8. `pb-122` → **`pb122-m-lax50`** (`s3 Electrode50 -> Snorlax50@Quick Claw`)
9. `sample-02` → **`s02-m-lax-curse`** (same shell: `s5 Snorlax50` `[Rest/Flame] -> [Curse/Self-Destruct]`)

## Reproduction

```sh
node tools/build-team-products-v4.js
```
