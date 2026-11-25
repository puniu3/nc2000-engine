# `team-pool-v4` (`policy-v2` Refresh & Certified 32-Shell Nash Equilibrium)

Generated deterministically by [`tools/build-team-products-v4.js`](../../tools/build-team-products-v4.js) from committed repo sources ([`data/team-pool-v3/team-pool.json`](../team-pool-v3/team-pool.json) + [`docs/reports/okimono-upgrades-v2.json`](../../docs/reports/okimono-upgrades-v2.json)) under production **`policy-v2`** (`CF-V_net v2` PUCT adaptive `243k` + `SteppedCfPreview 240k`, `c = 0.4`, strict `NASH_PREVIEW_CACHE=0` LIVE).

Full experimental evidence:
- **6-Round `7,776`-Game `policy-v2` Nash Equilibrium & 32-Shell Closure Certification**: [`docs/reports/policy-v2-nash-certification.md`](../../docs/reports/policy-v2-nash-certification.md)
- **Local Single-Slot Saturation & Decoy Audit (`9,112g`, `cf_party_ab`)**: [`docs/reports/cf-party-ab-v4-exploration.md`](../../docs/reports/cf-party-ab-v4-exploration.md)
- **Initial `v3 -> v4` Catalog Refresh & Step 4 True-CRN Audit (`10,152g`)**: [`docs/reports/policy-v2-team-pool-refresh.md`](../../docs/reports/policy-v2-team-pool-refresh.md)

## Summary

- **Catalog**: [`data/team-pool-v4/team-pool.json`](./team-pool.json) (`format: "nc2000-team-pool-v4"`, `policyEpoch: "policy-v2"`)
- **Certified Flagship Mixture**: [`data/team-pool-v4/flagship-mixture.json`](./flagship-mixture.json) (**6-shell certified `policy-v2` Bayes-Minimax Nash equilibrium**, `I1` Disclosed / Self-Consistent Shell regime, `7,776` strict NO-CACHE LIVE games / `15,552` live preview sides, `0/15` cross-round seed overlaps, uniform `96g/edge` zero-leakage Elite-8 matrix, `0/26` non-support detections [`max z = +0.25`, `min BH_q = 1.000`])
- **Exploitability Certificate (`50k` held-out posterior draws)**:
  - Uniform `96g/edge` Elite-8: **`E[true] = 2.68 pp`**, `p90 = 4.13 pp`, **`p95 = 4.66 pp`** (Jeffreys Beta-Binomial: **`2.64 / 4.07 / 4.57 pp`**; in-sample gap `0.46 pp`, `N_eff = 5.34`)
  - Full 32-Shell Empirical-Bayes Closure: **`E[true] = 2.75 pp`**, `p90 = 4.22 pp`, **`p95 = 4.78 pp`** (Beta: **`2.73 / 4.19 / 4.74 pp`**; conservative stress-test `p95 = 4.95–4.99 pp < 5.0 pp`)
- **Selected teams**: `33` across **`32` pairwise-distinct public Turn-0 shells** (`9` causally verified `policy-v2` upgrades replacing their `v3` parents + `23` preserved `v3` shells including 2-variant `vc-s0053-lefties`)
- **Distinct Level-55 Aces**: `22` (`Zapdos`, `Suicune`, `Marowak`, `Snorlax`, `Raikou`, `Porygon2`, `Tauros`, `Tyranitar`, `Miltank`, `Machamp`, `Gengar`, `Victreebel`, `Nidoking`, `Crobat`, `Heracross`, `Electabuzz`, `Alakazam`, `Kingdra`, `Dragonite`, `Entei`, `Poliwrath`, `Rhydon`)
- **Self-Referential `240k` Okimono Census (`step4b-v4-census-k4`, `K=4`, `4,356` previews)**:
  - **`0 / 33`** absolute `0.0%` okimonos (down from `2` in `v3`), **`2 / 33`** `< 5%` okimonos (down from `6`), mean minimum slot pick rate **`12.96% -> 15.39%`** (`+2.43 pp`)
- **Draw Weight Sum**: `1.000000` exact (`1/32 = 0.03125` per shell; `s0053-lefties` split `0.015625 + 0.015625`)
- **Legality & Training**: `33/33` WASM `Validator` PASS, strict Japanese NC2000 cartridge legality, `training-v3` normalized (`changedTeams: 0`).

## Certified 6-Shell `policy-v2` Flagship Equilibrium (`flagship-mixture.json`)

> [!NOTE]
> Under finite-sample payoff uncertainty (`SE ≈ 5.1 pp` at `96g/edge`), the unique 5-shell exact rational point-LP Nash (`40.25% pb66-m-wak55 / 20.34% mjj-2014-thdrcomb / 19.92% rraikou-m-cune50 / 17.37% sample-07 / 2.12% pb63-m-gar55`) over-concentrates on point estimates (`E[true] = 3.11 pp`, `p95 = 5.73 pp`). Deployed weights use the globally converged **Bayes-Minimax** equilibrium (`p_clean6`, `dual_gap <= 0.012 pp`), cutting posterior `p95` exploitability by **`-1.07 pp`** at only `0.46 pp` in-sample gap.

| Rank | Shell | Certified Weight (`p_clean6`) | Exact Point-LP | Boot Mean (`Supp%`) | Elite-8 RR (`96g/edge`) | EV vs `p_clean6` | Archetype / Role |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **`pb66-m-wak55`** | **`24.00%`** | `40.25%` | `19.49%` (`72.4%`) | `49.26%` | `50.46%` | All-around Lax55/Wak55 anchor; beats `mjj` (`52.1%`), `rraikou` (`52.1%`), `pb63` (`53.1%`) |
| 2 | **`mjj-2014-thdrcomb`** | **`21.50%`** | `20.34%` | `21.50%` (`86.5%`) | **`54.02%`** | `50.32%` | #1 RR & #1 boot support; Zapdos55/TTar55/Steelix/Venusaur; crushes `sample-07` (`60.4%`) & `pb122` (`70.8%`) |
| 3 | **`rraikou-m-cune50`** | **`21.00%`** | `19.92%` | `20.40%` (`85.4%`) | `52.68%` | `50.20%` | Raikou55/Wak55/Suicune/Fighters; crushes `pb-6` (`62.5%`), beats `mjj` (`55.2%`) & `pb84` (`59.4%`) |
| 4 | **`pb63-m-gar55`** | **`13.00%`** | `2.12%` | `13.45%` (`80.0%`) | `50.74%` | `49.46%` | Lax55/Gengar55/Blissey/Heracross; crushes `rraikou` (`60.4%`) & `pb84` (`65.6%`) |
| 5 | **`sample-07`** | **`13.00%`** | `17.37%` | `12.70%` (`69.3%`) | `51.04%` | `49.46%` | Triple-L55 Boom HO (`BG-Tauros55/Wak55/Zapdos55`); undefeated vs all Elite-8 except `mjj`; beats `pb66` (`55.2%`) & `pb-6` (`54.2%`) |
| 6 | **`pb-6`** | **`7.50%`** | `0.00%` | `7.59%` (`61.9%`) | `48.66%` | `48.92%` | Zapdos55/Machamp55/Umbreon/Perish-Gar; polices `pb63-m-gar55` (`65.6%`) & edges `pb66` (`52.1%`) |

*(Catalog `score` values record each shell's exact `policy-v2` expected win rate against `p_clean6`.)*

## All 9 Causally Verified `policy-v2` Upgrades (All `9/9` Positive `InΔ`)

1. `mjj-2020-rraikou` → **`rraikou-m-cune50`** (`s2 Cloyster@BJ -> Suicune50@Bright Powder`, `+5.2 pp`, `InΔ +10.4 pp`, `OutΔ 0.0 pp`, `v4 minPk 32.4%`) — **Certified Tier-S (`21.0%`)**
2. `pb-66` → **`pb66-m-wak55`** (`s6 Gengar50 -> Marowak55@Thick Club`, `+2.1 pp`, `InΔ +7.1 pp`, H2H `66.7%`, `v4 minPk 14.1%`) — **Certified Tier-S (`24.0%`, #1)**
3. `pb-63` → **`pb63-m-gar55`** (`s2 Venusaur55 -> Gengar55@Gold Berry [TBolt/Ice Punch/Explosion/DBond]`, **`59.4%`** in `128g` true-CRN, `InΔ +17.9 pp`, `OutΔ 0.0 pp`, `v4 minPk 10.9%`) — **Certified Tier-S (`13.0%`)**
4. `pb-84` → **`pb84-cln-skarm50`** (clean `Cloyster@Gold Berry Surf` + `Gligar@Bright Powder` + `s6 Umbreon50 -> Skarmory50@Leftovers`, **`53.9%`** in `128g` true-CRN, `InΔ +25.7 pp`, `v4` Elite-8 `47.4%` vs `p_clean6`)
5. `pb-122` → **`pb122-m-lax50`** (`s3 Electrode50 -> Snorlax50@Quick Claw`, **`+23.4 pp`**, `z = +3.30`, `InΔ +20.8 pp`; displaced from Tier-S by `mjj-2014-thdrcomb` [`29.2%`])
6. `mjj-2015-ayagen` → **`ayagen-m-zap50`** (`s5 Umbreon50 [0.0%] -> Zapdos50@Mint Berry`, **`+10.9 pp`**, `InΔ +14.3 pp`)
7. `mjj-2014-offantom` → **`offantom-m-wak50`** (`s2 Charizard55 [0.0%] -> Marowak50@Thick Club`, **`+14.1 pp`**, `InΔ +3.1 pp`, H2H `83.3%`)
8. `sample-02` → **`s02-m-lax-curse`** (same shell: `s5 Snorlax50` `[Rest/Flame] -> [Curse/Self-Destruct]`, **`+14.1 pp`**, `InΔ +23.3 pp`)
9. `pb-85` → **`pb85-m-cune50`** (`s6 Umbreon50 [3.9%] -> Suicune50@Bright Powder`, **`+6.25 pp`**, `InΔ +12.9 pp`, `OutΔ 0.0 pp`)

## Reproduction

```sh
node tools/build-team-products-v4.js
```
