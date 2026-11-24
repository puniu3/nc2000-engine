# `team-pool-v4` (`policy-v2` Refresh — True-CRN Audited)

Generated deterministically by [`tools/build-team-products-v4.js`](../../tools/build-team-products-v4.js) from committed repo sources ([`data/team-pool-v3/team-pool.json`](../team-pool-v3/team-pool.json) + [`docs/reports/okimono-upgrades-v2.json`](../../docs/reports/okimono-upgrades-v2.json)) under production **`policy-v2`** (`CF-V_net v2` PUCT adaptive `243k` + `SteppedCfPreview 240k`).

Full experimental evidence — including the `policy-v1 -> policy-v2` meta-shift audit, minor-Pokémon gem detection threshold recalibration, and the **1,840-game Step 4 true shared-seed CRN forensic audit** that retracted the earlier `Slowking L50` and `Espeon L55` minor-gem promotions — is documented in [`docs/reports/policy-v2-team-pool-refresh.md`](../../docs/reports/policy-v2-team-pool-refresh.md).

## Summary

- **Catalog**: [`data/team-pool-v4/team-pool.json`](./team-pool.json) (`format: "nc2000-team-pool-v4"`, `policyEpoch: "policy-v2"`)
- **Flagship Mixture**: [`data/team-pool-v4/flagship-mixture.json`](./flagship-mixture.json) (**provisional** uniform 5-shell Tier-S core, `0.200000` each; **not** a certified `policy-v2` Nash equilibrium — a full `>=48g/edge` `policy-v2` Nash round-robin including `pb63-m-gar55` and `pb84-cln-skarm50` is pending)
- **Selected teams**: `33` across **`32` pairwise-distinct public Turn-0 shells** (`9` causally verified `policy-v2` upgrades — all `9/9` with strictly positive in-battle CRN `InΔ` — replacing their `team-pool-v3` parents + `23` preserved clean/unbeaten `team-pool-v3` shells including 2-variant `vc-s0053-lefties`)
- **Retracted Minor-Gem Candidates (Excluded after Step 4 True-CRN Audit)**:
  - **`pb84-cln-sking50`** (`Slowking L50`): Retracted as a genetic hitchhiker / fake-CRN + missing-control artifact (`39.8%`, `9th/9` in `128g` true shared-seed CRN; `-14.06 pp` [`z = -2.46`] vs `pb84-cln-skarm50`, `-9.38 pp` vs `Suicune50`, `-8.59 pp` vs `Starmie50`; pooled `256g` vs clean `pb84-m-cloysurf` `+0.39 pp` with picked `InΔ = -1.75 pp`; Champs10 picked `36.8%` vs unpicked `52.8%`). Replaced by **`pb84-cln-skarm50`**.
  - **`pb63-m-esp55`** (`Espeon L55`): Retracted after failing true-CRN replication (`50.8%`, `-4.69 pp` vs parent `pb-63`, `-8.59 pp` [`z = -1.73`] vs `pb63-m-gar55`; pooled `192g` `-1.04 pp`, picked `InΔ = -5.71 pp`). Replaced by **`pb63-m-gar55`**.
- **Distinct Level-55 Aces**: `22` (`Zapdos`, `Suicune`, `Marowak`, `Snorlax`, `Raikou`, `Porygon2`, `Tauros`, `Tyranitar`, `Miltank`, `Machamp`, `Gengar`, `Victreebel`, `Nidoking`, `Crobat`, `Heracross`, `Electabuzz`, `Alakazam`, `Kingdra`, `Dragonite`, `Entei`, `Poliwrath`, `Rhydon`)
- **Self-Referential `240k` Okimono Census (`step4b-v4-census-k4`, `K=4`, `4,356` previews)**:
  - **`0 / 33`** absolute `0.0%` okimonos (down from `2` in `v3` under `policy-v2`)
  - **`2 / 33`** `< 5%` okimonos (cut from `6` in `v3`: only untouched `pb-87` `Miltank50 3.9%` and `hc75-top8-tsuru` `Golem50 4.7%`)
  - Mean minimum slot pick rate improved **`12.96% -> 15.39%`** (`+2.43 pp`)
- **Draw Weight Sum**: `1.000000` exact (`1/32 = 0.03125` per shell; `s0053-lefties` split `0.015625 + 0.015625`)
- **Legality & Training**: `33/33` WASM `Validator` PASS, strict Japanese NC2000 cartridge legality, `training-v3` normalized (`changedTeams: 0`).

## Provisional Tier-S 5-Shell Core (`flagship-mixture.json`)

> [!NOTE]
> These weights are a **provisional uniform `0.200000` split**, not a certified `policy-v2` Nash equilibrium. The earlier `12g/edge` 10×10 RR omitted `pb63-m-gar55` and `pb84-cln-skarm50` and had `90%` seat-pair bootstrap weight CIs starting at `0.0%` for every shell.

| Rank | Shell | Provisional Weight | Primary `policy-v2` Evidence | Key Feature |
|---:|---|---:|---|---|
| 1 | **`rraikou-m-cune50`** | `20.00%` | `60.2%` Champs10 RR (`108g`), `InΔ +10.4 pp`, `v4 minPk 32.4%` (#1!) | `s2 Cloyster@BJ -> Suicune50@Bright Powder` |
| 2 | **`pb122-m-lax50`** | `20.00%` | `62.5%` field (`+23.4 pp`, `z=+3.30`), `57.4%` Champs10 RR, `InΔ +20.8 pp` | `s3 Electrode50 -> Snorlax50@Quick Claw` |
| 3 | **`pb63-m-gar55`** | `20.00%` | **`59.4%`** (`28/32 >= 50%`) `128g` true-CRN (`InΔ +17.9 pp`, `OutΔ 0.0 pp`), `192g` pooled `+3.65 pp`, H2H `57.1%` | `s2 Venusaur55 -> Gengar55@Gold Berry` (`v4 minPk 10.9%`) |
| 4 | **`pb66-m-wak55`** | `20.00%` | `61.5%` field (`96g`), H2H `66.7%`, `54.6%` Champs10 RR, `InΔ +7.1 pp` | `s6 Gengar50 -> Marowak55@Thick Club` |
| 5 | **`sample-08`** | `20.00%` | `61.3%` 32×32 RR (`#1` 32×32 Nash), beats `rraikou-m-cune50` `66.7%` | Preserved human powerhouse |

## All 9 Causally Verified `policy-v2` Upgrades (All `9/9` Positive `InΔ`)

1. `mjj-2020-rraikou` → **`rraikou-m-cune50`** (`s2 Cloyster@BJ -> Suicune50@Bright Powder`, `+5.2 pp`, `InΔ +10.4 pp`, `OutΔ 0.0 pp`, `v4 minPk 32.4%`)
2. `pb-122` → **`pb122-m-lax50`** (`s3 Electrode50 -> Snorlax50@Quick Claw`, **`+23.4 pp`**, `z = +3.30`, `InΔ +20.8 pp`, `v4 minPk 21.9%`)
3. `pb-63` → **`pb63-m-gar55`** (`s2 Venusaur55 -> Gengar55@Gold Berry [TBolt/Ice Punch/Explosion/DBond]`, **`59.4%`** in `128g` true-CRN, `InΔ +17.9 pp`, `OutΔ 0.0 pp`, `192g` pooled `+3.65 pp`, H2H `57.1%`, `v4 minPk 10.9%`)
4. `pb-84` → **`pb84-cln-skarm50`** (clean `Cloyster@Gold Berry Surf` + `Gligar@Bright Powder` + `s6 Umbreon50 -> Skarmory50@Leftovers [Drill Peck/Curse/Whirlwind/Rest]`, **`53.9%`** in `128g` true-CRN, `+7.03 pp` vs clean base / `+14.06 pp` [`z = +2.46`] vs Slowking, **`InΔ +25.7 pp`**, `OutΔ 0.0 pp`, H2H `62.5%`, `s6 28.1%` in `v4` census)
5. `pb-66` → **`pb66-m-wak55`** (`s6 Gengar50 -> Marowak55@Thick Club`, `+2.1 pp`, `InΔ +7.1 pp`, H2H `66.7%`, `v4 minPk 14.1%`)
6. `mjj-2015-ayagen` → **`ayagen-m-zap50`** (`s5 Umbreon50 [0.0%] -> Zapdos50@Mint Berry`, **`+10.9 pp`**, `InΔ +14.3 pp`, `65.6%` all-32 WR, `v4 minPk 28.1%`)
7. `mjj-2014-offantom` → **`offantom-m-wak50`** (`s2 Charizard55 [0.0%] -> Marowak50@Thick Club`, **`+14.1 pp`**, `InΔ +3.1 pp`, H2H **`83.3%`**, `v4 minPk 15.2%`)
8. `sample-02` → **`s02-m-lax-curse`** (same shell: `s5 Snorlax50` `[Rest/Flame] -> [Curse/Self-Destruct]`, **`+14.1 pp`**, **`InΔ +23.3 pp`**, H2H `58.3%`, `v4 minPk 14.8%`)
9. `pb-85` → **`pb85-m-cune50`** (`s6 Umbreon50 [3.9%] -> Suicune50@Bright Powder`, **`+6.25 pp`**, `InΔ +12.9 pp`, `OutΔ 0.0 pp`, `v4 minPk 28.1%`)

## Reproduction

```sh
node tools/build-team-products-v4.js
```
