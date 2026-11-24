# `team-pool-v4` (`policy-v2` Refresh)

Generated deterministically by [`tools/build-team-products-v4.js`](../../tools/build-team-products-v4.js) from committed repo sources ([`data/team-pool-v3/team-pool.json`](../team-pool-v3/team-pool.json) + [`docs/reports/okimono-upgrades-v2.json`](../../docs/reports/okimono-upgrades-v2.json)) under production **`policy-v2`** (`CF-V_net v2` PUCT adaptive `243k` + `SteppedCfPreview 240k`).

Full experimental evidence, `policy-v1 -> policy-v2` meta-shift analysis, and calibrated minor-Pokémon gem detection thresholds are documented in [`docs/reports/policy-v2-team-pool-refresh.md`](../../docs/reports/policy-v2-team-pool-refresh.md).

## Summary

- **Catalog**: [`data/team-pool-v4/team-pool.json`](./team-pool.json) (`format: "nc2000-team-pool-v4"`, `policyEpoch: "policy-v2"`)
- **Flagship Mixture**: [`data/team-pool-v4/flagship-mixture.json`](./flagship-mixture.json) (6-shell `policy-v2` RM+ Nash core derived from 540g strict NO-CACHE LIVE 10×10 Super-Champions RR, `max_BR = 50.00%`)
- **Selected teams**: `33` across **`32` pairwise-distinct public Turn-0 shells** (`9` certified `policy-v2` upgrades replacing their `team-pool-v3` parents + `23` preserved clean/unbeaten `team-pool-v3` shells including 2-variant `vc-s0053-lefties`)
- **Certified Minor-Pokémon Gems Included (`2` shells, both in Tier-S Nash Core!)**:
  - **`pb84-cln-sking50`** (`Slowking L50 @ Leftovers [Surf / Fire Blast / Rest / Sleep Talk]`): `55.0%` over `160g` (`+9.4 pp` vs parent), **`18.36%`** Nash weight (#3), `23.8%` self-referential `v4` census pick rate.
  - **`pb63-m-esp55`** (`Espeon L55 @ PRZ Cure Berry [Psychic / Zap Cannon / Charm / Morning Sun]`): `+6.2 pp` on BOTH Panel A and disjoint Panel B (`56.2%` pooled), **`54.6%`** in 10×10 Champions RR (beat `#1 rraikou-m-cune50` `58.3%`, `#2 pb122-m-lax50` `58.3%`, `ayagen-m-zap50` `83.3%`), `22.7%` self-referential `v4` census pick rate.
- **Distinct Level-55 Aces**: `23` (adds **`Espeon`**; retains `Zapdos`, `Suicune`, `Marowak`, `Snorlax`, `Raikou`, `Porygon2`, `Tauros`, `Tyranitar`, `Miltank`, `Machamp`, `Gengar`, `Victreebel`, `Nidoking`, `Crobat`, `Heracross`, `Electabuzz`, `Alakazam`, `Kingdra`, `Dragonite`, `Entei`, `Poliwrath`, `Rhydon`)
- **Self-Referential `240k` Okimono Census (`step3b-v4-census-k4`, `K=4`, `4,356` previews)**:
  - **`0 / 33`** absolute `0.0%` okimonos (down from `2` in `v3` under `policy-v2`)
  - **`3 / 33`** `< 5%` okimonos (cut in half from `6`)
  - Mean minimum slot pick rate improved **`12.96% -> 15.08%`** (`+2.12 pp`)
- **Draw Weight Sum**: `1.000000` exact (`1/32 = 0.03125` per shell; `s0053-lefties` split `0.015625 + 0.015625`)
- **Legality & Training**: `33/33` WASM `Validator` PASS, strict Japanese NC2000 cartridge legality, `training-v3` normalized (`changedTeams: 0`).

## Tier-S `policy-v2` 6-Shell RM+ Nash Core (`flagship-mixture.json`)

| Rank | Shell | Nash Weight | 10×10 RR Mean (`108g`) | Upgrade Category / Key Feature |
|---:|---|---:|---:|---|
| 1 | **`rraikou-m-cune50`** | **`31.09%`** | **`60.2% ± 5.3`** | Major upgrade (`Cloyster -> Suicune50@Bright Powder`), `minPk 32.4%` (#1 pool balance) |
| 2 | **`sample-08`** | **`22.77%`** | `45.4% ± 6.2` | Preserved human powerhouse (`61.3%` in 32×32 RR; beats `rraikou-m-cune50` `66.7%`) |
| 3 | **`pb84-cln-sking50`** | **`18.36%`** | `50.0% ± 4.8` | **Certified Minor Gem (`Slowking50@Leftovers`)** (`55.0%` over 160g; `75%` vs `pb66-m-wak55`) |
| 4 | **`pb66-m-wak55`** | **`17.85%`** | `54.6% ± 7.9` | Major upgrade (`Gengar50 -> Marowak55@Thick Club`), H2H `66.7%` vs parent |
| 5 | **`pb122-m-lax50`** | **`9.13%`** | **`57.4% ± 6.9`** | Major upgrade (`Electrode50 -> Snorlax50@Quick Claw`), **`+23.4 pp` (`z=+3.28`)** vs parent |
| 6 | **`pb63-m-esp55`** | **`0.80%`** | **`54.6% ± 5.0`** | **Certified Minor Gem (`Espeon55@PRZ Cure Berry`)** (`+6.2 pp` PanA & PanB; `58.3%` vs `#1`) |

## All 9 Certified `policy-v2` Upgrades (vs `team-pool-v3`)

1. `mjj-2020-rraikou` → **`rraikou-m-cune50`** (`s2 Cloyster@BJ -> Suicune50@Bright Powder`, `+5.2 pp`, `#1` 10-Champ RR `60.2%`, `31.1%` Nash)
2. `pb-122` → **`pb122-m-lax50`** (`s3 Electrode50 -> Snorlax50@Quick Claw`, **`+23.4 pp`**, `z = +3.28`, `#2` 10-Champ RR `57.4%`)
3. `pb-84` → **`pb84-cln-sking50`** (**Minor Gem**: clean Cloyster/Gligar + `s6 Umbreon50 -> Slowking50@Leftovers`, **`+9.4 pp`** over `160g`, `18.4%` Nash)
4. `pb-63` → **`pb63-m-esp55`** (**Minor Gem**: `s2 Venusaur55 -> Espeon55@PRZ Cure Berry`, **`+6.2 pp`** on BOTH Panel A and disjoint Panel B, `54.6%` 10-Champ RR)
5. `pb-66` → **`pb66-m-wak55`** (`s6 Gengar50 -> Marowak55@Thick Club`, `+2.1 pp`, H2H `66.7%`, `54.6%` 10-Champ RR, `17.9%` Nash)
6. `mjj-2015-ayagen` → **`ayagen-m-zap50`** (`s5 Umbreon50 [0.0%] -> Zapdos50@Mint Berry`, **`+10.9 pp`**, `65.6%` all-32 WR, `minPk 0.0% -> 25.8%`)
7. `mjj-2014-offantom` → **`offantom-m-wak50`** (`s2 Charizard55 [0.0%] -> Marowak50@Thick Club`, **`+14.1 pp`**, **H2H `83.3%`**, `minPk 0.0% -> 14.4%`)
8. `sample-02` → **`s02-m-lax-curse`** (same shell: `s5 Snorlax50` `[Rest/Flame] -> [Curse/Self-Destruct]`, **`+14.1 pp`**, H2H `58.3%`, `minPk 14.1%`)
9. `pb-85` → **`pb85-m-cune50`** (`s6 Umbreon50 [3.9%] -> Suicune50@Bright Powder`, **`+6.3 pp`**, `minPk 3.9% -> 28.1%`)

## Reproduction

```sh
node tools/build-team-products-v4.js
```
