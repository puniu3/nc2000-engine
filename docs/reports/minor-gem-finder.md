# Minor-Pokémon Gem-Finder Verification Report (2026-11-22)

## Objective & Scoped Method

Evaluated whether a narrowed **Big-16 Anchor 1v1 + Adversarial Switch-In** table can cheaply surface unexplored minor Pokémon in NC2000 (6-pick-3, level sum ≤ 155, Item Clause, strict JP cart legality) without full all-vs-all combinatorial explosion:

1. **Stage 1 (Fast Root Screen)**: `gem_1v1 --mode root` (`400` iters lead, `140` iters × `5` samples per Turn-1 anchor switch-in move) across **44 Major species (317 sets)** + **99 non-dominated Minor species (931 sets)** × **16 canonical meta anchors** (`19,968` cells, ~25 s on 10 local threads). All sets canonicalized and validated via WASM `Validator` + `training-v3` normalizer; NYPC USA event moves, known incompatible Gen 2 egg combos, and pure evasion/Swagger cheese excluded from auto-generated candidates.
2. **Stage 2 (Full Played-Out 1v1 + Adversarial Switch-In)**: `gem_1v1 --mode play` (`16` full games/cell, `RmAgent` `100` iters lead / `80` iters switch-in, `max_turns = 120`) on **359 survivors** (`194` Majors + `165` Pareto-promising Minors across 46 species; `5,744` cells, `463.6 s`, `12.4 cells/s`).
3. **Stage 3 (Team Preview Activation Gate)**: Spliced top minor discoveries into flagship/pillar shells (`pb-84` Slot 6 `Umbreon50@Leftovers` and `s0053-lefties-hpi` Slot 1 `Electrode50` / Slot 2 `Suicune55`), evaluated `243k` Team Preview (`K = 2`) against an 8-shell panel from `data/team-pool-v3/team-pool.json` (`169.9 s`).
4. **Stage 4 (CRN 6v6 Sanity Check)**: Paired CRN `9k/81k` full battles (`32` games/arm, `352` games total, `236.5 s`) against the same 8-shell panel to separate true candidates from Team Preview rollout traps.

- **Tools**: [`crates/bot/examples/gem_1v1.rs`](../../crates/bot/examples/gem_1v1.rs), [`tools/gem-finder.py`](../../tools/gem-finder.py) (commit `c4d79571`).
- **Local raw outputs (gitignored)**: `target/gem-finder/{input.json,stage1-root.json,stage2-play.json,gem-report.json}`, `target/gem-finder/preview-gate/gate-report.json`, `target/gem-finder/ab-check-9k/out/`.

---

## Key Quantitative Findings

### 1. Stage 1 vs Stage 2 Calibration

- Over `5,744` overlapping cells: **Lead `r = 0.9192`**, **Switch-in (no-sleep) `r = 0.9208`**.
- Sanity check against all `317` Stage-1 Majors confirmed **5/5** best Majors for the top minor breakouts had been retained in Stage 2 (zero missing-major artifacts).

### 2. Top Stage-2 Pareto Breakouts (Same-Tier Switch-In vs All 44 Majors)

Big-16 anchors: `Lax55_Talk`, `Lax55_Left`, `Zap55_HPI`, `Zap55_DP`, `Raik55_Mag`, `Wak55_HPB`, `Ttar55_Mix`, `Cune55_Talk`, `P2_55_DT`, `P2_55_Ice`, `Lax50_Curse`, `Champ50_Talk`, `Venu50_Seed`, `Eggy50_Seed`, `Skarm50_Curse`, `Star50_Rec`.

| Tier | Minor Candidate | Notable Switch-In Breakout vs Best Same-Tier Major |
| :--- | :--- | :--- |
| `L50_NO_LEFT` | **Piloswine** `L50 @ Scope Lens` (`Earthquake / Ice Beam / Rest / Sleep Talk`) | Beats all 3 L55 Electric aces (`Zap55_HPI 0.75 / Zap55_DP 0.75 / Raik55_Mag 0.88`, lead `0.94 / 0.81 / 1.00`) vs Quagsire `0.31` (**`+44 pp`**). Note: `Ice Beam` (100% acc, 15 PP) strongly outperformed auto-picked `Blizzard` (70% acc, 5 PP). |
| `L50_NO_LEFT` | **Nidoqueen** `L50 @ Mint Berry` (`Earthquake / Curse / Rest / Sleep Talk`) | Beats `pb-84` duo `Raik55_Mag (0.81) + Champ50_Talk (0.88)` vs Heracross `0.19` (**`+62 pp`**), plus `Zap55_DP (0.62)` trio vs Quagsire `0.19` (**`+44 pp`**). |
| `L50_LEFT` / `L50_NO_LEFT` | **Shuckle** `L50 @ Leftovers` (`Toxic / Encore / Rest / Sleep Talk`) | Beats all 44 Majors in both L50 tiers against `offantom` Normal core `P2_55_DT (0.94) + P2_55_Ice (0.88) + Lax55_Left (0.56)` (lead `1.00 / 1.00 / 0.56`; vs Umbreon `0.62` on P2 pair, **`+25 pp`**). |
| `L55_ACE` | **Magneton** `L55 @ Leftovers` (`Thunder / Hidden Power Ice / Rest / Sleep Talk`) | `swMean = 0.45`, `ldMean = 0.53`, beats **7/16** anchors on switch-in (`Zap55_DP + Venu50 + Skarm50` `0.94` vs Electabuzz `0.56`, **`+38 pp`**). |
| `L55_ACE` | **Muk** `L55 @ Leftovers` (`Sludge Bomb / Dynamic Punch` or `Curse / Rest / Sleep Talk`) | Beats all 44 Majors on `pb-84` core (`Raik55 0.81 / Champ50 1.00 / Lax55 0.31`); Curse variant reaches `ldMean = 0.56`, `swMean = 0.41`. |
| Other strong | **Slowking / Slowbro** `L50`, **Ampharos** `L50/L55`, **Lanturn** `L50`, **Vaporeon** `L55`, **Espeon** `L55` (`ldMean = 0.59`) | Localized 2–3 anchor breakouts (e.g., `Slowking50` vs `Champ50/Skarm50/Star50/Eggy50`). |

### 3. Stage 3 (`243k` Preview Gate) & Stage 4 (`9k/81k` 32g CRN A/B)

Panel (8 shells from `team-pool-v3`): `s0053-lefties`, `pb-84`, `mjj-2014-offantom`, `mjj-2014-thdrcomb`, `pb-6`, `pb-122`, `pb-131`, `rental-cban-8`.

**L55 minor aces dropped into `s0053-lefties-hpi` Slot 2 (`Suicune55`) were eliminated at Preview Gate:**
- `Magneton55 @ Mint`: **`0.0%`** pick rate
- `Muk55 @ Mint`: **`0.0%`** pick rate
- `Vaporeon55 @ Mint`: **`0.0%`** pick rate
- `Espeon55 @ Miracle`: **`6.2%`** pick rate

**L50 minor drop-ins (`pb-84` Slot 6 `Umbreon50 @ Leftovers` and `s0053-lefties-hpi` Slot 1 `Electrode50`):**

| Candidate | 243k Pick% | 9k/81k WR (vs Base) | WR When Picked (Games) | Key Matchup Notes |
| :--- | ---: | ---: | ---: | :--- |
| `pb84_base` (`Umbreon50`) | 25.0% | 50.0% (`+0.0 pp`) | 0% (4g) | Baseline (dead Slot 6) |
| **`pb84_piloswine50`** (`EQ / Ice Beam / Rest / Talk`) | **50.0%** | **65.6% (`+15.6 pp`)** | **79% (14g)** | **Strongest candidate**: `100%` vs `pb-6 / pb-122 / pb-131`, `75%` vs `pb-84` |
| **`pb84_shuckle50`** (`Toxic / Encore / Rest / Talk`) | **43.8%** | **62.5% (`+12.5 pp`)** | 36% (14g) | `100%` (4–0) vs `offantom` (base `25%`), matchup-polarized |
| **`pb84_nidoqueen50`** (`EQ / Curse / Rest / Talk`) | **37.5%** | **62.5% (`+12.5 pp`)** | 46% (13g) | `100%` vs `pb-84 / pb-122`, `75%` vs `s0053 / thdr` |
| `pb84_slowking50` (`Surf / Fire Blast / Rest / Talk`) | 43.8% | 50.0% (`+0.0 pp`) | 50% (16g) | `100%` vs `s0053`, `0%` vs `offantom` |
| `pb84_ampharos50` (`Thunder / HP Ice / Rest / Talk`) | 75.0% | 43.8% (`-6.2 pp`) | 45% (22g) | Preview over-pick trap |
| `pb84_lanturn50` (`Thunder / Hydro Pump / Rest / Talk`) | 62.5% | 28.1% (`-21.9 pp`) | **17% (18g)** | Severe preview over-pick trap (8-turn rollout bias) |
| `s0053_base` (`Electrode50`) | 6.2% | 40.6% (`+0.0 pp`) | 33% (6g) | Baseline on 8-shell panel |
| **`s0053_shuck50_s1`** (`Shuckle50 @ Miracle`) | **68.8%** | **50.0% (`+9.4 pp`)** | 48% (21g) | `100%` picked vs `offantom`, `min_slot_pct` `6.2% -> 12.5%` |
| `s0053_queen50_s1` (`Nidoqueen50 @ Miracle`) | 62.5% | 37.5% (`-3.1 pp`) | 40% (25g) | Ground overlap zeroed `Marowak55` pick rate (`0.0%`) |
| `s0053_pilo50_s1` (`Piloswine50 @ Miracle`) | 75.0% | 31.2% (`-9.4 pp`) | 33% (21g) | Ground overlap zeroed `Marowak55` pick rate (`0.0%`) |

---

## Methodological Takeaways & Caveats

1. **Three-stage filtering works well as a minor-Pokémon ore separator**: Big-16 switch-in Pareto breakouts rapidly narrowed 99 minors to ~6 structural specialists; `243k` Preview Gate immediately rejected L55 minors that cannot displace L55 flagship aces; cheap `9k/81k` CRN games exposed bulky-RestTalk preview over-pick traps (`Lanturn50`, `Ampharos50`).
2. **Scope boundary**: Team-dependent utility/combo minors (Spikes, dual screens, Baton Pass, Perish-trap, Sleep leads) remain outside this 1v1-based screen by design.
3. **Production verification completed (2026-11-23)**: Follow-up strict NO-CACHE LIVE (`243k/729k`, `0` cache hits) verification across all 32 shells of `team-pool-v3` (see [`okimono-mutation-study.md`](./okimono-mutation-study.md) §3) showed that `pb84_piloswine50` (`37.5%` raw / `43.8%` clean, picked WR `33.3%`), `pb84_shuckle50` (`40.6%`, picked WR `34.5%`), and `pb84_nidoqueen50` (`42.2%`, picked WR `26.3%`) were `9k/81k` preview over-pick traps at `243k/729k`, and `s0053_shuck50_s1` (50/50 camo) regressed on independent 128g holdout (`50.0%` vs `54.7%` base, H2H `31.2%`). In contrast, **`pb84-cln-sking50`** (`Slowking L50 @ Leftovers [Surf / Fire Blast / Rest / Sleep Talk]` paired with `Cloyster @ Gold Berry Surf` + `Gligar @ Bright Powder`) **survived independent 192g LIVE holdout** (`48.4% ± 3.5` vs `39.1%` raw `pb-84`, `+3.6 pp` & `68.8%` H2H vs same-shell `pb84-m-cloysurf`, and `75.0%` vs `rraikou-m-cune50`).
4. **`policy-v2` (`CF-V_net v2` PUCT + `SteppedCfPreview 240k`) Threshold Recalibration (2026-11-24)**: Re-calibrating across **37 strictly NO-CACHE LIVE (`243k/240k`) minor-gem experiments** + a 58-candidate 7-role 3v3 screen (see [`policy-v2-team-pool-refresh.md`](./policy-v2-team-pool-refresh.md) §2) proved that detection thresholds shift from `policy-v1`'s `pick% >= 25%` to a **two-sided band `10% <= pick% <= 42%` + mandatory `InWR - OutWR >= +5 pp` gate** (`0/11` minors `> 42%` passed due to `CF-V_net v2` OOD over-valuation; negative control `pb6_typh50` failed disjoint Panel B `-18.8 pp`). Under `policy-v2`, BOTH **`pb84-cln-sking50` (`Slowking L50`, `55.0%` over `160g`, `18.4%` Nash)** AND **`pb63-m-esp55` (`Espeon L55`, `+6.2 pp` PanA & PanB, `54.6%` 10-Champ RR)** survived disjoint holdout and entered [`data/team-pool-v4/`](../../data/team-pool-v4/).

## Reproduction

```sh
cargo build --release -p nc2000-bot --example gem_1v1
python3 tools/gem-finder.py generate
python3 tools/gem-finder.py run --root-iters 400 --root-sw-iters 140 --sw-samples 5 \
  --play-iters 100 --play-sw-iters 80 --play-games 16 --max-turns 120 \
  --stage2-majors-per-tier 95 --stage2-minors-per-tier 55 --threads 10
python3 tools/gem-finder.py analyze
python3 tools/gem-finder.py preview-gate --preview-iters 243000 --k 2 --threads 10
```
