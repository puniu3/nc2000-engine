# Counterfactual Party A/B (`cf_party_ab`) & `team-pool-v4` Exploration Campaign (`policy-v2`, 2026-11-25)

## Executive Summary

Following the Step 4 `team-pool-v4` refresh ([`policy-v2-team-pool-refresh.md`](./policy-v2-team-pool-refresh.md)), we built, adversarially audited, and deployed a strictly unbiased **3-Stratum Maximal-Coupling Counterfactual Party A/B Estimator** ([`crates/bot/examples/cf_party_ab.rs`](../../crates/bot/examples/cf_party_ab.rs)) exposing read-only `SteppedCfPreview::execution_policy` with **0-ULP bit-exact preservation** of production `policy-v2` (`best_choice` / `PolicyMode::Legacy` / `nash_refresh.rs` untouched).

We then executed a **two-stage, 30-candidate `team-pool-v4` exploration campaign** across all 32 `v4` shells at full production budget (`240k` `SteppedCfPreview` + `243k adaptive` `CfValuePuct`, **`9,112` core battles** + **`2,432` pre-screen `240k` previews**, `16.42 CU` on `barn` `mini1 + mini2 + Colab V5E1`), independently audited at every step by `Model='inherit'` subagents:

1. **Zero promotions into `team-pool-v4` (`0/30` cleared strictly unbiased `K=6` certification at `z >= 1.96`)**: `data/team-pool-v4/` is empirically confirmed to be **locally saturated under `policy-v2` single-slot mutations** within `~3.0–4.0 pp` SE resolution. All 9 Step 1–4 upgrades already captured the large (`+5 to +23 pp`) single-slot gains available under `CF-V_net v2`, and Tier-S same-shell movesets (`sample-08` Mint Rest Snorlax `-14.06 pp [z = -2.77]` if converted to Pink Bow Curse; `pb84-cln-skarm50` RestTalk Snorlax55 `-8.79 pp [z = -1.68]` if stripped) proved well-tuned.
2. **Discovery of Non-Zero Hidden-Bench Decoy Effects (`Term 3` ≠ 0 in Disclosed Species Swaps)**: Explicit `term1` vs `term3` counterfactual decomposition uncovered a genuine structural trade-off invisible to standard A/B tests:
   - **`t3_pb63g_s3_venu50`** (`pb63-m-gar55` s3 `Blissey50 → Venusaur50 @ PRZ Cure Berry`): Replicated a statistically real **Stratum-1 win (`+5.21 ± 2.24 pp`, `z = +2.32`, `64W–44L–112T`** in independent Stage-2a; combined `K=8` `t1 = +6.64 ± 1.96 pp`, **`z = +3.39`**), **but lost `-3.26 ± 1.93 pp` (`z = -1.69`, `9W–17L–73T`, `9` negative vs `3` positive opps, `LOO-1 = -2.10 pp`) in Stratum 3** when neither Blissey nor Venusaur was picked — netting **`+1.95 ± 3.01 pp` (`z = +0.65`)** (not separated from Major control `ctrl_pb63g_s3_cune50` `-1.07 ± 3.39 pp`, diff `z = +0.67`).
   - **`t3_s06_s5_missy50`** (`sample-06` s5 `Raikou55 → Misdreavus50 @ Gold Berry`): Lost **`-4.88 ± 2.50 pp` (`z = -1.96`, `11W–23L–54T`) in Stratum 3** when removing the Lv55 `Raikou55` bench threat (`delta_u = -3.84 ± 3.82 pp`), whereas adding a physical-wall threat (`pb-6` `Umbreon50 → Skarmory50`) gained `+2.44 ± 2.41 pp` in Stratum 3 (`delta_u = +3.48 ± 3.62 pp`, `z = +0.96`).
   - **Methodological rule**: Stage-1 `--skip-term3` screening (`1.8–2.7×` VR) is an effective fast filter, but disclosed species swaps **must always pass strictly unbiased 3-stratum certification (Term 3 included)**.
3. **Empirically Calibrated `cf_party_ab --alloc neyman --preview-reps 2` Efficiency**: Across 36 production runs, observed compute-weighted VR matched analytical theory cleanly (`1.79×` unbiased Z1 moveset tweaks, `1.76–2.70×` `--skip-term3` screening, `1.14×` unbiased Lv50 disclosed species swaps bounded by high `tau ≈ 0.55–0.75` and irreducible `E[D3²] = 0.316`, with preview CPU only `16.6%` at `K=6`).

---

## 1. Counterfactual Estimator Architecture & Validation (`94053034`)

- **Read-only Turn-0 execution policy exposure**: [`crates/bot/src/preview_search.rs`](../../crates/bot/src/preview_search.rs), [`crates/bot/src/blind.rs`](../../crates/bot/src/blind.rs), [`crates/bot/src/import.rs`](../../crates/bot/src/import.rs) (`SteppedCfPreview::execution_policy`, `BlindSearch::preview_execution_policy`, `ProtocolAgent::preview_execution_policy`). Verified 0-ULP identical `best_choice` across 256 seeds in both pure and mixed modes, `583/583` `cargo t`, and `12/12` Node WASM contract tests.
- **3-Stratum Maximal Coupling ([`crates/bot/examples/cf_party_ab.rs`](../../crates/bot/examples/cf_party_ab.rs))**:
  - **Stratum 1 ($\tau = \text{TV}(\pi_1, \pi_0)$)**: Non-overlapping residual difference with secondary maximal $B$-coupling + tertiary exact-$A$/lead coupling.
  - **Stratum 2 ($m_2$, mutated slot $s \in a$)**: Same-index overlap evaluated under Turn 1+ Battle CRN.
  - **Stratum 3a ($m_{3z}$, provably bit-exact Zero-Term-3, 0 battles)**:
    - *Condition Z1*: Identical public Turn-0 frame `(species, level, gender, has_item)` + identical opponent belief pool + identical `party_order_matters` + deterministic genders.
    - *Condition Z2*: Exact 155-level-cap exclusion (e.g., Lv55 lead excluding Lv55 mutated slot) + strict singleton belief alignment (`sets_json_eq`) + `species_immunity_mask` guard against `mask.rs::foe_switchin_candidates` type-immunity flips.
    - Live `--verify-zero-term3` at `240k/243k` confirmed `0` `actionSha256` mismatches (`3/3` Z1, `11/11` Z2).
  - **Stratum 3b ($m_3$, hidden-bench belief)**: Unbiased sampling under default `--alloc neyman` (`p_j = w_j · (σ_j / σ_1)` with `(1.0, 0.80, 0.64)`, eliminating multi-stratum over-allocation and Jensen `E[w²]` taxes) + `--preview-reps R` unbiased policy averaging + Rao-Wu rescaled stratified bootstrap 95% CI.
- **Zero-bias validation vs `step4-true-crn-audit` (32 shells, `K=2`, `0.00 CU`, `target/nash-work/okimono-v2/cf-party-ab-bench/`)**:
  - `pb84-m-cloysurf` vs `v3:pb-84` (Z1 shared belief): `+1.20 ± 4.22 pp` vs `step4` `+0.78 ± 6.10 pp` (`z_diff = +0.06`).
  - `pb63-m-gar55` vs `v3:pb-63` (disclosed Lv55 Z2): `+8.76 ± 5.50 pp` vs `step4` `+3.91 ± 5.14 pp` (`z_diff = +0.63`).
  - `pb84-cln-skarm50` vs `pb84-m-cloysurf` (disclosed Lv50): **`+11.10 ± 3.98 pp` (`z = +2.79`)** vs `step4` `+7.03 ± 5.98 pp` (`z_diff = +0.57`, `delta_fight = +12.69 ± 5.17 pp`, `delta_adapt = -1.59 ± 2.90 pp`).

---

## 2. Stage-1 Exploration Screen (`30` Candidates × `32` `v4` Shells, `5,314` Core Battles, `6.17 CU`)

All 30 candidates passed WASM `Validator.canonicalizeTeam` → `training-v3` → `Validator.validateTeam` → `new wasm.Battle`, `castle-validation.json`, JP NC2000 legality, and pre-flight `240k K=2` (`64` samples/cand across all 32 `v4` shells) preview calibration (`10/10` Track-2 Majors `min_pick >= 12.5%`, `12/12` Track-3 distinct Minor species strictly in `[15.6%, 39.1%]` with `min_pick >= 14.1%`, `8/8` Track-1 same-shell tweaks live-verified `same_public_frame = true`).

Settings: `--alloc neyman --preview-reps 2 --clusters 2`; Track 1 used `--shared-opp-belief` (strictly unbiased Z1, `m3b = 0`); Tracks 2 & 3 used disclosed beliefs + `--skip-term3` screening. Raw data: `target/nash-work/okimono-v2/v4-cf-explore-stage1/`.

### 2.1 Track 1 — Tier-S / Top-8 Same-Shell Moveset & Item Tweaks (Strictly Unbiased Z1)

| Candidate | Parent (`Slot`) | Mutation | Unbiased `Δ ± SE` (`z`) | `tau` / `m2` / `m3z` | Battles | VR |
|---|---|---|---:|---|---:|---:|
| `t1_rraikou_champ_talk` | `rraikou-m-cune50` (`s6`) | `Machamp50@Scope [CC/RS/Rest/Talk]` | `+5.27 ± 6.64 pp` (`+0.79`) | `0.43 / 0.23 / 0.35` | `158` | `1.44×` |
| `t1_pb122_cloy_goldsurf` | `pb122-m-lax50` (`s6`) | `Cloyster50@Gold [Surf/IB/Spikes/Exp]` | `+3.13 ± 5.98 pp` (`+0.52`) | `0.42 / 0.32 / 0.27` | `156` | `1.80×` |
| `t1_pb122_raik_hpi` | `pb122-m-lax50` (`s2`) | `Raikou55@Mint [TBolt/HPI/Reflect/Rest]` | `+2.15 ± 4.97 pp` (`+0.43`) | `0.37 / 0.20 / 0.42` | `132` | `3.07×` |
| `t1_pb6_cloy_icebeam` | `pb-6` (`s5`) | `Cloyster50@Gold [Surf/IB/Spikes/Exp]` | `-0.39 ± 6.10 pp` (`-0.06`) | `0.45 / 0.44 / 0.12` | `202` | `1.33×` |
| `t1_pb63_zap_minttbolt` | `pb63-m-gar55` (`s6`) | `Zapdos50@Mint [TBolt/HPI/Rest/Talk]` | `-3.32 ± 4.93 pp` (`-0.67`) | `0.42 / 0.51 / 0.07` | `222` | `1.85×` |
| `t1_pb66_zap_talk` | `pb66-m-wak55` (`s3`) | `Zapdos50@Mint [TBolt/HPI/Rest/Talk]` | `-4.88 ± 5.53 pp` (`-0.88`) | `0.44 / 0.49 / 0.07` | `202` | `1.62×` |
| `t1_pb84_lax_curseboom` | `pb84-cln-skarm50` (`s1`) | `Snorlax55@Miracle [BS/EQ/Curse/SD]` | `-8.79 ± 5.23 pp` (`-1.68`) | `0.58 / 0.15 / 0.27` | `172` | `2.13×` |
| `t1_s08_lax_cursebow` | `sample-08` (`s5`) | `Snorlax50@PinkBow [BS/EQ/Curse/SD]` | **`-14.06 ± 5.09 pp` (`-2.77`)** | `0.43 / 0.42 / 0.15` | `194` | `1.99×` |

### 2.2 Track 2 — Untapped `< 8%` Okimono Major Upgrades (`--skip-term3` Screening)

| Candidate | Parent (`Slot`) | Replacement | Screen `Δ ± SE` (`z`) | `tau` / `m3b` | Battles | VR | Stage 2a? |
|---|---|---|---:|---|---:|---:|:---:|
| `t2_s06_s5_zap50` | `sample-06` (`s5`) | `Zapdos50@Gold [TBolt/HPI/Rest/Talk]` | `+8.59 ± 5.90 pp` (`+1.46`) | `0.77 / 0.23` | `206` | `1.40×` | **Yes (`K=6`)** |
| `t2_pb6_s4_star50` | `pb-6` (`s4`) | `Starmie50@QC [Surf/Psy/TWave/Rec]` | `+6.84 ± 4.79 pp` (`+1.43`) | `0.63 / 0.36` | `170` | `2.57×` | No (polarized) |
| `t2_pb6_s4_skarm50` | `pb-6` (`s4`) | `Skarmory50@Mint [DP/Curse/WW/Rest]` | `+6.25 ± 5.18 pp` (`+1.21`) | `0.53 / 0.47` | `138` | `2.70×` | **Yes (`K=6`)** |
| `t2_s06_s5_gar50` | `sample-06` (`s5`) | `Gengar50@Gold [TBolt/IP/Exp/DBond]` | `+4.69 ± 6.05 pp` (`+0.77`) | `0.86 / 0.14` | `224` | `1.22×` | No |
| `t2_pb84sk_s5_wak50` | `pb84-cln-skarm50` (`s5`) | `Marowak50@Club [EQ/RS/HPB/SD]` | `+1.56 ± 6.05 pp` (`+0.26`) | `0.85 / 0.14` | `226` | `1.21×` | No |
| `t2_pb63g_s3_eggy50` | `pb63-m-gar55` (`s3`) | `Exeggutor50@QC [Psy/SP/Seed/Exp]` | `+0.78 ± 4.35 pp` (`+0.18`) | `0.63 / 0.36` | `168` | `3.15×` | No |
| `t2_pb84sk_s5_eggy50` | `pb84-cln-skarm50` (`s5`) | `Exeggutor50@QC [Psy/SP/Seed/Exp]` | `-3.12 ± 6.15 pp` (`-0.51`) | `0.81 / 0.18` | `222` | `1.19×` | No |
| `t2_pb87_s3_champ50` | `pb-87` (`s3`) | `Machamp50@Scope [CC/RS/Rest/Talk]` | `-5.47 ± 4.88 pp` (`-1.12`) | `0.83 / 0.16` | `208` | `2.02×` | No |
| `t2_tsuru_s3_raik50` | `hc75-top8-tsuru` (`s3`) | `Raikou50@Mint [TBolt/HPI/Rest/Talk]` | `-7.81 ± 5.63 pp` (`-1.39`) | `0.84 / 0.16` | `216` | `1.46×` | No |
| `t2_pb87_s3_gar50` | `pb-87` (`s3`) | `Gengar50@PRZ [TBolt/IP/Exp/DBond]` | **`-11.72 ± 3.91 pp` (`-3.00`)** | `0.53 / 0.47` | `140` | `4.68×` | No |

### 2.3 Track 3 — Calibrated `policy-v2` Minor-Gem Challengers (`15.6%–39.1%`, `--skip-term3` Screening)

| Candidate | Parent (`Slot`) | Minor Set | Screen `Δ ± SE` (`z`, `z_can`) | `tau` / `m3b` | Battles | VR | Stage 2a? |
|---|---|---|---:|---|---:|---:|:---:|
| `t3_pb63g_s3_venu50` | `pb63-m-gar55` (`s3`) | `Venusaur50@PRZ [RL/SP/Seed/Synth]` | **`+10.94 ± 3.12 pp` (`+3.50`, `+2.33`)** | `0.56 / 0.44` | `132` | `7.76×` | **Yes (`K=6`)** |
| `t3_s06_s5_missy50` | `sample-06` (`s5`) | `Misdreavus50@Gold [Th/ML/DB/Perish]` | `+7.81 ± 4.56 pp` (`+1.71`, `+1.60`) | `0.61 / 0.39` | `156` | `3.09×` | **Yes (`K=6`)** |
| `t3_pb87_s3_ebuzz50` | `pb-87` (`s3`) | `Electabuzz50@Scope [TB/IP/CC/FP]` | `+5.47 ± 3.58 pp` (`+1.53`, `+1.30`) | `0.45 / 0.55` | `128` | `6.10×` | **Yes (`K=4`)** |
| `t3_pb63g_s3_vap50` | `pb63-m-gar55` (`s3`) | `Vaporeon50@Mint [Surf/IB/AA/Rest]` | `+5.47 ± 4.88 pp` (`+1.12`, `+1.06`) | `0.68 / 0.32` | `172` | `2.44×` | No |
| `t3_pb84sk_s5_quag50` | `pb84-cln-skarm50` (`s5`) | `Quagsire50@Mint [EQ/Surf/BD/Rest]` | `+5.27 ± 4.90 pp` (`+1.08`, `+1.08`) | `0.60 / 0.39` | `144` | `2.89×` | No |
| `t3_pb84sk_s5_jynx_thief` | `pb84-cln-skarm50` (`s5`) | `Jynx50@none [IB/Psy/Thief/LK]` | `+0.78 ± 5.24 pp` (`+0.15`) | `0.76 / 0.24` | `200` | `1.82×` | No |
| `t3_pb6_s4_forre_nospk` | `pb-6` (`s4`) | `Forretress50@QC [HPB/Spin/Curse/Exp]` | `-0.20 ± 3.46 pp` (`-0.06`) | `0.61 / 0.38` | `150` | `5.58×` | No |
| `t3_tsuru_s3_jolt50` | `hc75-top8-tsuru` (`s3`) | `Jolteon50@PRZ [TBolt/HPI/TW/Roar]` | `-1.56 ± 5.52 pp` (`-0.28`) | `0.80 / 0.20` | `202` | `1.62×` | No |
| `t3_pb87_s3_hera_curse` | `pb-87` (`s3`) | `Heracross50@PRZ [MH/EQ/Curse/Rest]` | `-1.56 ± 5.06 pp` (`-0.31`) | `0.61 / 0.38` | `140` | `2.79×` | No |
| `t3_pb6_s4_rhydon50` | `pb-6` (`s4`) | `Rhydon50@QC [EQ/RS/Curse/Roar]` | `-3.12 ± 4.69 pp` (`-0.67`) | `0.64 / 0.35` | `176` | `2.59×` | No |
| `t3_tsuru_s3_aero50` | `hc75-top8-tsuru` (`s3`) | `Aerodactyl50@PRZ [EQ/HPR/FB/WW]` | `-4.69 ± 5.74 pp` (`-0.82`) | `0.64 / 0.36` | `162` | `1.87×` | No |
| `t3_s06_s5_trode50` | `sample-06` (`s5`) | `Electrode50@PRZ [TB/Exp/LS/TW]` | `-5.47 ± 4.75 pp` (`-1.15`) | `0.72 / 0.28` | `196` | `2.26×` | No |

---

## 3. Stage-2a Strictly Unbiased Certification (`6` Candidates × `32` `v4` Shells, `3,798` Core Battles, `10.25 CU`)

Fresh independent seeds (`20262200+`), full 3-stratum `--alloc neyman --preview-reps 2` (**Term 3 included**, zero Stage-1 data reuse), plus same-slot Major control `ctrl_pb63g_s3_cune50` (`Suicune50@Mint`, `mut_pk=50.0%`, `min_pk=17.2%` in pre-flight `K=2` check). Raw data: `target/nash-work/okimono-v2/v4-cf-explore-stage2a/`.

| Candidate (Parent, `K`) | **Unbiased `ΔU ± SE` (`z`)** | Rao-Wu 95% CI | **Stratum 1 `t1 ± SE` (`z`)** | **Stratum 3 `t3 ± SE` (`z`)** | `S1` / `S3` W–L–T | Opp `+/−` | Battles | VR | Verdict |
|---|---:|---|---:|---:|---|---:|---:|---:|---|
| `t2_pb6_s4_skarm50` (`pb-6`, `K=6`) | `+3.48 ± 3.62 pp` (`+0.96`) | `[-3.7, +10.7]` | `+1.04 ± 2.70` (`+0.39`) | `+2.44 ± 2.41` (`+1.01`) | `48–44–119` / `21–15–69` | `16 / 11` | `632` | `1.20×` | Null (`z < 1.96`) |
| `t3_pb63g_s3_venu50` (`pb63-m-gar55`, `K=6`) | `+1.95 ± 3.01 pp` (`+0.65`) | `[-4.1, +7.8]` | **`+5.21 ± 2.24` (`+2.32`)** | `-3.26 ± 1.93` (`-1.69`) | `64–44–112` / `9–17–73` | `14 / 13` | `638` | `1.73×` | **Trade-off (`t1` vs `t3`), No Promo** |
| `t2_s06_s5_zap50` (`sample-06`, `K=6`) | `+1.86 ± 4.04 pp` (`+0.46`) | `[-5.9, +10.0]` | `+1.04 ± 3.52` (`+0.30`) | `+0.81 ± 1.78` (`+0.46`) | `85–81–141` / `10–8–36` | `15 / 13` | `722` | `0.85×` | Null (S1 Winner's Curse) |
| `t3_pb87_s3_ebuzz50` (`pb-87`, `K=4`) | `+1.22 ± 4.59 pp` (`+0.27`) | `[-7.8, +10.1]` | `+1.95 ± 2.98` (`+0.65`) | `-1.22 ± 2.99` (`-0.41`) | `33–28–53` / `10–12–70` | `14 / 11` | `414` | `1.15×` | Null |
| `ctrl_pb63g_s3_cune50` (`pb63-m-gar55`, `K=6`) | `-1.07 ± 3.39 pp` (`-0.32`) | `[-7.6, +5.5]` | `-0.26 ± 2.96` (`-0.09`) | `-0.81 ± 1.73` (`-0.47`) | `65–66–190` / `8–10–32` | `15 / 14` | `742` | `1.17×` | Major Control Null |
| `t3_s06_s5_missy50` (`sample-06`, `K=6`) | `-3.84 ± 3.82 pp` (`-1.01`) | `[-11.3, +3.5]` | `+1.04 ± 2.80` (`+0.37`) | **`-4.88 ± 2.50` (`-1.96`)** | `58–54–125` / `11–23–54` | `11 / 15` | `650` | `1.06×` | **Rejected (`t3` decoy loss)** |

---

## 4. Implications for the Three-Body Co-Evolution Loop

1. **Stop single-slot mutation mining on `team-pool-v4` under `policy-v2`**: `team-pool-v4` is empirically confirmed at a local optimum for single-slot changes under `CF-V_net v2` (`0/30` promoted at `K=6`, `±3.0–4.0 pp` SE).
2. **Next High-Leverage Moves**:
   - **Option A — Properly Powered `policy-v2` Nash Certification (`>=48g/edge`)**: Replace the provisional uniform 5-shell `flagship-mixture.json` with a bootstrap-stable `policy-v2` Nash equilibrium across `team-pool-v4`'s top ~8–10 shells (`rraikou-m-cune50`, `pb122-m-lax50`, `pb63-m-gar55`, `pb66-m-wak55`, `sample-08`, `ayagen-m-zap50`, `pb84-cln-skarm50`, `pb-6`, `s0053-lefties`).
   - **Option B — Close the Three-Body Loop (`CF-V_net v3` NN Retraining)**: Retrain `CF-V_net` on `team-pool-v4` + OOD minor-gem self-play (`Scizor`, `Clefable`, `Weezing`, `Vaporeon55`, `Muk`, `Venusaur50`, `Misdreavus50`, `Quagsire50`, `Electabuzz50`) to eliminate the `>42%` OOD over-valuation trap before searching for genuinely novel archetypes under `policy-v3`.
   - **Option C — Full 6-Mon Novel Archetype Synthesis**: Rather than single-slot okimono surgery on existing shells, synthesize brand-new 6-mon synergies specifically around `policy-v2`'s tactical strengths (Spikes/Explosion/Encore/Perish-Trap/dual-L55 splits) and evaluate them directly against the `team-pool-v4` field.
