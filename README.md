# Drought Severity Prediction

**Project:** NYCU Data Mining Spring 2026 Final Project — Natural Disaster Severity Prediction (Drought)
**Task:** Predict weekly drought severity scores (0–5) for the 5 weeks following each region's 91-day weather window
**Metric:** Mean Absolute Error (MAE), lower is better
**Evaluation:** Public LB on 40% of test data; Private LB on remaining 60% (hidden)
**Updated:** May 19, 2026

---

## Current Standing

| Bracket | Threshold | Points |
|---|---|---|
| Above Baseline 3 | LB < 0.8056 | 42–60 |
| Between Baselines 2 & 3 | 0.8056 ≤ LB < 0.8623 | 25–40 |
| Between Baselines 1 & 2 | 0.8623 ≤ LB < 0.9117 | 10–25 |
| Under Baseline 1 | LB ≥ 0.9117 | 5 |

**Best confirmed LB: ensemble_3way V5 + V7-10seed + V10-5seed at weights (0.10 / 0.50 / 0.40) = 0.7942** — clears Baseline 3 (0.8056) by 0.0114. Estimated points: 42–60 for Kaggle portion, exact placement depends on rank.

**Top two public-LB submissions (Kaggle auto-pick for final):**
1. (0.10 / 0.50 / 0.40) V5 / V7-10 / V10-5 → 0.7942
2. (0.15 / 0.45 / 0.40) V5 / V7-10 / V10-5 → 0.7944

Both share the same architectural shape — robust safety profile for private LB.

---

## Data Summary

- **Train:** 12,319,040 rows × 21 columns (2,248 regions × 5,480 days each)
- **Test:** 204,568 rows × 20 columns (2,248 regions × 91 days each, no `score` column)
- **Submission:** 2,248 rows × 6 columns
- **Score distribution (training):** 60% zeros, mean 0.836, median 0
- **Scores recorded weekly** on `day_idx % 7 == 6`
- **No missing weather features** in either train or test
- **Anonymized far-future dates** (years 3000+) — handled via string slicing + per-region day indices

---

## Submission History (Chronological)

| # | Model | Public LB | Notes |
|---|---|---|---|
| 1 | V1 — monthly mean lookup | 0.9506 | Established baseline |
| 2 | V1.5 — monthly median | 1.0271 | Median over-confident on hard cases |
| 3 | V2 — minimal LGBM | 0.9435 | Validation was leaky |
| 4 | V3 — LGBM with engineered features | 0.9168 | Honest buffered validation |
| 5 | V3 calibrated (blend=0.50) | 0.9673 | Quantile mapping hurt — ranking was the bottleneck |
| 6 | V5 — LGBM with climatology + score history | 0.9017 | First model above Baseline 1 |
| 7 | Ensemble V3 + V5 + V6 | 0.9035 | High pairwise correlation; ensembling didn't help |
| 8 | V7 — TCN | **0.8463** | Cleared Baseline 2; first sequence model |
| 9 | Ensemble V5 + V7 (mean) | 0.8353 | Cross-architecture ensembling worked |
| 10 | **V7 5-seed averaged** | **0.8071** | Variance reduction gave 0.039 jump |
| 11 | Ensemble 3-way V5 + V7-5seed + V10-single (v7stronger) | 0.8062 | Tied V7-5seed; V10 single-seed added little |
| 12 | **Ensemble 3-way V5 + V7-5seed + V10-5seed (v7stronger rebuilt)** | **0.7971** | Cleared Baseline 3 by 0.0085; V10-5seed contributed once seed-averaged |
| 13 | V10-TCN-static 5-seed avg (standalone) | 0.8159 | Standalone benchmark for V10; weaker than V7-5seed (0.8071) but contributes in ensemble |
| 14 | Ensemble V7-5seed + V10-5seed (60/40, no V5) | 0.7973 | Near-tie with 3-way (0.7971); V5 contributes ~0 to the ensemble |
| 15 | Per-week unconstrained Ridge stacker (V7-5seed + V10-5seed) | 0.7991 | Val-fit weights ~0.80/0.18; worse than 60/40 by 0.0020 — val/LB inversion confirmed |
| 16 | **Ensemble 3-way V5 + V7-10seed + V10-5seed** | **0.7948** | New best; 10-seed V7 cut another 0.0023 off LB. Seed averaging not yet at ceiling for V7. |
| 17 | Ensemble 3-way V5 + V7-10seed + V10-10seed (0.15/0.55/0.30) | 0.7965 | V10 5→10 seed averaging *hurt* the ensemble by 0.0017. V10 has hit its variance-reduction ceiling for ensembling. |
| 18 | Ensemble 3-way V5 + V7-10seed + V10-10seed (V7-heavy: 0.15/0.65/0.20) | 0.7969 | Tilting more weight to V7 didn't recover the loss — confirmed V10-10seed itself is the regression, not the weight balance. |
| 19 | **Ensemble 3-way V5 + V7-10seed + V10-5seed (V10-heavier: 0.15/0.50/0.35)** | **0.7945** | Weight sweep gain: trimming V7 by 0.05 toward V10-5seed gained 0.0003 LB. Confirmed V10-5seed (not V10-10seed) is the right diversifier. |
| 20 | **Ensemble 3-way V5 + V7-10seed + V10-5seed (more V10: 0.15/0.45/0.40)** | **0.7944** | Continuing the V10 gradient gained another 0.0001 LB. Step sizes shrinking — near plateau. |
| 21 | **Ensemble 3-way V5 + V7-10seed + V10-5seed (V5 trimmed: 0.10/0.50/0.40)** | **0.7942** | Trimming V5 from 0.15 to 0.10 gained 0.0002 LB. Current best. |
| 22 | Ensemble 2-way V7-10seed + V10-5seed (V5 dropped: 0/0.55/0.45) | 0.7961 | Dropping V5 entirely cost 0.0019 LB — V5 is **not** redundant at the new operating point. Reverses May 14 finding. |

---

## Full Model Catalog

### Tabular / Lookup-based

| # | Model | Val MAE | LB | Status |
|---|---|---|---|---|
| V1 | Per-region monthly mean | 0.79 | 0.9506 | Submitted |
| V1.5 | Per-region monthly median | — | 1.0271 | Submitted |
| V2 | LGBM, basic aggregates (58 features) | 0.524 (leaky) | 0.9435 | Submitted |
| V3 | LGBM with engineered features (90 features) + buffered val | 0.4634 | 0.9168 | Submitted |
| V3-calibrated | Quantile-mapped V3 (blend=0.5) | — | 0.9673 | Submitted (regression) |
| V5 | V3 + climatology anomalies + score history (105 features) | 0.4556 | 0.9017 | Submitted ⭐ |
| V6 | V5 + recent-drought features (117 features) | 0.4593 | — | Trained, not submitted |

### Sequence-based (NN architectures)

| # | Model | Val MAE | LB | Status |
|---|---|---|---|---|
| V7 | TCN (5 dilated blocks × 64 ch) | 0.4322 | 0.8463 | Submitted |
| V7.1 | TCN over-regularized | 0.4586 | — | Trained, not submitted |
| V7-large | TCN (6 blocks × 96 ch) | 0.4658 | — | Trained, not submitted |
| **V7 5-seed** | TCN averaged across seeds 42–46 | ~0.44 avg | **0.8071** | Submitted ⭐⭐ |
| **V7 10-seed** | TCN averaged across seeds 42–51 | (likely ~0.43) | — | Trained; used in best ensemble (LB 0.7942) ⭐⭐⭐ |
| V8 | P-sLSTM single-head (Kong et al. 2025) | 0.7128 | — | Trained, not submitted |
| V8.1 | P-sLSTM multi-head | 0.6918 | — | Trained, not submitted |
| V8.1 5-seed | V8.1 averaged across seeds 42–46 | 0.7124 avg per-seed | — | Trained, not submitted — seed averaging didn't fix structural bias |
| V9 | iTransformer (Liu et al. 2024) | 0.6791 | — | Trained, not submitted |
| V10 | TCN + V5 static features fused | 0.4498 | — | Trained, not submitted |
| **V10 5-seed** | V10 averaged across seeds 42–46 | ~0.456 avg | **0.8159** | Submitted ⭐⭐ |
| V10 10-seed | V10 averaged across seeds 42–51 | (lower variance) | — | Trained; hurt the ensemble (see row 17), not used in final picks |

### Ensembles

| Combination | LB | Notes |
|---|---|---|
| V3 + V5 + V6 (mean) | 0.9035 | All three tree-based, high correlation (0.91–0.98) |
| V5 + V7-single (mean) | 0.8353 | Cross-architecture diversity (correlation 0.73) |
| V5 + V7-5seed + V10-single (v7stronger) | 0.8062 | V10-single contributed marginally |
| **V5 + V7-5seed + V10-5seed (v7stronger rebuilt)** | **0.7971** | First submission above Baseline 3; V10-5seed contributed once noise was stripped |
| V7-5seed + V10-5seed (60/40, no V5) | 0.7973 | Near-tie with 3-way; V5 contributes ~0 once V10 is present |
| Per-week unconstrained Ridge stacker (V7-5seed + V10-5seed) | 0.7991 | Val-fit weights ~0.80/0.18; lost 0.0020 LB to 60/40 — val rankings disagree with LB |
| Per-week constrained tilt stacker (±0.15 around 60/40) | not submitted | Diagnostic showed all weeks hit +0.15 cap (uniformly V7-favoring); no per-week structure to exploit |
| **V5 + V7-10seed + V10-5seed** (0.15/0.55/0.30) | 0.7948 | First V7-10seed ensemble — replaced V7-5seed for a 0.0023 gain |
| V5 + V7-10seed + V10-10seed (0.15/0.55/0.30) | 0.7965 | V10 5→10 seed averaging *hurt* the ensemble (V10 hit its variance-reduction ceiling) |
| V5 + V7-10seed + V10-10seed (V7-heavy: 0.15/0.65/0.20) | 0.7969 | Weight tilt couldn't recover V10-10seed regression |
| V5 + V7-10seed + V10-5seed (V10-heavier: 0.15/0.50/0.35) | 0.7945 | Weight sweep direction confirmed (more V10 helps) |
| V5 + V7-10seed + V10-5seed (more V10: 0.15/0.45/0.40) | 0.7944 | Step shrinking — near plateau |
| **V5 + V7-10seed + V10-5seed (best: 0.10/0.50/0.40)** | **0.7942** | Current best ⭐⭐⭐ |
| V7-10seed + V10-5seed (V5 dropped: 0/0.55/0.45) | 0.7961 | V5 *is* contributing at the new operating point (cost 0.0019 to remove) |

---

## Failed Experiments (For Report)

### Failure 1 — V1.5 monthly median
Hypothesis: Median is MAE-optimal for skewed distributions.
Reality: Per-region-month medians are over-confident on chronic-drought cases (LB 1.03).
Lesson: MAE-optimal-median applies to single constant predictions, not conditional predictions on small groups.

### Failure 2 — V4 in-window score features
Hypothesis: 13 weekly scores within each 91-day window would improve predictions.
Reality: Val 0.20 but Kaggle 1.20 — test.csv has no score column, so features were all 0 at inference.
Lesson: Features must be computable identically at training and inference.

### Failure 3 — V3 calibration
Hypothesis: Distribution mismatch was causing the V3 LB gap.
Reality: Quantile mapping made it worse (0.9168 → 0.9673).
Lesson: Distribution mismatch was a symptom; ranking quality was the bottleneck.

### Failure 4 — V8 / V9 sequence architectures
Hypothesis: Sophisticated time-series architectures (sLSTM, iTransformer) would beat TCN.
Reality: All underperformed by 0.25+ val MAE.
Lesson: Channel independence (P-sLSTM, PatchTST) fails when channels are highly correlated. iTransformer's cross-channel attention didn't compensate — the 91-day sequence is too short for attention to extract richer signal than dilated convolution.

### Failure 5 — V7-large and V7.1
Hypothesis: V7 needed more capacity OR more regularization.
Reality: Both directions hurt (val 0.466 and 0.459 vs V7's 0.432).
Lesson: V7's 64ch × 5 blocks is a structural sweet spot.

### Failure 6 — V6 feature stacking and V3+V5+V6 ensembling
Hypothesis: More features and more models would compound improvements.
Reality: V5/V6 prediction correlation was 0.978; ensemble gained nothing (0.9017 → 0.9035).
Lesson: Diversity for ensembling must come from different inductive biases, not different features in the same model class.

### Failure 7 — V10 single-seed
Hypothesis: Fusing TCN with V5 static features inside one model would beat ensembling them.
Reality: Val 0.4498; converged in 2 epochs with near-zero train-val gap.
Lesson: Static features dominated training; TCN backbone contributed little additional signal.

---

## Key Findings

1. **Sequence modeling provides the biggest single jump.** TCN (V7) drove LB from 0.90 → 0.85, larger than any feature engineering iteration.

2. **Channel independence hurts on correlated features.** Our 14 weather variables share strong dependencies. Any architecture that processes channels independently throws away joint patterns.

3. **Seed averaging is the highest-leverage trick, and gains continue past 5 seeds.** V7-single → V7-5seed dropped LB by 0.039 — larger than any other single intervention. V7-5seed → V7-10seed (in the 3-way ensemble) added another 0.0023. Total seed-averaging contribution for V7: ≥0.041 LB. The pattern fits 1/√N variance reduction with no plateau through 10 seeds.

4. **Local-Kaggle gap is consistent ~2× for single models, but compresses with ensembling.** V5 val 0.46 → LB 0.90; V7 val 0.43 → LB 0.85; V7-5seed val ~0.44 → LB 0.81. The 3-way ensemble (V5+V7-5seed+V10-5seed) lands at LB 0.7971 — the gap is starting to bend slightly, suggesting ensembling reduces test-era distribution-shift sensitivity, not just variance.

5. **Cross-architecture diversity matters more than within-architecture diversity.** V3+V5+V6 (all tree-based) correlation 0.91–0.98 → 0% ensemble gain. V5+V7 (tree vs CNN) correlation 0.73 → 0.011 gain. V7-5seed+V10-5seed (CNN vs CNN+static) correlation 0.92 → expected minimal gain.

6. **V10-5seed contribution emerged only after seed averaging.** V10-single in the 3-way ensemble produced LB 0.8062 (no gain over V7-5seed alone). The same architecture with seed averaging produced LB 0.7971 (gain of 0.010). Seed averaging on the diversifier was as important as on the main model — single-seed V10 was too noisy for its complementary signal to survive the blend.

7. **V5's contribution is context-dependent — not "always redundant."** Earlier (May 14), dropping V5 from the 3-way blend at weights (0.15/0.55/0.30) cost only 0.0002 LB, suggesting V5 was redundant. After moving to V7-10seed and V10-5seed at weights (0.10/0.50/0.40), dropping V5 entirely cost 0.0019 LB. V5 contributes when the TCN-based components are smooth enough (V7-10seed has very low seed noise) that V5's tree-based predictions become the only genuinely orthogonal signal left. **V5's marginal value depends on what else is in the ensemble** — a more nuanced finding than "V5 is redundant" or "V5 always helps."

8. **V8/V9 failure is structural, not statistical — seed averaging cannot recover it.** V9 (iTransformer) converged to mean predictions of 0.57–0.61 vs ~1.0 for other models (low correlation r ≈ 0.18 = bias, not orthogonal signal). V8/V8.1 (P-sLSTM) avoided V9's failure mode but had higher val MAE (0.71–0.69 vs V7's 0.43) and underprediction bias (means ~0.75 vs ~1.05). We tested whether V8.1's apparent diversity (r = 0.47 with V7-5seed) hid orthogonal signal beneath seed noise: a 5-seed averaged V8.1 yielded per-seed val MAE in 0.69–0.74 range (avg 0.7124) and seed-averaged per-week means of 0.75–0.76 — the bias survived averaging unchanged. **Variance reduction cannot fix architectural mismatch.** TCN's locality bias is the right prior for 91-day → 5-week prediction with this distribution shift; sLSTM and iTransformer's inductive biases are not.

9. **Val/LB inversion blocks data-driven ensemble weighting.** Validation systematically under-credits V10's contribution to LB. On the val set (held-out tail of training data), V7-5seed has lower MAE (0.408) than the 60/40 blend (0.414), and V10-5seed alone is worst (0.433). On LB, the order inverts: 60/40 blend (0.7971) beats V7-5seed (0.8071), with V10-5seed alone weakest (0.8159). Two stacking experiments confirmed this: (a) unconstrained per-week Ridge picked weights ≈0.80/0.18 favoring V7 and scored 0.7991 LB (0.002 worse than 60/40); (b) constrained tilt stacker (±0.15 band around 60/40) had all 5 weeks hit the +0.15 cap, indicating uniformly V7-favoring tilts with no per-week structure to exploit. **Hand-tuned weights anchored to LB feedback outperformed any data-driven weight learned from val.** This is a direct consequence of the local-Kaggle gap (Finding 4) — the test era's distribution shift differs from even the most-recent training era, so val cannot serve as a proxy for test in ensemble optimization.

10. **Seed averaging benefits are architecture-specific and not monotonic.** For V7, the 5→10 seed jump cut ensemble LB by 0.0023 (0.7971 → 0.7948) — consistent with the 1/√N variance-reduction model and contradicting our May 13 working assumption of a plateau. For V10, the same 5→10 jump *hurt* the ensemble by 0.0017 (0.7948 → 0.7965 at the same weights). A V7-heavy weight tilt (0.65 / 0.20) couldn't recover the regression, confirming V10-10seed itself is the issue, not the weight balance. **Interpretation:** V10's value in the ensemble lives in the variance V10-5seed leaves behind. The diversifier's complementary signal is partly encoded in seed-to-seed disagreement; smoothing it away with more seeds makes V10 redundant with V7. This is the inverse of V7's behavior, where the dominant model benefits from maximal smoothing. **Asymmetric seed counts (10 for the main model, 5 for the diversifier) is the right pattern here.**

11. **Weight tuning matters even with high model correlation.** A three-day weight sweep on the (V5, V7-10seed, V10-5seed) ensemble moved LB from 0.7948 → 0.7942 across four submissions. The optimum shifted from the original (0.15/0.55/0.30) to (0.10/0.50/0.40) — meaningfully more weight on V10 and slightly less on V5. The directional signal was consistent (3 of 4 sweeps moved LB in the same direction) but per-slot gains shrank rapidly (0.0003 → 0.0001 → 0.0002), placing us firmly in diminishing-returns territory by the fourth sweep. **For models with 0.92 pairwise correlation, weight tuning still recovers ~0.001 LB but no more.**

12. **Best LB 0.7942 — 0.0114 above Baseline 3.** Final picks (Kaggle auto-selection of top two public): (0.10/0.50/0.40) at 0.7942 and (0.15/0.45/0.40) at 0.7944. Both share the same architectural shape (V5 + V7-10seed + V10-5seed) and differ only in V5/V7 weight split, giving robust private-LB safety. Stacking via val predictions is closed off (Finding 9); V8/V9 architecture path is closed off (Finding 8); V10-10seed path is closed off (Finding 10). Further LB gain would require fundamentally new orthogonal signal not present in any trained model.

---

## Pairwise Model Correlation (Average Across 5 Weeks)

| Pair | Correlation | Interpretation |
|---|---|---|
| V5 ↔ V7-single | 0.77 | Moderate — cross-architecture diversity |
| V5 ↔ V7-5seed | 0.77 | Stable across seed averaging |
| V5 ↔ V10-5seed | 0.78 | V5 ≈ V7 ≈ V10 in pairwise distance — V5 redundant once V10 present |
| V7-single ↔ V10-single | 0.88 | High — V10 mostly mimics V7 |
| **V7-5seed ↔ V10-5seed** | **0.92** | Very high — V10-5seed largely a static-feature-augmented V7 |
| V8 ↔ V7-5seed | 0.47 | Genuinely different signal, but V8 val MAE 0.71 = mostly noise |
| V8 ↔ V10-5seed | 0.44 | Same story; V8 disagrees but is too inaccurate to help |
| V9 ↔ V7-5seed | 0.19 | **Broken** — V9 means 0.6 vs others 1.0; bias not signal |
| V9 ↔ V10-5seed | 0.16 | **Broken** — same diagnosis |

**Insight:** Seed averaging strips out seed-specific noise; what remains is true model agreement, which is high for similar architectures. V5 is the most distinct *available* model and remains genuinely orthogonal to the TCN-based components, though its marginal value in the ensemble depends on how much pseudo-diversity the TCN seed variance is contributing (Finding 7). V8 and V9 reach genuinely different prediction regions but for the wrong reason (noise and bias respectively), not because they encode useful complementary structure (Finding 8).

---

## File Inventory

### Submission scripts (training)
- `baseline_v1_monthly_mean.py` — V1
- `baseline_v1_5_monthly_median.py` — V1.5
- `calibrate_v3.py` — V3 calibration step
- `baseline_v2_lgbm_minimal.py` — V2
- `baseline_v3_lgbm_features.py` — V3
- `baseline_v5_lgbm_score_history.py` — V5 ⭐
- `baseline_v6_lgbm_recent_drought.py` — V6
- `baseline_v7_tcn.py` — V7 ⭐
- `baseline_v7_1_tcn_reg.py` — V7.1
- `baseline_v7_large_tcn.py` — V7-large
- `baseline_v8_pslstm.py` — V8 P-sLSTM
- `baseline_v8_1_pslstm_mh.py` — V8.1 multi-head P-sLSTM
- `baseline_v9_itransformer.py` — V9
- `baseline_v10_tcn_static.py` — V10 (single seed)
- `run_v10_5seeds.py` — V10 5-seed runner ⭐⭐

### Shared utilities
- `data_pipeline_nn.py` — data prep for all NN models

### Ensemble scripts
- `ensemble_v3v5v6.py` — V3+V5+V6 (submitted, LB 0.9035)
- `ensemble_v5_v7.py` — V5+V7 single (submitted, LB 0.8353)
- `ensemble_v5_v7_5seed.py` — V5+V7-5seed
- `ensemble_v5_v7_5seed_v10.py` — V5+V7-5seed+V10 (3-way) ⭐

### Submission CSVs in Drive
- `baseline_v5_lgbm_score_history.csv` (LB 0.9017)
- `baseline_v7_tcn.csv` (LB 0.8463)
- `baseline_v7_5seed_avg.csv` (LB 0.8071) ⭐⭐
- `baseline_v10_5seed_avg.csv` (LB 0.8159) ⭐
- `ensemble_v5v7_mean.csv` (LB 0.8353)
- `ensemble_v5_v7_5seed_v7stronger.csv` (untested)
- `ensemble_3way_v7stronger.csv` (LB 0.7971) ⭐⭐ best
- `ensemble_3way_mean.csv`, `ensemble_3way_nov5.csv`, etc. (variants)

---
