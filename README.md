# Drought Severity Prediction

**NYCU Data Mining, Spring 2026 — Final Project (Natural Disaster Severity Prediction)**

Predict weekly drought severity scores (0–5) for the five weeks following each region's
91-day weather observation window. Evaluation metric is **Mean Absolute Error (MAE)**;
lower is better. The public leaderboard scores 40% of the test set, the private
leaderboard the remaining 60%.

---

## Result

| | |
|---|---|
| **Best public LB** | **0.7942** |
| Best model | 3-way ensemble — V5 (LightGBM) + V7-10seed (TCN) + V10-5seed (TCN + static) |
| Weights | 0.10 / 0.50 / 0.40 |
| Margin over Baseline 3 (0.8056) | 0.0114 |

The two submissions auto-selected for private-LB scoring are the (0.10/0.50/0.40)
blend at 0.7942 and the (0.15/0.45/0.40) blend at 0.7944. Both share the identical
architectural shape and differ only in the V5/V7 weight split, which gives a stable
private-LB safety profile.

| Bracket | Threshold | Status |
|---|---|---|
| Above Baseline 3 | LB < 0.8056 | **cleared (0.7942)** |
| Baselines 2–3 | 0.8056 ≤ LB < 0.8623 | — |
| Baselines 1–2 | 0.8623 ≤ LB < 0.9117 | — |
| Below Baseline 1 | LB ≥ 0.9117 | — |

---

## Data

- **Train:** 12.3M rows (2,248 regions × 5,480 days), 14 weather features + weekly `score`
- **Test:** 204k rows (2,248 regions × 91 days), no `score` column
- **Submission:** 2,248 rows × 6 columns (`region_id`, `pred_week1..5`)
- **Score distribution:** ~60% zeros, mean 0.836, median 0 — heavily zero-inflated
- Scores are recorded weekly (`day_idx % 7 == 6`); no missing weather values
- Dates are anonymized to far-future years and handled via per-region day indices

The 14 weather features are `prec`, `surf_pre`, `humidity`, `tmp`, `dp_tmp`, `wb_tmp`,
`tmp_max`, `tmp_min`, `tmp_range`, `surf_tmp`, `wind`, `wind_max`, `wind_min`,
`wind_range` — and they are strongly inter-correlated (temperature group, wind group,
humidity ↔ temperature). That correlation structure turns out to drive several of the
key findings below.

---

## Approach

The final model is a hand-weighted ensemble of three models from different
architecture families:

1. **V5 — LightGBM.** 105 engineered features including climatology anomalies and
   score history. Best tree-based model; standalone LB 0.9017.
2. **V7 — Temporal Convolutional Network.** 5 dilated residual blocks × 64 channels,
   averaged over 10 random seeds. The strongest single model.
3. **V10 — TCN + static features.** A TCN backbone fused with V5's static features,
   averaged over 5 seeds. Used as a diversifier.

All neural models share one pipeline (`data_pipeline_nn.py`): 91-day input windows,
five weekly targets at day offsets [7, 14, 21, 28, 35], and per-region z-score
normalization (RevIN-style) so the network sees deviations from each region's own
climate rather than raw weather values. Training uses AdamW + cosine LR, L1 loss
(matching the MAE metric), gradient clipping, and best-validation checkpointing.
Each model is trained from several random seeds and the raw test predictions are
averaged, then clipped to [0, 5].

Ensemble weights are tuned against leaderboard feedback rather than the validation
set — see Finding 7 for why.

---

## Model Catalog

### Tabular / lookup-based

| Model | Description | Val MAE | LB |
|---|---|---|---|
| V1 | Per-region monthly mean lookup | 0.79 | 0.9506 |
| V1.5 | Per-region monthly median lookup | — | 1.0271 |
| V2 | LightGBM, basic aggregates (58 feat.) | 0.524* | 0.9435 |
| V3 | LightGBM, engineered features (90 feat.) | 0.4634 | 0.9168 |
| **V5** | V3 + climatology anomalies + score history (105 feat.) | 0.4556 | **0.9017** |
| V6 | V5 + recent-drought features (117 feat.) | 0.4593 | not submitted |

\* V2 validation was leaky and not comparable.

### Sequence / neural architectures

| Model | Description | Val MAE | LB |
|---|---|---|---|
| V7 | TCN (5 dilated blocks × 64 ch) | 0.4322 | 0.8463 |
| **V7-5seed** | TCN averaged over seeds 42–46 | ~0.44 | **0.8071** |
| **V7-10seed** | TCN averaged over seeds 42–51 | ~0.43 | in best ensemble |
| V8 / V8.1 | P-sLSTM, single- and multi-head | 0.71 / 0.69 | failed |
| V9 | iTransformer | 0.6791 | failed |
| V10 | TCN + V5 static features fused | 0.4498 | — |
| **V10-5seed** | V10 averaged over seeds 42–46 | ~0.456 | **0.8159** |
| DLinear | Linear trend/seasonal decomposition | 0.6427 | not pursued |

### Leaderboard progression

| Milestone | LB | Note |
|---|---|---|
| V1 monthly mean | 0.9506 | first baseline |
| V5 LightGBM | 0.9017 | best tree model |
| V7 TCN | 0.8463 | first sequence model — biggest single jump |
| V7-5seed | 0.8071 | seed averaging cut 0.039 |
| 3-way (V5 + V7-5seed + V10-5seed) | 0.7971 | first submission above Baseline 3 |
| 3-way with V7-10seed | 0.7948 | 10-seed V7 cut a further 0.0023 |
| **3-way, tuned weights (0.10/0.50/0.40)** | **0.7942** | current best |


---

## Failed Experiments

Each entry is a hypothesis that did not work out, kept here because the negative
result is itself informative.

| # | Experiment | What happened | Lesson |
|---|---|---|---|
| 1 | V1.5 monthly median | LB 1.03 — over-confident on chronic-drought regions | MAE-optimal median applies to single constants, not conditional predictions on small groups |
| 2 | V4 in-window score features | Val 0.20, LB 1.20 | Features must be computable identically at train and inference time; `test.csv` has no score column |
| 3 | V3 quantile calibration | LB 0.917 → 0.967 | Distribution mismatch was a symptom; ranking quality was the real bottleneck |
| 4 | V8/V9 sequence architectures | All underperformed TCN by 0.25+ val MAE | Channel-independent architectures fail on correlated features; a 91-day window is too short for attention to beat dilated convolution |
| 5 | V7-large / V7.1 | More capacity *and* more regularization both hurt | V7's 64ch × 5 blocks is a structural sweet spot |
| 6 | V6 feature stacking + V3/V5/V6 ensemble | Correlation 0.978; ensemble gained nothing | Ensemble diversity needs different inductive biases, not more features |
| 7 | V10 single-seed | Converged in 2 epochs; static features dominated | The TCN backbone contributed little extra signal without seed averaging |
| 8 | Ridge / tilt stacking | Val-fit weights lost 0.002 LB to hand-tuned weights | Validation/LB inversion makes any val-fit weight unreliable |
| 9 | V10-10seed | 5 → 10 seeds *hurt* the ensemble by 0.0017 | The diversifier needs its seed variance; asymmetric seed counts are correct |
| 10 | DLinear | Phase-1 sanity check: val 0.643, prediction means decaying to 0.75 | A pre-registered stop criterion (means < 0.90 → abort) closed the path before a full run; the same bias as V8.1 |

---

## Model Correlation

Pairwise prediction correlation, averaged across the five weekly horizons:

| Pair | Correlation | Interpretation |
|---|---|---|
| V5 ↔ V7 | 0.77 | moderate — useful cross-architecture diversity |
| V5 ↔ V10-5seed | 0.78 | V5 remains the most distinct available model |
| V7-5seed ↔ V10-5seed | 0.92 | very high — V10 is largely a static-augmented V7 |
| V8 ↔ V7-5seed | 0.47 | genuinely different, but V8 is mostly noise (val 0.71) |
| V9 ↔ V7-5seed | 0.19 | broken — V9 predicts ~0.6, this is bias not signal |

Seed averaging strips seed-specific noise, leaving true model agreement — which is
high among similar architectures. V8 and V9 reach genuinely different prediction
regions, but for the wrong reasons (noise and bias), not because they encode useful
complementary structure.

---

## Repository Structure

```
.
├── data_pipeline_nn.py              # shared data prep for all neural models
├── baseline_v1_monthly_mean.py      # V1   — monthly-mean lookup
├── baseline_v2_lgbm_minimal.py      # V2   — minimal LightGBM
├── baseline_v3_lgbm_features.py     # V3   — engineered-feature LightGBM
├── baseline_v5_lgbm_score_history.py# V5   — best tree model
├── baseline_v6_lgbm_recent_drought.py# V6
├── baseline_v7_tcn.py               # V7   — TCN
├── baseline_v7_tcn_5seed_avg.py     # V7   — multi-seed TCN trainer
├── baseline_v8_pslstm.py            # V8   — P-sLSTM
├── baseline_v9_itransformer.py      # V9   — iTransformer
├── baseline_v10_tcn_static.py       # V10  — TCN + static features
├── baseline_v10_tcn_static_5seed_avg.py# V10  — multi-seed trainer
├── baseline_dlinear_5seed_avg.py    # DLinear — multi-seed trainer
├── ensemble/                        # ensemble blending scripts
├── FinalProject.ipynb               # For experiment reproducing
└── README.md
```

The multi-seed trainers (`*_5seed_avg.py`) take a configurable
comma-separated seed list, so the same script produces a single-seed sanity check, a
5-seed average, or a 10-seed average.

---

## Reproducing

All neural models share the data pipeline and a common training interface. Example —
training the 10-seed TCN that anchors the best ensemble:

```bash
python baseline_v7_tcn_5seed_avg.py \
    --base /path/to/data-mining-2026-final-project \
    --seeds 42,43,44,45,46,47,48,49,50,51 \
    --out baseline_v7_tcn_10seed_avg.csv
```

A single-seed sanity check before committing to a full run:

```bash
python baseline_dlinear_5seed_avg.py --seeds 42 --out dlinear_seed42.csv
```

Ensemble blends are produced by the `ensemble_*.py` scripts, which read the per-model
submission CSVs and combine them at the chosen weights.

---

## References

The architectures explored draw on the following work; full bibliographic details are
in the project report.

- **TCN** — Bai et al., *An Empirical Evaluation of Generic Convolutional and Recurrent
  Networks for Sequence Modeling*, 2018.
- **RevIN** — Kim et al., *Reversible Instance Normalization for Accurate Time-Series
  Forecasting against Distribution Shift*, ICLR 2022.
- **DLinear** — Zeng et al., *Are Transformers Effective for Time Series Forecasting?*,
  AAAI 2023.
- **iTransformer** — Liu et al., *iTransformer: Inverted Transformers Are Effective for
  Time Series Forecasting*, ICLR 2024.
- **P-sLSTM** — Kong et al., *Unlocking the Power of LSTM for Long-Term Time Series
  Forecasting*, 2025.
