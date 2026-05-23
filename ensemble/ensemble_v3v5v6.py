"""
Ensemble — Average predictions from V3, V5, V6 submissions.

Rationale
---------
After V6, val MAE is flat across V3/V5/V6 (~0.46), but each model emphasizes
different features:
  V3: weather aggregates + categorical region_int leans
  V5: + climatology anomalies + 3 score-history lookups
  V6: + recent-drought stats + region-month detail
Their errors should be partially decorrelated, so averaging the predictions
often gains 0.02-0.05 MAE — a robust trick on Kaggle when models are
similarly accurate but built on different features.

This script does NOT retrain anything. It just reads the three submission
CSVs that already exist, builds multiple ensemble variants, saves each as
a separate submission, and prints distribution stats for comparison.

Variants produced
-----------------
  ensemble_mean_v3v5v6.csv     Simple mean of all three
  ensemble_mean_v5v6.csv       Mean of V5 + V6 only
  ensemble_weighted_v5heavy.csv  0.5·V5 + 0.3·V6 + 0.2·V3
  ensemble_median_v3v5v6.csv   Element-wise median across all three

Usage
-----
    python ensemble_v3v5v6.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


WEEK_COLS = [f'pred_week{i}' for i in range(1, 6)]


def load_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert list(df.columns) == ['region_id'] + WEEK_COLS, \
        f'Unexpected columns in {path}: {list(df.columns)}'
    return df.sort_values('region_id').reset_index(drop=True)


def describe(arr: np.ndarray, label: str):
    print(f'  {label:30s} mean={arr.mean():.4f}  median={np.median(arr):.4f}  '
          f'std={arr.std():.4f}  <0.5={ (arr < 0.5).mean():.3f}  '
          f'>2.0={ (arr > 2.0).mean():.3f}')


def make_submission(template: pd.DataFrame, preds: np.ndarray, out_path: str):
    """Build a submission DataFrame with `preds` (N, 5) and save to disk."""
    out = template.copy()
    out[WEEK_COLS] = np.clip(preds, 0.0, 5.0)
    out.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    parser.add_argument('--v3', default='baseline_v3_lgbm_features.csv')
    parser.add_argument('--v5', default='baseline_v5_lgbm_score_history.csv')
    parser.add_argument('--v6', default='baseline_v6_lgbm_recent_drought.csv')
    args = parser.parse_args()

    print('[Loading submissions]')
    sub_v3 = load_submission(os.path.join(args.base, args.v3))
    sub_v5 = load_submission(os.path.join(args.base, args.v5))
    sub_v6 = load_submission(os.path.join(args.base, args.v6))

    # Sanity check: same region order
    assert (sub_v3['region_id'] == sub_v5['region_id']).all()
    assert (sub_v5['region_id'] == sub_v6['region_id']).all()
    print(f'  All three have {len(sub_v3):,} rows in matching order ✓')

    p_v3 = sub_v3[WEEK_COLS].values
    p_v5 = sub_v5[WEEK_COLS].values
    p_v6 = sub_v6[WEEK_COLS].values

    print('\n[Original prediction distributions]')
    describe(p_v3.flatten(), 'V3')
    describe(p_v5.flatten(), 'V5')
    describe(p_v6.flatten(), 'V6')

    # ── Variant 1: simple mean of all three
    e1 = (p_v3 + p_v5 + p_v6) / 3.0

    # ── Variant 2: mean of V5 + V6 (skip V3)
    e2 = (p_v5 + p_v6) / 2.0

    # ── Variant 3: weighted, favoring V5 (best LB so far)
    e3 = 0.5 * p_v5 + 0.3 * p_v6 + 0.2 * p_v3

    # ── Variant 4: element-wise median across all three
    stacked = np.stack([p_v3, p_v5, p_v6], axis=0)   # (3, N, 5)
    e4 = np.median(stacked, axis=0)

    print('\n[Ensemble variant distributions]')
    describe(e1.flatten(), 'mean V3+V5+V6')
    describe(e2.flatten(), 'mean V5+V6')
    describe(e3.flatten(), 'weighted (V5 heavy)')
    describe(e4.flatten(), 'median V3+V5+V6')

    print('\n[Pairwise correlations of model predictions, week-1]')
    corr_v3_v5 = np.corrcoef(p_v3[:, 0], p_v5[:, 0])[0, 1]
    corr_v3_v6 = np.corrcoef(p_v3[:, 0], p_v6[:, 0])[0, 1]
    corr_v5_v6 = np.corrcoef(p_v5[:, 0], p_v6[:, 0])[0, 1]
    print(f'  V3 vs V5: {corr_v3_v5:.4f}')
    print(f'  V3 vs V6: {corr_v3_v6:.4f}')
    print(f'  V5 vs V6: {corr_v5_v6:.4f}')
    print('  (lower correlation → more diverse → ensembling helps more)')

    print('\n[Saving variants]')
    for name, preds in [
        ('ensemble_mean_v3v5v6', e1),
        ('ensemble_mean_v5v6', e2),
        ('ensemble_weighted_v5heavy', e3),
        ('ensemble_median_v3v5v6', e4),
    ]:
        path = os.path.join(args.base, f'{name}.csv')
        make_submission(sub_v3, preds, path)
        print(f'  {name}.csv saved')

    print('\n[Recommendation]')
    print('  If V5–V6 correlation > 0.95: ensembling adds little; skip and move on.')
    print('  If V3–V5/V6 correlation < 0.90: V3 disagreement is informative; use mean_v3v5v6.')
    print('  Otherwise: mean_v5v6 is the safe pick (drops noisy V3).')


if __name__ == '__main__':
    main()
