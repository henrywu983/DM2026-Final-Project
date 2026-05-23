"""
Ensemble V5 + V7-5seed + V10 — Three-way architecturally diverse blend.

Strategy
--------
Three models with different inductive biases:
  V5       : LightGBM on 105 tabular features (LB 0.9017)
  V7-5seed : Pure TCN, 5-seed averaged (LB 0.8071)
  V10      : TCN with V5 static features fused inside the model

V5 and V7-5seed had pairwise correlation 0.77 across weeks. V10 represents
a third decision boundary — a single-seed TCN that internally combines
sequence and static signals, which is structurally distinct from both
pure-tabular V5 and pure-sequence V7-5seed.

The script computes pairwise correlations across all three models to
gauge effective diversity, then produces multiple weighted variants. The
best one depends on the actual correlation pattern.

Variants produced
-----------------
  ensemble_3way_mean.csv          1/3 each
  ensemble_3way_v7heavy.csv       0.2 V5 + 0.5 V7-5seed + 0.3 V10
  ensemble_3way_v7stronger.csv    0.15 V5 + 0.55 V7-5seed + 0.30 V10
  ensemble_3way_nov5.csv          0 V5 + 0.6 V7-5seed + 0.4 V10
  ensemble_3way_evenseq.csv       0.2 V5 + 0.4 V7-5seed + 0.4 V10

Decision rule for picking which to submit
-----------------------------------------
The cleanest signal will be the pairwise correlations:
  - If V10 corr with V7-5seed is < 0.85 (i.e., genuinely diverse) →
    nov5 or evenseq variants give V10 real weight and may surprise
  - If V10 corr with V7-5seed is > 0.92 (mostly redundant with V7) →
    v7stronger gives V10 less weight; close to v5_v7_5seed_v7stronger
  - Otherwise → mean variant is safest

Usage
-----
    python ensemble_v5_v7_5seed_v10.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --v7-5seed YOUR_5SEED_FILENAME.csv
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
    print(f'  {label:35s} mean={arr.mean():.4f}  median={np.median(arr):.4f}  '
          f'std={arr.std():.4f}  <0.5={(arr < 0.5).mean():.3f}  '
          f'>2.0={(arr > 2.0).mean():.3f}')


def make_submission(template: pd.DataFrame, preds: np.ndarray, out_path: str):
    out = template.copy()
    out[WEEK_COLS] = np.clip(preds, 0.0, 5.0)
    out.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    parser.add_argument('--v5', default='baseline_v5_lgbm_score_history.csv')
    parser.add_argument('--v7-5seed', default='baseline_v7_5seed_avg.csv',
                        help='Filename of the V7 5-seed averaged submission')
    parser.add_argument('--v10', default='baseline_v10_tcn_static.csv')
    args = parser.parse_args()

    print('[Loading submissions]')
    sub_v5 = load_submission(os.path.join(args.base, args.v5))
    sub_v7 = load_submission(os.path.join(args.base, args.v7_5seed))
    sub_v10 = load_submission(os.path.join(args.base, args.v10))
    assert (sub_v5['region_id'] == sub_v7['region_id']).all()
    assert (sub_v5['region_id'] == sub_v10['region_id']).all()
    print(f'  All three have {len(sub_v5):,} rows in matching order ✓')

    p_v5 = sub_v5[WEEK_COLS].values
    p_v7 = sub_v7[WEEK_COLS].values
    p_v10 = sub_v10[WEEK_COLS].values

    print('\n[Original prediction distributions]')
    describe(p_v5.flatten(), 'V5         (LB 0.9017)')
    describe(p_v7.flatten(), 'V7-5seed   (LB 0.8071)')
    describe(p_v10.flatten(), 'V10        (val 0.4498)')

    print('\n[Pairwise correlation by week]')
    print('  week  V5↔V7  V5↔V10  V7↔V10')
    for w in range(5):
        c_v5_v7 = np.corrcoef(p_v5[:, w], p_v7[:, w])[0, 1]
        c_v5_v10 = np.corrcoef(p_v5[:, w], p_v10[:, w])[0, 1]
        c_v7_v10 = np.corrcoef(p_v7[:, w], p_v10[:, w])[0, 1]
        print(f'   {w+1}    {c_v5_v7:.4f}  {c_v5_v10:.4f}  {c_v7_v10:.4f}')

    variants = {
        'ensemble_3way_mean':          (1/3) * p_v5 + (1/3) * p_v7 + (1/3) * p_v10,
        'ensemble_3way_v7heavy':       0.20 * p_v5 + 0.50 * p_v7 + 0.30 * p_v10,
        'ensemble_3way_v7stronger':    0.15 * p_v5 + 0.55 * p_v7 + 0.30 * p_v10,
        'ensemble_3way_nov5':          0.00 * p_v5 + 0.60 * p_v7 + 0.40 * p_v10,
        'ensemble_3way_evenseq':       0.20 * p_v5 + 0.40 * p_v7 + 0.40 * p_v10,
    }

    print('\n[Ensemble variant distributions]')
    for name, preds in variants.items():
        describe(preds.flatten(), name)

    print('\n[Saving variants]')
    for name, preds in variants.items():
        path = os.path.join(args.base, f'{name}.csv')
        make_submission(sub_v5, preds, path)
        print(f'  {name}.csv saved')

    print('\n[Recommendation]')
    print('  Look at V7↔V10 correlation:')
    print('    < 0.85: V10 is genuinely diverse → try ensemble_3way_evenseq')
    print('    0.85-0.92: moderate diversity → ensemble_3way_v7stronger is safest')
    print('    > 0.92: V10 mostly redundant → ensemble_3way_v7stronger or v7heavy')
    print('  V7-5seed alone (LB 0.8071) is already very close to Baseline 3 (0.8056).')
    print('  Best expected variant: one that weights V7-5seed at ~50-60%.')


if __name__ == '__main__':
    main()
