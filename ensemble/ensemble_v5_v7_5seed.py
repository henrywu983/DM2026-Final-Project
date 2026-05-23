"""
Ensemble V5 + V7-5seed — Strongest models, cross-architecture diversity.

Strategy
--------
V7-5seed (LB 0.8071) is our strongest single model — variance-reduced
through seed averaging. V5 (LB 0.9017) is our strongest tabular model and
has 0.73 correlation with V7, meaning it captures complementary signal.

V5 + V7 (single seed) ensemble already gained 0.011 on LB. Replacing the
V7 component with the better V7-5seed should preserve or amplify that gain.

Variants produced
-----------------
  ensemble_v5_v7_5seed_mean.csv         0.5·V5 + 0.5·V7_5seed
  ensemble_v5_v7_5seed_v7heavy.csv      0.4·V5 + 0.6·V7_5seed
  ensemble_v5_v7_5seed_v7stronger.csv   0.3·V5 + 0.7·V7_5seed

Default pick: v7stronger — V7-5seed is the stronger model (LB 0.81 vs 0.90),
so weighting it more should win. The other variants are hedges.

Usage
-----
    python ensemble_v5_v7_5seed.py \
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
    args = parser.parse_args()

    print('[Loading submissions]')
    sub_v5 = load_submission(os.path.join(args.base, args.v5))
    sub_v7 = load_submission(os.path.join(args.base, args.v7_5seed))
    assert (sub_v5['region_id'] == sub_v7['region_id']).all()
    print(f'  Both have {len(sub_v5):,} rows in matching order ✓')

    p_v5 = sub_v5[WEEK_COLS].values
    p_v7 = sub_v7[WEEK_COLS].values

    print('\n[Original prediction distributions]')
    describe(p_v5.flatten(), 'V5 (LB 0.9017)')
    describe(p_v7.flatten(), 'V7-5seed (LB 0.8071)')

    print('\n[Pairwise correlation by week]')
    for w in range(5):
        c = np.corrcoef(p_v5[:, w], p_v7[:, w])[0, 1]
        print(f'  week {w+1}: {c:.4f}')

    variants = {
        'ensemble_v5_v7_5seed_mean':       0.5 * p_v5 + 0.5 * p_v7,
        'ensemble_v5_v7_5seed_v7heavy':    0.4 * p_v5 + 0.6 * p_v7,
        'ensemble_v5_v7_5seed_v7stronger': 0.3 * p_v5 + 0.7 * p_v7,
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
    print('  V7-5seed (LB 0.81) is much stronger than V5 (LB 0.90), so')
    print('  weighting V7 more heavily should win. Default: v7stronger (0.3/0.7).')
    print('  If correlation across weeks is very low (<0.7), even the mean variant')
    print('  may surprise.')


if __name__ == '__main__':
    main()
