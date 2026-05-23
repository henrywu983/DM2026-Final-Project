"""
Ensemble V5 + V7 — Tree-based + sequence-based blend.

Why this should work
--------------------
The V3+V5+V6 ensemble barely moved the LB (0.9017 → 0.9035) because all three
were tree-based on overlapping tabular features (V5↔V6 correlation 0.978).
V7 (TCN) operates on raw 91-day sequences with a completely different
inductive bias, so its predictions should disagree with V5 in ways that
improve when averaged.

This script tries multiple blend ratios so we can pick the best one for
submission. Equal-weight averaging is the safe default; weighted variants
are included for completeness.

Variants produced
-----------------
  ensemble_v5v7_mean.csv          0.5·V5 + 0.5·V7
  ensemble_v5v7_v7heavy.csv       0.4·V5 + 0.6·V7  (favor stronger model)
  ensemble_v5v7_v5heavy.csv       0.6·V5 + 0.4·V7  (favor tabular signal)
  ensemble_v5v7_v7stronger.csv    0.3·V5 + 0.7·V7  (TCN dominant)

Usage
-----
    python ensemble_v5_v7.py \
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
    assert list(df.columns) == ['region_id'] + WEEK_COLS
    return df.sort_values('region_id').reset_index(drop=True)


def describe(arr: np.ndarray, label: str):
    print(f'  {label:30s} mean={arr.mean():.4f}  median={np.median(arr):.4f}  '
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
    parser.add_argument('--v7', default='baseline_v7_tcn.csv')
    args = parser.parse_args()

    print('[Loading submissions]')
    sub_v5 = load_submission(os.path.join(args.base, args.v5))
    sub_v7 = load_submission(os.path.join(args.base, args.v7))
    assert (sub_v5['region_id'] == sub_v7['region_id']).all()
    print(f'  Both have {len(sub_v5):,} rows in matching order ✓')

    p_v5 = sub_v5[WEEK_COLS].values
    p_v7 = sub_v7[WEEK_COLS].values

    print('\n[Original prediction distributions]')
    describe(p_v5.flatten(), 'V5 (LightGBM, LB 0.9017)')
    describe(p_v7.flatten(), 'V7 (TCN,      LB 0.8463)')

    print('\n[Pairwise correlation by week]')
    for w in range(5):
        c = np.corrcoef(p_v5[:, w], p_v7[:, w])[0, 1]
        print(f'  week {w+1}: {c:.4f}')

    # Variants
    variants = {
        'ensemble_v5v7_mean': 0.5 * p_v5 + 0.5 * p_v7,
        'ensemble_v5v7_v7heavy': 0.4 * p_v5 + 0.6 * p_v7,
        'ensemble_v5v7_v5heavy': 0.6 * p_v5 + 0.4 * p_v7,
        'ensemble_v5v7_v7stronger': 0.3 * p_v5 + 0.7 * p_v7,
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
    print('  Default pick: ensemble_v5v7_mean (equal weighting, most robust).')
    print('  If V7 correlation with V5 is < 0.85: any of the V7-heavy variants')
    print('  may outperform — but with only 3 submissions tomorrow, start with')
    print('  the mean and only escalate if it works.')


if __name__ == '__main__':
    main()
