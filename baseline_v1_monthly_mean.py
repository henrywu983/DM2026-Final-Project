"""
Baseline V1 — Per-region historical monthly mean.

Strategy
--------
For each region, compute the historical mean drought score grouped by
calendar month (using only score-bearing days). At inference time, for
each region's test window, determine the calendar month that each of the
next 5 weeks falls into and predict the corresponding region-month mean.

Fallbacks: region-month mean -> region overall mean -> global mean.

Usage
-----
    python baseline_v1_monthly_mean.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v1_monthly_mean.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd


# ───────────────────────── Date arithmetic (year-3000-safe) ─────────────────────────
# Python's pandas datetime64[ns] only supports 1677–2262, and datetime.strptime
# rejects fake leap days like 3000-02-29. We do everything manually on strings.

def is_leap(y: int) -> bool:
    """Gregorian leap-year rule. Year 3000 is NOT leap; year 2400 IS."""
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def days_in_month(y: int, m: int) -> int:
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if m in (4, 6, 9, 11):
        return 30
    return 29 if is_leap(y) else 28


def add_days(date_str: str, n: int) -> str:
    """Add n days to a 'YYYY-MM-DD' string. Handles arbitrary years."""
    y, m, d = (int(x) for x in date_str.split('-'))
    d += n
    while d > days_in_month(y, m):
        d -= days_in_month(y, m)
        m += 1
        if m > 12:
            m = 1
            y += 1
    while d < 1:
        m -= 1
        if m < 1:
            m = 12
            y -= 1
        d += days_in_month(y, m)
    return f'{y:04d}-{m:02d}-{d:02d}'


# ───────────────────────── Core baseline ─────────────────────────

def load_data(base: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(base, 'data', 'train.csv'))
    test = pd.read_csv(os.path.join(base, 'data', 'test.csv'))
    sub = pd.read_csv(os.path.join(base, 'sample_submission.csv'))
    return train, test, sub


def fit_lookup(train: pd.DataFrame) -> Tuple[pd.Series, pd.Series, float]:
    """Build region-month, region, and global lookups from training scores."""
    df = train[train['score'].notna()].copy()
    df['month'] = df['date'].str[5:7].astype(int)

    region_month_mean = df.groupby(['region_id', 'month'])['score'].mean()
    region_mean = df.groupby('region_id')['score'].mean()
    global_mean = float(df['score'].mean())

    print(f'  region-month entries : {len(region_month_mean):,}')
    print(f'  unique regions       : {region_mean.size:,}')
    print(f'  global mean score    : {global_mean:.4f}')
    return region_month_mean, region_mean, global_mean


def predict(
    test: pd.DataFrame,
    sub: pd.DataFrame,
    region_month_mean: pd.Series,
    region_mean: pd.Series,
    global_mean: float,
) -> pd.DataFrame:
    """For each region, predict 5 weekly scores after that region's last test day."""
    last_date = test.groupby('region_id')['date'].max()  # YYYY-MM-DD strings sort correctly

    rmm = region_month_mean  # alias for speed
    out_rows = []
    n_fallback_region = 0
    n_fallback_global = 0

    for region in sub['region_id']:
        last = last_date.get(region)
        if last is None:
            # Region not in test at all (shouldn't happen given EDA, but safe)
            preds = [global_mean] * 5
            n_fallback_global += 5
            out_rows.append([region, *preds])
            continue

        preds = []
        for week in range(1, 6):
            future_date = add_days(last, 7 * week)
            future_month = int(future_date[5:7])
            key = (region, future_month)

            if key in rmm.index:
                preds.append(rmm.loc[key])
            elif region in region_mean.index:
                preds.append(region_mean.loc[region])
                n_fallback_region += 1
            else:
                preds.append(global_mean)
                n_fallback_global += 1
        out_rows.append([region, *preds])

    print(f'  fallbacks (region mean): {n_fallback_region}')
    print(f'  fallbacks (global mean): {n_fallback_global}')

    return pd.DataFrame(
        out_rows,
        columns=['region_id', 'pred_week1', 'pred_week2',
                 'pred_week3', 'pred_week4', 'pred_week5'],
    )


def local_validation(train: pd.DataFrame) -> float:
    """
    Estimate MAE by holding out the last 5 weekly scores per region as a
    pseudo-test set, refitting on the remaining train, and predicting them
    with the same monthly-mean rule.
    """
    print('\n[Local validation] Holding out the last 5 score-weeks per region...')
    df = train[train['score'].notna()].copy()
    df['month'] = df['date'].str[5:7].astype(int)
    df = df.sort_values(['region_id', 'date'])

    # Last 5 score-rows per region = held-out targets
    df['rk'] = df.groupby('region_id').cumcount(ascending=False)
    holdout = df[df['rk'] < 5].copy()
    fit_df = df[df['rk'] >= 5].copy()

    rmm = fit_df.groupby(['region_id', 'month'])['score'].mean()
    rm = fit_df.groupby('region_id')['score'].mean()
    gm = float(fit_df['score'].mean())

    def lookup(r, m):
        if (r, m) in rmm.index:
            return rmm.loc[(r, m)]
        if r in rm.index:
            return rm.loc[r]
        return gm

    holdout['pred'] = [lookup(r, m) for r, m in zip(holdout['region_id'], holdout['month'])]
    mae = (holdout['score'] - holdout['pred']).abs().mean()
    print(f'  Local MAE on held-out 5-week tail: {mae:.4f}')
    return mae


# ───────────────────────── Main ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project',
        help='Project base directory (contains data/ and sample_submission.csv)',
    )
    parser.add_argument(
        '--out',
        default='baseline_v1_monthly_mean.csv',
        help='Output filename (saved inside --base)',
    )
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args()

    print('[Loading]')
    train, test, sub = load_data(args.base)
    print(f'  train: {train.shape}, test: {test.shape}, sub: {sub.shape}')

    print('\n[Fitting lookups]')
    rmm, rm, gm = fit_lookup(train)

    if not args.skip_validation:
        local_validation(train)

    print('\n[Predicting]')
    submission = predict(test, sub, rmm, rm, gm)

    # Sanity check: same regions and same order as sample_submission
    assert list(submission['region_id']) == list(sub['region_id']), \
        'region_id order does not match sample_submission'
    assert submission.shape == sub.shape, \
        f'shape mismatch: {submission.shape} vs {sub.shape}'

    out_path = os.path.join(args.base, args.out)
    submission.to_csv(out_path, index=False)
    print(f'\n[Saved] {out_path}')
    print('\nFirst 5 rows of submission:')
    print(submission.head())
    print('\nPrediction stats:')
    print(submission[['pred_week1', 'pred_week2', 'pred_week3',
                      'pred_week4', 'pred_week5']].describe().round(3))


if __name__ == '__main__':
    main()
