"""
Baseline V2 — Minimal LightGBM with 91-day weather aggregates.

Strategy
--------
Train a global gradient-boosted regression model on sliding 91-day windows
extracted from the training data. For each window, build features from
basic aggregates (mean/min/max/std) of all 14 weather columns over the
window, plus the window-end month and region ID. Train 5 independent
LightGBM models, one per week-ahead horizon (week 1 to week 5).

Why this should beat V1/V1.5
----------------------------
The seasonal-mean baselines ignored test-time weather, throwing away the
year-specific drought signal. This model uses 91 days of test weather
features, so it can detect "this region is currently in a precipitation
deficit" or "current temperature anomaly is high" — the actual drivers
of near-term drought severity.

Usage
-----
    pip install lightgbm  # usually preinstalled in Colab
    python baseline_v2_lgbm_minimal.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v2_lgbm_minimal.csv
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb


# ───────────────────────── Config ─────────────────────────

FEATURE_COLS: List[str] = [
    'prec', 'surf_pre', 'humidity', 'tmp', 'dp_tmp', 'wb_tmp',
    'tmp_max', 'tmp_min', 'tmp_range', 'surf_tmp',
    'wind', 'wind_max', 'wind_min', 'wind_range',
]
WINDOW_DAYS = 91
TARGET_OFFSETS = [7, 14, 21, 28, 35]   # day offsets after window-end for week 1..5
STRIDE = 35                             # window stride (non-overlapping targets)
AGG_FUNCS = ['mean', 'min', 'max', 'std']

LGB_PARAMS = {
    'objective': 'regression_l1',  # MAE — matches Kaggle metric
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'lambda_l2': 1.0,
    'verbose': -1,
    'n_jobs': -1,
}
NUM_BOOST_ROUND = 800
EARLY_STOP = 40


# ───────────────────────── Loading ─────────────────────────

def load_data(base: str):
    train = pd.read_csv(os.path.join(base, 'data', 'train.csv'))
    test = pd.read_csv(os.path.join(base, 'data', 'test.csv'))
    sub = pd.read_csv(os.path.join(base, 'sample_submission.csv'))

    # Date strings YYYY-MM-DD sort lexicographically == chronologically.
    train = train.sort_values(['region_id', 'date']).reset_index(drop=True)
    test = test.sort_values(['region_id', 'date']).reset_index(drop=True)
    return train, test, sub


def make_feature_names() -> List[str]:
    names = []
    for stat in AGG_FUNCS:
        for col in FEATURE_COLS:
            names.append(f'{col}_{stat}')
    names += ['month', 'region_int']
    return names


# ───────────────────────── Window construction ─────────────────────────

def aggregate_window(window: np.ndarray) -> np.ndarray:
    """Aggregate a (91, 14) window into a (56,) feature vector."""
    return np.concatenate([
        window.mean(axis=0),
        window.min(axis=0),
        window.max(axis=0),
        window.std(axis=0),
    ])


def build_train_windows(
    train: pd.DataFrame, region_to_int: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide windows across each region's 5480 days; return (X, y, region_int)."""
    feat_list, month_list, region_list, target_list = [], [], [], []

    n_regions = train['region_id'].nunique()
    print(f'  Processing {n_regions} regions...')
    t0 = time.time()

    for ri, (region_id, group) in enumerate(train.groupby('region_id', sort=False)):
        n = len(group)
        if n != 5480:
            raise ValueError(f'Region {region_id} has {n} days, expected 5480')

        feat_arr = group[FEATURE_COLS].values.astype(np.float32)
        score_arr = group['score'].values
        date_strs = group['date'].values

        # Window end day_idx: 90, 125, 160, ..., last valid (where e + 35 < 5480)
        end_days = np.arange(WINDOW_DAYS - 1, n - max(TARGET_OFFSETS), STRIDE)

        for e in end_days:
            window = feat_arr[e - WINDOW_DAYS + 1 : e + 1]   # (91, 14)
            agg = aggregate_window(window)
            month_e = int(date_strs[e][5:7])
            targets = [score_arr[e + off] for off in TARGET_OFFSETS]

            feat_list.append(agg)
            month_list.append(month_e)
            region_list.append(region_to_int[region_id])
            target_list.append(targets)

        if (ri + 1) % 500 == 0:
            print(f'    {ri + 1}/{n_regions} regions ({time.time() - t0:.0f}s)')

    X_feat = np.vstack(feat_list).astype(np.float32)
    months = np.array(month_list, dtype=np.float32).reshape(-1, 1)
    regions = np.array(region_list, dtype=np.float32).reshape(-1, 1)
    X = np.hstack([X_feat, months, regions])
    y = np.array(target_list, dtype=np.float32)

    print(f'  Built X: {X.shape}, y: {y.shape}')
    print(f'  NaN in targets: {np.isnan(y).sum()}  (should be 0)')
    return X, y, regions.flatten()


def build_test_features(
    test: pd.DataFrame, region_to_int: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """One feature vector per test region (91 days each)."""
    feat_list, month_list, region_list, region_ids = [], [], [], []

    for region_id, group in test.groupby('region_id', sort=False):
        if len(group) != 91:
            raise ValueError(f'Test region {region_id} has {len(group)} days, expected 91')

        feat_arr = group[FEATURE_COLS].values.astype(np.float32)
        date_strs = group['date'].values
        agg = aggregate_window(feat_arr)
        month_last = int(date_strs[-1][5:7])

        feat_list.append(agg)
        month_list.append(month_last)
        region_list.append(region_to_int[region_id])
        region_ids.append(region_id)

    X_feat = np.vstack(feat_list).astype(np.float32)
    months = np.array(month_list, dtype=np.float32).reshape(-1, 1)
    regions = np.array(region_list, dtype=np.float32).reshape(-1, 1)
    X = np.hstack([X_feat, months, regions])
    return X, np.array(region_ids)


# ───────────────────────── Train / val split ─────────────────────────

def time_based_split(X, y, regions, val_frac=0.1):
    """Hold out the last `val_frac` of windows per region (chronological)."""
    val_mask = np.zeros(len(X), dtype=bool)
    region_to_indices: Dict[float, List[int]] = {}
    for i, r in enumerate(regions):
        region_to_indices.setdefault(r, []).append(i)
    for r, idxs in region_to_indices.items():
        n_val = max(1, int(len(idxs) * val_frac))
        val_mask[idxs[-n_val:]] = True
    return X[~val_mask], X[val_mask], y[~val_mask], y[val_mask]


# ───────────────────────── Modelling ─────────────────────────

def train_models(X_tr, y_tr, X_val, y_val, feature_names):
    """Train 5 LightGBM models (one per week-ahead horizon)."""
    cat_idx = [feature_names.index('region_int'), feature_names.index('month')]
    models = []
    week_maes = []

    for week in range(5):
        print(f'\n  ── Week {week + 1} ──')
        dtrain = lgb.Dataset(X_tr, label=y_tr[:, week], feature_name=feature_names,
                             categorical_feature=cat_idx)
        dval = lgb.Dataset(X_val, label=y_val[:, week], reference=dtrain,
                           feature_name=feature_names, categorical_feature=cat_idx)

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            valid_names=['val'],
            callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(100)],
        )
        models.append(model)

        mae = float(np.abs(model.predict(X_val) - y_val[:, week]).mean())
        week_maes.append(mae)
        print(f'  Week {week + 1} val MAE: {mae:.4f}')

    overall = float(np.mean(week_maes))
    print(f'\n  Overall val MAE (mean across weeks): {overall:.4f}')
    return models, week_maes, overall


# ───────────────────────── Main ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project',
    )
    parser.add_argument('--out', default='baseline_v2_lgbm_minimal.csv')
    args = parser.parse_args()

    print('[Loading]')
    train, test, sub = load_data(args.base)
    print(f'  train: {train.shape}, test: {test.shape}, sub: {sub.shape}')

    # Region encoding (consistent across train and test)
    all_regions = sorted(set(train['region_id']) | set(test['region_id']))
    region_to_int = {r: i for i, r in enumerate(all_regions)}
    print(f'  unique regions: {len(region_to_int)}')

    print('\n[Building training windows]')
    X, y, regions_X = build_train_windows(train, region_to_int)

    print('\n[Train/val split]')
    X_tr, X_val, y_tr, y_val = time_based_split(X, y, regions_X, val_frac=0.1)
    print(f'  train: {X_tr.shape}, val: {X_val.shape}')

    feature_names = make_feature_names()

    print('\n[Training 5 LightGBM models]')
    models, week_maes, val_mae = train_models(X_tr, y_tr, X_val, y_val, feature_names)

    print('\n[Building test features]')
    X_test, region_ids_test = build_test_features(test, region_to_int)
    print(f'  test feature matrix: {X_test.shape}')

    print('\n[Predicting]')
    preds = np.column_stack([m.predict(X_test) for m in models])
    preds = np.clip(preds, 0.0, 5.0)   # scores live in [0, 5]

    region_to_pred = {r: p for r, p in zip(region_ids_test, preds)}
    rows = [[r, *region_to_pred[r]] for r in sub['region_id']]
    submission = pd.DataFrame(
        rows,
        columns=['region_id', 'pred_week1', 'pred_week2',
                 'pred_week3', 'pred_week4', 'pred_week5'],
    )

    out_path = os.path.join(args.base, args.out)
    submission.to_csv(out_path, index=False)
    print(f'\n[Saved] {out_path}')
    print(f'\n  Local val MAEs per week: '
          f'{[f"{m:.4f}" for m in week_maes]}')
    print(f'  Local val MAE (mean)   : {val_mae:.4f}')
    print('\nFirst 5 rows of submission:')
    print(submission.head())
    print('\nPrediction stats:')
    print(submission[[f'pred_week{i}' for i in range(1, 6)]].describe().round(3))

    # Feature importance from week-1 model (quick sanity check)
    print('\n[Top 15 features for week-1 model by gain]')
    imp = pd.DataFrame({
        'feature': feature_names,
        'gain': models[0].feature_importance(importance_type='gain'),
    }).sort_values('gain', ascending=False).head(15)
    print(imp.to_string(index=False))


if __name__ == '__main__':
    main()
