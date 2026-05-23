"""
Baseline V3 — LightGBM with richer features and honest time-based validation.

Changes vs V2
-------------
1. **Honest validation.** V2 held out "last 10% of windows per region" but,
   with 91-day windows and stride 35, consecutive windows shared 56 of 91
   days. Validation inputs nearly duplicated training inputs, producing an
   absurdly optimistic local MAE (0.52) vs the true Kaggle MAE (0.94).
   V3 holds out the last 5 windows per region as validation, with a
   3-window BUFFER (105 days) discarded between train and val. This
   guarantees no input or target overlap.

2. **Richer features.** V2 only had mean/min/max/std over the full 91 days,
   which washes out drought-specific timing signals. V3 adds:
   - Recency aggregates: mean & sum over last 7, 14, 28 days
     (for prec, humidity, tmp, surf_pre — the drought-relevant features)
   - Drought indicators: dry-day counts, longest consecutive dry streak
   - Linear trend slopes over the 91-day window
   This brings feature count from 58 → 90.

3. Same model topology: 5 independent LightGBM regressors, one per
   week-ahead horizon, MAE objective.

Usage
-----
    python baseline_v3_lgbm_features.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v3_lgbm_features.csv
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
KEY_FEATURES = ['prec', 'humidity', 'tmp', 'surf_pre']  # drought-relevant
RECENCY_DAYS = [7, 14, 28]
DRY_THRESHOLD = 1.0   # mm — defines a "dry day"

WINDOW_DAYS = 91
TARGET_OFFSETS = [7, 14, 21, 28, 35]
STRIDE = 35

# Validation: last 5 windows per region for val, 3 windows buffer (≈105 days)
N_VAL_WINDOWS = 5
N_BUFFER_WINDOWS = 3

LGB_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l2': 1.0,
    'verbose': -1,
    'n_jobs': -1,
}
NUM_BOOST_ROUND = 1000
EARLY_STOP = 50


# ───────────────────────── Loading ─────────────────────────

def load_data(base: str):
    train = pd.read_csv(os.path.join(base, 'data', 'train.csv'))
    test = pd.read_csv(os.path.join(base, 'data', 'test.csv'))
    sub = pd.read_csv(os.path.join(base, 'sample_submission.csv'))
    train = train.sort_values(['region_id', 'date']).reset_index(drop=True)
    test = test.sort_values(['region_id', 'date']).reset_index(drop=True)
    return train, test, sub


# ───────────────────────── Feature extraction ─────────────────────────

def make_feature_names() -> List[str]:
    names: List[str] = []
    # 1. Full-window aggregates
    for stat in ['mean', 'min', 'max', 'std']:
        for col in FEATURE_COLS:
            names.append(f'{col}_{stat}')
    # 2. Recency aggregates
    for d in RECENCY_DAYS:
        for col in KEY_FEATURES:
            names.append(f'{col}_mean_{d}d')
            names.append(f'{col}_sum_{d}d')
    # 3. Drought indicators
    names += ['prec_sum_91', 'dry_days_91', 'dry_days_28', 'max_dry_streak_91']
    # 4. Trends
    for col in KEY_FEATURES:
        names.append(f'{col}_slope')
    # 5. Categoricals
    names += ['month', 'region_int']
    return names


def aggregate_window(window: np.ndarray) -> np.ndarray:
    """Build the full feature vector from a (91, 14) window."""
    feats: List[np.ndarray | float] = []

    # 1. Full-window aggregates
    feats.append(window.mean(axis=0))
    feats.append(window.min(axis=0))
    feats.append(window.max(axis=0))
    feats.append(window.std(axis=0))

    # 2. Recency aggregates
    key_idx = [FEATURE_COLS.index(c) for c in KEY_FEATURES]
    rec_block = []
    for d in RECENCY_DAYS:
        recent = window[-d:, key_idx]   # (d, 4)
        rec_block.append(recent.mean(axis=0))
        rec_block.append(recent.sum(axis=0))
    # Each block is shape (4,); we need them interleaved: mean_d, sum_d per col
    # Above gives [mean_7(4), sum_7(4), mean_14(4), sum_14(4), mean_28(4), sum_28(4)]
    # but feature names expect grouping by (d, col) with mean/sum interleaved per col.
    # Re-arrange to match make_feature_names() order: for each d, for each col, mean then sum.
    recency_arr = np.empty(len(RECENCY_DAYS) * len(KEY_FEATURES) * 2, dtype=np.float64)
    pos = 0
    for di, d in enumerate(RECENCY_DAYS):
        mean_vec = rec_block[di * 2]      # (4,)
        sum_vec = rec_block[di * 2 + 1]   # (4,)
        for ci in range(len(KEY_FEATURES)):
            recency_arr[pos] = mean_vec[ci]; pos += 1
            recency_arr[pos] = sum_vec[ci]; pos += 1
    feats.append(recency_arr)

    # 3. Drought indicators
    prec = window[:, FEATURE_COLS.index('prec')]
    is_dry = prec < DRY_THRESHOLD
    # Longest consecutive dry streak (vectorised)
    if is_dry.any():
        # Run-length encoding via diff trick
        diff = np.diff(np.concatenate(([0], is_dry.view(np.int8), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        max_streak = int((ends - starts).max()) if len(starts) else 0
    else:
        max_streak = 0
    feats.append(np.array([
        prec.sum(),
        is_dry.sum(),
        is_dry[-28:].sum(),
        max_streak,
    ], dtype=np.float64))

    # 4. Linear slope over 91 days for key features (OLS slope)
    x = np.arange(WINDOW_DAYS, dtype=np.float64)
    xc = x - x.mean()
    x_var = (xc * xc).sum()
    slopes = np.empty(len(KEY_FEATURES), dtype=np.float64)
    for i, idx in enumerate([FEATURE_COLS.index(c) for c in KEY_FEATURES]):
        col = window[:, idx]
        slopes[i] = ((col - col.mean()) * xc).sum() / x_var
    feats.append(slopes)

    return np.concatenate([np.atleast_1d(f).ravel() for f in feats]).astype(np.float32)


# ───────────────────────── Window construction ─────────────────────────

def build_train_windows(
    train: pd.DataFrame, region_to_int: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_list, month_list, region_list, target_list, win_idx_list = [], [], [], [], []

    n_regions = train['region_id'].nunique()
    print(f'  Processing {n_regions} regions...')
    t0 = time.time()

    for ri, (region_id, group) in enumerate(train.groupby('region_id', sort=False)):
        n = len(group)
        feat_arr = group[FEATURE_COLS].values.astype(np.float32)
        score_arr = group['score'].values
        date_strs = group['date'].values
        end_days = np.arange(WINDOW_DAYS - 1, n - max(TARGET_OFFSETS), STRIDE)

        for wi, e in enumerate(end_days):
            agg = aggregate_window(feat_arr[e - WINDOW_DAYS + 1: e + 1])
            month_e = int(date_strs[e][5:7])
            targets = [score_arr[e + off] for off in TARGET_OFFSETS]

            feat_list.append(agg)
            month_list.append(month_e)
            region_list.append(region_to_int[region_id])
            target_list.append(targets)
            win_idx_list.append(wi)

        if (ri + 1) % 500 == 0:
            print(f'    {ri + 1}/{n_regions} regions ({time.time() - t0:.0f}s)')

    X_feat = np.vstack(feat_list).astype(np.float32)
    months = np.array(month_list, dtype=np.float32).reshape(-1, 1)
    regions = np.array(region_list, dtype=np.float32).reshape(-1, 1)
    X = np.hstack([X_feat, months, regions])
    y = np.array(target_list, dtype=np.float32)
    win_idx = np.array(win_idx_list, dtype=np.int32)

    print(f'  Built X: {X.shape}, y: {y.shape}')
    print(f'  NaN in targets: {np.isnan(y).sum()} (should be 0)')
    return X, y, regions.flatten(), win_idx


def build_test_features(
    test: pd.DataFrame, region_to_int: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    feat_list, month_list, region_list, region_ids = [], [], [], []
    for region_id, group in test.groupby('region_id', sort=False):
        feat_arr = group[FEATURE_COLS].values.astype(np.float32)
        date_strs = group['date'].values
        agg = aggregate_window(feat_arr)
        feat_list.append(agg)
        month_list.append(int(date_strs[-1][5:7]))
        region_list.append(region_to_int[region_id])
        region_ids.append(region_id)

    X_feat = np.vstack(feat_list).astype(np.float32)
    months = np.array(month_list, dtype=np.float32).reshape(-1, 1)
    regions = np.array(region_list, dtype=np.float32).reshape(-1, 1)
    X = np.hstack([X_feat, months, regions])
    return X, np.array(region_ids)


# ───────────────────────── Train / val split ─────────────────────────

def split_with_buffer(
    X, y, regions, win_idx,
    n_val: int = N_VAL_WINDOWS,
    n_buf: int = N_BUFFER_WINDOWS,
):
    """
    Per region, in chronological order:
      - last n_val windows  → validation
      - n_buf windows before val → DROPPED (buffer to prevent input/target overlap)
      - everything earlier → training
    """
    region_to_indices: Dict[float, List[int]] = {}
    for i, r in enumerate(regions):
        region_to_indices.setdefault(r, []).append(i)

    train_mask = np.zeros(len(X), dtype=bool)
    val_mask = np.zeros(len(X), dtype=bool)

    for r, idxs in region_to_indices.items():
        # idxs already in chronological order (sequential append)
        if len(idxs) < n_val + n_buf:
            train_mask[idxs] = True   # too few windows; keep all (rare)
            continue
        train_mask[idxs[:-(n_val + n_buf)]] = True
        # idxs[-(n_val+n_buf):-n_val] are the buffer (neither train nor val)
        val_mask[idxs[-n_val:]] = True

    n_dropped = (~train_mask & ~val_mask).sum()
    print(f'  train: {train_mask.sum():,}, val: {val_mask.sum():,}, '
          f'buffer dropped: {n_dropped:,}')
    return X[train_mask], X[val_mask], y[train_mask], y[val_mask]


# ───────────────────────── Modelling ─────────────────────────

def train_models(X_tr, y_tr, X_val, y_val, feature_names):
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
            LGB_PARAMS, dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval], valid_names=['val'],
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
    parser.add_argument('--out', default='baseline_v3_lgbm_features.csv')
    args = parser.parse_args()

    print('[Loading]')
    train, test, sub = load_data(args.base)
    print(f'  train: {train.shape}, test: {test.shape}, sub: {sub.shape}')

    all_regions = sorted(set(train['region_id']) | set(test['region_id']))
    region_to_int = {r: i for i, r in enumerate(all_regions)}

    print('\n[Building training windows]')
    X, y, regions_X, win_idx = build_train_windows(train, region_to_int)

    print('\n[Train/val split with buffer]')
    X_tr, X_val, y_tr, y_val = split_with_buffer(X, y, regions_X, win_idx)

    feature_names = make_feature_names()
    assert X.shape[1] == len(feature_names), \
        f'feature count mismatch: X has {X.shape[1]}, names have {len(feature_names)}'
    print(f'  feature count: {len(feature_names)}')

    print('\n[Training 5 LightGBM models]')
    models, week_maes, val_mae = train_models(X_tr, y_tr, X_val, y_val, feature_names)

    print('\n[Building test features]')
    X_test, region_ids_test = build_test_features(test, region_to_int)

    print('\n[Predicting]')
    preds = np.column_stack([m.predict(X_test) for m in models])
    preds = np.clip(preds, 0.0, 5.0)

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
    print(f'\n  Val MAE per week: {[f"{m:.4f}" for m in week_maes]}')
    print(f'  Val MAE (mean)  : {val_mae:.4f}')
    print('\nFirst 5 rows:')
    print(submission.head())
    print('\nPrediction stats:')
    print(submission[[f'pred_week{i}' for i in range(1, 6)]].describe().round(3))

    print('\n[Top 20 features for week-1 model]')
    imp = pd.DataFrame({
        'feature': feature_names,
        'gain': models[0].feature_importance(importance_type='gain'),
    }).sort_values('gain', ascending=False).head(20)
    print(imp.to_string(index=False))


if __name__ == '__main__':
    main()
