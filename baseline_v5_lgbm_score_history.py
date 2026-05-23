"""
Baseline V5 — Fix V4's train/test feature-availability mismatch.

Diagnosis from V4
-----------------
V4 reached val MAE 0.20 but Kaggle 1.20 — much worse than V3 (0.92). Cause:
the in-window score features (score_last, score_mean, score_max, score_slope,
score_nonzero_count) were the dominant features during training, but
test.csv contains NO score column. At inference time these features were
all set to 0 for every region, and the model — which had learned to weight
score_last heavily — predicted near-0 drought severity everywhere.

This is a classic train/test feature-availability mismatch. The score
column simply does not exist in the test set, so any feature that references
it during inference is unusable.

Fix in V5
---------
Drop the 5 in-window score features entirely. Replace them with three
features sourced exclusively from train.csv historical records — these
are computed once per region and do NOT reference any test-period scores:

  region_score_mean_all     : historical mean score over the entire training
                              record for that region
  region_score_mean_lastyear: mean score over the LAST year of training data
                              for that region (most recent climatology)
  region_month_score_mean   : mean score for that region in the calendar
                              month of the window's end day

These features encode "this region's drought baseline" without leaking any
future information. They are available identically at training and
inference time.

Everything else from V4 retained
--------------------------------
- Full-window aggregates, recency aggregates, drought indicators, trend
  slopes (90 features from V3)
- Climatology anomalies for 6 weather features (12 features from V4)
- Buffered time-based validation
- 5 LightGBM models, one per week-ahead horizon, MAE objective

Total feature count: 90 + 12 + 3 + 2 (categoricals) = 107

Usage
-----
    python baseline_v5_lgbm_score_history.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v5_lgbm_score_history.csv
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
KEY_FEATURES = ['prec', 'humidity', 'tmp', 'surf_pre']
CLIMATOLOGY_FEATURES = ['prec', 'humidity', 'tmp', 'surf_pre', 'tmp_max', 'tmp_min']
RECENCY_DAYS = [7, 14, 28]
DRY_THRESHOLD = 1.0

WINDOW_DAYS = 91
TARGET_OFFSETS = [7, 14, 21, 28, 35]
STRIDE = 35
N_VAL_WINDOWS = 5
N_BUFFER_WINDOWS = 3
LASTYEAR_DAYS = 365

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
NUM_BOOST_ROUND = 1500
EARLY_STOP = 60


# ───────────────────────── Loading ─────────────────────────

def load_data(base: str):
    train = pd.read_csv(os.path.join(base, 'data', 'train.csv'))
    test = pd.read_csv(os.path.join(base, 'data', 'test.csv'))
    sub = pd.read_csv(os.path.join(base, 'sample_submission.csv'))
    train = train.sort_values(['region_id', 'date']).reset_index(drop=True)
    test = test.sort_values(['region_id', 'date']).reset_index(drop=True)
    return train, test, sub


# ───────────────────────── Lookup tables (from train only) ─────────────────────────

def build_climatology(train: pd.DataFrame):
    print('  Building climatology lookup tables (weather)...')
    df = train.copy()
    df['month'] = df['date'].str[5:7].astype(int)
    grouped = df.groupby(['region_id', 'month'])[CLIMATOLOGY_FEATURES]
    clim_mean = grouped.mean()
    clim_std = grouped.std().fillna(1e-6)
    clim_std = clim_std.where(clim_std > 1e-6, 1e-6)
    print(f'  weather climatology shape: {clim_mean.shape}')
    return clim_mean, clim_std


def build_score_lookups(train: pd.DataFrame):
    """
    Build per-region score statistics from train.csv, used for both
    training and test feature extraction. These do NOT reference any
    test-period scores, so they are leakage-free.
    """
    print('  Building score-history lookup tables...')
    df = train[train['score'].notna()].copy()
    df['month'] = df['date'].str[5:7].astype(int)

    # 1. Region-wide mean score (overall historical average)
    region_score_mean = df.groupby('region_id')['score'].mean()

    # 2. Region (region, month) mean — drought climatology
    region_month_score_mean = df.groupby(['region_id', 'month'])['score'].mean()

    # 3. Mean over last 365 days of train per region — most-recent baseline
    region_lastyear_score_mean = (
        df.sort_values(['region_id', 'date'])
        .groupby('region_id')
        .tail(LASTYEAR_DAYS // 7 + 5)  # ~52 score-days in a year + buffer
        .groupby('region_id')['score']
        .mean()
    )

    print(f'  region_score_mean : {len(region_score_mean):,} entries')
    print(f'  region_month_mean : {len(region_month_score_mean):,} entries')
    print(f'  region_lastyr_mean: {len(region_lastyear_score_mean):,} entries')
    print(f'  global score mean : {df["score"].mean():.4f}')
    return region_score_mean, region_month_score_mean, region_lastyear_score_mean


# ───────────────────────── Feature names ─────────────────────────

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
    # 5. Climatology anomalies (6 features × 2 stats)
    for col in CLIMATOLOGY_FEATURES:
        names.append(f'{col}_anom_z')
        names.append(f'{col}_anom_diff')
    # 6. NEW: score history (3 features sourced from train.csv only)
    names += ['region_score_mean_all',
              'region_month_score_mean',
              'region_lastyear_score_mean']
    # 7. Categoricals
    names += ['month', 'region_int']
    return names


# ───────────────────────── Per-window feature extraction ─────────────────────────

def aggregate_window(
    window_feats: np.ndarray,
    end_month: int,
    region_id: str,
    clim_mean: pd.DataFrame,
    clim_std: pd.DataFrame,
    region_score_mean: pd.Series,
    region_month_score_mean: pd.Series,
    region_lastyear_score_mean: pd.Series,
    global_score_mean: float,
) -> np.ndarray:
    feats: List[np.ndarray | float] = []

    # 1. Full-window aggregates
    feats.append(window_feats.mean(axis=0))
    feats.append(window_feats.min(axis=0))
    feats.append(window_feats.max(axis=0))
    feats.append(window_feats.std(axis=0))

    # 2. Recency aggregates
    key_idx = [FEATURE_COLS.index(c) for c in KEY_FEATURES]
    recency_arr = np.empty(len(RECENCY_DAYS) * len(KEY_FEATURES) * 2, dtype=np.float64)
    pos = 0
    for d in RECENCY_DAYS:
        recent = window_feats[-d:, key_idx]
        rmean = recent.mean(axis=0)
        rsum = recent.sum(axis=0)
        for ci in range(len(KEY_FEATURES)):
            recency_arr[pos] = rmean[ci]; pos += 1
            recency_arr[pos] = rsum[ci]; pos += 1
    feats.append(recency_arr)

    # 3. Drought indicators
    prec = window_feats[:, FEATURE_COLS.index('prec')]
    is_dry = prec < DRY_THRESHOLD
    if is_dry.any():
        diff = np.diff(np.concatenate(([0], is_dry.view(np.int8), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        max_streak = int((ends - starts).max()) if len(starts) else 0
    else:
        max_streak = 0
    feats.append(np.array([prec.sum(), is_dry.sum(), is_dry[-28:].sum(), max_streak],
                          dtype=np.float64))

    # 4. Trend slopes
    x = np.arange(WINDOW_DAYS, dtype=np.float64)
    xc = x - x.mean()
    x_var = (xc * xc).sum()
    slopes = np.empty(len(KEY_FEATURES), dtype=np.float64)
    for i, idx in enumerate([FEATURE_COLS.index(c) for c in KEY_FEATURES]):
        col = window_feats[:, idx]
        slopes[i] = ((col - col.mean()) * xc).sum() / x_var
    feats.append(slopes)

    # 5. Climatology anomalies
    clim_idx = [FEATURE_COLS.index(c) for c in CLIMATOLOGY_FEATURES]
    window_means_clim = window_feats[:, clim_idx].mean(axis=0)
    try:
        c_mean = clim_mean.loc[(region_id, end_month)].values
        c_std = clim_std.loc[(region_id, end_month)].values
    except KeyError:
        c_mean = np.zeros(len(CLIMATOLOGY_FEATURES))
        c_std = np.ones(len(CLIMATOLOGY_FEATURES))
    diff = window_means_clim - c_mean
    z = diff / np.maximum(c_std, 1e-6)
    clim_arr = np.empty(2 * len(CLIMATOLOGY_FEATURES), dtype=np.float64)
    for ci in range(len(CLIMATOLOGY_FEATURES)):
        clim_arr[2 * ci] = z[ci]
        clim_arr[2 * ci + 1] = diff[ci]
    feats.append(clim_arr)

    # 6. Score-history features (from train.csv lookups — leakage-free)
    s_all = float(region_score_mean.get(region_id, global_score_mean))
    s_month = float(region_month_score_mean.get((region_id, end_month), s_all))
    s_lastyr = float(region_lastyear_score_mean.get(region_id, s_all))
    feats.append(np.array([s_all, s_month, s_lastyr], dtype=np.float64))

    return np.concatenate([np.atleast_1d(f).ravel() for f in feats]).astype(np.float32)


# ───────────────────────── Window construction ─────────────────────────

def build_train_windows(
    train: pd.DataFrame,
    region_to_int: Dict[str, int],
    clim_mean, clim_std,
    region_score_mean, region_month_score_mean, region_lastyear_score_mean,
    global_score_mean: float,
):
    feat_list, month_list, region_list, target_list = [], [], [], []
    n_regions = train['region_id'].nunique()
    print(f'  Processing {n_regions} regions...')
    t0 = time.time()

    for ri, (region_id, group) in enumerate(train.groupby('region_id', sort=False)):
        feat_arr = group[FEATURE_COLS].values.astype(np.float32)
        score_arr = group['score'].values
        date_strs = group['date'].values
        n = len(group)
        end_days = np.arange(WINDOW_DAYS - 1, n - max(TARGET_OFFSETS), STRIDE)

        for e in end_days:
            window_feats = feat_arr[e - WINDOW_DAYS + 1: e + 1]
            month_e = int(date_strs[e][5:7])
            agg = aggregate_window(
                window_feats, month_e, region_id,
                clim_mean, clim_std,
                region_score_mean, region_month_score_mean,
                region_lastyear_score_mean, global_score_mean,
            )
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
    print(f'  NaN in targets: {np.isnan(y).sum()} (should be 0)')
    return X, y, regions.flatten()


def build_test_features(
    test: pd.DataFrame,
    region_to_int: Dict[str, int],
    clim_mean, clim_std,
    region_score_mean, region_month_score_mean, region_lastyear_score_mean,
    global_score_mean: float,
):
    feat_list, month_list, region_list, region_ids = [], [], [], []
    for region_id, group in test.groupby('region_id', sort=False):
        feat_arr = group[FEATURE_COLS].values.astype(np.float32)
        date_strs = group['date'].values
        month_last = int(date_strs[-1][5:7])
        agg = aggregate_window(
            feat_arr, month_last, region_id,
            clim_mean, clim_std,
            region_score_mean, region_month_score_mean,
            region_lastyear_score_mean, global_score_mean,
        )
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

def split_with_buffer(X, y, regions, n_val=N_VAL_WINDOWS, n_buf=N_BUFFER_WINDOWS):
    region_to_indices: Dict[float, List[int]] = {}
    for i, r in enumerate(regions):
        region_to_indices.setdefault(r, []).append(i)
    train_mask = np.zeros(len(X), dtype=bool)
    val_mask = np.zeros(len(X), dtype=bool)
    for r, idxs in region_to_indices.items():
        if len(idxs) < n_val + n_buf:
            train_mask[idxs] = True
            continue
        train_mask[idxs[:-(n_val + n_buf)]] = True
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
    parser.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    parser.add_argument('--out', default='baseline_v5_lgbm_score_history.csv')
    args = parser.parse_args()

    print('[Loading]')
    train, test, sub = load_data(args.base)
    print(f'  train: {train.shape}, test: {test.shape}, sub: {sub.shape}')

    all_regions = sorted(set(train['region_id']) | set(test['region_id']))
    region_to_int = {r: i for i, r in enumerate(all_regions)}

    print('\n[Climatology — weather]')
    clim_mean, clim_std = build_climatology(train)

    print('\n[Climatology — scores (from train only, leakage-free)]')
    rs_mean, rms_mean, rls_mean = build_score_lookups(train)
    global_score_mean = float(train['score'].dropna().mean())

    print('\n[Building training windows]')
    X, y, regions_X = build_train_windows(
        train, region_to_int, clim_mean, clim_std,
        rs_mean, rms_mean, rls_mean, global_score_mean,
    )

    print('\n[Train/val split with buffer]')
    X_tr, X_val, y_tr, y_val = split_with_buffer(X, y, regions_X)

    feature_names = make_feature_names()
    assert X.shape[1] == len(feature_names), \
        f'feature mismatch: X has {X.shape[1]}, names have {len(feature_names)}'
    print(f'  feature count: {len(feature_names)}')

    print('\n[Training 5 LightGBM models]')
    models, week_maes, val_mae = train_models(X_tr, y_tr, X_val, y_val, feature_names)

    print('\n[Building test features]')
    X_test, region_ids_test = build_test_features(
        test, region_to_int, clim_mean, clim_std,
        rs_mean, rms_mean, rls_mean, global_score_mean,
    )

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

    print('\n[Top 25 features for week-1 model]')
    imp = pd.DataFrame({
        'feature': feature_names,
        'gain': models[0].feature_importance(importance_type='gain'),
    }).sort_values('gain', ascending=False).head(25)
    print(imp.to_string(index=False))


if __name__ == '__main__':
    main()
