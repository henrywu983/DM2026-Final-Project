"""
Baseline V6 — Richer per-region drought baseline features.

Motivation
----------
V5 reached LB 0.9017 (above Baseline 1 = 0.9117). V5's feature importance
showed climatology anomalies and region-level baselines doing the heavy
lifting, so V6 doubles down on this signal: instead of a single mean per
region or per (region, month), we add richer summaries — max, std,
percentiles, frequencies, trends, and recent-vs-baseline contrasts.

All new features are computed once from train.csv (using only score-day
rows) and looked up by region_id and end-month. They exist identically at
train and inference time. No score column is required in test.csv.

New features (12 total)
-----------------------
Recent (last ~52 score-weeks of training per region):
  recent_score_max         max score in recent year
  recent_drought_weeks     count of weeks with score >= 2
  recent_score_std         std of recent-year scores
  recent_score_trend       linear slope of recent-year scores
  recent_nonzero_frac      fraction of recent weeks with score > 0

Region-month detail (historical statistics per (region, month)):
  region_month_score_max     max score in that month historically
  region_month_score_std     std of scores in that month
  region_month_score_p75     75th percentile of scores in that month
  region_month_drought_freq  fraction of region-month obs with score >= 2
  region_month_zero_freq     fraction with score == 0

Cross-temporal contrasts:
  recent_vs_alltime_diff  recent_score_mean - region_score_mean_all
  recent_vs_month_diff    recent_score_mean - region_month_score_mean

Retained from V5
----------------
Everything else: full-window weather aggregates, recency aggregates,
drought indicators, trend slopes, climatology anomalies (12),
region_score_mean_all, region_month_score_mean, region_lastyear_score_mean,
buffered time-based validation, 5 LightGBM models per week-ahead horizon.

Total feature count: 105 (V5) + 12 (new) = 117

Usage
-----
    python baseline_v6_lgbm_recent_drought.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v6_lgbm_recent_drought.csv
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
RECENT_SCORE_WEEKS = 52   # ~1 year of weekly scores per region

WINDOW_DAYS = 91
TARGET_OFFSETS = [7, 14, 21, 28, 35]
STRIDE = 35
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


# ───────────────────────── Lookup tables ─────────────────────────

def build_climatology(train: pd.DataFrame):
    print('  Weather climatology...')
    df = train.copy()
    df['month'] = df['date'].str[5:7].astype(int)
    grouped = df.groupby(['region_id', 'month'])[CLIMATOLOGY_FEATURES]
    clim_mean = grouped.mean()
    clim_std = grouped.std().fillna(1e-6)
    clim_std = clim_std.where(clim_std > 1e-6, 1e-6)
    return clim_mean, clim_std


def build_score_lookups(train: pd.DataFrame):
    """V5's three lookups (kept for retention)."""
    df = train[train['score'].notna()].copy()
    df['month'] = df['date'].str[5:7].astype(int)
    region_score_mean = df.groupby('region_id')['score'].mean()
    region_month_score_mean = df.groupby(['region_id', 'month'])['score'].mean()
    region_lastyear_score_mean = (
        df.sort_values(['region_id', 'date'])
        .groupby('region_id')
        .tail(RECENT_SCORE_WEEKS + 5)
        .groupby('region_id')['score']
        .mean()
    )
    return region_score_mean, region_month_score_mean, region_lastyear_score_mean


def build_recent_score_features(train: pd.DataFrame) -> pd.DataFrame:
    """
    Per-region statistics over the last ~52 score-weeks of training.
    Returns DataFrame indexed by region_id with 5 columns.
    """
    print('  Recent score-history (last 52 weeks)...')
    df = train[train['score'].notna()].copy()
    df = df.sort_values(['region_id', 'date'])
    df = df.groupby('region_id').tail(RECENT_SCORE_WEEKS)
    df = df.reset_index(drop=True)

    def slope_per_group(g):
        y = g['score'].values.astype(float)
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y), dtype=float)
        xc = x - x.mean()
        denom = (xc * xc).sum()
        return ((y - y.mean()) * xc).sum() / denom if denom > 0 else 0.0

    grouped = df.groupby('region_id')
    out = pd.DataFrame({
        'recent_score_max': grouped['score'].max(),
        'recent_drought_weeks': grouped['score'].apply(lambda s: int((s >= 2).sum())),
        'recent_score_std': grouped['score'].std().fillna(0.0),
        'recent_score_trend': grouped.apply(slope_per_group),
        'recent_nonzero_frac': grouped['score'].apply(lambda s: float((s > 0).mean())),
    })
    return out


def build_region_month_detail(train: pd.DataFrame) -> pd.DataFrame:
    """Richer per-(region, month) statistics."""
    print('  Region-month detail (max/std/p75/frequencies)...')
    df = train[train['score'].notna()].copy()
    df['month'] = df['date'].str[5:7].astype(int)
    grouped = df.groupby(['region_id', 'month'])['score']
    out = pd.DataFrame({
        'region_month_score_max': grouped.max(),
        'region_month_score_std': grouped.std().fillna(0.0),
        'region_month_score_p75': grouped.quantile(0.75),
        'region_month_drought_freq': grouped.apply(lambda s: float((s >= 2).mean())),
        'region_month_zero_freq': grouped.apply(lambda s: float((s == 0).mean())),
    })
    return out


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
    # 5. Climatology anomalies
    for col in CLIMATOLOGY_FEATURES:
        names.append(f'{col}_anom_z')
        names.append(f'{col}_anom_diff')
    # 6. V5 score-history (3)
    names += ['region_score_mean_all', 'region_month_score_mean', 'region_lastyear_score_mean']
    # 7. NEW V6 — recent score features (5)
    names += ['recent_score_max', 'recent_drought_weeks', 'recent_score_std',
              'recent_score_trend', 'recent_nonzero_frac']
    # 8. NEW V6 — region-month detail (5)
    names += ['region_month_score_max', 'region_month_score_std',
              'region_month_score_p75', 'region_month_drought_freq',
              'region_month_zero_freq']
    # 9. NEW V6 — contrasts (2)
    names += ['recent_vs_alltime_diff', 'recent_vs_month_diff']
    # 10. Categoricals
    names += ['month', 'region_int']
    return names


# ───────────────────────── Per-window feature extraction ─────────────────────────

def aggregate_window(
    window_feats: np.ndarray,
    end_month: int,
    region_id: str,
    clim_mean, clim_std,
    rs_mean, rms_mean, rls_mean, global_score_mean,
    recent_df, regmon_df,
) -> np.ndarray:
    feats: List[np.ndarray | float] = []

    # 1. Full-window aggregates
    feats.append(window_feats.mean(axis=0))
    feats.append(window_feats.min(axis=0))
    feats.append(window_feats.max(axis=0))
    feats.append(window_feats.std(axis=0))

    # 2. Recency
    key_idx = [FEATURE_COLS.index(c) for c in KEY_FEATURES]
    recency_arr = np.empty(len(RECENCY_DAYS) * len(KEY_FEATURES) * 2, dtype=np.float64)
    pos = 0
    for d in RECENCY_DAYS:
        recent = window_feats[-d:, key_idx]
        rmean = recent.mean(axis=0); rsum = recent.sum(axis=0)
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

    # 4. Trends
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

    # 6. V5 score-history (3 features)
    s_all = float(rs_mean.get(region_id, global_score_mean))
    s_month = float(rms_mean.get((region_id, end_month), s_all))
    s_lastyr = float(rls_mean.get(region_id, s_all))
    feats.append(np.array([s_all, s_month, s_lastyr], dtype=np.float64))

    # 7. NEW: recent score features
    if region_id in recent_df.index:
        rrow = recent_df.loc[region_id]
        recent_arr = np.array([
            rrow['recent_score_max'],
            rrow['recent_drought_weeks'],
            rrow['recent_score_std'],
            rrow['recent_score_trend'],
            rrow['recent_nonzero_frac'],
        ], dtype=np.float64)
    else:
        recent_arr = np.zeros(5, dtype=np.float64)
    feats.append(recent_arr)

    # 8. NEW: region-month detail
    try:
        rm = regmon_df.loc[(region_id, end_month)]
        rm_arr = np.array([
            rm['region_month_score_max'],
            rm['region_month_score_std'],
            rm['region_month_score_p75'],
            rm['region_month_drought_freq'],
            rm['region_month_zero_freq'],
        ], dtype=np.float64)
    except KeyError:
        rm_arr = np.array([s_all, 0.0, s_all, 0.0, 1.0], dtype=np.float64)
    feats.append(rm_arr)

    # 9. NEW: contrasts
    feats.append(np.array([s_lastyr - s_all, s_lastyr - s_month], dtype=np.float64))

    return np.concatenate([np.atleast_1d(f).ravel() for f in feats]).astype(np.float32)


# ───────────────────────── Window construction ─────────────────────────

def build_train_windows(train, region_to_int, *lookups):
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
            agg = aggregate_window(window_feats, month_e, region_id, *lookups)
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
    return X, y, regions.flatten()


def build_test_features(test, region_to_int, *lookups):
    feat_list, month_list, region_list, region_ids = [], [], [], []
    for region_id, group in test.groupby('region_id', sort=False):
        feat_arr = group[FEATURE_COLS].values.astype(np.float32)
        date_strs = group['date'].values
        month_last = int(date_strs[-1][5:7])
        agg = aggregate_window(feat_arr, month_last, region_id, *lookups)
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
    models, week_maes = [], []
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
    parser.add_argument('--out', default='baseline_v6_lgbm_recent_drought.csv')
    args = parser.parse_args()

    print('[Loading]')
    train, test, sub = load_data(args.base)
    print(f'  train: {train.shape}, test: {test.shape}, sub: {sub.shape}')

    all_regions = sorted(set(train['region_id']) | set(test['region_id']))
    region_to_int = {r: i for i, r in enumerate(all_regions)}

    print('\n[Building lookup tables]')
    clim_mean, clim_std = build_climatology(train)
    rs_mean, rms_mean, rls_mean = build_score_lookups(train)
    recent_df = build_recent_score_features(train)
    regmon_df = build_region_month_detail(train)
    global_score_mean = float(train['score'].dropna().mean())
    print(f'  recent_df shape: {recent_df.shape}, regmon_df shape: {regmon_df.shape}')

    lookups = (clim_mean, clim_std, rs_mean, rms_mean, rls_mean,
               global_score_mean, recent_df, regmon_df)

    print('\n[Building training windows]')
    X, y, regions_X = build_train_windows(train, region_to_int, *lookups)

    print('\n[Train/val split with buffer]')
    X_tr, X_val, y_tr, y_val = split_with_buffer(X, y, regions_X)

    feature_names = make_feature_names()
    assert X.shape[1] == len(feature_names), \
        f'feature mismatch: X has {X.shape[1]}, names have {len(feature_names)}'
    print(f'  feature count: {len(feature_names)}')

    print('\n[Training 5 LightGBM models]')
    models, week_maes, val_mae = train_models(X_tr, y_tr, X_val, y_val, feature_names)

    print('\n[Building test features]')
    X_test, region_ids_test = build_test_features(test, region_to_int, *lookups)

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
