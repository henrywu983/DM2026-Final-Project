"""
4-way ensemble: V5 + V7-10seed + V10-5seed + V8.1-5seed.

Weights: 0.15 V5 + 0.50 V7-10seed + 0.30 V10-5seed + 0.05 V8.1-5seed

Run AFTER baseline_v8_1_pslstm_mh_5seed_avg.py completes and you've checked
the resulting val MAE. Decision rule before running this:

    V8.1-5seed val MAE ≤ 0.50  → run this, real shot at improvement
    V8.1-5seed val MAE 0.50–0.60 → marginal, run only if you want the data point
    V8.1-5seed val MAE > 0.60  → don't run; expected LB ≥ 0.800

Why 5% weight:
    V8.1 has higher noise (val MAE ~0.5–0.7 vs V7's 0.43) and biased mean
    predictions (~0.83 vs ~1.05 for V7/V10). At 5% weight, the bias drag is
    ~0.011 on mean predictions (small enough to tolerate). Any higher weight
    risks the bias dominating the diversity benefit.

    V5 stays at 0.15 because it's confirmed-cheap diversity (LB 0.7973 with
    V5 = LB 0.7971 without). Removing V5 to make room for V8.1 is the wrong
    swap — V5 is mostly free, V8.1 isn't.

Usage:
    !python {BASE}/ensemble_4way_with_v8.py {BASE}
"""

import sys
from pathlib import Path
import pandas as pd

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
print(f'BASE = {BASE.resolve()}\n')

PRED_COLS = ['pred_week1', 'pred_week2', 'pred_week3', 'pred_week4', 'pred_week5']

# Weights
W_V5    = 0.15
W_V7    = 0.50
W_V10   = 0.30
W_V8_1  = 0.05
total = W_V5 + W_V7 + W_V10 + W_V8_1
assert abs(total - 1.0) < 1e-9, f'weights must sum to 1.0, got {total}'

# Inputs
PATH_V5   = BASE / 'baseline_v5_lgbm_score_history.csv'
PATH_V7   = BASE / 'baseline_v7_tcn_10seed_avg.csv'
PATH_V10  = BASE / 'baseline_v10_tcn_static_5seed_avg.csv'
PATH_V8_1 = BASE / 'baseline_v8_1_pslstm_mh_5seed_avg.csv'   # output of step (2)
PATH_OUT  = BASE / 'ensemble_4way_v5_v7_10seed_v10_5seed_v8_1_5seed.csv'

# V7-10seed naming fallback
if not PATH_V7.exists():
    candidates = list(BASE.glob('*v7*10seed*.csv'))
    if len(candidates) == 1:
        PATH_V7 = candidates[0]
        print(f'V7-10seed auto-detected: {PATH_V7.name}\n')
    else:
        raise FileNotFoundError(
            f'V7-10seed file not found. Candidates: {[c.name for c in candidates]}'
        )

for p in [PATH_V5, PATH_V7, PATH_V10, PATH_V8_1]:
    if not p.exists():
        raise FileNotFoundError(f'Missing: {p}')

v5     = pd.read_csv(PATH_V5).sort_values('region_id').reset_index(drop=True)
v7     = pd.read_csv(PATH_V7).sort_values('region_id').reset_index(drop=True)
v10    = pd.read_csv(PATH_V10).sort_values('region_id').reset_index(drop=True)
v8_1   = pd.read_csv(PATH_V8_1).sort_values('region_id').reset_index(drop=True)

print('Loaded:')
print(f'  V5:          {PATH_V5.name}    ({len(v5)} rows)')
print(f'  V7-10seed:   {PATH_V7.name}    ({len(v7)} rows)')
print(f'  V10-5seed:   {PATH_V10.name}   ({len(v10)} rows)')
print(f'  V8.1-5seed:  {PATH_V8_1.name}  ({len(v8_1)} rows)\n')

# Sanity
for label, df in [('V7', v7), ('V10', v10), ('V8.1', v8_1)]:
    if not (v5['region_id'].values == df['region_id'].values).all():
        raise RuntimeError(f'region_id mismatch between V5 and {label}.')

# Mean-bias diagnostic — flag if V8.1's bias is severe
print('Per-week mean check (bias diagnostic — V8.1 expected ~0.20 below V7/V10):')
print(f'  {"week":8s} {"V5":>8s} {"V7-10":>8s} {"V10-5":>8s} {"V8.1-5":>8s}')
for col in PRED_COLS:
    print(f'  {col:8s} {v5[col].mean():8.4f} {v7[col].mean():8.4f} '
          f'{v10[col].mean():8.4f} {v8_1[col].mean():8.4f}')

mean_bias = v7[PRED_COLS].mean().mean() - v8_1[PRED_COLS].mean().mean()
print(f'\nV8.1 vs V7-10 mean bias: {mean_bias:.4f}')
if mean_bias > 0.30:
    print('  ⚠ Bias > 0.30 — V8.1 may drag the ensemble more than expected. '
          'Consider rerunning with W_V8_1 = 0.03 instead of 0.05.')

# Blend
out = v5[['region_id']].copy()
for col in PRED_COLS:
    out[col] = (
        W_V5    * v5[col].values
        + W_V7   * v7[col].values
        + W_V10  * v10[col].values
        + W_V8_1 * v8_1[col].values
    ).clip(0.0, 5.0)

out.to_csv(PATH_OUT, index=False)

print(f'\nWeights: V5 {W_V5}, V7-10seed {W_V7}, V10-5seed {W_V10}, V8.1-5seed {W_V8_1}')

print('\nBlend per-week means (compare to current best 0.7948 ensemble):')
for col in PRED_COLS:
    print(f'  {col}: {out[col].mean():.4f}')

print(f'\nOutput range: [{out[PRED_COLS].values.min():.3f}, '
      f'{out[PRED_COLS].values.max():.3f}]')
print(f'\n[Saved] {PATH_OUT}')
