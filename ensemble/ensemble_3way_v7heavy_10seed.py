"""
Build the V7-heavy 3-way ensemble: 0.15 V5 + 0.65 V7-10seed + 0.20 V10-10seed.

Hypothesis under test:
    The 0.7948 LB from V7-10seed + V10-5seed at weights (0.15/0.55/0.30) gained
    not from seed averaging per se, but from an implicit V7-heavy tilt — V7
    became stronger while V10 didn't. Going to V10-10seed at the same weights
    (0.7965) erased that tilt. If we re-introduce it explicitly via weights
    (0.15/0.65/0.20), we should recover ~0.794–0.795.

Outcome interpretations:
    LB ≤ 0.7950  → weight-tilt explanation confirmed; weight tuning > seed
                    averaging at this point on the LB surface.
    LB 0.7950–0.7965 → mixed; weight tilt helps but doesn't fully explain.
    LB > 0.7965  → V10-10seed has genuinely shifted predictions and even more
                    V7 weight can't fix it; V10 has hit its variance-reduction
                    ceiling.

Usage in Colab:
    !python {BASE}/ensemble_3way_v7heavy_10seed.py {BASE}
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
print(f'BASE = {BASE.resolve()}\n')

PRED_COLS = ['pred_week1', 'pred_week2', 'pred_week3', 'pred_week4', 'pred_week5']

# Weights — V7-heavy variant
W_V5  = 0.15
W_V7  = 0.65
W_V10 = 0.20
assert abs(W_V5 + W_V7 + W_V10 - 1.0) < 1e-9, 'weights must sum to 1.0'

# Inputs
PATH_V5  = BASE / 'baseline_v5_lgbm_score_history.csv'
PATH_V7  = BASE / 'baseline_v7_tcn_10seed_avg.csv'           # adjust if your V7-10seed file differs
PATH_V10 = BASE / 'baseline_v10_tcn_static_10seed_avg.csv'
PATH_OUT = BASE / 'ensemble_3way_v7heavy_10seed.csv'

# V7-10seed naming guess — fall back if the filename is different
if not PATH_V7.exists():
    candidates = list(BASE.glob('*v7*10seed*.csv'))
    if len(candidates) == 1:
        PATH_V7 = candidates[0]
        print(f'V7-10seed auto-detected: {PATH_V7.name}\n')
    elif len(candidates) > 1:
        raise FileNotFoundError(
            f'Multiple V7-10seed candidates found: {[c.name for c in candidates]}. '
            f'Edit PATH_V7 in this script to pick one.'
        )
    else:
        raise FileNotFoundError(
            f'V7-10seed file not found. Expected {PATH_V7.name} in {BASE}. '
            f'Edit PATH_V7 in this script if your filename differs.'
        )

# Load
for p in [PATH_V5, PATH_V7, PATH_V10]:
    if not p.exists():
        raise FileNotFoundError(f'Missing: {p}')

v5  = pd.read_csv(PATH_V5).sort_values('region_id').reset_index(drop=True)
v7  = pd.read_csv(PATH_V7).sort_values('region_id').reset_index(drop=True)
v10 = pd.read_csv(PATH_V10).sort_values('region_id').reset_index(drop=True)

print('Loaded:')
print(f'  V5:        {PATH_V5.name}  ({len(v5)} rows)')
print(f'  V7-10seed: {PATH_V7.name}  ({len(v7)} rows)')
print(f'  V10-10seed: {PATH_V10.name}  ({len(v10)} rows)\n')

# Sanity: same region order across all three
if not (v5['region_id'].values == v7['region_id'].values).all():
    raise RuntimeError('region_id mismatch between V5 and V7-10seed.')
if not (v5['region_id'].values == v10['region_id'].values).all():
    raise RuntimeError('region_id mismatch between V5 and V10-10seed.')

# Build the blend
out = v5[['region_id']].copy()
for col in PRED_COLS:
    out[col] = (
        W_V5  * v5[col].values
        + W_V7  * v7[col].values
        + W_V10 * v10[col].values
    ).clip(0.0, 5.0)

out.to_csv(PATH_OUT, index=False)

# Diagnostic
print(f'Weights: V5 {W_V5}, V7-10seed {W_V7}, V10-10seed {W_V10}\n')

print('Per-week means — comparing inputs and output:')
print(f'  {"week":8s} {"V5":>8s} {"V7-10":>8s} {"V10-10":>8s} {"blend":>8s}')
for col in PRED_COLS:
    print(f'  {col:8s} {v5[col].mean():8.4f} {v7[col].mean():8.4f} '
          f'{v10[col].mean():8.4f} {out[col].mean():8.4f}')

print(f'\nOutput range: [{out[PRED_COLS].values.min():.3f}, '
      f'{out[PRED_COLS].values.max():.3f}]')
print(f'\n[Saved] {PATH_OUT}')
print('\nReady to submit as the V7-heavy 10-seed ensemble.')
