"""
Weight sweep on current-best ensemble: more V10-5seed weight.

Current best: V5 + V7-10seed + V10-5seed at (0.15 / 0.55 / 0.30) → LB 0.7948
This script:   V5 + V7-10seed + V10-5seed at (0.15 / 0.50 / 0.35)

Hypothesis: the 0.30 V10 weight was tuned when V7 was 5-seed. Now that V7 is
10-seed (lower noise, stronger), V10 might deserve more relative weight to
preserve diversity.

Outcomes:
    LB ≤ 0.7945  → V10 was underweighted; consider another sweep
    LB ≈ 0.7948  → 0.7948 is the sweet spot at these weights
    LB > 0.7950  → V10 weight was already near optimal

Usage:
    !python {BASE}/ensemble_3way_v10heavier.py {BASE}
"""

import sys
from pathlib import Path
import pandas as pd

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
print(f'BASE = {BASE.resolve()}\n')

PRED_COLS = ['pred_week1', 'pred_week2', 'pred_week3', 'pred_week4', 'pred_week5']

# Weights — V10-heavier variant
W_V5  = 0.0
W_V7  = 0.55
W_V10 = 0.45
assert abs(W_V5 + W_V7 + W_V10 - 1.0) < 1e-9, 'weights must sum to 1.0'

# Inputs
PATH_V5  = BASE / 'baseline_v5_lgbm_score_history.csv'
PATH_V7  = BASE / 'baseline_v7_tcn_10seed_avg.csv'
PATH_V10 = BASE / 'baseline_v10_tcn_static_5seed_avg.csv'   # NOTE: 5-seed, not 10-seed
PATH_OUT = BASE / 'ensemble_3way_v10heavier_v7_10seed_no_v5.csv'

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

for p in [PATH_V5, PATH_V7, PATH_V10]:
    if not p.exists():
        raise FileNotFoundError(f'Missing: {p}')

v5  = pd.read_csv(PATH_V5).sort_values('region_id').reset_index(drop=True)
v7  = pd.read_csv(PATH_V7).sort_values('region_id').reset_index(drop=True)
v10 = pd.read_csv(PATH_V10).sort_values('region_id').reset_index(drop=True)

print('Loaded:')
print(f'  V5:        {PATH_V5.name}  ({len(v5)} rows)')
print(f'  V7-10seed: {PATH_V7.name}  ({len(v7)} rows)')
print(f'  V10-5seed: {PATH_V10.name}  ({len(v10)} rows)\n')

# Sanity
if not (v5['region_id'].values == v7['region_id'].values).all():
    raise RuntimeError('region_id mismatch between V5 and V7-10seed.')
if not (v5['region_id'].values == v10['region_id'].values).all():
    raise RuntimeError('region_id mismatch between V5 and V10-5seed.')

# Blend
out = v5[['region_id']].copy()
for col in PRED_COLS:
    out[col] = (
        W_V5  * v5[col].values
        + W_V7  * v7[col].values
        + W_V10 * v10[col].values
    ).clip(0.0, 5.0)

out.to_csv(PATH_OUT, index=False)

print(f'Weights: V5 {W_V5}, V7-10seed {W_V7}, V10-5seed {W_V10}\n')

print('Per-week means:')
print(f'  {"week":8s} {"V5":>8s} {"V7-10":>8s} {"V10-5":>8s} {"blend":>8s}')
for col in PRED_COLS:
    print(f'  {col:8s} {v5[col].mean():8.4f} {v7[col].mean():8.4f} '
          f'{v10[col].mean():8.4f} {out[col].mean():8.4f}')

print(f'\nOutput range: [{out[PRED_COLS].values.min():.3f}, '
      f'{out[PRED_COLS].values.max():.3f}]')
print(f'\n[Saved] {PATH_OUT}')
