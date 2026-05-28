"""
4-way ensemble builder: V5 (LightGBM) + V7-10seed (TCN) + V10-5seed (TCN+static)
+ V11-5seed (lean Informer).

Builds one submission per weight variant in a single run. Weights are
(w_v5, w_v7, w_v10, w_v11) and must sum to 1.0 (a small tolerance is allowed
and the script renormalizes + warns otherwise).

Blend math: per-region, per-week weighted average of the four prediction
matrices, then clip to [0, 5]. All four inputs are already clipped submissions,
so this is a convex combination and stays in range; the final clip is a guard.

Focus of this sweep
--------------------
V11-5seed (LB 0.8020) is the second-strongest standalone NN, behind V7-5seed
(0.8071) and ahead of V10-5seed (0.8159), and is less correlated with V7 (0.802)
than V10 is (0.92). Open question: do V10 and V11 stack (both add), or compete
(they correlate with each other, so weight on both ~ weight on one)? The variants
below hold V5+V7 roughly fixed and trade the remaining mass between V10 and V11
at comparable levels to read that off the LB directly.

Usage
-----
    python ensemble_4way_v10_v11_sweep.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --v5  baseline_v5_lgbm.csv \
        --v7  baseline_v7_10seed_avg.csv \
        --v10 baseline_v10_5seed_avg.csv \
        --v11 baseline_v11_informer_5seed_avg.csv

Edit VARIANTS below to add/remove blends. Each is (name, w_v5, w_v7, w_v10, w_v11).
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

PRED_COLS = [f'pred_week{i}' for i in range(1, 6)]

# (name, w_v5, w_v7, w_v10, w_v11)  — V10 and V11 balanced around equal weight.
VARIANTS = [
    # --- baselines for reference (no V11) ---
    ('current_best_3way',      0.10, 0.50, 0.40, 0.00),   # = your 0.7942 pick
    # --- V10 / V11 balanced, V5+V7 held at 0.55 total ---
    ('bal_2020',               0.10, 0.45, 0.20, 0.25),
    ('bal_2222',               0.10, 0.45, 0.225, 0.225),  # V10 ≈ V11
    ('bal_2520',               0.10, 0.45, 0.25, 0.20),
    # --- V10 / V11 balanced, more room (V7 down to 0.40) ---
    ('bal_v7low_2525',         0.10, 0.40, 0.25, 0.25),
    ('bal_v7low_3020',         0.10, 0.40, 0.30, 0.20),
    ('bal_v7low_2030',         0.10, 0.40, 0.20, 0.30),
    # --- drop V5, give its mass to the NN diversifiers equally ---
    ('nov5_bal',               0.00, 0.45, 0.275, 0.275),
    # --- single-diversifier controls (to detect V10/V11 redundancy) ---
    ('v11_only_div',           0.10, 0.45, 0.00, 0.45),
    ('v10_only_div',           0.10, 0.45, 0.45, 0.00),
]


def load_aligned(base, fname, ref_ids):
    path = fname if os.path.isabs(fname) else os.path.join(base, fname)
    df = pd.read_csv(path).set_index('region_id')
    df = df.reindex(ref_ids)              # align to submission order
    if df[PRED_COLS].isna().any().any():
        missing = int(df[PRED_COLS].isna().any(axis=1).sum())
        raise ValueError(f'{fname}: {missing} region_ids missing after align')
    return df[PRED_COLS].values.astype(np.float64)   # (n_regions, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    ap.add_argument('--v5',  required=True)
    ap.add_argument('--v7',  required=True)
    ap.add_argument('--v10', required=True)
    ap.add_argument('--v11', required=True)
    ap.add_argument('--outdir', default='ensembles_v10_v11')
    args = ap.parse_args()

    # Reference region order from sample_submission
    sub = pd.read_csv(os.path.join(args.base, 'sample_submission.csv'))
    ref_ids = sub['region_id'].values

    print('[Loading component CSVs]')
    P = {
        'v5':  load_aligned(args.base, args.v5,  ref_ids),
        'v7':  load_aligned(args.base, args.v7,  ref_ids),
        'v10': load_aligned(args.base, args.v10, ref_ids),
        'v11': load_aligned(args.base, args.v11, ref_ids),
    }
    for k, v in P.items():
        print(f'  {k:4s}: shape {v.shape}, mean {v.mean():.3f}')

    # Pairwise correlations (flattened) — context for reading the sweep
    print('\n[Pairwise correlation, flattened predictions]')
    keys = ['v5', 'v7', 'v10', 'v11']
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = P[keys[i]].ravel(), P[keys[j]].ravel()
            c = float(np.corrcoef(a, b)[0, 1])
            print(f'  {keys[i]:4s} ↔ {keys[j]:4s}: {c:.3f}')

    outdir = os.path.join(args.base, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print('\n[Building variants]')
    summary = []
    for name, w5, w7, w10, w11 in VARIANTS:
        w = np.array([w5, w7, w10, w11], dtype=np.float64)
        if abs(w.sum() - 1.0) > 1e-6:
            print(f'  !! {name}: weights sum to {w.sum():.4f}, renormalizing')
            w = w / w.sum()
        blend = (w[0] * P['v5'] + w[1] * P['v7'] +
                 w[2] * P['v10'] + w[3] * P['v11'])
        blend = np.clip(blend, 0.0, 5.0)

        out = pd.DataFrame(
            np.column_stack([ref_ids, blend]),
            columns=['region_id', *PRED_COLS],
        )
        for c in PRED_COLS:
            out[c] = out[c].astype(float)
        out_path = os.path.join(outdir, f'ensemble_4way_{name}.csv')
        out.to_csv(out_path, index=False)
        summary.append((name, *w, blend.mean()))
        print(f'  {name:20s}  w=({w[0]:.3f},{w[1]:.3f},{w[2]:.3f},{w[3]:.3f})  '
              f'mean {blend.mean():.3f}  -> {os.path.basename(out_path)}')

    print('\n[Summary]')
    sm = pd.DataFrame(summary, columns=['variant', 'w_v5', 'w_v7', 'w_v10',
                                        'w_v11', 'pred_mean'])
    print(sm.round(3).to_string(index=False))
    print(f'\nAll variants written to: {outdir}')
    print('Submit the spread, read the LB gradient. Compare against 0.7942.')


if __name__ == '__main__':
    main()
