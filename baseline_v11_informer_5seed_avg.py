"""
Baseline V11 — lean Informer, 5-seed averaged (phase 2).

Runs after phase 1 (baseline_v11_informer.py) clears its gates. Trains the
LeanInformer with N fresh seeds, averages the RAW (pre-clip) test predictions,
clips once to [0, 5], and saves a single submission. Same multi-seed pattern as
baseline_v7_tcn_5seed_avg.py: prepare data once, loop over seeds, average, clip.

Why average raw then clip once (not clip-then-average): clipping each seed first
would bias the mean upward near 0, where 60% of the targets live. Averaging the
unclipped predictions preserves the variance-reduction benefit; a single clip at
the end enforces the [0, 5] range.

Usage
-----
    python baseline_v11_informer_5seed_avg.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v11_informer_5seed_avg.csv \
        --seeds 42,43,44,45,46 \
        --v7-csv baseline_v7_10seed_avg.csv     # for the post-average correlation gate

Reminder on the correlation gate: the number that matters for ensembling is
V11-5seed ↔ V7-10seed, computed here on the AVERAGED predictions. Seed averaging
typically RAISES correlation between similar models (V7↔V10 went 0.88 → 0.92), so
expect this to come in above the single-seed 0.802. If it lands >0.85 the model
is likely redundant with V7 regardless of its val MAE.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from data_pipeline_nn import prepare_data                         # noqa: E402
from baseline_v11_informer import (                                # noqa: E402
    LeanInformer, train_epoch, eval_epoch, predict_test, acceptance_report,
)


def train_one_seed(bundle, args, seed: int, device) -> tuple[np.ndarray, float, np.ndarray]:
    """Train one model from scratch. Returns (raw_test_preds, best_val, best_week_mae)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LeanInformer(
        n_features=bundle.n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        n_regions=bundle.n_regions,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = args.epochs * len(bundle.train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_val = math.inf
    best_state = None
    best_week_mae = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, bundle.train_loader, opt, device, scheduler)
        val_mae, week_mae = eval_epoch(model, bundle.val_loader, device)
        dt = time.time() - t0

        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_week_mae = week_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            tag = '★'
        else:
            epochs_no_improve += 1
            tag = ' '

        print(f'    [seed {seed}] ep {epoch:3d}/{args.epochs}  tr {tr_loss:.4f}  '
              f'val {val_mae:.4f}  {dt:.0f}s  {tag}')

        if epochs_no_improve >= args.patience:
            print(f'    [seed {seed}] early stop @ ep {epoch}')
            break

    model.load_state_dict(best_state)
    raw_preds = predict_test(model, bundle.test_X, bundle.test_regions, device,
                             batch_size=args.batch_size)   # RAW, unclipped
    print(f'    [seed {seed}] best val {best_val:.4f}, '
          f'raw test mean {raw_preds.mean():.3f}')
    return raw_preds, best_val, best_week_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    parser.add_argument('--out', default='baseline_v11_informer_5seed_avg.csv')
    parser.add_argument('--seeds', default='42,43,44,45,46')
    parser.add_argument('--v7-csv', default=None)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--e-layers', type=int, default=2)
    parser.add_argument('--d-layers', type=int, default=1)
    parser.add_argument('--d-ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.10)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    print(f'[Seeds] {seeds}')

    bundle = prepare_data(args.base, batch_size=args.batch_size, num_workers=2)

    all_raw = []
    per_seed_val = []
    last_week_mae = None
    for seed in seeds:
        print(f'\n[Training seed {seed}]')
        raw, val, week_mae = train_one_seed(bundle, args, seed, device)
        all_raw.append(raw)
        per_seed_val.append(val)
        last_week_mae = week_mae

    # Average raw predictions across seeds, then clip ONCE
    avg_raw = np.mean(np.stack(all_raw, axis=0), axis=0)   # (n_test, 5)
    avg_preds = np.clip(avg_raw, 0.0, 5.0)

    print('\n[Seed summary]')
    for s, v in zip(seeds, per_seed_val):
        print(f'  seed {s}: val {v:.4f}')
    print(f'  mean per-seed val MAE: {np.mean(per_seed_val):.4f}')
    print(f'  averaged test mean {avg_preds.mean():.3f}, '
          f'median {np.median(avg_preds):.3f}')

    region_to_pred = {r: p for r, p in zip(bundle.test_region_ids, avg_preds)}
    rows = [[r, *region_to_pred[r]] for r in bundle.sub_template['region_id']]
    submission = pd.DataFrame(
        rows,
        columns=['region_id', 'pred_week1', 'pred_week2',
                 'pred_week3', 'pred_week4', 'pred_week5'],
    )
    out_path = os.path.join(args.base, args.out)
    submission.to_csv(out_path, index=False)
    print(f'\n[Saved] {out_path}')
    print(submission.head())

    # Acceptance report on the AVERAGED result. Use mean per-seed val MAE as the
    # val figure; the correlation gate now reflects the averaged predictions.
    acceptance_report(float(np.mean(per_seed_val)), last_week_mae,
                      avg_preds, submission, args.v7_csv, args.base)


if __name__ == '__main__':
    main()
