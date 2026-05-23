"""
Baseline V7-seed-ensemble — 5-seed average Temporal Convolutional Network (TCN).

This script trains the same V7 TCN architecture multiple times with different
random seeds, predicts the test set with each trained model, averages the test
predictions, clips the averaged predictions to [0, 5], and saves one Kaggle
submission file.

Default seeds: 42, 43, 44, 45, 46.

Usage
-----
    python baseline_v7_tcn_5seed_avg.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out baseline_v7_tcn_5seed_avg.csv \
        --epochs 60

Optional:
    python baseline_v7_tcn_5seed_avg.py --seeds 42,123,2026,7,999

Notes
-----
- The data pipeline is prepared once and shared across all seeds.
- Each seed trains a fresh TCN model from scratch.
- Early stopping is applied independently for each seed.
- The final prediction is the arithmetic mean of all seed predictions.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared pipeline. Both files should live in the same directory.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from data_pipeline_nn import prepare_data, TARGET_OFFSETS  # noqa: E402


# ───────────────────────── Model ─────────────────────────

class TemporalBlock(nn.Module):
    """Residual dilated conv block. Keeps sequence length via causal-equivalent padding."""

    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.norm1(self.conv1(x))))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return x + h   # residual


class TCN(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
        channels: int = 64,
        n_blocks: int = 5,
        n_regions: int = 2248,
        region_emb_dim: int = 8,
        n_outputs: int = 5,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input projection: 14 → 64 channels
        self.input_proj = nn.Conv1d(n_features, channels, kernel_size=1)

        # Dilated residual blocks
        self.blocks = nn.ModuleList([
            TemporalBlock(channels, kernel=kernel, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])

        # Region embedding
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(channels + region_emb_dim, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, n_outputs),
        )

    def forward(self, x: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, 91, 14)
        region: (B,) long
        Returns (B, 5)
        """
        x = x.transpose(1, 2)             # (B, 14, 91) — channels first for Conv1d
        x = self.input_proj(x)            # (B, 64, 91)
        for block in self.blocks:
            x = block(x)
        # Global average pool over time → (B, 64)
        h = x.mean(dim=2)
        # Concat with region embedding
        emb = self.region_emb(region)     # (B, 8)
        h = torch.cat([h, emb], dim=1)
        return self.head(h)               # (B, 5)


# ───────────────────────── Training ─────────────────────────

def train_epoch(model, loader, opt, device, scheduler=None):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, r, y in loader:
        x, r, y = x.to(device, non_blocking=True), r.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        pred = model(x, r)
        loss = F.l1_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(x)
        n_samples += len(x)
    return total_loss / n_samples


@torch.no_grad()
def eval_epoch(model, loader, device) -> Tuple[float, np.ndarray]:
    """Returns (mean MAE across weeks, per-week MAE array)."""
    model.eval()
    all_preds, all_y = [], []
    for x, r, y in loader:
        x, r = x.to(device, non_blocking=True), r.to(device, non_blocking=True)
        pred = model(x, r).cpu().numpy()
        all_preds.append(pred)
        all_y.append(y.numpy())
    P = np.concatenate(all_preds)
    Y = np.concatenate(all_y)
    week_mae = np.abs(P - Y).mean(axis=0)
    return float(week_mae.mean()), week_mae


@torch.no_grad()
def predict_test(model, test_X, test_regions, device, batch_size=512) -> np.ndarray:
    model.eval()
    preds = []
    n = len(test_X)
    for i in range(0, n, batch_size):
        x = test_X[i:i+batch_size].to(device)
        r = test_regions[i:i+batch_size].to(device)
        p = model(x, r).cpu().numpy()
        preds.append(p)
    return np.concatenate(preds)



# ───────────────────────── Multi-seed helpers ─────────────────────────

def parse_seeds(seed_string: str) -> list[int]:
    """Parse comma-separated seed list, e.g. '42,43,44,45,46'."""
    seeds = []
    for item in seed_string.split(','):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError('At least one seed must be provided.')
    return seeds


def set_all_seeds(seed: int) -> None:
    """Set common random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_seed(seed: int, args, bundle, device):
    """
    Train one TCN model with a given seed.

    Returns
    -------
    best_val : float
        Best validation MAE achieved by this seed.
    best_week_mae : np.ndarray
        Per-week validation MAE at the best epoch.
    test_pred : np.ndarray
        Raw, unclipped test predictions with shape (n_regions, 5).
    """
    set_all_seeds(seed)

    print(f'\n' + '=' * 80)
    print(f'[Seed {seed}] Training fresh TCN')
    print('=' * 80)

    model = TCN(
        n_features=bundle.n_features,
        channels=args.channels,
        n_blocks=args.n_blocks,
        n_regions=bundle.n_regions,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  TCN: {args.n_blocks} blocks × {args.channels} channels, {n_params:,} params')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(bundle.train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_val = math.inf
    best_week_mae = None
    best_state = None
    epochs_no_improve = 0

    print(f'\n[Seed {seed}] Training: {args.epochs} epochs max, patience {args.patience}')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, bundle.train_loader, opt, device, scheduler)
        val_mae, week_mae = eval_epoch(model, bundle.val_loader, device)
        dt = time.time() - t0

        improved = val_mae < best_val - 1e-4
        if improved:
            best_val = val_mae
            best_week_mae = week_mae.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            tag = '★'
        else:
            epochs_no_improve += 1
            tag = ' '

        print(f'  seed {seed} | ep {epoch:3d}/{args.epochs}  tr {tr_loss:.4f}  val {val_mae:.4f}  '
              f'[w1 {week_mae[0]:.3f} w2 {week_mae[1]:.3f} w3 {week_mae[2]:.3f} '
              f'w4 {week_mae[3]:.3f} w5 {week_mae[4]:.3f}]  {dt:.0f}s  {tag}')

        if epochs_no_improve >= args.patience:
            print(f'  [Seed {seed}] early stop: no improvement for {args.patience} epochs')
            break

    if best_state is None:
        raise RuntimeError(f'Seed {seed} did not produce a valid best_state.')

    print(f'\n[Seed {seed}] Best val MAE = {best_val:.4f}')
    model.load_state_dict(best_state)

    print(f'[Seed {seed}] Predicting on test')
    test_pred = predict_test(
        model,
        bundle.test_X,
        bundle.test_regions,
        device,
        batch_size=args.batch_size,
    )
    print(f'  raw test pred: shape {test_pred.shape}, mean {test_pred.mean():.3f}, '
          f'median {np.median(test_pred):.3f}')

    return best_val, best_week_mae, test_pred


# ───────────────────────── Main ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    parser.add_argument('--out', default='baseline_v7_tcn_5seed_avg.csv')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--n-blocks', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46',
                        help="Comma-separated random seeds, e.g. '42,43,44,45,46'.")
    parser.add_argument('--save-seed-preds', action='store_true',
                        help='Also save one CSV submission per individual seed.')
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    print(f'[Seeds] {seeds}')

    # Prepare data only once. The same train/val/test split is used for all seeds.
    bundle = prepare_data(args.base, batch_size=args.batch_size, num_workers=2)

    all_test_preds = []
    seed_summaries = []

    for seed in seeds:
        best_val, best_week_mae, test_pred = train_one_seed(seed, args, bundle, device)
        all_test_preds.append(test_pred)
        seed_summaries.append((seed, best_val, best_week_mae))

        if args.save_seed_preds:
            seed_pred_clipped = np.clip(test_pred, 0.0, 5.0)
            seed_region_to_pred = {r: p for r, p in zip(bundle.test_region_ids, seed_pred_clipped)}
            seed_rows = [[r, *seed_region_to_pred[r]] for r in bundle.sub_template['region_id']]
            seed_submission = pd.DataFrame(
                seed_rows,
                columns=['region_id', 'pred_week1', 'pred_week2',
                         'pred_week3', 'pred_week4', 'pred_week5'],
            )
            seed_out = os.path.join(args.base, f'seed_{seed}_' + args.out)
            seed_submission.to_csv(seed_out, index=False)
            print(f'[Seed {seed}] Saved individual submission: {seed_out}')

    print('\n' + '=' * 80)
    print('[Validation summary by seed]')
    for seed, best_val, week_mae in seed_summaries:
        print(f'  seed {seed}: best val {best_val:.4f}  '
              f'[w1 {week_mae[0]:.3f} w2 {week_mae[1]:.3f} w3 {week_mae[2]:.3f} '
              f'w4 {week_mae[3]:.3f} w5 {week_mae[4]:.3f}]')

    # Average raw predictions first, then clip once.
    avg_preds = np.mean(np.stack(all_test_preds, axis=0), axis=0)
    avg_preds = np.clip(avg_preds, 0.0, 5.0)

    print('\n[5-seed average prediction stats]')
    print(f'  shape {avg_preds.shape}, mean {avg_preds.mean():.3f}, median {np.median(avg_preds):.3f}')

    # Build submission in sample_submission order.
    region_to_pred = {r: p for r, p in zip(bundle.test_region_ids, avg_preds)}
    rows = [[r, *region_to_pred[r]] for r in bundle.sub_template['region_id']]
    submission = pd.DataFrame(
        rows,
        columns=['region_id', 'pred_week1', 'pred_week2',
                 'pred_week3', 'pred_week4', 'pred_week5'],
    )

    out_path = os.path.join(args.base, args.out)
    submission.to_csv(out_path, index=False)

    print(f'\n[Saved ensemble submission] {out_path}')
    print('\nFirst 5 rows:')
    print(submission.head())
    print('\nPrediction stats:')
    print(submission[[f'pred_week{i}' for i in range(1, 6)]].describe().round(3))


if __name__ == '__main__':
    main()
