"""
Baseline V7 — Temporal Convolutional Network (TCN) on raw 91-day sequences.

Why a TCN
---------
After V3/V5/V6 (all tree-based on 90-117 aggregated tabular features) plateaued
at LB ≈ 0.90, we need a model that uses the SEQUENTIAL structure of the 91-day
window — not just summary statistics. A 1D CNN with dilated convolutions
(Temporal Convolutional Network) is the natural baseline for this:

- Stacks of dilated 1D convs cover the full 91-day receptive field with
  few parameters and no recurrence.
- Treats the input as a multivariate time series (14 channels, 91 timesteps),
  letting kernels jointly fuse weather variables.
- Fast to train, easy to debug, well-understood — the right "is sequence
  modeling worth pursuing?" diagnostic before the more complex P-sLSTM /
  PatchTST in V8 / V9.

Architecture
------------
  Input:  (B, 14, 91)
  ┌─ Conv1d(14 → 64, k=3, dilation=1)  + GELU + Dropout
  ├─ Residual TemporalBlock(64, dilation=2,  k=3)
  ├─ Residual TemporalBlock(64, dilation=4,  k=3)
  ├─ Residual TemporalBlock(64, dilation=8,  k=3)
  ├─ Residual TemporalBlock(64, dilation=16, k=3)
  ├─ Residual TemporalBlock(64, dilation=32, k=3)
  ├─ Global-average pool over time
  ├─ Concat with region embedding (8d)
  └─ Linear → 5 outputs (week 1..5)

  Receptive field after 5 blocks: 1 + 2*(2+4+8+16+32) = 125 days → covers
  the full 91-day window with margin.

Training
--------
- Loss: L1 (MAE) — matches Kaggle metric, robust to score outliers (5s).
- Optimizer: AdamW, lr 1e-3, weight decay 1e-4.
- Schedule: cosine annealing.
- Early stopping on val MAE with patience 8.
- Predicts all 5 weeks jointly (one forward pass).

Usage
-----
    python baseline_v7_tcn.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v7_tcn.csv \
        --epochs 60
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


# ───────────────────────── Main ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    parser.add_argument('--out', default='baseline_v7_tcn.csv')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--n-blocks', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    bundle = prepare_data(args.base, batch_size=args.batch_size, num_workers=2)
    print(f'\n[Model]')
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
    best_state = None
    epochs_no_improve = 0
    history = []

    print(f'\n[Training] {args.epochs} epochs max, patience {args.patience}')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, bundle.train_loader, opt, device, scheduler)
        val_mae, week_mae = eval_epoch(model, bundle.val_loader, device)
        dt = time.time() - t0
        history.append((epoch, tr_loss, val_mae, *week_mae))

        improved = val_mae < best_val - 1e-4
        if improved:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            tag = '★'
        else:
            epochs_no_improve += 1
            tag = ' '

        print(f'  ep {epoch:3d}/{args.epochs}  tr {tr_loss:.4f}  val {val_mae:.4f}  '
              f'[w1 {week_mae[0]:.3f} w2 {week_mae[1]:.3f} w3 {week_mae[2]:.3f} '
              f'w4 {week_mae[3]:.3f} w5 {week_mae[4]:.3f}]  {dt:.0f}s  {tag}')

        if epochs_no_improve >= args.patience:
            print(f'  [early stop] no improvement for {args.patience} epochs')
            break

    print(f'\n[Best] val MAE = {best_val:.4f}')
    model.load_state_dict(best_state)

    print('\n[Predicting on test]')
    raw_preds = predict_test(model, bundle.test_X, bundle.test_regions, device,
                             batch_size=args.batch_size)
    raw_preds = np.clip(raw_preds, 0.0, 5.0)
    print(f'  test preds: shape {raw_preds.shape}, '
          f'mean {raw_preds.mean():.3f}, median {np.median(raw_preds):.3f}')

    # Build submission in sample_submission order
    region_to_pred = {r: p for r, p in zip(bundle.test_region_ids, raw_preds)}
    rows = [[r, *region_to_pred[r]] for r in bundle.sub_template['region_id']]
    submission = pd.DataFrame(
        rows,
        columns=['region_id', 'pred_week1', 'pred_week2',
                 'pred_week3', 'pred_week4', 'pred_week5'],
    )
    out_path = os.path.join(args.base, args.out)
    submission.to_csv(out_path, index=False)
    print(f'\n[Saved] {out_path}')
    print('\nFirst 5 rows:')
    print(submission.head())
    print('\nPrediction stats:')
    print(submission[[f'pred_week{i}' for i in range(1, 6)]].describe().round(3))


if __name__ == '__main__':
    main()
