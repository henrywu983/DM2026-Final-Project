"""
Baseline DLinear — multi-seed averaged DLinear (Zeng et al. 2023).

Trains the same DLinear architecture multiple times with different random seeds,
predicts the test set with each trained model, averages the raw test predictions,
clips to [0, 5], and saves one Kaggle submission file.

DLinear is the simplest possible time-series forecaster: moving-average
decomposition into trend + seasonal, then two per-channel linear layers
(91 → 5) summed back. Chosen here as a 4th architecture family — orthogonal
inductive bias to V7/V10 (CNN), V8/V8.1 (RNN), V9 (attention).

Default seeds: 42, 43, 44, 45, 46. Override with --seeds (comma list).

Usage
-----
    # 5-seed (default)
    python baseline_dlinear_5seed_avg.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out baseline_dlinear_5seed_avg.csv

    # Single-seed phase-1 sanity check (~30 min on T4)
    python baseline_dlinear_5seed_avg.py --seeds 42 --out baseline_dlinear_seed42.csv

    # 10-seed scaled up (if phase 2 passes and we want a stronger version)
    python baseline_dlinear_5seed_avg.py --seeds 42,43,44,45,46,47,48,49,50,51 \
        --out baseline_dlinear_10seed_avg.csv

    # No region embedding (cleanest "pure DLinear" architectural-diversity variant)
    python baseline_dlinear_5seed_avg.py --region-emb-dim 0

Notes
-----
- The data pipeline is prepared once and shared across all seeds.
- Each seed trains a fresh DLinear model from scratch.
- Early stopping is applied independently for each seed.
- Final prediction is the arithmetic mean of all seed predictions, clipped once.
- Matches V7's forward(x, region) signature so the training loop is unchanged.
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

class MovingAvg(nn.Module):
    """
    Moving-average smoother used in DLinear's series decomposition.

    Applies a length-`kernel` average pool over the time axis with edge padding
    to keep the output length equal to the input length (91 here).
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f'kernel_size must be odd, got {kernel_size}')
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        Returns: (B, T, C) — same length as input via edge-replication padding.
        """
        pad = (self.kernel_size - 1) // 2
        # Replicate-pad at both ends along the time axis so the moving average
        # is well-defined at the boundaries (matches the original DLinear code).
        front = x[:, :1, :].repeat(1, pad, 1)
        back = x[:, -1:, :].repeat(1, pad, 1)
        x_padded = torch.cat([front, x, back], dim=1)            # (B, T+2*pad, C)
        x_padded = x_padded.transpose(1, 2)                      # (B, C, T+2*pad)
        smoothed = self.avg(x_padded)                            # (B, C, T)
        return smoothed.transpose(1, 2)                          # (B, T, C)


class SeriesDecomp(nn.Module):
    """Decompose a series x into (seasonal, trend) where trend = moving_avg(x)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.ma = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C) → (seasonal: (B, T, C), trend: (B, T, C))
        """
        trend = self.ma(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear (Zeng et al. 2023, "Are Transformers Effective for Time Series Forecasting?").

    Architecture:
      1. Decompose input into trend + seasonal via moving average.
      2. Two Linear(seq_len → pred_len) heads — one for each component.
      3. Sum the two head outputs.
      4. Average across channels (channel-independent variant) OR use shared
         per-channel weights (the paper's `Individual=False` setting).

    `individual=True` gives each channel its own Linear(91→5) pair (~13K params).
    `individual=False` shares one Linear(91→5) pair across all channels (~910 params).
    The paper finds `individual=True` typically wins on multi-channel data.

    Optional region embedding head adds a learnable per-region bias (B, 5) on
    top of the DLinear output. Set `region_emb_dim=0` to disable entirely.
    """

    def __init__(
        self,
        n_features: int = 14,
        seq_len: int = 91,
        pred_len: int = 5,
        kernel_size: int = 25,
        individual: bool = True,
        n_regions: int = 2248,
        region_emb_dim: int = 8,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.region_emb_dim = region_emb_dim

        self.decomp = SeriesDecomp(kernel_size)

        if individual:
            # One Linear(91→5) per channel, per component (trend & seasonal).
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
        else:
            # Shared across channels.
            self.linear_trend = nn.Linear(seq_len, pred_len)
            self.linear_seasonal = nn.Linear(seq_len, pred_len)

        # Optional region embedding head — adds a per-region (B, 5) bias.
        if region_emb_dim > 0:
            self.region_emb = nn.Embedding(n_regions, region_emb_dim)
            self.region_head = nn.Linear(region_emb_dim, pred_len)
            # Zero-init the region head so it starts as a no-op and DLinear
            # only learns to use it if it's actually useful.
            nn.init.zeros_(self.region_head.weight)
            nn.init.zeros_(self.region_head.bias)
        else:
            self.region_emb = None
            self.region_head = None

    def forward(self, x: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, 91, 14)
        region: (B,) long
        Returns (B, 5)
        """
        seasonal, trend = self.decomp(x)                  # both (B, T, C)

        # Apply per-channel linear maps along the time axis.
        # We need to feed each channel a (B, T) input and get (B, pred_len).
        if self.individual:
            # (B, T, C) → list of (B, T) → per-channel (B, pred_len) → stack
            # Result: (B, pred_len, C)
            outs = []
            for c in range(self.n_features):
                s = self.linear_seasonal[c](seasonal[:, :, c])    # (B, pred_len)
                t = self.linear_trend[c](trend[:, :, c])          # (B, pred_len)
                outs.append(s + t)
            stacked = torch.stack(outs, dim=-1)                   # (B, pred_len, C)
        else:
            # Shared head: rearrange to (B*C, T), apply, rearrange back.
            B, T, C = x.shape
            s = seasonal.transpose(1, 2).reshape(B * C, T)        # (B*C, T)
            t = trend.transpose(1, 2).reshape(B * C, T)
            s_out = self.linear_seasonal(s).reshape(B, C, self.pred_len)
            t_out = self.linear_trend(t).reshape(B, C, self.pred_len)
            stacked = (s_out + t_out).transpose(1, 2)             # (B, pred_len, C)

        # Average across channels → (B, pred_len). This is the "ensemble-of-channels"
        # interpretation that composes well with the pipeline's per-region RevIN.
        out = stacked.mean(dim=-1)                                # (B, 5)

        # Optional additive region bias.
        if self.region_emb is not None:
            emb = self.region_emb(region)                         # (B, region_emb_dim)
            out = out + self.region_head(emb)                     # (B, 5)

        return out


# ───────────────────────── Training ─────────────────────────

def train_epoch(model, loader, opt, device, scheduler=None):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, r, y in loader:
        x = x.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
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
        x = test_X[i:i + batch_size].to(device)
        r = test_regions[i:i + batch_size].to(device)
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
    Train one DLinear model with a given seed.

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
    print(f'[Seed {seed}] Training fresh DLinear')
    print('=' * 80)

    model = DLinear(
        n_features=bundle.n_features,
        seq_len=91,
        pred_len=5,
        kernel_size=args.kernel_size,
        individual=args.individual,
        n_regions=bundle.n_regions,
        region_emb_dim=args.region_emb_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  DLinear: kernel={args.kernel_size}, individual={args.individual}, '
          f'region_emb_dim={args.region_emb_dim}, {n_params:,} params')

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
    # Phase-1 sanity diagnostic: per-week prediction means.
    week_means = test_pred.mean(axis=0)
    print(f'  raw test pred: shape {test_pred.shape}, mean {test_pred.mean():.3f}, '
          f'median {np.median(test_pred):.3f}')
    print(f'  per-week pred means: [w1 {week_means[0]:.3f} w2 {week_means[1]:.3f} '
          f'w3 {week_means[2]:.3f} w4 {week_means[3]:.3f} w5 {week_means[4]:.3f}]')

    return best_val, best_week_mae, test_pred


# ───────────────────────── Main ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    parser.add_argument('--out', default='baseline_dlinear_5seed_avg.csv')

    # Training schedule
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)

    # DLinear-specific
    parser.add_argument('--kernel-size', type=int, default=25,
                        help='Moving-average kernel size for trend extraction. '
                             'Must be odd. 25 ≈ 4-week smoothing.')
    parser.add_argument('--individual', action='store_true', default=True,
                        help='Per-channel linear heads (default). The paper finds '
                             'this typically beats shared weights on multi-channel data.')
    parser.add_argument('--shared', dest='individual', action='store_false',
                        help='Use shared linear heads across all channels '
                             '(the paper\'s Individual=False variant).')
    parser.add_argument('--region-emb-dim', type=int, default=8,
                        help='Region embedding dim. Set to 0 to disable the '
                             'region head entirely (cleanest "pure DLinear" variant).')

    # Multi-seed
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46',
                        help="Comma-separated random seeds, e.g. '42,43,44,45,46'. "
                             "For phase-1 sanity check use '42'.")
    parser.add_argument('--save-seed-preds', action='store_true',
                        help='Also save one CSV submission per individual seed.')

    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    print(f'[Seeds] {seeds}  (n={len(seeds)})')
    print(f'[Model] DLinear kernel={args.kernel_size} individual={args.individual} '
          f'region_emb_dim={args.region_emb_dim}')

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
    raw_week_means = avg_preds.mean(axis=0)
    avg_preds = np.clip(avg_preds, 0.0, 5.0)

    print(f'\n[{len(seeds)}-seed average prediction stats]')
    print(f'  shape {avg_preds.shape}, mean {avg_preds.mean():.3f}, median {np.median(avg_preds):.3f}')
    print(f'  per-week means (pre-clip): [w1 {raw_week_means[0]:.3f} w2 {raw_week_means[1]:.3f} '
          f'w3 {raw_week_means[2]:.3f} w4 {raw_week_means[3]:.3f} w5 {raw_week_means[4]:.3f}]')
    print(f'  acceptance check: V7 reference mean ~1.05; want each week in 0.95-1.20.')

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
