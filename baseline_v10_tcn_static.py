"""
Baseline V10 — TCN backbone fused with V5 static features.

Motivation
----------
V5 (LightGBM on 105 static features) and V7 (TCN on raw 91-day sequences)
each capture different signal:
  - V5 has long-term per-region drought baselines and climatology anomalies
  - V7 has short-term sequence dynamics and intra-window patterns
Their prediction correlation is 0.73 — meaning ~50% of variance is shared
but each model has unique signal the other lacks. The ensemble of their
predictions gains some (LB 0.8463 → 0.8353), but ensembling at the output
level can't recover signal that requires JOINT use of both feature types.

V10 fuses both signal types INSIDE a single model: the TCN extracts a
sequence representation, V5's full static feature set provides the
regional/climatology baseline, and a small MLP head decides how to combine
them for each prediction horizon.

Architecture
------------
  Input: (B, 91, 14) raw weather + (B, 105) V5 static features
                                                   │
  ┌─ TCN backbone (V7 config: 5 blocks × 64 ch) ─→ pooled features (B, 64)
  └─ Static feature norm + Linear(105→64) ─────→ static repr (B, 64)
                                                   │
                                concat → (B, 128) + region embedding (B, 8)
                                                   │
                                          MLP head → (B, 5)

Key choices
-----------
- TCN config identical to V7 (the sweet spot for sequence capacity).
- Static features normalized per-feature using training statistics. Reused
  V5's full 105-feature set (105 - 2 categoricals = 103 numeric features
  + region embedding as before).
- Categorical features (month, region_int) are excluded from the static
  block — region_int is already in the embedding, month is captured by the
  TCN's date-aware features implicitly through window structure.

Usage
-----
    python baseline_v10_tcn_static.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v10_tcn_static.csv

Requires baseline_v5_lgbm_score_history.py to be importable for feature
extraction; this script reuses its feature-building functions.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reuse V7's TCN block and V5's static feature extraction
from data_pipeline_nn import (  # noqa: E402
    FEATURE_COLS, WINDOW_DAYS, TARGET_OFFSETS, STRIDE,
    N_VAL_WINDOWS, N_BUFFER_WINDOWS,
    load_raw, compute_region_stats, normalize_window, split_with_buffer,
)
from baseline_v7_tcn import TemporalBlock  # noqa: E402
from baseline_v5_lgbm_score_history import (  # noqa: E402
    build_climatology, build_score_lookups,
    aggregate_window as v5_aggregate_window,
    make_feature_names as v5_make_feature_names,
)


# ───────────────────────── Config ─────────────────────────

# Static features: V5's 105 minus 2 categoricals (month, region_int)
V5_FEATURE_NAMES = v5_make_feature_names()
STATIC_FEATURE_NAMES = [n for n in V5_FEATURE_NAMES if n not in ('month', 'region_int')]
N_STATIC = len(STATIC_FEATURE_NAMES)


# ───────────────────────── Joint dataset ─────────────────────────

class JointDataset(Dataset):
    """Holds (X_seq, X_static, region, y) tuples."""

    def __init__(self, X_seq, X_static, regions, y=None):
        self.X_seq = torch.from_numpy(X_seq)
        self.X_static = torch.from_numpy(X_static)
        self.regions = torch.from_numpy(regions)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, i):
        if self.y is not None:
            return self.X_seq[i], self.X_static[i], self.regions[i], self.y[i]
        return self.X_seq[i], self.X_static[i], self.regions[i]


# ───────────────────────── Window construction ─────────────────────────

def build_joint_windows(
    df: pd.DataFrame,
    region_to_int: Dict[str, int],
    norm_stats: Dict,
    is_train: bool,
    clim_mean, clim_std, rs_mean, rms_mean, rls_mean, gsm,
):
    """Build (X_seq, X_static, regions, y, win_idx) jointly per region."""
    X_seq_list, X_static_list, region_list, y_list, win_idx_list = [], [], [], [], []
    n_regions = df['region_id'].nunique()
    print(f'  Building {"train" if is_train else "test"} windows across {n_regions} regions...')
    t0 = time.time()

    for ri, (region_id, group) in enumerate(df.groupby('region_id', sort=False)):
        feat_arr = group[FEATURE_COLS].values.astype(np.float32)
        date_strs = group['date'].values
        n = len(group)

        if is_train:
            score_arr = group['score'].values
            end_days = np.arange(WINDOW_DAYS - 1, n - max(TARGET_OFFSETS), STRIDE)
        else:
            end_days = [n - 1]

        for wi, e in enumerate(end_days):
            raw_window = feat_arr[e - WINDOW_DAYS + 1: e + 1]
            # Sequence input: per-region normalized
            seq_norm = normalize_window(raw_window, region_id, norm_stats)
            # Static features: V5's per-window aggregate (includes categoricals at end)
            month_e = int(date_strs[e][5:7])
            static_full = v5_aggregate_window(
                raw_window, month_e, region_id,
                clim_mean, clim_std, rs_mean, rms_mean, rls_mean, gsm,
            )
            # Drop the last 2 entries — those are categoricals (handled separately)
            # Actually V5's aggregate_window returns ONLY numeric features (categoricals
            # appended later in build_train_windows). So static_full IS pure numeric.
            # Length check below confirms.
            X_seq_list.append(seq_norm)
            X_static_list.append(static_full)
            region_list.append(region_to_int[region_id])
            win_idx_list.append(wi)
            if is_train:
                targets = np.array([score_arr[e + off] for off in TARGET_OFFSETS],
                                   dtype=np.float32)
                y_list.append(targets)

        if (ri + 1) % 500 == 0:
            print(f'    {ri + 1}/{n_regions} regions ({time.time() - t0:.0f}s)')

    X_seq = np.stack(X_seq_list).astype(np.float32)
    X_static = np.vstack(X_static_list).astype(np.float32)
    regions = np.array(region_list, dtype=np.int64)
    win_idx = np.array(win_idx_list, dtype=np.int32)

    if X_static.shape[1] != N_STATIC:
        raise RuntimeError(
            f'Static feature count mismatch: got {X_static.shape[1]}, expected {N_STATIC}. '
            f'Check that v5_aggregate_window returns pure numeric features.'
        )

    out = {'X_seq': X_seq, 'X_static': X_static, 'regions': regions, 'win_idx': win_idx}
    if is_train:
        out['y'] = np.stack(y_list).astype(np.float32)
    return out


# ───────────────────────── Static feature normalization ─────────────────────────

def fit_static_normalization(X_static: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standard z-score normalization across the training static features."""
    mean = X_static.mean(axis=0)
    std = X_static.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_static_normalization(X_static: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X_static - mean) / std).astype(np.float32)


# ───────────────────────── Model ─────────────────────────

class TCN_Static(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
        n_static: int = N_STATIC,
        channels: int = 64,
        n_blocks: int = 5,
        static_dim: int = 64,
        n_regions: int = 2248,
        region_emb_dim: int = 8,
        n_outputs: int = 5,
        kernel: int = 3,
        dropout: float = 0.15,
        static_dropout: float = 0.2,
    ):
        super().__init__()
        # Sequence path (V7 TCN config)
        self.input_proj = nn.Conv1d(n_features, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            TemporalBlock(channels, kernel=kernel, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])

        # Static path
        self.static_norm = nn.LayerNorm(n_static)
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, static_dim),
            nn.GELU(),
            nn.Dropout(static_dropout),
        )

        # Region embedding (same dim as V7)
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)

        # Fusion head
        fusion_dim = channels + static_dim + region_emb_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, n_outputs),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor,
                region: torch.Tensor) -> torch.Tensor:
        # Sequence branch
        x = x_seq.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        seq_repr = x.mean(dim=2)  # (B, channels)

        # Static branch
        static_repr = self.static_proj(self.static_norm(x_static))

        # Region embedding
        emb = self.region_emb(region)

        # Fuse and predict
        h = torch.cat([seq_repr, static_repr, emb], dim=1)
        return self.head(h)


# ───────────────────────── Training ─────────────────────────

def train_epoch(model, loader, opt, device, scheduler=None):
    model.train()
    total_loss, n_samples = 0.0, 0
    for x_seq, x_static, r, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        pred = model(x_seq, x_static, r)
        loss = F.l1_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(x_seq)
        n_samples += len(x_seq)
    return total_loss / n_samples


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_y = [], []
    for x_seq, x_static, r, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        pred = model(x_seq, x_static, r).cpu().numpy()
        all_preds.append(pred)
        all_y.append(y.numpy())
    P = np.concatenate(all_preds)
    Y = np.concatenate(all_y)
    week_mae = np.abs(P - Y).mean(axis=0)
    return float(week_mae.mean()), week_mae


@torch.no_grad()
def predict_test(model, X_seq, X_static, regions, device, batch_size=512):
    model.eval()
    preds = []
    n = len(X_seq)
    for i in range(0, n, batch_size):
        x_seq = X_seq[i:i+batch_size].to(device)
        x_static = X_static[i:i+batch_size].to(device)
        r = regions[i:i+batch_size].to(device)
        p = model(x_seq, x_static, r).cpu().numpy()
        preds.append(p)
    return np.concatenate(preds)


# ───────────────────────── Main ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    parser.add_argument('--out', default='baseline_v10_tcn_static.csv')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--n-blocks', type=int, default=5)
    parser.add_argument('--static-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--static-dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    print(f'[V10 config] channels={args.channels}, n_blocks={args.n_blocks}, '
          f'static_dim={args.static_dim}, n_static={N_STATIC}')

    # Load data
    print('\n[Loading raw data]')
    train, test, sub = load_raw(args.base)
    print(f'  train: {train.shape}, test: {test.shape}')

    all_regions = sorted(set(train['region_id']) | set(test['region_id']))
    region_to_int = {r: i for i, r in enumerate(all_regions)}
    n_regions = len(region_to_int)

    print('\n[Sequence normalization stats]')
    seq_stats = compute_region_stats(train)

    print('\n[V5 static-feature lookup tables]')
    clim_mean, clim_std = build_climatology(train)
    rs_mean, rms_mean, rls_mean = build_score_lookups(train)
    gsm = float(train['score'].dropna().mean())

    print('\n[Building joint training windows]')
    train_bundle = build_joint_windows(
        train, region_to_int, seq_stats, is_train=True,
        clim_mean=clim_mean, clim_std=clim_std,
        rs_mean=rs_mean, rms_mean=rms_mean, rls_mean=rls_mean, gsm=gsm,
    )
    print(f'  X_seq: {train_bundle["X_seq"].shape}, '
          f'X_static: {train_bundle["X_static"].shape}, '
          f'y: {train_bundle["y"].shape}')

    print('\n[Train/val split with buffer]')
    fake_bundle = {'X': train_bundle['X_seq'], 'regions': train_bundle['regions']}
    train_mask, val_mask = split_with_buffer(fake_bundle)

    # Fit static normalization on the training portion only (avoid leakage)
    print('\n[Static feature normalization]')
    static_mean, static_std = fit_static_normalization(train_bundle['X_static'][train_mask])
    X_static_norm = apply_static_normalization(train_bundle['X_static'], static_mean, static_std)

    train_ds = JointDataset(
        train_bundle['X_seq'][train_mask],
        X_static_norm[train_mask],
        train_bundle['regions'][train_mask],
        train_bundle['y'][train_mask],
    )
    val_ds = JointDataset(
        train_bundle['X_seq'][val_mask],
        X_static_norm[val_mask],
        train_bundle['regions'][val_mask],
        train_bundle['y'][val_mask],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f'  train batches: {len(train_loader)}, val batches: {len(val_loader)}')

    # Test windows
    print('\n[Building test windows]')
    test_bundle = build_joint_windows(
        test, region_to_int, seq_stats, is_train=False,
        clim_mean=clim_mean, clim_std=clim_std,
        rs_mean=rs_mean, rms_mean=rms_mean, rls_mean=rls_mean, gsm=gsm,
    )
    test_X_seq = torch.from_numpy(test_bundle['X_seq'])
    test_X_static = torch.from_numpy(
        apply_static_normalization(test_bundle['X_static'], static_mean, static_std)
    )
    test_regions = torch.from_numpy(test_bundle['regions'])
    int_to_region = {i: r for r, i in region_to_int.items()}
    test_region_ids = np.array([int_to_region[int(r)] for r in test_bundle['regions']])

    print(f'\n[Model]')
    model = TCN_Static(
        n_features=len(FEATURE_COLS),
        n_static=N_STATIC,
        channels=args.channels,
        n_blocks=args.n_blocks,
        static_dim=args.static_dim,
        n_regions=n_regions,
        dropout=args.dropout,
        static_dropout=args.static_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  TCN+Static: TCN({args.n_blocks}x{args.channels}) + Static({N_STATIC}→{args.static_dim}), '
          f'{n_params:,} params')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_val = math.inf
    best_state = None
    epochs_no_improve = 0

    print(f'\n[Training] {args.epochs} epochs max, patience {args.patience}')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, opt, device, scheduler)
        val_mae, week_mae = eval_epoch(model, val_loader, device)
        dt = time.time() - t0

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
    raw_preds = predict_test(model, test_X_seq, test_X_static, test_regions,
                             device, batch_size=args.batch_size)
    raw_preds = np.clip(raw_preds, 0.0, 5.0)
    print(f'  test preds: mean {raw_preds.mean():.3f}, median {np.median(raw_preds):.3f}')

    region_to_pred = {r: p for r, p in zip(test_region_ids, raw_preds)}
    rows = [[r, *region_to_pred[r]] for r in sub['region_id']]
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
