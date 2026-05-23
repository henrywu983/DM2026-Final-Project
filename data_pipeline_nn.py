"""
Shared data pipeline for NN models (TCN, P-sLSTM, PatchTST).

What this provides
------------------
1. Window construction (91-day inputs, 5 weekly targets per window).
2. Per-region instance normalization (RevIN-style): each window is
   normalized by the mean/std of THAT region computed over training. Drought
   is fundamentally about deviation from regional norms — feeding raw weather
   values to a NN obscures this signal, so we standardize before modeling.
3. Time-based train/val split with buffer (same as V3/V5/V6).
4. PyTorch Dataset and DataLoader wrappers.
5. Region encoding (for embedding layers).

What this does NOT provide
--------------------------
- Model definitions (TCN/P-sLSTM/PatchTST live in their own files)
- Training loop (lives in train_nn.py)

Usage in model scripts
----------------------
    from data_pipeline_nn import prepare_data

    bundle = prepare_data(base_dir='/path/to/data')
    train_loader = bundle['train_loader']
    val_loader = bundle['val_loader']
    test_inputs = bundle['test_inputs']
    n_regions = bundle['n_regions']
    ...

Notes
-----
- All data lives on CPU after preparation; the training loop moves batches
  to GPU. This keeps the prep code device-agnostic.
- Per-region normalization is FIT on training portion only, then applied
  to validation and test windows of that region. No leakage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ───────────────────────── Config ─────────────────────────

FEATURE_COLS: List[str] = [
    'prec', 'surf_pre', 'humidity', 'tmp', 'dp_tmp', 'wb_tmp',
    'tmp_max', 'tmp_min', 'tmp_range', 'surf_tmp',
    'wind', 'wind_max', 'wind_min', 'wind_range',
]
WINDOW_DAYS = 91
TARGET_OFFSETS = [7, 14, 21, 28, 35]
STRIDE = 35
N_VAL_WINDOWS = 5
N_BUFFER_WINDOWS = 3


# ───────────────────────── Loading ─────────────────────────

def load_raw(base: str):
    train = pd.read_csv(os.path.join(base, 'data', 'train.csv'))
    test = pd.read_csv(os.path.join(base, 'data', 'test.csv'))
    sub = pd.read_csv(os.path.join(base, 'sample_submission.csv'))
    train = train.sort_values(['region_id', 'date']).reset_index(drop=True)
    test = test.sort_values(['region_id', 'date']).reset_index(drop=True)
    return train, test, sub


# ───────────────────────── Per-region normalization ─────────────────────────

def compute_region_stats(train: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute per-region mean and std for each weather feature over the full
    training timeline. Used to normalize windows of that region.
    """
    print('  Computing per-region normalization stats...')
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for region_id, group in train.groupby('region_id', sort=False):
        feats = group[FEATURE_COLS].values.astype(np.float32)
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std = np.where(std < 1e-6, 1e-6, std)  # avoid div-by-zero
        stats[region_id] = (mean, std)
    return stats


def normalize_window(window: np.ndarray, region_id: str,
                     stats: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """(91, 14) window normalized using that region's train-time stats."""
    mean, std = stats[region_id]
    return (window - mean) / std


# ───────────────────────── Window construction ─────────────────────────

def build_windows(
    df: pd.DataFrame,
    region_to_int: Dict[str, int],
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    is_train: bool,
) -> Dict:
    """
    Returns dict with keys:
      X:        (N, 91, 14) float32 — normalized weather inputs
      y:        (N, 5) float32 — targets (only for train)
      regions:  (N,) int64 — region integer IDs
      win_idx:  (N,) int32 — per-region chronological window index (for splitting)
    """
    X_list, y_list, region_list, win_idx_list = [], [], [], []
    n_regions = df['region_id'].nunique()
    print(f'  Building {"train" if is_train else "test"} windows '
          f'across {n_regions} regions...')

    for region_id, group in df.groupby('region_id', sort=False):
        feat_arr = group[FEATURE_COLS].values.astype(np.float32)
        n = len(group)

        if is_train:
            score_arr = group['score'].values
            end_days = np.arange(WINDOW_DAYS - 1, n - max(TARGET_OFFSETS), STRIDE)
        else:
            end_days = [n - 1]   # one window per test region (the full 91 days)

        for wi, e in enumerate(end_days):
            raw_window = feat_arr[e - WINDOW_DAYS + 1: e + 1]
            norm_window = normalize_window(raw_window, region_id, stats)
            X_list.append(norm_window)
            region_list.append(region_to_int[region_id])
            win_idx_list.append(wi)

            if is_train:
                targets = np.array([score_arr[e + off] for off in TARGET_OFFSETS],
                                   dtype=np.float32)
                y_list.append(targets)

    X = np.stack(X_list).astype(np.float32)
    regions = np.array(region_list, dtype=np.int64)
    win_idx = np.array(win_idx_list, dtype=np.int32)
    out = {'X': X, 'regions': regions, 'win_idx': win_idx}
    if is_train:
        out['y'] = np.stack(y_list).astype(np.float32)
    return out


# ───────────────────────── Train / val split ─────────────────────────

def split_with_buffer(bundle: Dict,
                      n_val: int = N_VAL_WINDOWS,
                      n_buf: int = N_BUFFER_WINDOWS):
    """Hold out the last n_val windows per region; drop n_buf preceding ones as buffer."""
    regions = bundle['regions']
    # Per-region index lists (already chronological because we built sequentially)
    region_to_indices: Dict[int, List[int]] = {}
    for i, r in enumerate(regions):
        region_to_indices.setdefault(int(r), []).append(i)

    train_mask = np.zeros(len(regions), dtype=bool)
    val_mask = np.zeros(len(regions), dtype=bool)
    for r, idxs in region_to_indices.items():
        if len(idxs) < n_val + n_buf:
            train_mask[idxs] = True
            continue
        train_mask[idxs[:-(n_val + n_buf)]] = True
        val_mask[idxs[-n_val:]] = True

    n_dropped = (~train_mask & ~val_mask).sum()
    print(f'  train: {train_mask.sum():,}, val: {val_mask.sum():,}, '
          f'buffer dropped: {n_dropped:,}')
    return train_mask, val_mask


# ───────────────────────── PyTorch Dataset ─────────────────────────

class DroughtDataset(Dataset):
    def __init__(self, X: np.ndarray, regions: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X)        # (N, 91, 14)
        self.regions = torch.from_numpy(regions)  # (N,)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        r = self.regions[idx]
        if self.y is not None:
            return x, r, self.y[idx]
        return x, r


# ───────────────────────── Top-level preparation ─────────────────────────

@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_X: torch.Tensor          # (n_test_regions, 91, 14)
    test_regions: torch.Tensor    # (n_test_regions,)
    test_region_ids: np.ndarray   # original string IDs in submission order
    sub_template: pd.DataFrame
    n_regions: int
    n_features: int


def prepare_data(base: str,
                 batch_size: int = 512,
                 num_workers: int = 2) -> DataBundle:
    """
    Top-level entry point. Returns a DataBundle ready for any of the
    NN training scripts.
    """
    print('[Loading raw data]')
    train, test, sub = load_raw(base)
    print(f'  train: {train.shape}, test: {test.shape}')

    print('[Region encoding]')
    all_regions = sorted(set(train['region_id']) | set(test['region_id']))
    region_to_int = {r: i for i, r in enumerate(all_regions)}
    n_regions = len(region_to_int)

    print('[Normalization stats]')
    stats = compute_region_stats(train)

    print('[Building training windows]')
    train_bundle = build_windows(train, region_to_int, stats, is_train=True)
    print(f'  X: {train_bundle["X"].shape}, y: {train_bundle["y"].shape}')

    print('[Train/val split with buffer]')
    train_mask, val_mask = split_with_buffer(train_bundle)

    train_ds = DroughtDataset(
        train_bundle['X'][train_mask],
        train_bundle['regions'][train_mask],
        train_bundle['y'][train_mask],
    )
    val_ds = DroughtDataset(
        train_bundle['X'][val_mask],
        train_bundle['regions'][val_mask],
        train_bundle['y'][val_mask],
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print('[Building test windows]')
    test_bundle = build_windows(test, region_to_int, stats, is_train=False)
    test_X = torch.from_numpy(test_bundle['X'])
    test_regions = torch.from_numpy(test_bundle['regions'])

    # Recover original string IDs in the order they appear in test_X
    int_to_region = {i: r for r, i in region_to_int.items()}
    test_region_ids = np.array([int_to_region[int(r)] for r in test_bundle['regions']])

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_X=test_X,
        test_regions=test_regions,
        test_region_ids=test_region_ids,
        sub_template=sub,
        n_regions=n_regions,
        n_features=len(FEATURE_COLS),
    )


if __name__ == '__main__':
    # Smoke test
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else \
        '/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project'
    bundle = prepare_data(base, batch_size=128, num_workers=0)
    print(f'\n✓ Prepared {bundle.n_regions} regions, {bundle.n_features} features')
    print(f'✓ Train batches: {len(bundle.train_loader)}')
    print(f'✓ Val batches: {len(bundle.val_loader)}')
    print(f'✓ Test X: {bundle.test_X.shape}')

    # Inspect one batch
    for x, r, y in bundle.train_loader:
        print(f'✓ Sample batch: x={x.shape}, r={r.shape}, y={y.shape}')
        print(f'  x normalized? mean={x.mean():.3f}, std={x.std():.3f} (expect ≈ 0, 1)')
        break
