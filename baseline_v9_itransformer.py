"""
Baseline V9 — iTransformer for drought severity forecasting.

Based on
--------
Liu et al., "iTransformer: Inverted Transformers Are Effective for Time
Series Forecasting", ICLR 2024.

The core idea
-------------
Conventional time-series transformers treat each TIME STEP as a token
and all features as that token's embedding dim. iTransformer flips this:
each FEATURE/VARIABLE becomes a token, and the variable's full time
sequence becomes its embedding. Self-attention then runs ACROSS
VARIABLES, learning which variables interact with which.

Why this is the right architecture for our problem
--------------------------------------------------
We have 14 highly-correlated meteorological variables (tmp, tmp_max,
tmp_min, dp_tmp, wb_tmp, surf_tmp are all temperature measurements;
prec and humidity interact strongly). The V8 P-sLSTM failed because
channel independence threw away these cross-variable patterns.
iTransformer is the architectural opposite: it makes cross-variable
attention the centerpiece. If the V8 failure was indeed due to
channel-independent processing, iTransformer should succeed where it
failed.

Architecture
------------
  Input:  (B, 91, 14) — 91 days × 14 channels
  ├─ Transpose to (B, 14, 91) — each channel becomes a token
  ├─ Linear embed: 91 → d_model (project each channel's full time series)
  ├─ Add LayerNorm + dropout
  ├─ N × TransformerEncoderLayer (self-attention across the 14 tokens)
  ├─ Linear projection: d_model → 5 (one prediction head per channel)
  ├─ Mean across the 14 channel predictions → (B, 5)
  └─ Add region embedding contribution → (B, 5)

Key shapes
----------
  After embed:    (B, 14, d_model)
  After attention: (B, 14, d_model)  — attention is over the 14-dim axis
  After project:   (B, 14, 5)
  After mean:      (B, 5)

The attention sequence length is just 14 (number of variables), so
self-attention is extremely cheap — O(14²) per layer rather than O(91²)
as in conventional transformers. This makes iTransformer faster than
both PatchTST and our P-sLSTM despite using full attention.

Usage
-----
    python baseline_v9_itransformer.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v9_itransformer.csv
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
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from data_pipeline_nn import prepare_data  # noqa: E402


# ───────────────────────── Building blocks ─────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Standard pre-norm transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_tokens, d_model)
        a = self.norm1(x)
        a, _ = self.attn(a, a, a, need_weights=False)
        x = x + self.drop1(a)
        x = x + self.ff(self.norm2(x))
        return x


# ───────────────────────── iTransformer ─────────────────────────

class iTransformer(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
        seq_len: int = 91,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        n_regions: int = 2248,
        region_emb_dim: int = 16,
        n_outputs: int = 5,
        dropout: float = 0.15,
    ):
        super().__init__()
        # Each variable's 91-day sequence is projected to d_model
        self.embed = nn.Linear(seq_len, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        # N stacked encoder layers
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Per-channel forecast head: d_model → n_outputs
        self.head = nn.Linear(d_model, n_outputs)

        # Region embedding (parallel head)
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        self.region_head = nn.Linear(region_emb_dim, n_outputs)

    def forward(self, x: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, 91, 14)
        region: (B,)
        """
        # Transpose so each channel becomes a token: (B, 14, 91)
        x = x.transpose(1, 2)

        # Embed each channel's 91-day series to d_model
        h = self.embed(x)                       # (B, 14, d_model)
        h = self.embed_drop(self.embed_norm(h))

        # Transformer encoder — attention is over the 14 channel tokens
        for layer in self.encoder:
            h = layer(h)
        h = self.norm(h)                        # (B, 14, d_model)

        # Per-channel forecasts then average across channels
        out = self.head(h)                      # (B, 14, n_outputs)
        out = out.mean(dim=1)                   # (B, n_outputs)

        # Region embedding residual
        emb_out = self.region_head(self.region_emb(region))
        return out + emb_out


# ───────────────────────── Training (reuse) ─────────────────────────

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
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_y = [], []
    for x, r, y in loader:
        x = x.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        pred = model(x, r).cpu().numpy()
        all_preds.append(pred)
        all_y.append(y.numpy())
    P = np.concatenate(all_preds)
    Y = np.concatenate(all_y)
    week_mae = np.abs(P - Y).mean(axis=0)
    return float(week_mae.mean()), week_mae


@torch.no_grad()
def predict_test(model, test_X, test_regions, device, batch_size=512):
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
    parser.add_argument('--out', default='baseline_v9_itransformer.csv')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--d-ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    print(f'[iTransformer config]')
    print(f'  d_model={args.d_model}, n_heads={args.n_heads}, '
          f'n_layers={args.n_layers}, d_ff={args.d_ff}, dropout={args.dropout}')

    bundle = prepare_data(args.base, batch_size=args.batch_size, num_workers=2)
    print(f'\n[Model]')
    model = iTransformer(
        n_features=bundle.n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        n_regions=bundle.n_regions,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  iTransformer: {args.n_layers} layers × d_model={args.d_model}, '
          f'{n_params:,} params')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(bundle.train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_val = math.inf
    best_state = None
    epochs_no_improve = 0

    print(f'\n[Training] {args.epochs} epochs max, patience {args.patience}')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, bundle.train_loader, opt, device, scheduler)
        val_mae, week_mae = eval_epoch(model, bundle.val_loader, device)
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
    raw_preds = predict_test(model, bundle.test_X, bundle.test_regions, device,
                             batch_size=args.batch_size)
    raw_preds = np.clip(raw_preds, 0.0, 5.0)
    print(f'  test preds: mean {raw_preds.mean():.3f}, median {np.median(raw_preds):.3f}')

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
