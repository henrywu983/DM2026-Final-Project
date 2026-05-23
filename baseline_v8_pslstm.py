"""
Baseline V8 — P-sLSTM (Patched sLSTM) for drought severity forecasting.

Based on
--------
Kong et al., "Unlocking the Power of LSTM for Long Term Time Series
Forecasting", AAAI 2025. The paper proposes:

1. Channel independence — each weather variable is processed as a separate
   univariate series, batched together. Forces the model to learn signal
   per-variable rather than relying on cross-variable shortcuts. Originally
   from PatchTST.

2. Patching — divide the 91-day sequence into overlapping patches, project
   each patch to an embedding. Trades temporal resolution for richer per-
   step features, which sLSTM handles better than raw daily values.

3. sLSTM (scalar LSTM, from Beck et al. 2024's xLSTM family) — replaces
   sigmoid input/forget gates with exponential gating and adds a normalizer
   state for stability. The exponential gate can revise stored memory more
   aggressively than vanilla LSTM, fitting drought dynamics where conditions
   can shift quickly.

Architecture
------------
  Input:  (B, 91, 14) — 91 days, 14 weather channels
  ├─ Per-region instance normalization (already done in data pipeline)
  ├─ Channel independence: reshape to (B·14, 91, 1)
  ├─ Patching: (B·14, n_patches, patch_dim) with patch_size=16, stride=8 → 11 patches
  ├─ Linear projection: patch_dim=16 → d_model=64
  ├─ sLSTM block (2 layers, d_model=64)
  ├─ Flatten + Linear projection: (n_patches × d_model) → 5
  ├─ Reshape back: (B, 14, 5)
  ├─ Mean across channels → (B, 5)
  └─ Add region embedding contribution (parallel head) → (B, 5)

The region embedding is added as a small parallel residual to give the
model per-region priors without contaminating the channel-independent
sequence processing.

sLSTM implementation
--------------------
Single-layer, single-head for simplicity — the paper shows multi-head
matters most for very long sequences (96+ steps), but we only have 11
patches, so a single head suffices and is much faster.

Key sLSTM equations (Beck et al. 2024):
    f_t = exp(W_f x_t + R_f h_{t-1} + b_f)
    i_t = exp(W_i x_t + R_i h_{t-1} + b_i)
    z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)
    o_t = sigmoid(W_o x_t + R_o h_{t-1} + b_o)
    n_t = f_t * n_{t-1} + i_t
    c_t = f_t * c_{t-1} + i_t * z_t
    h_t = o_t * (c_t / n_t)

The normalizer state n_t stabilizes the exponential gating that would
otherwise blow up the cell state.

Usage
-----
    python baseline_v8_pslstm.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v8_pslstm.csv \
        --epochs 80
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

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from data_pipeline_nn import prepare_data  # noqa: E402


# ───────────────────────── sLSTM cell and block ─────────────────────────

class sLSTMCell(nn.Module):
    """
    Scalar LSTM cell with exponential gating (Beck et al. 2024).

    Compared to vanilla LSTM:
      - forget/input gates use exp() instead of sigmoid (unbounded above)
      - adds a normalizer state n_t to keep h_t numerically stable
      - subtracting the max log-gate value before exp() prevents overflow

    The "log-space stabilization" trick (m_t in the paper) is applied here.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 4 gates: input, forget, cell-input (z), output
        self.W = nn.Linear(input_dim, 4 * hidden_dim, bias=True)
        self.R = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, ...]
                ) -> Tuple[torch.Tensor, Tuple]:
        """
        x_t: (B, input_dim)
        state: (h, c, n, m) each (B, hidden_dim)
        """
        h_prev, c_prev, n_prev, m_prev = state
        gates = self.W(x_t) + self.R(h_prev)
        i_tilde, f_tilde, z_tilde, o_tilde = gates.chunk(4, dim=-1)

        # Log-space stabilizer m: tracks the max of f_tilde + m_prev and i_tilde
        # so that the exp() never overflows.
        m_t = torch.maximum(f_tilde + m_prev, i_tilde)
        i_t = torch.exp(i_tilde - m_t)
        f_t = torch.exp(f_tilde + m_prev - m_t)

        z_t = torch.tanh(z_tilde)
        o_t = torch.sigmoid(o_tilde)

        c_t = f_t * c_prev + i_t * z_t
        n_t = f_t * n_prev + i_t
        h_t = o_t * (c_t / (n_t + 1e-6))

        return h_t, (h_t, c_t, n_t, m_t)

    def init_state(self, batch_size: int, device) -> Tuple[torch.Tensor, ...]:
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
        return (zeros, zeros, torch.ones_like(zeros), zeros.clone())


class sLSTMLayer(nn.Module):
    """Single sLSTM layer over a (B, T, D) sequence."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = sLSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → (B, T, hidden_dim)"""
        B, T, _ = x.shape
        state = self.cell.init_state(B, x.device)
        outputs = []
        for t in range(T):
            h_t, state = self.cell(x[:, t], state)
            outputs.append(h_t)
        return torch.stack(outputs, dim=1)


class sLSTMBlock(nn.Module):
    """Stacked sLSTM layers with residual connections and LayerNorm."""

    def __init__(self, d_model: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([sLSTMLayer(d_model, d_model) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(norm(x))
            x = self.drop(x) + residual
        return x


# ───────────────────────── P-sLSTM ─────────────────────────

class PsLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
        seq_len: int = 91,
        patch_size: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
        n_lstm_layers: int = 2,
        n_regions: int = 2248,
        region_emb_dim: int = 8,
        n_outputs: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Pad sequence so it divides into evenly-sized patches.
        # n_patches = (seq_len - patch_size) / stride + 1
        pad = (patch_size - (seq_len % patch_stride or patch_stride)) % patch_stride
        self.pad_len = pad
        padded_len = seq_len + pad
        self.n_patches = (padded_len - patch_size) // patch_stride + 1

        # Patch projection (shared across channels via channel independence)
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.input_drop = nn.Dropout(dropout)

        # sLSTM backbone
        self.backbone = sLSTMBlock(d_model, n_layers=n_lstm_layers, dropout=dropout)

        # Forecast head: from (n_patches, d_model) per channel to n_outputs
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_patches * d_model, n_outputs),
        )

        # Region embedding (parallel head, added to channel-mean prediction)
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        self.region_head = nn.Linear(region_emb_dim, n_outputs)

    def forward(self, x: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 91, 14)
        region: (B,)
        """
        B, T, C = x.shape

        # Pad on the left so the most recent days fit cleanly into patches
        if self.pad_len > 0:
            pad = x[:, :1].expand(-1, self.pad_len, -1)
            x = torch.cat([pad, x], dim=1)

        # Channel independence: reshape to (B*C, T_padded, 1)
        x = x.permute(0, 2, 1).contiguous()  # (B, C, T_padded)
        x = x.view(B * C, x.shape[-1])       # (B*C, T_padded)

        # Patch: (B*C, n_patches, patch_size)
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)

        # Project: (B*C, n_patches, d_model)
        h = self.patch_proj(patches)
        h = self.input_drop(h)

        # sLSTM backbone
        h = self.backbone(h)  # (B*C, n_patches, d_model)

        # Per-channel forecast
        out = self.head(h)    # (B*C, n_outputs)
        out = out.view(B, C, -1).mean(dim=1)  # (B, n_outputs) — mean over channels

        # Region embedding residual
        emb_out = self.region_head(self.region_emb(region))
        return out + emb_out


# ───────────────────────── Training (shared logic) ─────────────────────────

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
def predict_test(model, test_X, test_regions, device, batch_size=256):
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
    parser.add_argument('--out', default='baseline_v8_pslstm.csv')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-lstm-layers', type=int, default=2)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--patch-stride', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    bundle = prepare_data(args.base, batch_size=args.batch_size, num_workers=2)
    print(f'\n[Model]')
    model = PsLSTM(
        n_features=bundle.n_features,
        d_model=args.d_model,
        n_lstm_layers=args.n_lstm_layers,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        n_regions=bundle.n_regions,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  P-sLSTM: d_model={args.d_model}, n_layers={args.n_lstm_layers}, '
          f'patch={args.patch_size}/{args.patch_stride}, n_patches={model.n_patches}, '
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
