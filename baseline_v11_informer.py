"""
Baseline V11 — lean Informer for drought severity forecasting (phase-1 sanity check).

Based on
--------
Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence
Time-Series Forecasting", AAAI 2021.

What this is
------------
A LEAN adaptation of Informer, not a faithful reimplementation. It keeps the
two pieces of Informer that are actually relevant at our sequence length, and
drops the three that are not. The point of this script is to settle one
question decisively: does TEMPORAL self-attention (each of the 91 days is a
token, attention runs over time) extract anything over the TCN locality prior,
at L=91 → T=5?

This is the architectural MIRROR of V9 (iTransformer). V9 made each of the 14
VARIABLES a token and ran attention across channels — it failed by converging
to mean predictions (~0.6 vs ~1.05 for working models, r≈0.17 with V7 = bias,
not signal). V11 makes each TIME STEP a token and runs attention across time,
the conventional transformer scheme that iTransformer inverted away from. So
this is a genuinely different bet on the same (attention) family, not a re-run
of V9.

What is KEPT from Informer
--------------------------
  - Conv1d token embedding (kernel 3, circular padding) — Informer's
    DataEmbedding.TokenEmbedding, verbatim.
  - Encoder-decoder structure with a GENERATIVE (one-shot, non-autoregressive)
    decoder — Informer's headline contribution. The decoder emits all 5 weekly
    forecasts in a single forward pass via cross-attention to the encoder
    memory. Here the "generative tokens" are 5 learnable per-week queries.

What is DROPPED from Informer, and why
--------------------------------------
  - ProbSparse self-attention (O(L log L)): exists to make attention tractable
    at long L (hundreds–thousands of steps). At L=91 full attention is ~8k ops;
    ProbSparse buys zero speedup and only adds approximation error. We use full
    attention so the test is about the architecture, not an approximation.
  - Self-attention distilling: halves the sequence between encoder layers to
    handle long inputs. At L=91 there is essentially nothing to distill.
  - label_len start-token decoder input + temporal (calendar) embeddings:
    Informer seeds its decoder with the tail of the input series and adds
    month/day/hour features. Neither applies here — our targets are a DIFFERENT
    variable (drought score) than the inputs (weather), so continuation-style
    start tokens are unnatural; and dates are anonymized (years 3000+), so
    calendar features don't exist. Learnable week queries replace the start
    tokens; per-region z-scoring (done in the pipeline) replaces value scaling.

Architecture
------------
  Input:  (B, 91, 14)
  ├─ Conv1d token embed: 14 → d_model, per step          (B, 91, d_model)
  ├─ + sinusoidal positional encoding (time order matters here, unlike V9)
  ├─ dropout
  ├─ e_layers × encoder self-attention (over the 91 time tokens)
  ├─ memory                                                (B, 91, d_model)
  │
  ├─ 5 learnable week queries                              (B, 5, d_model)
  ├─ d_layers × decoder (self-attn over queries + cross-attn to memory)
  ├─ head: d_model → 1 per query                           (B, 5)
  └─ + region embedding residual (parallel head, like V9)  (B, 5)

Acceptance gates (phase 1 — same as the DLinear brief / V9 diagnostics)
-----------------------------------------------------------------------
  Val MAE:               <= 0.55 PASS | 0.55–0.65 MARGINAL | > 0.65 FAIL
  Per-week pred means:   0.95–1.20 PASS | 0.85–0.95 MARGINAL | < 0.85 FAIL  (V7 ~1.05)
  Corr with V7-10seed:   0.50–0.70 PASS | 0.70–0.85 MARGINAL | >0.85 or <0.40 FAIL
The script prints all three at the end. If means collapse below ~0.90 the way
V9 did, abort — that is the structural failure, not a tuning problem.

Deliberate deviations from V9's hyperparameters (documented)
------------------------------------------------------------
  dropout 0.10 (V9: 0.15) and weight_decay 1e-4 (V9: 3e-4). Both V9 and V8.1
  failed by collapsing toward the zero-inflated mean; heavy regularization
  pushes a model that direction. Lighter regularization gives the lean Informer
  its fairest shot at NOT collapsing. Everything else (lr, batch, epochs,
  patience, L1 loss, grad clip, cosine schedule, region head) matches V9 so it
  is a fair head-to-head.

Usage
-----
    python baseline_v11_informer.py \
        --base /content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project \
        --out  baseline_v11_informer.csv \
        --v7-csv baseline_v7_5seed_avg.csv   # optional: auto-computes correlation

Note: --v7-csv ideally points at the V7-10seed CSV; if you only have the
V7-5seed CSV on hand, that is a fine proxy for the phase-1 correlation gate.
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


# ───────────────────────── Embeddings ─────────────────────────

class TokenEmbedding(nn.Module):
    """Informer's value embedding: Conv1d over time, kernel 3, circular padding."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            c_in, d_model, kernel_size=3, padding=1,
            padding_mode='circular', bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in',
                                nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, c_in) -> (B, L, d_model)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding. d_model assumed even."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


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


class EncoderLayer(nn.Module):
    """Pre-norm self-attention encoder layer (same shape as V9's)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.norm1(x)
        a, _ = self.attn(a, a, a, need_weights=False)
        x = x + self.drop1(a)
        x = x + self.ff(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    """Pre-norm decoder layer: self-attn over queries + cross-attn to memory."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # q: (B, 5, d_model)   memory: (B, 91, d_model)
        a = self.norm1(q)
        a, _ = self.self_attn(a, a, a, need_weights=False)
        q = q + self.drop1(a)

        b = self.norm2(q)
        b, _ = self.cross_attn(b, memory, memory, need_weights=False)
        q = q + self.drop2(b)

        q = q + self.ff(self.norm3(q))
        return q


# ───────────────────────── Lean Informer ─────────────────────────

class LeanInformer(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
        seq_len: int = 91,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 256,
        n_regions: int = 2248,
        region_emb_dim: int = 16,
        n_outputs: int = 5,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.n_outputs = n_outputs

        # Token embedding + positional encoding (time order is informative here)
        self.token_embed = TokenEmbedding(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(512, seq_len))
        self.embed_drop = nn.Dropout(dropout)

        # Encoder over the 91 temporal tokens
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Generative decoder: 5 learnable per-week queries cross-attend to memory
        self.week_query = nn.Embedding(n_outputs, d_model)
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(d_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        # Region embedding (parallel residual head, identical to V9)
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        self.region_head = nn.Linear(region_emb_dim, n_outputs)

    def forward(self, x: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, 91, 14)
        region: (B,)
        returns (B, 5)
        """
        # Encoder
        h = self.token_embed(x)                 # (B, 91, d_model)
        h = self.pos_enc(h)
        h = self.embed_drop(h)
        for layer in self.encoder:
            h = layer(h)
        memory = self.enc_norm(h)               # (B, 91, d_model)

        # Generative decoder
        B = x.size(0)
        idx = torch.arange(self.n_outputs, device=x.device)
        q = self.week_query(idx).unsqueeze(0).expand(B, -1, -1)  # (B, 5, d_model)
        for layer in self.decoder:
            q = layer(q, memory)
        q = self.dec_norm(q)                    # (B, 5, d_model)
        out = self.head(q).squeeze(-1)          # (B, 5)

        # Region residual
        emb_out = self.region_head(self.region_emb(region))
        return out + emb_out


# ───────────────────────── Training (reuse V9's loop) ─────────────────────────

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
        x = test_X[i:i + batch_size].to(device)
        r = test_regions[i:i + batch_size].to(device)
        p = model(x, r).cpu().numpy()
        preds.append(p)
    return np.concatenate(preds)


# ───────────────────────── Acceptance report ─────────────────────────

def _verdict(value, pass_lo, pass_hi, marg_lo, marg_hi):
    """Return PASS / MARGINAL / FAIL for a value against banded thresholds."""
    if pass_lo <= value <= pass_hi:
        return 'PASS'
    if marg_lo <= value <= marg_hi:
        return 'MARGINAL'
    return 'FAIL'


def acceptance_report(best_val, week_mae, raw_preds, submission, v7_csv, base):
    print('\n' + '=' * 64)
    print('ACCEPTANCE CHECK (phase 1)')
    print('=' * 64)

    # 1) Val MAE
    if best_val <= 0.55:
        v = 'PASS'
    elif best_val <= 0.65:
        v = 'MARGINAL'
    else:
        v = 'FAIL'
    print(f'  Val MAE              : {best_val:.4f}   [{v}]   '
          f'(<=0.55 pass | 0.55-0.65 marginal | >0.65 fail)')
    print(f'    per-week val MAE   : ' +
          ' '.join(f'w{i+1} {m:.3f}' for i, m in enumerate(week_mae)))

    # 2) Per-week test prediction means (the V9/V8.1 collapse signature)
    week_means = raw_preds.mean(axis=0)
    overall = float(raw_preds.mean())
    mean_verdict = _verdict(overall, 0.95, 1.20, 0.85, 0.95)
    print(f'  Test pred mean       : {overall:.3f}   [{mean_verdict}]   '
          f'(0.95-1.20 pass | 0.85-0.95 marginal | <0.85 fail; V7 ~1.05)')
    print(f'    per-week means     : ' +
          ' '.join(f'w{i+1} {m:.3f}' for i, m in enumerate(week_means)))
    if week_means.min() < 0.85:
        print(f'    !! week {int(week_means.argmin())+1} mean '
              f'{week_means.min():.3f} < 0.85 — collapse risk (cf. V9 ~0.6).')

    # 3) Correlation with V7 (if a reference CSV was provided)
    if v7_csv:
        v7_path = v7_csv if os.path.isabs(v7_csv) else os.path.join(base, v7_csv)
        if os.path.exists(v7_path):
            ref = pd.read_csv(v7_path)
            pred_cols = [f'pred_week{i}' for i in range(1, 6)]
            merged = submission.merge(ref, on='region_id', suffixes=('_v11', '_v7'))
            a = merged[[f'{c}_v11' for c in pred_cols]].values.ravel()
            b = merged[[f'{c}_v7' for c in pred_cols]].values.ravel()
            corr = float(np.corrcoef(a, b)[0, 1])
            if corr < 0.40 or corr > 0.85:
                cv = 'FAIL'
            elif 0.50 <= corr <= 0.70:
                cv = 'PASS'
            else:                       # 0.40-0.50 or 0.70-0.85
                cv = 'MARGINAL'
            print(f'  Corr with {os.path.basename(v7_path)} : {corr:.3f}   [{cv}]   '
                  f'(0.50-0.70 pass | 0.70-0.85 marginal | >0.85 or <0.40 fail)')
        else:
            print(f'  Corr with V7         : SKIPPED — file not found: {v7_path}')
    else:
        print('  Corr with V7         : SKIPPED — pass --v7-csv to compute')

    print('=' * 64)
    print('Decision: all PASS -> proceed to phase 2 (5-seed avg). Any mean < ~0.90')
    print('or val MAE > 0.65 -> abort; this is structural collapse, not tuning.')
    print('=' * 64 + '\n')


# ───────────────────────── Main ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',
        default='/content/drive/MyDrive/Data_Mining/Final_Project/data-mining-2026-final-project')
    parser.add_argument('--out', default='baseline_v11_informer.csv')
    parser.add_argument('--v7-csv', default=None,
                        help='Optional V7 (10seed preferred) CSV for correlation gate')
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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    print('[Lean Informer config]')
    print(f'  d_model={args.d_model}, n_heads={args.n_heads}, '
          f'e_layers={args.e_layers}, d_layers={args.d_layers}, '
          f'd_ff={args.d_ff}, dropout={args.dropout}')

    bundle = prepare_data(args.base, batch_size=args.batch_size, num_workers=2)

    print('\n[Model]')
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  LeanInformer: enc {args.e_layers} / dec {args.d_layers} layers '
          f'× d_model={args.d_model}, {n_params:,} params')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = args.epochs * len(bundle.train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_val = math.inf
    best_state = None
    best_week_mae = None
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
            best_week_mae = week_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
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
    print(f'  test preds: mean {raw_preds.mean():.3f}, '
          f'median {np.median(raw_preds):.3f}')

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

    acceptance_report(best_val, best_week_mae, raw_preds, submission,
                      args.v7_csv, args.base)


if __name__ == '__main__':
    main()
