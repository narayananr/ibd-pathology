"""Gated attention-MIL (ABMIL) head on the frozen tile embeddings — the *where*.

A slide is a **bag** of tile embeddings with ONE slide-level label (active/inactive). Mean-pool
(Step 4) gave the headline AUROC but averaged the location away. This head keeps the bag intact and
learns a per-tile attention weight ``a_i`` (softmax over the bag, so the weights sum to 1), pools
``z = sum_i a_i * h_i``, and classifies ``z``. **The attention vector ``a`` IS the heatmap** — it
says which tiles drove the "active" call.

Gated attention (Ilse, Tomczak & Welling, *Attention-based Deep MIL*, ICML 2018)::

    A_i = w^T ( tanh(V h_i)  ⊙  sigmoid(U h_i) )
    a   = softmax(A)            # over the tiles of ONE bag

The sigmoid gate can *suppress* a tile that merely looks busy (e.g. a dense-but-benign lymphoid
aggregate) — useful against the ``132_HE`` trap.

Torch — runs in the heavy ``.venv-embed`` env. Kept deliberately small (only 140 bags): a low-dim
projection + dropout + weight decay do the regularizing. This module owns both the network and the
train/predict loop so the Step-5 script stays thin and the tests can exercise the logic directly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GatedAttentionMIL(nn.Module):
    """Project -> gated attention pool -> linear classifier.

    forward(x, mask) takes a *batch of bags* ``x`` (B, N, in_dim) and an optional boolean ``mask``
    (B, N) where True = a real tile (False = padding). Returns ``(logit, attn)``:
      - ``logit`` (B,)   : pre-sigmoid slide score (active is the positive class)
      - ``attn``  (B, N) : attention weights, each row summing to 1 over its real tiles
    A single bag is just B=1; the Step-5 script runs one bag at a time, so ``mask`` is only needed
    when bags are padded together (and is exercised by the tests).
    """

    def __init__(self, in_dim: int = 1536, proj_dim: int = 128, attn_dim: int = 128,
                 dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.attn_V = nn.Linear(proj_dim, attn_dim)   # tanh branch
        self.attn_U = nn.Linear(proj_dim, attn_dim)   # sigmoid gate
        self.attn_w = nn.Linear(attn_dim, 1)          # -> one score per tile
        self.classifier = nn.Linear(proj_dim, 1)

    def attention(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Gated attention weights for projected tiles ``h`` (B, N, proj_dim) -> (B, N)."""
        gate = torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))   # (B, N, attn_dim)
        scores = self.attn_w(gate).squeeze(-1)                              # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))   # padded tiles can't win attention
        return torch.softmax(scores, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.proj(x)                                # (B, N, proj_dim)
        a = self.attention(h, mask)                     # (B, N)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)     # (B, proj_dim)  weighted bag vector
        logit = self.classifier(z).squeeze(-1)          # (B,)
        return logit, a


def _device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_mil(bags, targets, *, in_dim: int = 1536, epochs: int = 150, lr: float = 1e-3,
              weight_decay: float = 1e-3, dropout: float = 0.25, proj_dim: int = 128,
              batch_bags: int = 16, pos_weight: float | None = None, seed: int = 0,
              device=None) -> GatedAttentionMIL:
    """Train the head on a list of bags (one variable-length ``(n_i, in_dim)`` array each).

    Class-weighted BCE (``pos_weight`` handles the 54/86 active/inactive imbalance). Bags are fed
    one at a time (no padding); gradients are accumulated over ``batch_bags`` bags before each Adam
    step, which de-noises the per-bag signal. Returns the trained model in eval mode.
    """
    dev = _device(device)
    torch.manual_seed(seed)
    model = GatedAttentionMIL(in_dim=in_dim, proj_dim=proj_dim, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32, device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    tens = [torch.as_tensor(np.asarray(b), dtype=torch.float32, device=dev) for b in bags]
    tgt = torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=dev)
    gen = torch.Generator().manual_seed(seed)   # CPU generator just for shuffling indices

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(tens), generator=gen).tolist()
        opt.zero_grad()
        for j, i in enumerate(perm, 1):
            logit, _ = model(tens[i].unsqueeze(0))           # (1,)
            loss = bce(logit, tgt[i:i + 1]) / batch_bags
            loss.backward()
            if j % batch_bags == 0 or j == len(perm):
                opt.step()
                opt.zero_grad()
    model.eval()
    return model


@torch.no_grad()
def predict_bag(model: GatedAttentionMIL, bag, device=None):
    """One bag ``(n, in_dim)`` -> (p_active float, attention ``(n,)`` numpy array summing to 1)."""
    dev = _device(device)
    model.eval()
    x = torch.as_tensor(np.asarray(bag), dtype=torch.float32, device=dev).unsqueeze(0)
    logit, a = model(x)
    return torch.sigmoid(logit).item(), a.squeeze(0).detach().cpu().numpy()
