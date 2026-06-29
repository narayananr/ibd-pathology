"""Where tile embeddings live on disk, and how to read/write them.

Deliberately **torch-free** (only numpy) so the light 3.14 env (Steps 4–6) can load embeddings
that the heavy `.venv-embed` env produced in Step 3. The cache is keyed by encoder, one file per
slide:  artifacts/embeddings/<encoder>/<slide_id>.npy  -> (n_tiles, dim) float32, manifest order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import paths

EMBED_DIM = 1536   # H-optimus-0 / H-optimus-1 output width (H0-mini would be 768)


def embeddings_dir(encoder: str) -> Path:
    return paths.ARTIFACTS_DIR / "embeddings" / encoder


def slide_embedding_path(encoder: str, slide_id: str) -> Path:
    return embeddings_dir(encoder) / f"{slide_id}.npy"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize (each embedding -> unit length). Standard before a linear probe.
    A zero row stays zero (no divide-by-zero)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def save_slide_embedding(encoder: str, slide_id: str, feats: np.ndarray) -> Path:
    p = slide_embedding_path(encoder, slide_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, feats.astype(np.float32))
    return p


def load_slide_embedding(encoder: str, slide_id: str) -> np.ndarray:
    return np.load(slide_embedding_path(encoder, slide_id))
