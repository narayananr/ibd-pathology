"""Read the dataset's pixel-level epithelium masks and overlay them on tiles.

IBDColEpi ships an epithelium **segmentation mask** (`Labels_tif`) aligned 1:1 with each H&E tile
(`Images_tif`) — same 512×512 size, values {0, 1} (1 = epithelium). These masks are the **ground
truth** for the planned compartment-segmentation head. Pillow + numpy only (light env).

⚠️ Compartment segmentation is **deferred** scope — using these masks is part of that planned work.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def load_tile(zf, image_path) -> Image.Image:
    """The H&E tile as an RGB PIL image."""
    return Image.open(io.BytesIO(zf.read(image_path))).convert("RGB")


def load_mask(zf, mask_path) -> np.ndarray:
    """The epithelium mask as a (H, W) uint8 array of {0, 1} (1 = epithelium)."""
    arr = np.array(Image.open(io.BytesIO(zf.read(mask_path))))
    if arr.ndim == 3:                      # collapse if stored with channels
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def epithelium_fraction(mask: np.ndarray) -> float:
    """Share of the tile that is epithelium."""
    return float(np.asarray(mask).mean())


def overlay(tile: Image.Image, mask: np.ndarray, color=(51, 192, 141),
            alpha: float = 0.40, outline: bool = True) -> Image.Image:
    """Tint the epithelium region (and outline it) on the RGB tile → PIL image."""
    base = np.asarray(tile.convert("RGB")).astype(float)
    m = np.asarray(mask).astype(bool)
    col = np.array(color, dtype=float)
    base[m] = (1.0 - alpha) * base[m] + alpha * col
    out = base.astype(np.uint8)
    if outline:
        e = _edges(m)
        out[e] = col.astype(np.uint8)
    return Image.fromarray(out)


def _edges(mask_bool: np.ndarray) -> np.ndarray:
    """1-px boundary of a binary mask (where a pixel differs from a 4-neighbour)."""
    m = mask_bool
    edge = np.zeros_like(m)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[1:, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]
    edge[:, 1:] |= m[:, :-1] != m[:, 1:]
    return edge & _dilate_seed(m)  # keep only edges adjacent to epithelium (cleaner line)


def _dilate_seed(m: np.ndarray) -> np.ndarray:
    d = m.copy()
    d[:-1, :] |= m[1:, :]
    d[1:, :] |= m[:-1, :]
    d[:, :-1] |= m[:, 1:]
    d[:, 1:] |= m[:, :-1]
    return d
