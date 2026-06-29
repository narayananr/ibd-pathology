"""Stitch a slide's tiles into a thumbnail using their level-0 (x, y) coordinates.

The IBDColEpi patch set ships no WSI thumbnails, so we reconstruct a coarse slide image from the
tiles themselves. Used by the review gallery and (later) the Step-5 heatmap background. Pillow only
(works in either env).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def slide_thumbnail(rows, zf, target_px: int = 300, bg=(245, 245, 245)) -> Image.Image:
    """Reconstruct one slide as a small RGB image.

    `rows`: the manifest rows for ONE slide (columns x, y, w, h, image_path).
    `zf`:   an open zipfile.ZipFile of the patch zip.
    Tiles are downscaled and pasted at their scaled positions; the long side ~= target_px.
    """
    x = rows["x"].to_numpy()
    y = rows["y"].to_numpy()
    w = rows["w"].to_numpy()
    h = rows["h"].to_numpy()
    W = int((x + w).max())
    H = int((y + h).max())
    scale = target_px / max(W, H)
    canvas = Image.new("RGB", (max(1, round(W * scale)), max(1, round(H * scale))), bg)
    for xi, yi, wi, hi, path in zip(x, y, w, h, rows["image_path"]):
        tile = Image.open(io.BytesIO(zf.read(path))).convert("RGB")
        tw, th = max(1, round(wi * scale)), max(1, round(hi * scale))
        canvas.paste(tile.resize((tw, th)), (round(xi * scale), round(yi * scale)))
    return canvas


def heatmap_overlay(rows, weights, zf, *, target_px: int = 420, alpha: float = 0.62,
                    cmap: str = "turbo", vmax: float | None = None, bg=(245, 245, 245)) -> Image.Image:
    """Reconstructed slide thumbnail with a per-tile heat overlay (e.g. attention).

    `rows`:    manifest rows for ONE slide (x, y, w, h, image_path), in the SAME order as `weights`
               — i.e. sorted by ["y", "x"], the order Step 3 saved embeddings, so weight i lines up
               with the tile at rows.iloc[i].
    `weights`: per-tile scores (attention enrichment, P(active), ...), len == len(rows).
    `vmax`:    sets the colour scale. If None, weights are rescaled to each slide's own [min, max]
               (good for a single-slide view). If given, colour = clip(weight / vmax, 0, 1) with NO
               per-slide stretching — so a slide whose weights are all small stays *pale*. Pass a
               FIXED vmax across slides to make their heatmaps directly comparable (an inactive
               slide should then look faint, an active focus bright). A tile's opacity also scales
               with its (clipped) weight, so a focal hot spot pops out of an otherwise faint slide.

    Returns an RGB image: the tissue mosaic with the heat blended on top.
    """
    from matplotlib import colormaps   # local import keeps the module light unless this is used

    weights = np.asarray(weights, dtype=float)
    base = slide_thumbnail(rows, zf, target_px=target_px, bg=bg).convert("RGBA")
    heat = Image.new("RGBA", base.size, (0, 0, 0, 0))

    if vmax is not None:
        norm = np.clip(weights / vmax, 0.0, 1.0)
    else:
        lo, hi = float(weights.min()), float(weights.max())
        norm = (weights - lo) / (hi - lo) if hi > lo else np.zeros_like(weights)
    colormap = colormaps[cmap]

    x, y, w, h = (rows[c].to_numpy() for c in ("x", "y", "w", "h"))
    W, H = int((x + w).max()), int((y + h).max())
    scale = target_px / max(W, H)
    for xi, yi, wi, hi_, n in zip(x, y, w, h, norm):
        r, g, b, _ = colormap(float(n))
        box = (round(xi * scale), round(yi * scale),
               round((xi + wi) * scale), round((yi + hi_) * scale))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        a = int(255 * alpha * float(n))            # cold tiles ~transparent, hot tiles opaque
        patch = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]),
                          (int(255 * r), int(255 * g), int(255 * b), a))
        heat.paste(patch, (box[0], box[1]))
    return Image.alpha_composite(base, heat).convert("RGB")
