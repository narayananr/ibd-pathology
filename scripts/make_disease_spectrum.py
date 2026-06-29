#!/usr/bin/env python3
"""Make the 'disease is a spectrum' deck figure: three active-vs-calm slides coloured by HOW INFLAMED
EACH TILE LOOKS (the per-tile localizer P(inflamed)), on one shared scale — focal → patchy → diffuse.

This is the honest companion to the attention heatmap. Attention shows *where the model focused to
decide*; this shows *how much of the slide is actually inflamed* (the disease extent). It also makes the
key limitation visible: the genuinely FOCAL slide (`17_HE`, ~11% inflamed) is the one BOTH heads miss —
attention-MIL does not rescue it.

Per-tile localizer = a bare logistic regression on the slide-mean embeddings, applied to single tiles
(no scaler) — illustrative, exactly like the review gallery (the proper per-tile model is MIL attention).

Light env, from the repo root:  .venv/bin/python scripts/make_disease_spectrum.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ibdpath import embed, paths               # noqa: E402
from ibdpath.baseline import build_slide_dataset  # noqa: E402
from ibdpath.mosaic import heatmap_overlay      # noqa: E402

ENC = "hoptimus0"
# focal (hard, missed) -> patchy -> fully inflamed -> calm control
PANELS = [("17_HE", "FOCAL — rare & hard\n(both heads MISS this)"),
          ("137_HE", "PATCHY — partly inflamed"),
          ("142_HE", "DIFFUSE — inflamed end-to-end"),
          ("144_HE", "CALM — healed (control)")]


def main():
    man = pd.read_csv(paths.PATCH_MANIFEST_LABELED_CSV, dtype={"slide_id": str})
    X, y, groups, ids = build_slide_dataset(ENC)
    loc = LogisticRegression(max_iter=5000, class_weight="balanced").fit(X, y)
    zf = zipfile.ZipFile(paths.PATCH_ZIP)

    fig, ax = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 8))
    for a, (sid, desc) in zip(ax, PANELS):
        rows = man[man.slide_id == sid].sort_values(["y", "x"]).reset_index(drop=True)
        p = loc.predict_proba(embed.load_slide_embedding(ENC, sid))[:, 1]
        a.imshow(heatmap_overlay(rows, p, zf, target_px=620, vmax=1.0))
        a.axis("off")
        a.set_title(f"{sid}\n{desc}\n{(p > 0.5).mean():.0%} of tiles look inflamed", fontsize=11, pad=8)
    sm = ScalarMappable(norm=Normalize(0, 1), cmap="turbo")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["looks healthy", "looks inflamed"])
    cb.set_label("per-tile P(inflamed)", fontsize=11)
    fig.suptitle("Disease is a spectrum — colour = how inflamed each tile looks (one shared scale)",
                 fontsize=14, y=0.97)
    out = paths.ARTIFACTS_DIR / "disease_spectrum.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    (REPO_ROOT / "slides" / "images" / "disease_spectrum.png").write_bytes(out.read_bytes())
    print(f"✓ wrote {out.relative_to(REPO_ROOT)} and copied to slides/images/")


if __name__ == "__main__":
    main()
