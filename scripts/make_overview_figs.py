#!/usr/bin/env python3
"""Figure(s) for the short 5-slide overview deck (`slides/overview.html`).

`overview_problem.png` — states the problem in one picture: a whole reconstructed biopsy slide (raw H&E,
no model output) beside a tile that looks ACTIVE and a tile that looks HEALED. Tiles are chosen by the
illustrative per-tile localizer (slide-mean logistic regression applied to single tiles), purely to pick a
clear example of each look.

Light env, from the repo root:  .venv/bin/python scripts/make_overview_figs.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ibdpath import embed, paths               # noqa: E402
from ibdpath.baseline import build_slide_dataset  # noqa: E402
from ibdpath.mosaic import slide_thumbnail      # noqa: E402

ENC = "hoptimus0"
WHOLE_SLIDE = "137_HE"     # an active biopsy with clear structure
ACTIVE_FROM = "142_HE"     # fully-inflamed slide -> its top tile is an unmistakable "active" look
HEALED_FROM = "144_HE"     # clean inactive slide -> its calmest tile is an "healed" look


def tile_image(man, zf, sid, which):
    rows = man[man.slide_id == sid].sort_values(["y", "x"]).reset_index(drop=True)
    p = LOC.predict_proba(embed.load_slide_embedding(ENC, sid))[:, 1]
    idx = int(np.argmax(p) if which == "active" else np.argmin(p))
    return Image.open(io.BytesIO(zf.read(rows.image_path.iloc[idx]))).convert("RGB")


def main():
    global LOC
    man = pd.read_csv(paths.PATCH_MANIFEST_LABELED_CSV, dtype={"slide_id": str})
    X, y, groups, ids = build_slide_dataset(ENC)
    LOC = LogisticRegression(max_iter=5000, class_weight="balanced").fit(X, y)
    zf = zipfile.ZipFile(paths.PATCH_ZIP)

    whole = slide_thumbnail(man[man.slide_id == WHOLE_SLIDE].sort_values(["y", "x"]), zf, target_px=560)
    active = tile_image(man, zf, ACTIVE_FROM, "active")
    healed = tile_image(man, zf, HEALED_FROM, "healed")

    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    ax_slide = fig.add_subplot(gs[:, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1])

    ax_slide.imshow(whole); ax_slide.axis("off")
    ax_slide.set_title("ONE H&E biopsy slide — thousands of cells, ONE slide-level label\n"
                       "where (if anywhere) is the inflammation active?", fontsize=12.5, pad=10)
    ax_a.imshow(active); ax_a.axis("off")
    ax_a.set_title("ACTIVE  =  neutrophils, dense infiltrate, erosion", color="#c0392b", fontsize=12)
    ax_h.imshow(healed); ax_h.axis("off")
    ax_h.set_title("HEALED  =  orderly, quiet crypts", color="#1e8449", fontsize=12)

    fig.suptitle("The problem: tell ACTIVE from HEALED tissue across a whole slide — "
                 "with only a slide-level label, and disease that's often focal", fontsize=13.5, y=0.99)
    out = paths.ARTIFACTS_DIR / "overview_problem.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    (REPO_ROOT / "slides" / "images" / "overview_problem.png").write_bytes(out.read_bytes())
    print(f"✓ wrote {out.relative_to(REPO_ROOT)} and copied to slides/images/")


if __name__ == "__main__":
    main()
