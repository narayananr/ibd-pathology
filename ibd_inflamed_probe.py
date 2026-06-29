"""
ibd_inflamed_probe.py
=====================
Minimal v1 pipeline for "inflamed vs healed" region mapping on IBD H&E whole-slide
images. This is the ONE piece you have to write yourself; TRIDENT handles everything
upstream (segmentation, tiling, foundation-model embedding).

It assumes TRIDENT has already produced, per slide, under <job_dir>:
  - tile features + coordinates:
        <job_dir>/<mag>x_<ps>px_<ov>px_overlap/features_<encoder>/<slide>.h5
        h5 keys:  "features" (N, D) float32
                  "coords"   (N, 2) int   -> (x, y) TOP-LEFT of each tile at level 0
  - a tissue thumbnail (used only to draw the overlay):
        <job_dir>/thumbnails/<slide>.jpg

You supply, per annotated slide, a QuPath GeoJSON export whose polygon annotations are
classified (right-click -> Annotations -> set class) as one of CLASS_NAMES below.
QuPath exports polygon coordinates in level-0 pixels, which match TRIDENT's coords.

Run:
    python ibd_inflamed_probe.py \
        --feature_dir   trident_processed/20x_256px_0px_overlap/features_uni_v2 \
        --annotation_dir annotations \
        --thumbnail_dir  trident_processed/thumbnails \
        --out_dir        inflamed_maps

Deliberately NOT here (v1 scope): MIL, cell/crypt segmentation, Nancy/Geboes/Robarts.
Add those once this heatmap is something a pathologist signs off on. See notes at bottom.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Class 1 = positive = active inflammation; class 0 = healed / quiescent.
# Map every QuPath class name you use onto one of these two buckets (case-insensitive).
CLASS_NAMES = {
    "healed": 0, "quiescent": 0, "inactive": 0, "normal": 0,
    "inflamed": 1, "active": 1,
}


# ---------------------------------------------------------------------------
# 1. Load TRIDENT features + label tiles from QuPath polygons
# ---------------------------------------------------------------------------
def load_trident_h5(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        feats = f["features"][:]                      # (N, D) float32
        coords = f["coords"][:].astype(np.int64)      # (N, 2) top-left @ level 0
    return feats, coords


def tile_extent_lv0(coords: np.ndarray, fallback: int = 256) -> int:
    """Tile size in level-0 pixels, inferred from the spacing of the tile grid.
    TRIDENT stores coords at level 0, so the on-slide tile footprint is the modal
    stride between neighbouring tiles (== patch_size * level-0 downsample)."""
    xs = np.unique(coords[:, 0])
    if xs.size > 1:
        d = np.diff(np.sort(xs))
        d = d[d > 0]
        if d.size:
            return int(np.median(d))
    return fallback


def label_tiles_from_geojson(coords: np.ndarray, geojson_path: str,
                             extent_lv0: int) -> np.ndarray:
    """Return (N,) int labels: 0/1 for tiles whose CENTER lies in a classified polygon,
    -1 for unlabeled tiles (ignored during training). Requires shapely >= 2.0, whose
    STRtree.query returns integer indices into the input geometry list."""
    with open(geojson_path) as f:
        gj = json.load(f)
    features = gj["features"] if gj.get("type") == "FeatureCollection" else gj

    polys, labels = [], []
    for feat in features:
        cls = (feat.get("properties", {}).get("classification") or {}).get("name", "")
        cls = str(cls).strip().lower()
        if cls not in CLASS_NAMES:
            continue
        polys.append(shape(feat["geometry"]))
        labels.append(CLASS_NAMES[cls])

    out = np.full(len(coords), -1, dtype=np.int64)
    if not polys:
        return out

    tree = STRtree(polys)
    labels = np.asarray(labels)
    half = extent_lv0 / 2.0
    for i, (x, y) in enumerate(coords):
        c = Point(x + half, y + half)
        for j in np.atleast_1d(tree.query(c)):       # candidate polygons by bbox
            if polys[int(j)].contains(c):
                out[i] = int(labels[int(j)])
                break
    return out


def build_dataset(feature_dir: str, annotation_dir: str, slide_ids=None):
    """Walk all <slide>.h5 in feature_dir that have a matching <slide>.geojson.
    Returns X (M, D), y (M,), groups (M,) = slide id per row; only labeled tiles."""
    X, y, groups = [], [], []
    for h5_path in sorted(glob.glob(os.path.join(feature_dir, "*.h5"))):
        sid = Path(h5_path).stem
        if slide_ids is not None and sid not in slide_ids:
            continue
        gj = os.path.join(annotation_dir, sid + ".geojson")
        if not os.path.exists(gj):
            continue
        feats, coords = load_trident_h5(h5_path)
        lab = label_tiles_from_geojson(coords, gj, tile_extent_lv0(coords))
        m = lab >= 0
        if not m.any():
            continue
        X.append(feats[m]); y.append(lab[m]); groups.append(np.array([sid] * int(m.sum())))
        print(f"  {sid}: {int((lab == 1).sum())} inflamed / {int((lab == 0).sum())} healed")
    if not X:
        raise SystemExit("No labeled tiles found. Check that geojson stems match h5 stems "
                         "and that polygon classes are in CLASS_NAMES.")
    return np.concatenate(X), np.concatenate(y), np.concatenate(groups)


# ---------------------------------------------------------------------------
# 2. Train the probe (frozen embeddings -> logistic regression)
# ---------------------------------------------------------------------------
def train_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0):
    """Leave-slides-out validation for an honest read, then refit on all labeled tiles.
    Grouping by slide is essential: tiles from one slide are NOT independent, so a random
    tile split would leak and flatter the AUROC."""
    n_slides = len(set(groups))
    if n_slides >= 3:
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                        random_state=seed).split(X, y, groups))
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:, 1]
        print(f"\nHeld-out slides: {sorted(set(groups[va]))}")
        try:
            print(f"Tile-level AUROC: {roc_auc_score(y[va], p):.3f}")
        except ValueError:
            print("AUROC undefined (held-out slides had a single class).")
        print(classification_report(y[va], (p >= 0.5).astype(int),
                                    target_names=["healed", "inflamed"],
                                    digits=3, zero_division=0))
    else:
        print(f"\nOnly {n_slides} annotated slide(s): skipping held-out eval "
              "(annotate >=3 slides for a real validation number).")

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X, y)                                     # deploy model: all labeled data
    return clf


# ---------------------------------------------------------------------------
# 3. Predict + render the inflamed/healed overlay
# ---------------------------------------------------------------------------
def predict_heatmap(clf, h5_path: str, thumbnail_path: str, out_png: str,
                    alpha: float = 0.45):
    """Stitch per-tile P(inflamed) into a heatmap over the slide thumbnail."""
    feats, coords = load_trident_h5(h5_path)
    extent = tile_extent_lv0(coords)
    prob = clf.predict_proba(feats)[:, 1]

    thumb = plt.imread(thumbnail_path)
    H, W = thumb.shape[:2]
    x_max = int(coords[:, 0].max()) + extent          # level-0 extent of tissue
    y_max = int(coords[:, 1].max()) + extent
    sx, sy = W / x_max, H / y_max                      # level0 -> thumbnail scale

    heat = np.full((H, W), np.nan, dtype=np.float32)
    tw, th = max(1, round(extent * sx)), max(1, round(extent * sy))
    for (x, y), pr in zip(coords, prob):
        px, py = int(x * sx), int(y * sy)
        heat[py:py + th, px:px + tw] = pr

    norm = mcolors.Normalize(0, 1)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(thumb)
    ax.imshow(np.ma.masked_invalid(heat), cmap="turbo", norm=norm, alpha=alpha)
    ax.axis("off")
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="turbo"),
                        ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("P(active inflammation)")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature_dir", required=True,
                    help="TRIDENT features_<encoder> dir for the chosen mag/patch_size")
    ap.add_argument("--annotation_dir", required=True,
                    help="dir of QuPath <slide>.geojson exports (stems match the h5 files)")
    ap.add_argument("--thumbnail_dir", required=True, help="trident_processed/thumbnails")
    ap.add_argument("--out_dir", default="inflamed_maps")
    ap.add_argument("--predict_slide", default=None,
                    help="render only this slide stem (default: every slide with features)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Building labeled dataset from annotated slides:")
    X, y, groups = build_dataset(args.feature_dir, args.annotation_dir)
    print(f"\nTotal labeled tiles: {len(y)}  "
          f"(inflamed={int((y == 1).sum())}, healed={int((y == 0).sum())}, "
          f"slides={len(set(groups))})")
    clf = train_probe(X, y, groups)

    print("\nRendering heatmaps:")
    h5s = sorted(glob.glob(os.path.join(args.feature_dir, "*.h5")))
    if args.predict_slide:
        h5s = [p for p in h5s if Path(p).stem == args.predict_slide]
    for h5_path in h5s:
        sid = Path(h5_path).stem
        thumb = next((os.path.join(args.thumbnail_dir, sid + e)
                      for e in (".jpg", ".png", ".jpeg")
                      if os.path.exists(os.path.join(args.thumbnail_dir, sid + e))), None)
        if thumb is None:
            print(f"  skip {sid}: no thumbnail"); continue
        predict_heatmap(clf, h5_path, thumb, os.path.join(args.out_dir, sid + "_inflamed.png"))


if __name__ == "__main__":
    main()
