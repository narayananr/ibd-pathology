#!/usr/bin/env python3
"""
Step 1 — build the patch manifest.
==================================
Read the IBDColEpi HE patch zip (WITHOUT extracting it) and write a tidy table with one row
per H&E tile to  artifacts/patch_manifest.csv.

Why this exists: every later step (attach labels, embed tiles, baseline, MIL) needs a clean
list of "which tiles exist, which slide each belongs to, and where it sits." We build that
list once, here. No model, no GPU — pure zipfile + regex + pandas, a couple of seconds on CPU.

Run from the repo root:
    .venv/bin/python scripts/01_build_manifest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make `import ibdpath` work without pip-installing the package: add <root>/src to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibdpath import paths                       # noqa: E402  (import after sys.path tweak)
from ibdpath.manifest import build_patch_manifest   # noqa: E402


def main() -> None:
    print(f"Reading patch zip (file list only, not the pixels):\n  {paths.PATCH_ZIP}")
    df = build_patch_manifest(paths.PATCH_ZIP)

    paths.ensure_artifacts()
    df.to_csv(paths.PATCH_MANIFEST_CSV, index=False)

    # ---- summary -----------------------------------------------------------
    n_tiles = len(df)
    n_slides = df["slide_id"].nunique()
    print(f"\n✓ wrote {paths.PATCH_MANIFEST_CSV.relative_to(REPO_ROOT)}")
    print(f"  {n_tiles} tiles · {n_slides} distinct slides · "
          f"has_mask = {int(df['has_mask'].sum())}/{n_tiles}")
    sizes = sorted(int(v) for v in df["stored_px"].unique())   # plain ints -> clean print
    print(f"  stored tile size(s): {sizes} px")

    print("\nPer split (tiles / slides):")
    per = df.groupby("split").agg(tiles=("image_path", "size"),
                                  slides=("slide_id", "nunique"))
    for split, r in per.iterrows():
        print(f"  {split:14s} {int(r.tiles):6d} tiles   {int(r.slides):4d} slides")

    # ---- honesty check: does the authors' Train/Val split leak by slide? ----
    by_split = {s: set(g) for s, g in df.groupby("split")["slide_id"]}
    tr, va, te = by_split.get("Trainset", set()), by_split.get("Validationset", set()), by_split.get("Testset", set())
    print("\nSlide-level overlap between the authors' splits:")
    print(f"  Train ∩ Val  = {len(tr & va):3d} slides   (Val slides ALSO in Train → their split LEAKS by slide)")
    print(f"  Train ∩ Test = {len(tr & te):3d} slides")
    print(f"  Val   ∩ Test = {len(va & te):3d} slides")
    print("  → For our slide-level task we ignore this split and validate leave-slides-out by slide_id.")

    print("\nFirst 4 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
