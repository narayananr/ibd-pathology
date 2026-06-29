#!/usr/bin/env python3
"""
Step 2 — attach the active/inactive label to every tile.
========================================================
Join the curated slide labels (`metadata/slide_labels.csv`) onto the Step-1 manifest, adding
`label`, `slide_target` (active=1 / inactive=0), and `patient_id` to every tile row.
Output: artifacts/patch_manifest_labeled.csv.

`slide_target` is the SLIDE's label copied onto each of its tiles (the bag label for the
mean-pool baseline and attention-MIL) — NOT a per-tile claim, since an active slide is mostly
calm tissue. This step is pure bookkeeping (a table join) — no model, no GPU.

Run from the repo root:
    .venv/bin/python scripts/02_attach_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibdpath import paths                                    # noqa: E402
from ibdpath.labels import load_slide_labels, attach_labels  # noqa: E402


def main() -> None:
    manifest = pd.read_csv(paths.PATCH_MANIFEST_CSV, dtype={"slide_id": str})
    labels = load_slide_labels()
    print(f"manifest: {len(manifest)} tiles · {manifest['slide_id'].nunique()} slides")
    print(f"labels:   {len(labels)} slides {labels['label'].value_counts().to_dict()}")

    try:
        df = attach_labels(manifest, labels)
    except ValueError as e:                  # a join-key mismatch -> stop loudly
        raise SystemExit(f"label join failed: {e}")

    paths.ensure_artifacts()
    df.to_csv(paths.PATCH_MANIFEST_LABELED_CSV, index=False)
    print(f"\n✓ wrote {paths.PATCH_MANIFEST_LABELED_CSV.relative_to(REPO_ROOT)}  "
          f"({df.shape[0]} rows × {df.shape[1]} cols)")

    # ---- summary -----------------------------------------------------------
    slides = df.drop_duplicates("slide_id")
    print("\nSlides per class:", slides["label"].value_counts().to_dict(),
          f"  (positive rate = {slides['slide_target'].mean():.0%} active)")
    print("Tiles  per class:", df["label"].value_counts().to_dict())
    print(f"Patients: {df['patient_id'].nunique()} distinct (group by this for validation)")

    tps = df.groupby("slide_id").size()
    print(f"\nTiles per slide: min={tps.min()} median={int(tps.median())} max={tps.max()}")

    print("\nFirst 3 rows (key columns):")
    cols = ["split", "slide_id", "patient_id", "label", "slide_target", "x", "y", "stored_px"]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df[cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
