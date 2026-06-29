"""Load the curated slide -> active/inactive label table, plus a patient-grouping helper.

The labels were derived once from the IBDColEpi annotation filenames
(`ID-{id}_HE_{active|inactive}.tiff`) and committed to `metadata/slide_labels.csv`
(see PROGRESS.md / project memory for provenance). This module is the single place the rest
of the pipeline reads them.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import paths

# Our POSITIVE class is active inflammation (matches ibd_inflamed_probe.py: class 1 = active).
LABEL_TO_INT = {"inactive": 0, "active": 1}


def load_slide_labels(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Read the curated labels -> DataFrame[slide_id, label, source, slide_target].

    `slide_target` is the integer SLIDE-level label (active=1, inactive=0). The `slide_` prefix
    makes explicit it is a per-slide value — the bag label for mean-pool / attention-MIL, NOT a
    per-tile truth — and it never collides with the manifest's tile `y` coordinate when joined.
    Fails loudly if the file is missing or any label value isn't 'active'/'inactive'.
    """
    csv_path = paths.SLIDE_LABELS_CSV if csv_path is None else Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"slide labels not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"slide_id": str, "label": str})
    bad = set(df["label"]) - set(LABEL_TO_INT)
    if bad:
        raise ValueError(f"unexpected label value(s) in {csv_path}: {sorted(bad)}")
    df["slide_target"] = df["label"].map(LABEL_TO_INT).astype(int)
    return df


def patient_id(slide_id: str) -> str:
    """The patient/case id = the leading number of a slide_id.

    `107_HE` -> '107';  both `116_HE` and `116_2_HE` -> '116' (two biopsies, one patient).
    Group by THIS (not slide_id) for leave-patients-out validation, so a patient's two biopsies
    never straddle the train/test split.
    """
    m = re.match(r"(\d+)", slide_id)
    if m is None:
        raise ValueError(f"cannot parse a patient id from slide_id {slide_id!r}")
    return m.group(1)


def attach_labels(manifest: pd.DataFrame, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join per-slide labels onto a tile manifest -> adds `label`, `slide_target`, `patient_id`.

    `slide_target` is the SLIDE label replicated on every tile of that slide: the bag label for
    mean-pool / attention-MIL, NOT a per-tile truth (an active slide is mostly calm tiles). Every
    tile must match a labelled slide; raises ValueError listing any slide_ids that don't.
    """
    if labels is None:
        labels = load_slide_labels()
    out = manifest.merge(labels[["slide_id", "label", "slide_target"]], on="slide_id", how="left")
    out["patient_id"] = out["slide_id"].astype(str).map(patient_id)
    unlabeled = sorted(out.loc[out["label"].isna(), "slide_id"].unique())
    if unlabeled:
        raise ValueError(f"{len(unlabeled)} slide(s) have no label: {unlabeled[:10]}")
    out["slide_target"] = out["slide_target"].astype(int)
    return out
