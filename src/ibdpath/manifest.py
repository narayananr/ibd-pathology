"""Turn the IBDColEpi HE patch zip into a tidy per-tile table — WITHOUT extracting it.

Each H&E tile is a zip entry named like:

    Trainset/Images_tif/107_HE [d=4,x=1536,y=10752,w=2048,h=2048].tif
    └─split─┘ └─ kind ─┘ └slide┘ │   └── position ──┘ └ level-0 footprint ┘
                                 └ d = downsample (stored tile is w/d px)

We read only the zip's *file list* (fast — `zipfile` reads the small central directory, not
the 8 GB of pixels), parse each `Images_tif` name with one regex, and note whether a
same-named `Labels_tif` mask exists. Output: one row per H&E tile.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

from . import paths

# Verified against all 6,322 HE tile names (100% match). Case-insensitive for safety.
TILE_NAME_RE = re.compile(
    r"^(?P<slide>\w+?_HE)\s*"
    r"\[d=(?P<d>\d+),x=(?P<x>\d+),y=(?P<y>\d+),w=(?P<w>\d+),h=(?P<h>\d+)\]\.tif$",
    re.IGNORECASE,
)

IMAGES_KIND = "Images_tif"   # the H&E RGB tiles   <- what we want
LABELS_KIND = "Labels_tif"   # epithelium masks    <- DEFERRED scope; we only record presence

# The columns of the manifest, in order.
COLUMNS = ["split", "slide_id", "x", "y", "w", "h", "downsample", "stored_px",
           "has_mask", "image_path", "mask_path"]


def parse_tile_member(member: str) -> dict | None:
    """Parse one zip member path into a record dict, or return None if it isn't an HE tile.

    `member` looks like "Trainset/Images_tif/107_HE [d=4,x=...,...].tif". Anything that isn't
    an `Images_tif/*.tif` (a mask, the MIB-format copies, or a bare folder entry) returns None.
    """
    parts = member.split("/")
    if len(parts) < 3 or parts[1] != IMAGES_KIND:
        return None
    split, fname = parts[0], parts[-1]
    m = TILE_NAME_RE.match(fname)
    if m is None:
        return None
    d, w, h = int(m["d"]), int(m["w"]), int(m["h"])
    return {
        "split": split,                # Trainset / Validationset / Testset (PROVENANCE ONLY)
        "slide_id": m["slide"],        # e.g. 107_HE  -> our grouping key for leave-slides-out
        "x": int(m["x"]), "y": int(m["y"]),   # top-left of the crop, in level-0 pixels
        "w": w, "h": h,                # level-0 footprint of the crop (2048)
        "downsample": d,               # d=4
        "stored_px": w // d,           # 2048 / 4 = 512  -> the real image size on disk
        "image_path": member,          # zip member, so later steps can open the tile directly
    }


def build_patch_manifest(zip_path: str | Path | None = None) -> pd.DataFrame:
    """Scan the patch zip's file list and return one tidy row per H&E tile."""
    zip_path = Path(zip_path) if zip_path else paths.PATCH_ZIP
    if not zip_path.exists():
        raise FileNotFoundError(f"patch zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    # Index every mask by (split, filename) for an O(1) "does this tile have a mask?" lookup.
    mask_by_key: dict[tuple[str, str], str] = {}
    for n in names:
        p = n.split("/")
        if len(p) >= 3 and p[1] == LABELS_KIND and n.lower().endswith(".tif"):
            mask_by_key[(p[0], p[-1])] = n

    rows = []
    for n in names:
        rec = parse_tile_member(n)
        if rec is None:
            continue
        fname = rec["image_path"].split("/")[-1]          # mask shares the tile's exact filename
        mask = mask_by_key.get((rec["split"], fname))
        rec["has_mask"] = mask is not None
        rec["mask_path"] = mask or ""
        rows.append(rec)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["split", "slide_id", "y", "x"]).reset_index(drop=True)
    return df
