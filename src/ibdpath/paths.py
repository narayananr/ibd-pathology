"""Central place for every path the pipeline uses.

Putting paths in ONE module means no script hard-codes a directory, and if the data ever
moves we change it here once. All paths are absolute `pathlib.Path`s, resolved relative to
the repository root — this file lives at  <root>/src/ibdpath/paths.py.
"""
from __future__ import annotations

from pathlib import Path

# <root>/src/ibdpath/paths.py  ->  parents[0]=ibdpath, parents[1]=src, parents[2]=<root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- inputs (read-only) -----------------------------------------------------
DATA_DIR = REPO_ROOT / "data" / "ibdcolepi"
PATCH_ZIP = DATA_DIR / "patch-dataset-HE.zip"   # the 8.76 GB pre-tiled HE patch set

# --- curated reference data (small, committed — NOT disposable like data/) ---
METADATA_DIR = REPO_ROOT / "metadata"
SLIDE_LABELS_CSV = METADATA_DIR / "slide_labels.csv"   # slide_id -> active/inactive (Step 2 input)

# --- outputs (generated; always safe to delete and regenerate) --------------
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PATCH_MANIFEST_CSV = ARTIFACTS_DIR / "patch_manifest.csv"
PATCH_MANIFEST_LABELED_CSV = ARTIFACTS_DIR / "patch_manifest_labeled.csv"   # Step 2 output


def ensure_artifacts() -> Path:
    """Create the artifacts/ output folder if it doesn't exist yet; return it."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
