"""Tests for Step 1 — the patch-manifest builder (`ibdpath.manifest`).

We test the *logic*, not the runner's printing. Three layers:

1. UNIT tests on tiny made-up filenames           -> instant, need no data file.
2. A whole-pipeline test on a SYNTHETIC mini-zip   -> built in a temp folder, still no real data.
3. An INTEGRATION test against the real 8.76 GB zip -> auto-SKIPPED if the zip isn't present,
   so this suite passes on any machine.

Run from the repo root:
    .venv/bin/python -m unittest tests.test_manifest -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

# Make `import ibdpath` work without installing the package (same trick as the script).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibdpath import paths                                            # noqa: E402
from ibdpath.manifest import COLUMNS, build_patch_manifest, parse_tile_member  # noqa: E402


class TestParseTileMember(unittest.TestCase):
    """parse_tile_member: one zip path in -> one record dict (or None)."""

    def test_valid_image_tile_is_parsed(self):
        member = "Trainset/Images_tif/107_HE [d=4,x=1536,y=10752,w=2048,h=2048].tif"
        rec = parse_tile_member(member)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["split"], "Trainset")
        self.assertEqual(rec["slide_id"], "107_HE")
        self.assertEqual((rec["x"], rec["y"]), (1536, 10752))
        self.assertEqual((rec["w"], rec["h"], rec["downsample"]), (2048, 2048, 4))
        self.assertEqual(rec["stored_px"], 512)            # 2048 // 4
        self.assertEqual(rec["image_path"], member)        # kept verbatim, to open the tile later

    def test_mask_entry_is_ignored(self):
        # Labels_tif is an epithelium mask (DEFERRED scope) -> not an image tile -> None
        self.assertIsNone(parse_tile_member(
            "Trainset/Labels_tif/107_HE [d=4,x=1536,y=10752,w=2048,h=2048].tif"))

    def test_mib_format_is_ignored(self):
        self.assertIsNone(parse_tile_member(
            "Trainset/Images_mibImg/107_HE [d=4,x=1536,y=10752,w=2048,h=2048].mibImg"))

    def test_folder_entry_is_ignored(self):
        self.assertIsNone(parse_tile_member("Trainset/Images_tif/"))

    def test_match_is_case_insensitive(self):
        rec = parse_tile_member("Testset/Images_tif/9_HE [d=2,x=0,y=0,w=1024,h=1024].TIF")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["slide_id"], "9_HE")
        self.assertEqual(rec["stored_px"], 512)            # 1024 // 2

    def test_non_tif_extension_is_ignored(self):
        # under Images_tif but not a .tif (e.g. a stray .png) -> regex fails -> None
        self.assertIsNone(parse_tile_member(
            "Trainset/Images_tif/107_HE [d=4,x=0,y=0,w=2048,h=2048].png"))

    def test_non_he_slide_is_ignored(self):
        # the regex requires the slide id to end in _HE, so a CD3 tile is skipped
        self.assertIsNone(parse_tile_member(
            "Trainset/Images_tif/107_CD3 [d=4,x=0,y=0,w=2048,h=2048].tif"))

    def test_space_before_bracket_is_optional(self):
        # `\s*` in the regex means the space between slide id and "[" may be absent
        rec = parse_tile_member("Trainset/Images_tif/107_HE[d=4,x=0,y=0,w=2048,h=2048].tif")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["slide_id"], "107_HE")

    def test_downsample_one_keeps_full_size(self):
        # stored_px = w // d, so d=1 means the stored tile is the full footprint
        rec = parse_tile_member("Testset/Images_tif/5_HE [d=1,x=0,y=0,w=256,h=256].tif")
        self.assertEqual(rec["downsample"], 1)
        self.assertEqual(rec["stored_px"], 256)


class TestBuildPatchManifest(unittest.TestCase):
    """build_patch_manifest: a whole (tiny, synthetic) zip -> a tidy table."""

    @staticmethod
    def _make_zip(tmp: Path) -> Path:
        """Create a small zip that mimics the real layout (no pixels, just names)."""
        zpath = tmp / "mini.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            # 107_HE: an image tile WITH a matching mask
            zf.writestr("Trainset/Images_tif/107_HE [d=4,x=0,y=0,w=2048,h=2048].tif", b"")
            zf.writestr("Trainset/Labels_tif/107_HE [d=4,x=0,y=0,w=2048,h=2048].tif", b"")
            # 108_HE: an image tile with NO matching mask
            zf.writestr("Trainset/Images_tif/108_HE [d=4,x=512,y=512,w=2048,h=2048].tif", b"")
            # 200_HE in a different split, with a mask
            zf.writestr("Testset/Images_tif/200_HE [d=4,x=0,y=0,w=2048,h=2048].tif", b"")
            zf.writestr("Testset/Labels_tif/200_HE [d=4,x=0,y=0,w=2048,h=2048].tif", b"")
            # noise that MUST be ignored: a MIB-format copy + a bare folder entry
            zf.writestr("Trainset/Images_mibImg/107_HE [d=4,x=0,y=0,w=2048,h=2048].mibImg", b"")
            zf.writestr("Trainset/Images_tif/", b"")
        return zpath

    def test_builds_expected_rows(self):
        with tempfile.TemporaryDirectory() as d:
            df = build_patch_manifest(self._make_zip(Path(d)))

        # only the 3 real image tiles survive (the MIB copy + folder entry are excluded)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), COLUMNS)
        self.assertEqual(set(df["slide_id"]), {"107_HE", "108_HE", "200_HE"})

        # has_mask is correct per tile, and the matched mask path is recorded
        row107 = df[df["slide_id"] == "107_HE"].iloc[0]
        row108 = df[df["slide_id"] == "108_HE"].iloc[0]
        self.assertTrue(bool(row107["has_mask"]))
        self.assertEqual(row107["mask_path"],
                         "Trainset/Labels_tif/107_HE [d=4,x=0,y=0,w=2048,h=2048].tif")
        self.assertFalse(bool(row108["has_mask"]))
        self.assertEqual(row108["mask_path"], "")

    def test_missing_zip_raises(self):
        with self.assertRaises(FileNotFoundError):
            build_patch_manifest(Path("/no/such/file.zip"))

    def test_empty_zip_yields_empty_table_with_columns(self):
        # a zip with no image tiles -> 0 rows, but still the right columns (must not crash)
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "empty.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("Trainset/Labels_tif/1_HE [d=4,x=0,y=0,w=2048,h=2048].tif", b"")
            df = build_patch_manifest(zpath)
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), COLUMNS)

    def test_mask_must_match_the_same_split(self):
        # has_mask keys on (split, filename); a same-named mask in ANOTHER split must not count
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "split.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("Trainset/Images_tif/300_HE [d=4,x=0,y=0,w=2048,h=2048].tif", b"")
                zf.writestr("Testset/Labels_tif/300_HE [d=4,x=0,y=0,w=2048,h=2048].tif", b"")
            df = build_patch_manifest(zpath)
        self.assertEqual(len(df), 1)
        self.assertFalse(bool(df.iloc[0]["has_mask"]))
        self.assertEqual(df.iloc[0]["mask_path"], "")

    def test_rows_are_sorted_deterministically(self):
        # rows come out ordered by split -> slide_id -> y -> x, regardless of zip order
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "order.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("Trainset/Images_tif/2_HE [d=4,x=100,y=200,w=2048,h=2048].tif", b"")
                zf.writestr("Trainset/Images_tif/2_HE [d=4,x=0,y=200,w=2048,h=2048].tif", b"")
                zf.writestr("Trainset/Images_tif/1_HE [d=4,x=0,y=0,w=2048,h=2048].tif", b"")
                zf.writestr("Testset/Images_tif/9_HE [d=4,x=0,y=0,w=2048,h=2048].tif", b"")
            df = build_patch_manifest(zpath)
        got = [[t.split, t.slide_id, int(t.y), int(t.x)] for t in df.itertuples()]
        self.assertEqual(got, [
            ["Testset", "9_HE", 0, 0],
            ["Trainset", "1_HE", 0, 0],
            ["Trainset", "2_HE", 200, 0],
            ["Trainset", "2_HE", 200, 100],
        ])


@unittest.skipUnless(paths.PATCH_ZIP.exists(),
                     "real patch zip not present — integration test skipped")
class TestRealZipIntegration(unittest.TestCase):
    """Sanity checks against the ACTUAL dataset (skipped if the 8.76 GB zip is absent)."""

    @classmethod
    def setUpClass(cls):
        cls.df = build_patch_manifest(paths.PATCH_ZIP)

    def test_counts_match_what_we_measured(self):
        self.assertEqual(len(self.df), 6322)
        self.assertEqual(self.df["slide_id"].nunique(), 140)
        self.assertTrue(bool(self.df["has_mask"].all()))
        self.assertEqual(sorted(self.df["stored_px"].unique()), [512])

    def test_author_split_leaks_by_slide(self):
        by = {s: set(g) for s, g in self.df.groupby("split")["slide_id"]}
        # the whole reason we don't reuse their split: Val slides are a SUBSET of Train slides
        self.assertGreater(len(by["Trainset"] & by["Validationset"]), 0)
        self.assertEqual(len(by["Trainset"] & by["Testset"]), 0)
        self.assertEqual(len(by["Validationset"] & by["Testset"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
