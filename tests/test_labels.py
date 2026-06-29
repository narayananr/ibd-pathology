"""Tests for Step 2 — slide labels and the manifest join (`ibdpath.labels`).

Three layers, mirroring the Step-1 tests:
1. UNIT tests on the loader, the patient_id helper, and the join (synthetic data, no files).
2. (covered above) the join against a tiny in-memory manifest.
3. INTEGRATION test against the committed metadata/slide_labels.csv (skipped if absent).

Run from the repo root:
    .venv/bin/python -m unittest tests.test_labels -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibdpath import paths                                                   # noqa: E402
from ibdpath.labels import (LABEL_TO_INT, attach_labels,                    # noqa: E402
                            load_slide_labels, patient_id)


class TestPatientId(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(patient_id("107_HE"), "107")

    def test_rebiopsy_collapses_to_one_patient(self):
        self.assertEqual(patient_id("116_HE"), "116")
        self.assertEqual(patient_id("116_2_HE"), "116")

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            patient_id("HE_only")


class TestLoadSlideLabels(unittest.TestCase):
    def test_label_to_int_convention(self):
        self.assertEqual(LABEL_TO_INT, {"inactive": 0, "active": 1})  # active is the positive class

    def _write(self, d, **cols):
        p = Path(d) / "labels.csv"
        pd.DataFrame(cols).to_csv(p, index=False)
        return p

    def test_loads_and_adds_slide_target(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, slide_id=["1_HE", "2_HE"], label=["active", "inactive"], source=["x", "x"])
            df = load_slide_labels(p)
        self.assertEqual(list(df.columns), ["slide_id", "label", "source", "slide_target"])
        self.assertEqual(df.set_index("slide_id")["slide_target"].to_dict(), {"1_HE": 1, "2_HE": 0})

    def test_bad_label_value_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, slide_id=["1_HE"], label=["typo"], source=["x"])
            with self.assertRaises(ValueError):
                load_slide_labels(p)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_slide_labels(Path("/no/such/labels.csv"))


class TestAttachLabels(unittest.TestCase):
    @staticmethod
    def _labels():
        return pd.DataFrame({"slide_id": ["1_HE", "2_HE"], "label": ["active", "inactive"],
                             "source": ["x", "x"], "slide_target": [1, 0]})

    def test_join_adds_columns_and_patient_and_keeps_y(self):
        # manifest has its own `y` (tile coordinate) — it must survive the join untouched
        manifest = pd.DataFrame({"slide_id": ["1_HE", "1_HE", "2_HE"], "y": [0, 256, 0]})
        out = attach_labels(manifest, self._labels())
        self.assertEqual(len(out), 3)
        for col in ("label", "slide_target", "patient_id"):
            self.assertIn(col, out.columns)
        self.assertEqual(out.loc[out.slide_id == "1_HE", "slide_target"].tolist(), [1, 1])
        self.assertEqual(out.loc[out.slide_id == "1_HE", "patient_id"].tolist(), ["1", "1"])
        self.assertEqual(out["y"].tolist(), [0, 256, 0])     # tile coordinate intact, no collision

    def test_unlabeled_tile_raises(self):
        manifest = pd.DataFrame({"slide_id": ["1_HE", "999_HE"], "y": [0, 0]})
        with self.assertRaises(ValueError):
            attach_labels(manifest, self._labels())


@unittest.skipUnless(paths.SLIDE_LABELS_CSV.exists(),
                     "metadata/slide_labels.csv not present — integration test skipped")
class TestCommittedLabels(unittest.TestCase):
    def test_the_real_label_table(self):
        df = load_slide_labels()
        self.assertEqual(len(df), 140)
        self.assertEqual(df["label"].value_counts().to_dict(), {"inactive": 86, "active": 54})
        self.assertEqual(int(df["slide_target"].sum()), 54)
        # 140 slides collapse to 139 patients (the 116 re-biopsy)
        self.assertEqual(df["slide_id"].map(patient_id).nunique(), 139)


if __name__ == "__main__":
    unittest.main(verbosity=2)
