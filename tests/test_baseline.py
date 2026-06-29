"""Tests for the Step-4 baseline logic (`ibdpath.baseline`) — synthetic data, light env:
    .venv/bin/python -m unittest tests.test_baseline -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibdpath import baseline, embed   # noqa: E402


class TestBuildSlideDataset(unittest.TestCase):
    def test_meanpool_label_and_patient_grouping(self):
        enc = "tmp_baseline_enc"
        labels = pd.DataFrame({"slide_id": ["1_HE", "2_HE", "2_2_HE"], "slide_target": [1, 0, 0]})
        arrs = {}
        for i, sid in enumerate(labels["slide_id"]):
            a = np.random.RandomState(i).randn(7, embed.EMBED_DIM).astype(np.float32)
            arrs[sid] = a
            embed.save_slide_embedding(enc, sid, a)
        try:
            X, y, groups, ids = baseline.build_slide_dataset(enc, labels=labels)
            self.assertEqual(X.shape, (3, embed.EMBED_DIM))
            np.testing.assert_allclose(X[ids.index("1_HE")], arrs["1_HE"].mean(0), rtol=1e-5)
            self.assertEqual(list(y), [1, 0, 0])
            # 2_HE and 2_2_HE are the same patient ("2") -> same group
            self.assertEqual(groups[ids.index("2_HE")], groups[ids.index("2_2_HE")])
        finally:
            for sid in labels["slide_id"]:
                embed.slide_embedding_path(enc, sid).unlink(missing_ok=True)
            embed.embeddings_dir(enc).rmdir()


class TestAUROC(unittest.TestCase):
    def test_separable_data_scores_high(self):
        rng = np.random.RandomState(0)
        n, d = 40, 16
        y = np.array([0, 1] * (n // 2))
        X = rng.randn(n, d) + y[:, None] * 3.0          # active class shifted -> easily separable
        groups = np.arange(n)                            # all distinct patients
        auroc, oof = baseline.leave_patients_out_auroc(X, y, groups, n_splits=4)
        self.assertGreater(auroc, 0.9)
        self.assertEqual(oof.shape, (n,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
