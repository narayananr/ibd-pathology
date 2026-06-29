"""Tests for the embedding cache layer (`ibdpath.embed`).

This module is torch-free, so these run in the LIGHT env:
    .venv/bin/python -m unittest tests.test_embed -v
(The actual model forward pass is validated by running scripts/03_embed_tiles.py.)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibdpath import embed, paths   # noqa: E402


class TestL2Normalize(unittest.TestCase):
    def test_rows_become_unit_length(self):
        x = np.array([[3.0, 4.0], [1.0, 1.0]])          # norms 5 and sqrt(2)
        y = embed.l2_normalize(x)
        np.testing.assert_allclose(np.linalg.norm(y, axis=1), [1.0, 1.0], atol=1e-6)

    def test_zero_row_stays_zero(self):
        y = embed.l2_normalize(np.array([[0.0, 0.0]]))   # must not divide by zero
        np.testing.assert_array_equal(y, [[0.0, 0.0]])


class TestCachePaths(unittest.TestCase):
    def test_path_layout(self):
        p = embed.slide_embedding_path("hoptimus0", "116_2_HE")
        self.assertEqual(p.name, "116_2_HE.npy")
        self.assertEqual(p.parent.name, "hoptimus0")
        self.assertTrue(str(p).startswith(str(paths.ARTIFACTS_DIR)))


class TestSaveLoadRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        enc = "tmp_test_encoder"
        arr = np.arange(5 * embed.EMBED_DIM, dtype=np.float32).reshape(5, embed.EMBED_DIM)
        p = embed.save_slide_embedding(enc, "9_HE", arr)
        try:
            back = embed.load_slide_embedding(enc, "9_HE")
            self.assertEqual(back.shape, (5, embed.EMBED_DIM))
            np.testing.assert_array_equal(back, arr)
        finally:
            p.unlink(missing_ok=True)
            p.parent.rmdir()


if __name__ == "__main__":
    unittest.main(verbosity=2)
