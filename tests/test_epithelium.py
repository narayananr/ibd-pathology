"""Tests for the epithelium mask helpers (`ibdpath.epithelium`) — light env:
    .venv/bin/python -m unittest tests.test_epithelium -v
"""
from __future__ import annotations

import io
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibdpath import epithelium as ep   # noqa: E402


class TestPureFunctions(unittest.TestCase):
    def test_epithelium_fraction(self):
        m = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        self.assertAlmostEqual(ep.epithelium_fraction(m), 0.75)

    def test_overlay_keeps_size_and_rgb(self):
        tile = Image.fromarray(np.full((16, 16, 3), 200, np.uint8))
        mask = np.zeros((16, 16), np.uint8); mask[4:12, 4:12] = 1
        out = ep.overlay(tile, mask)
        self.assertEqual(out.size, (16, 16))
        self.assertEqual(out.mode, "RGB")
        # tinted region should differ from the untouched corner
        a = np.asarray(out)
        self.assertFalse(np.array_equal(a[8, 8], a[0, 0]))

    def test_edges_on_block(self):
        m = np.zeros((10, 10), bool); m[3:7, 3:7] = True
        e = ep._edges(m)
        self.assertTrue(e.any())            # a filled block has a boundary
        self.assertFalse(e[0, 0])           # far-away pixel is not an edge


class TestZipLoaders(unittest.TestCase):
    def _zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            b = io.BytesIO(); Image.fromarray(np.full((8, 8, 3), 120, np.uint8)).save(b, "TIFF")
            z.writestr("t.tif", b.getvalue())
            bm = io.BytesIO(); Image.fromarray(np.eye(8, dtype=np.uint8)).save(bm, "TIFF")
            z.writestr("m.tif", bm.getvalue())
        return zipfile.ZipFile(io.BytesIO(buf.getvalue()))

    def test_load_tile_and_mask(self):
        zf = self._zip()
        tile = ep.load_tile(zf, "t.tif")
        self.assertEqual(tile.size, (8, 8))
        self.assertEqual(tile.mode, "RGB")
        mask = ep.load_mask(zf, "m.tif")
        self.assertEqual(mask.shape, (8, 8))
        self.assertLessEqual(set(np.unique(mask).tolist()), {0, 1})
        self.assertEqual(int(mask.sum()), 8)   # identity matrix -> 8 epithelium pixels


if __name__ == "__main__":
    unittest.main(verbosity=2)
