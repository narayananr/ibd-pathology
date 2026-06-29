"""Tests for the Step-6 validation helpers (`ibdpath.validate`) — light env:
    .venv/bin/python -m unittest tests.test_validate -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibdpath import validate   # noqa: E402


class TestClassificationMetrics(unittest.TestCase):
    def test_known_confusion(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.9, 0.2, 0.8]          # pred @0.5 = [0, 1, 0, 1]
        m = validate.classification_metrics(y, p, threshold=0.5)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["sensitivity"], 0.5)   # tp/(tp+fn)
        self.assertAlmostEqual(m["specificity"], 0.5)   # tn/(tn+fp)
        self.assertAlmostEqual(m["accuracy"], 0.5)

    def test_threshold_shifts_calls(self):
        y = [0, 1]
        p = [0.4, 0.6]
        # at 0.5 both correct; at 0.7 the active one is called healed (a miss)
        self.assertEqual(validate.classification_metrics(y, p, 0.5)["tp"], 1)
        self.assertEqual(validate.classification_metrics(y, p, 0.7)["tp"], 0)


class TestBootstrapCI(unittest.TestCase):
    def test_separable_high_and_bracketed(self):
        rng = np.random.RandomState(0)
        y = np.array([0] * 12 + [1] * 12)
        p = y * 0.8 + rng.uniform(0, 0.2, size=len(y))   # cleanly separable
        groups = np.arange(len(y))                        # all distinct patients
        point, lo, hi, boots = validate.bootstrap_auroc_ci(y, p, groups, n_boot=300, seed=0)
        self.assertGreater(point, 0.95)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)
        self.assertLessEqual(hi, 1.0)
        self.assertGreater(len(boots), 0)

    def test_grouping_collapses_patient(self):
        # two slides from one patient must move together when resampled -> no crash, valid CI
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        groups = np.array(["a", "a", "b", "b"])
        point, lo, hi, boots = validate.bootstrap_auroc_ci(y, p, groups, n_boot=100, seed=1)
        self.assertTrue(0.0 <= lo <= hi <= 1.0)


class TestAgreement(unittest.TestCase):
    def test_fraction(self):
        p_a = [0.9, 0.1, 0.9]    # calls 1,0,1
        p_b = [0.9, 0.1, 0.1]    # calls 1,0,0
        self.assertAlmostEqual(validate.call_agreement(p_a, p_b), 2 / 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
