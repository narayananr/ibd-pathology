"""Tests for the Step-5 attention-MIL head (`ibdpath.mil`) — synthetic data.

Needs torch, so run in the HEAVY env:
    .venv-embed/bin/python -m unittest tests.test_mil -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from ibdpath import mil  # noqa: E402

DIM = 32


def _model():
    torch.manual_seed(0)
    return mil.GatedAttentionMIL(in_dim=DIM, proj_dim=16, attn_dim=16, dropout=0.0).eval()


class TestForward(unittest.TestCase):
    def test_output_shapes(self):
        model = _model()
        x = torch.randn(3, 9, DIM)          # batch of 3 bags, 9 tiles each
        logit, attn = model(x)
        self.assertEqual(tuple(logit.shape), (3,))
        self.assertEqual(tuple(attn.shape), (3, 9))

    def test_attention_sums_to_one_per_bag(self):
        model = _model()
        x = torch.randn(4, 11, DIM)
        _, attn = model(x)
        np.testing.assert_allclose(attn.sum(dim=1).detach().numpy(), np.ones(4), atol=1e-5)
        self.assertTrue((attn.detach().numpy() >= 0).all())   # softmax -> non-negative


class TestMasking(unittest.TestCase):
    def test_padded_tiles_get_zero_attention(self):
        model = _model()
        x = torch.randn(1, 10, DIM)
        mask = torch.ones(1, 10, dtype=torch.bool)
        mask[0, 6:] = False                                   # last 4 tiles are padding
        _, attn = model(x, mask=mask)
        a = attn.detach().numpy()[0]
        np.testing.assert_allclose(a[6:], 0.0, atol=1e-6)     # padding wins no attention
        np.testing.assert_allclose(a.sum(), 1.0, atol=1e-5)   # ...and the real tiles still sum to 1

    def test_padding_does_not_change_real_result(self):
        """A bag, vs the same bag with extra masked padding rows, gives the same logit + attention."""
        model = _model()
        real = torch.randn(1, 6, DIM)
        logit_a, attn_a = model(real)

        padded = torch.cat([real, torch.randn(1, 4, DIM)], dim=1)
        mask = torch.ones(1, 10, dtype=torch.bool)
        mask[0, 6:] = False
        logit_b, attn_b = model(padded, mask=mask)

        np.testing.assert_allclose(logit_a.detach().numpy(), logit_b.detach().numpy(), atol=1e-5)
        np.testing.assert_allclose(attn_a.detach().numpy()[0],
                                   attn_b.detach().numpy()[0, :6], atol=1e-5)


class TestTrainPredict(unittest.TestCase):
    def test_learns_separable_bags(self):
        """Active bags contain a 'hot' tile (shifted vector); inactive bags don't. The head should
        learn to separate them AND put its attention on the hot tile of active bags."""
        rng = np.random.RandomState(0)
        bags, y = [], []
        for k in range(24):
            label = k % 2
            bag = rng.randn(8, DIM).astype("float32")
            if label == 1:
                bag[3] += 6.0                      # one unmistakably "active" tile
            bags.append(bag)
            y.append(label)
        y = np.array(y)

        model = mil.train_mil(bags, y, in_dim=DIM, proj_dim=16, epochs=120,
                              batch_bags=8, seed=0, device="cpu")
        probs = np.array([mil.predict_bag(model, b, device="cpu")[0] for b in bags])
        # active bags should score clearly higher than inactive ones
        self.assertGreater(probs[y == 1].mean(), probs[y == 0].mean() + 0.3)

        p, a = mil.predict_bag(model, bags[1], device="cpu")   # bags[1] is active (k=1)
        self.assertAlmostEqual(float(a.sum()), 1.0, places=4)
        self.assertEqual(int(np.argmax(a)), 3)                 # attention found the hot tile

    def test_predict_bag_returns_valid_prob(self):
        model = _model()
        p, a = mil.predict_bag(model, np.random.randn(5, DIM).astype("float32"), device="cpu")
        self.assertTrue(0.0 <= p <= 1.0)
        self.assertEqual(a.shape, (5,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
