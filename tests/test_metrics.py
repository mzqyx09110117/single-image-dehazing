import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import entropy, psnr, ssim  # noqa: E402
from src.sky import dark_channel  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_psnr_identical(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(psnr(img, img), float("inf"))

    def test_psnr_gray_half(self):
        gt = np.zeros((8, 8, 3), dtype=np.float32)
        out = np.full((8, 8, 3), 0.5, dtype=np.float32)
        # MSE = 0.5^2 = 0.25，PSNR = 10*log10(4) ≈ 6.02 dB
        self.assertAlmostEqual(psnr(gt, out), 6.0206, places=3)

    def test_ssim_identical(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        self.assertAlmostEqual(ssim(img, img), 1.0, places=5)

    def test_entropy_float_matches_uint8(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        self.assertAlmostEqual(
            entropy(img), entropy(img.astype(np.float32) / 255.0), places=2
        )

    def test_dark_channel_shape(self):
        img = np.ones((20, 30, 3), dtype=np.float32)
        dc = dark_channel(img, patch=5)
        self.assertEqual(dc.shape, (20, 30))
        self.assertAlmostEqual(float(dc.mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
