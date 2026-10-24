import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dehaze import dehaze_image  # noqa: E402


class TestDehaze(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        for method in ["raw", "baseline", "improved"]:
            out, info = dehaze_image(img, method=method)
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(info["t"].shape, (48, 64))
            self.assertGreaterEqual(int(out.min()), 0)
            self.assertLessEqual(int(out.max()), 255)


if __name__ == "__main__":
    unittest.main()
