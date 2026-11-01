import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sky import atmospheric_light_sky, sky_mask  # noqa: E402


class TestSky(unittest.TestCase):
    def test_mask_only_in_upper_part(self):
        h, w = 120, 160
        img = np.full((h, w, 3), 0.25, dtype=np.float32)
        img[: int(0.4 * h)] = 0.92  # 上半部分是亮而均匀的天空
        keep, soft = sky_mask(img)
        self.assertEqual(keep.shape, (h, w))
        self.assertEqual(soft.shape, (h, w))
        self.assertGreater(int(keep[: int(0.5 * h)].sum()), 0)
        # 位置约束：下半部分不允许被选中
        self.assertEqual(int(keep[int(0.6 * h):].sum()), 0)

    def test_no_sky_gives_empty_mask_and_none(self):
        img = np.full((80, 80, 3), 0.2, dtype=np.float32)
        keep, _ = sky_mask(img)
        self.assertEqual(int(keep.sum()), 0)
        gray = img.mean(axis=2)
        self.assertIsNone(atmospheric_light_sky(img, gray, keep))


if __name__ == "__main__":
    unittest.main()