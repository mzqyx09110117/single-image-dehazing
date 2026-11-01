import csv
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synth import add_haze, make_depth  # noqa: E402


class TestSynth(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.clear = rng.integers(40, 220, (64, 80, 3), dtype=np.uint8)

    def test_deterministic(self):
        a, da, ta = add_haze(self.clear, beta=1.4, seed=7)
        b, db, tb = add_haze(self.clear, beta=1.4, seed=7)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(da, db)
        np.testing.assert_array_equal(ta, tb)

    def test_atmospheric_model_and_channel_order(self):
        hazy, _, t = add_haze(self.clear, beta=1.0, seed=3, noise=0.0)
        clear = self.clear.astype(np.float32) / 255.0
        A = np.array([0.86, 0.89, 0.92], dtype=np.float32)  # B, G, R
        expect = clear * t[..., None] + A * (1 - t[..., None])
        diff = np.abs(hazy.astype(np.float32) / 255.0 - expect).max()
        self.assertLess(float(diff), 1.5 / 255)

    def test_depth_range(self):
        depth = make_depth(self.clear)
        self.assertEqual(depth.shape, self.clear.shape[:2])
        self.assertGreaterEqual(float(depth.min()), 0.0)
        self.assertLessEqual(float(depth.max()), 1.0)

    def test_dataset_split(self):
        meta = ROOT / "data" / "synthetic" / "metadata.csv"
        if not meta.exists():
            self.skipTest("仓库里没有合成数据")
        with meta.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        groups = {}
        for r in rows:
            groups.setdefault(r["split"], set()).add(r["image_id"])
        self.assertEqual(len(rows), 36)
        self.assertEqual(len(groups.get("test", ())), 4)
        self.assertEqual(len(groups.get("tune", ())), 8)


if __name__ == "__main__":
    unittest.main()