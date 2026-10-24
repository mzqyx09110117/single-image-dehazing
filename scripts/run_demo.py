import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dehaze import dehaze_image  # noqa: E402
from src.metrics import avg_gradient, colorfulness, contrast, entropy  # noqa: E402


def measure(img):
    return {
        "contrast": round(contrast(img), 2),
        "gradient": round(avg_gradient(img), 2),
        "colorfulness": round(colorfulness(img), 2),
        "entropy": round(entropy(img), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="处理真实雾图并保存对比结果")
    parser.add_argument("--input", default=str(ROOT / "data" / "real"))
    parser.add_argument("--out", default=str(ROOT / "results" / "samples"))
    parser.add_argument("--metrics", default=str(ROOT / "results" / "real_metrics.csv"))
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png")))
    if not files:
        raise SystemExit(f"没有找到真实雾图: {in_dir}")

    rows = []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            print(f"读图失败: {path}")
            continue
        stem = path.stem

        rows.append({"method": "hazy", "image": stem, **measure(image), "time_ms": 0.0})
        cv2.imwrite(str(out_dir / f"real_{stem}_hazy.jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        for method in ["baseline", "improved"]:
            start = time.perf_counter()
            out, info = dehaze_image(image, method=method)
            cost = (time.perf_counter() - start) * 1000.0
            cv2.imwrite(str(out_dir / f"real_{stem}_{method}.jpg"), out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(
                {
                    "method": method,
                    "image": stem,
                    **measure(out),
                    "time_ms": round(cost, 1),
                }
            )
            if method == "improved":
                print(
                    f"{stem:12s} A={np.round(info['A'], 2)} "
                    f"omega={info['omega']:.2f} sky={info['sky_ratio']:.2f}"
                )

    out_path = Path(args.metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"写入 {out_dir}")
    print(f"写入 {out_path}")


if __name__ == "__main__":
    main()
