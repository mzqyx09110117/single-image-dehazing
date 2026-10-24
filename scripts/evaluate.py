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
from src.metrics import psnr, ssim  # noqa: E402


METHODS = ["raw", "baseline", "improved"]


def read_meta(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_one(gt, hazy, method, **overrides):
    start = time.perf_counter()
    if method == "hazy":
        out = hazy
    else:
        out, _ = dehaze_image(hazy, method=method, **overrides)
    cost = (time.perf_counter() - start) * 1000.0
    return out, cost


def main():
    parser = argparse.ArgumentParser(description="在合成数据上评测各种方法")
    parser.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_meta(data_dir / "metadata.csv")
    rows = []

    for item in meta:
        image = item["image"]
        beta = float(item["beta"])
        gt = cv2.imread(str(data_dir / "gt" / f"{image}.png"))
        hazy = cv2.imread(str(data_dir / "hazy" / f"{image}_b{beta}.png"))
        if gt is None or hazy is None:
            raise SystemExit(f"读图失败: {image} / beta={beta}")

        for method in ["hazy"] + METHODS:
            out, cost = run_one(gt, hazy, method)
            rows.append(
                {
                    "split": item["split"],
                    "method": method,
                    "image": image,
                    "level": item["level"],
                    "beta": beta,
                    "psnr": round(psnr(gt, out), 4),
                    "ssim": round(ssim(gt, out), 5),
                    "time_ms": round(cost, 2),
                }
            )

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 按 split / method / level 汇总
    summary = {}
    for row in rows:
        key = (row["split"], row["method"], row["level"])
        summary.setdefault(key, []).append(row)
    summary_rows = []
    for (split, method, level), group in sorted(summary.items()):
        summary_rows.append(
            {
                "split": split,
                "method": method,
                "level": level,
                "psnr_mean": round(float(np.mean([g["psnr"] for g in group])), 3),
                "psnr_std": round(float(np.std([g["psnr"] for g in group])), 3),
                "ssim_mean": round(float(np.mean([g["ssim"] for g in group])), 4),
                "ssim_std": round(float(np.std([g["ssim"] for g in group])), 4),
                "time_ms_mean": round(float(np.mean([g["time_ms"] for g in group])), 1),
            }
        )
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # 调参集上的消融实验
    variants = [
        ("baseline", {}),
        ("+sky", {"use_sky": True}),
        ("+omega", {"use_sky": True, "adaptive_omega": True, "omega_cap": 0.95}),
        (
            "+win41",
            {
                "use_sky": True,
                "adaptive_omega": True,
                "omega_cap": 0.95,
                "guided_win": 41,
                "guided_eps": 1e-3,
            },
        ),
        (
            "+t0",
            {
                "use_sky": True,
                "adaptive_omega": True,
                "omega_cap": 0.95,
                "guided_win": 41,
                "guided_eps": 1e-3,
                "t0": 0.35,
            },
        ),
        (
            "+gamma",
            {
                "use_sky": True,
                "adaptive_omega": True,
                "omega_cap": 0.95,
                "guided_win": 41,
                "guided_eps": 1e-3,
                "t0": 0.35,
                "gamma": 0.5,
            },
        ),
    ]

    abl_rows = []
    for name, overrides in variants:
        ps, ss, ts = [], [], []
        for item in meta:
            if item["split"] != "tune":
                continue
            image = item["image"]
            beta = float(item["beta"])
            gt = cv2.imread(str(data_dir / "gt" / f"{image}.png"))
            hazy = cv2.imread(str(data_dir / "hazy" / f"{image}_b{beta}.png"))
            out, cost = run_one(gt, hazy, "baseline", **overrides)
            ps.append(psnr(gt, out))
            ss.append(ssim(gt, out))
            ts.append(cost)
        abl_rows.append(
            {
                "variant": name,
                "psnr": round(float(np.mean(ps)), 3),
                "ssim": round(float(np.mean(ss)), 4),
                "time_ms": round(float(np.mean(ts)), 1),
            }
        )

    abl_path = out_dir / "ablation.csv"
    with open(abl_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(abl_rows[0].keys()))
        writer.writeheader()
        writer.writerows(abl_rows)

    print(f"写入 {metrics_path}")
    print(f"写入 {summary_path}")
    print(f"写入 {abl_path}")
    print("\n测试集平均:")
    for method in ["hazy", "raw", "baseline", "improved"]:
        group = [r for r in rows if r["split"] == "test" and r["method"] == method]
        if group:
            print(
                f"  {method:9s} PSNR {np.mean([g['psnr'] for g in group]):6.2f}  "
                f"SSIM {np.mean([g['ssim'] for g in group]):.3f}"
            )


if __name__ == "__main__":
    main()
