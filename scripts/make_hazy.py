import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synth import add_haze  # noqa: E402


BETAS = [0.8, 1.4, 2.0]
LEVELS = ["light", "medium", "heavy"]
LEVEL_SEEDS = [8, 14, 20]

TUNE_IDS = {1015, 1016, 1019, 1039, 1043, 11, 12, 16}
TEST_IDS = {1036, 1051, 49, 85}


def main():
    parser = argparse.ArgumentParser(description="生成合成的雾图/清晰图配对")
    parser.add_argument("--clear", default=str(ROOT / "data" / "clear"))
    parser.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    args = parser.parse_args()

    clear_dir = Path(args.clear)
    out_dir = Path(args.out)
    hazy_dir = out_dir / "hazy"
    gt_dir = out_dir / "gt"
    hazy_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(clear_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"没有找到图片: {clear_dir}")

    rows = []
    for path in files:
        try:
            image_id = int(path.stem.split("_")[-1])
        except ValueError:
            print(f"跳过文件名: {path.name}")
            continue

        split = "test" if image_id in TEST_IDS else "tune"
        clear = cv2.imread(str(path))
        if clear is None:
            print(f"读图失败: {path}")
            continue

        gt_path = gt_dir / f"{path.stem}.png"
        cv2.imwrite(str(gt_path), clear)

        for beta, level, level_seed in zip(BETAS, LEVELS, LEVEL_SEEDS):
            seed = 2026 + image_id + level_seed
            hazy, _, _ = add_haze(clear, beta=beta, seed=seed)
            hazy_path = hazy_dir / f"{path.stem}_b{beta}.png"
            cv2.imwrite(str(hazy_path), hazy)
            rows.append(
                {
                    "image": path.stem,
                    "image_id": image_id,
                    "split": split,
                    "beta": beta,
                    "level": level,
                    "seed": seed,
                    "A_r": 0.86,
                    "A_g": 0.89,
                    "A_b": 0.92,
                }
            )

    meta_path = out_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"清晰图: {len(files)} 张")
    print(f"合成雾图: {len(rows)} 对")
    print(f"调参集: {sum(r['split'] == 'tune' for r in rows)} 对")
    print(f"测试集: {sum(r['split'] == 'test' for r in rows)} 对")
    print(f"参数表: {meta_path}")


if __name__ == "__main__":
    main()
