import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dehaze import dehaze_image  # noqa: E402
from src.synth import add_haze, make_depth  # noqa: E402


def setup_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            break
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read(path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"读图失败: {path}")
    return img


def panel_image(img, mode):
    if mode == "color":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def grid(rows, titles, out_path, figsize, modes=None, ranges=None, row_labels=None,
         title_size=11, label_size=10):
    nrows = len(rows)
    ncols = len(rows[0])
    modes = modes or ["color"] * ncols
    ranges = ranges or [(None, None)] * ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            img = panel_image(rows[r][c], modes[c])
            vmin, vmax = ranges[c]
            if modes[c] == "color":
                ax.imshow(img)
            else:
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=title_size, pad=4)
            if c == 0 and row_labels:
                ax.text(
                    -0.03, 0.5, row_labels[r], transform=ax.transAxes,
                    ha="right", va="center", rotation=90, fontsize=label_size,
                )

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"写入 {out_path}")


def main():
    setup_font()
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 图 1：一张图的深度图和三个雾档
    clear = read(ROOT / "data" / "clear" / "picsum_1015.jpg")
    depth = make_depth(clear)
    light, _, _ = add_haze(clear, beta=0.8, seed=2026 + 1015 + 8)
    medium, _, _ = add_haze(clear, beta=1.4, seed=2026 + 1015 + 14)
    heavy, _, _ = add_haze(clear, beta=2.0, seed=2026 + 1015 + 20)
    grid(
        [[clear, depth, light, medium, heavy]],
        ["清晰图", "构造深度图", "轻雾 β=0.8", "中雾 β=1.4", "重雾 β=2.0"],
        fig_dir / "fig_synthesis.png",
        figsize=(8.0, 1.9),
        modes=["color", "gray", "color", "color", "color"],
        ranges=[(None, None), (0.0, 1.0), (None, None), (None, None), (None, None)],
    )

    # 图 2：测试集的合成图对比（中雾）
    test_ids = [1036, 1051, 49, 85]
    labels = ["雪景", "湖面", "白墙", "农田"]
    rows = []
    for image_id in test_ids:
        clear = read(ROOT / "data" / "clear" / f"picsum_{image_id}.jpg")
        hazy, _, _ = add_haze(clear, beta=1.4, seed=2026 + image_id + 14)
        base, _ = dehaze_image(hazy, method="baseline")
        imp, _ = dehaze_image(hazy, method="improved")
        rows.append([clear, hazy, base, imp])
    grid(
        rows,
        ["清晰图", "雾图", "基线", "改进"],
        fig_dir / "fig_synthetic.png",
        figsize=(7.2, 5.8),
        row_labels=labels,
    )

    # 图 3：真实雾图对比
    real_names = ["forest1", "tiananmen1", "mountain", "IMG_8766"]
    real_labels = ["森林", "天安门", "山景", "城市"]
    rows = []
    for name in real_names:
        hazy = read(ROOT / "data" / "real" / f"{name}.{'png' if name in ('tiananmen1', 'mountain') else 'jpg'}")
        base, _ = dehaze_image(hazy, method="baseline")
        imp, _ = dehaze_image(hazy, method="improved")
        rows.append([hazy, base, imp])
    grid(
        rows,
        ["雾图", "基线", "改进"],
        fig_dir / "fig_real.png",
        figsize=(6.8, 6.4),
        row_labels=real_labels,
    )

    # 图 4：中间量（暗通道、透射率、天空掩膜）
    clear = read(ROOT / "data" / "clear" / "picsum_49.jpg")
    hazy, _, _ = add_haze(clear, beta=1.4, seed=2026 + 49 + 14)
    _, info = dehaze_image(hazy, method="improved")
    dark = info["dark"]
    dark_max = max(float(dark.max()), 1e-6)
    grid(
        [[hazy, dark, info["t_raw"], info["t"], info["soft"]]],
        ["雾图", "暗通道", "原始透射率", "细化后透射率", "天空软掩膜"],
        fig_dir / "fig_transmission.png",
        figsize=(8.0, 1.9),
        modes=["color", "gray", "gray", "gray", "gray"],
        ranges=[(None, None), (0.0, dark_max), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    )

    # 图 5：失败案例（天安门，白色城墙+暖色天空）
    hazy = read(ROOT / "data" / "real" / "tiananmen1.png")
    base, _ = dehaze_image(hazy, method="baseline")
    imp, _ = dehaze_image(hazy, method="improved")
    grid(
        [[hazy, base, imp]],
        ["雾图", "基线", "改进"],
        fig_dir / "fig_failure.png",
        figsize=(6.8, 2.2),
    )


if __name__ == "__main__":
    main()
