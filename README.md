# 单幅图像去雾：暗通道先验的复现与改进

这是图像处理课程设计。基线是 He 等人 2009 年提出的暗通道先验（Dark Channel Prior, DCP），参考了 [He-Zhang/image_dehaze](https://github.com/He-Zhang/image_dehaze) 的 MIT 实现；代码按论文重新写成了几个小模块，并在此基础上做了几处修改。

## 环境

- Python 3.10+
- opencv-python
- numpy
- scikit-image
- matplotlib

```bash
pip install -r requirements.txt
```

## 运行

在仓库根目录执行：

```bash
# 1. 生成 36 对合成雾图（已经生成过的话可以跳过）
python scripts/make_hazy.py

# 2. 在调参集和测试集上评测，写 results/metrics.csv、summary.csv、ablation.csv
python scripts/evaluate.py

# 3. 处理 4 张真实雾图，结果在 results/samples/
python scripts/run_demo.py

# 4. 生成报告用的 5 张图，结果在 results/figures/
python scripts/make_figures.py
```

也可以直接跑 `bash scripts/run_all.sh`，它会把上面四步和测试一起跑一遍。

也可以直接处理一张图：

```python
import cv2
from src.dehaze import dehaze_image

img = cv2.imread("data/real/mountain.png")
out, info = dehaze_image(img, method="improved")
cv2.imwrite("out.jpg", out)
print(info["A"], info["omega"], info["sky_ratio"])
```

## 方法

基线的流程：

1. 三通道取最小，再做 15×15 的最小值滤波，得到暗通道；
2. 取暗通道最亮 0.1% 像素的颜色均值作为大气光 A；
3. 用 `t = 1 - 0.95 * dark(I/A)` 估计透射率；
4. 用灰度图作引导图，做 80×80 的引导滤波；
5. 按 `J = (I - A) / max(t, 0.30) + A` 恢复图像。

改进的地方：

- 天空掩膜：亮度、饱和度、局部方差、位置和连通域面积一起判断，只在图像上半部分找天空；
- 条件式大气光：先分别用暗通道和天空区域估计 A，只有暗通道估计明显偏亮（可能选到白色建筑/雪）时，才把天空估计混进来；
- 自适应 ω：按暗通道均值在 0.82–0.95 之间调整，轻雾少去除一点；
- 引导滤波窗口从 80×80 改成 41×41，减少过度平滑；
- 透射率下限从 0.30 提到 0.35，减少近处噪声放大。

另外试过自动 gamma，但 PSNR 掉得比较多，默认关闭，只在代码里留了开关。

## 数据

- `data/clear/`：12 张 800×600 的清晰户外图，来自 Lorem Picsum（Unsplash License），作者和链接见 `data/SOURCES.md`。
- `data/synthetic/`：用大气散射模型合成的 36 对雾图/清晰图，β = 0.8、1.4、2.0；深度图由垂直渐变和局部纹理构造；参数见 `metadata.csv`。
- `data/real/`：4 张真实雾图，来自 `He-Zhang/image_dehaze` 的 `input/` 目录（MIT），见 `NOTICE.md`。

调参集是 8 张清晰图（24 对），测试集是另外 4 张（12 对）。调参只用调参集，测试集最后跑一次。

## 结果

测试集上的平均结果（12 对）：

| 方法 | PSNR / dB | SSIM |
|---|---|---|
| 雾图 | 18.06 | 0.904 |
| 直接估计（无引导滤波） | 18.86 | 0.790 |
| 基线 | 19.51 | 0.825 |
| 改进 | **20.69** | **0.859** |

改进比基线高约 1.18 dB / 0.033 SSIM。分雾浓度看，中雾提升最大；重雾的 PSNR 略低，但 SSIM 更高；轻雾两种方法都低于雾图，说明轻雾不必强行去雾。

测试集各图平均（三档雾一起平均）：

| 图像 | 基线 PSNR / SSIM | 改进 PSNR / SSIM |
|---|---|---|
| 雪景 | 18.35 / 0.826 | 19.60 / 0.861 |
| 湖面 | 19.30 / 0.824 | 19.86 / 0.849 |
| 白墙 | 20.31 / 0.828 | 21.42 / 0.862 |
| 农田 | 20.07 / 0.823 | 21.88 / 0.863 |

调参集上的消融：

| 变体 | PSNR / dB | SSIM |
|---|---|---|
| 基线 | 22.20 | 0.832 |
| +天空大气光 | 22.26 | 0.832 |
| +自适应 ω | 22.31 | 0.829 |
| +窗口 41 | 22.56 | 0.839 |
| +t0=0.35 | 22.51 | 0.850 |
| +gamma | 20.94 | 0.824 |

![合成雾图对比](results/figures/fig_synthetic.png)

![真实雾图对比](results/figures/fig_real.png)

## 目录

```
src/                算法、指标、合成雾
scripts/            生成数据、评测、处理真实图、出图
data/               清晰图、合成图、真实雾图
results/            metrics.csv、summary.csv、ablation.csv、图片
tests/              几个简单的指标测试
```

## 局限

- 天空掩膜主要靠亮度和颜色，遇到大面积白色建筑、亮云、晚霞时会误判；天安门那张图两种方法都偏暗，是一个失败案例。
- 合成雾的深度图是人工构造的，和真实雾的分布有差距，合成数据上的结论不能直接套到所有真实场景。
- 只做了传统方法，没有和深度学习去雾做对比。

## 参考

1. K. He, J. Sun, X. Tang. Single Image Haze Removal Using Dark Channel Prior. CVPR 2009.
2. K. He, J. Sun, X. Tang. Guided Image Filtering. ECCV 2010.
3. He-Zhang/image_dehaze: https://github.com/He-Zhang/image_dehaze

## License

MIT，见 `LICENSE`。第三方图片和基线来源见 `NOTICE.md`。
