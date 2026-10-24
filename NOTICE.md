# 来源说明

## 基线实现

去雾的基线流程参考了下面这个 MIT 协议的项目：

- He-Zhang / image_dehaze  
  https://github.com/He-Zhang/image_dehaze

原项目是一个 Python 脚本，使用暗通道先验和引导滤波。本仓库没有直接复制整份脚本，而是按论文公式重写成了 `src/` 下的函数，并把参数、评测和出图拆开。原项目的 MIT 协议和版权声明仍然适用。

## 真实雾图

`data/real/` 里的 4 张真实雾图来自上面这个项目的 `input/` 目录，同样遵循 MIT 协议：

- forest1.jpg
- tiananmen1.png
- mountain.png
- IMG_8766.jpg

## 清晰图

`data/clear/` 里的 12 张清晰图来自 Lorem Picsum（图片本身来自 Unsplash，遵循 Unsplash License），用于合成雾图和配对评价。具体作者和原图链接见 `data/SOURCES.md`。

## 参考论文

1. K. He, J. Sun, X. Tang. Single Image Haze Removal Using Dark Channel Prior. CVPR 2009.
2. K. He, J. Sun, X. Tang. Guided Image Filtering. ECCV 2010.
