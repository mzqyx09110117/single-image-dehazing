import cv2
import numpy as np


def make_depth(clear_bgr):
    """构造一个简单的相对深度图。

    上面远、下面近，再叠一点图像自身的纹理：纹理多的地方认为离镜头近。
    这不是真实的深度，只是用来合成雾图，方便做配对评价。
    """
    h, w = clear_bgr.shape[:2]
    gray = cv2.cvtColor(clear_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    mean = cv2.boxFilter(gray, cv2.CV_32F, (15, 15), borderType=cv2.BORDER_REFLECT)
    mean2 = cv2.boxFilter(gray * gray, cv2.CV_32F, (15, 15), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0))
    texture = np.clip(std / 0.15, 0, 1)

    yy = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    depth = 0.65 * (1 - yy) + 0.35 * (1 - texture)
    depth = cv2.GaussianBlur(depth, (0, 0), 9)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return (0.05 + 0.90 * depth).astype(np.float32)


def add_haze(clear_bgr, beta, seed, A=(0.86, 0.89, 0.92), noise=0.005):
    """按大气散射模型 I = J*t + A*(1-t) 合成雾图。

    beta 越大雾越重。返回值是 uint8 雾图、深度图和透射率。
    """
    rng = np.random.default_rng(seed)
    clear = clear_bgr.astype(np.float32) / 255.0
    depth = make_depth(clear_bgr)
    t = np.exp(-beta * depth).astype(np.float32)

    A = np.array(A, dtype=np.float32)
    hazy = clear * t[..., None] + A * (1 - t[..., None])
    hazy += rng.normal(0, noise, hazy.shape).astype(np.float32)
    hazy = np.clip(hazy, 0, 1)
    return (hazy * 255).round().astype(np.uint8), depth, t
