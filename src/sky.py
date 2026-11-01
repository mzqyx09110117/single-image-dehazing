import cv2
import numpy as np


def dark_channel(img, patch=15):
    """求暗通道：三通道取最小后做最小值滤波。"""
    dc = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(dc, kernel)


def atmospheric_light_dark(img, dark):
    """原论文的做法：暗通道最亮 0.1% 像素的颜色均值。"""
    n = max(int(dark.size * 0.001), 1)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    return img.reshape(-1, 3)[idx].mean(axis=0)


def sky_mask(img):
    """用亮度、饱和度、局部方差和位置做一个粗略天空掩膜。

    返回二值掩膜 keep 和经过高斯模糊的软掩膜 soft。
    天空面积太小的时候，调用方应该回退到原来的大气光估计。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)

    mean = cv2.boxFilter(gray, cv2.CV_32F, (15, 15), borderType=cv2.BORDER_REFLECT)
    mean2 = cv2.boxFilter(gray * gray, cv2.CV_32F, (15, 15), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0))

    yy = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    cand = ((yy < 0.55) & (sat < 0.35) & (mx > 0.55) & (std < 0.06)).astype(np.uint8)

    # 先闭后开，补掉亮云的小洞，再去掉零碎点
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.zeros_like(cand)
    top_limit = max(1, int(0.02 * h))
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area >= 0.02 * h * w and y <= top_limit:
            keep[labels == i] = 1

    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 7)
    return keep, soft


def atmospheric_light_sky(img, gray, keep):
    """在天空区域里取最亮 10% 像素的颜色均值。"""
    vals = gray[keep == 1]
    if vals.size == 0:
        return None
    thr = np.percentile(vals, 90)
    sel = (keep == 1) & (gray >= thr)
    return img[sel].mean(axis=0)
