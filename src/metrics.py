import cv2
import numpy as np
from skimage.metrics import structural_similarity


def to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def psnr(gt, out):
    """峰值信噪比，输入 uint8 或 [0,1] 浮点图。"""
    gt = to_float(gt)
    out = to_float(out)
    mse = np.mean((gt - out) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def ssim(gt, out):
    """结构相似度，彩色图按通道算完取平均。"""
    gt = to_float(gt)
    out = to_float(out)
    return float(structural_similarity(gt, out, channel_axis=2, data_range=1.0))


def contrast(img):
    """灰度标准差，越大通常表示对比度越高。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(np.std(gray.astype(np.float32)))


def avg_gradient(img):
    """平均梯度，用 Sobel 的 dx/dy 估计。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt((gx * gx + gy * gy) / 2.0)))


def colorfulness(img):
    """Hasler-Süsstrunk 色彩丰富度，输入 BGR。"""
    b, g, r = cv2.split(img.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def entropy(img):
    """灰度直方图熵，输入 uint8 或 [0,1] 浮点图。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    is_uint8 = gray.dtype == np.uint8
    gray = gray.astype(np.float32)
    # cv2.calcHist 的区间按 0-255 的灰阶来解释，浮点图先换算回 0-255
    if not is_uint8 and float(gray.max()) <= 1.0:
        gray *= 255.0
    gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))
