import cv2
import numpy as np


def guided_filter(guide, src, win=41, eps=1e-3):
    """用灰度引导图对单通道图做引导滤波。

    guide: H×W 的 float32 灰度图，范围 [0, 1]
    src:   H×W 的 float32 单通道图，比如透射率
    win:   滤波窗口大小（奇数或偶数都行，内部直接用 win×win 的方框）
    eps:   正则项，越大越平滑
    """
    win = int(win)
    ksize = (win, win)

    mean_i = cv2.boxFilter(guide, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_ii = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    cov_ip = mean_ip - mean_i * mean_p
    var_i = mean_ii - mean_i * mean_i
    var_i = np.maximum(var_i, 0)

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    return mean_a * guide + mean_b
