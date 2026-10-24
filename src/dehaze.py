from dataclasses import dataclass, replace

import cv2
import numpy as np

from .guided_filter import guided_filter
from .sky import (
    atmospheric_light_dark,
    atmospheric_light_sky,
    dark_channel,
    sky_mask,
)


@dataclass
class DehazeConfig:
    guided: bool = True
    patch: int = 15
    omega: float = 0.95
    adaptive_omega: bool = False
    omega_cap: float = 0.95
    t0: float = 0.30
    guided_win: int = 80
    guided_eps: float = 4e-4
    use_sky: bool = False
    sky_mix: float = 0.5
    sky_thr: float = 0.01
    gamma: float = 0.0


PRESETS = {
    "raw": DehazeConfig(guided=False, t0=0.30),
    "baseline": DehazeConfig(),
    "improved": DehazeConfig(
        guided=True,
        guided_win=41,
        guided_eps=1e-3,
        t0=0.35,
        use_sky=True,
        sky_mix=0.5,
        sky_thr=0.01,
        adaptive_omega=True,
        omega_cap=0.95,
    ),
}


def get_config(method="baseline", **overrides):
    if isinstance(method, DehazeConfig):
        cfg = replace(method)
    else:
        if method not in PRESETS:
            raise ValueError(f"unknown method: {method}")
        cfg = replace(PRESETS[method])

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"unknown config field: {key}")
        setattr(cfg, key, value)
    return cfg


def estimate_transmission(img, A, patch=15, omega=0.95):
    norm = img / np.maximum(A, 1e-6)
    return np.clip(1.0 - omega * dark_channel(norm, patch), 0.01, 1.0)


def recover(img, t, A, t0=0.30):
    t = np.maximum(t, t0)
    J = (img - A) / np.maximum(t[..., None], 1e-3) + A
    return np.clip(J, 0.0, 1.0)


def auto_gamma(img, target=0.5, strength=0.5):
    """只把偏暗的结果往亮处拉一点，避免把本来就亮的图压暗。"""
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    mean = float(gray.astype(np.float32).mean() / 255.0)
    if mean >= target or strength <= 0:
        return img

    gamma = float(np.clip(np.log(target) / np.log(max(mean, 1e-3)), 0.7, 1.0))
    out = np.power(np.clip(img, 0, 1), gamma)
    return np.clip((1 - strength) * img + strength * out, 0, 1)


def dehaze_image(img_bgr, method="baseline", **overrides):
    """对一张 BGR 图去雾，返回结果图和中间量。

    返回值：
      out:  uint8 BGR 图
      info: dict，包含 A、A0、omega、sky_ratio、dark、t_raw、t、keep、soft
    """
    cfg = get_config(method, **overrides)
    img = img_bgr.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dark = dark_channel(img, cfg.patch)
    A0 = atmospheric_light_dark(img, dark)

    keep = np.zeros(gray.shape, dtype=np.uint8)
    soft = np.zeros(gray.shape, dtype=np.float32)
    A = A0

    if cfg.use_sky:
        keep, soft = sky_mask(img)
        if keep.sum() > 0:
            A_sky = atmospheric_light_sky(img, gray, keep)
            # 只有原方法的大气光明显偏亮时，才怀疑它选到了白色物体，
            # 这时候才把天空估计混进来；否则保留原估计，避免偏色。
            if (
                A_sky is not None
                and float(A_sky.min()) >= 0.60
                and float(A0.mean() - A_sky.mean()) > cfg.sky_thr
            ):
                A = cfg.sky_mix * A_sky + (1 - cfg.sky_mix) * A0

    A = np.clip(A, 0.65, 0.98)

    omega = cfg.omega
    if cfg.adaptive_omega:
        rho = float(dark.mean())
        omega = float(np.clip(0.70 + 0.50 * rho, 0.82, cfg.omega_cap))

    t_raw = estimate_transmission(img, A, cfg.patch, omega)
    if cfg.guided:
        t = guided_filter(gray, t_raw, cfg.guided_win, cfg.guided_eps)
    else:
        t = t_raw

    out = recover(img, t, A, cfg.t0)
    if cfg.gamma > 0:
        out = auto_gamma(out, strength=cfg.gamma)

    info = {
        "A": A,
        "A0": A0,
        "omega": omega,
        "sky_ratio": float(keep.mean()),
        "dark": dark,
        "t_raw": t_raw,
        "t": np.maximum(t, cfg.t0),
        "keep": keep,
        "soft": soft,
    }
    return (out * 255).round().astype(np.uint8), info
