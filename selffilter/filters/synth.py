"""Synthetic fun-selfie filters with coverage + placement taxonomy."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

PLACEMENTS: Dict[str, Tuple[float, float, float, float]] = {
    "eyes": (0.22, 0.48, 0.12, 0.88),
    "nose": (0.40, 0.68, 0.32, 0.68),
    "mouth": (0.62, 0.88, 0.28, 0.72),
    "total": (0.10, 0.92, 0.10, 0.90),
}

COVERAGE_LEVELS = {
    "low": 0.10,
    "medium": 0.28,
    "high": 0.50,
}


def _box(h: int, w: int, placement: str):
    y0f, y1f, x0f, x1f = PLACEMENTS[placement]
    return int(y0f * h), int(y1f * h), int(x0f * w), int(x1f * w)


def apply_filter(
    images: torch.Tensor,
    placement: str = "eyes",
    coverage: str = "medium",
    color: Tuple[float, float, float] = (1.0, 0.2, 0.6),
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Overlay a tinted sticker pattern. Returns (filtered, mask, coverage_intensity).
    images: (B,3,H,W) in ~[-1,1].
    """
    b, _, h, w = images.shape
    y0, y1, x0, x1 = _box(h, w, placement)
    region_h, region_w = max(y1 - y0, 1), max(x1 - x0, 1)
    mask = torch.zeros(b, 1, h, w, device=images.device, dtype=images.dtype)
    mask[:, :, y0:y1, x0:x1] = 1.0

    target = COVERAGE_LEVELS[coverage]
    cur = float(mask.mean().item())
    if cur > target + 1e-6:
        keep = (target / cur) ** 0.5
        nh, nw = max(1, int(region_h * keep)), max(1, int(region_w * keep))
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        y0, x0 = max(0, cy - nh // 2), max(0, cx - nw // 2)
        y1, x1 = min(h, y0 + nh), min(w, x0 + nw)
        mask.zero_()
        mask[:, :, y0:y1, x0:x1] = 1.0

    tint = (torch.tensor(color, device=images.device, dtype=images.dtype).view(1, 3, 1, 1) * 2.0 - 1.0)
    yy = torch.arange(y1 - y0, device=images.device).view(-1, 1)
    xx = torch.arange(x1 - x0, device=images.device).view(1, -1)
    dots = (((yy // 4) + (xx // 4)) % 2).float().view(1, 1, y1 - y0, x1 - x0)
    patch = tint * (0.55 + 0.45 * dots)
    patch = patch.expand(b, 3, y1 - y0, x1 - x0)

    out = images.clone()
    alpha = 0.85
    out[:, :, y0:y1, x0:x1] = (1 - alpha) * images[:, :, y0:y1, x0:x1] + alpha * patch
    active = mask.expand_as(out)
    out = images * (1 - active) + out * active
    intensity = float(((out - images).abs().mean(dim=1, keepdim=True) > 1e-3).float().mean().item())
    return out, mask, intensity


def remove_with_mask(images: torch.Tensor, mask: torch.Tensor, ksize: int = 7) -> torch.Tensor:
    """Mask-guided blur inpainting surrogate (paper uses GAN)."""
    pad = ksize // 2
    blurred = F.avg_pool2d(images, kernel_size=ksize, stride=1, padding=pad)
    m = mask.expand_as(images)
    return images * (1 - m) + blurred * m
