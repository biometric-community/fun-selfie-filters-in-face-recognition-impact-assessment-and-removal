"""Synthetic fun-selfie filters with the paper's coverage + placement taxonomy."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PLACEMENTS: Dict[str, Tuple[float, float, float, float]] = {
    "eyes": (0.22, 0.48, 0.12, 0.88),
    "nose": (0.40, 0.68, 0.32, 0.68),
    "mouth": (0.62, 0.88, 0.28, 0.72),
    "total": (0.10, 0.92, 0.10, 0.90),
}

# Facial region of an aligned crop, used as the denominator of Eq. 1.
FACE_REGION = PLACEMENTS["total"]

# Target coverage intensity per group, inside the Table IV bands
# (low < 15%, medium 15-40%, high > 40%).
COVERAGE_LEVELS = {
    "low": 0.08,
    "medium": 0.27,
    "high": 0.45,
}


def _face_mask(h: int, w: int, device, dtype) -> torch.Tensor:
    y0f, y1f, x0f, x1f = FACE_REGION
    mask = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
    mask[:, :, int(y0f * h) : int(y1f * h), int(x0f * w) : int(x1f * w)] = 1.0
    return mask


def coverage_intensity(original: torch.Tensor, filtered: torch.Tensor) -> float:
    """Eq. 1: average pixel intensity change per facial pixel, as a fraction of the range.

    Images are in [-1, 1], so a change of 2 is the full intensity range.
    """
    _, _, h, w = original.shape
    face = _face_mask(h, w, original.device, original.dtype)[:, 0] > 0
    change = (filtered - original).abs().mean(dim=1) / 2.0  # (B,H,W)
    return float(change[:, face[0]].mean().item())


def _box(h: int, w: int, placement: str, scale: float) -> Tuple[int, int, int, int]:
    """Box around the placement centre, scaled by `scale` and clipped to the face region."""
    y0f, y1f, x0f, x1f = PLACEMENTS[placement]
    cy, cx = (y0f + y1f) / 2, (x0f + x1f) / 2
    hy, hx = scale * (y1f - y0f) / 2, scale * (x1f - x0f) / 2
    fy0, fy1, fx0, fx1 = FACE_REGION
    y0, y1 = max(fy0, cy - hy), min(fy1, cy + hy)
    x0, x1 = max(fx0, cx - hx), min(fx1, cx + hx)
    return int(y0 * h), max(int(y1 * h), int(y0 * h) + 1), int(x0 * w), max(int(x1 * w), int(x0 * w) + 1)


def apply_filter(
    images: torch.Tensor,
    placement: str = "eyes",
    coverage: str = "medium",
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Overlay an opaque sticker centred on `placement`, grown until its Eq. 1 coverage
    intensity reaches the target for `coverage`. Returns (filtered, mask, intensity).
    images: (B,3,H,W) in ~[-1,1].
    """
    b, _, h, w = images.shape
    target = COVERAGE_LEVELS[coverage]
    face = _face_mask(h, w, images.device, images.dtype)
    face_mean = (images * face).sum(dim=(1, 2, 3)) / (3 * face.sum())
    # Solid sticker colour that contrasts with each face
    fill = torch.where(face_mean > 0, -1.0, 1.0).to(images.dtype).view(b, 1, 1, 1)

    def overlay(scale: float):
        y0, y1, x0, x1 = _box(h, w, placement, scale)
        mask = torch.zeros(b, 1, h, w, device=images.device, dtype=images.dtype)
        mask[:, :, y0:y1, x0:x1] = 1.0
        out = images * (1 - mask) + fill * mask
        return out, mask, coverage_intensity(images, out)

    # Large enough for a sticker grown from any placement to cover the whole face. The
    # intensity grows with the sticker, so bisect for the smallest one reaching the target.
    scales = np.linspace(0.05, 8.0, 400)
    lo, hi = 0, len(scales) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if overlay(float(scales[mid]))[2] >= target:
            hi = mid
        else:
            lo = mid + 1
    return overlay(float(scales[lo]))


def remove_with_mask(images: torch.Tensor, mask: torch.Tensor, iterations: int = 300) -> torch.Tensor:
    """Fill the filtered region from its surroundings (diffusion inpainting).

    The paper uses a segmentation network and a GAN; this training-free surrogate
    reconstructs the masked pixels from the visible face around them.
    """
    hole = mask.expand_as(images) > 0
    known = (~hole).to(images.dtype)
    start = (images * known).sum(dim=(2, 3), keepdim=True) / known.sum(dim=(2, 3), keepdim=True).clamp_min(1)
    out = torch.where(hole, start.expand_as(images), images)
    for _ in range(iterations):
        smooth = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        out = torch.where(hole, smooth, images)
    return out
