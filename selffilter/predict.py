"""Apply filter / removal to one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from selffilter.filters.synth import apply_filter, remove_with_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--placement", default="eyes")
    ap.add_argument("--coverage", default="high")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0)
    fx, m, inten = apply_filter(x, placement=args.placement, coverage=args.coverage)
    rx = remove_with_mask(fx, m)
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # denorm for save
    def den(t):
        return (t * 0.5 + 0.5).clamp(0, 1)

    save_image(den(fx)[0], out_dir / "filtered.png")
    save_image(den(rx)[0], out_dir / "removed.png")
    meta = {"coverage_intensity": inten, "placement": args.placement, "coverage": args.coverage}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out_dir} inten={inten:.3f}")


if __name__ == "__main__":
    main()
