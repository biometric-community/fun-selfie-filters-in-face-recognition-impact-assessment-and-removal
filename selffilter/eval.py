"""LFW impact by coverage/placement + mask-guided removal recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from selffilter.data.pairs import build_calfw, build_cplfw, build_lfw
from selffilter.filters.synth import PLACEMENTS, apply_filter, remove_with_mask
from selffilter.models.fr_net import FaceEmbedder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer = 0.0, 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer


@torch.no_grad()
def score_pairs(model, loader, device, transform_fn=None) -> Tuple[np.ndarray, np.ndarray, float]:
    scores, labels = [], []
    intensities = []
    model.eval()
    for a, b, y in loader:
        a, b = a.to(device), b.to(device)
        if transform_fn is not None:
            a, ma, ia = transform_fn(a)
            b, mb, ib = transform_fn(b)
            intensities.append(0.5 * (ia + ib))
        ea = model.embed(a)
        eb = model.embed(b)
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    mean_i = float(np.mean(intensities)) if intensities else 0.0
    return s, y, mean_i


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = FaceEmbedder(
        int(blob["num_classes"]), emb_dim=int(blob.get("emb_dim", 128)), pretrained=False
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    image_size = int(cfg["model"].get("image_size", 112))
    bs = int(cfg["eval"].get("batch_size", 128))
    nw = int(cfg["train"].get("num_workers", 4))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None

    results: Dict = {"verification": {}, "filter_impact": {}, "removal": {}}
    for name, builder in (("lfw", build_lfw), ("calfw", build_calfw), ("cplfw", build_cplfw)):
        try:
            ds = builder(cfg, image_size, max_p)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
        s, y, _ = score_pairs(model, loader, device)
        acc, eer = best_acc_eer(s, y)
        results["verification"][name] = {"accuracy": acc, "eer": eer, "n_pairs": int(len(y))}
        print(f"[{name}] clean acc={acc:.4f} eer={eer:.4f}")

    # Impact on LFW
    lfw = build_lfw(cfg, image_size, max_p)
    loader = DataLoader(lfw, batch_size=min(bs, 64), shuffle=False, num_workers=nw)
    coverages = list(cfg["eval"].get("coverages", ["low", "medium", "high"]))
    placements = list(cfg["eval"].get("placements", ["eyes", "nose", "mouth", "total"]))

    for cov in coverages:
        for place in placements:
            def _tf(x, _c=cov, _p=place):
                return apply_filter(x, placement=_p, coverage=_c)

            s, y, inten = score_pairs(model, loader, device, transform_fn=_tf)
            acc, eer = best_acc_eer(s, y)
            key = f"{cov}/{place}"
            results["filter_impact"][key] = {
                "accuracy": acc,
                "eer": eer,
                "coverage_intensity": inten,
                "n_pairs": int(len(y)),
            }
            print(f"[impact {key}] acc={acc:.4f} eer={eer:.4f} inten={inten:.3f}")

    # Removal recovery: high/total and high/eyes
    for place in ("eyes", "total"):
        def _filt(x, _p=place):
            fx, m, i = apply_filter(x, placement=_p, coverage="high")
            return remove_with_mask(fx, m), m, i

        s, y, inten = score_pairs(model, loader, device, transform_fn=_filt)
        acc, eer = best_acc_eer(s, y)
        key = f"high/{place}"
        base = results["filter_impact"].get(key, {})
        results["removal"][key] = {
            "accuracy": acc,
            "eer": eer,
            "coverage_intensity": inten,
            "acc_before": base.get("accuracy"),
            "eer_before": base.get("eer"),
            "acc_delta": float(acc - base["accuracy"]) if base else None,
        }
        print(f"[removal {key}] acc={acc:.4f} (before={base.get('accuracy')})")

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
