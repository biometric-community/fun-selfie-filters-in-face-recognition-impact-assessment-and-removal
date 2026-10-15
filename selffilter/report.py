"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from selffilter.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("FR backbone (CelebA)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_impact(impact: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # group by coverage: mean EER across placements
    from collections import defaultdict

    by_cov = defaultdict(list)
    for k, v in impact.items():
        cov = k.split("/")[0]
        by_cov[cov].append(v["eer"] * 100)
    keys = [c for c in ("low", "medium", "high") if c in by_cov]
    vals = [float(np.mean(by_cov[c])) for c in keys]
    x = np.arange(len(keys))
    ax.bar(x, vals, color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Mean EER (%)")
    ax.set_title("Filter impact by coverage (LFW)")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_removal(removal: dict, impact: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    keys = list(removal.keys())
    before = [impact.get(k, {}).get("eer", 0) * 100 for k in keys]
    after = [removal[k]["eer"] * 100 for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w / 2, before, w, label="filtered", color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.bar(x + w / 2, after, w, label="removed", color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("EER (%)")
    ax.set_title("Removal recovery (LFW)")
    ax.legend()
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    impact = eval_res.get("filter_impact", {})
    removal = eval_res.get("removal", {})
    lines = [
        "# REPORT — Fun Selfie Filters in Face Recognition",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin: synthetic selfie filters (coverage × placement) on LFW,",
        "impact on verification, and mask-guided removal recovery.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Clean verification",
        "",
        "| Dataset | Accuracy | EER | N |",
        "|---------|----------|-----|---|",
    ]
    for k, v in sorted(eval_res.get("verification", {}).items()):
        lines.append(f"| {k} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {v['n_pairs']} |")
    lines += [
        "",
        "## Filter impact (LFW)",
        "",
        "| Condition | Accuracy | EER | Intensity |",
        "|-----------|----------|-----|-----------|",
    ]
    for k in sorted(impact.keys()):
        v = impact[k]
        lines.append(
            f"| {k} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {v['coverage_intensity']:.3f} |"
        )
    lines += [
        "",
        f"![Impact]({fig_dir.as_posix()}/filter_impact.svg)",
        "",
        "## Removal",
        "",
        "| Condition | EER before | EER after | Acc Δ |",
        "|-----------|------------|-----------|-------|",
    ]
    for k, v in sorted(removal.items()):
        lines.append(
            f"| {k} | {v.get('eer_before', float('nan'))*100:.2f}% | {v['eer']*100:.2f}% | "
            f"{(v.get('acc_delta') or 0)*100:.2f}pp |"
        )
    lines += [
        "",
        f"![Removal]({fig_dir.as_posix()}/removal.svg)",
        "",
        "## Notes",
        "",
        "- Paper uses FRGC/FERET app filters + GAN removal; we use CelebA/LFW + synthetic overlays.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["history"], fig_dir / "train_loss.svg")
    plot_impact(eval_res.get("filter_impact", {}), fig_dir / "filter_impact.svg")
    plot_removal(eval_res.get("removal", {}), eval_res.get("filter_impact", {}), fig_dir / "removal.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
