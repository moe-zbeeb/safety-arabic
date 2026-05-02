#!/usr/bin/env python3
"""
Visualize per-layer safety directions saved by extract_directions.py.

For each model with a saved <name>_all_directions.pt, this writes
plots/<name>_diagnostics.png with three panels:

    (a) Cosine similarity heatmap, layer x layer.
        Bright square in the middle  ==  the direction is stable across
        a span of layers.  Hot diagonal only  ==  direction rotates a lot.

    (b) Cosine similarity to the EXP5 best-layer direction.
        A single line; the chosen guard layer is marked.

    (c) Cosine similarity between consecutive layers.
        Where this dips is where the direction is changing fastest.

Also writes plots/cross_model_cosine_to_best.png  --  one curve per model
on a normalized depth axis (layer / n_layers), so the 5 architectures can
be compared side-by-side.

We can only plot what's stored in the .pt files (the per-layer normalized
directions r_hat).  We do *not* have the raw activations, so projection-
score and AUC-per-layer plots aren't possible from this script alone.

Usage:
    python plot_directions.py
    python plot_directions.py --models Qwen2.5-3B-Instruct
    python plot_directions.py --dirs-dir /path/to/directions --out plots/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_directions(pt_path: Path):
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    raw = blob["directions"]
    # Stack into a matrix R (n_layers, d_model)
    layers = sorted(int(k) for k in raw.keys())
    R = torch.stack([raw[l].float() for l in layers], dim=0)
    return {
        "model_name": blob["model_name"],
        "n_layers": blob["n_layers"],
        "d_model": blob["d_model"],
        "exp5_best_layer": blob.get("exp5_best_layer"),
        "R": R,  # (n_layers, d_model), each row already unit-normalized
    }


def cosine_matrix(R: torch.Tensor) -> np.ndarray:
    R = R / (R.norm(dim=1, keepdim=True) + 1e-12)
    return (R @ R.T).numpy()


def plot_one_model(info: dict, out_path: Path):
    R = info["R"]
    n = info["n_layers"]
    best = info["exp5_best_layer"]
    name = info["model_name"]

    C = cosine_matrix(R)
    cos_to_best = C[best] if best is not None else None
    cos_consec = np.array([C[l, l + 1] for l in range(n - 1)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) heatmap
    ax = axes[0]
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.set_title(f"{name}\ncos(r_hat[i], r_hat[j])  --  {n} layers")
    ax.set_xlabel("layer j")
    ax.set_ylabel("layer i")
    if best is not None:
        ax.axhline(best, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(best, color="k", lw=0.6, ls="--", alpha=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) cosine to EXP5 best layer
    ax = axes[1]
    if cos_to_best is not None:
        ax.plot(np.arange(n), cos_to_best, marker="o", ms=3)
        ax.axvline(best, color="r", ls="--", lw=1, label=f"EXP5 best layer = {best}")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(1, color="k", lw=0.3, ls=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"cos(r_hat[layer], r_hat[{best}])")
        ax.set_xlabel("layer")
        ax.set_ylabel("cosine similarity")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no exp5_best_layer in metadata",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # (c) consecutive cosine
    ax = axes[2]
    ax.plot(np.arange(n - 1) + 0.5, cos_consec, marker="o", ms=3, color="C2")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="k", lw=0.3, ls=":")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("cos(r_hat[l], r_hat[l+1])  --  rate of rotation")
    ax.set_xlabel("layer boundary")
    ax.set_ylabel("cosine similarity")
    if best is not None:
        ax.axvline(best, color="r", ls="--", lw=1, alpha=0.6)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_cross_model(infos: List[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for info in infos:
        if info["exp5_best_layer"] is None:
            continue
        R = info["R"]
        n = info["n_layers"]
        best = info["exp5_best_layer"]
        C = cosine_matrix(R)
        cos_to_best = C[best]
        depth = np.arange(n) / max(n - 1, 1)
        ax.plot(depth, cos_to_best, marker="o", ms=3,
                label=f"{info['model_name']} (best layer {best}/{n})")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="k", lw=0.3, ls=":")
    ax.set_xlabel("normalized depth (layer / n_layers)")
    ax.set_ylabel("cos(r_hat[layer], r_hat[best])")
    ax.set_title("Direction alignment with the EXP5 best layer, across models")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs-dir", type=str, default=None,
                    help="root holding <model>/<model>_all_directions.pt; "
                         "default: ./directions next to this script")
    ap.add_argument("--out", type=str, default=None,
                    help="output dir; default: ./plots next to this script")
    ap.add_argument("--models", nargs="+", default=None,
                    help="optional subset of model display names")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    dirs_dir = Path(args.dirs_dir) if args.dirs_dir else here / "directions"
    out_dir = Path(args.out) if args.out else here / "plots"

    if not dirs_dir.exists():
        raise SystemExit(f"directions dir not found: {dirs_dir}")

    pts = sorted(dirs_dir.glob("*/*_all_directions.pt"))
    if args.models:
        pts = [p for p in pts if any(m in p.name for m in args.models)]
    if not pts:
        raise SystemExit(f"no *_all_directions.pt files under {dirs_dir}")

    infos = []
    for pt in pts:
        print(f"loading {pt.relative_to(dirs_dir)}")
        info = load_directions(pt)
        infos.append(info)
        plot_one_model(info, out_dir / f"{info['model_name']}_diagnostics.png")

    if len(infos) > 1:
        plot_cross_model(infos, out_dir / "cross_model_cosine_to_best.png")

    print(f"\nDone. {len(infos)} model(s) plotted into {out_dir}")


if __name__ == "__main__":
    main()
