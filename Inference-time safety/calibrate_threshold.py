#!/usr/bin/env python3
"""
Calibrate the inference-time guard threshold on the 256-safe + 256-unsafe set.

For each model, this script:

  1. Loads the per-layer directions saved by extract_directions.py.
  2. Runs the 256 safe + 256 unsafe prompts through the model, capturing the
     residual stream activation at the LAST token, at EVERY layer.
  3. Per-layer sweep: projects activations onto r_hat(l), computes AUC-ROC
     -> answers "is the EXP5 best layer also the best guard layer?".
  4. At the EXP5 best layer specifically, fits three thresholds tau by
     operating-point rules (not by max-accuracy):
        tau_fpr01  -- false-refusal rate <= 1%   on the safe pool
        tau_fpr05  -- false-refusal rate <= 5%
        tau_fpr10  -- false-refusal rate <= 10%
     plus tau_youden (Youden's J max) as a comparison point.
  5. Saves per-layer AUCs, threshold table, raw projection scores, and plots.

Outputs (per model):
  directions/<model>/<model>_threshold.json   -- thresholds + score stats
  directions/<model>/<model>_layer_sweep.json -- per-layer AUCs
  plots/<model>_threshold.png                 -- score histograms + ROC
  plots/<model>_layer_auc.png                 -- per-layer AUC line plot

Run on RunPod after extract_directions.py finishes:

    cd /workspace/inference_time_guard
    python calibrate_threshold.py
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

# Reuse the model registry, formatter, and extractors from extract_directions.py
from extract_directions import (
    MODELS,
    ChatFormatter,
    HFHookExtractor,
    PromptData,
    TLensExtractor,
    load_jsonl,
)

FPR_TARGETS = [0.01, 0.05, 0.10]


# ============================================================================
# Threshold fitting
# ============================================================================

def fit_thresholds(safe_scores: np.ndarray, unsafe_scores: np.ndarray) -> Dict:
    """Fit thresholds at multiple FPR operating points + Youden's J."""
    y = np.concatenate([np.zeros_like(safe_scores), np.ones_like(unsafe_scores)])
    s = np.concatenate([safe_scores, unsafe_scores])

    auc = float(roc_auc_score(y, s))

    # Operating-point thresholds: tau = (1 - FPR) quantile of safe scores
    rows = []
    for fpr in FPR_TARGETS:
        tau = float(np.quantile(safe_scores, 1.0 - fpr))
        # Use score > tau (strict) for the predicted-unsafe rule
        empirical_fpr = float((safe_scores > tau).mean())
        tpr = float((unsafe_scores > tau).mean())
        preds = (s > tau).astype(int)
        rows.append({
            "rule": f"FPR<={fpr:.2f}",
            "fpr_target": fpr,
            "tau": tau,
            "empirical_fpr": empirical_fpr,
            "tpr": tpr,
            "fnr": 1 - tpr,
            "accuracy": float((preds == y).mean()),
            "f1": float(f1_score(y, preds, zero_division=0)),
        })

    # Youden's J: argmax_t (TPR - FPR)
    fpr_arr, tpr_arr, thr_arr = roc_curve(y, s)
    j = tpr_arr - fpr_arr
    j_idx = int(np.argmax(j))
    tau_j = float(thr_arr[j_idx])
    preds = (s > tau_j).astype(int)
    rows.append({
        "rule": "Youden_J",
        "fpr_target": None,
        "tau": tau_j,
        "empirical_fpr": float(fpr_arr[j_idx]),
        "tpr": float(tpr_arr[j_idx]),
        "fnr": float(1 - tpr_arr[j_idx]),
        "accuracy": float((preds == y).mean()),
        "f1": float(f1_score(y, preds, zero_division=0)),
    })

    return {
        "auc": auc,
        "thresholds": rows,
        "score_stats": {
            "safe_mean": float(safe_scores.mean()),
            "safe_std": float(safe_scores.std()),
            "unsafe_mean": float(unsafe_scores.mean()),
            "unsafe_std": float(unsafe_scores.std()),
            "mean_gap": float(unsafe_scores.mean() - safe_scores.mean()),
        },
    }


# ============================================================================
# Plots
# ============================================================================

def plot_threshold_panel(model_name: str,
                         best_layer: int,
                         safe_scores: np.ndarray,
                         unsafe_scores: np.ndarray,
                         calib: Dict,
                         out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histograms
    ax = axes[0]
    bins = np.linspace(
        min(safe_scores.min(), unsafe_scores.min()),
        max(safe_scores.max(), unsafe_scores.max()),
        40,
    )
    ax.hist(safe_scores, bins=bins, alpha=0.55, label=f"safe (n={len(safe_scores)})", color="C0")
    ax.hist(unsafe_scores, bins=bins, alpha=0.55, label=f"unsafe (n={len(unsafe_scores)})", color="C3")
    colors = {"FPR<=0.01": "k", "FPR<=0.05": "gray", "FPR<=0.10": "lightgray", "Youden_J": "C2"}
    for row in calib["thresholds"]:
        c = colors.get(row["rule"], "k")
        ax.axvline(row["tau"], color=c, ls="--", lw=1.2,
                   label=f"{row['rule']}  TPR={row['tpr']:.2f}")
    ax.set_xlabel(f"projection score onto r_hat[layer {best_layer}]")
    ax.set_ylabel("count")
    ax.set_title(f"{model_name}  --  calibration set scores  (AUC={calib['auc']:.4f})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Right: ROC
    ax = axes[1]
    y = np.concatenate([np.zeros_like(safe_scores), np.ones_like(unsafe_scores)])
    s = np.concatenate([safe_scores, unsafe_scores])
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, lw=2, color="C0")
    ax.plot([0, 1], [0, 1], lw=0.8, ls=":", color="k")
    for row in calib["thresholds"]:
        ax.scatter(row["empirical_fpr"], row["tpr"], s=40, zorder=5,
                   label=f"{row['rule']} (FPR={row['empirical_fpr']:.2%}, TPR={row['tpr']:.2%})")
    ax.set_xlabel("false positive rate (over-refusal on safe)")
    ax.set_ylabel("true positive rate (catches unsafe)")
    ax.set_title(f"ROC  --  AUC = {calib['auc']:.4f}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_layer_auc(model_name: str,
                   best_layer: int,
                   layer_aucs: Dict[int, float],
                   out_path: Path):
    layers = sorted(layer_aucs.keys())
    aucs = [layer_aucs[l] for l in layers]
    best_at = max(layers, key=lambda l: layer_aucs[l])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(layers, aucs, marker="o", ms=3, color="C0")
    ax.axvline(best_layer, color="r", ls="--", lw=1, label=f"EXP5 best layer = {best_layer}  (AUC={layer_aucs[best_layer]:.4f})")
    ax.axvline(best_at, color="C2", ls=":", lw=1, label=f"calibration best = {best_at}  (AUC={layer_aucs[best_at]:.4f})")
    ax.axhline(0.5, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("AUC on 256+256 calibration set")
    ax.set_title(f"{model_name}  --  per-layer guard quality")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main per-model
# ============================================================================

def calibrate_one_model(spec: dict,
                        safe: List[PromptData],
                        unsafe: List[PromptData],
                        dirs_dir: Path,
                        plots_dir: Path,
                        batch_size: int,
                        device: str) -> Dict:
    name = spec["name"]
    print(f"\n{'='*72}\n{name}  ({spec['hf']}, backend={spec['backend']})\n{'='*72}")
    t0 = time.time()

    # 1. Load saved directions
    pt = dirs_dir / name / f"{name}_all_directions.pt"
    if not pt.exists():
        print(f"  SKIP: {pt} not found  (run extract_directions.py first)")
        return None
    blob = torch.load(pt, map_location="cpu", weights_only=False)
    directions = {int(l): blob["directions"][l].float() for l in blob["directions"]}
    n_layers = int(blob["n_layers"])
    best_layer = int(blob["exp5_best_layer"])
    print(f"  loaded directions: {n_layers} layers, best={best_layer}")

    # 2. Load model + extract activations at every layer
    formatter = ChatFormatter(spec["hf"])
    safe_fmt = [formatter.format_prompt(p.prompt) for p in safe]
    unsafe_fmt = [formatter.format_prompt(p.prompt) for p in unsafe]

    print(f"  loading model ({spec['backend']})...")
    dtype = torch.bfloat16
    if spec["backend"] == "tlens":
        ext = TLensExtractor(spec["hf"], device=device, dtype=dtype)
    elif spec["backend"] == "hf":
        ext = HFHookExtractor(spec["hf"], device=device, dtype=dtype)
    else:
        raise ValueError(spec["backend"])
    print(f"  layers={ext.n_layers}  d_model={ext.d_model}")

    print("  extracting safe activations...")
    safe_acts = ext.extract(safe_fmt, batch_size)
    print("  extracting unsafe activations...")
    unsafe_acts = ext.extract(unsafe_fmt, batch_size)
    ext.cleanup()
    del ext
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Per-layer AUC sweep
    print("  sweeping per-layer AUC...")
    layer_aucs: Dict[int, float] = {}
    for l in range(n_layers):
        r_hat = directions[l]
        s_safe_l = (safe_acts[l] @ r_hat).numpy()
        s_unsafe_l = (unsafe_acts[l] @ r_hat).numpy()
        y = np.concatenate([np.zeros_like(s_safe_l), np.ones_like(s_unsafe_l)])
        s = np.concatenate([s_safe_l, s_unsafe_l])
        layer_aucs[l] = float(roc_auc_score(y, s))
    best_at = max(layer_aucs, key=lambda l: layer_aucs[l])
    print(f"    EXP5 best layer {best_layer}: AUC={layer_aucs[best_layer]:.4f}")
    print(f"    calibration argmax  layer {best_at}: AUC={layer_aucs[best_at]:.4f}")

    # 4. Threshold fitting at EXP5 best layer
    print("  fitting thresholds at EXP5 best layer...")
    r_hat = directions[best_layer]
    s_safe = (safe_acts[best_layer] @ r_hat).numpy()
    s_unsafe = (unsafe_acts[best_layer] @ r_hat).numpy()
    calib = fit_thresholds(s_safe, s_unsafe)

    for row in calib["thresholds"]:
        print(f"    {row['rule']:<12}  tau={row['tau']:+.4f}  "
              f"FPR={row['empirical_fpr']:.3f}  TPR={row['tpr']:.3f}  "
              f"F1={row['f1']:.3f}")

    # 5. Save artifacts
    threshold_json = {
        "model_name": name,
        "hf_name": spec["hf"],
        "best_layer": best_layer,
        "calibration_set": {"n_safe": len(safe), "n_unsafe": len(unsafe)},
        "auc_at_best_layer": calib["auc"],
        "score_stats": calib["score_stats"],
        "thresholds": calib["thresholds"],
        "raw_scores": {
            "safe": s_safe.tolist(),
            "unsafe": s_unsafe.tolist(),
        },
    }
    threshold_path = dirs_dir / name / f"{name}_threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_json, f, indent=2, ensure_ascii=False)

    sweep_json = {
        "model_name": name,
        "n_layers": n_layers,
        "exp5_best_layer": best_layer,
        "calibration_argmax_layer": best_at,
        "layer_aucs": {str(l): layer_aucs[l] for l in sorted(layer_aucs)},
    }
    sweep_path = dirs_dir / name / f"{name}_layer_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_json, f, indent=2, ensure_ascii=False)

    plot_threshold_panel(name, best_layer, s_safe, s_unsafe, calib,
                         plots_dir / f"{name}_threshold.png")
    plot_layer_auc(name, best_layer, layer_aucs,
                   plots_dir / f"{name}_layer_auc.png")

    elapsed = round(time.time() - t0, 1)
    print(f"  saved threshold + sweep + plots  (elapsed {elapsed}s)")

    return {
        "model_name": name,
        "auc_at_best_layer": calib["auc"],
        "calibration_argmax_layer": best_at,
        "calibration_argmax_auc": layer_aucs[best_at],
        "best_layer_auc": layer_aucs[best_layer],
        "thresholds": {row["rule"]: row for row in calib["thresholds"]},
        "elapsed_seconds": elapsed,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--dirs-dir", type=str, default=None,
                    help="root containing extract_directions.py output")
    ap.add_argument("--plots-dir", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else here / "data"
    dirs_dir = Path(args.dirs_dir) if args.dirs_dir else here / "directions"
    plots_dir = Path(args.plots_dir) if args.plots_dir else here / "plots"

    safe = load_jsonl(data_dir / "threshold_safe.jsonl")
    unsafe = load_jsonl(data_dir / "threshold_unsafe.jsonl")
    print(f"loaded calibration set: {len(safe)} safe + {len(unsafe)} unsafe from {data_dir}")
    print(f"directions dir: {dirs_dir}")
    print(f"plots dir:      {plots_dir}")
    print(f"device: {args.device}  batch_size: {args.batch_size}")

    selected = MODELS if not args.models else [m for m in MODELS if m["name"] in args.models]

    summaries = []
    for spec in selected:
        if args.skip_existing:
            tp = dirs_dir / spec["name"] / f"{spec['name']}_threshold.json"
            if tp.exists():
                print(f"\nSKIP {spec['name']} (already calibrated at {tp})")
                continue
        try:
            s = calibrate_one_model(spec, safe, unsafe, dirs_dir, plots_dir,
                                    args.batch_size, args.device)
            if s:
                summaries.append(s)
        except Exception as e:
            import traceback
            print(f"\nERROR {spec['name']}: {e}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    summary_path = dirs_dir / "calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")

    # Print compact table
    print("\n" + "=" * 100)
    print(f"{'Model':<28} {'AUC':>7} {'argmax':>7} {'argmaxAUC':>10}  "
          f"{'TPR@1%':>7} {'TPR@5%':>7} {'TPR@10%':>8}")
    print("-" * 100)
    for s in summaries:
        ths = s["thresholds"]
        print(f"{s['model_name']:<28} "
              f"{s['auc_at_best_layer']:>7.4f} "
              f"{s['calibration_argmax_layer']:>7d} "
              f"{s['calibration_argmax_auc']:>10.4f}  "
              f"{ths['FPR<=0.01']['tpr']:>7.3f} "
              f"{ths['FPR<=0.05']['tpr']:>7.3f} "
              f"{ths['FPR<=0.10']['tpr']:>8.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
