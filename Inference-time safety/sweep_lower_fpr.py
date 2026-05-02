#!/usr/bin/env python3
"""
Sweep additional (mostly stricter) FPR targets and re-evaluate on AraSafe.

This is post-processing only -- no model reload required:
  - Calibration safe/unsafe scores come from
        directions/<model>/<model>_threshold.json   (raw_scores)
  - AraSafe per-prompt scores come from
        eval/<model>_arasafe_scores.jsonl

For each model and each FPR target, we:
  1. Fit tau on the calibration safe scores: tau = quantile(safe, 1 - FPR).
  2. Apply tau to the AraSafe scores -> empirical FPR, TPR, F1, AUC.
  3. Write out tables (one CSV per rule + a combined long CSV).
  4. Plot the AraSafe ROC curve and overlay every fit-on-calibration
     operating point so we can see the Pareto frontier vs where our
     thresholds actually land.

Run locally on a Mac, no GPU needed.

    python3 sweep_lower_fpr.py
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
DIRS_DIR = HERE / "directions"
EVAL_DIR = HERE / "eval"
TABLES_DIR = HERE / "tables"
PLOTS_DIR = HERE / "plots"

# Lower FPR targets (to add) plus the existing ones for a single, clean sweep.
# Order matters for plotting (legend order tracks this).
FPR_TARGETS = [0.001, 0.002, 0.005, 0.01, 0.05, 0.10]

# Pretty label / filename slug per FPR target
def label(fpr: float) -> str:
    pct = fpr * 100
    if pct == int(pct):
        return f"{int(pct)}pct"
    if pct >= 1:
        return f"{pct:.0f}pct"
    return f"{pct:.1f}pct".rstrip("0").rstrip(".")

def slug(fpr: float) -> str:
    return f"over_refusal_{label(fpr)}"


# ============================================================================
# IO
# ============================================================================

def load_calibration_scores(model: str) -> Tuple[np.ndarray, np.ndarray]:
    th_path = DIRS_DIR / model / f"{model}_threshold.json"
    with open(th_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    safe = np.array(blob["raw_scores"]["safe"])
    unsafe = np.array(blob["raw_scores"]["unsafe"])
    return safe, unsafe


def load_arasafe_scores(model: str) -> Tuple[np.ndarray, np.ndarray]:
    path = EVAL_DIR / f"{model}_arasafe_scores.jsonl"
    safe, unsafe = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            (safe if r["label"] == "normal" else unsafe).append(r["score"])
    return np.array(safe), np.array(unsafe)


def list_models() -> List[str]:
    out = []
    for p in sorted(EVAL_DIR.glob("*_arasafe_scores.jsonl")):
        m = p.name.replace("_arasafe_scores.jsonl", "")
        if (DIRS_DIR / m / f"{m}_threshold.json").exists():
            out.append(m)
    return out


# ============================================================================
# Metrics
# ============================================================================

def evaluate(safe_arasafe: np.ndarray,
             unsafe_arasafe: np.ndarray,
             tau: float) -> Dict:
    fpr = float((safe_arasafe > tau).mean())
    tpr = float((unsafe_arasafe > tau).mean())
    s = np.concatenate([safe_arasafe, unsafe_arasafe])
    y = np.concatenate([np.zeros_like(safe_arasafe), np.ones_like(unsafe_arasafe)])
    preds = (s > tau).astype(int)
    return {
        "tau": float(tau),
        "fpr": fpr,
        "tpr": tpr,
        "f1": float(f1_score(y, preds, zero_division=0)),
        "accuracy": float((preds == y).mean()),
        "n_refused_safe": int((safe_arasafe > tau).sum()),
        "n_caught_unsafe": int((unsafe_arasafe > tau).sum()),
        "missed_unsafe": int((unsafe_arasafe <= tau).sum()),
    }


def fit_tau(safe_calib: np.ndarray, fpr: float) -> float:
    return float(np.quantile(safe_calib, 1.0 - fpr))


# ============================================================================
# Plot
# ============================================================================

def plot_roc(model: str,
             safe_arasafe: np.ndarray,
             unsafe_arasafe: np.ndarray,
             ops: List[Dict],
             out_path: Path):
    s = np.concatenate([safe_arasafe, unsafe_arasafe])
    y = np.concatenate([np.zeros_like(safe_arasafe), np.ones_like(unsafe_arasafe)])
    fpr_curve, tpr_curve, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: full ROC
    ax = axes[0]
    ax.plot(fpr_curve, tpr_curve, lw=2, color="C0")
    ax.plot([0, 1], [0, 1], lw=0.7, ls=":", color="k")
    cmap = plt.get_cmap("viridis")
    for i, op in enumerate(ops):
        c = cmap(i / max(len(ops) - 1, 1))
        ax.scatter(op["fpr"], op["tpr"], s=60, color=c, zorder=5,
                   edgecolors="k", linewidths=0.6,
                   label=f"calib FPR={op['fpr_target']*100:g}%  →  ({op['fpr']:.3f}, {op['tpr']:.3f})")
    ax.set_xlabel("AraSafe FPR  (over-refusal of safe)")
    ax.set_ylabel("AraSafe TPR  (catch on unsafe)")
    ax.set_title(f"{model}  --  AraSafe ROC  (AUC={auc:.4f})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # Right panel: zoomed top-left (operating-point region)
    ax = axes[1]
    ax.plot(fpr_curve, tpr_curve, lw=2, color="C0")
    for i, op in enumerate(ops):
        c = cmap(i / max(len(ops) - 1, 1))
        ax.scatter(op["fpr"], op["tpr"], s=70, color=c, zorder=5,
                   edgecolors="k", linewidths=0.6,
                   label=f"calib FPR={op['fpr_target']*100:g}%")
        ax.annotate(f"{op['tpr']:.2f}", (op["fpr"], op["tpr"]),
                    textcoords="offset points", xytext=(6, -2), fontsize=8)
    ax.set_xlabel("AraSafe FPR")
    ax.set_ylabel("AraSafe TPR")
    ax.set_title("Zoomed: operating points from calibration thresholds")
    xs = [op["fpr"] for op in ops]
    ys = [op["tpr"] for op in ops]
    pad = 0.02
    ax.set_xlim(max(0, min(xs) - pad), min(1, max(xs) + 0.05))
    ax.set_ylim(max(0, min(ys) - 0.05), min(1, max(ys) + pad))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

COLUMNS = [
    "model_name", "rule", "fpr_target", "tau",
    "n_safe", "n_unsafe",
    "safe_refusal_rate", "unsafe_refusal_rate",
    "n_refused_safe", "n_caught_unsafe", "missed_unsafe",
    "accuracy", "f1", "auc_arasafe",
]


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    models = list_models()
    if not models:
        raise SystemExit("no models found with both threshold.json and arasafe_scores.jsonl")
    print(f"models: {models}\n")

    by_rule: Dict[str, List[Dict]] = {slug(f): [] for f in FPR_TARGETS}
    all_rows: List[Dict] = []

    for m in models:
        safe_calib, _ = load_calibration_scores(m)
        safe_ev, unsafe_ev = load_arasafe_scores(m)

        s = np.concatenate([safe_ev, unsafe_ev])
        y = np.concatenate([np.zeros_like(safe_ev), np.ones_like(unsafe_ev)])
        auc = float(roc_auc_score(y, s))

        ops_for_plot: List[Dict] = []
        print(f"=== {m}  (n_safe={len(safe_ev)}, n_unsafe={len(unsafe_ev)}, AraSafe AUC={auc:.4f}) ===")
        print(f"{'rule':<22} {'tau':>10} {'AraSafe FPR':>13} {'AraSafe TPR':>13} {'F1':>7}")
        print("-" * 70)

        for fpr_target in FPR_TARGETS:
            tau = fit_tau(safe_calib, fpr_target)
            ev = evaluate(safe_ev, unsafe_ev, tau)
            row = {
                "model_name":          m,
                "rule":                slug(fpr_target),
                "fpr_target":          fpr_target,
                "tau":                 round(tau, 6),
                "n_safe":              int(len(safe_ev)),
                "n_unsafe":            int(len(unsafe_ev)),
                "safe_refusal_rate":   round(ev["fpr"], 6),
                "unsafe_refusal_rate": round(ev["tpr"], 6),
                "n_refused_safe":      ev["n_refused_safe"],
                "n_caught_unsafe":     ev["n_caught_unsafe"],
                "missed_unsafe":       ev["missed_unsafe"],
                "accuracy":            round(ev["accuracy"], 6),
                "f1":                  round(ev["f1"], 6),
                "auc_arasafe":         round(auc, 6),
            }
            by_rule[slug(fpr_target)].append(row)
            all_rows.append(row)
            ops_for_plot.append({
                "fpr_target": fpr_target,
                "fpr": ev["fpr"],
                "tpr": ev["tpr"],
                "tau": tau,
            })
            print(f"{slug(fpr_target):<22} {tau:>10.4f} "
                  f"{ev['fpr']:>13.4f} {ev['tpr']:>13.4f} {ev['f1']:>7.3f}")

        plot_roc(m, safe_ev, unsafe_ev, ops_for_plot,
                 PLOTS_DIR / f"{m}_arasafe_roc.png")
        print()

    # Per-rule CSVs
    for rule_slug, rows in by_rule.items():
        rows.sort(key=lambda r: r["model_name"])
        out = TABLES_DIR / f"{rule_slug}.csv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}  ({len(rows)} rows)")

    # Combined long-format CSV
    all_rows.sort(key=lambda r: (r["fpr_target"], r["model_name"]))
    combined = TABLES_DIR / "over_refusal_sweep_all.csv"
    with open(combined, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {combined}  ({len(all_rows)} rows)")
    print(f"plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
