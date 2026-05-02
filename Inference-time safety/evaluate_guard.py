#!/usr/bin/env python3
"""
Evaluate the inference-time guard on AraSafe Real (~12K human-labeled prompts).

For each model, this script:
  1. Loads the saved per-layer directions and the calibrated thresholds.
  2. Runs every AraSafe prompt through the model UP TO the best layer only
     (early-exit; no generation).
  3. Reads the residual at that layer at the last token, projects onto r_hat,
     and produces a single scalar score per prompt.
  4. Compares each score to each saved tau (FPR<=0.01, 0.05, 0.10, Youden_J)
     and records refuse / continue decisions.
  5. Aggregates per-class metrics:
        - on AraSafe safe   prompts -> empirical FPR (deployment over-refusal)
        - on AraSafe unsafe prompts -> empirical TPR (deployment catch rate)
     Plus AraSafe AUC (threshold-free), F1 at each tau, and score stats.
  6. Saves per-prompt scores and decisions for downstream analysis.
  7. Plots:
        - AraSafe score histograms with calibration thresholds drawn
        - AraSafe vs calibration distributions overlaid (transfer check)

Outputs (per model):
  eval/<model>_arasafe_scores.jsonl    -- one row per prompt
  eval/<model>_arasafe_metrics.json    -- aggregated metrics at each tau
  plots/<model>_arasafe.png            -- score histograms + transfer overlay

Cross-model:
  eval/arasafe_summary.json            -- one row per model

RunPod:
    export HF_HOME=/workspace/hf_cache
    export HF_TOKEN=hf_xxx
    cd /workspace/inference_time_guard
    # Upload arasafe_human.jsonl into data/  (or pass --arasafe-path)
    python evaluate_guard.py
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from extract_directions import (
    MODELS,
    ChatFormatter,
    PromptData,
)

# AraSafe label literal -> we treat anything else as unsafe
ARASAFE_SAFE_LABEL = "آمن/غير ضار"

# Field names: human-labeled file uses "User_Prompt"; synthetic uses "Prompt".
# Try both for robustness.
PROMPT_FIELDS = ["User_Prompt", "Prompt", "prompt"]
LABEL_FIELDS = ["Safety_Label", "label"]


# ============================================================================
# AraSafe loading
# ============================================================================

def load_arasafe(path: Path, limit: Optional[int] = None) -> List[PromptData]:
    out: List[PromptData] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompt = next((r[k] for k in PROMPT_FIELDS if k in r), None)
            label_raw = next((r[k] for k in LABEL_FIELDS if k in r), None)
            if prompt is None or label_raw is None:
                raise ValueError(f"Unexpected schema; first record keys = {list(r.keys())}")
            label = "normal" if label_raw == ARASAFE_SAFE_LABEL else "unsafe"
            out.append(PromptData(prompt=prompt.strip(), label=label))
    if limit:
        # balanced subsample
        safe = [d for d in out if d.label == "normal"][: limit // 2]
        unsafe = [d for d in out if d.label == "unsafe"][: limit // 2]
        out = safe + unsafe
    return out


# ============================================================================
# Per-prompt scoring with early-exit
# ============================================================================

class TLensScorer:
    """Scorer using TransformerLens with stop_at_layer for true early-exit."""

    def __init__(self, hf_name: str, device: str, dtype, layer: int):
        from transformer_lens import HookedTransformer
        self.model = HookedTransformer.from_pretrained_no_processing(
            hf_name, device=device, dtype=dtype, trust_remote_code=True
        )
        self.model.eval()
        self.layer = layer
        self.hook_name = f"blocks.{layer}.hook_resid_post"

    def score(self, formatted: str, r_hat: torch.Tensor) -> float:
        tokens = self.model.to_tokens(formatted, prepend_bos=False)
        captured = {}

        # TransformerLens calls hooks as fn(activation, hook=HookPoint),
        # i.e. the second arg is passed as kwarg `hook`. Param name must match.
        def capture(activation, hook):
            captured["x"] = activation[0, -1].float().cpu()
            return activation

        with torch.no_grad():
            self.model.run_with_hooks(
                tokens,
                fwd_hooks=[(self.hook_name, capture)],
                stop_at_layer=self.layer + 1,
            )
        return float(captured["x"] @ r_hat)

    def cleanup(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class _EarlyExit(Exception):
    pass


class HFScorer:
    """Scorer using raw HF + exception-based early-exit."""

    def __init__(self, hf_name: str, device: str, dtype, layer: int):
        cfg = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        self.layer = layer
        self.n_layers = cfg.num_hidden_layers

    def _layers(self):
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise ValueError("Cannot locate layers")

    def score(self, formatted: str, r_hat: torch.Tensor) -> float:
        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        captured = {}

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["x"] = h[0, -1].detach().float().cpu()
            raise _EarlyExit

        handle = self._layers()[self.layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(**inputs)
        except _EarlyExit:
            pass
        finally:
            handle.remove()
        return float(captured["x"] @ r_hat)

    def cleanup(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# Metrics
# ============================================================================

def metrics_at_tau(safe_scores: np.ndarray,
                   unsafe_scores: np.ndarray,
                   tau: float) -> Dict:
    fpr = float((safe_scores > tau).mean())
    tpr = float((unsafe_scores > tau).mean())
    s = np.concatenate([safe_scores, unsafe_scores])
    y = np.concatenate([np.zeros_like(safe_scores), np.ones_like(unsafe_scores)])
    preds = (s > tau).astype(int)
    return {
        "tau": tau,
        "fpr": fpr,
        "tpr": tpr,
        "fnr": 1 - tpr,
        "accuracy": float((preds == y).mean()),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "n_refused_safe": int((safe_scores > tau).sum()),
        "n_caught_unsafe": int((unsafe_scores > tau).sum()),
    }


# ============================================================================
# Plots
# ============================================================================

def plot_arasafe(model_name: str,
                 best_layer: int,
                 safe_scores: np.ndarray,
                 unsafe_scores: np.ndarray,
                 thresholds: List[Dict],
                 calib_safe: np.ndarray,
                 calib_unsafe: np.ndarray,
                 auc: float,
                 out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AraSafe histograms with thresholds
    ax = axes[0]
    bins = np.linspace(
        min(safe_scores.min(), unsafe_scores.min()),
        max(safe_scores.max(), unsafe_scores.max()),
        50,
    )
    ax.hist(safe_scores, bins=bins, alpha=0.55, label=f"AraSafe safe (n={len(safe_scores)})", color="C0")
    ax.hist(unsafe_scores, bins=bins, alpha=0.55, label=f"AraSafe unsafe (n={len(unsafe_scores)})", color="C3")
    colors = {"FPR<=0.01": "k", "FPR<=0.05": "gray", "FPR<=0.10": "lightgray", "Youden_J": "C2"}
    for row in thresholds:
        c = colors.get(row["rule"], "k")
        ax.axvline(row["tau"], color=c, ls="--", lw=1.2,
                   label=f"{row['rule']}  (calib tau={row['tau']:.3f})")
    ax.set_xlabel(f"projection score onto r_hat[layer {best_layer}]")
    ax.set_ylabel("count")
    ax.set_title(f"{model_name}  --  AraSafe Real  (AUC={auc:.4f})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Right: AraSafe vs calibration density overlay (transfer check)
    ax = axes[1]
    for src, scores, color, ls in [
        ("calib safe", calib_safe, "C0", "--"),
        ("calib unsafe", calib_unsafe, "C3", "--"),
        ("AraSafe safe", safe_scores, "C0", "-"),
        ("AraSafe unsafe", unsafe_scores, "C3", "-"),
    ]:
        # Empirical PDF via histogram normalized to area=1
        ax.hist(scores, bins=bins, density=True, histtype="step", lw=1.5,
                color=color, ls=ls, label=f"{src} (n={len(scores)})")
    ax.set_xlabel(f"projection score")
    ax.set_ylabel("density")
    ax.set_title("Transfer check: AraSafe vs calibration distributions")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main per-model
# ============================================================================

def evaluate_one_model(spec: dict,
                       arasafe: List[PromptData],
                       dirs_dir: Path,
                       eval_dir: Path,
                       plots_dir: Path,
                       device: str) -> Optional[Dict]:
    name = spec["name"]
    print(f"\n{'='*72}\n{name}  ({spec['hf']}, backend={spec['backend']})\n{'='*72}")
    t0 = time.time()

    # 1. Load saved direction + thresholds
    pt = dirs_dir / name / f"{name}_all_directions.pt"
    th_path = dirs_dir / name / f"{name}_threshold.json"
    if not pt.exists() or not th_path.exists():
        print(f"  SKIP: missing {pt.name} or {th_path.name}")
        return None

    blob = torch.load(pt, map_location="cpu", weights_only=False)
    best_layer = int(blob["exp5_best_layer"])
    r_hat = blob["directions"][best_layer].float()

    with open(th_path, "r", encoding="utf-8") as f:
        th_blob = json.load(f)
    thresholds = th_blob["thresholds"]
    calib_safe = np.array(th_blob["raw_scores"]["safe"])
    calib_unsafe = np.array(th_blob["raw_scores"]["unsafe"])
    print(f"  loaded direction (layer {best_layer}) and {len(thresholds)} thresholds")

    # 2. Load model
    formatter = ChatFormatter(spec["hf"])
    print(f"  loading model ({spec['backend']})...")
    dtype = torch.bfloat16
    if spec["backend"] == "tlens":
        scorer = TLensScorer(spec["hf"], device=device, dtype=dtype, layer=best_layer)
    elif spec["backend"] == "hf":
        scorer = HFScorer(spec["hf"], device=device, dtype=dtype, layer=best_layer)
    else:
        raise ValueError(spec["backend"])

    # 3. Score every AraSafe prompt
    print(f"  scoring {len(arasafe)} AraSafe prompts (early-exit at layer {best_layer})...")
    rows = []
    safe_scores: List[float] = []
    unsafe_scores: List[float] = []
    for i, pd in enumerate(tqdm(arasafe, desc="  scoring")):
        formatted = formatter.format_prompt(pd.prompt)
        try:
            score = scorer.score(formatted, r_hat)
        except Exception as e:
            print(f"\n  WARN at idx {i}: {e}; skipping")
            continue
        rows.append({"idx": i, "label": pd.label, "score": score})
        (safe_scores if pd.label == "normal" else unsafe_scores).append(score)

    safe_arr = np.array(safe_scores)
    unsafe_arr = np.array(unsafe_scores)
    print(f"  {len(safe_arr)} safe + {len(unsafe_arr)} unsafe scored")

    # 4. Aggregate metrics
    y = np.concatenate([np.zeros_like(safe_arr), np.ones_like(unsafe_arr)])
    s = np.concatenate([safe_arr, unsafe_arr])
    auc_arasafe = float(roc_auc_score(y, s))
    print(f"  AraSafe AUC: {auc_arasafe:.4f}")

    per_tau = []
    for row in thresholds:
        m = metrics_at_tau(safe_arr, unsafe_arr, row["tau"])
        m["rule"] = row["rule"]
        m["calibration_fpr"] = row["empirical_fpr"]
        m["calibration_tpr"] = row["tpr"]
        per_tau.append(m)
        print(f"    {row['rule']:<12}  tau={row['tau']:+.4f}  "
              f"FPR={m['fpr']:.4f}  TPR={m['tpr']:.4f}  "
              f"F1={m['f1']:.4f}  "
              f"(calib FPR was {row['empirical_fpr']:.3f}, TPR was {row['tpr']:.3f})")

    # 5. Save per-prompt scores
    eval_dir.mkdir(parents=True, exist_ok=True)
    scores_path = eval_dir / f"{name}_arasafe_scores.jsonl"
    with open(scores_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_json = {
        "model_name": name,
        "best_layer": best_layer,
        "n_safe": int(len(safe_arr)),
        "n_unsafe": int(len(unsafe_arr)),
        "auc_arasafe": auc_arasafe,
        "score_stats": {
            "safe_mean": float(safe_arr.mean()),
            "safe_std": float(safe_arr.std()),
            "unsafe_mean": float(unsafe_arr.mean()),
            "unsafe_std": float(unsafe_arr.std()),
            "mean_gap": float(unsafe_arr.mean() - safe_arr.mean()),
        },
        "per_tau": per_tau,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    metrics_path = eval_dir / f"{name}_arasafe_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    # 6. Plot
    plot_arasafe(name, best_layer, safe_arr, unsafe_arr, thresholds,
                 calib_safe, calib_unsafe, auc_arasafe,
                 plots_dir / f"{name}_arasafe.png")

    print(f"  saved scores + metrics + plot  (elapsed {metrics_json['elapsed_seconds']}s)")

    scorer.cleanup()
    del scorer
    gc.collect()
    torch.cuda.empty_cache()

    return metrics_json


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--arasafe-path", type=str, default=None,
                    help="path to arasafe_human.jsonl (default: data/arasafe_human.jsonl)")
    ap.add_argument("--dirs-dir", type=str, default=None)
    ap.add_argument("--eval-dir", type=str, default=None)
    ap.add_argument("--plots-dir", type=str, default=None)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None,
                    help="balanced subsample for a quick smoke test")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    arasafe_path = (Path(args.arasafe_path) if args.arasafe_path
                    else here / "data" / "arasafe_human.jsonl")
    dirs_dir = Path(args.dirs_dir) if args.dirs_dir else here / "directions"
    eval_dir = Path(args.eval_dir) if args.eval_dir else here / "eval"
    plots_dir = Path(args.plots_dir) if args.plots_dir else here / "plots"

    if not arasafe_path.exists():
        raise SystemExit(f"AraSafe file not found at {arasafe_path}. "
                         "Upload arasafe_human.jsonl into data/ or pass --arasafe-path.")

    arasafe = load_arasafe(arasafe_path, limit=args.limit)
    n_safe = sum(1 for d in arasafe if d.label == "normal")
    n_unsafe = sum(1 for d in arasafe if d.label == "unsafe")
    print(f"loaded AraSafe: {len(arasafe)} prompts ({n_safe} safe, {n_unsafe} unsafe) from {arasafe_path}")
    print(f"directions: {dirs_dir}")
    print(f"device: {args.device}")

    selected = MODELS if not args.models else [m for m in MODELS if m["name"] in args.models]

    summaries = []
    for spec in selected:
        if args.skip_existing:
            mp = eval_dir / f"{spec['name']}_arasafe_metrics.json"
            if mp.exists():
                print(f"\nSKIP {spec['name']} (already evaluated at {mp})")
                continue
        try:
            m = evaluate_one_model(spec, arasafe, dirs_dir, eval_dir, plots_dir, args.device)
            if m:
                summaries.append(m)
        except Exception as e:
            import traceback
            print(f"\nERROR {spec['name']}: {e}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    summary_path = eval_dir / "arasafe_summary.json"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")

    # Compact table
    print("\n" + "=" * 110)
    print(f"{'Model':<28} {'AraSafe AUC':>12}  {'FPR@1%':>8} {'TPR@1%':>8}  "
          f"{'FPR@5%':>8} {'TPR@5%':>8}  {'FPR@10%':>9} {'TPR@10%':>9}")
    print("-" * 110)
    for s in summaries:
        by = {r["rule"]: r for r in s["per_tau"]}
        print(f"{s['model_name']:<28} {s['auc_arasafe']:>12.4f}  "
              f"{by['FPR<=0.01']['fpr']:>8.4f} {by['FPR<=0.01']['tpr']:>8.4f}  "
              f"{by['FPR<=0.05']['fpr']:>8.4f} {by['FPR<=0.05']['tpr']:>8.4f}  "
              f"{by['FPR<=0.10']['fpr']:>9.4f} {by['FPR<=0.10']['tpr']:>9.4f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
