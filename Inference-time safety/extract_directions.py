#!/usr/bin/env python3
"""
Extract per-layer safety directions for the 5 EXP5 models on the new
1000-safe + 1000-unsafe dataset.

For each model, this script:
  1. Loads the model (TransformerLens for Qwen/Llama, HF + hooks for Fanar/ALLaM).
  2. Formats every prompt with the model's chat template + Arabic system prompt
     and add_generation_prompt=True.
  3. Extracts residual stream activation at the LAST token, at EVERY layer.
  4. Computes r(l) = mean(unsafe_acts) - mean(safe_acts) per layer.
  5. Normalizes each direction to unit length.
  6. Saves all per-layer directions + metadata.

We do NOT re-pick a "best layer" — we keep the EXP5 best-layer choices.
The guard later loads r_hat[best_layer] from the saved tensor dict.

================================================================================
RunPod setup (RTX PRO 4500, 32 GB VRAM)
================================================================================

Once. In the pod terminal:

    # Optional but strongly recommended: put HF cache on the persistent volume
    export HF_HOME=/workspace/hf_cache
    mkdir -p $HF_HOME

    # Auth (Llama-3 is gated; Fanar may require accepting the license once)
    export HF_TOKEN=hf_xxx_your_token_xxx
    huggingface-cli login --token $HF_TOKEN  # optional, env var is enough

    cd /workspace/inference_time_guard
    pip install -r requirements.txt

Run all 5 models:

    python extract_directions.py

Run a subset / custom batch size:

    python extract_directions.py --models Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct
    python extract_directions.py --batch-size 2 --skip-existing

Outputs land in `inference_time_guard/directions/<model>/`:
    - <model>_all_directions.pt   # dict: {layer_idx: r_hat (d_model,)}
    - <model>_metadata.json       # config, in-sample AUC sanity check
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

ARABIC_SYSTEM_PROMPT = "أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا."

# (display_name, hf_name, backend, exp5_best_layer)
# backend: "tlens" uses TransformerLens; "hf" uses raw transformers + forward hooks.
MODELS = [
    {"name": "Qwen2.5-3B-Instruct",     "hf": "Qwen/Qwen2.5-3B-Instruct",     "backend": "tlens", "exp5_best_layer": 23},
    {"name": "Qwen2.5-7B-Instruct",     "hf": "Qwen/Qwen2.5-7B-Instruct",     "backend": "tlens", "exp5_best_layer": 20},
    {"name": "Meta-Llama-3-8B-Instruct","hf": "meta-llama/Meta-Llama-3-8B-Instruct", "backend": "tlens", "exp5_best_layer": 14},
    {"name": "Fanar-1-9B",              "hf": "QCRI/Fanar-1-9B",              "backend": "hf",    "exp5_best_layer": 26},
    {"name": "allam",                   "hf": "ALLaM-AI/ALLaM-7B-Instruct-preview", "backend": "hf", "exp5_best_layer": 23},
]


# ============================================================================
# Data
# ============================================================================

@dataclass
class PromptData:
    prompt: str
    label: str  # "normal" | "unsafe"


def load_jsonl(path: Path) -> List[PromptData]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append(PromptData(prompt=r["prompt"], label=r["label"]))
    return out


# ============================================================================
# Chat formatting
# ============================================================================

class ChatFormatter:
    def __init__(self, hf_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": ARABIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Some templates don't support system role
                merged = [{"role": "user",
                           "content": f"{ARABIC_SYSTEM_PROMPT}\n\n{user_prompt}"}]
                return self.tokenizer.apply_chat_template(
                    merged, tokenize=False, add_generation_prompt=True
                )
        # Fallback
        return (f"### System:\n{ARABIC_SYSTEM_PROMPT}\n\n"
                f"### User:\n{user_prompt}\n\n### Assistant:\n")


# ============================================================================
# Activation extraction — TransformerLens backend
# ============================================================================

class TLensExtractor:
    def __init__(self, hf_name: str, device: str, dtype):
        from transformer_lens import HookedTransformer
        # Use *_no_processing to avoid the CPU-RAM blow-up from weight folding
        # (TransformerLens otherwise holds original + processed weights at once,
        # which OOMs on 7B+ models with typical RunPod RAM limits). The skipped
        # transforms are unitary on resid_post — directions are equivalent.
        self.model = HookedTransformer.from_pretrained_no_processing(
            hf_name, device=device, dtype=dtype, trust_remote_code=True
        )
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model

    def extract(self, formatted_prompts: List[str], batch_size: int) -> Dict[int, torch.Tensor]:
        per_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in range(self.n_layers)}
        with torch.no_grad():
            for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="  fwd"):
                batch = formatted_prompts[i:i + batch_size]
                tokens = self.model.to_tokens(batch, prepend_bos=False)
                _, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=lambda n: "resid_post" in n,
                    pos_slice=-1,
                )
                for l in range(self.n_layers):
                    a = cache[("resid_post", l)]
                    if a.dim() == 3:
                        a = a.squeeze(1)
                    per_layer[l].append(a.float().cpu())
                del cache
                torch.cuda.empty_cache()
        return {l: torch.cat(per_layer[l], dim=0) for l in range(self.n_layers)}

    def cleanup(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# Activation extraction — raw HuggingFace + forward hooks
# ============================================================================

class HFHookExtractor:
    def __init__(self, hf_name: str, device: str, dtype):
        cfg = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
        self.n_layers = cfg.num_hidden_layers
        self.d_model = cfg.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        self.device = device
        self._captured: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def _layers(self):
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        raise ValueError("Could not locate transformer layers on model")

    def _make_hook(self, idx: int) -> Callable:
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._captured[idx] = h[:, -1, :].detach().cpu().float()
        return hook

    def _register(self):
        for i, layer in enumerate(self._layers()):
            self._hooks.append(layer.register_forward_hook(self._make_hook(i)))

    def _unregister(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def extract(self, formatted_prompts: List[str], batch_size: int) -> Dict[int, torch.Tensor]:
        per_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in range(self.n_layers)}
        self._register()
        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="  fwd"):
                    batch = formatted_prompts[i:i + batch_size]
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048,
                    ).to(self.model.device)
                    self._captured = {}
                    _ = self.model(**inputs)
                    for l in range(self.n_layers):
                        if l in self._captured:
                            per_layer[l].append(self._captured[l])
                    del inputs
                    torch.cuda.empty_cache()
        finally:
            self._unregister()
        return {l: torch.cat(per_layer[l], dim=0) for l in range(self.n_layers)}

    def cleanup(self):
        self._unregister()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# Direction math
# ============================================================================

def compute_directions(safe_acts: Dict[int, torch.Tensor],
                       unsafe_acts: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    out = {}
    for l in safe_acts:
        mu_n = safe_acts[l].mean(dim=0).squeeze()
        mu_u = unsafe_acts[l].mean(dim=0).squeeze()
        r = (mu_u - mu_n).squeeze()
        out[l] = r / (torch.norm(r) + 1e-8)
    return out


def in_sample_auc(safe_acts: torch.Tensor, unsafe_acts: torch.Tensor, r_hat: torch.Tensor) -> float:
    """Smoke test only — projection scores on the same prompts the direction was fit to."""
    s_safe = (safe_acts @ r_hat).numpy()
    s_unsafe = (unsafe_acts @ r_hat).numpy()
    y = np.array([0] * len(s_safe) + [1] * len(s_unsafe))
    return float(roc_auc_score(y, np.concatenate([s_safe, s_unsafe])))


# ============================================================================
# Main per-model pipeline
# ============================================================================

def run_one_model(spec: dict, safe: List[PromptData], unsafe: List[PromptData],
                  out_dir: Path, batch_size: int, device: str) -> Optional[dict]:
    name = spec["name"]
    print(f"\n{'='*72}\n{name}  ({spec['hf']}, backend={spec['backend']})\n{'='*72}")
    t0 = time.time()

    formatter = ChatFormatter(spec["hf"])
    print("  formatting prompts...")
    safe_fmt = [formatter.format_prompt(p.prompt) for p in safe]
    unsafe_fmt = [formatter.format_prompt(p.prompt) for p in unsafe]

    print(f"  loading model ({spec['backend']})...")
    dtype = torch.bfloat16
    if spec["backend"] == "tlens":
        ext = TLensExtractor(spec["hf"], device=device, dtype=dtype)
    elif spec["backend"] == "hf":
        ext = HFHookExtractor(spec["hf"], device=device, dtype=dtype)
    else:
        raise ValueError(f"unknown backend {spec['backend']}")
    print(f"  layers={ext.n_layers}  d_model={ext.d_model}")

    print("  extracting safe activations...")
    safe_acts = ext.extract(safe_fmt, batch_size)
    print("  extracting unsafe activations...")
    unsafe_acts = ext.extract(unsafe_fmt, batch_size)

    print("  computing directions...")
    directions = compute_directions(safe_acts, unsafe_acts)

    best_layer = spec["exp5_best_layer"]
    if best_layer >= ext.n_layers:
        print(f"  WARN: EXP5 best layer {best_layer} >= n_layers {ext.n_layers}; clamping")
        best_layer = ext.n_layers - 1

    auc_best = in_sample_auc(safe_acts[best_layer], unsafe_acts[best_layer], directions[best_layer])
    print(f"  in-sample AUC at EXP5 best layer {best_layer}: {auc_best:.4f}")

    # Save
    model_out = out_dir / name
    model_out.mkdir(parents=True, exist_ok=True)
    pt_path = model_out / f"{name}_all_directions.pt"
    torch.save(
        {
            "directions": {int(l): directions[l].clone() for l in directions},
            "n_layers": ext.n_layers,
            "d_model": ext.d_model,
            "model_name": name,
            "hf_name": spec["hf"],
            "exp5_best_layer": best_layer,
        },
        pt_path,
    )

    metadata = {
        "model_name": name,
        "hf_name": spec["hf"],
        "backend": spec["backend"],
        "n_layers": ext.n_layers,
        "d_model": ext.d_model,
        "exp5_best_layer": best_layer,
        "in_sample_auc_at_best_layer": round(auc_best, 4),
        "n_safe_prompts": len(safe),
        "n_unsafe_prompts": len(unsafe),
        "system_prompt": ARABIC_SYSTEM_PROMPT,
        "elapsed_seconds": round(time.time() - t0, 1),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    meta_path = model_out / f"{name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  saved: {pt_path.name}, {meta_path.name}  (elapsed {metadata['elapsed_seconds']}s)")

    ext.cleanup()
    del safe_acts, unsafe_acts, directions, ext
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of model display names; default: all 5")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else here / "data"
    out_dir = Path(args.out_dir) if args.out_dir else here / "directions"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = load_jsonl(data_dir / "direction_safe.jsonl")
    unsafe = load_jsonl(data_dir / "direction_unsafe.jsonl")
    print(f"loaded {len(safe)} safe + {len(unsafe)} unsafe prompts from {data_dir}")
    print(f"device: {args.device}  batch_size: {args.batch_size}")
    if args.device == "cpu":
        print("WARN: running on CPU will be extremely slow for 7B+ models")

    selected = MODELS if not args.models else [m for m in MODELS if m["name"] in args.models]
    if not selected:
        raise SystemExit(f"No models match {args.models}. Available: {[m['name'] for m in MODELS]}")

    summary = []
    for spec in selected:
        if args.skip_existing:
            pt = out_dir / spec["name"] / f"{spec['name']}_all_directions.pt"
            if pt.exists():
                print(f"\nSKIP {spec['name']} (already exists at {pt})")
                continue
        try:
            meta = run_one_model(spec, safe, unsafe, out_dir, args.batch_size, args.device)
            if meta:
                summary.append(meta)
        except Exception as e:
            import traceback
            print(f"\nERROR processing {spec['name']}: {e}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    summary_path = out_dir / "extract_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
