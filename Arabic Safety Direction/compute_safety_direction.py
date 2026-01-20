#!/usr/bin/env python3
"""
Compute Arabic Safety Direction using TransformerLens.

This script:
1. Loads JSONL prompts (train/val for normal + unsafe)
2. Formats prompts with the model's chat template
3. Runs the model and caches residual stream activations at the last token
4. Computes direction per layer: r(l) = mean_unsafe(l) - mean_normal(l)
5. Picks the best layer by AUC-ROC on validation set
6. Saves the best r_hat to disk

Usage:
    python compute_safety_direction.py --model Qwen/Qwen2.5-3B-Instruct
    python compute_safety_direction.py --model Qwen/Qwen2.5-3B-Instruct --batch-size 4

Supported models (TransformerLens compatible):
    - Qwen/Qwen2.5-3B-Instruct
    - Qwen/Qwen2.5-7B-Instruct
    - meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime

# TransformerLens imports
from transformer_lens import HookedTransformer

# HuggingFace imports for tokenizer
from transformers import AutoTokenizer


# ============================================================================
# Data Loading
# ============================================================================

@dataclass
class PromptData:
    prompt: str
    label: str  # "normal" or "unsafe"


def load_jsonl(path: str) -> List[PromptData]:
    """Load prompts from a JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(PromptData(prompt=item["prompt"], label=item["label"]))
    return data


def load_all_data(data_dir: str) -> Tuple[List[PromptData], List[PromptData]]:
    """Load train and validation splits."""
    data_dir = Path(data_dir)

    train_data = (
        load_jsonl(data_dir / "train_normal.jsonl") +
        load_jsonl(data_dir / "train_unsafe.jsonl")
    )

    val_data = (
        load_jsonl(data_dir / "val_normal.jsonl") +
        load_jsonl(data_dir / "val_unsafe.jsonl")
    )

    return train_data, val_data


# ============================================================================
# Chat Template Formatting
# ============================================================================

# Arabic system prompt (consistent with existing project patterns)
ARABIC_SYSTEM_PROMPT = "أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا."


class ChatFormatter:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model_path = model_path

    def format_prompt(self, user_prompt: str) -> str:
        """
        Format a user prompt using the model's chat template.
        Returns the formatted string ready for tokenization.
        """
        messages = [
            {"role": "system", "content": ARABIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use the model's native chat template with generation prompt
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # Critical: adds assistant header
            )
        else:
            # Fallback for models without chat template
            formatted = self._manual_format(messages)

        return formatted

    def _manual_format(self, messages: List[Dict]) -> str:
        """Fallback formatting for models without chat templates."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"### {role.capitalize()}:\n{content}")
        parts.append("### Assistant:\n")
        return "\n\n".join(parts)


# ============================================================================
# Activation Extraction
# ============================================================================

class ActivationExtractor:
    def __init__(self, model: HookedTransformer, formatter: ChatFormatter):
        self.model = model
        self.formatter = formatter
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model

    def get_last_token_activations(
        self,
        prompts: List[str],
        batch_size: int = 1,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract residual stream activations at the last token position.

        Returns:
            Dict mapping layer_idx -> tensor of shape (n_prompts, d_model)
        """
        all_activations = {layer: [] for layer in range(self.n_layers)}

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting activations"):
                batch_prompts = prompts[i:i + batch_size]

                # Format each prompt
                formatted = [self.formatter.format_prompt(p) for p in batch_prompts]

                # Tokenize - TransformerLens handles this
                tokens = self.model.to_tokens(formatted, prepend_bos=False)

                # Run with cache, get activations at last position
                _, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=lambda name: "resid_post" in name,
                    pos_slice=-1,  # Only last token position
                )

                # Extract per-layer activations
                for layer in range(self.n_layers):
                    # Shape might be (batch, 1, d_model) or (batch, d_model)
                    layer_act = cache[("resid_post", layer)]
                    # Squeeze out any extra dimensions to get (batch, d_model)
                    if layer_act.dim() == 3:
                        layer_act = layer_act.squeeze(1)
                    all_activations[layer].append(layer_act.float().cpu())

                # Clear cache to free memory
                del cache
                torch.cuda.empty_cache()

        # Concatenate all batches
        result = {}
        for layer in range(self.n_layers):
            result[layer] = torch.cat(all_activations[layer], dim=0)

        return result

    def extract_class_activations(
        self,
        data: List[PromptData],
        batch_size: int = 1,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Extract activations separately for normal and unsafe prompts.

        Returns:
            (normal_activations, unsafe_activations) - each is Dict[layer -> Tensor]
        """
        normal_prompts = [d.prompt for d in data if d.label == "normal"]
        unsafe_prompts = [d.prompt for d in data if d.label == "unsafe"]

        print(f"Extracting activations for {len(normal_prompts)} normal prompts...")
        normal_acts = self.get_last_token_activations(normal_prompts, batch_size)

        print(f"Extracting activations for {len(unsafe_prompts)} unsafe prompts...")
        unsafe_acts = self.get_last_token_activations(unsafe_prompts, batch_size)

        return normal_acts, unsafe_acts


# ============================================================================
# Direction Computation
# ============================================================================

class DirectionComputer:
    def __init__(self, n_layers: int, d_model: int):
        self.n_layers = n_layers
        self.d_model = d_model

    def compute_mean_directions(
        self,
        normal_acts: Dict[int, torch.Tensor],
        unsafe_acts: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Compute the safety direction for each layer.

        Direction r(l) = mean(unsafe) - mean(normal)
        Positive projection = more unsafe
        """
        directions = {}

        for layer in range(self.n_layers):
            # Compute means
            mu_normal = normal_acts[layer].mean(dim=0).squeeze()  # (d_model,)
            mu_unsafe = unsafe_acts[layer].mean(dim=0).squeeze()  # (d_model,)

            # Direction: unsafe - normal
            r = mu_unsafe - mu_normal
            directions[layer] = r.squeeze()  # Ensure 1D

        return directions

    def normalize_directions(
        self,
        directions: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Normalize direction vectors to unit length: r_hat = r / ||r||"""
        normalized = {}
        for layer, r in directions.items():
            norm = torch.norm(r)
            normalized[layer] = r / (norm + 1e-8)
        return normalized


# ============================================================================
# Layer Selection
# ============================================================================

@dataclass
class LayerMetrics:
    layer: int
    auc_roc: float
    accuracy: float
    mean_gap: float  # mean(unsafe_scores) - mean(normal_scores)
    threshold: float


class LayerSelector:
    def __init__(self):
        pass

    def evaluate_layer(
        self,
        normal_scores: torch.Tensor,
        unsafe_scores: torch.Tensor,
        layer: int,
    ) -> LayerMetrics:
        """Evaluate how well a layer's projections separate classes."""
        n_normal = len(normal_scores)
        n_unsafe = len(unsafe_scores)

        all_scores = torch.cat([normal_scores, unsafe_scores]).numpy()
        labels = np.array([0] * n_normal + [1] * n_unsafe)  # 0=normal, 1=unsafe

        # AUC-ROC (primary metric)
        auc = roc_auc_score(labels, all_scores)

        # Find optimal threshold
        threshold = self._find_optimal_threshold(all_scores, labels)
        predictions = (all_scores > threshold).astype(int)
        acc = accuracy_score(labels, predictions)

        # Mean gap
        mean_gap = float(unsafe_scores.mean() - normal_scores.mean())

        return LayerMetrics(
            layer=layer,
            auc_roc=auc,
            accuracy=acc,
            mean_gap=mean_gap,
            threshold=threshold,
        )

    def _find_optimal_threshold(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Find threshold that maximizes accuracy."""
        sorted_scores = np.sort(np.unique(scores))
        best_acc = 0
        best_thresh = 0

        for thresh in sorted_scores:
            preds = (scores > thresh).astype(int)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        return float(best_thresh)

    def select_best_layer(
        self,
        normal_activations: Dict[int, torch.Tensor],
        unsafe_activations: Dict[int, torch.Tensor],
        directions: Dict[int, torch.Tensor],
    ) -> Tuple[int, List[LayerMetrics]]:
        """
        Evaluate all layers and select the best one.

        Returns:
            (best_layer_idx, all_metrics)
        """
        all_metrics = []
        n_layers = len(directions)

        print("\nEvaluating layers on validation set:")
        print("-" * 60)

        for layer in range(n_layers):
            r_hat = directions[layer]

            # Compute projection scores
            normal_scores = torch.matmul(normal_activations[layer], r_hat)
            unsafe_scores = torch.matmul(unsafe_activations[layer], r_hat)

            metrics = self.evaluate_layer(normal_scores, unsafe_scores, layer)
            all_metrics.append(metrics)

            print(f"Layer {layer:2d}: AUC={metrics.auc_roc:.4f}, "
                  f"Acc={metrics.accuracy:.4f}, Gap={metrics.mean_gap:.4f}")

        # Select best by AUC-ROC
        best = max(all_metrics, key=lambda m: m.auc_roc)
        print("-" * 60)
        print(f"Best layer: {best.layer} (AUC={best.auc_roc:.4f}, Acc={best.accuracy:.4f})")

        return best.layer, all_metrics


# ============================================================================
# Saving Results
# ============================================================================

class DirectionSaver:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_direction(
        self,
        direction: torch.Tensor,
        layer: int,
        model_name: str,
        metrics: LayerMetrics,
        all_metrics: List[LayerMetrics],
    ) -> str:
        """Save the best direction vector to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name}_arabic_safety_direction_layer{layer}"

        # Save direction vector (.pt format)
        pt_path = self.output_dir / f"{base_name}.pt"
        torch.save({
            "direction": direction,
            "layer": layer,
            "model_name": model_name,
            "auc_roc": metrics.auc_roc,
            "accuracy": metrics.accuracy,
            "mean_gap": metrics.mean_gap,
            "threshold": metrics.threshold,
            "timestamp": timestamp,
        }, pt_path)

        # Save as numpy too
        npy_path = self.output_dir / f"{base_name}.npy"
        np.save(npy_path, direction.numpy())

        # Save metadata JSON
        metadata = {
            "model_name": model_name,
            "best_layer": layer,
            "direction_shape": list(direction.shape),
            "metrics": {
                "auc_roc": metrics.auc_roc,
                "accuracy": metrics.accuracy,
                "mean_gap": metrics.mean_gap,
                "threshold": metrics.threshold,
            },
            "all_layers_metrics": [
                {
                    "layer": m.layer,
                    "auc_roc": m.auc_roc,
                    "accuracy": m.accuracy,
                    "mean_gap": m.mean_gap,
                }
                for m in all_metrics
            ],
            "timestamp": timestamp,
            "data_sources": {
                "train_normal": "train_normal.jsonl (512 prompts)",
                "train_unsafe": "train_unsafe.jsonl (512 prompts)",
                "val_normal": "val_normal.jsonl (128 prompts)",
                "val_unsafe": "val_unsafe.jsonl (128 prompts)",
            }
        }

        meta_path = self.output_dir / f"{base_name}_metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save all directions for analysis
        all_dirs_path = self.output_dir / f"{model_name}_all_directions.pt"
        torch.save({
            "directions": {layer: directions[layer] for layer, directions in enumerate([direction])},
            "model_name": model_name,
        }, all_dirs_path)

        return str(pt_path)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute Arabic Safety Direction using TransformerLens"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Local path to tokenizer (optional, defaults to model name)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for activation extraction (use 1 for large models)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (defaults to ./data relative to this script)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for saved directions (defaults to ./directions)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()

    # Set default paths relative to script location
    script_dir = Path(__file__).parent
    if args.data_dir is None:
        args.data_dir = script_dir / "data"
    if args.output_dir is None:
        args.output_dir = script_dir / "directions"

    # Handle model name - extract short name for output files
    model_name = args.model.replace("/", "_")

    # Use local tokenizer path if provided, otherwise use model name
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model

    print("=" * 70)
    print(f"Computing Arabic Safety Direction")
    print(f"Model: {args.model}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    train_data, val_data = load_all_data(args.data_dir)
    print(f"  Train: {len(train_data)} prompts ({len([d for d in train_data if d.label == 'normal'])} normal, {len([d for d in train_data if d.label == 'unsafe'])} unsafe)")
    print(f"  Val: {len(val_data)} prompts ({len([d for d in val_data if d.label == 'normal'])} normal, {len([d for d in val_data if d.label == 'unsafe'])} unsafe)")

    # 2. Load model with TransformerLens
    print(f"\n[2/6] Loading model with TransformerLens...")
    model = HookedTransformer.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  Layers: {n_layers}")
    print(f"  Hidden dim: {d_model}")

    # 3. Initialize formatter
    print(f"\n[3/6] Initializing chat formatter...")
    formatter = ChatFormatter(tokenizer_path)

    # Show example formatted prompt
    example_formatted = formatter.format_prompt("مرحبا")
    print(f"  Example formatted prompt:\n  {example_formatted[:200]}...")

    # 4. Extract activations
    print(f"\n[4/6] Extracting activations...")
    extractor = ActivationExtractor(model, formatter)

    # Training set activations (for computing direction)
    print("\nProcessing training set:")
    train_normal_acts, train_unsafe_acts = extractor.extract_class_activations(
        train_data,
        batch_size=args.batch_size
    )

    # Validation set activations (for layer selection)
    print("\nProcessing validation set:")
    val_normal_acts, val_unsafe_acts = extractor.extract_class_activations(
        val_data,
        batch_size=args.batch_size
    )

    # 5. Compute directions
    print(f"\n[5/6] Computing safety directions...")
    computer = DirectionComputer(n_layers, d_model)

    # Compute raw directions from training data
    raw_directions = computer.compute_mean_directions(
        train_normal_acts,
        train_unsafe_acts
    )

    # Normalize to unit vectors
    normalized_directions = computer.normalize_directions(raw_directions)
    print(f"  Computed {len(normalized_directions)} direction vectors (one per layer)")

    # 6. Select best layer using validation data
    print(f"\n[6/6] Selecting best layer...")
    selector = LayerSelector()
    best_layer, all_metrics = selector.select_best_layer(
        val_normal_acts,
        val_unsafe_acts,
        normalized_directions,
    )

    # 7. Save results
    print(f"\nSaving results...")
    saver = DirectionSaver(args.output_dir)

    best_metrics = next(m for m in all_metrics if m.layer == best_layer)
    saved_path = saver.save_direction(
        direction=normalized_directions[best_layer],
        layer=best_layer,
        model_name=model_name,
        metrics=best_metrics,
        all_metrics=all_metrics,
    )

    # Also save all normalized directions
    all_dirs_path = Path(args.output_dir) / f"{model_name}_all_directions.pt"
    torch.save({
        "directions": normalized_directions,
        "metrics": [
            {"layer": m.layer, "auc_roc": m.auc_roc, "accuracy": m.accuracy, "mean_gap": m.mean_gap}
            for m in all_metrics
        ],
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
    }, all_dirs_path)

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Best layer: {best_layer}")
    print(f"AUC-ROC: {best_metrics.auc_roc:.4f}")
    print(f"Accuracy: {best_metrics.accuracy:.4f}")
    print(f"Mean gap: {best_metrics.mean_gap:.4f}")
    print(f"\nSaved to:")
    print(f"  Direction: {saved_path}")
    print(f"  All directions: {all_dirs_path}")


if __name__ == "__main__":
    main()
