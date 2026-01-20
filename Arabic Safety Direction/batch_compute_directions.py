#!/usr/bin/env python3
"""
Batch Compute Arabic Safety Directions for Multiple Models.

This script:
1. Iterates over all models in the models directory
2. Computes safety directions using training data
3. Evaluates on validation data
4. Evaluates on AraSafe benchmark
5. Saves results in an organized directory structure

Usage:
    python batch_compute_directions.py
    python batch_compute_directions.py --models Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct
    python batch_compute_directions.py --batch-size 2 --skip-existing
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from datetime import datetime
import traceback
import gc

# TransformerLens imports
from transformer_lens import HookedTransformer

# HuggingFace imports
from transformers import AutoTokenizer


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("/home/zbibm/Safety-Arabic")
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "Arabic Safety Direction" / "data"
DIRECTIONS_DIR = BASE_DIR / "Arabic Safety Direction" / "directions"
ARASAFE_PATH = BASE_DIR / "Arasafe" / "arasafe_human.jsonl"

# Arabic system prompt
ARABIC_SYSTEM_PROMPT = "أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا."

# AraSafe safe label
ARASAFE_SAFE_LABEL = "آمن/غير ضار"

# Models to exclude (non-instruct models)
EXCLUDED_MODELS = {"Qwen3Guard-Gen-4B"}

# Mapping from local directory names to HuggingFace model names
# TransformerLens requires official HF model names for loading
MODEL_NAME_MAPPING = {
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Fanar-1-9B": "QCRI/Fanar-1-9B",
    "allam": "humain-ai/ALLaM-7B-Instruct-preview",
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PromptData:
    prompt: str
    label: str  # "normal"/"unsafe" or AraSafe label


@dataclass
class LayerMetrics:
    layer: int
    auc_roc: float
    accuracy: float
    mean_gap: float
    threshold: float
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class EvaluationResult:
    dataset_name: str
    n_safe: int
    n_unsafe: int
    best_layer: int
    auc_roc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    mean_gap: float
    threshold: float
    all_layers_metrics: List[Dict] = field(default_factory=list)


@dataclass
class ModelResult:
    model_name: str
    model_path: str
    n_layers: int
    d_model: int
    best_layer: int
    direction_shape: List[int]
    training_metrics: LayerMetrics
    validation_result: EvaluationResult
    arasafe_result: EvaluationResult
    timestamp: str
    success: bool = True
    error_message: str = ""


# ============================================================================
# Data Loading
# ============================================================================

def load_jsonl(path: str) -> List[PromptData]:
    """Load prompts from a JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(PromptData(prompt=item["prompt"], label=item["label"]))
    return data


def load_training_data() -> Tuple[List[PromptData], List[PromptData]]:
    """Load train and validation splits."""
    train_data = (
        load_jsonl(DATA_DIR / "train_normal.jsonl") +
        load_jsonl(DATA_DIR / "train_unsafe.jsonl")
    )
    val_data = (
        load_jsonl(DATA_DIR / "val_normal.jsonl") +
        load_jsonl(DATA_DIR / "val_unsafe.jsonl")
    )
    return train_data, val_data


def load_arasafe() -> List[PromptData]:
    """Load AraSafe benchmark data."""
    data = []
    with open(ARASAFE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # Convert AraSafe labels to normal/unsafe
            label = "normal" if item["Safety_Label"] == ARASAFE_SAFE_LABEL else "unsafe"
            data.append(PromptData(prompt=item["User_Prompt"], label=label))
    return data


# ============================================================================
# Chat Template Formatting
# ============================================================================

class ChatFormatter:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model_path = model_path

    def format_prompt(self, user_prompt: str) -> str:
        """Format a user prompt using the model's chat template."""
        messages = [
            {"role": "system", "content": ARABIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
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
        desc: str = "Extracting"
    ) -> Dict[int, torch.Tensor]:
        """Extract residual stream activations at the last token position."""
        all_activations = {layer: [] for layer in range(self.n_layers)}

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
                batch_prompts = prompts[i:i + batch_size]
                formatted = [self.formatter.format_prompt(p) for p in batch_prompts]
                tokens = self.model.to_tokens(formatted, prepend_bos=False)

                _, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=lambda name: "resid_post" in name,
                    pos_slice=-1,
                )

                for layer in range(self.n_layers):
                    layer_act = cache[("resid_post", layer)]
                    if layer_act.dim() == 3:
                        layer_act = layer_act.squeeze(1)
                    all_activations[layer].append(layer_act.float().cpu())

                del cache
                torch.cuda.empty_cache()

        result = {}
        for layer in range(self.n_layers):
            result[layer] = torch.cat(all_activations[layer], dim=0)

        return result

    def extract_class_activations(
        self,
        data: List[PromptData],
        batch_size: int = 1,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Extract activations separately for normal and unsafe prompts."""
        normal_prompts = [d.prompt for d in data if d.label == "normal"]
        unsafe_prompts = [d.prompt for d in data if d.label == "unsafe"]

        normal_acts = self.get_last_token_activations(
            normal_prompts, batch_size, desc=f"Normal ({len(normal_prompts)})"
        )
        unsafe_acts = self.get_last_token_activations(
            unsafe_prompts, batch_size, desc=f"Unsafe ({len(unsafe_prompts)})"
        )

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
        """Compute safety direction: r(l) = mean(unsafe) - mean(normal)"""
        directions = {}
        for layer in range(self.n_layers):
            mu_normal = normal_acts[layer].mean(dim=0).squeeze()
            mu_unsafe = unsafe_acts[layer].mean(dim=0).squeeze()
            r = mu_unsafe - mu_normal
            directions[layer] = r.squeeze()
        return directions

    def normalize_directions(
        self,
        directions: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Normalize direction vectors to unit length."""
        normalized = {}
        for layer, r in directions.items():
            norm = torch.norm(r)
            normalized[layer] = r / (norm + 1e-8)
        return normalized


# ============================================================================
# Evaluation
# ============================================================================

class Evaluator:
    def __init__(self):
        pass

    def compute_scores(
        self,
        activations: Dict[int, torch.Tensor],
        direction: torch.Tensor,
        layer: int
    ) -> torch.Tensor:
        """Compute projection scores for activations onto direction."""
        return torch.matmul(activations[layer], direction)

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
        labels = np.array([0] * n_normal + [1] * n_unsafe)

        # AUC-ROC
        auc = roc_auc_score(labels, all_scores)

        # Find optimal threshold
        threshold = self._find_optimal_threshold(all_scores, labels)
        predictions = (all_scores > threshold).astype(int)

        # Metrics
        acc = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )

        mean_gap = float(unsafe_scores.mean() - normal_scores.mean())

        return LayerMetrics(
            layer=layer,
            auc_roc=auc,
            accuracy=acc,
            mean_gap=mean_gap,
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1=f1,
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

    def evaluate_all_layers(
        self,
        normal_activations: Dict[int, torch.Tensor],
        unsafe_activations: Dict[int, torch.Tensor],
        directions: Dict[int, torch.Tensor],
    ) -> Tuple[int, List[LayerMetrics]]:
        """Evaluate all layers and select best by AUC-ROC."""
        all_metrics = []
        n_layers = len(directions)

        for layer in range(n_layers):
            r_hat = directions[layer]
            normal_scores = torch.matmul(normal_activations[layer], r_hat)
            unsafe_scores = torch.matmul(unsafe_activations[layer], r_hat)
            metrics = self.evaluate_layer(normal_scores, unsafe_scores, layer)
            all_metrics.append(metrics)

        best = max(all_metrics, key=lambda m: m.auc_roc)
        return best.layer, all_metrics

    def evaluate_on_dataset(
        self,
        data: List[PromptData],
        extractor: ActivationExtractor,
        direction: torch.Tensor,
        layer: int,
        dataset_name: str,
        batch_size: int = 1,
    ) -> EvaluationResult:
        """Evaluate direction on a dataset."""
        print(f"\n  Evaluating on {dataset_name}...")

        # Extract activations
        normal_acts, unsafe_acts = extractor.extract_class_activations(data, batch_size)

        # Compute scores
        normal_scores = torch.matmul(normal_acts[layer], direction)
        unsafe_scores = torch.matmul(unsafe_acts[layer], direction)

        # Get metrics
        metrics = self.evaluate_layer(normal_scores, unsafe_scores, layer)

        return EvaluationResult(
            dataset_name=dataset_name,
            n_safe=len([d for d in data if d.label == "normal"]),
            n_unsafe=len([d for d in data if d.label == "unsafe"]),
            best_layer=layer,
            auc_roc=metrics.auc_roc,
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            mean_gap=metrics.mean_gap,
            threshold=metrics.threshold,
        )


# ============================================================================
# Results Saving
# ============================================================================

class ResultsSaver:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_model_results(
        self,
        model_name: str,
        direction: torch.Tensor,
        all_directions: Dict[int, torch.Tensor],
        result: ModelResult,
        all_metrics: List[LayerMetrics],
    ):
        """Save all results for a model."""
        # Create model-specific directory
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        model_dir = self.base_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)

        layer = result.best_layer
        base_name = f"{safe_name}_arabic_safety_direction_layer{layer}"

        # 1. Save best direction (.pt)
        pt_path = model_dir / f"{base_name}.pt"
        torch.save({
            "direction": direction,
            "layer": layer,
            "model_name": model_name,
            "auc_roc": result.training_metrics.auc_roc,
            "accuracy": result.training_metrics.accuracy,
            "mean_gap": result.training_metrics.mean_gap,
            "threshold": result.training_metrics.threshold,
            "timestamp": result.timestamp,
        }, pt_path)

        # 2. Save as numpy
        npy_path = model_dir / f"{base_name}.npy"
        np.save(npy_path, direction.numpy())

        # 3. Save all directions
        all_dirs_path = model_dir / f"{safe_name}_all_directions.pt"
        torch.save({
            "directions": all_directions,
            "metrics": [
                {"layer": m.layer, "auc_roc": m.auc_roc, "accuracy": m.accuracy, "mean_gap": m.mean_gap}
                for m in all_metrics
            ],
            "model_name": model_name,
            "n_layers": result.n_layers,
            "d_model": result.d_model,
        }, all_dirs_path)

        # 4. Save metadata JSON (simplified - best layer only)
        metadata = {
            "model_name": model_name,
            "model_path": result.model_path,
            "n_layers": result.n_layers,
            "d_model": result.d_model,
            "best_layer": result.best_layer,
            "direction_shape": result.direction_shape,
            "timestamp": result.timestamp,
            "validation": {
                "n_safe": result.validation_result.n_safe,
                "n_unsafe": result.validation_result.n_unsafe,
                "auc_roc": round(result.validation_result.auc_roc, 4),
                "accuracy": round(result.validation_result.accuracy, 4),
                "threshold": round(result.validation_result.threshold, 4),
            },
            "arasafe": {
                "n_safe": result.arasafe_result.n_safe,
                "n_unsafe": result.arasafe_result.n_unsafe,
                "auc_roc": round(result.arasafe_result.auc_roc, 4),
                "accuracy": round(result.arasafe_result.accuracy, 4),
                "threshold": round(result.arasafe_result.threshold, 4),
            },
        }

        meta_path = model_dir / f"{base_name}_metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"  Saved to: {model_dir}")
        return model_dir

    def save_summary(self, all_results: List[ModelResult]):
        """Save a summary of all models with a clean results table."""
        summary = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_models": len(all_results),
            "successful": len([r for r in all_results if r.success]),
            "failed": len([r for r in all_results if not r.success]),
            "models": []
        }

        for result in all_results:
            model_summary = {
                "model_name": result.model_name,
                "success": result.success,
            }

            if result.success:
                model_summary.update({
                    "best_layer": result.best_layer,
                    "n_layers": result.n_layers,
                    "validation": {
                        "auc_roc": round(result.validation_result.auc_roc, 4),
                        "accuracy": round(result.validation_result.accuracy, 4),
                    },
                    "arasafe": {
                        "auc_roc": round(result.arasafe_result.auc_roc, 4),
                        "accuracy": round(result.arasafe_result.accuracy, 4),
                    },
                })
            else:
                model_summary["error"] = result.error_message

            summary["models"].append(model_summary)

        # Save summary JSON
        summary_path = self.base_dir / "batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Create and save results table as text
        table_lines = []
        table_lines.append("=" * 110)
        table_lines.append("ARABIC SAFETY DIRECTION - RESULTS SUMMARY")
        table_lines.append("=" * 110)
        table_lines.append("")

        # Header
        header = f"{'Model':<30} │ {'Layer':>5} │ {'Val AUC':>8} │ {'Val Acc':>8} │ {'AraSafe AUC':>11} │ {'AraSafe Acc':>11}"
        table_lines.append(header)
        table_lines.append("─" * 30 + "─┼─" + "─" * 5 + "─┼─" + "─" * 8 + "─┼─" + "─" * 8 + "─┼─" + "─" * 11 + "─┼─" + "─" * 11)

        for result in all_results:
            if result.success:
                row = (f"{result.model_name:<30} │ {result.best_layer:>5} │ "
                       f"{result.validation_result.auc_roc:>8.4f} │ {result.validation_result.accuracy:>8.4f} │ "
                       f"{result.arasafe_result.auc_roc:>11.4f} │ {result.arasafe_result.accuracy:>11.4f}")
                table_lines.append(row)
            else:
                row = f"{result.model_name:<30} │ {'FAIL':>5} │ {'-':>8} │ {'-':>8} │ {'-':>11} │ {'-':>11}"
                table_lines.append(row)

        table_lines.append("=" * 110)
        table_lines.append("")
        table_lines.append("Notes:")
        table_lines.append("  - Layer: Best layer selected by AUC-ROC on validation set")
        table_lines.append("  - Val: Validation dataset (128 safe + 128 unsafe prompts)")
        table_lines.append("  - AraSafe: AraSafe benchmark (10823 safe + 1254 unsafe prompts)")
        table_lines.append("  - Accuracy computed using optimal threshold per model")
        table_lines.append("")

        # Print and save table
        table_text = "\n".join(table_lines)
        print("\n" + table_text)

        table_path = self.base_dir / "results_table.txt"
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(table_text)

        print(f"Summary saved to: {summary_path}")
        print(f"Table saved to: {table_path}")


# ============================================================================
# Main Processing
# ============================================================================

def get_available_models(models_dir: Path) -> List[str]:
    """Get list of available model directories (excluding non-instruct models)."""
    models = []
    for item in models_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Skip excluded models
            if item.name in EXCLUDED_MODELS:
                continue
            # Check if it has model files
            if any(item.glob("*.safetensors")) or any(item.glob("*.bin")):
                models.append(item.name)
    return sorted(models)


def process_model(
    model_name: str,
    model_path: Path,
    train_data: List[PromptData],
    val_data: List[PromptData],
    arasafe_data: List[PromptData],
    batch_size: int = 1,
    device: str = "cuda",
) -> Tuple[ModelResult, Optional[torch.Tensor], Optional[Dict], Optional[List]]:
    """Process a single model: compute direction and evaluate."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")

        # 1. Load model
        print(f"\n[1/5] Loading model...")

        # Get HuggingFace model name from mapping
        hf_model_name = MODEL_NAME_MAPPING.get(model_name)
        if hf_model_name is None:
            raise ValueError(f"No HuggingFace model mapping found for '{model_name}'")

        print(f"  HF model: {hf_model_name}")
        print(f"  Local tokenizer: {model_path}")

        model = HookedTransformer.from_pretrained(
            hf_model_name,
            device=device,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        print(f"  Layers: {n_layers}, Hidden dim: {d_model}")

        # 2. Initialize formatter and extractor
        print(f"\n[2/5] Initializing components...")
        # Use local path for tokenizer to ensure correct chat template
        formatter = ChatFormatter(str(model_path))
        extractor = ActivationExtractor(model, formatter)
        computer = DirectionComputer(n_layers, d_model)
        evaluator = Evaluator()

        # 3. Extract training activations and compute direction
        print(f"\n[3/5] Computing safety direction from training data...")
        train_normal_acts, train_unsafe_acts = extractor.extract_class_activations(
            train_data, batch_size
        )

        raw_directions = computer.compute_mean_directions(train_normal_acts, train_unsafe_acts)
        normalized_directions = computer.normalize_directions(raw_directions)

        # 4. Select best layer on validation
        print(f"\n[4/5] Selecting best layer on validation data...")
        val_normal_acts, val_unsafe_acts = extractor.extract_class_activations(
            val_data, batch_size
        )

        best_layer, all_metrics = evaluator.evaluate_all_layers(
            val_normal_acts, val_unsafe_acts, normalized_directions
        )
        best_metrics = next(m for m in all_metrics if m.layer == best_layer)
        best_direction = normalized_directions[best_layer]

        print(f"  Best layer: {best_layer} (AUC={best_metrics.auc_roc:.4f}, Acc={best_metrics.accuracy:.4f})")

        # 5. Evaluate on datasets
        print(f"\n[5/5] Evaluating on benchmarks...")

        # Validation evaluation
        val_result = EvaluationResult(
            dataset_name="validation",
            n_safe=len([d for d in val_data if d.label == "normal"]),
            n_unsafe=len([d for d in val_data if d.label == "unsafe"]),
            best_layer=best_layer,
            auc_roc=best_metrics.auc_roc,
            accuracy=best_metrics.accuracy,
            precision=best_metrics.precision,
            recall=best_metrics.recall,
            f1=best_metrics.f1,
            mean_gap=best_metrics.mean_gap,
            threshold=best_metrics.threshold,
        )

        # AraSafe evaluation
        arasafe_result = evaluator.evaluate_on_dataset(
            arasafe_data, extractor, best_direction, best_layer,
            "AraSafe", batch_size
        )

        # Clean up
        del model, extractor, train_normal_acts, train_unsafe_acts
        del val_normal_acts, val_unsafe_acts
        gc.collect()
        torch.cuda.empty_cache()

        # Create result
        result = ModelResult(
            model_name=model_name,
            model_path=str(model_path),
            n_layers=n_layers,
            d_model=d_model,
            best_layer=best_layer,
            direction_shape=list(best_direction.shape),
            training_metrics=best_metrics,
            validation_result=val_result,
            arasafe_result=arasafe_result,
            timestamp=timestamp,
            success=True,
        )

        return result, best_direction, normalized_directions, all_metrics

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        traceback.print_exc()

        # Clean up on error
        gc.collect()
        torch.cuda.empty_cache()

        result = ModelResult(
            model_name=model_name,
            model_path=str(model_path),
            n_layers=0,
            d_model=0,
            best_layer=0,
            direction_shape=[],
            training_metrics=LayerMetrics(0, 0, 0, 0, 0),
            validation_result=EvaluationResult("validation", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            arasafe_result=EvaluationResult("AraSafe", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            timestamp=timestamp,
            success=False,
            error_message=str(e),
        )

        return result, None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Batch compute Arabic safety directions for multiple models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to process (default: all in models directory)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for activation extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have computed directions"
    )
    parser.add_argument(
        "--arasafe-limit",
        type=int,
        default=None,
        help="Limit AraSafe evaluation to N samples (for faster testing)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Batch Arabic Safety Direction Computation")
    print("=" * 70)
    print(f"Models directory: {MODELS_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {DIRECTIONS_DIR}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Get models to process
    available_models = get_available_models(MODELS_DIR)
    print(f"\nAvailable models: {available_models}")

    if args.models:
        models_to_process = [m for m in args.models if m in available_models]
        if len(models_to_process) != len(args.models):
            missing = set(args.models) - set(models_to_process)
            print(f"Warning: Models not found: {missing}")
    else:
        models_to_process = available_models

    if args.skip_existing:
        existing = []
        for model_name in models_to_process:
            safe_name = model_name.replace("/", "_").replace(" ", "_")
            model_dir = DIRECTIONS_DIR / safe_name
            if model_dir.exists() and any(model_dir.glob("*.pt")):
                existing.append(model_name)
        models_to_process = [m for m in models_to_process if m not in existing]
        if existing:
            print(f"Skipping existing: {existing}")

    print(f"\nModels to process: {models_to_process}")

    # Load data
    print("\nLoading data...")
    train_data, val_data = load_training_data()
    print(f"  Training: {len(train_data)} prompts")
    print(f"  Validation: {len(val_data)} prompts")

    arasafe_data = load_arasafe()
    if args.arasafe_limit:
        # Sample balanced subset
        safe = [d for d in arasafe_data if d.label == "normal"][:args.arasafe_limit // 2]
        unsafe = [d for d in arasafe_data if d.label == "unsafe"][:args.arasafe_limit // 2]
        arasafe_data = safe + unsafe
    print(f"  AraSafe: {len(arasafe_data)} prompts ({len([d for d in arasafe_data if d.label == 'normal'])} safe, {len([d for d in arasafe_data if d.label == 'unsafe'])} unsafe)")

    # Process models
    saver = ResultsSaver(DIRECTIONS_DIR)
    all_results = []

    for i, model_name in enumerate(models_to_process):
        print(f"\n[{i+1}/{len(models_to_process)}] Processing {model_name}")
        model_path = MODELS_DIR / model_name

        result, direction, all_directions, all_metrics = process_model(
            model_name=model_name,
            model_path=model_path,
            train_data=train_data,
            val_data=val_data,
            arasafe_data=arasafe_data,
            batch_size=args.batch_size,
            device=args.device,
        )

        all_results.append(result)

        if result.success and direction is not None:
            saver.save_model_results(
                model_name, direction, all_directions, result, all_metrics
            )

    # Save summary
    saver.save_summary(all_results)


if __name__ == "__main__":
    main()
