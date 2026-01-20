#!/usr/bin/env python3
"""
Compute Arabic Safety Directions using HuggingFace Transformers (for unsupported models).

This script uses forward hooks to extract residual stream activations from models
that are not supported by TransformerLens.

Usage:
      python transformers_compute_directions.py --models Fanar-1-9B allam
    python transformers_compute_directions.py --batch-size 2
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from datetime import datetime
import traceback
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


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

# Models to process with this script (not supported by TransformerLens)
UNSUPPORTED_MODELS = {
    "Fanar-1-9B": {
        "layer_module": "model.layers",  # Gemma2 uses model.layers
        "n_layers": 42,
        "hidden_size": 3584,
    },
    "allam": {
        "layer_module": "model.layers",  # Llama uses model.layers
        "n_layers": 32,
        "hidden_size": 4096,
    },
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PromptData:
    prompt: str
    label: str


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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_path = model_path

    def format_prompt(self, user_prompt: str) -> str:
        """Format a user prompt using the model's chat template."""
        messages = [
            {"role": "system", "content": ARABIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Some models don't support system messages
                messages_no_sys = [
                    {"role": "user", "content": f"{ARABIC_SYSTEM_PROMPT}\n\n{user_prompt}"}
                ]
                formatted = self.tokenizer.apply_chat_template(
                    messages_no_sys,
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

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and return input_ids."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )


# ============================================================================
# Activation Extraction using Forward Hooks
# ============================================================================

class ActivationExtractor:
    def __init__(self, model: AutoModelForCausalLM, formatter: ChatFormatter, n_layers: int, hidden_size: int):
        self.model = model
        self.formatter = formatter
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.activations = {}
        self.hooks = []

    def _get_layers(self):
        """Get the transformer layers from the model."""
        # Try different common patterns
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            return self.model.gpt_neox.layers
        else:
            raise ValueError("Could not find transformer layers in model")

    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook for a specific layer."""
        def hook(module, input, output):
            # Output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store the last token's activation
            self.activations[layer_idx] = hidden_states[:, -1, :].detach().cpu().float()
        return hook

    def register_hooks(self):
        """Register forward hooks on all layers."""
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_last_token_activations(
        self,
        prompts: List[str],
        batch_size: int = 1,
        desc: str = "Extracting"
    ) -> Dict[int, torch.Tensor]:
        """Extract residual stream activations at the last token position."""
        all_activations = {layer: [] for layer in range(self.n_layers)}

        self.model.eval()
        self.register_hooks()

        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
                    batch_prompts = prompts[i:i + batch_size]
                    formatted = [self.formatter.format_prompt(p) for p in batch_prompts]

                    # Tokenize
                    inputs = self.formatter.tokenizer(
                        formatted,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048,
                    ).to(self.model.device)

                    # Forward pass (hooks will capture activations)
                    self.activations = {}
                    _ = self.model(**inputs)

                    # Collect activations from hooks
                    for layer in range(self.n_layers):
                        if layer in self.activations:
                            all_activations[layer].append(self.activations[layer])

                    # Clear cache
                    del inputs
                    torch.cuda.empty_cache()

        finally:
            self.remove_hooks()

        # Concatenate all batches
        result = {}
        for layer in range(self.n_layers):
            if all_activations[layer]:
                result[layer] = torch.cat(all_activations[layer], dim=0)
            else:
                result[layer] = torch.zeros(len(prompts), self.hidden_size)

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

        # 4. Save metadata JSON
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

    def update_summary_and_table(self, new_results: List[ModelResult]):
        """Update the batch summary and results table with new model results."""
        # Load existing summary
        summary_path = self.base_dir / "batch_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            summary = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "total_models": 0,
                "successful": 0,
                "failed": 0,
                "models": []
            }

        # Update or add new results
        existing_names = {m["model_name"] for m in summary["models"]}

        for result in new_results:
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

            if result.model_name in existing_names:
                # Update existing entry
                for i, m in enumerate(summary["models"]):
                    if m["model_name"] == result.model_name:
                        summary["models"][i] = model_summary
                        break
            else:
                summary["models"].append(model_summary)

        # Recalculate totals
        summary["total_models"] = len(summary["models"])
        summary["successful"] = len([m for m in summary["models"] if m["success"]])
        summary["failed"] = len([m for m in summary["models"] if not m["success"]])
        summary["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save updated summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Regenerate results table
        self._generate_results_table(summary["models"])

        print(f"Updated summary: {summary_path}")

    def _generate_results_table(self, models: List[Dict]):
        """Generate results table from model data."""
        table_lines = []
        table_lines.append("=" * 110)
        table_lines.append("ARABIC SAFETY DIRECTION - RESULTS SUMMARY")
        table_lines.append("=" * 110)
        table_lines.append("")

        # Header
        header = f"{'Model':<30} │ {'Layer':>5} │ {'Val AUC':>8} │ {'Val Acc':>8} │ {'AraSafe AUC':>11} │ {'AraSafe Acc':>11}"
        table_lines.append(header)
        table_lines.append("─" * 30 + "─┼─" + "─" * 5 + "─┼─" + "─" * 8 + "─┼─" + "─" * 8 + "─┼─" + "─" * 11 + "─┼─" + "─" * 11)

        # Sort models by name
        sorted_models = sorted(models, key=lambda m: m["model_name"])

        for model in sorted_models:
            if model["success"]:
                row = (f"{model['model_name']:<30} │ {model['best_layer']:>5} │ "
                       f"{model['validation']['auc_roc']:>8.4f} │ {model['validation']['accuracy']:>8.4f} │ "
                       f"{model['arasafe']['auc_roc']:>11.4f} │ {model['arasafe']['accuracy']:>11.4f}")
                table_lines.append(row)
            else:
                row = f"{model['model_name']:<30} │ {'FAIL':>5} │ {'-':>8} │ {'-':>8} │ {'-':>11} │ {'-':>11}"
                table_lines.append(row)

        table_lines.append("=" * 110)
        table_lines.append("")
        table_lines.append("Notes:")
        table_lines.append("  - Layer: Best layer selected by AUC-ROC on validation set")
        table_lines.append("  - Val: Validation dataset (128 safe + 128 unsafe prompts)")
        table_lines.append("  - AraSafe: AraSafe benchmark (10823 safe + 1254 unsafe prompts)")
        table_lines.append("  - Accuracy computed using optimal threshold per model")
        table_lines.append("")

        # Print and save
        table_text = "\n".join(table_lines)
        print("\n" + table_text)

        table_path = self.base_dir / "results_table.txt"
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(table_text)

        print(f"Table saved to: {table_path}")


# ============================================================================
# Main Processing
# ============================================================================

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
    model_config = UNSUPPORTED_MODELS.get(model_name, {})
    n_layers = model_config.get("n_layers", 32)
    hidden_size = model_config.get("hidden_size", 4096)

    try:
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")

        # 1. Load model
        print(f"\n[1/5] Loading model from {model_path}...")

        # Load config to get actual values
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        n_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print(f"  Layers: {n_layers}, Hidden dim: {hidden_size}")

        # 2. Initialize formatter and extractor
        print(f"\n[2/5] Initializing components...")
        formatter = ChatFormatter(str(model_path))
        extractor = ActivationExtractor(model, formatter, n_layers, hidden_size)
        computer = DirectionComputer(n_layers, hidden_size)
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
            d_model=hidden_size,
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

        gc.collect()
        torch.cuda.empty_cache()

        result = ModelResult(
            model_name=model_name,
            model_path=str(model_path),
            n_layers=n_layers,
            d_model=hidden_size,
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
        description="Compute Arabic safety directions using HuggingFace Transformers"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(UNSUPPORTED_MODELS.keys()),
        help="Models to process (default: Fanar-1-9B, allam)"
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
        "--arasafe-limit",
        type=int,
        default=None,
        help="Limit AraSafe evaluation to N samples (for faster testing)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Transformers-based Arabic Safety Direction Computation")
    print("=" * 70)
    print(f"Models directory: {MODELS_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {DIRECTIONS_DIR}")
    print(f"Device: {args.device}")
    print(f"Models to process: {args.models}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_data, val_data = load_training_data()
    print(f"  Training: {len(train_data)} prompts")
    print(f"  Validation: {len(val_data)} prompts")

    arasafe_data = load_arasafe()
    if args.arasafe_limit:
        safe = [d for d in arasafe_data if d.label == "normal"][:args.arasafe_limit // 2]
        unsafe = [d for d in arasafe_data if d.label == "unsafe"][:args.arasafe_limit // 2]
        arasafe_data = safe + unsafe
    print(f"  AraSafe: {len(arasafe_data)} prompts ({len([d for d in arasafe_data if d.label == 'normal'])} safe, {len([d for d in arasafe_data if d.label == 'unsafe'])} unsafe)")

    # Process models
    saver = ResultsSaver(DIRECTIONS_DIR)
    all_results = []

    for i, model_name in enumerate(args.models):
        print(f"\n[{i+1}/{len(args.models)}] Processing {model_name}")
        model_path = MODELS_DIR / model_name

        if not model_path.exists():
            print(f"  Model path does not exist: {model_path}")
            continue

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

    # Update summary and table
    saver.update_summary_and_table(all_results)

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
