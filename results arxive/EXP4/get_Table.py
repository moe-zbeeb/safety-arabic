import json
from pathlib import Path

# Define models and checkpoints
models = ['allam', 'Fanar', 'llama38b', 'Qwen2.53b', 'Qwen2.57B']
checkpoints = [50, 100, 150, 200, 250, 300, 313]
datasets = ['refusals', 'mix']

base_path = Path('/home/zbibm/Safety-Arabic/results arxive/EXP3')
exp2_path = Path('/home/zbibm/Safety-Arabic/results arxive/Exp2')

# Mapping of model names to their base model summary files
base_model_files = {
    'allam': 'allam_summary.json',
    'Fanar': 'Fanar-1-9B_summary.json',
    'llama38b': 'Meta-Llama-3-8B-Instruct_summary.json',
    'Qwen2.53b': 'Qwen2.5-3B-Instruct_summary.json',
    'Qwen2.57B': 'Qwen2.5-7B-Instruct_summary.json'
}

# Load base model data
base_data = {}
for model, filename in base_model_files.items():
    json_file = exp2_path / filename
    if json_file.exists():
        with open(json_file, 'r') as f:
            summary = json.load(f)
            base_data[model] = {
                'safe_refusal_rate': summary['safe_refusal_rate'],
                'unsafe_refusal_rate': summary['unsafe_refusal_rate']
            }
    else:
        base_data[model] = {
            'safe_refusal_rate': None,
            'unsafe_refusal_rate': None
        }

# Collect data from EXP3
data = {dataset: {model: {} for model in models} for dataset in datasets}

# Special handling for folder naming inconsistency (Qwen2.57B vs Qwen2.57b)
def get_model_folder_name(model, dataset):
    if model == 'Qwen2.57B' and dataset == 'refusals':
        return 'Qwen2.57b-refusals'
    return f"{model}-{dataset}"

for dataset in datasets:
    for model in models:
        folder_name = get_model_folder_name(model, dataset)
        model_dir = base_path / folder_name

        for checkpoint in checkpoints:
            json_file = model_dir / f"checkpoint-{checkpoint}_summary.json"

            if json_file.exists():
                with open(json_file, 'r') as f:
                    summary = json.load(f)
                    data[dataset][model][checkpoint] = {
                        'safe_refusal_rate': summary['safe_refusal_rate'],
                        'unsafe_refusal_rate': summary['unsafe_refusal_rate']
                    }
            else:
                data[dataset][model][checkpoint] = {
                    'safe_refusal_rate': None,
                    'unsafe_refusal_rate': None
                }

# Generate tables for each model
output = []
output.append("# Refusal Rates Comparison Across Checkpoints\n")
output.append("Comparison of safe and unsafe refusal rates for models trained on refusals-only data vs mixed data.\n")
output.append("The 'Base Model' row shows the performance of the original pretrained model before fine-tuning.\n\n")

for model in models:
    output.append(f"## {model}\n")
    output.append("| Checkpoint | **Refusals Data** |                  | **Mix Data**      |                  |")
    output.append("|------------|-------------------|------------------|-------------------|------------------|")
    output.append("|            | Safe Refusal %    | Unsafe Refusal % | Safe Refusal %    | Unsafe Refusal % |")

    # Add base model row
    base_safe = base_data[model]['safe_refusal_rate']
    base_unsafe = base_data[model]['unsafe_refusal_rate']
    base_safe_str = f"{base_safe:.2f}" if base_safe is not None else "N/A"
    base_unsafe_str = f"{base_unsafe:.2f}" if base_unsafe is not None else "N/A"

    output.append(f"| **Base Model** | {base_safe_str:>17} | {base_unsafe_str:>16} | {base_safe_str:>17} | {base_unsafe_str:>16} |")

    # Add checkpoint rows
    for checkpoint in checkpoints:
        refusals_safe = data['refusals'][model][checkpoint]['safe_refusal_rate']
        refusals_unsafe = data['refusals'][model][checkpoint]['unsafe_refusal_rate']
        mix_safe = data['mix'][model][checkpoint]['safe_refusal_rate']
        mix_unsafe = data['mix'][model][checkpoint]['unsafe_refusal_rate']

        # Format values or show N/A if missing
        refusals_safe_str = f"{refusals_safe:.2f}" if refusals_safe is not None else "N/A"
        refusals_unsafe_str = f"{refusals_unsafe:.2f}" if refusals_unsafe is not None else "N/A"
        mix_safe_str = f"{mix_safe:.2f}" if mix_safe is not None else "N/A"
        mix_unsafe_str = f"{mix_unsafe:.2f}" if mix_unsafe is not None else "N/A"

        output.append(f"| {checkpoint:<10} | {refusals_safe_str:>17} | {refusals_unsafe_str:>16} | {mix_safe_str:>17} | {mix_unsafe_str:>16} |")

    output.append("\n")

# Write to file
output_text = "\n".join(output)
output_file = base_path / 'refusal_rates_tables.md'
with open(output_file, 'w') as f:
    f.write(output_text)

print(output_text)
print(f"\nTables saved to: {output_file}")
