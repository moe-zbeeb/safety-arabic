import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

INPUT_FILE = "Arabic boundary-set_MSA.json"
BATCH_SIZE = 32
MAX_LENGTH = 512

MODELS = {
    "EGY": {
        "name": "QCRI/AraDiCE-msa-to-egy",
        "bos_token": "arz_Arab",
        "gpu": 0,
        "output_file": "Arabic boundary-set_EGY.json",
    },
    "LEV": {
        "name": "QCRI/AraDiCE-msa-to-lev",
        "bos_token": "ajp_Arab",
        "gpu": 1,
        "output_file": "Arabic boundary-set_LEV.json",
    },
}


def load_model(model_cfg):
    device = f"cuda:{model_cfg['gpu']}"
    print(f"Loading {model_cfg['name']} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["name"]).to(device)
    model.eval()
    forced_bos_id = tokenizer.convert_tokens_to_ids(model_cfg["bos_token"])
    print(f"Loaded {model_cfg['name']} on {device}")
    return tokenizer, model, device, forced_bos_id


def translate_batch(prompts, tokenizer, model, device, forced_bos_id):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            max_length=MAX_LENGTH,
        )
    return tokenizer.batch_decode(translated, skip_special_tokens=True)


def translate_dialect(data, model_cfg, loaded_model):
    tokenizer, model, device, forced_bos_id = loaded_model
    prompts = [item["Prompt"] for item in data]
    translations = []

    for i in tqdm(
        range(0, len(prompts), BATCH_SIZE),
        desc=f"{model_cfg['name']}",
    ):
        batch = prompts[i : i + BATCH_SIZE]
        translated = translate_batch(batch, tokenizer, model, device, forced_bos_id)
        translations.extend(translated)

    result = []
    for item, translated_prompt in zip(data, translations):
        result.append(
            {
                "Safe/Unsafe": item["Safe/Unsafe"],
                "Category": item["Category"],
                "Prompt": translated_prompt,
            }
        )

    with open(model_cfg["output_file"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved {model_cfg['output_file']}")


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} prompts")

    # Load models SEQUENTIALLY to avoid CPU RAM exhaustion
    print("Loading models sequentially...")
    loaded_models = {}
    for dialect, cfg in MODELS.items():
        loaded_models[dialect] = load_model(cfg)

    # Run inference in PARALLEL (each model on its own GPU)
    print("Running inference in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            dialect: executor.submit(translate_dialect, data, cfg, loaded_models[dialect])
            for dialect, cfg in MODELS.items()
        }
        for dialect, future in futures.items():
            future.result()
            print(f"Done: {dialect}")

    print("All translations complete!")


if __name__ == "__main__":
    main()
