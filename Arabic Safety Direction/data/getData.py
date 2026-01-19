import json
import random
import re
from datasets import load_dataset

SEED = 42
random.seed(SEED)

# ---- config: sizes you asked about ----
N_TRAIN_PER_CLASS = 512   # good default; use 128 if you want a quick pilot
N_VAL_PER_CLASS   = 128

# Optional: length filters to avoid extreme outliers (tokenization varies by model)
MIN_CHARS = 15
MAX_CHARS = 800

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def keep_text(s: str) -> bool:
    s = clean_text(s)
    return (MIN_CHARS <= len(s) <= MAX_CHARS)

def save_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- 1) NORMAL prompts from Hala-4.6M-SFT ----
# Structure shown: {"conversations": [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
hala = load_dataset("hammh0a/Hala-4.6M-SFT", split="train")

normal_prompts = []
for ex in hala:
    conv = ex.get("conversations", None)
    if not isinstance(conv, list) or len(conv) == 0:
        continue

    # take the first human turn as the instruction prompt
    first = conv[0]
    if isinstance(first, dict) and first.get("from") == "human":
        p = clean_text(first.get("value", ""))
        if keep_text(p):
            normal_prompts.append(p)

# Deduplicate while preserving order
seen = set()
normal_prompts = [p for p in normal_prompts if not (p in seen or seen.add(p))]

print(f"Normal prompts extracted: {len(normal_prompts)}")

# ---- 2) UNSAFE prompts from Arabic_Unsafe_BeaverTails ----
unsafe_ds = load_dataset("zbeeb/Arabic_Unsafe_BeaverTails", split="train")

unsafe_prompts = []
for ex in unsafe_ds:
    p = clean_text(ex.get("prompt_ar", ""))
    if keep_text(p):
        unsafe_prompts.append(p)

seen = set()
unsafe_prompts = [p for p in unsafe_prompts if not (p in seen or seen.add(p))]

print(f"Unsafe prompts extracted: {len(unsafe_prompts)}")

# ---- 3) Sample splits (balanced) ----
def make_splits(prompts, n_train, n_val):
    if len(prompts) < (n_train + n_val):
        raise ValueError(f"Not enough prompts: have {len(prompts)}, need {n_train+n_val}")
    idxs = list(range(len(prompts)))
    random.shuffle(idxs)
    train = [prompts[i] for i in idxs[:n_train]]
    val   = [prompts[i] for i in idxs[n_train:n_train+n_val]]
    return train, val

train_normal, val_normal = make_splits(normal_prompts, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS)
train_unsafe, val_unsafe = make_splits(unsafe_prompts, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS)

# ---- 4) Save in a simple JSONL schema ----
# Keep schema minimal: {"prompt": "...", "label": "..."}
save_jsonl("train_normal.jsonl", [{"prompt": p, "label": "normal"} for p in train_normal])
save_jsonl("val_normal.jsonl",   [{"prompt": p, "label": "normal"} for p in val_normal])
save_jsonl("train_unsafe.jsonl", [{"prompt": p, "label": "unsafe"} for p in train_unsafe])
save_jsonl("val_unsafe.jsonl",   [{"prompt": p, "label": "unsafe"} for p in val_unsafe])

print("Wrote: train_normal.jsonl, val_normal.jsonl, train_unsafe.jsonl, val_unsafe.jsonl")
