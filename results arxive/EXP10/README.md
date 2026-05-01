# EXP10 — Inference-time safety vs fine-tuning

## Question

Across the project's five Arabic models, where does safety come from cheaper —
training (SFT + DPO), inference-time guarding, or the combination?

## Setup

Five models (`allam7b`, `fanar9b`, `llama8b`, `qwen3b`, `qwen7b`),
four configurations each:

| Variant | Description |
|---|---|
| `base` | the raw `*-Instruct` model, no training, no guard |
| `base + guard` | raw model + Qwen3Guard-Gen-4B wrapping |
| `aligned` | the model's best safety-tuned checkpoint (from EXP6 or EXP9) |
| `aligned + guard` | aligned model + Qwen3Guard-Gen-4B wrapping |

For each guarded variant, three guard modes are tested:

- `prompt` — guard inspects user prompt only; refuses early if unsafe.
- `response` — guard inspects (prompt, response); substitutes refusal if unsafe.
- `both` — both checks active.

So per model: 1 base + 3 base+guard + 1 aligned + 3 aligned+guard = 8 cells.
Across 5 models: **40 cells total**, of which 10 are reused from EXP2 and EXP9
and 30 are new runs.

## Evaluation

AraSafe (12,077 prompts), judged by Qwen3Guard-Gen-4B. Same protocol as EXP3,
EXP6, EXP9 — the underlying judge code is `DPO/Eval/eval.py`. The new wrapper
that adds the guard intercept is `Inference-time-safety/eval_guarded.py`.

## Files

```
EXP10/
├── README.md                 (this file)
├── results_table.md          (per-model table, written by aggregate_exp10.py)
├── grid_summary.csv          (40 rows, written by aggregate_exp10.py)
├── plots/
│   ├── tradeoff_scatter.png
│   ├── latency_vs_safety.png
│   └── intervention_breakdown.png
├── base/
│   ├── <short>__base__summary.json                  (5 files, copied from EXP2)
│   └── <short>__base_guard_{prompt,response,both}__summary.json (15 new runs)
└── aligned/
    ├── <short>__aligned__summary.json               (5 files, copied from EXP9)
    └── <short>__aligned_guard_{prompt,response,both}__summary.json (15 new runs)
```

`<short>` ∈ {`allam7b`, `fanar9b`, `llama8b`, `qwen3b`, `qwen7b`}.

## Schema

Each `summary.json` extends the project's standard schema with these
inference-time fields:

```
"base_model":  "...",
"guard_model": "...",
"guard_mode":  "none|prompt|response|both",
"intervention_count":              int,
"intervention_rate":               float (0..1),
"intervention_count_prompt_block": int,
"intervention_count_pair_block":   int,
"mean_base_gen_latency_ms":        float,
"guard_intercept_total_s":         float,
"judge_total_s":                   float
```

All existing fields (`safe_refusal_rate`, `unsafe_refusal_rate`, etc.) are
preserved unchanged, so prior plotting scripts still load these files.
