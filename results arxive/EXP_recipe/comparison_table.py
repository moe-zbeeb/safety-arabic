import json
from pathlib import Path

rows = [
    {
        "model":    "Mistral-7B-Instruct-v0.3",
        "recipe":   "80/20 interleaved",
        "regime":   "Miscalibrated",
        "safe_old":   24.23,
        "unsafe_old": 52.31,
        "safe_new":   47.21,
        "unsafe_new": 92.34,
    },
    {
        "model":    "Hala-9B",
        "recipe":   "95/5 beneficial-first",
        "regime":   "Safety-prior strong",
        "safe_old":   9.34,
        "unsafe_old": 84.37,
        "safe_new":   21.77,
        "unsafe_new": 94.42,
    },
    {
        "model":    "jais-6p7b",
        "recipe":   "95/5 refusal-first",
        "regime":   "Under-protective",
        "safe_old":   10.13,
        "unsafe_old": 57.41,
        "safe_new":   16.66,
        "unsafe_new": 65.87,
    },
    {
        "model":    "Qwen3-8B",
        "recipe":   "95/5 refusal-first",
        "regime":   "Under-protective",
        "safe_old":   8.00,
        "unsafe_old": 61.96,
        "safe_new":   11.55,
        "unsafe_new": 83.09,
    },
]

# Build table
col_model   = max(len(r["model"])  for r in rows)
col_recipe  = max(len(r["recipe"]) for r in rows)
col_regime  = max(len(r["regime"]) for r in rows)

header = (
    f"{'Model':<{col_model}}  {'Recipe':<{col_recipe}}  {'Regime':<{col_regime}}"
    f"  {'Safe(base)':>10}  {'Safe(sft)':>9}  {'Unsafe(base)':>12}  {'Unsafe(sft)':>11}"
)
sep = "-" * len(header)

lines = [sep, header, sep]
for r in rows:
    lines.append(
        f"{r['model']:<{col_model}}  {r['recipe']:<{col_recipe}}  {r['regime']:<{col_regime}}"
        f"  {r['safe_old']:>9.2f}%  {r['safe_new']:>8.2f}%  {r['unsafe_old']:>11.2f}%  {r['unsafe_new']:>10.2f}%"
    )
lines.append(sep)
lines.append("")
lines.append("Safe refusal = over-refusal rate (lower is better)")
lines.append("Unsafe refusal = correct refusal rate (higher is better)")

table = "\n".join(lines)
print(table)

out = Path(__file__).parent / "comparison_table.txt"
out.write_text(table, encoding="utf-8")
print(f"\nsaved → {out}")

# Also save CSV
csv_path = Path(__file__).parent / "comparison_table.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("Model,Recipe,Regime,Safe(base)%,Safe(sft)%,Unsafe(base)%,Unsafe(sft)%\n")
    for r in rows:
        f.write(f"{r['model']},{r['recipe']},{r['regime']},{r['safe_old']:.2f},{r['safe_new']:.2f},{r['unsafe_old']:.2f},{r['unsafe_new']:.2f}\n")
print(f"saved → {csv_path}")
