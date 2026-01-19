# Curriculum Learning Results

| Model | Stage | Safe Refusal % | Unsafe Refusal % |
|-------|-------|----------------|------------------|
| **allam** | Base | 19.36 | 86.20 |
| | 70ben_30ref | 28.11 | 94.74 |
| | 75ben_25ref | 27.74 | 94.42 |
| | 80ben_20ref | 25.82 | 94.66 |
| | 85ben_15ref | 24.12 | 94.34 |
| | 90ben_10ref | 24.25 | 94.82 |
| | *95ben_5ref* | *17.65* | *92.42* |
| **Fanar-1-9B** | Base | 28.78 | 74.72 |
| | *75ben_25ref* | *14.88* | *88.20* |
| | 80ben_20ref | 13.27 | 86.76 |
| | 85ben_15ref | 13.87 | 86.04 |
| | 90ben_10ref | 11.31 | 84.05 |
| | 95ben_5ref | 8.62 | 79.03 |
| **Meta-Llama-3-8B-Instruct** | Base | 10.55 | 79.59 |
| | 70ben_30ref | 35.87 | 95.45 |
| | 75ben_25ref | 37.17 | 95.77 |
| | 80ben_20ref | 34.85 | 96.17 |
| | 85ben_15ref | 32.75 | 95.22 |
| | 90ben_10ref | 31.65 | 94.74 |
| | *95ben_5ref* | *26.06* | *93.46* |
| **Qwen2.5-3B-Instruct** | Base | 9.52 | 71.13 |
| | 70ben_30ref | 35.30 | 96.33 |
| | 75ben_25ref | 33.69 | 95.69 |
| | 80ben_20ref | 32.42 | 95.77 |
| | 85ben_15ref | 30.61 | 95.53 |
| | 90ben_10ref | 28.97 | 94.42 |
| | *95ben_5ref* | *22.58* | *92.03* |
| **Qwen2.5-7B-Instruct** | Base | 8.26 | 69.86 |
| | 70ben_30ref | 30.58 | 96.01 |
| | 75ben_25ref | 29.39 | 95.77 |
| | 80ben_20ref | 27.76 | 95.77 |
| | 85ben_15ref | 28.26 | 95.93 |
| | 90ben_10ref | 25.09 | 94.98 |
| | *95ben_5ref* | *18.81* | *91.95* |

**Best combinations** (italicized): Selected based on lowest safe refusal rate while maintaining unsafe refusal > 90%. For Fanar-1-9B, 75ben_25ref was chosen as it has the highest unsafe refusal rate (88.20%) among all stages.
