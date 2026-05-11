# -*- coding: utf-8 -*-
"""
Convert Arabic Boundary-Set JSON to Arabizi via the OpenAI API.

Reads a JSON array of objects with at least a `Prompt` field, replaces
the Prompt with its Arabizi version, and writes the result to a new JSON
file. All other fields in each row are preserved.
"""

import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm.auto import tqdm

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_NAME = "gpt-4o-mini"

INPUT_JSON_PATH  = "/Users/mohamad22/Desktop/Safety-Arabic/boundary_set/Arabic Boundary-set_MSA.json"
OUTPUT_JSON_PATH = "/Users/mohamad22/Desktop/Safety-Arabic/boundary_set/Arabic-boundary-set-arabizi.json"

REQUEST_DELAY_SECONDS = 0.4
MAX_RETRIES = 3

# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = '''Only convert Arabic text into Arabizi.
Do not answer the prompts. Do not analyse meaning.
Return only the Arabizi version of the input text.
Preserve the original meaning exactly.
Do not translate to English.
Do not summarize or explain.
Keep punctuation and formatting when possible, especially quotations.'''


def to_arabizi(text: str) -> str:
    """Send one prompt to the model and return the Arabizi version."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 * attempt)
    raise last_error


# --------------------------------------------------------------------------
# Load input
# --------------------------------------------------------------------------
json_path = Path(INPUT_JSON_PATH)
if not json_path.exists():
    raise FileNotFoundError(f"JSON file not found: {json_path}")

with json_path.open("r", encoding="utf-8") as f:
    rows = json.load(f)

if not isinstance(rows, list):
    raise ValueError("JSON input must be a top-level array.")

print(f"Loaded {len(rows)} rows from {json_path.name}")

# --------------------------------------------------------------------------
# Convert
# --------------------------------------------------------------------------
converted_rows = []
for row in tqdm(rows, desc="Converting"):
    prompt = str(row.get("Prompt", "") or "").strip()
    new_row = dict(row)  # preserve every existing field
    if prompt:
        new_row["Prompt"] = to_arabizi(prompt)
        time.sleep(REQUEST_DELAY_SECONDS)
    else:
        new_row["Prompt"] = ""
    converted_rows.append(new_row)

print(f"Converted {len(converted_rows)} prompts")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
output_path = Path(OUTPUT_JSON_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(converted_rows, f, ensure_ascii=False, indent=2)

print(f"Saved JSON to {output_path}")
