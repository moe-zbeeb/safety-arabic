# Arabic Boundary-Set (MSA)

This folder contains the canonical Modern Standard Arabic (MSA) version of the Arabic boundary-set used for the Arabic boundary-set experiment.

The boundary-set was created to evaluate safety behavior near the safe/unsafe decision boundary in Arabic, with a particular focus on over-refusal. It is designed to test whether a model can avoid refusing safe prompts that contain risky vocabulary or sensitive contexts, while still refusing genuinely unsafe prompts phrased in similar language.

## Intended Use

This dataset is an evaluation set, not a training corpus.

It is intended to complement the project’s existing safety evaluation setup by adding a safe-but-sensitive Arabic benchmark that more directly measures false positives and over-refusal.

The intended evaluation setup is to report model behavior on:

- AraSafe unsafe split
- normal safe split
- this new boundary safe split

Although the full constructed dataset contains both `Safe` and `Unsafe` examples, the highest-priority use case for this benchmark is evaluating safe prompts that are likely to trigger unnecessary refusal.

## Variants

This folder contains the canonical MSA source version of the dataset.

Additional language variants derived from this MSA set are stored separately in `boundary_set_2/`. These include dialectal and script-converted versions of the same prompts. The MSA file in this folder should be treated as the source reference version.

## Files In This Folder

| File | Variant | Description |
|------|---------|-------------|
| `Arabic boundary-set_MSA.json` | Modern Standard Arabic | Canonical source boundary-set in MSA |
| `Arabic boundary-set.csv` | Modern Standard Arabic | CSV version of the same MSA dataset |
| `Arabic boundary-set.json` | Modern Standard Arabic | JSON version of the same MSA dataset |

## File Format

The dataset is provided in JSON and CSV formats.

Each example contains the following fields:

- `Safe/Unsafe`: the ground-truth label, either `Safe` or `Unsafe`
- `Category`: the boundary-case category
- `Prompt`: the Arabic prompt text in MSA

Example schema:

```json
{
  "Safe/Unsafe": "Safe",
  "Category": "safety-policy discussions",
  "Prompt": "..."
}
```

## Data Composition

The MSA boundary-set contains `480` prompts in total.

### Overall Label Distribution

- `270` Safe prompts
- `210` Unsafe prompts

### Number of Categories

The dataset contains `6` categories.

### Categories

The 6 categories are:

1. `safety-policy discussions`
2. `Quoted Content`
3. `safe educational questions`
4. `reporting / summarization of unsafe text`
5. `medical / legal / historical contexts with risky vocabulary`
6. `neutral prompts containing keywords that often trigger refusal`

### Per-Category Label Distribution

Each category contains exactly `80` prompts, with the same label split in every category:

- `45` Safe prompts
- `35` Unsafe prompts

Detailed breakdown:

| Category | Safe | Unsafe | Total |
|---|---:|---:|---:|
| safety-policy discussions | 45 | 35 | 80 |
| Quoted Content | 45 | 35 | 80 |
| safe educational questions | 45 | 35 | 80 |
| reporting / summarization of unsafe text | 45 | 35 | 80 |
| medical / legal / historical contexts with risky vocabulary | 45 | 35 | 80 |
| neutral prompts containing keywords that often trigger refusal | 45 | 35 | 80 |

## Category Descriptions

- `safety-policy discussions`
  Prompts about AI safety policy, moderation, alignment, governance, or responsible disclosure that may mention harmful topics without requesting harmful assistance.

- `Quoted Content`
  Prompts that quote unsafe-seeming text for translation, analysis, explanation, or discussion rather than for execution.

- `safe educational questions`
  Benign educational or academic questions involving sensitive domains where the user intent remains non-harmful.

- `reporting / summarization of unsafe text`
  Prompts asking to summarize, analyze, rewrite, classify, or discuss unsafe material in a non-endorsing context.

- `medical / legal / historical contexts with risky vocabulary`
  Prompts grounded in legitimate medical, legal, historical, or documentary contexts that contain terms often associated with refusal behavior.

- `neutral prompts containing keywords that often trigger refusal`
  Benign prompts that include high-risk keywords but do not request harmful action.

## Labeling Notes

The authoritative label is the `Safe/Unsafe` field.

Do not infer the label from the category alone. Categories group prompt types, but each category contains both `Safe` and `Unsafe` examples.

## Note on Final Size

The planning discussion for this experiment referenced a target of `250` safe prompts and `200` unsafe prompts. The final MSA dataset released here contains `270` safe prompts and `210` unsafe prompts, for a total of `480` examples.

## Evaluation Goal

This dataset is intended to expose two important failure modes:

- false positives: refusing prompts that are actually safe
- false negatives: answering prompts that are actually unsafe

Because this is a boundary-focused dataset, false positives on safe prompts are especially important when assessing over-refusal.

## Recommended Metrics

When evaluating models on this dataset, report at least:

- Safe Refusal %
- Unsafe Refusal %
- overall accuracy
- false positive rate on `Safe` prompts
- false negative rate on `Unsafe` prompts

For consistency with earlier project reporting, results can also be presented in the same style as EXP2 tables, especially with:

- `Safe Refusal %`
- `Unsafe Refusal %`

## Project Context

This dataset was created as part of the Arabic boundary-set experiment to address a key bottleneck observed in earlier model results: over-refusal on safe but sensitive Arabic prompts. The MSA version in this folder serves as the reference version from which the other variants are derived.
