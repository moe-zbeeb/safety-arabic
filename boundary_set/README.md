# Arabic Boundary-Set (MSA)

This folder contains the canonical Modern Standard Arabic (MSA) version of the Arabic boundary-set used for the Arabic boundary-set experiment.

The boundary-set was created to evaluate safety behavior near the safe/unsafe decision boundary in Arabic, with a particular focus on over-refusal. It is designed to test whether a model can avoid refusing safe prompts that contain risky vocabulary or sensitive contexts, while still refusing genuinely unsafe prompts phrased in similar language.

## Variants

This folder contains the canonical MSA source version of the dataset.

Additional language variants derived from this MSA set are stored separately in `boundary_set_2/`. These include dialectal and script-converted versions of the same prompts. The MSA file in this folder should be treated as the source reference version.

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