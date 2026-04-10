# Arabic Boundary-Set

This folder contains the final Arabic boundary-set used for the Arabic boundary-set experiment.

The dataset is designed to evaluate safety behavior near the safe/unsafe boundary in Arabic. Its purpose is to test whether a model can avoid over-refusing safe prompts that contain risky vocabulary or sensitive contexts, while still refusing genuinely unsafe prompts phrased in similar language.

## Intended Use

It is intended to complement the project’s existing safety evaluation setup by adding a boundary-focused split that more directly measures over-refusal in Arabic.

The intended evaluation setup is to report model behavior on:

- AraSafe unsafe split
- normal safe split
- this Arabic boundary-set

## File Format

The dataset is stored as a single JSON file containing a top-level array of examples.

Each example has the following fields:

- `Safe/Unsafe`: the ground-truth label, either `Safe` or `Unsafe`
- `Category`: the boundary-case category
- `Prompt`: the Arabic prompt text

## Data Composition

The Arabic boundary-set contains `480` prompts in total.

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

The final dataset released here contains `270` safe prompts and `210` unsafe prompts, for a total of `480` examples.
