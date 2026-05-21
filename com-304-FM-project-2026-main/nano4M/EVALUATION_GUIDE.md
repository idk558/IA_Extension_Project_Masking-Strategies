# nano4M Evaluation Guide

This document explains how to:

1. Generate `scene_desc` predictions from a trained `nano4M` checkpoint on Kuma
2. Score those predictions with automatic text metrics
3. Interpret the main scores for comparing different trainings

The workflow below is the one we used for the baseline random masking checkpoint.

## What This Evaluation Does

The model takes visual token modalities as input:

- `tok_rgb@256`
- `tok_depth@256`
- `tok_normal@256`

and generates the text modality:

- `scene_desc`

So the evaluation is:

- input: tokenized visual representation of the scene
- output: predicted text description of the scene
- comparison: predicted text vs reference text from the dataset

## Files Used

Main scripts:

- [generate_scene_desc_predictions.py](/Users/aymanbelbachir/IA_Extension_Project_Masking-Strategies/com-304-FM-project-2026-main/nano4M/generate_scene_desc_predictions.py:1)
- [score_scene_desc_predictions.py](/Users/aymanbelbachir/IA_Extension_Project_Masking-Strategies/com-304-FM-project-2026-main/nano4M/score_scene_desc_predictions.py:1)

Typical checkpoint path on Kuma:

- `/home/belbachi/COM-304-FM/checkpoints/nano4M_baseline_random_masking/checkpoint-final.pth`

Dataset path on Kuma:

- `/work/com-304/datasets/clevr_com_304/`

## Step 1: Connect to Kuma

```bash
ssh -X belbachi@kuma.hpc.epfl.ch
```

Then start or attach to a GPU session and activate the environment:

```bash
source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate fourm
```

## Step 2: Go to the Project Folder

```bash
cd /home/belbachi/COM-304-FM/extension/IA_Extension_Project_Masking-Strategies/com-304-FM-project-2026-main/nano4M
```

## Step 3: Generate Predictions

Example with 200 validation examples:

```bash
python generate_scene_desc_predictions.py \
  --checkpoint /home/belbachi/COM-304-FM/checkpoints/nano4M_baseline_random_masking/checkpoint-final.pth \
  --root-dir /work/com-304/datasets/clevr_com_304/ \
  --split val \
  --start-idx 0 \
  --num-samples 200 \
  --device cuda \
  --temperature 0 \
  --output-json baseline_scene_desc_predictions_200.json
```

This creates:

- `baseline_scene_desc_predictions_200.json`

Each item contains:

- `dataset_index`
- `input_modalities`
- `reference_text`
- `predicted_text`
- `reference_token_count`
- `predicted_token_count`

## Step 4: Score the Predictions

```bash
python score_scene_desc_predictions.py \
  --input-json baseline_scene_desc_predictions_200.json \
  --output-json baseline_scene_desc_predictions_200_scored.json
```

This creates:

- `baseline_scene_desc_predictions_200_scored.json`

and prints a summary with average scores.

## Example Output

Example summary:

```json
{
  "num_examples": 200,
  "averages": {
    "bleu_1_avg": 86.905864415499,
    "bleu_2_avg": 81.03707821291376,
    "bleu_3_avg": 74.7997968722313,
    "bleu_4_avg": 69.59108294982825,
    "cider_avg": 37.0729842545967,
    "dataset_index_avg": 99.5,
    "exact_match_avg": 4.0,
    "predicted_token_count_avg": 116.42,
    "reference_token_count_avg": 119.02,
    "rouge_l_avg": 84.31299196093454
  }
}
```

## What The Scores Mean

### Main scores to keep

These are the most useful metrics for comparing different trainings:

- `cider_avg`
- `bleu_4_avg`
- `rouge_l_avg`
- `exact_match_avg`

### BLEU scores

- `bleu_1_avg`
  - compares single words
  - high score means many individual words match

- `bleu_2_avg`
  - compares 2-word sequences
  - stricter than BLEU-1

- `bleu_3_avg`
  - compares 3-word sequences
  - captures more local structure

- `bleu_4_avg`
  - compares 4-word sequences
  - one of the strictest BLEU variants
  - useful for checking whether the full structured description matches well

### ROUGE-L

- `rouge_l_avg`
  - measures global similarity using the longest common subsequence
  - useful for checking whether the prediction follows the overall structure of the reference

### CIDEr

- `cider_avg`
  - this is the CIDEr-style score used by the scoring script
  - it rewards matching informative n-grams
  - it is often more meaningful than BLEU alone for description generation
  - in this project, it is especially useful for comparing different trainings against each other

Note:
- the script uses a lightweight CIDEr-style implementation designed for relative model comparison
- it is good for comparing multiple checkpoints or masking strategies consistently

### Exact Match

- `exact_match_avg`
  - percentage of examples where the predicted description exactly matches the reference text
  - example: `4.0` means about 4% exact matches

### Token count averages

- `predicted_token_count_avg`
  - average length of predicted descriptions

- `reference_token_count_avg`
  - average length of reference descriptions

These are useful as diagnostics, but not as primary quality metrics.

### dataset_index_avg

- `dataset_index_avg`
  - not a quality metric
  - just the average index of the evaluated subset
  - ignore it in the final comparison

## Recommended Comparison Table

When comparing several trainings, keep a table like this:

| Training | # examples | CIDEr | BLEU-4 | ROUGE-L | Exact Match |
|---|---:|---:|---:|---:|---:|
| Baseline random masking | 200 | 37.07 | 69.59 | 84.31 | 4.00 |
| Structured masking | 200 | ... | ... | ... | ... |
| Mixed masking | 200 | ... | ... | ... | ... |

## Recommended Interpretation

Use this logic:

- `CIDEr` as the main score
- `BLEU-4` as a strict complementary score
- `ROUGE-L` as a structural similarity score
- `Exact Match` as a very strict correctness score

If one training has:

- higher `CIDEr`
- higher `BLEU-4`
- similar or higher `ROUGE-L`
- higher `Exact Match`

then it is usually the stronger training for `scene_desc` generation.

## Optional Next Step: LLM Judge

After automatic scoring, you can also run an LLM judge on:

- `reference_text`
- `predicted_text`

This gives a more semantic or human-readable evaluation, but the automatic metrics above should remain the main quantitative comparison.
