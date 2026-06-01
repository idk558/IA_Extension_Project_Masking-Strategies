# nano4M Text-to-Image Evaluation Guide

This document explains how to:

1. Generate CLEVR images from `scene_desc` text using a trained `nano4M` checkpoint
2. Score the generated images with the CLEVR verifier
3. Interpret the main scores for comparing different trainings

The workflow below mirrors the image-to-text evaluation guide, but evaluates the opposite direction:

- input: `scene_desc`
- output: generated `tok_rgb@256` image
- comparison: generated image vs reference scene description

The recommended evaluation size is the same as the image-to-text guide:

- `split`: `val`
- `num_samples`: `200`

## What This Evaluation Does

The model takes the text modality as input:

- `scene_desc`

and generates the image token modality:

- `tok_rgb@256`

The generated image tokens are decoded into an RGB image with the Cosmos image tokenizer. The image is then evaluated with:

- Grounding DINO for object-level CLEVR fidelity
- CLIP for global text-image alignment

So the evaluation is:

- input: structured text description of the scene
- output: generated RGB image
- comparison: generated image vs requested scene composition

## Files Used

Main scripts:

- `scripts/eval_clevr_fidelity.py`
- `nanofm/eval/clevr_verifier.py`

Dataset path:

- `/work/com-304/datasets/clevr_com_304/`

Cosmos tokenizer path:

- `/home/nalaoui/cosmos-tokenizer/Cosmos-0.1-Tokenizer-DI16x16`

Recommended checkpoint folder:

- `/home/nalaoui/COM-304-FM/trained_models/`

Example baseline checkpoint:

- `/home/nalaoui/COM-304-FM/trained_models/nano4M_baseline_random_masking/checkpoint-final.pth`

## Step 1: Check the Model Checkpoints

List all available trained model checkpoints:

```bash
find /home/nalaoui/COM-304-FM/trained_models -name "*.safetensors" -o -name "*.pth"
```

For this verifier, prefer `checkpoint-final.pth` when available.

Expected model folders:

- `nano4M_baseline_random_masking`
- `V1_text-span_image-random`
- `V2_textrandom_imageblock`
- `V3_text_span_image_block`
- `V4_text-span-random_image-block-random`

## Step 2: Run a Small Smoke Test

Before evaluating 200 samples, run 5 samples to make sure the checkpoint, GPU, dataset, tokenizer, Grounding DINO, and CLIP all work.

Example for the baseline checkpoint:

```bash
PYTHONPATH=. python scripts/eval_clevr_fidelity.py \
  --checkpoint /home/nalaoui/COM-304-FM/trained_models/nano4M_baseline_random_masking/checkpoint-final.pth \
  --root-dir /work/com-304/datasets/clevr_com_304 \
  --split val \
  --num-samples 5 \
  --device cuda \
  --detector-device cuda \
  --clip-device cuda \
  --output-dir outputs/eval_text_to_image/baseline_5 \
  --image-tokenizer-dir /home/nalaoui/cosmos-tokenizer/Cosmos-0.1-Tokenizer-DI16x16
```

This creates:

- `outputs/eval_text_to_image/baseline_5/report.json`
- `outputs/eval_text_to_image/baseline_5/visual_report.html`
- `outputs/eval_text_to_image/baseline_5/candidate/*.png`
- `outputs/eval_text_to_image/baseline_5/candidate/*_detections.png`

Each item contains:

- `idx`
- `caption`
- `image_path`
- `annotated_image_path`
- `score`
- `clip_score`
- `category_score`
- `hallucination_penalty`
- `expected`
- `detected`
- `per_category_breakdown`

## Step 3: Run the Full 200-Sample Evaluation

This is the text-to-image equivalent of the image-to-text guide's 200 validation examples.

Example for the baseline checkpoint:

```bash
PYTHONPATH=. python scripts/eval_clevr_fidelity.py \
  --checkpoint /home/nalaoui/COM-304-FM/trained_models/nano4M_baseline_random_masking/checkpoint-final.pth \
  --root-dir /work/com-304/datasets/clevr_com_304 \
  --split val \
  --num-samples 200 \
  --device cuda \
  --detector-device cuda \
  --clip-device cuda \
  --output-dir outputs/eval_text_to_image/baseline_200 \
  --image-tokenizer-dir /home/nalaoui/cosmos-tokenizer/Cosmos-0.1-Tokenizer-DI16x16
```

This creates:

- `outputs/eval_text_to_image/baseline_200/report.json`
- `outputs/eval_text_to_image/baseline_200/visual_report.html`

The terminal prints a summary similar to:

```json
[
  {
    "label": "candidate",
    "checkpoint": "/home/nalaoui/COM-304-FM/trained_models/nano4M_baseline_random_masking/checkpoint-final.pth",
    "mean_score": 0.56,
    "std_score": 0.27,
    "mean_clip_score": 0.63,
    "std_clip_score": 0.02,
    "worst_score": 0.0,
    "best_score": 1.0
  }
]
```

## Step 4: Evaluate the Other Trainings

Use the same command for every model. Only change:

- `--checkpoint`
- `--output-dir`

### V1: Text Span, Image Random

```bash
PYTHONPATH=. python scripts/eval_clevr_fidelity.py \
  --checkpoint /home/nalaoui/COM-304-FM/trained_models/V1_text-span_image-random/checkpoint-final.pth \
  --root-dir /work/com-304/datasets/clevr_com_304 \
  --split val \
  --num-samples 200 \
  --device cuda \
  --detector-device cuda \
  --clip-device cuda \
  --output-dir outputs/eval_text_to_image/v1_200 \
  --image-tokenizer-dir /home/nalaoui/cosmos-tokenizer/Cosmos-0.1-Tokenizer-DI16x16
```

### V2: Text Random, Image Block

```bash
PYTHONPATH=. python scripts/eval_clevr_fidelity.py \
  --checkpoint /home/nalaoui/COM-304-FM/trained_models/V2_textrandom_imageblock/checkpoint-final.pth \
  --root-dir /work/com-304/datasets/clevr_com_304 \
  --split val \
  --num-samples 200 \
  --device cuda \
  --detector-device cuda \
  --clip-device cuda \
  --output-dir outputs/eval_text_to_image/v2_200 \
  --image-tokenizer-dir /home/nalaoui/cosmos-tokenizer/Cosmos-0.1-Tokenizer-DI16x16
```

### V3: Text Span, Image Block

```bash
PYTHONPATH=. python scripts/eval_clevr_fidelity.py \
  --checkpoint /home/nalaoui/COM-304-FM/trained_models/V3_text_span_image_block/checkpoint-final.pth \
  --root-dir /work/com-304/datasets/clevr_com_304 \
  --split val \
  --num-samples 200 \
  --device cuda \
  --detector-device cuda \
  --clip-device cuda \
  --output-dir outputs/eval_text_to_image/v3_200 \
  --image-tokenizer-dir /home/nalaoui/cosmos-tokenizer/Cosmos-0.1-Tokenizer-DI16x16
```

### V4: Text Span-Random, Image Block-Random

```bash
PYTHONPATH=. python scripts/eval_clevr_fidelity.py \
  --checkpoint /home/nalaoui/COM-304-FM/trained_models/V4_text-span-random_image-block-random/checkpoint-final.pth \
  --root-dir /work/com-304/datasets/clevr_com_304 \
  --split val \
  --num-samples 200 \
  --device cuda \
  --detector-device cuda \
  --clip-device cuda \
  --output-dir outputs/eval_text_to_image/v4_200 \
  --image-tokenizer-dir /home/nalaoui/cosmos-tokenizer/Cosmos-0.1-Tokenizer-DI16x16
```

If a folder does not contain `checkpoint-final.pth`, find the exact checkpoint path with:

```bash
find /home/nalaoui/COM-304-FM/trained_models/<MODEL_FOLDER> -name "*.safetensors" -o -name "*.pth"
```

Then replace the checkpoint path in the command.

## Step 5: Open the Visual Reports

Each run writes a visual report inside its output folder. For example:

```bash
outputs/eval_text_to_image/baseline_200/visual_report.html
```

Open this file in a browser to inspect the generated images and detections.

The visual report shows:

- requested text
- generated image
- Grounding DINO detections
- CLEVR score
- CLIP score
- expected objects
- detected objects
- per-category score breakdown

## What The Scores Mean

### Main scores to keep

These are the most useful metrics for comparing different text-to-image trainings:

- `mean_score`
- `mean_clip_score`

### CLEVR fidelity score

- JSON field: `mean_score`
- Per-example field: `score`
- Range: `[0, 1]`

This is the object-level score computed from Grounding DINO detections.

For each expected `(shape, color)` category:

```text
n_expected[c] = count in caption
n_detected[c] = count from Grounding DINO
match[c] = min(n_expected[c], n_detected[c]) / max(n_expected[c], n_detected[c])
```

Then:

```text
category_score = mean(match[c] for expected categories)
hallucination_penalty = max(0, 1 - hallucinated / max(total_expected, 1))
final_score = category_score * hallucination_penalty
```

Higher is better.

### CLIP score

- JSON field: `mean_clip_score`
- Per-example field: `clip_score`
- Range: `[0, 1]`

This score measures global text-image alignment with CLIP. It is complementary to the CLEVR fidelity score:

- CLEVR score checks whether DINO detects the requested object categories.
- CLIP score checks whether the generated image and text are globally aligned.

Higher is better.

### Standard deviations

- `std_score`
- `std_clip_score`

These describe how variable the scores are across the 200 validation examples.

### Worst and best examples

- `worst_10`
- `best_10`

These are useful for qualitative inspection in the JSON report and visual report.

## Recommended Comparison Table

When comparing several trainings, keep a table like this:

| Training | # examples | CLEVR fidelity | CLIP score |
|---|---:|---:|---:|
| Baseline random masking | 200 | ... | ... |
| V1 text span, image random | 200 | ... | ... |
| V2 text random, image block | 200 | ... | ... |
| V3 text span, image block | 200 | ... | ... |
| V4 text span-random, image block-random | 200 | ... | ... |

## Recommended Interpretation

Use this logic:

- `mean_score` as the main CLEVR composition fidelity score
- `mean_clip_score` as a complementary global text-image alignment score
- visual reports to verify whether the automatic scores match human inspection

If one training has:

- higher `mean_score`
- similar or higher `mean_clip_score`
- fewer visually bad generated examples

then it is usually the stronger training for text-to-image CLEVR generation.

## Notes and Common Warnings

The following warnings are usually not blocking:

```text
Missing keys when loading ... ['dec_context_proj.bias']
Could not load the custom kernel for multi-scale deformable attention
```

The run is successful if it prints:

```text
Wrote visual report to ...
Wrote report to ...
```

If CUDA is not available, request or attach to a GPU session before running the evaluation.
