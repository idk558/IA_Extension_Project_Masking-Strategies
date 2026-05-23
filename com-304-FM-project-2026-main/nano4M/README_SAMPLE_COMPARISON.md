# Sample Comparison Workflow

This README explains how to reproduce the single-sample comparison workflow for `nano4M`.

The goal is to:
- load one CLEVR sample
- generate the reference scene description
- generate predictions from `Baseline`, `V1`, `V2`, `V3`, and `V4`
- render a PNG from the reference scene description
- save a Markdown report with all outputs

## Files Used

Main scripts:
- [compare_one_sample_all_models.py](/Users/aymanbelbachir/IA_Extension_Project_Masking-Strategies/com-304-FM-project-2026-main/nano4M/compare_one_sample_all_models.py:1)
- [render_scene_desc_to_png.py](/Users/aymanbelbachir/IA_Extension_Project_Masking-Strategies/com-304-FM-project-2026-main/nano4M/render_scene_desc_to_png.py:1)

## Expected Checkpoints

Place these files in:

```bash
/home/belbachi/COM-304-FM/checkpoints/
```

Expected filenames:
- `checkpoint-final_baseline.pth`
- `checkpoint-final_V1.pth`
- `checkpoint-final_V2.pth`
- `checkpoint-final_V3.pth`
- `checkpoint-final_V4.pth`

## Pull Latest Code

On Kuma:

```bash
cd /home/belbachi/COM-304-FM/extension/IA_Extension_Project_Masking-Strategies
git stash push -u -m "temp before pull"
git pull --rebase origin main
```

## Run The Comparison

Go to `nano4M`:

```bash
cd /home/belbachi/COM-304-FM/extension/IA_Extension_Project_Masking-Strategies/com-304-FM-project-2026-main/nano4M
```

Launch the comparison:

```bash
python compare_one_sample_all_models.py \
  --root-dir /work/com-304/datasets/clevr_com_304/ \
  --split val \
  --index 42 \
  --device cuda \
  --skip-image-reconstruction \
  --baseline-checkpoint /home/belbachi/COM-304-FM/checkpoints/checkpoint-final_baseline.pth \
  --v1-checkpoint /home/belbachi/COM-304-FM/checkpoints/checkpoint-final_V1.pth \
  --v2-checkpoint /home/belbachi/COM-304-FM/checkpoints/checkpoint-final_V2.pth \
  --v3-checkpoint /home/belbachi/COM-304-FM/checkpoints/checkpoint-final_V3.pth \
  --v4-checkpoint /home/belbachi/COM-304-FM/checkpoints/checkpoint-final_V4.pth
```

## Output Files

The script creates:

```bash
sample_comparisons/sample_42/
```

Inside this folder you should get:
- `comparison.md`
- `comparison.json`
- `scene_desc_render.png`

If a real RGB image exists in the dataset and is available to the script, it may save `rgb.png` instead of relying only on the rendered scene description fallback.

## Read The Results

To read the report in the terminal:

```bash
cat sample_comparisons/sample_42/comparison.md
```

or:

```bash
sed -n '1,240p' sample_comparisons/sample_42/comparison.md
```

## Important: How To See The Image

To actually view the PNG image, use **JupyterLab** on Kuma.

Open the folder:

```bash
sample_comparisons/sample_42/
```

Then click on:
- `scene_desc_render.png`

or, if available:
- `rgb.png`

This is the easiest way to inspect the image visually.

## What The Report Contains

The Markdown report contains:
- the rendered scene image
- the reference scene description
- the prediction from each model
- for each version, a `Text To Render As Image` block

Those text blocks are designed so that someone else can copy the reference or predicted text and generate a corresponding image externally.

## Notes

- `--skip-image-reconstruction` is recommended here because the 4M token detokenization path may fail depending on the local environment and token compatibility.
- The fallback renderer still gives a useful PNG by drawing a CLEVR-like image directly from the reference `scene_desc`.
- For dataset-level evaluation over many samples, use the separate scoring scripts and aggregated metrics instead of this single-sample report.
