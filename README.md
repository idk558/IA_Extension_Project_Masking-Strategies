# IA_Extension_Project_Masking-Strategies

Week 10 implementation for alternate nano4M masking strategies:

- span masking for sequence/text modalities such as `scene_desc`
- block masking for image-token modalities such as `tok_rgb@256`, `tok_depth@256`, and `tok_normal@256`
- optional random/structured mixing through `structured_mask_probability`

The extracted base project lives in `com-304-FM-project-2026-main/`. The main implementation is in:

- `nano4M/nanofm/data/multimodal/masking.py`
- `nano4M/nanofm/data/multimodal/__init__.py`
- `nano4M/cfgs/nano4M/multiclevr_d6-6w512_structured_masking.yaml`

Run the local tests from the nano4M folder:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/tahrihassani/miniconda3/bin/python -m unittest discover -s tests
```

The visual masking sanity check is generated at:

`com-304-FM-project-2026-main/nano4M/tests/visual_outputs/masking_comparison.png`

## Evaluation process

For the evaluation workflow, go to the `nano4M` folder:

```bash
cd com-304-FM-project-2026-main/nano4M
```

Use the detailed guides already provided in the project:

- `TEXT_TO_IMAGE_EVALUATION_GUIDE.md` for text-to-image evaluation with CLEVR fidelity and CLIP scores.
- `README_SAMPLE_COMPARISON.md` for single-sample qualitative comparison across Baseline, V1, V2, V3, and V4.

Recommended order:

1. Run the text-to-image 5-sample smoke test from `TEXT_TO_IMAGE_EVALUATION_GUIDE.md`.
2. If it works, run the full 200-sample evaluation for each model.
3. Run the single-sample comparison from `README_SAMPLE_COMPARISON.md`.
4. Report `mean_score`, `mean_clip_score`, the generated `visual_report.html`, and the sample `comparison.md`.

Before running evaluations, check the dataset path, checkpoint paths, and Cosmos tokenizer path. Do not commit large evaluation outputs unless explicitly required.
