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
