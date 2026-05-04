import random
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from nanofm.data.multimodal.masking import SimpleMultimodalMasking


OUTPUT_DIR = Path(__file__).resolve().parent / "visual_outputs"
OUTPUT_PATH = OUTPUT_DIR / "masking_comparison.png"
SPAN_OUTPUT_PATH = OUTPUT_DIR / "span_masking_tokens_comparison.png"
BLOCK_OUTPUT_PATH = OUTPUT_DIR / "block_masking_tokens_comparison.png"


def build_masker(**kwargs):
    defaults = dict(
        modalities=["tok_rgb@256", "scene_desc"],
        vocab_sizes=[64, 64],
        max_seq_lens=[256, 32],
        input_alphas=[1.0, 1.0],
        target_alphas=[1.0, 1.0],
        input_tokens_range=(128, 128),
        target_tokens_range=(64, 64),
        image_modalities=["tok_rgb@256"],
        sequence_modalities=["scene_desc"],
        image_token_grid_sizes={"tok_rgb@256": (16, 16)},
    )
    defaults.update(kwargs)
    return SimpleMultimodalMasking(**defaults)


def draw_label(draw, xy, text):
    draw.text(xy, text, fill=(28, 33, 40), font=ImageFont.load_default())


def draw_image_grid(draw, origin, masked_positions, label, cell=9, rows=16, cols=16):
    x0, y0 = origin
    masked = set(masked_positions)
    draw_label(draw, (x0, y0 - 16), label)
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            fill = (231, 235, 240)
            if idx in masked:
                fill = (236, 103, 90)
            x = x0 + col * cell
            y = y0 + row * cell
            draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=fill)


def draw_text_strip(draw, origin, masked_positions, label, cell=14, n_tokens=32):
    x0, y0 = origin
    masked = set(masked_positions)
    draw_label(draw, (x0, y0 - 16), label)
    for idx in range(n_tokens):
        fill = (231, 235, 240)
        if idx in masked:
            fill = (236, 103, 90)
        x = x0 + idx * cell
        draw.rectangle((x, y0, x + cell - 3, y0 + 22), fill=fill)
        if idx % 4 == 0:
            draw.text((x + 1, y0 + 25), str(idx), fill=(85, 92, 105), font=ImageFont.load_default())


def draw_token_cells(draw, origin, masked_positions, label, n_tokens=24, cell_w=34, cell_h=30):
    x0, y0 = origin
    masked = set(masked_positions)
    draw_label(draw, (x0, y0 - 18), label)
    font = ImageFont.load_default()
    for idx in range(n_tokens):
        x = x0 + idx * (cell_w + 2)
        fill = (231, 235, 240)
        text_fill = (50, 56, 66)
        if idx in masked:
            fill = (236, 103, 90)
            text_fill = (255, 255, 255)
        draw.rectangle((x, y0, x + cell_w, y0 + cell_h), fill=fill, outline=(255, 255, 255))
        draw.text((x + 7, y0 + 9), f"t{idx:02d}", fill=text_fill, font=font)


def draw_numbered_grid(draw, origin, masked_positions, label, rows=8, cols=8, cell=38):
    x0, y0 = origin
    masked = set(masked_positions)
    draw_label(draw, (x0, y0 - 18), label)
    font = ImageFont.load_default()
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            fill = (231, 235, 240)
            text_fill = (50, 56, 66)
            if idx in masked:
                fill = (236, 103, 90)
                text_fill = (255, 255, 255)
            x = x0 + col * cell
            y = y0 + row * cell
            draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=fill, outline=(255, 255, 255))
            draw.text((x + 10, y + 13), f"{idx:02d}", fill=text_fill, font=font)


class VisualMaskingTest(unittest.TestCase):
    def test_visual_masking_comparison_is_generated(self):
        random.seed(7)
        torch.manual_seed(7)

        random_masker = build_masker(masking_strategy="random")
        structured_masker = build_masker(masking_strategy="structured", block_min_size=4, block_max_fraction=0.5)
        structured_masker._sample_geometric_span_length = lambda: 10

        image_random = random_masker.sample_random_positions(256, 64).tolist()
        image_block = structured_masker.sample_block_positions(
            256,
            64,
            modality="tok_rgb@256",
            grid_size=(16, 16),
        ).tolist()
        text_random = random_masker.sample_random_positions(32, 10).tolist()
        text_span = structured_masker.sample_span_positions(32, 10).tolist()

        image = Image.new("RGB", (560, 430), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw_label(draw, (24, 20), "Visual masking sanity check")
        draw_label(draw, (24, 38), "Grey = visible/original, red = masked target tokens")

        draw_image_grid(draw, (24, 86), [], "Image without masking")
        draw_image_grid(draw, (206, 86), image_random, "Image random masking")
        draw_image_grid(draw, (388, 86), image_block, "Image block masking")

        draw_text_strip(draw, (24, 292), [], "Text without masking")
        draw_text_strip(draw, (24, 342), text_random, "Text random masking")
        draw_text_strip(draw, (24, 392), text_span, "Text span masking")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        image.save(OUTPUT_PATH)

        self.assertTrue(OUTPUT_PATH.exists())
        self.assertGreater(OUTPUT_PATH.stat().st_size, 0)

    def test_span_masking_token_visual_is_generated(self):
        random.seed(11)
        torch.manual_seed(11)

        span_masker = build_masker(masking_strategy="span", span_geometric_p=0.2)
        span_masker._sample_geometric_span_length = lambda: 8
        span_positions = span_masker.sample_span_positions(num_tokens=24, num_positions=8).tolist()

        image = Image.new("RGB", (920, 180), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw_label(draw, (24, 20), "Span masking on text tokens")
        draw_label(draw, (24, 38), "Grey = visible/original. Red = masked decoder targets.")
        draw_token_cells(draw, (24, 80), [], "Without span masking")
        draw_token_cells(draw, (24, 135), span_positions, "With span masking")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        image.save(SPAN_OUTPUT_PATH)

        self.assertTrue(SPAN_OUTPUT_PATH.exists())
        self.assertGreater(SPAN_OUTPUT_PATH.stat().st_size, 0)

    def test_block_masking_token_visual_is_generated(self):
        random.seed(17)
        torch.manual_seed(17)

        block_masker = build_masker(
            masking_strategy="block",
            image_token_grid_sizes={"tok_rgb@256": (8, 8)},
            block_min_size=4,
            block_max_fraction=0.5,
        )
        block_positions = block_masker.sample_block_positions(
            num_tokens=64,
            num_positions=16,
            modality="tok_rgb@256",
            grid_size=(8, 8),
        ).tolist()

        image = Image.new("RGB", (720, 410), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw_label(draw, (24, 20), "Block masking on image-token grid")
        draw_label(draw, (24, 38), "Grey = visible/original. Red = masked decoder targets.")
        draw_numbered_grid(draw, (24, 82), [], "Without block masking")
        draw_numbered_grid(draw, (384, 82), block_positions, "With block masking")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        image.save(BLOCK_OUTPUT_PATH)

        self.assertTrue(BLOCK_OUTPUT_PATH.exists())
        self.assertGreater(BLOCK_OUTPUT_PATH.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
