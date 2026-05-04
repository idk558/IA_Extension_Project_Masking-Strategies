import argparse
import io
import math
import re
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset
from nanofm.data.multimodal.masking import SimpleMultimodalMasking


MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"]
VOCAB_SIZES = [64000, 64000, 64000, 50304]
MAX_SEQ_LENS = [256, 256, 256, 256]
IMAGE_MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256"]
SEQUENCE_MODALITIES = ["scene_desc"]
IMAGE_TOKEN_GRID_SIZES = {
    "tok_rgb@256": (16, 16),
    "tok_depth@256": (16, 16),
    "tok_normal@256": (16, 16),
}
DEFAULT_DEMO_TEXT = (
    "A shiny red sphere is left of a small blue cube, while a large green cylinder "
    "sits behind two metal objects on the floor."
)


def build_dataset(args):
    return SimpleMultimodalDataset(
        root_dir=args.root_dir,
        split=args.split,
        modalities=MODALITIES,
        transforms=None,
        sample_from_k_augmentations=args.sample_from_k_augmentations,
        text_tokenizer_path=args.text_tokenizer_path,
        text_max_length=256,
    )


def build_masker(strategy):
    return SimpleMultimodalMasking(
        modalities=MODALITIES,
        vocab_sizes=VOCAB_SIZES,
        max_seq_lens=MAX_SEQ_LENS,
        input_alphas=[1.0, 1.0, 1.0, 1.0],
        target_alphas=[1.0, 1.0, 1.0, 1.0],
        input_tokens_range=(128, 128),
        target_tokens_range=(128, 128),
        overlap_vocab=True,
        overlap_posembs=True,
        masking_strategy=strategy,
        structured_mask_probability=1.0,
        image_modalities=IMAGE_MODALITIES,
        sequence_modalities=SEQUENCE_MODALITIES,
        image_token_grid_sizes=IMAGE_TOKEN_GRID_SIZES,
        span_geometric_p=0.2,
        block_min_size=1,
        block_max_fraction=0.5,
    )


def sample_text_positions(masker, strategy, n_tokens, n_mask_tokens):
    if strategy == "random":
        return masker.sample_random_positions(n_tokens, n_mask_tokens)
    return masker.sample_span_positions(n_tokens, n_mask_tokens)


def sample_image_positions(masker, strategy, modality, n_tokens, n_mask_tokens):
    if strategy == "random":
        return masker.sample_random_positions(n_tokens, n_mask_tokens)
    return masker.sample_block_positions(
        n_tokens,
        n_mask_tokens,
        modality=modality,
        grid_size=IMAGE_TOKEN_GRID_SIZES[modality],
    )


def text_to_display_tokens(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def print_demo_text_view(text, positions):
    tokens = text_to_display_tokens(text)
    masked = ["[MASK]" if idx in set(positions.tolist()) else token for idx, token in enumerate(tokens)]

    print("\n" + "=" * 80)
    print("HARDCODED TEXT - WITHOUT MASKING")
    print("=" * 80)
    print(" ".join(tokens))

    print("\n" + "=" * 80)
    print("HARDCODED TEXT - WITH SPAN MASKING")
    print("=" * 80)
    print(f"Masked token positions: {positions.tolist()}")
    print(" ".join(masked))


def print_text_view(dataset, original_tokens, positions):
    masked_tokens = original_tokens.clone()
    masked_tokens[positions] = 62

    print("\n" + "=" * 80)
    print("TEXT FROM DATASET - WITHOUT MASKING")
    print("=" * 80)
    print(dataset.text_tokenizer.decode(original_tokens))

    print("\n" + "=" * 80)
    print("TEXT FROM DATASET - WITH SPAN MASKING")
    print("=" * 80)
    print(f"Masked token positions: {positions.tolist()}")
    print(dataset.text_tokenizer.decode(masked_tokens))


def load_demo_image(image_url=None, image_size=256):
    if image_url:
        with urllib.request.urlopen(image_url) as response:
            image_bytes = response.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        torch.manual_seed(4)
        image = torch.rand(3, image_size, image_size)
        return image

    image = image.resize((image_size, image_size))
    image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    image_tensor = image_tensor.reshape(image_size, image_size, 3).permute(2, 0, 1).float() / 255
    return image_tensor


def apply_patch_mask(image_tensor, positions, grid_size=(16, 16), color=(1.0, 0.1, 0.05)):
    masked = image_tensor.clone()
    token_mask = torch.zeros(math.prod(grid_size), dtype=torch.bool)
    token_mask[positions.long()] = True
    pixel_mask = token_mask.reshape(1, 1, *grid_size).float()
    pixel_mask = F.interpolate(pixel_mask, image_tensor.shape[-2:], mode="nearest")[0, 0].bool()
    masked[:, pixel_mask] = torch.tensor(color).reshape(3, 1)
    return masked


def show_demo_image_view(original_image, masked_image, positions):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(original_image.permute(1, 2, 0))
    axes[0].set_title("Without block masking")
    axes[0].axis("off")
    axes[1].imshow(masked_image.permute(1, 2, 0))
    axes[1].set_title("With block masking")
    axes[1].axis("off")
    fig.suptitle(f"Masked token positions: {positions.tolist()}")
    plt.tight_layout()
    plt.show()


def format_grid(tokens, positions, grid_size):
    masked = set(positions.tolist())
    width = max(4, len(str(int(tokens.max().item()))) if tokens.numel() else 4)
    rows = []
    for row in range(grid_size[0]):
        cells = []
        for col in range(grid_size[1]):
            idx = row * grid_size[1] + col
            value = str(int(tokens[idx].item())).rjust(width)
            cells.append(f"[{value}]" if idx in masked else f" {value} ")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def print_image_token_view(tokens, positions, modality):
    grid_size = IMAGE_TOKEN_GRID_SIZES[modality]
    expected_tokens = math.prod(grid_size)
    if tokens.numel() != expected_tokens:
        raise ValueError(f"{modality} expected {expected_tokens} tokens, got {tokens.numel()}")

    print("\n" + "=" * 80)
    print(f"{modality} TOKENS FROM DATASET - WITHOUT MASKING")
    print("=" * 80)
    print(format_grid(tokens, torch.empty(0, dtype=torch.long), grid_size))

    print("\n" + "=" * 80)
    print(f"{modality} TOKENS FROM DATASET - WITH BLOCK MASKING")
    print("=" * 80)
    print(f"Masked token positions: {positions.tolist()}")
    print(format_grid(tokens, positions, grid_size))


def main():
    parser = argparse.ArgumentParser(description="Show masking on nano4M CLEVR samples using the original dataset loader.")
    parser.add_argument("--source", choices=["demo", "dataset"], default="demo")
    parser.add_argument("--root-dir", default="/work/com-304/datasets/clevr_com_304/")
    parser.add_argument("--split", default="train")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--sample-from-k-augmentations", type=int, default=10)
    parser.add_argument("--text-tokenizer-path", default="gpt2")
    parser.add_argument("--modality", choices=["scene_desc", "tok_rgb@256", "tok_depth@256", "tok_normal@256"], default="scene_desc")
    parser.add_argument("--strategy", choices=["random", "span", "block", "structured"], default="structured")
    parser.add_argument("--num-mask-tokens", type=int, default=None)
    parser.add_argument("--text", default=DEFAULT_DEMO_TEXT)
    parser.add_argument("--image-url", default=None)
    args = parser.parse_args()

    masker = build_masker(args.strategy)

    if args.source == "demo":
        if args.modality == "scene_desc":
            tokens = text_to_display_tokens(args.text)
            n_mask_tokens = min(args.num_mask_tokens or 8, len(tokens))
            positions = sample_text_positions(masker, args.strategy, len(tokens), n_mask_tokens)
            print_demo_text_view(args.text, positions)
            return

        image = load_demo_image(args.image_url)
        n_tokens = math.prod(IMAGE_TOKEN_GRID_SIZES[args.modality])
        n_mask_tokens = min(args.num_mask_tokens or 64, n_tokens)
        positions = sample_image_positions(masker, args.strategy, args.modality, n_tokens, n_mask_tokens)
        masked_image = apply_patch_mask(image, positions, IMAGE_TOKEN_GRID_SIZES[args.modality])
        show_demo_image_view(image, masked_image, positions)
        return

    if not Path(args.root_dir).exists():
        raise FileNotFoundError(
            f"Dataset root not found: {args.root_dir}\n"
            "Use the SCITAS/course environment, or pass --root-dir to your local clevr_com_304 copy."
        )

    dataset = build_dataset(args)
    sample = dataset[args.idx]

    if args.modality == "scene_desc":
        tokens = sample["scene_desc"].cpu()
        n_mask_tokens = min(args.num_mask_tokens or 32, tokens.numel())
        positions = sample_text_positions(masker, args.strategy, tokens.numel(), n_mask_tokens)
        print_text_view(dataset, tokens, positions)
        return

    tokens = sample[args.modality].cpu().flatten()
    n_mask_tokens = min(args.num_mask_tokens or 64, tokens.numel())
    positions = sample_image_positions(masker, args.strategy, args.modality, tokens.numel(), n_mask_tokens)
    print_image_token_view(tokens, positions, args.modality)


if __name__ == "__main__":
    main()
