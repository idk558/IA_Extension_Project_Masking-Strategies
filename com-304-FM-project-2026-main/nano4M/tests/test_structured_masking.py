import random
import unittest

import torch

from nanofm.data.multimodal.masking import SimpleMultimodalMasking
from nanofm.models.fourm import FourM


def build_masker(**kwargs):
    defaults = dict(
        modalities=["tok_rgb@256", "scene_desc"],
        vocab_sizes=[64, 64],
        max_seq_lens=[16, 16],
        input_alphas=[1.0, 1.0],
        target_alphas=[1.0, 1.0],
        input_tokens_range=(8, 8),
        target_tokens_range=(8, 8),
        image_modalities=["tok_rgb@256"],
        sequence_modalities=["scene_desc"],
        image_token_grid_sizes={"tok_rgb@256": (4, 4)},
    )
    defaults.update(kwargs)
    return SimpleMultimodalMasking(**defaults)


class StructuredMaskingTest(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)

    def test_span_masking_selects_contiguous_span(self):
        masker = build_masker(masking_strategy="span", span_geometric_p=0.2)
        masker._sample_geometric_span_length = lambda: 8

        positions = masker.sample_span_positions(num_tokens=32, num_positions=8)

        self.assertEqual(positions.numel(), 8)
        self.assertTrue(torch.all(positions[1:] - positions[:-1] == 1))

    def test_block_masking_selects_rectangular_block(self):
        masker = build_masker(masking_strategy="block", block_min_size=2, block_max_fraction=0.5)

        positions = masker.sample_block_positions(
            num_tokens=16,
            num_positions=4,
            modality="tok_rgb@256",
            grid_size=(4, 4),
        )

        rows = torch.unique(positions // 4)
        cols = torch.unique(positions % 4)
        rectangle = {
            int(row * 4 + col)
            for row in rows.tolist()
            for col in cols.tolist()
        }
        self.assertEqual(positions.numel(), 4)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(cols), 2)
        self.assertEqual(set(positions.tolist()), rectangle)

    def test_structured_masking_keeps_inputs_and_targets_disjoint(self):
        masker = build_masker(masking_strategy="structured")
        data_dict = {
            "tok_rgb@256": torch.arange(16).long(),
            "scene_desc": torch.arange(16, 32).long(),
        }

        masked = masker.perform_random_masking(
            data_dict,
            input_token_budget=[4, 4],
            target_token_budget=[4, 4],
        )

        for mod_idx in range(2):
            enc_mask = (masked["enc_modalities"] == mod_idx) & masked["enc_pad_mask"]
            dec_mask = (masked["dec_modalities"] == mod_idx) & masked["dec_pad_mask"]
            enc_positions = set(masked["enc_positions"][enc_mask].tolist())
            dec_positions = set(masked["dec_positions"][dec_mask].tolist())
            self.assertTrue(enc_positions.isdisjoint(dec_positions))

        self.assertEqual(int(masked["enc_pad_mask"].sum().item()), 8)
        self.assertEqual(int(masked["dec_pad_mask"].sum().item()), 8)

    def test_tiny_fourm_forward_accepts_structured_masked_batch(self):
        masker = build_masker(masking_strategy="structured")
        data_dict = {
            "tok_rgb@256": torch.arange(16).long(),
            "scene_desc": torch.arange(16, 32).long(),
        }
        masked = masker.perform_random_masking(
            data_dict,
            input_token_budget=[4, 4],
            target_token_budget=[4, 4],
        )
        batch = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in masked.items()
        }

        model = FourM(
            enc_tokens_read_key="enc_tokens",
            dec_tokens_read_key="dec_tokens",
            enc_modalities_read_key="enc_modalities",
            dec_modalities_read_key="dec_modalities",
            enc_positions_read_key="enc_positions",
            dec_positions_read_key="dec_positions",
            enc_pad_mask_read_key="enc_pad_mask",
            dec_pad_mask_read_key="dec_pad_mask",
            modalities=["tok_rgb@256", "scene_desc"],
            vocab_sizes=[64, 64],
            max_seq_lens=[16, 16],
            dim=16,
            enc_depth=1,
            dec_depth=1,
            head_dim=8,
            per_modality_loss_avg=True,
        )

        loss, metrics = model(batch)

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("tok_rgb@256", metrics)
        self.assertIn("scene_desc", metrics)


if __name__ == "__main__":
    unittest.main()
