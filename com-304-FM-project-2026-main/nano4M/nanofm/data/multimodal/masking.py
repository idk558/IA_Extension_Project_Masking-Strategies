# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Any, Union, Optional
import math
import random
from timm.models.layers import to_2tuple
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from .utils import to_unified_multimodal_vocab


class SimpleMultimodalMasking(object):
    def __init__(
            self,
            modalities: List[str],
            vocab_sizes: List[int],
            max_seq_lens: List[int],
            input_alphas: List[str],
            target_alphas: List[str],
            input_tokens_range: Union[int, Tuple[int, int]],
            target_tokens_range: Union[int, Tuple[int, int]],
            overlap_vocab: bool = True,
            overlap_posembs: bool = True,
            include_unmasked_data_dict: bool = False,
            masking_strategy: str = "random",
            structured_mask_probability: float = 1.0,
            sequence_modalities: Optional[List[str]] = None,
            image_modalities: Optional[List[str]] = None,
            image_token_grid_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
            span_geometric_p: float = 0.2,
            block_min_size: int = 1,
            block_max_fraction: float = 0.5,
        ):
        """
        Simple multimodal masking class for sampling input and target masks for each modality.
        Operates on a dictionary of modalities, where each entry is a dictionary with 
        a 'tokens' key containing the token tensor.

        Args:
            modalities: List of modality names
            vocab_sizes: Vocabulary size of each modality. Used to create a unified vocabolary.
            max_seq_lens: Maximum sequence length for each modality
            input_alphas: List of Dirichlet alphas for the input modalities
            target_alphas: List of Dirichlet alphas for the target modalities
            input_tokens_range: Range of number of input tokens to sample from
            target_tokens_range: Range of number of target tokens to sample from
            overlap_vocab: Whether to use a unified vocabulary across modalities.
            overlap_posembs: Whether to reuse position indices/embeddings across modalities.
            include_unmasked_data_dict: If True, adds the unmasked data dictionary to the output
                using the key 'unmasked_data_dict'.
            masking_strategy: Strategy used for target positions. Supports:
                'random', 'span', 'block', 'structured', and 'mixed'. Structured uses
                span masking for sequence modalities and block masking for image modalities.
            structured_mask_probability: Probability of using the structured strategy when
                it applies to a modality. Values below 1.0 mix structured and random masking.
            sequence_modalities: Optional explicit list of sequence-like modalities.
            image_modalities: Optional explicit list of image-like modalities.
            image_token_grid_sizes: Optional per-modality grid sizes, e.g.
                {'tok_rgb@256': (16, 16)}. If omitted, square grids are inferred.
            span_geometric_p: Success probability for geometric span lengths. p=0.2 has
                mean span length close to 5.
            block_min_size: Minimum sampled block height/width in tokens.
            block_max_fraction: Maximum sampled block height/width as a fraction of the grid.
        """
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.vocab_sizes = vocab_sizes
        self.max_seq_lens = max_seq_lens
        self.input_alphas = torch.tensor(input_alphas)
        self.target_alphas = torch.tensor(target_alphas)
        self.input_tokens_range = to_2tuple(input_tokens_range)
        self.target_tokens_range = to_2tuple(target_tokens_range)
        self.overlap_vocab = overlap_vocab
        self.overlap_posembs = overlap_posembs
        self.include_unmasked_data_dict = include_unmasked_data_dict
        self.masking_strategy = masking_strategy.lower()
        self.structured_mask_probability = structured_mask_probability
        self.sequence_modalities = set(sequence_modalities or [])
        self.image_modalities = set(image_modalities or [])
        self.image_token_grid_sizes = {
            mod: tuple(grid_size)
            for mod, grid_size in (image_token_grid_sizes or {}).items()
        }
        self.span_geometric_p = span_geometric_p
        self.block_min_size = block_min_size
        self.block_max_fraction = block_max_fraction

        valid_strategies = {"random", "span", "block", "structured", "mixed"}
        if self.masking_strategy not in valid_strategies:
            raise ValueError(f"Unknown masking_strategy '{masking_strategy}'. Expected one of {sorted(valid_strategies)}")
        if not 0.0 <= self.structured_mask_probability <= 1.0:
            raise ValueError("structured_mask_probability must be in [0, 1]")
        if not 0.0 < self.span_geometric_p <= 1.0:
            raise ValueError("span_geometric_p must be in (0, 1]")
        if self.block_min_size < 1:
            raise ValueError("block_min_size must be >= 1")
        if not 0.0 < self.block_max_fraction <= 1.0:
            raise ValueError("block_max_fraction must be in (0, 1]")

        self.max_seq_len_shifts = torch.tensor(max_seq_lens).cumsum(0) - max_seq_lens[0]

        # Dirichlet sampling
        eps = 1e-9
        self.input_dirichlet = Dirichlet(torch.clamp(self.input_alphas, min=eps))
        self.target_dirichlet = Dirichlet(torch.clamp(self.target_alphas, min=eps))

    def is_sequence_modality(self, modality: str) -> bool:
        if modality in self.sequence_modalities:
            return True
        if modality in self.image_modalities:
            return False
        return "scene_desc" in modality or "caption" in modality or "text" in modality

    def is_image_modality(self, modality: str) -> bool:
        if modality in self.image_modalities:
            return True
        if modality in self.sequence_modalities:
            return False
        return modality.startswith("tok_") or "@256" in modality

    def infer_grid_size(self, modality: str, num_tokens: int) -> Optional[Tuple[int, int]]:
        if modality in self.image_token_grid_sizes:
            grid_h, grid_w = self.image_token_grid_sizes[modality]
            if grid_h * grid_w != num_tokens:
                raise ValueError(
                    f"Grid size {grid_h}x{grid_w} for modality '{modality}' "
                    f"does not match {num_tokens} tokens"
                )
            return grid_h, grid_w

        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size == num_tokens:
            return grid_size, grid_size
        return None

    def sample_random_positions(
            self,
            num_tokens: int,
            num_positions: int,
            exclude_positions: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """Sample sorted token positions uniformly at random, optionally excluding positions."""
        if num_positions <= 0 or num_tokens <= 0:
            return torch.empty(0, dtype=torch.long)

        available = torch.ones(num_tokens, dtype=torch.bool)
        if exclude_positions is not None and exclude_positions.numel() > 0:
            available[exclude_positions.long()] = False
        available_positions = torch.arange(num_tokens, dtype=torch.long)[available]
        num_positions = min(num_positions, available_positions.numel())
        if num_positions <= 0:
            return torch.empty(0, dtype=torch.long)

        perm = torch.randperm(available_positions.numel())[:num_positions]
        return available_positions[perm].sort()[0]

    def _sample_geometric_span_length(self) -> int:
        length = 1
        while random.random() > self.span_geometric_p:
            length += 1
        return length

    @staticmethod
    def _available_runs(selected: torch.Tensor) -> List[Tuple[int, int]]:
        """Return contiguous runs where selected is False as (start, length)."""
        runs = []
        start = None
        for idx, is_selected in enumerate(selected.tolist()):
            if not is_selected and start is None:
                start = idx
            elif is_selected and start is not None:
                runs.append((start, idx - start))
                start = None
        if start is not None:
            runs.append((start, selected.numel() - start))
        return runs

    def sample_span_positions(self, num_tokens: int, num_positions: int) -> torch.Tensor:
        """Sample target positions as non-overlapping contiguous spans."""
        if num_positions <= 0 or num_tokens <= 0:
            return torch.empty(0, dtype=torch.long)

        num_positions = min(num_positions, num_tokens)
        selected = torch.zeros(num_tokens, dtype=torch.bool)

        while int(selected.sum().item()) < num_positions:
            remaining = num_positions - int(selected.sum().item())
            runs = self._available_runs(selected)
            if not runs:
                break

            run_weights = [run_len for _, run_len in runs]
            run_start, run_len = random.choices(runs, weights=run_weights, k=1)[0]
            span_len = min(self._sample_geometric_span_length(), remaining, run_len)
            max_start = run_start + run_len - span_len
            start = random.randint(run_start, max_start)
            selected[start:start + span_len] = True

        return torch.where(selected)[0].long()

    def sample_block_positions(
            self,
            num_tokens: int,
            num_positions: int,
            modality: Optional[str] = None,
            grid_size: Optional[Tuple[int, int]] = None,
        ) -> torch.Tensor:
        """Sample target positions as rectangular blocks on an image-token grid."""
        if num_positions <= 0 or num_tokens <= 0:
            return torch.empty(0, dtype=torch.long)

        num_positions = min(num_positions, num_tokens)
        if grid_size is None:
            grid_size = self.infer_grid_size(modality or "", num_tokens)
        if grid_size is None:
            return self.sample_random_positions(num_tokens, num_positions)

        grid_h, grid_w = grid_size
        selected = torch.zeros(grid_h, grid_w, dtype=torch.bool)
        min_h = min(self.block_min_size, grid_h)
        min_w = min(self.block_min_size, grid_w)
        max_h = max(min_h, int(grid_h * self.block_max_fraction))
        max_w = max(min_w, int(grid_w * self.block_max_fraction))
        max_h = min(max_h, grid_h)
        max_w = min(max_w, grid_w)

        attempts = 0
        max_attempts = max(32, num_positions * 8)
        while int(selected.sum().item()) < num_positions and attempts < max_attempts:
            attempts += 1
            remaining = num_positions - int(selected.sum().item())
            h = random.randint(min_h, max_h)
            w = random.randint(min_w, max_w)
            h = min(h, grid_h)
            w = min(w, grid_w)
            top = random.randint(0, grid_h - h)
            left = random.randint(0, grid_w - w)

            block = selected[top:top + h, left:left + w]
            new_cells = (~block).nonzero(as_tuple=False)
            if new_cells.numel() == 0:
                continue
            if new_cells.shape[0] > remaining:
                new_cells = new_cells[torch.randperm(new_cells.shape[0])[:remaining]]
            selected[top + new_cells[:, 0], left + new_cells[:, 1]] = True

        if int(selected.sum().item()) < num_positions:
            flat_selected = selected.flatten()
            missing = num_positions - int(flat_selected.sum().item())
            fill = self.sample_random_positions(num_tokens, missing, exclude_positions=torch.where(flat_selected)[0])
            flat_selected[fill] = True
            selected = flat_selected.reshape(grid_h, grid_w)

        return torch.where(selected.flatten())[0].long()

    def sample_target_positions(self, modality: str, num_tokens: int, num_positions: int) -> torch.Tensor:
        """Sample decoder target positions according to the configured masking strategy."""
        use_structured = random.random() < self.structured_mask_probability
        if self.masking_strategy == "random" or not use_structured:
            return self.sample_random_positions(num_tokens, num_positions)

        if self.masking_strategy in {"span", "structured", "mixed"} and self.is_sequence_modality(modality):
            return self.sample_span_positions(num_tokens, num_positions)

        if self.masking_strategy in {"block", "structured", "mixed"} and self.is_image_modality(modality):
            return self.sample_block_positions(num_tokens, num_positions, modality=modality)

        return self.sample_random_positions(num_tokens, num_positions)
        
    def input_token_budget(self, num_input_tokens: int, max_tokens: torch.Tensor) -> List[int]:
        """Sample the number of input tokens for each modality, i.e. the
        per-modality token budget.

        Args:
            num_input_tokens: Number of tokens in the input
            max_tokens: Maximum number of tokens per modality

        Returns:
            Token budget for the input
        """
        # Get the number of tokens for each modality
        input_token_budget = (self.input_dirichlet.sample() * num_input_tokens).floor().int()
        diff = num_input_tokens - input_token_budget.sum()
        # Adds the remaining tokens by sampling from the Dirichlet and taking the argmax
        # This avoids adding tokens to modalities that shouldn't be sampled (i.e. with alphas ~=0)
        input_token_budget += torch.bincount(self.input_dirichlet.sample((diff,)).argmax(dim=-1), minlength=len(input_token_budget))

        # If token budget is over max tokens for a given modality, set it to max
        input_token_budget = torch.clamp(input_token_budget, max=max_tokens)

        return input_token_budget.tolist()

    def target_token_budget(
            self, 
            input_token_budget: List[int], 
            num_target_tokens: int,
            max_tokens: torch.Tensor,
        ) -> List[int]:
        """Sample the number of target tokens for each modality, i.e. the
        per-modality token budget.

        Args:
            input_token_budget: Token budget for the input modalities
            num_target_tokens: Number of tokens in the target
            max_tokens: Maximum number of tokens per modality

        Returns:
            Token budget for the target
        """
        max_tokens_remaining = max_tokens - torch.tensor(input_token_budget)

        target_token_budget = (self.target_dirichlet.sample() * num_target_tokens).floor().int()
        diff = num_target_tokens - target_token_budget.sum()
        # Adds the remaining tokens by sampling from the Dirichlet and taking the argmax
        # This avoids adding tokens to modalities that shouldn't be sampled (i.e. with alphas ~=0)
        target_token_budget += torch.bincount(self.target_dirichlet.sample((diff,)).argmax(dim=-1), minlength=len(target_token_budget))

        # If token budget is over max tokens for a given modality, set it to max
        target_token_budget = torch.clamp(target_token_budget, max=max_tokens_remaining)

        return target_token_budget.tolist()

    def perform_random_masking(
            self, 
            data_dict: Dict[str, Any],
            input_token_budget: List[int],
            target_token_budget: List[int],
        ) -> Dict[str, Any]:
        """
        Applies input and target masking to a dictionary of modalities.

        Args:
            data_dict: Dictionary of modalities and the corresponding tokens
            input_token_budget: Token budget for the input modalities
            target_token_budget: Token budget for the target modalities
        Returns:
            Dictionary containing the masked modality information
        """
        enc_tokens, enc_positions, enc_modalities = [], [], []
        dec_tokens, dec_positions, dec_modalities = [], [], []

        for mod_idx, mod in enumerate(self.modalities):
            num_tokens = data_dict[mod].shape[0]
            n_input_tokens = input_token_budget[mod_idx]
            n_target_tokens = target_token_budget[mod_idx]
            
            # Sample structured target positions first, then draw visible input tokens
            # from the remaining positions so encoder and decoder tokens never overlap.
            target_pos = self.sample_target_positions(mod, num_tokens, n_target_tokens)
            input_pos = self.sample_random_positions(num_tokens, n_input_tokens, exclude_positions=target_pos)
            # Optionally shift the position indices such that each modality learns unique position embeddings
            pos_idx_shift = 0 if self.overlap_posembs else self.max_seq_len_shifts[mod_idx]
            enc_positions.append(input_pos + pos_idx_shift)
            dec_positions.append(target_pos + pos_idx_shift)

            # Get the corresponding input and target tokens
            input_tokens, target_tokens = data_dict[mod][input_pos], data_dict[mod][target_pos]
            enc_tokens.append(input_tokens)
            dec_tokens.append(target_tokens)

            # In case n_input_tokens+n_target_tokens was larger than num_tokens, let's recompute 
            # the actual number of input and target tokens
            n_input_tokens, n_target_tokens = input_pos.shape[0], target_pos.shape[0]
            
            # To decide which token to predict in the encoder and decoder, we pass modality indices 
            # that are transformed into a modality embedding
            enc_modalities.append(mod_idx * torch.ones(n_input_tokens, dtype=torch.long))
            dec_modalities.append(mod_idx * torch.ones(n_target_tokens, dtype=torch.long))
                        
        # Concatenate all lists into tensors
        enc_tokens, dec_tokens = torch.cat(enc_tokens), torch.cat(dec_tokens)
        enc_positions, dec_positions = torch.cat(enc_positions), torch.cat(dec_positions)
        enc_modalities, dec_modalities = torch.cat(enc_modalities), torch.cat(dec_modalities)

        # For batching, all sequences need the same length.
        max_input_tokens, max_target_tokens = self.input_tokens_range[1], self.target_tokens_range[1]
        enc_pad_length = max_input_tokens - enc_tokens.shape[0]
        dec_pad_length = max_target_tokens - dec_tokens.shape[0]
        enc_tokens = F.pad(enc_tokens, (0, enc_pad_length), mode='constant', value=0)
        enc_positions = F.pad(enc_positions, (0, enc_pad_length), mode='constant', value=0)
        enc_modalities = F.pad(enc_modalities, (0, enc_pad_length), mode='constant', value=0)
        dec_positions = F.pad(dec_positions, (0, dec_pad_length), mode='constant', value=0)
        dec_tokens = F.pad(dec_tokens, (0, dec_pad_length), mode='constant', value=-100)
        dec_modalities = F.pad(dec_modalities, (0, dec_pad_length), mode='constant', value=0)

        # Create attention masks for encoder and decoder
        enc_pad_mask = torch.ones(max_input_tokens, dtype=torch.bool)
        if enc_pad_length > 0:
            enc_pad_mask[-enc_pad_length:] = False
        dec_pad_mask = torch.ones(max_target_tokens, dtype=torch.bool)
        if dec_pad_length > 0:
            dec_pad_mask[-dec_pad_length:] = False

        masked_data_dict = {
            'enc_tokens': enc_tokens,
            'enc_positions': enc_positions,
            'enc_modalities': enc_modalities,
            'enc_pad_mask': enc_pad_mask,
            'dec_tokens': dec_tokens,
            'dec_positions': dec_positions,
            'dec_modalities': dec_modalities,
            'dec_pad_mask': dec_pad_mask,
        }

        return masked_data_dict

    def __call__(self, data_dict):
        """Applies input and target masking to a dictionary of modalities

        Args:
            data_dict: Dictionary of modalities

        Returns:
            Dictionary containing the masked modalities
        """
        if not self.overlap_vocab:
            # Unify the vocabulary for all modalities, making sure the indices for each modality 
            # are non-overlapping with other modalities.
            data_dict = to_unified_multimodal_vocab(data_dict, self.modalities, self.vocab_sizes)

        # Get maximum number of tokens for each modality
        max_tokens = torch.tensor(self.max_seq_lens)
        
        # Sample number of input and target tokens
        num_input_tokens = random.randint(*self.input_tokens_range)
        num_target_tokens = random.randint(*self.target_tokens_range)
        
        # Get input and target per-modality token budgets
        input_token_budget = self.input_token_budget(num_input_tokens, max_tokens)
        target_token_budget = self.target_token_budget(input_token_budget, num_target_tokens, max_tokens)
            
        # Apply input and target masking
        masked_data_dict = self.perform_random_masking(data_dict, input_token_budget, target_token_budget)

        if self.include_unmasked_data_dict:
            masked_data_dict['unmasked_data_dict'] = data_dict
            
        return masked_data_dict
