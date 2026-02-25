# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for megatron.bridge.training.vocab_slice module."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# Import vocab_slice directly from file to avoid triggering the heavy
# megatron.bridge.__init__ import chain (which requires transformer_engine etc.)
_module_path = Path(__file__).resolve().parents[3] / "src" / "megatron" / "bridge" / "training" / "vocab_slice.py"
_spec = importlib.util.spec_from_file_location("vocab_slice", _module_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["megatron.bridge.training.vocab_slice"] = _mod
_spec.loader.exec_module(_mod)

collect_active_vocab_ids = _mod.collect_active_vocab_ids
install_vocab_slice = _mod.install_vocab_slice
create_vocab_sliced_forward_step = _mod.create_vocab_sliced_forward_step


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class FakeSFTDataset(Dataset):
    """Minimal SFT dataset for testing collect_active_vocab_ids."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class FakeGPTModel(nn.Module):
    """Minimal model that mimics GPTModel's output layer and loss interface."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # output_layer is a stub; install_vocab_slice replaces its forward
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

    def shared_embedding_or_output_weight(self):
        return self.embedding.weight

    def compute_language_model_loss(self, labels, logits):
        # Simple cross-entropy per token (seq-first like Megatron)
        # logits: [S, B, V], labels: [S, B]
        S, B, V = logits.shape
        return F.cross_entropy(
            logits.reshape(-1, V), labels.reshape(-1), reduction="none"
        ).reshape(S, B)


# ---------------------------------------------------------------------------
# Tests: collect_active_vocab_ids
# ---------------------------------------------------------------------------


class TestCollectActiveVocabIds:
    def test_standard_sft_dataset(self):
        """Test collecting active IDs from standard (non-packed) SFT examples."""
        examples = [
            # Context tokens: [100, 200, 300], answer: [5, 10]
            {"input_ids": [100, 200, 300, 5, 10], "answer_start_idx": 3},
            # Context: [400], answer: [5, 15, 20]
            {"input_ids": [400, 5, 15, 20], "answer_start_idx": 1},
        ]
        ds = FakeSFTDataset(examples)
        active_ids = collect_active_vocab_ids(ds)

        expected = torch.tensor([5, 10, 15, 20], dtype=torch.long)
        assert torch.equal(active_ids, expected)

    def test_packed_dataset(self):
        """Test collecting active IDs from packed SFT examples."""
        # Packed example with 2 sub-sequences:
        # Sub-seq 0: boundaries [0, 5], loss_mask active at positions 3,4
        #   -> labels at positions 3,4 are input_ids[4] and input_ids[5-1=4]
        # Sub-seq 1: boundaries [5, 8], loss_mask active at position 5,6
        examples = [
            {
                "input_ids": [100, 200, 300, 7, 8, 500, 12, 13],
                "seq_boundaries": [0, 5, 8],
                "loss_mask": [0, 0, 0, 1, 0, 0, 1, 0],
                # Sub-seq 0: labels = input_ids[1:5] = [200, 300, 7, 8]
                #   loss_mask[0:4] = [0, 0, 0, 1] -> active label = input_ids[3+1]=8? No...
                # Wait, let me recalculate:
                # Sub-seq 0: start=0, end=5, range(4):
                #   k=0: loss_mask[0]=0, skip
                #   k=1: loss_mask[1]=0, skip
                #   k=2: loss_mask[2]=0, skip
                #   k=3: loss_mask[3]=1 -> input_ids[0+3+1]=input_ids[4]=8
                # Sub-seq 1: start=5, end=8, range(2):
                #   k=0: loss_mask[5]=0, skip
                #   k=1: loss_mask[6]=1 -> input_ids[5+1+1]=input_ids[7]=13
            }
        ]
        ds = FakeSFTDataset(examples)
        active_ids = collect_active_vocab_ids(ds)

        expected = torch.tensor([8, 13], dtype=torch.long)
        assert torch.equal(active_ids, expected)

    def test_max_samples_limit(self):
        """Test that max_samples limits scanning."""
        examples = [
            {"input_ids": [100, 5], "answer_start_idx": 1},
            {"input_ids": [200, 10], "answer_start_idx": 1},
            {"input_ids": [300, 15], "answer_start_idx": 1},
        ]
        ds = FakeSFTDataset(examples)

        # Only scan first 2 examples
        active_ids = collect_active_vocab_ids(ds, max_samples=2)
        expected = torch.tensor([5, 10], dtype=torch.long)
        assert torch.equal(active_ids, expected)

    def test_returns_sorted_unique(self):
        """Test that returned IDs are sorted and deduplicated."""
        examples = [
            {"input_ids": [100, 20, 5, 20, 5], "answer_start_idx": 1},
        ]
        ds = FakeSFTDataset(examples)
        active_ids = collect_active_vocab_ids(ds)

        expected = torch.tensor([5, 20], dtype=torch.long)
        assert torch.equal(active_ids, expected)

    def test_empty_dataset(self):
        """Test with empty dataset."""
        ds = FakeSFTDataset([])
        active_ids = collect_active_vocab_ids(ds)
        assert len(active_ids) == 0


# ---------------------------------------------------------------------------
# Tests: install_vocab_slice
# ---------------------------------------------------------------------------


class TestInstallVocabSlice:
    def test_logits_shape_reduced(self):
        """Test that output logits have reduced vocab dimension."""
        vocab_size = 1000
        hidden_size = 32
        model = FakeGPTModel(vocab_size, hidden_size)

        active_ids = torch.tensor([5, 10, 15, 20, 25])
        install_vocab_slice(model, active_ids)

        # Simulate forward: output_layer now returns [*, N] not [*, V]
        hidden = torch.randn(4, 8, hidden_size)  # [S, B, H]
        logits, _ = model.output_layer(hidden, weight=model.shared_embedding_or_output_weight())

        assert logits.shape == (4, 8, 5), f"Expected (4, 8, 5), got {logits.shape}"

    def test_logits_match_full_subset(self):
        """Test that sliced logits exactly match the corresponding rows of full logits."""
        vocab_size = 100
        hidden_size = 16
        model = FakeGPTModel(vocab_size, hidden_size)

        active_ids = torch.tensor([3, 7, 42, 99])

        # Full logits before patching
        hidden = torch.randn(2, 4, hidden_size)
        weight = model.shared_embedding_or_output_weight()
        logits_full = F.linear(hidden, weight)

        # Install and compute sliced logits
        install_vocab_slice(model, active_ids)
        logits_sliced, _ = model.output_layer(hidden, weight=model.shared_embedding_or_output_weight())

        # Compare
        logits_full_subset = logits_full[:, :, active_ids]
        assert torch.allclose(logits_full_subset, logits_sliced, atol=1e-6)

    def test_label_remap(self):
        """Test that labels are correctly remapped for loss computation."""
        vocab_size = 100
        hidden_size = 16
        model = FakeGPTModel(vocab_size, hidden_size)

        active_ids = torch.tensor([10, 20, 30])
        install_vocab_slice(model, active_ids)

        # Create logits [S, B, 3] and labels [S, B] with original IDs
        logits = torch.randn(2, 1, 3)
        labels = torch.tensor([[20], [30]])  # Original IDs

        # After remap: 20 -> 1, 30 -> 2
        loss = model.compute_language_model_loss(labels, logits)
        assert loss.shape == (2, 1)

        # Verify manually
        expected = F.cross_entropy(
            logits.reshape(-1, 3),
            torch.tensor([1, 2]),  # Remapped
            reduction="none",
        ).reshape(2, 1)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_masked_labels_safe(self):
        """Test that labels outside active_ids (at masked positions) map to 0."""
        vocab_size = 100
        hidden_size = 16
        model = FakeGPTModel(vocab_size, hidden_size)

        active_ids = torch.tensor([10, 20])
        install_vocab_slice(model, active_ids)

        # Label 50 is not in active_ids -> maps to 0 (safe, loss will be masked)
        logits = torch.randn(1, 1, 2)
        labels = torch.tensor([[50]])

        # Should not crash
        loss = model.compute_language_model_loss(labels, logits)
        assert loss.shape == (1, 1)

    def test_unwraps_ddp(self):
        """Test that install_vocab_slice correctly unwraps DDP-like wrappers."""
        vocab_size = 50
        hidden_size = 8
        inner_model = FakeGPTModel(vocab_size, hidden_size)

        # Simulate DDP wrapping
        wrapper = MagicMock()
        wrapper.module = inner_model

        active_ids = torch.tensor([1, 2, 3])
        install_vocab_slice(wrapper, active_ids)

        # Verify the inner model was patched
        hidden = torch.randn(1, 1, hidden_size)
        logits, _ = inner_model.output_layer(hidden, weight=inner_model.shared_embedding_or_output_weight())
        assert logits.shape[-1] == 3

    def test_without_shared_weight(self):
        """Test that output_layer works when weight is not passed (uses shared weight)."""
        vocab_size = 50
        hidden_size = 8
        model = FakeGPTModel(vocab_size, hidden_size)

        active_ids = torch.tensor([0, 5, 10])
        install_vocab_slice(model, active_ids)

        # Call without passing weight (should use shared_embedding_or_output_weight)
        hidden = torch.randn(1, 1, hidden_size)
        logits, _ = model.output_layer(hidden)
        assert logits.shape[-1] == 3


# ---------------------------------------------------------------------------
# Tests: create_vocab_sliced_forward_step
# ---------------------------------------------------------------------------


class TestCreateVocabSlicedForwardStep:
    def test_wraps_base_forward_step(self):
        """Test that the wrapper calls the base forward_step."""
        call_count = 0

        def mock_forward_step(state, data_iterator, model, return_schedule_plan=False):
            nonlocal call_count
            call_count += 1
            return torch.tensor(0.0), None

        active_ids = torch.tensor([1, 2, 3])

        with patch("megatron.bridge.training.vocab_slice.install_vocab_slice") as mock_install:
            fwd = create_vocab_sliced_forward_step(active_ids, base_forward_step=mock_forward_step)

            # Call twice
            fwd(None, None, MagicMock())
            fwd(None, None, MagicMock())

        assert call_count == 2, "Base forward_step should be called each time"
        assert mock_install.call_count == 1, "install_vocab_slice should only be called once"

    def test_installs_on_first_call_only(self):
        """Test that vocab slice is installed on first call and not subsequent ones."""
        active_ids = torch.tensor([1, 2, 3])

        with patch("megatron.bridge.training.vocab_slice.install_vocab_slice") as mock_install:
            mock_base = MagicMock(return_value=(torch.tensor(0.0), None))
            fwd = create_vocab_sliced_forward_step(active_ids, base_forward_step=mock_base)

            model = MagicMock()
            fwd(None, None, model)
            fwd(None, None, model)
            fwd(None, None, model)

        # install called exactly once, with the model from the first call
        mock_install.assert_called_once_with(model, active_ids)

    def test_passes_all_arguments(self):
        """Test that all arguments are forwarded to the base forward_step."""
        received_args = {}

        def mock_forward_step(state, data_iterator, model, return_schedule_plan=False):
            received_args["state"] = state
            received_args["data_iterator"] = data_iterator
            received_args["model"] = model
            received_args["return_schedule_plan"] = return_schedule_plan
            return torch.tensor(0.0), None

        active_ids = torch.tensor([1, 2, 3])

        with patch("megatron.bridge.training.vocab_slice.install_vocab_slice"):
            fwd = create_vocab_sliced_forward_step(active_ids, base_forward_step=mock_forward_step)

            state = MagicMock()
            data_iter = MagicMock()
            model = MagicMock()
            fwd(state, data_iter, model, return_schedule_plan=True)

        assert received_args["state"] is state
        assert received_args["data_iterator"] is data_iter
        assert received_args["model"] is model
        assert received_args["return_schedule_plan"] is True


# ---------------------------------------------------------------------------
# Tests: end-to-end correctness
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_sliced_logits_match_full_for_active_tokens(self):
        """End-to-end: sliced logits exactly match full logits for active token positions."""
        vocab_size = 200
        hidden_size = 32
        n_active = 10
        seq_len = 16

        model = FakeGPTModel(vocab_size, hidden_size)
        active_ids = torch.arange(n_active)

        hidden = torch.randn(seq_len, 1, hidden_size)

        # Full logits before patching
        weight = model.shared_embedding_or_output_weight()
        logits_full = F.linear(hidden, weight)  # [S, 1, V]
        logits_full_active = logits_full[:, :, active_ids]  # [S, 1, N]

        # Install slice and compute
        install_vocab_slice(model, active_ids)
        logits_sliced, _ = model.output_layer(
            hidden, weight=model.shared_embedding_or_output_weight()
        )

        # Logits for active tokens should be identical
        assert torch.allclose(logits_full_active, logits_sliced, atol=1e-6)

    def test_sliced_loss_is_finite_and_positive(self):
        """End-to-end: sliced model produces finite, positive loss."""
        vocab_size = 200
        hidden_size = 32
        n_active = 10
        seq_len = 16

        model = FakeGPTModel(vocab_size, hidden_size)
        active_ids = torch.arange(n_active)
        install_vocab_slice(model, active_ids)

        hidden = torch.randn(seq_len, 1, hidden_size)
        labels = torch.randint(0, n_active, (seq_len, 1))

        logits, _ = model.output_layer(hidden, weight=model.shared_embedding_or_output_weight())
        loss_per_token = model.compute_language_model_loss(labels, logits)

        loss = loss_per_token.sum()
        assert loss.isfinite()
        assert loss > 0

    def test_gradient_flows_to_active_embeddings(self):
        """Test that gradients only flow to active embedding rows via output projection."""
        vocab_size = 50
        hidden_size = 8

        model = FakeGPTModel(vocab_size, hidden_size)
        active_ids = torch.tensor([5, 10, 15])
        install_vocab_slice(model, active_ids)

        hidden = torch.randn(2, 1, hidden_size)
        labels = torch.tensor([[10], [15]])  # -> remapped to [1, 2]
        weight = model.shared_embedding_or_output_weight()

        logits, _ = model.output_layer(hidden, weight=weight)
        loss = model.compute_language_model_loss(labels, logits).sum()
        loss.backward()

        # Only active rows should have gradients from the output projection
        grad = model.embedding.weight.grad
        assert grad is not None

        # Active rows should have non-zero gradient
        for idx in active_ids:
            assert grad[idx].abs().sum() > 0, f"Active ID {idx} should have gradient"

        # Non-active rows should have zero gradient (from output path only)
        inactive_mask = torch.ones(vocab_size, dtype=torch.bool)
        inactive_mask[active_ids] = False
        inactive_grad_norm = grad[inactive_mask].abs().sum()
        assert inactive_grad_norm == 0, "Inactive embeddings should have zero gradient from output"
