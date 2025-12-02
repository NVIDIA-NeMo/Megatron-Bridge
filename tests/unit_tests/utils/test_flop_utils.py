#!/usr/bin/env python3
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

"""Tests for flop_utils module."""

import importlib

import pytest

from megatron.bridge.training.utils.flop_utils import num_floating_point_operations
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


class TestFlops:
    @pytest.mark.parametrize(
        "model_family,model_config_func_name, seq_length, vocab_size, expected_flops",
        [
            ("llama", "llama3_8b_pretrain_config", 8192, 151936, 4.22e14),
            ("llama", "llama3_70b_pretrain_config", 8192, 151936, 3.68e15),
            ("llama", "llama31_405b_pretrain_config", 8192, 151936, 2.07e16),
            ("qwen", "qwen3_30b_a3b_pretrain_config", 4096, 151643, 9.42e13),
            ("qwen", "qwen3_235b_a22b_pretrain_config", 4096, 151643, 6.06e14),
        ],
    )
    def test_flops(self, model_family, model_config_func_name, seq_length, vocab_size, expected_flops):
        model_family_module = importlib.import_module(f"megatron.bridge.recipes.{model_family}")
        cfg = getattr(model_family_module, model_config_func_name)()
        cfg.model.finalize()

        cfg.model.seq_length = seq_length
        cfg.tokenizer.vocab_size = vocab_size

        # Calculate padded vocab size to ensure it's divisible by tensor parallel size
        cfg.tokenizer.padded_vocab_size = calculate_padded_vocab_size(
            cfg.tokenizer.vocab_size,
            cfg.model.make_vocab_size_divisible_by,
            cfg.model.tensor_model_parallel_size,
        )

        actual_num_flops = num_floating_point_operations(cfg, batch_size=1)
        actual_num_flops_rounded = float(f"{actual_num_flops:.2e}")

        assert actual_num_flops_rounded == expected_flops, (
            f"Expected TFLops: {expected_flops} but got {actual_num_flops} with Padded Vocab Size: {cfg.tokenizer.padded_vocab_size} and Sequence len: {cfg.model.seq_length}"
        )
