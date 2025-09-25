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
import math
from megatron.bridge.training.utils.flop_utils import num_floating_point_operations
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size

class TestFlops:
    # @pytest.mark.parametrize("recipe, expected_flops", [
    #    ("llama.llama3_8b", 4.22e14),
    #    ("llama.llama3_70b", 3.68e15),
    #    ("llama.llama31_405b", 2.07e16),
    #    ("llama.llama4_e16", 8.92e14),
    #    ("llama.llama4_e128", 8.92e14),
    # ])
    # def test_llama_flops(self, recipe, expected_flops):
    #     module = importlib.import_module(f"megatron.bridge.recipes.{recipe}")
    #     cfg = module.pretrain_config()
    #     cfg.model.finalize()
        
    #     cfg.model.seq_length = 8192

    #     # Calculate padded vocab size to ensure it's divisible by tensor parallel size
    #     cfg.tokenizer.padded_vocab_size = calculate_padded_vocab_size(
    #         cfg.tokenizer.vocab_size,
    #         cfg.model.make_vocab_size_divisible_by,
    #         cfg.model.tensor_model_parallel_size,
    #     )

    #     num_flops = num_floating_point_operations(cfg)
    #     assert (
    #         math.floor(num_flops / 1e12) == expected_flops
    #     ), f"Expected TFLops: {expected_flops:.2e} but got {num_flops:.2e} with {cfg.tokenizer.padded_vocab_size} Seqlen: {cfg.model.seq_length}"


    @pytest.mark.parametrize("recipe, expected_flops", [
       #("deepseek.deepseek_v3", 8.92e14),
       ("qwen.qwen3_30b_a3b", 7.32E+13),
    ])
    def test_moe_flops(self, recipe, expected_flops):
        module = importlib.import_module(f"megatron.bridge.recipes.{recipe}")
        cfg = module.pretrain_config()
        cfg.model.finalize()
        
        cfg.model.seq_length = 4096
        cfg.tokenizer.vocab_size = 151643

        # Calculate padded vocab size to ensure it's divisible by tensor parallel size
        cfg.tokenizer.padded_vocab_size = calculate_padded_vocab_size(
            cfg.tokenizer.vocab_size,
            cfg.model.make_vocab_size_divisible_by,
            cfg.model.tensor_model_parallel_size,
        )

        num_flops = num_floating_point_operations(cfg)
        assert (
            math.floor(num_flops / 1e12) == expected_flops
        ), f"Expected TFLops: {expected_flops:.2e} but got {num_flops:.2e} with {cfg.tokenizer.padded_vocab_size} Seqlen: {cfg.model.seq_length}"
