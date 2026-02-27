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

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.training.model_load_save import load_megatron_model, load_tokenizer


def init_distributed():
    """Initialize process group, model parallel, rng seed for single GPU."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(torch.distributed.get_rank())

    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(42)


class TestModelLoad:
    """Test instantiating and loading model and tokenizer from Megatron Bridge and MegatronLM checkpoints."""

    @pytest.mark.pleasefixme
    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "ckpt_path",
        [
            "/home/TestData/megatron_bridge/checkpoints/llama3_145m-mbridge_saved-distckpt",
            "/home/TestData/megatron_bridge/checkpoints/llama3_145m-mlm_saved-distckpt",
        ],
    )
    def test_model_load_and_forward(self, ckpt_path: str):
        """Load model and tokenizer, run a forward pass, and check outputted token id."""

        init_distributed()

        try:
            tokenizer = load_tokenizer(ckpt_path)
            model = load_megatron_model(ckpt_path, model_type="gpt", use_cpu_init=False)
            model = model[0]

            # This test expects tokenizer to be a SentencePiece tokenizer
            token_ids = tokenizer.tokenize("NVIDIA NeMo is an end-to-end platform for")
            input_batch = torch.tensor([token_ids]).cuda()
            position_ids = torch.arange(input_batch.size(1), dtype=torch.long, device=input_batch.device)
            attention_mask = torch.ones_like(input_batch, dtype=torch.bool)

            with torch.no_grad():
                output = model.forward(input_ids=input_batch, position_ids=position_ids, attention_mask=attention_mask)

            next_token_id = torch.argmax(output[:, -1], dim=-1).item()
            expected_id = 267
            assert next_token_id == expected_id, (
                f"Model checkpoint at {ckpt_path} did not produce expected next token. Expected: {expected_id}, Actual: {next_token_id}"
            )
        finally:
            parallel_state.destroy_model_parallel()
