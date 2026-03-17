# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import datetime
import os

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import Qwen3OmniMoeForConditionalGeneration

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from tests.functional_tests.models.qwen_omni.utils import (
    SMOKE_MODEL_CACHE_PATH,
    SMOKE_LOCK_DIR,
    create_qwen3_omni_smoke_model,
    build_real_sample_inputs,
    move_inputs_to_device,
    smoke_assets_available,
)


pytestmark = [
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(not smoke_assets_available(), reason="Qwen3-Omni local smoke assets are unavailable"),
]


class TestQwen3OmniSmoke:
    @pytest.fixture(scope="class")
    def qwen3_omni_smoke_model_path(self, tmp_path_factory):
        del tmp_path_factory
        return str(create_qwen3_omni_smoke_model(SMOKE_MODEL_CACHE_PATH))

    @staticmethod
    def _init_dist() -> None:
        if dist.is_initialized():
            return
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29515"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=1,
            rank=0,
            timeout=datetime.timedelta(minutes=30),
        )

    @staticmethod
    def _init_model_parallel() -> None:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            model_parallel_cuda_manual_seed(123)
        else:
            torch.manual_seed(123)

    def test_hf_thinker_e2e_smoke(self, qwen3_omni_smoke_model_path):
        inputs = build_real_sample_inputs(qwen3_omni_smoke_model_path)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            qwen3_omni_smoke_model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        prepared = move_inputs_to_device(inputs, device, dtype=model.dtype)

        with torch.no_grad():
            outputs = model.thinker(**prepared)

        assert outputs.logits.shape[:2] == prepared["input_ids"].shape
        assert outputs.logits.shape[-1] == model.config.thinker_config.text_config.vocab_size

    def test_megatron_e2e_smoke(self, qwen3_omni_smoke_model_path):
        self._init_dist()
        self._init_model_parallel()

        try:
            inputs = build_real_sample_inputs(qwen3_omni_smoke_model_path)
            SMOKE_LOCK_DIR.mkdir(parents=True, exist_ok=True)
            os.environ["MEGATRON_CONFIG_LOCK_DIR"] = str(SMOKE_LOCK_DIR)
            bridge = AutoBridge.from_hf_pretrained(qwen3_omni_smoke_model_path, dtype=torch.bfloat16)
            provider = bridge.to_megatron_provider(load_weights=True)
            provider.tensor_model_parallel_size = 1
            provider.pipeline_model_parallel_size = 1
            provider.pipeline_dtype = torch.bfloat16
            provider.params_dtype = torch.bfloat16
            provider.finalize()
            model = provider.provide_distributed_model(wrap_with_ddp=False)
            if isinstance(model, list):
                model = model[0]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            prepared = move_inputs_to_device(
                inputs,
                device,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            prepared["labels"] = prepared["input_ids"].clone()

            with torch.no_grad():
                outputs = model(**prepared)

            assert outputs is not None
        finally:
            os.environ.pop("MEGATRON_CONFIG_LOCK_DIR", None)
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
