# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU regression for HF import through native BF16/FSDP wrappers."""

import os
from types import SimpleNamespace

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.utils import unwrap_model

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import DirectMapping
from megatron.bridge.training.gpt_step import _prepare_packed_padding_mask


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data(tmp_path_factory):
    """This test uses in-memory weights and does not download model assets."""
    yield tmp_path_factory.mktemp("fsdp_import")


class _TinyModel(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pre_process = True
        self.linear = torch.nn.Linear(8, 8, bias=False, device="cuda")

    def forward(self, inputs):
        return self.linear(inputs)


class _TinyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        return None

    def mapping_registry(self):
        return MegatronMappingRegistry(DirectMapping("linear.weight", "weight"))

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        model = megatron_model[0]
        return [
            WeightConversionTask(
                param_name="linear.weight",
                global_param_name="linear.weight",
                mapping=DirectMapping("linear.weight", "weight"),
                megatron_module=model.linear,
                param_weight=model.linear.weight,
            )
        ]


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("checkpoint_unwrapped", [False, True])
@pytest.mark.parametrize("strategy", ["optim_grads", "optim_grads_params"])
def test_bf16_fsdp_hf_import(checkpoint_unwrapped, strategy):
    """Imported FP32 shards must populate BF16 forward weights in both entry paths."""
    if "RANK" not in os.environ:
        pytest.skip("Run with two torch.distributed.run processes")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
        )
        model = _TinyModel(config)
        wrapped = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                data_parallel_sharding_strategy=strategy,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                average_in_collective=False,
                megatron_fsdp_main_params_dtype=torch.float32,
            ),
            module=Float16Module(config, model),
            fsdp_unit_modules=[torch.nn.Linear],
        )
        models = unwrap_model([wrapped]) if checkpoint_unwrapped else [wrapped]
        hf = SimpleNamespace(state={"weight": torch.full((8, 8), 2.0)}, model_name_or_path="in-memory")
        result = _TinyBridge().load_weights_hf_to_megatron(hf, models)
        assert result is models
        padding_mask = torch.tensor([[False, False, True, True]], device="cuda")
        assert (
            _prepare_packed_padding_mask(
                padding_mask, config=config, model=models[0], pg_collection=model.pg_collection
            )
            is padding_mask
        )
        wrapped.zero_grad_buffer()
        inputs = torch.ones(4, 8, device="cuda", requires_grad=True)
        output = wrapped(inputs)
        torch.testing.assert_close(output, torch.full_like(output, 16.0), rtol=0, atol=0)
        output.square().mean().backward()
        wrapped.finish_grad_sync()
        assert torch.isfinite(inputs.grad).all()
    finally:
        parallel_state.destroy_model_parallel()
