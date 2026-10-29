# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Functional tests for MegatronMIMO conversion checkpoint I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
import torch
import torch.distributed as dist
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.megatron_mimo import build_megatron_mimo_model
from megatron.bridge.models.megatron_mimo.conversion.mimo_model_io import (
    load_megatron_mimo_model,
    save_megatron_mimo_model,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
from tests.functional_tests.utils import broadcast_path, initialize_distributed


_IMAGE_TOKEN_ID = 63
register_allowed_target_prefix(f"{__name__}.")


@dataclass
class TinyStandardMIMOProvider:
    """Tiny modality-aware provider used to test conversion-generated MIMO providers."""

    vocab_size: int = 64
    hidden_size: int = 16
    num_layers: int = 1
    num_attention_heads: int = 4
    seq_length: int = 16
    image_size: int = 16
    patch_dim: int = 8
    modality_keys: ClassVar[dict[str, str]] = {"images": "clip"}
    special_token_ids: ClassVar[dict[str, int]] = {"images": _IMAGE_TOKEN_ID}

    def _transformer_config(self) -> TransformerConfig:
        return TransformerConfig(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.hidden_size * 4,
            num_attention_heads=self.num_attention_heads,
            bf16=True,
            variable_seq_lengths=True,
            moe_token_dispatcher_type="alltoall",
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )

    def build_language_model_spec(self, pp_rank: int = 0) -> ModuleSpec:
        return ModuleSpec(
            module=GPTModel,
            params={
                "config": self._transformer_config(),
                "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
                "vocab_size": self.vocab_size,
                "max_sequence_length": self.seq_length,
                "pre_process": pp_rank == 0,
                "post_process": True,
            },
        )

    def build_vision_encoder_spec(self) -> ModuleSpec:
        return ModuleSpec(
            module=CLIPViTModel,
            params={
                "transformer_config": self._transformer_config(),
                "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
                "patch_dim": self.patch_dim,
                "img_h": self.image_size,
                "img_w": self.image_size,
            },
        )


def _parallelism_config() -> MegatronMIMOParallelismConfig:
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            MIMO_LANGUAGE_MODULE_KEY: ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=0,
            ),
            "images": ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=1,
            ),
        }
    )


def _mimo_provider() -> MegatronMIMOProvider:
    return MegatronMIMOProvider.from_standard_provider(
        standard_provider=TinyStandardMIMOProvider(),
        megatron_mimo_parallelism_config=_parallelism_config(),
    )


def _active_submodule(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    if module_name == MIMO_LANGUAGE_MODULE_KEY:
        return model.language_model
    return model.modality_submodules[module_name]


def _fill_active_submodule(model: torch.nn.Module, module_name: str) -> None:
    submodule = _active_submodule(model, module_name)
    fill_value = float(dist.get_rank() + 1)
    with torch.no_grad():
        for param in submodule.parameters():
            param.fill_(fill_value)


def _active_parameter_sum(model: torch.nn.Module, module_name: str) -> float:
    submodule = _active_submodule(model, module_name)
    return float(sum(param.detach().float().sum().cpu() for param in submodule.parameters()))


@pytest.mark.run_only_on("GPU")
def test_megatron_mimo_conversion_checkpoint_save_load_roundtrip(tmp_path):
    initialize_distributed()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this functional test")
    if dist.get_world_size() != 2:
        pytest.skip("This functional test requires exactly 2 ranks")

    checkpoint_path = Path(broadcast_path(tmp_path / "mimo_conversion_ckpt"))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    provider = _mimo_provider()
    model, infra = build_megatron_mimo_model(
        provider,
        bf16=True,
        wrap_with_ddp=False,
        data_parallel_random_init=False,
    )
    active_module = infra.participating_modules[0]
    _fill_active_submodule(model, active_module)
    expected_sum = _active_parameter_sum(model, active_module)

    save_megatron_mimo_model(model, infra, provider, checkpoint_path)
    dist.barrier()

    loaded_model, loaded_infra, loaded_provider = load_megatron_mimo_model(
        checkpoint_path,
        parallelism_config=_parallelism_config(),
        bf16=True,
        wrap_with_ddp=False,
        data_parallel_random_init=False,
    )
    loaded_active_module = loaded_infra.participating_modules[0]

    assert loaded_active_module == active_module
    assert loaded_provider.standard_provider is not None
    assert loaded_provider.language_model_spec is not None
    assert loaded_provider.modality_submodules_spec
    assert _active_parameter_sum(loaded_model, loaded_active_module) == pytest.approx(expected_sum)
