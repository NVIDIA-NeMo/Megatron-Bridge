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
"""CUDA RNG smoke test for colocated heterogeneous MegatronMIMO.

Runs on the 2-GPU Stage-2 colocated shape:

* vision: ``TP=1, DP=2``
* language: ``TP=2, DP=1``

The test probes MCore's CUDA RNG tracker inside the module scopes wired by
``MegatronMIMOProvider``. This avoids depending on a specific fused dropout
implementation while still verifying the observable contract required when one
physical rank hosts modules with different TP coordinates.

Run:
    torchrun --nproc_per_node=2 -m pytest -v -s -x \\
        tests/functional_tests/test_groups/training/megatron_mimo/test_colocated_heterogeneous_rng_smoke.py
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from typing import Any, Iterator


os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")

import pytest
import torch
import torch.distributed as dist
from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_local_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from tests.functional_tests.utils import initialize_distributed


_HIDDEN_SIZE = 64
_FFN_HIDDEN_SIZE = 256
_NUM_HEADS = 4
_NUM_LAYERS = 2
_VOCAB_SIZE = 1000
_SEQ_LENGTH = 32
_IMG_SIZE = 32
_PATCH_DIM = 16
_SPECIAL_TOKEN_ID = 999

_LANGUAGE = MIMO_LANGUAGE_MODULE_KEY
_VISION = "vision"
_VISION_ENCODER = "clip"
_PROBE_SHAPE = (16,)


def _make_transformer_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        calculate_per_token_loss=True,
    )
    cfg.hidden_dropout = 0.5
    cfg.attention_dropout = 0.5
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    return cfg


def _build_model_specs() -> tuple[ModuleSpec, dict[str, ModuleSpec], dict[str, int]]:
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": _make_transformer_config(),
            "transformer_layer_spec": get_vit_layer_with_local_spec(),
            "patch_dim": _PATCH_DIM,
            "img_h": _IMG_SIZE,
            "img_w": _IMG_SIZE,
        },
    )
    vision_submodule = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={"encoders": {_VISION_ENCODER: vision_encoder}},
    )
    language_model = ModuleSpec(
        module=GPTModel,
        params={
            "config": _make_transformer_config(),
            "transformer_layer_spec": get_gpt_layer_local_spec(),
            "vocab_size": _VOCAB_SIZE,
            "max_sequence_length": _SEQ_LENGTH,
        },
    )
    return language_model, {_VISION: vision_submodule}, {_VISION: _SPECIAL_TOKEN_ID}


def _build_parallelism_config() -> MegatronMIMOParallelismConfig:
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            _LANGUAGE: ModuleParallelismConfig(
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=0,
            ),
            _VISION: ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                data_parallel_size=2,
                rank_offset=0,
            ),
        },
    )


def _build_provider() -> MegatronMIMOProvider:
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()
    provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=_build_parallelism_config(),
        topology={_VISION: [_LANGUAGE], _LANGUAGE: []},
        bf16=False,
    )
    return provider


def _clone_rng_state(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {key: _clone_rng_state(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_clone_rng_state(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_rng_state(value) for value in obj)
    return copy.deepcopy(obj)


def _restore_module_rng_states(infra: Any, snapshot: dict[str, Any]) -> None:
    infra.cuda_rng_states_per_module.clear()
    infra.cuda_rng_states_per_module.update(_clone_rng_state(snapshot))


def _draw_tracker_probe() -> torch.Tensor:
    with tensor_parallel.get_cuda_rng_tracker().fork():
        return torch.empty(_PROBE_SHAPE, device="cuda").uniform_()


def _draw_module_probe(model: torch.nn.Module, module_name: str) -> torch.Tensor:
    with model._scope(module_name):
        return _draw_tracker_probe()


def _gather_probe(probe: torch.Tensor) -> list[torch.Tensor]:
    gathered = [torch.empty_like(probe) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, probe)
    return gathered


def _assert_close(left: torch.Tensor, right: torch.Tensor, message: str) -> None:
    assert torch.equal(left, right), f"{message}: max_abs_diff={(left.float() - right.float()).abs().max().item():.6e}"


def _assert_different(left: torch.Tensor, right: torch.Tensor, message: str) -> None:
    assert not torch.equal(left, right), message


class TestColocatedHeterogeneousRngSmoke:
    """Functional smoke test for module-local CUDA RNG state."""

    @pytest.mark.run_only_on("GPU")
    def test_module_rng_scopes_are_module_local(self) -> None:
        initialize_distributed()
        if dist.get_world_size() != 2:
            pytest.skip(f"Requires exactly 2 GPUs, got {dist.get_world_size()}")

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)

        from megatron.core import parallel_state

        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()

        provider = _build_provider()
        provider.initialize_model_parallel(seed=1234)
        infra = provider.build_infra()
        model = provider.provide()
        model = model.cuda(torch.cuda.current_device())
        model.train()

        assert set(infra.cuda_rng_states_per_module) == {_LANGUAGE, _VISION}

        baseline_rng_states = _clone_rng_state(infra.cuda_rng_states_per_module)

        _restore_module_rng_states(infra, baseline_rng_states)
        vision_probe = _draw_module_probe(model, _VISION)
        vision_by_rank = _gather_probe(vision_probe)
        _assert_close(
            vision_by_rank[0],
            vision_by_rank[1],
            "vision TP=1 DP siblings should draw identical module-local RNG probes",
        )

        _restore_module_rng_states(infra, baseline_rng_states)
        language_probe = _draw_module_probe(model, _LANGUAGE)
        language_by_rank = _gather_probe(language_probe)
        _assert_different(
            language_by_rank[0],
            language_by_rank[1],
            "language TP=2 siblings should draw different module-local RNG probes",
        )

        _restore_module_rng_states(infra, baseline_rng_states)
        vision_a_first = _draw_module_probe(model, _VISION)
        vision_a_second = _draw_module_probe(model, _VISION)

        _restore_module_rng_states(infra, baseline_rng_states)
        vision_b_first = _draw_module_probe(model, _VISION)
        _draw_module_probe(model, _LANGUAGE)
        vision_b_second = _draw_module_probe(model, _VISION)

        _assert_close(vision_a_first, vision_b_first, "first vision probe should be reproducible after reset")
        _assert_close(
            vision_a_second,
            vision_b_second,
            "language RNG activity should not perturb the vision module RNG stream",
        )
        _assert_different(
            vision_a_first,
            vision_a_second,
            "vision RNG stream did not advance between consecutive vision probes",
        )

        _restore_module_rng_states(infra, baseline_rng_states)
        entered_scopes: list[str] = []
        original_scope = model._scope

        @contextmanager
        def recording_scope(module_name: str) -> Iterator[None]:
            with original_scope(module_name):
                entered_scopes.append(module_name)
                if module_name == _LANGUAGE:
                    _draw_tracker_probe()
                yield

        model._scope = recording_scope  # type: ignore[method-assign]
        try:
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device="cuda").unsqueeze(0)
            text_embeddings = model.get_text_embeddings(input_ids, position_ids, model.special_token_ids)
        finally:
            model._scope = original_scope  # type: ignore[method-assign]

        assert text_embeddings.shape == (input_ids.numel(), _HIDDEN_SIZE)
        assert _LANGUAGE in entered_scopes, "get_text_embeddings did not enter the language RNG scope"
