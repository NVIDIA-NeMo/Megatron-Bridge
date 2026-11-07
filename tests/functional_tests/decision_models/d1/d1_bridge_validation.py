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

"""Strict full-state conversion checks for explicitly invoked D1 validators."""

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry


@contextmanager
def standalone_cuda() -> Iterator[torch.device]:
    """Own and clean the one-GPU groups created by a standalone validator.

    Existing distributed jobs are rejected rather than reconfigured or destroyed.
    Importing this module does not initialize CUDA, distributed groups or downloads.
    """
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if torch.distributed.is_initialized() or parallel_state.model_parallel_is_initialized():
        raise RuntimeError("Run this one-GPU validator outside an existing distributed process group")
    if not torch.cuda.is_available():
        raise RuntimeError("D1 Bridge publisher validation requires the supported CUDA environment")
    torch.cuda.set_device(0)
    with TemporaryDirectory(prefix="d1-parity-") as temporary:
        try:
            torch.distributed.init_process_group(
                "nccl", init_method=f"file://{Path(temporary) / 'rendezvous'}", rank=0, world_size=1
            )
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            model_parallel_cuda_manual_seed(7)
            yield torch.device("cuda", 0)
        finally:
            try:
                if parallel_state.model_parallel_is_initialized():
                    parallel_state.destroy_model_parallel()
            finally:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()


def assert_complete_round_trip(
    original: Mapping[str, torch.Tensor], consumed: set[str], exported: Mapping[str, torch.Tensor]
) -> None:
    """Require every full publisher state-dict key, dtype and value to round-trip."""
    assert consumed == set(original), {
        "unconsumed": sorted(set(original) - consumed),
        "unexpected": sorted(consumed - set(original)),
    }
    assert set(exported) == set(original), {
        "unexported": sorted(set(original) - set(exported)),
        "unexpected": sorted(set(exported) - set(original)),
    }
    for name, value in original.items():
        assert exported[name].dtype == value.dtype, name
        assert torch.equal(exported[name], value), name


def copy_and_round_trip(
    native: torch.nn.Module,
    registry: "MegatronMappingRegistry",
    original: Mapping[str, torch.Tensor],
    buffers: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Load parameters and persistent buffers through actual native mappings.

    The production caller supplies Bridge's persistent_buffers iterator. The
    complete reference state_dict, including tied aliases, is the coverage oracle.
    """
    consumed: set[str] = set()
    exported: dict[str, torch.Tensor] = {}
    for name, parameter in (*native.named_parameters(), *buffers):
        mapping = registry.megatron_to_hf_lookup(name)
        assert mapping is not None, name
        if isinstance(mapping.hf_param, str):
            source = original[mapping.hf_param]
            consumed.add(mapping.hf_param)
        else:
            source = {kind: original[key] for kind, key in mapping.hf_param.items()}
            consumed.update(mapping.hf_param.values())
        owner = native.get_submodule(name.rpartition(".")[0])
        converted = mapping.hf_to_megatron(source, owner)
        assert converted.shape == parameter.shape, name
        with torch.no_grad():
            parameter.copy_(converted)
        converted_back = mapping.megatron_to_hf(parameter.detach(), owner)
        assert not set(converted_back) & set(exported), name
        exported.update(converted_back)
    if "lm_head.weight" in original:
        assert native.lm_head.weight is native.get_input_embeddings().weight
        assert torch.equal(original["lm_head.weight"], original["model.language_model.embed_tokens.weight"])
        exported["lm_head.weight"] = native.lm_head.weight.detach()
        consumed.add("lm_head.weight")
    assert_complete_round_trip(original, consumed, exported)


def build_native(
    config: "PretrainedConfig", reference: torch.nn.Module
) -> tuple[torch.nn.Module, "MegatronMappingRegistry"]:
    """Construct current native providers and strictly convert the full state dict."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.utils import persistent_buffers

    config.dtype = "float32"
    bridge = AutoBridge.from_hf_config(config)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.sequence_parallel = False
    provider.gradient_accumulation_fusion = False
    provider.use_cpu_initialization = False
    provider.params_dtype = torch.float32
    provider.bf16 = False
    provider.fp16 = False
    provider.apply_overrides_and_finalize()
    native = provider.provide().cuda().eval()
    registry = bridge._model_bridge.mapping_registry()
    copy_and_round_trip(native, registry, reference.state_dict(), persistent_buffers(native))
    return native, registry


def export_gradients(native: torch.nn.Module, registry: "MegatronMappingRegistry") -> dict[str, torch.Tensor]:
    """Convert native gradients to publisher layouts for independent comparison."""
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in native.named_parameters():
        if parameter.grad is not None:
            mapping = registry.megatron_to_hf_lookup(name)
            assert mapping is not None, name
            owner = native.get_submodule(name.rpartition(".")[0])
            gradients.update(mapping.megatron_to_hf(parameter.grad, owner))
    return gradients
