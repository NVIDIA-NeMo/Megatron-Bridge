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

import os
from contextlib import contextmanager
from typing import Iterator

import torch

from megatron.bridge.utils.common_utils import get_local_rank_preinit


def _cuda_is_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _select_distributed_backend(*, use_cpu_initialization: bool | None) -> str:
    if use_cpu_initialization or not _cuda_is_available():
        return "gloo"
    return "nccl"


def _initialize_default_process_group(*, backend: str) -> None:
    os.environ["RANK"] = os.environ.get("RANK", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    if backend == "nccl":
        torch.cuda.set_device(get_local_rank_preinit())

    torch.distributed.init_process_group(backend)


def _disable_cpu_offloading_for_cpu_only_initialization(provider: object) -> None:
    for attr in (
        "cpu_offloading",
        "cpu_offloading_activations",
        "cpu_offloading_weights",
        "cpu_offloading_double_buffering",
    ):
        if hasattr(provider, attr):
            setattr(provider, attr, False)


def _disable_te_only_features_for_cpu_only_initialization(provider: object) -> None:
    _disable_cpu_offloading_for_cpu_only_initialization(provider)
    if hasattr(provider, "persist_layer_norm"):
        setattr(provider, "persist_layer_norm", False)


@contextmanager
def _disable_te_cpu_offload_context_for_cpu_only_initialization(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    try:
        from megatron.core.transformer import transformer_block
    except ImportError:
        yield
        return

    get_cpu_offload_context = getattr(transformer_block, "get_cpu_offload_context", None)
    if get_cpu_offload_context is None:
        yield
        return

    transformer_block.get_cpu_offload_context = None
    try:
        yield
    finally:
        transformer_block.get_cpu_offload_context = get_cpu_offload_context
