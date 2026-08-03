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

from collections.abc import Callable

import torch
import torch.nn as nn
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule


ModelList = list[MegatronModule]
ModelHook = Callable[[ModelList], ModelList | None]


class LinearForLastLayer(nn.Linear):
    """Replicated projection head compatible with Megatron output-layer calls."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        sequence_parallel: bool,
        bias: bool = False,
        dropout: float = 0.0,
        *,
        output_in_fp32: bool = True,
        tp_group: torch.distributed.ProcessGroup | None = None,
        init_method: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
        perform_initialization: bool = True,
    ) -> None:
        """Initialize a replicated final projection.

        Args:
            input_size: Hidden dimension of the transformer output.
            output_size: Output dimension of the projection head.
            sequence_parallel: Whether to gather sequence-parallel activations.
            bias: Whether to add a trainable bias.
            dropout: Dropout probability applied before the projection.
            output_in_fp32: Whether to cast the projection output to FP32.
            tp_group: Tensor-parallel process group used for sequence gathering.
            init_method: Optional weight initializer for fresh-model construction.
            perform_initialization: Whether to initialize the head parameters.
        """
        self._init_method = init_method
        self.perform_initialization = perform_initialization
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = sequence_parallel
        self.dropout = nn.Dropout(dropout)
        self.output_in_fp32 = output_in_fp32
        self.tp_group = tp_group
        if sequence_parallel:
            setattr(self.weight, "sequence_parallel", True)
            if self.bias is not None:
                setattr(self.bias, "sequence_parallel", True)

    def reset_parameters(self) -> None:
        """Apply the configured initializer, including after meta-device materialization."""
        if not getattr(self, "perform_initialization", True):
            return
        init_method = getattr(self, "_init_method", None)
        if init_method is None:
            super().reset_parameters()
        else:
            init_method(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Run the final projection and return Megatron-style ``(output, bias)``."""
        del weight, runtime_gather_output
        logits = super().forward(self.dropout(input_))
        if self.output_in_fp32:
            logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits,
                tensor_parallel_output_grad=False,
                group=self.tp_group,
            )
        return logits, None


def create_value_head_hook(hidden_size: int, sequence_parallel: bool, output_size: int = 1) -> ModelHook:
    """Create a pre-wrap hook that replaces the final pipeline-stage output head.

    Args:
        hidden_size: Hidden dimension of the transformer output.
        sequence_parallel: Whether the model uses sequence parallelism.
        output_size: Number of outputs produced by the final head.

    Returns:
        A model hook suitable for external trainer provider construction.
    """
    return _create_last_layer_hook(
        hidden_size=hidden_size,
        sequence_parallel=sequence_parallel,
        output_size=output_size,
        output_layer_path="output_layer",
        bias=False,
        dropout=0.0,
        output_in_fp32=True,
    )


def _create_last_layer_hook(
    *,
    hidden_size: int,
    sequence_parallel: bool,
    output_size: int,
    output_layer_path: str,
    bias: bool,
    dropout: float,
    output_in_fp32: bool,
) -> ModelHook:
    def hook(model: ModelList | MegatronModule) -> ModelList:
        model_chunks = ensure_model_list(model)
        model_post_process: list[bool] = []
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            for vp_stage in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                model_post_process.append(
                    parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
                )
        else:
            model_post_process.append(parallel_state.is_pipeline_last_stage())

        if len(model_post_process) != len(model_chunks):
            raise ValueError(
                "Model list length and pipeline post-process list length must match. "
                f"Got {len(model_chunks)} model chunks and {len(model_post_process)} post-process flags."
            )

        for index, model_chunk in enumerate(model_chunks):
            if model_post_process[index]:
                output_parent = model_chunk
                path_parts = output_layer_path.split(".")
                for part in path_parts[:-1]:
                    output_parent = getattr(output_parent, part)
                setattr(
                    output_parent,
                    path_parts[-1],
                    LinearForLastLayer(
                        input_size=hidden_size,
                        output_size=output_size,
                        sequence_parallel=sequence_parallel,
                        bias=bias,
                        dropout=dropout,
                        output_in_fp32=output_in_fp32,
                        tp_group=getattr(getattr(model_chunk, "pg_collection", None), "tp", None),
                    ),
                )

        return model_chunks

    return hook


def ensure_model_list(model: ModelList | MegatronModule) -> ModelList:
    """Normalize a single model chunk or a model-chunk list to a list."""
    return model if isinstance(model, list) else [model]
