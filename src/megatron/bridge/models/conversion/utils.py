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


from typing import Optional

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from rich.table import Table


def weights_verification_table(bridge, megatron_model) -> Table:
    """
    Returns a table comparing weights between a Hugging Face model and a Megatron-LM model.

    Args:
        bridge (AutoBridge): The bridge object containing model information.
        megatron_model: The Megatron-LM model instance.

    Returns:
        Table: A rich Table object with the comparison.
    """
    table = Table(title="Hugging Face Weights Verification")
    table.add_column("Weight Name", style="cyan")
    table.add_column("Shape")
    table.add_column("DType")
    table.add_column("Device")
    table.add_column("Matches Original", justify="center")

    # Check each weight against the original HF-model
    for name, param in bridge(megatron_model, show_progress=True):
        original_param = bridge.hf_pretrained.state[name]
        table.add_row(
            name,
            str(tuple(param.shape)),
            str(param.dtype).replace("torch.", ""),
            str(param.device),
            "✅" if torch.allclose(param, original_param.to(param.device), atol=1e-6) else "❌",
        )

    return table


# Adapted from MCore for few additional changes, consider upstream later.
# Update argument to able to take in pipeline_rank
def get_transformer_layer_offset(config: TransformerConfig, pipeline_rank: int = 0, vp_stage: Optional[int] = None):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    from megatron.core.transformer.enums import LayerType

    if config.pipeline_model_parallel_size > 1:
        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        elif (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0 if config.num_layers_in_first_pipeline_stage is None else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0 if config.num_layers_in_last_pipeline_stage is None else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers - num_layers_in_first_pipeline_stage - num_layers_in_last_pipeline_stage
            )

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert vp_stage is not None, "vp_stage must be provided if virtual pipeline model parallel size is set"

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage = middle_num_layers // vp_size

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pipeline_rank == 0:
                    offset = vp_stage * total_virtual_chunks
                else:
                    offset = (
                        vp_stage * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + (pipeline_rank - 1)
                        * (num_layers_per_vritual_model_chunk_in_middle_pipeline_stage // middle_pipeline_stages)
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                middle_pipeline_rank = (
                    pipeline_rank if config.num_layers_in_first_pipeline_stage is None else pipeline_rank - 1
                )

                if pipeline_rank == 0:
                    offset = 0
                else:
                    offset = (middle_pipeline_rank * num_layers_per_pipeline_rank) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert vp_stage is not None, "vp_stage must be provided if virtual pipeline model parallel size is set"

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_stage * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split:
                    raise NotImplementedError()
            else:
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split:
                    raise NotImplementedError()
    else:
        offset = 0
    return offset
