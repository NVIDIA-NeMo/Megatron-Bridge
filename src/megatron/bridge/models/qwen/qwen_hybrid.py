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

"""Shared HybridModel layout helpers for Qwen3 and newer models."""

from collections.abc import Sequence
from dataclasses import dataclass

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider


def qwen_pipeline_layer_pattern(
    hybrid_layer_pattern: str,
    pipeline_model_parallel_size: int,
    *,
    account_for_embedding: bool = False,
    account_for_loss: bool = False,
) -> str:
    """Split a Qwen Hybrid pattern across PP stages without separating logical blocks."""
    if pipeline_model_parallel_size <= 0:
        raise ValueError("pipeline_model_parallel_size must be positive")

    main_pattern, separator, mtp_pattern = hybrid_layer_pattern.partition(Symbols.MTP_SEPARATOR)
    if Symbols.PIPE in main_pattern:
        segment_count = main_pattern.count(Symbols.PIPE) + 1
        if segment_count != pipeline_model_parallel_size:
            raise ValueError(
                f"Qwen hybrid_layer_pattern defines {segment_count} pipeline segments, "
                f"but pipeline_model_parallel_size is {pipeline_model_parallel_size}."
            )
        return hybrid_layer_pattern
    if pipeline_model_parallel_size == 1:
        return hybrid_layer_pattern
    if len(main_pattern) % 2:
        raise ValueError("Qwen Hybrid patterns must contain two physical layers per logical block.")

    logical_blocks = [main_pattern[index : index + 2] for index in range(0, len(main_pattern), 2)]
    attention_symbols = {Symbols.ATTENTION, Symbols.GDN}
    mlp_symbols = {Symbols.MLP, Symbols.MOE}
    invalid_blocks = [
        block for block in logical_blocks if block[0] not in attention_symbols or block[1] not in mlp_symbols
    ]
    if invalid_blocks:
        raise ValueError(f"Unsupported Qwen logical blocks in hybrid_layer_pattern: {invalid_blocks}")

    total_pipeline_units = len(logical_blocks) + int(account_for_embedding) + int(account_for_loss)
    units_per_stage, extra_units = divmod(total_pipeline_units, pipeline_model_parallel_size)
    logical_blocks_per_stage = [
        units_per_stage + int(stage_index < extra_units) for stage_index in range(pipeline_model_parallel_size)
    ]
    logical_blocks_per_stage[0] -= int(account_for_embedding)
    logical_blocks_per_stage[-1] -= int(account_for_loss)
    if any(count <= 0 for count in logical_blocks_per_stage):
        raise ValueError(
            "Every pipeline stage must receive at least one Qwen logical block after embedding/loss balancing."
        )

    segments = []
    block_offset = 0
    for block_count in logical_blocks_per_stage:
        segments.append("".join(logical_blocks[block_offset : block_offset + block_count]))
        block_offset += block_count
    if block_offset != len(logical_blocks):
        raise RuntimeError("Failed to assign every Qwen logical block to a pipeline stage.")

    segmented_pattern = Symbols.PIPE.join(segments)
    return segmented_pattern + (separator + mtp_pattern if separator else "")


@dataclass
class QwenHybridModelProvider(HybridModelProvider):
    """HybridModel provider that keeps each Qwen logical block on one PP stage."""

    def finalize(self) -> None:
        if (
            self.hybrid_layer_pattern is not None
            and self.pipeline_model_parallel_size > 1
            and self.pipeline_model_parallel_layout is None
            and self.num_layers_in_first_pipeline_stage is None
            and self.num_layers_in_last_pipeline_stage is None
        ):
            segmented_pattern = qwen_pipeline_layer_pattern(
                self.hybrid_layer_pattern,
                self.pipeline_model_parallel_size,
                account_for_embedding=bool(self.account_for_embedding_in_pipeline_split),
                account_for_loss=bool(self.account_for_loss_in_pipeline_split),
            )
            if segmented_pattern != self.hybrid_layer_pattern:
                self.hybrid_layer_pattern = segmented_pattern
                # The explicit segments already include embedding/loss-aware balancing.
                self.account_for_embedding_in_pipeline_split = False
                self.account_for_loss_in_pipeline_split = False
        super().finalize()


def qwen_attention_symbols(
    num_layers: int,
    linear_attention_freq: int | Sequence[int] | None = None,
) -> list[str]:
    """Translate a logical Qwen attention schedule to HybridModel symbols."""
    if linear_attention_freq is None:
        return [Symbols.ATTENTION] * num_layers

    if isinstance(linear_attention_freq, int):
        if linear_attention_freq <= 0:
            raise ValueError("linear_attention_freq must be positive")
        return [
            Symbols.ATTENTION if (layer_idx + 1) % linear_attention_freq == 0 else Symbols.GDN
            for layer_idx in range(num_layers)
        ]

    linear_attention_pattern = list(linear_attention_freq)
    if len(linear_attention_pattern) != num_layers:
        raise ValueError(
            "linear_attention_freq has "
            f"{len(linear_attention_pattern)} entries, but num_hidden_layers is {num_layers}."
        )
    invalid_values = sorted(set(linear_attention_pattern) - {0, 1})
    if invalid_values:
        raise ValueError(f"Unsupported linear attention pattern values: {invalid_values}. Expected only 0 or 1.")
    return [Symbols.GDN if is_linear else Symbols.ATTENTION for is_linear in linear_attention_pattern]


def qwen_hybrid_layer_pattern(
    num_layers: int,
    *,
    mlp_symbols: str | Sequence[str],
    linear_attention_freq: int | Sequence[int] | None = None,
) -> str:
    """Build a two-physical-layer HybridModel pattern for each logical Qwen block."""
    if isinstance(mlp_symbols, str):
        mlp_pattern = [mlp_symbols] * num_layers
    else:
        mlp_pattern = list(mlp_symbols)
        if len(mlp_pattern) != num_layers:
            raise ValueError(f"MLP pattern has {len(mlp_pattern)} entries, but num_hidden_layers is {num_layers}.")

    invalid_symbols = sorted(set(mlp_pattern) - {Symbols.MLP, Symbols.MOE})
    if invalid_symbols:
        raise ValueError(
            f"Unsupported Qwen MLP symbols: {invalid_symbols}. Expected '{Symbols.MLP}' or '{Symbols.MOE}'."
        )

    attention_pattern = qwen_attention_symbols(num_layers, linear_attention_freq)
    return "".join(
        attention_symbol + mlp_symbol for attention_symbol, mlp_symbol in zip(attention_pattern, mlp_pattern)
    )


def qwen_moe_layer_symbols(
    num_layers: int,
    *,
    decoder_sparse_step: int = 1,
    mlp_only_layers: Sequence[int] = (),
) -> list[str]:
    """Translate Hugging Face Qwen MoE placement fields to Hybrid symbols."""
    if decoder_sparse_step <= 0:
        raise ValueError("decoder_sparse_step must be positive")
    dense_layers = set(mlp_only_layers)
    invalid_layers = sorted(layer_idx for layer_idx in dense_layers if not 0 <= layer_idx < num_layers)
    if invalid_layers:
        raise ValueError(f"mlp_only_layers contains out-of-range indices: {invalid_layers}")
    return [
        Symbols.MOE
        if logical_layer_idx not in dense_layers and (logical_layer_idx + 1) % decoder_sparse_step == 0
        else Symbols.MLP
        for logical_layer_idx in range(num_layers)
    ]


def configure_qwen_hybrid_layers(
    provider: HybridModelProvider,
    *,
    num_logical_layers: int,
    mlp_symbols: str | Sequence[str],
    linear_attention_freq: int | Sequence[int] | None = None,
    mtp_mlp_symbol: str | None = None,
) -> None:
    """Configure main and optional MTP physical layer patterns on a Qwen provider."""
    provider.hybrid_layer_pattern = qwen_hybrid_layer_pattern(
        num_logical_layers,
        mlp_symbols=mlp_symbols,
        linear_attention_freq=linear_attention_freq,
    )
    provider.num_layers = len(provider.hybrid_layer_pattern)

    if mtp_mlp_symbol is not None:
        if mtp_mlp_symbol not in {Symbols.MLP, Symbols.MOE}:
            raise ValueError(
                f"Unsupported Qwen MTP MLP symbol: {mtp_mlp_symbol}. Expected '{Symbols.MLP}' or '{Symbols.MOE}'."
            )
        provider.mtp_hybrid_override_pattern = Symbols.ATTENTION + mtp_mlp_symbol


def qwen_logical_layer_count(hybrid_layer_pattern: str | None) -> int | None:
    """Return the number of logical Qwen blocks encoded by a HybridModel pattern."""
    if not hybrid_layer_pattern:
        return None
    main_pattern = hybrid_layer_pattern.split(Symbols.MTP_SEPARATOR)[0].replace(Symbols.PIPE, "")
    attention_symbols = {Symbols.ATTENTION, Symbols.GDN}
    return sum(symbol in attention_symbols for symbol in main_pattern)


def qwen_physical_layer_indices(logical_layer_idx: int) -> tuple[int, int]:
    """Return attention and MLP/MoE physical indices for one logical Qwen block."""
    attention_layer_idx = 2 * logical_layer_idx
    return attention_layer_idx, attention_layer_idx + 1
