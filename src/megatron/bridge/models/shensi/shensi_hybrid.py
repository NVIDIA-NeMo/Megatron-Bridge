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

"""Shensi decoder stack and HybridModel integration."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.shensi.modeling_shensi import (
    SHENSI_INPUT_IDS,
    SHENSI_LABELS,
    ShensiModel,
    ShensiStreamState,
    get_shensi_decoder_block_spec,
    get_shensi_layer_spec,
    get_shensi_mtp_block_spec,
    pack_shensi_payload,
    record_final_blocks,
    rosa,
    shensi_topk_payload_rows,
    unpack_shensi_payload,
)


def shensi_hybrid_layer_pattern(model_cfg, *, logical_layers_per_stage: Optional[Sequence[int]] = None) -> str:
    """Set the Shensi hybrid layer pattern and pipeline layout.

    Args:
        model_cfg: Shensi model config to update in place.
        logical_layers_per_stage: Optional decoder-layer counts in VPP-major, PP-minor order.

    Returns:
        The pattern written to ``model_cfg.hybrid_layer_pattern``.
    """
    pp_size = int(model_cfg.pipeline_model_parallel_size or 1)
    vp_size = int(getattr(model_cfg, "virtual_pipeline_model_parallel_size", None) or 1)
    num_layers = int(model_cfg.num_layers)
    mtp_layers = int(getattr(model_cfg, "mtp_num_layers", 0) or 0)
    mtp_pattern = Symbols.MTP_SEPARATOR + Symbols.ATTENTION * mtp_layers if mtp_layers else ""

    if pp_size <= 1 and vp_size <= 1:
        pattern = Symbols.ATTENTION * num_layers + mtp_pattern
        model_cfg.hybrid_layer_pattern = pattern
        return pattern

    num_stages = pp_size * vp_size
    if logical_layers_per_stage is None:
        counts = [
            int(get_num_layers_to_build(model_cfg, vp_stage=stage // pp_size, pp_rank=stage % pp_size))
            for stage in range(num_stages)
        ]
    else:
        counts = [int(count) for count in logical_layers_per_stage]
        if len(counts) != num_stages or any(count < 0 for count in counts) or sum(counts) != num_layers:
            raise ValueError(f"Expected {num_stages} nonnegative stage counts summing to {num_layers}, got {counts}.")

    layout: list[list[str]] = []
    segments: list[str] = []
    for stage in range(num_stages):
        entries = ["embedding"] if stage == 0 else []
        entries.extend(["decoder"] * counts[stage])
        if stage == num_stages - 1:
            entries.extend(["mtp"] * mtp_layers)
            entries.append("loss")
        layout.append(entries)
        segments.append(Symbols.ATTENTION * counts[stage])

    model_cfg.pipeline_model_parallel_layout = layout
    pattern = Symbols.PIPE.join(segments) + mtp_pattern
    model_cfg.hybrid_layer_pattern = pattern
    return pattern


class ShensiDecoderStack(TransformerBlock):
    """Shensi decoder stack used as the ``HybridModel`` decoder."""

    def __init__(
        self,
        config,
        submodules: Optional[HybridStackSubmodules] = None,
        *,
        layer_config_list: Optional[Sequence[Any]] = None,
        pp_layer_offset: int = 0,
        pre_process: bool = True,
        post_process: bool = True,
        dtype=None,
        pg_collection=None,
        hash_moe_layer_threshold: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: Optional[str] = None,
        **_ignored: Any,
    ) -> None:
        self.output_contract: Optional[Callable[[Any], Any]] = None
        self.stream_components: Optional[dict] = None
        self.num_attn_res_blocks = int(
            sum(1 for block_type in config.attn_res_block_layer_types if block_type == "block_write_layer")
        )
        pp_rank = None
        if pg_collection is not None and getattr(pg_collection, "pp", None) is not None:
            pp_rank = int(pg_collection.pp.rank())
        vp_stage = _vp_stage_for_layer_offset(config, int(pp_layer_offset), pp_rank)
        spec = get_shensi_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage, pp_rank=pp_rank)
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=False,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            name=name,
        )

    def set_output_contract(self, contract: Optional[Callable[[Any], Any]]) -> None:
        """Attach the model's attention-residual output contract (applied when ``post_process``)."""
        self.output_contract = contract

    def set_stream_components(
        self,
        *,
        hc_head,
        stream_embeddings,
        bank,
        output_attn_res,
        norm,
    ) -> None:
        """Bind the model-level stream components the stack applies at its boundaries."""
        self.stream_components = {
            "hc_head": hc_head,
            "stream_embeddings": stream_embeddings,
            "bank": bank,
            "output_attn_res": output_attn_res,
            "norm": norm,
        }

    def _mask_2d(self, attention_mask, inputs_embeds) -> torch.Tensor:
        batch, seq_len = inputs_embeds.shape[1], inputs_embeds.shape[0]
        tp_size = int(self.tp_group.size()) if self.config.sequence_parallel else 1
        full_seq_len = seq_len * tp_size
        if attention_mask is None:
            return inputs_embeds.new_ones((batch, full_seq_len), dtype=torch.bool)
        mask = attention_mask.bool()
        if mask.dim() != 2:
            raise ValueError(f"Shensi expects a 2D attention mask, got shape {tuple(mask.shape)}")
        if mask.shape == (batch, full_seq_len):
            return mask
        if mask.shape == (full_seq_len, batch):
            return mask.transpose(0, 1)
        raise ValueError(
            f"attention mask shape {tuple(mask.shape)} does not match the activation shape "
            f"({seq_len} local x {tp_size} = {full_seq_len}, batch {batch})"
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        *,
        input_ids=None,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
        **kwargs: Any,
    ):
        """Run the Shensi stream loop over this stage's local layers."""
        del packed_seq_params_by_layout, cp_layout_plan, kwargs
        if self.stream_components is None:
            raise RuntimeError(
                "Shensi stream components are not bound: construct the model through ShensiModel "
                "or call set_stream_components() first."
            )
        bank = self.stream_components["bank"]
        stream_embeddings = self.stream_components["stream_embeddings"]
        hc_head = self.stream_components["hc_head"]
        output_attn_res = self.stream_components["output_attn_res"]
        norm = self.stream_components["norm"]
        input_ids = input_ids if input_ids is not None else SHENSI_INPUT_IDS.get()
        topk_rows = shensi_topk_payload_rows(self.config.index_topk_tokens, self.config.hidden_size)

        if not self.pre_process:
            payload = self.input_tensor if hidden_states is None else hidden_states
            if payload is None:
                raise RuntimeError(
                    "this Shensi stage is not first in the pipeline but received no input tensor: "
                    "the schedule must deliver the previous stage's payload through set_input_tensor"
                )
            hidden_states, state = unpack_shensi_payload(
                payload,
                num_streams=int(self.config.hc_mult),
                num_blocks=self.num_attn_res_blocks,
                topk_rows=topk_rows,
                index_topk=int(self.config.index_topk_tokens),
            )
            inputs_embeds = state.inputs_embeds
            prefix_sum = state.prefix_sum
            blocks = state.blocks
            rosa_ids = state.rosa_ids
            topk_indices = state.topk_indices
            if topk_indices is not None and self.config.sequence_parallel and self.tp_group.size() > 1:
                topk_indices = gather_from_sequence_parallel_region(
                    topk_indices.transpose(0, 1).contiguous(), group=self.tp_group
                ).transpose(0, 1)
            mask = self._mask_2d(attention_mask, inputs_embeds)
            # The bank's token bias is rebuilt from this stage's embeddings; the committed block
            # rows ride the payload.
            bank.initialize(inputs_embeds, self.num_attn_res_blocks)
            streams = bank.encode_streams(state.streams)
            wide_first = False
        else:
            inputs_embeds = hidden_states
            mask = self._mask_2d(attention_mask, inputs_embeds)
            streams = stream_embeddings(inputs_embeds, input_ids)
            blocks = bank.initialize(inputs_embeds, self.num_attn_res_blocks)
            prefix_sum = torch.zeros_like(inputs_embeds)
            rosa_ids = None
            topk_indices = None
            wide_first = True
            if input_ids is not None and self.config.rosa_layer_ids:
                rosa_ids = rosa(
                    input_ids,
                    attention_mask=mask if input_ids.shape == mask.shape else None,
                    eos_token_id=self.config.rosa_reset_token_id,
                )
                if self.config.sequence_parallel and self.tp_group.size() > 1:
                    rank, local = self.tp_group.rank(), inputs_embeds.shape[0]
                    rosa_ids = rosa_ids[:, rank * local : (rank + 1) * local]

        # Shensi owns the whole layer body, so Megatron-Core's selective recompute (which only
        # wraps its own attention/MLP submodules) cannot reach it: treat `selective` like `full`
        # and checkpoint one stream-loop iteration.
        use_checkpoint = (
            self.training
            and torch.is_grad_enabled()
            and getattr(self.config, "recompute_granularity", "none") in ("full", "selective")
        )

        # Labels ride into the checkpoint region as an explicit argument: the context variable is
        # reset before checkpoint recompute runs (during backward), and the ROSA auxiliary loss
        # must attach identically in both passes.
        labels = SHENSI_LABELS.get()

        last_local = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            keep_wide = self.post_process and index == last_local
            if use_checkpoint:
                hidden_states, streams, prefix_sum, blocks, topk_indices = torch.utils.checkpoint.checkpoint(
                    self._checkpointed_layer,
                    layer,
                    bank,
                    inputs_embeds,
                    streams,
                    prefix_sum,
                    blocks,
                    topk_indices,
                    mask,
                    input_ids,
                    rosa_ids,
                    labels,
                    wide_first and index == 0,
                    keep_wide,
                    use_reentrant=False,
                )
                continue
            layer_streams = streams if (wide_first and index == 0) else bank.decode_streams(streams)
            hidden_states, prefix_sum, blocks, topk_indices = layer(
                layer_streams,
                attention_mask=mask,
                input_ids=input_ids,
                partial_block=prefix_sum,
                blocks=blocks,
                prev_topk_indices=topk_indices,
                rosa_token_ids=rosa_ids,
                labels=labels,
            )
            streams = hidden_states if keep_wide else bank.encode_streams(hidden_states)

        if self.post_process:
            # Encode the completed rows, flush the last partial block, and read the bank once.
            residual = bank.finalize(blocks, prefix_sum, self.num_attn_res_blocks)
            record_final_blocks(residual)
            hidden_states = hc_head(streams)
            hidden_states, _ = output_attn_res(
                torch.zeros_like(prefix_sum),
                hidden_states,
                residual,
                num_blocks=1 + self.num_attn_res_blocks,
            )
            return norm(hidden_states)

        packed_topk = topk_indices
        if packed_topk is not None and self.config.sequence_parallel and self.tp_group.size() > 1:
            rank, local = self.tp_group.rank(), inputs_embeds.shape[0]
            packed_topk = packed_topk[:, rank * local : (rank + 1) * local]
        state = ShensiStreamState(
            inputs_embeds=inputs_embeds,
            streams=hidden_states,
            prefix_sum=prefix_sum,
            blocks=blocks,
            rosa_ids=rosa_ids,
            topk_indices=packed_topk,
        )
        # The activation row only carries the stage shape; the state rows carry the streams.
        stage_output = hidden_states.mean(dim=2) if hidden_states.dim() == 4 else hidden_states
        return pack_shensi_payload(
            stage_output,
            state,
            topk_rows=topk_rows,
            index_topk=int(self.config.index_topk_tokens),
        )

    def _checkpointed_layer(
        self,
        layer,
        bank,
        inputs_embeds,
        streams,
        prefix_sum,
        blocks,
        topk_indices,
        mask,
        input_ids,
        rosa_ids,
        labels,
        wide_streams,
        keep_wide,
    ):
        """One stream-loop iteration, checkpointed as a unit under full recompute."""
        layer_streams = streams if wide_streams else bank.decode_streams(streams)
        hidden_states, prefix_sum, blocks, topk_indices = layer(
            layer_streams,
            attention_mask=mask,
            input_ids=input_ids,
            partial_block=prefix_sum,
            blocks=blocks,
            prev_topk_indices=topk_indices,
            rosa_token_ids=rosa_ids,
            labels=labels,
        )
        streams = hidden_states if keep_wide else bank.encode_streams(hidden_states)
        return hidden_states, streams, prefix_sum, blocks, topk_indices


def _vp_stage_for_layer_offset(config, pp_layer_offset: int, pp_rank: Optional[int]) -> Optional[int]:
    """Return the virtual pipeline stage owning ``pp_layer_offset``."""
    vp_size = int(getattr(config, "virtual_pipeline_model_parallel_size", None) or 1)
    if vp_size <= 1:
        return None
    for vp_stage in range(vp_size):
        if int(get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)) == pp_layer_offset:
            return vp_stage
    return None


def get_shensi_hybrid_stack_spec(
    config,
    *,
    use_transformer_engine: Optional[bool] = None,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> ModuleSpec:
    """Return the ``HybridModel`` stack spec and the MTP template."""
    if use_transformer_engine is None:
        use_transformer_engine = getattr(config, "transformer_impl", "local") == "transformer_engine"
    num_layers = int(config.num_layers)
    mtp_layer_recipe = get_shensi_layer_spec(
        use_te=use_transformer_engine,
        config=config,
        layer_number=num_layers,
        is_mtp_layer=True,
    )
    mtp_block_spec = None
    if int(getattr(config, "mtp_num_layers", 0) or 0):
        mtp_block_spec = get_shensi_mtp_block_spec(
            config, use_transformer_engine=use_transformer_engine, vp_stage=vp_stage, pp_rank=pp_rank
        )
        for layer_spec in getattr(mtp_block_spec, "layer_specs", None) or []:
            if hasattr(layer_spec.submodules, "mtp_model_layer"):
                layer_spec.submodules.mtp_model_layer = None

    return ModuleSpec(
        module=ShensiDecoderStack,
        submodules=HybridStackSubmodules(
            attention_layer=mtp_layer_recipe,
            mtp_block_spec=mtp_block_spec,
        ),
    )


def build_shensi_model(
    config,
    pg_collection,
    *,
    vocab_size: int,
    max_sequence_length: int,
    vp_stage: Optional[int] = None,
    pre_process: Optional[bool] = None,
    post_process: Optional[bool] = None,
    **hybrid_kwargs: Any,
) -> ShensiModel:
    """Build one Shensi pipeline stage.

    Args:
        config: Shensi transformer config.
        pg_collection: Process groups for this stage.
        vocab_size: Padded vocabulary size.
        max_sequence_length: Maximum sequence length.
        vp_stage: Virtual pipeline stage index.
        pre_process: Whether this stage owns the embedding.
        post_process: Whether this stage owns the output layer.
        **hybrid_kwargs: Extra ``HybridModel`` arguments.

    Returns:
        The constructed Shensi stage.
    """
    if getattr(config, "hybrid_layer_pattern", None) is None:
        shensi_hybrid_layer_pattern(config)
    stack_spec = get_shensi_hybrid_stack_spec(config, vp_stage=vp_stage, pp_rank=int(pg_collection.pp.rank()))
    vp_size = getattr(config, "virtual_pipeline_model_parallel_size", None)
    if pre_process is None:
        pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
    if post_process is None:
        post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)
    return ShensiModel(
        config=config,
        hybrid_stack_spec=stack_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        hybrid_layer_pattern=config.hybrid_layer_pattern,
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
        **hybrid_kwargs,
        **logit_dtype_kwarg(HybridModel, hybrid_kwargs.pop("logit_dtype", None)),
    )


__all__ = [
    "ShensiDecoderStack",
    "build_shensi_model",
    "get_shensi_hybrid_stack_spec",
    "shensi_hybrid_layer_pattern",
]
