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

"""Shensi on Megatron-Core's ``HybridModel``: decoder stack, stack spec and pipeline pattern.

The container, the layer pattern and the MTP machinery are Megatron-Core's; the decoder is the
bridge-side `ShensiDecoderStack` because ``HybridStack`` dispatches on MCore's own layer-config classes
over a closed symbol table and cannot build a model-defined layer type. The stack keeps the GPT-style
layer layout, so the attention-residual pipeline plan and the checkpoint paths are unaffected.

Every shensi layer is the neutral ``*`` (attention) symbol: the stack builds its layers from its own
spec list, and only the container reads the pattern, for layer accounting. A trailing ``/`` section
gives each MTP head one symbol, and for PP > 1 the stage split is written to
``pipeline_model_parallel_layout`` from MCore's own per-stage layer counts, so the container, the stack
and the attention-residual stage plan see the same layout.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.shensi.modeling_shensi import (
    ShensiModel,
    get_shensi_decoder_block_spec,
    get_shensi_layer_spec,
    get_shensi_mtp_block_spec,
    publish_hash_input_ids,
    reset_hash_input_ids,
)


def shensi_hybrid_layer_pattern(model_cfg, *, logical_layers_per_stage: Optional[Sequence[int]] = None) -> str:
    """Set ``hybrid_layer_pattern`` (and the pipeline layout for PP > 1) in place.

    The main pattern is one ``*`` per decoder layer with ``|`` separators at the pipeline stages; a
    trailing ``/`` section gives each MTP depth its own ``*``. For PP > 1 the stage split is written
    to ``pipeline_model_parallel_layout`` (which the attention-residual stage plan also reads) using
    MCore's own per-stage layer counts, so both sides see the same layout.

    Args:
        model_cfg: Shensi model config to configure in place.
        logical_layers_per_stage: Optional per-stage decoder-layer counts in VPP-major, PP-minor
            order. Defaults to MCore's own per-stage counts.

    Returns:
        The pattern that was written to ``model_cfg.hybrid_layer_pattern``.
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
    """Bridge-side shensi decoder stack behind ``HybridModel``'s stack contract.

    ``HybridModel`` builds its decoder with hybrid arguments (per-layer configs, the pipeline layer
    offset, the hash-MoE threshold) and calls it with two layout arguments; this stack consumes none of
    them and builds the shensi layer specs instead, sliced per stage by the block itself.
    """

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
        """TransformerBlock forward; hybrid-only layout arguments are accepted and ignored."""
        token = publish_hash_input_ids(input_ids) if input_ids is not None else None
        try:
            hidden_states = super().forward(hidden_states, attention_mask, **kwargs)
        finally:
            if token is not None:
                reset_hash_input_ids(token)
        if self.post_process and self.output_contract is not None:
            hidden_states = self.output_contract(hidden_states)
        return hidden_states


def _vp_stage_for_layer_offset(config, pp_layer_offset: int, pp_rank: Optional[int]) -> Optional[int]:
    """Virtual pipeline stage owning ``pp_layer_offset``; ``None`` outside a VP run.

    ``HybridModel`` hands the stack its segment's per-layer configs and that segment's global layer
    offset, but not the virtual stage index the block needs to slice its own specs.
    """
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
    use_transformer_engine: bool = True,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> ModuleSpec:
    """Stack spec for ``HybridModel``: the shensi stack plus the MTP template of the pattern.

    The MTP inner model layer is built by MCore from this spec's ``attention_layer`` recipe (the
    symbol that the pattern's MTP section repeats), so that recipe is the shensi layer variant the MTP
    head uses - the hash MLP only when the last decoder layer is a hash layer - and the MTP block
    spec's own ``mtp_model_layer`` is cleared, because the hybrid path builds it from the pattern.
    """
    num_layers = int(config.num_layers)
    n_hash = int(config.moe_n_hash_layers)
    mtp_layer_recipe = get_shensi_layer_spec(
        use_te=use_transformer_engine,
        config=config,
        layer_number=num_layers + 1,
        is_hash_layer=(num_layers - 1) < n_hash,
    )
    mtp_block_spec = None
    if int(getattr(config, "mtp_num_layers", 0) or 0):
        mtp_block_spec = get_shensi_mtp_block_spec(
            config, use_transformer_engine=use_transformer_engine, vp_stage=vp_stage, pp_rank=pp_rank
        )
        # ``get_shensi_mtp_block_spec`` returns the block submodules (what the GPT path hands to
        # ``MultiTokenPredictionBlock``); the hybrid path builds the MTP inner model layer from the
        # pattern, so the per-layer recipe must not pin its own ``mtp_model_layer``.
        for layer_spec in getattr(mtp_block_spec, "layer_specs", None) or []:
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
    """Build one Shensi stage on ``HybridModel``.

    Shared by the provider and the model builder so both paths construct the same model: the pipeline
    pattern and the stack spec are derived here and the remaining keyword arguments go to
    ``HybridModel`` (``logit_dtype`` through the version-guarded helper).

    Args:
        config: Shensi transformer config; the layer pattern is written in place.
        pg_collection: Process groups used for the stage's construction.
        vocab_size: Vocabulary size, already padded.
        max_sequence_length: Sequence length for the stage.
        vp_stage: Virtual pipeline stage index.
        pre_process: Whether this stage owns the embedding; defaults to this rank's pipeline position.
        post_process: Whether this stage owns the output layer; defaults to this rank's pipeline position.
        **hybrid_kwargs: ``HybridModel`` keyword arguments (rope, output layer, ...).

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
