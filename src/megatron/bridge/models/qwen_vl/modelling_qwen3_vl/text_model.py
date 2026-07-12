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


"""
Copied from https://github.com/Thaurun/mbridge/blob/4462d1e284626d2ed9d3e3e
3e5a40f2ee42a2c74/mbridge/models/qwen3_vl/gpt_model.py
"""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from typing import Literal, Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor
from torch import Tensor

from megatron.bridge.models.hybrid.hybrid_provider import get_default_hybrid_stack_spec
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_q_cu_seqlens


def _get_mtp_packed_seq_params(packed_seq_params: PackedSeqParams | None) -> PackedSeqParams | None:
    """Use physical padded offsets for MTP token rolling without changing attention metadata."""
    if packed_seq_params is None or packed_seq_params.cu_seqlens_q_padded is None:
        return packed_seq_params

    _, cu_seqlens_q = get_packed_seq_q_cu_seqlens(packed_seq_params)
    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
    if cu_seqlens_kv is None:
        cu_seqlens_kv = cu_seqlens_q
    return replace(
        packed_seq_params,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
    )


class Qwen3VLGPTModel(GPTModel):
    """Qwen3-VL GPT model with vision-language capabilities."""

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope", "mrope", "none"] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )

        # rebuild rope
        self.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            cp_group=self.pg_collection.cp,
        )
        self.mrope_section = self.config.mrope_section
        assert self.mrope_section is not None, (
            "mrope require mrope_section setting, but we got None from TransformerConfig"
        )

        # rebuild the transformer block
        self.decoder = Qwen3VLTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )

    def tie_embeddings_and_output_weights_state_dict(
        self,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
        metadata: dict | None = None,
    ) -> None:
        """Tie embedding/output checkpoint entries for Qwen3-VL MTP pipeline stages."""
        if getattr(self, "mtp_process", False) and not self.pre_process:
            sharded_state_dict.pop(output_layer_weight_key, None)
            return

        super().tie_embeddings_and_output_weights_state_dict(
            sharded_state_dict,
            output_layer_weight_key,
            first_stage_word_emb_key,
            metadata if metadata is not None else {},
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

         forward pass is overridden to add support for deepstack visual embeddings.

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # `_preprocess` can optionally return an extra fused cos/sin buffer (for
        # flash decode). Match the upstream GPTModel handling to avoid unpack
        # errors when six values are returned.
        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ) = preproc_output[:5]

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            # Qwen3 VL blocks do not currently consume fused cos/sin; pass along
            # the standard components only.
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        # MTP calls self.embedding directly (bypassing the manual SP scatter that
        # model.py does for the combined VL embeddings). Temporarily wrap the embedding
        # to apply the SP scatter so its output shape matches hidden_states.
        # We write to self.__dict__ directly to bypass nn.Module.__setattr__'s type
        # check, which rejects non-Module values for registered child modules.
        _shadow_embedding = False
        if self.mtp_process and self.config.sequence_parallel:
            _original_embedding = self.embedding

            def _sp_scatter_embedding(input_ids, position_ids):
                out = _original_embedding(input_ids=input_ids, position_ids=position_ids)
                return tensor_parallel.scatter_to_sequence_parallel_region(out, group=self.pg_collection.tp)

            _sp_scatter_embedding.word_embeddings = _original_embedding.word_embeddings
            self.__dict__["embedding"] = _sp_scatter_embedding
            _shadow_embedding = True

        postprocess_packed_seq_params = (
            _get_mtp_packed_seq_params(packed_seq_params) if self.mtp_process else packed_seq_params
        )
        result = self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=postprocess_packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

        if _shadow_embedding:
            del self.__dict__["embedding"]

        return result


class Qwen3VLHybridStack(HybridStack):
    """Hybrid stack that injects Qwen DeepStack embeddings after logical blocks."""

    def set_multimodal_context(
        self,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
    ) -> None:
        """Set per-forward multimodal inputs consumed by :meth:`forward`."""
        self._qwen_visual_pos_masks = visual_pos_masks
        self._qwen_deepstack_visual_embeds = deepstack_visual_embeds

    def clear_multimodal_context(self) -> None:
        """Release references to per-forward multimodal tensors."""
        self._qwen_visual_pos_masks = None
        self._qwen_deepstack_visual_embeds = None

    @staticmethod
    def _deepstack_process(
        hidden_states: Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> Tensor:
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        return hidden_states.transpose(0, 1).contiguous()

    def _maybe_add_deepstack_embedding(
        self,
        hidden_states: Tensor,
        local_layer_idx: int,
        visual_pos_masks: torch.Tensor,
        deepstack_visual_embeds: tuple[torch.Tensor, ...],
    ) -> Tensor:
        if self.layer_type_list[local_layer_idx] not in {Symbols.MLP, Symbols.MOE}:
            return hidden_states
        logical_layer_idx = (self.layers[local_layer_idx].layer_number - 1) // 2
        if logical_layer_idx >= len(deepstack_visual_embeds):
            return hidden_states
        hidden_states = self._deepstack_process(
            hidden_states,
            visual_pos_masks,
            deepstack_visual_embeds[logical_layer_idx],
        )
        return make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

    def _checkpointed_forward_with_deepstack(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams | None,
        padding_mask: Tensor | None,
        visual_pos_masks: torch.Tensor,
        deepstack_visual_embeds: tuple[torch.Tensor, ...],
    ) -> Tensor:
        use_inner_quantization_context = bool(
            (self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed) or self.config.fp4
        )

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                padding_mask,
                visual_pos_masks,
                *visual_embeds,
            ):
                for local_layer_idx in range(start, end):
                    layer = self.layers[local_layer_idx]
                    if use_inner_quantization_context and self.config.fp8:
                        quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                    elif use_inner_quantization_context and self.config.fp4:
                        quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                    else:
                        quantization_context = nullcontext()

                    with quantization_context:
                        if isinstance(layer, TransformerLayer):
                            hidden_states, _ = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                inference_context=None,
                                packed_seq_params=packed_seq_params,
                                padding_mask=padding_mask,
                            )
                        else:
                            hidden_states = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=None,
                                packed_seq_params=packed_seq_params,
                            )
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    hidden_states = self._maybe_add_deepstack_embedding(
                        hidden_states,
                        local_layer_idx,
                        visual_pos_masks,
                        visual_embeds,
                    )
                return hidden_states

            return custom_forward

        def run_chunk(start: int, end: int, use_checkpoint: bool) -> None:
            nonlocal hidden_states
            forward_func = custom(start, end)
            args = (
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                padding_mask,
                visual_pos_masks,
                *deepstack_visual_embeds,
            )
            if use_checkpoint:
                if self.config.fp8 or self.config.fp4:
                    from megatron.core.extensions.transformer_engine import te_checkpoint

                    hidden_states = te_checkpoint(
                        forward_func,
                        self.config.distribute_saved_activations,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        self.pg_collection.tp,
                        *args,
                    )
                else:
                    hidden_states = tensor_parallel.checkpoint(
                        forward_func,
                        self.config.distribute_saved_activations,
                        *args,
                    )
            else:
                hidden_states = forward_func(*args)

        if self.config.recompute_method == "uniform":
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                chunk_end = min(
                    layer_idx + self.config.recompute_num_layers,
                    self.num_layers_per_pipeline_rank,
                )
                run_chunk(layer_idx, chunk_end, True)
                layer_idx = chunk_end
        elif self.config.recompute_method == "block":
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                use_checkpoint = (
                    recompute_skip_num_layers
                    <= layer_idx
                    < self.config.recompute_num_layers + recompute_skip_num_layers
                )
                run_chunk(layer_idx, layer_idx + 1, use_checkpoint)
        else:
            raise ValueError(f"Unsupported recompute method: {self.config.recompute_method}")

        return hidden_states

    def forward(
        self,
        hidden_states: Tensor | WrappedTensor,
        attention_mask: Tensor,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: Tensor | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Run the Hybrid stack with mRoPE and optional DeepStack injection."""
        visual_pos_masks = getattr(self, "_qwen_visual_pos_masks", None)
        deepstack_visual_embeds = tuple(getattr(self, "_qwen_deepstack_visual_embeds", None) or ())
        if not deepstack_visual_embeds:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )

        if not self.pre_process:
            raise ValueError("DeepStack embeddings must be consumed on the first language pipeline stage.")
        if visual_pos_masks is None:
            raise ValueError("visual_pos_masks is required when DeepStack embeddings are provided.")

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        if inference_context and inference_context.is_static_batching():
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset

        if (
            (self.config.cuda_graph_impl == "local" or self.config.flash_decode)
            and inference_context
            and inference_context.is_static_batching()
            and InferenceMode.is_active()
        ):
            current_batch_size = hidden_states.shape[1]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            sequence_len_offset = None

        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()
        with outer_fp8_context:
            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward_with_deepstack(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    packed_seq_params,
                    padding_mask,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
            else:
                for local_layer_idx, layer in enumerate(self.layers):
                    if self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed:
                        quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                    elif self.config.fp4:
                        quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                    else:
                        quantization_context = nullcontext()
                    with quantization_context:
                        if isinstance(layer, TransformerLayer):
                            hidden_states, _ = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=inference_context,
                                rotary_pos_emb=rotary_pos_emb,
                                sequence_len_offset=sequence_len_offset,
                                packed_seq_params=packed_seq_params,
                                padding_mask=padding_mask,
                            )
                        else:
                            hidden_states = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=inference_context,
                                packed_seq_params=packed_seq_params,
                            )
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    hidden_states = self._maybe_add_deepstack_embedding(
                        hidden_states,
                        local_layer_idx,
                        visual_pos_masks,
                        deepstack_visual_embeds,
                    )

        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)
        return make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )


def get_qwen3_vl_hybrid_stack_spec(config: TransformerConfig) -> ModuleSpec:
    """Return the default Hybrid stack with Qwen's absolute mRoPE attention."""
    stack_spec = deepcopy(get_default_hybrid_stack_spec(config))
    stack_spec.module = Qwen3VLHybridStack
    stack_spec.submodules.attention_layer.submodules.self_attention.module = Qwen3VLSelfAttention
    return stack_spec


class Qwen3VLHybridModel(HybridModel):
    """Qwen3-VL language model implemented on top of HybridModel."""

    def __init__(
        self,
        config: TransformerConfig,
        hybrid_stack_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        hybrid_layer_pattern: str,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        scatter_embedding_sequence_parallel: bool = False,
        seq_len_interpolation_factor: float | None = None,
        vp_stage: int | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_layer_pattern=hybrid_layer_pattern,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="none",
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        self.position_embedding_type = "mrope"
        self.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            cp_group=self.pg_collection.cp,
        )
        self.mrope_section = self.config.mrope_section
        if self.mrope_section is None:
            raise ValueError("mrope_section must be configured for Qwen3 multimodal models.")

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor | None = None,
        labels: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        extra_block_kwargs: dict | None = None,
        runtime_gather_output: bool | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        loss_mask: Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> Tensor:
        """Run HybridModel with explicit multimodal positions and DeepStack inputs."""
        if position_ids is None:
            raise ValueError("Qwen3 multimodal HybridModel requires explicit position_ids.")
        extra_block_kwargs = dict(extra_block_kwargs or {})
        padding_mask = extra_block_kwargs.pop("padding_mask", None)
        if extra_block_kwargs or kwargs:
            unexpected = sorted(set(extra_block_kwargs) | set(kwargs))
            raise TypeError(f"Unexpected Qwen3 multimodal language-model arguments: {unexpected}")

        self.rotary_pos_emb.set_forward_context(
            position_ids,
            self.mrope_section,
        )
        self.decoder.set_multimodal_context(
            visual_pos_masks,
            deepstack_visual_embeds,
        )
        position_embedding_type = self.position_embedding_type
        self.position_embedding_type = "rope"
        try:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_context=inference_context,
                runtime_gather_output=runtime_gather_output,
                inference_params=inference_params,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )
        finally:
            self.position_embedding_type = position_embedding_type
            self.rotary_pos_emb.clear_forward_context()
            self.decoder.clear_multimodal_context()
