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

"""Provider fields and model construction for the Bridge-local Qwen4-Exp model."""

import contextlib
from dataclasses import dataclass
from typing import Callable

import torch
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.qwen.modeling_qwen4_exp.layer_specs import get_qwen4_exp_block_spec
from megatron.bridge.models.qwen.modeling_qwen4_exp.model import Qwen4ExpGPTModel
from megatron.bridge.utils import fusions
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


@dataclass
class Qwen4ExpModelProvider(GPTModelProvider):
    """Language-model provider without a model-specific Megatron-Core dependency.

    Fields mirror the Qwen4-Exp checkpoint and are included in Bridge config
    serialization. Call ``finalize`` after changing model or parallelism fields.
    """

    transformer_layer_spec: ModuleSpec | Callable[..., TransformerBlockSubmodules] = get_qwen4_exp_block_spec
    linear_attention_output_gate_activation: str | None = None
    mhc_variant: str = "gated_residual"
    mhc_gated_residual_rank: int | None = None
    qsa_indexer_n_heads: int | None = None
    qsa_indexer_kv_heads: int = 1
    qsa_indexer_head_dim: int | None = None
    qsa_indexer_budget: int | None = None
    qsa_indexer_compress_ratio: int | None = None
    qsa_force_sparse: bool = False
    ple_layer_ids: list[int] | None = None
    ple_embed_dim: int | None = None
    ple_conv_kernel_size: int = 4
    ple_ngram_size: int = 3
    ple_heads_per_ngram: int = 8
    ple_ngram_vocab_size_base: int = 20_000_000
    ple_ngram_vocab_divisible_by: int = 128
    ple_seed: int = 1234
    ple_eos_token_id: int | None = None
    ple_unigram_vocab_size: int | None = None

    def _validate_qwen4_exp(self) -> None:
        if self.enable_mhc_connections:
            raise ValueError("Disable Core mHC expansion/contraction for the local Qwen4-Exp model.")
        if self.pipeline_model_parallel_size != 1 or self.context_parallel_size != 1:
            raise NotImplementedError("Qwen4-Exp currently requires PP=CP=1.")
        if self.virtual_pipeline_model_parallel_size is not None:
            raise NotImplementedError("Qwen4-Exp virtual pipeline stages are not supported.")
        if self.cuda_graph_impl != "none" or self.recompute_granularity is not None:
            raise NotImplementedError("Qwen4-Exp CUDA graphs and activation recomputation are not supported yet.")
        if self.mhc_recompute_layer_num is not None:
            raise NotImplementedError("Core mHC recomputation is not supported by Qwen4-Exp gated residuals.")
        if self.mtp_num_layers or self.fp8 or self.use_transformer_engine_full_layer_spec:
            raise NotImplementedError("Qwen4-Exp MTP, FP8, and full-TE layers are not supported yet.")
        if (
            self.mhc_variant != "gated_residual"
            or not self.mhc_gated_residual_rank
            or self.mhc_gated_residual_rank < 1
        ):
            raise ValueError("Qwen4-Exp requires a positive gated-residual mixer rank.")
        if self.mhc_num_residual_streams < 1:
            raise ValueError("Qwen4-Exp requires at least one residual stream.")
        if self.linear_attention_output_gate_activation not in (None, "silu", "sigmoid"):
            raise ValueError("Qwen4-Exp output gate must be silu or sigmoid.")
        if self.qsa_indexer_n_heads is not None:
            for name in (
                "qsa_indexer_n_heads",
                "qsa_indexer_head_dim",
                "qsa_indexer_budget",
                "qsa_indexer_compress_ratio",
            ):
                value = getattr(self, name)
                if value is None or value <= 0:
                    raise ValueError(f"Qwen4-Exp requires a positive {name}.")
            if self.qsa_indexer_kv_heads != 1:
                raise ValueError("QSA requires one indexer key head.")
            # Narrow optional checkpoint fields after validating them above.
            assert self.qsa_indexer_budget is not None
            assert self.qsa_indexer_compress_ratio is not None
            assert self.qsa_indexer_head_dim is not None
            if self.qsa_indexer_budget % self.qsa_indexer_compress_ratio:
                raise ValueError("QSA budget must be divisible by its compression ratio.")
            if int(self.kv_channels * self.rotary_percent) > self.qsa_indexer_head_dim:
                raise ValueError("The attention RoPE width must fit in the QSA indexer head.")
        if self.ple_layer_ids:
            if len(set(self.ple_layer_ids)) != len(self.ple_layer_ids):
                raise ValueError("PLE layer IDs must be unique; their order defines the hash-table layout.")
            if any(layer < 1 or layer > self.num_layers for layer in self.ple_layer_ids):
                raise ValueError("PLE layer IDs must be one-indexed decoder layer numbers.")
            if self.ple_eos_token_id is None or not self.ple_unigram_vocab_size or self.ple_unigram_vocab_size < 1:
                raise ValueError("PLE requires EOS and unigram vocabulary settings.")
            heads = (self.ple_ngram_size - 1) * self.ple_heads_per_ngram
            if heads < 1 or (self.ple_embed_dim or self.hidden_size) % heads:
                raise ValueError("PLE embedding width must be divisible by its n-gram head count.")
            if min(self.ple_conv_kernel_size, self.ple_ngram_vocab_size_base, self.ple_ngram_vocab_divisible_by) < 1:
                raise ValueError("PLE convolution and vocabulary dimensions must be positive.")

    def finalize(self) -> None:
        super().finalize()
        self._validate_qwen4_exp()

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> Qwen4ExpGPTModel:
        """Build the local model with the same GPT checkpoint parameter names."""
        self._validate_qwen4_exp()
        if vp_stage is not None or pre_process is False or post_process is False:
            raise NotImplementedError("Qwen4-Exp requires a complete decoder on each PP=1 rank.")
        if self._pg_collection is None:
            raise ValueError("Qwen4-Exp requires an explicit process-group collection.")
        if self.vocab_size is None:
            raise ValueError("vocab_size must be configured before calling provide().")
        if not fusions.validate_rope_fusion_compatibility(self):
            self.apply_rope_fusion = False
        vocab_size = self.vocab_size
        if self.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        layer_spec = self.transformer_layer_spec
        if callable(layer_spec):
            layer_spec = layer_spec(self, vp_stage=None)
        self._vp_stage = None
        init_context = torch.device("meta") if self.init_model_with_meta_device else contextlib.nullcontext()
        with init_context:
            return Qwen4ExpGPTModel(
                self,
                transformer_layer_spec=layer_spec,
                vocab_size=vocab_size,
                max_sequence_length=self.seq_length,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                **logit_dtype_kwarg(Qwen4ExpGPTModel, self.logit_dtype),
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                rope_scaling=self.rope_scaling,
                rope_scaling_factor=self.rope_scaling_factor,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=True,
                post_process=True,
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
                pg_collection=self._pg_collection,
            )
