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

import contextlib
import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import torch
import torch.distributed
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo_lm.utils.import_utils import safe_import

_, HAVE_TE = safe_import("transformer_engine")

# Gradient accumulation fusion may be enabled if available, for more information see:
# https://github.com/NVIDIA/Megatron-LM/blob/01945b98d1ea3a2acb5e8301e181a328104f4856/megatron/core/tensor_parallel/layers.py#L575
# TODO: Clean this up with a getter and install instructions
_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda  # noqa: F401  # pylint: disable=unused-import
except ImportError:
    _grad_accum_fusion_available = False

if TYPE_CHECKING:
    from transformers import GenerationConfig

logger = logging.getLogger(__name__)


def transformer_engine_layer_spec(config: "GPTConfig") -> ModuleSpec:
    """Create a Transformer Engine layer specification based on the provided config.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for Transformer Engine based layers
    """
    from megatron.core.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        fp8=bool(config.num_moe_experts and (config.fp8 is not None)),
    )


def transformer_engine_full_layer_spec(config: "GPTConfig") -> ModuleSpec:
    """Create a full Transformer Engine layer specification with autocast support.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for full TE layers
    """
    from nemo.collections.nlp.models.language_modeling.megatron.gpt_full_te_layer_autocast_spec import (
        get_gpt_full_te_layer_autocast_spec,
    )

    return get_gpt_full_te_layer_autocast_spec(transformer_config=config)


def local_layer_spec(config: "GPTConfig") -> ModuleSpec:
    """Create a local layer specification without Transformer Engine.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for local implementation layers
    """
    from megatron.core.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        normalization=config.normalization,
    )


def default_layer_spec(config: "GPTConfig") -> ModuleSpec:
    """Determine the most appropriate layer specification based on availability.

    Uses Transformer Engine specs if available, otherwise falls back to local implementation.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: The selected module specification
    """
    if HAVE_TE:
        if config.use_transformer_engine_full_layer_spec:
            return transformer_engine_full_layer_spec(config)
        else:
            return transformer_engine_layer_spec(config)
    else:
        return local_layer_spec(config)


def mtp_block_spec(config: "GPTConfig") -> Optional[ModuleSpec]:
    """Pass in the MTP block spec if model has MTP layers.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: The MTP module specification
    """
    if getattr(config, "mtp_num_layers", None):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

        if isinstance(config.transformer_layer_spec, Callable):
            spec = config.transformer_layer_spec(config)
        else:
            spec = config.transformer_layer_spec
        return get_gpt_mtp_block_spec(config, spec, use_transformer_engine=HAVE_TE)
    else:
        return None


def torch_dtype_from_mcore_config(config: TransformerConfig) -> torch.dtype:
    """Extract the appropriate torch dtype from a Megatron Core configuration.

    Args:
        config: Megatron Core Transformer configuration

    Returns:
        torch.dtype: The appropriate torch dtype (float16, bfloat16, or float32)
    """
    if config.fp16:
        return torch.float16
    elif config.bf16:
        return torch.bfloat16
    else:
        return torch.float


def torch_dtype_from_dict_config(config: dict[str, Any]) -> torch.dtype:
    """Extract the appropriate torch dtype from a dictionary configuration.

    Args:
        config: Dictionary containing configuration parameters

    Returns:
        torch.dtype: The appropriate torch dtype (float16, bfloat16, or float32)
    """
    if config["fp16"]:
        return torch.float16
    elif config["bf16"]:
        return torch.bfloat16
    else:
        return torch.float


def get_vocab_size(
    config,
    vocab_size: int,
    make_vocab_size_divisible_by: int = 128,
) -> int:
    """returns `vocab size + padding` to make sure sum is dividable by `make_vocab_size_divisible_by`"""
    after = vocab_size
    multiple = make_vocab_size_divisible_by * config.tensor_model_parallel_size
    after = ((after + multiple - 1) // multiple) * multiple
    logger.info(
        f"Padded vocab_size: {after}, original vocab_size: {vocab_size}, dummy tokens:" f" {after - vocab_size}."
    )

    return after


@dataclass
class GPTConfig(TransformerConfig):
    """Configuration class for GPT models.

    Extends TransformerConfig with additional parameters specific to GPT models
    and provides utility methods for model configuration.
    """

    # From megatron.core.models.gpt.gpt_model.GPTModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024
    attention_softmax_in_fp32: bool = False
    masked_softmax_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    gradient_accumulation_fusion: bool = _grad_accum_fusion_available
    deallocate_pipeline_outputs: bool = True
    scatter_embedding_sequence_parallel: bool = True
    tp_only_amax_red: bool = False
    tp_comm_overlap_cfg: Optional[str] = None
    """Config file when tp_comm_overlap is enabled."""

    use_transformer_engine_full_layer_spec: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = default_layer_spec

    generation_config: Optional["GenerationConfig"] = None

    vocab_size: Optional[int] = None
    tp_comm_overlap_cfg: Optional[Union[str, dict[str, Any]]] = None

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core GPT model based on this configuration.

        Args:
            tokenizer: Tokenizer used with the model
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        if self.enable_cuda_graph:
            assert HAVE_TE, "Transformer Engine is required for cudagraphs."
            assert getattr(self, "use_te_rng_tracker", False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = getattr(self, "account_for_embedding_in_pipeline_split", False) or getattr(
            self, "account_for_loss_in_pipeline_split", False
        )
        if vp_size and not is_pipeline_asymmetric:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)

        if self.vocab_size is not None:
            vocab_size = self.vocab_size
            if tokenizer is not None:
                logger.info(
                    f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                    f" {vocab_size - tokenizer.vocab_size}."
                )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)

        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(torch.device, device="meta")

        import inspect

        if "mtp_block_spec" in inspect.signature(MCoreGPTModel.__init__).parameters:
            kwargs = {"mtp_block_spec": mtp_block_spec(self)}
        else:
            kwargs = {}
        with model_init_device_context():
            model = MCoreGPTModel(
                self,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=vocab_size,
                max_sequence_length=self.seq_length,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
                post_process=post_process or parallel_state.is_pipeline_last_stage(),
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
                **kwargs,
            )

        # If using full TE layer, need to set TP, CP group since the module call
        # is not routed through megatron core, which normally handles passing the
        # TP, CP group to the TE modules.
        # Deep iterate but skip self to avoid infinite recursion.
        if HAVE_TE and self.use_transformer_engine_full_layer_spec:
            # Copied from:
            # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                for index, child in enumerate(model.modules()):
                    if index == 0:
                        continue
                    if hasattr(child, "set_tensor_parallel_group"):
                        tp_group = parallel_state.get_tensor_model_parallel_group()
                        child.set_tensor_parallel_group(tp_group)

            if parallel_state.get_context_parallel_world_size() > 1:
                cp_stream = torch.cuda.Stream()
                for module in self.get_model_module_list():
                    for index, child in enumerate(module.modules()):
                        if index == 0:
                            continue
                        if hasattr(child, "set_context_parallel_group"):
                            child.set_context_parallel_group(
                                parallel_state.get_context_parallel_group(),
                                parallel_state.get_context_parallel_global_ranks(),
                                cp_stream,
                            )

        return model


@dataclass
class GPTConfig126M(GPTConfig):
    """Configuration for a 126M parameter GPT model.

    Predefined configuration for a small GPT model with 12 layers,
    768 hidden size, and 12 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig5B(GPTConfig):
    """Configuration for a 5B parameter GPT model.

    Predefined configuration for a medium-sized GPT model with 24 layers,
    4096 hidden size, and 32 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 24
    hidden_size: int = 4096
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig7B(GPTConfig):
    """Configuration for a 7B parameter GPT model.

    Predefined configuration for a medium-sized GPT model with 32 layers,
    4096 hidden size, and 32 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 10880
    num_attention_heads: int = 32
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig20B(GPTConfig):
    """Configuration for a 20B parameter GPT model.

    Predefined configuration for a large GPT model with 44 layers,
    6144 hidden size, and 48 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 44
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 48
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig40B(GPTConfig):
    """Configuration for a 40B parameter GPT model.

    Predefined configuration for a large GPT model with 48 layers,
    8192 hidden size, and 64 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 48
    hidden_size: int = 8192
    ffn_hidden_size: int = 32768
    num_attention_heads: int = 64
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig175B(GPTConfig):
    """Configuration for a 175B parameter GPT model.

    Predefined configuration for a massive GPT model with 96 layers,
    12288 hidden size, and 96 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 96
    hidden_size: int = 12288
    ffn_hidden_size: int = 49152
    num_attention_heads: int = 96
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True
    layernorm_zero_centered_gamma: bool = True


def get_batch_on_this_context_parallel_rank(batch) -> dict[str, torch.Tensor]:
    """Process batch data for the current context parallel rank.

    Handles the slicing of batch data across context parallel dimensions.

    Args:
        batch: Input batch

    Returns:
        dict[str, torch.Tensor]: Processed batch for the current context parallel rank
    """
    from megatron.core import parallel_state

    if (cp_size := parallel_state.get_context_parallel_world_size()) > 1:
        num_valid_tokens_in_ub = None
        if "loss_mask" in batch and batch["loss_mask"] is not None:
            num_valid_tokens_in_ub = batch["loss_mask"].sum()

        cp_rank = parallel_state.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != "attention_mask" else 2
                _val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).to(
                    _val.device, non_blocking=True
                )
                _val = _val.index_select(seq_dim, index)
                _val = _val.view(*val.shape[0:seq_dim], -1, *_val.shape[(seq_dim + 2) :])
                batch[key] = _val
        batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub
    return batch


def get_packed_seq_params(batch):
    """Extract packed sequence parameters from the batch.

    Creates and returns a PackedSeqParams object with appropriate parameters
    for packed sequence processing.

    Args:
        batch: Input batch containing packed sequence information

    Returns:
        PackedSeqParams: Parameters for packed sequence processing
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = batch["cu_seqlens"].squeeze()  # remove batch size dimension (mbs=1)
    # remove -1 "paddings" added in collate_fn
    if (cu_seqlens_argmin := batch.get("cu_seqlens_argmin", None)) is not None:
        # pre-compute cu_seqlens_argmin in dataset class for perf
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    # pre-compute max_seqlens in dataset class for perf
    max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None

    # these args are passed eventually into TEDotProductAttention.forward()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )
