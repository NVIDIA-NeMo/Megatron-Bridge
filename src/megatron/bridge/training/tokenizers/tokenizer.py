# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from copy import copy

from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer as build_mcore_tokenizer

from megatron.bridge.training.tokenizers.config import TokenizerConfig


def _config_for_mcore_build_tokenizer(config: TokenizerConfig) -> TokenizerConfig:
    """Return a tokenizer config compatible with MCore's tokenizer builder."""
    mcore_config = copy(config)

    # TODO: remove this guard when Megatron-LM tokenizer build_tokenizer either restores
    # pad_vocab_size handling or no longer requires padding attrs on TokenizerConfig.
    if not hasattr(mcore_config, "make_vocab_size_divisible_by"):
        setattr(mcore_config, "make_vocab_size_divisible_by", 1)
    if not hasattr(mcore_config, "tensor_model_parallel_size"):
        setattr(mcore_config, "tensor_model_parallel_size", 1)
    if not hasattr(mcore_config, "rank"):
        setattr(mcore_config, "rank", 0)

    return mcore_config


def build_tokenizer(config: TokenizerConfig, **kwargs) -> MegatronTokenizer:
    """Initialize tokenizer from megatron.core.tokenizers based on the provided configuration.

    Args:
        config (TokenizerConfig): Configuration object specifying the tokenizer
                                            type, paths to vocab/model files, and other
                                            tokenizer-specific settings.

    Returns:
        MegatronTokenizer: An instance of the initialized tokenizer.
    """
    from megatron.bridge.utils.common_utils import warn_rank_0

    warn_rank_0(
        "`build_tokenizer` is deprecated and will be removed soon. "
        "Please, use `megatron.core.tokenizers.utils.build_tokenizer` instead."
    )

    return build_mcore_tokenizer(_config_for_mcore_build_tokenizer(config), **kwargs)
