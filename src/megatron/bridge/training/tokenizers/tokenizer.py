# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from megatron.bridge.utils.common_utils import warn_rank_0

from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer as build_mcore_tokenizer


def build_tokenizer(config: TokenizerConfig, **kwargs) -> MegatronTokenizer:
    """Initialize tokenizer from megatron.core.tokenizers based on the provided configuration.

    Args:
        config (TokenizerConfig): Configuration object specifying the tokenizer
                                            type, paths to vocab/model files, and other
                                            tokenizer-specific settings.

    Returns:
        MegatronTokenizer: An instance of the initialized tokenizer.
    """

    warn_rank_0(
        "`build_tokenizer` is deprecated and will be removed soon. "
        "Please, use `megatron.core.tokenizers.utils.build_tokenizer` instead."
    )

    return build_mcore_tokenizer(config, **kwargs)

