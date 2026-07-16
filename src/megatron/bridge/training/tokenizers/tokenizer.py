# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

import dataclasses
from typing import Optional

from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer as build_mcore_tokenizer

from megatron.bridge.training.tokenizers.config import TokenizerConfig


def _resolve_chat_template(config: TokenizerConfig) -> Optional[str]:
    """Resolve the effective chat template from ``config``.

    Returns ``chat_template`` verbatim when set, or the contents of the file at
    ``chat_template_path`` (local path or ``msc://`` URL) otherwise.

    Args:
        config: Tokenizer configuration.

    Returns:
        The chat template string, or ``None`` when neither field is set.

    Raises:
        ValueError: If both ``chat_template`` and ``chat_template_path`` are set.
    """
    if config.chat_template_path is None:
        return config.chat_template
    if config.chat_template is not None:
        raise ValueError("Set only one of `chat_template` or `chat_template_path`, not both.")

    path = config.chat_template_path
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        with msc.open(str(path), "r") as f:
            return f.read()
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


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

    if config.chat_template_path is not None:
        config = dataclasses.replace(config, chat_template=_resolve_chat_template(config), chat_template_path=None)

    return build_mcore_tokenizer(config, **kwargs)
