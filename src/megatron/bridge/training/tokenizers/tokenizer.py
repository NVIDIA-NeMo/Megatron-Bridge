# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

import dataclasses
from copy import copy
from pathlib import Path
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


_HF_TOKENIZER_SNAPSHOT_ALLOW_PATTERNS = (
    "config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab*",
    "merges*",
    "*.model",
    "*.spm",
    "*.tiktoken",
    "*.py",
    "*.jinja",
    "chat_template*",
    "chat_templates/*",
)
_HF_MODEL_WEIGHT_IGNORE_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.msgpack",
    "*.gguf",
    "*.onnx",
    "*.npz",
)


def _is_local_tokenizer_path(tokenizer_model: object) -> bool:
    """Return whether a tokenizer reference is explicitly a local path."""
    if isinstance(tokenizer_model, Path):
        return True
    if not isinstance(tokenizer_model, str):
        return False

    path = Path(tokenizer_model).expanduser()
    return path.exists() or path.is_absolute() or tokenizer_model.startswith(("./", "../", "~"))


def _resolve_hf_tokenizer_revision(config: TokenizerConfig) -> TokenizerConfig:
    """Resolve an immutable Hugging Face tokenizer revision without mutating persisted config."""
    if config.tokenizer_type != "HuggingFaceTokenizer":
        return config

    hf_tokenizer_kwargs = config.hf_tokenizer_kwargs or {}
    revision = hf_tokenizer_kwargs.get("revision")
    if revision is None:
        return config
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("hf_tokenizer_kwargs.revision must be a non-empty string")

    tokenizer_model = config.tokenizer_model
    if not isinstance(tokenizer_model, str) or _is_local_tokenizer_path(tokenizer_model):
        return config

    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=tokenizer_model,
        revision=revision,
        allow_patterns=list(_HF_TOKENIZER_SNAPSHOT_ALLOW_PATTERNS),
        ignore_patterns=list(_HF_MODEL_WEIGHT_IGNORE_PATTERNS),
    )

    resolved_config = copy(config)
    resolved_config.tokenizer_model = snapshot_path
    if "use_fast" in hf_tokenizer_kwargs:
        resolved_config.tokenizer_hf_no_use_fast = not hf_tokenizer_kwargs["use_fast"]
    if "include_special_tokens" in hf_tokenizer_kwargs:
        resolved_config.tokenizer_hf_no_include_special_tokens = not hf_tokenizer_kwargs["include_special_tokens"]
    if "trust_remote_code" in hf_tokenizer_kwargs:
        resolved_config.trust_remote_code = hf_tokenizer_kwargs["trust_remote_code"]
    return resolved_config


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

    return build_mcore_tokenizer(_resolve_hf_tokenizer_revision(config), **kwargs)
