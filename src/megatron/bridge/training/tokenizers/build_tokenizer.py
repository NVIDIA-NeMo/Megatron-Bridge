# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

import math

from dataclasses import asdict

from megatron.core.tokenizers import MegatronTokenizer

from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.multimodal_tokenizer import MultimodalTokenizer
from megatron.bridge.utils.common_utils import get_rank_safe


MAIN_ARGS = [
    "tokenizer_path",
    "metadata_path",
    "multimodal_tokenizer",
    "write_metadata",
    "overwrite_metadata",
    "tokenizer_library",
    "model_type",
]


def build_tokenizer(
    tokenizer_config: TokenizerConfig, make_vocab_size_divisible_by: int, tensor_model_parallel_size: int, **kwargs
):
    """Initialize tokenizer based on the provided configuration.

    This function serves as a factory to instantiate various tokenizer types
    supported by NeMo, such as BERT, GPT2, SentencePiece, HuggingFace, etc.
    It also handles padding the vocabulary size to be GPU-friendly.

    Args:
        tokenizer_config (TokenizerConfig): Configuration object specifying the tokenizer
                                            type, paths to vocab/model files, and other
                                            tokenizer-specific settings.
        **kwargs: Additional keyword arguments that might be specific to certain tokenizers
                  (e.g., passed to HuggingFace AutoTokenizer).

    Returns:
        MegatronTokenizer: An instance of the initialized tokenizer.

    Raises:
        NotImplementedError: If the specified tokenizer_type in tokenizer_config is not supported.
        ImportError: If a required library (e.g., transformers for MultimodalTokenizer) is not installed.
    """
    if get_rank_safe() == 0:
        print("> building {} tokenizer ...".format(tokenizer_config.tokenizer_library), flush=True)

    if tokenizer_config.multimodal_tokenizer:
        try:
            import transformers as _transformers
        except ImportError as exc:
            raise ImportError("MultimodalTokenizer currently requires transformers library to be installed") from exc
        kwargs = {}
        if tokenizer_config.tokenizer_prompt_format == "nvlm-yi-34b":
            kwargs = {"from_slow": True, "legacy": False, "add_bos_token": True}
        underlying_tokenizer = _transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_config.tokenizer_path, **kwargs
        )
        tokenizer = MultimodalTokenizer(
            underlying_tokenizer,
            tokenizer_config.tokenizer_prompt_format,
            tokenizer_config.special_tokens,
            tokenizer_config.image_tag_type,
        )
    else:
        if tokenizer_config.write_metadata:
            MegatronTokenizer.write_metadata(
                tokenizer_path=tokenizer_config.tokenizer_path,
                tokenizer_library=tokenizer_config.tokenizer_library,
                model_type=tokenizer_config.model_type,
                tokenizer_class=tokenizer_config.tokenizer_class,
                overwrite=tokenizer_config.overwrite_metadata,
                metadata_path=tokenizer_config.metadata_path,
                chat_template=tokenizer_config.chat_template,
            )

        additional_args = {k: v for k, v in asdict(tokenizer_config).items() if v is not None and k not in MAIN_ARGS}
        if tokenizer_config.additional_special_tokens:
            additional_args['additional_special_tokens'] = tokenizer_config.additional_special_tokens

        special_tokens = tokenizer_config.special_tokens
        if type(special_tokens) is list or special_tokens is None:
            tokenizer = MegatronTokenizer.from_pretrained(
                tokenizer_path=tokenizer_config.tokenizer_path,
                metadata_path=tokenizer_config.metadata_path,
                **additional_args,
            )
        else:
            special_tokens = additional_args.pop('special_tokens')
            tokenizer = MegatronTokenizer.from_pretrained(
                tokenizer_path=tokenizer_config.tokenizer_path,
                metadata_path=tokenizer_config.metadata_path,
                **additional_args,
                **special_tokens,
            )

    # Add vocab size (if not already set from a checkpoint).
    if getattr(tokenizer_config, "padded_vocab_size", None) is None:
        tokenizer_config.padded_vocab_size = _vocab_size_with_padding(
            tokenizer.vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size
        )

    return tokenizer


def _vocab_size_with_padding(
    orig_vocab_size: int,
    make_vocab_size_divisible_by: int,
    tensor_model_parallel_size: int,
    logging_enabled: bool = True,
):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
    after = int(math.ceil(after / multiple) * multiple)
    if get_rank_safe() == 0 and logging_enabled:
        print(
            " > padded vocab (size: {}) with {} dummy tokens (new size: {})".format(
                orig_vocab_size, after - orig_vocab_size, after
            ),
            flush=True,
        )
    return after
