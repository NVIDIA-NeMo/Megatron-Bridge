# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from megatron.core.tokenizers import MegatronTokenizer

from megatron.bridge.training.tokenizers.config import TokenizerConfig


MEGATRON_TOKENIZERS = ["BertWordPieceLowerCase", "BertWordPieceCase", "GPT2BPETokenizer"]

SP_TOKENIZERS = ["SentencePieceTokenizer", "GPTSentencePieceTokenizer", "Llama2Tokenizer"]


def _compute_space_sensitive(tokenizer_instance: "MegatronTokenizer", default: bool = True) -> bool:
    """
    Determine if a tokenizer is space-sensitive.

    A tokenizer is space-sensitive if tokenizing "x y" produces different token sequences
    than concatenating tokenize("x") + tokenize("y"). This affects how prompt templates
    handle spaces in the dataset preprocessing pipeline.

    Args:
        tokenizer_instance: Tokenizer instance with a `tokenize` method
        default: Fallback value if computation fails (True for SentencePiece, False for others)

    Returns:
        bool: True if the tokenizer is space-sensitive, False otherwise

    Example:
        # A space-sensitive tokenizer (e.g., many BPE tokenizers):
        # tokenize("x y") -> [87, 331]
        # tokenize("x") + tokenize("y") -> [87, 379]  # Different!

        # A non-space-sensitive tokenizer would produce the same result
    """
    try:
        test_tokens_with_space = tokenizer_instance.tokenize("x y")
        test_tokens_concat = tokenizer_instance.tokenize("x") + tokenizer_instance.tokenize("y")
        return test_tokens_with_space != test_tokens_concat
    except Exception:
        # If tokenization fails for any reason, use the default
        return default


def build_tokenizer(config: TokenizerConfig, **kwargs) -> MegatronTokenizer:
    """Initialize tokenizer from megatron.core.tokenizers based on the provided configuration.

    Args:
        config (TokenizerConfig): Configuration object specifying the tokenizer
                                            type, paths to vocab/model files, and other
                                            tokenizer-specific settings.

    Returns:
        MegatronTokenizer: An instance of the initialized tokenizer.
    """
    kwargs = {}
    tokenizer_library = None
    tokenizer_path = None
    if config.tokenizer_type in MEGATRON_TOKENIZERS:
        tokenizer_library = "megatron"
        tokenizer_path = config.tokenizer_type
        kwargs["additional_special_tokens"] = config.special_tokens if config.special_tokens else []
        if tokenizer_path == "BertWordPieceCase":
            special_tokens = {}
            special_tokens["additional_special_tokens"] = [f"<extra_id_{i}>" for i in range(100)]
            kwargs = special_tokens
        kwargs["vocab_file"] = config.vocab_file
        kwargs["merges_file"] = config.merge_file
        if config.hf_tokenizer_kwargs:
            kwargs.update(config.hf_tokenizer_kwargs)
    elif config.tokenizer_type in SP_TOKENIZERS:
        tokenizer_library = "sentencepiece"
        tokenizer_path = config.tokenizer_model
        kwargs["chat_template"] = config.chat_template
        kwargs["special_tokens"] = config.special_tokens
        kwargs.update(config.sp_tokenizer_kwargs)
    elif config.tokenizer_type == "TikTokenizer":
        tokenizer_library = "tiktoken"
        tokenizer_path = config.tokenizer_model
        kwargs["chat_template"] = config.chat_template
        if config.tiktoken_pattern:
            kwargs["pattern"] = config.tiktoken_pattern
        if config.vocab_size:
            kwargs["vocab_size"] = config.vocab_size
        kwargs["num_special_tokens"] = config.tiktoken_num_special_tokens
        kwargs["special_tokens"] = config.special_tokens
        kwargs["vocab_size"] = config.vocab_size
    elif config.tokenizer_type == "HuggingFaceTokenizer":
        tokenizer_library = "huggingface"
        tokenizer_path = config.tokenizer_model
        kwargs["chat_template"] = config.chat_template
        kwargs["vocab_file"] = config.vocab_file
        kwargs["merges_file"] = config.merge_file
        kwargs["additional_special_tokens"] = config.special_tokens if config.special_tokens else []
        if config.hf_tokenizer_kwargs:
            kwargs.update(config.hf_tokenizer_kwargs)
    elif config.tokenizer_type == 'MultimodalTokenizer':
        tokenizer_library = 'multimodal'
        kwargs['prompt_format'] = config.tokenizer_prompt_format
        kwargs['special_tokens'] = config.special_tokens
        kwargs['image_tag_type'] = config.image_tag_type
        kwargs['force_system_message'] = config.force_system_message
    elif config.tokenizer_type == 'SFTTokenizer':
        tokenizer_library = 'sft'
        tokenizer_path = config.tokenizer_model
        kwargs['prompt_format'] = config.tokenizer_prompt_format
    elif config.tokenizer_type in ['NullTokenizer', 'NullMultimodalTokenizer']:
        tokenizer_library = (
            'null-text' if config.tokenizer_type == 'NullTokenizer' else 'null-multimodal'
        )
        metadata = {"library": tokenizer_library}
        if config.vocab_size:
            kwargs["vocab_size"] = config.vocab_size - 1
        tokenizer = MegatronTokenizer.from_pretrained(metadata_path=metadata, **kwargs)

        return tokenizer

    if config.metadata_path:
        metadata = config.metadata_path
    else:
        metadata = {"library": tokenizer_library}
    tokenizer = MegatronTokenizer.from_pretrained(tokenizer_path=tokenizer_path, metadata_path=metadata, **kwargs)

    return tokenizer
