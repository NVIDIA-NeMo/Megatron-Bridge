# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

from megatron.bridge.training.tokenizers.bert_tokenization import FullTokenizer as FullBertTokenizer
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.gpt2_tokenization import GPT2Tokenizer
from megatron.bridge.training.tokenizers.multimodal_tokenizer import MultimodalTokenizer
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


try:
    from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer as MegatronTokenizerCore
except ImportError:
    # Fallback to old path
    from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer as MegatronTokenizerCore


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


class MegatronTokenizer(MegatronTokenizerCore):
    """Base tokenizer class, extending the MegatronTokenizer from megatron core.

    This class provides a common interface for various tokenizers used within the NeMo framework.
    """

    def __call__(self, *args, **kwargs):
        """Makes the tokenizer instance callable, synonym for `tokenize`."""
        return self.tokenize(*args, **kwargs)

    def text_to_ids(self, text: str) -> list[int]:
        """Converts text to a list of token IDs."""
        return self.tokenize(text)

    @property
    def eod_id(self):
        """ID for the end-of-document token."""
        return self.eod

    @property
    def bos_id(self):
        """ID for the beginning-of-sentence token."""
        return self.bos

    @property
    def eos_id(self):
        """ID for the end-of-sentence token."""
        return self.eos

    @property
    def mask_id(self):
        """ID for the mask token."""
        return self.mask


def build_tokenizer(tokenizer_config: TokenizerConfig, **kwargs) -> MegatronTokenizer:
    """Initialize tokenizer based on the provided configuration.

    This function serves as a factory to instantiate various tokenizer types
    supported by NeMo Framework, such as BERT, GPT2, SentencePiece, HuggingFace, etc.
    It also handles padding the vocabulary size to be GPU-friendly.

    Args:
        tokenizer_config (TokenizerConfig): Configuration object specifying the tokenizer
                                            type, paths to vocab/model files, and other
                                            tokenizer-specific settings.
        **kwargs: Additional keyword arguments that might be specific to certain tokenizers
                  (e.g., passed to HuggingFace AutoTokenizer). These will override any
                  kwargs specified in tokenizer_config.hf_tokenizer_kwargs.

    Returns:
        MegatronTokenizer: An instance of the initialized tokenizer.

    Raises:
        NotImplementedError: If the specified tokenizer_type in tokenizer_config is not supported.
        ImportError: If a required library (e.g., transformers for MultimodalTokenizer) is not installed.
    """
    if get_rank_safe() == 0:
        print("> building {} tokenizer ...".format(tokenizer_config.tokenizer_type), flush=True)

    # Select and instantiate the tokenizer.
    if tokenizer_config.tokenizer_type == "BertWordPieceLowerCase":
        assert tokenizer_config.vocab_file is not None
        tokenizer = _BertWordPieceTokenizer(
            vocab_file=tokenizer_config.vocab_file, lower_case=True, vocab_extra_ids=tokenizer_config.vocab_extra_ids
        )
    elif tokenizer_config.tokenizer_type == "BertWordPieceCase":
        assert tokenizer_config.vocab_file is not None
        tokenizer = _BertWordPieceTokenizer(
            vocab_file=tokenizer_config.vocab_file, lower_case=False, vocab_extra_ids=tokenizer_config.vocab_extra_ids
        )
    elif tokenizer_config.tokenizer_type == "GPT2BPETokenizer":
        assert tokenizer_config.vocab_file is not None
        assert tokenizer_config.merge_file is not None
        tokenizer = _GPT2BPETokenizer(tokenizer_config.vocab_file, tokenizer_config.merge_file)
    elif tokenizer_config.tokenizer_type == "SentencePieceTokenizer":
        assert tokenizer_config.tokenizer_model is not None
        tokenizer = _SentencePieceTokenizer(
            tokenizer_config.tokenizer_model, vocab_extra_ids=tokenizer_config.vocab_extra_ids
        )
    elif tokenizer_config.tokenizer_type == "GPTSentencePieceTokenizer":
        assert tokenizer_config.tokenizer_model is not None
        tokenizer = _GPTSentencePieceTokenizer(tokenizer_config.tokenizer_model)
    elif tokenizer_config.tokenizer_type == "HuggingFaceTokenizer":
        # Merge config-based kwargs with any passed kwargs (passed kwargs take precedence)
        hf_kwargs = tokenizer_config.hf_tokenizer_kwargs or {}
        hf_kwargs.update(kwargs)
        tokenizer = _HuggingFaceTokenizer(tokenizer_config.tokenizer_model, **hf_kwargs)
    elif tokenizer_config.tokenizer_type == "Llama2Tokenizer":
        assert tokenizer_config.tokenizer_model is not None
        tokenizer = _Llama2Tokenizer(tokenizer_config.tokenizer_model)
    elif tokenizer_config.tokenizer_type == "TikTokenizer":
        assert tokenizer_config.tokenizer_model is not None
        assert tokenizer_config.tiktoken_pattern is not None
        assert tokenizer_config.tiktoken_pattern in {"v1", "v2"}
        pattern_tiktoken = r"[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
        pattern_tiktoken_v2 = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"  # pylint: disable=line-too-long
        pattern = pattern_tiktoken if tokenizer_config.tiktoken_pattern == "v1" else pattern_tiktoken_v2
        tokenizer = CustomTikTokenizer(
            path=tokenizer_config.tokenizer_model,
            pattern=pattern,
            vocab_size=tokenizer_config.vocab_size,
            num_special_tokens=tokenizer_config.tiktoken_num_special_tokens,
            special_tokens=tokenizer_config.tiktoken_special_tokens,
        )
    elif tokenizer_config.tokenizer_type == "NullTokenizer":
        assert tokenizer_config.vocab_size is not None
        tokenizer = _NullTokenizer(tokenizer_config.vocab_size)
    elif tokenizer_config.tokenizer_type == "MultimodalTokenizer":
        try:
            import transformers as _transformers
        except ImportError as exc:
            raise ImportError("MultimodalTokenizer currently requires transformers library to be installed") from exc
        kwargs = {}
        if tokenizer_config.tokenizer_prompt_format == "nvlm-yi-34b":
            kwargs = {"from_slow": True, "legacy": False, "add_bos_token": True}
        underlying_tokenizer = _transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_config.tokenizer_model, **kwargs
        )
        tokenizer = MultimodalTokenizer(
            underlying_tokenizer,
            tokenizer_config.tokenizer_prompt_format,
            tokenizer_config.special_tokens,
            tokenizer_config.image_tag_type,
        )
    elif tokenizer_config.tokenizer_type == "NullMultimodalTokenizer":
        assert tokenizer_config.vocab_size is not None
        tokenizer = _NullMultimodalTokenizer(tokenizer_config.vocab_size)
    else:
        raise NotImplementedError("{} tokenizer is not implemented.".format(tokenizer_config.tokenizer_type))

    return tokenizer


class _HuggingFaceTokenizer(MegatronTokenizer):
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__(pretrained_model_name_or_path, **kwargs)
        try:
            import transformers
        except ImportError:
            raise EnvironmentError("The transformers library must be installed to use huggingface_tokenizer_provider")

        # Extract chat_template if provided, to set it explicitly after loading
        chat_template = kwargs.pop("chat_template", None)

        # TODO(bnorick): download tokenizer once to lustre
        # and use force offline to make sure all tasks read it from there
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )

        # Explicitly set chat_template if provided
        if chat_template is not None:
            if getattr(self._tokenizer, "chat_template", None) is not None:
                print_rank_0("You are overwriting tokenizer's chat template, confirm this is intended.")
            self._tokenizer.chat_template = chat_template
            self._tokenizer.chat_template_format = "jinja"

        self._vocab = self._tokenizer.get_vocab()
        self._inv_vocab = {token_id: token for token, token_id in self._vocab.items()}

        # Compute space_sensitive attribute for template handling in datasets
        self.space_sensitive = _compute_space_sensitive(self, default=False)

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self._tokenizer)

    @property
    def vocab(self):
        """Returns the vocabulary (token to ID mapping)."""
        return self._vocab

    @property
    def inv_vocab(self):
        """Returns the inverse vocabulary (ID to token mapping)."""
        return self._inv_vocab

    @property
    def decoder(self):
        """Alias for inv_vocab, for compatibility."""
        return self._inv_vocab

    def tokenize(self, text, **kwargs):
        """Tokenizes a string of text into a list of token IDs."""
        return self._tokenizer(text, **kwargs).input_ids

    def detokenize(self, token_ids, **kwargs):
        """Converts a list of token IDs back into a string."""
        return self._tokenizer.decode(token_ids, **kwargs)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Calculates the character offsets for each token ID in the given text."""
        retok_ids = self._tokenizer(text)
        offsets, next_start_idx = [], 0
        for i in range(len(ids)):
            span = retok_ids.token_to_chars(i)
            if span is not None:
                offsets.append(span.start)
                next_start_idx = span.end
            else:
                offsets.append(next_start_idx)
        return offsets

    @property
    def eod(self):
        """Returns the end-of-document token ID."""
        return self._tokenizer.eos_token_id

    @property
    def bos(self):
        """Returns the beginning-of-sentence token ID."""
        return self._tokenizer.bos_token_id

    @property
    def eos(self):
        """Returns the end-of-sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def mask(self):
        """Returns the mask token ID."""
        return self._tokenizer.mask_token_id


class _BertWordPieceTokenizer(MegatronTokenizer):
    """Original BERT wordpiece tokenizer adapted for Megatron.

    This tokenizer uses the `FullBertTokenizer` from `bert_tokenization`.
    It handles lower/upper casing and adds special tokens like [CLS], [SEP],
    [PAD], [MASK], [BOS], and [EOS]. It also supports adding extra vocabulary IDs.

    Args:
        vocab_file (str): Path to the BERT vocabulary file.
        lower_case (bool, optional): Whether to convert text to lower case. Defaults to True.
        vocab_extra_ids (int, optional): Number of extra IDs to add to the vocabulary,
                                       often used for sentinel tokens in T5-style models.
                                       Defaults to 0.
    """

    def __init__(self, vocab_file, lower_case=True, vocab_extra_ids=0):
        super().__init__(vocab_file, lower_case=lower_case, vocab_extra_ids=vocab_extra_ids)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab["[CLS]"]
        self.sep_id = self.tokenizer.vocab["[SEP]"]
        self.pad_id = self.tokenizer.vocab["[PAD]"]
        self.mask_id = self.tokenizer.vocab["[MASK]"]
        self._additional_special_tokens = []

        # (dsachan) Add BOS and EOS tokens
        # SPECIAL_TOKENS = {"eos_token": "[EOS]", "bos_token": "[BOS]"}
        self._bos_token = "[BOS]"
        self.add_token(self._bos_token)
        self._bos_token_id = self.vocab.get(self._bos_token)

        self._eos_token = "[EOS]"
        self.add_token(self._eos_token)
        self._eos_token_id = self.vocab.get(self._eos_token)

        # (dsachan) Add additional special tokens
        # These can be used as sentinel tokens in T5 model inputs
        additional_special_tokens = []
        additional_special_tokens.extend(["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)])
        self.add_additional_special_tokens(additional_special_tokens)

    def add_token(self, token):
        """Adds a single token to the vocabulary if it doesn't already exist."""
        if token not in self.vocab:
            self.inv_vocab[self.vocab_size] = token
            # self.vocab_size comes from len(vocab)
            # and it will increase as we add elements
            self.vocab[token] = self.vocab_size

    def add_additional_special_tokens(self, tokens_list):
        """Adds a list of special tokens to the vocabulary."""
        setattr(self, "additional_special_tokens", tokens_list)
        for value in tokens_list:
            self.add_token(value)

    @property
    def vocab_size(self):
        """Returns the current size of the vocabulary."""
        return self.tokenizer.vocab_size()

    @property
    def vocab(self):
        """Returns the vocabulary (token to ID mapping)."""
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        """Returns the inverse vocabulary (ID to token mapping)."""
        return self.tokenizer.inv_vocab

    def tokenize(self, text):
        """Tokenizes a string of text into a list of token IDs."""
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    def decode(self, ids):
        """Converts a list of token IDs back to a string, cleaning up ## prefixes."""
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def detokenize(self, token_ids):
        """Converts a list of token IDs back to a string. Alias for decode()."""
        return self.decode(token_ids)

    def decode_token_ids(self, token_ids):
        """Converts token IDs to a string, excluding [PAD] and [CLS] and handling ## prefixes."""
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ["[PAD]", "[CLS]"]
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    @property
    def cls(self):
        """Returns the [CLS] token ID."""
        return self.cls_id

    @property
    def sep(self):
        """Returns the [SEP] token ID."""
        return self.sep_id

    @property
    def pad(self):
        """Returns the [PAD] token ID."""
        return self.pad_id

    @property
    def mask(self):
        """Returns the [MASK] token ID."""
        return self.mask_id

    @property
    def bos(self):
        """Returns the beginning-of-sentence ([BOS]) token ID."""
        return self._bos_token_id

    @property
    def eos(self):
        """Returns the end-of-sentence token ID."""
        return self._eos_token_id

    @property
    def eod(self):
        """Alias for eos, as BERT models typically use EOS for end-of-document."""
        return self.eos

    @property
    def bos_token(self):
        """Returns the beginning-of-sentence token string ([BOS])."""
        return self._bos_token

    @property
    def eos_token(self):
        """Returns the end-of-sentence token string ([EOS])."""
        return self._eos_token

    @property
    def additional_special_tokens(self):
        """Returns a list of additional special token strings added to the tokenizer."""
        return self._additional_special_tokens

    @property
    def additional_special_tokens_ids(self):
        """Returns a list of IDs for the additional special tokens."""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value


class _GPT2BPETokenizer(MegatronTokenizer):
    """Original GPT-2 BPE tokenizer adapted for Megatron.

    This tokenizer uses the `GPT2Tokenizer` from `gpt2_tokenization`.
    It handles BPE tokenization based on a vocabulary file and a merges file.
    The primary special token is <|endoftext|>.

    Args:
        vocab_file (str): Path to the GPT-2 vocabulary file (e.g., vocab.json).
        merge_file (str): Path to the GPT-2 merges file (e.g., merges.txt).
    """

    def __init__(self, vocab_file, merge_file):
        super().__init__(vocab_file, merge_file)

        self.tokenizer = GPT2Tokenizer(vocab_file, merge_file, errors="replace", special_tokens=[], max_len=None)
        self.eod_id = self.tokenizer.encoder["<|endoftext|>"]

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        """Returns the vocabulary (token to ID mapping)."""
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        """Returns the inverse vocabulary (ID to token mapping)."""
        return self.tokenizer.decoder

    def tokenize(self, text):
        """Tokenizes a string of text into a list of token IDs."""
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        """Converts a list of token IDs back into a string."""
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        """Returns the end-of-document (<|endoftext|>) token ID."""
        return self.eod_id

    @property
    def eos(self):
        """Returns the EOS token ID."""
        return self.eod_id


class _SentencePieceTokenizer(MegatronTokenizer):
    """A wrapper for SentencePiece tokenizers used with Megatron.

    This class interfaces with a pre-trained SentencePiece model.
    It defines and manages several special tokens such as <CLS>, <SEP>, <EOD>,
    <MASK>, <PAD>, <BOS>, and <EOS>. It also supports adding extra vocabulary
    IDs, typically for T5-style sentinel tokens.

    Args:
        model_file (str): Path to the SentencePiece model file (e.g., tokenizer.model).
        vocab_extra_ids (int, optional): Number of extra IDs to add to the vocabulary.
                                       Defaults to 0.
    """

    def __init__(self, model_file, vocab_extra_ids=0):
        super().__init__(model_file, vocab_extra_ids=vocab_extra_ids)

        import sentencepiece

        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    def _populate_vocab(self):
        self._vocab = {}
        self._inv_vocab = {}

        for i in range(len(self.tokenizer)):
            t = self.tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()
        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token("<CLS>")
        self._cls_id = self._vocab["<CLS>"]
        _add_special_token("<SEP>")
        self._sep_id = self._vocab["<SEP>"]
        _add_special_token("<EOD>")
        self._eod_id = self._vocab["<EOD>"]
        _add_special_token("<MASK>")
        self._mask_id = self._vocab["<MASK>"]

        pad_id = self.tokenizer.pad_id()
        try:
            pad_token = self.tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = "<PAD>"
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        bos_id = self.tokenizer.bos_id()
        try:
            bos_token = self.tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = "<BOS>"
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self.tokenizer.eos_id()
        try:
            eos_token = self.tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = "<EOS>"
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]

        # Compute space_sensitive attribute for template handling in datasets
        # SentencePiece tokenizers are typically space-sensitive (default=True)
        self.space_sensitive = _compute_space_sensitive(self, default=True)

    @property
    def vocab_size(self):
        """Returns the current size of the vocabulary, including added special tokens."""
        return len(self._vocab)

    @property
    def vocab(self):
        """Returns the vocabulary (token to ID mapping)."""
        return self._vocab

    @property
    def inv_vocab(self):
        """Returns the inverse vocabulary (ID to token mapping)."""
        return self._inv_vocab

    @property
    def decoder(self):
        """Alias for inv_vocab."""
        return self._inv_vocab

    @property
    def encoder(self):
        """Alias for vocab."""
        return self._vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89  # pylint: disable=line-too-long
    def tokenize(self, text):
        """Tokenizes a string, handling special tokens separately.

        This method first finds occurrences of special tokens (defined during
        initialization) and tokenizes the text segments around them using the
        SentencePiece model. Special tokens are inserted as their pre-defined IDs.

        Args:
            text (str): The input string to tokenize.

        Returns:
            list[int]: A list of token IDs.
        """
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125  # pylint: disable=line-too-long
    def detokenize(self, ids):
        """Converts a list of token IDs back to a string, handling special tokens.

        This method reconstructs the text by decoding segments of regular token IDs
        using the SentencePiece model and inserting the string representations of
        special tokens where their IDs appear.

        Args:
            ids (list[int]): A list of token IDs.

        Returns:
            str: The detokenized string.
        """
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Calculates the character starting offsets for each token ID."""
        return [p.begin for p in self.tokenizer.decode_ids_as_immutable_proto(ids).pieces]

    @property
    def cls(self):
        """Returns the <CLS> token ID."""
        return self._cls_id

    @property
    def sep(self):
        """Returns the <SEP> token ID."""
        return self._sep_id

    @property
    def pad(self):
        """Returns the padding token ID (e.g., <PAD>)."""
        return self._pad_id

    @property
    def bos(self):
        """Returns the beginning-of-sentence token ID (e.g., <BOS>)."""
        return self._bos_id

    @property
    def eod(self):
        """Returns the end-of-document (<EOD>) token ID."""
        return self._eod_id

    @property
    def eos(self):
        """Returns the end-of-sentence token ID (e.g., <EOS>)."""
        return self._eos_id

    @property
    def mask(self):
        """Returns the <MASK> token ID."""
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        """Returns a list of IDs for T5-style <extra_id_*> sentinel tokens."""
        return [self.vocab[k] for k in self._t5_tokens]


class _GPTSentencePieceTokenizer(_SentencePieceTokenizer):
    """A specialized SentencePiece tokenizer for GPT-style models.

    This class inherits from `_SentencePieceTokenizer` but simplifies the special
    token handling. It primarily uses the BOS, EOS, and PAD IDs defined by the
    SentencePiece model itself, without adding extra tokens like <CLS>, <SEP>, etc.
    The `eod` (end-of-document) token is mapped to the `eos_id`.
    Args:
        model_file (str): Path to the SentencePiece model file.
    """

    def __init__(self, model_file):
        super().__init__(model_file, vocab_extra_ids=0)

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        self._pad_id = self.tokenizer.pad_id()
        self._bos_id = self.tokenizer.bos_id()
        self._eos_id = self.tokenizer.eos_id()

    def tokenize(self, text):
        """Tokenizes a string of text directly using SentencePiece encode_as_ids."""
        return self.tokenizer.encode_as_ids(text)

    def detokenize(self, ids):
        """Converts a list of token IDs back to a string using SentencePiece decode_ids."""
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        """Returns -1 as [CLS] is not typically used in this tokenizer."""
        return -1

    @property
    def sep(self):
        """Returns -1 as [SEP] is not typically used in this tokenizer."""
        return -1

    @property
    def mask(self):
        """Returns -1 as [MASK] is not typically used in this tokenizer."""
        return -1

    @property
    def eod(self):
        """Returns the end-of-sentence token ID, used as end-of-document."""
        return self._eos_id

    @property
    def eos(self):
        """Returns the EOS token ID."""
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        """Returns None as no additional special tokens are added by default."""
        return None


class _Llama2Tokenizer(_SentencePieceTokenizer):
    """A tokenizer specifically for Llama-2 style models, using SentencePiece.
    This class inherits from `_SentencePieceTokenizer` and is configured for Llama-2's
    specific use of BOS and EOS tokens. It uses the BOS/EOS/PAD IDs directly from
    the SentencePiece model.
    Args:
        model_file (str): Path to the SentencePiece model file for Llama-2.
    """

    def __init__(self, model_file):
        super().__init__(model_file, vocab_extra_ids=0)

        self.n_words: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()
        assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()

    def tokenize(self, s: str, bos=True, eos=False):
        """Tokenizes a string, with options to add BOS and EOS tokens.
        Args:
            s (str): The input string to tokenize.
            bos (bool, optional): Whether to prepend the BOS token. Defaults to True.
            eos (bool, optional): Whether to append the EOS token. Defaults to False.
        Returns:
            list[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.tokenizer.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def detokenize(self, ids):
        """Converts a list of token IDs back into a string."""
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        """Returns -1 as [CLS] is not typically used in this tokenizer."""
        return -1

    @property
    def sep(self):
        """Returns -1 as [SEP] is not typically used in this tokenizer."""
        return -1

    @property
    def mask(self):
        """Returns -1 as [MASK] is not typically used in this tokenizer."""
        return -1

    @property
    def eod(self):
        """Returns the end-of-sentence token ID, used as end-of-document."""
        return self.eos_id

    @property
    def eos(self):
        """Returns the EOS token ID."""
        return self.eos_id

    @property
    def additional_special_tokens_ids(self):
        """Returns None as no additional special tokens are added by default."""
        return None


def reload_mergeable_ranks(path: str, max_vocab: Optional[int] = None) -> Dict[bytes, int]:
    """
    Reloads a tokenizer vocabulary from a JSON file (NeMo format) and converts it
    into the mergeable ranks format required by Tiktoken.
    The input JSON file is expected to be a list of dictionaries, each with
    "rank", "token_bytes" (base64 encoded), and "token_str" keys.
    Args:
        path (str): Path to the JSON vocabulary file.
        max_vocab (Optional[int], optional): If provided, truncates the vocabulary
                                           to this maximum size. Defaults to None.
    Returns:
        Dict[bytes, int]: A dictionary mapping token bytes to their ranks.
    """
    assert path.endswith(".json")

    # reload vocab
    with open(path, "r") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    print_rank_0(f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
        print_rank_0(f"Cutting vocab to first {len(vocab)} tokens.")

    # build ranks
    ranks: Dict[bytes, int] = {}
    for i, x in enumerate(vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i])
        ranks[merge] = x["rank"]

    # sanity check
    assert len(ranks) == len(vocab)
    assert set(ranks.values()) == set(range(len(ranks)))

    return ranks


class CustomTikTokenizer(MegatronTokenizer):
    """A custom tokenizer using the Tiktoken library with a NeMo-style vocabulary file.
    This tokenizer loads a vocabulary from a JSON file (processed by
    `reload_mergeable_ranks`) and uses it with Tiktoken for encoding and decoding.
    It supports a configurable number of special tokens, which are placed at the
    beginning of the vocabulary ID space.
    Args:
        path (str): Path to the JSON vocabulary file (NeMo format).
        pattern (str): The regex pattern string for Tiktoken.
        vocab_size (Optional[int]): The target vocabulary size. If None, defaults to 2^17.
        num_special_tokens (int): The total number of special tokens to reserve.
        special_tokens (Optional[List[str]]): A list of initial special token strings.
                                            Must include "<unk>", "<s>", "</s>".
                                            If shorter than `num_special_tokens`,
                                            it will be padded with "<SPECIAL_id>".
    """

    def __init__(
        self,
        path: str,
        pattern: str,
        vocab_size: Optional[int],
        num_special_tokens: int,
        special_tokens: Optional[List[str]],
    ):
        super().__init__(
            path,
            pattern=pattern,
            vocab_size=vocab_size,
            num_special_tokens=num_special_tokens,
            special_tokens=special_tokens,
        )
        import tiktoken

        if vocab_size is None:
            vocab_size = 2**17  # Fallback vocab size is 131072.
        self._vocab_size = vocab_size

        SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]
        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS.copy()
        assert len(special_tokens) == len(set(special_tokens)), f"Special tokens should be unique: {special_tokens}"
        assert len(special_tokens) <= num_special_tokens < self._vocab_size
        assert set(SPECIAL_TOKENS) <= set(special_tokens), f"Custom special tokens should include {SPECIAL_TOKENS}"

        special_filler = ["<SPECIAL_{id}>".format(id=i) for i in range(len(special_tokens), num_special_tokens)]
        if special_filler:
            print_rank_0(f"Adding special tokens {special_filler[0]}, ..., {special_filler[-1]}")
        special_tokens = special_tokens + special_filler
        assert len(set(special_tokens)) == len(special_tokens) == num_special_tokens, special_tokens
        inner_vocab_size = self._vocab_size - num_special_tokens

        token_to_id_without_special_tokens = reload_mergeable_ranks(path, max_vocab=inner_vocab_size)
        # Create space for special tokens.
        token_to_id_without_special_tokens = {
            t: i + num_special_tokens for t, i in token_to_id_without_special_tokens.items()
        }

        special_tokens = {t: i for i, t in enumerate(special_tokens)}
        self._unk_id = special_tokens["<unk>"]
        self._bos_id = special_tokens["<s>"]
        self._eos_id = special_tokens["</s>"]

        # Create tiktoken model.
        self._model = tiktoken.Encoding(
            name=Path(path).parent.name,
            pat_str=pattern,
            mergeable_ranks=token_to_id_without_special_tokens,
            special_tokens=special_tokens,
        )

        # Create final _id_to_token and _token_to_id data structures with special tokens inserted
        # into appropriate locations.
        assert set(token_to_id_without_special_tokens.keys()).isdisjoint(set(special_tokens.keys()))
        self._token_to_id = token_to_id_without_special_tokens.copy()
        self._token_to_id.update(special_tokens)
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        assert set(range(self._vocab_size)) == set(self._id_to_token.keys())

    @property
    def bos(self) -> int:
        """Returns the beginning-of-sentence (<s>) token ID."""
        return self._bos_id

    @property
    def eos(self) -> int:
        """Returns the end-of-sentence (</s>) token ID."""
        return self._eos_id

    @property
    def unk(self) -> int:
        """Returns the unknown (<unk>) token ID."""
        return self._unk_id

    @property
    def eod(self) -> int:
        """Returns the end-of-document token ID (same as EOS for this tokenizer)."""
        return self._eos_id

    @property
    def vocab(self):
        """Returns the vocabulary (token string/bytes to ID mapping)."""
        return self._token_to_id

    @property
    def inv_vocab(self):
        """Returns the inverse vocabulary (ID to token string/bytes mapping)."""
        return self._id_to_token

    def tokenize(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """Tokenizes a string, with options to add BOS and EOS tokens.
        Args:
            s (str): The input string to tokenize.
            bos (bool, optional): Whether to prepend the BOS token. Defaults to False.
            eos (bool, optional): Whether to append the EOS token. Defaults to False.
        Returns:
            List[int]: A list of token IDs.
        """
        tokens = self._model.encode_ordinary(s)
        if bos:
            tokens = [self.bos, *tokens]
        if eos:
            tokens = [*tokens, self.eos]

        return tokens

    def detokenize(self, tokens: List[int]) -> str:
        """Converts a list of token IDs back into a string."""
        return self._model.decode(tokens)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Calculates the character starting offsets for each token ID."""
        return self._model.decode_with_offsets(ids)[1]

    @property
    def vocab_size(self) -> int:
        """Returns the total vocabulary size, including special tokens."""
        return self._vocab_size

    @property
    def encoder(self):
        """Alias for vocab."""
        return self._token_to_id

    @property
    def decoder(self):
        """Alias for inv_vocab."""
        return self._id_to_token


class _NullTokenizer(MegatronTokenizer):
    """A simple tokenizer that splits text by spaces and converts tokens to integers.
    This tokenizer is primarily for testing or placeholder purposes where actual
    linguistic tokenization is not required. It assumes tokens are space-separated
    integers.
    Args:
        vocab_size (int): The vocabulary size, excluding the EOD token.
                          The EOD token will be assigned `vocab_size` as its ID.
    """

    def __init__(self, vocab_size):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size = int(vocab_size)
        self._eod_id = self._vocab_size - 1

    def tokenize(self, text):
        """Tokenizes by splitting the string by spaces and converting parts to integers."""
        return [int(x) for x in text.split(" ")]

    def detokenize(self, ids):
        """Converts a list of integer IDs back to a space-separated string."""
        text = [str(x) for x in ids]
        return " ".join(text)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Calculates character offsets, assuming space-separated integer tokens."""
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    @property
    def vocab_size(self):
        """Returns the vocabulary size, including the EOD token."""
        return self._vocab_size

    @property
    def vocab(self):
        """Not implemented for NullTokenizer."""
        raise NotImplementedError

    @property
    def inv_vocab(self):
        """Not implemented for NullTokenizer."""
        raise NotImplementedError

    @property
    def cls(self):
        """Returns -1 as [CLS] is not used."""
        return -1

    @property
    def sep(self):
        """Returns -1 as [SEP] is not used."""
        return -1

    @property
    def mask(self):
        """Returns -1 as [MASK] is not used."""
        return -1

    @property
    def eod(self):
        """Returns the end-of-document token ID."""
        return self._eod_id

    @property
    def eos(self):
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        """Returns None as no additional special tokens are used."""
        return None


class _NullMultimodalTokenizer(MegatronTokenizerCore):
    def __init__(self, vocab_size, image_token=None, image_token_id=None):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size = int(vocab_size)
        self._eod_id = self._vocab_size - 1

        from megatron.core.models.multimodal.llava_model import (
            DEFAULT_IMAGE_TOKEN_INDEX,
            IMAGE_TOKEN,
        )

        self._image_token = image_token if image_token is not None else IMAGE_TOKEN
        self._image_token_id = image_token_id if image_token_id is not None else DEFAULT_IMAGE_TOKEN_INDEX

    def tokenize(self, text):
        return [int(x) for x in text.split(" ")]

    def detokenize(self, ids):
        text = [str(x) for x in ids]
        return " ".join(text)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    def convert_tokens_to_ids(self, tokens):
        ids = [(int(t) if t != self._image_token else self._image_token_id) for t in tokens.split("  ")]
        return ids if len(ids) > 1 else ids[0]

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos(self):
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        return None
