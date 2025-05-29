# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

import base64
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer as MegatronTokenizerCore

from nemo_lm.tokenizers.bert_tokenization import FullTokenizer as FullBertTokenizer
from nemo_lm.tokenizers.gpt2_tokenization import GPT2Tokenizer
from nemo_lm.tokenizers.multimodal_tokenizer import MultimodalTokenizer
from nemo_lm.training.config import TokenizerConfig
from nemo_lm.utils.common_utils import get_rank_safe, print_rank_0, ensure_bool


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
        make_vocab_size_divisible_by (int): Ensures the vocabulary size is a multiple of this value.
        tensor_model_parallel_size (int): The tensor model parallel size, used for further
                                          adjusting vocabulary size for distributed training.
        **kwargs: Additional keyword arguments that might be specific to certain tokenizers
                  (e.g., passed to HuggingFace AutoTokenizer).

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
        tokenizer = _HuggingFaceTokenizer(tokenizer_config.tokenizer_model, **kwargs)
    elif tokenizer_config.tokenizer_type == "Llama2Tokenizer":
        assert tokenizer_config.tokenizer_model is not None
        tokenizer = _Llama2Tokenizer(tokenizer_config.tokenizer_model)
    elif tokenizer_config.tokenizer_type == "TikTokenizer":
        assert tokenizer_config.tokenizer_model is not None
        assert tokenizer_config.tiktoken_pattern is not None
        assert tokenizer_config.tiktoken_pattern in {"v1", "v2"}
        pattern = PATTERN_TIKTOKEN if tokenizer_config.tiktoken_pattern == "v1" else PATTERN_TIKTOKEN_V2
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
    else:
        raise NotImplementedError("{} tokenizer is not implemented.".format(tokenizer_config.tokenizer_type))

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


class _HuggingFaceTokenizer(MegatronTokenizer):
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__(pretrained_model_name_or_path, **kwargs)
        try:
            import transformers
        except ImportError:
            raise EnvironmentError("The transformers library must be installed to use huggingface_tokenizer_provider")

        # TODO(bnorick): download tokenizer once to lustre
        # and use force offline to make sure all tasks read it from there
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )
        self._vocab = self._tokenizer.get_vocab()
        self._inv_vocab = {token_id: token for token, token_id in self._vocab.items()}

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

    This class inherits from `