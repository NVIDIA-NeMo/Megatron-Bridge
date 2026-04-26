# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for pure helpers in megatron_mimo_training_llava_audio.py.

The full training script imports megatron.core extensions and a sibling
training module that aren't available in CPU-only test envs. The pure helpers
are extracted via AST so they can be exercised in isolation.
"""

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from megatron.bridge.data.megatron_mimo.dataset import MegatronMIMODataset


TRAINING_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "models"
    / "megatron_mimo"
    / "megatron_mimo_training_llava_audio.py"
)


def _extract_node(source_path: Path, name: str, kind, *, globals_extra=None):
    """Compile and return one top-level function or class from a Python file.

    Skips the module's import side effects so a heavy dep chain doesn't gate
    pure helpers.
    """
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, kind) and getattr(node, "name", None) == name:
            ns: dict = {"torch": torch, "os": os}
            if globals_extra:
                ns.update(globals_extra)
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(source_path), "exec"), ns)
            return ns[name]
    raise RuntimeError(f"{name} not found in {source_path}")


@pytest.fixture(scope="module")
def llava_preprocess():
    return _extract_node(TRAINING_PATH, "_llava_preprocess", ast.FunctionDef)


@pytest.fixture(scope="module")
def find_token_span():
    return _extract_node(TRAINING_PATH, "_find_token_span", ast.FunctionDef)


@pytest.fixture(scope="module")
def answer_masked_dataset_cls(find_token_span):
    return _extract_node(
        TRAINING_PATH,
        "_AnswerMaskedMegatronMIMODataset",
        ast.ClassDef,
        globals_extra={
            "MegatronMIMODataset": MegatronMIMODataset,
            "_find_token_span": find_token_span,
        },
    )


# ---------------------------------------------------------------------------
# _find_token_span
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindTokenSpan:
    def test_pattern_at_start(self, find_token_span):
        assert find_token_span(torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2])) == (0, 2)

    def test_pattern_in_middle(self, find_token_span):
        assert find_token_span(torch.tensor([1, 2, 3, 4, 5]), torch.tensor([3, 4])) == (2, 4)

    def test_pattern_at_end(self, find_token_span):
        assert find_token_span(torch.tensor([1, 2, 3, 4, 5]), torch.tensor([4, 5])) == (3, 5)

    def test_pattern_not_found(self, find_token_span):
        assert find_token_span(torch.tensor([1, 2, 3]), torch.tensor([4, 5])) == (-1, -1)

    def test_empty_pattern(self, find_token_span):
        assert find_token_span(torch.tensor([1, 2, 3]), torch.tensor([], dtype=torch.long)) == (-1, -1)

    def test_pattern_longer_than_seq(self, find_token_span):
        assert find_token_span(torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4])) == (-1, -1)

    def test_strict_first_token_mismatch_fails(self, find_token_span):
        assert find_token_span(torch.tensor([1, 2, 3, 4]), torch.tensor([99, 3, 4])) == (-1, -1)

    def test_allow_first_mismatch_matches_when_only_first_differs(self, find_token_span):
        """SentencePiece boundary case — pattern's first token differs but rest matches."""
        seq = torch.tensor([1, 2, 3, 4])
        # Match starts at index 1 because match[1:] == [3, 4] aligns with seq[2:4].
        assert find_token_span(seq, torch.tensor([99, 3, 4]), allow_first_mismatch=True) == (1, 4)

    def test_start_idx_skips_earlier_match(self, find_token_span):
        seq = torch.tensor([1, 2, 1, 2, 3])
        assert find_token_span(seq, torch.tensor([1, 2]), start_idx=1) == (2, 4)


# ---------------------------------------------------------------------------
# _llava_preprocess
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLlavaPreprocess:
    def test_concatenates_conversation_values(self, llava_preprocess):
        ex = {"conversations": [{"value": "hello"}, {"value": "world"}]}
        out = llava_preprocess(ex, dataset_root="/data")
        assert out["text"] == "hello world"

    def test_strips_image_marker(self, llava_preprocess):
        ex = {"conversations": [{"value": "describe <image>"}, {"value": "this scene"}]}
        out = llava_preprocess(ex, dataset_root="/data")
        assert "<image>" not in out["text"]
        assert "describe" in out["text"]

    def test_strips_audio_marker(self, llava_preprocess):
        ex = {"conversations": [{"value": "<audio> sound here"}]}
        out = llava_preprocess(ex, dataset_root="/data")
        assert "<audio>" not in out["text"]

    def test_relative_image_path_resolved_to_absolute(self, llava_preprocess):
        ex = {"conversations": [], "image": "subdir/img.jpg"}
        out = llava_preprocess(ex, dataset_root="/data")
        assert out["image"] == os.path.join("/data", "subdir/img.jpg")

    def test_absolute_image_path_left_unchanged(self, llava_preprocess):
        ex = {"conversations": [], "image": "/abs/path/img.jpg"}
        out = llava_preprocess(ex, dataset_root="/data")
        assert out["image"] == "/abs/path/img.jpg"

    def test_audio_dict_array_extracted(self, llava_preprocess):
        arr = np.array([0.1, 0.2, 0.3])
        ex = {"conversations": [], "audio": {"array": arr, "sampling_rate": 16000}}
        out = llava_preprocess(ex, dataset_root="/data")
        assert out["audio"] is arr

    def test_audio_path_loaded_via_soundfile(self, llava_preprocess, monkeypatch):
        arr = np.array([0.1, 0.2, 0.3])
        sf_mock = MagicMock()
        sf_mock.read.return_value = (arr, 16000)
        monkeypatch.setitem(sys.modules, "soundfile", sf_mock)

        ex = {"conversations": [], "audio": "rel/sound.wav"}
        out = llava_preprocess(ex, dataset_root="/data")

        sf_mock.read.assert_called_once_with(os.path.join("/data", "rel/sound.wav"))
        assert out["audio"] is arr

    def test_audio_absolute_path_not_joined(self, llava_preprocess, monkeypatch):
        sf_mock = MagicMock()
        sf_mock.read.return_value = (np.array([0.0]), 16000)
        monkeypatch.setitem(sys.modules, "soundfile", sf_mock)

        ex = {"conversations": [], "audio": "/abs/sound.wav"}
        llava_preprocess(ex, dataset_root="/data")

        sf_mock.read.assert_called_once_with("/abs/sound.wav")

    def test_audio_non_16khz_raises_value_error(self, llava_preprocess, monkeypatch):
        sf_mock = MagicMock()
        sf_mock.read.return_value = (np.array([0.0]), 22050)
        monkeypatch.setitem(sys.modules, "soundfile", sf_mock)

        ex = {"conversations": [], "audio": "/abs/sound.wav"}
        with pytest.raises(ValueError, match="16 kHz"):
            llava_preprocess(ex, dataset_root="/data")

    def test_no_image_or_audio_keys_does_not_crash(self, llava_preprocess):
        ex = {"conversations": [{"value": "just text"}]}
        out = llava_preprocess(ex, dataset_root="/data")
        assert out["text"] == "just text"
        assert "image" not in out
        assert "audio" not in out

    def test_empty_conversations(self, llava_preprocess):
        ex = {"conversations": []}
        out = llava_preprocess(ex, dataset_root="/data")
        assert out["text"] == ""

    def test_missing_conversations_key(self, llava_preprocess):
        out = llava_preprocess({}, dataset_root="/data")
        assert out["text"] == ""


# ---------------------------------------------------------------------------
# _AnswerMaskedMegatronMIMODataset
# ---------------------------------------------------------------------------


class _DeterministicTokenizer:
    """Maps each unique whitespace-split word to a stable int ID.

    Lets tests reason about exact `input_ids` positions without committing to
    a real tokenizer's vocab.
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self._word_to_id: dict[str, int] = {}
        self._next_id = 100

    def _ids_for(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._word_to_id:
                self._word_to_id[word] = self._next_id
                self._next_id += 1
            ids.append(self._word_to_id[word])
        return ids

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ):
        ids = self._ids_for(text)[:max_length]
        return {"input_ids": torch.tensor([ids])}


@pytest.mark.unit
class TestAnswerMaskedDataset:
    def _build(self, dataset_cls, examples, *, tokenizer=None, seq_length=16):
        return dataset_cls(
            examples=examples,
            processors={},
            tokenizer=tokenizer or _DeterministicTokenizer(),
            seq_length=seq_length,
            special_token_ids={},
            encoder_seq_lengths={},
            modality_columns={},
        )

    def test_loss_mask_only_on_answer_predicting_positions(self, answer_masked_dataset_cls):
        # Tokens in order: ["describe","this","image","a","cat"] → ids [100,101,102,103,104].
        # Answer "a cat" → [103, 104] found at indices [3, 5).
        # labels[max(0,3-1):5-1] = labels[2:4] = input_ids[3:5] = [103, 104]; rest = -100.
        examples = [
            {
                "text": "describe this image a cat",
                "conversations": [
                    {"from": "human", "value": "describe this image"},
                    {"from": "gpt", "value": "a cat"},
                ],
            }
        ]
        dataset = self._build(answer_masked_dataset_cls, examples)
        item = dataset[0]
        assert item["loss_mask"].sum().item() == 2
        assert item["loss_mask"][2].item() == 1
        assert item["loss_mask"][3].item() == 1
        assert item["labels"][2].item() == 103
        assert item["labels"][3].item() == 104
        for i in (0, 1, 4, 5, 10, 15):
            assert item["labels"][i].item() == -100

    def test_no_gpt_turns_falls_back_to_base_behavior(self, answer_masked_dataset_cls):
        examples = [
            {
                "text": "no gpt turn",
                "conversations": [{"from": "human", "value": "no gpt turn"}],
            }
        ]
        dataset = self._build(answer_masked_dataset_cls, examples, seq_length=8)
        item = dataset[0]
        # Base loss_mask is positive on real text positions; the override returns early.
        assert item["loss_mask"].sum().item() > 0

    def test_truncated_answer_skipped_silently(self, answer_masked_dataset_cls):
        """An answer that doesn't appear in input_ids leaves labels all -100, no crash."""
        examples = [
            {
                "text": "describe this image",
                "conversations": [
                    {"from": "human", "value": "describe this image"},
                    {"from": "gpt", "value": "completely different answer"},
                ],
            }
        ]
        dataset = self._build(answer_masked_dataset_cls, examples, seq_length=12)
        item = dataset[0]
        assert item["loss_mask"].sum().item() == 0
        assert torch.all(item["labels"] == -100)

    def test_image_marker_stripped_from_answer_before_search(self, answer_masked_dataset_cls):
        """`<image>` inside a gpt turn must be removed before tokenizing the answer."""
        examples = [
            {
                "text": "describe this image a cat",
                "conversations": [
                    {"from": "human", "value": "describe this image"},
                    {"from": "gpt", "value": "a <image> cat"},
                ],
            }
        ]
        dataset = self._build(answer_masked_dataset_cls, examples)
        item = dataset[0]
        # After cleaning, "a cat" tokens still match the same positions.
        assert item["loss_mask"].sum().item() == 2

    def test_multiple_gpt_turns_advance_search_idx(self, answer_masked_dataset_cls):
        """Each successive answer searches starting after the previous match.

        Uses multi-token answers because `_find_token_span(..., allow_first_mismatch=True)`
        matches vacuously on single-token patterns (`match[1:]` is empty).
        """
        examples = [
            {
                "text": "ask first answer one ask second answer two",
                "conversations": [
                    {"from": "human", "value": "ask first"},
                    {"from": "gpt", "value": "answer one"},
                    {"from": "human", "value": "ask second"},
                    {"from": "gpt", "value": "answer two"},
                ],
            }
        ]
        dataset = self._build(answer_masked_dataset_cls, examples)
        item = dataset[0]
        # tokens: ["ask","first","answer","one","ask","second","answer","two"]
        #     → ids [100, 101, 102, 103, 100, 104, 102, 105]
        # answer "answer one" → [102, 103] found at [2, 4):
        #     labels[1:3] = input_ids[2:4] = [102, 103]
        # answer "answer two" → [102, 105] found at [6, 8) (start_idx=4):
        #     labels[5:7] = input_ids[6:8] = [102, 105]
        assert item["loss_mask"].sum().item() == 4
        for i in (1, 2, 5, 6):
            assert item["loss_mask"][i].item() == 1, f"position {i} should be masked"
        assert item["labels"][1].item() == 102
        assert item["labels"][2].item() == 103
        assert item["labels"][5].item() == 102
        assert item["labels"][6].item() == 105


# ---------------------------------------------------------------------------
# _build_parallelism_config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parallelism_config_fn():
    cfg_mod = pytest.importorskip("megatron.bridge.models.megatron_mimo.megatron_mimo_config")
    return _extract_node(
        TRAINING_PATH,
        "_build_parallelism_config",
        ast.FunctionDef,
        globals_extra={
            "MegatronMIMOParallelismConfig": cfg_mod.MegatronMIMOParallelismConfig,
            "ModuleParallelismConfig": cfg_mod.ModuleParallelismConfig,
        },
    )


_PARALLELISM_ENV_VARS = (
    "MIMO_LLM_TP",
    "MIMO_LLM_PP",
    "MIMO_LLM_DP",
    "MIMO_LLM_OFFSET",
    "MIMO_VISION_TP",
    "MIMO_VISION_PP",
    "MIMO_VISION_DP",
    "MIMO_VISION_OFFSET",
    "MIMO_AUDIO_TP",
    "MIMO_AUDIO_PP",
    "MIMO_AUDIO_DP",
    "MIMO_AUDIO_OFFSET",
)


@pytest.mark.unit
class TestBuildParallelismConfig:
    def test_defaults_when_no_env_set(self, parallelism_config_fn, monkeypatch):
        for v in _PARALLELISM_ENV_VARS:
            monkeypatch.delenv(v, raising=False)
        cfg = parallelism_config_fn()
        # 8-GPU layout: LLM TP=4 ranks 0-3, vision TP=2 ranks 4-5, audio TP=2 ranks 6-7.
        llm = cfg.module_parallelisms["language"]
        assert (
            llm.tensor_model_parallel_size,
            llm.pipeline_model_parallel_size,
            llm.data_parallel_size,
            llm.rank_offset,
        ) == (4, 1, 1, 0)
        vision = cfg.module_parallelisms["images"]
        assert (vision.tensor_model_parallel_size, vision.rank_offset) == (2, 4)
        audio = cfg.module_parallelisms["audios"]
        assert (audio.tensor_model_parallel_size, audio.rank_offset) == (2, 6)

    def test_env_overrides_apply(self, parallelism_config_fn, monkeypatch):
        for v in _PARALLELISM_ENV_VARS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("MIMO_LLM_TP", "8")
        monkeypatch.setenv("MIMO_LLM_OFFSET", "10")
        monkeypatch.setenv("MIMO_VISION_TP", "1")
        monkeypatch.setenv("MIMO_AUDIO_DP", "4")
        cfg = parallelism_config_fn()
        assert cfg.module_parallelisms["language"].tensor_model_parallel_size == 8
        assert cfg.module_parallelisms["language"].rank_offset == 10
        assert cfg.module_parallelisms["images"].tensor_model_parallel_size == 1
        assert cfg.module_parallelisms["audios"].data_parallel_size == 4


# ---------------------------------------------------------------------------
# _make_audio_config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def make_audio_config_fn():
    tc_mod = pytest.importorskip("megatron.core.transformer.transformer_config")
    return _extract_node(
        TRAINING_PATH,
        "_make_audio_config",
        ast.FunctionDef,
        globals_extra={"TransformerConfig": tc_mod.TransformerConfig},
    )


@pytest.mark.unit
class TestMakeAudioConfig:
    def test_whisper_base_dimensions(self, make_audio_config_fn):
        """Whisper-base: 6 layers × 512 hidden × 8 heads × 2048 ffn."""
        cfg = make_audio_config_fn()
        assert cfg.num_layers == 6
        assert cfg.hidden_size == 512
        assert cfg.ffn_hidden_size == 2048
        assert cfg.num_attention_heads == 8

    def test_runs_in_bf16(self, make_audio_config_fn):
        cfg = make_audio_config_fn()
        assert cfg.bf16 is True
        assert cfg.pipeline_dtype == torch.bfloat16

    def test_dropouts_disabled_and_per_token_loss(self, make_audio_config_fn):
        """Pin the non-default training knobs that the audio config sets."""
        cfg = make_audio_config_fn()
        assert cfg.hidden_dropout == 0.0
        assert cfg.attention_dropout == 0.0
        assert cfg.gated_linear_unit is False
        assert cfg.calculate_per_token_loss is True
        assert cfg.normalization == "LayerNorm"


# ---------------------------------------------------------------------------
# _wrap_iter — audio batch packing logic
# ---------------------------------------------------------------------------


_AUDIO_SPECIAL_TOKEN_ID = 32002


@pytest.fixture(scope="module")
def wrap_iter_fn():
    return _extract_node(
        TRAINING_PATH,
        "_wrap_iter",
        ast.FunctionDef,
        globals_extra={"AUDIO_SPECIAL_TOKEN_ID": _AUDIO_SPECIAL_TOKEN_ID},
    )


@pytest.fixture
def cuda_is_noop(monkeypatch):
    """`_wrap_iter` calls `.cuda(non_blocking=True)`; make it a no-op so the test runs on CPU."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)


def _consume_one(wrap_iter_fn, batch, **kwargs):
    return next(wrap_iter_fn([batch], **kwargs))


@pytest.mark.unit
class TestWrapIter:
    def test_attention_mask_cleared(self, wrap_iter_fn, cuda_is_noop):
        batch = {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        out = _consume_one(wrap_iter_fn, batch)
        assert out["attention_mask"] is None

    def test_loss_mask_defaulted_to_ones_when_absent(self, wrap_iter_fn, cuda_is_noop):
        batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long)}
        out = _consume_one(wrap_iter_fn, batch)
        assert torch.equal(out["loss_mask"], torch.ones(2, 4, dtype=torch.float))

    def test_existing_loss_mask_preserved(self, wrap_iter_fn, cuda_is_noop):
        existing = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long), "loss_mask": existing}
        out = _consume_one(wrap_iter_fn, batch)
        assert torch.equal(out["loss_mask"], existing)

    def test_vision_pixel_values_remapped_to_clip_x(self, wrap_iter_fn, cuda_is_noop):
        pv = torch.randn(1, 3, 224, 224)
        batch = {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "modality_inputs": {"images": {"pixel_values": pv}},
        }
        out = _consume_one(wrap_iter_fn, batch)
        clip = out["modality_inputs"]["images"]["clip"]
        assert clip["x"].dtype == torch.bfloat16  # _wrap_iter hardcodes bf16 cast
        assert clip["x"].shape == pv.shape

    def test_audio_seq_lengths_full_signal(self, wrap_iter_fn, cuda_is_noop):
        """All-nonzero mel: seq_lengths = (mel_frames - 1) // 2 + 1."""
        af = torch.ones(1, 80, 6)  # all frames have energy
        batch = {
            "input_ids": torch.full((1, 10), _AUDIO_SPECIAL_TOKEN_ID, dtype=torch.long),
            "modality_inputs": {"audios": {"input_features": af}},
        }
        out = _consume_one(wrap_iter_fn, batch)
        whisper = out["modality_inputs"]["audios"]["whisper"]
        assert torch.equal(whisper["seq_lengths"], torch.tensor([3]))  # (6-1)//2 + 1

    def test_audio_seq_lengths_count_only_non_zero_frames(self, wrap_iter_fn, cuda_is_noop):
        """Trailing zero-energy frames (Whisper padding) are excluded."""
        af = torch.zeros(1, 80, 8)
        af[0, :, :4] = 1.0  # only first 4 frames carry signal
        batch = {
            "input_ids": torch.full((1, 10), _AUDIO_SPECIAL_TOKEN_ID, dtype=torch.long),
            "modality_inputs": {"audios": {"input_features": af}},
        }
        out = _consume_one(wrap_iter_fn, batch)
        whisper = out["modality_inputs"]["audios"]["whisper"]
        assert torch.equal(whisper["seq_lengths"], torch.tensor([2]))  # (4-1)//2 + 1

    def test_excess_audio_placeholders_replaced_with_pad(self, wrap_iter_fn, cuda_is_noop):
        """Audio placeholder tokens beyond the valid count get zeroed out."""
        af = torch.zeros(1, 80, 6)
        af[0, :, :2] = 1.0  # valid_frames=2 → seq_lengths = (2-1)//2 + 1 = 1
        input_ids = torch.full((1, 5), _AUDIO_SPECIAL_TOKEN_ID, dtype=torch.long)
        batch = {
            "input_ids": input_ids,
            "modality_inputs": {"audios": {"input_features": af}},
        }
        out = _consume_one(wrap_iter_fn, batch)
        # First placeholder kept, the other 4 zeroed.
        assert out["input_ids"][0, 0].item() == _AUDIO_SPECIAL_TOKEN_ID
        for i in range(1, 5):
            assert out["input_ids"][0, i].item() == 0

    def test_audio_input_features_cast_to_bf16(self, wrap_iter_fn, cuda_is_noop):
        """`_wrap_iter` casts audio input_features to bfloat16 to match model dtype."""
        af = torch.ones(1, 80, 4, dtype=torch.float32)
        batch = {
            "input_ids": torch.full((1, 4), _AUDIO_SPECIAL_TOKEN_ID, dtype=torch.long),
            "modality_inputs": {"audios": {"input_features": af}},
        }
        out = _consume_one(wrap_iter_fn, batch)
        assert out["modality_inputs"]["audios"]["whisper"]["input_features"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# _make_checkpoint_loader_hook
# ---------------------------------------------------------------------------


@pytest.fixture
def ckpt_hook_factory():
    """Returns (factory, fn_globals) so each test gets fresh dist + loader mocks.

    Use `factory.__globals__` (the function's actual globals dict) so any
    reassignment of mock entries propagates into the running function — the
    `globals_extra` dict passed in is copied into a fresh exec namespace.
    """
    factory = _extract_node(
        TRAINING_PATH,
        "_make_checkpoint_loader_hook",
        ast.FunctionDef,
        globals_extra={"dist": MagicMock(), "_load_tp_rank_weights": MagicMock()},
    )
    return factory, factory.__globals__


def _build_mimo_model(*, language=True, vision_clip=True, audio_whisper=True):
    """Construct a mocked MIMO model object with the attributes the hook reads."""
    model = MagicMock()
    model.language_model = MagicMock() if language else None
    grids = {
        "language": MagicMock(),
        "images": MagicMock(),
        "audios": MagicMock(),
    }
    model.mimo_config.module_to_grid_map = grids
    submodules = {}
    if vision_clip:
        sub = MagicMock()
        sub.encoders.clip = MagicMock()
        submodules["images"] = sub
    if audio_whisper:
        sub = MagicMock()
        sub.encoders.whisper = MagicMock()
        submodules["audios"] = sub
    model.modality_submodules = submodules
    return model


@pytest.mark.unit
class TestMakeCheckpointLoaderHook:
    def test_no_checkpoints_is_noop(self, ckpt_hook_factory):
        factory, ns = ckpt_hook_factory
        hook = factory()  # all None
        model = _build_mimo_model()
        result = hook([model])
        assert result == [model]
        ns["_load_tp_rank_weights"].assert_not_called()

    def test_language_checkpoint_loaded_when_present(self, ckpt_hook_factory):
        factory, ns = ckpt_hook_factory
        hook = factory(language_model_ckpt="/ckpts/llm")
        model = _build_mimo_model()
        hook([model])
        ns["_load_tp_rank_weights"].assert_called_once()
        # Second positional arg is the ckpt path.
        assert ns["_load_tp_rank_weights"].call_args.args[1] == "/ckpts/llm"

    def test_language_checkpoint_skipped_when_language_model_none(self, ckpt_hook_factory):
        """Encoder-only ranks have language_model=None — guard must skip the load."""
        factory, ns = ckpt_hook_factory
        hook = factory(language_model_ckpt="/ckpts/llm")
        model = _build_mimo_model(language=False)
        hook([model])
        ns["_load_tp_rank_weights"].assert_not_called()

    def test_vision_checkpoint_loaded_when_clip_present(self, ckpt_hook_factory):
        factory, ns = ckpt_hook_factory
        hook = factory(vision_encoder_ckpt="/ckpts/vit")
        model = _build_mimo_model()
        hook([model])
        ns["_load_tp_rank_weights"].assert_called_once()
        assert ns["_load_tp_rank_weights"].call_args.args[1] == "/ckpts/vit"

    def test_audio_checkpoint_loaded_when_whisper_present(self, ckpt_hook_factory):
        factory, ns = ckpt_hook_factory
        hook = factory(audio_encoder_ckpt="/ckpts/whisper")
        model = _build_mimo_model()
        hook([model])
        ns["_load_tp_rank_weights"].assert_called_once()
        assert ns["_load_tp_rank_weights"].call_args.args[1] == "/ckpts/whisper"

    def test_all_three_checkpoints_loaded_together(self, ckpt_hook_factory):
        factory, ns = ckpt_hook_factory
        hook = factory(
            language_model_ckpt="/ckpts/llm",
            vision_encoder_ckpt="/ckpts/vit",
            audio_encoder_ckpt="/ckpts/whisper",
        )
        hook([_build_mimo_model()])
        assert ns["_load_tp_rank_weights"].call_count == 3

    def test_skipped_when_modality_absent(self, ckpt_hook_factory):
        """LLM-only ranks have no images/audios submodules — the hook must not crash."""
        factory, ns = ckpt_hook_factory
        hook = factory(vision_encoder_ckpt="/ckpts/vit", audio_encoder_ckpt="/ckpts/whisper")
        model = _build_mimo_model(vision_clip=False, audio_whisper=False)
        hook([model])
        ns["_load_tp_rank_weights"].assert_not_called()


# ---------------------------------------------------------------------------
# _log
# ---------------------------------------------------------------------------


@pytest.fixture
def log_fn():
    """Returns (log, fn_globals) so tests can flip `_rank_log_file` and inspect dist mocks.

    Returning `fn.__globals__` (the function's actual globals dict) instead of the
    local `globals_extra` arg — `_extract_node` copies the latter into a fresh dict
    used as the exec namespace, so reassignments to the original wouldn't propagate.
    """
    fn = _extract_node(
        TRAINING_PATH,
        "_log",
        ast.FunctionDef,
        globals_extra={"dist": MagicMock(), "_rank_log_file": None},
    )
    return fn, fn.__globals__


@pytest.mark.unit
class TestLog:
    def test_uses_question_mark_when_dist_uninitialized(self, log_fn, capsys):
        log, ns = log_fn
        ns["dist"].is_initialized.return_value = False
        log("hello")
        out = capsys.readouterr().out
        assert "[Rank ?]" in out and "hello" in out

    def test_uses_get_rank_when_dist_initialized(self, log_fn, capsys):
        log, ns = log_fn
        ns["dist"].is_initialized.return_value = True
        ns["dist"].get_rank.return_value = 3
        log("hi")
        out = capsys.readouterr().out
        assert "[Rank 3]" in out

    def test_writes_to_file_when_set(self, log_fn, capsys):
        log, ns = log_fn
        ns["dist"].is_initialized.return_value = False
        fake_file = MagicMock()
        ns["_rank_log_file"] = fake_file
        log("hello")
        fake_file.write.assert_called_once()
        fake_file.flush.assert_called_once()
        # Still prints to stdout regardless.
        assert "hello" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _build_data_iterators
# ---------------------------------------------------------------------------


@pytest.fixture
def build_data_iterators_fn(monkeypatch):
    """Returns (build_fn, mocks) so tests can configure the mocked dataloader/TrainState."""
    # Stub the heavy modules the function imports inside its body. Use monkeypatch
    # so the stubs are reverted at test teardown — otherwise they leak across the
    # session and can break unrelated tests that import the real modules later.
    loaders_mock = MagicMock()
    state_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "megatron.bridge.data.megatron_mimo.loaders", loaders_mock)
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.state", state_mock)

    # Inject _wrap_iter as a passthrough so the test asserts wiring, not transformation.
    # Accepts arbitrary kwargs so the test isn't sensitive to whether the source passes
    # `model_dtype=` or not.
    captured_wrap_calls = []

    def fake_wrap_iter(loader_iter, **kwargs):
        captured_wrap_calls.append(loader_iter)
        return iter([])  # dummy iterator

    fn = _extract_node(
        TRAINING_PATH,
        "_build_data_iterators",
        ast.FunctionDef,
        globals_extra={
            "torch": torch,
            "_wrap_iter": fake_wrap_iter,
        },
    )
    return fn, loaders_mock, state_mock, captured_wrap_calls


@pytest.mark.unit
class TestBuildDataIterators:
    def _cfg(self, *, train_iters=10, global_batch_size=4):
        cfg = SimpleNamespace()
        cfg.train = SimpleNamespace(train_iters=train_iters, global_batch_size=global_batch_size)
        cfg.model = SimpleNamespace(bf16=True)
        cfg.dataset = MagicMock()
        return cfg

    def test_returns_train_iter_and_none_valid(self, build_data_iterators_fn):
        fn, loaders_mock, state_mock, captured = build_data_iterators_fn
        loaders_mock.build_megatron_mimo_data_loaders.return_value = (MagicMock(), None, None)
        train_iter, valid_iter = fn(self._cfg(), MagicMock())
        assert valid_iter is None
        assert train_iter is not None
        # Loader was passed through to _wrap_iter.
        assert len(captured) == 1

    def test_default_train_state_constructed_when_not_given(self, build_data_iterators_fn):
        fn, loaders_mock, state_mock, captured = build_data_iterators_fn
        loaders_mock.build_megatron_mimo_data_loaders.return_value = (MagicMock(), None, None)
        fn(self._cfg(), MagicMock())
        # TrainState was instantiated since train_state defaulted to None.
        state_mock.TrainState.assert_called_once()

    def test_passed_train_state_used_directly(self, build_data_iterators_fn):
        fn, loaders_mock, state_mock, captured = build_data_iterators_fn
        loaders_mock.build_megatron_mimo_data_loaders.return_value = (MagicMock(), None, None)
        existing_state = MagicMock()
        fn(self._cfg(), MagicMock(), train_state=existing_state)
        # Default TrainState NOT constructed when one was provided.
        state_mock.TrainState.assert_not_called()
        # The provided state is forwarded.
        kwargs = loaders_mock.build_megatron_mimo_data_loaders.call_args.kwargs
        assert kwargs["train_state"] is existing_state

    def test_train_samples_floor_is_10(self, build_data_iterators_fn):
        """Even tiny iter counts produce at least 10 samples (sanity floor)."""
        fn, loaders_mock, _, _ = build_data_iterators_fn
        loaders_mock.build_megatron_mimo_data_loaders.return_value = (MagicMock(), None, None)
        # train_iters=1, global_batch_size=1 → product=1; max(1, 10) = 10.
        fn(self._cfg(train_iters=1, global_batch_size=1), MagicMock())
        kwargs = loaders_mock.build_megatron_mimo_data_loaders.call_args.kwargs
        assert kwargs["train_samples"] == 10

    def test_no_loader_returns_none_train_iter(self, build_data_iterators_fn):
        fn, loaders_mock, _, _ = build_data_iterators_fn
        loaders_mock.build_megatron_mimo_data_loaders.return_value = (None, None, None)
        train_iter, valid_iter = fn(self._cfg(), MagicMock())
        assert train_iter is None
        assert valid_iter is None
