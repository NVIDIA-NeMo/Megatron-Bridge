# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for pure helpers in synthesize_llava_pretrain_audio.py.

The script's heavy dependencies (NeMo TTS) are deferred to function bodies, so
the module itself imports cleanly with stdlib only and can be loaded directly.
"""

import importlib.util
from pathlib import Path

import pytest


SYNTH_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "models"
    / "megatron_mimo"
    / "synthesize_llava_pretrain_audio.py"
)


@pytest.fixture(scope="module")
def synth():
    spec = importlib.util.spec_from_file_location("synthesize_llava_pretrain_audio_under_test", SYNTH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
class TestExtractHumanText:
    def test_returns_first_human_turn_only(self, synth):
        record = {
            "conversations": [
                {"from": "human", "value": "describe this"},
                {"from": "gpt", "value": "a cat"},
                {"from": "human", "value": "second turn"},
            ]
        }
        assert synth.extract_human_text(record) == "describe this"

    def test_strips_image_and_audio_markers(self, synth):
        record = {"conversations": [{"from": "human", "value": "describe <image> and <audio>"}]}
        assert synth.extract_human_text(record) == "describe and"

    def test_collapses_whitespace_runs(self, synth):
        record = {"conversations": [{"from": "human", "value": "describe   <image>   this"}]}
        assert synth.extract_human_text(record) == "describe this"

    def test_no_human_turn_returns_none(self, synth):
        record = {"conversations": [{"from": "gpt", "value": "answer"}]}
        assert synth.extract_human_text(record) is None

    def test_empty_after_stripping_returns_none(self, synth):
        record = {"conversations": [{"from": "human", "value": "<image>"}]}
        assert synth.extract_human_text(record) is None

    def test_no_conversations_key(self, synth):
        assert synth.extract_human_text({}) is None

    def test_skips_non_human_before_finding_human(self, synth):
        record = {
            "conversations": [
                {"from": "system", "value": "ignore me"},
                {"from": "gpt", "value": "ignore me too"},
                {"from": "human", "value": "the real prompt"},
            ]
        }
        assert synth.extract_human_text(record) == "the real prompt"


@pytest.mark.unit
class TestAudioRelPath:
    def test_uses_image_dir_basename_as_prefix(self, synth):
        record = {"id": "abc12345", "image": "data/00453/img.jpg"}
        assert synth.audio_rel_path(record, "audio") == "audio/00453/abc12345.flac"

    def test_works_with_absolute_image_path(self, synth):
        """Absolutized image paths still yield a relative output under <output_subdir>/<prefix>/."""
        record = {"id": "abc12345", "image": "/abs/path/00453/foo.jpg"}
        assert synth.audio_rel_path(record, "audio") == "audio/00453/abc12345.flac"

    def test_falls_back_to_id_prefix_when_no_image(self, synth):
        record = {"id": "abc12345"}
        assert synth.audio_rel_path(record, "audio") == "audio/abc12/abc12345.flac"

    def test_empty_image_string_falls_back_to_id_prefix(self, synth):
        record = {"id": "abc12345", "image": ""}
        assert synth.audio_rel_path(record, "audio") == "audio/abc12/abc12345.flac"

    def test_custom_output_subdir(self, synth):
        record = {"id": "abc12345", "image": "data/00453/img.jpg"}
        assert synth.audio_rel_path(record, "tts_out") == "tts_out/00453/abc12345.flac"
