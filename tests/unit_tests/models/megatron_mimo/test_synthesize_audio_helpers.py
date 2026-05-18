# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for pure helpers in synthesize_llava_pretrain_audio.py.

The script's heavy dependencies (NeMo TTS) are deferred to function bodies, so
the module itself imports cleanly with stdlib only and can be loaded directly.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


def _write_input_json(root: Path, records: list[dict], name: str = "input.json") -> str:
    path = root / name
    path.write_text(json.dumps(records))
    return name


def _write_shard(manifest_dir: Path, shard_idx: int, entries: list[dict]) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / f"shard_{shard_idx:05d}.jsonl").open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _merge_args(
    dataset_root: Path, *, input_json="input.json", output_json="out.json", manifest_subdir="manifest"
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=str(dataset_root),
        input_json=input_json,
        output_json=output_json,
        manifest_subdir=manifest_subdir,
    )


@pytest.mark.unit
class TestRunMerge:
    """Tests for run_merge: combine per-shard JSONL manifests into the augmented JSON."""

    def test_records_with_matching_id_get_audio_field(self, synth, tmp_path):
        _write_input_json(
            tmp_path,
            [
                {"id": "a", "text": "alpha"},
                {"id": "b", "text": "beta"},
            ],
        )
        _write_shard(
            tmp_path / "manifest",
            0,
            [
                {"id": "a", "audio": "audio/a.flac"},
                {"id": "b", "audio": "audio/b.flac"},
            ],
        )
        synth.run_merge(_merge_args(tmp_path))
        out = json.loads((tmp_path / "out.json").read_text())
        assert out == [
            {"id": "a", "text": "alpha", "audio": "audio/a.flac"},
            {"id": "b", "text": "beta", "audio": "audio/b.flac"},
        ]

    def test_records_without_matching_shard_entry_are_dropped(self, synth, tmp_path):
        _write_input_json(
            tmp_path,
            [
                {"id": "a", "text": "alpha"},
                {"id": "missing", "text": "no audio"},
                {"id": "b", "text": "beta"},
            ],
        )
        _write_shard(
            tmp_path / "manifest",
            0,
            [
                {"id": "a", "audio": "a.flac"},
                {"id": "b", "audio": "b.flac"},
            ],
        )
        synth.run_merge(_merge_args(tmp_path))
        out = json.loads((tmp_path / "out.json").read_text())
        assert [r["id"] for r in out] == ["a", "b"]

    def test_multiple_shard_files_merged(self, synth, tmp_path):
        _write_input_json(tmp_path, [{"id": str(i)} for i in range(4)])
        _write_shard(
            tmp_path / "manifest",
            0,
            [
                {"id": "0", "audio": "0.flac"},
                {"id": "1", "audio": "1.flac"},
            ],
        )
        _write_shard(
            tmp_path / "manifest",
            1,
            [
                {"id": "2", "audio": "2.flac"},
                {"id": "3", "audio": "3.flac"},
            ],
        )
        synth.run_merge(_merge_args(tmp_path))
        out = json.loads((tmp_path / "out.json").read_text())
        assert {r["id"] for r in out} == {"0", "1", "2", "3"}
        assert all(r.get("audio") for r in out)

    def test_empty_manifest_dir_drops_all_records(self, synth, tmp_path):
        _write_input_json(tmp_path, [{"id": "a"}, {"id": "b"}])
        (tmp_path / "manifest").mkdir()
        synth.run_merge(_merge_args(tmp_path))
        assert json.loads((tmp_path / "out.json").read_text()) == []

    def test_record_without_id_is_dropped(self, synth, tmp_path):
        _write_input_json(tmp_path, [{"text": "no id"}, {"id": "a"}])
        _write_shard(tmp_path / "manifest", 0, [{"id": "a", "audio": "a.flac"}])
        synth.run_merge(_merge_args(tmp_path))
        out = json.loads((tmp_path / "out.json").read_text())
        assert [r["id"] for r in out] == ["a"]

    def test_shard_overrides_audio_when_id_repeats(self, synth, tmp_path):
        """Later shard entries with the same id overwrite earlier ones (last-write-wins)."""
        _write_input_json(tmp_path, [{"id": "a"}])
        _write_shard(tmp_path / "manifest", 0, [{"id": "a", "audio": "first.flac"}])
        _write_shard(tmp_path / "manifest", 1, [{"id": "a", "audio": "second.flac"}])
        synth.run_merge(_merge_args(tmp_path))
        out = json.loads((tmp_path / "out.json").read_text())
        assert out[0]["audio"] == "second.flac"


@pytest.mark.unit
class TestMain:
    """Tests for main(): dispatches to run_synth or run_merge based on --mode."""

    def test_synth_mode_calls_run_synth(self, synth, monkeypatch):
        captured = {}
        monkeypatch.setattr(synth, "parse_args", lambda: SimpleNamespace(mode="synth"))
        monkeypatch.setattr(synth, "run_synth", lambda args: captured.setdefault("synth", args))
        monkeypatch.setattr(synth, "run_merge", lambda args: captured.setdefault("merge", args))
        synth.main()
        assert "synth" in captured and "merge" not in captured

    def test_merge_mode_calls_run_merge(self, synth, monkeypatch):
        captured = {}
        monkeypatch.setattr(synth, "parse_args", lambda: SimpleNamespace(mode="merge"))
        monkeypatch.setattr(synth, "run_synth", lambda args: captured.setdefault("synth", args))
        monkeypatch.setattr(synth, "run_merge", lambda args: captured.setdefault("merge", args))
        synth.main()
        assert "merge" in captured and "synth" not in captured
