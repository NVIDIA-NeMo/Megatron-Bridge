"""Tests for the portable Nemotron ECD subset preparation example.

These use only pytest and the stdlib, without Torch, Transformers, or Energon.
The real processor/collator execution is a separate runtime qualification.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/models/qwen/qwen4_exp/prepare_nemotron_image_v3_subset.py"
SPEC = importlib.util.spec_from_file_location("prepare_nemotron_subset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class TestNemotronSubsetPreparation(unittest.TestCase):
    def test_complete_aligned_sample_at_cap(self):
        self.assertEqual(prep.classify(length=2041, supervised=3, alignment=8, limit=2048), ("accepted", 2048))

    def test_overlength_sample_is_skipped_without_truncation(self):
        self.assertEqual(prep.classify(length=2049, supervised=3, alignment=8, limit=2048), ("overlength", 2056))

    def test_invalid_supervision_and_geometry_fail(self):
        for length, supervised, alignment in ((20, 0, 8), (20, 21, 8), (20, 2, 0)):
            with self.subTest(length=length, supervised=supervised, alignment=alignment):
                with self.assertRaises(ValueError):
                    prep.classify(length=length, supervised=supervised, alignment=alignment, limit=2048)

    def _args(self, checkpoint: Path, *extra: str):
        return prep.parse_args(
            ["--source-dir", "downloads", "--output-dir", "prepared", "--checkpoint", str(checkpoint), *extra]
        )

    def test_portable_checkout_discovery_and_validated_defaults(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            args = self._args(Path(checkpoint))
        self.assertEqual(args.bridge_root, ROOT)
        self.assertEqual((args.max_samples, args.num_workers), (8192, 4))
        self.assertEqual(
            (args.max_seq_length, args.min_pixels, args.max_pixels, args.alignment), (2048, 65536, 262144, 8)
        )

    def test_small_subset_and_one_worker_are_supported(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            args = self._args(Path(checkpoint), "--max-samples", "1", "--num-workers", "1")
        self.assertEqual((args.max_samples, args.num_workers), (1, 1))

    def test_out_of_scope_bounds_fail_before_download(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            for option, value in (
                ("--max-samples", "0"),
                ("--max-samples", "8193"),
                ("--num-workers", "0"),
                ("--num-workers", "5"),
                ("--min-pixels", "16384"),
                ("--max-seq-length", "4096"),
                ("--alignment", "1"),
            ):
                with self.subTest(option=option, value=value):
                    with self.assertRaises(ValueError):
                        self._args(Path(checkpoint), option, value)

    def test_missing_checkout_or_local_processor_fails(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            with self.assertRaises(ValueError):
                self._args(Path(checkpoint), "--bridge-root", checkpoint)
            with self.assertRaises(ValueError):
                self._args(Path(checkpoint) / "absent")

    def test_repository_source_hashes_are_observed_not_old_revision_pins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in prep.SOURCE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"# arbitrary checked-out source\n")
            initial = prep._source_dependencies(root)
            changed = root / prep.SOURCE_FILES[0]
            changed.write_bytes(b"# revised converter\n")
            observed = prep._source_dependencies(root)
            self.assertNotEqual(initial, observed)
            self.assertEqual(observed[prep.SOURCE_FILES[0]], hashlib.sha256(changed.read_bytes()).hexdigest())
            changed.unlink()
            with self.assertRaises(FileNotFoundError):
                prep._source_dependencies(root)

    def test_actual_converter_writer_keeps_images_and_messages_in_correct_fields(self):
        # Execute the real stdlib writer function, not a mocked framework implementation.
        path = ROOT / prep.SOURCE_FILES[0]
        node = next(
            n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_write_sample"
        )
        namespace = {"json": json, "pickle": pickle}
        tree = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
        records = []
        images = [b"\x89PNG\r\n\x1a\nimage-content"]
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "chart?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
        prep._emit_sample(
            SimpleNamespace(_write_sample=namespace["_write_sample"]),
            SimpleNamespace(write=records.append),
            ("ecd__sample", messages, images),
        )
        self.assertEqual(records[0]["__key__"], "ecd__sample")
        self.assertEqual(pickle.loads(records[0]["jpgs"]), images)
        self.assertEqual(json.loads(records[0]["json"]), messages)


if __name__ == "__main__":
    unittest.main()
