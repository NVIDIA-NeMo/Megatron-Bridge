# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only selection contracts of the agentic SFT filter (``02_5_filter_coderforge.py``).

Adapted from the Megatron-LM agentic SFT example by Zhongbo Zhu.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


pytestmark = pytest.mark.unit

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "agentic_sft"


def load_source(name, filename):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


filtering = load_source("agentic_sft_filtering", "02_5_filter_coderforge.py")
jsonl_rows = sys.modules["agentic_sft_data"]
SEQUENCE_LENGTH = 65536


def measured(identifier, length, row_index=0, original_length=None):
    return {
        "trajectory_id": identifier,
        "row_index": row_index,
        "runtime_tokens": length,
        "original_runtime_tokens": length if original_length is None else original_length,
        "supervised_tokens": min(length, 17),
    }


class SelectionTests(unittest.TestCase):
    def test_minimum_cp_boundaries_use_postshift_runtime_length(self):
        for length, bucket in ((1, 0), (16384, 0), (16385, 1), (32768, 1), (32769, 2), (65536, 2)):
            with self.subTest(length=length):
                self.assertEqual(filtering.length_bucket(length, SEQUENCE_LENGTH), bucket)
        for length, sequence_length in (
            (0, 65536),
            (65537, 65536),
            (1, 7),
            (1, 8),
            (1, 24),
            (1, 65535),
        ):
            with self.subTest(length=length, sequence_length=sequence_length):
                with self.assertRaises(ValueError):
                    filtering.length_bucket(length, sequence_length)

    def test_natural_preserves_every_index_even_with_missing_buckets(self):
        ids, lengths = ["third", "first", "second"], [65000, 33000, 40000]
        before = deepcopy((ids, lengths))
        self.assertEqual(filtering.select_indices(ids, lengths, "natural", 1234, 65536), [0, 1, 2])
        self.assertEqual((ids, lengths), before)

    def test_largest_remainder_quotas_and_invalid_ratios(self):
        self.assertEqual(filtering.bucket_quotas(5120, (0.5, 0.3, 0.2)), [2560, 1536, 1024])
        self.assertEqual(filtering.bucket_quotas(256, (0.5, 0.3, 0.2)), [128, 77, 51])
        self.assertEqual(filtering.bucket_quotas(2, (1 / 3, 1 / 3, 1 / 3)), [1, 1, 0])
        for count, ratios in (
            (0, (1, 0, 0)),
            (3, (0.5, 0.2, 0.2)),
            (3, (-0.1, 0.1, 1)),
            (3, (float("nan"), 0, 1)),
            (3, (1, 0)),
        ):
            with self.subTest(count=count, ratios=ratios), self.assertRaises(ValueError):
                filtering.bucket_quotas(count, ratios)

    def test_mixed_prefers_native_then_unique_verified_prefixes_and_shuffles(self):
        records = [measured(f"id-{i}", length, i) for i, length in enumerate([10000] + [20000] * 4 + [60000] * 20)]
        before = deepcopy(records)
        tasks_seen = []

        def resolve(tasks):
            tasks_seen.extend(tasks)
            return [
                dict(
                    measured(records[i]["trajectory_id"], 8000, i),
                    is_prefix=True,
                    end_message_index=1,
                )
                for i, bucket, target in tasks
            ]

        selected = filtering.select_mixed_records(records, [5, 3, 2], 1234, 65536, resolve)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({r["source_id"] for r in selected}), 10)
        self.assertEqual(
            Counter(filtering.length_bucket(r["runtime_tokens"], 65536) for r in selected),
            {0: 5, 1: 3, 2: 2},
        )
        self.assertEqual(sum(r["is_prefix"] for r in selected), 4)
        self.assertIn("id-0", [r["source_id"] for r in selected])
        self.assertEqual(selected, filtering.select_mixed_records(records, [5, 3, 2], 1234, 65536, resolve))
        self.assertEqual(
            [r["source_id"] for r in selected],
            [r["source_id"] for r in filtering.select_mixed_records(records[::-1], [5, 3, 2], 1234, 65536, resolve)],
        )
        self.assertEqual(records, before)
        self.assertTrue(all(bucket == 0 and 1 <= target <= 16384 for _, bucket, target in tasks_seen))
        self.assertGreater(len({target for _, _, target in tasks_seen}), 3)
        self.assertNotEqual(
            [filtering.length_bucket(r["runtime_tokens"], 65536) for r in selected],
            sorted(filtering.length_bucket(r["runtime_tokens"], 65536) for r in selected),
        )

    def test_mixed_shortages_and_invalid_verified_prefixes_fail(self):
        records = [measured(str(i), 60000, i) for i in range(6)]
        for quotas, resolve, error in (
            ([4, 3, 1], lambda tasks: [], "unique trainable"),
            ([2, 1, 1], lambda tasks: [None] * len(tasks), "prefix search shortage"),
            ([0, 0, 7], lambda tasks: [], "unique trainable"),
            (
                [2, 1, 1],
                lambda tasks: [dict(measured(str(i), 20000, i), is_prefix=True) for i, _, _ in tasks],
                "Invalid verified",
            ),
        ):
            with (
                self.subTest(quotas=quotas, error=error),
                self.assertRaisesRegex(ValueError, error),
            ):
                filtering.select_mixed_records(records, quotas, 1234, 65536, resolve)
        with self.assertRaisesRegex(ValueError, "long bucket shortage"):
            filtering.select_mixed_records(
                [measured(str(i), 1000, i) for i in range(6)],
                [1, 1, 2],
                1234,
                65536,
                lambda tasks: [],
            )

    def test_prefixes_end_at_complete_assistant_messages_including_tool_calls(self):
        row = {
            "messages": [
                {"role": "system", "content": "system", "tokens": 100},
                {"role": "user", "content": "task", "tokens": 100},
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "search"}}],
                    "tokens": 5000,
                },
                {"role": "tool", "content": "response", "tokens": 3000},
                {"role": "assistant", "content": "reasoning and response", "tokens": 3000},
                {"role": "user", "content": "later", "tokens": 20000},
                {"role": "assistant", "content": "last", "tokens": 1000},
            ]
        }
        before = deepcopy(row)
        seen = []

        def encode(messages):
            seen.append(deepcopy(messages))
            length = sum(message["tokens"] for message in messages)
            return [0] * (length + 1), [-100] * length + [42]

        result = filtering.find_prefix(row, 0, 5200, 65536, encode)
        self.assertEqual(result["end_message_index"], 2)
        self.assertEqual(result["runtime_tokens"], 5200)
        self.assertEqual(result["original_runtime_tokens"], 5200)
        self.assertEqual(result["supervised_tokens"], 1)
        self.assertEqual(row, before)
        self.assertTrue(
            all(
                messages == row["messages"][: len(messages)] and messages[-1]["role"] == "assistant"
                for messages in seen
            )
        )
        self.assertLessEqual(len(seen), filtering.MAX_PREFIX_ENCODINGS)

    def test_prefix_verification_rejects_out_of_bucket_and_empty_targets(self):
        row = {"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]}
        self.assertIsNone(filtering.find_prefix(row, 0, 5000, 65536, lambda messages: ([0] * 17001, [1] * 17001)))
        self.assertIsNone(filtering.find_prefix(row, 0, 5000, 65536, lambda messages: ([0] * 5001, [-100] * 5001)))
        with self.assertRaisesRegex(ValueError, "configuration"):
            filtering.find_prefix(row, 0, 5000, 65536, Mock(side_effect=ValueError("bad configuration")))
        self.assertIsNone(
            filtering.find_prefix(
                row,
                0,
                5000,
                65536,
                Mock(side_effect=ValueError("No next-token assistant targets in this conversation")),
            )
        )

    def test_invalid_selection_inputs_and_duplicate_split_ids_are_rejected(self):
        for ids, lengths, distribution in (
            (["a"], [], "natural"),
            (["a", "a"], [1, 2], "natural"),
            (["a"], [1], "invalid"),
        ):
            with self.subTest(ids=ids, distribution=distribution):
                with self.assertRaises(ValueError):
                    filtering.select_indices(ids, lengths, distribution, 1234, 65536)
        for splits in (
            {"training": [measured("a", 1), measured("a", 2)]},
            {"training": [measured("a", 1)], "validation": [measured("a", 2)]},
            {"training": [measured("", 1)]},
        ):
            with self.subTest(splits=splits):
                with self.assertRaisesRegex(ValueError, "duplicate trajectory ID"):
                    filtering.validate_unique_ids(splits)

    def test_summary_reconciles_capped_lengths_and_supervision(self):
        lengths = [1, 16384, 16385, 32768, 32769, 65536]
        records = [measured(str(i), length, i) for i, length in enumerate(lengths)]
        records[-1]["original_runtime_tokens"] = 70000
        summary = filtering.summarize_lengths(records, 65536)
        self.assertEqual(summary["samples"], 6)
        self.assertEqual(summary["runtime_tokens"], sum(lengths))
        self.assertEqual(summary["original_runtime_tokens"], sum(lengths[:-1]) + 70000)
        self.assertEqual(summary["supervised_tokens"], 1 + 5 * 17)
        self.assertEqual(summary["truncated_samples"], 1)
        self.assertEqual([bucket["samples"] for bucket in summary["buckets"]], [2, 2, 2])
        self.assertEqual([bucket["minimum_cp_size"] for bucket in summary["buckets"]], [1, 2, 4])
        self.assertAlmostEqual(sum(bucket["token_fraction"] for bucket in summary["buckets"]), 1)
        self.assertAlmostEqual(sum(bucket["sample_fraction"] for bucket in summary["buckets"]), 1)


class FileSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.data_dir = self.directory / "input"
        self.data_dir.mkdir()
        self.lines = {}
        self.index = {"fingerprints": {"inputs": {}}, "splits": {}}
        for split, lengths in (
            ("training", [10000] * 4 + [20000] * 3 + [50000] * 2),
            ("validation", [10000] * 2 + [20000] * 2 + [50000]),
        ):
            self.lines[split] = []
            self.index["splits"][split] = []
            for i, length in enumerate(lengths):
                identifier = f"{split}-{i}"
                row = {
                    "trajectory_id": identifier,
                    "task_id": f"{split}-task-{i // 2}",
                    "messages": [
                        {"role": "user", "content": "完整输入 " * 20},
                        {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "Keep historical reasoning.",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": "test", "arguments": '{"x":1}'},
                                }
                            ],
                        },
                        {"role": "tool", "content": "passed", "tool_call_id": "call-1"},
                        {"role": "assistant", "content": "All done."},
                    ],
                    "tools": [{"type": "function", "function": {"name": "test"}}],
                    "metadata": {"license": "source-license", "nested": [1, "keep"]},
                }
                self.lines[split].append((json.dumps(row, ensure_ascii=False) + "\n").encode())
                self.index["splits"][split].append(measured(identifier, length, i))
            path = self.data_dir / f"{split}.jsonl"
            path.write_bytes(b"\n" + b"\n".join(self.lines[split]))
            self.index["fingerprints"]["inputs"][split] = filtering.sha256_file(path)

    def assert_no_published_outputs(self, output_dir):
        for name in ("training.jsonl", "validation.jsonl", "manifest.json"):
            self.assertFalse((output_dir / name).exists(), name)

    def test_writer_keeps_complete_original_rows_and_split_membership(self):
        for distribution in ("natural",):
            with self.subTest(distribution=distribution):
                output_dir = self.directory / distribution
                outputs = filtering.write_selection(self.data_dir, output_dir, self.index, distribution, 1234, 65536)
                for split, records in self.index["splits"].items():
                    selected = filtering.select_indices(
                        [r["trajectory_id"] for r in records],
                        [r["runtime_tokens"] for r in records],
                        distribution,
                        1234,
                        65536,
                    )
                    path = output_dir / f"{split}.jsonl"
                    self.assertEqual(path.read_bytes(), b"".join(self.lines[split][i] for i in selected))
                    self.assertEqual(outputs[split]["sha256"], filtering.sha256_file(path))
                    self.assertEqual(outputs[split]["rows"], len(selected))
                    self.assertEqual(outputs[split]["candidate_lengths"]["samples"], len(records))
                    self.assertEqual(outputs[split]["selected_lengths"]["samples"], len(selected))

    def test_unverified_mixed_fails_before_publishing_either_split(self):
        output_dir = self.directory / "unverified"
        with self.assertRaisesRegex(ValueError, "verified prefix"):
            filtering.write_selection(self.data_dir, output_dir, self.index, "mixed", 1234, 65536)
        self.assert_no_published_outputs(output_dir)

    def test_mixed_writer_preserves_original_fields_prefixes_and_sidecar_order(self):
        selections = {}
        for split, records in self.index["splits"].items():
            native = dict(
                records[2],
                source_id=records[2]["trajectory_id"],
                is_prefix=False,
                source_original_runtime_tokens=records[2]["original_runtime_tokens"],
            )
            prefix = dict(
                records[0],
                source_id=records[0]["trajectory_id"],
                is_prefix=True,
                source_original_runtime_tokens=records[0]["original_runtime_tokens"],
                end_message_index=1,
                runtime_tokens=5000,
                original_runtime_tokens=5000,
            )
            selections[split] = [native, prefix]
        output_dir = self.directory / "mixed_prefix"
        outputs = filtering.write_selection(
            self.data_dir, output_dir, self.index, "mixed", 1234, 65536, mixed_selections=selections
        )
        sidecar = json.loads((output_dir / "selection.json").read_text())
        for split in self.index["splits"]:
            written = [json.loads(line) for line in (output_dir / f"{split}.jsonl").read_text().splitlines()]
            for row, measurement in zip(written, sidecar["splits"][split]):
                original = json.loads(self.lines[split][measurement["row_index"]])
                self.assertEqual(row["trajectory_id"], original["trajectory_id"])
                provenance = row.pop("selection_provenance")
                self.assertEqual(provenance["source_id"], original["trajectory_id"])
                self.assertEqual(
                    row.pop("messages"),
                    original.pop("messages")[: measurement["end_message_index"] + 1],
                )
                self.assertEqual(row, original)
            self.assertEqual(outputs[split]["prefix_rows"], 1)
            self.assertEqual([row["row_index"] for row in sidecar["splits"][split]], [2, 0])
        self.assertFalse(
            {r["source_id"] for r in sidecar["splits"]["training"]}
            & {r["source_id"] for r in sidecar["splits"]["validation"]}
        )

    def test_mixed_duplicate_parent_and_nonassistant_boundary_fail_without_publication(self):
        for mode in ("duplicate", "bad_boundary"):
            selections = {}
            for split, records in self.index["splits"].items():
                row = dict(
                    records[0],
                    source_id=records[0]["trajectory_id"],
                    is_prefix=True,
                    source_original_runtime_tokens=records[0]["original_runtime_tokens"],
                    end_message_index=2,
                )
                selections[split] = [row, deepcopy(row)] if mode == "duplicate" else [row]
            output_dir = self.directory / mode
            with self.assertRaises(ValueError):
                filtering.write_selection(
                    self.data_dir,
                    output_dir,
                    self.index,
                    "mixed",
                    1234,
                    65536,
                    mixed_selections=selections,
                )
            self.assert_no_published_outputs(output_dir)

    def test_same_row_count_source_change_is_rejected_before_publication(self):
        path = self.data_dir / "validation.jsonl"
        path.write_bytes(path.read_bytes().replace(b"All done.", b"Modified."))
        output_dir = self.directory / "changed"
        with self.assertRaises((ValueError, RuntimeError)):
            filtering.write_selection(self.data_dir, output_dir, self.index, "natural", 1234, 65536)
        self.assert_no_published_outputs(output_dir)

    def measure_with_error(self, error, stage):
        rows = jsonl_rows.JsonlRows(self.data_dir / "training.jsonl")
        encoder = Mock()
        encoder.encode.return_value = ([1, 2], [-100, 2])
        truncate = Mock(side_effect=error)
        if stage == "tokenize":
            encoder.encode.side_effect = error
        try:
            with (
                patch.object(filtering, "_ROWS", rows),
                patch.object(filtering, "_ENCODER", encoder),
                patch.object(filtering, "truncate_chat", truncate),
                patch.object(filtering, "_SEQUENCE_LENGTH", SEQUENCE_LENGTH),
            ):
                return filtering._measure_row(1)
        finally:
            rows.close()

    def test_known_untrainable_rows_are_reported_with_physical_source_line(self):
        cases = (
            (
                "No next-token assistant targets in this conversation",
                "tokenize",
                "no_assistant_targets",
            ),
            (
                "No assistant targets remain after truncation to 65536; "
                "increase --seq-length or filter the source trajectory",
                "truncate",
                "no_assistant_targets_after_truncation",
            ),
        )
        for message, stage, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(
                    self.measure_with_error(ValueError(message), stage),
                    {
                        "trajectory_id": "training-1",
                        "row_index": 1,
                        "source_line": 4,
                        "rejected_reason": reason,
                    },
                )

    def test_unknown_errors_are_fatal_even_if_the_message_resembles_a_known_rejection(self):
        known = "No next-token assistant targets in this conversation"
        errors = (
            ValueError("Unexpected tokenizer failure"),
            RuntimeError(known),
            ValueError(known + "; unexpected additional failure"),
            ValueError("No assistant targets remain after truncation to broken-policy"),
            ValueError(
                "No assistant targets remain after truncation to 32768; "
                "increase --seq-length or filter the source trajectory"
            ),
        )
        for error in errors:
            with self.subTest(error=repr(error)):
                with self.assertRaisesRegex(ValueError, "Length measurement failed") as caught:
                    self.measure_with_error(error, "truncate")
                self.assertIs(caught.exception.__cause__, error)
                self.assertEqual(
                    str(caught.exception),
                    f"Length measurement failed at {self.data_dir / 'training.jsonl'}:4",
                )

    def test_rejected_middle_rows_do_not_shift_selected_source_positions(self):
        for split, records in self.index["splits"].items():
            records[1] = {
                "trajectory_id": f"{split}-1",
                "row_index": 1,
                "source_line": 4,
                "rejected_reason": "no_assistant_targets",
            }
        for distribution in ("natural",):
            with self.subTest(distribution=distribution):
                output_dir = self.directory / f"rejected_{distribution}"
                outputs = filtering.write_selection(self.data_dir, output_dir, self.index, distribution, 1234, 65536)
                for split, records in self.index["splits"].items():
                    eligible = [r for r in records if "rejected_reason" not in r]
                    selected = filtering.select_indices(
                        [r["trajectory_id"] for r in eligible],
                        [r["runtime_tokens"] for r in eligible],
                        distribution,
                        1234,
                        65536,
                    )
                    self.assertEqual(
                        (output_dir / f"{split}.jsonl").read_bytes(),
                        b"".join(self.lines[split][eligible[i]["row_index"]] for i in selected),
                    )
                    output = outputs[split]
                    self.assertEqual(output["candidate_rows"], len(records))
                    self.assertEqual(output["candidate_lengths"]["samples"], len(eligible))
                    self.assertEqual(output["rows"], len(selected))
                    self.assertEqual(output["selected_lengths"]["samples"], len(selected))
                    self.assertEqual(
                        output["rejected_rows"],
                        [
                            {
                                "trajectory_id": f"{split}-1",
                                "source_line": 4,
                                "reason": "no_assistant_targets",
                            }
                        ],
                    )
                    self.assertEqual(output["rejected_rows_by_reason"], {"no_assistant_targets": 1})
                    self.assertEqual(
                        output["candidate_rows"],
                        output["candidate_lengths"]["samples"] + len(output["rejected_rows"]),
                    )

    def test_fully_rejected_validation_fails_without_publishing_training(self):
        self.index["splits"]["validation"] = [
            {
                "trajectory_id": record["trajectory_id"],
                "row_index": i,
                "source_line": 2 + 2 * i,
                "rejected_reason": "no_assistant_targets_after_truncation",
            }
            for i, record in enumerate(self.index["splits"]["validation"])
        ]
        output_dir = self.directory / "all_rejected"
        with self.assertRaisesRegex(ValueError, "validation: no trainable trajectories"):
            filtering.write_selection(self.data_dir, output_dir, self.index, "natural", 1234, 65536)
        self.assert_no_published_outputs(output_dir)

    def test_cached_lengths_reuse_and_invalidate_on_each_semantic_input(self):
        tokenizer = self.directory / "tokenizer"
        tokenizer.mkdir()
        tokenizer_file = tokenizer / "tokenizer.json"
        tokenizer_file.write_text("original tokenizer")
        named_template = tokenizer / "chat_templates" / "tool_use.jinja"
        named_template.parent.mkdir()
        named_template.write_text("original named template")
        fake_root = self.directory / "source"
        source_paths = ("02_5_filter_coderforge.py", "agentic_sft_data.py")
        for relative_path in source_paths:
            path = fake_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original source")
        measurements = {r["trajectory_id"]: r for records in self.index["splits"].values() for r in records}
        measurements["training-1"] = {
            "trajectory_id": "training-1",
            "row_index": 1,
            "source_line": 4,
            "rejected_reason": "no_assistant_targets",
        }
        worker = {}

        def initialize(rows, tokenizer_path, sequence_length, loss_mode):
            worker["rows"] = rows

        def measure(row_index):
            return deepcopy(measurements[worker["rows"][row_index]["trajectory_id"]])

        with (
            patch.object(filtering, "EXAMPLE_DIR", fake_root),
            patch.object(filtering, "_initialize_worker", side_effect=initialize),
            patch.object(filtering, "_measure_row", side_effect=measure) as measurement,
        ):
            expected_calls = len(measurements)
            first, first_path = filtering.load_length_index(self.data_dir, tokenizer, 65536, 1)
            self.assertEqual(measurement.call_count, expected_calls)
            self.assertEqual(first["splits"]["training"][1], measurements["training-1"])
            self.assertEqual(
                [r["row_index"] for r in first["splits"]["training"]],
                list(range(len(self.lines["training"]))),
            )
            cached, cached_path = filtering.load_length_index(self.data_dir, tokenizer, 65536, 1)
            self.assertEqual(cached, first)
            self.assertEqual(cached_path, first_path)
            self.assertEqual(measurement.call_count, expected_calls)
            changes = (
                lambda: tokenizer_file.write_text("changed tokenizer"),
                lambda: named_template.write_text("changed named template"),
                lambda: (fake_root / source_paths[1]).write_text("changed chat encoding and truncation policy"),
                lambda: (self.data_dir / "training.jsonl").write_bytes(
                    b"\n" + (self.data_dir / "training.jsonl").read_bytes()
                ),
            )
            for change in changes:
                change()
                filtering.load_length_index(self.data_dir, tokenizer, 65536, 1)
                expected_calls += len(measurements)
                self.assertEqual(measurement.call_count, expected_calls)
            _, changed_length_path = filtering.load_length_index(self.data_dir, tokenizer, 131072, 1)
            self.assertNotEqual(first_path, changed_length_path)
            self.assertEqual(measurement.call_count, expected_calls + len(measurements))
            expected_calls += len(measurements)
            full, full_path = filtering.load_length_index(self.data_dir, tokenizer, 65536, 1, loss_mode="full")
            self.assertNotEqual(first_path, full_path)
            self.assertEqual(full["fingerprints"]["loss_mode"], "full")
            self.assertEqual(measurement.call_count, expected_calls + len(measurements))


if __name__ == "__main__":
    unittest.main()
