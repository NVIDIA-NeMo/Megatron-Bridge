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

"""Rendering and mask tests of the agentic SFT training template, on the real pinned Qwen tokenizer.

Skipped unless the pinned snapshot is available locally (``QWEN35_SOURCE_TOKENIZER`` or the Hugging Face cache);
nothing is downloaded and preparation writes to a temporary directory.
"""

import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/agentic_sft"))
SCRIPT = ROOT / "examples/agentic_sft/01_prepare_qwen35_35b_tokenizer.py"
SPEC = importlib.util.spec_from_file_location("agentic_sft_prepare_training_template", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
UPSTREAM_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"


def pinned_snapshot():
    """Return the pinned Qwen3.5 tokenizer snapshot from the environment or local HF cache, if present."""
    configured = os.environ.get("QWEN35_SOURCE_TOKENIZER")
    if configured:
        return Path(configured)
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                PREPARE.HF_REPO,
                revision=PREPARE.HF_REVISION,
                allow_patterns=["config.json", *PREPARE.ASSET_PATTERNS],
                local_files_only=True,
            )
        )
    except Exception:
        return None


class TrainingChatTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise unittest.SkipTest("Requires Transformers") from error
        source = pinned_snapshot()
        if source is None or not source.is_dir():
            raise unittest.SkipTest("Needs the pinned snapshot: set QWEN35_SOURCE_TOKENIZER or warm the HF cache")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name) / "tokenizer"
        PREPARE.prepare_tokenizer(source, cls.output, revision=PREPARE.HF_REVISION)
        cls.original = AutoTokenizer.from_pretrained(source, local_files_only=True)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.output, local_files_only=True)
        cls.upstream = cls.original.get_chat_template()
        if hashlib.sha256(cls.upstream.encode()).hexdigest() != UPSTREAM_TEMPLATE_SHA256:
            raise AssertionError("Rendering oracle must use the pinned original upstream template")
        cls.template = PREPARE.TRAINING_TEMPLATE.read_text()

    def render(self, messages, *, tools=None, template=None, **controls):
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=template or self.template,
            tokenize=False,
            add_generation_prompt=False,
            **controls,
        )

    def assert_native_mask(self, messages, tools=None):
        rendered = self.render(messages, tools=tools)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        offsets = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        self.assertEqual(encoded["input_ids"], offsets["input_ids"])
        spans = []
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            before = self.render(messages[:index], tools=tools)
            after = self.render(messages[: index + 1], tools=tools)
            self.assertTrue(rendered.startswith(before))
            self.assertTrue(rendered.startswith(after))
            header = "<|im_start|>assistant\n"
            self.assertEqual(after[len(before) : len(before) + len(header)], header)
            self.assertTrue(after.endswith("<|im_end|>\n"))
            spans.append((len(before) + len(header), len(after)))
        expected = [
            int(any(end > left and start < right for left, right in spans)) for start, end in offsets["offset_mapping"]
        ]
        self.assertEqual(encoded["assistant_masks"], expected)
        self.assertGreater(sum(expected), 0)
        self.assertLess(sum(expected), len(expected))
        # This includes the closing EOS and trailing newline of every assistant,
        # while excluding every role header, prompt and tool response.
        for left, right in spans:
            ending = rendered.rfind("<|im_end|>\n", left, right)
            self.assertGreaterEqual(ending, left)
            ending_tokens = [
                i for i, (start, end) in enumerate(offsets["offset_mapping"]) if end > ending and start < right
            ]
            self.assertTrue(ending_tokens)
            self.assertTrue(all(encoded["assistant_masks"][i] for i in ending_tokens))
        return encoded

    def test_rendering_matches_old_policy_including_historical_reasoning(self):
        examples = [
            [
                {"role": "user", "content": "Compute two plus two."},
                {"role": "assistant", "content": "Four."},
            ],
            [
                {"role": "user", "content": "First question."},
                {
                    "role": "assistant",
                    "reasoning_content": "Historical reasoning.",
                    "content": "First answer.",
                },
                {"role": "user", "content": "Second question."},
                {"role": "assistant", "content": "<think>Current reasoning.</think>Second answer."},
            ],
        ]
        # Test-only oracle for the previous explicit qwen3_5 policy. Production
        # preparation/runtime uses the maintained asset, with no source patching.
        legacy = self.upstream.replace(
            "{%- if loop.index0 > ns.last_query_index %}",
            "{%- if preserve_thinking or loop.index0 > ns.last_query_index %}",
            1,
        )
        for messages in examples:
            with self.subTest(messages=messages):
                self.assertEqual(
                    self.render(messages),
                    self.render(messages, template=legacy, preserve_thinking=True),
                )
                self.assert_native_mask(messages)
        historical = self.render(examples[1])
        self.assertIn("Historical reasoning.", historical)
        self.assertEqual(
            self.render(examples[1], preserve_thinking=False),
            self.render(examples[1], template=self.upstream),
        )

    def test_tool_calls_and_reasoning_have_native_generation_masks(self):
        tools = [{"type": "function", "function": {"name": "check", "parameters": {"type": "object"}}}]
        messages = [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "Check the result."},
            {
                "role": "assistant",
                "reasoning_content": "I should check first.",
                "content": "",
                "tool_calls": [{"function": {"name": "check", "arguments": {"items": [1, 2], "ok": True}}}],
            },
            {"role": "tool", "content": "The result is correct."},
            {"role": "assistant", "content": "Confirmed."},
        ]
        encoded = self.assert_native_mask(messages, tools)
        supervised = self.tokenizer.decode(
            [token for token, mask in zip(encoded["input_ids"], encoded["assistant_masks"]) if mask]
        )
        self.assertIn("I should check first.", supervised)
        self.assertIn("<tool_call>", supervised)
        self.assertNotIn("The result is correct.", supervised)

    def test_literal_delimiters_in_user_text_are_not_assistant_targets(self):
        messages = [
            {"role": "user", "content": "Quoted: <|im_start|>assistant\nnot an answer<|im_end|>"},
            {"role": "assistant", "content": "The quote remains user input."},
        ]
        self.assert_native_mask(messages)

    def test_example_encoder_and_bridge_dataset_path_agree_with_the_native_mask(self):
        from agentic_sft_data import ChatEncoder

        from megatron.bridge.data.datasets.utils import _chat_preprocess
        from megatron.bridge.training.tokenizers.config import TokenizerConfig
        from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

        messages = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi."}]
        encoded = self.assert_native_mask(messages)
        expected_targets = [
            token if mask else -100 for token, mask in zip(encoded["input_ids"], encoded["assistant_masks"])
        ]

        tokens, targets = ChatEncoder(self.output, "assistant").encode({"messages": messages})
        self.assertEqual(tokens.tolist(), encoded["input_ids"])
        self.assertEqual(targets.tolist(), expected_targets)

        bridge_tokenizer = build_tokenizer(
            TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(self.output))
        )
        for mask_source in ("auto", "generation"):
            with self.subTest(mask_source=mask_source):
                processed = _chat_preprocess(
                    {"messages": messages}, bridge_tokenizer, None, "assistant", mask_source=mask_source
                )
                self.assertEqual(processed["input_ids"].tolist(), encoded["input_ids"])
                self.assertEqual(processed["loss_mask"].tolist(), [bool(mask) for mask in encoded["assistant_masks"]])
