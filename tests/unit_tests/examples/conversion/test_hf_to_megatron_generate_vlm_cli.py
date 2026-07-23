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

"""Focused tests for the VLM generation CLI."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path


def _load_cli_symbols():
    script = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "hf_to_megatron_generate_vlm.py"
    tree = ast.parse(script.read_text(), filename=str(script))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_completion_output",
            "_hf_revision_kwargs",
            "_should_stop_generation",
            "build_parser",
        }
    ]
    module = ast.fix_missing_locations(
        ast.Module(
            body=[ast.Import(names=[ast.alias(name="argparse")]), *selected],
            type_ignores=[],
        )
    )
    namespace = {"__name__": "test_hf_to_megatron_generate_vlm_cli"}
    exec(compile(module, str(script), "exec"), namespace)
    return namespace


def _load_kimi_patch():
    script = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "vlm_generate_utils.py"
    tree = ast.parse(script.read_text(), filename=str(script))
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "patch_kimi_vision_processor"
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"__name__": "test_vlm_generate_utils"}
    exec(compile(module, str(script), "exec"), namespace)
    return namespace["patch_kimi_vision_processor"]


def test_hf_revision_is_optional():
    symbols = _load_cli_symbols()

    args = symbols["build_parser"]().parse_args(["--hf_model_path", "Qwen/model"])

    assert args.hf_revision is None
    assert args.exact_new_tokens is False
    assert symbols["_hf_revision_kwargs"](args.hf_revision) == {}


def test_hf_revision_is_forwarded_as_a_transformers_kwarg():
    symbols = _load_cli_symbols()
    revision = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret

    args = symbols["build_parser"]().parse_args(["--hf_model_path", "Qwen/model", "--hf-revision", revision])

    assert args.hf_revision == revision
    assert symbols["_hf_revision_kwargs"](args.hf_revision) == {"revision": revision}


def test_exact_new_tokens_ignores_early_eos():
    symbols = _load_cli_symbols()
    should_stop = symbols["_should_stop_generation"]

    assert should_stop(2, [2], exact_new_tokens=False) is True
    assert should_stop(2, [2], exact_new_tokens=True) is False
    assert should_stop(3, [2], exact_new_tokens=True) is False


def test_completion_output_slices_prompt_and_emits_every_token_id():
    symbols = _load_cli_symbols()

    class _Completion:
        def tolist(self):
            return [17, 23, 42]

    class _Generated:
        def __getitem__(self, key):
            assert key == (0, slice(5, None))
            return _Completion()

    class _Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [17, 23, 42]
            assert skip_special_tokens is True
            return "completion only"

    assert symbols["_completion_output"](_Generated(), 5, _Tokenizer()) == (
        [17, 23, 42],
        "completion only",
    )


def test_kimi_patch_forwards_revision_and_respects_trust_policy(monkeypatch):
    patch_kimi = _load_kimi_patch()
    calls = []

    class _KimiProcessor:
        pass

    def get_class_from_dynamic_module(class_name, model_path, **kwargs):
        calls.append((class_name, model_path, kwargs))
        return _KimiProcessor

    transformers = types.ModuleType("transformers")
    transformers.__path__ = []
    dynamic_module_utils = types.ModuleType("transformers.dynamic_module_utils")
    dynamic_module_utils.get_class_from_dynamic_module = get_class_from_dynamic_module
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", dynamic_module_utils)

    patch_kimi("org/model", revision="immutable", trust_remote_code=False)
    assert calls == []

    patch_kimi("org/model", revision="immutable", trust_remote_code=True)
    assert calls == [
        (
            "kimi_k25_vision_processing.KimiK25VisionProcessor",
            "org/model",
            {"revision": "immutable"},
        )
    ]
