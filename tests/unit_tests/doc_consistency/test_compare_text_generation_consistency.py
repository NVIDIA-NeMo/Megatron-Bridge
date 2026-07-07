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
"""Consistency checks for examples/conversion/compare_text_generation.py.

trust_remote_code is a single --trust-remote-code CLI flag (default off) threaded
through every HF load in the pipeline. The checkpoint tokenizer load must forward
that flag rather than issue a bare load_tokenizer(), which would trip the checkpoint
trust gate whenever the checkpoint's tokenizer config requires remote code.

Deliberately stdlib-only (ast over the script source) so it runs without
torch/GPU.
"""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples" / "conversion" / "compare_text_generation.py"


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {SCRIPT}")


def test_checkpoint_tokenizer_load_opts_into_remote_code():
    """load_tokenizer() in megatron_generate_from_checkpoint forwards trust_remote_code."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    fn = _find_function(tree, "megatron_generate_from_checkpoint")
    calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", "") == "load_tokenizer" or getattr(node.func, "attr", "") == "load_tokenizer")
    ]
    assert calls, "megatron_generate_from_checkpoint no longer calls load_tokenizer"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert "trust_remote_code" in kwargs, (
            "load_tokenizer() called without trust_remote_code — the checkpoint is imported "
            "with trust_remote_code=True, so this call trips the tokenizer trust gate"
        )


if __name__ == "__main__":
    # Allow standalone RED-GREEN without pytest/torch:  python3 test_compare_text_generation_consistency.py
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
