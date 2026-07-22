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

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).parents[3] / "examples" / "evaluation" / "run_lm_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_lm_eval", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run_lm_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_lm_eval)


@pytest.mark.unit
def test_build_lm_eval_command_preserves_legacy_mmlu_configuration(tmp_path: Path) -> None:
    command = run_lm_eval.build_lm_eval_command(parallelism=4, output_dir=tmp_path)

    assert command[command.index("--tasks") + 1] == "mmlu_str"
    assert command[command.index("--num_fewshot") + 1] == "5"
    assert command[command.index("--limit") + 1] == "100"
    assert command[command.index("--model") + 1] == "local-completions"
    model_args = command[command.index("--model_args") + 1]
    assert "base_url=http://0.0.0.0:8000/v1/completions/" in model_args
    assert "model=megatron_model" in model_args
    assert "num_concurrent=4" in model_args
    assert "timeout=1000" in model_args
    assert "--gen_kwargs=temperature=1e-07,top_p=0.9999999" in command


@pytest.mark.unit
def test_copy_evaluation_results_excludes_lm_cache(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    model_dir = source_dir / "megatron_model"
    model_dir.mkdir(parents=True)
    (model_dir / "results_2026-01-01.json").write_text("{}", encoding="utf-8")
    (source_dir / "lm_cache_rank0.db").write_text("cache", encoding="utf-8")

    run_lm_eval.copy_evaluation_results(source_dir, destination_dir)

    assert (destination_dir / "megatron_model" / "results_2026-01-01.json").is_file()
    assert not (destination_dir / "lm_cache_rank0.db").exists()


@pytest.mark.unit
def test_wait_for_endpoint_retries_until_server_is_ready() -> None:
    assert run_lm_eval._READINESS_REQUEST_TIMEOUT == run_lm_eval._REQUEST_TIMEOUT

    class DelayedReadinessHandler(BaseHTTPRequestHandler):
        attempts = 0

        def do_POST(self) -> None:  # noqa: N802
            type(self).attempts += 1
            content_length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(content_length))
            assert payload == {"model": "megatron_model", "max_tokens": 1, "prompt": "hello, my name is"}
            status = 200 if type(self).attempts >= 3 else 503
            self.send_response(status)
            self.end_headers()

        def log_message(self, _format: str, *args: object) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), DelayedReadinessHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        run_lm_eval.wait_for_endpoint(
            f"http://127.0.0.1:{server.server_port}/v1/completions/",
            max_attempts=5,
            retry_interval=0.01,
            request_timeout=1,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert DelayedReadinessHandler.attempts == 3


@pytest.mark.unit
def test_load_evaluation_results_is_scoped_to_run_id(tmp_path: Path) -> None:
    current_results_dir = tmp_path / "results" / "current-run" / "megatron_model"
    stale_results_dir = tmp_path / "results" / "stale-run" / "megatron_model"
    current_results_dir.mkdir(parents=True)
    stale_results_dir.mkdir(parents=True)
    current = current_results_dir / "results_current.json"
    stale = stale_results_dir / "results_stale.json"
    current.write_text(json.dumps({"marker": "current"}), encoding="utf-8")
    stale.write_text(json.dumps({"marker": "stale"}), encoding="utf-8")

    results_path, results = run_lm_eval.load_evaluation_results(tmp_path, "current-run")

    assert results_path == current
    assert results == {"marker": "current"}


@pytest.mark.unit
def test_iter_evaluation_metrics_reads_task_and_group_scores() -> None:
    results = {
        "results": {
            "mmlu_str": {"exact_match,none": 0.8, "exact_match_stderr,none": 0.02},
            "mmlu_str_abstract_algebra": {
                "alias": "abstract algebra",
                "exact_match,strict-match": 0.75,
                "exact_match_stderr,strict-match": 0.1,
            },
        },
        "groups": {"mmlu_str": {"exact_match,none": 0.8, "exact_match_stderr,none": "N/A"}},
    }

    assert list(run_lm_eval.iter_evaluation_metrics(results)) == [
        ("results", "mmlu_str_abstract_algebra", "exact_match__strict-match", 0.75, 0.1),
        ("groups", "mmlu_str", "exact_match", 0.8, None),
    ]
