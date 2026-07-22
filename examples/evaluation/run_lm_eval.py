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

"""Run the legacy five-shot MMLU evaluation against a deployed endpoint."""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator, Sequence
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


logger = logging.getLogger(__name__)

_ENDPOINT_URL = "http://0.0.0.0:8000/v1/completions/"
_MODEL_ID = "megatron_model"
_MMLU_TASK = "mmlu_str"
_NUM_FEWSHOT = 5
_LIMIT_SAMPLES = 100.0
_REQUEST_TIMEOUT = 1000
_TEMPERATURE = 1e-7
_TOP_P = 0.9999999
_MAX_RETRIES = 5
_READINESS_MAX_ATTEMPTS = 600
_READINESS_RETRY_INTERVAL = 2.0
_READINESS_REQUEST_TIMEOUT = float(_REQUEST_TIMEOUT)
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

EvaluationMetric = tuple[str, str, str, float, float | None]


def _parse_run_id(value: str) -> str:
    if value in {".", ".."} or _RUN_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "run-id must contain only letters, numbers, periods, underscores, and hyphens"
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory in which to store evaluation results")
    parser.add_argument("parallelism", type=int, help="Number of concurrent requests sent to the endpoint")
    parser.add_argument(
        "--run-id",
        type=_parse_run_id,
        default=uuid.uuid4().hex,
        help="Unique identifier used to isolate this run's results",
    )
    parser.add_argument("--endpoint-url", default=_ENDPOINT_URL, help="Full OpenAI-compatible completions URL")
    parser.add_argument("--eval-task", default=_MMLU_TASK, help="lm-eval task or task group")
    parser.add_argument(
        "--limit-samples",
        type=float,
        default=_LIMIT_SAMPLES,
        help="Maximum samples per task; values below one select a fraction",
    )
    args = parser.parse_args(argv)
    if args.parallelism < 1:
        parser.error("parallelism must be at least 1")
    if args.limit_samples <= 0:
        parser.error("limit-samples must be positive")
    return args


def wait_for_endpoint(
    endpoint_url: str,
    *,
    model_id: str = _MODEL_ID,
    max_attempts: int = _READINESS_MAX_ATTEMPTS,
    retry_interval: float = _READINESS_RETRY_INTERVAL,
    request_timeout: float = _READINESS_REQUEST_TIMEOUT,
) -> None:
    """Wait until an OpenAI-compatible completions endpoint accepts a request."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if retry_interval < 0:
        raise ValueError("retry_interval must be non-negative")
    if request_timeout <= 0:
        raise ValueError("request_timeout must be positive")

    parsed_url = urlsplit(endpoint_url)
    if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname is None:
        raise ValueError(f"Unsupported endpoint URL: {endpoint_url}")
    connection_type = http.client.HTTPSConnection if parsed_url.scheme == "https" else http.client.HTTPConnection
    request_target = urlunsplit(("", "", parsed_url.path or "/", parsed_url.query, ""))
    request_body = json.dumps({"model": model_id, "max_tokens": 1, "prompt": "hello, my name is"})
    request_headers = {"Accept": "application/json", "Content-Type": "application/json"}

    for attempt in range(1, max_attempts + 1):
        connection = connection_type(parsed_url.hostname, parsed_url.port, timeout=request_timeout)
        try:
            connection.request("POST", request_target, body=request_body, headers=request_headers)
            response = connection.getresponse()
            response.read()
            if response.status == HTTPStatus.OK:
                logger.info("Endpoint is ready after %d attempt(s)", attempt)
                return
            logger.info("Endpoint readiness attempt %d/%d returned HTTP %d", attempt, max_attempts, response.status)
        except (OSError, http.client.HTTPException) as error:
            logger.info("Endpoint readiness attempt %d/%d failed: %s", attempt, max_attempts, error)
        finally:
            connection.close()

        if attempt < max_attempts:
            time.sleep(retry_interval)

    raise RuntimeError(f"Endpoint did not become ready after {max_attempts} attempts: {endpoint_url}")


def build_lm_eval_command(
    *,
    parallelism: int,
    output_dir: Path,
    endpoint_url: str = _ENDPOINT_URL,
    eval_task: str = _MMLU_TASK,
    limit_samples: float = _LIMIT_SAMPLES,
) -> list[str]:
    """Build the command matching the legacy NeMo Evaluator MMLU configuration."""
    model_args = ",".join(
        [
            f"base_url={endpoint_url}",
            f"model={_MODEL_ID}",
            "tokenized_requests=False",
            "tokenizer_backend=None",
            f"num_concurrent={parallelism}",
            f"timeout={_REQUEST_TIMEOUT}",
            f"max_retries={_MAX_RETRIES}",
            "stream=False",
        ]
    )
    return [
        "lm-eval",
        "--tasks",
        eval_task,
        "--num_fewshot",
        str(_NUM_FEWSHOT),
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--log_samples",
        "--output_path",
        str(output_dir),
        "--use_cache",
        str(output_dir / "lm_cache"),
        "--limit",
        str(int(limit_samples) if limit_samples.is_integer() else limit_samples),
        "--trust_remote_code",
        f"--gen_kwargs=temperature={_TEMPERATURE},top_p={_TOP_P}",
    ]


def copy_evaluation_results(source_dir: Path, destination_dir: Path) -> None:
    """Copy evaluation artifacts while leaving the local SQLite cache behind."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        if source.name.startswith("lm_cache"):
            continue
        destination = destination_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)


def find_latest_results_file(output_dir: Path, run_id: str) -> Path:
    """Find the lm-eval aggregate results file for one isolated run."""
    run_results_dir = output_dir / "results" / _parse_run_id(run_id)
    candidates = list(run_results_dir.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No lm-eval results file found under {run_results_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def load_evaluation_results(output_dir: Path, run_id: str) -> tuple[Path, dict[str, object]]:
    """Load the aggregate results for one isolated evaluation run."""
    results_path = find_latest_results_file(output_dir, run_id)
    with results_path.open(encoding="utf-8") as results_file:
        results = json.load(results_file)
    if not isinstance(results, dict):
        raise ValueError(f"Expected an object in {results_path}")
    return results_path, results


def iter_evaluation_metrics(results: dict[str, object]) -> Iterator[EvaluationMetric]:
    """Yield numeric task and group metrics from an lm-eval result object."""
    groups = results.get("groups")
    group_names = set(groups) if isinstance(groups, dict) else set()
    for category in ("results", "groups"):
        entries = results.get(category)
        if not isinstance(entries, dict):
            continue
        for entry_name, entry in entries.items():
            if not isinstance(entry_name, str) or not isinstance(entry, dict):
                continue
            if category == "results" and entry_name in group_names:
                continue
            for metric_key, value in entry.items():
                if not isinstance(metric_key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                metric_name, separator, filter_name = metric_key.partition(",")
                if metric_name.endswith("_stderr"):
                    continue
                stderr_key = f"{metric_name}_stderr,{filter_name}" if separator else f"{metric_name}_stderr"
                stderr_value = entry.get(stderr_key)
                stderr = (
                    float(stderr_value)
                    if isinstance(stderr_value, (int, float)) and not isinstance(stderr_value, bool)
                    else None
                )
                reported_metric_name = (
                    f"{metric_name}__{filter_name}" if separator and filter_name != "none" else metric_name
                )
                yield category, entry_name, reported_metric_name, float(value), stderr


def main(argv: Sequence[str] | None = None) -> None:
    """Run five-shot MMLU and persist all non-cache artifacts."""
    args = parse_args(argv)
    results_dir = args.output_dir / "results" / args.run_id
    if results_dir.exists():
        raise FileExistsError(f"Refusing to mix evaluation artifacts into existing run directory: {results_dir}")
    wait_for_endpoint(args.endpoint_url)
    with tempfile.TemporaryDirectory(prefix="megatron-bridge-eval-") as temporary_dir:
        local_output_dir = Path(temporary_dir)
        command = build_lm_eval_command(
            parallelism=args.parallelism,
            output_dir=local_output_dir,
            endpoint_url=args.endpoint_url,
            eval_task=args.eval_task,
            limit_samples=args.limit_samples,
        )
        logger.info("Running five-shot lm-eval task %s against %s", args.eval_task, args.endpoint_url)
        subprocess.run(command, check=True)
        copy_evaluation_results(local_output_dir, results_dir)
    logger.info("Results for run %s copied to %s", args.run_id, results_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
