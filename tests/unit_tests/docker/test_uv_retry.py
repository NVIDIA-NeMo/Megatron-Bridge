# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Docker download wrapper without network access or CUDA."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "docker/common/uv_retry.sh"


class TestUvRetry(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        uv = self.directory / "uv"
        uv.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "record = Path(os.environ['UV_TEST_RECORD'])\n"
            "calls = json.loads(record.read_text()) if record.exists() else []\n"
            "calls.append(sys.argv[1:])\n"
            "record.write_text(json.dumps(calls))\n"
            "if len(calls) <= int(os.environ['UV_TEST_FAILURES']):\n"
            "    print(os.environ['UV_TEST_ERROR'], file=sys.stderr, flush=True)\n"
            "    raise SystemExit(7)\n"
            "print('download complete', flush=True)\n"
        )
        uv.chmod(0o755)
        sleep = self.directory / "sleep"
        sleep.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$UV_TEST_SLEEPS"\n')
        sleep.chmod(0o755)

    def run_wrapper(self, error: str, failures: int) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
        record = self.directory / "calls.json"
        sleeps = self.directory / "sleeps"
        record.unlink(missing_ok=True)
        sleeps.unlink(missing_ok=True)
        args = ["sync", "--locked", "--no-install-package", "mamba-ssm", "argument with spaces"]
        result = subprocess.run(
            ["bash", str(WRAPPER), "uv", *args],
            env={
                **os.environ,
                "PATH": f"{self.directory}{os.pathsep}{os.environ['PATH']}",
                "UV_TEST_RECORD": str(record),
                "UV_TEST_SLEEPS": str(sleeps),
                "UV_TEST_ERROR": error,
                "UV_TEST_FAILURES": str(failures),
            },
            text=True,
            capture_output=True,
            timeout=10,
        )
        calls = json.loads(record.read_text())
        self.assertTrue(all(call == args for call in calls))
        waits = sleeps.read_text().splitlines() if sleeps.exists() else []
        self.assertEqual(waits, ["10"] * (len(calls) - 1))
        return result, calls

    def test_success_needs_no_retry(self) -> None:
        result, calls = self.run_wrapper("", 0)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("download complete", result.stdout)

    def test_transient_download_recovers(self) -> None:
        for status in [429, 500, 502, 503, 504]:
            with self.subTest(status=status):
                kind = "client" if status == 429 else "server"
                message = f"Failed to download `example==1`\nHTTP status {kind} error ({status} failure)"
                result, calls = self.run_wrapper(message, 1)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(len(calls), 2)
                self.assertIn(message, result.stdout)

    def test_exhaustion_preserves_failure(self) -> None:
        message = "Failed to download `example==1`\nHTTP status server error (504 Gateway Timeout)"
        result, calls = self.run_wrapper(message, 10)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result.stdout.count("504 Gateway Timeout"), 3)

    def test_last_attempt_can_succeed(self) -> None:
        message = "Failed to download example\nHTTP status server error (503 Service Unavailable)"
        result, calls = self.run_wrapper(message, 2)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(calls), 3)

    def test_deterministic_errors_are_not_retried(self) -> None:
        for message in [
            "No solution found when resolving dependencies",
            "The lockfile needs to be updated, but --locked was provided",
            "Failed to build example\ncompiler exited with status 1",
            "Failed to download example\nHash mismatch",
            "Failed to download example\nHTTP status client error (404 Not Found)",
            "HTTP status server error (504 Gateway Timeout)",
        ]:
            with self.subTest(message=message):
                result, calls = self.run_wrapper(message, 10)
                self.assertEqual(result.returncode, 7)
                self.assertEqual(len(calls), 1)
                self.assertIn(message, result.stdout)

    def test_docker_uses_wrapper_for_every_sync(self) -> None:
        dockerfile = (ROOT / "docker/Dockerfile.ci").read_text()
        self.assertIn("COPY docker/common/uv_retry.sh /opt/uv_retry.sh", dockerfile)
        self.assertNotIn("        uv sync ", dockerfile)
        self.assertEqual(dockerfile.count("bash /opt/uv_retry.sh uv sync "), 6)


if __name__ == "__main__":
    unittest.main()
