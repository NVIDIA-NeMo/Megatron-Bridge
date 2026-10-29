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

"""Dependency-free checks for the Docker-only Mamba source installer."""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "docker/common/install_mamba.sh"
PATCH = ROOT / "docker/patches/mamba.patch"
SETUP_SOURCE = """        }
    else:
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
"""


class TestInstallMamba(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.archive = self.directory / "mamba.tar.gz"
        self.lock = self.directory / "uv.lock"
        self.record = self.directory / "install.json"
        self._write_archive(SETUP_SOURCE)
        self._write_lock()

        # Execute the installer's real lock parser, but never run uv or compile CUDA.
        uv = self.directory / "uv"
        uv.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "if args[:4] == ['run', '--no-project', '--no-sync', 'python']:\n"
            "    os.execv(sys.executable, [sys.executable, *args[4:]])\n"
            "if args[:2] != ['pip', 'install']:\n"
            "    raise SystemExit(f'Unexpected uv invocation: {args}')\n"
            "record = {'args': args, 'force_build': os.environ.get('MAMBA_FORCE_BUILD'),\n"
            "          'nvcc_append_flags': os.environ.get('NVCC_APPEND_FLAGS'),\n"
            "          'source': (Path(args[-1]) / 'setup.py').read_text()}\n"
            "Path(os.environ['MAMBA_TEST_RECORD']).write_text(json.dumps(record))\n"
        )
        uv.chmod(0o755)

    def _write_archive(self, source):
        data = source.encode()
        with tarfile.open(self.archive, "w:gz") as archive:
            info = tarfile.TarInfo("mamba_ssm-2.3.1/setup.py")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    def _write_lock(self, *, version="2.3.1", source=None, digest=None):
        if source is None:
            source = '{ registry = "https://pypi.org/simple" }'
        if digest is None:
            digest = "sha256:" + hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.lock.write_text(
            '[[package]]\nname = "mamba-ssm"\n'
            f'version = "{version}"\nsource = {source}\n'
            f'sdist = {{ url = "{self.archive.as_uri()}", hash = "{digest}" }}\n'
        )

    def _run(self, **extra_env):
        env = {
            key: value for key, value in os.environ.items() if key not in {"MAMBA_NVCC_THREADS", "NVCC_APPEND_FLAGS"}
        }
        return subprocess.run(
            ["bash", str(INSTALLER), str(self.lock), str(PATCH)],
            env={
                **env,
                "PATH": f"{self.directory}{os.pathsep}{os.environ['PATH']}",
                "MAMBA_TEST_RECORD": str(self.record),
                "MAMBA_FORCE_BUILD": "FALSE",
                **extra_env,
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def test_installs_only_patched_source_without_dependencies(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(self.record.read_text())
        self.assertNotIn("-std=c++17", record["source"])
        self.assertIn('"cxx": ["-O3"]', record["source"])
        self.assertEqual(record["force_build"], "TRUE")
        self.assertIsNone(record["nvcc_append_flags"], "Default build must keep Mamba's own nvcc threads")
        self.assertEqual(
            record["args"][:-1],
            ["pip", "install", "--no-build-isolation", "--no-deps", "--reinstall"],
        )
        self.assertFalse(Path(record["args"][-1]).exists(), "Temporary source was not cleaned up")

    def test_optional_nvcc_threads_override_mambas_hardcoded_value(self):
        result = self._run(MAMBA_NVCC_THREADS="9", NVCC_APPEND_FLAGS="-lineinfo")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(self.record.read_text())
        self.assertEqual(record["nvcc_append_flags"], "-lineinfo --threads 9")
        self.assertEqual(record["force_build"], "TRUE")

    def test_rejects_invalid_nvcc_threads(self):
        result = self._run(MAMBA_NVCC_THREADS="nine")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MAMBA_NVCC_THREADS must be a positive integer", result.stderr)
        self.assertFalse(self.record.exists())

    def test_rejects_checksum_mismatch(self):
        self._write_lock(digest="sha256:" + "0" * 64)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.record.exists())

    def test_rejects_new_version(self):
        self._write_lock(version="2.4.0")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Mamba source/version changed", result.stderr)
        self.assertFalse(self.record.exists())

    def test_pyproject_pins_the_version_the_installer_accepts(self):
        installer = INSTALLER.read_text()
        accepted = re.search(r'package\["version"\] != "([^"]+)"', installer).group(1)
        self.assertEqual(set(re.findall(r"mamba_ssm-([\w.]+)", installer)), {accepted})
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        requirements = [*project["dependencies"], *sum(project["optional-dependencies"].values(), [])]
        mamba = [requirement for requirement in requirements if re.match(r"mamba[-_]ssm\b", requirement)]
        self.assertEqual(mamba, [f"mamba-ssm=={accepted}"])

    def test_rejects_new_source(self):
        self._write_lock(source='{ git = "https://example.invalid/mamba.git" }')
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Mamba source/version changed", result.stderr)
        self.assertFalse(self.record.exists())

    def test_rejects_missing_package(self):
        self.lock.write_text('[[package]]\nname = "another-package"\n')
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected exactly one locked mamba-ssm", result.stderr)
        self.assertFalse(self.record.exists())

    def test_rejects_unsupported_hash(self):
        self._write_lock(digest="sha512:" + "0" * 128)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected a SHA-256 digest", result.stderr)
        self.assertFalse(self.record.exists())

    def test_rejects_incompatible_patch(self):
        self._write_archive("# Incompatible upstream setup.py\n")
        self._write_lock()
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed", (result.stdout + result.stderr).lower())
        self.assertFalse(self.record.exists())

    def test_docker_syncs_skip_mamba_before_explicit_install(self):
        dockerfile = (ROOT / "docker/Dockerfile.ci").read_text()
        syncs = [line for line in dockerfile.splitlines() if "uv sync " in line and not line.lstrip().startswith("#")]
        self.assertEqual(len(syncs), 6)
        for line in syncs:
            self.assertIn("--no-install-package mamba-ssm", line)
        self.assertIn(
            "fi && \\\n    bash /opt/install_mamba.sh /opt/Megatron-Bridge/uv.lock /opt/mamba.patch &&",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
