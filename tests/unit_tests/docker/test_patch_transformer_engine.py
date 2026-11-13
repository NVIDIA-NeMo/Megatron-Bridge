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

"""Dependency-free checks for the Docker-only TransformerEngine backport."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "docker/common/patch_transformer_engine.sh"
PATCH = ROOT / "docker/patches/transformer-engine-layernorm-linear-dummy-wgrad.patch"
# The TransformerEngine 2.21.0 code the backport targets.
SOURCE = """        # Handle custom DDP from mcore.
        if args.fuse_wgrad_accumulation and hasattr(origin_weight, "grad_added_to_main_grad"):
            origin_weight.grad_added_to_main_grad = True
            if getattr(origin_weight, "zero_out_wgrad", False):
                wgrad = get_dummy_wgrad(
                    list(main_grad.shape),
                    origin_weight.dtype,
                    zero=True,
                )
            else:
                wgrad = get_dummy_wgrad(
                    list(main_grad.shape),
                    origin_weight.dtype,
                    zero_tensor=getattr(origin_weight, "zero_tensor_dummy_wgrad", False),
                )
"""


class TestPatchTransformerEngine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        package = self.directory / "site-packages/transformer_engine"
        self.source = package / "pytorch/module/layernorm_linear.py"
        self.marker = package / "MBRIDGE_DOCKER_PATCHES"
        self.source.parent.mkdir(parents=True)
        self.source.write_text(SOURCE)
        # The real package loads CUDA libraries on import, so the script must not import it.
        (package / "__init__.py").write_text('raise ImportError("transformer_engine must not be imported")\n')

        python = self.directory / "bin/python"
        python.parent.mkdir()
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(0o755)

    def _run(self, **extra_env):
        env = {key: value for key, value in os.environ.items() if key != "MBRIDGE_PRESERVE_BASE_RUNTIME"}
        return subprocess.run(
            ["bash", str(SCRIPT), str(PATCH)],
            env={
                **env,
                "PATH": f"{self.directory / 'bin'}{os.pathsep}{os.environ['PATH']}",
                "PYTHONPATH": str(self.directory / "site-packages"),
                **extra_env,
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def test_applies_backport_without_importing_transformer_engine(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        patched = self.source.read_text()
        self.assertNotIn("main_grad.shape", patched)
        self.assertEqual(patched.count("list(origin_weight.shape),"), 2)
        self.assertEqual(self.marker.read_text(), f"{PATCH.name}\n")

    def test_skips_transformer_engine_that_already_has_the_fix(self):
        self.assertEqual(self._run().returncode, 0)
        patched = self.source.read_text()
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("already includes", result.stdout)
        self.assertEqual(self.source.read_text(), patched)
        self.assertEqual(self.marker.read_text(), f"{PATCH.name}\n")

    def test_rejects_transformer_engine_the_patch_does_not_fit(self):
        self.source.write_text("# Refactored upstream module\n")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.marker.exists())

    def test_leaves_preserved_base_runtime_untouched(self):
        result = self._run(MBRIDGE_PRESERVE_BASE_RUNTIME="True")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.source.read_text(), SOURCE)
        self.assertFalse(self.marker.exists())

    def test_docker_applies_backport_after_last_environment_change(self):
        dockerfile = (ROOT / "docker/Dockerfile.ci").read_text()
        backport = dockerfile.index(f"bash /opt/patch_transformer_engine.sh /opt/{PATCH.name}")
        self.assertIn(f"source=docker/patches/{PATCH.name},target=/opt/{PATCH.name}", dockerfile)
        for installer in ("uv sync ", "uv pip install ", "install_mamba.sh", "install_diffusion_deps.sh"):
            self.assertLess(dockerfile.rindex(installer), backport, installer)
        self.assertLess(backport, dockerfile.index("COPY --chown=1000:1000 . /opt/Megatron-Bridge"))


if __name__ == "__main__":
    unittest.main()
