# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise the image installer's compiler and cubin gates without compiling CUDA."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "docker/common/install_mok.sh"
REVISION = "a" * 40
TARGETS = "sm_100a\nsm_103a\nsm_107a\n"


class TestInstallMok(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        self.record = self.directory / "commands.jsonl"
        tool = self.bin / "tool"
        tool.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "name, args = Path(sys.argv[0]).name, sys.argv[1:]\n"
            "with open(os.environ['MOK_TEST_RECORD'], 'a') as out:\n"
            "    out.write(json.dumps({'name': name, 'args': args, 'arch': os.environ.get('MOK_ARCH')}) + '\\n')\n"
            "if name == 'realpath':\n"
            "    sys.stdout.write(str(Path(args[-1]).resolve()) + '\\n')\n"
            "elif name == 'nvcc':\n"
            "    if args == ['--list-gpu-code']:\n"
            "        sys.stdout.write(os.environ['MOK_TEST_LISTED_TARGETS'])\n"
            "    else:\n"
            "        assert len(args) == 6 and args[:2] == ['-cubin', '-gencode'] and args[4] == '-o', args\n"
            "        target = Path(args[5]).stem\n"
            "        assert args[2] == 'arch=compute_' + target.removeprefix('sm_') + ',code=' + target, args\n"
            "        assert '__global__ void mok_arch_probe()' in Path(args[3]).read_text()\n"
            "        if target not in os.environ['MOK_TEST_COMPILER_TARGETS'].split():\n"
            "            sys.exit(1)\n"
            "        Path(args[5]).write_bytes(b'' if os.environ.get('MOK_TEST_EMPTY_PROBE') else b'compiled cubin')\n"
            "elif name == 'git':\n"
            "    if args[0] == 'init':\n"
            "        scripts = Path(args[1]) / 'scripts'\n"
            "        scripts.mkdir(parents=True)\n"
            "        (scripts / 'prepare_thunderkittens.sh').write_text('exit 0\\n')\n"
            "    else:\n"
            "        assert args[:1] == ['-C'] and args[2] in ('remote', 'fetch', 'checkout'), args\n"
            "elif name == 'uv':\n"
            "    if args[0] == 'build':\n"
            "        wheels = Path(args[args.index('--out-dir') + 1])\n"
            "        wheels.mkdir(parents=True, exist_ok=True)\n"
            "        (wheels / 'mok.whl').touch()\n"
            "    elif args[0] == 'run':\n"
            "        sys.stdout.write('/fake/mok/_C.so')\n"
            "    else:\n"
            "        assert args[:2] == ['pip', 'install'], args\n"
            "elif name == 'cuobjdump':\n"
            "    assert args == ['--list-elf', '/fake/mok/_C.so'], args\n"
            "    for target in os.environ['MOK_TEST_WHEEL_TARGETS'].split():\n"
            "        sys.stdout.write('ELF file 1: kernel.' + target + '.cubin\\n')\n"
            "else:\n"
            "    raise AssertionError(name)\n"
        )
        tool.chmod(0o755)
        for name in ("realpath", "git", "uv", "nvcc", "cuobjdump"):
            (self.bin / name).symlink_to(tool)

    def _run(self, *, revision=REVISION, **overrides):
        self.record.unlink(missing_ok=True)
        env = {key: value for key, value in os.environ.items() if key != "MOK_ARCH"}
        return subprocess.run(
            ["bash", str(INSTALLER), revision, str(self.directory / "wheels")],
            env={
                **env,
                "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                "CUDA_HOME": str(self.directory),
                "MOK_TEST_RECORD": str(self.record),
                "MOK_TEST_COMPILER_TARGETS": TARGETS,
                "MOK_TEST_LISTED_TARGETS": "sm_100\nsm_103\nsm_107\n",
                "MOK_TEST_WHEEL_TARGETS": TARGETS,
                **overrides,
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def _commands(self):
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def test_default_builds_all_targets_from_exact_revision(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self._commands()
        fetch = next(command for command in commands if "fetch" in command["args"])
        self.assertEqual(fetch["args"][-4:], ["--depth", "1", "origin", REVISION])
        build = next(command for command in commands if command["name"] == "uv" and command["args"][0] == "build")
        self.assertEqual(build["arch"], "ALL")
        self.assertIn("--no-build-isolation", build["args"])
        self.assertFalse(Path(build["args"][-1]).exists(), "Temporary source must be cleaned up")

    def test_each_requested_architecture_can_be_built_alone(self):
        for arch, cubin in (("SM100", "sm_100a"), ("SM103", "sm_103a"), ("SM107", "sm_107a")):
            with self.subTest(arch=arch):
                result = self._run(MOK_ARCH=arch, MOK_TEST_COMPILER_TARGETS=cubin, MOK_TEST_WHEEL_TARGETS=cubin)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                probes = [command for command in self._commands() if command["name"] == "nvcc"]
                self.assertEqual(len(probes), 1)
                self.assertEqual(probes[0]["args"][2], f"arch=compute_{cubin.removeprefix('sm_')},code={cubin}")

    def test_compiles_specific_targets_when_listing_only_reports_base_names(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self._commands()
        probes = [command for command in commands if command["name"] == "nvcc"]
        self.assertEqual(len(probes), 3)
        self.assertEqual(
            [command["args"][2] for command in probes],
            [f"arch=compute_{target.removeprefix('sm_')},code={target}" for target in TARGETS.split()],
        )
        self.assertTrue(all(command["args"][0] == "-cubin" for command in probes))
        self.assertTrue(all(not Path(command["args"][3]).parent.exists() for command in probes))
        first_git = next(index for index, command in enumerate(commands) if command["name"] == "git")
        self.assertTrue(all(command["name"] != "nvcc" for command in commands[first_git:]))

    def test_empty_compiler_output_fails_before_fetching_source(self):
        result = self._run(MOK_TEST_EMPTY_PROBE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot build sm_100a", result.stderr)
        commands = self._commands()
        self.assertFalse(any(command["name"] == "git" for command in commands))
        probe = next(command for command in commands if command["name"] == "nvcc")
        self.assertFalse(Path(probe["args"][3]).parent.exists(), "Failed probe must be cleaned up")

    def test_missing_rubin_cubin_fails_default_build(self):
        result = self._run(MOK_TEST_WHEEL_TARGETS="sm_100a sm_103a")
        self.assertNotEqual(result.returncode, 0)

    def test_missing_blackwell_cubin_fails_default_build(self):
        result = self._run(MOK_TEST_WHEEL_TARGETS="sm_103a sm_107a")
        self.assertNotEqual(result.returncode, 0)

    def test_unsupported_compiler_fails_before_fetching_source(self):
        result = self._run(MOK_TEST_COMPILER_TARGETS="sm_100a\nsm_103a\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot build sm_107a", result.stderr)
        commands = self._commands()
        self.assertFalse(any(command["name"] == "git" for command in commands))
        probes = [command for command in commands if command["name"] == "nvcc"]
        self.assertEqual(len(probes), 3)
        self.assertTrue(all(not Path(command["args"][3]).parent.exists() for command in probes))

    def test_unsupported_architecture_fails_before_fetching_source(self):
        result = self._run(MOK_ARCH="SM90")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported MOK_ARCH", result.stderr)
        self.assertFalse(any(command["name"] == "git" for command in self._commands()))

    def test_nonimmutable_revision_fails_before_fetching_source(self):
        result = self._run(revision="main")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MOK_COMMIT must be a full Git commit", result.stderr)
        self.assertFalse(any(command["name"] == "git" for command in self._commands()))
