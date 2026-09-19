"""CPU tests of direct-Enroot rendering and the rank-local storage gate."""

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("container_runtime_test", ROOT / "scripts/common/container_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def arguments(**overrides):
    values = dict(
        container_runtime="enroot",
        enroot_root="/shared/runtime",
        container_image="/shared/model.sqsh",
        executor="slurm",
        srun_args=[],
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def render(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_require_lustre", lambda paths: None)
    nemo = types.ModuleType("nemo_run")
    nemo.Script = lambda **kwargs: types.SimpleNamespace(**kwargs)
    config = types.ModuleType("nemo_run.config")
    config.get_nemorun_home = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "nemo_run", nemo)
    monkeypatch.setitem(sys.modules, "nemo_run.config", config)

    def make(args=None, *, mounts=None, env_names=None):
        executor = types.SimpleNamespace(
            container_image="image",
            container_mounts=["mount"],
            container_env=["TOKEN"],
            additional_parameters={"segment": 1},
        )
        task = types.SimpleNamespace(
            entrypoint="python",
            path="/opt/Megatron-Bridge/worker.py",
            env={"PYTHONPATH": "/opt/Megatron-Bridge/src:$PYTHONPATH"},
            args=[shlex.quote("a '$HOME'; $(touch injected)\nnext")],
        )
        wrapped = runtime.apply_container_runtime(
            args or arguments(), executor=executor, task=task, mounts=mounts or [], env_names=env_names or []
        )
        return wrapped, executor, task

    return make


def test_submission_gate_rejects_non_lustre_nemorun_home(monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **kw: "ext2/ext3\n")
    with pytest.raises(ValueError, match="including NEMORUN_HOME"):
        runtime._require_lustre(["/home/example/.nemo_run"])


def test_submission_gate_checks_every_path(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda command, **kw: calls.append(command) or "lustre\n")
    runtime._require_lustre(["/storage/image.sqsh", "/storage/runtime"])
    assert [command[-1] for command in calls] == ["/storage/image.sqsh", "/storage/runtime"]


def test_pyxis_is_unchanged(render):
    wrapped, executor, task = render(arguments(container_runtime="pyxis", enroot_root=None))
    assert wrapped is task
    assert executor.container_image == "image"
    assert executor.container_mounts == ["mount"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"enroot_root": None},
        {"enroot_root": "/"},
        {"enroot_root": "relative"},
        {"enroot_root": "/data/../tmp"},
        {"container_image": "docker://registry/model"},
        {"container_image": "/image.tar"},
        {"executor": "local"},
        {"srun_args": ["--container-image=other"]},
        {"srun_args": ["--export=ALL"]},
        {"container_runtime": "pyxis"},
        {"container_runtime": "bad"},
    ],
)
def test_rejects_ambiguous_or_incompatible_options(overrides):
    with pytest.raises(ValueError):
        runtime.validate_container_runtime(arguments(**overrides))


def test_environment_defaults_and_explicit_selection(monkeypatch):
    monkeypatch.setenv("BRIDGE_CONTAINER_RUNTIME", "enroot")
    monkeypatch.setenv("BRIDGE_ENROOT_ROOT", "/shared/runtime")
    parser = argparse.ArgumentParser()
    runtime.add_container_runtime_args(parser.add_argument_group("Execution"))
    assert parser.parse_args([]).container_runtime == "enroot"
    assert parser.parse_args([]).enroot_root == "/shared/runtime"
    assert parser.parse_args(["--container-runtime", "pyxis"]).container_runtime == "pyxis"


def test_wrapper_disables_pyxis_preserves_resources_and_secret_names(render, monkeypatch):
    monkeypatch.setenv("TOKEN", "not-to-be-serialized")
    wrapped, executor, task = render(env_names=["TOKEN"])
    assert executor.container_image is None
    assert executor.container_mounts == executor.container_env == []
    assert executor.additional_parameters == {"segment": 1, "export": "PATH,TOKEN"}
    assert wrapped.env == task.env
    assert "not-to-be-serialized" not in wrapped.inline
    assert "TOKEN" in wrapped.inline
    assert "enroot start" in wrapped.inline
    assert "enroot import" not in wrapped.inline
    subprocess.run(["bash", "-n"], input=wrapped.inline, text=True, check=True)


@pytest.mark.parametrize(
    "mount",
    [
        "relative:/opt",
        "/src:relative",
        "/src:/tmp",
        "/src:/",
        "/src:/opt:bad",
        "/src:/x/../tmp",
        "/src:/tmp/",
        "/src:/var//tmp",
    ],
)
def test_rejects_unsafe_mounts(render, mount):
    with pytest.raises(ValueError):
        render(mounts=[mount])


@pytest.mark.parametrize("name", ["TOKEN=value", "BAD-NAME", "ENROOT_RUNTIME_PATH"])
def test_rejects_environment_injection(render, name):
    with pytest.raises(ValueError):
        render(env_names=[name])


@pytest.fixture
def fake_host(tmp_path, monkeypatch):
    tools = tmp_path / "bin"
    tools.mkdir()
    stat = tools / "stat"
    stat.write_text('#!/bin/bash\necho "${FAKE_FS:-lustre}"\n')
    stat.chmod(0o755)
    enroot = tools / "enroot"
    enroot.write_text("""#!/bin/bash
set -eu
if [[ "$1" == info ]]; then
  for name in ENROOT_DATA_PATH ENROOT_RUNTIME_PATH ENROOT_CACHE_PATH ENROOT_TEMP_PATH ENROOT_CONFIG_PATH ENROOT_MOUNT_HOME ENROOT_ROOTFS_WRITABLE ENROOT_LOGIN_SHELL; do
    if [[ "$name" == "${BAD_SETTING:-}" ]]; then
      echo " $name=/raid/unexpected"
    elif [[ "$name" == ENROOT_MOUNT_HOME || "$name" == ENROOT_ROOTFS_WRITABLE || "$name" == ENROOT_LOGIN_SHELL ]]; then
      case "${BOOLEAN_FORMAT-n}" in n|no|false|FALSE) ;; *) echo " $name=${BOOLEAN_FORMAT}" ;; esac
    else
      echo " $name=${!name}"
    fi
  done
else
  printf '%s\\0' "$@" > "$ENROOT_CAPTURE"
  exit "${WORKER_EXIT:-0}"
fi
""")
    enroot.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("PYTHONPATH", "test-source")
    monkeypatch.setenv("ENROOT_CAPTURE", str(tmp_path / "capture"))
    image = tmp_path / "fixture.sqsh"
    # Empty fake file, not a container image; enroot itself is a local test double.
    image.touch()
    return arguments(enroot_root=str(tmp_path), container_image=str(image))


def test_rank_wrapper_executes_with_lustre_paths_and_exact_arguments(render, fake_host, tmp_path):
    wrapped, _, _ = render(fake_host, mounts=[f"{tmp_path}:/opt/Megatron-Bridge:ro"])
    result = subprocess.run(["bash"], input=wrapped.inline, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    captured = (tmp_path / "capture").read_bytes().decode().split("\0")[:-1]
    assert captured[0] == "start"
    assert "--container-image" not in captured
    assert f"{tmp_path}:/opt/Megatron-Bridge:none:bind,ro" in captured
    assert any("/tmp:none:bind,rw" in arg for arg in captured)
    assert "SLURM_PROCID" in captured
    assert "PYTHONPATH" in captured
    assert "TMPDIR=/tmp" in captured
    assert not (tmp_path / "injected").exists()
    command = captured[-1].split("exec ", 1)[1]
    assert shlex.split(command) == ["python", "/opt/Megatron-Bridge/worker.py", "a '$HOME'; $(touch injected)\nnext"]
    roots = list(tmp_path.glob("job-123-rank-2-*"))
    assert len(roots) == 1
    assert roots[0].stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "setting",
    ["ENROOT_DATA_PATH", "ENROOT_RUNTIME_PATH", "ENROOT_CACHE_PATH", "ENROOT_TEMP_PATH", "ENROOT_CONFIG_PATH"],
)
def test_site_path_override_fails_before_image_start(render, fake_host, tmp_path, monkeypatch, setting):
    monkeypatch.setenv("BAD_SETTING", setting)
    wrapped, _, _ = render(fake_host)
    result = subprocess.run(["bash"], input=wrapped.inline, text=True, capture_output=True)
    assert result.returncode != 0
    assert f"overrides {setting}" in result.stderr
    assert not (tmp_path / "capture").exists()


def test_non_lustre_storage_fails_before_creating_runtime(render, fake_host, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FS", "ext2/ext3")
    wrapped, _, _ = render(fake_host)
    result = subprocess.run(["bash"], input=wrapped.inline, text=True, capture_output=True)
    assert result.returncode != 0
    assert "not on Lustre" in result.stderr
    assert not list(tmp_path.glob("job-*"))
    assert not (tmp_path / "capture").exists()


def test_worker_failure_propagates_without_cleanup(render, fake_host, tmp_path, monkeypatch):
    monkeypatch.setenv("WORKER_EXIT", "17")
    wrapped, _, _ = render(fake_host)
    result = subprocess.run(["bash"], input=wrapped.inline, text=True, capture_output=True)
    assert result.returncode == 17
    assert len(list(tmp_path.glob("job-*"))) == 1


def test_explicit_cache_is_validated_and_mounted(render, fake_host, tmp_path, monkeypatch):
    cache = tmp_path / "model-cache"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))
    wrapped, _, _ = render(fake_host, env_names=["HF_HOME"])
    result = subprocess.run(["bash"], input=wrapped.inline, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    captured = (tmp_path / "capture").read_bytes().decode().split("\0")
    assert f"{cache}:{cache}:none:bind,rw" in captured
    assert "HF_HOME" in captured
    assert not any(value.startswith("HF_HOME=") for value in captured)


def test_explicit_missing_cache_fails_before_start(render, fake_host, tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path / "missing"))
    wrapped, _, _ = render(fake_host, env_names=["FLASHINFER_WORKSPACE_BASE"])
    result = subprocess.run(["bash"], input=wrapped.inline, text=True, capture_output=True)
    assert result.returncode != 0
    assert "FLASHINFER_WORKSPACE_BASE must name" in result.stderr
    assert not (tmp_path / "capture").exists()


def test_managed_mounts_follow_user_ancestor_mounts(render, fake_host, tmp_path):
    wrapped, _, _ = render(fake_host, mounts=[f"{tmp_path}:/var:ro", f"{tmp_path}:{tmp_path.parent}:ro"])
    result = subprocess.run(["bash"], input=wrapped.inline, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    captured = (tmp_path / "capture").read_bytes().decode().split("\0")
    mounts = [captured[i + 1] for i, arg in enumerate(captured) if arg == "--mount"]
    assert mounts[0].endswith(":/var:none:bind,ro")
    assert mounts[-3].split(":")[0] == mounts[-3].split(":")[1]
    assert mounts[-2].endswith(":/tmp:none:bind,rw")
    assert mounts[-1].endswith(":/var/tmp:none:bind,rw")


@pytest.mark.parametrize("value", ["", "n", "no", "false", "FALSE", "0", "yes", "true", "1"])
def test_boolean_normalization_does_not_enable_writable_root(render, fake_host, tmp_path, monkeypatch, value):
    monkeypatch.setenv("BOOLEAN_FORMAT", value)
    wrapped, _, _ = render(fake_host)
    result = subprocess.run(["bash"], input=wrapped.inline, text=True, capture_output=True)
    disabled = value in ("n", "no", "false", "FALSE")
    assert (result.returncode == 0) == disabled
    assert (tmp_path / "capture").exists() == disabled


@pytest.mark.parametrize(
    "family,loader",
    [
        ("scripts/inference/test_setup_inference.py", "_load_setup_inference_module"),
        ("scripts/training/test_setup_experiment.py", "_load_setup_experiment_module"),
        ("conversion/launcher/test_setup_conversion.py", "_load_setup_conversion_module"),
    ],
)
def test_all_public_launchers_apply_the_backend(monkeypatch, tmp_path, family, loader):
    spec = importlib.util.spec_from_file_location("launcher_test_support", ROOT / "tests/unit_tests" / family)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    module = getattr(support, loader)()
    module.run.Script = lambda **kw: types.SimpleNamespace(**kw)
    module.run.Packager = lambda: None
    module.run.LocalTunnel = lambda **kw: types.SimpleNamespace(**kw)
    module.run.SlurmExecutor = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setattr(module, "get_nemorun_home", lambda: str(tmp_path))
    calls = []
    sentinel = object()

    def apply(args, **kw):
        assert args.container_runtime == "enroot"
        assert args.enroot_root == "/storage/runtime"
        assert "--container-runtime" not in kw["task"].args
        calls.append(kw)
        return sentinel

    class Experiment:
        def __init__(self, *args, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def add(self, task, **kw):
            assert task is sentinel

        def dryrun(self):
            calls.append("dryrun")

    module.run.Experiment = Experiment
    monkeypatch.setattr(module, "apply_container_runtime", apply)
    argv = [
        "--container-runtime",
        "enroot",
        "--enroot-root",
        "/storage/runtime",
        "--container-image",
        "/storage/model.sqsh",
        "--account",
        "account",
        "--partition",
        "batch",
        "--gpus-per-node",
        "1",
        "--submission-dry-run",
    ]
    if family.startswith("conversion"):
        argv = [
            "import",
            "--executor",
            "slurm",
            "--device",
            "gpu",
            "--hf-model",
            "org/model",
            "--megatron-path",
            "work/checkpoint",
            *argv,
        ]
    elif "training" in family:
        argv += ["--recipe", "gpt_oss_20b_sft_config", "--mode", "sft"]
    else:
        argv += ["--prompt", "test"]
    module.main(argv)
    assert len(calls) == 2 and calls[-1] == "dryrun"
