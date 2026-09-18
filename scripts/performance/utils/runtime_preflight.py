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

"""Opt-in runtime import diagnostics, runnable before training dependencies load.

MBRIDGE_RUNTIME_PREFLIGHT=only checks Torch/TE and exits without training.
MBRIDGE_RUNTIME_PREFLIGHT=check also checks DeepEP/ModelOpt, then permits
training only if every import succeeds. Both require an artifact directory in
MBRIDGE_RUNTIME_PREFLIGHT_DIR. No packages or environment settings are changed.
"""

import importlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


def _command(argv: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "error": str(error)}
    return {"command": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _provenance() -> dict[str, object]:
    names = {"torch", "deep-ep", "nvidia-modelopt", "nvidia-cutlass-dsl"}
    packages = []
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name in names or name.startswith("transformer-engine"):
            packages.append({"name": name, "version": dist.version, "location": str(dist.locate_file(""))})
    sources = {}
    for path in ("/opt/transformerengine", "/opt/transformer-engine", "/opt/TransformerEngine"):
        if Path(path).is_dir():
            sources[path] = {
                "revision": _command(["git", "-C", path, "rev-parse", "HEAD"]),
                "status": _command(["git", "-C", path, "status", "--short"]),
            }
    binaries = {}
    for root in dict.fromkeys(sys.path):
        if not root:
            continue
        patterns = (
            "transformer_engine/wheel_lib/transformer_engine_torch*.so",
            "transformer_engine_torch*.so",
            "torch/lib/libc10_cuda.so",
        )
        for pattern in patterns:
            for path in Path(root).glob(pattern):
                if str(path) in binaries:
                    continue
                symbols = _command(["nm", "-D", str(path)])
                symbols["stdout"] = "\n".join(
                    line
                    for line in str(symbols.get("stdout", "")).splitlines()
                    if "c10_cuda_check_implementation" in line
                )
                binaries[str(path)] = {"ldd": _command(["ldd", str(path)]), "cuda_check_symbols": symbols}
    return {
        "python": sys.executable,
        "machine": platform.machine(),
        "sys_path": sys.path,
        # Deliberately never dump the full environment or command-line arguments.
        "library_environment": {
            name: os.environ.get(name) for name in ("LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_HOME", "VIRTUAL_ENV")
        },
        "packages": packages,
        "te_sources": sources,
        "binaries": binaries,
    }


def _probe(module_name: str) -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        # Preload Torch exactly as the training path does before loading TE.
        torch = importlib.import_module("torch")
        logger.info(
            "Torch version=%s git=%s cuda=%s path=%s",
            torch.__version__,
            torch.version.git_version,
            torch.version.cuda,
            torch.__file__,
        )
        module = importlib.import_module(module_name)
        logger.info("Imported %s from %s", module_name, getattr(module, "__file__", None))
        if module_name == "torch":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in the runtime probe")
            logger.info("GPU=%s capability=%s", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    except Exception:
        # Import failures, including native extension errors, are the output of this diagnostic.
        logger.exception("Runtime import failed: %s", module_name)
        return 1
    finally:
        maps = Path("/proc/self/maps")
        if maps.exists():
            paths = {
                line.split(maxsplit=5)[-1]
                for line in maps.read_text().splitlines()
                if any(name in line for name in ("libc10", "libtorch", "transformer_engine"))
            }
            logger.info("Loaded libraries: %s", sorted(paths))
    return 0


def _run_probe(module_name: str) -> dict[str, object]:
    command = [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), "--probe", module_name]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except subprocess.TimeoutExpired:
        return {"module": module_name, "returncode": 124, "error": "Import exceeded 120 seconds"}
    except OSError as error:
        return {"module": module_name, "returncode": 127, "error": str(error)}
    return {"module": module_name, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run_from_environment() -> None:
    """Save per-rank evidence and optionally exit before any training imports.

    Raises:
        ValueError: If enabled with an invalid mode, directory, or rank.
        SystemExit: In diagnostic-only mode, or when a required import fails.
    """
    mode = os.environ.get("MBRIDGE_RUNTIME_PREFLIGHT")
    if not mode:
        return
    if mode not in {"only", "check"}:
        raise ValueError("MBRIDGE_RUNTIME_PREFLIGHT must be only or check")
    directory = os.environ.get("MBRIDGE_RUNTIME_PREFLIGHT_DIR")
    if not directory:
        raise ValueError("MBRIDGE_RUNTIME_PREFLIGHT_DIR is required")
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID"))
    if rank is None or not rank.isdecimal():
        raise ValueError("Runtime preflight requires a nonnegative RANK or SLURM_PROCID")

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"runtime_preflight_rank-{int(rank)}.json"
    probes: list[dict[str, object]] = []
    report = {"mode": mode, "rank": int(rank), "provenance": _provenance(), "probes": probes}
    # Persist each stage before risking a native crash or peer-rank cancellation.
    path.write_text(json.dumps(report, indent=2) + "\n")
    modules = ["torch", "transformer_engine.pytorch"]
    if mode == "check":
        modules.extend(["deep_ep", "modelopt.torch"])
    for name in modules:
        probes.append(_run_probe(name))
        path.write_text(json.dumps(report, indent=2) + "\n")
    failed = [probe["module"] for probe in probes if probe["returncode"] != 0]
    logger.warning("Runtime preflight rank=%s failed=%s report=%s", rank, failed, path)
    if failed or mode == "only":
        raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--probe":
        raise SystemExit("Internal usage: runtime_preflight.py --probe MODULE")
    raise SystemExit(_probe(sys.argv[2]))
