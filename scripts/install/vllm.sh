#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# # MBridge 0.7.0 in the vLLM base container
#
# Container: `dl/dgx/vllm:main-py3.83861681-base` (registry hostname omitted).
# Run inside that Linux x86_64 image, from the patched MBridge checkout:
# `NVTE_CUDA_ARCHS=90 bash scripts/install/vllm.sh .`
#
# ## Expected dependencies
#
# | Component | Version |
# | --- | --- |
# | Python / CUDA toolkit | 3.14.4 / 13.5 |
# | PyTorch / Triton | 2.15.0a0+fd23299d96.nvinternal.main.83205032 / 3.9.0+git20469696.nv26.11 |
# | vLLM | 0.34.0+e51c829d.nvinternal.main.83861681 |
# | FlashInfer Python, cubin and JIT caches | 0.7.5+24ac7794.nvinternal.main.cu.83753500 |
# | cuDNN frontend / backend | 1.33.0 / 9.29.0.29 |
# | MBridge / pinned MCore | 0.7.0 / 0.20.0+503867317 |
# | Transformers / Tokenizers / HF Hub | 5.12.1 / 0.22.2 / 1.33.0 |
# | Transformer Engine | 2.20.2+44f31bbe |
#
# Requires git, uv, curl, tar, sha256sum and the image's CUDA build tools.
# Installs Maturin in .venv; installs cargo/rustc through apt if cargo is absent.
# Python packages go into .venv, which inherits the image's runtime packages.
# This is a forced compatibility experiment: Transformers 5.12.1 is below
# MBridge/vLLM's declared minimums; Muse Glimmer is unavailable. GPU execution
# is untested. Uses uv pip install; the release uv lock stays on Python 3.12.
# See README.md in this directory for overrides, provenance and limitations.
set -euo pipefail

export UV_PYTHON_DOWNLOADS=never
export MAX_JOBS="${MAX_JOBS:-4}"
export TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-13}"
export NVTE_CMAKE_EXTRA_ARGS="${NVTE_CMAKE_EXTRA_ARGS:-} -DCUDNN_SKIP_PRECOMPILED_LINK=ON"
export NVTE_BUILD_MAX_JOBS="${NVTE_BUILD_MAX_JOBS:-4}"
repo_dir="${1:-$PWD}"
bridge_ref="${MBRIDGE_REF:-ko3n1g/fix/mbridge-v070-python-upper-bound}"
export MBRIDGE_TRANSFORMERS_VERSION="${MBRIDGE_TRANSFORMERS_VERSION:-5.12.1}"

command -v uv
command -v git
if [[ ! -e "$repo_dir" ]]; then
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "$repo_dir"
    git -C "$repo_dir" checkout "$bridge_ref"
fi
cd "$repo_dir"
git rev-parse --show-toplevel
git rev-parse HEAD
git submodule update --init 3rdparty/Megatron-LM
git -C 3rdparty/Megatron-LM rev-parse HEAD

if [[ ! -d .venv ]]; then
    uv venv --python /usr/bin/python3 --system-site-packages .venv
fi
uv --no-config pip install --python .venv/bin/python packaging pybind11 wheel 'maturin>=1.14.1,<2'
if ! command -v cargo; then
    apt-get update
    apt-get install --no-install-recommends -y cargo rustc
fi

runtime_dir="$(mktemp -d)"
trap 'rm -rf "$runtime_dir"' EXIT
build_dir="$PWD/.venv/mbridge-build"
mkdir -p "$build_dir"
if [[ ! -d "$build_dir/nvtx" ]]; then
    git clone --depth 1 --branch v3.4.0 https://github.com/NVIDIA/NVTX.git "$build_dir/nvtx"
fi
curl --fail --silent --show-error \
    https://developer.download.nvidia.com/compute/cuda/redist/cuda_profiler_api/linux-x86_64/cuda_profiler_api-linux-x86_64-13.0.85-archive.tar.xz \
    --output "$runtime_dir/profiler.tar.xz"
printf '%s  %s\n' dc233d88a5cafa095b197e6246b4c468a4581c128da8f951d67e063cdd6bca4c "$runtime_dir/profiler.tar.xz" | sha256sum --check
mkdir "$runtime_dir/profiler"
tar -xJf "$runtime_dir/profiler.tar.xz" -C "$runtime_dir/profiler" --strip-components=1
cp -a "$runtime_dir/profiler/include/." "$build_dir/nvtx/c/include/"
export CPATH="$build_dir/nvtx/c/include${CPATH:+:$CPATH}"
export NVTE_CMAKE_BUILD_DIR="${NVTE_CMAKE_BUILD_DIR:-$build_dir/te-cmake}"
export NVTE_CMAKE_EXTRA_ARGS="$NVTE_CMAKE_EXTRA_ARGS -DCMAKE_CXX_FLAGS=-I$build_dir/nvtx/c/include -DCMAKE_CUDA_FLAGS=-I$build_dir/nvtx/c/include"
export MBRIDGE_RUNTIME_DIR="$runtime_dir"
uv --no-config run --no-project --python /usr/bin/python3 python - <<'PY'
"""Generate temporary uv configuration and preserve the image runtime versions."""

import importlib.metadata as metadata
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

runtime_dir = Path(os.environ["MBRIDGE_RUNTIME_DIR"])
overridden = ["flashinfer-python", "flashinfer-cubin", "nvidia-cudnn-frontend"]
overridden += sorted(
    {distribution.metadata["Name"] for distribution in metadata.distributions()
     if distribution.metadata.get("Name", "").startswith("flashinfer-jit-cache")}
)
preserved = ["torch", "torchvision", "triton", "vllm", "numpy"]
versions = {name: metadata.version(name) for name in preserved + overridden}
for name, version in versions.items():
    logger.info("Preserving %s==%s", name, version)
(runtime_dir / "constraints.txt").write_text(
    "".join(f"{name}=={version}\n" for name, version in versions.items())
)
(runtime_dir / "overrides.txt").write_text(
    "".join(f"{name}=={versions[name]}\n" for name in overridden)
    + f"transformers=={os.environ['MBRIDGE_TRANSFORMERS_VERSION']}\n"
    + "multi-storage-client @ git+https://github.com/NVIDIA/multi-storage-client.git@6c82181c1b32cfbdc5cf0a40a0dd4ff035927d66#subdirectory=multi-storage-client\n"
)
if os.environ["MBRIDGE_TRANSFORMERS_VERSION"] == "5.12.1":
    with (runtime_dir / "overrides.txt").open("a") as overrides:
        overrides.write("tokenizers==0.22.2\nhuggingface-hub==1.33.0\n")
if wheel_path := os.environ.get("TRANSFORMER_ENGINE_WHEEL"):
    with (runtime_dir / "overrides.txt").open("a") as overrides:
        overrides.write(f"transformer-engine @ {Path(wheel_path).resolve().as_uri()}\n")
(runtime_dir / "excludes.txt").write_text("".join(f"{name}\n" for name in versions))
(runtime_dir / "flashinfer-cache.txt").write_text(
    "".join(f"{name}\n" for name in overridden if name.startswith("flashinfer-jit-cache"))
)
source = Path("pyproject.toml").read_text()
config = source.split("[tool.uv]\n", 1)[1].split("[tool.uv.sources]", 1)[0]
config = config.replace("tool.uv.", "")
config = re.sub(r"^(managed|environments) = .*\n", "", config, flags=re.MULTILINE)
config = config.replace("onnx>=1.21.0; python_version < '3.13'", "onnx>=1.21.0")
config = config.replace(
    "nvidia-cudnn-frontend==1.29.1",
    f"nvidia-cudnn-frontend=={versions['nvidia-cudnn-frontend']}",
)
(runtime_dir / "uv.toml").write_text(config)
PY

uv --no-config pip uninstall --python .venv/bin/python \
    --requirements "$runtime_dir/flashinfer-cache.txt"
uv --config-file "$runtime_dir/uv.toml" pip install \
    --python .venv/bin/python \
    --no-build-isolation \
    --overrides "$runtime_dir/overrides.txt" \
    --excludes "$runtime_dir/excludes.txt" \
    --constraints "$runtime_dir/constraints.txt" \
    -e . -e 3rdparty/Megatron-LM vllm

uv --no-config run --no-project --python .venv/bin/python python - <<'PY'
"""Report installed versions and probe imports without running GPU workloads."""

import importlib
import importlib.metadata as metadata
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

constraints = Path(os.environ["MBRIDGE_RUNTIME_DIR"], "constraints.txt").read_text()
for requirement in constraints.splitlines():
    name, expected = requirement.split("==", 1)
    actual = metadata.version(name)
    if actual != expected:
        raise RuntimeError(f"Runtime changed: {name}: {expected} -> {actual}")
logger.info("Image runtime versions preserved")
for name in ("megatron-bridge", "megatron-core", "torch", "transformers", "vllm"):
    logger.info("%s==%s", name, metadata.version(name))
for module in ("torch", "transformers", "flashinfer", "transformer_engine.pytorch", "megatron.core", "megatron.bridge", "vllm"):
    importlib.import_module(module)
    logger.info("Import OK: %s", module)
PY
