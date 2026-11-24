# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build or load the precompiled native TP1 BF16 multimax/CE forward kernel."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import Tensor

_SOURCE = Path(__file__).with_name("chunked_multimax_cuda.cu")
_ENV = "ERNIE55_CUDA_HEAD_LIBRARY"


@lru_cache(maxsize=1)
def _fingerprint() -> str:
    return hashlib.sha256(_SOURCE.read_bytes()).hexdigest()


def _default_path(arch: str) -> Path:
    return _SOURCE.with_name(f"libernie55_multimax_{arch.replace('_', '')}.so")


def _verify(path: Path, arch: str) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.ernie55_multimax_ce_arch.restype = ctypes.c_char_p
    library.ernie55_multimax_ce_source.restype = ctypes.c_char_p
    if library.ernie55_multimax_ce_arch().decode() != arch:
        raise ValueError(f"Kernel in {path} was compiled for a different GPU target (expected {arch})")
    if library.ernie55_multimax_ce_source().decode() != _fingerprint():
        raise ValueError(f"CUDA kernel in {path} does not match Bridge source {_SOURCE}")
    launch = library.ernie55_multimax_ce_forward
    launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    return library


def build_library(arch: str, output: Path) -> None:
    """Compile one architecture-specific library and atomically publish it for training ranks."""
    if not re.fullmatch(r"sm_[0-9]{2,3}", arch):
        raise ValueError("Expected a numeric CUDA target such as sm_103.")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # The persistent lock file also coordinates multiple independent jobs sharing this output.
    with output.with_suffix(".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            try:
                _verify(output, arch)
            except (OSError, AttributeError, ValueError):
                pass
            else:
                return
        with tempfile.TemporaryDirectory(prefix=".ernie55-head-", dir=output.parent) as tmp:
            artifact = Path(tmp) / output.name
            subprocess.run(
                [
                    "nvcc",
                    "-O3",
                    f"-arch={arch}",
                    "-std=c++17",
                    "--shared",
                    "-Xcompiler=-fPIC",
                    f'-DERNIE55_CUDA_HEAD_SOURCE="{_fingerprint()}"',
                    f'-DERNIE55_CUDA_HEAD_ARCH="{arch}"',
                    str(_SOURCE),
                    "-o",
                    str(artifact),
                ],
                check=True,
            )
            _verify(artifact, arch)
            os.replace(artifact, output)


@lru_cache(maxsize=None)
def _get_library(device: int) -> ctypes.CDLL:
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    path = Path(os.environ[_ENV]) if _ENV in os.environ else _default_path(arch)
    if not path.is_absolute():
        raise ValueError(f"{_ENV} must name an absolute file path. Got {path}")
    try:
        return _verify(path, arch)
    except (OSError, AttributeError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot use ERNIE CUDA head {path} ({exc}). Build it once in the target container with "
            f"python {__file__} --arch {arch} --output {path} or set {_ENV} to a matching library."
        ) from exc


def forward(
    raw: Tensor,
    rt: Tensor,
    labels: Tensor,
    maxima: Tensor,
    denom: Tensor,
    loss: Tensor,
    preds: Tensor | None,
) -> None:
    """Reduce a contiguous [tokens, vocabulary] BF16 projection into FP32 CE and backward statistics."""
    import torch

    rows, vocab = raw.shape
    if not raw.is_cuda or raw.dtype != torch.bfloat16 or vocab < 2048 or vocab % 8:
        raise ValueError("CUDA multimax requires CUDA BF16 logits with a vocabulary divisible by 8 and >= 2048.")
    if not 0 < rows <= 2**31 - 1 or vocab > 2**31 - 1 or raw.data_ptr() % 16:
        raise ValueError("CUDA multimax requires 16-byte-aligned rows and int32-compatible positive dimensions.")
    expected = [
        (rt, 8, torch.float32),
        (labels, rows, torch.int64),
        (maxima, rows, torch.float32),
        (denom, rows, torch.float32),
        (loss, rows, torch.float32),
    ]
    if preds is not None:
        expected.append((preds, rows, torch.int64))
    if not raw.is_contiguous() or any(
        t.device != raw.device or t.shape != (length,) or t.dtype != dtype or not t.is_contiguous()
        for t, length, dtype in expected
    ):
        raise ValueError("CUDA multimax requires same-device contiguous FP32 statistics and int64 labels/predictions.")
    with torch.cuda.device(raw.device):
        library = _get_library(raw.device.index)
        stream = torch.cuda.current_stream(raw.device).cuda_stream
        code = library.ernie55_multimax_ce_forward(
            raw.data_ptr(),
            rt.data_ptr(),
            labels.data_ptr(),
            maxima.data_ptr(),
            denom.data_ptr(),
            loss.data_ptr(),
            preds.data_ptr() if preds is not None else None,
            rows,
            vocab,
            stream,
        )
    if code:
        raise RuntimeError(f"ERNIE CUDA multimax kernel launch failed with CUDA error {code}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="CUDA architecture, e.g. sm_103 for the D2-Z GB300")
    parser.add_argument(
        "--output", type=Path, help="Publish path. Defaults to the architecture-specific package path."
    )
    args = parser.parse_args()
    build_library(args.arch, args.output or _default_path(args.arch))
