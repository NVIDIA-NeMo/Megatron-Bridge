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

"""Checks for running GatedDeltaNet (GDN) layers on cuDNN's kernels.

``model.gdn_kernel_backend="transformer_engine"`` runs GatedDeltaNet through Transformer Engine's
GatedDeltaNetAttention on the cuDNN frontend kernels in ``cudnn.linear_attention``.
"""

import re

from megatron.bridge.utils.common_utils import warn_rank_0
from megatron.bridge.utils.import_utils import get_distribution_version, is_module_available


TRANSFORMER_ENGINE_GDN_MIN_VERSION = (2, 19)  # first Transformer Engine with GatedDeltaNetAttention
# Transformer Engine 2.19's GatedDeltaNetAttention raises a ValueError under Transformer Engine's FP8 autocast, which
# NVFP4 also uses. Megatron-Core enters it around the layers for model.fp8. For model.fp4, Megatron-Core's own
# transformer and hybrid blocks, CUDA-graph capture and EP-overlap schedule enter it, but Bridge's Qwen3-VL block does
# not. Transformer Engine 2.20.2's linear-attention modules ignore FP8 autocast.
_TRANSFORMER_ENGINE_GDN_FP8_RAISING_RELEASE = (2, 19)
TRANSFORMER_ENGINE_GDN_FP8_VERSION = (2, 20, 2)
CUDNN_FRONTEND_GDN_MIN_VERSION = (1, 27, 0)  # first cuDNN frontend with cudnn.linear_attention
CUDNN_FRONTEND_GDN_NAN_FREE_VERSION = (1, 29, 0)  # earlier GatedDeltaNet kernels can return NaN
# CUTEDSL_MIN_VERSION in cudnn/frost/buffers.py of cuDNN frontend 1.29.0. Below it the frontend routes
# GatedDeltaNet to its cuTile engine instead of the FROST engine, and cuTile is slower than FLA.
CUTLASS_DSL_GDN_MIN_VERSION = (4, 7, 0)
_LEADING_INTEGER = re.compile(r"^(\d+)")


def _release_tuple(version: str) -> tuple[int, ...]:
    """Parse ``version`` the way cuDNN frontend 1.29 compares CuTe DSL versions.

    Keeps the leading integer of each of the first three dot components, drops a local label and zero-pads
    short versions ("4.6" -> (4, 6, 0)). Returns ``()`` when nothing parses.
    """
    parts: list[int] = []
    for component in version.split("+", 1)[0].split(".")[:3]:
        match = _LEADING_INTEGER.match(component)
        if match is None:
            break
        parts.append(int(match.group(1)))
    if not parts:
        return ()
    return tuple(parts + [0] * (3 - len(parts)))


def _version_str(version: tuple[int, ...]) -> str:
    """Render a release tuple such as ``(4, 7, 0)`` as ``"4.7.0"``."""
    return ".".join(str(part) for part in version)


def cudnn_gdn_stack_issues(
    *,
    have_te_gdn: bool,
    cudnn_frontend_version: str | None,
    cutlass_importable: bool,
    cutlass_dsl: tuple[str, str] | None,
) -> list[str]:
    """List why ``gdn_kernel_backend="transformer_engine"`` would not run on cuDNN's fast GatedDeltaNet kernels.

    Args:
        have_te_gdn: Megatron-Core's ``HAVE_TE_GDN``: Transformer Engine provides GatedDeltaNetAttention.
        cudnn_frontend_version: Installed ``nvidia-cudnn-frontend`` version, or None.
        cutlass_importable: Whether the CuTe DSL ``cutlass`` package resolves.
        cutlass_dsl: ``(distribution, version)`` of the first installed of ``nvidia-cutlass-dsl`` and
            ``nvidia-cutlass-dsl-internal`` (what cuDNN frontend's ``cutedsl_state()`` reads), or None.

    Returns:
        One message per gap; empty when the installed packages pass cuDNN frontend 1.29's FROST support gate
        (``cutedsl_state`` / ``cutedsl_too_old``). Two cases still fall back to the slower cuTile engine without a
        message: a GPU outside SM100-SM103 and SM107 (checking it would initialize CUDA), and a ``cutlass`` package
        that does not match the ``nvidia-cutlass-dsl`` metadata, such as a ``PYTHONPATH`` copy of the wheel without
        its ``dsl_packages`` directory.
    """
    issues: list[str] = []
    if not have_te_gdn:
        issues.append(
            "Transformer Engine has no GatedDeltaNetAttention (added in transformer-engine "
            f"{_version_str(TRANSFORMER_ENGINE_GDN_MIN_VERSION)}), so Megatron-Core will fail to build the model"
        )
    if cudnn_frontend_version is None:
        issues.append("nvidia-cudnn-frontend is not installed, so the first GatedDeltaNet forward will fail")
        return issues
    frontend = _release_tuple(cudnn_frontend_version)
    if frontend and frontend < CUDNN_FRONTEND_GDN_MIN_VERSION:
        issues.append(
            f"nvidia-cudnn-frontend {cudnn_frontend_version} has no cudnn.linear_attention (added in "
            f"{_version_str(CUDNN_FRONTEND_GDN_MIN_VERSION)}), so the first GatedDeltaNet forward will fail"
        )
        return issues
    if frontend and frontend < CUDNN_FRONTEND_GDN_NAN_FREE_VERSION:
        issues.append(
            f"nvidia-cudnn-frontend {cudnn_frontend_version} is older than "
            f"{_version_str(CUDNN_FRONTEND_GDN_NAN_FREE_VERSION)}, and its GatedDeltaNet kernels can return NaN"
        )
        return issues
    # cuDNN frontend 1.29's FROST gate (cutedsl_state / cutedsl_too_old): the cutlass package must resolve and the
    # public wheel must be at least the floor; internal builds and missing or unparsable metadata pass. Older
    # frontends gate differently and are reported above.
    if not cutlass_importable:
        issues.append(
            "the CuTe DSL package cutlass (nvidia-cutlass-dsl) is not importable, so cuDNN frontend runs "
            "GatedDeltaNet on its cuTile engine, which is slower than FLA"
        )
    elif cutlass_dsl is not None and cutlass_dsl[0] == "nvidia-cutlass-dsl":
        dsl = _release_tuple(cutlass_dsl[1])
        if dsl and dsl < CUTLASS_DSL_GDN_MIN_VERSION:
            issues.append(
                f"nvidia-cutlass-dsl {cutlass_dsl[1]} is older than {_version_str(CUTLASS_DSL_GDN_MIN_VERSION)}, "
                "so cuDNN frontend runs GatedDeltaNet on its cuTile engine, which is slower than FLA"
            )
    return issues


def te_gdn_fp8_issues(*, transformer_engine_version: str | None, fp8: bool, fp4: bool) -> list[str]:
    """List why Transformer Engine's GatedDeltaNetAttention would fail under the model's FP8 or FP4 autocast.

    Transformer Engine 2.19 raises "GatedDeltaNetAttention does not support FP8 autocast or FP8 calibration." whenever
    GatedDeltaNet runs under Transformer Engine's FP8 autocast, which NVFP4 also uses. With ``model.fp8`` Megatron-Core
    runs the layers under it, so training fails in the first forward pass. With only ``model.fp4`` it fails where
    Megatron-Core's own transformer and hybrid blocks, CUDA-graph capture or EP-overlap schedule enter that autocast;
    Bridge's Qwen3-VL block does not. Transformer Engine 2.20.2 and later ignore FP8 autocast in GatedDeltaNet. Only
    the 2.19 family is flagged; releases without GatedDeltaNetAttention are reported by ``cudnn_gdn_stack_issues``.

    Args:
        transformer_engine_version: Installed ``transformer-engine`` version, or None.
        fp8: Whether the model config sets ``fp8``.
        fp4: Whether the model config sets ``fp4``.

    Returns:
        One message for Transformer Engine 2.19 with FP8 or FP4; empty otherwise, including unparsable versions.
    """
    if not (fp8 or fp4) or transformer_engine_version is None:
        return []
    if _release_tuple(transformer_engine_version)[:2] != _TRANSFORMER_ENGINE_GDN_FP8_RAISING_RELEASE:
        return []
    if fp8:
        where = "model.fp8 runs the layers under it, so training fails in the first forward pass"
    else:
        where = (
            "model.fp4 runs GatedDeltaNet under it in Megatron-Core's transformer and hybrid blocks, CUDA-graph "
            "capture and EP-overlap schedules, so GatedDeltaNet fails there"
        )
    return [
        f"transformer-engine {transformer_engine_version}'s GatedDeltaNetAttention raises under FP8 autocast, and "
        f"{where} (transformer-engine {_version_str(TRANSFORMER_ENGINE_GDN_FP8_VERSION)} and later ignore FP8 "
        "autocast in GatedDeltaNet)"
    ]


def is_te_gdn_available() -> bool:
    """Return Megatron-Core's own check that Transformer Engine provides GatedDeltaNetAttention.

    Returns:
        ``megatron.core.extensions.transformer_engine.HAVE_TE_GDN``, or False when Megatron-Core does not define it.
    """
    try:
        from megatron.core.extensions.transformer_engine import HAVE_TE_GDN
    except ImportError:
        return False
    return bool(HAVE_TE_GDN)


def probe_cudnn_gdn_stack_issues() -> list[str]:
    """Run ``cudnn_gdn_stack_issues`` against this environment without importing cudnn or cutlass.

    Returns:
        The issues ``cudnn_gdn_stack_issues`` reports for the installed packages; empty when the stack is complete.
    """
    cutlass_dsl: tuple[str, str] | None = None
    for name in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-internal"):
        version = get_distribution_version(name)
        if version is not None:
            cutlass_dsl = (name, version)
            break
    return cudnn_gdn_stack_issues(
        have_te_gdn=is_te_gdn_available(),
        cudnn_frontend_version=get_distribution_version("nvidia-cudnn-frontend"),
        cutlass_importable=is_module_available("cutlass"),
        cutlass_dsl=cutlass_dsl,
    )


def validate_cudnn_gdn_stack(model_config: object) -> None:
    """Warn when ``gdn_kernel_backend="transformer_engine"`` cannot reach cuDNN's fast GatedDeltaNet kernels.

    ``ConfigContainer.validate()`` calls this on the final model config, after every override. Warn-only: the
    backend is never changed, so a recipe builds the same config in every environment, and the message names the
    fix and the FLA override. Setups that cannot run at all also fail later in Megatron-Core or Transformer Engine;
    the silent cases are the cuTile fallback (slower than FLA) and cuDNN frontend releases before 1.29.0
    (GatedDeltaNet NaN). With ``fp8`` or ``fp4`` set it also flags Transformer Engine 2.19, whose
    GatedDeltaNetAttention raises under FP8 autocast (``te_gdn_fp8_issues``), and the install hint then asks for
    Transformer Engine 2.20.2. The check reads installed package metadata only; ``cudnn_gdn_stack_issues`` lists what
    it cannot see.

    Args:
        model_config: The model provider or config; read for ``gdn_kernel_backend``, ``fp8`` and ``fp4``.
    """
    if getattr(model_config, "gdn_kernel_backend", None) != "transformer_engine":
        return
    fp8 = bool(getattr(model_config, "fp8", None))
    fp4 = bool(getattr(model_config, "fp4", None))
    issues = probe_cudnn_gdn_stack_issues()
    if fp8 or fp4:
        te_version = get_distribution_version("transformer-engine")
        issues = issues + te_gdn_fp8_issues(transformer_engine_version=te_version, fp8=fp8, fp4=fp4)
    if issues:
        te_floor = TRANSFORMER_ENGINE_GDN_FP8_VERSION if fp8 or fp4 else TRANSFORMER_ENGINE_GDN_MIN_VERSION
        warn_rank_0(
            "model.gdn_kernel_backend='transformer_engine' cannot run GatedDeltaNet on cuDNN's fast kernels in this "
            f"environment: {'; '.join(issues)}. Install transformer-engine>={_version_str(te_floor)}, "
            f"nvidia-cudnn-frontend>={_version_str(CUDNN_FRONTEND_GDN_NAN_FREE_VERSION)} and "
            f"nvidia-cutlass-dsl[cu13]>={_version_str(CUTLASS_DSL_GDN_MIN_VERSION)}, "
            "or set model.gdn_kernel_backend=fla."
        )
