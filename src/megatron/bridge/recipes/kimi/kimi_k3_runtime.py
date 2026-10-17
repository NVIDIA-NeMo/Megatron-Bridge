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
"""Apply recorded FLA forward configurations without changing kernel arithmetic."""

import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import torch


def config_dict(config) -> dict:
    """Serialize the fields used by the installed Triton configuration API."""
    return {
        "kwargs": config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
        "maxnreg": config.maxnreg,
    }


def install_controls(output: Path, rank: int) -> Callable[[], dict]:
    """Constrain forward tuning and return a verifier of configurations actually used."""
    import triton
    from fla.modules import fused_norm_gate
    from fla.ops.utils.cache import FLA_CACHE_MODE, FlaCacheMode
    from fla.utils import autotune_cache_kwargs

    assert FLA_CACHE_MODE is FlaCacheMode.ALWAYS
    records = {
        "fla.modules.l2norm.l2norm_fwd_kernel/0": {
            "kwargs": {"BT": 8},
            "num_warps": 8,
            "num_ctas": 1,
            "num_stages": 3,
            "maxnreg": None,
        },
        "fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64/1": {
            "kwargs": {"BV": 32},
            "num_warps": 2,
            "num_ctas": 1,
            "num_stages": 3,
            "maxnreg": None,
        },
        "fla.ops.gla.chunk.chunk_gla_fwd_kernel_o/1": {
            "kwargs": {"BK": 64, "BV": 64},
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 2,
            "maxnreg": None,
        },
        "fla.ops.kda.wy_fast.recompute_w_u_fwd_kda_kernel/1": {
            "kwargs": {},
            "num_warps": 8,
            "num_ctas": 1,
            "num_stages": 4,
            "maxnreg": None,
        },
        "fla.ops.kda.chunk_intra.chunk_kda_fwd_kernel_inter_solve_fused/1": {
            "kwargs": {"BK": 64},
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 3,
            "maxnreg": None,
        },
        "fla.ops.kda.chunk_intra.chunk_kda_fwd_kernel_intra_sub_chunk/1": {
            "kwargs": {},
            "num_warps": 1,
            "num_ctas": 1,
            "num_stages": 3,
            "maxnreg": None,
        },
        "fla.ops.kda.gate.kda_gate_chunk_cumsum_vector_kernel/1": {
            "kwargs": {"BS": 32},
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 3,
            "maxnreg": None,
        },
    }
    config_dir = output / f"configs-rank-{rank}"
    config_dir.mkdir(exist_ok=True)
    os.environ["FLA_CONFIG_DIR"] = str(config_dir)
    objects = {}
    for path, config in records.items():
        name, depth = path.split("/")
        module, attr = name.rsplit(".", 1)
        obj = getattr(importlib.import_module(module), attr)
        for _ in range(int(depth)):
            obj = obj.fn
        assert type(obj).__name__ == "CachedAutotuner"
        (config_dir / f"{obj.kernel_name}.json").write_text(
            json.dumps({"kernel_name": obj.kernel_name, "default_config": config})
        )
        objects[path] = obj
    norm_config = {"kwargs": {"BT": 16}, "num_warps": 4, "num_stages": 3, "num_ctas": 1, "maxnreg": None}
    original = fused_norm_gate.layer_norm_gated_fwd_kernel.fn
    assert type(original).__name__ == "Autotuner"
    assert norm_config in [config_dict(config) for config in original.configs]
    keys = ["D", "NB", "IS_RMS_NORM", "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_WEIGHT"]
    # Keep the original JIT body and outer pointer-presence heuristics.
    tuner = triton.autotune(configs=[triton.Config(**norm_config)], key=keys, **autotune_cache_kwargs)(original.fn)
    fused_norm_gate.layer_norm_gated_fwd_kernel.fn = tuner

    def verify() -> dict:
        actual = {}
        for path, obj in objects.items():
            selected = config_dict(obj.best_config)
            assert selected == records[path], (path, selected)
            actual[path] = selected
        actual["gated_norm"] = config_dict(tuner.best_config)
        assert actual["gated_norm"] == norm_config
        return actual

    return verify


def configure_kimi_k3_kernels(models: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """Install early-recipe forward controls before the first training forward."""
    output = Path(tempfile.gettempdir()) / "kimi-k3-kernels"
    output.mkdir(parents=True, exist_ok=True)
    verify = install_controls(output, torch.distributed.get_rank())

    def verify_forward(module, inputs, result):
        verify()

    for model in models:
        model.register_forward_hook(verify_forward)
    return models
