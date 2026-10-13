# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Apply recorded FLA forward configurations without changing kernel arithmetic."""

import importlib
import json
import os
from pathlib import Path
from typing import Callable


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
    records = json.loads(Path(__file__).with_name("kda-configs.json").read_text())
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
    norm_config = json.loads(Path(__file__).with_name("norm-config.json").read_text())
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
