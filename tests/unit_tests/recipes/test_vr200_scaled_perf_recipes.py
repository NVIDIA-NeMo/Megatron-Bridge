"""Tests for scaled VR200 performance recipes."""

from collections.abc import Callable

import pytest

from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v3_pretrain_256gpu_vr200_fp8mx_config,
    deepseek_v3_pretrain_256gpu_vr200_nvfp4_config,
    deepseek_v3_pretrain_512gpu_vr200_fp8mx_config,
    deepseek_v3_pretrain_512gpu_vr200_nvfp4_config,
    deepseek_v3_pretrain_1024gpu_vr200_fp8mx_config,
    deepseek_v3_pretrain_1024gpu_vr200_nvfp4_config,
)
from megatron.bridge.perf_recipes.gpt_oss import (
    gpt_oss_120b_pretrain_64gpu_vr200_fp8mx_config,
    gpt_oss_120b_pretrain_128gpu_vr200_fp8mx_config,
    gpt_oss_120b_pretrain_256gpu_vr200_fp8mx_config,
    gpt_oss_120b_pretrain_512gpu_vr200_fp8mx_config,
)
from megatron.bridge.perf_recipes.nemotronh import (
    nemotron_3_super_pretrain_64gpu_vr200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_vr200_nvfp4_config,
    nemotron_3_super_pretrain_128gpu_vr200_fp8mx_config,
    nemotron_3_super_pretrain_128gpu_vr200_nvfp4_config,
    nemotron_3_super_pretrain_256gpu_vr200_fp8mx_config,
    nemotron_3_super_pretrain_256gpu_vr200_nvfp4_config,
    nemotron_3_super_pretrain_512gpu_vr200_fp8mx_config,
    nemotron_3_super_pretrain_512gpu_vr200_nvfp4_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    ("base_factory", "base_num_gpus", "scaled_factory", "num_gpus"),
    [
        (
            gpt_oss_120b_pretrain_64gpu_vr200_fp8mx_config,
            64,
            gpt_oss_120b_pretrain_128gpu_vr200_fp8mx_config,
            128,
        ),
        (
            gpt_oss_120b_pretrain_64gpu_vr200_fp8mx_config,
            64,
            gpt_oss_120b_pretrain_256gpu_vr200_fp8mx_config,
            256,
        ),
        (
            gpt_oss_120b_pretrain_64gpu_vr200_fp8mx_config,
            64,
            gpt_oss_120b_pretrain_512gpu_vr200_fp8mx_config,
            512,
        ),
        (
            nemotron_3_super_pretrain_64gpu_vr200_fp8mx_config,
            64,
            nemotron_3_super_pretrain_128gpu_vr200_fp8mx_config,
            128,
        ),
        (
            nemotron_3_super_pretrain_64gpu_vr200_fp8mx_config,
            64,
            nemotron_3_super_pretrain_256gpu_vr200_fp8mx_config,
            256,
        ),
        (
            nemotron_3_super_pretrain_64gpu_vr200_fp8mx_config,
            64,
            nemotron_3_super_pretrain_512gpu_vr200_fp8mx_config,
            512,
        ),
        (
            deepseek_v3_pretrain_256gpu_vr200_fp8mx_config,
            256,
            deepseek_v3_pretrain_512gpu_vr200_fp8mx_config,
            512,
        ),
        (
            deepseek_v3_pretrain_256gpu_vr200_fp8mx_config,
            256,
            deepseek_v3_pretrain_1024gpu_vr200_fp8mx_config,
            1024,
        ),
        (
            nemotron_3_super_pretrain_64gpu_vr200_nvfp4_config,
            64,
            nemotron_3_super_pretrain_128gpu_vr200_nvfp4_config,
            128,
        ),
        (
            nemotron_3_super_pretrain_64gpu_vr200_nvfp4_config,
            64,
            nemotron_3_super_pretrain_256gpu_vr200_nvfp4_config,
            256,
        ),
        (
            nemotron_3_super_pretrain_64gpu_vr200_nvfp4_config,
            64,
            nemotron_3_super_pretrain_512gpu_vr200_nvfp4_config,
            512,
        ),
        (
            deepseek_v3_pretrain_256gpu_vr200_nvfp4_config,
            256,
            deepseek_v3_pretrain_512gpu_vr200_nvfp4_config,
            512,
        ),
        (
            deepseek_v3_pretrain_256gpu_vr200_nvfp4_config,
            256,
            deepseek_v3_pretrain_1024gpu_vr200_nvfp4_config,
            1024,
        ),
    ],
)
def test_vr200_recipe_scales_global_batch_size(
    base_factory: Callable[[], ConfigContainer],
    base_num_gpus: int,
    scaled_factory: Callable[[], ConfigContainer],
    num_gpus: int,
) -> None:
    base_cfg = base_factory()
    scaled_cfg = scaled_factory()

    expected_global_batch_size = int(base_cfg.train.global_batch_size / base_num_gpus * num_gpus)
    assert scaled_cfg.train.global_batch_size == expected_global_batch_size
    assert scaled_cfg.train.micro_batch_size == base_cfg.train.micro_batch_size
    assert scaled_cfg.env_vars == base_cfg.env_vars
