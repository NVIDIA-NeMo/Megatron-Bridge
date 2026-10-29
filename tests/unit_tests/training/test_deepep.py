# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from scripts.performance.utils.utils import finalize_config_overrides

from megatron.bridge.training.flex_dispatcher_backend import (
    apply_flex_dispatcher_backend,
    validate_flex_dispatcher_backend,
)


pytestmark = pytest.mark.unit


def _model_config(**overrides):
    fields = dict(
        num_moe_experts=8,
        moe_token_dispatcher_type="alltoall",
        moe_flex_dispatcher_backend=None,
        moe_shared_expert_overlap=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("backend", ["deepep", "hybridep", "ncclep"])
@pytest.mark.parametrize("cuda_available", [False, True])
def test_apply_preserves_requested_backend_without_cuda_probe(backend, cuda_available):
    config = _model_config(moe_token_dispatcher_type="allgather")
    with (
        patch("torch.cuda.is_available", return_value=cuda_available) as available,
        patch("torch.cuda.get_device_properties", side_effect=AssertionError("unexpected CUDA probe")) as probe,
    ):
        apply_flex_dispatcher_backend(config, backend)

    assert config.moe_token_dispatcher_type == "flex"
    assert config.moe_flex_dispatcher_backend == backend
    assert config.moe_shared_expert_overlap is False
    available.assert_not_called()
    probe.assert_not_called()


@pytest.mark.parametrize("dispatcher", ["flex", "alltoall", "allgather"])
def test_none_explicitly_selects_alltoall_without_cuda_probe(dispatcher):
    config = _model_config(moe_token_dispatcher_type=dispatcher, moe_flex_dispatcher_backend="deepep")
    with patch("torch.cuda.get_device_properties") as probe:
        apply_flex_dispatcher_backend(config, None)
        validate_flex_dispatcher_backend(config)

    assert config.moe_token_dispatcher_type == "alltoall"
    assert config.moe_flex_dispatcher_backend is None
    assert config.moe_shared_expert_overlap is True
    probe.assert_not_called()


def test_apply_rejects_unknown_backend_without_mutation():
    config = _model_config()
    before = vars(config).copy()
    with patch("torch.cuda.get_device_properties") as probe:
        with pytest.raises(ValueError, match="Unknown flex dispatcher backend"):
            apply_flex_dispatcher_backend(config, "typo")
    assert vars(config) == before
    probe.assert_not_called()


@pytest.mark.parametrize("experts", [None, 0])
def test_non_moe_model_is_unchanged(experts):
    config = _model_config(num_moe_experts=experts)
    before = vars(config).copy()
    with (
        patch("megatron.bridge.training.flex_dispatcher_backend.get_rank_safe", return_value=0),
        patch("megatron.bridge.training.flex_dispatcher_backend.logger") as logger,
        patch("torch.cuda.get_device_properties") as probe,
    ):
        apply_flex_dispatcher_backend(config, "deepep")
    assert vars(config) == before
    logger.warning.assert_called_once()
    probe.assert_not_called()


@pytest.mark.parametrize(
    ("backend", "major", "device_name"),
    [
        ("deepep", 8, "NVIDIA A100"),
        ("deepep", 9, "NVIDIA H100"),
        ("deepep", 10, "NVIDIA B200"),
        ("deepep", 10, "NVIDIA B300 SXM6 AC"),
        ("hybridep", 8, "NVIDIA A100"),
        ("hybridep", 9, "NVIDIA H100"),
        ("hybridep", 10, "NVIDIA GB200"),
        ("hybridep", 10, "NVIDIA GB300"),
        ("ncclep", 9, "NVIDIA H100"),
        ("ncclep", 10, "NVIDIA GB200"),
    ],
)
def test_supported_backend_validation_does_not_mutate_config(backend, major, device_name):
    config = _model_config()
    apply_flex_dispatcher_backend(config, backend)
    before = vars(config).copy()
    with patch(
        "torch.cuda.get_device_properties", return_value=SimpleNamespace(major=major, name=device_name)
    ) as probe:
        validate_flex_dispatcher_backend(config)
    assert vars(config) == before
    probe.assert_called_once_with(0)


@pytest.mark.parametrize(
    ("backend", "major", "device_name"),
    [
        ("deepep", 6, "NVIDIA P100"),
        ("deepep", 7, "NVIDIA V100"),
        ("deepep", 10, "NVIDIA GB200"),
        ("deepep", 10, "NVIDIA GB300"),
        ("deepep", 11, "NVIDIA Future GPU"),
        ("hybridep", 11, "NVIDIA Future GPU"),
        ("ncclep", 8, "NVIDIA A100"),
    ],
)
@pytest.mark.parametrize("dispatcher", ["alltoall", "flex", "allgather"])
def test_unsupported_request_survives_finalization_and_fails_validation(backend, major, device_name, dispatcher):
    """Regression for #5207: finalization must never hide an unsupported request."""
    config = SimpleNamespace(
        model=_model_config(moe_token_dispatcher_type=dispatcher, moe_flex_dispatcher_backend=backend),
        ddp=SimpleNamespace(nccl_ub=False, fsdp_manual_registration=False),
    )
    with patch(
        "torch.cuda.get_device_properties", return_value=SimpleNamespace(major=major, name=device_name)
    ) as probe:
        apply_flex_dispatcher_backend(config.model, backend)
        finalize_config_overrides(config)
        probe.assert_not_called()
        before = vars(config.model).copy()
        with pytest.raises(ValueError, match="explicitly set moe_token_dispatcher_type='alltoall'"):
            validate_flex_dispatcher_backend(config.model)

    assert vars(config.model) == before
    assert config.model.moe_token_dispatcher_type == "flex"
    assert config.model.moe_flex_dispatcher_backend == backend
    probe.assert_called_once_with(0)


@pytest.mark.parametrize("backend", [None, "typo"])
def test_invalid_flex_config_fails_without_mutation_or_hardware_probe(backend):
    config = _model_config(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend=backend)
    before = vars(config).copy()
    with patch("torch.cuda.get_device_properties") as probe:
        with pytest.raises(ValueError, match="requires moe_flex_dispatcher_backend"):
            validate_flex_dispatcher_backend(config)
    assert vars(config) == before
    probe.assert_not_called()


@pytest.mark.parametrize("dispatcher", ["alltoall", "allgather"])
def test_non_flex_dispatcher_does_not_validate_unused_backend(dispatcher):
    config = _model_config(moe_token_dispatcher_type=dispatcher, moe_flex_dispatcher_backend="deepep")
    before = vars(config).copy()
    with patch("torch.cuda.get_device_properties") as probe:
        validate_flex_dispatcher_backend(config)
    assert vars(config) == before
    probe.assert_not_called()


def test_validation_preserves_cuda_probe_error():
    config = _model_config(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="deepep")
    before = vars(config).copy()
    with patch("torch.cuda.get_device_properties", side_effect=RuntimeError("CUDA initialization failed")):
        with pytest.raises(RuntimeError, match="CUDA initialization failed"):
            validate_flex_dispatcher_backend(config)
    assert vars(config) == before


def test_child_recipe_can_replace_parent_backend_before_validation():
    config = _model_config()
    with patch(
        "torch.cuda.get_device_properties", return_value=SimpleNamespace(major=10, name="NVIDIA GB200")
    ) as probe:
        apply_flex_dispatcher_backend(config, "deepep")
        apply_flex_dispatcher_backend(config, "hybridep")
        probe.assert_not_called()
        validate_flex_dispatcher_backend(config)
    assert config.moe_flex_dispatcher_backend == "hybridep"
    probe.assert_called_once_with(0)


def test_explicit_none_override_still_disables_flex_during_finalization():
    config = SimpleNamespace(model=_model_config(), ddp=SimpleNamespace(nccl_ub=False))
    apply_flex_dispatcher_backend(config.model, "deepep")
    config.model.moe_flex_dispatcher_backend = None
    finalize_config_overrides(config)
    with patch("torch.cuda.get_device_properties") as probe:
        validate_flex_dispatcher_backend(config.model)
    assert config.model.moe_token_dispatcher_type == "alltoall"
    assert config.model.moe_flex_dispatcher_backend is None
    probe.assert_not_called()
