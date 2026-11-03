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

from types import SimpleNamespace

from megatron.training.determinism import DETERMINISM_ENV_VAR_DEFAULTS

from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides


def test_overrides_follow_megatron_core():
    """Only the aux-loss fusion is disabled; env is Megatron-Core's defaults plus the SSM pins."""
    cfg = SimpleNamespace(
        model=SimpleNamespace(moe_router_fusion=True, moe_router_aux_loss_fusion=None),
        env_vars={},
        comm_overlap=SimpleNamespace(tp_comm_overlap=True),
    )

    apply_determinism_overrides(cfg)

    assert cfg.model.deterministic_mode is True
    assert cfg.model.cross_entropy_loss_fusion is False
    assert cfg.model.moe_router_aux_loss_fusion is False
    assert cfg.model.moe_router_fusion is True
    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.env_vars == {
        **DETERMINISM_ENV_VAR_DEFAULTS,
        "MAMBA_DETERMINISTIC": "1",
        "CAUSAL_CONV1D_DETERMINISTIC": "1",
    }
