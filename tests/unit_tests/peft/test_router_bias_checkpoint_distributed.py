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

"""One-GPU native-router PEFT checkpoint regression.

Run with uv run python -m torch.distributed.run --nproc_per_node=1 -m pytest
tests/unit_tests/peft/test_router_bias_checkpoint_distributed.py.
"""

import os

import pytest
import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from torch.distributed.checkpoint.api import CheckpointException

from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.utils import load_peft_adapter_checkpoint
from megatron.bridge.training.checkpointing import _generate_model_state_dict, apply_peft_adapter_filter_to_state_dict


class _RouterModel(MegatronModule):
    def __init__(self, config, groups):
        super().__init__(config)
        self.proj = nn.Linear(16, 16, bias=False).cuda()
        self.router = TopKRouter(config, pg_collection=groups).cuda()
        self.router.set_layer_number(1)

    def forward(self, inputs):
        return self.router(self.proj(inputs))


@pytest.mark.unit
@pytest.mark.gpu
def test_native_router_bias_survives_peft_checkpoint_and_merge(tmp_path):
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("requires a one-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_dist = not dist.is_initialized()
    owns_parallel = not parallel_state.model_parallel_is_initialized()
    if owns_dist:
        dist.init_process_group("nccl")
    if owns_parallel:
        parallel_state.initialize_model_parallel()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp", "tp_cp", "tp_dp_cp", "dp_cp"])
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            num_moe_experts=4,
            add_bias_linear=False,
            moe_router_topk=2,
            moe_router_score_function="sigmoid",
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=0.01,
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0.001,
        )
        base = _RouterModel(config, groups)
        base_state = {key: value.clone() for key, value in base.state_dict().items()}
        peft = LoRA(target_modules=["proj"], dim=2, alpha=2, dropout=0.0)
        model = peft(base, training=True)
        peft.set_params_to_save(model)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        inputs = torch.randn(1, 1, 16, device="cuda").expand(16, 1, 16).contiguous()
        probs, _ = model(inputs)
        (probs * torch.arange(4, device="cuda")).sum().backward()
        _update_router_expert_bias([model], config, tp_dp_cp_group=groups.tp_dp_cp)
        optimizer.step()
        expected_bias = model.router.expert_bias.clone()
        assert not model.router.weight.requires_grad
        assert not torch.equal(expected_bias, base_state["router.expert_bias"])
        assert torch.count_nonzero(model.proj.adapter.linear_out.weight) > 0

        checkpoint = tmp_path / "adapter"
        checkpoint.mkdir()
        saved = apply_peft_adapter_filter_to_state_dict(_generate_model_state_dict([model], {}), peft)
        assert "router.expert_bias" in saved["model"]
        dist_checkpointing.save(saved, str(checkpoint))

        for training in (True, False):
            restored_base = _RouterModel(config, groups)
            restored_base.load_state_dict(base_state)
            restored_peft = LoRA(target_modules=["proj"], dim=2, alpha=2, dropout=0.0)
            restored = restored_peft(restored_base, training=training)
            # Deliberately leave params_to_save empty, as the merge path does.
            assert not restored_peft.params_to_save
            load_peft_adapter_checkpoint(restored, checkpoint, restored_peft, fully_parallel_load=False)
            assert torch.equal(restored.router.expert_bias, expected_bias)
            for key, value in model.state_dict().items():
                assert torch.equal(restored.state_dict()[key], value), key

            # LoRALinear.weight uses the same native LoRAMerge helper as export.
            merged = _RouterModel(config, groups)
            merged.load_state_dict(base_state)
            with torch.no_grad():
                merged.proj.weight.copy_(restored.proj.weight)
                merged.router.expert_bias.copy_(restored.router.expert_bias)
                restored.eval()
                merged.eval()
                actual_probs, actual_routes = restored(inputs)
                merged_probs, merged_routes = merged(inputs)
            torch.testing.assert_close(actual_probs, merged_probs, rtol=1e-5, atol=1e-6)
            assert torch.equal(actual_routes, merged_routes)
            assert torch.equal(merged.router.expert_bias, expected_bias)

        # Older adapter checkpoints omitted this non-parameter training state.
        # Loading must fail rather than claim recovery with the base-model bias.
        legacy_checkpoint = tmp_path / "legacy-adapter"
        legacy_checkpoint.mkdir()
        legacy_state = {
            **saved,
            "model": {key: value for key, value in saved["model"].items() if key != "router.expert_bias"},
        }
        dist_checkpointing.save(legacy_state, str(legacy_checkpoint))
        restored.router.expert_bias.copy_(base_state["router.expert_bias"])
        with pytest.raises(CheckpointException, match="router.expert_bias"):
            load_peft_adapter_checkpoint(restored, legacy_checkpoint, restored_peft, fully_parallel_load=False)
        assert torch.equal(restored.router.expert_bias, base_state["router.expert_bias"])
    finally:
        if owns_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_dist and dist.is_initialized():
            dist.destroy_process_group()
