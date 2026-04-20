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
"""Functional test for OmniModal heterogeneous parallel training.

Exercises pretrain_omni_modal -> setup_omni_modal -> train_omni_modal on 2 GPUs with
synthetic data. Requires torchrun with --nproc_per_node=2.

Run:
    torchrun --nproc_per_node=2 -m pytest -v -s -x \
        tests/functional_tests/test_groups/training/test_pretrain_omni_modal.py
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.data.omni_modal.mock_provider import MockOmniModalProvider
from megatron.bridge.models.omni_modal.omni_modal_config import ModuleParallelismConfig, OmniModalParallelismConfig
from megatron.bridge.models.omni_modal.omni_modal_provider import OmniModalProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.training.omni_modal_step import forward_step as omni_modal_forward_step
from megatron.bridge.training.pretrain_omni_modal import pretrain_omni_modal
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from tests.functional_tests.utils import initialize_distributed


# ── Constants ────────────────────────────────────────────────────────────────

_ENCODER_SEQ_LEN = 197  # (224/16)^2 = 196 patches + 1 class token
_SPECIAL_TOKEN_ID = 32000
_VOCAB_SIZE = 50304
_SEQ_LENGTH = 256
_IMG_SIZE = 224
_PATCH_DIM = 16
_TRAIN_ITERS = 5


# ── Model helpers ────────────────────────────────────────────────────────────


def _make_vision_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
    )
    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    return cfg


def _make_language_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        cross_entropy_loss_fusion=True,
    )


def _build_model_specs():
    """Return (language_model_spec, modality_submodules_spec, special_token_ids)."""
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": _make_vision_config(),
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "patch_dim": _PATCH_DIM,
            "img_h": _IMG_SIZE,
            "img_w": _IMG_SIZE,
        },
    )
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={"encoders": {"clip": vision_encoder}},
    )
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": _make_language_config(),
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": _VOCAB_SIZE,
            "max_sequence_length": _SEQ_LENGTH,
        },
    )
    return language_model_spec, {"vision": vision_submodule_spec}, {"vision": _SPECIAL_TOKEN_ID}


# ── Data helpers ─────────────────────────────────────────────────────────────


class _CLIPImageProcessor:
    """Minimal image processor that produces pixel_values in the shape CLIP ViT expects.

    Avoids depending on the openai/clip-vit-base-patch16 HF processor which may
    not be available in all CI environments.
    """

    def __call__(self, image, return_tensors="pt"):
        # CLIP ViT expects [3, img_h, img_w] normalized float tensors.
        import numpy as np

        arr = np.array(image, dtype=np.float32) / 255.0  # [H, W, 3]
        arr = arr.transpose(2, 0, 1)  # [3, H, W]
        t = torch.tensor(arr)
        if return_tensors == "pt":
            t = t.unsqueeze(0)  # [1, 3, H, W] — batch dim removed by OmniModalDataset
        return {"pixel_values": t}


def _build_mock_data_provider() -> MockOmniModalProvider:
    provider = MockOmniModalProvider(
        seq_length=_SEQ_LENGTH,
        processor_paths={},
        tokenizer_path="gpt2",
        special_token_ids={"vision": _SPECIAL_TOKEN_ID},
        encoder_seq_lengths={"vision": _ENCODER_SEQ_LEN},
        modality_configs={"vision": {"type": "image", "width": _IMG_SIZE, "height": _IMG_SIZE}},
    )
    provider.drop_last = True
    # Inject our minimal CLIP-compatible processor so OmniModalDataset uses it.
    object.__setattr__(provider, "_processors", {"vision": _CLIPImageProcessor()})
    return provider


def _wrap_iter(loader_iter):
    """Adapt data-loader batches for the OmniModal model.

    Remaps modality_inputs["vision"]["pixel_values"] to
    modality_inputs["vision"]["clip"]["x"] for CLIPViTModel.
    """
    for batch in loader_iter:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda(non_blocking=True)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v.cuda(non_blocking=True)
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, torch.Tensor):
                                value[k][kk] = vv.cuda(non_blocking=True)

        mi = batch.get("modality_inputs")
        if mi and "vision" in mi:
            pv = mi["vision"].get("pixel_values")
            if pv is not None:
                mi["vision"] = {"clip": {"x": pv.to(torch.bfloat16)}}

        if "loss_mask" not in batch or batch["loss_mask"] is None:
            batch["loss_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.float)

        batch["attention_mask"] = None
        yield batch


def _build_data_iterators(cfg, omni_modal_infra):
    """Build data iterators compatible with pretrain_omni_modal's build_data_iterators_fn."""
    from megatron.bridge.data.omni_modal.loaders import build_omni_modal_data_loaders
    from megatron.bridge.training.state import TrainState

    train_state = TrainState()
    train_samples = cfg.train.train_iters * cfg.train.global_batch_size

    train_loader, _, _ = build_omni_modal_data_loaders(
        cfg=cfg,
        train_state=train_state,
        omni_modal_provider=cfg.dataset,
        train_samples=max(train_samples, 100),
        valid_samples=0,
        test_samples=0,
    )

    train_iter = _wrap_iter(train_loader) if train_loader is not None else None
    return train_iter, None


# ── Config builder ───────────────────────────────────────────────────────────


def _build_config(
    parallelism_config: OmniModalParallelismConfig,
    train_iters: int = _TRAIN_ITERS,
) -> ConfigContainer:
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()

    omni_modal_provider = OmniModalProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        omni_modal_parallelism_config=parallelism_config,
        topology={"vision": ["language"], "language": []},
    )
    if not hasattr(omni_modal_provider, "num_moe_experts"):
        omni_modal_provider.num_moe_experts = None

    train_cfg = TrainingConfig(
        micro_batch_size=1,
        global_batch_size=1,
        train_iters=train_iters,
    )
    train_cfg.num_microbatches = 1

    opt_config = OptimizerConfig(
        bf16=True,
        use_distributed_optimizer=True,
        lr=1e-4,
        min_lr=0.0,
    )

    return ConfigContainer(
        train=train_cfg,
        model=omni_modal_provider,
        optimizer=opt_config,
        scheduler=SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0),
        dataset=_build_mock_data_provider(),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(),
    )


# ── Test class ───────────────────────────────────────────────────────────────


class TestOmniModalTraining:
    """Functional tests for OmniModal heterogeneous parallel training.

    Requires 2 GPUs. Run with:
        torchrun --nproc_per_node=2 -m pytest -v -s -x \\
            tests/functional_tests/test_groups/training/test_pretrain_omni_modal.py
    """

    @pytest.mark.run_only_on("GPU")
    def test_omni_modal_tp1_both(self):
        """Smoke test: OmniModal training with TP=1 for both LLM and vision.

        LLM on rank 0 (TP=1, DP=1), vision on rank 1 (TP=1, DP=1).
        Trains for 5 iterations with synthetic data and verifies completion.
        """
        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"OmniModal test requires exactly 2 GPUs, got {world_size}")

        # Monkey-patch: report_theoretical_memory crashes on OmniModal models
        # because cfg.model is OmniModalProvider (no kv_channels).
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        par_cfg = OmniModalParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=1,
                ),
            },
        )

        cfg = _build_config(par_cfg)

        pretrain_omni_modal(
            cfg=cfg,
            forward_step_func=omni_modal_forward_step,
            build_data_iterators_fn=_build_data_iterators,
        )
