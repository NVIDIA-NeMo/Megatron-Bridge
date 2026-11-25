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
"""Functional tests for Qwen3.5-VL Multi-Token Prediction (MTP) in MegatronMIMO.

A tiny Qwen3.5-VL keeps its language model, including one MTP layer, on rank 0 and
its vision encoder on rank 1. The tests train it on synthetic batches and round-trip
its MTP weights through MegatronMIMO conversion. Requires exactly 2 GPUs.

Run:
    torchrun --nproc_per_node=2 -m pytest -v -s -x \
        tests/functional_tests/test_groups/megatron_mimo/test_qwen35_vl_mtp.py
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.core.utils import unwrap_model
from safetensors.torch import load_file

from megatron.bridge import AutoBridge
from megatron.bridge.data.megatron_mimo.mock_provider import MockMegatronMIMOProvider
from megatron.bridge.models.megatron_mimo.conversion.orchestrator import MegatronMIMOBridge
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import _TRANSFORMERS_HAS_QWEN3_5
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.training.megatron_mimo_parallel_utils import unwrap_megatron_mimo_model
from megatron.bridge.training.megatron_mimo_step import forward_step as megatron_mimo_forward_step
from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from tests.functional_tests.test_groups.megatron_mimo.qwen35_vl_toy import (
    IMAGE_TOKEN_ID,
    PATCH_DIM,
    SPATIAL_MERGE_SIZE,
    VIDEO_TOKEN_ID,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
    hf_toy_config,
    save_hf_toy_checkpoint_with_mtp,
)
from tests.functional_tests.utils import broadcast_path, initialize_distributed


pytestmark = pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5, reason="transformers does not have Qwen3.5 support")


# ── Constants ────────────────────────────────────────────────────────────────

_TEXT_VOCAB_SIZE = 2000
_SEQ_LENGTH = 64
_IMAGE_GRID_THW = (1, 4, 4)  # 16 patches -> 4 image tokens after the 2x2 merge
_TRAIN_ITERS = 3


def _parallelism_config() -> MegatronMIMOParallelismConfig:
    """Non-colocated layout: the language model on rank 0, the vision encoder on rank 1."""
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=0,
            ),
            "images": ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=1,
            ),
        }
    )


def _require_two_ranks() -> None:
    initialize_distributed()
    if dist.get_world_size() != 2:
        pytest.skip(f"Qwen3.5-VL MegatronMIMO MTP tests require exactly 2 GPUs, got {dist.get_world_size()}")


# ── Training ─────────────────────────────────────────────────────────────────


def _synthetic_batch(index: int) -> dict:
    """Build one Qwen-format MIMO micro-batch with a single image."""
    generator = torch.Generator().manual_seed(1234 + index)
    num_patches = math.prod(_IMAGE_GRID_THW)
    num_image_tokens = num_patches // SPATIAL_MERGE_SIZE**2
    prompt = torch.randint(0, _TEXT_VOCAB_SIZE, (8,), generator=generator)
    response_length = _SEQ_LENGTH - prompt.numel() - num_image_tokens - 2
    response = torch.randint(0, _TEXT_VOCAB_SIZE, (response_length,), generator=generator)
    input_ids = torch.cat(
        [
            prompt,
            torch.tensor([VISION_START_TOKEN_ID]),
            torch.full((num_image_tokens,), IMAGE_TOKEN_ID),
            torch.tensor([VISION_END_TOKEN_ID]),
            response,
        ]
    ).unsqueeze(0)
    image_grid_thw = torch.tensor([_IMAGE_GRID_THW])
    position_ids, _ = get_rope_index(
        SPATIAL_MERGE_SIZE,
        IMAGE_TOKEN_ID,
        VIDEO_TOKEN_ID,
        VISION_START_TOKEN_ID,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=torch.ones_like(input_ids),
    )
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = 0
    # Supervise the response only, as in SFT: positions whose next token is response text.
    loss_mask = torch.zeros(input_ids.shape, dtype=torch.float)
    loss_mask[:, _SEQ_LENGTH - response_length - 1 : -1] = 1.0
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
        "labels": labels,
        "loss_mask": loss_mask,
        "modality_inputs": {
            "images": {
                "qwen_visual": {
                    "hidden_states": torch.randn(num_patches, PATCH_DIM, generator=generator),
                    "grid_thw": image_grid_thw,
                }
            }
        },
    }


def _build_data_iterators(cfg, megatron_mimo_infra, *, train_state=None):
    """Every rank reads the same synthetic micro-batches in the same order."""
    return (_synthetic_batch(index) for index in itertools.count()), None


def _build_config() -> ConfigContainer:
    standard_provider = AutoBridge.from_hf_config(hf_toy_config()).to_megatron_provider(load_weights=False)
    standard_provider.seq_length = _SEQ_LENGTH
    standard_provider.language_max_sequence_length = _SEQ_LENGTH
    assert standard_provider.mtp_num_layers == 1, "The toy config must give the language model one MTP layer"

    megatron_mimo_provider = MegatronMIMOProvider.from_standard_provider(
        standard_provider=standard_provider,
        megatron_mimo_parallelism_config=_parallelism_config(),
    )
    if not hasattr(megatron_mimo_provider, "num_moe_experts"):
        megatron_mimo_provider.num_moe_experts = None

    return ConfigContainer(
        # Two micro-batches per step exercise the per-microbatch MTP loss scaling.
        train=TrainingConfig(micro_batch_size=1, global_batch_size=2, train_iters=_TRAIN_ITERS),
        model=megatron_mimo_provider,
        optimizer=OptimizerConfig(bf16=True, use_distributed_optimizer=True, lr=1e-3, min_lr=0.0),
        scheduler=SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0),
        # The batches come from _build_data_iterators; this only carries the sequence length.
        dataset=MockMegatronMIMOProvider(seq_length=_SEQ_LENGTH),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(),
    )


@pytest.mark.run_only_on("GPU")
def test_qwen35_vl_megatron_mimo_trains_mtp(monkeypatch):
    """MTP is built on the language stage, its loss reaches every rank and its weights train."""
    _require_two_ranks()

    import megatron.bridge.training.train_megatron_mimo as train_megatron_mimo_module
    import megatron.bridge.training.utils.train_utils as train_utils

    # report_theoretical_memory expects a language provider, but cfg.model is a MegatronMIMOProvider.
    monkeypatch.setattr(train_utils, "report_theoretical_memory", lambda *args, **kwargs: None)

    loss_dicts = []
    language_models = []
    initial_mtp_params = {}
    train_step = train_megatron_mimo_module.train_step_megatron_mimo

    def recording_train_step(*args, **kwargs):
        mimo_model = unwrap_megatron_mimo_model(kwargs["model"])
        if mimo_model.language_model is not None and not language_models:
            language_model = unwrap_model(mimo_model.language_model)
            language_models.append(language_model)
            if language_model.mtp_process:
                initial_mtp_params.update(
                    {name: param.detach().clone() for name, param in language_model.mtp.named_parameters()}
                )
        result = train_step(*args, **kwargs)
        loss_dicts.append({key: value.item() for key, value in result[0].items()})
        return result

    monkeypatch.setattr(train_megatron_mimo_module, "train_step_megatron_mimo", recording_train_step)

    pretrain_megatron_mimo(
        cfg=_build_config(),
        forward_step_func=megatron_mimo_forward_step,
        build_data_iterators_fn=_build_data_iterators,
    )

    # The language last stage computes the losses; the logging rank (the image rank) receives them.
    assert len(loss_dicts) == _TRAIN_ITERS
    for loss_dict in loss_dicts:
        assert {"lm loss", "mtp_1 loss"} <= loss_dict.keys(), loss_dict
        assert math.isfinite(loss_dict["lm loss"]) and loss_dict["lm loss"] > 0
        assert math.isfinite(loss_dict["mtp_1 loss"]) and loss_dict["mtp_1 loss"] > 0

    if dist.get_rank() == 0:
        assert language_models, "rank 0 must own the language model"
        language_model = language_models[0]
        assert language_model.mtp_process is True
        assert initial_mtp_params
        changed = [
            name
            for name, param in language_model.mtp.named_parameters()
            if not torch.equal(param.detach(), initial_mtp_params[name])
        ]
        assert changed, "No MTP parameter changed after training"


# ── Conversion ───────────────────────────────────────────────────────────────


@pytest.mark.run_only_on("GPU")
def test_qwen35_vl_megatron_mimo_conversion_round_trips_mtp(tmp_path):
    """MegatronMIMO conversion imports the HF ``mtp.*`` weights and exports them unchanged."""
    _require_two_ranks()

    hf_path = Path(broadcast_path(tmp_path / "qwen35_vl_mtp_toy"))
    if dist.get_rank() == 0:
        save_hf_toy_checkpoint_with_mtp(hf_path)
    dist.barrier()

    bridge = MegatronMIMOBridge.from_hf_pretrained(hf_path, parallelism_config=_parallelism_config())
    (model,) = bridge.to_megatron_model(load_weights=True, wrap_with_ddp=False, data_parallel_random_init=False)
    exported = {name: weight for name, weight in bridge.export_hf_weights(model, cpu=True, show_progress=False)}

    if dist.get_rank() == 0:
        source = load_file(hf_path / "model.safetensors")
        mtp_names = sorted(name for name in source if name.startswith("mtp."))
        assert len(mtp_names) == 15, mtp_names
        assert sorted(name for name in exported if name.startswith("mtp.")) == mtp_names
        for name in mtp_names:
            torch.testing.assert_close(exported[name], source[name], rtol=0, atol=0, msg=name)
    dist.barrier()
