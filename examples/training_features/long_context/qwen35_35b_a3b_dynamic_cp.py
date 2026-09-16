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

"""Fine-tune Qwen3.5-35B-A3B on long variable-length data with global-batch packing and dynamic CP.

This is a real launch script (unlike ``dynamic_context_parallel.py``, the packing
demo next to it). It builds the ``qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config``
recipe (CoderForge SWE-Rebench trajectories, TP1 PP1 CP4 EP8, 32k cap), applies optional
YAML / dotted CLI overrides, and calls ``pretrain``.

Requires the Megatron-Core ``dev`` submodule pin::

    ./scripts/switch_mcore.sh dev && uv sync

Full recipe on eight GB200 GPUs::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py \\
        checkpoint.pretrained_checkpoint=/path/to/megatron/qwen3.5-35b-a3b \\
        train.train_iters=20 logger.log_interval=1

Static-CP baseline with the same bins (``dp_balanced`` at CP4)::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py \\
        model.dynamic_context_parallel=false model.sequence_packing_scheduler=dp_balanced

Smoke test with a tiny Qwen3.5-shaped model and synthetic lognormal lengths on two GPUs::

    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py --tiny-model

Dotted overrides use the ``ConfigContainer`` field paths, e.g. ``train.global_batch_size=128``
or ``model.max_seqlen_per_dp_cp_rank=4096``.
"""

from __future__ import annotations

import argparse
import json
import logging

from omegaconf import OmegaConf

from megatron.bridge.recipes.qwen.gb200.qwen35 import qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config
from megatron.bridge.training.config import ConfigContainer, MockGPTDatasetConfig
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger = logging.getLogger(__name__)


def parse_cli_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the script's own flags and pass the rest through as config overrides."""
    parser = argparse.ArgumentParser(
        description="Qwen3.5-35B-A3B with online sequence packing and dynamic context parallelism.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Optional YAML file with ConfigContainer overrides, merged before dotted CLI overrides.",
    )
    parser.add_argument(
        "--tiny-model",
        action="store_true",
        help="Shrink the model to a Qwen3.5-shaped smoke-test size (4 layers, 8 experts, 8k cap) that runs "
        "on two GPUs with CP2 and the cuDNN attention backend.",
    )
    return parser.parse_known_args()


def apply_tiny_model_overrides(cfg: ConfigContainer) -> None:
    """Shrink the recipe for a two-GPU smoke test while keeping the global-batch packing path.

    The model keeps Qwen3.5's shape (Gated DeltaNet 3:1 with gated attention, MoE) at a
    fraction of the size, and the CoderForge dataset is replaced by Megatron's mock
    variable-length dataset with a lognormal length distribution so the run needs no
    download and no checkpoint.
    """
    cfg.model.num_layers = 4
    # The Qwen3.5 bridge expresses the 3:1 GDN:attention layout as a per-layer list in
    # linear_attention_freq; keep the first four entries for the shrunken depth.
    if isinstance(cfg.model.linear_attention_freq, list):
        cfg.model.linear_attention_freq = list(cfg.model.linear_attention_freq[:4])
    cfg.model.hidden_size = 512
    cfg.model.ffn_hidden_size = 1024
    cfg.model.num_attention_heads = 8
    cfg.model.num_query_groups = 2
    # 128-wide heads keep the cuDNN fused-attention backend available for THD bins.
    cfg.model.kv_channels = 128
    cfg.model.linear_num_key_heads = 4
    cfg.model.linear_num_value_heads = 8
    cfg.model.linear_key_head_dim = 64
    cfg.model.linear_value_head_dim = 64
    cfg.model.num_moe_experts = 8
    cfg.model.moe_router_topk = 2
    cfg.model.moe_ffn_hidden_size = 256
    cfg.model.moe_shared_expert_intermediate_size = 256
    cfg.model.expert_model_parallel_size = 2
    cfg.model.context_parallel_size = 2
    cfg.model.seq_length = 8192
    cfg.model.max_seqlen_per_dp_cp_rank = 4096
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.dataset = MockGPTDatasetConfig(
        seq_length=8192,
        random_seed=1234,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        enable_global_batch_packing=True,
        fold_alignment_padding=False,
        varlen_mock_dataset_config_json=json.dumps(
            {
                "mode": "distribution",
                "type": "lognormal",
                "min_seq_len": 64,
                "max_seq_len": 8192,
                "mean_seq_len": 1024,
                "lognormal_sigma": 1.0,
            }
        ),
        dataloader_type="single",
        num_workers=2,
    )
    cfg.train.global_batch_size = 16
    cfg.train.train_iters = 5
    cfg.validation.eval_iters = 0
    cfg.checkpoint.pretrained_checkpoint = None


def main() -> None:
    """Build the recipe, apply overrides, and launch training."""
    args, cli_overrides = parse_cli_args()
    cfg: ConfigContainer = qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config()
    if args.tiny_model:
        apply_tiny_model_overrides(cfg)

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)
    if args.config_file:
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, OmegaConf.load(args.config_file))
    if cli_overrides:
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
    apply_overrides(cfg, OmegaConf.to_container(merged_omega_conf, resolve=True), excluded_fields)

    if get_rank_safe() == 0:
        logger.info(
            "Global-batch packing: scheduler=%s dynamic_cp=%s cp=%s max_seqlen_per_dp_cp_rank=%s seq_length=%s",
            cfg.model.sequence_packing_scheduler,
            cfg.model.dynamic_context_parallel,
            cfg.model.context_parallel_size,
            cfg.model.max_seqlen_per_dp_cp_rank,
            cfg.model.seq_length,
        )
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
