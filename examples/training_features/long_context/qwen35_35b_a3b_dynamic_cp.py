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

"""Dynamic context parallelism (DCP v2) with global-batch packing on mock variable-length data.

This is a real launch script (unlike ``dynamic_context_parallel.py``, the packing demo next to it).
It builds the ``qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config`` recipe, applies optional
YAML / dotted CLI overrides, and calls ``pretrain``. It runs on the Megatron-Core ``main`` pin this
repository ships; no submodule switch is needed.

``--tiny-model`` is the supported way to run it today. It shrinks the recipe to a Qwen3.5-shaped
attention + MoE model with full-attention layers only and replaces the dataset with a generated
variable-length SFT file, so it needs no download and no checkpoint. The full recipe has Gated
DeltaNet layers, whose context-parallel exchange Megatron-Core ``main`` sizes from the static CP
group, and a multi-token-prediction layer that hangs under dynamic CP, so dynamic CP rejects it at
validation until Megatron-Core supports runtime groups for both.

The tiny model runs ``world_size / 2`` data-parallel replicas of CP2 with EP2. Dynamic CP schedules
each packed bin onto a group of 1, 2, ... ranks drawn from the ``dp x cp`` pool, so a bin that fits
one rank's ``max_seqlen_per_dp_cp_rank`` pays no context-parallel communication.

Dynamic CP on two GPUs::

    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py --tiny-model

Static-CP baseline on the same packing path (``dp_balanced`` at CP2), for throughput comparisons::

    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py --tiny-model --static-cp

For a steadier comparison add ``train.train_iters=40 logger.log_interval=1`` and drop the first
iterations, which pay one-time compilation and communicator setup. Dotted overrides use the
``ConfigContainer`` field paths, e.g. ``train.global_batch_size=128`` or
``model.max_seqlen_per_dp_cp_rank=4096``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.recipes.qwen.gb200.qwen35 import qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe, get_world_size_safe


logger = logging.getLogger(__name__)

_TINY_NUM_LAYERS = 4
_TINY_CP_SIZE = 2
_TINY_EP_SIZE = 2
_TINY_SEQ_LENGTH = 8192
_TINY_MAX_SEQLEN_PER_RANK = 4096
_TINY_SAMPLES_PER_GPU = 8
_MOCK_NUM_ROWS = 2048
_MOCK_SEED = 1234


def parse_cli_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the script's own flags and pass the rest through as config overrides."""
    parser = argparse.ArgumentParser(
        description="Qwen3.5 recipe with online sequence packing and dynamic context parallelism.",
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
        help="Shrink the recipe to a Qwen3.5-shaped attention + MoE smoke-test size (4 layers, 8 experts, 8k cap) "
        "on generated variable-length data. Runs on any even GPU count with CP2 per replica.",
    )
    parser.add_argument(
        "--static-cp",
        action="store_true",
        help="Run the static-CP baseline on the same packing path (dp_balanced scheduler, dynamic CP off).",
    )
    return parser.parse_known_args()


def write_tiny_sft_dataset(root: Path, *, num_rows: int, seed: int) -> Path:
    """Write a prompt/completion JSONL with lognormal answer lengths for the mock-data runs.

    Most answers fit within one rank's 4K-token share and some need a larger CP group, so dynamic CP
    has both kinds of bins to schedule. Every rank writes the same seeded file and renames it into
    place, so concurrent writers are harmless.

    Args:
        root: Directory that receives ``training.jsonl``.
        num_rows: Number of prompt/completion rows to generate.
        seed: Seed of the length and token generator.

    Returns:
        ``root``, which holds ``training.jsonl``.
    """
    train_file = root / "training.jsonl"
    if train_file.exists():
        return root
    rng = random.Random(seed)
    lines = []
    for index in range(num_rows):
        num_words = min(6000, max(16, int(rng.lognormvariate(6.0, 1.0))))
        answer = " ".join(f"w{rng.randrange(1000)}" for _ in range(num_words))
        lines.append(json.dumps({"input": f"Question {index}: repeat the words.", "output": answer}) + "\n")
    root.mkdir(parents=True, exist_ok=True)
    partial = root / f"training.jsonl.{os.getpid()}.tmp"
    partial.write_text("".join(lines), encoding="utf-8")
    os.replace(partial, train_file)
    return root


def apply_tiny_model_overrides(cfg: ConfigContainer, *, world_size: int) -> None:
    """Shrink the recipe for a mock-data run while keeping the global-batch packing path.

    The model keeps Qwen3.5's attention and MoE shape (gated attention, shared expert) at a fraction
    of the size, with full-attention layers only: Megatron-Core ``main`` supports dynamic CP for
    attention and MoE layers but not yet for Gated DeltaNet or multi-token prediction, so the recipe's
    MTP layer is dropped too. The CoderForge dataset is replaced by a generated prompt/completion file
    with lognormal lengths (see :func:`write_tiny_sft_dataset`), so the run needs no download and no
    checkpoint.

    Args:
        cfg: The recipe configuration to shrink in place.
        world_size: Total number of ranks; must be a multiple of the per-replica CP size.

    Raises:
        ValueError: If ``world_size`` cannot hold whole CP2 replicas.
    """
    if world_size < _TINY_CP_SIZE or world_size % _TINY_CP_SIZE:
        raise ValueError(
            f"--tiny-model runs replicas of CP{_TINY_CP_SIZE}; the world size must be a positive multiple of "
            f"{_TINY_CP_SIZE}, got {world_size}."
        )
    cfg.model.num_layers = _TINY_NUM_LAYERS
    # In linear_attention_freq a 1 marks a Gated DeltaNet layer and a 0 a full-attention layer.
    cfg.model.linear_attention_freq = [0] * _TINY_NUM_LAYERS
    # Multi-token prediction hangs under dynamic CP on the pinned Megatron-Core, so the recipe's MTP layer is dropped.
    cfg.model.mtp_num_layers = None
    cfg.model.hidden_size = 512
    cfg.model.ffn_hidden_size = 1024
    cfg.model.num_attention_heads = 8
    cfg.model.num_query_groups = 2
    # 128-wide heads keep the cuDNN fused-attention backend available for THD bins.
    cfg.model.kv_channels = 128
    cfg.model.num_moe_experts = 8
    cfg.model.moe_router_topk = 2
    cfg.model.moe_ffn_hidden_size = 256
    cfg.model.moe_shared_expert_intermediate_size = 256
    cfg.model.expert_model_parallel_size = _TINY_EP_SIZE
    cfg.model.context_parallel_size = _TINY_CP_SIZE
    cfg.model.seq_length = _TINY_SEQ_LENGTH
    cfg.model.max_seqlen_per_dp_cp_rank = _TINY_MAX_SEQLEN_PER_RANK
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    mock_root = Path(tempfile.gettempdir()) / f"qwen35_dynamic_cp_tiny_sft_{_MOCK_NUM_ROWS}_{_MOCK_SEED}"
    cfg.dataset = GPTSFTDatasetConfig(
        dataset_root=str(write_tiny_sft_dataset(mock_root, num_rows=_MOCK_NUM_ROWS, seed=_MOCK_SEED)),
        seq_length=_TINY_SEQ_LENGTH,
        # A cyclic loader starts a new epoch instead of ending the run when train_iters * global_batch_size outgrows the
        # generated rows.
        dataloader_type="cyclic",
        num_workers=2,
        do_validation=False,
        do_test=False,
        enable_global_batch_packing=True,
    )
    cfg.train.global_batch_size = _TINY_SAMPLES_PER_GPU * world_size
    cfg.train.train_iters = 5
    cfg.validation.eval_iters = 0
    # Mock runs neither write nor resume checkpoints, so repeated and dynamic-versus-static runs always start fresh.
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None


def main() -> None:
    """Build the recipe, apply overrides, and launch training."""
    args, cli_overrides = parse_cli_args()
    cfg: ConfigContainer = qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config()
    if args.tiny_model:
        apply_tiny_model_overrides(cfg, world_size=get_world_size_safe())
    if args.static_cp:
        # One static CP group per replica on the same packing path: the baseline dynamic CP is measured against.
        cfg.model.dynamic_context_parallel = False
        cfg.model.sequence_packing_scheduler = "dp_balanced"

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
