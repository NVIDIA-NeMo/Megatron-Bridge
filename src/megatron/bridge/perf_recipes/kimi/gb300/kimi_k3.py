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
"""Reduced-depth BF16 Kimi K3 benchmark built from the canonical HF configuration."""

from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.recipes.kimi.kimi_k3 import kimi_k3_pretrain_config
from megatron.bridge.training.config import ConfigContainer, MockGPTDatasetConfig
from megatron.bridge.training.tokenizers.config import TokenizerConfig


def kimi_k3_pretrain_64gpu_gb300_bf16_proxy_config() -> ConfigContainer:
    """Build a random-init, mock-data 24-block proxy with explicitly forced routing.

    Retain the first 24 attention blocks from the HF architecture and replace
    their feed-forward layers with experts. For a source with three KDA blocks
    then MLA, this produces the ``KEKEKE+E`` repeated six times used by #7066.
    Other HF architectures retain their own attention ordering. This is a BF16
    Adam setup; it does not reproduce #7066's MXFP8/MoK/Muon measurements or
    establish full-model throughput, memory fit, or convergence.
    """
    cfg = kimi_k3_pretrain_config()
    layers = cfg.model.hybrid_layer_pattern.replace("|", "")
    if len(layers) < 48:
        raise ValueError("The 24-block Kimi K3 proxy requires an HF model with at least 24 blocks.")
    cfg.model.hybrid_layer_pattern = "".join(layers[i] + "E" for i in range(0, 48, 2))
    cfg.model.num_layers = 48
    cfg.model.pipeline_block_counts = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_num_layers = 1
    cfg.mixed_precision = _perf_precision("bf16")
    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.grad_reduce_in_fp32 = True
    # The first attention residual has unused trainable query/norm parameters.
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.dist.enable_megatron_core_experimental = True
    cfg.dataset = MockGPTDatasetConfig(
        seq_length=cfg.model.seq_length,
        random_seed=20260901,
        split="100,0,0",
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        num_dataset_builder_threads=1,
        data_sharding=True,
        dataloader_type="single",
        create_attention_mask=False,
        skip_getting_attention_mask_from_dataset=True,
    )
    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="NullTokenizer", vocab_size=cfg.model.vocab_size, use_tokenizer_vocab_size=False
    )
    cfg.rng.seed = 20260901
    return cfg
