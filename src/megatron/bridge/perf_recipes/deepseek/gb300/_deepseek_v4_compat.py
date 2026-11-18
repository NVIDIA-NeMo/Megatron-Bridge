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

"""Compatibility helpers for the DSv4 main and development-MCore benchmarks.

The fixed THD dataset reproduces the one-sequence-per-pack shape of the supplied
fixed-length mock runs. It does not implement MCore's general variable-length
``dp_balanced`` data scheduler or reproduce its random token stream.
"""

import logging
from dataclasses import dataclass, field, fields

import torch
from megatron.core.transformer.spec_utils import ModuleSpec
from torch.utils.data import Dataset

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.packing.config import PackedSequenceSpecs
from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import DeepSeekV4HybridModelProvider


logger = logging.getLogger(__name__)

_MCORE_CAPABILITIES = (
    "is_hybrid_model",
    "moe_megakernel_backend",
    "moe_megakernel_backend_config",
    "moe_hybridep_routing_map_mode",
    "mhc_recompute_layer_num",
    "mhc_recompute_attn_cuda_graph_split",
    "sequence_packing_scheduler",
    "max_seqlen_per_dp_cp_rank",
    "pad_packed_seq_alignment",
    "attention_cp_layout",
    "linear_cp_layout",
)
_DEV_MCORE_CAPABILITIES = (
    "cuda_graph_dynamic_microbatches",
    "pipeline_p2p_fixed_shape",
    "thd_max_packed_sequences",
    "cp_partition_mode",
    "dsa_indexer_precision",
    "o_groups",
    "o_lora_rank",
    "moe_n_hash_layers",
    "actual_vocab_size",
)
_MAIN_MCORE_CAPABILITIES = (
    "cuda_graph_thd_max_packed_sequences",
    "output_projection_groups",
    "output_projection_lora_rank",
    "moe_num_hash_layers",
    "hash_moe_vocab_size",
)


def dsv4_hybrid_stack_spec(config: DeepSeekV4HybridModelProvider) -> ModuleSpec:
    """Resolve native main's static spec or development MCore's spec factory."""
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec

    if isinstance(hybrid_dsv4_stack_spec, ModuleSpec):
        return hybrid_dsv4_stack_spec
    return hybrid_dsv4_stack_spec(config)


def configure_dsv4_perf_model(model: DeepSeekV4HybridModelProvider) -> None:
    """Configure declared main/dev APIs without weakening the graph requirements.

    Native main uses BF16 packed indexer scoring; development MCore retains the
    measured compact MXFP8 indexer. Both require mHC recompute with attention
    graphs (MCore #7337 on main). Call after choosing the main and MTP patterns.
    """
    declared_fields = {item.name for item in fields(model)}
    development_api = "o_groups" in declared_fields
    api_fields = _DEV_MCORE_CAPABILITIES if development_api else _MAIN_MCORE_CAPABILITIES
    missing = sorted(set(_MCORE_CAPABILITIES + api_fields) - declared_fields)
    if missing:
        raise RuntimeError(
            "The DSv4 performance recipes require MCore with DSv4 THD, MoK, and "
            "mHC recompute with attention CUDA graphs. Native main requires the "
            "mHC graph support from https://github.com/NVIDIA/Megatron-LM/pull/7337. "
            "See examples/models/deepseek_v4/README.md for exact runtime revisions "
            "and the native BF16 indexer difference. "
            f"The installed MCore is missing configuration fields: {', '.join(missing)}."
        )

    if "enable_hyper_connections" in declared_fields:
        if "num_residual_streams" not in declared_fields:
            raise RuntimeError("The installed MCore does not declare num_residual_streams for mHC.")
        model.enable_hyper_connections = True
        model.num_residual_streams = 4
    elif "enable_mhc_connections" in declared_fields:
        if "mhc_num_residual_streams" not in declared_fields:
            raise RuntimeError("The installed MCore does not declare mhc_num_residual_streams for mHC.")
        model.enable_mhc_connections = True
        model.mhc_num_residual_streams = 4
    else:
        raise RuntimeError("The installed MCore does not declare the DSv4 mHC configuration fields.")

    # Match native argument processing for initialization and hash-layer counting.
    model.is_hybrid_model = True
    model.attention_cp_layout = "contiguous"
    model.linear_cp_layout = "contiguous"
    model.hybrid_stack_spec = dsv4_hybrid_stack_spec
    if development_api:
        model.o_groups = 16
        model.o_lora_rank = 1024
        model.moe_n_hash_layers = 3
        model.actual_vocab_size = 129280
        model.cp_partition_mode = "contiguous"
        model.pipeline_p2p_fixed_shape = True
        model.thd_max_packed_sequences = 8
        model.cuda_graph_dynamic_microbatches = True
        model.dsa_indexer_precision = "mxfp8"
    else:
        model.output_projection_groups = 16
        model.output_projection_lora_rank = 1024
        model.moe_num_hash_layers = 3
        model.hash_moe_vocab_size = 129280
        model.cuda_graph_thd_max_packed_sequences = 8
        # Main infers the fixed microbatch graph count and uses native PP shape
        # exchange. It has no compact MXFP8 CSA indexer precision selector.
        logger.warning(
            "DSv4 native-main functional variant: packed indexer scoring uses BF16; "
            "model training remains MXFP8. This differs from the reference compact "
            "MXFP8 indexer. See examples/models/deepseek_v4/README.md."
        )
    if model.hybrid_layer_pattern is None:
        raise ValueError("Set the DSv4 main hybrid pattern before configuring its stack.")
    main_pattern = model.hybrid_layer_pattern.split("/", 1)[0].replace("|", "")
    mtp_pattern = model.mtp_hybrid_override_pattern or ""
    symbols = main_pattern + mtp_pattern * (model.mtp_num_layers or 0)
    ratios = {"H": 128, "C": 4, "W": 0, "E": 0}
    if set(symbols) - ratios.keys():
        raise ValueError("The DSv4 performance recipe requires explicit H/C/W attention and E MoE symbols.")
    model.csa_compress_ratios = [ratios[symbol] for symbol in symbols]


class _FixedTHDMockDataset(Dataset):
    """Deterministic synthetic next-token samples with static packed metadata."""

    def __init__(
        self, *, num_samples: int, seq_length: int, vocab_size: int, seed: int, eod: int, metadata_capacity: int
    ) -> None:
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed
        self.eod = eod
        self.metadata_capacity = metadata_capacity

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        # MCore MockVarlenDataset samples length-1 token IDs and appends EOD.
        # Next-token shifting leaves length-1 valid targets; pad the final slot.
        token_stream = torch.randint(self.vocab_size, (self.seq_length + 1,), generator=generator)
        token_stream[-2:] = self.eod
        tokens = token_stream[:-1].clone()
        labels = token_stream[1:].clone()
        tokens[-1] = labels[-1] = 0
        loss_mask = torch.ones(self.seq_length, dtype=torch.float32)
        loss_mask[-1] = 0.0
        position_ids = torch.arange(self.seq_length, dtype=torch.int64)
        position_ids[-1] = 0
        padding_mask = torch.zeros(self.seq_length, dtype=torch.bool)
        padding_mask[-1] = True
        cu_seqlens = torch.full((self.metadata_capacity + 1,), self.seq_length - 1, dtype=torch.int32)
        cu_seqlens[0] = 0
        cu_seqlens_padded = torch.full((self.metadata_capacity + 1,), self.seq_length, dtype=torch.int32)
        cu_seqlens_padded[0] = 0
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "padding_mask": padding_mask,
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_kv": cu_seqlens.clone(),
            "cu_seqlens_q_padded": cu_seqlens_padded,
            "cu_seqlens_kv_padded": cu_seqlens_padded.clone(),
            "max_seqlen_q": torch.tensor(self.seq_length, dtype=torch.int32),
            "max_seqlen_kv": torch.tensor(self.seq_length, dtype=torch.int32),
            "pad_between_seqs": torch.tensor(True),
        }


@dataclass(kw_only=True)
class FixedTHDMockDatasetConfig(DatasetProvider):
    """One fixed synthetic sequence per MBS1 pack for the DSv4 benchmarks.

    The offline-packing metadata declaration makes Bridge retain boundaries on
    intermediate PP stages. Samples are generated in memory; no pack file is used.
    """

    seq_length: int
    vocab_size: int
    seed: int
    metadata_capacity: int
    dataloader_type: str = "single"
    num_workers: int = 0
    persistent_workers: bool = False
    skip_getting_attention_mask_from_dataset: bool = True
    enable_offline_packing: bool = field(init=False, default=True)
    pad_to_max_length: bool = field(init=False, default=True)
    offline_packing_specs: PackedSequenceSpecs = field(init=False)

    def __post_init__(self) -> None:
        self.finalize()

    def finalize(self) -> None:
        """Validate the fixed pack dimensions and expose static graph metadata."""
        super().finalize()
        if self.seq_length < 2 or self.vocab_size < 2 or self.metadata_capacity < 1:
            raise ValueError("DSv4 fixed THD data requires seq_length/vocab_size >= 2 and metadata_capacity >= 1.")
        self.offline_packing_specs = PackedSequenceSpecs(
            packed_sequence_size=self.seq_length,
            max_single_sequence_length=self.seq_length,
            pad_cu_seqlens=True,
            pad_seq_to_mult=1,
        )

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Dataset | None, Dataset | None, Dataset | None]:
        """Build reproducible train, validation, and test sample streams."""
        if context.tokenizer is None:
            raise ValueError("DSv4 fixed THD mock data requires the configured tokenizer.")
        eod = context.tokenizer.eod
        if not 0 <= eod < self.vocab_size:
            raise ValueError("The tokenizer EOD ID must be within the configured model vocabulary.")

        def build(num_samples: int, split: int) -> Dataset | None:
            if not num_samples:
                return None
            return _FixedTHDMockDataset(
                num_samples=num_samples,
                seq_length=self.seq_length,
                vocab_size=self.vocab_size,
                seed=self.seed + split * 1_000_000_000,
                eod=eod,
                metadata_capacity=self.metadata_capacity,
            )

        return build(context.train_samples, 0), build(context.valid_samples, 1), build(context.test_samples, 2)
