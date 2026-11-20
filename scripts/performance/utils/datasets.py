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
def create_mock_dataset_config(seq_length, num_workers=8, pin_memory=False, persistent_workers=False):
    """Create mock dataset configuration for Megatron-Bridge."""
    from megatron.bridge.training.config import MockGPTDatasetConfig

    # Create mock dataset using MockGPTDatasetConfig which enforces blend=None, blend_per_split=None
    return MockGPTDatasetConfig(
        seq_length=seq_length,
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        num_dataset_builder_threads=1,
        split="99990,8,2",  # Standard train/val/test split
        # Dataloader config parameters
        data_sharding=True,
        dataloader_type="single",
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def create_rp2_dataset_config(
    dataset_paths, seq_length, index_mapping_dir=None, num_workers=1, pin_memory=False, persistent_workers=True
):
    """Create RedPajama2 dataset configuration for Megatron-Bridge."""
    from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
    from megatron.bridge.training.config import GPTDatasetConfig

    # Get blend configuration for rp2 data paths
    blend, blend_per_split, split = get_blend_fields_from_data_paths(data_paths=dataset_paths, mock=False)

    return GPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        seq_length=seq_length,
        num_dataset_builder_threads=1,
        blend=blend,
        blend_per_split=blend_per_split,
        split=split or "99990,8,2",
        path_to_cache=index_mapping_dir,
        # Dataloader config parameters
        data_sharding=True,
        dataloader_type="single",
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def create_c4_dataset_config(
    seq_length,
    c4_root,
    train_shards=(6,),
    num_workers=1,
    pin_memory=False,
    persistent_workers=True,
    index_mapping_dir=None,
):
    """Create C4 dataset configuration for Megatron-Bridge.

    Uses Megatron-mmap C4 files at:
      <c4_root>/c4-train.en_<shard>_text_document.{bin,idx}
      <c4_root>/c4-validation-91205-samples.en_text_document.{bin,idx}

    Produces a blend_per_split with (train, val, test) tuples where val and
    test point to the same validation file (matches NVIDIA's MLPerf DSV3
    reference layout).
    """
    from megatron.bridge.training.config import GPTDatasetConfig

    val_prefix = f"{c4_root}/c4-validation-91205-samples.en_text_document"
    train_prefixes = [f"{c4_root}/c4-train.en_{i}_text_document" for i in train_shards]
    train_weights = [50.0] * len(train_prefixes)

    # GPTDatasetConfig.blend_per_split format: list of 3 (prefixes, weights) tuples
    # for (train, val, test). Mirrors NVIDIA's MLPerf DSV3 reference setup with
    # val and test pointing to the same validation file.
    blend_per_split = [
        (train_prefixes, train_weights),
        ([val_prefix], None),
        ([val_prefix], None),
    ]

    return GPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        seq_length=seq_length,
        num_dataset_builder_threads=1,
        blend=None,
        blend_per_split=blend_per_split,
        split=None,  # blend_per_split takes precedence; split must be None.
        path_to_cache=index_mapping_dir,
        data_sharding=True,
        dataloader_type="single",
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def _gov_report_pack_metadata(seq_length: int) -> list[dict[str, int]]:
    """Return packing metadata for official GovReport npy (one sequence per pack)."""
    return [
        {
            "dataset_max_seqlen": seq_length,
            "max_samples_per_bin": 1,
            "min_packed_seqlen": seq_length,
        }
    ]


def _ensure_gov_report_pack_metadata(*, dataset_root, seq_length: int):
    """Write static packing metadata for CUDA-graph-compatible packed npy.

    Official GovReport packs are already a single ``seq_length`` sequence per
    row (``seq_start_id=[0]``). The metadata file is required when
    ``pad_cu_seqlens=True``. Prefer ``dataset_root``; fall back to a temp dir
    when that tree is read-only.
    """
    import json
    import tempfile
    from pathlib import Path

    serialized = json.dumps(_gov_report_pack_metadata(seq_length))
    candidates = [
        Path(dataset_root) / "packed_metadata.json",
        Path(tempfile.gettempdir()) / f"gov_report_packed_metadata_{seq_length}.json",
    ]
    last_error = None
    for path in candidates:
        try:
            path.write_text(serialized)
            return path
        except OSError as error:
            if path.is_file():
                return path
            last_error = error
    raise OSError(
        f"Unable to write GovReport packing metadata under {dataset_root} or {tempfile.gettempdir()}"
    ) from last_error


def create_gov_report_dataset_config(
    *,
    dataset_root: str,
    seq_length: int,
    seed: int,
    max_train_samples: int | None,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
):
    """Create a packed SCROLLS GovReport dataset config for Llama 2 PEFT.

    Expects the packed MLPerf-format artifacts produced by
    ``scripts/performance/download_gov_report.py``:

      ``<dataset_root>/train.npy``
      ``<dataset_root>/validation.npy``
    """
    from pathlib import Path

    from megatron.bridge.data.builders import GPTSFTDatasetConfig
    from megatron.bridge.data.packing import PackedSequenceSpecs

    root = Path(dataset_root)
    train_path = root / "train.npy"
    val_path = root / "validation.npy"
    missing = [str(path) for path in (train_path, val_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "GovReport packed files not found: "
            + ", ".join(missing)
            + ". Download them with scripts/performance/download_gov_report.py "
            + f"--data_dir {root}"
        )

    metadata_path = _ensure_gov_report_pack_metadata(dataset_root=root, seq_length=seq_length)
    return GPTSFTDatasetConfig(
        seq_length=seq_length,
        dataset_root=root,
        seed=seed,
        do_validation=True,
        do_test=False,
        enable_offline_packing=True,
        offline_packing_specs=PackedSequenceSpecs(
            packed_sequence_size=seq_length,
            packed_train_data_path=train_path,
            packed_val_data_path=val_path,
            packed_metadata_path=metadata_path,
            pad_cu_seqlens=True,
        ),
        dataset_kwargs={"return_cu_seqlen": False, "pad_to_max_length": True},
        dataloader_type="batch",
        max_train_samples=max_train_samples,
        num_workers=num_workers,
        data_sharding=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def create_squad_dataset_config(
    dataset_root, seq_length, packed=False, pad_seq_to_mult=1, num_workers=2, pin_memory=True, persistent_workers=False
):
    """Create SQuAD dataset configuration for Megatron-Bridge using HF text SFT data."""
    from megatron.bridge.data.builders import (
        GPTSFTDatasetConfig,
        HFDatasetSourceConfig,
        PromptCompletionSFTPreprocessingConfig,
    )
    from megatron.bridge.data.packing import PackedSequenceSpecs

    dataset_kwargs = {}
    offline_packing_specs = None
    if packed:
        dataset_kwargs["pad_to_max_length"] = True
        offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

    return GPTSFTDatasetConfig(
        seq_length=seq_length,
        hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
        hf_output_root=dataset_root,
        hf_validation_proportion=0.1,
        seed=5678,
        do_validation=True,
        do_test=False,
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="input",
            completion_column="output",
            separator=" ",
        ),
        dataset_kwargs=dataset_kwargs,
        enable_offline_packing=packed,
        offline_packing_specs=offline_packing_specs,
        dataloader_type="single",
        num_workers=num_workers,
        data_sharding=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
