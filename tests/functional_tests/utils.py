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

import os
import shutil
from pathlib import Path

import torch

from megatron.bridge.training.utils.checkpoint_utils import (
    TRACKER_PREFIX,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_checkpoint_train_state_filename,
)


def initialize_distributed() -> None:
    """Initialize global process group for distributed execution."""
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))

    device_count = torch.cuda.device_count()
    if device_count > 0:
        torch.cuda.set_device(local_rank)

    # Call the init process
    init_process_group_kwargs = {
        "backend": "nccl",
        "world_size": world_size,
        "rank": rank,
    }
    torch.distributed.init_process_group(**init_process_group_kwargs)
    torch.distributed.barrier(device_ids=[local_rank])


def broadcast_path(path: str | Path) -> str:
    """
    Broadcast a path from rank 0 to all ranks. This function assumes that torch.distributed is already initialized.

    Args:
        path: Path to broadcast

    Returns:
        str: Broadcasted path
    """
    assert torch.distributed.is_initialized(), "Distributed is not initialized"

    if torch.distributed.get_world_size() == 1:
        return path

    # Create a shared directory path - rank 0 creates it, then broadcasts to all ranks
    if torch.distributed.get_rank() == 0:
        ret_path = str(path)
    else:
        ret_path = None

    shared_dir_list = [ret_path]
    torch.distributed.broadcast_object_list(shared_dir_list, src=0)
    shared_path = shared_dir_list[0]
    return shared_path


def get_directory_size(path: str) -> int:
    """Calculate the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def clear_directories(path: str) -> None:
    """Delete a directory on rank 0."""
    if not torch.distributed.is_initialized():
        if os.path.exists(path):
            shutil.rmtree(path)
        return

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            if os.path.exists(path):
                shutil.rmtree(path)
        torch.distributed.barrier()


def verify_checkpoint_files(
    checkpoint_dir: str,
    iteration_count: int,
    ckpt_format: str = "torch_dist",
    storage_writers_per_rank: int = 1,
) -> None:
    """Verify that checkpoint files were created correctly.

    Args:
        checkpoint_dir: Directory containing checkpoints
        iteration_count: Expected iteration number for the checkpoint
        ckpt_format: Checkpoint format ("torch_dist", "fsdp_dtensor", etc.)
        storage_writers_per_rank: Storage writers per rank (torch_dist only).
            Pass config.checkpoint.storage_writers_per_rank.
            Affects expected file count: world_size * storage_writers_per_rank.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        # Verify Megatron-Bridge tracker file
        latest_tracker_file = get_checkpoint_train_state_filename(checkpoint_dir, prefix=TRACKER_PREFIX)
        assert os.path.exists(latest_tracker_file), "Latest checkpoint tracker file not found"

        # Verify Megatron-LM compatibility tracker file
        megatron_lm_tracker = get_checkpoint_tracker_filename(checkpoint_dir)
        assert os.path.exists(megatron_lm_tracker), "Megatron-LM tracker file not found"

        # Verify the tracker file contains the correct iteration
        with open(megatron_lm_tracker, "r") as f:
            saved_iteration = f.read().strip()
        assert saved_iteration == str(iteration_count), (
            f"Megatron-LM tracker file contains '{saved_iteration}', expected '{iteration_count}'"
        )

        final_iter_dir = get_checkpoint_name(checkpoint_dir, iteration_count, release=False)
        assert os.path.exists(final_iter_dir), f"Final checkpoint directory not found at {final_iter_dir}"

        metadata_file = os.path.join(final_iter_dir, ".metadata")
        assert os.path.exists(metadata_file), "Checkpoint metadata file not found"

        # Both formats use torch.distributed.checkpoint but may create different numbers of .distcp files
        distcp_files = [f for f in os.listdir(final_iter_dir) if f.endswith(".distcp")]

        if ckpt_format == "torch_dist":
            num_expected_files = storage_writers_per_rank * torch.distributed.get_world_size()
        elif ckpt_format == "fsdp_dtensor":
            # fsdp_dtensor format creates .distcp files (one per rank)
            num_expected_files = torch.distributed.get_world_size()
        else:
            raise ValueError(f"Unsupported checkpoint format for verification: {ckpt_format}")

        assert len(distcp_files) == num_expected_files, (
            f"Expected {num_expected_files} .distcp files for {ckpt_format}, found {len(distcp_files)}: {distcp_files}"
        )


def verify_peft_checkpoint_smaller(pretrain_checkpoint_dir, peft_checkpoint_dir, pretrain_iters, peft_iters) -> None:
    """Verify that PEFT checkpoint is smaller than pretrained checkpoint (adapter weights only)."""
    if torch.distributed.get_rank() == 0:
        pretrain_iter_dir = os.path.join(pretrain_checkpoint_dir, f"iter_{pretrain_iters:07d}")
        peft_iter_dir = os.path.join(peft_checkpoint_dir, f"iter_{peft_iters:07d}")

        assert os.path.exists(pretrain_iter_dir), f"Pretrain checkpoint directory not found at {pretrain_iter_dir}"
        assert os.path.exists(peft_iter_dir), f"PEFT checkpoint directory not found at {peft_iter_dir}"

        pretrain_size = get_directory_size(pretrain_iter_dir)
        peft_size = get_directory_size(peft_iter_dir)

        # PEFT checkpoint should be significantly smaller (only adapter weights)
        assert peft_size < pretrain_size * 0.6, (
            f"PEFT checkpoint ({peft_size}) should be smaller than 60% of pretrain checkpoint ({pretrain_size})"
        )


def compare_provider_configs(converted_provider, predefined_provider, model_id, skip_fields=None):
    """Compare ALL configuration attributes between converted and predefined providers.

    Args:
        converted_provider: The provider converted from HuggingFace
        predefined_provider: The predefined provider class
        model_id: Model identifier for error messages
        skip_fields: Optional set of field names to skip comparison for this specific model
    """
    if skip_fields is None:
        skip_fields = set()

    # Get all attributes from both providers
    converted_attrs = vars(converted_provider)
    predefined_attrs = vars(predefined_provider)

    # First check that both providers have the same set of attributes
    converted_keys = set(converted_attrs.keys())
    predefined_keys = set(predefined_attrs.keys())

    missing_in_converted = predefined_keys - converted_keys
    missing_in_predefined = converted_keys - predefined_keys

    if missing_in_converted:
        raise AssertionError(f"Converted provider for {model_id} is missing attributes: {missing_in_converted}")

    if missing_in_predefined:
        raise AssertionError(f"Predefined provider for {model_id} is missing attributes: {missing_in_predefined}")

    # Compare all attribute values
    mismatched_attrs = []
    excluded_attrs = set()

    for attr_name in sorted(converted_keys):
        # Skip excluded attributes
        if "init_method" in attr_name or attr_name in {"generation_config", "vocab_size", "hf_model_id"}:
            excluded_attrs.add(attr_name)
            continue

        # Skip model-specific fields
        if attr_name in skip_fields:
            excluded_attrs.add(attr_name)
            continue

        converted_value = converted_attrs[attr_name]
        predefined_value = predefined_attrs[attr_name]

        # Handle special comparison cases for different types
        if converted_value != predefined_value:
            # For functions, compare by name/identity since they might be the same function
            # but not pass == comparison
            if callable(converted_value) and callable(predefined_value):
                if (
                    hasattr(converted_value, "__name__")
                    and hasattr(predefined_value, "__name__")
                    and converted_value.__name__ == predefined_value.__name__
                ):
                    continue
                elif converted_value is predefined_value:
                    continue

            mismatched_attrs.append(f"  {attr_name}: converted={converted_value} vs predefined={predefined_value}")

    if mismatched_attrs:
        raise AssertionError(f"Configuration mismatch for {model_id}:\n" + "\n".join(mismatched_attrs))


def autoconfig_roundtrip(
    local_model_path: str, tmp_path: Path, trust_remote_code: bool = False, atol: float = 2e-2
) -> None:
    """Run a full HF → Megatron → HF auto-config roundtrip and assert correctness.

    This is the shared implementation for all ``test_<model>_autoconfig_roundtrip``
    tests.  It performs:

    1. ``AutoBridge.import_ckpt`` (HF → Megatron)
    2. ``AutoBridge.from_auto_config`` + ``export_ckpt`` (Megatron → HF)
    3. Config diff report (printed)
    4. Weight key & value comparison (asserted bit-exact)
    5. Forward-pass logit comparison (asserted within *atol*)

    Args:
        local_model_path: Path to the toy HF model directory.
        tmp_path: Pytest ``tmp_path`` fixture.
        trust_remote_code: Passed to ``import_ckpt`` and ``from_pretrained``.
            When True, also copies ``*.py`` from *local_model_path* into the
            export directory so that custom modeling code is available.
        atol: Absolute tolerance for the forward-pass logit comparison.
    """
    import json
    import shutil

    import megatron.core.parallel_state as parallel_state
    import torch.distributed as dist
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForMultimodalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSpeechSeq2Seq,
    )

    from megatron.bridge import AutoBridge

    megatron_root = str(tmp_path / "megatron")
    export_path = tmp_path / "hf_export"

    # Read original config
    with open(Path(local_model_path) / "config.json") as f:
        original_config = json.load(f)

    # HF → Megatron
    AutoBridge.import_ckpt(
        hf_model_id=local_model_path,
        megatron_path=megatron_root,
        trust_remote_code=trust_remote_code,
    )

    # Tear down distributed state between import and export
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()

    # Megatron → HF via auto-config export
    bridge = AutoBridge.from_auto_config(megatron_root, local_model_path)
    bridge.export_ckpt(
        megatron_path=megatron_root,
        hf_path=str(export_path),
        show_progress=True,
        strict=False,
    )

    # Copy custom modeling files for trust_remote_code models
    if trust_remote_code:
        for py_file in Path(local_model_path).glob("*.py"):
            shutil.copy(py_file, export_path)

    # ── Config diff ──────────────────────────────────────────────
    with open(export_path / "config.json") as f:
        exported_config = json.load(f)

    print_config_diff(original_config, exported_config)

    # ── Debug: inspect exported weights before loading ─────────
    from safetensors.torch import load_file as load_safetensors

    export_weights_files = list(export_path.glob("*.safetensors"))
    if export_weights_files:
        print("\n" + "=" * 70)
        print("DEBUG: Exported weight shapes (from safetensors)")
        print("=" * 70)
        for wf in sorted(export_weights_files):
            sd = load_safetensors(str(wf))
            for k in sorted(sd.keys()):
                print(f"  {k}: {sd[k].shape} {sd[k].dtype}")
        print("=" * 70)

    print(f"\nDEBUG: exported config model_type = {exported_config.get('model_type')}")
    print(f"DEBUG: exported config architectures = {exported_config.get('architectures')}")

    # ── Weight & forward-pass comparison ─────────────────────────
    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust_remote_code)

    loader_candidates = (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForMultimodalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSpeechSeq2Seq,
    )
    errors = []
    original = None
    exported = None

    for loader in loader_candidates:
        try:
            original = loader.from_pretrained(local_model_path, **load_kwargs)
            exported = loader.from_pretrained(str(export_path), **load_kwargs)
            break
        except Exception as exc:  # pragma: no cover - exercised by model-specific mappings
            errors.append(f"{loader.__name__}: {exc}")

    if original is None or exported is None:
        joined_errors = "\n".join(errors)
        raise RuntimeError(f"Unable to load roundtrip models with available auto loaders.\nTried:\n{joined_errors}")

    assert_weights_equal(original, exported)
    assert_forward_pass_equal(original, exported, atol=atol)


def print_config_diff(original_config: dict, exported_config: dict) -> None:
    """Print a detailed diff between two HF config dicts."""
    all_keys = sorted(set(list(original_config.keys()) + list(exported_config.keys())))
    orig_only = []
    export_only = []
    value_diffs = []
    matched = []

    for key in all_keys:
        in_orig = key in original_config
        in_exp = key in exported_config
        if in_orig and not in_exp:
            orig_only.append(key)
        elif in_exp and not in_orig:
            export_only.append(key)
        elif original_config[key] != exported_config[key]:
            value_diffs.append((key, original_config[key], exported_config[key]))
        else:
            matched.append(key)

    print("\n" + "=" * 70)
    print("CONFIG DIFF REPORT")
    print("=" * 70)
    print(f"\nTotal keys — original: {len(original_config)}, exported: {len(exported_config)}")
    print(f"Matched:       {len(matched)}")
    print(f"Value diffs:   {len(value_diffs)}")
    print(f"Only original: {len(orig_only)}")
    print(f"Only exported: {len(export_only)}")

    if orig_only:
        print("\n--- Keys ONLY in original (missing from export) ---")
        for key in orig_only:
            print(f"  {key}: {original_config[key]}")

    if export_only:
        print("\n--- Keys ONLY in export (new/unexpected) ---")
        for key in export_only:
            print(f"  {key}: {exported_config[key]}")

    if value_diffs:
        print("\n--- Keys with DIFFERENT values ---")
        for key, orig_val, exp_val in value_diffs:
            print(f"  {key}:")
            print(f"    original: {orig_val!r}")
            print(f"    exported: {exp_val!r}")

    if matched:
        print(f"\n--- Matched keys ({len(matched)}) ---")
        for key in matched:
            print(f"  {key}: {original_config[key]!r}")

    print("=" * 70)


def assert_weights_equal(original, exported) -> None:
    """Assert that two models have bit-exact matching weights, with detailed diff on failure."""
    pure_sd = {k: v.cpu() for k, v in original.state_dict().items()}
    conv_sd = {k: v.cpu() for k, v in exported.state_dict().items()}

    orig_keys = set(pure_sd.keys())
    conv_keys = set(conv_sd.keys())
    only_in_orig = orig_keys - conv_keys
    only_in_conv = conv_keys - orig_keys
    common_keys = orig_keys & conv_keys

    print("\n" + "=" * 70)
    print("WEIGHT KEY DIFF REPORT")
    print("=" * 70)
    print(f"Original keys: {len(orig_keys)}, Exported keys: {len(conv_keys)}")

    if only_in_orig:
        print(f"\n--- Keys ONLY in original weights ({len(only_in_orig)}) ---")
        for k in sorted(only_in_orig):
            print(f"  {k}  shape={pure_sd[k].shape}  dtype={pure_sd[k].dtype}")

    if only_in_conv:
        print(f"\n--- Keys ONLY in exported weights ({len(only_in_conv)}) ---")
        for k in sorted(only_in_conv):
            print(f"  {k}  shape={conv_sd[k].shape}  dtype={conv_sd[k].dtype}")

    mismatched_weights = []
    matched_weights = 0
    for key in sorted(common_keys):
        if not torch.equal(pure_sd[key], conv_sd[key]):
            max_diff = (pure_sd[key].float() - conv_sd[key].float()).abs().max().item()
            mismatched_weights.append((key, pure_sd[key].shape, max_diff))
        else:
            matched_weights += 1

    print(f"\nCommon keys: {len(common_keys)}")
    print(f"  Bit-exact match: {matched_weights}")
    print(f"  Value mismatch:  {len(mismatched_weights)}")

    if mismatched_weights:
        print("\n--- Weight value mismatches ---")
        for key, shape, max_diff in mismatched_weights:
            print(f"  {key}  shape={shape}  max_diff={max_diff:.6e}")

    print("=" * 70)

    assert not only_in_orig, f"Keys only in original: {only_in_orig}"
    assert not only_in_conv, f"Keys only in exported: {only_in_conv}"
    assert not mismatched_weights, f"Weight mismatches: {[m[0] for m in mismatched_weights]}"


def assert_forward_pass_equal(original, exported, atol: float = 2e-2) -> None:
    """Assert that two models produce matching logits for a simple input."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=next(original.parameters()).device)
    with torch.no_grad():
        orig_logits = original(input_ids).logits.cpu()
        export_logits = exported(input_ids).logits.cpu()
    max_diff = (orig_logits - export_logits).abs().max().item()
    mean_diff = (orig_logits - export_logits).abs().mean().item()
    print(f"\nFORWARD PASS: max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")
    assert torch.allclose(orig_logits, export_logits, atol=atol), f"Forward pass mismatch: max diff {max_diff}"
