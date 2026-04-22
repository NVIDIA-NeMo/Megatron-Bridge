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

"""
Standalone forward/backward test for ERNIE 4.5 VL MoE Megatron model.

This script must be launched with torchrun (for multi-GPU tests):
    torchrun --nproc_per_node=N ernie45_vl_fwd_bwd.py --hf-model-path <path> --tp T --pp P --ep E

It performs:
1. Load HF toy model -> convert to Megatron via AutoBridge
2. Forward pass with text-only or text+vision input
3. Backward pass: compute loss, check gradients exist
4. Print PASS/FAIL status

With --with-vision, constructs a dummy image input that exercises the full
vision pipeline: ViT patch embedding -> vision transformer -> resampler ->
embedding injection -> language model forward.

Exit code 0 = PASS, non-zero = FAIL.
"""

import argparse
import os


# Disable torch.compile to avoid triton compatibility issues in some environments.
# Must be set before importing torch.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.distributed as dist

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main


def _is_rank_0() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def print_rank_0(msg: str):
    """Print a message only from rank 0."""
    if _is_rank_0():
        print(msg, flush=True)


def run_forward_backward(
    hf_model_path: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    seq_len: int = 16,
    backward: bool = True,
    with_vision: bool = False,
    prompt: str | None = None,
):
    """Run forward (and optionally backward) pass on the ERNIE 4.5 VL MoE toy model.

    Args:
        hf_model_path: Path to the HF toy model directory.
        tp: Tensor parallelism size.
        pp: Pipeline parallelism size.
        ep: Expert parallelism size.
        seq_len: Sequence length for the dummy input.
        backward: Whether to also run backward pass.
        with_vision: Whether to include a dummy image in the input to exercise
            the vision tower and resampler forward path.
    """
    print_rank_0("=== ERNIE 4.5 VL Forward/Backward Test ===")
    print_rank_0(f"  HF model: {hf_model_path}")
    print_rank_0(f"  TP={tp}, PP={pp}, EP={ep}, seq_len={seq_len}, backward={backward}, with_vision={with_vision}")

    # ------------------------------------------------------------------ #
    # 1. Build Megatron model from HF checkpoint via AutoBridge
    # ------------------------------------------------------------------ #
    print_rank_0("Step 1: Loading HF model and converting to Megatron...")

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    # Megatron requires sequence parallelism when MoE + TP > 1 during training
    if tp > 1 and backward:
        model_provider.sequence_parallel = True
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=42)

    megatron_models = model_provider.provide_distributed_model(wrap_with_ddp=False)

    # Disable deallocate_pipeline_outputs: the no-pipelining schedule does not
    # call deallocate_output_tensor(), so backward_step's custom_backward()
    # would fail with "output should be pseudo-'freed' in schedule".
    for m in megatron_models:
        if hasattr(m, "config"):
            m.config.deallocate_pipeline_outputs = False

    # Put models in train mode for backward pass
    if backward:
        for m in megatron_models:
            m.train()
    else:
        for m in megatron_models:
            m.eval()

    print_rank_0(f"  Model built successfully. {len(megatron_models)} component(s).")

    # ------------------------------------------------------------------ #
    # 2. Prepare input (text-only or text+vision)
    # ------------------------------------------------------------------ #
    print_rank_0(f"Step 2: Preparing {'text+vision' if with_vision else 'text-only'} input...")

    # Use vocab_size from model config to stay in valid range
    vocab_size = getattr(model_provider, "padded_vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(model_provider, "vocab_size", None)
    if vocab_size is None:
        vocab_size = 2048  # toy model default

    pixel_values = None
    image_grid_thw = None

    if with_vision:
        # Read vision config and special token IDs from the model provider / HF config
        import json

        with open(os.path.join(hf_model_path, "config.json")) as f:
            hf_cfg = json.load(f)

        image_token_id = hf_cfg.get("image_token_id", 100295)
        image_start_token_id = hf_cfg.get("image_start_token_id", 101304)
        image_end_token_id = hf_cfg.get("image_end_token_id", 101305)

        # For toy models with small vocab_size, remap special token IDs to fit
        # within the embedding table.  This allows the vision pipeline to execute
        # without index-out-of-range errors in the embedding layer.
        text_cfg = hf_cfg.get("text_config", hf_cfg)
        cfg_vocab_size = text_cfg.get("vocab_size", vocab_size)
        if (
            image_token_id >= cfg_vocab_size
            or image_start_token_id >= cfg_vocab_size
            or image_end_token_id >= cfg_vocab_size
        ):
            # Use the last 3 tokens in the vocab as placeholders
            image_token_id = cfg_vocab_size - 3
            image_start_token_id = cfg_vocab_size - 2
            image_end_token_id = cfg_vocab_size - 1
            print_rank_0(
                f"  Remapped special token IDs to fit vocab_size={cfg_vocab_size}: "
                f"image_token={image_token_id}, start={image_start_token_id}, end={image_end_token_id}"
            )

            # Also update model config so get_placeholder_mask / get_rope_index use the remapped IDs
            for m in megatron_models:
                if hasattr(m, "config"):
                    m.config.image_token_id = image_token_id
                    m.config.image_start_token_id = image_start_token_id
                    m.config.image_end_token_id = image_end_token_id

        # Vision config for computing pixel_values shape
        vis_cfg = hf_cfg.get("vision_config", {})
        patch_size = vis_cfg.get("patch_size", 14)
        in_channels = vis_cfg.get("in_channels", 3)
        spatial_merge_size = vis_cfg.get("spatial_merge_size", 2)

        # Use minimal valid image grid: 2x2 patches (must be divisible by spatial_merge_size)
        grid_h, grid_w = 2, 2
        image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long, device="cuda")

        # pixel_values: [T*H*W, in_channels * patch_size * patch_size]
        num_patches = grid_h * grid_w  # 4
        patch_dim = in_channels * patch_size * patch_size  # 3*14*14 = 588
        pixel_values = torch.randn(num_patches, patch_dim, dtype=torch.bfloat16, device="cuda")

        # Number of image placeholder tokens after resampler spatial merge
        num_image_tokens = num_patches // (spatial_merge_size**2)  # 4 // 4 = 1

        # Build input_ids: [text..., image_start, <image_placeholders>, image_end, text...]
        # Ensure seq_len is large enough
        min_seq_len = num_image_tokens + 4  # at least: text + start + placeholders + end + text
        actual_seq_len = max(seq_len, min_seq_len)

        # Number of text tokens before and after the image block
        num_text_before = 2
        num_text_after = actual_seq_len - num_text_before - 1 - num_image_tokens - 1
        if num_text_after < 1:
            num_text_after = 1
            actual_seq_len = num_text_before + 1 + num_image_tokens + 1 + num_text_after

        # Construct input_ids
        # Use token IDs that don't collide with special tokens
        max_text_token = min(vocab_size, 1024, image_token_id)
        text_before = torch.randint(1, max(2, max_text_token), (num_text_before,), device="cuda")
        img_start = torch.tensor([image_start_token_id], device="cuda")
        img_placeholders = torch.full((num_image_tokens,), image_token_id, device="cuda")
        img_end = torch.tensor([image_end_token_id], device="cuda")
        text_after = torch.randint(1, max(2, max_text_token), (num_text_after,), device="cuda")
        input_ids = torch.cat([text_before, img_start, img_placeholders, img_end, text_after]).unsqueeze(0)
        actual_seq_len = input_ids.shape[1]

        # mm_token_type_ids: 0=text, 1=image placeholder
        mm_token_type_ids = torch.zeros(1, actual_seq_len, dtype=torch.int32, device="cuda")
        img_start_pos = num_text_before + 1  # position of first image placeholder
        mm_token_type_ids[0, img_start_pos : img_start_pos + num_image_tokens] = 1

        # Labels and loss mask
        labels = torch.randint(0, min(vocab_size, 1024), (1, actual_seq_len), device="cuda")
        loss_mask = torch.ones(1, actual_seq_len, dtype=torch.float32, device="cuda")

        print_rank_0(f"  input_ids shape: {input_ids.shape}, vocab_size: {vocab_size}")
        print_rank_0(f"  pixel_values shape: {pixel_values.shape}")
        print_rank_0(f"  image_grid_thw: {image_grid_thw.tolist()}")
        print_rank_0(f"  num_image_tokens (placeholders): {num_image_tokens}")
        seq_len = actual_seq_len
    else:
        # Text-only input
        if prompt is not None:
            # Use real text with tokenizer for meaningful loss measurement.
            # Labels are the next-token targets (input shifted right by 1).
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            # Need at least 2 tokens for next-token prediction
            assert len(token_ids) >= 2, f"Prompt too short: {len(token_ids)} tokens"
            input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
            seq_len = input_ids.shape[1]
            # Labels: shifted right by 1 (predict next token)
            # For position i, label[i] = input_ids[i+1]; last position has no valid label
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # ignore last position (no next token)
            loss_mask = torch.ones(1, seq_len, dtype=torch.float32, device="cuda")
            loss_mask[:, -1] = 0  # don't compute loss on last position
            print_rank_0(f"  Prompt: {prompt!r}")
            print_rank_0(f"  Tokenized: {len(token_ids)} tokens")
        else:
            input_ids = torch.randint(0, min(vocab_size, 1024), (1, seq_len), device="cuda")
            labels = torch.randint(0, min(vocab_size, 1024), (1, seq_len), device="cuda")
            loss_mask = torch.ones(1, seq_len, dtype=torch.float32, device="cuda")
        mm_token_type_ids = torch.zeros(1, seq_len, dtype=torch.int32, device="cuda")
        print_rank_0(f"  input_ids shape: {input_ids.shape}, vocab_size: {vocab_size}")

    # ------------------------------------------------------------------ #
    # 3. Forward pass
    # ------------------------------------------------------------------ #
    print_rank_0("Step 3: Running forward pass...")

    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    # Build a data iterator compatible with get_forward_backward_func
    class SingleBatchIterator:
        def __init__(self, batch):
            self.batch = batch
            self._yielded = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._yielded:
                raise StopIteration
            self._yielded = True
            return self.batch

    def ernie_vl_forward_step(data_iterator, model, **kwargs):
        """Forward step for ERNIE 4.5 VL model.

        Follows the Megatron forward_step / loss_func protocol:
        - forward_step returns (output_tensor, loss_func)
        - loss_func(output_tensor) returns a 3-tuple (scalar_loss, num_tokens, report)
          or 2-tuple (scalar_loss, report) for backward compatibility.
        - The scheduler calls loss_func on the last pipeline stage, normalises
          the scalar loss, then calls .backward() on it.

        GPTModel.forward(labels=...) returns per-token cross-entropy loss of
        shape [batch, seq_len].  loss_func must reduce it to a differentiable
        scalar via loss_mask.
        """
        batch = next(data_iterator)

        forward_args = {
            "input_ids": batch["tokens"],
            "mm_token_type_ids": batch["mm_token_type_ids"],
            "attention_mask": None,  # Let Megatron auto-generate causal mask
        }

        # Pass vision inputs if present
        if batch.get("pixel_values") is not None:
            forward_args["pixel_values"] = batch["pixel_values"]
        if batch.get("image_grid_thw") is not None:
            forward_args["image_grid_thw"] = batch["image_grid_thw"]

        if backward:
            forward_args["labels"] = batch["labels"]
            forward_args["loss_mask"] = batch["loss_mask"]

        output = model(**forward_args)

        if backward:
            # output is per-token loss [batch, seq_len] from GPTModel
            per_token_loss = output
            cur_loss_mask = batch["loss_mask"]

            def loss_func(output_tensor, **kwargs):
                # Reduce per-token loss to a mean scalar using loss_mask
                losses = output_tensor.view(-1).float()
                mask = cur_loss_mask.view(-1).float()
                num_tokens = mask.sum()
                loss = torch.sum(losses * mask) / torch.clamp(num_tokens, min=1)
                return loss, {"lm loss": loss.clone().detach()}

            return per_token_loss, loss_func
        else:
            if isinstance(output, tuple):
                output = output[0]

            def loss_func(output_tensor, **kwargs):
                # forward-only: return a dummy scalar loss and the logits as report
                dummy_loss = output_tensor.sum() * 0  # zero-grad scalar on same device
                return dummy_loss, {"logits": output_tensor}

            return output, loss_func

    batch = {
        "tokens": input_ids,
        "mm_token_type_ids": mm_token_type_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    fwd_bwd_func = get_forward_backward_func()
    iterator = SingleBatchIterator(batch)

    output = fwd_bwd_func(
        forward_step_func=ernie_vl_forward_step,
        data_iterator=iterator,
        model=megatron_models,
        num_microbatches=1,
        forward_only=not backward,
        seq_length=seq_len,
        micro_batch_size=1,
    )

    print_rank_0("  Forward pass completed successfully.")

    # ------------------------------------------------------------------ #
    # 4. Verify forward output
    # ------------------------------------------------------------------ #
    print_rank_0("Step 4: Verifying forward output...")

    from megatron.core import parallel_state

    is_last_stage = not dist.is_initialized() or parallel_state.is_pipeline_last_stage()

    if is_last_stage:
        if isinstance(output, list) and len(output) > 0:
            result = output[0]
        else:
            result = output

        if backward:
            # In backward mode, output contains loss info
            if isinstance(result, dict):
                print_rank_0(f"  Loss output: {result}")
            elif isinstance(result, torch.Tensor):
                print_rank_0(f"  Loss value: {result.item():.6f}")
                assert torch.isfinite(result), f"Loss is not finite: {result.item()}"
            else:
                print_rank_0(f"  Output type: {type(result)}, value: {result}")
        else:
            # In forward-only mode, output is stored as {"logits": tensor} from loss_func
            if isinstance(result, dict) and "logits" in result:
                logits = result["logits"]
                print_rank_0(f"  Logits shape: {logits.shape}")
                print_rank_0(f"  Logits stats: mean={logits.float().mean():.4f}, std={logits.float().std():.4f}")
                assert torch.isfinite(logits).all(), "Logits contain non-finite values"
            elif isinstance(result, torch.Tensor):
                print_rank_0(f"  Output shape: {result.shape}")
                print_rank_0(f"  Output stats: mean={result.float().mean():.4f}, std={result.float().std():.4f}")
                assert torch.isfinite(result).all(), "Output contains non-finite values"
            else:
                print_rank_0(f"  Output type: {type(result)}")

    print_rank_0("  Forward output verification passed.")

    # ------------------------------------------------------------------ #
    # 5. Verify gradients (backward pass)
    # ------------------------------------------------------------------ #
    if backward:
        print_rank_0("Step 5: Verifying gradients from backward pass...")

        # Check that at least some parameters have gradients
        total_params = 0
        params_with_grad = 0
        params_with_nonzero_grad = 0

        for m in megatron_models:
            for _name, param in m.named_parameters():
                if param.requires_grad:
                    total_params += 1
                    if param.grad is not None:
                        params_with_grad += 1
                        if param.grad.abs().sum() > 0:
                            params_with_nonzero_grad += 1

        print_rank_0(f"  Total trainable params: {total_params}")
        print_rank_0(f"  Params with gradient: {params_with_grad}")
        print_rank_0(f"  Params with non-zero gradient: {params_with_nonzero_grad}")

        # At least some params should have gradients
        # (Not all will have gradients due to PP - only the local stage's params)
        assert params_with_grad > 0, (
            f"No parameters have gradients! total_params={total_params}, params_with_grad={params_with_grad}"
        )

        # When vision is enabled, verify vision tower and resampler got gradients
        if with_vision:
            vision_params_with_grad = 0
            for m in megatron_models:
                for name, param in m.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if "vision_tower" in name or "resampler" in name:
                            if param.grad.abs().sum() > 0:
                                vision_params_with_grad += 1
            print_rank_0(f"  Vision params with non-zero gradient: {vision_params_with_grad}")
            assert vision_params_with_grad > 0, (
                "Vision tower/resampler parameters have no gradients! The vision forward path may not be exercised."
            )

        print_rank_0("  Gradient verification passed.")

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    print_rank_0("=== ALL CHECKS PASSED ===")


@torchrun_main
def _run(
    hf_model_path: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    seq_len: int = 16,
    forward_only: bool = False,
    with_vision: bool = False,
    prompt: str | None = None,
):
    """Entry point for torchrun-launched forward/backward test."""
    run_forward_backward(
        hf_model_path=hf_model_path,
        tp=tp,
        pp=pp,
        ep=ep,
        seq_len=seq_len,
        backward=not forward_only,
        with_vision=with_vision,
        prompt=prompt,
    )


def main():
    """Parse CLI arguments and launch the forward/backward test."""
    parser = argparse.ArgumentParser(description="ERNIE 4.5 VL MoE forward/backward test")
    parser.add_argument("--hf-model-path", required=True, help="Path to HF toy model directory")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--forward-only", action="store_true", help="Skip backward pass")
    parser.add_argument(
        "--with-vision",
        action="store_true",
        help="Include a dummy image input to exercise the vision tower and resampler",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Use real text prompt instead of random tokens for meaningful loss measurement",
    )
    args = parser.parse_args()

    _run(
        hf_model_path=args.hf_model_path,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        seq_len=args.seq_len,
        forward_only=args.forward_only,
        with_vision=args.with_vision,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
