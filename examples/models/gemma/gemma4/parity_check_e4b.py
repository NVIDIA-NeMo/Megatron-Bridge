#!/usr/bin/env python3
"""
Logit parity check: Megatron Gemma-4 E4B vs HF Gemma-4 E4B.

Loads the converted Megatron checkpoint (TP=2), runs a forward pass, gathers
the full vocab logits from both ranks, then on rank 0 runs the same tokens
through the HF model and reports max/mean absolute difference.

Run from Megatron-Bridge root via:
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 \
        examples/models/gemma/gemma4/parity_check_e4b.py \
        --hf-dir ~/models/gemma-4-E4B-it \
        --megatron-ckpt /path/to/gemma4-e4b-megatron
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
MEGATRON_LM_ROOT = os.environ.get("MEGATRON_LM_ROOT", os.getcwd())

sys.path.insert(0, os.path.join(BRIDGE_ROOT, "src"))
sys.path.insert(0, MEGATRON_LM_ROOT)

import torch
import torch.distributed as dist

SEQ = 16
BATCH = 1
FULL_VOCAB = 262144       # HF vocab size
LOGIT_SOFTCAP = 30.0     # Gemma-4 final_logit_softcapping


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--megatron-ckpt", required=True)
    p.add_argument("--atol", type=float, default=1.0,
                   help="Max absolute logit difference. ~1.0 is typical for bf16.")
    p.add_argument("--tp", type=int, default=2, choices=[1, 2],
                   help="Tensor parallel size.")
    p.add_argument("--bf16", action="store_true",
                   help="Use bf16 (default: float32).")
    return p.parse_args()


def _build_megatron_argv(ckpt, tp=2, bf16=False):
    return [
        "parity",
        "--use-mcore-models",
        "--num-layers", "42", "--hidden-size", "2560",
        "--ffn-hidden-size", "10240", "--num-attention-heads", "8",
        "--group-query-attention", "--num-query-groups", "2",
        "--kv-channels", "256", "--global-kv-channels", "512",
        "--num-global-query-groups", "2",
        "--seq-length", str(SEQ), "--max-position-embeddings", "131072",
        "--position-embedding-type", "rope", "--rotary-percent", "1.0",
        "--sliding-window-rope-base", "10000",
        "--full-attention-rope-base", "1000000",
        "--full-attention-rope-partial-factor", "0.25",
        "--window-size", "511,0", "--window-attn-skip-freq", "6",
        "--num-kv-shared-layers", "18",
        "--geglu-tanh", "--normalization", "RMSNorm", "--norm-epsilon", "1e-6",
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--disable-bias-linear",
        "--vocab-size", "262143", "--make-vocab-size-divisible-by", "128",
        "--scale-embeddings-by-hidden-size",
        "--per-layer-embed-vocab-size", "262144", "--per-layer-embed-dim", "256",
        "--spec", "megatron.bridge.models.gemma.gemma4_layer_specs", "gemma4_layer_spec",
        "--transformer-impl", "local", "--attention-backend", "unfused",
        "--tensor-model-parallel-size", str(tp), "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--no-rope-fusion", "--no-persist-layer-norm", "--no-masked-softmax-fusion",
        "--no-gradient-accumulation-fusion",
        "--load", ckpt, "--finetune", "--no-load-optim", "--no-load-rng",
        "--init-method-std", "0.02",
        "--micro-batch-size", str(BATCH), "--global-batch-size", str(BATCH),
        "--train-iters", "1",
        "--tokenizer-type", "NullTokenizer", "--mock-data",
        "--no-create-attention-mask-in-dataloader", "--no-mmap-bin-files",
        "--num-workers", "0", "--lr", "1e-4",
        "--distributed-timeout-minutes", "10",
        "--log-interval", "1", "--eval-iters", "0", "--eval-interval", "1000",
        "--no-save-optim", "--no-save-rng",
    ] + (["--bf16"] if bf16 else [])


def main():
    args = _parse()

    pretrain_gpt = os.path.join(MEGATRON_LM_ROOT, "pretrain_gpt.py")
    if not os.path.isfile(pretrain_gpt):
        sys.exit(f"Error: Megatron-LM root not found: {MEGATRON_LM_ROOT}")
    os.chdir(MEGATRON_LM_ROOT)

    sys.argv = _build_megatron_argv(args.megatron_ckpt, tp=args.tp, bf16=args.bf16)

    from megatron.core import mpu
    from megatron.core.enums import ModelType
    from megatron.training import get_model
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron

    parse_and_validate_args()
    initialize_megatron()
    rank = dist.get_rank()

    from functools import partial

    from gpt_builders import gpt_builder
    from pretrain_gpt import model_provider
    models = get_model(partial(model_provider, gpt_builder), ModelType.encoder_or_decoder)
    model = models[0]

    # gpt_model.py calls wire_gemma4_kv_sharing from megatron.core, but this parity
    # script uses the Bridge spec whose Gemma4SelfAttention is a different class.
    # Re-wire explicitly using the Bridge's version so isinstance() matches.
    from megatron.bridge.models.gemma.gemma4_layer_specs import wire_gemma4_kv_sharing
    wire_gemma4_kv_sharing(model)

    load_checkpoint(models, None, None)
    model.eval()

    # Fixed tokens for reproducibility: [0, 1, 2, ..., SEQ-1]
    tokens = torch.arange(SEQ, dtype=torch.long).unsqueeze(0).cuda()  # [1, SEQ]

    with torch.no_grad():
        out = model(input_ids=tokens, position_ids=None, attention_mask=None)

    logits = out[0] if isinstance(out, tuple) else out
    # mcore GPTModel returns [batch, seq, vocab/tp]; handle seq-first just in case
    if logits.shape[0] == SEQ and logits.shape[1] == BATCH:
        logits = logits.permute(1, 0, 2)

    # All-gather vocab shard from each TP rank
    tp = mpu.get_tensor_model_parallel_world_size()
    if tp > 1:
        parts = [torch.zeros_like(logits) for _ in range(tp)]
        dist.all_gather(parts, logits.contiguous(),
                        group=mpu.get_tensor_model_parallel_group())
        logits = torch.cat(parts, dim=-1)  # [BATCH, SEQ, full_vocab_padded]

    # Gemma-4 applies final_logit_softcapping in HF but Megatron doesn't implement it yet.
    # Apply it here so both sides are compared at the same level.
    raw_megatron = logits[..., :FULL_VOCAB].cpu().float()
    megatron_logits = torch.tanh(raw_megatron / LOGIT_SOFTCAP) * LOGIT_SOFTCAP

    del model, models, logits, out
    torch.cuda.empty_cache()

    # Broadcast FAIL signal from rank 0 so all ranks exit cleanly together.
    fail_flag = torch.tensor([0], dtype=torch.int32).cuda()

    if rank == 0:
        from transformers import AutoModelForCausalLM
        print(f"\nLoading HF model from {args.hf_dir} ...")
        hf_dtype = torch.bfloat16 if args.bf16 else torch.float32
        hf = AutoModelForCausalLM.from_pretrained(
            args.hf_dir, torch_dtype=hf_dtype, device_map="cuda:0"
        )
        hf.eval()
        with torch.no_grad():
            hf_logits = hf(input_ids=tokens, output_hidden_states=False).logits
        hf_logits = hf_logits[..., :FULL_VOCAB].cpu().float()
        del hf
        torch.cuda.empty_cache()

        diff = (megatron_logits - hf_logits).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Show top-3 positions with highest per-token max diff
        per_token_max = diff[0].max(dim=-1).values  # [SEQ]
        top3 = per_token_max.topk(3)

        print(f"\n{'='*60}")
        print(f"  Parity: Megatron Gemma-4 E4B  vs  HF Gemma-4 E4B")
        print(f"  (Megatron logits softcapped at {LOGIT_SOFTCAP} before comparison)")
        print(f"  seq={SEQ}  batch={BATCH}  vocab={FULL_VOCAB}")
        print(f"  max |diff|  : {max_diff:.6f}  (atol={args.atol})")
        print(f"  mean |diff| : {mean_diff:.6f}")
        print(f"  worst token positions: {top3.indices.tolist()} "
              f"(diffs: {[f'{v:.4f}' for v in top3.values.tolist()]})")
        status = "PASSED" if max_diff <= args.atol else "FAILED"
        print(f"  --> {status}")
        print(f"{'='*60}\n")

        if status == "FAILED":
            fail_flag.fill_(1)

    dist.broadcast(fail_flag, src=0)
    dist.barrier()
    if fail_flag.item() == 1:
        sys.exit(1)


if __name__ == "__main__":
    main()
