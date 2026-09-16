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

"""HF Qwen4-Exp (toy, random) -> Megatron-Bridge -> GPTModel logits parity + HF export round trip.

Launch under ``torch.distributed.run`` with ``--nproc_per_node=<tp>``::

    python -m torch.distributed.run --nproc_per_node=2 qwen4_exp_parity.py --hf-model-path <dir> --tp 2

The toy model enables every Qwen4-Exp feature: the GDN / QSA hybrid, 4-stream gated residuals, a
PLE layer with a sharded n-gram table, and the MoE with a gated shared expert. It is compared in
fp32 (TE unfused attention) so that mismatches are visible above numerical noise.
"""

import argparse
import glob
import os
import sys

import torch
import torch.distributed as dist


HF_QWEN4_EXP_TOY_MODEL_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 16,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "hc_count": 4,
    "hc_lowrank": 8,
    "ple_layer_ids": [2],
    "ple_embed_dim": 64,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 97,
    "make_ngram_vocab_size_divisible_by": 8,
    "seed": 1234,
    "split_ngram_parts": 4,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 16,
    "indexer_budget": 8,
    "indexer_compress_ratio": 4,
    "output_gate_type": "sigmoid",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "tie_word_embeddings": False,
    "initializer_range": 0.05,
}


def build_toy_checkpoint(path: str, seed: int = 0):
    """Create and save a random text-only Qwen4-Exp toy model (fp32 safetensors)."""
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    torch.manual_seed(seed)
    config = Qwen4ExpTextConfig(**HF_QWEN4_EXP_TOY_MODEL_CONFIG, dtype="float32")
    model = Qwen4ExpForCausalLM(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Zero-initialized gains / PLE conv would hide mapping bugs: give them signal.
            if param.ndim > 0 and "A_log" not in name and "dt_bias" not in name and param.abs().max() == 0:
                param.normal_(0, 0.1)
            if "ple.conv1d" in name:
                param.normal_(0, 0.3)
    model.save_pretrained(path, safe_serialization=True)
    return model


def _load_safetensors(directory: str) -> dict:
    from safetensors.torch import load_file

    tensors = {}
    for file in glob.glob(os.path.join(directory, "*.safetensors")):
        tensors.update(load_file(file))
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--seq", type=int, default=40, help="sequence length (multiple of tp)")
    parser.add_argument("--force-sparse", action="store_true", help="always run the QSA sparse kernel")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--tolerance", type=float, default=2e-2, help="relative tolerance on logits")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    from transformers import Qwen4ExpForCausalLM

    hf_model = Qwen4ExpForCausalLM.from_pretrained(args.hf_model_path, dtype=torch.float32).cuda().eval()
    cfg = hf_model.config

    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend

    from megatron.bridge import AutoBridge

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=args.tp)
    model_parallel_cuda_manual_seed(123)

    bridge = AutoBridge.from_hf_pretrained(args.hf_model_path)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = args.tp
    provider.pipeline_model_parallel_size = 1
    provider.sequence_parallel = args.tp > 1
    provider.params_dtype = torch.float32
    provider.bf16 = False
    provider.pipeline_dtype = torch.float32
    provider.autocast_dtype = torch.float32
    provider.attention_backend = AttnBackend.unfused
    provider.gradient_accumulation_fusion = False
    provider.moe_router_dtype = "fp32"
    provider.variable_seq_lengths = True
    provider.qsa_force_sparse = args.force_sparse
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)
    gpt = model[0].eval()

    ok = True
    torch.manual_seed(7)
    b, s = 2, args.seq
    input_ids = torch.randint(3, cfg.vocab_size, (b, s), device="cuda")
    input_ids[0, 9] = cfg.eos_token_id  # exercise the n-gram context reset
    dist.broadcast(input_ids, 0)
    position_ids = torch.arange(s, device="cuda").unsqueeze(0).expand(b, -1)

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits.float()
        mg_logits = gpt(input_ids, position_ids, None, runtime_gather_output=True).float()
    scale = max(1.0, hf_logits.abs().max().item())
    diff = (hf_logits - mg_logits).abs().max().item()
    argmax_agreement = (hf_logits.argmax(-1) == mg_logits.argmax(-1)).float().mean().item()
    if rank == 0:
        print(
            f"[TP={args.tp}] bshd: max|diff|={diff:.3e} (logit range {scale:.3f}), argmax agreement {argmax_agreement:.3f}"
        )
    ok &= diff < args.tolerance * scale

    # Packed (THD) sequences: two documents, positions restart per document.
    lens = [s, s - 8]
    packed = torch.cat([input_ids[0], input_ids[1, : lens[1]]]).unsqueeze(0)
    cu_seqlens = torch.tensor([0, lens[0], lens[0] + lens[1]], device="cuda", dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(lens),
        max_seqlen_kv=max(lens),
    )
    packed_positions = torch.cat([torch.arange(lens[0]), torch.arange(lens[1])]).unsqueeze(0).cuda()
    with torch.no_grad():
        mg_thd = gpt(packed, packed_positions, None, packed_seq_params=packed_seq_params, runtime_gather_output=True)
        ref = torch.cat([hf_logits[0], hf_model(input_ids=input_ids[1:, : lens[1]]).logits.float()[0]], dim=0)
    diff_thd = (mg_thd.float()[0] - ref).abs().max().item()
    if rank == 0:
        print(f"[TP={args.tp}] thd: max|diff|={diff_thd:.3e}")
    ok &= diff_thd < args.tolerance * scale

    # Backward: every tensor-parallel rank must run the same collectives (the PLE halo exchange
    # and the vocab-parallel n-gram table both take part in autograd); compare a parameter
    # gradient against HF to check the gated-residual / QSA / PLE backward paths.
    gpt.train()
    hf_model.train()
    mg_out = gpt(input_ids, position_ids, None, runtime_gather_output=True).float()
    hf_out = hf_model(input_ids=input_ids).logits.float()
    targets = torch.roll(input_ids, -1, dims=1)
    mg_loss = torch.nn.functional.cross_entropy(mg_out.reshape(-1, mg_out.shape[-1]), targets.reshape(-1))
    hf_loss = torch.nn.functional.cross_entropy(hf_out.reshape(-1, hf_out.shape[-1]), targets.reshape(-1))
    mg_loss.backward()
    hf_loss.backward()
    mg_grad = gpt.embedding.word_embeddings.weight.grad
    hf_grad = hf_model.model.embed_tokens.weight.grad
    # Vocab-parallel embedding: compare this rank's rows.
    rows = hf_grad.shape[0] // args.tp
    hf_rows = hf_grad[rank * rows : (rank + 1) * rows]
    grad_diff = (mg_grad.float() - hf_rows).abs().max().item()
    grad_scale = max(1e-6, hf_rows.abs().max().item())
    if rank == 0:
        print(
            f"[TP={args.tp}] backward: loss mg={mg_loss.item():.5f} hf={hf_loss.item():.5f}, "
            f"embedding grad max|diff|={grad_diff:.3e} (grad range {grad_scale:.3e})"
        )
    ok &= abs(mg_loss.item() - hf_loss.item()) < args.tolerance and grad_diff < 5e-2 * grad_scale
    gpt.eval()
    hf_model.eval()

    # Export: Megatron -> HF safetensors must reproduce the source tensors.
    export_dir = args.export_dir or (args.hf_model_path.rstrip("/") + f"_export_tp{args.tp}")
    bridge.save_hf_weights(model, export_dir, show_progress=False)
    dist.barrier()
    if rank == 0:
        source, exported = _load_safetensors(args.hf_model_path), _load_safetensors(export_dir)
        missing = sorted(set(source) - set(exported))
        worst = max(
            ((exported[n].float() - t.float()).abs().max().item() for n, t in source.items() if n in exported),
            default=0.0,
        )
        print(f"[TP={args.tp}] export: {len(exported)} tensors, missing={len(missing)}, max|diff|={worst:.3e}")
        ok &= not missing and worst < 1e-6
        print("PARITY", "OK" if ok else "FAILED")
    ok_tensor = torch.tensor(int(ok), device="cuda")
    dist.broadcast(ok_tensor, 0)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok_tensor.item() else 1


if __name__ == "__main__":
    sys.exit(main())
