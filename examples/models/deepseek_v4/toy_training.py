#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small DeepSeek-V4 training smoke used for optimizer/precision bring-up.

The model is built from the real DeepSeek-V4 Flash HF config, but shrunk to a
few layers and small dimensions. It still keeps the DSv4-specific surfaces that
matter for this smoke: hybrid CSA, DSA indexer, hyper-connections, hash-routed
MoE, and one MTP layer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
    distributed_muon_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import bf16_mixed, bf16_with_mxfp8_mixed
from megatron.bridge.training.pretrain import pretrain


CASES = ("adam", "muon", "adam_mxfp8", "muon_mxfp8")


def _print_rank0(message: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(message, flush=True)


def _make_hf_config(hf_config_path: str, seq_length: int, model_size: str):
    hf_cfg = AutoConfig.from_pretrained(hf_config_path, trust_remote_code=True)

    if model_size == "full":
        return hf_cfg

    # Keep major dimensions aligned to 32 so MXFP8 param paths have legal shapes.
    overrides = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 3,
        "hidden_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "o_groups": 4,
        "qk_rope_head_dim": 64,
        "vocab_size": 1024,
        "actual_vocab_size": 1024,
        "max_position_embeddings": max(seq_length, 1024),
        "n_routed_experts": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 128,
        "num_hash_layers": 1,
        "num_nextn_predict_layers": 1,
        # Main layers: window-only, learned CSA indexer, all-compressed CSA.
        # MTP layer: window-only, so the MTP path is present without adding a second indexer.
        "compress_ratios": [0, 4, 128, 0],
        "sliding_window": 32,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 16,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 8,
        "routed_scaling_factor": 1.5,
        "norm_topk_prob": True,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "swiglu_limit": 10.0,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.006,
    }
    for key, value in overrides.items():
        setattr(hf_cfg, key, value)

    hf_cfg.rope_theta = 10000
    hf_cfg.compress_rope_theta = 160000
    hf_cfg.rope_scaling = {
        "type": "yarn",
        "factor": 1.0,
        "original_max_position_embeddings": max(seq_length, 1024),
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    return hf_cfg


def _pipeline_layout(num_layers: int, mtp_num_layers: int, pp_size: int) -> str | None:
    if pp_size <= 1:
        return None

    base_layers, extra_layers = divmod(num_layers, pp_size)
    stages = []
    for pp_rank in range(pp_size):
        stage = "E" if pp_rank == 0 else ""
        decoder_layers = base_layers + int(pp_rank < extra_layers)
        if decoder_layers == 1:
            stage += "t"
        elif decoder_layers > 1:
            stage += f"t*{decoder_layers}"

        if pp_rank == pp_size - 1:
            if mtp_num_layers == 1:
                stage += "m"
            elif mtp_num_layers > 1:
                stage += f"m*{mtp_num_layers}"
            stage += "L"
        stages.append(stage)
    return "|".join(stages)


def _build_model_provider(args: argparse.Namespace):
    hf_cfg = _make_hf_config(args.hf_config_path, args.seq_length, args.model_size)
    provider = AutoBridge.from_hf_config(hf_cfg).to_megatron_provider(load_weights=False)

    provider.tensor_model_parallel_size = args.tensor_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_parallel_size
    provider.pipeline_model_parallel_layout = _pipeline_layout(
        provider.num_layers,
        provider.mtp_num_layers,
        args.pipeline_parallel_size,
    )
    provider.context_parallel_size = args.context_parallel_size
    provider.expert_model_parallel_size = args.expert_parallel_size
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = args.tensor_parallel_size > 1
    provider.seq_length = args.seq_length
    if args.model_size == "toy":
        provider.max_position_embeddings = max(args.seq_length, 1024)
    provider.pipeline_dtype = torch.bfloat16
    provider.params_dtype = torch.bfloat16

    provider.apply_rope_fusion = args.use_fused_kernels
    provider.use_fused_mhc = args.use_fused_kernels
    provider.csa_backend = args.csa_backend
    if args.model_size == "toy":
        provider.csa_window_size = 32
        provider.csa_compress_rotary_base = 160000.0
    provider.dsa_indexer_loss_coeff = 0.0
    provider.dsa_indexer_use_sparse_loss = True

    provider.moe_token_dispatcher_type = args.moe_token_dispatcher_type
    provider.moe_grouped_gemm = True
    provider.moe_permute_fusion = True
    provider.moe_router_fusion = False
    provider.moe_router_force_load_balancing = False
    provider.moe_router_padding_for_fp8 = False
    provider.moe_shared_expert_overlap = False
    provider.moe_aux_loss_coeff = 0.0

    provider.transformer_impl = "transformer_engine"
    provider.attention_backend = None
    provider.cross_entropy_loss_fusion = True
    provider.cross_entropy_fusion_impl = "te"
    if args.model_size == "full":
        provider.recompute_granularity = "selective"
        provider.recompute_modules = ["moe_act", "mhc"]
    else:
        provider.recompute_granularity = None
        provider.recompute_modules = None
    provider.recompute_method = None
    provider.recompute_num_layers = None
    provider.fine_grained_activation_offloading = False
    provider.offload_modules = None
    provider.cuda_graph_impl = "none"
    return provider


def _configure_optimizer_and_precision(cfg, args: argparse.Namespace) -> None:
    case = args.case
    if case.startswith("adam"):
        optimizer_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
            lr_warmup_iters=1,
            lr_decay_iters=args.train_iters,
            max_lr=args.adam_max_lr,
            min_lr=args.adam_min_lr,
            weight_decay=args.adam_weight_decay,
            clip_grad=1.0,
        )
        cfg.ddp.use_distributed_optimizer = True
        cfg.ddp.overlap_param_gather = True
        cfg.ddp.grad_reduce_in_fp32 = False
        cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        if args.model_size == "full":
            optimizer_cfg.use_precision_aware_optimizer = True
            optimizer_cfg.main_grads_dtype = torch.float32
            optimizer_cfg.main_params_dtype = torch.float32
            optimizer_cfg.exp_avg_dtype = torch.bfloat16
            optimizer_cfg.exp_avg_sq_dtype = torch.bfloat16
            cfg.ddp.overlap_param_gather = False
            cfg.ddp.overlap_grad_reduce = False
    else:
        optimizer_cfg, scheduler_cfg = distributed_muon_with_cosine_annealing(
            muon_momentum=args.muon_momentum,
            muon_use_nesterov=args.muon_use_nesterov,
            muon_scale_mode=args.muon_scale_mode,
            muon_fp32_matmul_prec=args.muon_fp32_matmul_prec,
            muon_num_ns_steps=args.muon_num_ns_steps,
            muon_extra_scale_factor=args.muon_extra_scale_factor,
            lr_warmup_iters=1,
            lr_decay_iters=args.train_iters,
            max_lr=args.muon_max_lr,
            min_lr=args.muon_min_lr,
            weight_decay=args.muon_weight_decay,
            clip_grad=1.0,
        )
        optimizer_cfg.optimizer = args.muon_optimizer_name
        optimizer_cfg.muon_coefficient_type = args.muon_coefficient_type
        # Match existing Bridge Muon recipes: Muon uses regular DDP, no dist optimizer.
        cfg.ddp.use_distributed_optimizer = False
        cfg.ddp.overlap_param_gather = False
        cfg.ddp.grad_reduce_in_fp32 = True
        cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    cfg.optimizer = optimizer_cfg
    cfg.scheduler = scheduler_cfg
    if not (case.startswith("adam") and args.model_size == "full"):
        cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_megatron_fsdp = False

    if case.endswith("mxfp8"):
        cfg.mixed_precision = bf16_with_mxfp8_mixed()
    else:
        cfg.mixed_precision = bf16_mixed()

    if case.startswith("muon"):
        cfg.mixed_precision.grad_reduce_in_fp32 = True


def build_config(args: argparse.Namespace):
    cfg = _pretrain_common()
    cfg.model = _build_model_provider(args)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.seq_length = args.seq_length
    cfg.dataset.num_workers = 0
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.dataset.dataloader_type = "single"

    cfg.train.train_iters = args.train_iters
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 1
    cfg.train.manual_gc_eval = 1
    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0

    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = str(Path(args.output_dir) / args.case / "tb")

    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = None

    cfg.dist.enable_megatron_core_experimental = True
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    _configure_optimizer_and_precision(cfg, args)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--model-size", choices=("toy", "full"), default="toy")
    parser.add_argument("--hf-config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--train-iters", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=4)
    parser.add_argument("--csa-backend", choices=("unfused", "cudnn_dsa"), default="cudnn_dsa")
    parser.add_argument(
        "--moe-token-dispatcher-type",
        choices=("alltoall", "allgather", "alltoall_seq", "flex"),
        default="alltoall",
    )
    parser.add_argument("--adam-max-lr", type=float, default=1e-4)
    parser.add_argument("--adam-min-lr", type=float, default=1e-5)
    parser.add_argument("--adam-weight-decay", type=float, default=0.01)
    parser.add_argument("--muon-max-lr", type=float, default=1e-6)
    parser.add_argument("--muon-min-lr", type=float, default=1e-7)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-scale-mode", choices=("shape_scaling", "spectral", "unit_rms_norm"), default="unit_rms_norm")
    parser.add_argument("--muon-fp32-matmul-prec", choices=("highest", "high", "medium"), default="highest")
    parser.add_argument("--muon-num-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-coefficient-type",
        choices=("simple", "quintic", "polar_express", "aol"),
        default="quintic",
    )
    parser.add_argument("--muon-optimizer-name", choices=("muon", "dist_muon"), default="muon")
    parser.add_argument("--muon-extra-scale-factor", type=float, default=0.2)
    parser.add_argument("--muon-use-nesterov", dest="muon_use_nesterov", action="store_true")
    parser.add_argument("--no-muon-use-nesterov", dest="muon_use_nesterov", action="store_false")
    parser.set_defaults(muon_use_nesterov=False)
    parser.add_argument("--use-fused-kernels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    _print_rank0(
        "DSv4 training smoke: "
        f"model_size={args.model_size} case={args.case} "
        f"layers={cfg.model.num_layers} mtp={cfg.model.mtp_num_layers} "
        f"hidden={cfg.model.hidden_size} heads={cfg.model.num_attention_heads} "
        f"experts={cfg.model.num_moe_experts} topk={cfg.model.moe_router_topk} "
        f"compress={cfg.model.csa_compress_ratios} optimizer={cfg.optimizer.optimizer} "
        f"lr={cfg.optimizer.lr} min_lr={cfg.optimizer.min_lr} "
        f"csa_backend={cfg.model.csa_backend} "
        f"pp_layout={cfg.model.pipeline_model_parallel_layout} "
        f"fp8={getattr(cfg.mixed_precision, 'fp8', None)} "
        f"fp8_recipe={getattr(cfg.mixed_precision, 'fp8_recipe', None)}"
    )
    pretrain(config=cfg, forward_step_func=forward_step)
    _print_rank0(f"DSv4 training smoke model_size={args.model_size} case={args.case}: PASS")


if __name__ == "__main__":
    main()
