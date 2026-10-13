# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run bounded checkpoint-initialized Kimi K3 training and record rank diagnostics."""

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from runtime_controls import install_controls

from megatron.bridge import AutoBridge


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Read explicit checkpoint, token-stream and training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=128)
    parser.add_argument("--first-stage-layers", type=int, default=24)
    parser.add_argument("--peak-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--decay-steps", type=int, default=10000)
    parser.add_argument("--stop-after-seconds", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    """Import once, allocate distributed Adam, and attempt real RP2 updates."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "rank": rank,
        "node": os.uname().nodename,
        "passed": False,
        "steps": [],
        "memory": [],
        "independent_loss_reference_qualified": False,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    started = time.monotonic()

    def save() -> None:
        temporary = args.output / f"training-rank-{rank}.tmp"
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(args.output / f"training-rank-{rank}.json")

    def memory(phase: str) -> None:
        free, total = torch.cuda.mem_get_info()
        entry = {
            "phase": phase,
            "seconds": time.monotonic() - started,
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "free": free,
            "total": total,
        }
        result["memory"].append(entry)
        logger.info("rank=%s memory=%s", rank, entry)
        save()

    try:
        torch.cuda.set_device(local_rank)
        assert args.steps > 0 and args.microbatches > 0 and args.seq_length >= 4096
        assert 0 < args.warmup_steps < args.decay_steps
        assert os.environ.get("NCCL_PXN_DISABLE") == "1", "Validated transport control is required"
        verify_controls = install_controls(args.output, rank)
        dist.init_process_group("nccl", timeout=timedelta(minutes=12))
        bridge = AutoBridge.from_hf_pretrained(args.snapshot, trust_remote_code=True, local_files_only=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        overrides = dict(
            tensor_model_parallel_size=args.tp,
            pipeline_model_parallel_size=args.pp,
            expert_model_parallel_size=args.ep,
            expert_tensor_parallel_size=1,
            context_parallel_size=1,
            num_layers_in_first_pipeline_stage=args.first_stage_layers,
            sequence_parallel=True,
            seq_length=args.seq_length,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            fp16=False,
            gradient_accumulation_fusion=False,
            use_cpu_initialization=False,
            perform_initialization=False,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        for name, value in overrides.items():
            assert hasattr(provider, name), name
            setattr(provider, name, value)
        provider.finalize()
        provider.initialize_model_parallel(seed=1234)
        result["parallelism"] = {
            "TP": ps.get_tensor_model_parallel_world_size(),
            "PP": ps.get_pipeline_model_parallel_world_size(),
            "DP": ps.get_data_parallel_world_size(),
            "EP": ps.get_expert_model_parallel_world_size(),
        }
        result["global_batch_size"] = args.microbatches * ps.get_data_parallel_world_size()
        memory("before_model")

        def load_checkpoint(models: list) -> list:
            with torch.no_grad():
                for chunk in models:
                    for parameter in chunk.parameters():
                        parameter.fill_(float("nan"))
            bridge.load_hf_weights(models)
            with torch.no_grad():
                for chunk in models:
                    for name, parameter in chunk.named_parameters():
                        assert torch.isfinite(parameter).all().item(), name
            result["all_checkpoint_parameters_finite"] = True
            memory("checkpoint_loaded_before_DDP")
            return models

        models = provider.provide_distributed_model(
            wrap_with_ddp=True,
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                use_distributed_optimizer=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                check_for_nan_in_grad=True,
            ),
            pre_wrap_hook=load_checkpoint,
        )
        bridge = None
        gc.collect()
        torch.cuda.empty_cache()
        model = models[0]
        model.train()
        memory("DDP_and_gradient_buffers")
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=args.peak_lr / args.warmup_steps,
                min_lr=0.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-8,
                weight_decay=0.1,
                clip_grad=1.0,
            ),
            models,
        )
        provider.finalize_model_grads_func = finalize_model_grads
        provider.grad_scale_func = optimizer.scale_loss
        memory("optimizer_constructed")
        # The preparation pipeline writes an int32 stream with EOS between RP2 documents.
        stream = np.memmap(args.data_prefix + ".bin", mode="r", dtype=np.int32)
        dp_rank, dp_size = ps.get_data_parallel_rank(), ps.get_data_parallel_world_size()
        required_tokens = args.steps * dp_size * args.microbatches * args.seq_length + 1
        assert len(stream) >= required_tokens, (len(stream), required_tokens)
        schedule = get_forward_backward_func()
        for step in range(args.steps):
            remaining = torch.tensor(int(time.monotonic() - started < args.stop_after_seconds), device="cuda")
            dist.all_reduce(remaining, op=dist.ReduceOp.MIN)
            if not remaining.item():
                result["stopped_for_walltime"] = True
                break
            step_started = time.monotonic()
            lr_fraction = (
                (step + 1) / args.warmup_steps
                if step < args.warmup_steps
                else 0.5
                * (
                    1
                    + math.cos(
                        math.pi * min(1.0, (step + 1 - args.warmup_steps) / (args.decay_steps - args.warmup_steps))
                    )
                )
            )
            lr = args.peak_lr * lr_fraction
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_mult", 1.0)
            samples = []
            for microbatch in range(args.microbatches):
                offset = ((step * dp_size + dp_rank) * args.microbatches + microbatch) * args.seq_length
                values = np.array(stream[offset : offset + args.seq_length + 1], dtype=np.int64)
                assert values.min() >= 0 and values.max() < provider.vocab_size
                samples.append(torch.from_numpy(values).cuda()[None])
            sample_hash = hashlib.sha256(b"".join(x.cpu().numpy().tobytes() for x in samples)).hexdigest()
            positions = torch.arange(args.seq_length, device="cuda")[None]
            iterator = iter(samples)

            def forward_step(data_iterator, chunk):
                tokens = next(data_iterator)
                losses = chunk(tokens[:, :-1], positions, None, labels=tokens[:, 1:])

                def loss_fn(value):
                    loss = value.float().mean()
                    assert torch.isfinite(loss).item(), "Nonfinite LM loss"
                    return loss, {"lm_loss": loss.detach().clone()}

                return losses, loss_fn

            model.zero_grad_buffer()
            optimizer.zero_grad()
            before = {
                name: parameter.detach().reshape(-1)[:64].clone() for name, parameter in model.named_parameters()
            }
            losses = schedule(
                forward_step_func=forward_step,
                data_iterator=iterator,
                model=models,
                num_microbatches=args.microbatches,
                seq_length=args.seq_length,
                micro_batch_size=1,
                forward_only=False,
            )
            memory(f"step_{step}_backward")
            grad_count = 0
            for name, parameter in model.named_parameters():
                gradient = getattr(parameter, "main_grad", None)
                if gradient is not None:
                    assert torch.isfinite(gradient).all().item(), name
                    grad_count += 1
            assert grad_count > 0
            success, grad_norm, zeros = optimizer.step()
            assert success and grad_norm is not None and math.isfinite(float(grad_norm)) and float(grad_norm) > 0, (
                success,
                grad_norm,
            )
            memory(f"step_{step}_Adam_update")
            changed = 0
            for name, parameter in model.named_parameters():
                assert torch.isfinite(parameter).all().item(), name
                changed += int(not torch.equal(before[name], parameter.detach().reshape(-1)[:64]))
            changed_global = torch.tensor(changed, device="cuda", dtype=torch.long)
            dist.all_reduce(changed_global)
            assert changed_global.item() > 0, "No sampled model parameter changed"
            local_loss = sum(float(value["lm_loss"]) for value in losses) / len(losses) if losses else 0.0
            aggregate = torch.tensor([local_loss, float(bool(losses))], device="cuda", dtype=torch.float64)
            dist.all_reduce(aggregate)
            assert aggregate[1].item() > 0
            global_loss = float((aggregate[0] / aggregate[1]).item())
            selected = verify_controls()
            row = {
                "step": step,
                "sequence_length": args.seq_length,
                "learning_rate": lr,
                "lm_loss": global_loss,
                "grad_norm": float(grad_norm),
                "zero_gradients": float(zeros) if zeros is not None else None,
                "gradient_tensors_checked": grad_count,
                "changed_parameter_samples_local": changed,
                "changed_parameter_samples_global": int(changed_global.item()),
                "input_hash": sample_hash,
                "seconds": time.monotonic() - step_started,
                "kernel_configs": selected,
            }
            result["steps"].append(row)
            logger.info("rank=%s training_step=%s", rank, row)
            save()
        assert result["steps"], "No training step completed"
        result.update(
            passed=True,
            completed=True,
            successful_steps=len(result["steps"]),
            target_steps_completed=len(result["steps"]) == args.steps,
        )
        memory("finished")
        dist.barrier()
    except Exception as exc:
        result["error"] = repr(exc)
        result["oom"] = isinstance(exc, torch.cuda.OutOfMemoryError)
        logger.exception("Kimi K3 training failed on rank %s", rank)
        save()
        # Let Slurm terminate peers instead of waiting through distributed teardown.
        os._exit(1)
    finally:
        save()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
