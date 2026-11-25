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

"""Observe one native GA8 microbatch from a frozen, shared Hugging Face model state."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    """Capture the first native GA8 microbatch from the requested shared model state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf",
        type=Path,
        required=True,
        help="HF model-only checkpoint with the step-two weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory on a shared filesystem.",
    )
    parser.add_argument(
        "--reference-router-bias",
        type=Path,
        help="Optional CPU tensor dict keyed by HF router names.",
    )
    parser.add_argument("--index-sha256", help="Expected SHA256 of the shared HF shard index.")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Pass recipe overrides following --.",
    )
    args = parser.parse_args()
    hf, output = args.hf.resolve(), args.output.resolve()
    index_hash = hashlib.sha256((hf / "model.safetensors.index.json").read_bytes()).hexdigest()
    if args.index_sha256 and args.index_sha256 != index_hash:
        parser.error("The HF index differs from the requested shared checkpoint.")
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides

    import megatron.bridge
    from megatron.bridge.recipes.ernie import (
        ernie55_pretrain_256k_64gpu_gb300_bf16_config,
    )

    repo = Path(megatron.bridge.__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts/training"))
    from recipe_runner import apply_cli_overrides, bootstrap_recipe_environment

    config = ernie55_pretrain_256k_64gpu_gb300_bf16_config(hf_path=str(hf))
    config = apply_cli_overrides(config, overrides)
    config = bootstrap_recipe_environment(config, script_path=str(Path(__file__).resolve()), argv=sys.argv[1:])

    import numpy as np
    import torch
    import torch.distributed as dist
    from head_parity_256k import install
    from shared_state_weights_256k import WeightAudit

    from megatron.bridge import AutoBridge
    from megatron.bridge.data.utils import get_dataset_provider
    from megatron.bridge.models.ernie import ernie55_step
    from megatron.bridge.training.config import runtime_config_update
    from megatron.bridge.training.setup import setup
    from megatron.bridge.training.state import GlobalState
    from megatron.bridge.training.train import _finish_train, train

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    topology = (
        int(os.environ["WORLD_SIZE"]),
        config.model.context_parallel_size,
        config.model.expert_model_parallel_size,
        config.model.tensor_model_parallel_size,
        config.model.pipeline_model_parallel_size,
        config.train.global_batch_size,
        config.train.micro_batch_size,
    )
    if topology != (64, 16, 64, 1, 1, 32, 1):
        raise ValueError(f"Expected 64 GPUs, CP16 EP64 TP1 PP1 GBS32 MBS1, received {topology}")
    if any(
        (
            config.checkpoint.pretrained_checkpoint,
            config.checkpoint.load,
            config.checkpoint.save,
        )
    ):
        raise ValueError("Set all three checkpoint paths to null when reading a shared HF model directly.")
    if config.train.train_iters != 1 or not config.dataset.pack or config.model.cp_partition_mode != "zigzag":
        raise ValueError("This example needs a single scheduled iteration and fixed CP16 zigzag packed inputs.")
    receipt = output / "receipt"
    if (receipt / f"mapping-{rank:02}.json.gz").exists() or (output / "heads" / f"rank{rank:02}.npz").exists():
        raise FileExistsError(f"Provide a fresh shared-state output directory: {output}")
    adjoints = {}

    def observe(name, tensor, labels):
        if name in adjoints or not tensor.requires_grad:
            raise RuntimeError(f"Unexpected {name} head loss in frozen-state capture")
        valid = torch.as_tensor(np.asarray(labels).reshape(-1) >= 0, device=tensor.device)

        def record(gradient):
            g = gradient.detach().reshape(-1)
            if g.shape != valid.shape or not torch.isfinite(g).all():
                raise ValueError(f"Invalid {name} CE cotangent")

            def summary(x):
                values, counts = torch.unique(x, return_counts=True)
                return dict(
                    sum=x.double().sum().item(),
                    values=values.float().tolist(),
                    counts=counts.tolist(),
                )

            adjoints[name] = dict(
                dtype=str(g.dtype),
                physical=g.numel(),
                valid=summary(g[valid]),
                ignored=summary(g[~valid]),
            )

        tensor.register_hook(record)

    audit = None

    def load(model):
        nonlocal audit
        auto = AutoBridge.from_hf_pretrained(str(hf), torch_dtype=torch.bfloat16)
        audit = WeightAudit(auto, hf, receipt)
        audit.load(model)
        if args.reference_router_bias:
            source = torch.load(args.reference_router_bias, map_location="cpu", weights_only=True)
            if source.keys() != audit.bias.keys() or any(
                not torch.equal(v, audit.bias[k].detach().cpu()) for k, v in source.items()
            ):
                raise ValueError("A loaded router bias differs from the live exporting trainer.")
        return model

    # Run after distributed wrapping and before optimizer construction. The optimizer cannot
    # later populate its working shards from unrelated initialized weights.
    config.model.post_wrap_hooks.append(load)
    runtime_config_update(config)
    state = GlobalState()
    state.cfg = config
    initialized = setup(state, get_dataset_provider(config.dataset))
    if audit is None:
        raise RuntimeError("The distributed model builder did not call its post-wrap hook.")
    save = install(output / "heads", model_parts=initialized.model, observe_ce_gradient=observe)
    native_batch = ernie55_step.get_batch
    calls = 0

    class CapturedMicrobatch(Exception):
        pass

    def next_batch(*args, **kwargs):
        nonlocal calls
        if calls == 0:
            calls += 1
            audit.check_bias("before-forward")
            batch = native_batch(*args, **kwargs)
            bounds = batch.packed_seq_params.cu_seqlens_q_padded.detach().cpu().tolist()
            if bounds != list(range(0, 262145, 32768)):
                raise ValueError(f"Expected exactly eight 32768-token documents, received boundaries {bounds}")
            return batch
        # With PP1 the next batch read follows the first complete native backward. The
        # schedule used its real eight-microbatch normalization throughout that backward.
        save()
        if set(adjoints) != {"main", "mtp"}:
            raise RuntimeError("A head's CE cotangent was never observed.")
        with (receipt / f"adjoint-{rank:02}.json").open("x") as stream:
            json.dump(adjoints, stream)
        raise CapturedMicrobatch

    ernie55_step.get_batch = next_batch
    try:
        train(
            ernie55_step.forward_step,
            initialized.model,
            initialized.optimizer,
            initialized.scheduler,
            initialized.train_data_iterator,
            initialized.valid_data_iterator,
            state,
            initialized.checkpoint_manager,
            initialized.pg_collection,
        )
    except CapturedMicrobatch:
        if state.train_state.step != 0:
            raise RuntimeError("An optimizer update occurred during the frozen-state probe.")
        dist.barrier()
        if rank == 0 and len(list((output / "heads").glob("rank*.npz"))) != 64:
            raise RuntimeError("Missing head captures after all ranks finished.")
        if state.wandb_logger:
            import wandb

            if wandb.run is not None:
                (output / "shared-wandb.json").write_text(json.dumps(dict(id=wandb.run.id, url=wandb.run.url)))
                wandb.log(
                    {
                        "shared_state/observed_microbatches": 1,
                        "shared_state/ga": 8,
                        "shared_state/optimizer_steps": 0,
                    }
                )
        _finish_train(state, initialized.checkpoint_manager)
        dist.barrier()
        if rank == 0:
            print("SHARED_STATE_HEADS_COMPLETE", output / "heads", flush=True)
        dist.destroy_process_group()
    else:
        raise RuntimeError("The PP1 schedule returned without the frozen-state capture.")
    finally:
        ernie55_step.get_batch = native_batch


if __name__ == "__main__":
    main()
