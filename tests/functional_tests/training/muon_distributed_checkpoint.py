# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Two-GPU numerical and save/resume regression for distributed optimizers.

Run with torch.distributed.run. Each case uses a fresh process; --compare checks
all rank artifacts without initializing distributed training.
"""

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.recipes.gpt.vanilla_gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import OptimizerConfig
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.train import force_param_sync


logger = logging.getLogger(__name__)


class RecordTraining(Callback):
    def __init__(self, output: Path) -> None:
        self.output = output
        self.records = []
        self.start_step = None

    def on_train_start(self, context: CallbackContext) -> None:
        self.start_step = context.state.train_state.step
        self.model = context.model

    def on_train_step_end(self, context: CallbackContext) -> None:
        assert not context.skipped_iter
        losses = {name: float(value) for name, value in context.loss_dict.items()}
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        self.records.append(dict(losses=losses, grad_norm=float(context.grad_norm)))

    def on_checkpoint_save(self, context: CallbackContext) -> None:
        # Saving must materialize every parameter before serialization. A later
        # resume can otherwise hide stale model weights by restoring main params.
        before = [p.detach().cpu().clone() for chunk in self.model for p in chunk.parameters()]
        force_param_sync(self.model, context.optimizer)
        after = [p.detach().cpu() for chunk in self.model for p in chunk.parameters()]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def on_train_end(self, context: CallbackContext) -> None:
        force_param_sync(context.model, context.optimizer)
        rank = torch.distributed.get_rank()
        self.output.mkdir(parents=True, exist_ok=True)
        tensors = {
            f"{i}.{name}": param.detach().cpu().clone()
            for i, chunk in enumerate(context.model)
            for name, param in chunk.named_parameters()
        }
        assert all(torch.isfinite(value).all() for value in tensors.values())
        torch.save(tensors, self.output / f"weights-rank{rank}.pt")
        (self.output / f"records-rank{rank}.json").write_text(
            json.dumps(
                dict(
                    start_step=self.start_step,
                    end_step=context.state.train_state.step,
                    records=self.records,
                    parameters=sum(p.numel() for p in tensors.values()),
                ),
                indent=2,
            )
        )


def run_case(root: Path, case: str) -> None:
    cfg = vanilla_gpt_pretrain_config()
    cfg.model = GPTModelProvider(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=32,
        vocab_size=1024,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
    )
    if "builder" in case:
        from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
        from megatron.bridge.models.transformer_config import TransformerConfig

        cfg.model = GPTModelConfig(
            transformer=TransformerConfig(
                num_layers=2,
                hidden_size=128,
                ffn_hidden_size=512,
                num_attention_heads=4,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                gradient_accumulation_fusion=False,
                bias_activation_fusion=False,
                bias_dropout_fusion=False,
                masked_softmax_fusion=False,
            ),
            seq_length=32,
            vocab_size=1024,
            share_embeddings_and_output_weights=True,
        )
    # The resize source saves at step 2 and continues through step 3, providing
    # an uninterrupted two-rank control for the one-rank resume.
    cfg.train.train_iters = 3 if case.startswith("muon_dp") else 6
    cfg.train.global_batch_size = 4
    cfg.train.micro_batch_size = 1
    # Keep hash checks separate: they force a gather and could hide a broken
    # save-time synchronization gate in the checkpoint regression cases.
    cfg.train.check_weight_hash_across_dp_replicas_interval = 1 if case == "muon_hash" else None
    cfg.validation.eval_iters = 1
    cfg.validation.eval_interval = 3
    cfg.dataset.seq_length = 32
    cfg.dataset.num_workers = 0
    cfg.tokenizer.vocab_size = 1024
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = 6
    cfg.scheduler.start_weight_decay = 0.01
    cfg.scheduler.end_weight_decay = 0.01
    cfg.logger.tensorboard_dir = None
    cfg.logger.log_interval = 1
    cfg.checkpoint.save = str(root / case / "checkpoints")
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = 2
    cfg.checkpoint.async_save = False
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.dist_ckpt_optim_fully_reshardable = "reshard" in case
    cfg.checkpoint.load_optim = True
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.load_rng = True
    cfg.checkpoint.save_rng = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.bucket_size = 100000
    cfg.ddp.use_distributed_optimizer = case.startswith("adam") or case == "muon_inherited"
    overlap = not case.endswith("_off")
    name = "adam" if case.startswith("adam") else "dist_muon" if case == "muon_alias" else "muon"
    cfg.optimizer = OptimizerConfig(
        optimizer=name,
        lr=1e-3,
        min_lr=0.0,
        clip_grad=1.0,
        use_distributed_optimizer=cfg.ddp.use_distributed_optimizer,
        use_layer_wise_distributed_optimizer=case not in ("muon_alias", "muon_inherited") and name == "muon",
    )
    if case == "muon_legacy_on":
        # Recreate the compact optimizer representation used before the layout option.
        cfg.optimizer.use_layer_wise_param_layout = False
    elif case != "muon_legacy_resume":
        # Keep the existing full-layout matrix as explicit opt-in coverage.
        cfg.optimizer.use_layer_wise_param_layout = True
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=True,
        overlap_param_gather=overlap,
        bucket_size=100000,
    )
    if case == "muon_hash":
        # Exercise the lifecycle gate independently of CommOverlapConfig.
        cfg.ddp.overlap_param_gather = True
        cfg.optimizer.overlap_param_gather = True
    if case == "muon_dp1_resume":
        cfg.checkpoint.load = str(root / "muon_dp2_save" / "checkpoints" / "iter_0000002")
    elif case.endswith("_resume"):
        source = case.removesuffix("_resume") + "_on"
        cfg.checkpoint.load = str(root / source / "checkpoints" / "iter_0000002")
    # Older compact checkpoints omitted distributed-optimizer sharding metadata.
    # Save that exact representation, then resume using today's untouched default.
    legacy_metadata = (
        patch(
            "megatron.bridge.training.checkpointing._uses_distributed_optimizer",
            return_value=False,
        )
        if case == "muon_legacy_on"
        else nullcontext()
    )
    with legacy_metadata:
        pretrain(cfg, forward_step, callbacks=[RecordTraining(root / case)])


def compare(root: Path) -> None:
    pairs = [
        ("muon_on", "muon_inherited"),
        ("muon_on", "muon_alias"),
        ("muon_on", "muon_off"),
        ("muon_on", "muon_resume"),
        ("adam_on", "adam_off"),
        ("adam_on", "adam_resume"),
        ("muon_on", "muon_hash"),
        ("muon_builder_on", "muon_builder_off"),
        ("muon_builder_on", "muon_builder_resume"),
        ("muon_reshard_on", "muon_reshard_resume"),
        ("muon_legacy_on", "muon_legacy_resume"),
    ]
    for reference, candidate in pairs:
        for rank in range(2):
            a = json.loads((root / reference / f"records-rank{rank}.json").read_text())
            b = json.loads((root / candidate / f"records-rank{rank}.json").read_text())
            assert a["start_step"] == 0 and a["end_step"] == b["end_step"] == 6
            resumed = candidate.endswith("_resume")
            assert b["start_step"] == (2 if resumed else 0)
            assert len(a["records"]) == 6 and len(b["records"]) == (4 if resumed else 6)
            assert a["records"][2 if resumed else 0 :] == b["records"], (reference, candidate, rank, a, b)
            weights = [
                torch.load(root / c / f"weights-rank{rank}.pt", map_location="cpu", weights_only=True)
                for c in (reference, candidate)
            ]
            torch.testing.assert_close(*weights, rtol=0, atol=0)
    compare_dp_resize(root)
    logger.info("All optimizer modes, overlap pairs and fresh-process resumes match their expected tolerances.")


def compare_dp_resize(root: Path) -> None:
    """Compare a one-rank resume with its uninterrupted two-rank source run."""
    dp_resize = json.loads((root / "muon_dp1_resume" / "records-rank0.json").read_text())
    dp_controls = [json.loads((root / "muon_dp2_save" / f"records-rank{rank}.json").read_text()) for rank in range(2)]
    assert dp_resize["start_step"] == 2
    assert dp_resize["end_step"] == 3
    assert len(dp_resize["records"]) == 1
    for control in dp_controls:
        assert control["start_step"] == 0
        assert control["end_step"] == 3
        assert len(control["records"]) == 3
        resumed_record = dp_resize["records"][0]
        control_record = control["records"][2]
        assert resumed_record["grad_norm"] == control_record["grad_norm"]
        assert resumed_record["losses"].keys() == control_record["losses"].keys()
        for name, resumed_loss in resumed_record["losses"].items():
            # Changing DP size changes the loss-reduction order, while the
            # restored optimizer trajectory and final BF16 model remain exact.
            torch.testing.assert_close(resumed_loss, control_record["losses"][name], rtol=0, atol=1e-6)
    weights = [
        torch.load(root / case / "weights-rank0.pt", map_location="cpu", weights_only=True)
        for case in ("muon_dp2_save", "muon_dp1_resume")
    ]
    torch.testing.assert_close(*weights, rtol=0, atol=0)
    dp2_rank1 = torch.load(root / "muon_dp2_save" / "weights-rank1.pt", map_location="cpu", weights_only=True)
    torch.testing.assert_close(weights[0], dp2_rank1, rtol=0, atol=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=[
            "muon_on",
            "muon_off",
            "muon_inherited",
            "muon_alias",
            "muon_resume",
            "adam_on",
            "adam_off",
            "adam_resume",
            "muon_hash",
            "muon_builder_on",
            "muon_builder_off",
            "muon_builder_resume",
            "muon_reshard_on",
            "muon_reshard_resume",
            "muon_dp2_save",
            "muon_dp1_resume",
            "muon_legacy_on",
            "muon_legacy_resume",
        ],
    )
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare(args.root)
    else:
        assert args.case is not None
        run_case(args.root, args.case)
