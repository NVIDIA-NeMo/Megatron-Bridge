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

"""Post-train Cloudflare Clef (Qwen3.5 backbone + joint schema head) with Megatron-Bridge.

Clef turns a state and a schema of typed questions into one logit per allowed option in a single forward pass: the
Qwen3.5 backbone (hybrid Gated DeltaNet / attention) produces final hidden states and a small transformer "joint schema
head" scores every option of every question jointly.

This script converts the Hugging Face checkpoint with ``AutoBridge`` (``Qwen35VLBridge`` -> ``Qwen3VLModel``) and runs
a Megatron-Core training loop:

* LoRA (``megatron.bridge.peft.LoRA``) on the language model's attention, MLP and Gated DeltaNet projections; the
  vision tower and LM head stay frozen.  Tensor parallelism (+ sequence parallelism) shards the 27B backbone.
* The joint schema head is attached through ``output_processor``: Megatron returns the final hidden states (gathered
  over the sequence-parallel region) instead of vocabulary logits and the head turns them into option logits.  The
  head is replicated in fp32 on every rank, trained with its own AdamW, and its gradients are averaged over the
  data-parallel group.
* ``--mode eval`` scores the model (the *before* numbers); ``--mode train`` runs SFT (``--loss-mode sft``) or RL
  (``--loss-mode rl``).  For RL the reference policy's log-probabilities are pre-computed in one inference pass.
* Every stage writes ``adapter_tp{rank}.safetensors`` (LoRA by parameter name) and the head in the release layout,
  so stages chain with ``--init-from``; ``--export-hf`` writes a merged Hugging Face checkpoint plus
  ``joint_head.*`` - the same layout as ``Cloudflare/clef``, loadable with its ``load_release_model``.

Examples::

    uv run python -m torch.distributed.run --nproc_per_node=8 examples/decision/clef_posttrain.py \\
        --mode eval --tp 4 --data-dir data/soc_triage --output-dir runs/bridge_clef/base
    uv run python -m torch.distributed.run --nproc_per_node=8 examples/decision/clef_posttrain.py \\
        --mode train --loss-mode sft --tp 4 --data-dir data/soc_triage --output-dir runs/bridge_clef/sft
    uv run python -m torch.distributed.run --nproc_per_node=8 examples/decision/clef_posttrain.py \\
        --mode train --loss-mode rl --init-from runs/bridge_clef/sft/best --lr 2e-5 --head-lr 1e-5 --epochs 1 \\
        --eval-metric reward --tp 4 --data-dir data/soc_triage --output-dir runs/bridge_clef/rl
"""

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import finalize_model_grads
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.utils import get_model_config
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.decision.clef_head import (
    CLEF_HF_REPO,
    CLEF_HF_REVISION,
    EncodedRecord,
    LexicalTable,
    clef_option_orders,
    collate_records,
    encode_record,
    load_joint_head,
    option_token_ids,
    save_joint_head,
)
from megatron.bridge.decision.losses import DecisionLossConfig, build_targets, pad_logits
from megatron.bridge.decision.megatron_glue import (
    all_reduce_mean_grads,
    canonical_name,
    gather_vocab_parallel_rows,
    load_trainable_state,
    save_trainable_state,
)
from megatron.bridge.decision.metrics import DecisionEvaluator, save_summary
from megatron.bridge.decision.records import load_records, question_targets, validate_record
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.initialize import initialize_megatron
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


logger = logging.getLogger(__name__)

LORA_TARGETS = [
    "*language_model.*.linear_qkv",
    "*language_model.*.linear_proj",
    "*language_model.*.linear_fc1",
    "*language_model.*.linear_fc2",
    "*language_model.*.in_proj",
    "*language_model.*.out_proj",
]

Item = tuple[dict[str, Any], EncodedRecord]


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["eval", "train"], default="train")
    p.add_argument("--loss-mode", choices=["sft", "rl"], default="sft")
    p.add_argument("--rl-objective", choices=["expected_reward", "grpo"], default="expected_reward")
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--anchor-weight", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--brier-weight", type=float, default=1.0)
    p.add_argument("--ordinal-weight", type=float, default=0.5)
    p.add_argument("--hf-model", default=CLEF_HF_REPO)
    p.add_argument("--revision", default=None, help="pinned Hub model, tokenizer, and head revision")
    p.add_argument("--head-only", action="store_true", help="freeze the entire backbone and train only the joint head")
    p.add_argument("--init-from", default=None, help="previous stage dir (adapter_tp*.safetensors + joint_head.*)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--slice-keys", nargs="*", default=["evidence", "verdict", "family"])
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--recompute", action="store_true", help="full activation recompute (saves memory)")
    p.add_argument("--lora-dim", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--eval-metric", default="accuracy", choices=["accuracy", "exact_match", "reward"])
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--export-hf", default=None, help="write a merged HF checkpoint + joint head here at the end")
    p.add_argument("--seed", type=int, default=1234)
    return p


class ClefTrainer:
    """Megatron-Core LoRA + joint-schema-head post-training loop for Clef (see module docstring)."""

    def __init__(self, args: argparse.Namespace):
        """Build the Megatron model, head, data and optimizers."""
        self.args = args
        self.tag = "eval" if args.mode == "eval" else args.loss_mode
        self.output_dir = Path(args.output_dir)

        self.bridge = AutoBridge.from_hf_pretrained(args.hf_model, revision=args.revision, torch_dtype=torch.bfloat16)
        provider = self.bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = args.tp
        provider.pipeline_model_parallel_size = 1
        provider.context_parallel_size = 1
        provider.sequence_parallel = args.tp > 1
        provider.seq_length = args.max_length
        provider.freeze_vision_model = True
        provider.freeze_vision_projection = True
        if args.recompute:
            provider.recompute_granularity = "full"
            provider.recompute_method = "uniform"
            provider.recompute_num_layers = 1
        self.lora = LoRA(
            target_modules=LORA_TARGETS, dim=args.lora_dim, alpha=args.lora_alpha, dropout=args.lora_dropout
        )

        def apply_lora(model: list) -> list:
            if args.head_only:
                for module in model:
                    module.requires_grad_(False)
            else:
                model = self.lora(model, training=True)
                self.lora.set_params_to_save(model)
            return model

        provider.register_pre_wrap_hook(apply_lora)  # runs after the HF weight-loading hook

        cfg = ConfigContainer(
            model=provider,
            train=TrainingConfig(
                micro_batch_size=args.micro_batch_size, global_batch_size=args.global_batch_size, train_iters=1
            ),
            optimizer=OptimizerConfig(
                optimizer="adam",
                lr=args.lr,
                min_lr=0.0,
                weight_decay=0.0,
                adam_beta1=0.9,
                adam_beta2=0.95,
                clip_grad=1.0,
                bf16=True,
                use_distributed_optimizer=True,
            ),
            # The LR schedule (warm-up + cosine) is applied manually per step in ``train``.
            scheduler=SchedulerConfig(
                lr_decay_style="constant",
                lr_warmup_iters=0,
                start_weight_decay=0.0,
                end_weight_decay=0.0,
                override_opt_param_scheduler=True,
            ),
            ddp=DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=True),
            tokenizer=TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=args.hf_model),
            checkpoint=CheckpointConfig(
                save=None,
                load=None,
                save_interval=0,
                async_save=False,
                fully_parallel_save=False,
                fully_parallel_load=False,
            ),
            logger=LoggerConfig(),
            dataset=None,
        )
        cfg.validate()
        initialize_megatron(cfg=cfg)
        self.cfg = cfg
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        # The provider applies registered HF weight-loading and LoRA hooks before DDP wrapping.
        self.model = cfg.model.provide_distributed_model(
            ddp_config=cfg.ddp,
            wrap_with_ddp=not args.head_only,
            overlap_param_gather_with_optimizer_step=False,
            use_torch_fsdp2=False,
            data_parallel_random_init=False,
            pg_collection=pg_collection,
        )
        if not args.head_only and not self.lora.params_to_save:
            raise RuntimeError("LoRA construction hook did not produce any trainable adapter tensors")
        trainable = sum(p.numel() for p in self.model[0].parameters() if p.requires_grad)
        print_rank_0(f"LoRA trainable parameters per TP rank: {trainable} ({len(self.lora.params_to_save)} tensors)")
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        self.dp_group = parallel_state.get_data_parallel_group()
        self.dp_rank = parallel_state.get_data_parallel_rank()
        self.dp_size = parallel_state.get_data_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.device = torch.device("cuda", torch.cuda.current_device())

        if args.init_from and not args.head_only:
            n = load_trainable_state(self.model[0], args.init_from, tp_rank=self.tp_rank)
            print_rank_0(f"loaded {n} LoRA tensors from {args.init_from}")
        self.head = load_joint_head(
            args.init_from or args.hf_model,
            dtype=torch.float32,
            dropout=args.head_dropout,
            revision=args.revision,
        )
        self.head = self.head.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_model, revision=args.revision)
        self.pad_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )
        self.splits: dict[str, list[Item]] = {}
        for name in ("train", "validation", "test"):
            path = Path(args.data_dir) / f"{name}.jsonl"
            if path.exists():
                records = load_records(path)
                for record in records:
                    validate_record(record)
                self.splits[name] = [
                    (r, encode_record(self.tokenizer, r, max_length=args.max_length)) for r in records
                ]
        self.lexical = self._build_lexical_table()

        self.optimizer = self.scheduler = None
        if args.mode == "train" and not args.head_only:
            self.optimizer, self.scheduler = setup_optimizer(
                optimizer_config=cfg.optimizer,
                scheduler_config=cfg.scheduler,
                model=self.model,
                use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
            )
            config = get_model_config(self.model[0])
            config.finalize_model_grads_func = finalize_model_grads
            config.grad_scale_func = self.optimizer.scale_loss
        if args.mode == "train":
            self.head_optimizer = torch.optim.AdamW(self.head.parameters(), lr=args.head_lr, weight_decay=0.0)
        self.loss_config = DecisionLossConfig(
            mode=args.loss_mode,
            label_smoothing=args.label_smoothing,
            brier_weight=args.brier_weight,
            ordinal_weight=args.ordinal_weight,
            rl_objective=args.rl_objective,
            kl_coef=args.kl_coef,
            anchor_weight=args.anchor_weight,
        )
        self.loss_fn = self.loss_config.build()
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed + self.dp_rank)

    # ------------------------------------------------------------------ model plumbing
    def _build_lexical_table(self) -> LexicalTable:
        """Gather the frozen LM-head rows of every option token in the data (once)."""
        ids = option_token_ids([e for items in self.splits.values() for _, e in items])
        weight = next(
            p for n, p in self.model[0].named_parameters() if canonical_name(n).endswith("output_layer.weight")
        )
        rows = gather_vocab_parallel_rows(weight, ids, tp_group=self.tp_group)
        return LexicalTable(ids, rows.float()).to(self.device)

    def _processor(self, batch: dict[str, Any]):
        """``output_processor`` that runs the joint head and returns padded option logits [questions, options]."""

        def processor(*, hidden_states: torch.Tensor, config: Any = None, **_: Any) -> torch.Tensor:
            """Map ``hidden_states`` of shape [sequence (/ tp if SP), batch, hidden] to logits [questions, options]."""
            if config is not None and config.sequence_parallel:
                # The head runs replicated on every TP rank, so its gradient must be split, not reduce-scattered.
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
                )
            hidden = hidden_states.transpose(0, 1)[:, : batch["input_ids"].shape[1]]
            per_record = self.head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], self.lexical)
            logits, _ = pad_logits([q.float() for rec in per_record for q in rec])
            return logits

        return processor

    def _make_batch(self, items: list[Item], reference: dict | None = None) -> dict[str, Any]:
        collated = collate_records([e for _, e in items], self.pad_id, self.device)
        width = collated["input_ids"].shape[1]
        multiple = 64 * self.args.tp
        padded = (width + multiple - 1) // multiple * multiple  # sequence parallelism needs seq % tp == 0
        model_ids = torch.full((len(items), padded), self.pad_id, dtype=torch.long, device=self.device)
        model_ids[:, :width] = collated["input_ids"]
        flat_targets, ref_rows = [], [] if reference is not None else None
        for record, encoded in items:
            flat_targets.extend(question_targets(record, clef_option_orders(encoded)))
            if ref_rows is not None:
                ref_rows.extend(reference[str(record["id"])])
        return {
            "model_input_ids": model_ids,
            "input_ids": collated["input_ids"],
            "attention_mask": collated["attention_mask"],
            "records": collated["records"],
            "targets": build_targets(flat_targets, self.loss_config.question_weights, ref_rows).to(self.device),
        }

    def _forward(self, batch: dict[str, Any]) -> torch.Tensor:
        # Right padding + causal attention / recurrence: no mask is needed, padded positions never affect real ones.
        return self.model[0](
            input_ids=batch["model_input_ids"],
            position_ids=None,
            attention_mask=None,
            output_processor=self._processor(batch),
        )

    def _shard(self, items: list[Item], *, shuffle: bool, epoch: int = 0) -> list[Item]:
        """Equal-length data-parallel shard (identical across tensor-parallel ranks)."""
        items = list(items)
        if shuffle:
            random.Random(self.args.seed + epoch).shuffle(items)
        per_rank = math.ceil(len(items) / self.dp_size)
        items = items + items[: per_rank * self.dp_size - len(items)]
        return items[self.dp_rank :: self.dp_size]

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def score(self, items: list[Item]) -> dict[str, list[list[float]]]:
        """``{record_id: [[logp per option] per question]}`` for ``items``, gathered on every rank."""
        for m in self.model:
            m.eval()
        self.head.eval()
        local: dict[str, list[list[float]]] = {}
        shard = self._shard(items, shuffle=False)
        for b in range(0, len(shard), self.args.eval_batch_size):
            chunk = shard[b : b + self.args.eval_batch_size]
            batch = self._make_batch(chunk)
            logp = torch.log_softmax(self._forward(batch).float().masked_fill(~batch["targets"].option_mask, -1e9), -1)
            k = 0
            for record, encoded in chunk:
                n = len(encoded.questions)
                local[str(record["id"])] = [
                    logp[k + j, : len(encoded.questions[j].option_ids)].tolist() for j in range(n)
                ]
                k += n
        gathered: list[dict] = [None] * self.dp_size
        dist.all_gather_object(gathered, local, group=self.dp_group)
        merged: dict[str, list[list[float]]] = {}
        for part in gathered:
            merged.update(part)
        return merged

    def evaluate(self, split: str) -> dict[str, Any]:
        """Metrics summary for ``split`` (identical on every rank)."""
        logprobs = self.score(self.splits[split])
        evaluator = DecisionEvaluator(slice_keys=tuple(self.args.slice_keys))
        for record, encoded in self.splits[split]:
            targets = question_targets(record, clef_option_orders(encoded))
            probs = [[math.exp(x) for x in q] for q in logprobs[str(record["id"])]]
            evaluator.add_batch([record], [targets], [probs])
        return evaluator.summary()

    def report(self, split: str) -> None:
        """Evaluate ``split`` and write ``metrics_<split>.json``."""
        summary = self.evaluate(split)
        summary["run"] = {
            "stage": self.tag,
            "split": split,
            "init_from": self.args.init_from,
            "model": self.args.hf_model,
            "revision": self.args.revision,
            "head_only": self.args.head_only,
        }
        if get_rank_safe() == 0:
            save_summary(summary, self.output_dir / f"metrics_{split}.json")
            o = summary["overall"]
            print_rank_0(
                f"[{self.tag}/{split}] accuracy={o['accuracy']:.4f} exact_match={o['exact_match']:.4f} "
                f"reward={o.get('reward', float('nan')):.4f} brier={o['brier']:.4f} ece={o['ece']:.4f}"
            )

    # ------------------------------------------------------------------ stage I/O
    def save_stage(self, path: Path) -> None:
        """Write LoRA tensors (one file per TP rank) and the head (release layout) to ``path``."""
        if not self.args.head_only:
            save_trainable_state(self.model[0], path, tp_rank=self.tp_rank, write=self.dp_rank == 0)
        path.mkdir(parents=True, exist_ok=True)
        if get_rank_safe() == 0:
            save_joint_head(self.head, path, dtype=torch.float32)
            (path / "stage.json").write_text(
                json.dumps({"stage": self.tag, "tp": self.args.tp, "head_only": self.args.head_only}, indent=2)
            )
        dist.barrier()

    def load_stage(self, path: Path) -> None:
        """Restore LoRA tensors and the head from ``path`` (same TP size)."""
        if not self.args.head_only:
            load_trainable_state(self.model[0], path, tp_rank=self.tp_rank)
        self.head.load_state_dict(load_joint_head(path).state_dict())
        # The distributed optimizer keeps fp32 main copies; re-sync them from the restored bf16 params.
        if self.optimizer is not None and hasattr(self.optimizer, "reload_model_params"):
            self.optimizer.reload_model_params()

    # ------------------------------------------------------------------ training
    def _val_score(self) -> float | None:
        if "validation" not in self.splits:
            return None
        overall = self.evaluate("validation")["overall"]
        if self.args.eval_metric not in overall:
            raise ValueError(f"validation has no {self.args.eval_metric!r} metric; choose a metric with supervision")
        return overall[self.args.eval_metric]

    def train(self) -> None:
        """SFT / RL loop with Megatron's forward-backward schedule."""
        args = self.args
        train_items = self.splits["train"]
        batch_multiple = args.micro_batch_size * self.dp_size
        if args.global_batch_size < batch_multiple or args.global_batch_size % batch_multiple:
            raise ValueError("global-batch-size must be a positive multiple of micro-batch-size times DP size")
        accum = args.global_batch_size // batch_multiple
        reference = None
        if self.loss_config.mode == "rl" and self.loss_config.kl_coef > 0:
            start = time.perf_counter()
            reference = self.score(train_items)
            print_rank_0(
                f"pre-computed reference log-probs for {len(reference)} records ({time.perf_counter() - start:.0f}s)"
            )
        steps_per_epoch = len(self._shard(train_items, shuffle=False)) // (args.micro_batch_size * accum)
        if steps_per_epoch == 0:
            raise ValueError("training split has no full batch; reduce global-batch-size or add records")
        total = args.epochs * steps_per_epoch
        warmup = max(1, int(total * args.warmup_ratio))

        def lr_scale(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))

        forward_backward = get_forward_backward_func()
        best = self._val_score()
        print_rank_0(f"[{self.tag}] step 0 val {args.eval_metric}={best} ({total} steps, {accum} micro-batches/step)")
        self.save_stage(self.output_dir / "best")
        log = open(self.output_dir / "training_log.jsonl", "w") if get_rank_safe() == 0 else None
        step = 0
        for epoch in range(args.epochs):
            shard = self._shard(train_items, shuffle=True, epoch=epoch)
            micro = [shard[i : i + args.micro_batch_size] for i in range(0, len(shard), args.micro_batch_size)]
            for s in range(0, len(micro) - accum + 1, accum):
                for m in self.model:
                    m.train(not args.head_only)
                    if not args.head_only:
                        m.zero_grad_buffer()
                self.head.train()
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                self.head_optimizer.zero_grad(set_to_none=True)
                for group in self.optimizer.param_groups if self.optimizer is not None else ():
                    group["lr"] = args.lr * lr_scale(step)
                for group in self.head_optimizer.param_groups:
                    group["lr"] = args.head_lr * lr_scale(step)
                batches = [self._make_batch(mb, reference) for mb in micro[s : s + accum]]
                window_weight = torch.stack(
                    [self.loss_fn.normalization_weight(batch["targets"]) for batch in batches]
                ).sum()
                if self.dp_size > 1:
                    dist.all_reduce(window_weight, op=dist.ReduceOp.SUM, group=self.dp_group)
                if window_weight.item() <= 0:
                    raise ValueError("global optimizer window has no eligible labels or reward tables")

                def forward_step(data_iterator: Iterator[dict], model: torch.nn.Module):
                    batch = next(data_iterator)
                    logits = model(
                        input_ids=batch["model_input_ids"],
                        position_ids=None,
                        attention_mask=None,
                        output_processor=self._processor(batch),
                    )

                    def loss_func(output: torch.Tensor):
                        loss, metrics = self.loss_fn(output, batch["targets"], generator=self.generator)
                        local_weight = self.loss_fn.normalization_weight(batch["targets"])
                        # MCore's schedule divides by num_microbatches. Cancel
                        # that averaging before DP averaging, so questions in
                        # the whole optimizer window have their declared weight.
                        scale = accum * self.dp_size * local_weight / window_weight
                        return loss * scale, {"loss": loss.detach(), **metrics}

                    return logits, loss_func

                if args.head_only:
                    losses = []
                    for batch in batches:
                        loss, metrics = self.loss_fn(self._forward(batch), batch["targets"], generator=self.generator)
                        local_weight = self.loss_fn.normalization_weight(batch["targets"])
                        (loss * self.dp_size * local_weight / window_weight).backward()
                        losses.append({"loss": loss.detach(), **metrics})
                else:
                    losses = forward_backward(
                        forward_step_func=forward_step,
                        data_iterator=iter(batches),
                        model=self.model,
                        num_microbatches=accum,
                        seq_length=batches[0]["model_input_ids"].shape[1],
                        micro_batch_size=args.micro_batch_size,
                        decoder_seq_length=batches[0]["model_input_ids"].shape[1],
                        forward_only=False,
                    )
                all_reduce_mean_grads(self.head, group=self.dp_group)
                head_grad_norm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                grad_norm = head_grad_norm
                if self.optimizer is not None:
                    _, grad_norm, _ = self.optimizer.step()
                self.head_optimizer.step()
                step += 1
                if log is not None:
                    row = {"epoch": epoch, "step": step, "grad_norm": float(grad_norm or 0.0)}
                    for key in losses[0]:
                        row[key] = sum(float(x[key]) for x in losses) / len(losses)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if step % 10 == 0:
                        print_rank_0(
                            f"[{self.tag}] step {step}/{total} " + json.dumps({k: round(v, 4) for k, v in row.items()})
                        )
                if step % args.val_every == 0 or step == total:
                    score = self._val_score()
                    print_rank_0(f"[{self.tag}] step {step} val {args.eval_metric}={score} (best {best})")
                    if best is None or (score is not None and score > best):
                        best = score
                        self.save_stage(self.output_dir / "best")
        self.save_stage(self.output_dir / "last")
        if log is not None:
            log.close()
        self.load_stage(self.output_dir / "best")

    def export_hf(self, path: Path) -> None:
        """Write a merged HF checkpoint plus the joint head (``Cloudflare/clef`` release layout)."""
        self.bridge.save_hf_pretrained(self.model, path, merge_adapter_weights=True)
        if get_rank_safe() == 0:
            save_joint_head(self.head, path, dtype=torch.bfloat16)
            self.tokenizer.save_pretrained(path)
        dist.barrier()


def main() -> None:
    """Evaluate or post-train Clef, then report validation / test metrics."""
    args = build_parser().parse_args()
    if args.revision is None and args.hf_model == CLEF_HF_REPO:
        args.revision = CLEF_HF_REVISION
    if args.head_only and args.init_from:
        stage = Path(args.init_from) / "stage.json"
        if not stage.exists() or not json.loads(stage.read_text()).get("head_only", False):
            raise ValueError("head-only continuation requires a head-only stage with stage.json")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    trainer = ClefTrainer(args)
    trainer.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        trainer.train()
    for split in ("validation", "test"):
        if split in trainer.splits:
            trainer.report(split)
    if args.export_hf:
        trainer.export_hf(Path(args.export_hf))


if __name__ == "__main__":
    main()
