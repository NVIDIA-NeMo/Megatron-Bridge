# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Train and evaluate CLM projection heads on cached encoder embeddings.

The CLM encoder is frozen, so after every state / option text has been embedded once (with Megatron, see
``examples/decision/clm_posttrain.py``) post-training only touches the ~20M-parameter heads, which train in seconds per
epoch on one GPU.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from megatron.bridge.decision.clm_heads import CLMHeads, EmbeddingTable, clm_batch
from megatron.bridge.decision.losses import DecisionLossConfig, build_targets, pad_logits
from megatron.bridge.decision.metrics import DecisionEvaluator
from megatron.bridge.decision.records import question_targets


logger = logging.getLogger(__name__)


@dataclass
class CLMTrainConfig:
    """Optimisation settings for CLM head post-training.

    Attributes:
        epochs: Maximum number of epochs.
        batch_size: Records per optimisation step (every question of every record is one decision).
        lr: Peak AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of steps used for linear warm-up before cosine decay.
        grad_clip: Gradient-norm clip.
        patience: Early-stopping patience in epochs (on the validation metric).
        eval_metric: Validation metric used for model selection (``accuracy``, ``exact_match`` or ``reward``).
        train_logit_scale: Whether the temperature (``logit_scale``) is trained.
        seed: Random seed.
    """

    epochs: int = 20
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    patience: int = 5
    eval_metric: str = "accuracy"
    train_logit_scale: bool = True
    seed: int = 1234


def _forward(heads: CLMHeads, records: list[dict[str, Any]], table: EmbeddingTable, device: torch.device):
    """Return ``(flat per-question logits, per-record targets)`` for a batch (see :meth:`CLMHeads.question_logits`)."""
    states, options, counts, orders = clm_batch(records, table, device)
    logits = heads.question_logits(states, options, counts)
    return logits, [question_targets(r, o) for r, o in zip(records, orders)]


@torch.no_grad()
def evaluate_heads(
    heads: CLMHeads,
    records: list[dict[str, Any]],
    table: EmbeddingTable,
    *,
    device: torch.device,
    slice_keys: tuple[str, ...] = (),
    batch_size: int = 256,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score ``records`` and return ``(summary, per-question predictions)``."""
    heads.eval()
    evaluator = DecisionEvaluator(slice_keys=slice_keys)
    for b in range(0, len(records), batch_size):
        batch = records[b : b + batch_size]
        logits, targets = _forward(heads, batch, table, device)
        probs, k = [], 0
        for ts in targets:
            probs.append([logits[k + j].float().softmax(-1).tolist() for j in range(len(ts))])
            k += len(ts)
        evaluator.add_batch(batch, targets, probs)
    return evaluator.summary(), evaluator.predictions()


def train_heads(
    heads: CLMHeads,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]] | None,
    table: EmbeddingTable,
    *,
    loss_config: DecisionLossConfig,
    train_config: CLMTrainConfig,
    device: torch.device,
    log_path: Path | None = None,
) -> CLMHeads:
    """Post-train ``heads`` (SFT or RL per ``loss_config.mode``) and return the best copy by validation metric.

    For RL the heads the stage starts from are frozen and used as the KL reference policy.
    """
    cfg = train_config
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    loss_fn = loss_config.build()
    reference = None
    if loss_config.mode == "rl":
        reference = copy.deepcopy(heads).eval().requires_grad_(False)
    params = [p for n, p in heads.named_parameters() if cfg.train_logit_scale or n != "logit_scale"]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_records = list(train_records)
    steps_per_epoch = math.ceil(len(train_records) / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    warmup = max(1, int(total * cfg.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (
            (s + 1) / warmup
            if s < warmup
            else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup))))
        ),
    )
    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    def val_score(h: CLMHeads) -> float | None:
        if not val_records:
            return None
        overall = evaluate_heads(h, val_records, table, device=device)[0]["overall"]
        if cfg.eval_metric not in overall:
            raise ValueError(f"validation has no {cfg.eval_metric!r} metric; choose a metric with supervision")
        return overall[cfg.eval_metric]

    best_score, best, stale, step = val_score(heads), copy.deepcopy(heads), 0, 0
    logger.info("[%s] epoch 0 val %s=%s", loss_config.mode, cfg.eval_metric, best_score)
    log = open(log_path, "w") if log_path is not None else None
    for epoch in range(1, cfg.epochs + 1):
        heads.train()
        rng.shuffle(train_records)
        for b in range(0, len(train_records), cfg.batch_size):
            batch = train_records[b : b + cfg.batch_size]
            logits_list, targets = _forward(heads, batch, table, device)
            ref = None
            if reference is not None:
                with torch.no_grad():
                    ref = [x.float().log_softmax(-1).tolist() for x in _forward(reference, batch, table, device)[0]]
            flat = [q for ts in targets for q in ts]
            t = build_targets(flat, loss_config.question_weights, ref).to(device)
            if loss_fn.normalization_weight(t).item() <= 0:
                raise ValueError("training batch has no eligible labels or reward tables")
            logits, _ = pad_logits(logits_list)
            loss, metrics = loss_fn(logits, t, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if log is not None:
                row = {"epoch": epoch, "step": step, "loss": loss.item(), "grad_norm": grad_norm.item()}
                row.update({k: v.item() for k, v in metrics.items()})
                log.write(json.dumps(row) + "\n")
        score = val_score(heads)
        logger.info("[%s] epoch %d loss=%.4f val %s=%s", loss_config.mode, epoch, loss.item(), cfg.eval_metric, score)
        if log is not None:
            log.write(json.dumps({"epoch": epoch, f"val_{cfg.eval_metric}": score}) + "\n")
            log.flush()
        if score is None or best_score is None or score > best_score:
            best_score, best, stale = score, copy.deepcopy(heads), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                logger.info("[%s] early stopping after %d stale epochs", loss_config.mode, stale)
                break
    if log is not None:
        log.close()
    return best.eval()
