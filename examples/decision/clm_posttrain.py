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

"""Post-train Contrastive Language Model (CLM) decision heads with a Megatron-Core encoder.

CLM-v0.1-8B is a frozen Qwen3-8B encoder (last-token pooled, L2-normalised final hidden state) plus two projection
heads scored with ``exp(logit_scale) * cosine``.  The encoder is frozen by design (the released heads, the
``clm-serve`` action cache and the vLLM pooling server all live in its embedding space), so this script:

1. Converts Qwen3-8B from Hugging Face with ``AutoBridge`` and embeds every unique state / option text with Megatron
   (tensor parallel within a model replica, texts sharded over data-parallel replicas).  The decoder's final hidden
   states are read through ``output_processor`` instead of the vocabulary projection.  Embeddings are cached.
2. Trains (``--mode train``) or evaluates (``--mode eval``) the heads on rank 0, with SFT (``--loss-mode sft``) or RL
   (``--loss-mode rl``) objectives from :mod:`megatron.bridge.decision.losses`.

The output ``heads.pt`` is in the ``contrastive-lm`` checkpoint format (``CLM_CKPT=heads.pt clm-serve``).

Examples::

    # before: evaluate the released heads
    uv run python -m torch.distributed.run --nproc_per_node=8 examples/decision/clm_posttrain.py \\
        --mode eval --data-dir data/soc_triage --output-dir runs/bridge_clm/base
    # SFT
    uv run python -m torch.distributed.run --nproc_per_node=8 examples/decision/clm_posttrain.py \\
        --mode train --loss-mode sft --data-dir data/soc_triage --output-dir runs/bridge_clm/sft
    # RL from the SFT heads
    uv run python -m torch.distributed.run --nproc_per_node=8 examples/decision/clm_posttrain.py \\
        --mode train --loss-mode rl --heads runs/bridge_clm/sft/heads.pt --lr 1e-4 --epochs 10 \\
        --eval-metric reward --data-dir data/soc_triage --output-dir runs/bridge_clm/rl
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.decision.clm_heads import (
    CLM_ENCODER,
    CLM_ENCODER_REVISION,
    CLM_HF_REVISION,
    CLM_MAX_TOKENS,
    EmbeddingTable,
    collect_clm_texts,
    load_clm_heads,
    save_clm_heads,
)
from megatron.bridge.decision.clm_trainer import CLMTrainConfig, evaluate_heads, train_heads
from megatron.bridge.decision.losses import DecisionLossConfig
from megatron.bridge.decision.megatron_glue import final_hidden_states_processor, last_token_pool
from megatron.bridge.decision.metrics import save_summary
from megatron.bridge.decision.records import load_records, validate_record
from megatron.bridge.utils.common_utils import get_rank_safe, maybe_initialize_distributed, print_rank_0


logger = logging.getLogger(__name__)


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
    p.add_argument("--encoder", default=CLM_ENCODER, help="HF encoder the heads were trained on")
    p.add_argument("--encoder-revision", default=None, help="pinned Hub revision of the encoder")
    p.add_argument("--heads-revision", default=None, help="pinned Hub revision of the released heads")
    p.add_argument("--heads", default=None, help="heads checkpoint (.pt); default: released CLM-v0.1-8B heads")
    p.add_argument("--data-dir", required=True, help="directory with train/validation/test.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default=None, help="embedding cache (default: <output-dir>/../embedding_cache)")
    p.add_argument("--slice-keys", nargs="*", default=["evidence", "verdict", "family"])
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size of the encoder")
    p.add_argument("--max-length", type=int, default=CLM_MAX_TOKENS)
    p.add_argument("--embed-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--eval-metric", default="accuracy", choices=["accuracy", "exact_match", "reward"])
    p.add_argument("--seed", type=int, default=1234)
    return p


def cache_path(
    cache_dir: Path, encoder: str, texts: list[str], *, revision: str | None = None, max_length: int = CLM_MAX_TOKENS
) -> Path:
    """Content-addressed cache file for ``texts`` embedded by ``encoder``."""
    identity = [encoder, revision or "", str(max_length), *texts]
    digest = hashlib.sha256(("\n\x00".join(identity)).encode()).hexdigest()[:16]
    return cache_dir / f"embeddings_{digest}.pt"


@torch.no_grad()
def embed_with_megatron(texts: list[str], args: argparse.Namespace, cache: Path) -> None:
    """Embed ``texts`` with the Megatron encoder and write an :class:`EmbeddingTable` to ``cache`` (global rank 0)."""
    bridge = AutoBridge.from_hf_pretrained(args.encoder, revision=args.encoder_revision, torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = args.tp
    provider.pipeline_model_parallel_size = 1
    provider.sequence_parallel = False
    provider.apply_overrides_and_finalize()
    provider.initialize_model_parallel(seed=args.seed)
    model = provider.provide_distributed_model(wrap_with_ddp=False)[0].cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, revision=args.encoder_revision)
    # vLLM's ``truncate_prompt_tokens`` keeps the *last* tokens; match it so embeddings equal clm-serve's.
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    processor = final_hidden_states_processor(tp_group=parallel_state.get_tensor_model_parallel_group())

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    mine = order[dp_rank::dp_size]
    rows = []
    start = time.perf_counter()
    for b in range(0, len(mine), args.embed_batch_size):
        chunk = [texts[i] for i in mine[b : b + args.embed_batch_size]]
        ids = [
            tokenizer(t, add_special_tokens=True, truncation=True, max_length=args.max_length).input_ids for t in chunk
        ]
        lengths = torch.tensor([len(x) for x in ids])
        width = int(lengths.max())
        # Pad to a multiple of 8 for efficient kernels; right padding never changes earlier positions (causal).
        width = (width + 7) // 8 * 8
        input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
        input_ids = input_ids.cuda()
        position_ids = torch.arange(width, device="cuda").unsqueeze(0).expand_as(input_ids)
        hidden = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None, output_processor=processor)
        rows.append(F.normalize(last_token_pool(hidden, lengths).float(), dim=-1).cpu())
        if (b // args.embed_batch_size) % 50 == 0:
            print_rank_0(
                f"embedded {b + len(chunk)}/{len(mine)} texts per replica ({time.perf_counter() - start:.0f}s)"
            )
    shard = torch.cat(rows) if rows else torch.empty((0, provider.hidden_size))

    cache.parent.mkdir(parents=True, exist_ok=True)
    if parallel_state.get_tensor_model_parallel_rank() == 0:
        torch.save({"indices": mine, "embeddings": shard}, cache.with_suffix(f".dp{dp_rank}"))
    dist.barrier()
    if get_rank_safe() == 0:
        embeddings = torch.empty((len(texts), shard.shape[1]))
        for r in range(dp_size):
            part = torch.load(cache.with_suffix(f".dp{r}"), weights_only=True)
            if part["embeddings"].numel():
                embeddings[torch.tensor(part["indices"])] = part["embeddings"]
            cache.with_suffix(f".dp{r}").unlink()
        EmbeddingTable(texts, embeddings).save(cache)
        print_rank_0(f"cached {len(texts)} embeddings (dim {embeddings.shape[1]}) at {cache}")
    dist.barrier()
    del model
    torch.cuda.empty_cache()


def main() -> None:
    """Embed (once), then evaluate or post-train the CLM heads."""
    args = build_parser().parse_args()
    if args.encoder_revision is None and args.encoder == CLM_ENCODER:
        args.encoder_revision = CLM_ENCODER_REVISION
    if args.heads is None and args.heads_revision is None:
        args.heads_revision = CLM_HF_REVISION
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    maybe_initialize_distributed()
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    splits = {}
    for name in ("train", "validation", "test"):
        if (data_dir / f"{name}.jsonl").exists():
            splits[name] = load_records(data_dir / f"{name}.jsonl")
            for record in splits[name]:
                validate_record(record)
    texts = sorted({t for records in splits.values() for t in collect_clm_texts(records)})
    cache = cache_path(
        Path(args.cache_dir) if args.cache_dir else output_dir.parent / "embedding_cache",
        args.encoder,
        texts,
        revision=args.encoder_revision,
        max_length=args.max_length,
    )
    if not cache.exists():
        embed_with_megatron(texts, args, cache)
    if get_rank_safe() != 0:
        dist.barrier()
        return

    device = torch.device("cuda")
    table = EmbeddingTable.load(cache)
    heads = load_clm_heads(args.heads, revision=args.heads_revision).to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "eval" if args.mode == "eval" else args.loss_mode
    if args.mode == "train":
        loss_config = DecisionLossConfig(
            mode=args.loss_mode,
            label_smoothing=args.label_smoothing,
            brier_weight=args.brier_weight,
            ordinal_weight=args.ordinal_weight,
            rl_objective=args.rl_objective,
            kl_coef=args.kl_coef,
            anchor_weight=args.anchor_weight,
        )
        train_config = CLMTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            eval_metric=args.eval_metric,
            seed=args.seed,
        )
        heads = train_heads(
            heads,
            splits["train"],
            splits.get("validation"),
            table,
            loss_config=loss_config,
            train_config=train_config,
            device=device,
            log_path=output_dir / "training_log.jsonl",
        )
        save_clm_heads(heads, output_dir / "heads.pt", extra={"posttrain_mode": tag})
        print_rank_0(f"saved post-trained heads to {output_dir / 'heads.pt'}")
    for split in ("validation", "test"):
        if split not in splits:
            continue
        summary, predictions = evaluate_heads(
            heads, splits[split], table, device=device, slice_keys=tuple(args.slice_keys)
        )
        summary["run"] = {
            "stage": tag,
            "split": split,
            "init_checkpoint": args.heads or "release",
            "encoder": args.encoder,
            "encoder_revision": args.encoder_revision,
            "heads_revision": args.heads_revision,
            "max_length": args.max_length,
        }
        save_summary(summary, output_dir / f"metrics_{split}.json")
        with open(output_dir / f"predictions_{split}.jsonl", "w") as f:
            for row in predictions:
                f.write(json.dumps(row) + "\n")
        o = summary["overall"]
        print_rank_0(
            f"[{tag}/{split}] accuracy={o['accuracy']:.4f} exact_match={o['exact_match']:.4f} "
            f"reward={o.get('reward', float('nan')):.4f} brier={o['brier']:.4f} ece={o['ece']:.4f}"
        )
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
