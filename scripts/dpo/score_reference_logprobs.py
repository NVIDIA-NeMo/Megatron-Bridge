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

"""Score reference-model logprobs for every DPO preference pair (offline, one-time).

Runs the reference model forward over the training recipe's own pair dataset and writes a
``pair_id``-keyed artifact the trainer loads instead of the reference weights. The dataset,
tokenizer, sequence length, precision and parallel layout (TP, EP, ETP, sequence parallelism)
are the recipe's, so both jobs see identical rows and token streams; ``scoring_metadata.json``
records them and the trainer checks it. Ranks beyond one TP group form data-parallel replicas
that each score a share of the batches; DP only decides which pairs share a batch, so it is not
recorded. The scorer always runs at PP=1, and its artifact is valid for any training PP.

Trailing ``KEY=VALUE`` overrides are applied to the recipe before anything is read from it.

Example (one GPU, the recipe's dataset, first 1024 pairs):
    python scripts/dpo/score_reference_logprobs.py \\
        --recipe qwen3_30b_a3b_dpo_config --output /tmp/uf_ref_logprobs dataset.num_pairs=1024

With the recipe's parallel layout (launch TP x DP processes; rank 0 gathers and writes):
    python -m torch.distributed.run --nproc_per_node=8 \\
        scripts/dpo/score_reference_logprobs.py --recipe qwen3_30b_a3b_dpo_config \\
        --model /checkpoints/qwen3-30b-a3b-hf --output /data/uf_ref_logprobs
"""

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from megatron.core import parallel_state
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.pipeline_parallel import get_forward_backward_func
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders.dpo import DPODatasetConfig, build_preference_split, hf_chat_tokenizer
from megatron.bridge.data.datasets.preference import (
    ScoringFingerprint,
    pack_pairs_by_token_budget,
    pair_token_lengths,
    write_ref_logprobs,
    write_scoring_metadata,
)
from megatron.bridge.data.datasets.preference_pair import PreferencePairDataset
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.dpo import sequence_logprob_sums, split_pair_rows
from megatron.bridge.training.dpo_train import expected_scoring_metadata
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.checkpoint_utils import is_hf_checkpoint_dir
from megatron.bridge.utils.common_utils import (
    get_rank_safe,
    get_world_size_safe,
    maybe_initialize_distributed,
    print_rank_0,
)


# Recipe resolution and override parsing are the training launcher's.
TRAINING_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "training"
if str(TRAINING_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_SCRIPT_DIR))

from recipe_runner import apply_cli_overrides, load_recipe, sync_model_dataset_sequence_length  # noqa: E402


# One artifact row: pair_id plus the chosen/rejected logprob sums and token counts.
PairRecord = dict[str, float | int]


@dataclass(kw_only=True)
class ScoringOptions:
    """What the scorer decides on top of the recipe: where the artifact goes and how pairs are batched."""

    output: str
    split: Literal["train", "validation"] = "train"
    model: str | None = None
    """HF path or id of the reference weights; see ``reference_weights_path`` for the default."""
    micro_batch_size: int | None = 8
    token_budget: int | None = None
    max_pairs_per_batch: int = 64
    seed: int = 0


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--recipe",
        required=True,
        help="DPO recipe function name, e.g. qwen3_30b_a3b_dpo_config. Its dataset, tokenizer, sequence length, "
        "precision and TP/EP/ETP/SP layout are the scoring run's (PP is always 1)",
    )
    parser.add_argument("--output", required=True, help="msc:// URL or local directory for the artifact")
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="train",
        help="Which of the recipe's preference splits to score; validation needs dataset.validation_source",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HF path or id of the reference weights (default: checkpoint.pretrained_checkpoint when it is an HF "
        "checkpoint, else the recipe's HF model)",
    )
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Pairs per forward pass, each pair is 2 rows (default 8 when --token-budget is unset)",
    )
    batch_group.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Length-sorted batches capped at this many padded tokens instead of a fixed pair count",
    )
    parser.add_argument(
        "--max-pairs-per-batch",
        type=int,
        default=64,
        help="Pair cap per --token-budget batch; ignored otherwise",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Recipe config overrides, e.g. dataset.num_pairs=1024 model.tensor_model_parallel_size=2",
    )

    args = parser.parse_args()
    for override in args.overrides:
        if override.startswith("-") or "=" not in override:
            parser.error(f"Expected a recipe override in KEY=VALUE form, got {override!r}")
    if args.micro_batch_size is None and args.token_budget is None:
        args.micro_batch_size = 8
    return args


def load_scoring_config(recipe: str, overrides: list[str]) -> ConfigContainer:
    """The training recipe with the CLI overrides applied; must carry a DPO dataset."""
    config = sync_model_dataset_sequence_length(apply_cli_overrides(load_recipe(recipe), overrides))
    if not isinstance(config.dataset, DPODatasetConfig):
        raise SystemExit(
            f"--recipe {recipe} is not a DPO recipe: its dataset is {type(config.dataset).__name__}, "
            "not DPODatasetConfig."
        )
    return config


def reference_weights_path(config: ConfigContainer, override: str | None) -> str:
    """Where the reference weights come from: ``--model``, else the recipe's pretrained checkpoint, else its HF model."""
    path = override or config.checkpoint.pretrained_checkpoint or config.tokenizer.tokenizer_model
    if not path:
        raise SystemExit("No reference weights: pass --model or set checkpoint.pretrained_checkpoint.")
    if os.path.isdir(path) and not is_hf_checkpoint_dir(path):
        raise SystemExit(
            f"{path} is not a HuggingFace model directory. The scorer loads the reference in HF format; pass "
            "--model with the HF checkpoint the policy initializes from."
        )
    return path


def validate_scorer_world_size(world_size: int, *, tp: int, ep: int, etp: int) -> int:
    """Return the data-parallel size. TP and the EP x ETP expert mesh must both tile the world."""
    expert_mesh = ep * etp
    if world_size % tp != 0 or world_size % expert_mesh != 0:
        raise SystemExit(
            f"World size {world_size} must be a multiple of both TP {tp} and EP x ETP = {ep} x {etp} = "
            f"{expert_mesh}. The ranks beyond one TP group form the data-parallel replicas that split the pairs."
        )
    return world_size // tp


def shard_batches(batch_plan: list[list[int]], *, rank: int, size: int) -> list[list[int]]:
    """Deal whole batches round-robin to the data-parallel replicas, so a length-sorted plan stays balanced."""
    return batch_plan[rank::size]


class ReferenceLogprobScorer:
    """Forward the reference model over every pair of one split and write the ref-logprob artifact."""

    def __init__(self, config: ConfigContainer, options: ScoringOptions) -> None:
        self.config = config
        self.options = options
        self.dataset_config: DPODatasetConfig = config.dataset

    def run(self) -> None:
        """Load data + model, score all pairs, write the artifact."""
        model_path = reference_weights_path(self.config, self.options.model)
        fingerprint = expected_scoring_metadata(self.config, self.options.split)
        provider = self.config.model
        self.dataset_config.pad_seq_length_to_mult = (
            provider.tensor_model_parallel_size if provider.sequence_parallel else 1
        )

        maybe_initialize_distributed()
        print_rank_0(f"loading {fingerprint.dataset} and tokenizer {fingerprint.tokenizer}...")

        # The trainer's own builder, so both sides see identical rows, token streams, and pair_ids.
        tokenizer = hf_chat_tokenizer(build_tokenizer(self.config.tokenizer))
        dataset = build_preference_split(self.dataset_config, self.options.split, tokenizer)
        model = self._build_model(model_path)
        records = self._score_all(model, dataset)

        if get_rank_safe() == 0:
            write_ref_logprobs(records, self.options.output)
            self._commit_artifact(fingerprint, model_path, num_pairs=len(dataset))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # don't tear down the process group under the writer

    def _score_all(self, model: list[torch.nn.Module], dataset: PreferencePairDataset) -> list[PairRecord]:
        """Score this replica's batches, then gather every replica's records; complete and pair_id-sorted on all ranks."""
        loader = self._pair_loader(dataset)
        replica_pairs = sum(len(batch) for batch in loader.batch_sampler)
        log_every = max(1, len(loader) // 10)
        records: list[PairRecord] = []
        for step, batch in enumerate(loader):
            records.extend(self._score_batch(model, batch))
            if step % log_every == 0 or step == len(loader) - 1:
                print_rank_0(f"scored {len(records)}/{replica_pairs} pairs on replica 0 of {self._dp_size()}")
        return sorted(self._gather_records(records), key=lambda record: record["pair_id"])

    @staticmethod
    def _dp_size() -> int:
        return parallel_state.get_data_parallel_world_size() if torch.distributed.is_initialized() else 1

    def _gather_records(self, records: list[PairRecord]) -> list[PairRecord]:
        """Concatenate the replicas' records; TP ranks within a replica hold identical copies, so DP suffices."""
        dp_size = self._dp_size()
        if dp_size == 1:
            return records
        shards = [None] * dp_size
        torch.distributed.all_gather_object(shards, records, group=parallel_state.get_data_parallel_group())
        return [record for shard in shards for record in shard]

    def _expert_layout(self) -> tuple[int, int]:
        provider = self.config.model
        return (
            provider.expert_model_parallel_size or 1,
            provider.expert_tensor_parallel_size or provider.tensor_model_parallel_size,
        )

    def _build_model(self, model_path: str) -> list[torch.nn.Module]:
        """Forward-only Megatron bring-up of the recipe's provider, with the reference weights loaded from HF."""
        provider = self.config.model
        ep, etp = self._expert_layout()
        validate_scorer_world_size(get_world_size_safe(), tp=provider.tensor_model_parallel_size, ep=ep, etp=etp)
        self._configure_provider(provider)
        provider.finalize()
        provider.initialize_model_parallel(seed=self.options.seed)
        model = [m.cuda() for m in provider.provide_distributed_model(wrap_with_ddp=False)]
        bridge = AutoBridge.from_hf_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.config.checkpoint.hf_trust_remote_code,
        )
        bridge.load_hf_weights(model)
        for m in model:
            m.eval()
        return model

    def _configure_provider(self, provider) -> None:
        """Apply the recipe's precision, collapse PP to 1, and pin the no-MTP contract."""
        precision = self.config.mixed_precision
        if isinstance(precision, str):
            precision = get_mixed_precision_config(precision)
        if precision is not None:
            precision.setup(provider)
        provider.pipeline_model_parallel_size = 1  # the artifact is valid for any training PP
        provider.virtual_pipeline_model_parallel_size = None
        provider.mtp_num_layers = None  # an MTP head would add its own next-token loss on top of the DPO loss

    def _pair_loader(self, dataset: PreferencePairDataset) -> DataLoader:
        """This replica's share of a deterministic full-coverage batch plan; records carry pair_id, so order is free."""
        batch_plan = self._batch_plan(dataset)
        if torch.distributed.is_initialized():
            batch_plan = shard_batches(
                batch_plan,
                rank=parallel_state.get_data_parallel_rank(),
                size=parallel_state.get_data_parallel_world_size(),
            )
        return DataLoader(dataset, batch_sampler=batch_plan, collate_fn=dataset.collate_fn, pin_memory=True)

    def _batch_plan(self, dataset: PreferencePairDataset) -> list[list[int]]:
        """Every pair index exactly once: length-sorted token-budget batches, or fixed-size sequential ones."""
        if self.options.token_budget:
            print_rank_0(f"token-budget batching: measuring {len(dataset)} pair lengths (one CPU pass)...")
            batch_plan = pack_pairs_by_token_budget(
                pair_token_lengths(dataset),
                budget_tokens=self.options.token_budget,
                max_pairs_per_batch=self.options.max_pairs_per_batch,
            )
            print_rank_0(
                f"token-budget batching: {len(batch_plan)} batches under {self.options.token_budget} padded tokens"
            )
            return batch_plan
        sampler = BatchSampler(SequentialSampler(dataset), batch_size=self.options.micro_batch_size, drop_last=False)
        return [list(batch) for batch in sampler]

    @staticmethod
    def _score_batch(model: list[torch.nn.Module], batch: dict[str, torch.Tensor]) -> list[PairRecord]:
        """One forward pass; returns one record per pair via the shared DPO sum math."""
        rows = {key: batch[key].cuda() for key in ("tokens", "labels", "loss_mask", "position_ids")}

        def forward_step(data_iterator, megatron_model):
            data = next(data_iterator)

            def collect(per_token_nll, non_loss_data=True):
                sums, counts = sequence_logprob_sums(per_token_nll, data["loss_mask"])
                return sums.cpu(), counts.cpu()

            output = megatron_model(
                input_ids=data["tokens"],
                position_ids=data["position_ids"],
                attention_mask=None,
                labels=data["labels"],
            )
            return output, collect

        with torch.no_grad():
            collected = get_forward_backward_func()(
                forward_step_func=forward_step,
                data_iterator=iter([rows]),
                model=model,
                num_microbatches=1,
                forward_only=True,
                collect_non_loss_data=True,
                seq_length=rows["tokens"].size(1),
                micro_batch_size=rows["tokens"].size(0),
            )

        sums, counts = collected[0]
        chosen_sums, rejected_sums = split_pair_rows(sums)
        chosen_counts, rejected_counts = split_pair_rows(counts)
        pair_ids = batch["pair_id"][::2]
        return [
            {
                "pair_id": int(pair_ids[i]),
                "ref_chosen_logprob_sum": float(chosen_sums[i]),
                "ref_chosen_num_tokens": int(chosen_counts[i]),
                "ref_rejected_logprob_sum": float(rejected_sums[i]),
                "ref_rejected_num_tokens": int(rejected_counts[i]),
            }
            for i in range(len(pair_ids))
        ]

    def _expert_layout_metadata(self) -> dict[str, int]:
        """The (EP, ETP) keys, omitted when they match the implied EP=1 / ETP=TP layout."""
        ep, etp = self._expert_layout()
        if ep == 1 and etp == self.config.model.tensor_model_parallel_size:
            return {}
        return {"expert_model_parallel_size": ep, "expert_tensor_parallel_size": etp}

    def _commit_artifact(self, fingerprint: ScoringFingerprint, model_path: str, *, num_pairs: int) -> None:
        """Publish the scoring metadata; the trainer refuses a table without it."""
        precision = self.config.mixed_precision
        write_scoring_metadata(
            {
                **asdict(fingerprint),
                **self._expert_layout_metadata(),
                "model": model_path,
                "num_pairs": num_pairs,
                "mixed_precision": precision if isinstance(precision, str) else None,
            },
            self.options.output,
        )
        print_rank_0(f"artifact committed to {self.options.output} ({num_pairs} pairs)")


def main() -> None:
    """Score the reference model over the recipe's preference dataset and publish the artifact."""
    MultiStorageClientFeature.enable()
    args = parse_args()
    options = ScoringOptions(
        output=args.output,
        split=args.split,
        model=args.model,
        micro_batch_size=args.micro_batch_size,
        token_budget=args.token_budget,
        max_pairs_per_batch=args.max_pairs_per_batch,
        seed=args.seed,
    )
    ReferenceLogprobScorer(load_scoring_config(args.recipe, args.overrides), options).run()


if __name__ == "__main__":
    main()
