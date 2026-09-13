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

Runs the reference model forward over the trainer's own pair dataset and writes a
``pair_id``-keyed artifact the trainer loads instead of the reference weights. Data, tokenizer, ``--max-seq-length`` and
the parallel layout (``--tp``, ``--sequence-parallel``) must equal the training run's;
``scoring_metadata.json`` records them and the trainer checks it. Ranks beyond one TP group
form data-parallel replicas that each score a share of the batches; DP only decides which
pairs share a batch, so it is not recorded. The scorer always runs at PP=1, and its artifact
is valid for any training PP.

Example:
    python scripts/dpo/score_reference_logprobs.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output /tmp/capybara_ref_logprobs --num-pairs 1024

With tensor and data parallelism (launch ``--tp`` x DP processes; rank 0 gathers and writes):
    python -m torch.distributed.run --nproc_per_node=8 \\
        scripts/dpo/score_reference_logprobs.py --tp 2 \\
        --model Qwen/Qwen2.5-7B-Instruct --output /tmp/uf_ref_logprobs
"""

import argparse
import os
from dataclasses import asdict

import torch
from megatron.core import parallel_state
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.pipeline_parallel import get_forward_backward_func
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders.dpo import DPODatasetConfig, PreferenceSource, build_preference_split
from megatron.bridge.data.datasets.preference import (
    ScoringFingerprint,
    pack_pairs_by_token_budget,
    pair_token_lengths,
    write_ref_logprobs,
    write_scoring_metadata,
)
from megatron.bridge.data.datasets.preference_pair import PreferencePairDataset
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.data.sources.jsonl import JSONLSourceConfig
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.training.dpo import sequence_logprob_sums, split_pair_rows
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.utils.common_utils import (
    get_rank_safe,
    get_world_size_safe,
    maybe_initialize_distributed,
    print_rank_0,
)


# The trainer's default recipe; applied here too so the reference forward runs the policy's numerics
# whatever precision the checkpoint declares.
MIXED_PRECISION_RECIPE = "bf16_mixed"

# One artifact row: pair_id plus the chosen/rejected logprob sums and token counts.
PairRecord = dict[str, float | int]


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset",
        help="HF hub id, local path, or a single JSONL file (local or msc://); must match training",
    )
    parser.add_argument("--split", default="train")
    source_group.add_argument(
        "--jsonl",
        nargs="+",
        default=None,
        help="JSONL preference rows (local or msc://) read in place, instead of --dataset",
    )
    parser.add_argument(
        "--index-mapping-dir",
        default=os.path.expanduser("~/.cache/megatron_bridge/dpo_index"),
        help="Where JSONL memmap index sidecars go; required for msc:// paths",
    )
    parser.add_argument("--chosen-key", default="chosen")
    parser.add_argument("--rejected-key", default="rejected")
    parser.add_argument(
        "--prompt-key",
        default=None,
        help="Field holding the shared prompt when chosen/rejected hold only the completion turns; "
        "must match training",
    )
    parser.add_argument("--model", required=True, help="Reference model (HF path/id the policy initializes from)")
    parser.add_argument("--tokenizer", default=None, help="Defaults to --model; must match training")
    parser.add_argument("--output", required=True, help="msc:// URL or local directory for the artifact")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Must match training")
    parser.add_argument("--num-pairs", type=int, default=0, help="Score only the first N source rows (0 = all)")
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
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size; must match training. Ranks beyond one TP group become data-parallel replicas",
    )
    parser.add_argument(
        "--sequence-parallel",
        action="store_true",
        help="Shard activations along the sequence across the TP group (required for MoE at TP > 1); "
        "must match training",
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size (MoE only)")
    parser.add_argument("--etp", type=int, default=None, help="Expert tensor parallel size (default: --tp)")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.tp < 1 or args.ep < 1 or (args.etp is not None and args.etp < 1):
        parser.error(f"--tp, --ep and --etp must be >= 1; got --tp {args.tp} --ep {args.ep} --etp {args.etp}.")
    if args.etp is None:
        args.etp = args.tp
    if args.micro_batch_size is None and args.token_budget is None:
        args.micro_batch_size = 8
    return args


def validate_scorer_world_size(world_size: int, *, tp: int, ep: int, etp: int) -> int:
    """Return the data-parallel size. TP and the EP x ETP expert mesh must both tile the world."""
    expert_mesh = ep * etp
    if world_size % tp != 0 or world_size % expert_mesh != 0:
        raise SystemExit(
            f"World size {world_size} must be a multiple of both --tp {tp} and EP x ETP = {ep} x {etp} = "
            f"{expert_mesh}. The ranks beyond one TP group form the data-parallel replicas that split the pairs."
        )
    return world_size // tp


def shard_batches(batch_plan: list[list[int]], *, rank: int, size: int) -> list[list[int]]:
    """Deal whole batches round-robin to the data-parallel replicas, so a length-sorted plan stays balanced."""
    return batch_plan[rank::size]


def dpo_source(args: argparse.Namespace) -> PreferenceSource:
    """The preference source these CLI args select; a path-shaped --dataset is routed to JSONL by the config."""
    if args.jsonl:
        return JSONLSourceConfig(paths=list(args.jsonl), index_mapping_dir=args.index_mapping_dir)
    return HFDatasetSourceConfig(path_or_dataset=args.dataset, split=args.split)


class ReferenceLogprobScorer:
    """Forward the reference model over every pair and write the ref-logprob artifact."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tokenizer_name: str = args.tokenizer or args.model
        self.dataset_config = DPODatasetConfig(
            tokenizer_name=self.tokenizer_name,
            seq_length=args.max_seq_length,
            source=dpo_source(args),
            index_mapping_dir=args.index_mapping_dir,
            ref_artifact=args.output,
            num_pairs=args.num_pairs,
            chosen_key=args.chosen_key,
            rejected_key=args.rejected_key,
            prompt_key=args.prompt_key,
            pad_seq_length_to_mult=args.tp if args.sequence_parallel else 1,
        )

    def run(self) -> None:
        """Load data + model, score all pairs, write the artifact."""
        maybe_initialize_distributed()
        source_key, _ = self.dataset_config.source_identity(self.dataset_config.source)
        print_rank_0(f"loading {source_key} and tokenizer {self.tokenizer_name}...")

        # The trainer's own builder, so both sides see identical rows, token streams, and pair_ids.
        dataset = build_preference_split(self.dataset_config, "train", self.dataset_config.load_tokenizer())
        model = self._build_model()
        records = self._score_all(model, dataset)

        if get_rank_safe() == 0:
            write_ref_logprobs(records, self.args.output)
            self._commit_artifact(num_pairs=len(dataset))
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

    def _build_model(self) -> list[torch.nn.Module]:
        """Forward-only Megatron bring-up (no ConfigContainer/optimizer)."""
        validate_scorer_world_size(get_world_size_safe(), tp=self.args.tp, ep=self.args.ep, etp=self.args.etp)
        bridge = AutoBridge.from_hf_pretrained(self.args.model, torch_dtype=torch.bfloat16)
        provider = bridge.to_megatron_provider(load_weights=True)
        self._configure_provider(provider)
        provider.finalize()
        provider.initialize_model_parallel(seed=self.args.seed)
        model = provider.provide_distributed_model(wrap_with_ddp=False)
        model = [m.cuda() for m in model]
        for m in model:
            m.eval()
        return model

    def _configure_provider(self, provider: GPTModelProvider) -> None:
        """Apply the trainer's precision recipe, the scorer's parallel layout, and the no-MTP pin."""
        get_mixed_precision_config(MIXED_PRECISION_RECIPE).setup(provider)
        overrides = {
            "tensor_model_parallel_size": self.args.tp,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": self.args.ep,
            "expert_tensor_parallel_size": self.args.etp,
            "sequence_parallel": self.args.sequence_parallel,
            "seq_length": self.args.max_seq_length,
            "mtp_num_layers": None,  # an MTP head would add its own next-token loss on top of the DPO loss
        }
        for name, value in overrides.items():
            if not hasattr(provider, name):
                raise ValueError(f"Model provider {type(provider).__name__} has no '{name}' to pin.")
            setattr(provider, name, value)

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
        if self.args.token_budget:
            print_rank_0(f"token-budget batching: measuring {len(dataset)} pair lengths (one CPU pass)...")
            batch_plan = pack_pairs_by_token_budget(
                pair_token_lengths(dataset),
                budget_tokens=self.args.token_budget,
                max_pairs_per_batch=self.args.max_pairs_per_batch,
            )
            print_rank_0(
                f"token-budget batching: {len(batch_plan)} batches under {self.args.token_budget} padded tokens"
            )
            return batch_plan
        sampler = BatchSampler(SequentialSampler(dataset), batch_size=self.args.micro_batch_size, drop_last=False)
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
        if self.args.ep == 1 and self.args.etp == self.args.tp:
            return {}
        return {"expert_model_parallel_size": self.args.ep, "expert_tensor_parallel_size": self.args.etp}

    def _commit_artifact(self, num_pairs: int) -> None:
        """Publish the scoring metadata; the trainer refuses a table without it."""
        dataset_key, split = self.dataset_config.source_identity(self.dataset_config.source)
        fingerprint = ScoringFingerprint(
            dataset=dataset_key,
            split=split,
            tokenizer=self.tokenizer_name,
            max_seq_length=self.args.max_seq_length,
            prompt_key=self.args.prompt_key,
            tensor_model_parallel_size=self.args.tp,
            sequence_parallel=bool(self.args.sequence_parallel),
        )
        write_scoring_metadata(
            {
                **asdict(fingerprint),
                **self._expert_layout_metadata(),
                "model": self.args.model,
                "num_pairs": num_pairs,
                "mixed_precision": MIXED_PRECISION_RECIPE,
            },
            self.args.output,
        )
        print_rank_0(f"artifact committed to {self.args.output} ({num_pairs} pairs)")


def main() -> None:
    """Score the reference model over the preference dataset and publish the artifact."""
    MultiStorageClientFeature.enable()
    ReferenceLogprobScorer(parse_args()).run()


if __name__ == "__main__":
    main()
