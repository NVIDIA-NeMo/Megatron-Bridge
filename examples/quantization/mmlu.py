# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
MMLU evaluation for quantized Megatron-Bridge checkpoints.

Prerequisites:
First, run quantization to create a quantized checkpoint:
    torchrun --nproc_per_node 8 examples/quantization/quantize.py \
        --hf-model-id /models/Llama-3.2-1B-Instruct \
        --export-quant-cfg fp8 \
        --megatron-save-path /models/Llama-3.2-1B-Instruct_fp8_mlm \
        --tp 4 --pp 2

Usage:
    torchrun --nproc_per_node 8 examples/quantization/mmlu.py \
        --hf-model-id /models/Llama-3.2-1B-Instruct \
        --megatron-load-path /models/Llama-3.2-1B-Instruct_fp8_mlm \
        --tp 8 \
        --mmlu-dataset /hf-local/cais/mmlu \
        --fraction 0.05 \
        --lower-bound 0.45
"""

import argparse
import os
import sys
import warnings

import torch
from datasets import load_dataset
from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.core.utils import unwrap_model
from modelopt.torch.utils.plugins.megatron_generate import megatron_generate
from rich.console import Console

from ptq_generate import _validate_quantized_model

warnings.filterwarnings("ignore")

HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
console = Console()

# Hardcoded subject list for offline use (mirrors cais/mmlu test splits).
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def _format_example(example, include_answer: bool = True):
    """Format an MMLU example into a multiple-choice prompt."""
    prompt = example["question"]
    for choice, answer in zip(["A", "B", "C", "D"], example["choices"]):
        prompt += f"\n{choice}. {answer}"
    if include_answer:
        prompt += "\nAnswer: {}\n\n".format(["A", "B", "C", "D"][example["answer"]])
    else:
        prompt += "\nAnswer:"
    return prompt


def _generate_prompt(test_example, dev_examples, few_shots: int = 0):
    """Build a few-shot prompt for a test example."""
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        " ".join(test_example["subject"].split("_"))
    )
    for i in range(few_shots):
        prompt += _format_example(dev_examples[i])
    prompt += _format_example(test_example, include_answer=False)
    return prompt


def run_mmlu(model, tokenizer, mmlu_dataset, fraction, few_shots, lower_bound, disable_tqdm):
    """Run MMLU evaluation and optionally assert a lower bound on average accuracy.

    Args:
        model: Unwrapped quantized Megatron model.
        tokenizer: HuggingFace tokenizer.
        mmlu_dataset: HuggingFace dataset name or local path to the MMLU dataset.
        fraction: Fraction of each subject's test split to evaluate (0.0–1.0).
        few_shots: Number of few-shot examples to prepend to each prompt.
        lower_bound: If set, assert that average accuracy exceeds this value.
        disable_tqdm: Suppress per-subject progress output on rank 0.
    """
    is_rank_0 = torch.distributed.get_rank() == 0
    all_correct = {}

    if is_rank_0:
        console.print(
            f"\n[green]MMLU ({fraction * 100:.1f}%, {few_shots}-shot) evaluation started...[/green]\n"
        )
        print("{:48} | (ACC) | Count/Total".format("Subject"), flush=True)
        print("{:48} | {:5} | {:11}".format("-" * 48, "-" * 5, "-" * 11), flush=True)

    for subject in MMLU_SUBJECTS:
        test_data = load_dataset(mmlu_dataset, subject, split="test")
        dev_data = load_dataset(mmlu_dataset, subject, split="dev")

        correct = []
        for idx, test_example in enumerate(test_data):
            if idx >= fraction * len(test_data):
                break
            prompt = _generate_prompt(test_example, dev_data, few_shots)
            label = ["A", "B", "C", "D"][test_example["answer"]]
            tokens = tokenizer(prompt, return_tensors="pt")
            generated_ids = megatron_generate(
                model, tokens.input_ids.cuda(), osl=2, disable_tqdm=True
            )
            predict = tokenizer.batch_decode(generated_ids)[0].strip()
            correct.append(predict.startswith(label))

        all_correct[subject] = correct

        if is_rank_0 and not disable_tqdm:
            print(
                f"{subject:48} | {sum(correct) / len(correct):.3f} | {sum(correct):5}/{len(correct):5}",
                flush=True,
            )

    avg_correct = [c for correct in all_correct.values() for c in correct]
    avg_acc = sum(avg_correct) / len(avg_correct)

    if is_rank_0:
        print("{:48} | {:5} | {:11}".format("-" * 48, "-" * 5, "-" * 11), flush=True)
        print(
            "{:48} | {:.3f} | {:5}/{:5}".format(
                "average", avg_acc, sum(avg_correct), len(avg_correct)
            ),
            flush=True,
        )
        if lower_bound is not None:
            assert avg_acc > lower_bound, (
                f"MMLU average accuracy {avg_acc:.3f} is below lower bound {lower_bound}"
            )

    return avg_acc


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    megatron_load_path: str = "./quantized_megatron_checkpoint",
    mmlu_dataset: str = "cais/mmlu",
    fraction: float = 0.05,
    few_shots: int = 0,
    lower_bound: float | None = None,
    disable_tqdm: bool = False,
    trust_remote_code: bool | None = None,
) -> None:
    """Load a quantized Megatron-Bridge checkpoint and evaluate on MMLU."""
    if os.environ.get("WORLD_SIZE") is None:
        console.print("This script must be launched with torchrun. Please run:")
        console.print(f"torchrun --nproc_per_node <gpus> {sys.argv[0]}")
        sys.exit(1)

    if not os.path.exists(megatron_load_path):
        console.print(
            f"[red]Error: Quantized checkpoint path {megatron_load_path} does not exist![/red]"
        )
        sys.exit(1)

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=is_safe_repo(
            trust_remote_code=trust_remote_code,
            hf_path=hf_model_id,
        ),
    )

    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = torch.bfloat16

    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    megatron_model = bridge.load_megatron_model(
        megatron_load_path,
        mp_overrides={
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
        },
        wrap_with_ddp=False,
    )
    megatron_model = [m.cuda() for m in megatron_model]

    is_rank_0 = torch.distributed.get_rank() == 0

    if is_rank_0:
        console.print(f"[green]Loaded quantized model from: {megatron_load_path}[/green]")

    unwrapped_model = unwrap_model(megatron_model)[0]
    unwrapped_model.eval()

    _validate_quantized_model(unwrapped_model, is_rank_0)

    run_mmlu(
        unwrapped_model,
        bridge.hf_pretrained.tokenizer,
        mmlu_dataset,
        fraction,
        few_shots,
        lower_bound,
        disable_tqdm,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MMLU evaluation for quantized Megatron-Bridge checkpoints"
    )
    parser.add_argument(
        "--hf-model-id", type=str, default=HF_MODEL_ID,
        help="HuggingFace model ID or local path for tokenizer and model structure",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument(
        "--megatron-load-path", type=str, default="./quantized_megatron_checkpoint",
        help="Path to the quantized Megatron checkpoint",
    )
    parser.add_argument(
        "--mmlu-dataset", type=str, default="cais/mmlu",
        help="HuggingFace dataset name or local path to the MMLU dataset",
    )
    parser.add_argument(
        "--fraction", type=float, default=0.05,
        help="Fraction of each subject's test split to evaluate (default: 0.05)",
    )
    parser.add_argument(
        "--few-shots", type=int, default=0,
        help="Number of few-shot examples to prepend to each prompt",
    )
    parser.add_argument(
        "--lower-bound", type=float, default=None,
        help="Assert that average accuracy exceeds this value (fails CI if not)",
    )
    parser.add_argument("--disable-tqdm", action="store_true", help="Suppress per-subject output")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")

    args = parser.parse_args()
    main(
        args.hf_model_id,
        args.tp,
        args.pp,
        args.ep,
        args.etp,
        args.megatron_load_path,
        args.mmlu_dataset,
        args.fraction,
        args.few_shots,
        args.lower_bound,
        args.disable_tqdm,
        args.trust_remote_code,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
