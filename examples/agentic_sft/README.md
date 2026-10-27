# Agentic SFT: full Qwen3.5-35B-A3B on 4×GB200

Supervised fine-tuning of the **full 40-layer Qwen3.5-35B-A3B** (plus one MTP layer) on long agentic
tool-use trajectories, on a single four-GPU GB200 node. Tools are conversation data and are never executed.

This is a Megatron Bridge port of the Megatron-LM agentic SFT example by Zhongbo Zhu
([`examples/agentic_sft`, commit `420686e`](https://github.com/zhongbozhu/Megatron-LM/commit/420686e1a7e9d51d422e8ac1a18a35818843bb86)).
The data contract, selection logic and training settings are theirs; the tokenization and training stack is Bridge's.
The performance numbers in that example were measured with native Megatron-LM and are **not** Bridge measurements.

Run the commands from the repository root. All generated assets stay in a workspace
(`agentic_sft_workspace/` by default), and preparation downloads no model weights.

## What is here

| File | Purpose |
| --- | --- |
| `01_prepare_qwen35_35b_tokenizer.py` | Pins the Qwen3.5-35B-A3B tokenizer and text config, and installs the generation-tagged training template |
| `02_prepare_coderforge.py` | Downloads a bounded, revision-pinned CoderForge SWE-Rebench subset and splits it by task |
| `02_5_filter_coderforge.py` | Selects trajectories by effective training length: `natural` or `mixed` |
| `04_train_qwen35_35b_a3b.py` | Builds the Bridge `ConfigContainer` and trains |
| `agentic_sft_data.py` | Shared helpers: JSONL reader, chat encoding, truncation |
| `templates/qwen3_5_sft_v1.jinja` | Training chat template with `{% generation %}` blocks |

The original example's offline packing step (`03_pack_coderforge.py`) and its `offline_dynamic_cp` mode are not
part of this port; see [Not included](#not-included).

## Requirements

* Four GB200 GPUs and a container with Transformer Engine, CuTeDSL and DeepEP HybridEP.
* Large host memory: the default FP32 CPU optimizer offload keeps master weights and Adam state on the host. The
  full-model smoke run below completed within a 900 GB allocation; peak host usage was not measured.
* Megatron-Core: `online_static_cp` runs on the pinned Megatron-Core. `online_dynamic_cp` additionally needs
  [NVIDIA/Megatron-LM#7048](https://github.com/NVIDIA/Megatron-LM/pull/7048) until it is merged and pinned. See
  [Dynamic CP and Megatron-Core](#dynamic-cp-and-megatron-core).

## Model and modes

The model has **40 decoder layers**: ten cycles of 3 GDN + 1 attention, plus **MTP1 (loss coefficient 0.1)**. It is
built from the config that script 01 writes next to the tokenizer, so the tokenizer, template and architecture come
from one verified directory. Hidden size 2048, 16 attention heads and 2 query groups, 256 top-8 experts (FFN 512),
a gated shared expert, vocabulary 248320.

Both modes use **64K context, 16384 tokens per rank, TP1/PP1/base CP4/EP4/ETP1, micro batch 1** and per-token loss.

| Mode | Scheduler | Runtime CP | Default global batch |
| --- | --- | --- | --- |
| `online_static_cp` | `dp_balanced` | 4 | 32 trajectories |
| `online_dynamic_cp` | `default_dynamic_cp` | 1, 2, 4 | 32 trajectories |

With four ranks the base data-parallel size is 1, so static mode does not measure multi-DP balancing.

By default the decoder and MTP layers are initialized from scratch. To start from a converted, shape-compatible
Megatron checkpoint pass `--init-mode checkpoint --pretrained-checkpoint PATH`; only weights are loaded. Conversion is
outside this example. `--resume` continues a run from its own `--save-dir`, and a fresh run refuses to overwrite an
existing checkpoint.

## Prepare the data

Script 02 defaults to 5% of the pinned CoderForge SWE_Rebench split. The mixed distribution below needs enough unique
trajectories, so request more. Sampling uses seed-shuffled shards and a bounded prefix; it is not uniform row sampling.
See the [dataset card](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) for its licenses.

* `natural` keeps every trainable trajectory, preserving the long-heavy distribution.
* `mixed` targets **5,120 training and 256 validation samples** with **50% needing at most CP1 capacity (≤16K tokens),
  30% CP2 (16–32K) and 20% CP4 (above 32K)**. It prefers complete trajectories and fills short and medium shortages
  with real prefixes that end at complete assistant messages, including tool calls. Every source trajectory contributes
  at most one sample and stays in its original split. The ratios are sample counts, not token shares.

```bash
EXAMPLE=examples/agentic_sft
WORKSPACE=agentic_sft_workspace
TOKENIZER="${WORKSPACE}/models/qwen35_35b_a3b/tokenizer"

uv run python "${EXAMPLE}/01_prepare_qwen35_35b_tokenizer.py" --workspace "${WORKSPACE}"
uv run python "${EXAMPLE}/02_prepare_coderforge.py" --percentage 50 \
    --output-dir "${WORKSPACE}/data/coderforge_candidates"
for distribution in natural mixed; do
    uv run python "${EXAMPLE}/02_5_filter_coderforge.py" \
        --data-dir "${WORKSPACE}/data/coderforge_candidates" \
        --output-dir "${WORKSPACE}/data/coderforge_${distribution}" \
        --distribution "${distribution}" \
        --tokenizer "${TOKENIZER}"
done
```

`02_5_filter_coderforge.py` defaults to 32 tokenizer processes and a 64K sequence length. Mixed-only options are
`--num-train-samples`, `--num-validation-samples` and `--length-bucket-ratios`. Insufficient unique parents or usable
prefixes fail explicitly. The manifest records actual lengths, token shares, prefix endpoints and source IDs. Rows
without any assistant target are excluded and reported, and any other error aborts. Use a fresh output directory
whenever the selection, supervision or tokenizer provenance changes.

## Train

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 \
    examples/agentic_sft/04_train_qwen35_35b_a3b.py online_dynamic_cp \
    --data-dir agentic_sft_workspace/data/coderforge_mixed \
    --train-iters 75
```

`--dry-run` builds and prints the configuration without launching. Arguments the script does not recognize are applied
as dotted `ConfigContainer` overrides, for example `train.train_iters=20`. Run `--help` for every option.

## Supervision

Targets come from the **chat template's own `{% generation %}` mask**, selected with
`ChatSFTPreprocessingConfig(loss_mode="assistant", mask_source="generation")`. The template installed by script 01
supervises assistant reasoning, tool calls, body and end-of-turn tokens, while role headers and other roles are
context. Historical reasoning is retained. No assistant boundaries are guessed from rendered text, and a template
without generation blocks fails explicitly instead of silently training on nothing.

Each trajectory is rendered once, truncated on the right to the sequence length plus one token, given no added EOS, and
shifted once. A trajectory with no assistant target left is rejected by the filter rather than trained as an empty row.

`mask_source="generation"` is an opt-in setting of Bridge's chat preprocessing. The default, `auto`, infers turn
boundaries by rendering each message prefix; on these trajectories that is about 135 renders per row (roughly 12 s)
versus one render (about 0.2 s), with identical tokens and masks on the rows checked.

## Performance settings

| Option | Default | Effect |
| --- | --- | --- |
| `--precision` | `mxfp8` | MXFP8 compute with the TE op fuser, grouped tensors and CuTeDSL; `bf16` disables this bundle |
| `--optimizer-cpu-offload` | on | CPU Adam with FP32 master weights, gradients and moments; BF16 parameter gather. Off keeps the optimizer on the GPU and enables FP8 parameter gather with MXFP8 |
| `--hybrid-ep` | on | HybridEP with uneven-input padding; off selects all-to-all |
| `--full-recompute` | on | Uniform full recompute; off recomputes only GDN norm and MoE |
| `--ddp-overlap` | on | Overlap gradient reduction and parameter gather |

Both precisions use BF16 as the base precision. Per-token loss, FP32 router and gradient accumulation, native fused
cross entropy, zero dropout and an MoE auxiliary-loss coefficient of 0.001 stay on. With CPU offload the script sets
12 OpenMP threads per rank, otherwise one.

Differences from the original example that can matter for performance:

* Bridge derives the per-sequence alignment padding itself (`2 × data-parallel × CP`, 8 here); the original used a fixed 32.
* Sequence alignment padding is folded into each sequence (`fold_alignment_padding`), which Qwen3.5's 256-wide
  attention heads need so that FlashAttention sees no padding between packed sequences.
* The GDN kernel backend is the Megatron-Core default; the existing 16-GPU recipe selects `transformer_engine` instead.

## Dynamic CP and Megatron-Core

`online_dynamic_cp` regroups the four ranks as CP1, CP2 or CP4 for each packed bin. On the Megatron-Core this branch
pins, Gated DeltaNet layers still assume the static CP size, so a bin that runs on a smaller group fails:

```text
GDN: cu_seqlens_q[-1]=16384 does not match total_sequence_length=65536
```

[NVIDIA/Megatron-LM#7048](https://github.com/NVIDIA/Megatron-LM/pull/7048) propagates the runtime CP group through
hybrid layers and normalizes the MTP loss across it. Until it is pinned, apply it to your Megatron-Core checkout.

A dataset whose samples all need the full CP4 group (for example, only >32K-token trajectories) never forms a smaller
group and so also runs on the pinned Megatron-Core. The `mixed` distribution does form them.

## Not included

* `03_pack_coderforge.py` and the `offline_dynamic_cp` mode (training on pre-packed Parquet rows). Bridge already has
  the offline packing building blocks; the mode additionally needs scheduler changes for packed rows under dynamic CP,
  so it is left to a follow-up.
* The original BF16 numerical diagnostic, which was documented as failing its own tolerances.

## Validation status

Smoke-tested on one GB200 node (four GPUs) in a NeMo 26.10 container, Megatron-Core main at
`ed9c2c8c1`, from-scratch initialization, MXFP8, CPU optimizer offload and HybridEP. Each run used 2-3 optimizer steps on
a few real CoderForge trajectories. They show that the paths run end to end with finite, decreasing loss; they are not
performance measurements or convergence evidence, and step times include one-time compilation or are from tiny runs.

| Run | Data | Result |
| --- | --- | --- |
| `online_static_cp`, 8 layers, pinned Megatron-Core | natural | 3 steps; lm loss 12.850, 12.155, 11.571 |
| `online_dynamic_cp`, 8 layers, with #7048 | natural | 3 steps; lm loss 12.850, 12.157, 11.575 |
| `online_dynamic_cp`, 8 layers, with #7048 | mixed (CP1/CP2/CP4 bins) | 2 steps; lm loss 12.842, 12.024 |
| `online_dynamic_cp`, 8 layers, pinned Megatron-Core | mixed | fails with the GDN error above |
| `online_dynamic_cp`, 8 layers, with #7048, evaluation on the validation split | natural | 2 steps then evaluation; validation lm loss 11.77 |
| `online_dynamic_cp`, **full 40 layers + MTP1**, with #7048 | natural | 3 steps; lm loss 12.833, 12.245, 11.434; MTP loss 12.81, 12.77, 12.69; peak 113.6 GB allocated per GPU; 464 GB checkpoint (with optimizer state) written in about 97 s |

Static and dynamic mode have identical first-step loss on the same data (same initialization and batch), and nearly
equal losses afterwards, as expected from different microbatch grouping. A static-versus-dynamic throughput
comparison has not been measured with Bridge. The 75-step CP-scheduling measurements in the original example were
made with native Megatron-LM and should not be read as Bridge results.

## Tests

CPU-only, no GPU needed:

```bash
uv run python -m pytest tests/unit_tests/examples/test_agentic_sft_*.py
```

The template tests use the pinned local tokenizer and skip when it is unavailable; set `QWEN35_SOURCE_TOKENIZER` to
its snapshot directory when it lives outside the Hugging Face cache.
