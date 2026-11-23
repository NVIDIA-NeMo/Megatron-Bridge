# Qwen3.8-Flash-Next VLM

This guide covers `Qwen/Qwen3.8-Flash-Next` with `text_only=False` (the default):
the language model, vision encoder and projector. The [text-only guide](qwen3.8-flash-next-text-only.md)
records the separate standalone language workflow and fixed-4K pretraining result.
Both modes share the hybrid decoder, QSA, GDN, MoE, gated residual and PLE modules.

The image-SFT recipe is **EARLY, UNOPTIMIZED**. It trains the language model,
encoder and projector on 64 GB200 GPUs with TP4/SP, EP64, sequence 2048 and Adam.
Natural routing and HybridEP with MoE fusions are enabled.

Released NeMo 26.10 verification completed on 2026-11-22. Full-VLM GPU
conversion preserved all 1,627 supported tensors exactly, including 333 vision
tensors; unsupported MTP tensors are excluded. The auxiliary export-checker
failure and separate successful asset recheck are recorded in the card.

| Workload | GPUs / sequence / updates | Initial → final loss | Final-ten mean step | TFLOP/s/GPU | Token slots/s/GPU |
| --- | --- | --- | --- | --- | --- |
| Full image SFT / Adam | 64 / 2048 / 100 | 0.351589 → 0.167965 | 5.458 s | 5.89 | 93.81 |

All ranks completed finite, unskipped updates. These are bounded functional
observations, not optimized benchmarks. Full-VLM pretraining is unverified.
The historical 8,192-sample, 5,349-step image-SFT loss curve remains separate evidence.

See the [VLM verification card](../../../examples/model_verification_cards/qwen3.8-flash-next/card.yaml)
for exact source/container revisions and conversion/training limits, and the
[training instructions](../../../examples/models/qwen/qwen4_exp/README.md) for
public commands and data preparation. Native VLM and text checkpoints have
different namespaces and must be imported separately. Functional training does
not establish cross-framework trajectory parity, resume or held-out quality.

## Frozen PLE table sharding

`model.ple_embedding_sharding="ep"` distributes the frozen n-gram table over
the expert-parallel group, independently of attention tensor parallelism.
Enable it together with `model.ple_freeze_ngram_embeddings=True` and
`model.expert_tensor_parallel_size=1`. The default `"tp"` preserves the
existing tensor-parallel table layout. Only the lookup table is frozen;
the PLE projections, normalization, and convolution remain trainable.
The shared decoder implements this option for both text-only and VL models.

The owner-routed lookup is adapted from BestJuly's
[Megatron-LM PR #7553](https://github.com/NVIDIA/Megatron-LM/pull/7553), specifically
[`distributed_embedding.py` at `1ce7e86a`](https://github.com/BestJuly/Megatron-LM/blob/1ce7e86a073c18bcb6ea2ca5542c91e270d06a12/megatron/core/models/engram/distributed_embedding.py).
Bridge retains its single padded table and HF checkpoint keys rather than
adopting upstream's separate table per hash head. Each EP rank owns an equal
contiguous row range; the padded row count must be divisible by EP.
Lookup requests and embeddings travel by all-to-all. Hashing precedes SP
slicing so n-grams retain context across tensor-parallel boundaries.

HF import reads the checkpoint shards overlapping each destination's row
range. HF export streams one shard at a time over the table's TP or EP group.
The shared conversion layer applies an explicit floating-point export dtype
as each tensor is emitted, preserving streaming and integer buffer dtypes. Start a fresh optimizer when
freezing a previously trainable table; this option does not migrate optimizer
state from trainable PLE checkpoints. Trainable EP tables, ETP greater than
one, CP, and CUDA graph capture are outside this initial implementation.

This is a memory-layout optimization, not a throughput guarantee. The
approximately 95 GiB BF16 n-gram table occupies about 3 GiB per rank at EP32,
making lower attention TP practical to evaluate. The 384–430 TFLOP/s/GPU
numbers in #7553 use a reduced 8-layer, 32K-token GB300 proxy with additional
kernel and graph optimizations; they are not verified results for this port.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-qwen3.8-flash-next"></a>
### Run a configuration

Choose a workflow, precision, and exact recorded combination. The command and expected result update below.

<div class="verification-model-explorer" data-model-explorer>
  <div class="verification-model-controls" hidden>
    <div class="verification-capability-tabs" role="tablist" aria-label="Workflow">
      <button type="button" role="tab" aria-selected="true" data-capability-tab="import-export">Import & Export</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="pretrain">Pretrain</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="benchmark" disabled>Benchmark</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="sft">SFT</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="lora">LoRA</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="long-context">Long Context</button>
    </div>
    <div class="verification-filter-row">
      <div class="verification-precision-controls" aria-label="Precision filter">
        <span>Precision</span>
        <button type="button" class="is-active" data-precision="">All</button>
        <button type="button" data-precision="bf16">BF16</button>
        <button type="button" data-precision="fp8_mx">FP8 MX</button>
        <button type="button" data-precision="nvfp4">NVFP4</button>
      </div>
      <div class="verification-hardware-controls" aria-label="GPU filter">
        <span>GPU</span>
        <button type="button" class="is-active" data-hardware="">All</button>
        <button type="button" data-hardware="GB200">GB200</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="qwen3-8-flash-next-hf-to-megatron-cpu" aria-controls="qwen3-8-flash-next-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-8-flash-next-hf-to-megatron-gpu" aria-controls="qwen3-8-flash-next-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="qwen3-8-flash-next-megatron-to-hf-cpu" aria-controls="qwen3-8-flash-next-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-8-flash-next-megatron-to-hf-gpu" aria-controls="qwen3-8-flash-next-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="qwen3-8-flash-next-pretrain-gb200" aria-controls="qwen3-8-flash-next-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="qwen3-8-flash-next-sft-gb200" aria-controls="qwen3-8-flash-next-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="qwen3-8-flash-next-sft-long-context-gb200" aria-controls="qwen3-8-flash-next-sft-long-context-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="qwen3-8-flash-next-peft-gb200" aria-controls="qwen3-8-flash-next-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="qwen3-8-flash-next-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="qwen3-8-flash-next-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU import and persisted-checkpoint reload have not been verified.</p>
      </section>
    </article>
    <article id="qwen3-8-flash-next-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="qwen3-8-flash-next-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-22</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 2 --gpus-per-node 4 --tp 4 --pp 1 --ep 8 --etp 1 --hf-model Qwen/Qwen3.8-Flash-Next --hf-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --torch-dtype bfloat16 --megatron-path work/checkpoints/qwen3.8-flash-next-vl</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Full language/vision import persisted a native checkpoint that reloaded in a fresh export process. The supported full-VL roundtrip retained exact tensor keys, shapes, dtypes and bytes. MTP is unsupported and excluded.</p>
      </section>
    </article>
    <article id="qwen3-8-flash-next-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="qwen3-8-flash-next-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU export, reload, and weight equality have not been verified.</p>
      </section>
    </article>
    <article id="qwen3-8-flash-next-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="qwen3-8-flash-next-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-22</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 2 --gpus-per-node 4 --tp 4 --pp 1 --ep 8 --etp 1 --hf-model Qwen/Qwen3.8-Flash-Next --hf-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --torch-dtype bfloat16 --export-weight-dtype bfloat16 --distributed-save --megatron-path work/checkpoints/qwen3.8-flash-next-vl --hf-path work/exports/exported-hf</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Fresh native reload and export completed. All 1,627 supported tensors match the pinned HF checkpoint exactly by key, shape, dtype and bytes: 333 vision tensors, 1,624 BF16 tensors and three I64 PLE buffers. The 31 unsupported MTP tensors are excluded. Reloaded config fields match except provenance metadata and unsupported MTP configuration; tokenizer and image-processor configuration and a processed image batch agree. An auxiliary asset checker failed on integer-key diagnostic formatting after successful export and weight comparison, causing the original wrapper to exit unsuccessfully. The repaired checker passed separately on the same preserved export in the released container. This does not verify inference or training-trajectory equivalence.</p>
      </section>
    </article>
    <article id="qwen3-8-flash-next-pretrain-gb200" class="verification-model-detail" data-entry-detail="qwen3-8-flash-next-pretrain-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Full-VLM pretraining with image inputs has not been verified. The fixed-4096-token frozen-PLE Muon run belongs to the separate text-only card.</p>
      </section>
    </article>
    <article id="qwen3-8-flash-next-sft-gb200" class="verification-model-detail" data-entry-detail="qwen3-8-flash-next-sft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-22</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.35158914</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.16796531</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>5,457.587 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>5.890 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>93.814 tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 16 --gpus-per-node 4 --recipe qwen38_flash_next_sft_64gpu_gb200_bf16_config --mode sft --max_steps 100 --pretrained_checkpoint work/cache/qwen3.8-flash-next dataset.path=work/data/nemotron-image-v3-ecd logger.save_config_filepath=work/model-verification/qwen3.8-flash-next/sft/resolved-config.yaml logger.tensorboard_dir=work/model-verification/qwen3.8-flash-next/sft/tensorboard</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>EARLY, UNOPTIMIZED recipe. Completed 100 native Energon-loader updates with the full pinned HF checkpoint and 8,192 complete image conversations prepared by scripts/data/prepare_nemotron_image_v3_subset.py from nvidia/Nemotron-Image-Training-v3 ECD revision 7656391d4d4cb11ec3722b34f10d499435de0460. All language, vision and projector parameters are trainable. Configuration: 64 GB200 GPUs, TP4/SP, EP64/ETP1, PP1/CP1, padded unpacked sequence 2048, MBS1/GBS16, BF16 with FP32 gradient reduction, distributed Adam, constant LR 1e-6, zero weight decay and clip 1. Natural routing, HybridEP and MoE fusions are enabled; CUDA graphs, recomputation, shared-expert overlap, evaluation and checkpoint saving are disabled. Final-ten metrics use updates 91-100. These are short functional observations on a bounded subset, not held-out quality, full convergence or numerical equivalence to another parallel layout. Throughput counts padded slots, not supervised tokens. TFLOPs estimate model matrix work rather than hardware utilization. At sequence 2048 the dense QSA shortcut bypasses the indexer. All 64 ranks completed 100 finite, unskipped updates with zero NaN iterations. Image inputs and encoder execution were observed; selected language/vision parameters and FP32 Adam samples changed. The run consumed 1,600 conversations and 1,156,890 supervised tokens, counted once across dense-DP representatives. Preclip gradient norms ranged from 1.694735 to 28.194944. Native timings include lightweight verification callbacks; no profiler was active.</p>
      </section>
    </article>
    <article id="qwen3-8-flash-next-sft-long-context-gb200" class="verification-model-detail" data-entry-detail="qwen3-8-flash-next-sft-long-context-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Long-context SFT with packing and context parallelism has not been verified.</p>
      </section>
    </article>
    <article id="qwen3-8-flash-next-peft-gb200" class="verification-model-detail" data-entry-detail="qwen3-8-flash-next-peft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Parameter-efficient finetuning has not been verified.</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
