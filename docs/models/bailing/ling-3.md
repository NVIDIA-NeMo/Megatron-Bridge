# Ling 3.0

[Ling 3.0](https://huggingface.co/collections/inclusionAI/ling-30) is a hybrid Mixture-of-Experts language model family from inclusionAI. Megatron Bridge supports the public Tiny, Tiny Base, and Flash checkpoints through the Bailing bridge with automatic Hugging Face configuration and bidirectional weight conversion.

Tiny and Tiny Base have 24 logical KDA/MLA blocks, 128 routed experts, and low-rank-Q MLA; Tiny Base additionally includes one MTP layer. Flash has 42 logical blocks, 512 routed experts, direct-Q MLA, and one MTP layer. Each logical block maps to two MCore HybridModel positions. All variants require Hugging Face remote code.

<a id="megatron-core-requirements"></a>

**Megatron-Core requirements.** Conversion and SFT require native HybridModel KDA and head-wise MLA output gating, which the default main pin does not yet provide. Selecting `bash scripts/switch_mcore.sh dev` alone is also insufficient: the latest PR validation was blocked by shared Bridge imports (`set_default_log_ranks` and `FullyShardedDataParallelV1`) unavailable in that dev pin. Unsupported native capabilities are rejected before Ling provider construction.

The Tiny Base verification card (`examples/model_verification_cards/ling-3.0-tiny-base/card.yaml`) records the exact historical Bridge/MCore combination used for successful conversion. These results do not validate the current Bridge with its dev pin or with the historical MCore revision alone. SFT remains unverified; Megatron KDA text generation is unsupported. Ling uses native MCore layers rather than Kimi K3 because the direct KDA forget-gate projection, RoPE-enabled MLA, and head-wise output gate have different layouts and semantics.

See the [Bailing examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/bailing) for conversion commands and `src/megatron/bridge/recipes/bailing/h100/ling_v3_tiny_base.py` for the eight-GPU BF16 full-parameter SFT recipe.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-ling-3.0-tiny-base"></a>
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
        <button type="button" data-hardware="H200">H200</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="ling-3-0-tiny-base-hf-to-megatron-cpu" aria-controls="ling-3-0-tiny-base-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="ling-3-0-tiny-base-hf-to-megatron-gpu" aria-controls="ling-3-0-tiny-base-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="ling-3-0-tiny-base-megatron-to-hf-cpu" aria-controls="ling-3-0-tiny-base-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="ling-3-0-tiny-base-megatron-to-hf-gpu" aria-controls="ling-3-0-tiny-base-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H200" data-status="unverified" data-entry="ling-3-0-tiny-base-pretrain-h200" aria-controls="ling-3-0-tiny-base-pretrain-h200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H200" data-status="unverified" data-entry="ling-3-0-tiny-base-sft-h200" aria-controls="ling-3-0-tiny-base-sft-h200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H200" data-status="unverified" data-entry="ling-3-0-tiny-base-sft-long-context-h200" aria-controls="ling-3-0-tiny-base-sft-long-context-h200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H200" data-status="unverified" data-entry="ling-3-0-tiny-base-peft-h200" aria-controls="ling-3-0-tiny-base-peft-h200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="ling-3-0-tiny-base-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model inclusionAI/Ling-3.0-tiny-base --hf-revision bab7297fa02713af237e378bf21107718b8e0e1a --megatron-path work/model-verification/ling-3.0-tiny-base/cpu-megatron --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU import exits successfully and creates a reloadable iter_0000000 distributed checkpoint from the pinned public revision.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model inclusionAI/Ling-3.0-tiny-base --hf-revision bab7297fa02713af237e378bf21107718b8e0e1a --megatron-path work/model-verification/ling-3.0-tiny-base/gpu-megatron --torch-dtype bfloat16 --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Distributed import exits successfully on 8 GPUs at TP1/PP1/EP8/ETP1 and creates a reloadable iter_0000000 checkpoint with MLA, KDA, MoE, and MTP weights intact.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model inclusionAI/Ling-3.0-tiny-base --hf-revision bab7297fa02713af237e378bf21107718b8e0e1a --megatron-path work/model-verification/ling-3.0-tiny-base/cpu-megatron/iter_0000000 --hf-path work/model-verification/ling-3.0-tiny-base/cpu-hf-export --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU export exits successfully and writes the complete indexed Hugging Face checkpoint. All 9,686 tensors match the pinned source exactly in keys, shapes, dtypes, and values with max_abs_diff=0.0. AutoConfig, AutoTokenizer, and BailingMoeV3ForCausalLM reload successfully under Transformers 4.56.2; the full model has 8,209,994,528 parameters.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model inclusionAI/Ling-3.0-tiny-base --hf-revision bab7297fa02713af237e378bf21107718b8e0e1a --megatron-path work/model-verification/ling-3.0-tiny-base/gpu-megatron/iter_0000000 --hf-path work/model-verification/ling-3.0-tiny-base/gpu-hf-export --torch-dtype bfloat16 --distributed-save --save-every-n-ranks 1 --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Distributed export exits successfully on 8 GPUs at TP1/PP1/EP8/ETP1. All 9,686 tensors match the pinned source exactly in keys, shapes, dtypes, and values with max_abs_diff=0.0, including the source FP32 KDA and router state. BailingMoeV3ForCausalLM reloads successfully under Transformers 4.56.2 with 8,209,994,528 parameters.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-pretrain-h200" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-pretrain-h200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H200</dd></div>
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
        <p>Verification requires a bounded public-data pretraining run with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable final checkpoint.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-sft-h200" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-sft-h200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H200</dd></div>
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
        <p>The 8-GPU validation loaded the pinned Tiny Base checkpoint, transformed 10,000 pinned public SQuAD rows into 898 sequence-length-2048 packs at 99.12 percent efficiency, built TP1/PP1/CP1/EP8/ETP1 with the MTP layer, and completed the first forward/backward pass. It then stopped before a complete optimizer step: Bridge training_log() passed num_moe_layers to megatron.core.transformer.moe.moe_utils.track_moe_metrics(), but the tested MCore dev commit 4d197408bfd8513289cc69a9112f37798e6fdd4c did not accept that keyword. This describes the historical dev validation, not the default main pin. Verified status requires an unpatched 100-step run with finite loss, all five metrics, and a reloadable final checkpoint.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-sft-long-context-h200" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-sft-long-context-h200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H200</dd></div>
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
        <p>Verification requires a public long-context Tiny Base SFT run with supported sequence packing and context parallelism, finite loss, no skipped or NaN iterations, and all five metrics.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-peft-h200" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-peft-h200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H200</dd></div>
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
        <p>Verification requires a public Tiny Base PEFT recipe with an audited adapter target set, finite loss, no skipped or NaN iterations, all five metrics, and a reloadable adapter checkpoint.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
