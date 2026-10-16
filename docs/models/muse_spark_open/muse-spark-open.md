# Muse-Spark-Open-816B-A42B

Muse-Spark-Open-816B-A42B is a sparse mixture-of-experts vision-and-audio-language model.
Megatron Bridge models the complete checkpoint: the vision tower, vision adapter and
projection, the audio tower, and the text decoder all participate in checkpoint
conversion. A text-only mode converts the language model and its separate LM head alone.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-muse-spark-open-816b-a42b"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="muse-spark-open-816b-a42b-hf-to-megatron-cpu" aria-controls="muse-spark-open-816b-a42b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="muse-spark-open-816b-a42b-hf-to-megatron-gpu" aria-controls="muse-spark-open-816b-a42b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="muse-spark-open-816b-a42b-megatron-to-hf-cpu" aria-controls="muse-spark-open-816b-a42b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="muse-spark-open-816b-a42b-megatron-to-hf-gpu" aria-controls="muse-spark-open-816b-a42b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="muse-spark-open-816b-a42b-pretrain-gb200" aria-controls="muse-spark-open-816b-a42b-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="muse-spark-open-816b-a42b-sft-gb200" aria-controls="muse-spark-open-816b-a42b-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="muse-spark-open-816b-a42b-sft-long-context-gb200" aria-controls="muse-spark-open-816b-a42b-sft-long-context-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="muse-spark-open-816b-a42b-peft-gb200" aria-controls="muse-spark-open-816b-a42b-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="muse-spark-open-816b-a42b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-hf-to-megatron-cpu" tabindex="-1">
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
        <p>CPU import of the complete 816B checkpoint has not been attempted. GPU import is recorded separately.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-10-16</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 4 --gpus-per-node 4 --hf-model meta-models/Muse-Spark-Open-816B-A42B --hf-revision 0fb02478b02565b2ce7fc57e6051c91172774ac9 --megatron-path work/model-verification/muse-spark-open-816b-a42b/gpu-megatron --torch-dtype bfloat16 --tp 4 --pp 1 --ep 16 --etp 1 --low-memory-save --distributed-timeout-minutes 110</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>With 16 GB200 GPUs the command builds the complete native HybridModel with both Hugging Face perception encoders, loads the checkpoint at TP4/PP1/EP16/ETP1, and persists a 1.5 TB iter_0000000 distributed checkpoint from all 16 ranks. The exported audit under megatron_to_hf_gpu establishes its tensor content, and the training items warm-start from this checkpoint.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-megatron-to-hf-cpu" tabindex="-1">
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
        <p>CPU export of the complete 816B checkpoint has not been attempted. GPU export is recorded separately.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-10-16</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 8 --gpus-per-node 4 --hf-model meta-models/Muse-Spark-Open-816B-A42B --hf-revision 0fb02478b02565b2ce7fc57e6051c91172774ac9 --megatron-path work/model-verification/muse-spark-open-816b-a42b/gpu-megatron/iter_0000000 --hf-path work/model-verification/muse-spark-open-816b-a42b/gpu-hf-export --torch-dtype bfloat16 --tp 2 --pp 1 --ep 32 --etp 1 --distributed-save --distributed-timeout-minutes 110</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>With 32 GB200 GPUs the TP4/EP16 import reloads at TP2/PP1/EP32/ETP1, exercising tensor- and expert-parallel resharding, and strict distributed export writes all 65 safetensors shards plus configuration and tokenizer files from all 32 ranks. An exhaustive comparison of the export against the source covers all 2,000 tensors and 822,036,529,664 elements with no missing, extra, or reshaped keys. All values match exactly except 1,204 of 8,192 entries of model.language_model.norm.weight (maximum absolute difference 0.015625), the known BF16 rounding of the zero-centered final-norm conversion. The 124 residual-gate tensors are promoted from BF16 to FP32 with unchanged values, and the FP32 router tensors keep FP32. Do not pass --export-weight-dtype bfloat16: it would round the FP32 router tensors.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-pretrain-gb200" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-pretrain-gb200" tabindex="-1">
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
        <p>Not yet verified on the pinned public revision. The 192-GPU recipe must warm-start from the imported checkpoint and complete the bounded 100-step text pretraining run with finite loss, no skipped or NaN iterations, all five metrics, and full-state saves at steps 50 and 100.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-sft-gb200" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-sft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB200</h4>
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
        <p>Not yet verified on the pinned public revision. The 192-GPU recipe must complete one 85-step epoch over the 100K Tulu 3 conversations packed to 4K with finite loss, no skipped or NaN iterations, and all five metrics.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-sft-long-context-gb200" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-sft-long-context-gb200" tabindex="-1">
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
        <p>Not yet verified on the pinned public revision. The 192-GPU recipe must complete the 20-step packed 32K Tulu 3 run with context parallelism with finite loss, no skipped or NaN iterations, and all five metrics.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-peft-gb200" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-peft-gb200" tabindex="-1">
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
        <p>Not yet verified on the pinned public revision. The 64-GPU LoRA recipe must complete one 85-step epoch over the 100K Tulu 3 conversations packed to 4K with finite loss, no skipped or NaN iterations, and all five metrics.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
