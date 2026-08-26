# Nemotron 3 Nano
[Nemotron 3 Nano](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) is a large language model (LLM) trained from scratch by NVIDIA, and designed as a unified model for both reasoning and non-reasoning tasks. The model employs a hybrid Mixture-of-Experts (MoE) architecture, consisting of 23 Mamba-2 and MoE layers, along with 6 Attention layers. Each MoE layer includes 128 experts plus 1 shared expert, with 6 experts activated per token. The model has 3.5B active parameters and 30B parameters in total.

NeMo Megatron Bridge supports pretraining, full parameters finetuning, and LoRA finetuning this model. The finetuned model can be converted back to the 🤗 Hugging Face format for downstream evaluation.

```{important}
Run all commands from `/opt/Megatron-Bridge` (e.g. `docker run -w /opt/Megatron-Bridge ...`)
```

```{tip}
We use the following environment variables throughout this page
- `HF_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `MEGATRON_MODEL_PATH=/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (feel free to set your own path)
```

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-nemotron-3-nano-30b-a3b"></a>
### Run a configuration

Choose a workflow, precision, and exact recorded combination. The command and expected result update below.

<div class="verification-model-explorer" data-model-explorer>
  <div class="verification-model-controls" hidden>
    <div class="verification-capability-tabs" role="tablist" aria-label="Workflow">
      <button type="button" role="tab" aria-selected="true" data-capability-tab="import-export">Import & Export</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="pretrain">Pretrain</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="benchmark">Benchmark</button>
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
        <button type="button" data-hardware="H100">H100</button>
        <button type="button" data-hardware="GB300">GB300</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="nemotron-3-nano-30b-a3b-hf-to-megatron-cpu" aria-controls="nemotron-3-nano-30b-a3b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-nano-30b-a3b-hf-to-megatron-gpu" aria-controls="nemotron-3-nano-30b-a3b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="nemotron-3-nano-30b-a3b-megatron-to-hf-cpu" aria-controls="nemotron-3-nano-30b-a3b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="nemotron-3-nano-30b-a3b-megatron-to-hf-gpu" aria-controls="nemotron-3-nano-30b-a3b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="nemotron-3-nano-30b-a3b-pretrain-h100" aria-controls="nemotron-3-nano-30b-a3b-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="nemotron-3-nano-30b-a3b-sft-h100" aria-controls="nemotron-3-nano-30b-a3b-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="nemotron-3-nano-30b-a3b-sft-long-context-h100" aria-controls="nemotron-3-nano-30b-a3b-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="nemotron-3-nano-30b-a3b-peft-h100" aria-controls="nemotron-3-nano-30b-a3b-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="fp8_mx" data-hardware="GB300" data-status="verified" data-entry="nemotron-3-nano-30b-a3b-pretrain-performance-gb300" aria-controls="nemotron-3-nano-30b-a3b-pretrain-performance-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · GB300</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">FP8 MX</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="nemotron-3-nano-30b-a3b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-30b-a3b-hf-to-megatron-cpu" tabindex="-1">
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
        <p>This workflow remains unverified and requires a public run against the pinned Hugging Face revision with all applicable verification gates.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-30b-a3b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-30b-a3b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-25</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --hf-revision 2d59de1cbd51c0adf384eb906b766d1aee0e0517 --megatron-path work/model-verification/nemotron-3-nano-30b-a3b/import-gpu --torch-dtype bfloat16 --tp 1 --pp 1 --ep 8 --etp 1 --distributed-timeout-minutes 120 --trust-remote-code --low-memory-save</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The pinned one-node 8-H100 import exits successfully at TP1/PP1/EP8/ETP1, persists iter_0000000, and reloads it for inference. A separate public GPU round-trip at the same topology exhaustively compares all 6,243 exported BF16 weights against the immutable source: 6,243 of 6,243 match with no skipped FP8 tensors or mismatches.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-30b-a3b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-30b-a3b-megatron-to-hf-cpu" tabindex="-1">
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
        <p>This workflow remains unverified and requires a public run against the pinned Hugging Face revision with all applicable verification gates.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-30b-a3b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-30b-a3b-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
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
        <p>This workflow remains unverified and requires a public run against the pinned Hugging Face revision with all applicable verification gates.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-30b-a3b-pretrain-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-30b-a3b-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
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
        <p>This workflow remains unverified and requires a bounded public run with the model&#x27;s pinned revision and all applicable verification gates.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-30b-a3b-sft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-30b-a3b-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
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
        <p>This workflow remains unverified and requires a bounded public run with the model&#x27;s pinned revision and all applicable verification gates.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-30b-a3b-sft-long-context-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-30b-a3b-sft-long-context-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H100</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
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
        <p>This workflow remains unverified and requires a bounded public run with the model&#x27;s pinned revision and all applicable verification gates.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-30b-a3b-peft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-30b-a3b-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
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
        <p>This workflow remains unverified and requires a bounded public run with the model&#x27;s pinned revision and all applicable verification gates.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-30b-a3b-pretrain-performance-gb300" class="verification-model-detail" data-entry-detail="nemotron-3-nano-30b-a3b-pretrain-performance-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · GB300</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>FP8 MX</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-18</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.18025</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.004964447</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>12,455.140 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>937.770 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>42,094.107 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 2 --gpus-per-node 4 --recipe nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config --mode pretrain --max_steps 50 --seq_length 8192 logger.save_config_filepath=work/model-verification/nemotron-3-nano-30b-a3b/gb300-performance/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On exactly 8 GB300s, the canonical MXFP8 mock-data recipe completes exactly 50 optimizer steps at TP1/PP1/CP1/EP8/ETP1, GBS/MBS 512/4, and sequence length 8192. All 50 keyed rows have finite loss with zero skipped or NaN iterations. Loss moves from 12.18025 to 0.004964447; the final ten steps average 12455.140 ms, 937.770 TFLOP/s/GPU, and 42094.107 tokens/s/GPU. The resolved configuration persists.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
