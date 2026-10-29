# Kimi K3

[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) is a large sparse MoE model from Moonshot AI. Megatron Bridge supports the **language backbone** of the published multimodal checkpoint through the `KimiK3Bridge`.

```{note}
Support for this model is in progress. Conversion (HF → Megatron) and Megatron greedy inference are verified; strict full-checkpoint export, exact round-trip parity, and every training workflow are not. See Known Limitations and the machine-readable [verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/kimi-k3/card.yaml) before relying on this path.
```

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-kimi-k3"></a>
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
        <button type="button" data-hardware="GB300">GB300</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="kimi-k3-hf-to-megatron-cpu" aria-controls="kimi-k3-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="kimi-k3-hf-to-megatron-gpu" aria-controls="kimi-k3-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="kimi-k3-megatron-to-hf-cpu" aria-controls="kimi-k3-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="kimi-k3-megatron-to-hf-gpu" aria-controls="kimi-k3-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB300" data-status="unverified" data-entry="kimi-k3-pretrain-gb300" aria-controls="kimi-k3-pretrain-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB300</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB300" data-status="unverified" data-entry="kimi-k3-sft-gb300" aria-controls="kimi-k3-sft-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB300</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB300" data-status="unverified" data-entry="kimi-k3-sft-long-context-gb300" aria-controls="kimi-k3-sft-long-context-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB300</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB300" data-status="unverified" data-entry="kimi-k3-peft-gb300" aria-controls="kimi-k3-peft-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB300</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="kimi-k3-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="kimi-k3-hf-to-megatron-cpu" tabindex="-1">
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
        <p>No CPU Hugging Face-to-Megatron conversion result is claimed. Future verification must import the complete pinned language backbone with the public conversion launcher, persist iter_0000000, reload it, and audit all mapped keys, shapes, dtypes, and values.
</p>
      </section>
    </article>
    <article id="kimi-k3-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="kimi-k3-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-28</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 12 --gpus-per-node 4 --hf-model moonshotai/Kimi-K3 --hf-revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 --megatron-path work/model-verification/kimi-k3/imported-megatron --torch-dtype bfloat16 --tp 2 --pp 3 --ep 8 --etp 2 --distributed-timeout-minutes 180 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 48x GB200, the pinned complete checkpoint imports at TP2/PP3/EP8/ETP2 and exits successfully in 27:01. Published MXFP4 routed-expert weights are dequantized to BF16, and iter_0000000 is persisted with 48 distributed-checkpoint shards, metadata, and the recorded 93-layer run configuration. Exact Hugging Face round-trip parity is tracked separately by megatron_to_hf_gpu.
</p>
      </section>
    </article>
    <article id="kimi-k3-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="kimi-k3-megatron-to-hf-cpu" tabindex="-1">
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
        <p>No CPU Megatron-to-Hugging-Face conversion result is claimed. Future verification depends on a complete persisted Megatron import and must write, strictly reload, and exactly audit the exported Hugging Face language-backbone weights against the pinned source revision.
</p>
      </section>
    </article>
    <article id="kimi-k3-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="kimi-k3-megatron-to-hf-gpu" tabindex="-1">
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
        <p>No GPU Megatron-to-Hugging-Face conversion result is claimed. All 69 source KDA A_log tensors were audited: each stores 96 active heads plus 32 zero-valued inactive entries, for 2,208 checked padding values. Import validation drops only this zero padding, and export unit tests restore it with zeros. A four-layer diagnostic proxy completed low-memory GPU import and distributed streaming export. Its seven safetensors shards and index, totaling 58,482,941,904 bytes, were byte-identical to an export from the previous default-save import; this validates the low-memory I/O behavior, not full-model parity. Against the published proxy source, all 16,402 keys and shapes matched and A_log padding was valid. Three o_norm tensors changed from FP32 to BF16, and 16,128 value mismatches were concentrated in MXFP4 expert packed weights and scales because import dequantizes them to BF16 before export requantizes them. Full verification therefore still requires the pinned complete model to export and strictly reload, exact comparison of lossless tensors, and a declared numerical criterion for lossy MXFP4 reconstruction.
</p>
      </section>
    </article>
    <article id="kimi-k3-pretrain-gb300" class="verification-model-detail" data-entry-detail="kimi-k3-pretrain-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB300</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
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
        <p>Training was intentionally excluded from this model-only verification pass. A future public Kimi K3 language-backbone recipe must complete a bounded 100-step pretrain run with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable final checkpoint.
</p>
      </section>
    </article>
    <article id="kimi-k3-sft-gb300" class="verification-model-detail" data-entry-detail="kimi-k3-sft-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB300</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
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
        <p>Training was intentionally excluded from this model-only verification pass. A future public Kimi K3 language-backbone recipe must complete a pinned-data 100-step full-SFT run with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable final checkpoint.
</p>
      </section>
    </article>
    <article id="kimi-k3-sft-long-context-gb300" class="verification-model-detail" data-entry-detail="kimi-k3-sft-long-context-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB300</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
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
        <p>Training was intentionally excluded from this model-only verification pass. A future public long-context SFT recipe must define supported sequence packing and context parallelism, complete 100 steps with finite loss and no skipped or NaN iterations, and record all five metrics.
</p>
      </section>
    </article>
    <article id="kimi-k3-peft-gb300" class="verification-model-detail" data-entry-detail="kimi-k3-peft-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB300</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
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
        <p>Training was intentionally excluded from this model-only verification pass. A future Kimi K3 PEFT recipe with an audited adapter target set must complete 100 steps with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable adapter checkpoint.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
