# Qwen3-235B-A22B

[Qwen](https://huggingface.co/Qwen) is a family of large language models developed by Alibaba Cloud, including dense models (Qwen2, Qwen2.5, Qwen3) and Mixture-of-Experts models (Qwen3 MoE, Qwen3-Next). The models feature innovations like QK layernorm, Gated-Delta Networks, and Zero-Centered RMSNorm for improved training stability and performance.

Qwen family models are supported via the Bridge system with auto-detected configuration and weight mapping.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-qwen3-235b-a22b"></a>
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
        <button type="button" data-hardware="GB200">GB200</button>
        <button type="button" data-hardware="GB300">GB300</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="qwen3-235b-a22b-hf-to-megatron-cpu" aria-controls="qwen3-235b-a22b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="qwen3-235b-a22b-hf-to-megatron-gpu" aria-controls="qwen3-235b-a22b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="qwen3-235b-a22b-megatron-to-hf-cpu" aria-controls="qwen3-235b-a22b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="qwen3-235b-a22b-megatron-to-hf-gpu" aria-controls="qwen3-235b-a22b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="qwen3-235b-a22b-pretrain-h100" aria-controls="qwen3-235b-a22b-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="fp8_mx" data-hardware="GB200" data-status="verified" data-entry="qwen3-235b-a22b-pretrain-gb200" aria-controls="qwen3-235b-a22b-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">FP8 MX</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="qwen3-235b-a22b-sft-h100" aria-controls="qwen3-235b-a22b-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="qwen3-235b-a22b-sft-long-context-h100" aria-controls="qwen3-235b-a22b-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="qwen3-235b-a22b-peft-h100" aria-controls="qwen3-235b-a22b-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="fp8_mx" data-hardware="GB300" data-status="verified" data-entry="qwen3-235b-a22b-pretrain-performance-gb300" aria-controls="qwen3-235b-a22b-pretrain-performance-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · GB300</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">FP8 MX</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="fp8_mx" data-hardware="GB200" data-status="verified" data-entry="qwen3-235b-a22b-pretrain-performance-gb200" aria-controls="qwen3-235b-a22b-pretrain-performance-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">FP8 MX</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="qwen3-235b-a22b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-hf-to-megatron-cpu" tabindex="-1">
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
    <article id="qwen3-235b-a22b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
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
    <article id="qwen3-235b-a22b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-megatron-to-hf-cpu" tabindex="-1">
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
    <article id="qwen3-235b-a22b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-megatron-to-hf-gpu" tabindex="-1">
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
    <article id="qwen3-235b-a22b-pretrain-h100" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-pretrain-h100" tabindex="-1">
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
    <article id="qwen3-235b-a22b-pretrain-gb200" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-pretrain-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>FP8 MX</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-28</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>11.17684</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>6.856652</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>72,553.930 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>262.340 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,806.546 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 64 --gpus-per-node 4 --recipe qwen3_235b_a22b_256gpu_gb200_fp8mx_pretrain_config --mode pretrain --dataset megatron-indexed --seq_length 4096 --max_steps 100 --lr 3e-4 --min_lr 3e-5 --warmup_iters 40 &#x27;dataset.blend=[[&quot;work/data/rp2/head_01&quot;],null]&#x27; dataset.path_to_cache=work/cache/qwen3-235b-a22b/rp2 tokenizer.tokenizer_type=SentencePieceTokenizer tokenizer.tokenizer_model=work/data/rp2/tokenizer.model scheduler.lr_decay_iters=100 model.moe_router_force_load_balancing=false ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.load=null validation.eval_iters=0 validation.eval_interval=0 dataset.random_seed=1234 dataset.num_workers=8 rng.seed=1234 dist.distributed_timeout_minutes=30 --save_dir work/model-verification/qwen3-235b-a22b/pretrain-mxfp8-gb200-reference-checkpoints --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/qwen3-235b-a22b/pretrain-mxfp8-gb200-reference-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On exactly 256 GB200s, this support-verification workload completes exactly 100 bounded RP2 optimizer steps with TP1/PP16/CP1/EP16/ETP1, DP1, SP off, GBS/MBS 8192/1, and sequence length 4096. MXFP8 compute, natural routing, DeepEP, and Transformer Engine CUDA graphs for moe_router and moe_preprocess remain active. Loss is finite from 11.176840 to 6.856652 with no skipped or NaN iterations, all five metrics are recorded, the post-setup configuration persists, and complete iter_0000050 and iter_0000100 checkpoints are saved. GBS8192 makes this support verification rather than cross-model convergence evidence for the GBS1024 cohort.
</p>
      </section>
    </article>
    <article id="qwen3-235b-a22b-sft-h100" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-sft-h100" tabindex="-1">
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
    <article id="qwen3-235b-a22b-sft-long-context-h100" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-sft-long-context-h100" tabindex="-1">
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
    <article id="qwen3-235b-a22b-peft-h100" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-peft-h100" tabindex="-1">
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
    <article id="qwen3-235b-a22b-pretrain-performance-gb300" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-pretrain-performance-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · GB300</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>FP8 MX</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-17</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.75473</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>8.126335</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>14,920.840 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>1,300.440 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>8,784.492 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 64 --gpus-per-node 4 --recipe qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config --mode pretrain --max_steps 50 --seq_length 4096 logger.save_config_filepath=work/model-verification/qwen3-235b-a22b/gb300-performance/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On exactly 256 GB300s, the canonical MXFP8 mock-data recipe completes exactly 50 optimizer steps at TP1/PP4/CP1/EP32/ETP1, GBS/MBS 8192/2, and sequence length 4096. All 50 keyed rows have finite loss with zero skipped or NaN iterations. Loss moves from 12.75473 to 8.126335; the final ten steps average 14920.840 ms, 1300.440 TFLOP/s/GPU, and 8784.492 tokens/s/GPU. The resolved configuration persists.
</p>
      </section>
    </article>
    <article id="qwen3-235b-a22b-pretrain-performance-gb200" class="verification-model-detail" data-entry-detail="qwen3-235b-a22b-pretrain-performance-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>FP8 MX</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-19</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.75152</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>8.126326</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>17,905.340 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>1,083.660 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>7,320.274 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 64 --gpus-per-node 4 --recipe qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config --mode pretrain --max_steps 50 --seq_length 4096 logger.save_config_filepath=work/model-verification/qwen3-235b-a22b/gb200-performance/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On exactly 256 GB200s, the canonical MXFP8 mock-data recipe completes exactly 50 optimizer steps at TP1/PP8/CP1/EP32/ETP1, VPP3, GBS/MBS 8192/1, and sequence length 4096. All 50 keyed rows have finite loss with zero skipped or NaN iterations. Loss moves from 12.75152 to 8.126326; the final ten steps average 17905.340 ms, 1083.660 TFLOP/s/GPU, and 7320.274 tokens/s/GPU. The resolved configuration persists.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
