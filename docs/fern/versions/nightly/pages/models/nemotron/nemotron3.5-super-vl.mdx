# Nemotron 3.5 Super VL

Nemotron 3.5 Super VL combines a Nemotron-H hybrid language decoder with a RADIO vision encoder, a vision projector, and a separate temporal video embedder. It reuses the Nemotron Omni image/video stack without an audio encoder and retains shared Multi-Token Prediction (MTP) weights.

Megatron Bridge provides checkpoint conversion and pretraining, SFT, and PEFT recipes. The verification records below distinguish verified configurations from pending checks; adding a recipe does not imply that every workflow is verified.

The `freeze_vision_model`, `freeze_vision_projection`, and `freeze_language_model` provider options independently control which model components are trained. The Super-VL SFT recipes freeze the vision encoder while keeping the projection and language model trainable.

The records below contain refreshed conversion, inference, and training verification with the corrected training callbacks and shared-MTP objective. Verification is scoped to each recorded workflow and hardware configuration; the GB200 pretrain/resume comparison and stronger visual diagnostics are not all passing. The public checkpoint revision remains unbound, and functional throughput measurements are not optimized-performance claims.

For language-only conversion and training without media components, see [the text-only companion guide](nemotron3.5-super-vl-text-only.md).

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-nemotron-3.5-super-vl-120b-a12b"></a>
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
        <button type="button" data-hardware="H100">H100</button>
        <button type="button" data-hardware="GB200">GB200</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="nemotron-3-5-super-vl-120b-a12b-hf-to-megatron-cpu" aria-controls="nemotron-3-5-super-vl-120b-a12b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-5-super-vl-120b-a12b-hf-to-megatron-gpu" aria-controls="nemotron-3-5-super-vl-120b-a12b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="nemotron-3-5-super-vl-120b-a12b-megatron-to-hf-cpu" aria-controls="nemotron-3-5-super-vl-120b-a12b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-5-super-vl-120b-a12b-megatron-to-hf-gpu" aria-controls="nemotron-3-5-super-vl-120b-a12b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-5-super-vl-120b-a12b-pretrain-h100" aria-controls="nemotron-3-5-super-vl-120b-a12b-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="nemotron-3-5-super-vl-120b-a12b-pretrain-gb200" aria-controls="nemotron-3-5-super-vl-120b-a12b-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-5-super-vl-120b-a12b-sft-h100" aria-controls="nemotron-3-5-super-vl-120b-a12b-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-super-vl-120b-a12b-sft-gb200" aria-controls="nemotron-3-5-super-vl-120b-a12b-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="nemotron-3-5-super-vl-120b-a12b-sft-long-context-h100" aria-controls="nemotron-3-5-super-vl-120b-a12b-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="nemotron-3-5-super-vl-120b-a12b-sft-long-context-gb200" aria-controls="nemotron-3-5-super-vl-120b-a12b-sft-long-context-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-5-super-vl-120b-a12b-peft-h100" aria-controls="nemotron-3-5-super-vl-120b-a12b-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-super-vl-120b-a12b-peft-gb200" aria-controls="nemotron-3-5-super-vl-120b-a12b-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="nemotron-3-5-super-vl-120b-a12b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-hf-to-megatron-cpu" tabindex="-1">
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
        <p>CPU-only import is not claimed in this card; the distributed GPU import below is the verified conversion path for this checkpoint.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-23</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --hf-model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --megatron-path work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron --torch-dtype bfloat16 --tp 1 --pp 2 --ep 2 --etp 1 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Distributed BF16 GPU import and paired export passed with all 43,079 source tensors exact in keys, shapes, dtypes and values. Compared 248,906,833,960 source bytes, including language, MTP, image encoder, projector, temporal embedder and the vision summary-index buffer. This establishes conversion, not training or numerical forward parity.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-megatron-to-hf-cpu" tabindex="-1">
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
        <p>CPU-only export is not claimed in this card; the distributed GPU export below is the verified conversion path for this checkpoint.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-23</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --hf-model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --megatron-path work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron --hf-path work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-hf-export --torch-dtype bfloat16 --tp 1 --pp 2 --ep 2 --etp 1 --trust-remote-code --distributed-save</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Saved round-trip comparison passed for all 43,079 tensors with no missing or extra keys and zero value differences. The exported package reloaded through native Transformers and performed bounded image generation. Exact conversion is separate from forward-logit correlation.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-pretrain-h100" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-25</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>3.368271</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.918807</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>60,028.320 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>114.330 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,364.689 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 8 --gpus-per-node 8 --recipe nemotron_35_super_vl_pretrain_64gpu_h100_bf16_config --mode pretrain --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --max_steps 200 --save_interval 100 --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-pretrain dataset.path=work/data/datacomp-525k-energon dataset.do_validation=false dataset.num_workers=2 model.tensor_model_parallel_size=2 model.sequence_parallel=true model.cross_entropy_fusion_impl=native validation.eval_iters=0 validation.eval_interval=0 scheduler.lr_decay_iters=200 checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false checkpoint.async_save=false checkpoint.exit_on_missing_checkpoint=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-pretrain-config/ConfigContainer.yaml ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed 200 finite optimizer steps on 64 H100s with zero skipped or NaN iterations, full checkpoints at steps 100 and 200, and a fresh final-checkpoint native inference reload. DataComp image-caption training warm-started from converted pretrained weights at sequence length 4096, GBS/MBS 1280/1, TP2/PP2/EP32/ETP1, natural HybridEP routing and FP32 optimizer state. Shared MTP uses two prediction depths and loss scale 0.3. Native finalization and update preflights passed for this workload. Rates are padded token slots per GPU, not actual supervised-token counts; production supervised totals were not recorded. This is bounded support verification, not full convergence.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-pretrain-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-pretrain-gb200" tabindex="-1">
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
        <p>Refreshed DataComp pretraining used 64 GB200 GPUs, pretrained weights, natural HybridEP routing, FP32 optimizer state, TP2/PP1/EP64/ETP1, sequence length 8192, GBS/MBS 512/1 and selective moe_act recompute. The original 500-step schedule reached step 475 before its four-hour limit. A fresh-root continuation from step 250 reached step 500 and saved a complete final checkpoint, whose native model reload passed. This is not an uninterrupted 500-step reference. One of 1125 matched loss observations exceeded the unchanged resume tolerance; independent replay remains pending, so no verified pretrain/resume pair is claimed.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-sft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-23</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>3.375419</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.958105</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>42,907.800 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>159.840 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,909.210 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 8 --gpus-per-node 8 --recipe nemotron_35_super_vl_sft_64gpu_h100_bf16_config --mode sft --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --max_steps 200 --save_interval 100 --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-sft dataset.path=work/data/datacomp-525k-energon dataset.do_validation=false dataset.num_workers=2 model.tensor_model_parallel_size=2 model.sequence_parallel=true model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 model.recompute_modules=[] validation.eval_iters=0 validation.eval_interval=0 scheduler.lr_decay_iters=200 checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false checkpoint.async_save=false checkpoint.exit_on_missing_checkpoint=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-sft-config/ConfigContainer.yaml ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed 200 finite optimizer steps on 64 H100 GPUs with zero skipped or NaN iterations and a complete final checkpoint. DataComp image-caption training uses sequence length 4096 and GBS/MBS 1280/1, natural HybridEP routing, FP32 optimizer state, and two shared MTP prediction depths at loss scale 0.3. The vision encoder is frozen; the language model and projection are trainable. Native finalization/update preflights passed. Final export and native HF reload are checked separately. Token rates are padded slots, not measured supervised-token totals. This verifies image-caption training, not video training or full convergence.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-sft-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-sft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-23</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>3.428863</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.934663</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>55,224.620 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>124.210 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,483.396 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 16 --gpus-per-node 4 --recipe nemotron_35_super_vl_sft_64gpu_gb200_bf16_config --mode sft --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --max_steps 200 --save_interval 100 --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-sft dataset.path=work/data/datacomp-525k-energon dataset.do_validation=false dataset.num_workers=2 validation.eval_iters=0 validation.eval_interval=0 scheduler.lr_decay_iters=200 checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false checkpoint.async_save=false checkpoint.exit_on_missing_checkpoint=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-sft-config/ConfigContainer.yaml ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed 200 finite optimizer steps on 64 GB200 GPUs with zero skipped or NaN iterations and a complete final checkpoint. DataComp image-caption training uses sequence length 4096 and GBS/MBS 1280/1, natural HybridEP routing, FP32 optimizer state, and two shared MTP prediction depths at loss scale 0.3. The vision encoder is frozen; the language model and projection are trainable. Native finalization/update preflights passed. Final export and native HF reload are checked separately. Token rates are padded slots, not measured supervised-token totals. This verifies image-caption training, not video training or full convergence.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-sft-long-context-h100" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-sft-long-context-h100" tabindex="-1">
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
        <p>No H100 long-context SFT result is claimed in this card.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-sft-long-context-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-sft-long-context-gb200" tabindex="-1">
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
        <p>No GB200 long-context SFT result is claimed in this card.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-peft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-24</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.483866</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.596869</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>5,634.590 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>60.860 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>726.938 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 2 --gpus-per-node 8 --recipe nemotron_35_super_vl_peft_16gpu_h100_bf16_config --mode lora --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --max_steps 500 --save_interval 250 --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-peft dataset.path=work/data/datacomp-525k-energon dataset.do_validation=false dataset.num_workers=2 validation.eval_iters=0 validation.eval_interval=0 scheduler.lr_decay_iters=500 checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false checkpoint.async_save=false checkpoint.exit_on_missing_checkpoint=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-peft-config/ConfigContainer.yaml ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed 500 finite LoRA optimizer steps on 16 H100 GPUs with zero skipped or NaN iterations and a complete final native adapter checkpoint. DataComp image-caption training uses sequence length 4096, GBS/MBS 16/1, rank 32, alpha 32, zero adapter dropout and language-only targets. Native save/reload and merge preserve updated router expert-bias buffers as well as adapter tensors. Full merged HF reload and bounded image generation passed; stronger post-merge numerical diagnostics are not all passing. Standalone HF adapter export does not preserve these trained router buffers; use native reload and full-model merge/export. Token rates are padded slots, not measured supervised-token totals.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-peft-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-peft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-24</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>3.45039</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.059837</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>3,640.690 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>94.230 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,125.061 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 4 --gpus-per-node 4 --recipe nemotron_35_super_vl_peft_16gpu_gb200_bf16_config --mode lora --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --max_steps 500 --save_interval 250 --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-peft dataset.path=work/data/datacomp-525k-energon dataset.do_validation=false dataset.num_workers=2 validation.eval_iters=0 validation.eval_interval=0 scheduler.lr_decay_iters=500 checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false checkpoint.async_save=false checkpoint.exit_on_missing_checkpoint=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-peft-config/ConfigContainer.yaml ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed 500 finite LoRA optimizer steps on 16 GB200 GPUs with zero skipped or NaN iterations and a complete final native adapter checkpoint. DataComp image-caption training uses sequence length 4096, GBS/MBS 16/1, rank 32, alpha 32, zero adapter dropout and language-only targets. Native save/reload and merge preserve updated router expert-bias buffers as well as adapter tensors. Full merged HF reload and bounded image generation passed; stronger post-merge numerical diagnostics are not all passing. Standalone HF adapter export does not preserve these trained router buffers; use native reload and full-model merge/export. Token rates are padded slots, not measured supervised-token totals.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
