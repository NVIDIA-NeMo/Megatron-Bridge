# Muse Glimmer 30B

[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) is a dense
vision-language model with a roughly 2B-parameter vision encoder and a 28B-parameter
decoder. Megatron Bridge models the complete checkpoint: the vision tower, adapter,
projection, and text decoder all participate in checkpoint conversion.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-muse-glimmer-30b"></a>
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
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="muse-glimmer-30b-hf-to-megatron-cpu" aria-controls="muse-glimmer-30b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="muse-glimmer-30b-hf-to-megatron-gpu" aria-controls="muse-glimmer-30b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="muse-glimmer-30b-megatron-to-hf-cpu" aria-controls="muse-glimmer-30b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="muse-glimmer-30b-megatron-to-hf-gpu" aria-controls="muse-glimmer-30b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="muse-glimmer-30b-pretrain-h100" aria-controls="muse-glimmer-30b-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="muse-glimmer-30b-sft-h100" aria-controls="muse-glimmer-30b-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="muse-glimmer-30b-sft-long-context-h100" aria-controls="muse-glimmer-30b-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="muse-glimmer-30b-peft-h100" aria-controls="muse-glimmer-30b-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="muse-glimmer-30b-pretrain-performance-h100" aria-controls="muse-glimmer-30b-pretrain-performance-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="muse-glimmer-30b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="muse-glimmer-30b-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model meta-models/Muse-Glimmer-30B --hf-revision f84ecc3a0ea984a4c04542a84269e3d065350a6e --megatron-path work/model-verification/muse-glimmer-30b/cpu-megatron --torch-dtype bfloat16 --overwrite</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On a CPU-only node with no CUDA runtime, the command builds a native 52-layer HybridModel, completes all 1,228 mappings for 29,776,626,688 parameters, and persists iter_0000000. The checkpoint reloads for CPU export, whose exact audit covers all 1,436 source tensors and 29,776,626,688 elements with zero missing, unexpected, shape, dtype, or value mismatches.
</p>
      </section>
    </article>
    <article id="muse-glimmer-30b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="muse-glimmer-30b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model meta-models/Muse-Glimmer-30B --hf-revision f84ecc3a0ea984a4c04542a84269e3d065350a6e --megatron-path work/model-verification/muse-glimmer-30b/gpu-megatron --torch-dtype bfloat16 --tp 8 --overwrite</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The eight-GPU command builds the complete native HybridModel, completes all 1,228 mappings, validates the TP8 distributed state, and persists iter_0000000. The checkpoint reloads at TP8 for strict GPU export; its complete round-trip audit is bitwise exact across all 1,436 tensors and 29,776,626,688 elements.
</p>
      </section>
    </article>
    <article id="muse-glimmer-30b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="muse-glimmer-30b-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model meta-models/Muse-Glimmer-30B --hf-revision f84ecc3a0ea984a4c04542a84269e3d065350a6e --megatron-path work/model-verification/muse-glimmer-30b/cpu-megatron/iter_0000000 --hf-path work/model-verification/muse-glimmer-30b/cpu-hf-export --torch-dtype bfloat16 --overwrite</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Strict export exits successfully. All 1,436 BF16 tensors and 29,776,626,688 elements match the pinned source exactly in keys, shapes, dtypes, and values (maximum absolute difference 0). Transformers 5.15.0 reloads the result as MuseGlimmerForConditionalGeneration with no missing, unexpected, mismatched, or error keys and the same parameter count.
</p>
      </section>
    </article>
    <article id="muse-glimmer-30b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="muse-glimmer-30b-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model meta-models/Muse-Glimmer-30B --hf-revision f84ecc3a0ea984a4c04542a84269e3d065350a6e --megatron-path work/model-verification/muse-glimmer-30b/gpu-megatron/iter_0000000 --hf-path work/model-verification/muse-glimmer-30b/gpu-hf-export --torch-dtype bfloat16 --export-weight-dtype bfloat16 --distributed-save --tp 8 --overwrite</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Strict distributed export exits successfully. Every language, vision, adapter, and projection tensor matches the pinned source bitwise: 1,436 tensors and 29,776,626,688 elements with zero missing, unexpected, shape, dtype, or value mismatches. Transformers 5.15.0 strictly reloads the result with the same class and parameter count.
</p>
      </section>
    </article>
    <article id="muse-glimmer-30b-pretrain-h100" class="verification-model-detail" data-entry-detail="muse-glimmer-30b-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-12</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.23977</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.4103</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>1,035.340 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>42.490 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>247.262 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 8 --recipe muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config --mode pretrain --max_steps 100 --save_dir work/model-verification/muse-glimmer-30b/pretrain-reference --save_interval 50 checkpoint.load=null checkpoint.finetune=false checkpoint.save_optim=true checkpoint.save_rng=true checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/muse-glimmer-30b/pretrain-reference-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 32 H100 GPUs at TP8/PP2/DP2, the command trains the complete randomly initialized HybridModel for exactly 100 optimizer steps on the revision-pinned public CORD-v2 multimodal dataset. All losses and gradient norms remain finite, with zero skipped and zero NaN iterations. Loss decreases from 12.23977 to 1.410300; steps 91-100 average 1,035.34 ms, 42.49 model TFLOP/s/GPU, and 247.262 token slots/s/GPU. The persisted post-setup config records the exact runtime. Complete step-50 and step-100 checkpoints each contain 32 nonempty distributed shards and 1,550 state entries, including model, optimizer, scheduler, and RNG state, plus metadata, train state, and the native HybridModel run config.
</p>
      </section>
    </article>
    <article id="muse-glimmer-30b-sft-h100" class="verification-model-detail" data-entry-detail="muse-glimmer-30b-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-12</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.9772241</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.02695329</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>1,826.640 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>94.030 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>560.592 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 8 --recipe muse_glimmer_30b_sft_32gpu_h100_bf16_config --mode sft --pretrained_checkpoint work/model-verification/muse-glimmer-30b/gpu-megatron/iter_0000000 --max_steps 100 --save_dir work/model-verification/muse-glimmer-30b/sft-checkpoints --save_interval 100 checkpoint.load=null checkpoint.save_optim=false checkpoint.save_rng=false checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/muse-glimmer-30b/sft-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 32 H100 GPUs at TP8/PP2/DP2, the command loads the complete imported HybridModel and performs exactly 100 full-model image-conditioned SFT optimizer steps on the revision-pinned public CORD-v2 dataset. All losses and gradient norms remain finite, with zero skipped and zero NaN iterations. Loss decreases from 0.9772241 to 0.02695329; steps 91-100 average 1,826.64 ms, 94.03 model TFLOP/s/GPU, and 560.592 token slots/s/GPU. The complete 32-shard step-100 full-model checkpoint, metadata, train state, and resolved config are saved and reloadable.
</p>
      </section>
    </article>
    <article id="muse-glimmer-30b-sft-long-context-h100" class="verification-model-detail" data-entry-detail="muse-glimmer-30b-sft-long-context-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-12</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.034497</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.03382061</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>1,419.910 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>16.070 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,442.345 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 8 --recipe muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config --mode sft --pretrained_checkpoint work/model-verification/muse-glimmer-30b/gpu-megatron/iter_0000000 --max_steps 100 --save_dir work/model-verification/muse-glimmer-30b/long-sft-checkpoints --save_interval 100 checkpoint.load=null checkpoint.save_optim=false checkpoint.save_rng=false checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/muse-glimmer-30b/long-sft-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 32 H100 GPUs, the command loads the complete imported HybridModel and completes exactly 100 full-model SFT optimizer steps at sequence length 8192 with in-batch packing, TP1/PP4/CP2, A2A context-parallel communication, and selective core-attention recompute. Every loss and gradient norm is finite, with zero skipped and zero NaN iterations. Loss decreases from 1.034497 to 0.03382061; steps 91-100 average 1,419.91 ms, 16.07 model TFLOP/s/GPU, and 1,442.345 token slots/s/GPU. The complete step-100 full-model checkpoint, metadata, train state, and resolved config are saved and reloadable.
</p>
      </section>
    </article>
    <article id="muse-glimmer-30b-peft-h100" class="verification-model-detail" data-entry-detail="muse-glimmer-30b-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-12</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.7263421</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.03298394</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>4,823.070 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>283.360 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,698.503 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 8 --recipe muse_glimmer_30b_peft_8gpu_h100_bf16_config --mode lora --pretrained_checkpoint work/model-verification/muse-glimmer-30b/gpu-megatron/iter_0000000 --max_steps 100 --save_dir work/model-verification/muse-glimmer-30b/peft-checkpoints --save_interval 100 checkpoint.load=null checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/muse-glimmer-30b/peft-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command loads the complete imported native HybridModel checkpoint, freezes the base model, and inserts LoRA dim 8 / alpha 16 adapters only into the native linear_qkv and linear_proj attention projections. All 100 optimizer-step rows are present exactly once with finite loss, zero skipped iterations, and zero NaN iterations. Loss decreases from 0.7263421 to 0.03298394; steps 91-100 average 4,823.07 ms, 283.36 model TFLOP/s/GPU, and 1,698.503 token slots/s/GPU. The persisted post-setup config records the exact runtime. The complete eight-shard iter_0000100 checkpoint contains all 208 expected adapter tensors, its run config, metadata, and train state. A direct reload restores model and optimizer at step 100, completes finite step 101 with zero skipped or NaN iterations, and saves a distinct complete eight-shard checkpoint.
</p>
      </section>
    </article>
    <article id="muse-glimmer-30b-pretrain-performance-h100" class="verification-model-detail" data-entry-detail="muse-glimmer-30b-pretrain-performance-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-12</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.26882</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.442728</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>9,373.060 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>428.240 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>2,621.983 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 8 --recipe muse_glimmer_30b_pretrain_32gpu_h100_fp8cs_config --mode pretrain --max_steps 50 checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/muse-glimmer-30b/performance-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The exact 32-H100 command completes all 50 fixed-shape mock-token dense decoder steps at TP4/PP4/CP2, a 9|15|15|13 split of all 52 language layers, sequence length 4096, MBS6/GBS192, BF16 model and gradient state, and FP8 current-scaling Transformer GEMMs. The vision and projection towers are frozen and excluded from the fixed-shape decoder workload. Loss remains finite from 12.26882 to 2.442728 with zero skipped or NaN iterations. Steps 41-50 average 9,373.06 ms, 428.24 model TFLOP/s/GPU, and 2,621.983 token slots/s/GPU. The persisted post-setup config records the exact runtime, and the process exits successfully without checkpoint output.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
