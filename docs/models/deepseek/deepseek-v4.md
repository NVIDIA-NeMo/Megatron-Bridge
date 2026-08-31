# DeepSeek V4

[DeepSeek-V4](https://github.com/deepseek-ai/DeepSeek-V4) is the next-generation Mixture-of-Experts language model from DeepSeek-AI. It extends the V3 design with **Hyper-Connections (mHC)** for multi-stream residual mixing, **Compressed Sparse Attention (CSA)** with a learned token-importance indexer (DSA), **hash-routed MoE layers** for the first few decoder blocks, and a refined **Multi-Token Prediction (MTP)** head with separate `e_proj` / `h_proj` projections.

DeepSeek V4 models are supported via the Bridge system with auto-detected configuration and weight mapping.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-deepseek-v4-flash"></a>
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
        <button type="button" data-hardware="GB200">GB200</button>
        <button type="button" data-hardware="GB300">GB300</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="deepseek-v4-flash-hf-to-megatron-cpu" aria-controls="deepseek-v4-flash-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="deepseek-v4-flash-hf-to-megatron-gpu" aria-controls="deepseek-v4-flash-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="deepseek-v4-flash-megatron-to-hf-cpu" aria-controls="deepseek-v4-flash-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="deepseek-v4-flash-megatron-to-hf-gpu" aria-controls="deepseek-v4-flash-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="deepseek-v4-flash-pretrain-gb200" aria-controls="deepseek-v4-flash-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="deepseek-v4-flash-sft-gb200" aria-controls="deepseek-v4-flash-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="deepseek-v4-flash-sft-long-context-gb200" aria-controls="deepseek-v4-flash-sft-long-context-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="unverified" data-entry="deepseek-v4-flash-peft-gb200" aria-controls="deepseek-v4-flash-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="fp8_mx" data-hardware="GB200" data-status="verified" data-entry="deepseek-v4-flash-pretrain-performance-gb200" aria-controls="deepseek-v4-flash-pretrain-performance-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">FP8 MX</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="fp8_mx" data-hardware="GB300" data-status="verified" data-entry="deepseek-v4-flash-pretrain-performance-gb300" aria-controls="deepseek-v4-flash-pretrain-performance-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · GB300</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">FP8 MX</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="deepseek-v4-flash-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-hf-to-megatron-cpu" tabindex="-1">
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
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model deepseek-ai/DeepSeek-V4-Flash --hf-revision 60d8d70770c6776ff598c94bb586a859a38244f1 --megatron-path work/model-verification/dsv4-flash/import-cpu --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU import requires approximately 570 GB merely for the full BF16 weight materialization, plus conversion overhead. Prior attempts exhausted the available node memory, but that resource limit does not establish a product-level lack of support. A high-memory-node run must complete and produce a reloadable checkpoint before this item is promoted.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-04</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --ep 4 --hf-model deepseek-ai/DeepSeek-V4-Flash --megatron-path work/model-verification/dsv4-flash/import-gpu --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully and creates iter_0000000. The checkpoint is reloadable and paired GPU export produces correct HF weights.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-megatron-to-hf-cpu" tabindex="-1">
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
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model deepseek-ai/DeepSeek-V4-Flash --hf-revision 60d8d70770c6776ff598c94bb586a859a38244f1 --megatron-path work/model-verification/dsv4-flash/import-gpu/iter_0000000 --hf-path work/model-verification/dsv4-flash/export-cpu --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU export requires approximately 570 GB merely for the full BF16 weight set, plus tensor-merging overhead. Prior attempts exhausted the available node memory, but that resource limit does not establish a product-level lack of support. A high-memory-node run must complete and produce a reloadable Hugging Face checkpoint before this item is promoted.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-05</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 2 --gpus-per-node 4 --ep 8 --pp 4 --hf-model deepseek-ai/DeepSeek-V4-Flash --megatron-path work/model-verification/dsv4-flash/import-gpu/iter_0000000 --hf-path work/model-verification/dsv4-flash/export-gpu --torch-dtype bfloat16 --export-weight-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully and writes 46 BF16 safetensors shards covering all transformer-layer parameters. The export completes 4176/4176 weight conversions with zero errors.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-pretrain-gb200" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-pretrain-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-04</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.64786</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>6.05023</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>10,415.760 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>142.910 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,573.001 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 16 --gpus-per-node 4 --recipe deepseek_v4_flash_pretrain_64gpu_gb200_bf16_config --dataset megatron-indexed --max_steps 100 &#x27;dataset.blend=[[&quot;work/data/rp2/head_01_text_document&quot;],null]&#x27; dataset.path_to_cache=work/cache/rp2 dataset.num_workers=0 dataset.random_seed=1234 rng.seed=1234 scheduler.lr_warmup_iters=40 scheduler.lr_decay_iters=100 validation.eval_interval=0 validation.eval_iters=0 ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.exit_on_missing_checkpoint=false checkpoint.load=null --save_dir work/model-verification/dsv4-flash/pretrain-ref --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=60</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 64 GB200 (16 nodes x 4 GPUs), the recipe runs with TP=1/PP=8/EP=8/CP=1/DP=1 and GBS/MBS 256/1. Data is RP2 tokenized with Llama SentencePiece (token IDs 0-31999 are within DSV4 vocab 0-129279; NullTokenizer is used). The run completes 100 steps from random initialisation with finite loss, no skipped or NaN iterations, and complete checkpoints at steps 50 and 100. Note: set dataset.num_workers=0 to avoid an NCCL pipeline timeout during data loading.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-sft-gb200" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-sft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-07</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.94383</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2837</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>13,806.500 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>13.620 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>148.336 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 16 --gpus-per-node 4 --recipe deepseek_v4_flash_sft_openmath_thinking_packed_config --step-func dsv4_step --pretrained_checkpoint work/models/deepseek-v4-flash-megatron --save_dir work/model-verification/dsv4-flash/sft-ref --save_interval 100 --max_steps 100 model.pipeline_model_parallel_size=4 &#x27;model.pipeline_model_parallel_layout=Et*11|t*11|t*11|t*10mL&#x27; model.context_parallel_size=2 model.expert_model_parallel_size=8 model.moe_grouped_gemm=false model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 dataset.seq_length=1024 scheduler.lr_warmup_iters=10 scheduler.lr_decay_iters=100 validation.eval_interval=0 validation.eval_iters=0 rng.seed=5678 ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.exit_on_missing_checkpoint=false checkpoint.load=null ddp.overlap_grad_reduce=false logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=180</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 64 GB200 (16 nodes x 4 GPUs), TP=1/PP=4/EP=8/CP=2/DP=1, GBS/MBS 128/1. OpenMathInstruct-2 thinking data with offline packing, seq_length=1024. Full recompute (granularity=full) is required to fit in memory at seq=1024 with CP=2. HybridEP dispatcher and DSA kernel fusion are enabled via the GB200-specific recipe. The run completes 100 steps with finite LM and MTP losses, no skipped or NaN iterations, and a complete step-100 checkpoint. moe_grouped_gemm=False is preserved for export compatibility. dist.distributed_timeout_minutes must be 180 or above to allow offline data packing (approximately 63 minutes) before training begins.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-sft-long-context-gb200" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-sft-long-context-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-07</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.94383</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2837</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>13,806.500 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>13.620 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>148.336 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 16 --gpus-per-node 4 --recipe deepseek_v4_flash_sft_openmath_thinking_packed_config --step-func dsv4_step --pretrained_checkpoint work/models/deepseek-v4-flash-megatron --save_dir work/model-verification/dsv4-flash/sft-ref --save_interval 100 --max_steps 100 model.pipeline_model_parallel_size=4 &#x27;model.pipeline_model_parallel_layout=Et*11|t*11|t*11|t*10mL&#x27; model.context_parallel_size=2 model.cp_partition_mode=contiguous model.expert_model_parallel_size=8 model.moe_grouped_gemm=false model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 dataset.seq_length=1024 scheduler.lr_warmup_iters=10 scheduler.lr_decay_iters=100 validation.eval_interval=0 validation.eval_iters=0 rng.seed=5678 ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.exit_on_missing_checkpoint=false checkpoint.load=null ddp.overlap_grad_reduce=false logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=180</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Same run as sft.GB200. CP=2 with contiguous partitioning and offline-packed OpenMathInstruct-2 at seq_length=1024 demonstrates sequence packing and context parallelism working together over 100 training steps with finite loss and no skipped or NaN iterations.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-peft-gb200" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-peft-gb200" tabindex="-1">
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
        <p>No exact DeepSeek V4 PEFT recipe is currently exported. Recipe absence does not establish an architectural lack of support; this item remains unverified until a public recipe and bounded run pass the PEFT gates.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-pretrain-performance-gb200" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-pretrain-performance-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>FP8 MX</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-06</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>13.5875</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>3.62683</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>8,134.600 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>731.400 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>8,056.450 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 32 --gpus-per-node 4 --recipe deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_config --max_steps 50 scheduler.lr_warmup_iters=5 scheduler.lr_decay_iters=50 validation.eval_interval=0 validation.eval_iters=0 checkpoint.exit_on_missing_checkpoint=false logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=120</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 128 GB200 (32 nodes x 4 GPUs, --segment=16 for NVLink locality), PP=1/EP=64/CP=1/TP=1/DP=2, GBS/MBS 2048/1, FP8-MX with HybridEP dispatcher and full-iteration CUDA graphs. The 50-step run completes with finite losses, no skipped or NaN iterations, and all five metrics recorded. First iteration takes approximately 24 minutes for CUDA graph compilation and kernel warmup; subsequent iterations run at approximately 8.1 seconds per step at steady state. The --segment=16 flag is required to ensure nodes are allocated within NVLink domains for MNNVL communication.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-pretrain-performance-gb300" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-pretrain-performance-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · GB300</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>FP8 MX</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-13</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.59616</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>3.160061</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>7,840.320 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>759.280 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>8,358.842 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 32 --gpus-per-node 4 --recipe deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config --mode pretrain --max_steps 50 --seq_length 4096 logger.save_config_filepath=work/model-verification/deepseek-v4-flash/gb300-performance/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On exactly 128 GB300s, the canonical MXFP8 mock-data recipe completes exactly 50 optimizer steps at TP1/PP1/CP1/EP64/ETP1, GBS/MBS 2048/1, and sequence length 4096. All 50 keyed rows have finite loss with zero skipped or NaN iterations. Loss moves from 12.59616 to 3.160061; the final ten steps average 7840.320 ms, 759.280 TFLOP/s/GPU, and 8358.842 tokens/s/GPU. The resolved configuration persists.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
