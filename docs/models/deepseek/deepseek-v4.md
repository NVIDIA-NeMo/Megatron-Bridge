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
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="fp8_mx" data-hardware="GB200" data-status="verified" data-entry="deepseek-v4-flash-pretrain-gb200" aria-controls="deepseek-v4-flash-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">FP8 MX</span>
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
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="deepseek-v4-flash-peft-gb200" aria-controls="deepseek-v4-flash-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
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
        <p>CPU import requires approximately 570 GB of CPU RAM for the full BF16 weight materialisation (285B parameters x 2 bytes). Verification requires a high-memory node with enough additional headroom for conversion workspace and strict checkpoint reload.
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --ep 4 --hf-model deepseek-ai/DeepSeek-V4-Flash --hf-revision 60d8d70770c6776ff598c94bb586a859a38244f1 --megatron-path work/model-verification/dsv4-flash/import-gpu --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully and creates a reloadable iter_0000000 Megatron checkpoint.
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model deepseek-ai/DeepSeek-V4-Flash --hf-revision 60d8d70770c6776ff598c94bb586a859a38244f1 --megatron-path work/model-verification/dsv4-flash/import-cpu/iter_0000000 --hf-path work/model-verification/dsv4-flash/export-cpu --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU export requires a high-memory node with enough headroom for the full BF16 weight set and tensor-merging workspace, followed by strict Hugging Face reload.
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
        <div><dt>Last verified</dt><dd>2026-08-21</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 2 --gpus-per-node 4 --tp 1 --pp 1 --ep 8 --etp 1 --hf-model deepseek-ai/DeepSeek-V4-Flash --hf-revision 60d8d70770c6776ff598c94bb586a859a38244f1 --megatron-path work/model-verification/dsv4-flash/import-gpu/iter_0000000 --hf-path work/model-verification/dsv4-flash/export-gpu --torch-dtype bfloat16 --export-weight-dtype bfloat16 --trust-remote-code --not-strict</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Two GB200 nodes produced 46 complete safetensors shards containing all 35,020 expected non-scale source keys. BF16 export intentionally omitted 34,167 quantization-scale companions; 35,017 exported tensors are BF16 and the three I32 tid2eid routing tables exactly match their source values after the expected integer cast. Two independent Transformers processes each strictly reloaded all 1,500 model modules across four GPUs with no CPU or disk placement and produced the same bounded greedy token. --not-strict permits only the intentional scale omission; exact key, shard, dtype, routing-table value, and reload checks remain required correctness gates.
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
        <div><dt>Precision</dt><dd>FP8 MX</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-27</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>7.250204</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>3.280722</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>7,856.370 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>189.690 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>2,085.441 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 16 --gpus-per-node 4 --recipe deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_library_config --mode pretrain --dataset megatron-indexed --max_steps 100 --pretrained_checkpoint work/model-verification/dsv4-flash/import-gpu/iter_0000000 &#x27;dataset.blend=[[&quot;work/data/the-pile/my-gpt3_08_text_document&quot;],null]&#x27; dataset.path_to_cache=work/cache/the-pile dataset.num_workers=0 dataset.random_seed=1234 rng.seed=1234 scheduler.lr_warmup_iters=10 scheduler.lr_decay_iters=100 validation.eval_interval=0 validation.eval_iters=0 ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.exit_on_missing_checkpoint=false checkpoint.finetune=false checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false checkpoint.save_optim=true checkpoint.save_rng=true checkpoint.async_save=false --save_dir work/model-verification/dsv4-flash/pretrain-ref --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=120</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 64 GB200 (16 nodes x 4 GPUs), TP1/PP4/VPP4/EP16/CP1 uses dense DP16, expert DP1, and GBS/MBS 256/1. The command loads the imported model weights while starting optimizer and RNG state fresh. It uses HybridEP natural routing, grouped GEMM, TE fused cross entropy, selective recompute over mhc and mla_up_proj, and MXFP8 parameter gather and gradient-buffer reuse without activation offload. Expert capacity, paged stash, CUDA graphs, and forced load balancing remain disabled. The run completed 100 finite LM/MTP-loss steps with no skipped or NaN iterations and wrote complete grouped-MXFP8 checkpoints containing model, optimizer, scheduler, data-order, and RNG state at steps 50 and 100. The item-specific Bridge revision pins the compatible Megatron-LM dev revision used by this workload without changing the card&#x27;s default environment for other verification items.
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
        <div><dt>Last verified</dt><dd>2026-08-20</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.9435847</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2834835</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>8,575.850 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>42.180 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>477.620 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>184.960 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>187.040 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 8 --gpus-per-node 4 --recipe deepseek_v4_flash_sft_openmath_thinking_packed_gb200_config --mode sft --step-func dsv4_step --pretrained_checkpoint work/models/deepseek-v4-flash-megatron --save_dir work/model-verification/dsv4-flash/sft-ref --save_interval 100 --max_steps 100 --seq_length 1024 --pipeline_model_parallel_size 4 --context_parallel_size 1 --expert_model_parallel_size 8 &#x27;model.pipeline_model_parallel_layout=Et*11|t*11|t*11|t*10mL&#x27; scheduler.lr_warmup_iters=10 scheduler.lr_decay_iters=100 validation.eval_interval=0 validation.eval_iters=0 rng.seed=5678 ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.exit_on_missing_checkpoint=false checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false checkpoint.save_optim=false ddp.overlap_grad_reduce=false logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=180</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 32 GB200 (8 nodes x 4 GPUs), TP1/PP4/EP8/CP1 with dense DP8 and expert DP1, GBS/MBS 128/1. OpenMathInstruct-2 thinking data uses offline packing at seq_length=1024 with fixed token and cumulative-boundary shapes. The GB200 recipe uses selective recompute over moe, mhc, mla_up_proj, and layernorm, attention activation offload, HybridEP dispatch, and DSA kernel fusion without enabling the optional DSA indexer loss. The run completed 100 steps with finite LM and MTP losses, no skipped or NaN iterations, and a complete model checkpoint at step 100. A fresh process reloaded that checkpoint and completed finite step 101. moe_grouped_gemm=True is a recipe default. dist.distributed_timeout_minutes must be 180 or above to allow offline data packing (approximately 63 minutes) before training begins.
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 16 --gpus-per-node 4 --recipe deepseek_v4_flash_sft_openmath_thinking_packed_config --step-func dsv4_step --pretrained_checkpoint work/models/deepseek-v4-flash-megatron --save_dir work/model-verification/dsv4-flash/sft-long-context-ref --save_interval 100 --max_steps 100 model.pipeline_model_parallel_size=4 &#x27;model.pipeline_model_parallel_layout=Et*11|t*11|t*11|t*10mL&#x27; model.context_parallel_size=2 model.cp_partition_mode=contiguous model.expert_model_parallel_size=8 model.moe_grouped_gemm=false model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 dataset.seq_length=1024 scheduler.lr_warmup_iters=10 scheduler.lr_decay_iters=100 validation.eval_interval=0 validation.eval_iters=0 rng.seed=5678 ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.exit_on_missing_checkpoint=false checkpoint.load=null ddp.overlap_grad_reduce=false logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=180</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CP=2 with contiguous partitioning and offline-packed OpenMathInstruct-2 at seq_length=1024 demonstrates sequence packing and context parallelism working together over 100 training steps with finite loss and no skipped or NaN iterations.
</p>
      </section>
    </article>
    <article id="deepseek-v4-flash-peft-gb200" class="verification-model-detail" data-entry-detail="deepseek-v4-flash-peft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-25</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.9435847</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2914867</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>6,944.330 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>52.000 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>589.834 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>53.712 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>53.953 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 8 --gpus-per-node 4 --recipe deepseek_v4_flash_peft_openmath_thinking_packed_gb200_config --mode lora --step-func dsv4_step --pretrained_checkpoint work/models/deepseek-v4-flash-megatron --save_dir work/model-verification/dsv4-flash/peft-ref --save_interval 100 --max_steps 100 --seq_length 1024 --pipeline_model_parallel_size 4 --context_parallel_size 1 --expert_model_parallel_size 8 &#x27;model.pipeline_model_parallel_layout=Et*11|t*11|t*11|t*10mL&#x27; scheduler.lr_warmup_iters=10 scheduler.lr_decay_iters=100 validation.eval_interval=0 validation.eval_iters=0 rng.seed=5678 ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.exit_on_missing_checkpoint=false checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false checkpoint.save_optim=false ddp.overlap_grad_reduce=false logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=180</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 32 GB200 (8 nodes x 4 GPUs), TP1/PP4/EP8/CP1 with dense DP8 and expert DP1, GBS/MBS 128/1. The recipe applies rank-32, alpha-32 LoRA with zero dropout to linear_q_down_proj, linear_q_up_proj, linear_kv_proj, linear_proj, linear_fc1, and linear_fc2. Routed experts use separate per-expert adapters; shared experts are also adapted. The run preserves the packed OpenMath thinking data, objective, natural routing, seed, and warmup/decay horizon used by the verified SFT item while freezing the base model and using PEFT&#x27;s own 1e-4 peak learning rate. It disables activation recompute and offload because the reduced PEFT training-state footprint fits with 53.712 GiB peak allocated memory. Transformer Engine CUDA graphs cover moe_router and moe_preprocess, while HybridEP uses 32 flex-dispatcher SMs, 8 ranks per NVLink domain, 128-token combine chunks, a 72-GPU domain, and MNNVL. All 100 steps completed with finite LM/MTP losses and zero skipped or NaN iterations. LM/MTP losses were 0.9435847/0.2250465 at step 1 and 0.2914867/0.02638751 at step 100. The adapter-only checkpoint reloaded after the same base checkpoint in a fresh process and completed finite step 101 with LM loss 0.2827368 and MTP loss 0.02594211.
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
        <p>On 128 GB200, PP1/EP64/CP1/TP1/DP2 with GBS/MBS 2048/1, FP8-MX, HybridEP dispatch, and full-iteration CUDA graphs completes 50 steps with finite losses, no skipped or NaN iterations, and all five metrics recorded. This benchmark uses forced routing, a static expert-rank capacity, paged stash, and a different batch/objective contract, so its metrics must not be compared with natural-routing library pretraining as convergence evidence.
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
