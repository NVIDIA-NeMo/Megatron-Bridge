# Nemotron 3.5 Lightning
[NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) is a large language model trained by NVIDIA for reasoning, agentic, and conversational workloads. It uses a hybrid Mixture-of-Experts architecture with interleaved Mamba-2, MoE, and Attention layers, together with Multi-Token Prediction (MTP). This BF16 reference checkpoint has 30B total parameters and 3B active parameters.

NeMo Megatron Bridge supports pretraining, full-parameter fine-tuning, and LoRA fine-tuning of this model. The fine-tuned model can be converted back to the 🤗 Hugging Face format for downstream evaluation.

```{important}
Run all commands from `/opt/Megatron-Bridge` (e.g. `docker run -w /opt/Megatron-Bridge ...`)
```

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-nemotron-3.5-lightning"></a>
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
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-5-lightning-hf-to-megatron-cpu" aria-controls="nemotron-3-5-lightning-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-5-lightning-hf-to-megatron-gpu" aria-controls="nemotron-3-5-lightning-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-5-lightning-megatron-to-hf-cpu" aria-controls="nemotron-3-5-lightning-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-5-lightning-megatron-to-hf-gpu" aria-controls="nemotron-3-5-lightning-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-5-lightning-pretrain-h100" aria-controls="nemotron-3-5-lightning-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-lightning-pretrain-gb200" aria-controls="nemotron-3-5-lightning-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-lightning-pretrain-fsdp-gb200-bf16" aria-controls="nemotron-3-5-lightning-pretrain-fsdp-gb200-bf16" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · FSDP · GB200 · bf16</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="fp8_mx" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-lightning-pretrain-fsdp-gb200-fp8-mx" aria-controls="nemotron-3-5-lightning-pretrain-fsdp-gb200-fp8-mx" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · FSDP · GB200 · fp8_mx</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">FP8 MX</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-5-lightning-sft-h100" aria-controls="nemotron-3-5-lightning-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-lightning-sft-gb200" aria-controls="nemotron-3-5-lightning-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-5-lightning-sft-long-context-h100" aria-controls="nemotron-3-5-lightning-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-lightning-sft-long-context-gb200" aria-controls="nemotron-3-5-lightning-sft-long-context-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-5-lightning-peft-h100" aria-controls="nemotron-3-5-lightning-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-lightning-peft-gb200" aria-controls="nemotron-3-5-lightning-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-5-lightning-pretrain-performance-h100" aria-controls="nemotron-3-5-lightning-pretrain-performance-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-lightning-pretrain-performance-gb200" aria-controls="nemotron-3-5-lightning-pretrain-performance-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="nemotron-3-5-lightning-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-23</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --hf-revision b3caaabed0263651a17dc1f2d4ce97e794f76c44 --megatron-path work/model-verification/nemotron-3.5-lightning/cpu-megatron --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The pinned checkpoint imports on CPU, creates a reloadable iter_0000000, and preserves every MTP tensor for the paired CPU export. After export, all 6,513 tensors and 32,913,266,240 parameters match the pinned source exactly in name, shape, dtype, and value.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-23</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --tp 1 --pp 1 --ep 8 --etp 1 --hf-model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --hf-revision b3caaabed0263651a17dc1f2d4ce97e794f76c44 --megatron-path work/model-verification/nemotron-3.5-lightning/imported-megatron --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully and creates a reloadable iter_0000000. After the paired GPU export, all 6,513 tensors and 32,913,266,240 parameters match the pinned source exactly in name, shape, dtype, and value, including the MTP layers.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-23</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --hf-revision b3caaabed0263651a17dc1f2d4ce97e794f76c44 --megatron-path work/model-verification/nemotron-3.5-lightning/cpu-megatron/iter_0000000 --hf-path work/model-verification/nemotron-3.5-lightning/cpu-hf-export --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Strict CPU export writes a native NemotronHForCausalLM checkpoint that reloads with Transformers. All 6,513 tensors and 32,913,266,240 parameters match the pinned source bitwise with maximum difference zero.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-23</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --tp 1 --pp 1 --ep 8 --etp 1 --hf-model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --hf-revision b3caaabed0263651a17dc1f2d4ce97e794f76c44 --megatron-path work/model-verification/nemotron-3.5-lightning/imported-megatron/iter_0000000 --hf-path work/model-verification/nemotron-3.5-lightning/hf-export --torch-dtype bfloat16 --distributed-save</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Strict export exits successfully; all 6,513 tensors match the pinned source bitwise with original dtypes preserved, maximum difference zero, and Transformers reloads the output natively as NemotronHForCausalLM.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-pretrain-h100" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-28</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.18927</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>7.239111</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>36,869.270 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>200.180 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>7,110.095 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>68.547 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>73.893 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe nemotron_3_5_lightning_pretrain_config --mode pretrain --dataset megatron-indexed --seq_length 8192 --max_steps 20 --lr 3e-4 --min_lr 3e-5 --warmup_iters 8 &#x27;dataset.blend=[[&quot;work/data/rp2/perp_middle/middle_01_text_document&quot;],null]&#x27; dataset.path_to_cache=work/cache/nemotron-3.5-lightning-h100/rp2 rng.seed=1234 dataset.random_seed=1234 scheduler.lr_decay_iters=20 validation.eval_iters=0 validation.eval_interval=0 model.moe_router_force_load_balancing=false checkpoint.load=null --save_dir work/model-verification/nemotron-3.5-lightning-h100/pretrain-reference --save_interval 10 checkpoint.async_save=false logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=120</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On two complete eight-H100 nodes, the real-data RP2 CP2/p2p run completes exactly 20 sequence-length-8,192, GBS/MBS 512/1 optimizer steps with natural routing and finite LM/MTP-1/MTP-2 losses. LM loss is 12.18927 -&gt; 7.239111, steps 11-20 average 36,869.270 ms / 200.180 TFLOP/s/GPU, no iteration is skipped or NaN, and complete 20-file, 430-GiB checkpoints are written at steps 10 and 20.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-pretrain-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-pretrain-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-28</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.17197</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>5.798433</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>21,483.630 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>686.920 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>24,404.070 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>158.220 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>162.290 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --additional-slurm-params segment=2 --recipe nemotron_3_5_lightning_pretrain_8k_config --mode pretrain --dataset megatron-indexed --seq_length 8192 --max_steps 100 --lr 3e-4 --min_lr 3e-5 --warmup_iters 8 &#x27;dataset.blend=[[&quot;work/data/rp2/perp_middle/middle_01_text_document&quot;],null]&#x27; dataset.path_to_cache=work/cache/nemotron-3.5-lightning-gb200/rp2 rng.seed=1234 dataset.random_seed=1234 scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 model.moe_router_force_load_balancing=false checkpoint.load=null checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-gb200/pretrain-reference logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=120</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On two complete four-GPU GB200 nodes in one segment-2 NVLink domain, the sequence-length-8,192, GBS/MBS 512/2 real-data RP2 convergence-verification run completes exactly 100 finite-loss optimizer steps with natural routing, Transformer Engine fused cross entropy, CUDA graphs disabled, finite LM/MTP-1/MTP-2 losses, and no skipped or NaN iterations. LM loss is 12.171970 -&gt; 5.798433; real-data steps 91-100 average 21,483.630 ms / 686.920 TFLOP/s/GPU with 158.220 GiB peak allocated and 162.290 GiB peak reserved memory. Complete 15-file, 430-GiB checkpoints with distributed metadata are written at steps 50 and 100.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-pretrain-fsdp-gb200-bf16" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-pretrain-fsdp-gb200-bf16" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · FSDP · GB200 · bf16</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-28</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.17197</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>5.797082</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>26,665.830 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>553.430 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>19,661.417 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>160.760 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>166.230 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --additional-slurm-params segment=2 --recipe nemotron_3_5_lightning_pretrain_8k_fsdp_config --mode pretrain --dataset megatron-indexed --seq_length 8192 --max_steps 100 --lr 3e-4 --min_lr 3e-5 --warmup_iters 8 &#x27;dataset.blend=[[&quot;work/data/rp2/perp_middle/middle_01_text_document&quot;],null]&#x27; dataset.path_to_cache=work/cache/nemotron-3.5-lightning-gb200/rp2 rng.seed=1234 dataset.random_seed=1234 scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 model.moe_router_force_load_balancing=false checkpoint.load=null checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-gb200/pretrain-fsdp logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null dist.distributed_timeout_minutes=120</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On two complete four-GPU GB200 nodes in one segment-2 NVLink domain, the sequence-length-8,192, GBS/MBS 512/2 BF16 real-data RP2 FSDP convergence-verification run completes exactly 100 finite-loss optimizer steps with natural routing, Transformer Engine fused cross entropy, active MTP-1/MTP-2 losses, and no skipped or NaN iterations. LM loss is 12.171970 -&gt; 5.797082; real-data steps 91-100 average 26,665.830 ms / 553.430 TFLOP/s/GPU with 160.760 GiB peak allocated and 166.230 GiB peak reserved memory. Complete 14-file, 369-GiB FSDP checkpoints with distributed metadata are written at steps 50 and 100. Do not use this run to claim convergence neutrality against ordinary BF16 pretraining: 188 of 300 shared LM/MTP loss values exceed the 1% loss gate, with a maximum relative difference of 12.95%, despite the similar final loss.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-pretrain-fsdp-gb200-fp8-mx" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-pretrain-fsdp-gb200-fp8-mx" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · FSDP · GB200 · fp8_mx</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>FP8 MX</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-24</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.19034</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>3.913218</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>13,917.000 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>795.390 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>28,254.365 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>169.540 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>173.860 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 2 --gpus-per-node 4 --additional-slurm-params segment=2 --recipe nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp_config --mode pretrain --max_steps 20</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On two complete four-GPU GB200 nodes in one segment-2 NVLink domain, the sequence-length-8,192 MXFP8 FSDP recipe completes exactly 20 finite-loss optimizer steps with GBS/MBS 384/3, active MTP-1 and MTP-2 losses, and no skipped or NaN iterations. LM loss is 12.19034 -&gt; 3.913218, and steps 11-20 average 13,917.000 ms / 795.390 TFLOP/s/GPU with 169.54 GiB peak allocated and 173.86 GiB peak reserved memory. This run must not be compared with the BF16 real-data pretrain or BF16 FSDP runs: it uses MXFP8, mock data, forced routing, GBS/MBS 384/3, and no CUDA graphs.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-sft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-03</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.4406566</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2757806</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>9,887.800 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>79.240 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,313.983 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe nemotron_3_5_lightning_sft_openmathinstruct2_packed_config --mode sft --max_steps 100 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning-h100/imported-megatron/iter_0000000 --save_dir work/model-verification/nemotron-3.5-lightning-h100/sft-checkpoints</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Packed OpenMathInstruct-2 full SFT completes 100 finite-loss steps with active MTP gradients, LM loss 0.4406566 -&gt; 0.2757806, finite MTP-1 loss 0.7371141 -&gt; 0.4133309, and no skipped or NaN iterations. Steps 91-100 average 9,887.800 ms / 79.240 TFLOP/s/GPU, and the complete reloadable full-model checkpoint is written at step 100.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-sft-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-sft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB200</h4>
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
            <dd>0.4407797</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2762039</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>6,795.830 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>256.990 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>9,643.561 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>101.510 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>103.960 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --additional-slurm-params segment=2 --recipe nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config --mode sft --max_steps 100 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning-gb200/imported-megatron/iter_0000000 --save_dir work/model-verification/nemotron-3.5-lightning-gb200/sft-checkpoints</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On two complete four-GPU GB200 nodes in one segment-2 NVLink domain, packed OpenMathInstruct-2 full SFT completes exactly 100 finite-loss optimizer steps with active MTP gradients. LM loss is 0.4407797 -&gt; 0.2762039, MTP-1 loss is 0.7373838 -&gt; 0.4146959, and MTP-2 loss is 0.8092746 -&gt; 0.4744196. Steps 91-100 average 6,795.830 ms / 256.990 TFLOP/s/GPU with 101.510 GiB peak allocated and 103.960 GiB peak reserved memory. No iteration is skipped or NaN, and the complete 12-file reloadable full-model checkpoint is written at step 100.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-sft-long-context-h100" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-sft-long-context-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-03</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.4268321</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2650192</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>40,863.010 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>188.950 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>6,415.191 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe nemotron_3_5_lightning_sft_config --mode sft --dataset openmathinstruct2 --max_steps 100 --seq_length 32768 --lr 5e-6 --min_lr 0 --warmup_iters 10 --tensor_model_parallel_size 2 --context_parallel_size 2 --expert_tensor_parallel_size 1 --expert_model_parallel_size 8 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning-h100/imported-megatron/iter_0000000 &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;469216e3f46f4dacf476b382e192485ea51a143e&quot;}&#x27; dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:32768,pad_seq_to_mult:4}&#x27; model.sequence_parallel=true model.cp_comm_type=p2p model.cross_entropy_loss_fusion=true model.calculate_per_token_loss=true model.recompute_granularity=full model.recompute_modules=null model.recompute_method=uniform model.recompute_num_layers=1 model.moe_router_force_load_balancing=false ddp.average_in_collective=false ddp.overlap_param_gather=false optimizer.overlap_param_gather=false scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null checkpoint.save_optim=false checkpoint.save_rng=false checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-h100/long-sft-checkpoints --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Packed OpenMathInstruct-2 CP2 SFT completes exactly 100 finite-loss 32K steps with LM loss 0.4268321 -&gt; 0.2650192 and MTP-1 loss 0.7322379 -&gt; 0.4439273. All 100 steps have finite LM/MTP losses and no skipped or NaN iterations. Steps 91-100 average 40,863.010 ms / 188.950 TFLOP/s/GPU. Complete step-50 and step-100 checkpoints each contain 20 files, distributed metadata, saved run configuration and training state, with latest marker 100.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-sft-long-context-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-sft-long-context-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB200</h4>
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
            <dd>0.4268995</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2649925</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>129,692.530 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>119.030 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>4,042.546 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --recipe nemotron_3_5_lightning_sft_config --mode sft --dataset openmathinstruct2 --max_steps 100 --seq_length 32768 --lr 5e-6 --min_lr 0 --warmup_iters 10 --context_parallel_size 2 --expert_tensor_parallel_size 1 --expert_model_parallel_size 8 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning-gb200/imported-megatron/iter_0000000 model.moe_flex_dispatcher_backend=hybridep &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;469216e3f46f4dacf476b382e192485ea51a143e&quot;}&#x27; dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:32768,pad_seq_to_mult:4}&#x27; model.cp_comm_type=a2a model.cross_entropy_loss_fusion=false model.calculate_per_token_loss=true model.recompute_granularity=selective &#x27;model.recompute_modules=[moe,layernorm,core_attn,mlp]&#x27; model.recompute_method=null model.recompute_num_layers=null model.moe_router_force_load_balancing=false ddp.average_in_collective=false ddp.overlap_param_gather=false optimizer.overlap_param_gather=false scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null checkpoint.save_optim=false checkpoint.save_rng=false checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-gb200/long-sft-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Packed OpenMathInstruct-2 CP2 SFT completes exactly 100 finite-loss 32K steps with LM loss 0.4268995 -&gt; 0.2649925 and MTP-1 loss 0.7322624 -&gt; 0.4439029. All 100 steps have finite LM/MTP losses and no skipped or NaN iterations. Steps 91-100 average 129,692.530 ms / 119.030 TFLOP/s/GPU. The complete step-100 checkpoint has distributed metadata, saved run configuration and training state, 12 files, and a latest marker of 100.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-peft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-03</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.4406696</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2854756</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>16,900.360 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>92.890 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,877.787 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 8 --recipe nemotron_3_5_lightning_peft_config --mode lora --dataset openmathinstruct2 --max_steps 100 --seq_length 4096 --lr 1e-4 --min_lr 0 --warmup_iters 10 --tensor_model_parallel_size 2 --expert_tensor_parallel_size 1 --expert_model_parallel_size 8 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning-h100/imported-megatron/iter_0000000 &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;469216e3f46f4dacf476b382e192485ea51a143e&quot;}&#x27; dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:4096,pad_seq_to_mult:2}&#x27; model.sequence_parallel=true ddp.overlap_param_gather=false optimizer.overlap_param_gather=false scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-h100/peft-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Packed rank-32, alpha-32, zero-dropout LoRA completes 100 finite-loss steps with adapters on the MTP layers, LM loss 0.4406696 -&gt; 0.2854756, finite MTP-1 loss 0.7370828 -&gt; 0.4362708, and no skipped or NaN iterations. The complete adapter checkpoint is written at step 100 and reloads over the pinned base checkpoint at step 100.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-peft-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-peft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB200</h4>
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
            <dd>0.4406088</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2855029</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>19,230.400 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>81.780 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,407.937 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --recipe nemotron_3_5_lightning_peft_config --mode lora --dataset openmathinstruct2 --max_steps 100 --seq_length 4096 --lr 1e-4 --min_lr 0 --warmup_iters 10 --tensor_model_parallel_size 2 --expert_tensor_parallel_size 1 --expert_model_parallel_size 8 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning-gb200/imported-megatron/iter_0000000 model.moe_flex_dispatcher_backend=hybridep &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;469216e3f46f4dacf476b382e192485ea51a143e&quot;}&#x27; dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:4096,pad_seq_to_mult:2}&#x27; model.sequence_parallel=true ddp.overlap_param_gather=false optimizer.overlap_param_gather=false scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-gb200/peft-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Packed rank-32, alpha-32, zero-dropout LoRA completes 100 finite-loss steps with 48 rank-local adapter attachment records covering MTP attention, expert, and shared-expert layers, LM loss 0.4406088 -&gt; 0.2855029, finite MTP-1 loss 0.7371238 -&gt; 0.4362065, and no skipped or NaN iterations. The complete adapter checkpoint is written at step 100 and reloads over the pinned base checkpoint at step 100.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-pretrain-performance-h100" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-pretrain-performance-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-03</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.16197</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.01217863</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>18,467.280 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>399.580 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>14,195.052 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>72.066 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>75.340 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config --max_steps 50</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 16x H100, the exact mock-data, force-balanced BF16 performance recipe completes exactly 50 steps with finite LM/MTP-1/MTP-2 losses and no skipped or NaN iterations. Transformer Engine CUDA graphs cover attention and Mamba scopes, while expert-FC1 activations are offloaded. Steps 41-50 average 18,467.280 ms and 399.580 model TFLOP/s/GPU; peak rank-0 allocated/reserved memory is 72.066/75.340 GiB. This benchmark-only run stores Adam moments in BF16, so its losses are not convergence-comparable with FP32 optimizer-state runs.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-lightning-pretrain-performance-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-lightning-pretrain-performance-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-28</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.1825</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.03325584</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>18,236.060 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>809.220 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>28,750.070 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>156.850 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>161.260 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --recipe nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config --max_steps 50</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 8x GB200, the exact mock-data BF16 performance recipe completes exactly 50 steps with finite LM/MTP-1/MTP-2 losses and no skipped or NaN iterations. Transformer Engine CUDA graphs cover attention, Mamba, MoE-router, and MoE-preprocess scopes. Steps 41-50 average 18,236.060 ms and 809.220 model TFLOP/s/GPU; peak rank-0 allocated/reserved memory is 156.850/161.260 GiB.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
