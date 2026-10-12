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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.18931</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>7.238761</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>36,909.780 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>199.960 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>7,102.291 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>67.753 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>70.758 GiB</dd>
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
        <p>On two complete eight-H100 nodes, the real-data RP2 CP2/p2p run completes exactly 20 sequence-length-8,192, GBS/MBS 512/1 optimizer steps with natural routing and finite LM/MTP-1/MTP-2 losses. LM loss is 12.18931 -&gt; 7.238761, steps 11-20 average 36,909.780 ms / 199.960 TFLOP/s/GPU, no iteration is skipped or NaN, and complete 23-file, 430-GiB checkpoints are written at steps 10 and 20.
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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
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
            <dd>5.785674</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>21,296.790 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>692.930 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>24,618.170 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>162.430 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>166.630 GiB</dd>
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
        <p>On two complete four-GPU GB200 nodes in one segment-2 NVLink domain, the sequence-length-8,192, GBS/MBS 512/2 real-data RP2 convergence-verification run completes exactly 100 finite-loss optimizer steps with natural routing, Transformer Engine fused cross entropy, CUDA graphs disabled, finite LM/MTP-1/MTP-2 losses, and no skipped or NaN iterations. LM loss is 12.171970 -&gt; 5.785674; real-data steps 91-100 average 21,296.790 ms / 692.930 TFLOP/s/GPU with 162.430 GiB peak allocated and 166.630 GiB peak reserved memory. Complete 15-file, 430-GiB checkpoints with distributed metadata are written at steps 50 and 100.
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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
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
            <dd>5.812089</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>26,049.240 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>566.530 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>20,126.806 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>164.230 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>169.840 GiB</dd>
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
        <p>On two complete four-GPU GB200 nodes in one segment-2 NVLink domain, the sequence-length-8,192, GBS/MBS 512/2 BF16 real-data RP2 FSDP convergence-verification run completes exactly 100 finite-loss optimizer steps with natural routing, Transformer Engine fused cross entropy, active MTP-1/MTP-2 losses, and no skipped or NaN iterations. LM loss is 12.171970 -&gt; 5.812089; real-data steps 91-100 average 26,049.240 ms / 566.530 TFLOP/s/GPU with 164.230 GiB peak allocated and 169.840 GiB peak reserved memory. Complete 14-file, 369-GiB FSDP checkpoints with distributed metadata are written at steps 50 and 100. Do not use this run to claim convergence neutrality against ordinary BF16 pretraining: 155 of 300 shared LM/MTP loss values exceed the 1% loss gate, with a maximum relative difference of 9.80%, despite the similar final loss.
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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
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
            <dd>1.745152</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>14,203.880 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>779.280 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>27,683.703 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>175.150 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>180.030 GiB</dd>
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
        <p>On two complete four-GPU GB200 nodes in one segment-2 NVLink domain, the sequence-length-8,192 MXFP8 FSDP recipe completes exactly 20 finite-loss optimizer steps with GBS/MBS 384/3, active MTP-1 and MTP-2 losses, and no skipped or NaN iterations. LM loss is 12.19034 -&gt; 1.745152, and steps 11-20 average 14,203.880 ms / 779.280 TFLOP/s/GPU with 175.150 GiB peak allocated and 180.030 GiB peak reserved memory. This run must not be compared with the BF16 real-data pretrain or BF16 FSDP runs: it uses MXFP8, mock data, forced routing, GBS/MBS 384/3, and no CUDA graphs.
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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.4406465</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2761714</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>9,414.880 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>92.740 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,480.448 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe nemotron_3_5_lightning_sft_openmathinstruct2_packed_config --mode sft --max_steps 100 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning/imported-megatron/iter_0000000 --save_dir work/model-verification/nemotron-3.5-lightning-h100/sft-checkpoints</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Packed OpenMathInstruct-2 full SFT completes 100 finite-loss steps with active MTP gradients, LM loss 0.4406465 -&gt; 0.2761714, finite MTP-1 loss 0.7372081 -&gt; 0.4146557, and no skipped or NaN iterations. Steps 91-100 average 9,414.880 ms / 92.740 TFLOP/s/GPU, and the complete reloadable full-model checkpoint is written at step 100.
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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
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
            <dd>0.2761015</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>9,533.730 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>183.250 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>6,874.120 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>101.510 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>103.910 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --additional-slurm-params segment=2 --recipe nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config --mode sft --max_steps 100 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning/imported-megatron/iter_0000000 --save_dir work/model-verification/nemotron-3.5-lightning-gb200/sft-checkpoints</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On two complete four-GPU GB200 nodes in one segment-2 NVLink domain, packed OpenMathInstruct-2 full SFT completes exactly 100 finite-loss optimizer steps with active MTP gradients. LM loss is 0.4407797 -&gt; 0.2761015, MTP-1 loss is 0.7373838 -&gt; 0.4145924, and MTP-2 loss is 0.8092746 -&gt; 0.4743617. Steps 91-100 average 9,533.730 ms / 183.250 TFLOP/s/GPU with 101.510 GiB peak allocated and 103.910 GiB peak reserved memory. No iteration is skipped or NaN, and the complete 12-file reloadable full-model checkpoint is written at step 100.
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe nemotron_3_5_lightning_sft_config --mode sft --dataset openmathinstruct2 --max_steps 100 --seq_length 32768 --lr 5e-6 --min_lr 0 --warmup_iters 10 --tensor_model_parallel_size 2 --context_parallel_size 2 --expert_tensor_parallel_size 1 --expert_model_parallel_size 8 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning/imported-megatron/iter_0000000 &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;469216e3f46f4dacf476b382e192485ea51a143e&quot;}&#x27; dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:32768,pad_seq_to_mult:4}&#x27; model.sequence_parallel=true model.cp_comm_type=p2p model.cross_entropy_loss_fusion=true model.calculate_per_token_loss=true model.recompute_granularity=full model.recompute_modules=null model.recompute_method=uniform model.recompute_num_layers=1 model.moe_router_force_load_balancing=false ddp.average_in_collective=false ddp.overlap_param_gather=false optimizer.overlap_param_gather=false scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null checkpoint.save_optim=false checkpoint.save_rng=false checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-h100/long-sft-checkpoints --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
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
        <div><dt>Last verified</dt><dd>2026-10-12</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.4268849</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2653473</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>41,101.830 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>420.700 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>12,755.831 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --recipe nemotron_3_5_lightning_sft_config --mode sft --dataset openmathinstruct2 --max_steps 100 --seq_length 32768 --lr 5e-6 --min_lr 0 --warmup_iters 10 --context_parallel_size 2 --expert_tensor_parallel_size 1 --expert_model_parallel_size 8 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning/imported-megatron/iter_0000000 model.moe_flex_dispatcher_backend=hybridep &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;469216e3f46f4dacf476b382e192485ea51a143e&quot;}&#x27; dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:32768,pad_seq_to_mult:4}&#x27; model.cp_comm_type=a2a model.cross_entropy_loss_fusion=false model.calculate_per_token_loss=true model.recompute_granularity=selective &#x27;model.recompute_modules=[moe,layernorm,core_attn,mlp]&#x27; model.recompute_method=null model.recompute_num_layers=null model.moe_router_force_load_balancing=false ddp.average_in_collective=false ddp.overlap_param_gather=false optimizer.overlap_param_gather=false scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null checkpoint.save_optim=false checkpoint.save_rng=false checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-gb200/long-sft-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Packed OpenMathInstruct-2 CP2 SFT completes exactly 100 finite-loss 32K steps with LM loss 0.4268849 -&gt; 0.2653473 and MTP-1 loss 0.7321853 -&gt; 0.4469699. All 100 steps have finite LM/MTP losses and no skipped or NaN iterations. Steps 91-100 average 41,101.830 ms / 420.700 TFLOP/s/GPU. The complete step-100 checkpoint has distributed metadata, saved run configuration and training state, 12 files, and a latest marker of 100.
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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.4407612</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2858826</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>19,558.310 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>89.580 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,350.801 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 8 --recipe nemotron_3_5_lightning_peft_config --mode lora --dataset openmathinstruct2 --max_steps 100 --seq_length 4096 --lr 1e-4 --min_lr 0 --warmup_iters 10 --tensor_model_parallel_size 2 --expert_tensor_parallel_size 1 --expert_model_parallel_size 8 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning/imported-megatron/iter_0000000 &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;469216e3f46f4dacf476b382e192485ea51a143e&quot;}&#x27; dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:4096,pad_seq_to_mult:2}&#x27; model.sequence_parallel=true ddp.overlap_param_gather=false optimizer.overlap_param_gather=false scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-h100/peft-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Packed rank-32, alpha-32, zero-dropout LoRA completes 100 finite-loss steps with 48 rank-local adapter attachment records covering MTP attention, expert, and shared-expert layers, LM loss 0.4407612 -&gt; 0.2858826, finite MTP-1 loss 0.7371775 -&gt; 0.4370083, and no skipped or NaN iterations. Steps 91-100 average 19,558.310 ms / 89.580 TFLOP/s/GPU, and the complete 12-file adapter checkpoint is written at step 100 with a latest marker of 100.
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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.4406565</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.2857773</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>21,602.620 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>80.940 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,033.706 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --recipe nemotron_3_5_lightning_peft_config --mode lora --dataset openmathinstruct2 --max_steps 100 --seq_length 4096 --lr 1e-4 --min_lr 0 --warmup_iters 10 --tensor_model_parallel_size 2 --expert_tensor_parallel_size 1 --expert_model_parallel_size 8 --pretrained_checkpoint work/model-verification/nemotron-3.5-lightning/imported-megatron/iter_0000000 model.moe_flex_dispatcher_backend=hybridep &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;469216e3f46f4dacf476b382e192485ea51a143e&quot;}&#x27; dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:4096,pad_seq_to_mult:2}&#x27; model.sequence_parallel=true ddp.overlap_param_gather=false optimizer.overlap_param_gather=false scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null checkpoint.async_save=false --save_dir work/model-verification/nemotron-3.5-lightning-gb200/peft-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Packed rank-32, alpha-32, zero-dropout LoRA completes 100 finite-loss steps with 48 rank-local adapter attachment records covering MTP attention, expert, and shared-expert layers, LM loss 0.4406565 -&gt; 0.2857773, finite MTP-1 loss 0.7370399 -&gt; 0.4370756, and no skipped or NaN iterations. Steps 91-100 average 21,602.620 ms / 80.940 TFLOP/s/GPU, and the complete 12-file adapter checkpoint is written at step 100 with a latest marker of 100.
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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
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
            <dd>0.01383163</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>18,073.460 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>408.260 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>14,504.362 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>74.170 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>77.500 GiB</dd>
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
        <p>On 16x H100, the exact mock-data, force-balanced BF16 performance recipe completes exactly 50 steps with finite LM/MTP-1/MTP-2 losses and no skipped or NaN iterations. Transformer Engine CUDA graphs cover attention and Mamba scopes, while expert-FC1 activations are offloaded. Steps 41-50 average 18,073.460 ms and 408.260 model TFLOP/s/GPU; peak rank-0 allocated/reserved memory is 74.170/77.500 GiB. This benchmark-only run stores Adam moments in BF16, so its losses are not convergence-comparable with FP32 optimizer-state runs.
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
        <div><dt>Last verified</dt><dd>2026-10-09</dd></div>
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
            <dd>0.04329855</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>18,902.370 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>780.730 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>27,736.628 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>161.060 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>165.390 GiB</dd>
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
        <p>On 8x GB200, the exact mock-data BF16 performance recipe completes exactly 50 steps with finite LM/MTP-1/MTP-2 losses and no skipped or NaN iterations. Transformer Engine CUDA graphs cover attention, Mamba, MoE-router, and MoE-preprocess scopes. Steps 41-50 average 18,902.370 ms and 780.730 model TFLOP/s/GPU; peak rank-0 allocated/reserved memory is 161.060/165.390 GiB.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
