# Nemotron 3.5 Super VL

Nemotron 3.5 Super VL combines a Nemotron-H hybrid language decoder with a RADIO vision encoder, a vision projector, and a separate temporal video embedder. It reuses the Nemotron Omni image/video stack without an audio encoder and retains shared Multi-Token Prediction (MTP) weights.

Megatron Bridge provides checkpoint conversion and pretraining, SFT, and PEFT recipes. The verification records below distinguish verified configurations from pending checks; adding a recipe does not imply that every workflow is verified.

The `freeze_vision_model`, `freeze_vision_projection`, and `freeze_language_model` provider options independently control which model components are trained. The Super-VL SFT recipes freeze the vision encoder while keeping the projection and language model trainable.

> Verification scope: the recorded training results below are historical and apply only to the Bridge commits recorded in the [authoritative verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/nemotron-3.5-super-vl-120b-a12b/card.yaml). They used one MTP prediction depth and predate the current two-depth shared-MTP configuration, FP32 optimizer-state and weight-decay defaults, and language-only PEFT adapter targets. Verification of those current training settings remains pending; the historical “Verified” badges do not certify the current recipes. Recorded throughput is functional-verification evidence, not a canonical optimized performance result.

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
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="nemotron-3-5-super-vl-120b-a12b-pretrain-gb200" aria-controls="nemotron-3-5-super-vl-120b-a12b-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
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
        <div><dt>Last verified</dt><dd>2026-08-25</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B-SourceOfTruth --hf-revision e86197a3bad449de618a5835f26835ce770c6f10 --megatron-path work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron --torch-dtype bfloat16 --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code --low-memory-save</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The pinned eight-GPU BF16 import exits successfully after mapping 6,281 parameters per rank and creates a TP1/PP1/EP8/ETP1 iter_0000000 checkpoint containing the complete language, MTP, image, projector, and temporal-video weights. The paired export reloads this saved checkpoint successfully.
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
        <div><dt>Last verified</dt><dd>2026-08-25</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B-SourceOfTruth --hf-revision e86197a3bad449de618a5835f26835ce770c6f10 --megatron-path work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --hf-path work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-hf-export --torch-dtype bfloat16 --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code --distributed-save --save-every-n-ranks 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Strict distributed export exits successfully in 63 indexed shards. All 43,078 source tensors match in keys, shapes, dtypes, and values, with maximum difference zero. AutoConfig, AutoProcessor, and AutoModelForImageTextToText reload the local-only export natively as NemotronH_Omni_Reasoning_V3 across all eight visible H100s.
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
        <div><dt>Last verified</dt><dd>2026-08-29</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.999004</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.104368</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>49,743.200 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>130.340 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,646.858 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 8 --gpus-per-node 8 --recipe nemotron_35_super_vl_pretrain_64gpu_h100_bf16_config --mode pretrain --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/vision-pretrained/iter_0000000 --max_steps 120 dataset.path=work/data/datacomp-525k-energon checkpoint.load=null --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-pretrain --save_interval 60 logger.log_interval=1 logger.log_throughput=true optimizer.optimizer_cpu_offload=true optimizer.optimizer_offload_fraction=0.25 optimizer.overlap_cpu_optimizer_d2h_h2d=false model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 model.recompute_modules=null logger.log_device_memory_used=true logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-pretrain-config/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Exactly 64 H100s completed all 120 optimizer steps from the learned vision-pretrained checkpoint on DataComp image-caption samples at TP1/PP2/CP1/EP32/ETP1 and GBS/MBS 1280/1. Natural dropless routing and full/uniform one-layer language recompute, selective vision recompute, and 25% optimizer-state CPU offload plus the recipe&#x27;s BF16 precision-aware optimizer produced finite loss, zero skipped or NaN iterations, and the resolved ConfigContainer. Full optimizer/RNG checkpoints at steps 60 and 120 each contain all 64 distributed model shards and 32 Energon data-parallel states.
</p>
      </section>
    </article>
    <article id="nemotron-3-5-super-vl-120b-a12b-pretrain-gb200" class="verification-model-detail" data-entry-detail="nemotron-3-5-super-vl-120b-a12b-pretrain-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-28</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>3.062157</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.038754</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>20,385.850 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>257.330 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,214.779 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 16 --gpus-per-node 4 --recipe nemotron_35_super_vl_pretrain_64gpu_gb200_bf16_config --mode pretrain --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/vision-pretrained/iter_0000000 --max_steps 120 dataset.path=work/data/datacomp-525k-energon checkpoint.load=null --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-pretrain --save_interval 60 logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-pretrain-config/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On exactly 64 GB200s within one NVL72 domain, complete all 120 optimizer steps from the learned vision-pretrained checkpoint on DataComp image-caption samples at TP2/PP1/CP1/EP64/ETP1 and GBS/MBS 512/1. Natural dropless routing, finite loss, zero skipped or NaN iterations, no active language or vision recompute, scoped Transformer Engine CUDA-graph capture, and the resolved ConfigContainer are required. Full optimizer/RNG checkpoints at steps 60 and 120 each contain all 64 distributed model shards and 32 Energon data-parallel states.
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
        <div><dt>Last verified</dt><dd>2026-08-28</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>5.412096</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.293077</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>30,878.250 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>209.980 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>2,653.000 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 8 --gpus-per-node 8 --recipe nemotron_35_super_vl_sft_64gpu_h100_bf16_config --mode sft --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --max_steps 100 dataset.path=work/data/datacomp-525k-energon dataset.do_validation=false dataset.do_test=false validation.eval_iters=0 validation.eval_interval=0 scheduler.lr_decay_iters=100 checkpoint.load=null --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-sft --save_interval 100 checkpoint.save_optim=false checkpoint.save_rng=false logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-sft-config/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Exactly 64 H100s completed 100 BF16 SFT optimizer steps from the immutable source checkpoint on 128,000 DataComp image-caption samples at TP1/PP2/CP1/EP32/ETP1 and GBS/MBS 1280/1. The shifted production loss masks contain 2,900,779 supervised tokens. Natural routing, HybridEP flex dispatch, fixed expert capacity, selective language recompute with vision recompute disabled, and the precision-aware optimizer produced finite loss with zero skipped or NaN iterations. The model-only step-100 checkpoint contains all 64 distributed shards and 32 Energon states and reloaded successfully. This dataset verifies image-caption batching; it does not contain video samples.
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
        <div><dt>Last verified</dt><dd>2026-08-29</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>5.393576</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.303741</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>47,924.550 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>135.300 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,709.354 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 16 --gpus-per-node 4 --recipe nemotron_35_super_vl_sft_64gpu_gb200_bf16_config --mode sft --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --max_steps 100 dataset.path=work/data/datacomp-525k-energon dataset.do_validation=false dataset.do_test=false validation.eval_iters=0 validation.eval_interval=0 scheduler.lr_decay_iters=100 checkpoint.load=null --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-sft --save_interval 100 checkpoint.save_optim=false checkpoint.save_rng=false logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-sft-config/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Exactly 64 GB200s within one NVL72 domain completed 100 BF16 SFT optimizer steps from the immutable source checkpoint on 128,000 DataComp image-caption samples at TP2/PP1/CP1/EP64/ETP1 and GBS/MBS 1280/1. The shifted production loss masks contain 2,900,779 supervised tokens. Natural routing, eager all-to-all dispatch, fixed expert capacity, and no active language or vision recompute produced finite loss with zero skipped or NaN iterations. The model-only step-100 checkpoint contains all 64 distributed shards and 32 Energon states and reloaded successfully. This dataset verifies image-caption batching; it does not contain video samples.
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
        <div><dt>Last verified</dt><dd>2026-08-29</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>3.857522</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.157877</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>5,427.270 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>59.740 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>754.707 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 2 --gpus-per-node 8 --recipe nemotron_35_super_vl_peft_16gpu_h100_bf16_config --mode lora --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --max_steps 100 dataset.path=work/data/datacomp-525k-energon dataset.do_validation=false dataset.do_test=false validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-peft --save_interval 100 checkpoint.save_optim=false checkpoint.save_rng=false logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/h100-peft-config/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Exactly 16 H100s completed the rank-32, alpha-32, zero-dropout LoRA objective for 100 optimizer steps from the immutable source checkpoint on 1,600 DataComp image-caption samples at TP4/PP2/CP1/EP8/ETP1 and GBS/MBS 16/1. The shifted production loss masks contain 35,540 supervised tokens. Frozen base weights, natural routing, dropless all-to-all dispatch, and no active language or vision recompute produced finite loss with zero skipped or NaN iterations. The step-100 adapter contains all 16 distributed shards and two Energon states and reloaded successfully. This dataset does not contain video samples. This historical recipe used unqualified adapter targets that also matched vision and projector modules; it does not verify the current language-only adapter placement.
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
        <div><dt>Last verified</dt><dd>2026-08-28</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>5.485836</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.260179</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>3,472.770 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>93.550 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,179.462 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 4 --gpus-per-node 4 --recipe nemotron_35_super_vl_peft_16gpu_gb200_bf16_config --mode lora --pretrained_checkpoint work/model-verification/nemotron-3.5-super-vl-120b-a12b/gpu-megatron/iter_0000000 --max_steps 100 dataset.path=work/data/datacomp-525k-energon dataset.do_validation=false dataset.do_test=false validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-peft --save_interval 100 checkpoint.save_optim=false checkpoint.save_rng=false logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/nemotron-3.5-super-vl-120b-a12b/gb200-peft-config/ConfigContainer.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Exactly 16 GB200s completed the rank-32, alpha-32, zero-dropout LoRA objective for 100 optimizer steps from the immutable source checkpoint on 1,600 DataComp image-caption samples at TP2/PP1/CP1/EP16/ETP1 and GBS/MBS 16/1. The shifted production loss masks contain 34,537 supervised tokens. Frozen base weights, natural routing, dropless all-to-all dispatch, and no active language or vision recompute produced finite loss with zero skipped or NaN iterations. The step-100 adapter contains all 16 distributed shards and eight Energon states and reloaded successfully. This dataset does not contain video samples. This historical recipe used unqualified adapter targets that also matched vision and projector modules; it does not verify the current language-only adapter placement.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
