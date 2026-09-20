# BAGEL-7B-MoT

Megatron Bridge supports BAGEL pretraining and fine-tuning with text-to-image,
image editing, and vision-language objectives through Megatron MIMO. The H100
recipes use Megatron FSDP; the verification records below distinguish completed
runs from unsupported or unverified capabilities.

BAGEL requires the tested Megatron-LM `dev` revision and the `bagel` extra.
The Megatron-LM revision pinned by Bridge `main` does not support its runtime.
Follow the [BAGEL training guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/bagel/README.md)
for dependency setup, checkpoint initialization, training, and direct inference.
See the [BAGEL data tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/bagel/README.md)
for dataset preparation.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-bagel"></a>
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
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="" data-hardware="" data-status="unsupported" data-entry="bagel-hf-to-megatron-cpu" aria-controls="bagel-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </span>
      <span class="verification-combination-meta">Precision not specified</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="bagel-hf-to-megatron-gpu" aria-controls="bagel-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="" data-hardware="" data-status="unsupported" data-entry="bagel-megatron-to-hf-cpu" aria-controls="bagel-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </span>
      <span class="verification-combination-meta">Precision not specified</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="" data-hardware="" data-status="unsupported" data-entry="bagel-megatron-to-hf-gpu" aria-controls="bagel-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </span>
      <span class="verification-combination-meta">Precision not specified</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="bagel-pretrain-h100" aria-controls="bagel-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="bagel-sft-h100" aria-controls="bagel-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="" data-hardware="H100" data-status="unsupported" data-entry="bagel-sft-long-context-h100" aria-controls="bagel-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </span>
      <span class="verification-combination-meta">Precision not specified</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="" data-hardware="H100" data-status="unsupported" data-entry="bagel-peft-h100" aria-controls="bagel-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </span>
      <span class="verification-combination-meta">Precision not specified</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="bagel-pretrain-fsdp-h100" aria-controls="bagel-pretrain-fsdp-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · FSDP · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="bagel-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="bagel-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>NOT SPECIFIED</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>BAGEL native-checkpoint initialization currently runs as part of GPU model construction and is not registered with the maintained public CPU conversion launcher.
</p>
      </section>
    </article>
    <article id="bagel-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="bagel-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-01</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 1 --hf-model ByteDance-Seed/BAGEL-7B-MoT --hf-revision 5019f57d168e5816e8f3f701b17cc816bb7cf24b --megatron-path work/model-verification/bagel/hf-to-megatron --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The public GPU importer consumed the pinned official EMA state, reported 1,223 source tensors consumed and 943 target tensors verified, and wrote a standalone torch_dist checkpoint. A clean one-GPU reload completed without missing or unexpected checkpoint keys; all 56 MoT language-layer input-norm tensors used as regression sentinels exactly matched the official EMA values.
</p>
      </section>
    </article>
    <article id="bagel-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="bagel-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>NOT SPECIFIED</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Exporting a BAGEL Megatron checkpoint to a complete reloadable Hugging Face checkpoint is not implemented by the public CPU converter.
</p>
      </section>
    </article>
    <article id="bagel-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="bagel-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>NOT SPECIFIED</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Exporting a BAGEL Megatron checkpoint to a complete reloadable Hugging Face checkpoint is not implemented by the public GPU converter.
</p>
      </section>
    </article>
    <article id="bagel-pretrain-h100" class="verification-model-detail" data-entry-detail="bagel-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-01</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.9585528</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.242196</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>16,383.960 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>195.540 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>2,250.005 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>57.060 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>70.154 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 8 --recipe bagel_7b_pretrain_8gpu_h100_bf16_config --mode pretrain --max_steps 10 model.bagel_repo=work/dependencies/Bagel model.model_path=work/model-verification/bagel/native model.vae_path=work/model-verification/bagel/native/ae.safetensors model.native_model_checkpoint=work/model-verification/bagel/native/ema.safetensors model.native_model_seed=336 model.native_world_size=8 model.validate_native_checkpoint_metadata=false model.reference_training_seed=336 model.reference_training_world_size=8 model.reset_reference_training_rng=true dataset.dataset_root=work/data/bagel-wds dataset.bagel_repo=work/dependencies/Bagel dataset.tokenizer_model=work/model-verification/bagel/native dataset.seed=42 dataset.data_seed=42 rng.seed=42 checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false dataset.dataloader_load=null checkpoint.save=work/model-verification/bagel/pretrain/checkpoints dataset.dataloader_save=work/model-verification/bagel/pretrain/checkpoints checkpoint.save_interval=10 checkpoint.async_save=false validation.eval_iters=0 validation.eval_interval=null logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/bagel/pretrain/run_config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The clean public-data run completes 10 BF16 optimizer steps on 8 H100s at TP1/PP1/CP1/DP8 and GBS/MBS 8/1. Every step has finite CE, MSE, and total loss with no skipped or NaN iterations. The final step reports CE 0.9156674, MSE 0.3265290, and total loss 1.242196. The final-10 averages include first-step compilation and are 16,383.960 ms and 195.540 model TFLOP/s/GPU; 2,250.005 token slots/GPU/s uses the fixed 36,864 maximum packed sequence slots rather than variable logical tokens. The post-setup configuration and iteration-10 model, optimizer, RNG, and per-DP dataloader state are persisted. A separate same-topology public-launcher load restored step 10 and exited without executing an additional optimizer step.
</p>
      </section>
    </article>
    <article id="bagel-sft-h100" class="verification-model-detail" data-entry-detail="bagel-sft-h100" tabindex="-1">
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
        <p>The BAGEL fine-tuning recipe must complete a bounded pinned-data run with finite losses, all required metrics, and a reloadable checkpoint.
</p>
      </section>
    </article>
    <article id="bagel-sft-long-context-h100" class="verification-model-detail" data-entry-detail="bagel-sft-long-context-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H100</h4>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>NOT SPECIFIED</dd></div>
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
        <p>The current BAGEL integration does not expose a maintained packed long-context fine-tuning recipe with context parallelism.
</p>
      </section>
    </article>
    <article id="bagel-peft-h100" class="verification-model-detail" data-entry-detail="bagel-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>NOT SPECIFIED</dd></div>
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
        <p>BAGEL LoRA and other parameter-efficient fine-tuning workflows are not implemented by the current integration.
</p>
      </section>
    </article>
    <article id="bagel-pretrain-fsdp-h100" class="verification-model-detail" data-entry-detail="bagel-pretrain-fsdp-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · FSDP · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-31</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>14.68331</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>10.809916</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>7,697.160 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>234.268 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>4,789.299 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>58.594 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>62.919 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 8 --recipe bagel_7b_pretrain_32gpu_h100_bf16_config --mode pretrain --max_steps 30 model.bagel_repo=work/dependencies/Bagel model.model_path=work/model-verification/bagel/native model.vae_path=work/model-verification/bagel/native/ae.safetensors model.reference_training_seed=1344 model.reference_training_world_size=32 model.reset_reference_training_rng=true dataset.dataset_root=work/data/bagel-wds dataset.bagel_repo=work/dependencies/Bagel dataset.tokenizer_model=work/model-verification/bagel/native dataset.seed=42 dataset.data_seed=42 rng.seed=42 checkpoint.load=work/model-verification/bagel/seed42-mcore-init checkpoint.load_optim=true checkpoint.load_rng=true dataset.dataloader_load=work/model-verification/bagel/seed42-mcore-init ddp.overlap_grad_reduce=false ddp.overlap_param_gather=true ddp.fsdp_double_buffer=false checkpoint.save=null checkpoint.save_interval=0 validation.eval_iters=0 validation.eval_interval=0 logger.log_interval=1 logger.log_throughput=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 32-GPU TP1/PP1/CP1/DP32 real-data run completes exactly 30 BF16 steps at GBS/MBS 32/1 with one microbatch, block-23 language-model recompute, full vision recompute, Megatron FSDP, finite CE/MSE/total losses, and no skipped or NaN iterations. Steps 21-30 average 7,697.160 ms, 234.268 model TFLOP/s/GPU, 23.687% MFU, and 4,789.299 token slots/GPU/s using 36,864 maximum packed sequence slots and GBS 32 across 32 GPUs. The separately measured logical throughput is 4,428.810 tokens/GPU/s from 1,090,856.3 logical tokens per step. Logical tokens include text, ViT, VAE-latent, and special tokens and exclude physical padding. The variable packed sequence length averages 34,070.15 tokens/GPU over steps 6-30. Peak allocated/reserved memory is 58.594/62.919 GiB. The same steps 6-30 workload reports standalone raw metrics of 10,167.941 ms, 176.936 model TFLOP/s/GPU, and 17.890% MFU under official BAGEL. Do not compute a relative speedup: that run uses full language-model recompute and performs an EMA update, while the Bridge run uses block-23 recompute without EMA.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
