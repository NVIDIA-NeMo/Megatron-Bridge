# Inkling 2

[Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) is Thinking Machines' image-and-text
mixture-of-experts model with 594B total and about 17B active parameters. Its 48 decoder layers replace the residual
stream with Block Attention Residuals, interleave 1,024-token sliding-window attention with full attention on every
sixth layer, and route each token to 8 of 384 experts. The release stores the routed experts in NVFP4 and MXFP8.
Megatron Bridge builds it as a native Megatron-Core `HybridModel` through the builder-backed `Inkling2ModelConfig`,
converts the complete checkpoint including the image encoder, and keeps the speculative-decoding draft network outside
the Megatron model.

- Examples: [`examples/models/inkling`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/inkling)
  (conversion, GB200 recipes, image-text SFT)
- Implementation: [`src/megatron/bridge/models/inkling`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/inkling)
  (architecture, quantized-expert conversion, parallelism and limits)
- Verification: [model verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/inkling-2-small/card.yaml)

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-inkling-2-small"></a>
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
        <button type="button" data-hardware="GB200">GB200</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="inkling-2-small-hf-to-megatron-cpu" aria-controls="inkling-2-small-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="inkling-2-small-hf-to-megatron-gpu" aria-controls="inkling-2-small-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="inkling-2-small-megatron-to-hf-cpu" aria-controls="inkling-2-small-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="inkling-2-small-megatron-to-hf-gpu" aria-controls="inkling-2-small-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="inkling-2-small-pretrain-gb200" aria-controls="inkling-2-small-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="inkling-2-small-sft-gb200" aria-controls="inkling-2-small-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="inkling-2-small-sft-long-context-gb200" aria-controls="inkling-2-small-sft-long-context-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="inkling-2-small-peft-gb200" aria-controls="inkling-2-small-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="inkling-2-small-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="inkling-2-small-hf-to-megatron-cpu" tabindex="-1">
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
        <p>CPU import of the complete 594B checkpoint has not been attempted. GPU import is recorded separately.
</p>
      </section>
    </article>
    <article id="inkling-2-small-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="inkling-2-small-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-12</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 4 --gpus-per-node 4 --hf-model thinkingmachines/Inkling-2-Small --hf-revision 78cb6c0ef68e1f089b77e78cfd94783d05b5c41b --megatron-path work/model-verification/inkling-2-small/gpu-megatron --torch-dtype bfloat16 --tp 1 --pp 1 --ep 16 --etp 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed with 16 GB200 GPUs at TP1/PP1/EP16/ETP1: the release loads by its Hub ID, builds the complete HybridModel with the image encoder, and dequantizes the NVFP4 and MXFP8 experts to BF16 one layer at a time, in 200.6 s. The export audit under megatron_to_hf_gpu establishes its tensor content. Caveats: this ran through a private srun wrapper that imports and exports in memory rather than through this launcher, so it persisted no iter_0000000 checkpoint. The training items start from a distributed checkpoint of the same weights that a pre-release tree of this support imported and saved.
</p>
      </section>
    </article>
    <article id="inkling-2-small-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="inkling-2-small-megatron-to-hf-cpu" tabindex="-1">
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
        <p>CPU export of the complete 594B checkpoint has not been attempted. GPU export is recorded separately.
</p>
      </section>
    </article>
    <article id="inkling-2-small-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="inkling-2-small-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-12</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 4 --gpus-per-node 4 --hf-model thinkingmachines/Inkling-2-Small --hf-revision 78cb6c0ef68e1f089b77e78cfd94783d05b5c41b --megatron-path work/model-verification/inkling-2-small/gpu-megatron/iter_0000000 --hf-path work/model-verification/inkling-2-small/gpu-hf-export --torch-dtype bfloat16 --tp 1 --pp 1 --ep 16 --etp 1 --distributed-save</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed with 16 GB200 GPUs at TP1/PP1/EP16/ETP1: the export re-quantizes the routed experts against the source scales, and an exhaustive comparison against the release finds all 1,770 tensors byte-identical, including the NVFP4 and MXFP8 experts and their scales, with no missing or extra keys. The tokenizer files, chat template, processor and quantization configurations, and the MTP draft are byte-identical to the release; config.json keeps every release key and adds dtype, image_token_id, tie_word_embeddings, and transformers_version. Caveats: the export ran in memory from the model imported under hf_to_megatron_gpu, through a private srun wrapper, rather than from a saved iter_0000000 checkpoint through this launcher. Export from a saved checkpoint was exercised by the plain BF16 export of an SFT checkpoint described under sft_export_inference.
</p>
      </section>
    </article>
    <article id="inkling-2-small-pretrain-gb200" class="verification-model-detail" data-entry-detail="inkling-2-small-pretrain-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-12</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.827345</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.745319</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>18,019.690 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>203.270 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,818.455 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 32 --gpus-per-node 4 --recipe inkling2_small_pretrain_128gpu_gb200_bf16_config --mode pretrain --pretrained_checkpoint work/model-verification/inkling-2-small/gpu-megatron --max_steps 50 &#x27;dataset.blend=[[&quot;work/data/rp2/head_01_inkling2_text_document&quot;],null]&#x27; checkpoint.save=null checkpoint.load=null checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/inkling-2-small/pretrain-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed with 128 GB200 GPUs at TP1/PP4/EP32/ETP1 (pipeline stages of 13, 13, 12, and 10 decoder blocks; each EP32 group on eight consecutive nodes of one NVL72 rack; HybridEP dispatch; full activation recompute; compiled Inkling2 operators; expandable segments set in the environment as the example README instructs): the command warm-starts from the imported checkpoint with natural expert routing, keeps the image encoder frozen, and runs the first 50 of the recipe&#x27;s 100 steps on the RedPajama2 head_01 prefix at GBS 1024 and sequence length 4096, at the recipe&#x27;s peak LR of 3e-5 after 40 warmup steps (2.82e-5 at step 50). All 50 steps finish with finite losses and 0 skipped or NaN iterations. Caveats: run on 2026-11-12 through a private srun wrapper around scripts/training/run_recipe.py rather than train.sh, on a pre-release tree whose model code and recipe match the recorded commit. Checkpoint saving was off, so the run saved no step-50 or step-100 checkpoint and checkpoint_resume stays unverified. An earlier 100-step run of the same data and schedule on older code reached a loss of 1.7301 at step 100.
</p>
      </section>
    </article>
    <article id="inkling-2-small-sft-gb200" class="verification-model-detail" data-entry-detail="inkling-2-small-sft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-11</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.323619</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.6976618</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>1,481.150 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>37.740 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>345.677 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 32 --gpus-per-node 4 --recipe inkling2_small_sft_128gpu_gb200_bf16_config --mode sft --pretrained_checkpoint work/model-verification/inkling-2-small/gpu-megatron --max_steps 30 checkpoint.save=null checkpoint.load=null checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/inkling-2-small/sft-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed with 128 GB200 GPUs at TP4/PP4/EP32/ETP1 with sequence parallelism, no activation recompute, HybridEP dispatch, and compiled Inkling2 operators: the command warm-starts from the imported checkpoint and runs the first 30 of the recipe&#x27;s 100 steps of full SFT on the first 10,000 Tulu 3 conversations, rendered with the release&#x27;s chat template and packed to 8,192 tokens, at GBS 8 (eight data-parallel replicas with one pack each) and peak LR 5e-6 after 10 warmup steps. All 30 steps finish with finite losses and 0 skipped or NaN iterations. Caveats: run on 2026-11-11 through a private srun wrapper around scripts/training/run_recipe.py rather than train.sh, with checkpoint saving off, on a pre-release tree whose model code matches the recorded commit; its recipe reached the recorded recipe&#x27;s settings through recompute overrides and rendered the data with a copy of the release&#x27;s chat template that gives the same tokens and loss masks. A 100-step run of the same data and schedule on older code reached a loss of 0.5446 at step 100; sft_export_inference describes the export of its checkpoint.
</p>
      </section>
    </article>
    <article id="inkling-2-small-sft-long-context-gb200" class="verification-model-detail" data-entry-detail="inkling-2-small-sft-long-context-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-11</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.153563</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.8579385</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>4,711.280 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>47.460 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>434.701 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 32 --gpus-per-node 4 --recipe inkling2_small_sft_128gpu_gb200_bf16_long_context_config --mode sft --pretrained_checkpoint work/model-verification/inkling-2-small/gpu-megatron --max_steps 20 checkpoint.save=null checkpoint.load=null checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/inkling-2-small/sft-long-context-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed with 128 GB200 GPUs at TP4/PP4/EP32/CP4/ETP1 with sequence parallelism, Inkling2&#x27;s own context parallelism, recompute of the first 20 of 24 layers per stage, HybridEP dispatch, and compiled Inkling2 operators: the command warm-starts from the imported checkpoint and runs all 20 recipe steps of full SFT on Tulu 3 packed to 32,768 tokens at GBS 8 (two data-parallel replicas running four packs each), at peak LR 1e-6 after 2 warmup steps with decay to 0 at step 20. All 20 steps finish with finite losses and 0 skipped or NaN iterations. Caveats: run on 2026-11-11 through a private srun wrapper around scripts/training/run_recipe.py rather than train.sh, on a pre-release tree whose model code matches the recorded commit; its recipe reached the recorded recipe&#x27;s settings through recompute overrides and rendered the data with a copy of the release&#x27;s chat template that gives the same tokens and loss masks. The pre-release checkpoint&#x27;s configuration starts full-attention length scaling at 8,192 tokens instead of this revision&#x27;s 128,000, so tokens past position 8,192 of a document were scaled where this revision would not scale them.
</p>
      </section>
    </article>
    <article id="inkling-2-small-peft-gb200" class="verification-model-detail" data-entry-detail="inkling-2-small-peft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-11</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.324133</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.7994129</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>1,427.080 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>142.310 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,435.098 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 8 --gpus-per-node 4 --recipe inkling2_small_peft_32gpu_gb200_bf16_config --mode lora --pretrained_checkpoint work/model-verification/inkling-2-small/gpu-megatron --max_steps 30 checkpoint.save=null checkpoint.load=null checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/inkling-2-small/peft-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed with 32 GB200 GPUs at TP4/PP1/EP32/ETP1 with sequence parallelism, no activation recompute, HybridEP dispatch, and compiled Inkling2 operators: the command loads the imported checkpoint, freezes the base model, and trains rank-8, alpha-16 LoRA adapters on linear_q, linear_k, linear_v, and linear_proj for the first 30 of the recipe&#x27;s 100 steps on the packed 8K Tulu 3 data of the sft item at GBS 8 and peak LR 1e-4 after 10 warmup steps. All 30 steps finish with finite losses and 0 skipped or NaN iterations. The TFLOP/s value is Bridge&#x27;s generic estimate, which counts a full backward pass. Caveats: run on 2026-11-11 through a private srun wrapper around scripts/training/run_recipe.py rather than train.sh, with checkpoint saving off, on a pre-release tree whose model code matches the recorded commit; its recipe reached the recorded recipe&#x27;s settings through recompute overrides and rendered the data with a copy of the release&#x27;s chat template that gives the same tokens and loss masks. A 100-step run of the same data and schedule on older code reached a loss of 0.7560 at step 100.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
