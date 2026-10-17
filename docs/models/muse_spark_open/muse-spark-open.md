# Muse-Spark-Open-816B-A42B

Muse-Spark-Open-816B-A42B is a sparse mixture-of-experts vision-and-audio-language model.
Megatron Bridge models the complete checkpoint: the vision tower, vision adapter and
projection, the audio tower, and the text decoder all participate in checkpoint
conversion. A text-only mode converts the language model and its separate LM head alone.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-muse-spark-open-816b-a42b"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="muse-spark-open-816b-a42b-hf-to-megatron-cpu" aria-controls="muse-spark-open-816b-a42b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="muse-spark-open-816b-a42b-hf-to-megatron-gpu" aria-controls="muse-spark-open-816b-a42b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="muse-spark-open-816b-a42b-megatron-to-hf-cpu" aria-controls="muse-spark-open-816b-a42b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="muse-spark-open-816b-a42b-megatron-to-hf-gpu" aria-controls="muse-spark-open-816b-a42b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="muse-spark-open-816b-a42b-pretrain-gb200" aria-controls="muse-spark-open-816b-a42b-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="muse-spark-open-816b-a42b-sft-gb200" aria-controls="muse-spark-open-816b-a42b-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="muse-spark-open-816b-a42b-sft-long-context-gb200" aria-controls="muse-spark-open-816b-a42b-sft-long-context-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="muse-spark-open-816b-a42b-peft-gb200" aria-controls="muse-spark-open-816b-a42b-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="muse-spark-open-816b-a42b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-hf-to-megatron-cpu" tabindex="-1">
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
        <p>CPU import of the complete 816B checkpoint has not been attempted. GPU import is recorded separately.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-10-16</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 4 --gpus-per-node 4 --hf-model meta-models/Muse-Spark-Open-816B-A42B --hf-revision 0fb02478b02565b2ce7fc57e6051c91172774ac9 --megatron-path work/model-verification/muse-spark-open-816b-a42b/gpu-megatron --torch-dtype bfloat16 --tp 4 --pp 1 --ep 16 --etp 1 --low-memory-save --distributed-timeout-minutes 110</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>With 16 GB200 GPUs the command builds the complete native HybridModel with both Hugging Face perception encoders, loads the checkpoint at TP4/PP1/EP16/ETP1, and persists a 1.5 TB iter_0000000 distributed checkpoint from all 16 ranks. The exported audit under megatron_to_hf_gpu establishes its tensor content, and the training items warm-start from this checkpoint.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-megatron-to-hf-cpu" tabindex="-1">
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
        <p>CPU export of the complete 816B checkpoint has not been attempted. GPU export is recorded separately.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-10-16</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 8 --gpus-per-node 4 --hf-model meta-models/Muse-Spark-Open-816B-A42B --hf-revision 0fb02478b02565b2ce7fc57e6051c91172774ac9 --megatron-path work/model-verification/muse-spark-open-816b-a42b/gpu-megatron/iter_0000000 --hf-path work/model-verification/muse-spark-open-816b-a42b/gpu-hf-export --torch-dtype bfloat16 --tp 2 --pp 1 --ep 32 --etp 1 --distributed-save --distributed-timeout-minutes 110</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>With 32 GB200 GPUs the TP4/EP16 import reloads at TP2/PP1/EP32/ETP1, exercising tensor- and expert-parallel resharding, and strict distributed export writes all 65 safetensors shards plus configuration and tokenizer files from all 32 ranks. An exhaustive comparison of the export against the source covers all 2,000 tensors and 822,036,529,664 elements with no missing, extra, or reshaped keys. All values match exactly except 1,204 of 8,192 entries of model.language_model.norm.weight (maximum absolute difference 0.015625), the known BF16 rounding of the zero-centered final-norm conversion. The 124 residual-gate tensors are promoted from BF16 to FP32 with unchanged values, and the FP32 router tensors keep FP32. Do not pass --export-weight-dtype bfloat16: it would round the FP32 router tensors.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-pretrain-gb200" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-pretrain-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-10-16</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.841344</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.724736</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>23,978.620 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>254.050 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>911.034 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 48 --gpus-per-node 4 --recipe muse_spark_open_pretrain_192gpu_gb200_bf16_config --mode pretrain --dataset megatron-indexed --pretrained_checkpoint work/model-verification/muse-spark-open-816b-a42b/gpu-megatron --max_steps 100 &#x27;dataset.blend=[[&quot;work/data/rp2/head_01_musespark_text_document&quot;],null]&#x27; dataset.path_to_cache=work/cache/muse-spark-open-816b-a42b/rp2-head-01-gb200 dataset.random_seed=1234 dataset.num_workers=8 rng.seed=1234 scheduler.lr_decay_iters=100 checkpoint.load=null --save_dir work/model-verification/muse-spark-open-816b-a42b/pretrain-checkpoints --save_interval 50 checkpoint.async_save=false ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/muse-spark-open-816b-a42b/pretrain-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed on the pinned public revision with 192 GB200 GPUs at TP1/PP6/EP32/ETP1 (pipeline stages of 9, 11, 11, 11, 11, and 9 decoder blocks; 32 data-parallel replicas running 32 microbatches each per step): the command warm-starts every language-model parameter from the imported checkpoint with natural expert routing, full activation recompute (uniform, one layer per recompute group), BF16 gradient reduction, and the distributed optimizer; the perception towers stay frozen because the data is text only. MoE tokens are dispatched with HybridEP (flex dispatcher, 32 SMs): each EP32 group is one pipeline stage&#x27;s 32 contiguous ranks on 8 nodes in one NVL72 NVLink domain, and the recipe sets NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32, NVLINK_DOMAIN_SIZE=72, and USE_MNNVL=1. Data is the first 604,180 documents of RedPajama2 head_01 (750,001,072 tokens in the shared LLaMA SentencePiece tokenization), decoded and re-tokenized with the model tokenizer and an appended end-of-document token: 677,845,358 tokens, enough for 100 steps at GBS 1024 and sequence length 4096 without repetition. Deviations from the bounded pretraining contract for this warm-started model: the peak learning rate is 3e-5 (40 warmup steps, cosine decay to 3e-6 at step 100) instead of 3e-4 to 3e-5, and the precision-aware optimizer is disabled; the effective weight decay is 0.033. All 100 optimizer-step rows are present exactly once with finite loss and gradient norm and zero skipped or NaN iterations. Loss decreases from 1.841344 to 1.724736 (minimum 1.709827 at step 99) with no rise at the learning-rate peak (1.761883 at step 40); steps 11-100 stay between 1.7098 and 1.8046, and gradient-norm spikes (48.1 at step 46, 9.0 at step 24) are clipped at 1.0. Steps 91-100 average 23,978.62 ms, 254.05 model TFLOP/s/GPU, and 911.034 token slots/s/GPU. Peak allocated memory per pipeline stage is 97.2, 104.8, 104.8, 104.8, 104.8, and 93.3 GiB. Full model, optimizer, and RNG state, about 11.5 TB per checkpoint, is saved at steps 50 and 100; the step-50 checkpoint is checkpoint_resume&#x27;s starting point. The post-setup config persists.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-sft-gb200" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-sft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-10-16</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.428229</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.7338403</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>4,195.300 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>178.800 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>650.887 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 48 --gpus-per-node 4 --recipe muse_spark_open_sft_192gpu_gb200_bf16_config --mode sft --pretrained_checkpoint work/model-verification/muse-spark-open-816b-a42b/gpu-megatron --max_steps 85 tokenizer.chat_template_path=work/templates/muse_spark_open_chat_template_pinned_date.jinja dataset.hf_output_root=work/data/tulu3/muse-spark-open-100k-4k checkpoint.save=null checkpoint.load=null checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/muse-spark-open-816b-a42b/sft-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed on the pinned public revision with 192 GB200 GPUs at TP4/PP3/EP64/ETP1 with sequence parallelism (pipeline stages of 21, 22, and 19 decoder blocks; 16 data-parallel replicas running 8 microbatches each per step): the command warm-starts every language-model parameter from the imported checkpoint with no activation recompute and the distributed optimizer with FP32 gradient reduction at peak LR 5e-6; the perception towers stay frozen because the data is text only. MoE tokens are dispatched with HybridEP (flex dispatcher, 32 SMs): each EP64 group is one pipeline stage&#x27;s 64 contiguous ranks on 16 nodes in one NVL72 NVLink domain, and the recipe sets NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64, NVLINK_DOMAIN_SIZE=72, and USE_MNNVL=1. Data is the first 100,000 conversations of the revision-pinned Tulu 3 SFT mixture, rendered with the model chat template whose current-date default is fixed to 2026-10-04 and offline-packed to 4,096 tokens with pad_seq_to_mult 4: 10,951 packs of 9.1 conversations on average, 90 conversations (0.09%) truncated at 4,092 tokens. Every micro-batch is padded to the full 4,096 tokens (pad_to_max_length), and padding carries no loss. GBS 128 for 85 steps is one epoch over 10,880 distinct packs (max_train_samples leaves the other 71 packs unused), with 10 warmup steps and cosine decay to 0 at step 85, model seed 5678, and data seed 1234. All 85 optimizer-step rows are present exactly once with finite loss and gradient norm and zero skipped or NaN iterations. Loss moves from 1.428229 to 0.7338403 (minimum 0.6910 at step 42) and averages 0.7870 over steps 51-85; gradient norms of 67.8 at step 1 and 108.0 at step 15 are clipped at 1.0. Steps 76-85 average 4,195.30 ms, 178.80 model TFLOP/s/GPU, and 650.887 token slots/s/GPU. Peak allocated memory per pipeline stage is 156.2, 142.5, and 108.8 GiB. The post-setup config persists; no checkpoint is saved. Not comparable with other cards: this deviates from the qwen3_30b_a3b_convergence_v2 SFT/PEFT contract (Tulu 3 train[:10000], 8,192-token packs, GBS 8, 100 steps), so it is support verification rather than cross-model convergence evidence. The mixture is stored grouped by source, so train[:100000] is 90.0% FLAN v2, 7.1% OASST1, 2.6% No Robots, and 0.2% hard-coded prompts.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-sft-long-context-gb200" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-sft-long-context-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-10-16</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.082231</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.8572639</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>5,614.890 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>69.230 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>243.163 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 48 --gpus-per-node 4 --recipe muse_spark_open_sft_192gpu_gb200_bf16_long_context_config --mode sft --pretrained_checkpoint work/model-verification/muse-spark-open-816b-a42b/gpu-megatron --max_steps 20 tokenizer.chat_template_path=work/templates/muse_spark_open_chat_template_pinned_date.jinja dataset.hf_output_root=work/data/tulu3/muse-spark-open-full-sft-gb200-32k checkpoint.save=null checkpoint.load=null checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/muse-spark-open-816b-a42b/sft-long-context-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed on the pinned public revision with 192 GB200 GPUs at TP4/PP6/CP2/EP32/ETP1 with sequence parallelism and all-to-all context parallelism (pipeline stages of 9, 11, 11, 11, 11, and 9 decoder blocks; four data-parallel replicas running two microbatches each per step): the command warm-starts every language-model parameter from the imported checkpoint with full activation recompute, the all-to-all MoE dispatcher, and the distributed optimizer with FP32 gradient reduction; the perception towers stay frozen because the data is text only. Data is the revision-pinned Tulu 3 SFT mixture train[:10000], rendered with the model chat template whose current-date default is fixed to 2026-10-04 and offline-packed to 32,768 tokens with pad_seq_to_mult 8 (2 x CP, and TP x CP for sequence parallelism): 143 packs averaging 69.9 sequences, none truncated, so 20 steps at GBS 8 make about 1.1 passes. The longest sequence has 8,688 tokens, so the run exercises packed context parallelism rather than attention across one 32K-token sequence. Peak LR is 1e-6 with 2 warmup steps and cosine decay to 0 at step 20; model seed 5678 and data seed 1234. All 20 optimizer-step rows are present exactly once with finite loss and gradient norm and zero skipped or NaN iterations. Loss moves from 1.082231 to 0.8572639 (minimum 0.8273457 at step 13); gradient norm is between 6.4 and 20.2 over steps 1-5, 118.6 at step 6, and between 1.5 and 3.5 from step 7 on, and is clipped at 1.0. Steps 11-20 average 5,614.89 ms, 69.23 model TFLOP/s/GPU, and 243.163 token slots/s/GPU. Peak allocated memory per pipeline stage is 98.4, 105.8, 107.8, 104.4, 107.2, and 89.4 GiB. The post-setup config persists; no checkpoint is saved.
</p>
      </section>
    </article>
    <article id="muse-spark-open-816b-a42b-peft-gb200" class="verification-model-detail" data-entry-detail="muse-spark-open-816b-a42b-peft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-10-16</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.428229</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.74849</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>11,099.810 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>201.840 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>738.031 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 16 --gpus-per-node 4 --recipe muse_spark_open_peft_64gpu_gb200_bf16_config --mode lora --pretrained_checkpoint work/model-verification/muse-spark-open-816b-a42b/gpu-megatron --max_steps 85 tokenizer.chat_template_path=work/templates/muse_spark_open_chat_template_pinned_date.jinja dataset.hf_output_root=work/data/tulu3/muse-spark-open-100k-4k checkpoint.save=null checkpoint.load=null checkpoint.async_save=false ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true logger.log_interval=1 logger.log_throughput=true logger.log_device_memory_used=true logger.tensorboard_dir=null logger.save_config_filepath=work/model-verification/muse-spark-open-816b-a42b/peft-resolved.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed on the pinned public revision with 64 GB200 GPUs at TP4/PP1/EP64/ETP1 with sequence parallelism (16 data-parallel replicas running 8 microbatches each per step): the command loads the imported checkpoint, freezes the base model, and trains LoRA rank 32 / alpha 32 / dropout 0 adapters on the linear_qkv and linear_proj attention projections, on linear_fc1 and linear_fc2 of every routed expert (one adapter per expert), and on the fc1_latent_proj and fc2_latent_proj latent MoE projections, at peak LR 1e-4 with FP32 gradient reduction and no activation recompute. MoE tokens are dispatched with HybridEP (flex dispatcher, 32 SMs): the EP64 group is all 64 ranks on 16 nodes in one NVL72 NVLink domain, and the recipe sets NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64, NVLINK_DOMAIN_SIZE=72, and USE_MNNVL=1. Data is the first 100,000 conversations of the revision-pinned Tulu 3 SFT mixture, rendered with the model chat template whose current-date default is fixed to 2026-10-04 and offline-packed to 4,096 tokens with pad_seq_to_mult 4: 10,951 packs of 9.1 conversations on average, 90 conversations (0.09%) truncated at 4,092 tokens. Every micro-batch is padded to the full 4,096 tokens (pad_to_max_length), and padding carries no loss. GBS 128 for 85 steps is one epoch over 10,880 distinct packs (max_train_samples leaves the other 71 packs unused), with 10 warmup steps and cosine decay to 0 at step 85, model seed 5678, and data seed 1234. All 85 optimizer-step rows are present exactly once with finite loss and gradient norm and zero skipped or NaN iterations. Loss moves from 1.428229 to 0.74849 (minimum 0.6982 at step 42) and averages 0.8023 over steps 51-85; gradient norms up to 7.8 are clipped at 1.0. Steps 76-85 average 11,099.81 ms, 201.84 model TFLOP/s/GPU, and 738.031 token slots/s/GPU. Peak allocated memory is 103.2 GiB. The post-setup config persists; no adapter checkpoint is saved. Not comparable with other cards: this deviates from the qwen3_30b_a3b_convergence_v2 SFT/PEFT contract (Tulu 3 train[:10000], 8,192-token packs, GBS 8, 100 steps), so it is support verification rather than cross-model convergence evidence. The mixture is stored grouped by source, so train[:100000] is 90.0% FLAN v2, 7.1% OASST1, 2.6% No Robots, and 0.2% hard-coded prompts.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
