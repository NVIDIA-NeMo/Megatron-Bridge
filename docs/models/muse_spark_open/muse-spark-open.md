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
        <div><dt>Last verified</dt><dd>2026-10-06</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 4 --gpus-per-node 4 --hf-model meta-models/Muse-Spark-Open-816B-A42B --hf-revision unpinned --megatron-path work/model-verification/muse-spark-open-816b-a42b/gpu-megatron --torch-dtype bfloat16 --tp 4 --pp 1 --ep 16 --etp 1 --low-memory-save --distributed-timeout-minutes 110</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed on the unpinned local checkpoint with 16 GB200 GPUs: the command builds the complete native HybridModel with both Hugging Face perception encoders, loads all 2,000 source tensors at TP4/PP1/EP16/ETP1, and persists a 1.5 TB iter_0000000 distributed checkpoint from all 16 ranks. The exported audit under megatron_to_hf_gpu establishes its tensor content. Verified on a local copy of the checkpoint whose public Hugging Face revision is not yet pinned: model.hf_revision stays unpinned, which the card validator rejects for verified items. Rerun against the pinned revision once it is published.
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
        <div><dt>Last verified</dt><dd>2026-10-06</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 8 --gpus-per-node 4 --hf-model meta-models/Muse-Spark-Open-816B-A42B --hf-revision unpinned --megatron-path work/model-verification/muse-spark-open-816b-a42b/gpu-megatron/iter_0000000 --hf-path work/model-verification/muse-spark-open-816b-a42b/gpu-hf-export --torch-dtype bfloat16 --tp 2 --pp 1 --ep 32 --etp 1 --distributed-save --distributed-timeout-minutes 110</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Observed on the unpinned local checkpoint with 32 GB200 GPUs: the TP4/EP16 import reloads at TP2/PP1/EP32/ETP1, exercising tensor- and expert-parallel resharding, and strict distributed export writes all 65 safetensors shards plus configuration and tokenizer files from all 32 ranks. An exhaustive comparison of the export against the source covers all 2,000 tensors and 822,036,529,664 elements with no missing, extra, or reshaped keys. All values match exactly except 1,204 of 8,192 entries of model.language_model.norm.weight (maximum absolute difference 0.015625), the known BF16 rounding of the zero-centered final-norm conversion. The 124 residual-gate tensors are promoted from BF16 to FP32 with unchanged values, and the FP32 router tensors keep FP32. Do not pass --export-weight-dtype bfloat16: it would round the FP32 router tensors. Export at TP1/EP16 exhausts GPU memory during checkpoint resharding. Verified on a local copy of the checkpoint whose public Hugging Face revision is not yet pinned: model.hf_revision stays unpinned, which the card validator rejects for verified items. Rerun against the pinned revision once it is published.
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
        <div><dt>Last verified</dt><dd>2026-10-14</dd></div>
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
            <dd>1.720235</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>22,561.470 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>267.290 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>968.258 tokens/s/GPU</dd>
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
        <p>Observed on the unpinned local checkpoint with 192 GB200 GPUs at TP1/PP6/EP32/ETP1 (pipeline stages of 9, 11, 11, 11, 11, and 9 decoder blocks): the command warm-starts every language-model parameter from the imported checkpoint with natural expert routing, full activation recompute, BF16 gradient reduction, and the distributed optimizer; the perception towers stay frozen because the data is text only. MoE tokens are dispatched with HybridEP (flex dispatcher, 32 SMs): each EP32 group is one pipeline stage&#x27;s 32 contiguous ranks on 8 nodes, which must share one NVL72 NVLink domain, and the recipe sets NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32, NVLINK_DOMAIN_SIZE=72, and USE_MNNVL=1. Data is the first 604,180 documents of RedPajama2 head_01 (750,001,072 tokens in the shared LLaMA SentencePiece tokenization), decoded and re-tokenized with the model tokenizer and an appended end-of-document token: 677,845,358 tokens, enough for 100 steps at GBS 1024 and sequence length 4096 without repetition. Deviation from the bounded pretraining contract: the peak learning rate is 3e-5 (40 warmup steps, cosine decay to 3e-6 at step 100) instead of 3e-4 to 3e-5. In an earlier run with the contract&#x27;s from-scratch 3e-4 peak, all-to-all dispatch, and the precision-aware optimizer enabled (every other setting unchanged), the warm-started model&#x27;s loss rose from 1.841362 to 3.26 at the learning-rate peak (step 40) and to a maximum of 5.34 at step 44, stayed above 3.5 through step 62, and ended at 2.264579. All 100 optimizer-step rows are present exactly once with finite loss and gradient norm and zero skipped or NaN iterations. Loss decreases from 1.841344 to 1.720235 (minimum 1.7057 at step 99) with no rise at the learning-rate peak (1.7721 at step 40); steps 11-100 stay between 1.7057 and 1.7937, and gradient-norm spikes (59.7 at step 6, 13.3 at step 16) are clipped at 1.0. Steps 91-100 average 22,561.47 ms, 267.29 model TFLOP/s/GPU, and 968.258 token slots/s/GPU. Peak allocated memory per pipeline stage is 97.2, 104.8, 104.8, 104.8, 104.8, and 93.3 GiB. Full model, optimizer, and RNG state, about 11.4 TB per checkpoint, is saved at steps 50 and 100; the step-50 checkpoint is checkpoint_resume&#x27;s starting point. This run used the recipe as is (HybridEP and its environment variables come from the recipe) on the branch&#x27;s Megatron-Core pin, which includes the distributed-optimizer fix NVIDIA/Megatron-LM#7336; without it, saving optimizer state fails because the FP32 router weights and residual gates share optimizer groups with BF16 parameters. With the precision-aware optimizer enabled and everything else unchanged, the same command ended at 1.728448 (steps 91-100 average 1.7304 against 1.7221 here) at the same step time and peak memory. An earlier run without saves and with the precision-aware optimizer enabled, on an earlier Megatron-Core with HybridEP selected by command-line overrides, ended at 1.726057 with 22,483.32 ms per step over steps 91-100 (its model TFLOP/s/GPU, 243.93, used that version&#x27;s FLOP count). The same command on 64 nodes with muse_spark_open_pretrain_256gpu_gb200_bf16_config, which keeps the all-to-all dispatcher (PP8, stages of 7, 8, 8, 8, 8, 8, 8, and 7 blocks; the same data-parallel size and batches), run without saves, ends at 1.724089, averages 19,776.13 ms, 228.69 model TFLOP/s/GPU, and 828.474 token slots/s/GPU over steps 91-100, and peaks at 80.1 GiB per stage. The post-setup config persists. Verified on a local copy of the checkpoint whose public Hugging Face revision is not yet pinned: model.hf_revision stays unpinned, which the card validator rejects for verified items. Rerun against the pinned revision once it is published.
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
        <div><dt>Last verified</dt><dd>2026-10-12</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.41593</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.7347853</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>6,688.570 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>101.550 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>408.259 tokens/s/GPU</dd>
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
        <p>Observed on the unpinned local checkpoint with 192 GB200 GPUs: all 85 optimizer-step rows are present exactly once with finite loss and gradient norm and zero skipped or NaN iterations. Loss moves from 1.41593 to 0.7347853 (minimum 0.6878 at step 42) and averages 0.7875 over steps 51-85; gradient norms of 91.4 at step 1 and up to 178.8 at step 4 are clipped at 1.0. Steps 76-85 average 6,688.57 ms, 101.55 model TFLOP/s/GPU, and 408.259 token slots/s/GPU; each data-parallel replica runs 16 microbatches per step. Peak allocated memory per pipeline stage is 87.7, 95.6, 95.6, 94.4, 94.4, and 80.9 GiB. The post-setup config persists; no checkpoint is saved, so sft_export_inference still refers to an earlier checkpoint. The command warm-starts every language-model parameter from the imported checkpoint at TP4/PP6/EP32/ETP1 with sequence parallelism (pipeline stages of 9, 11, 11, 11, 11, and 9 decoder blocks, eight data-parallel replicas), full activation recompute, and the distributed optimizer with FP32 gradient reduction at peak LR 5e-6; the perception towers stay frozen because the data is text only. MoE tokens are dispatched with HybridEP (flex dispatcher, 32 SMs): each EP32 group is one pipeline stage&#x27;s 32 contiguous ranks on 8 nodes, which must share one NVL72 NVLink domain, and the recipe sets NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32, NVLINK_DOMAIN_SIZE=72, and USE_MNNVL=1. Packed sequences give the EP ranks different token counts, which the recipe pads to the group maximum (moe_hybridep_pad_uneven_dispatch_inputs); the data and loss mask are unchanged. Data is the first 100,000 conversations of the revision-pinned Tulu 3 SFT mixture, rendered with the model chat template whose current-date default is fixed to 2026-10-04 and offline-packed to 4,096 tokens with pad_seq_to_mult 4: 10,951 packs of 9.1 conversations on average, 89 conversations (0.09%) truncated at 4,092 tokens. GBS 128 for 85 steps is one epoch over 10,880 distinct packs (max_train_samples leaves the other 71 packs unused), with 10 warmup steps and cosine decay to 0 at step 85, model seed 5678, and data seed 1234. Not comparable with other cards: this deviates from the qwen3_30b_a3b_convergence_v2 SFT/PEFT contract (Tulu 3 train[:10000], 8,192-token packs, GBS 8, 100 steps), so it is support verification rather than cross-model convergence evidence. The mixture is stored grouped by source, so train[:100000] is 90.0% FLAN v2, 7.1% OASST1, 2.6% No Robots, and 0.2% hard-coded prompts. muse_spark_open_sft_256gpu_gb200_bf16_config keeps the contract shape. With those contract settings and all-to-all dispatch on the same 192 GPUs the recipe moved the loss from 1.155696 to 0.5453231, and the 256-GPU recipe (TP4/PP8/EP32, which keeps the all-to-all dispatcher) from 1.155695 to 0.5597777. The run used the earlier recipe module with command-line overrides for the data, batch, schedule, and dispatcher and exported the HybridEP environment variables; the current recipe resolves to the same data, batch, schedule, optimizer, parallelism, recompute, and dispatcher settings as the run&#x27;s saved post-setup configuration, apart from the unused checkpoint save interval. The image-text recipe muse_spark_open_sft_192gpu_gb200_bf16_multimodal_config (CORD-v2, TP1/PP6/EP32) completes 100 steps on the same 192 GPUs (loss 0.8243734 to 0.02772351) with zero skipped or NaN iterations. Verified on a local copy of the checkpoint whose public Hugging Face revision is not yet pinned: model.hf_revision stays unpinned, which the card validator rejects for verified items. Rerun against the pinned revision once it is published.
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
        <div><dt>Last verified</dt><dd>2026-10-14</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.07948</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.8569268</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>5,527.290 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>70.030 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>247.017 tokens/s/GPU</dd>
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
        <p>Observed on the unpinned local checkpoint with 192 GB200 GPUs at TP4/PP6/CP2/EP32/ETP1 with sequence parallelism and all-to-all context parallelism (pipeline stages of 9, 11, 11, 11, 11, and 9 decoder blocks, four data-parallel replicas): all 20 optimizer-step rows are present exactly once with finite loss and gradient norm and zero skipped or NaN iterations. Loss moves from 1.07948 to 0.8569268 (minimum 0.822026 at step 13); gradient norm falls from 19.3 at step 1 to between 1.3 and 5.6 from step 7 on and is clipped at 1.0. Steps 11-20 average 5,527.29 ms, 70.03 model TFLOP/s/GPU, and 247.017 token slots/s/GPU. Peak allocated memory per pipeline stage is 98.5, 105.8, 107.8, 104.4, 107.5, and 89.5 GiB. The post-setup config persists; no checkpoint is saved. The command warm-starts every language-model parameter from the imported checkpoint with full activation recompute and the distributed optimizer with FP32 gradient reduction; the perception towers stay frozen because the data is text only. Data is the revision-pinned Tulu 3 SFT mixture train[:10000], rendered with the model chat template whose current-date default is fixed to 2026-10-04 and offline-packed to 32,768 tokens with pad_seq_to_mult 8 (2 x CP, and TP x CP for sequence parallelism): 143 packs averaging 69.9 sequences, so 20 steps at GBS 8 make about 1.1 passes. Peak LR is 1e-6 with 2 warmup steps and cosine decay to 0 at step 20; model seed 5678 and data seed 1234. Tulu 3 rows are shorter than the 2,048-token sliding window, so this run exercises packed CP plumbing rather than attention across long single sequences. muse_spark_open_sft_192gpu_gb200_bf16_coderforge_64k_config covers long single sequences on the same 192 GPUs at TP4/PP6/CP4/EP32: 600 CoderForge SWE-Rebench trajectories (median 46K tokens) packed to 65,536 tokens complete 20 steps with zero skipped or NaN iterations, loss 0.1977274 to 0.131147, and steps 11-20 averaging 10,372.72 ms and 73.98 model TFLOP/s/GPU. Verified on a local copy of the checkpoint whose public Hugging Face revision is not yet pinned: model.hf_revision stays unpinned, which the card validator rejects for verified items. Rerun against the pinned revision once it is published.
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
        <div><dt>Last verified</dt><dd>2026-10-12</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.415098</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.7489657</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>15,306.700 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>132.970 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>535.190 tokens/s/GPU</dd>
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
        <p>Observed on the unpinned local checkpoint with 64 GB200 GPUs at TP4/PP4/EP16/ETP1 with sequence parallelism (pipeline stages of 15, 16, 16, and 15 decoder blocks, four data-parallel replicas running 32 microbatches each per step): the command loads the imported 816B checkpoint, freezes the base, and trains LoRA rank 32 / alpha 32 / dropout 0 adapters on the linear_qkv and linear_proj attention projections, on linear_fc1 and linear_fc2 of every routed expert (one adapter per expert), and on the fc1_latent_proj and fc2_latent_proj latent MoE projections, at peak LR 1e-4 with FP32 gradient reduction. MoE tokens are dispatched with HybridEP (flex dispatcher, 32 SMs): each EP16 group is one pipeline stage&#x27;s 16 contiguous ranks on 4 nodes, which must share one NVL72 NVLink domain, and the recipe sets NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=16, NVLINK_DOMAIN_SIZE=72, and USE_MNNVL=1. Packed sequences give the EP ranks different token counts, which the recipe pads to the group maximum (moe_hybridep_pad_uneven_dispatch_inputs); the data and loss mask are unchanged. Data is the first 100,000 conversations of the revision-pinned Tulu 3 SFT mixture, rendered with the model chat template whose current-date default is fixed to 2026-10-04 and offline-packed to 4,096 tokens with pad_seq_to_mult 4: 10,951 packs of 9.1 conversations on average, 89 conversations (0.09%) truncated at 4,092 tokens. GBS 128 for 85 steps is one epoch over 10,880 distinct packs (max_train_samples leaves the other 71 packs unused), with 10 warmup steps and cosine decay to 0 at step 85, model seed 5678, and data seed 1234. All 85 optimizer-step rows are present exactly once with finite loss and zero skipped or NaN iterations. Loss moves from 1.415098 to 0.7489657 (minimum 0.6997 at step 42) and averages 0.8031 over steps 51-85; gradient norms up to 8.4 are clipped at 1.0. Steps 76-85 average 15,306.70 ms, 132.97 model TFLOP/s/GPU, and 535.190 token slots/s/GPU. Peak allocated memory per pipeline stage is 98.2, 83.1, 65.4, and 49.0 GiB; the expert adapters keep the routed expert inputs for their weight gradients, which is why the recipe uses four pipeline stages. The post-setup config persists; no adapter checkpoint is saved (with these targets the adapter with its optimizer state is about 140 GB). Not comparable with other cards: this deviates from the qwen3_30b_a3b_convergence_v2 SFT/PEFT contract (Tulu 3 train[:10000], 8,192-token packs, GBS 8, 100 steps), so it is support verification rather than cross-model convergence evidence. The mixture is stored grouped by source, so train[:100000] is 90.0% FLAN v2, 7.1% OASST1, 2.6% No Robots, and 0.2% hard-coded prompts. With those contract settings and all-to-all dispatch this recipe moved the loss from 1.155864 to 0.8608636 and saved its adapter. The run used the earlier recipe module with command-line overrides for the data, batch, schedule, and dispatcher and exported the HybridEP environment variables; the current recipe resolves to the same data, batch, schedule, optimizer, PEFT, parallelism, recompute, and dispatcher settings as the run&#x27;s saved post-setup configuration, apart from the unused checkpoint save interval. The image-text recipe muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config (CORD-v2) also trains with zero skipped or NaN iterations at TP4/EP16 (100 steps, loss 0.8218 to 0.0193), TP4/EP32, and TP4/CP2/EP16 with A2A (30 steps each). Verified on a local copy of the checkpoint whose public Hugging Face revision is not yet pinned: model.hf_revision stays unpinned, which the card validator rejects for verified items. Rerun against the pinned revision once it is published.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
