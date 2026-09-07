# Gemma 4 VL (26B-A4B MoE)

[Google's Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it)
is a Mixture-of-Experts vision-language model (26B total, 4B active
parameters). It pairs a 128-expert top-k=8 MoE language backbone with a
SigLIP vision tower, dual sliding/global attention, and K=V tying on the
full-attention layers.

NeMo Megatron Bridge supports HF↔Megatron conversion, full SFT, and LoRA
PEFT on image-text datasets. The finetuned model can be re-exported to
🤗 Hugging Face format for downstream evaluation or deployment.

For the full setup, conversion, inference, training, and LoRA merge /
adapter export workflows, see
[`examples/models/gemma/gemma4_vl/README.md`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/gemma/gemma4_vl/README.md).

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-gemma-4-26b-a4b-it"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="gemma-4-26b-a4b-it-hf-to-megatron-cpu" aria-controls="gemma-4-26b-a4b-it-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="gemma-4-26b-a4b-it-hf-to-megatron-gpu" aria-controls="gemma-4-26b-a4b-it-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="gemma-4-26b-a4b-it-megatron-to-hf-cpu" aria-controls="gemma-4-26b-a4b-it-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="gemma-4-26b-a4b-it-megatron-to-hf-gpu" aria-controls="gemma-4-26b-a4b-it-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="" data-hardware="" data-status="unsupported" data-entry="gemma-4-26b-a4b-it-pretrain-all" aria-controls="gemma-4-26b-a4b-it-pretrain-all" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain</strong>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </span>
      <span class="verification-combination-meta">Precision not specified</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="gemma-4-26b-a4b-it-sft-h100" aria-controls="gemma-4-26b-a4b-it-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="gemma-4-26b-a4b-it-sft-long-context-h100" aria-controls="gemma-4-26b-a4b-it-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="gemma-4-26b-a4b-it-peft-h100" aria-controls="gemma-4-26b-a4b-it-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="gemma-4-26b-a4b-it-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="gemma-4-26b-a4b-it-hf-to-megatron-cpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model google/gemma-4-26B-A4B-it --hf-revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --megatron-path work/model-verification/gemma-4-26b-a4b-it/cpu-megatron --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command completed all 8,548 conversion mappings and saved iter_0000000. All 383 tensor entries in the distributed-checkpoint metadata were torch.bfloat16. The run configuration recorded bf16=true, fp16=false, and BF16 parameter and autocast dtypes. The artifact reloaded successfully for CPU export.
</p>
      </section>
    </article>
    <article id="gemma-4-26b-a4b-it-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="gemma-4-26b-a4b-it-hf-to-megatron-gpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model google/gemma-4-26B-A4B-it --hf-revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --megatron-path work/model-verification/gemma-4-26b-a4b-it/imported-megatron --torch-dtype bfloat16 --tp 4 --pp 2 --ep 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command completed successfully at TP4/PP2/EP1. It loaded 8,548 conversion mappings from the immutable Hugging Face revision and saved the full multimodal checkpoint as iter_0000000. All 383 tensor entries in the distributed-checkpoint metadata were torch.bfloat16. The artifact then reloaded successfully for distributed export and direct-model inference.
</p>
      </section>
    </article>
    <article id="gemma-4-26b-a4b-it-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="gemma-4-26b-a4b-it-megatron-to-hf-cpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model google/gemma-4-26B-A4B-it --hf-revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --megatron-path work/model-verification/gemma-4-26b-a4b-it/cpu-megatron/iter_0000000 --hf-path work/model-verification/gemma-4-26b-a4b-it/cpu-hf-export --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command completed all 8,548 conversion mappings and wrote 11 BF16 shards. All 1,013 exported tensor entries matched the immutable source exactly in key, shape, dtype, and value. Native Gemma4ForConditionalGeneration reload reported model.dtype=torch.bfloat16 with no missing, unexpected, mismatched, or error keys.
</p>
      </section>
    </article>
    <article id="gemma-4-26b-a4b-it-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="gemma-4-26b-a4b-it-megatron-to-hf-gpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model google/gemma-4-26B-A4B-it --hf-revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --megatron-path work/model-verification/gemma-4-26b-a4b-it/imported-megatron/iter_0000000 --hf-path work/model-verification/gemma-4-26b-a4b-it/hf-export-verified --torch-dtype bfloat16 --export-weight-dtype bfloat16 --distributed-save --tp 2 --pp 1 --ep 4 --etp 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Distributed TP2/PP1/EP4 export completed all 2,788 conversion mappings. The exported config preserved sliding-attention rope_theta 10000 and full-attention rope_theta 1000000. All 1,013 exported tensor entries were BF16 and matched the immutable source exactly in key, shape, dtype, and value. Native Gemma4ForConditionalGeneration reload reported model.dtype=torch.bfloat16 with no missing, unexpected, mismatched, or error keys.
</p>
      </section>
    </article>
    <article id="gemma-4-26b-a4b-it-pretrain-all" class="verification-model-detail" data-entry-detail="gemma-4-26b-a4b-it-pretrain-all" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain</h4>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>all</dd></div>
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
        <p>Megatron Bridge does not publish a pretraining recipe for this Gemma 4 conditional-generation variant; the current model package provides SFT and PEFT recipes only.
</p>
      </section>
    </article>
    <article id="gemma-4-26b-a4b-it-sft-h100" class="verification-model-detail" data-entry-detail="gemma-4-26b-a4b-it-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-11</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.282612</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.09127883</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>15,936.250 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>32.200 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,028.096 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>65.223 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>72.356 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 8 --recipe gemma4_vl_26b_sft_config --mode sft --dataset cord-v2 --pretrained_checkpoint work/model-verification/gemma-4-26b-a4b-it/imported-megatron/iter_0000000 --max_steps 10 --seq_length 4096 &#x27;dataset.source.load_kwargs={revision:&quot;7f0115a4b758a71d6473b8d085751692da2fef98&quot;}&#x27; validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/gemma-4-26b-a4b-it/sft-checkpoints --save_interval 10 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/gemma-4-26b-a4b-it/sft-post-setup.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 2026-08-11 the single-node TP4/PP1/EP8/ETP1 run loaded the immutable BF16 checkpoint and real CORD-v2 revision, then completed exactly 10 optimizer steps at MBS1 and GBS32. The persisted post-setup config confirmed a frozen vision encoder, trainable projection and language model, expandable CUDA allocator segments, and full-layer uniform recompute with one layer per recompute unit directly from the recipe defaults, while CUDA graphs and CPU offload remained disabled. Loss moved from 1.282612 to 0.09127883, final grad norm was 3.090, and all 10 steps reported zero skipped and zero NaN iterations. After iteration 2, logger-reported ranks returned to 62.134-62.145 GiB allocated, while cumulative peak allocated and reserved memory remained at or below 65.223 and 72.356 GiB. The run saved a complete iter_0000010 torch_dist checkpoint with eight rank shards, finalized metadata, run config, training state, and a latest-checkpoint tracker containing iteration 10.
</p>
      </section>
    </article>
    <article id="gemma-4-26b-a4b-it-sft-long-context-h100" class="verification-model-detail" data-entry-detail="gemma-4-26b-a4b-it-sft-long-context-h100" tabindex="-1">
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
        <p>No Gemma 4 VL long-context recipe or run was verified. A future workload must establish sequence-packing and context-parallel settings, complete at least 10 optimizer steps with finite loss and no skipped or NaN iterations, report all five required metrics, and save a reloadable final checkpoint.
</p>
      </section>
    </article>
    <article id="gemma-4-26b-a4b-it-peft-h100" class="verification-model-detail" data-entry-detail="gemma-4-26b-a4b-it-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-04</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.288328</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.1072377</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>14,163.250 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>65.010 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>2,313.593 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>44.252 GiB</dd>
          </div>
          <div>
            <dt>Peak reserved memory</dt>
            <dd>45.637 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 4 --recipe gemma4_vl_26b_peft_config --mode lora --dataset cord-v2 --pretrained_checkpoint work/model-verification/gemma-4-26b-a4b-it/imported-megatron/iter_0000000 --max_steps 10 --seq_length 4096 &#x27;dataset.source.load_kwargs={revision:&quot;7f0115a4b758a71d6473b8d085751692da2fef98&quot;}&#x27; validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/gemma-4-26b-a4b-it/peft-checkpoints --save_interval 10 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/gemma-4-26b-a4b-it/peft-post-setup.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 2026-08-04 the 4-GPU TP2/PP1/EP4/ETP1 LoRA run loaded the immutable BF16 checkpoint and real CORD-v2 revision, then completed exactly 10 optimizer steps at MBS1 and GBS32. The persisted post-setup config recorded rank-32 adapters with alpha 32 and zero dropout on linear_qkv, linear_proj, linear_fc1, and linear_fc2. Runtime transformation froze the base model and reported 22,272,000 trainable adapter parameters, 0.30% of the local model shard. Loss moved from 1.288328 to 0.1072377, final grad norm was 22.994, and every step reported zero skipped and zero NaN iterations. Across all four ranks, allocated memory returned to 14.582 GiB after each step, while cumulative peak allocated and reserved memory remained at or below 44.252 and 45.637 GiB. The run saved a complete iter_0000010 adapter checkpoint with four rank shards, finalized metadata, training state, and a latest-checkpoint tracker containing iteration 10.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
