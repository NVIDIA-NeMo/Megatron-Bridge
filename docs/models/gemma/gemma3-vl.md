# Gemma 3 VL (Vision-Language)

[Google's Gemma 3 VL](https://huggingface.co/collections/google/gemma-3-release) is a family of vision-language models built on the same research and technology used to create Gemini models. The Gemma 3 VL architecture combines the text-generation capabilities of Gemma 3 with a SigLIP vision encoder for robust visual understanding.

Gemma 3 VL models support multimodal tasks including image captioning, visual question answering, OCR, and general vision-language understanding.

Gemma family models are supported via the Bridge system with auto-detected configuration and weight mapping.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-gemma-3-4b-it"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="gemma-3-4b-it-hf-to-megatron-cpu" aria-controls="gemma-3-4b-it-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="gemma-3-4b-it-hf-to-megatron-gpu" aria-controls="gemma-3-4b-it-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="gemma-3-4b-it-megatron-to-hf-cpu" aria-controls="gemma-3-4b-it-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="gemma-3-4b-it-megatron-to-hf-gpu" aria-controls="gemma-3-4b-it-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="" data-hardware="" data-status="unsupported" data-entry="gemma-3-4b-it-pretrain-all" aria-controls="gemma-3-4b-it-pretrain-all" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain</strong>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </span>
      <span class="verification-combination-meta">Precision not specified</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="gemma-3-4b-it-sft-h100" aria-controls="gemma-3-4b-it-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="" data-hardware="H100" data-status="unsupported" data-entry="gemma-3-4b-it-sft-long-context-h100" aria-controls="gemma-3-4b-it-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </span>
      <span class="verification-combination-meta">Precision not specified</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="gemma-3-4b-it-peft-h100" aria-controls="gemma-3-4b-it-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="gemma-3-4b-it-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="gemma-3-4b-it-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-24</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model google/gemma-3-4b-it --hf-revision 093f9f388b31de276ce2de164bdc2081324b9767 --megatron-path work/model-verification/gemma-3-4b-it/cpu-megatron --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Exact-revision BF16 CPU import loaded all 781 Bridge tensors and saved one reloadable DCP shard. An independent Gloo reload mapped all 883 source tensors with exact keys, shapes, dtypes, and torch.equal values, including all 437 legacy vision-tower keys.
</p>
      </section>
    </article>
    <article id="gemma-3-4b-it-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="gemma-3-4b-it-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-30</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --hf-model google/gemma-3-4b-it --hf-revision 093f9f388b31de276ce2de164bdc2081324b9767 --megatron-path work/model-verification/gemma-3-4b-it/imported-megatron --torch-dtype bfloat16 --tp 2 --pp 2 --ep 1 --etp 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Exact-revision distributed BF16 import completed with TP2/PP2/EP1/ETP1. All 781 Bridge tensors loaded and all four ranks saved a reloadable checkpoint with the vision tower and multimodal projector on the owning pipeline stage. Independent TP2/PP2 and Gloo reloads retained FusedAttention and matched all 883 mapped tensors exactly.
</p>
      </section>
    </article>
    <article id="gemma-3-4b-it-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="gemma-3-4b-it-megatron-to-hf-cpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model google/gemma-3-4b-it --hf-revision 093f9f388b31de276ce2de164bdc2081324b9767 --megatron-path work/model-verification/gemma-3-4b-it/cpu-megatron/iter_0000000 --hf-path work/model-verification/gemma-3-4b-it/cpu-hf-export --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>A fresh exact-revision CPU import and export from Bridge commit 2d0175b116af6cb785bb8efab7ee053435496266 completed on 2026-07-24, but the source/export key gate reported 437 missing legacy vision_tower.vision_model.* keys and 437 extra flat vision_tower.* keys. This leaf remains unverified pending a source-namespace-preserving design; changing the general config-only export path would regress pruned and custom checkpoint exports.
</p>
      </section>
    </article>
    <article id="gemma-3-4b-it-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="gemma-3-4b-it-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-30</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --hf-model google/gemma-3-4b-it --hf-revision 093f9f388b31de276ce2de164bdc2081324b9767 --megatron-path work/model-verification/gemma-3-4b-it/imported-megatron/iter_0000000 --hf-path work/model-verification/gemma-3-4b-it/hf-export --torch-dtype bfloat16 --export-weight-dtype bfloat16 --distributed-save --tp 2 --pp 2 --ep 1 --etp 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>TP2/PP2 distributed export completed without a missing PP owner. Source and export contained the same 883 keys with exact shapes, dtypes, and torch.equal values, including all 437 legacy vision_tower.vision_model.* tensors. Native Gemma3ForConditionalGeneration strict reload reported no missing, unexpected, mismatched, or errored tensors.
</p>
      </section>
    </article>
    <article id="gemma-3-4b-it-pretrain-all" class="verification-model-detail" data-entry-detail="gemma-3-4b-it-pretrain-all" tabindex="-1">
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
        <p>Megatron Bridge does not provide a public Gemma 3 VL 4B pretraining recipe. The available exact-model recipes cover SFT and PEFT, so a supported pretraining command and checkpoint contract cannot be stated.
</p>
      </section>
    </article>
    <article id="gemma-3-4b-it-sft-h100" class="verification-model-detail" data-entry-detail="gemma-3-4b-it-sft-h100" tabindex="-1">
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
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 1 --recipe gemma3_vl_4b_sft_config --mode sft --dataset cord-v2 --pretrained_checkpoint work/model-verification/gemma-3-4b-it/imported-megatron/iter_0000000 --max_steps 10 --seq_length 4096 &#x27;dataset.source.load_kwargs={revision:&quot;7f0115a4b758a71d6473b8d085751692da2fef98&quot;}&#x27; validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/gemma-3-4b-it/sft-checkpoints --save_interval 10 logger.log_interval=1 logger.log_throughput=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Pending exact-revision CORD-v2 full-SFT verification with the public Gemma 3 VL 4B H100 recipe. The run must complete at least ten optimizer steps with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable final full-model checkpoint.
</p>
      </section>
    </article>
    <article id="gemma-3-4b-it-sft-long-context-h100" class="verification-model-detail" data-entry-detail="gemma-3-4b-it-sft-long-context-h100" tabindex="-1">
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
        <p>Gemma 3 VL image-bidirectional attention uses dense local/global additive biases with FusedAttention. The provider rejects context parallelism greater than one, and the public recipes disable in-batch THD packing, so there is no supported exact-model long-context SFT contract.
</p>
      </section>
    </article>
    <article id="gemma-3-4b-it-peft-h100" class="verification-model-detail" data-entry-detail="gemma-3-4b-it-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
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
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 1 --recipe gemma3_vl_4b_peft_config --mode lora --dataset cord-v2 --pretrained_checkpoint work/model-verification/gemma-3-4b-it/imported-megatron/iter_0000000 --max_steps 10 --seq_length 4096 &#x27;dataset.source.load_kwargs={revision:&quot;7f0115a4b758a71d6473b8d085751692da2fef98&quot;}&#x27; validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/gemma-3-4b-it/peft-checkpoints --save_interval 10 logger.log_interval=1 logger.log_throughput=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Pending exact-revision CORD-v2 LoRA verification with the public Gemma 3 VL 4B H100 recipe. The run must complete at least ten optimizer steps with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable adapter checkpoint.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
