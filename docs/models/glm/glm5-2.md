# GLM-5.2

[GLM-5](https://huggingface.co/zai-org/GLM-5), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1), and [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) are large sparse MoE language models with Multi-Latent Attention and Dynamic Sparse Attention. Megatron Bridge supports these checkpoints through the shared `GLM5Bridge`.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-glm5-2"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="glm5-2-hf-to-megatron-cpu" aria-controls="glm5-2-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="glm5-2-hf-to-megatron-gpu" aria-controls="glm5-2-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="glm5-2-megatron-to-hf-cpu" aria-controls="glm5-2-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="glm5-2-megatron-to-hf-gpu" aria-controls="glm5-2-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="glm5-2-pretrain-h100" aria-controls="glm5-2-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="glm5-2-pretrain-gb200" aria-controls="glm5-2-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="glm5-2-sft-h100" aria-controls="glm5-2-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="glm5-2-sft-gb200" aria-controls="glm5-2-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="glm5-2-sft-long-context-h100" aria-controls="glm5-2-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="glm5-2-sft-long-context-gb200" aria-controls="glm5-2-sft-long-context-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="glm5-2-peft-h100" aria-controls="glm5-2-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="glm5-2-peft-gb200" aria-controls="glm5-2-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="glm5-2-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="glm5-2-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-22</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model zai-org/GLM-5.2 --hf-revision 4d67f66cc64d3219133b767c253b2ad1425c6c88 --megatron-path work/model-verification/glm5-2/cpu-megatron --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Import exits successfully and creates a reloadable CPU checkpoint. Its strict round-trip comparison matches all 59,585 BF16 tensors and 753,329,940,480 elements bitwise against the immutable HF revision.
</p>
      </section>
    </article>
    <article id="glm5-2-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="glm5-2-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-22</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 13 --gpus-per-node 8 --hf-model zai-org/GLM-5.2 --hf-revision 4d67f66cc64d3219133b767c253b2ad1425c6c88 --megatron-path work/model-verification/glm5-2/gpu-megatron --torch-dtype bfloat16 --tp 1 --pp 13 --ep 8 --etp 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Import exits successfully at TP1/PP13/EP8/ETP1 and creates a reloadable iter_0000000 checkpoint. Its strict round-trip comparison matches all 59,585 BF16 tensors and 753,329,940,480 elements bitwise against the immutable HF revision.
</p>
      </section>
    </article>
    <article id="glm5-2-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="glm5-2-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-22</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model zai-org/GLM-5.2 --hf-revision 4d67f66cc64d3219133b767c253b2ad1425c6c88 --megatron-path work/model-verification/glm5-2/cpu-megatron --hf-path work/model-verification/glm5-2/cpu-hf-export --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Export exits successfully; all 59,585 BF16 tensors and 753,329,940,480 elements match the immutable HF revision bitwise. Transformers 5.12.1 strictly reloads the output as GlmMoeDsaForCausalLM with 743,377,000,704 parameters.
</p>
      </section>
    </article>
    <article id="glm5-2-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="glm5-2-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-22</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 13 --gpus-per-node 8 --hf-model zai-org/GLM-5.2 --hf-revision 4d67f66cc64d3219133b767c253b2ad1425c6c88 --megatron-path work/model-verification/glm5-2/gpu-megatron --hf-path work/model-verification/glm5-2/gpu-hf-export-exact --torch-dtype bfloat16 --distributed-save --tp 1 --pp 13 --ep 8 --etp 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Distributed export exits successfully; all 59,585 BF16 tensors and 753,329,940,480 elements match the immutable HF revision bitwise. Transformers strictly reloads the output as GlmMoeDsaForCausalLM with 743,377,000,704 parameters.
</p>
      </section>
    </article>
    <article id="glm5-2-pretrain-h100" class="verification-model-detail" data-entry-detail="glm5-2-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-23</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>13.15907</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>6.385243</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>31,204.920 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>54.595 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>190.926 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 44 --gpus-per-node 8 --recipe glm52_pretrain_416gpu_h100_bf16_config --mode pretrain --dataset megatron-indexed --deterministic --seq_length 2048 --max_steps 100 --lr 3e-4 --min_lr 3e-5 --warmup_iters 40 &#x27;dataset.blend=[[&quot;work/data/wikitext103/wikitext103_glm52_text_document&quot;],null]&#x27; dataset.path_to_cache=work/cache/glm5-2/wikitext-pretrain dataset.random_seed=1234 rng.seed=1234 scheduler.lr_decay_iters=100 model.pipeline_model_parallel_size=11 model.virtual_pipeline_model_parallel_size=null model.microbatch_group_size_per_vp_stage=null model.pipeline_model_parallel_layout=&#x27;Etttttt|tttttttt|tttttttt|tttttttt|tttttttt|tttttttt|tttttttt|tttttttt|tttttttt|tttt|ttttmL&#x27; --save_dir work/model-verification/glm5-2/pretrain-reference/checkpoints --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The uninterrupted 352-H100 TP1/PP11/CP1/EP32/ETP1 run completed 100 real-data steps at GBS/MBS 1024/1 with cuDNN DSA, sparse indexer loss coefficient 0.001, MTP1, and flex/DeepEP dispatch. Loss remained finite from 13.15907 to 6.385243, final grad norm was 0.796, and every step recorded zero skipped and NaN iterations. The final 10 steps averaged 31,204.920 ms and 54.595 TFLOPS/GPU; peak observed GPU memory was 80,999 MiB. The matching step-50 and step-100 checkpoints contain model, optimizer, scheduler, data-order, and RNG state; the step-50 checkpoint reloads successfully.
</p>
      </section>
    </article>
    <article id="glm5-2-pretrain-gb200" class="verification-model-detail" data-entry-detail="glm5-2-pretrain-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB200</h4>
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
            <dd>13.16815</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>6.523522</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>64,786.540 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>102.520 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>337.189 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 48 --gpus-per-node 4 --recipe glm52_pretrain_192gpu_gb200_bf16_config --mode pretrain --dataset megatron-indexed --deterministic --seq_length 4096 --max_steps 100 --lr 3e-4 --min_lr 3e-5 --warmup_iters 40 &#x27;dataset.blend=[[&quot;work/data/rp2/head_01_text_document&quot;],null]&#x27; dataset.path_to_cache=work/cache/glm5-2/rp2-head-01-gb200 dataset.random_seed=1234 dataset.num_workers=8 tokenizer.tokenizer_type=SentencePieceTokenizer tokenizer.tokenizer_model=work/data/rp2/tokenizer/tokenizer.model tokenizer.use_tokenizer_vocab_size=false rng.seed=1234 scheduler.lr_decay_iters=100 model.moe_router_force_load_balancing=false checkpoint.load=null checkpoint.pretrained_checkpoint=null checkpoint.save_optim=true checkpoint.save_rng=true checkpoint.load_optim=true checkpoint.load_rng=true checkpoint.finetune=false --save_dir work/model-verification/glm5-2/pretrain-rp2-gb200/checkpoints --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 192-GB200 TP1/PP6/CP1/EP32/ETP1 run completed exactly 100 RP2 head_01 pretraining steps at GBS/MBS 1024/1 with cuDNN DSA, sparse indexer loss coefficient 0.001, MTP1, and all-to-all expert dispatch. The RP2 Llama SentencePiece tokenizer is used while retaining GLM-5.2&#x27;s configured vocabulary size, so this is support verification rather than a cross-tokenizer convergence claim. Loss remained finite from 13.168150 to 6.523522, final grad norm was 2.775, and every step recorded zero skipped and NaN iterations. The final 10 steps averaged 64,786.540 ms and 102.520 TFLOPS/GPU. Step-50 and step-100 checkpoints contain model, optimizer, scheduler, data-order, and RNG state.
</p>
      </section>
    </article>
    <article id="glm5-2-sft-h100" class="verification-model-detail" data-entry-detail="glm5-2-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-22</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.909875</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.6634203</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>4,436.290 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>10.154 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>35.511 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 52 --gpus-per-node 8 --recipe glm52_sft_416gpu_h100_bf16_config --mode sft --pretrained_checkpoint work/model-verification/glm5-2/gpu-megatron/iter_0000000 --max_steps 100 --seq_length 2048 dataset.hf_output_root=work/data/glm5-2/tulu3-full-sft-pad32 model.context_parallel_size=2 model.virtual_pipeline_model_parallel_size=null model.pipeline_model_parallel_layout=&#x27;Etttttt|tttt|tttttttt|tttttttt|tttttttt|tttttttt|tttttttt|tttttttt|tttt|tttt|tttt|tttt|ttttmL&#x27; --save_dir work/model-verification/glm5-2/sft-functional/checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The immutable-revision 416-H100 run completes exactly 100 full-SFT steps at TP1/PP13/CP2/EP32/ETP1, GBS/MBS 32/1, with cuDNN DSA, sparse indexer loss coefficient 0.001, MTP1, and natural routing. Pad-32 packing is 99.60% efficient and contains 4,097,215 supervised tokens. Loss remains finite from 1.909875 to 0.6634203 with zero skipped or NaN iterations; the final grad norm is 0.895 and the step-100 checkpoint reloads for export.
</p>
      </section>
    </article>
    <article id="glm5-2-sft-gb200" class="verification-model-detail" data-entry-detail="glm5-2-sft-gb200" tabindex="-1">
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
            <dd>1.704979</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.3682597</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>8,593.990 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>12.446 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>39.718 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 48 --gpus-per-node 4 --recipe glm52_sft_192gpu_gb200_bf16_config --mode sft --pretrained_checkpoint work/model-verification/glm5-2/hf-4d67f66cc64d3219133b767c253b2ad1425c6c88 --max_steps 100 tokenizer.chat_template_path=work/templates/glm52_chat_template_generation.jinja --save_dir work/model-verification/glm5-2/sft-gb200-cudnn/checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The immutable-revision 192-GB200 run completed exactly 100 full-SFT steps at TP1/PP6/CP4/EP32/ETP1, DP8, and GBS/MBS 8/1 without gradient accumulation. It used the pinned Tulu 3 train[:10000] split with the GLM assistant-only chat template, 8,192-token offline packs, pad-8 sequence alignment, an 8,184-token single-sequence cap, constant-shape cu_seqlens, cuDNN DSA, sparse indexer loss coefficient 0.001, MTP1, natural routing, and HybridEP. Packing is 99.72% efficient across 421 rows with a 23.75 packing factor. The 100-step window contains 6,553,600 token slots and 4,283,561 supervised tokens. Metadata and Parquet SHA-256 values are cbd964c686b74f82804ceea2df75bb00fbe28648f0458a641b9db27977a55b1b and b2c6f71a6b5add331750956622073cc8ced0bb6b2742e1d7a0e1a9f763cbaa0c. Loss remained finite from 1.704979 to 0.3682597, final grad norm was 0.873, and every step recorded zero skipped and NaN iterations. The final 10 steps averaged 8,593.990 ms and 12.446 TFLOPS/GPU. A complete step-100 distributed checkpoint was written for the future GB200 export/reload gate.
</p>
      </section>
    </article>
    <article id="glm5-2-sft-long-context-h100" class="verification-model-detail" data-entry-detail="glm5-2-sft-long-context-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-22</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>8.283127</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>2.557671</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>36,603.370 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>39.648 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>116.828 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 76 --gpus-per-node 8 --recipe glm52_sft_608gpu_h100_bf16_200k_config --mode sft --pretrained_checkpoint work/model-verification/glm5-2/gpu-megatron/iter_0000000 --max_steps 20 --seq_length 200000 dataset.dataset_root=work/data/glm5-2/synthetic-200k-260 train.empty_unused_memory_level=2 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 608-H100 TP1/PP19/CP32/EP32/ETP1 run completed exactly 20 optimizer steps over 200,000-token packed rows with cuDNN DSA, sparse indexer loss coefficient 0.001, and MTP1. Loss remained finite from 8.283127 to 2.557671, final grad norm was 10838.703, and every step recorded zero skipped and NaN iterations. The last 10 steps averaged 36,603.370 ms and 39.648 TFLOPS/GPU; peak observed GPU memory was 81,005 MiB.
</p>
      </section>
    </article>
    <article id="glm5-2-sft-long-context-gb200" class="verification-model-detail" data-entry-detail="glm5-2-sft-long-context-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-05</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>0.4159648</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.000511673</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>143,248.820 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>88.848 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>266.874 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 48 --gpus-per-node 4 --recipe glm52_sft_192gpu_gb200_bf16_128k_config --mode sft --pretrained_checkpoint work/model-verification/glm5-2/hf-4d67f66cc64d3219133b767c253b2ad1425c6c88 --max_steps 20 tokenizer.chat_template_path=work/templates/glm52_chat_template_generation.jinja dataset.dataset_root=work/data/glm5-2/synthetic-long-sft-128k dataset.max_train_samples=1120 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The immutable-revision 192-GB200 run completed exactly 20 optimizer steps at TP1/PP6/CP32/EP32/ETP1, DP1, and GBS/MBS 56/1 with cuDNN DSA, MTP1, HybridEP, and full activation recompute. The explicit PP6 decoder layout [14, 16, 12, 12, 12, 12] completed without OOM using 32 single-sequence, 131,072-token synthetic packs. Loss remained finite from 0.4159648 to 0.000511673 with zero skipped or NaN iterations across 146,786,080 supervised tokens. The final 10 steps averaged 143,248.820 ms and 88.848 TFLOPS/GPU. This verifies long-context execution only; the 8K pinned-Tulu run provides real-data convergence evidence.
</p>
      </section>
    </article>
    <article id="glm5-2-peft-h100" class="verification-model-detail" data-entry-detail="glm5-2-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-22</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.909421</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.8835775</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>4,638.550 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>19.423 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>67.926 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 26 --gpus-per-node 8 --recipe glm52_peft_208gpu_h100_bf16_config --mode lora --pretrained_checkpoint work/model-verification/glm5-2/gpu-megatron/iter_0000000 --max_steps 100 --seq_length 2048 dataset.hf_output_root=work/data/glm5-2/tulu3-peft-pad4 --save_dir work/model-verification/glm5-2/peft-lora/checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The immutable-revision 208-H100 run completes exactly 100 LoRA steps at TP1/PP13/CP1/EP16/ETP1, GBS/MBS 32/1, with cuDNN DSA, sparse indexer loss coefficient 0.001, MTP1, and natural routing. Pad-4 packing is 99.45% efficient and contains 4,251,129 supervised tokens. Loss remains finite from 1.909421 to 0.8835775 with zero skipped or NaN iterations. The final grad norm is 0.101; the checkpoint contains exactly 790 BF16 rank-8 adapter tensors and no base-model weights.
</p>
      </section>
    </article>
    <article id="glm5-2-peft-gb200" class="verification-model-detail" data-entry-detail="glm5-2-peft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-24</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.8311</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.8957523</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>68,588.270 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>1.423 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>4.977 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 48 --gpus-per-node 4 --recipe glm52_peft_192gpu_gb200_bf16_config --mode lora --pretrained_checkpoint work/model-verification/glm5-2/gpu-megatron/iter_0000000 --max_steps 100 tokenizer.chat_template_path=work/templates/glm52_chat_template_generation.jinja --save_dir work/model-verification/glm5-2/peft-gb200-cudnn/checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The immutable-revision 192-GB200 run completed exactly 100 LoRA steps at TP1/PP6/CP1/EP32/ETP1, GBS/MBS 32/1, with offline-packed Tulu 3 data, cuDNN DSA, sparse indexer loss coefficient 0.001, MTP1, and all-to-all expert dispatch. Loss remained finite from 1.831100 to 0.8957523, final grad norm was 0.101, and every step recorded zero skipped and NaN iterations. The step-100 adapter checkpoint was written for reload validation.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
