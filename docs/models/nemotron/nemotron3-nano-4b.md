# Nemotron 3 Nano 4B
[Nemotron 3 Nano](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) is a large language model (LLM) trained from scratch by NVIDIA, and designed as a unified model for both reasoning and non-reasoning tasks. The model employs a hybrid Mixture-of-Experts (MoE) architecture, consisting of 23 Mamba-2 and MoE layers, along with 6 Attention layers. Each MoE layer includes 128 experts plus 1 shared expert, with 6 experts activated per token. The model has 3.5B active parameters and 30B parameters in total.

NeMo Megatron Bridge supports pretraining, full parameters finetuning, and LoRA finetuning this model. The finetuned model can be converted back to the 🤗 Hugging Face format for downstream evaluation.

```{important}
Run all commands from `/opt/Megatron-Bridge` (e.g. `docker run -w /opt/Megatron-Bridge ...`)
```

```{tip}
We use the following environment variables throughout this page
- `HF_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `MEGATRON_MODEL_PATH=/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (feel free to set your own path)
```

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-nemotron-3-nano-4b"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-nano-4b-hf-to-megatron-cpu" aria-controls="nemotron-3-nano-4b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-nano-4b-hf-to-megatron-gpu" aria-controls="nemotron-3-nano-4b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-nano-4b-megatron-to-hf-cpu" aria-controls="nemotron-3-nano-4b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-nano-4b-megatron-to-hf-gpu" aria-controls="nemotron-3-nano-4b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-nano-4b-pretrain-h100" aria-controls="nemotron-3-nano-4b-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-nano-4b-sft-h100" aria-controls="nemotron-3-nano-4b-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-nano-4b-sft-long-context-h100" aria-controls="nemotron-3-nano-4b-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-nano-4b-peft-h100" aria-controls="nemotron-3-nano-4b-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="nemotron-3-nano-4b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-4b-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-20</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --hf-revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f --megatron-path work/model-verification/nemotron-3-nano-4b/cpu-megatron --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully and creates iter_0000000. After the paired CPU export, all 263 tensors and 3,973,556,832 parameters match the pinned source exactly in name, shape, dtype, and value, with maximum difference zero.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-4b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-4b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-20</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 1 --tp 1 --hf-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --hf-revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f --megatron-path work/model-verification/nemotron-3-nano-4b/imported-megatron --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully and creates a reloadable iter_0000000. After the paired GPU export, all 263 tensors and 3,973,556,832 parameters match the pinned source exactly in name, shape, dtype, and value.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-4b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-4b-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-20</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --hf-revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f --megatron-path work/model-verification/nemotron-3-nano-4b/cpu-megatron --hf-path work/model-verification/nemotron-3-nano-4b/cpu-hf-export --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Strict export exits successfully; all 263 BF16 tensors match the pinned source bitwise, the 171-byte generation configuration is preserved byte-for-byte, and Transformers reloads the output natively as NemotronHForCausalLM.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-4b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-4b-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-20</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 1 --tp 1 --hf-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --hf-revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f --megatron-path work/model-verification/nemotron-3-nano-4b/imported-megatron --hf-path work/model-verification/nemotron-3-nano-4b/hf-export --torch-dtype bfloat16</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Strict export exits successfully; all 263 BF16 tensors match the pinned source bitwise, the generation configuration is exact, and Transformers reloads the output natively as NemotronHForCausalLM. A_log, D, dt_bias, and out_proj are exact for all 21 Mamba layers.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-4b-pretrain-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-4b-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-20</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.4325</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>5.511951</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>26,775.110 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>402.510 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>19,581.171 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 1 --gpus-per-node 8 --recipe nemotron_3_nano_4b_pretrain_8gpu_h100_bf16_config --mode pretrain --dataset megatron-indexed --seq_length 4096 --max_steps 100 --lr 3e-4 --min_lr 3e-5 --warmup_iters 40 &#x27;dataset.blend=[[&quot;work/data/rp2/head_01&quot;],null]&#x27; dataset.path_to_cache=work/cache/nemotron-3-nano-4b/rp2 tokenizer.tokenizer_type=SentencePieceTokenizer tokenizer.tokenizer_model=work/data/rp2/tokenizer.model rng.seed=1234 dataset.random_seed=1234 scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 dist.distributed_timeout_minutes=30 ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.load=null --save_dir work/model-verification/nemotron-3-nano-4b/pretrain-reference-checkpoints --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The uninterrupted bounded random-initialization RP2 reference finishes all 100 steps at recipe-owned GBS/MBS 1024/1 with the five recorded metrics, finite loss, and no skipped or NaN iterations. It reaches peak learning rate 3e-4 at step 40 and completes cosine decay to 3e-5 at step 100. Both iter_0000050 and iter_0000100 contain all eight distributed shards plus metadata, optimizer/RNG train state, run config, and tokenizer; iter_0000100 reloads at step 100 without an extra optimizer step.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-4b-sft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-4b-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-20</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.839282</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.234906</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>627.000 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>262.890 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>13,065.391 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 1 --gpus-per-node 8 --recipe nemotron_3_nano_4b_sft_8gpu_h100_bf16_config --mode sft --dataset tulu3 --pretrained_checkpoint work/model-verification/nemotron-3-nano-4b/imported-megatron/iter_0000000 --max_steps 100 --seq_length 2048 --lr 5e-6 --min_lr 0 --warmup_iters 10 &#x27;dataset.hf_dataset.split=&quot;train[:10000]&quot;&#x27; &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;b14afda60f1bbebe55d5d2fa1e4df5042f97f8be&quot;}&#x27; dataset.hf_output_root=work/data/tulu3/nemotron-3-nano-4b-sft-b14afda60f1b dataset.hf_rewrite=true dataset.seed=1234 rng.seed=5678 dataset.do_validation=false dataset.hf_validation_proportion=null dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:2048,pad_seq_to_mult:1}&#x27; scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/nemotron-3-nano-4b/sft-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Pad-1 offline packing is 99.27% efficient. The TP1/DP8 run uses four microbatches per optimizer step and completes all 100 steps with the four recorded metrics, finite loss, and no skipped or NaN iterations. Across 6,553,600 token slots, assistant-only loss masks contain 4,187,630 supervised tokens. The complete eight-shard iter_0000100 full-model checkpoint is saved and reloads for export.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-4b-sft-long-context-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-4b-sft-long-context-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-20</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.65866</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.100197</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>5,575.590 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>142.060 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>5,877.046 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 1 --gpus-per-node 8 --recipe nemotron_3_nano_4b_sft_8gpu_h100_bf16_32k_config --mode sft --dataset tulu3 --pretrained_checkpoint work/model-verification/nemotron-3-nano-4b/imported-megatron/iter_0000000 --max_steps 100 --seq_length 32768 --tensor_model_parallel_size 2 --context_parallel_size 2 --lr 5e-6 --min_lr 0 --warmup_iters 10 &#x27;dataset.hf_dataset.split=&quot;train[:10000]&quot;&#x27; &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;b14afda60f1bbebe55d5d2fa1e4df5042f97f8be&quot;}&#x27; dataset.hf_output_root=work/data/tulu3/nemotron-3-nano-4b-long-b14afda60f1b dataset.hf_rewrite=true dataset.seed=1234 rng.seed=5678 dataset.do_validation=false dataset.hf_validation_proportion=null dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:32768,pad_seq_to_mult:4}&#x27; model.sequence_parallel=true model.cp_comm_type=a2a model.cross_entropy_loss_fusion=false model.calculate_per_token_loss=true ddp.average_in_collective=false scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/nemotron-3-nano-4b/long-sft-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Pad-4 offline packing is 99.35% efficient. The TP2/CP2/SP-on/DP2 run uses four microbatches per optimizer step and completes all 100 32K steps with the five recorded metrics, finite loss, and no skipped or NaN iterations. Across 26,214,400 token slots, assistant-only loss masks contain 16,734,347 supervised tokens. The complete eight-shard iter_0000100 checkpoint is saved and reloads at the same topology.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-4b-peft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-4b-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-20</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.769892</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.256916</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>381.690 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>431.900 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>21,462.443 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --wait --nodes 1 --gpus-per-node 8 --recipe nemotron_3_nano_4b_peft_8gpu_h100_bf16_config --mode lora --dataset tulu3 --pretrained_checkpoint work/model-verification/nemotron-3-nano-4b/imported-megatron/iter_0000000 --max_steps 100 --seq_length 2048 --lr 1e-4 --min_lr 0 --warmup_iters 10 &#x27;dataset.hf_dataset.split=&quot;train[:10000]&quot;&#x27; &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;b14afda60f1bbebe55d5d2fa1e4df5042f97f8be&quot;}&#x27; dataset.hf_output_root=work/data/tulu3/nemotron-3-nano-4b-peft-b14afda60f1b dataset.hf_rewrite=true dataset.seed=1234 rng.seed=5678 dataset.do_validation=false dataset.hf_validation_proportion=null dataset.enable_offline_packing=true &#x27;dataset.offline_packing_specs={packed_sequence_size:2048,pad_seq_to_mult:4}&#x27; scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/nemotron-3-nano-4b/peft-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Pad-4 offline packing is 99.36% efficient. The frozen-base LoRA run uses rank 8, alpha 16, zero dropout, and linear_qkv/linear_proj targets. Its TP1/DP8 layout uses four microbatches per optimizer step and completes all 100 steps with finite loss and no skipped or NaN iterations. Across 6,553,600 token slots, assistant-only loss masks contain 4,178,503 supervised tokens. The complete eight-shard iter_0000100 adapter checkpoint reloads over the pinned base model at step 100.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
