# Nemotron-3 Nano Omni

[NVIDIA Nemotron-3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
is a 30B-A3B MoE multimodal reasoning model that jointly processes image,
video, audio, and text inputs. It pairs a MoE Mamba/attention hybrid
language backbone with a RADIO
vision tower (static-resolution image path or dynamic-resolution
temporal video embedder) and a Parakeet sound encoder.

NeMo Megatron Bridge supports HF↔Megatron conversion, full SFT, and LoRA
PEFT on image-text and audio-video-text datasets. The finetuned model can
be re-exported to 🤗 Hugging Face format for downstream evaluation or
deployment.

```{important}
**Day-0 release.** This model is supported on dedicated public branches:

| Repo | Branch |
|---|---|
| Megatron-Bridge | [`nemotron_3_omni`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/nemotron_3_omni) |
| Megatron-LM (submodule at `3rdparty/Megatron-LM`) | [`nemotron_3_omni`](https://github.com/NVIDIA/Megatron-LM/tree/nemotron_3_omni) |
```

For the full setup, conversion, inference, training, evaluation, and LoRA
merge / adapter export workflows, see
[`examples/models/nemotron/nemotron_3_omni/README.md`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/nemotron/nemotron_3_omni/README.md).

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-nemotron-3-nano-omni-30b-a3b-reasoning"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-nano-omni-30b-a3b-reasoning-hf-to-megatron-cpu" aria-controls="nemotron-3-nano-omni-30b-a3b-reasoning-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="nemotron-3-nano-omni-30b-a3b-reasoning-hf-to-megatron-gpu" aria-controls="nemotron-3-nano-omni-30b-a3b-reasoning-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="nemotron-3-nano-omni-30b-a3b-reasoning-megatron-to-hf-cpu" aria-controls="nemotron-3-nano-omni-30b-a3b-reasoning-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="nemotron-3-nano-omni-30b-a3b-reasoning-megatron-to-hf-gpu" aria-controls="nemotron-3-nano-omni-30b-a3b-reasoning-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="" data-hardware="" data-status="unsupported" data-entry="nemotron-3-nano-omni-30b-a3b-reasoning-pretrain-all" aria-controls="nemotron-3-nano-omni-30b-a3b-reasoning-pretrain-all" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain</strong>
        <span class="verification-status verification-status--unsupported" title="Unsupported">× Unsupported</span>
      </span>
      <span class="verification-combination-meta">Precision not specified</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-nano-omni-30b-a3b-reasoning-sft-h100" aria-controls="nemotron-3-nano-omni-30b-a3b-reasoning-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="nemotron-3-nano-omni-30b-a3b-reasoning-sft-long-context-h100" aria-controls="nemotron-3-nano-omni-30b-a3b-reasoning-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="nemotron-3-nano-omni-30b-a3b-reasoning-peft-h100" aria-controls="nemotron-3-nano-omni-30b-a3b-reasoning-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="nemotron-3-nano-omni-30b-a3b-reasoning-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-omni-30b-a3b-reasoning-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --hf-revision 24e67ea000b7c2837fc8f9488aa2008524fac8ba --megatron-path work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/cpu-megatron-clean --torch-dtype bfloat16 --tp 1 --pp 1 --ep 1 --etp 1 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The offline, immutable-revision CPU import exits successfully after mapping 7,333 model parameters and creates a reloadable iter_0000000 torch_dist checkpoint with 33,015,546,816 parameters on the single model parallel rank.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-omni-30b-a3b-reasoning-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-omni-30b-a3b-reasoning-hf-to-megatron-gpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --hf-revision 24e67ea000b7c2837fc8f9488aa2008524fac8ba --megatron-path work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/gpu-megatron-clean --torch-dtype bfloat16 --tp 2 --pp 1 --ep 4 --etp 1 --trust-remote-code --low-memory-save</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully at TP2/PP1/EP4/ETP1 and creates a reloadable iter_0000000 checkpoint. The paired strict GPU export contains the source checkpoint&#x27;s exact 7,349-key set, shapes, dtypes, and values.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-omni-30b-a3b-reasoning-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-omni-30b-a3b-reasoning-megatron-to-hf-cpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --hf-revision 24e67ea000b7c2837fc8f9488aa2008524fac8ba --megatron-path work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/cpu-megatron-clean/iter_0000000 --hf-path work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/cpu-hf-export-clean --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU export completes in 14 indexed shards. Its exact comparison contains all 7,349 source tensors (7,300 BF16, 24 int64, and 25 float32), with identical keys, shapes, dtypes, and values and maximum difference zero. The item remains unverified because the Transformers 5.8.0 local custom-code loader omits transitive configuration modules before from_pretrained can reload the otherwise bitwise-identical export.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-omni-30b-a3b-reasoning-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="nemotron-3-nano-omni-30b-a3b-reasoning-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --hf-revision 24e67ea000b7c2837fc8f9488aa2008524fac8ba --megatron-path work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/gpu-megatron-clean/iter_0000000 --hf-path work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/gpu-hf-export-clean --torch-dtype bfloat16 --tp 2 --pp 1 --ep 4 --etp 1 --trust-remote-code --distributed-save --save-every-n-ranks 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Strict export completed in 17 indexed shards. All 7,349 tensors match the immutable HF source in keys, shapes, dtypes, and values, with maximum difference zero. The item remains unverified because Transformers 5.8.0 local custom-code loading omits the transitive configuration_nemotron_h and configuration_radio modules from the model cache, preventing a native from_pretrained reload even though those files exist in the export.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-omni-30b-a3b-reasoning-pretrain-all" class="verification-model-detail" data-entry-detail="nemotron-3-nano-omni-30b-a3b-reasoning-pretrain-all" tabindex="-1">
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
        <p>Megatron Bridge does not publish a pretraining recipe for this multimodal conditional-generation model; the supported package provides supervised CORD v2 SFT and LoRA PEFT workflows.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-omni-30b-a3b-reasoning-sft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-omni-30b-a3b-reasoning-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-30</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.123339</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.4893276</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>24,454.110 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>60.730 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>669.990 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe nemotron_omni_cord_v2_sft_4gpu_h100_bf16_config --mode sft --step_func nemotron_omni_step --pretrained_checkpoint work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/gpu-megatron-clean/iter_0000000 --max_steps 10 --tensor_model_parallel_size 2 --pipeline_model_parallel_size 2 --expert_model_parallel_size 4 --expert_tensor_parallel_size 1 --save_dir work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/sft-checkpoints-clean --save_interval 10 &#x27;dataset.source.load_kwargs={revision:&quot;7f0115a4b758a71d6473b8d085751692da2fef98&quot;}&#x27; dataset.do_validation=false dataset.do_test=false validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null logger.log_interval=1 logger.log_throughput=true rng.seed=5678</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The immutable-revision CORD v2 run completes exactly 10 full-SFT optimizer steps on 16 H100 GPUs at TP2/PP2/CP1/EP4/ETP1, GBS/MBS 64/1. LM loss is finite from 1.123339 to 0.4893276, all ten recorded steps average 24,454.11 ms and 60.73 TFLOP/s/GPU including first-step compilation, no iteration is skipped or NaN, and a complete iter_0000010 checkpoint is saved.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-omni-30b-a3b-reasoning-sft-long-context-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-omni-30b-a3b-reasoning-sft-long-context-h100" tabindex="-1">
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
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 8 --recipe nemotron_omni_cord_v2_long_context_sft_8gpu_h100_bf16_config --mode sft --step_func nemotron_omni_step --pretrained_checkpoint work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/gpu-megatron-clean/iter_0000000 --max_steps 10 --save_dir work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/long-context-checkpoints-clean --save_interval 10 &#x27;dataset.source.load_kwargs={revision:&quot;7f0115a4b758a71d6473b8d085751692da2fef98&quot;}&#x27; dataset.do_validation=false dataset.do_test=false validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null logger.log_interval=1 logger.log_throughput=true rng.seed=5678</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 8K TP4/PP1/CP2/EP1/ETP4, MBS2 in-batch-packing run uses precision-aware Adam with FP16 main parameters and stored FP32 remainders, BF16 gradients, and BF16 moments. Step 1 completes with finite LM loss 1.142561 in 170,179.0 ms at 8.6 TFLOP/s/GPU with no skipped or NaN iteration, but step 2 encounters rank-divergent H100 memory exhaustion and does not produce the required 10-step checkpoint.
</p>
      </section>
    </article>
    <article id="nemotron-3-nano-omni-30b-a3b-reasoning-peft-h100" class="verification-model-detail" data-entry-detail="nemotron-3-nano-omni-30b-a3b-reasoning-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-30</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.098394</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.3166811</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>41,897.410 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>22.220 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>782.101 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 8 --recipe nemotron_omni_cord_v2_peft_4gpu_h100_bf16_config --mode lora --step_func nemotron_omni_step --pretrained_checkpoint work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/gpu-megatron-clean/iter_0000000 --max_steps 10 --save_dir work/model-verification/nemotron-3-nano-omni-30b-a3b-reasoning/peft-checkpoints-clean --save_interval 10 &#x27;dataset.source.load_kwargs={revision:&quot;7f0115a4b758a71d6473b8d085751692da2fef98&quot;}&#x27; dataset.do_validation=false dataset.do_test=false validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null logger.log_interval=1 logger.log_throughput=true rng.seed=5678</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The ten-step TP4/PP1/CP1/EP1 LoRA run exits successfully and saves a complete eight-shard iter_0000010 adapter checkpoint. LM loss is finite from 1.098394 to 0.3166811, all ten steps average 41,897.41 ms and 22.22 TFLOP/s/GPU including first-step compilation, and no iteration is skipped or NaN.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
