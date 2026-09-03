# Qwen3.8-27B

[Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) is a family of vision-language models supporting multimodal understanding across text, images, and videos. Qwen3.5-VL includes both dense models and Mixture-of-Experts (MoE) variants for improved efficiency at scale.

[Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) shares the same architecture as Qwen3.5 VL MoE (`Qwen3_5MoeForConditionalGeneration`) and is supported through the same bridge implementation.

Qwen 3.5/3.6 models feature a hybrid architecture combining GDN (Gated DeltaNet) layers with standard attention layers, SwiGLU activations, and RMSNorm. MoE variants use top-k routing with shared experts for better quality.

Qwen 3.5/3.6 models are supported via Megatron Bridge with auto-detected configuration and weight mapping.

```{important}
Use `transformers` >= 5.2.0 for Qwen3.5 and >= 5.8.1 for Qwen3.6.
```

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-qwen3.8-27b"></a>
### Run a configuration

Choose a workflow, precision, and exact recorded combination. The command and expected result update below.

<div class="verification-model-explorer" data-model-explorer>
  <div class="verification-model-controls" hidden>
    <div class="verification-capability-tabs" role="tablist" aria-label="Workflow">
      <button type="button" role="tab" aria-selected="true" data-capability-tab="import-export">Import & Export</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="pretrain">Pretrain</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="benchmark">Benchmark</button>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-8-27b-hf-to-megatron-cpu" aria-controls="qwen3-8-27b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-8-27b-hf-to-megatron-gpu" aria-controls="qwen3-8-27b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-8-27b-megatron-to-hf-cpu" aria-controls="qwen3-8-27b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-8-27b-megatron-to-hf-gpu" aria-controls="qwen3-8-27b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="qwen3-8-27b-pretrain-gb200" aria-controls="qwen3-8-27b-pretrain-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="qwen3-8-27b-sft-gb200" aria-controls="qwen3-8-27b-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="qwen3-8-27b-sft-long-context-gb200" aria-controls="qwen3-8-27b-sft-long-context-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="qwen3-8-27b-peft-gb200" aria-controls="qwen3-8-27b-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="qwen3-8-27b-pretrain-performance-gb200" aria-controls="qwen3-8-27b-pretrain-performance-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="qwen3-8-27b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="qwen3-8-27b-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --gpus-per-node 1 --mem 512G --hf-model Qwen/Qwen3.8-27B --hf-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --megatron-path work/model-verification/qwen3.8-27b/cpu-megatron</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The pinned-revision CPU import completed in 3m11s and created a reloadable iter_0000000 containing 8 nonempty files totaling 55,586,366,001 bytes. The converter selected Qwen35VLBridge, completed all 956 mappings, kept model weights on CPU, and reported 27,781,427,952 parameters. One shared runtime GB200 was used for Transformer Engine vision-module construction.
</p>
      </section>
    </article>
    <article id="qwen3-8-27b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="qwen3-8-27b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --hf-model Qwen/Qwen3.8-27B --hf-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --megatron-path work/model-verification/qwen3.8-27b/gpu-megatron --tp 4</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The pinned-revision four-GB200 import completed in 2m01s at TP4/PP1 and created an 11-file iter_0000000 totaling 55,592,033,255 bytes, including four distributed checkpoint shards. All 956 conversion mappings completed, and subsequent export and two independent inference runs reloaded it.
</p>
      </section>
    </article>
    <article id="qwen3-8-27b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="qwen3-8-27b-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --gpus-per-node 1 --mem 512G --hf-model Qwen/Qwen3.8-27B --hf-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --megatron-path work/model-verification/qwen3.8-27b/cpu-megatron/iter_0000000 --hf-path work/model-verification/qwen3.8-27b/cpu-hf-export</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU export completed in 2m16s, wrote every source tensor, and produced 26 files totaling 55,583,128,197 bytes. The index preserves the original 1,199-key set and exact 18-shard placement with 55,562,855,904 BF16 tensor bytes. Transformers strictly reloaded the export as Qwen3_5ForConditionalGeneration with empty missing, unexpected, mismatched, and error sets, then completed processor-native GB200 inference. Export normalizes vision_config.model_type from qwen3_5 to qwen3_5_vision and updates producer-version metadata without changing the architecture.
</p>
      </section>
    </article>
    <article id="qwen3-8-27b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="qwen3-8-27b-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --hf-model Qwen/Qwen3.8-27B --hf-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --megatron-path work/model-verification/qwen3.8-27b/gpu-megatron/iter_0000000 --hf-path work/model-verification/qwen3.8-27b/gpu-hf-export --tp 4</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The four-GB200 distributed export completed in 1m55s, wrote every source tensor, and produced 26 files totaling 55,583,128,222 bytes. Its index has the same 1,199 keys, 18-shard placement, and 55,562,855,904 BF16 tensor bytes as the pinned source and CPU export. Transformers strictly reloaded it with empty missing, unexpected, mismatched, and error sets and completed processor-native inference. Relative to CPU export, config.json only adds the metadata field vision_config.dtype=bfloat16.
</p>
      </section>
    </article>
    <article id="qwen3-8-27b-pretrain-gb200" class="verification-model-detail" data-entry-detail="qwen3-8-27b-pretrain-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-20</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>6.858159</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>3.955657</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>41,391.140 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>33.240 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>197.917 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 4 --recipe qwen35_vl_27b_pretrain_16gpu_h100_bf16_mock_config --mode pretrain --deterministic --pretrained_checkpoint work/model-verification/qwen3.8-27b/gpu-megatron/iter_0000000 --max_steps 20 --warmup_iters 2 --save_dir work/model-verification/qwen3.8-27b/pretrain/checkpoints --save_interval 10 dataset.hf_processor_path=work/model-verification/qwen3.8-27b/gpu-hf-export tokenizer.tokenizer_model=work/model-verification/qwen3.8-27b/gpu-hf-export model.hf_model_id=Qwen/Qwen3.8-27B model.bos_token_id=248044 checkpoint.load=null validation.eval_iters=0 validation.eval_interval=0 ddp.check_for_large_grads=true logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.8-27b/pretrain/resolved-config.yaml scheduler.lr_decay_iters=20</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 16-GB200 MockVLM pretrain completed exactly 20 deterministic steps at TP4/PP4, sequence length 4096, and MBS2/GBS32. The language model and multimodal vision merger trained while the vision encoder remained frozen. Loss remained finite from 6.858159 to 3.955657 with finite gradients and zero skipped or NaN iterations. Steps 11-20 averaged 41,391.140 ms, 33.240 corrected model TFLOP/s/GPU, and 197.917 token slots/s/GPU. Checkpoints at iterations 10 and 20 each contain 20 files totaling 399,257,378,741 bytes, and the tracker records iteration 20. Both built-in 32-iteration validation and test passes completed at loss 3.907797. The process exited successfully in 26m31s.
</p>
      </section>
    </article>
    <article id="qwen3-8-27b-sft-gb200" class="verification-model-detail" data-entry-detail="qwen3-8-27b-sft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.116404</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.429567</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>24,160.490 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>58.010 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>339.066 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 4 --recipe qwen35_vl_27b_sft_16gpu_h100_bf16_config --mode sft --dataset medpix --deterministic --pretrained_checkpoint work/model-verification/qwen3.8-27b/gpu-megatron/iter_0000000 --max_steps 100 --warmup_iters 10 --save_interval 50 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=work/model-verification/qwen3.8-27b/gpu-hf-export dataset.do_validation=false dataset.pad_to_max_length=true tokenizer.tokenizer_model=work/model-verification/qwen3.8-27b/gpu-hf-export model.hf_model_id=Qwen/Qwen3.8-27B model.bos_token_id=248044 model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 model.recompute_modules=null scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 ddp.check_for_large_grads=true --save_dir work/model-verification/qwen3.8-27b/sft-recompute/checkpoints logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.8-27b/sft-recompute/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 16-GB200 run completed exactly 100 full-SFT MedPix steps at TP4/PP4, MBS4/GBS32, deterministic mode, and one-layer full-uniform activation recompute. Loss remained finite from 2.116404 to 1.429567 with zero skipped or NaN iterations. Steps 91-100 averaged 24,160.490 ms and 58.010 model TFLOP/s/GPU. Step-50 and step-100 saves each contain 20 files totaling 404,422,605,133 bytes, the tracker selects iteration 100, and the process exited successfully in 45m42s. The unchanged stock TP4/PP4 recipe without activation recompute reached the training loop but OOMed near the approximately 184.3-GiB per-GPU device limit; the successful command records the required config-only recompute overrides.
</p>
      </section>
    </article>
    <article id="qwen3-8-27b-sft-long-context-gb200" class="verification-model-detail" data-entry-detail="qwen3-8-27b-sft-long-context-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.343622</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.378064</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>8,164.790 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>362.530 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>2,006.665 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 4 --recipe qwen35_vl_27b_sft_16gpu_h100_bf16_config --mode sft --dataset medpix --pretrained_checkpoint work/model-verification/qwen3.8-27b/gpu-megatron/iter_0000000 --max_steps 20 --warmup_iters 2 --seq_length 8192 --pipeline_model_parallel_size 2 --context_parallel_size 2 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=work/model-verification/qwen3.8-27b/gpu-hf-export dataset.do_validation=false dataset.enable_in_batch_packing=true dataset.defer_in_batch_packing_to_step=true dataset.in_batch_packing_pad_to_multiple_of=4 tokenizer.tokenizer_model=work/model-verification/qwen3.8-27b/gpu-hf-export model.hf_model_id=Qwen/Qwen3.8-27B model.bos_token_id=248044 model.calculate_per_token_loss=true model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 ddp.average_in_collective=false scheduler.lr_decay_iters=20 checkpoint.load=null checkpoint.save=null validation.eval_iters=0 validation.eval_interval=0 ddp.check_for_large_grads=true logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.8-27b/sft-long/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 16-GB200 run completed exactly 20 full-SFT steps at TP4/PP2/CP2, MBS4/GBS32, 8192 tokens, deferred processor-native in-batch packing with pad-to-multiple-of-4, per-token loss, and one-layer full-uniform activation recompute. Loss remained finite from 2.343622 to 1.378064 with zero skipped or NaN iterations. Steps 11-20 averaged 8,164.790 ms and 362.530 model TFLOP/s/GPU. The process exited successfully without checkpoint output. An earlier otherwise identical attempt with --deterministic reached the training loop but triggered the explicit guard that packed sequence does not support deterministic mode; the successful command omits that incompatible flag, as does the Qwen 3.5 MoE reference card.
</p>
      </section>
    </article>
    <article id="qwen3-8-27b-peft-gb200" class="verification-model-detail" data-entry-detail="qwen3-8-27b-peft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-14</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.115661</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.558339</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>74,048.550 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>75.790 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>442.520 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 4 --recipe qwen35_vl_27b_peft_2gpu_h100_bf16_config --mode lora --tensor_model_parallel_size 4 --dataset medpix --deterministic --pretrained_checkpoint work/model-verification/qwen3.8-27b/gpu-megatron/iter_0000000 --max_steps 100 --warmup_iters 10 --save_interval 50 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=work/model-verification/qwen3.8-27b/gpu-hf-export dataset.do_validation=false dataset.pad_to_max_length=true tokenizer.tokenizer_model=work/model-verification/qwen3.8-27b/gpu-hf-export model.hf_model_id=Qwen/Qwen3.8-27B model.bos_token_id=248044 model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 model.recompute_modules=null scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 ddp.check_for_large_grads=true --save_dir work/model-verification/qwen3.8-27b/peft-tp4-recompute/checkpoints logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.8-27b/peft-tp4-recompute/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The four-GB200 run completed exactly 100 deterministic MedPix LoRA steps at TP4/PP1, MBS4/GBS32, and one-layer full-uniform activation recompute. Loss remained finite from 2.115661 to 1.558339 with zero skipped or NaN iterations. Steps 91-100 averaged 74,048.550 ms and 75.790 model TFLOP/s/GPU. Step-50 and step-100 adapter checkpoints each contain 8 files totaling 2,294,105,245 bytes, the tracker selects iteration 100, and the process exited successfully in 2h01m43s. The unchanged stock TP2 two-GPU recipe without activation recompute OOMed; a TP4 retry without recompute also OOMed near the approximately 184.3-GiB per-GPU device limit. The successful command records only the required topology and recompute config overrides, with no code change.
</p>
      </section>
    </article>
    <article id="qwen3-8-27b-pretrain-performance-gb200" class="verification-model-detail" data-entry-detail="qwen3-8-27b-pretrain-performance-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-20</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>6.852387</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>3.884668</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>1,679.460 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>819.190 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>4,877.758 tokens/s/GPU</dd>
          </div>
          <div>
            <dt>Peak allocated memory</dt>
            <dd>174.120 GiB</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 4 --recipe qwen35_vl_27b_pretrain_16gpu_gb200_bf16_mock_config --mode pretrain --pretrained_checkpoint work/model-verification/qwen3.8-27b/gpu-megatron/iter_0000000 --max_steps 20 --warmup_iters 2 dataset.hf_processor_path=work/model-verification/qwen3.8-27b/gpu-hf-export tokenizer.tokenizer_model=work/model-verification/qwen3.8-27b/gpu-hf-export model.hf_model_id=Qwen/Qwen3.8-27B model.bos_token_id=248044 logger.save_config_filepath=work/model-verification/qwen3.8-27b/pretrain-performance/resolved-config.yaml scheduler.lr_decay_iters=20</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 16 GB200 GPUs, the exact canonical BF16 language-and-projector MockVLM recipe completes 20 steps at TP2/PP1/CP1, DP8, sequence length 4096, and MBS2/GBS32. The language model and multimodal vision merger train while the vision encoder remains frozen. The recipe uses fused Gated DeltaNet execution without activation recompute and disables evaluation and checkpoint output for the bounded run. Loss remains finite from 6.852387 to 3.884668 with finite gradients and zero skipped or NaN iterations. Excluding first-iteration kernel compilation, steps 11-20 average 1,679.460 ms, 819.190 corrected model TFLOP/s/GPU, and 4,877.758 token slots/s/GPU; peak allocated memory is 174.120 GiB. The process exits successfully. This mock-data result is throughput evidence, not convergence evidence.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
