# Qwen3.6-35B-A3B

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

<a id="verified-qwen3.6-35b-a3b"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-6-35b-a3b-hf-to-megatron-cpu" aria-controls="qwen3-6-35b-a3b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-6-35b-a3b-hf-to-megatron-gpu" aria-controls="qwen3-6-35b-a3b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-6-35b-a3b-megatron-to-hf-cpu" aria-controls="qwen3-6-35b-a3b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-6-35b-a3b-megatron-to-hf-gpu" aria-controls="qwen3-6-35b-a3b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="qwen3-6-35b-a3b-pretrain-h100" aria-controls="qwen3-6-35b-a3b-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="qwen3-6-35b-a3b-sft-h100" aria-controls="qwen3-6-35b-a3b-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="qwen3-6-35b-a3b-sft-gb200" aria-controls="qwen3-6-35b-a3b-sft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="qwen3-6-35b-a3b-sft-long-context-h100" aria-controls="qwen3-6-35b-a3b-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="qwen3-6-35b-a3b-peft-h100" aria-controls="qwen3-6-35b-a3b-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB200" data-status="verified" data-entry="qwen3-6-35b-a3b-peft-gb200" aria-controls="qwen3-6-35b-a3b-peft-gb200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB200</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="qwen3-6-35b-a3b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-hf-to-megatron-cpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --gpus-per-node 1 --mem 512G --hf-model Qwen/Qwen3.6-35B-A3B --hf-revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --megatron-path work/model-verification/qwen3.6-35b-a3b/cpu-megatron</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The pinned-revision import exited successfully in 14m39s and created iter_0000000. Model weights remained on CPU; exactly one shared runtime GPU was required because Transformer Engine constructs the Qwen vision modules through CUDA even for CPU-weight conversion. The resulting checkpoint was reloadable by the subsequent verified CPU export.
</p>
      </section>
    </article>
    <article id="qwen3-6-35b-a3b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-hf-to-megatron-gpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 2 --gpus-per-node 8 --hf-model Qwen/Qwen3.6-35B-A3B --hf-revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --megatron-path work/model-verification/qwen3.6-35b-a3b/imported-megatron --tp 2 --ep 8</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 16-H100 pinned-revision import exited successfully at TP2/PP1/EP8/ETP1 and created a 16-shard iter_0000000 distributed checkpoint. Subsequent verified GPU export and inference workloads reloaded this checkpoint successfully.
</p>
      </section>
    </article>
    <article id="qwen3-6-35b-a3b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-megatron-to-hf-cpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --gpus-per-node 1 --mem 512G --hf-model Qwen/Qwen3.6-35B-A3B --hf-revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --megatron-path work/model-verification/qwen3.6-35b-a3b/cpu-megatron/iter_0000000 --hf-path work/model-verification/qwen3.6-35b-a3b/cpu-hf-export</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Export exited successfully in 6m42s with weights on CPU and exactly one shared runtime GPU for Transformer Engine vision-module construction. Transformers strictly reloaded all 1,045 BF16 tensors in 26 indexed shards as Qwen3_5MoeForConditionalGeneration. The exported config preserves 40 language layers, 256 experts, top-8 routing, 27 vision layers, and one MTP layer with dedicated embeddings disabled.
</p>
      </section>
    </article>
    <article id="qwen3-6-35b-a3b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-megatron-to-hf-gpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 2 --gpus-per-node 8 --hf-model Qwen/Qwen3.6-35B-A3B --hf-revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --megatron-path work/model-verification/qwen3.6-35b-a3b/imported-megatron/iter_0000000 --hf-path work/model-verification/qwen3.6-35b-a3b/hf-export --tp 2 --ep 8</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 16-H100 distributed export exited successfully and wrote 26 indexed shards containing 1,045 BF16 tensors and 71,903,645,408 serialized tensor bytes. Transformers independently reloaded the persisted export as Qwen3_5MoeForConditionalGeneration with empty missing, unexpected, mismatched, and error sets.
</p>
      </section>
    </article>
    <article id="qwen3-6-35b-a3b-pretrain-h100" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-05</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>3.604611</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>3.458849</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>22,271.220 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>135.610 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>5,885.264 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe qwen35_vl_35b_a3b_pretrain_config --dataset energon --pretrained_checkpoint work/model-verification/qwen3.6-35b-a3b/imported-megatron/iter_0000000 --max_steps 50 --warmup_iters 10 dataset.path=work/data/datacomp/energon dataset.task_encoder.hf_processor_path=Qwen/Qwen3.6-35B-A3B dataset.task_encoder.hf_processor_revision=995ad96eacd98c81ed38be0c5b274b04031597b0 dataset.task_encoder.max_pixels=200704 dataset.do_validation=false dataset.pad_to_max_length=true scheduler.lr_decay_iters=50 model.hf_model_id=Qwen/Qwen3.6-35B-A3B model.bos_token_id=248044 checkpoint.load=null validation.eval_iters=0 validation.eval_interval=0 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.6-35b-a3b/h100-performance/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 16-H100 command completed all 50 full-model DataComp Energon steps at TP1/PP2/CP1/EP8/ETP1 and MBS1/GBS512 after warm-starting the immutable imported checkpoint for natural routing. Loss remained finite from 3.604611 to 3.458849 with zero skipped or NaN iterations. The 519,827/5,173 train/validation split was prepared with the official DataComp downloader at commit 4a8df1992566, DataComp-1B metadata revision 086ebeee20d4, img2dataset 1.40.0, and preparation manifest 6e273a96a756. This verifies bounded full-model training, not canonical DataComp/CLIP convergence or checkpoint resume.
</p>
      </section>
    </article>
    <article id="qwen3-6-35b-a3b-sft-h100" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-05</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.211317</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.006886</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>3,268.140 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>57.690 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>2,506.625 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config --dataset medpix --pretrained_checkpoint work/model-verification/qwen3.6-35b-a3b/imported-megatron/iter_0000000 --max_steps 100 --warmup_iters 10 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=Qwen/Qwen3.6-35B-A3B &#x27;dataset.hf_processor_kwargs={revision:995ad96eacd98c81ed38be0c5b274b04031597b0}&#x27; dataset.do_validation=false dataset.pad_to_max_length=true scheduler.lr_decay_iters=100 model.hf_model_id=Qwen/Qwen3.6-35B-A3B model.bos_token_id=248044 validation.eval_iters=0 validation.eval_interval=0 ddp.check_for_large_grads=true --save_dir work/model-verification/qwen3.6-35b-a3b/sft-support-v11/checkpoints --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.6-35b-a3b/sft-support-v11/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The exact 16-H100 command completed all 100 MedPix SFT steps at TP1/PP2/CP1/EP8/ETP1, MBS1/GBS32, natural expert routing, HybridEP, one-layer full-uniform recompute, graph-free language and vision stacks, FP32 optimizer state, and Transformer Engine fused cross entropy. Loss was finite from 2.211317 to 1.006886 with zero skipped or NaN iterations. Steps 91-100 averaged 3,268.140 ms and 57.690 model TFLOP/s/GPU. Step-50 and step-100 saves both completed; the final checkpoint contains 20 nonempty files, including 16 distributed shards, totaling 509,636,716,209 bytes, and the tracker selects iteration 100.
</p>
      </section>
    </article>
    <article id="qwen3-6-35b-a3b-sft-gb200" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-sft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-30</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.212866</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.9978729</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>9,899.640 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>38.180 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>1,655.010 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --recipe qwen35_vl_35b_a3b_sft_8gpu_gb200_bf16_functional_config --dataset medpix --pretrained_checkpoint work/model-verification/qwen3.6-35b-a3b/gb200-imported/iter_0000000 --max_steps 100 --warmup_iters 10 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=Qwen/Qwen3.6-35B-A3B &#x27;dataset.hf_processor_kwargs={revision:995ad96eacd98c81ed38be0c5b274b04031597b0}&#x27; scheduler.lr_decay_iters=100 model.hf_model_id=Qwen/Qwen3.6-35B-A3B model.bos_token_id=248044 logger.save_config_filepath=work/model-verification/qwen3.6-35b-a3b/gb200-sft/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 8x GB200, the shared Qwen3.5/Qwen3.6-VL functional recipe completed exactly 100 MedPix SFT steps at TP1/PP1/CP1/EP8/ETP1, MBS1/GBS32, natural routing, HybridEP, no activation recompute, and BF16 gradient reduction. Loss was finite from 2.212866 to 0.9978729 with zero skipped or NaN iterations. Steps 91-100 averaged 9,899.640 ms and 38.180 model TFLOP/s/GPU. The recipe intentionally disables checkpoint output for this bounded functional run; reload/export coverage remains represented by the H100 SFT and sft_export_inference items.
</p>
      </section>
    </article>
    <article id="qwen3-6-35b-a3b-sft-long-context-h100" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-sft-long-context-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-28</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.997653</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.997894</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>36,847.110 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>84.830 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,557.185 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 8 --recipe qwen35_vl_35b_a3b_sft_long_context_32gpu_h100_bf16_config --dataset medpix --pretrained_checkpoint work/model-verification/qwen3.6-35b-a3b/imported-megatron/iter_0000000 --max_steps 20 --warmup_iters 2 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=Qwen/Qwen3.6-35B-A3B &#x27;dataset.hf_processor_kwargs={revision:995ad96eacd98c81ed38be0c5b274b04031597b0}&#x27; dataset.do_validation=false scheduler.lr_decay_iters=20 model.hf_model_id=Qwen/Qwen3.6-35B-A3B model.bos_token_id=248044 checkpoint.load=null checkpoint.save=null validation.eval_iters=0 validation.eval_interval=0 ddp.check_for_large_grads=true logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.6-35b-a3b/medpix-long-context/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The pinned 32-H100 run completed exactly 20 full-SFT steps at TP1/PP4/CP2/EP8/ETP1, dense DP4 and expert DP1, MBS2/GBS512, and 64-way gradient accumulation. It used 8192-token processor-native deferred in-batch packing with pad-to-multiple-of-4, fixed 448-by-448 MedPix images, per-token loss, and one-layer full-uniform activation recompute. The persisted post-setup config confirms every setting. Loss was finite from 1.997653 to 0.997894 with zero skipped or NaN iterations. After the compile-heavy first step, steps 11-20 averaged 36,847.110 ms and 84.830 TFLOP/s/GPU; the process exited successfully.
</p>
      </section>
    </article>
    <article id="qwen3-6-35b-a3b-peft-h100" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-05</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.212696</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.270382</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>1,714.300 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>109.940 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>4,778.627 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe qwen35_vl_35b_a3b_peft_16gpu_h100_bf16_config --dataset medpix --pretrained_checkpoint work/model-verification/qwen3.6-35b-a3b/imported-megatron/iter_0000000 --max_steps 100 --warmup_iters 10 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=Qwen/Qwen3.6-35B-A3B &#x27;dataset.hf_processor_kwargs={revision:995ad96eacd98c81ed38be0c5b274b04031597b0}&#x27; dataset.do_validation=false dataset.pad_to_max_length=true scheduler.lr_decay_iters=100 model.hf_model_id=Qwen/Qwen3.6-35B-A3B model.bos_token_id=248044 validation.eval_iters=0 validation.eval_interval=0 ddp.check_for_large_grads=true --save_dir work/model-verification/qwen3.6-35b-a3b/peft-support-v5/checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.6-35b-a3b/peft-support-v5/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The exact 16-H100 command completed all 100 MedPix LoRA steps at TP1/PP2/CP1/EP8/ETP1, MBS1/GBS32, natural expert routing, HybridEP, scoped Transformer Engine CUDA graphs, and no activation recompute. The recipe disables dispatch-backward/expert-wgrad overlap because LoRA-wrapped expert linears do not expose Megatron Core&#x27;s backward_dw hook. Loss was finite from 2.212696 to 1.270382 with zero skipped or NaN iterations. Steps 91-100 averaged 1,714.300 ms and 109.940 model TFLOP/s/GPU. The step-100 checkpoint contains 20 nonempty files, including 16 distributed shards, totaling 5,004,722,270 bytes.
</p>
      </section>
    </article>
    <article id="qwen3-6-35b-a3b-peft-gb200" class="verification-model-detail" data-entry-detail="qwen3-6-35b-a3b-peft-gb200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB200</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-30</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.214827</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.254138</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>5,511.530 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>68.410 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>2,972.677 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --recipe qwen35_vl_35b_a3b_peft_8gpu_gb200_bf16_functional_config --dataset medpix --pretrained_checkpoint work/model-verification/qwen3.6-35b-a3b/gb200-imported/iter_0000000 --max_steps 100 --warmup_iters 10 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=Qwen/Qwen3.6-35B-A3B &#x27;dataset.hf_processor_kwargs={revision:995ad96eacd98c81ed38be0c5b274b04031597b0}&#x27; scheduler.lr_decay_iters=100 model.hf_model_id=Qwen/Qwen3.6-35B-A3B model.bos_token_id=248044 logger.save_config_filepath=work/model-verification/qwen3.6-35b-a3b/gb200-peft/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On the same 8x GB200 topology and MedPix schedule as the controlled SFT item, LoRA completed exactly 100 finite steps with zero skipped or NaN iterations. The persisted configs differ only in the evidence-output path, the intended PEFT LR/min-LR, and the LoRA block; model, data, topology, batch, schedule, and safety settings match. Step-1 loss 2.214827 differs from SFT&#x27;s 2.212866 by only 0.001961, or 0.088618%, ruling out an initial-forward configuration mismatch. The later loss difference reflects LoRA&#x27;s trainable parameter set and learning rate. Steps 91-100 averaged 5,511.530 ms and 68.410 model TFLOP/s/GPU.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
