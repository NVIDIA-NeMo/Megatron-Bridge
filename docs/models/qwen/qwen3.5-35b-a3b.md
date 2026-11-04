# Qwen3.5-35B-A3B

[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) is a vision-language MoE model. Its 40 language blocks combine Gated DeltaNet or gated self-attention with routed and shared experts. The HybridModel implementation represents each block as two consecutive layers and preserves the existing vision encoder and multimodal rotary positions.

The migration is under validation. The verification card records the status of the exact model; synthetic performance workloads do not establish pretrained-model quality or convergence.

The canonical eight-GB300 performance recipes are `qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config` and `qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config`. Both use sequence length 4096, micro batch size 4, global batch size 512, and expert parallelism 8. BF16 uses NCCL EP; MXFP8 uses HybridEP. Both use mock multimodal inputs and forced load balancing.

The BF16 recipe enables Transformer Engine's native grouped-tensor path, which requires cuBLASLt 13.3 or later on Blackwell. Older libraries silently fall back to per-expert execution with host synchronization. Avoid cuBLAS 13.7.0 through 13.8.0: the [Transformer Engine release notes](https://docs.nvidia.com/deeplearning/transformer-engine/release_notes.html) document incorrect grouped-GEMM results with concurrent NCCL kernels and recommend 13.8.1 or later.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-qwen3.5-35b-a3b"></a>
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
        <button type="button" data-hardware="GB300">GB300</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-5-35b-a3b-hf-to-megatron-cpu" aria-controls="qwen3-5-35b-a3b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-5-35b-a3b-hf-to-megatron-gpu" aria-controls="qwen3-5-35b-a3b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-5-35b-a3b-megatron-to-hf-cpu" aria-controls="qwen3-5-35b-a3b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="qwen3-5-35b-a3b-megatron-to-hf-gpu" aria-controls="qwen3-5-35b-a3b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB300" data-status="verified" data-entry="qwen3-5-35b-a3b-pretrain-gb300" aria-controls="qwen3-5-35b-a3b-pretrain-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB300</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB300" data-status="verified" data-entry="qwen3-5-35b-a3b-sft-gb300" aria-controls="qwen3-5-35b-a3b-sft-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB300</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB300" data-status="verified" data-entry="qwen3-5-35b-a3b-sft-long-context-gb300" aria-controls="qwen3-5-35b-a3b-sft-long-context-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB300</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB300" data-status="verified" data-entry="qwen3-5-35b-a3b-peft-gb300" aria-controls="qwen3-5-35b-a3b-peft-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB300</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="benchmark" data-precision="bf16" data-hardware="GB300" data-status="unverified" data-entry="qwen3-5-35b-a3b-pretrain-performance-gb300" aria-controls="qwen3-5-35b-a3b-pretrain-performance-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Benchmark · GB300</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="qwen3-5-35b-a3b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="qwen3-5-35b-a3b-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --gpus-per-node 4 --mem 512G --hf-model Qwen/Qwen3.5-35B-A3B --hf-revision 59d61f3ce65a6d9863b86d2e96597125219dc754 --megatron-path work/model-verification/qwen3.5-35b-a3b/cpu-megatron</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Imported the complete pinned checkpoint with model weights on CPU in one process and saved iter_0000000. Transformer Engine vision-module construction required one runtime GPU; four GB300 GPUs were allocated to satisfy the minimum allocation. The subsequent CPU export reloaded the checkpoint successfully. The complete BF16 roundtrip preserves all 1,811 keys and shapes, with precision changes documented in megatron_to_hf_cpu; it is not a lossless mixed-dtype roundtrip.
</p>
      </section>
    </article>
    <article id="qwen3-5-35b-a3b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="qwen3-5-35b-a3b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --ep 4 --hf-model Qwen/Qwen3.5-35B-A3B --hf-revision 59d61f3ce65a6d9863b86d2e96597125219dc754 --megatron-path work/model-verification/qwen3.5-35b-a3b/gpu-megatron</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Imported the complete pinned checkpoint on four GB300 GPUs at TP1/PP1/EP4/ETP1 and saved iter_0000000 with four distributed shards. Subsequent GPU export, forward comparison, inference, SFT, and PEFT successfully reloaded the checkpoint. The BF16 roundtrip preserves all 1,811 tensor keys and shapes, with the precision changes documented in megatron_to_hf_gpu; it is not a lossless mixed-dtype roundtrip.
</p>
      </section>
    </article>
    <article id="qwen3-5-35b-a3b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="qwen3-5-35b-a3b-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --gpus-per-node 4 --mem 512G --hf-model Qwen/Qwen3.5-35B-A3B --hf-revision 59d61f3ce65a6d9863b86d2e96597125219dc754 --megatron-path work/model-verification/qwen3.5-35b-a3b/cpu-megatron/iter_0000000 --hf-path work/model-verification/qwen3.5-35b-a3b/cpu-hf-export</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Export completed with weights on CPU in one process and one runtime GPU from a four-GB300 allocation. Transformers independently reloaded the complete export with no missing, unexpected, mismatched, or error entries. All 1,811 keys and shapes match the pinned source. The source&#x27;s 30 GDN A_log tensors and 30 GDN output norm tensors change from FP32 to BF16; all other dtypes are preserved. Values of 1,781 tensors match exactly, including A_log. The 30 norm tensors exactly match BF16(FP32_source - 1) + BF16(1). Their maximum absolute source difference is 0.00390625, within an absolute tolerance of 1/256; no other tensor values differ. This verifies BF16 conversion, not lossless retention of the source&#x27;s mixed dtypes.
</p>
      </section>
    </article>
    <article id="qwen3-5-35b-a3b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="qwen3-5-35b-a3b-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 4 --ep 4 --hf-model Qwen/Qwen3.5-35B-A3B --hf-revision 59d61f3ce65a6d9863b86d2e96597125219dc754 --megatron-path work/model-verification/qwen3.5-35b-a3b/gpu-megatron/iter_0000000 --hf-path work/model-verification/qwen3.5-35b-a3b/gpu-hf-export</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>GPU export completed on four GB300 GPUs at TP1/PP1/EP4/ETP1. All 1,811 keys and shapes match the pinned source. Transformers strictly reloaded the complete export with no missing, unexpected, mismatched, or error entries. The source&#x27;s 30 GDN A_log tensors and 30 GDN output norm tensors change from FP32 to BF16; all other dtypes are preserved. Values of 1,781 tensors match exactly, including A_log. The 30 norm tensors exactly match BF16(FP32_source - 1) + BF16(1), the implemented zero-centered norm roundtrip. Their maximum absolute source difference is 0.00390625, within an absolute tolerance of 1/256; no other tensor values differ. This verifies the BF16 conversion, not lossless retention of the source&#x27;s mixed dtypes.
</p>
      </section>
    </article>
    <article id="qwen3-5-35b-a3b-pretrain-gb300" class="verification-model-detail" data-entry-detail="qwen3-5-35b-a3b-pretrain-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB300</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-04</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.997142</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.03088333</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>28,666.450 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>420.700 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>18,289.255 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 4 --gpus-per-node 4 --wait --recipe qwen35_vl_35b_a3b_pretrain_16gpu_gb300_bf16_config --mode pretrain --dataset energon --pretrained_checkpoint work/model-verification/qwen3.5-35b-a3b/gpu-megatron/iter_0000000 --max_steps 100 --warmup_iters 40 --lr 3e-4 --min_lr 3e-5 dataset.path=work/data/medpix/energon-pinned dataset.task_encoder.hf_processor_path=Qwen/Qwen3.5-35B-A3B dataset.task_encoder.hf_processor_revision=59d61f3ce65a6d9863b86d2e96597125219dc754 dataset.task_encoder.max_pixels=200704 dataset.do_validation=false dataset.pad_to_max_length=true scheduler.lr_decay_iters=100 checkpoint.load=null --save_dir work/model-verification/qwen3.5-35b-a3b/pretrain/checkpoints --save_interval 50 validation.eval_iters=0 validation.eval_interval=0 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.5-35b-a3b/pretrain/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed all 100 full-model continuation steps on 16 GB300 GPUs, warm-started from the pinned published weights, with finite losses and zero skipped or NaN iterations. The Energon input contains all 17,420 training rows of mmoukouba/MedPix-VQA revision 69eb219d477ab8699296da1fd6b89beb768042d3, reused for 204,800 consumed examples. This is a bounded VQA continuation cohort, not the RedPajama2 or DataComp pretraining cohorts and not full convergence. The saved runtime config records TP1/PP1/CP1/EP4/ETP1, MBS4/GBS2048, sequence length 4096, natural routing, FLA GDN, FP32 optimizer master parameters and gradients, BF16 Adam moments, selective core_attn and gdn_norm_out recompute, and separate vision attn/mlp CUDA graphs. Learning rate warms up for 40 steps to 3e-4 and decays to 3e-5 at step 100. Metrics average steps 91-100. Both step-50 and step-100 checkpoints saved all 16 distributed shards, optimizer/RNG state, and per-rank Energon data positions. A separate continuation successfully restored step 50; its loss-consistency result is tracked in checkpoint_resume.
</p>
      </section>
    </article>
    <article id="qwen3-5-35b-a3b-sft-gb300" class="verification-model-detail" data-entry-detail="qwen3-5-35b-a3b-sft-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB300</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-03</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.151638</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>0.9914345</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>5,216.920 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>72.280 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,140.550 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --wait --recipe qwen35_vl_35b_a3b_sft_8gpu_gb200_bf16_functional_config --mode sft --dataset medpix --pretrained_checkpoint work/model-verification/qwen3.5-35b-a3b/gpu-megatron/iter_0000000 --max_steps 100 --warmup_iters 10 scheduler.lr_decay_iters=100 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=Qwen/Qwen3.5-35B-A3B &#x27;dataset.hf_processor_kwargs={revision:59d61f3ce65a6d9863b86d2e96597125219dc754}&#x27; env_vars.NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=64 checkpoint.load=null --save_dir work/model-verification/qwen3.5-35b-a3b/sft/checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.5-35b-a3b/sft/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed all 100 full-model MedPix-VQA SFT steps on eight GB300 GPUs, including language, vision, projection, and one MTP layer, with finite losses and zero skipped or NaN iterations. The persisted runtime config records TP1/PP1/CP1/EP8/ETP1, sequence length 4096, MBS1/GBS32, natural routing, no CUDA graphs, FP32 optimizer states, and assistant-only loss. Adam learning rate warms up for 10 steps to 2e-5 and decays to 2e-6 at step 100. The combine chunk is explicitly 64 to match HybridEP fused dispatch and preprocessing; the inherited 128 fails its runtime check. Step 100 saved all eight distributed shards, optimizer and RNG state, and the checkpoint tracker selects iteration 100. These are bounded support-verification observations, not full convergence or optimized performance results.
</p>
      </section>
    </article>
    <article id="qwen3-5-35b-a3b-sft-long-context-gb300" class="verification-model-detail" data-entry-detail="qwen3-5-35b-a3b-sft-long-context-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB300</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-03</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>1.966061</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.024277</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>115,083.290 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>108.650 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>4,555.727 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --wait --recipe qwen35_vl_35b_a3b_sft_long_context_32gpu_h100_bf16_config --mode sft --dataset medpix --pretrained_checkpoint work/model-verification/qwen3.5-35b-a3b/gpu-megatron/iter_0000000 --max_steps 20 --warmup_iters 2 scheduler.lr_decay_iters=20 model.pipeline_model_parallel_size=1 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=Qwen/Qwen3.5-35B-A3B &#x27;dataset.hf_processor_kwargs={revision:59d61f3ce65a6d9863b86d2e96597125219dc754}&#x27; dataset.do_validation=false dataset.seq_length=8192 dataset.enable_in_batch_packing=true dataset.defer_in_batch_packing_to_step=true dataset.in_batch_packing_pad_to_multiple_of=4 checkpoint.load=null checkpoint.save=null validation.eval_iters=0 validation.eval_interval=0 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.5-35b-a3b/long_context/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed all 20 MedPix-VQA SFT steps on eight GB300 GPUs with finite losses and zero skipped or NaN iterations. The H100 long-context recipe was resized to PP1 while retaining its MBS2/GBS512; the persisted runtime config records TP1/PP1/CP2/EP8/ETP1, alltoall dispatch, no CUDA graphs, one-layer full-uniform recompute, and learning rate 2e-5 to 2e-6 with two warmup steps. Qwen&#x27;s step-time THD packing pads each example to 8192 slots and preserves separate attention boundaries; CP2 was active throughout. Token-slot throughput uses 8192 times GBS512 per step. Metrics average steps 11-20. This verifies padded 8K execution with packing and CP, not long-document answer quality or full convergence; no checkpoint was requested for this item.
</p>
      </section>
    </article>
    <article id="qwen3-5-35b-a3b-peft-gb300" class="verification-model-detail" data-entry-detail="qwen3-5-35b-a3b-peft-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB300</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-03</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>2.151638</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.313625</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>4,884.420 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>77.200 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>3,354.339 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --wait --recipe qwen35_vl_35b_a3b_peft_8gpu_gb200_bf16_functional_config --mode lora --dataset medpix --pretrained_checkpoint work/model-verification/qwen3.5-35b-a3b/gpu-megatron/iter_0000000 --max_steps 100 --warmup_iters 10 scheduler.lr_decay_iters=100 &#x27;dataset.source.load_kwargs={revision:69eb219d477ab8699296da1fd6b89beb768042d3}&#x27; &#x27;dataset.source.adapter_kwargs={resized_height:448,resized_width:448}&#x27; dataset.hf_processor_path=Qwen/Qwen3.5-35B-A3B &#x27;dataset.hf_processor_kwargs={revision:59d61f3ce65a6d9863b86d2e96597125219dc754}&#x27; env_vars.NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=64 checkpoint.load=null --save_dir work/model-verification/qwen3.5-35b-a3b/peft/checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.5-35b-a3b/peft/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed all 100 MedPix-VQA LoRA steps on eight GB300 GPUs with finite losses and zero skipped or NaN iterations. The persisted runtime config records TP1/PP1/CP1/EP8/ETP1, sequence length 4096, MBS1/GBS32, natural routing, no CUDA graphs, rank 32, alpha 32, and targets linear_qkv, linear_proj, linear_fc1, and linear_fc2. Learning rate warms up for 10 steps to 2e-4 and decays to 3e-5 at step 100. Step 100 saved all eight distributed adapter/optimizer shards and the checkpoint tracker selects iteration 100. This verifies bounded adapter training, not full convergence.
</p>
      </section>
    </article>
    <article id="qwen3-5-35b-a3b-pretrain-performance-gb300" class="verification-model-detail" data-entry-detail="qwen3-5-35b-a3b-pretrain-performance-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Benchmark · GB300</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>12.76551</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>3.010811</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>11,420.740 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>528.120 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>22,953.329 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --wait --recipe qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config --mode pretrain --max_steps 50 logger.log_interval=1 logger.log_throughput=true logger.save_config_filepath=work/model-verification/qwen3.5-35b-a3b/performance/resolved-config.yaml</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Completed all 50 optimizer steps of the exact canonical BF16 recipe on eight GB300 GPUs with finite losses and zero skipped or NaN iterations. The saved runtime config records random weights, mock multimodal data, forced router load balancing, sequence length 4096, MBS4/GBS512, TP1/PP1/CP1/ETP1/EP8, NCCL EP, Transformer Engine GDN, and attn/moe_router/moe_preprocess CUDA graphs. Metrics average steps 41-50. This synthetic workload does not save weights or establish convergence and is not comparable to the real-data training items&#x27; loss trajectories. The recorded run omitted Qwen rotary-position inputs from language attention graphs. Its throughput is retained as historical evidence, not an accepted correct-model performance result for the current revision.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
