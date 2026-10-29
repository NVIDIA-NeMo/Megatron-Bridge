# Moonlight

[Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B) is a 16B-parameter Mixture-of-Experts (MoE) model from **Moonshot AI** trained with 5.7T tokens using the innovative **Muon optimizer**. While Moonlight shares the same architecture as DeepSeek-V3 (featuring Multi-head Latent Attention and MoE), it is a distinct model that advances the Pareto frontier of performance vs training FLOPs through the use of Muon, which is ~2× more sample efficient than Adam with compute optimal training.

The model features 27 decoder layers with 64 routed experts and 8 shared experts per layer, with 3B activated parameters per forward pass out of 16B total parameters.

Moonlight models are supported via the Bridge system with specialized configurations for MoE and MLA optimizations.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-moonlight-16b-a3b"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="moonlight-16b-a3b-hf-to-megatron-cpu" aria-controls="moonlight-16b-a3b-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="moonlight-16b-a3b-hf-to-megatron-gpu" aria-controls="moonlight-16b-a3b-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="moonlight-16b-a3b-megatron-to-hf-cpu" aria-controls="moonlight-16b-a3b-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="moonlight-16b-a3b-megatron-to-hf-gpu" aria-controls="moonlight-16b-a3b-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="moonlight-16b-a3b-pretrain-h100" aria-controls="moonlight-16b-a3b-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="moonlight-16b-a3b-sft-h100" aria-controls="moonlight-16b-a3b-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="moonlight-16b-a3b-sft-long-context-h100" aria-controls="moonlight-16b-a3b-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="verified" data-entry="moonlight-16b-a3b-peft-h100" aria-controls="moonlight-16b-a3b-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="moonlight-16b-a3b-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="moonlight-16b-a3b-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-19</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model moonshotai/Moonlight-16B-A3B --megatron-path work/model-verification/moonlight-16b-a3b/cpu-megatron --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully, creates iter_0000000, and the checkpoint round-trips through CPU export with exact model configuration. After native reload, all 377 registered parameters and persistent buffers match bitwise by key, shape, dtype, and value. In the serialized files, 26 router correction-bias buffers widen losslessly from BF16 to FP32.
</p>
      </section>
    </article>
    <article id="moonlight-16b-a3b-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="moonlight-16b-a3b-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-19</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model moonshotai/Moonlight-16B-A3B --megatron-path work/model-verification/moonlight-16b-a3b/imported-megatron --torch-dtype bfloat16 --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully, creates a reloadable iter_0000000, and an exact audit finds all 5,344 serialized source tensors and the model configuration unchanged after the paired GPU export.
</p>
      </section>
    </article>
    <article id="moonlight-16b-a3b-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="moonlight-16b-a3b-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-19</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model moonshotai/Moonlight-16B-A3B --megatron-path work/model-verification/moonlight-16b-a3b/cpu-megatron/iter_0000000 --hf-path work/model-verification/moonlight-16b-a3b/cpu-hf-export --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The command exits successfully, the model configuration is exact, and all 377 registered parameters and persistent buffers match bitwise by key, shape, dtype, and value after native reload. Transformers strictly reloads the output as DeepseekV3ForCausalLM with trust_remote_code disabled. In the serialized files, 26 router correction-bias buffers widen losslessly from BF16 to FP32 and 27 stale, nonpersistent source inv_freq tensors are omitted.
</p>
      </section>
    </article>
    <article id="moonlight-16b-a3b-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="moonlight-16b-a3b-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-07-19</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model moonshotai/Moonlight-16B-A3B --megatron-path work/model-verification/moonlight-16b-a3b/imported-megatron/iter_0000000 --hf-path work/model-verification/moonlight-16b-a3b/hf-export --torch-dtype bfloat16 --export-weight-dtype bfloat16 --distributed-save --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Strict distributed export exits successfully, all 5,344 serialized tensor keys, shapes, dtypes, and values match the recorded HF revision, and native Transformers strictly reloads the output as DeepseekV3ForCausalLM with trust_remote_code disabled and no missing or unexpected keys.
</p>
      </section>
    </article>
    <article id="moonlight-16b-a3b-pretrain-h100" class="verification-model-detail" data-entry-detail="moonlight-16b-a3b-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
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
            <dd>12.43248</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>5.15842</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>11,636.350 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>386.590 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>22,528.026 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 2 --gpus-per-node 8 --recipe moonlight_16b_pretrain_16gpu_h100_bf16_config --mode pretrain --dataset megatron-indexed --seq_length 4096 --max_steps 100 --lr 3e-4 --min_lr 3e-5 --warmup_iters 40 &#x27;dataset.blend=[[&quot;work/data/rp2/head_01&quot;],null]&#x27; dataset.path_to_cache=work/cache/moonlight-16b-a3b/rp2-convergence-v1 tokenizer.tokenizer_type=SentencePieceTokenizer tokenizer.tokenizer_model=work/data/rp2/tokenizer.model dataset.random_seed=1234 rng.seed=1234 scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 ddp.check_for_nan_in_grad=true ddp.check_for_large_grads=true rerun_state_machine.check_for_nan_in_loss=true checkpoint.load=null dist.distributed_timeout_minutes=30 --save_dir work/model-verification/moonlight-16b-a3b/pretrain-convergence-v1-reference-checkpoints --save_interval 50 logger.log_interval=1 logger.log_throughput=true logger.tensorboard_dir=null</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The uninterrupted 16-GPU bounded RP2 run completes exactly 100 steps at TP1/PP1/CP1/EP8/ETP1, DP16, GBS/MBS 1024/2, and 32-way gradient accumulation with natural MoE routing and the HybridEP flex dispatcher. It reaches peak learning rate at step 40 and completes cosine decay at step 100. LM loss is finite from 12.43248 to 5.158420 with no skipped or NaN iterations, all five metrics are recorded, the post-setup configuration persists, and complete 16-shard iter_0000050 and iter_0000100 checkpoints are saved.
</p>
      </section>
    </article>
    <article id="moonlight-16b-a3b-sft-h100" class="verification-model-detail" data-entry-detail="moonlight-16b-a3b-sft-h100" tabindex="-1">
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
            <dd>1.770799</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.180197</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>562.140 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>229.500 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>14,572.882 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 8 --recipe moonlight_16b_sft_8gpu_h100_bf16_tp1_config --mode sft --dataset tulu3 --pretrained_checkpoint work/model-verification/moonlight-16b-a3b/imported-megatron/iter_0000000 --max_steps 100 --seq_length 8192 --lr 5e-6 --min_lr 0 --warmup_iters 10 &#x27;dataset.hf_dataset.split=&quot;train[:10000]&quot;&#x27; &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;b14afda60f1bbebe55d5d2fa1e4df5042f97f8be&quot;}&#x27; &#x27;++tokenizer.hf_tokenizer_kwargs.revision=&quot;476b36a473d4467f94469414bef6cee75c9c8172&quot;&#x27; dataset.hf_output_root=work/data/tulu3/moonlight-16b-a3b-sft-hybridep-b14afda60f1b dataset.hf_rewrite=true dataset.seed=1234 rng.seed=5678 dataset.do_validation=false dataset.hf_validation_proportion=null dataset.enable_offline_packing=true +dataset.offline_packing_specs.pad_seq_to_mult=1 &#x27;dataset.dataset_kwargs={pad_to_max_length:true}&#x27; scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/moonlight-16b-a3b/sft-convergence-hybridep-v3-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The immutable-revision 8-GPU run completes exactly 100 full-SFT steps at TP1/PP1/CP1/EP8/ETP1, DP8, SP off, GBS/MBS 8/1, and no gradient accumulation, using the HybridEP flex dispatcher with overlapped expert communication and delayed weight-gradient compute. Fixed-width 8K, pad-1 offline packing is 99.73% efficient and averages 21.550 source sequences per pack while retaining 6,553,600 token slots across the 100 updates. LM loss is finite from 1.770799 to 1.180197 with no skipped or NaN iterations, all five metrics are recorded, and the complete eight-shard iter_0000100 full-model checkpoint saves successfully.
</p>
      </section>
    </article>
    <article id="moonlight-16b-a3b-sft-long-context-h100" class="verification-model-detail" data-entry-detail="moonlight-16b-a3b-sft-long-context-h100" tabindex="-1">
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
            <dd>1.30644</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.228019</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>60,599.310 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>40.840 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>2,162.929 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 8 --recipe moonlight_16b_sft_8k_config --mode sft --dataset tulu3 --pretrained_checkpoint work/model-verification/moonlight-16b-a3b/imported-megatron/iter_0000000 --max_steps 20 --seq_length 8192 --context_parallel_size 2 --lr 1e-6 --min_lr 0 --warmup_iters 2 &#x27;dataset.hf_dataset.split=&quot;train[:10000]&quot;&#x27; &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;b14afda60f1bbebe55d5d2fa1e4df5042f97f8be&quot;}&#x27; &#x27;++tokenizer.hf_tokenizer_kwargs.revision=&quot;476b36a473d4467f94469414bef6cee75c9c8172&quot;&#x27; dataset.hf_output_root=work/data/tulu3/moonlight-16b-a3b-long-context-b14afda60f1b dataset.hf_rewrite=true dataset.seed=1234 rng.seed=5678 dataset.do_validation=false dataset.hf_validation_proportion=null dataset.enable_offline_packing=true +dataset.offline_packing_specs.pad_seq_to_mult=4 scheduler.lr_decay_iters=20 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null checkpoint.save=null logger.log_interval=1 logger.log_throughput=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The immutable-revision 8-GPU run completes exactly 20 Tulu3 SFT steps at the model&#x27;s 8192-token context limit with recipe-owned TP2/PP1/CP2/EP8/SP-on, GBS/MBS 128/1, and explicit pad-4 offline packing. LM loss is 1.306440 to 1.228019; skipped/NaN totals are 0/0. The persisted post-setup runtime config matches the command, packing is 99.69%, and the sampled training window contains 13,765,732 actual supervised tokens.
</p>
      </section>
    </article>
    <article id="moonlight-16b-a3b-peft-h100" class="verification-model-detail" data-entry-detail="moonlight-16b-a3b-peft-h100" tabindex="-1">
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
            <dd>1.264324</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>1.100755</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>3,822.840 ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>67.380 TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>4,285.819 tokens/s/GPU</dd>
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
          <pre><code class="language-bash">./scripts/training/train.sh --nodes 1 --gpus-per-node 4 --recipe moonlight_16b_peft_4gpu_h100_bf16_config --mode lora --dataset tulu3 --pretrained_checkpoint work/model-verification/moonlight-16b-a3b/imported-megatron/iter_0000000 --max_steps 100 --seq_length 2048 --lr 1e-4 --min_lr 0 --warmup_iters 10 &#x27;dataset.hf_dataset.split=&quot;train[:10000]&quot;&#x27; &#x27;dataset.hf_dataset.load_kwargs={revision:&quot;b14afda60f1bbebe55d5d2fa1e4df5042f97f8be&quot;}&#x27; &#x27;++tokenizer.hf_tokenizer_kwargs.revision=&quot;476b36a473d4467f94469414bef6cee75c9c8172&quot;&#x27; dataset.hf_output_root=work/data/tulu3/moonlight-16b-a3b-peft-b14afda60f1b dataset.hf_rewrite=true dataset.seed=1234 rng.seed=5678 dataset.do_validation=false dataset.hf_validation_proportion=null dataset.enable_offline_packing=true +dataset.offline_packing_specs.pad_seq_to_mult=4 scheduler.lr_decay_iters=100 validation.eval_iters=0 validation.eval_interval=0 checkpoint.load=null --save_dir work/model-verification/moonlight-16b-a3b/peft-convergence-v1-checkpoints --save_interval 100 logger.log_interval=1 logger.log_throughput=true</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The immutable-revision 4-GPU run completes exactly 100 PEFT steps at TP1/PP1/CP1/EP4/ETP1, DP4, SP off, GBS/MBS 32/1, and eight-way gradient accumulation. Pad-4 offline packing is 99.29% efficient, and the sampled 6,553,600 token slots contain 4,291,091 supervised tokens after label masking. Only rank-8, alpha-16, zero-dropout LoRA on linear_q_proj, linear_kv_down_proj, linear_kv_up_proj, and linear_proj is trainable. LM loss is finite from 1.264324 to 1.100755 with no skipped or NaN iterations, all five metrics are recorded, and the complete four-shard iter_0000100 adapter checkpoint covers all 216 expected adapter entries.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
