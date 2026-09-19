# GLM-5

[GLM-5](https://huggingface.co/zai-org/GLM-5), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1), [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), and [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) use the shared `GLM5Bridge` for their MoE, Multi-Latent Attention, and Dynamic Sparse Attention architecture. GLM-5.3 shares GLM-5.2's architecture and import mappings; full-model GLM-5.3 verification remains pending.


## GLM-5.2 / GLM-5.3 compatibility

The pinned publisher configs for [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json) and [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/config.json) have identical architecture fields: 78 decoder layers, 256 routed experts with top-8 routing, MLA, and the same DSA IndexShare pattern. All 59,585 base checkpoint tensor names and shapes also match. AutoBridge resolves both to `GlmMoeDsaForCausalLM` and `GLM5Bridge`; no separate provider is needed. **GLM-5.3-Flash is a different architecture and is outside this support statement.**

The checkpoints have important differences:

- **Precision:** GLM-5.2 stores BF16 weights (plus FP32 router biases). GLM-5.3 stores 59,044 weights as E4M3 FP8 with FP32 scales per 128×128 block. The existing GLM bridge dequantizes these weights to BF16 on import. This is distinct from enabling FP8 training.
- **Config metadata:** GLM-5.3 adds `quantization_config` and records Transformers 5.15.0 instead of 5.12.0. Use the repository's supported dependency versions.
- **Tokenizer and generation defaults:** `tokenizer.json`, `tokenizer_config.json`, and `generation_config.json` are identical at those revisions, including token IDs and sampling defaults.
- **Chat formatting:** GLM-5.3's template adds low reasoning effort, retains historical reasoning by default, always opens a thinking response (it does not honor `enable_thinking=False`), and changes structured tool-result handling and ordering. Load the template from the GLM-5.3 checkpoint; do not substitute GLM-5.2's template.

For config/provider loading without weights:

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(
    "zai-org/GLM-5.3", revision="aca966e4e02791568aa6a4ced368624b3d897f42"  # pragma: allowlist secret (public HF revision)
)
provider = bridge.to_megatron_provider(load_weights=False)
```

For **BF16 export** using the FP8 source as the reference, explicitly select `--export-weight-dtype bfloat16` with the GPU or distributed-CPU conversion backend (API: `weight_dtype=torch.bfloat16`). This removes FP8 scale tensors from strict source-key validation and writes a BF16 artifact rather than claiming a bitwise FP8 round trip. The serial-CPU backend does not support this option. MTP is disabled by default; the appended MTP layer is outside that default inference graph.

### Verification limits

The recorded commands and results below belong to the checkpoint named in each verification card. GLM-5.2 results do **not** establish GLM-5.3 full-checkpoint conversion/export, forward parity, generation, SFT/PEFT, resume, long-context behavior, or performance. Those GLM-5.3 checks remain unverified. The `glm52_*` recipes are pinned to GLM-5.2 and retain that name and attribution; changing only their display name would not select GLM-5.3 weights or its chat template.

[Production qualification tracker #5476](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5476) remains open. It includes repeated-MTP training semantics, precision controls, dynamic context parallelism, production recipes, checkpoint continuity, convergence, and hardware-specific performance qualification. See also the open [export context-length fix #5996](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5996), [router-bias fix #6036](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6036), and [recipe-selection fix #5608](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5608).

## HybridModel migration

The shared GLM-5/5.1/5.2/5.3 bridge constructs `HybridModel` through
`GLM5ModelProvider`. The outer language model is Core's native `HybridModel`;
GLM-specific runtime adaptations are supplied through the stack specification.
Each HF block becomes a `D-` (dense) or `DE` (MoE) pair;
78 HF blocks therefore become 156 physical Hybrid layers. Core's native
`DSAttention` is used directly; the stack translates the HF index-sharing cadence
into physical layer coordinates on private per-layer config copies. HF export retains the
original layer count, and MTP uses a separate `/DE` pattern when enabled.

Import the pinned HF checkpoint into a **new** Hybrid checkpoint directory.
The generic Core GPT-to-Hybrid converter does not support this DSA architecture;
old GPT optimizer/checkpoint resume is outside this migration. `get_model_config()`
is not supported for this family yet; use `to_megatron_provider()`.

The initial recipes use BF16 and non-VPP pipeline segments. Pattern boundaries
must begin on DSA compute layers. Full recomputation replays an entire pipeline
stage to preserve forward-local top-k sharing, so memory and throughput must be
remeasured. Selective `core_attn` recomputation is rejected because its backward
replay does not preserve the forward-local DSA sharing state. GPU round-trip,
parity, MTP, packed CP, training and Hybrid checkpoint
resume require fresh verification. Historical GPT results do not verify Hybrid.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-glm5"></a>
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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="glm5-hf-to-megatron-cpu" aria-controls="glm5-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="glm5-hf-to-megatron-gpu" aria-controls="glm5-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="glm5-megatron-to-hf-cpu" aria-controls="glm5-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="glm5-megatron-to-hf-gpu" aria-controls="glm5-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="glm5-pretrain-h100" aria-controls="glm5-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="glm5-sft-h100" aria-controls="glm5-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="glm5-sft-long-context-h100" aria-controls="glm5-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="glm5-peft-h100" aria-controls="glm5-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="glm5-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="glm5-hf-to-megatron-cpu" tabindex="-1">
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
        <p>CPU HF-to-Hybrid import must cover every enabled parameter and save a strictly reloadable checkpoint. Hybrid execution has not yet been verified.</p>
      </section>
    </article>
    <article id="glm5-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="glm5-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-19</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 4 --gpus-per-node 4 --hf-model zai-org/GLM-5 --hf-revision 4e6698ba8e85059d749020e3c4d2123719f23926 --megatron-path work/model-verification/glm5/hybrid/gpu-megatron --torch-dtype bfloat16 --tp 1 --pp 2 --ep 8 --etp 1 --distributed-timeout-minutes 60 --low-memory-save</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Distributed HF-to-Hybrid import on 16 GB200 GPUs maps all 6201 pinned source tensors through GLM5Bridge and saves a 1.4 TB distributed checkpoint (iter_0000000 with run_config.yaml) in about 10 minutes. The checkpoint is strictly reloadable, as exercised by the verified GPU export. The appended MTP layer stays disabled by default.</p>
      </section>
    </article>
    <article id="glm5-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="glm5-megatron-to-hf-cpu" tabindex="-1">
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
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 4 --cpu-processes-per-node 8 --cpus-per-task 16 --mem 0 --exclusive --hf-model zai-org/GLM-5 --hf-revision 4e6698ba8e85059d749020e3c4d2123719f23926 --megatron-path work/model-verification/glm5/hybrid/gpu-megatron/iter_0000000 --hf-path work/model-verification/glm5/hybrid/cpu-hf-export --torch-dtype bfloat16 --tp 1 --pp 2 --ep 8 --etp 2 --distributed-timeout-minutes 240 --distributed-save --save-every-n-ranks 1 --no-progress</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU Hybrid-to-HF export must match the pinned source keys, shapes, dtypes and values exactly within the enabled MTP scope. Hybrid execution has not yet been verified.</p>
      </section>
    </article>
    <article id="glm5-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="glm5-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-19</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 4 --gpus-per-node 4 --hf-model zai-org/GLM-5 --hf-revision 4e6698ba8e85059d749020e3c4d2123719f23926 --megatron-path work/model-verification/glm5/hybrid/gpu-megatron/iter_0000000 --hf-path work/model-verification/glm5/hybrid/gpu-hf-export --torch-dtype bfloat16 --tp 1 --pp 2 --ep 8 --etp 1 --distributed-timeout-minutes 60 --distributed-save --save-every-n-ranks 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Distributed Hybrid-to-HF export on 16 GB200 GPUs strictly reloads the imported checkpoint and writes 280 safetensors shards plus model.safetensors.index.json in about 10 minutes. All 59079 exported tensors match the pinned source keys, shapes, dtypes and values exactly; the 791 tensors of the appended MTP layer (model.layers.78.*) are excluded because MTP is disabled by default. generation_config.json is preserved from the source because Transformers strict validation rejects re-saving it.</p>
      </section>
    </article>
    <article id="glm5-pretrain-h100" class="verification-model-detail" data-entry-detail="glm5-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
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
        <p>Complete 100 real-data Hybrid pretraining steps with finite losses and zero skipped or NaN iterations, preserving the existing convergence contract. Save a full step-50 checkpoint and the post-setup config. Hybrid execution has not yet been verified.</p>
      </section>
    </article>
    <article id="glm5-sft-h100" class="verification-model-detail" data-entry-detail="glm5-sft-h100" tabindex="-1">
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
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Complete 100 real-data Hybrid full-SFT steps with finite losses, unchanged data/masking/packing semantics, and a full reloadable checkpoint. Hybrid execution has not yet been verified.</p>
      </section>
    </article>
    <article id="glm5-sft-long-context-h100" class="verification-model-detail" data-entry-detail="glm5-sft-long-context-h100" tabindex="-1">
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
        <p>Complete the hardware recipe long-context Hybrid workload with packed CP, finite losses and zero skipped or NaN iterations. Verify causal positions and top-k sharing before scaling up. Hybrid execution has not yet been verified.</p>
      </section>
    </article>
    <article id="glm5-peft-h100" class="verification-model-detail" data-entry-detail="glm5-peft-h100" tabindex="-1">
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
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Complete 100 Hybrid LoRA steps with the original targets and hyperparameters; reload the adapter checkpoint and verify the trainable parameter set. Hybrid execution has not yet been verified.</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
