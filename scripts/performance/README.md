# Performance Recipes

## NOTE: This directory will change a lot over the coming weeks

New runs of exact exported flat recipes should use `scripts/training/train.sh --recipe <function_name>`. The launcher
discovers text pretraining, text SFT/PEFT, Qwen-VL pretraining, and Wan pretraining recipes automatically and selects
their forward step. This directory remains the compatibility path for selector-based invocation, dataset replacement,
topology resizing, and specialized benchmark controls. The training launcher preserves total GPU-count validation,
the recipe process environment, and mock-data defaults for text SFT/PEFT; it does not inject offline defaults. The
performance compatibility launcher continues to own its benchmark offline environment.
Cluster-specific CPU/NUMA binding, Slurm segment sizing, NCCL fabric settings, and `srun` arguments remain user
supplied.

- Scripts defined in `scripts/performance` launch performance-optimized experiments on Slurm-based clusters.

## Performance recipe configs

Performance-optimized recipes live in `src/megatron/bridge/perf_recipes`. The performance
launcher resolves recipes from that package by model, task, GPU count, GPU type, precision,
and config variant.

`setup_experiment.py` launches `bootstrap.py` on each rank. The bootstrap resolves and
applies recipe-owned process settings before importing the training loop, then replaces itself with
either `run_script.py` for flat performance recipes or `run_recipe.py` for model recipes.
Each training entrypoint therefore executes only once.

- Prefer command-line overrides for one-off changes.
- Add or update flat perf recipe functions in `src/megatron/bridge/perf_recipes` for reusable benchmark configs.

## Setup Instructions

Follow the steps below on a Slurm based login node to launch Megatron-Bridge experiments.

### Step 1. Clone Megatron-Bridge Repo

We need to clone the repo to access and run performance benchmarking experiments using `Megatron-Bridge/scripts/performance/setup_experiment.py` (more details in Step 2.).

`setup_experiment.py` uses [NeMo/Run](https://github.com/NVIDIA-NeMo/Run) to launch experiments. This script generates and runs a sbatch script. The experiment is ultimately run on compute node(s) inside a container specified by the user. The experiment uses Megatron-Bridge code that comes pre-packaged with the container.

```
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
```

Next, we need to switch to a branch that was used to build the container you want to use for your experiments.

```
cd Megatron-Bridge
git switch <branch> 
```
Example: If using 26.04 container, then execute- `git switch r0.4.0`

### Step 2. Run instructions

#### <ins>Examples</ins>

The following line shows an example of how you can launch a pre-training benchmark/experiment-

`uv run python scripts/performance/setup_experiment.py --account <your_slurm_account> --partition <your_slurm_partition> --gpu gb200 --model_family_name <model name> --model_recipe_name <model_recipe_name> -ng <num gpus>`

You can also create a bash file to define the experiment arguments and launch it. For e.g. The bash file will look as follows-

```
CONTAINER="nvcr.io/nvidia/nemo:26.04"
MBRIDGE_PATH="</path/to/mbridge>"

JOB_NAME="dsv3_gb300"
RESULTS_DIR="${MBRIDGE_PATH}/results/${JOB_NAME}"

uv run python scripts/performance/setup_experiment.py \
  --account <slurm_account> \
  -i ${CONTAINER} \
  --partition <slurm_partition> \
  -m deepseek \
  -mr deepseek_v3 \
  --log_dir ${RESULTS_DIR} \
  --num_gpus 256 \
  --gpus_per_node 4 \
  -t "00:15:00" \
  -g gb300 \ 
  -c fp8_mx \
  -hf <HF_TOKEN>
```


  Generate your personal HuggingFace Access Token from <https://huggingface.co/settings/tokens/new?>

#### <ins>Mandatory arguments</ins>
- `-m/--model_family_name`
- `-mr/--model_recipe_name`
- `-ng/--num_gpus`
- `-g/--gpu`
- `-a/--account` (Mandatory for Slurm based clusters)
- `-p/--partition` (Mandatory for Slurm based clusters)

#### <ins>Configuration Options</ins>

##### Container Image

- `-i/--container_image`: NeMo container image to launch. For release container XX.YY use nvcr.io/nvidia/nemo:XX.YY.
  For 26.04, use nvcr.io/nvidia/nemo:26.04. For the complete list of NGC containers refer <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags>.
  Defaults to `nvcr.io/nvidia/nemo:dev`.

##### General arguments

- `-m/--model_family_name`: Model family name to use for experiment. E.g. `llama` (not llama3).
- `-mr/--model_recipe_name`: Model recipe name to use for experiment. E.g. `llama31_405b`.
- `--use_recipes`: Use model recipes instead of flat performance recipes. Disabled by default.
- `-nh/--nemo_home`: Directory to expose as `NEMO_HOME` on the compute node. Defaults to `~/.cache/nemo`.
- `--detach`: Detach the experiment from the terminal. Pass `true` or `false`. Default `true`.
- `--max_retries`: Maximum number of retries. Default `2`.
- `-ng/--num_gpus`: Number of GPUs.
- `-d/--dryrun`: Print the generated `sbatch` script without launching.

##### Training arguments

- `--task`: Workflow to run (`pretrain`, `sft`, `peft`). Default `pretrain`.
- `-ms/--max_steps`: Maximum number of training steps.
- `-gb/--global_batch_size`: Override global batch size.
- `-mb/--micro_batch_size`: Override micro-batch size.
- `-sl/--seq_length`: Override model sequence length and the LLM dataset sequence length.

##### Optimizer arguments

- `--lr`: Learning rate.
- `--min_lr`: Minimum learning rate.
- `--warmup_iters`: Warmup iterations. Default `10`.

##### Checkpointing arguments

- `--pretrained_checkpoint`: Path to pretrained checkpoint.
- `--save_dir`: Directory to save checkpoints.
- `--load_dir`: Directory to load checkpoints.
- `--save_interval`: Number of iterations between checkpoint saves.
- `--most_recent_k`: Number of latest checkpoints to keep.

##### Data arguments

- `--data`: Dataset type to use (`mock`, `rp2`, `squad`, `squad_packed`). Default `mock`.
- `--dataset_paths`: Dataset paths (for rp2 dataset). Accepts multiple paths.
- `--dataset_root`: Dataset root directory (for squad datasets).
- `--index_mapping_dir`: Index mapping directory (for rp2 dataset).
- `--dataset_name`: Dataset name (deprecated).
- `--packed_sequence`: Enable packed sequences.
- `--head_only`: Use only head data (for rp2 dataset).

##### Tokenizer arguments

- `--tokenizer_type`: Tokenizer type (`NullTokenizer`, `HuggingFaceTokenizer`, `SentencePieceTokenizer`).
- `--tokenizer_model`: Path to tokenizer model (automatically provided by launcher).
- `--vocab_size`: Vocabulary size for NullTokenizer. Default `32000`.
- `-hf/--hf_token`: HuggingFace token for accessing tokenizers and checkpoints.
  - User can generate a token from- huggingface.co/settings/tokens (click on "Create new token" button)
  - For a "Fine-grained" token, only "User permissions" are needed. Under "User permissions", make selections for "Repositories", "Webhooks" and "Collections".
- `--offline`: Set `HF_HUB_OFFLINE=1` (Slurm launcher path).
  - Cannot be used together with `--hf_token`.

##### HuggingFace connectivity and cache behavior (Slurm launcher)

This launcher uses split defaults:

- `TRANSFORMERS_OFFLINE=1`
- `HF_HUB_OFFLINE=0`

What each variable controls in this workflow:

- `TRANSFORMERS_OFFLINE`: Transformers calls (for example `AutoTokenizer`) stay offline unless `--hf_token` is provided.
- `HF_HUB_OFFLINE`: HuggingFace Hub calls (for example Hub-backed config/model resolution such as `AutoConfig`) stay online unless `--offline` is provided.

Why this split exists:

- Most benchmark recipes use `NullTokenizer`, so `TRANSFORMERS_OFFLINE=1` avoids unnecessary network traffic.
- Most performance model families (`llama`, `qwen`, `qwen_vl`, `deepseek`, `gpt_oss`) use HF-backed config/model lookup paths.

Flag mapping:

- `--hf_token` sets `HF_TOKEN` and `TRANSFORMERS_OFFLINE=0`.
- `--offline` sets `HF_HUB_OFFLINE=1`.
- `--hf_token` and `--offline` are mutually exclusive.

Practical guidance:

1. Prefetch required model/tokenizer/config files into a local HF cache.
2. Mount that cache into the container with `-cm/--custom_mounts`.
3. Set `HF_HOME` to that mounted cache path before launch (Slurm exports env vars by default), for example `export HF_HOME=/path/to/hf_cache`.
4. If needed, explicitly override `HF_HOME` with `-ce/--custom_env_vars`.
5. Pass `--offline` to block Hub network checks.

Mounting cached files is not enough by itself. If `HF_HUB_OFFLINE` remains `0`, Hub-backed code paths may still perform network checks and hit HuggingFace rate limits.

##### Parallelism arguments

- `-tp/--tensor_model_parallel_size`: Tensor parallel degree. Intra-layer model parallelism; splits tensors across GPU ranks.
- `-pp/--pipeline_model_parallel_size`: Pipeline parallel degree. Inter-layer model parallelism; splits transformer layers across GPU ranks.
- `-cp/--context_parallel_size`: Context parallel degree. Splits network input along sequence dimension across GPU ranks.
- `-vp/--virtual_pipeline_model_parallel_size`: Number of virtual blocks per pipeline model parallel rank. Accepts `None` or an integer value.
- `-ep/--expert_model_parallel_size`: MoE expert parallel degree. Distributes MoE experts across sub data parallel dimension.
- `-et/--expert_tensor_parallel_size`: Expert tensor parallel degree. Intra-layer tensor model parallelism for expert layer. Use `-et` (no value) for `None` or `-et <int>`.

##### Slurm launcher behavior

- The launcher always adds `--container-writable` to `srun`.
- This avoids benchmark failures on clusters using Enroot defaults, where `ENROOT_ROOTFS_WRITABLE=no`.

##### Slurm arguments

- `-a/--account`: Slurm account to use for experiment.
- `-p/--partition`: Slurm partition to use for experiment.
- `-t/--time_limit`: Maximum time limit before the Slurm job is cancelled. Format `HH:MM:SS`. Default `00:30:00`.
- `-gn/--gpus_per_node`: GPUs per node. Default `None`. If not provided, it is inferred from the GPU type.
- `-cm/--custom_mounts`: Comma-separated list of host mounts to expose inside the container.
- `-ce/--custom_env_vars`: Comma-separated string of environment variables (format: `key1=value1,key2=value2`).
- `-E/--env`: Set environment variable (repeatable arg). This is an alternative to `--custom_env_vars`. (`--custom_env_vars` is preferred for most cases). Example: `-E var1=value1,value2 -E var2=value3"`.
- `-cs/--custom_srun_args`: Comma-separated string of srun arguments.
- `--gres`: Slurm generic resources to request (e.g., `gpu:4`).
- `--additional_slurm_params`: Additional SLURM parameters as key=value pairs. Use semicolons (`;`) to separate parameters when values contain commas. Examples: `nodelist=node001,node002;constraint=gpu` or `reservation=my_res;exclusive`.
- `--packager`: How code is packaged for the job. `git` snapshots the repo at submission time (default). `none` skips snapshotting — use when code is pre-installed in the container image or available via a shared filesystem.

##### Kubeflow arguments

- `--kubeflow_namespace`: Kubernetes namespace for the Kubeflow TrainJob. Setting this routes the experiment through the Kubeflow executor instead of Slurm.
- `--csp`: cloud provider whose fabric plugin to apply to the Kubeflow executor — `aws` applies `EKSEnvPlugin` (EFA: `FI_PROVIDER=efa`, `FI_EFA_USE_HUGE_PAGE=0`, EFA device requests + privileged container) and `gcp` applies `GKEEnvPlugin` (gIB RDMA-NIC pod annotations). No-op for the Slurm executor.
- `--kubeflow_workdir_pvc`: PVC name for syncing the job workdir (launch scripts, packaged code) into the cluster before launch.
- `--kubeflow_workdir_pvc_path`: Mount path for the workdir PVC inside the training pod. Default `/nemo_run`.
- `--kubeflow_workdir_local_path`: Local directory whose contents nemo-run's `KubeflowExecutor.package()` rsyncs into the workdir PVC via a temporary alpine pod before launch. Used to overlay a `--mbridge-ref` checkout onto `/opt/Megatron-Bridge` in the trainer container without rebuilding the image.
- `--kubeflow_image_pull_secrets`: Comma-separated list of Kubernetes image pull secret names.
- `--kubeflow_volumes_json`: JSON-encoded list of Kubernetes `Volume` dicts attached to the training pod (PVC, emptyDir, hostPath).
- `--kubeflow_volume_mounts_json`: JSON-encoded list of Kubernetes `VolumeMount` dicts applied to the training container (must match a name in `--kubeflow_volumes_json`).
- `--kubeflow_tolerations_json`: JSON-encoded list of Kubernetes `Toleration` dicts applied to the training pods (e.g. to land on lease-tainted nodes such as `gpu-wrangler.nvidia.com/lease`).
- `--kubeflow_affinity_json`: JSON-encoded Kubernetes `Affinity` dict applied to the training pods (e.g. node affinity onto GPULease-allocated nodes).
- `--kubeflow_env_list_json`: JSON-encoded list of Kubernetes `EnvVar` dicts (supports `valueFrom.secretKeyRef` for secret-backed env vars such as `WANDB_API_KEY` / `HF_TOKEN`).
- `--kubeflow_extra_resource_requests_json`: JSON-encoded dict of extra container resource requests (e.g. `{"vpc.amazonaws.com/efa": "32"}` for EFA on AWS).
- `--kubeflow_extra_resource_limits_json`: JSON-encoded dict of extra container resource limits (paired with the requests above).
- `--kubeflow_pod_spec_overrides_json`: JSON-encoded dict merged into the pod spec — escape hatch for `nodeSelector`, `hostNetwork`, etc.
- `--kubeflow_container_kwargs_json`: JSON-encoded dict of extra fields set on the training container (e.g. `{"securityContext": {"privileged": true}}` for EFA / RDMA).
- `--kubeflow_pod_annotations_json`: JSON-encoded dict of annotations applied to the trainer pod template metadata (e.g. GKE `networking.gke.io/interfaces` to attach the RDMA NICs for gIB). Usually set for you by `--csp gcp`.
- `--kubeflow_labels_json`: JSON-encoded dict of labels applied to the TrainJob's pods.

##### Performance arguments

- `-g/--gpu`: Target GPU type (`h100`, `b200`, `gb200`, `gb300`, `b300`).
- `-c/--compute_dtype`: Compute precision (`bf16`, `fp8_cs`, `fp8_mx`, `fp8_sc`, `nvfp4`). Default `bf16`.
- `-vb/--enable_vboost`: Enable VBoost (tensor core power steering). Pass `true` or `false`. Disabled by default.
- `-lgc/--lock_gpu_freq`: Lock GPU graphics clock to a fixed frequency in MHz (e.g. `1200`). Used for silicon simulation correlation studies. Disabled by default.
- `-lmc/--peak_mem_clk`: Lock GPU memory clock to a fixed peak frequency in MHz (e.g. `2600`). Used for silicon simulation correlation studies. Defaults to `4752` MHz for VR200 and is disabled by default for other GPUs. Pass `-lmc -1` or `--peak_mem_clk -1` to disable the VR200 default.
- `-en/--enable_nsys`: Enable Nsight Systems profiling. Disabled by default.
- `-pyp/--pytorch_profiler`: Enable PyTorch profiler. Pass `true` or `false`. Disabled by default.
- `--profiling_start_step`: Defines start step for profiling. Default `10`.
- `--profiling_stop_step`: Defines stop step for profiling. Default `11`.
- `-mh/--record_memory_history`: Enable PyTorch profiler memory history recording. Pass `true` or `false`. Enabled by default (if pytorch_profiler is enabled).
- `--profiling_gpu_metrics`: Enable nsys GPU metrics. Disabled by default.
- `--profiling_ranks`: Comma-separated list of ranks to target for profiling. Defaults to just the first rank.
- `--use_tokendrop`: Enable token drop (currently DeepSeek v3 only). Pass `true` or `false`. Disabled by default.
- `--use_megatron_fsdp`: Enable Megatron FSDP integration. Pass `true` or `false`. Disabled by default.
- `--nccl_ub`: Enable NCCL user buffer for FSDP communication. Pass `true` or `false`. Disabled by default.
- `--cuda_graph_impl`: CUDA graph implementation (`none`, `local`, `transformer_engine`).
- `--cuda_graph_scope`: CUDA graph capture scope (`full_iteration`, `attn`, `mlp`, `moe`, `moe_router`, `moe_preprocess`, `mamba`). Comma-separated list of scopes is allowed.
- `--moe_a2a_overlap`: Set the `moe_a2a_overlap` configuration flag. Pass `true` or `false`.
- `-rl/--recompute_num_layers`: Number of transformer layers to recompute (intermediate activations).
- `-ol/--activation_offload_layers`: Number of transformer layers to offload activations to CPU memory.
- `--recompute_modules`: Comma-separated list of modules to recompute.

##### Logging arguments

- `-l/--log_dir`: Directory for logging experiment results. Defaults to `NEMORUN_HOME`.
  - Make sure the environment variable `NEMORUN_HOME=<log_dir>` is accessible and set correctly in your virtual environment.
  - You can run `export NEMORUN_HOME=<log_dir>` in your terminal. You can add it your bashrc file (or equivalent for your OS/Linux distro) for setting it permanently.
- `-wdk/--wandb_key`: Weights & Biases API key for remote logging.
- `-wdp/--wandb_project_name`: Weights & Biases project name.
- `-wde/--wandb_entity_name`: Weights & Biases entity name.
- `-wdj/--wandb_experiment_name`: Weights & Biases experiment/run name.
- `-wds/--wandb_save_dir`: Weights & Biases save directory.
- - `--save_config_filepath`: Path to save the task configuration file.

##### Config variant arguments

- `-cv/--config_variant`: Config variant to use. Omit to use the suffix-less canonical flat perf recipe. Named variants such as `"large_scale"` are supported when a matching flat recipe exists. Use `--list_config_variants` to see available options.
- `--list_config_variants`: List available config variants for the specified model/task/gpu/dtype and interactively select one (with 15s timeout).

##### Testing arguments

- `--is_long_convergence_run`: If set, runs a long convergence run.
- `--golden_values_path`: Path to golden values file.
- `--timing_threshold`: Step timing validation threshold. Default `0.05` (5%).
- `--skip_first_percent_time`: Percentage of iterations to skip for timing comparison. Default `0.70` (70%).
- `--correlation_threshold`: Correlation threshold for loss curve validation. Default `0.95`.
- `--high_loss_tolerance`: Tolerance for high loss values (>2.0). Default `0.10`.
- `--medium_loss_tolerance`: Tolerance for medium loss values (0.5-2.0). Default `0.05`.
- `--low_loss_tolerance`: Tolerance for low loss values (<0.5). Default `0.02`.
- `--final_loss_tolerance`: Tolerance for final loss value. Default `0.05`.
- `--max_outlier_ratio`: Maximum ratio of outliers allowed. Default `0.1`.
- `--outlier_threshold`: Outlier detection threshold (sigma). Default `3.0`.
- `--skip_first_percent_loss`: Percentage of loss points to skip from beginning for convergence analysis. Default `0.20` (20%).

## Determinism

Deterministic training guarantees that two runs with identical inputs produce identical outputs at every step.  It is useful for debugging (isolating regressions) and for reproducibility studies.

### What `--deterministic` does

**Environment variables** (stored in `cfg.env_vars` and applied before the training process imports Torch):

| Variable | Value | Reason |
|---|---|---|
| `NCCL_ALGO` | `Ring` | Disables tree/NVLink collectives that are non-deterministic |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | Forces TE to use deterministic algorithms |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Disables cuBLAS heuristic workspace selection |

**Model config overrides** (applied by `apply_determinism_overrides` in the recipe layer):

| Field | Value |
|---|---|
| `model.deterministic_mode` | `True` |
| `model.cross_entropy_loss_fusion` | `False` |
| `comm_overlap.tp_comm_overlap` | `False` |

### Example commands

```bash
# Llama 3 70B — deterministic, H100 64-GPU
python scripts/performance/setup_experiment.py \
  --account <account> --partition <partition> \
  --gpu h100 -m llama3 -s 70b -ng 64 -gn 8 \
  --container_image <image> --task pretrain \
  --deterministic

# Llama 3.1 405B — deterministic, H100 512-GPU
python scripts/performance/setup_experiment.py \
  --account <account> --partition <partition> \
  --gpu h100 -m llama31 -s 405b -ng 512 -gn 8 \
  --container_image <image> --task pretrain \
  --deterministic
```

### Using model recipes directly

`apply_determinism_overrides` is also importable for use outside the performance script layer:

```python
from megatron.bridge.recipes.llama.h100 import llama3_70b_pretrain_32gpu_h100_bf16_deterministic_config

cfg = llama3_70b_pretrain_32gpu_h100_bf16_deterministic_config()

# Or, apply overrides to any existing recipe:
from megatron.bridge.recipes.utils import apply_determinism_overrides
from megatron.bridge.recipes.llama.h100 import llama3_70b_pretrain_32gpu_h100_bf16_config

cfg = llama3_70b_pretrain_32gpu_h100_bf16_config()
apply_determinism_overrides(cfg)
```

`apply_determinism_overrides(cfg)` adds both the model overrides and these runtime environment defaults to the recipe.
Explicit shell or launcher environment values retain precedence when the recipe is launched.
