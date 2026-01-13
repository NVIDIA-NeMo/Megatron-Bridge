# Performance Recipes

## NOTE: This directory will change a lot over the coming weeks

- Scripts defined in `scripts/performance` are recipes optimized for performance. These scripts can launch pre-training experiments on Slurm based clusters.

## Configuration files

There are configuration files- `workload_base_configs.py` for supported models in `scripts/performance/configs`.

- You can override the default configs using these files using command line arguments (recommended) or directly updating these files  

## Setup Instructions

### Step 1. Virtual Environment

- Create a virtual env at your preferred location on login node on a Slurm cluster and install the NeMo-Run package-

  ```
  pip install git+https://github.com/NVIDIA-NeMo/Run.git
  ```

- The YAML config files are resolved on compute node inside the container.

### Step 2. Clone the Repo and Pick the corresponding release branch to the container

  ```
  git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
  git switch <branch> 
  Example: If using 25.11 Container ```git switch r0.2.0
  ```
  
  To find out which branch is used to build the container, refer <https://docs.nvidia.com/nemo-framework/user-guide/latest/softwarecomponentversions.html>

  Why? This is required because when running a job the version of Megatron-Bridge in the setup and the one built into the container should match.

## Step 3. Run instructions

The following line shows an example of how you can launch a pre-training benchmark/experiment-

`python scripts/performance/setup_experiment.py --account <your_slurm_account> --partition <your_slurm_partition> --gpu gb200 --model_family_name <model name> --model_recipe_name <model_recipe_name> -ng <num gpus>`

#### Mandatory arguments
- `-m/--model_family_name`
- `-mr/--model_recipe_name`
- `-ng/--num_gpus`
- `-g/--gpu`
- `-a/--account` (Mandatory for Slurm based clusters)
- `-p/--partition` (Mandatory for Slurm based clusters)

### Configuration Options

#### Container Image

- `-i/--container_image`: NeMo container image to launch. For release container XX.YY use nvcr.io/nvidia/nemo:XX.YY.
  For 25.09, use nvcr.io/nvidia/nemo:25.09. For the complete list of NGC containers refer <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags>.
  Defaults to `nvcr.io/nvidia/nemo:dev`.

#### General arguments

- `-m/--model_family_name`: Model family name to use for experiment. E.g. `llama` (not llama3).
- `-mr/--model_recipe_name`: Model recipe name to use for experiment. E.g. `llama31_405b`.
- `--use_recipes`: Use library recipes. Disabled by default.
- `-nh/--nemo_home`: Directory to expose as `NEMO_HOME` on the compute node. Defaults to `~/.cache/nemo`.
- `--detach`: Detach the experiment from the terminal. Pass `true` or `false`. Default `true`.
- `--max_retries`: Maximum number of retries. Default `2`.
- `-ng/--num_gpus`: Number of GPUs.
- `-d/--dryrun`: Print the generated `sbatch` script without launching.

#### Training arguments

- `--task`: Workflow to run (`pretrain`, `sft`, `lora`). Default `pretrain`.
- `-ms/--max_steps`: Maximum number of training steps.
- `-gb/--global_batch_size`: Override global batch size.
- `-mb/--micro_batch_size`: Override micro-batch size.
- `-sl/--seq_length`: Sequence length.

#### Optimizer arguments

- `--lr`: Learning rate.
- `--min_lr`: Minimum learning rate.
- `--warmup_iters`: Warmup iterations. Default `10`.

#### Checkpointing arguments

- `--pretrained_checkpoint`: Path to pretrained checkpoint.
- `--save_dir`: Directory to save checkpoints.
- `--load_dir`: Directory to load checkpoints.
- `--save_interval`: Number of iterations between checkpoint saves.
- `--most_recent_k`: Number of latest checkpoints to keep.
- `--save_config_filepath`: Path to save the task configuration file.

#### Data arguments

- `--data`: Dataset type to use (`mock`, `rp2`, `squad`, `squad_packed`). Default `mock`.
- `--dataset_paths`: Dataset paths (for rp2 dataset). Accepts multiple paths.
- `--dataset_root`: Dataset root directory (for squad datasets).
- `--index_mapping_dir`: Index mapping directory (for rp2 dataset).
- `--dataset_name`: Dataset name (deprecated).
- `--packed_sequence`: Enable packed sequences.
- `--head_only`: Use only head data (for rp2 dataset).

#### Tokenizer arguments

- `--tokenizer_type`: Tokenizer type (`NullTokenizer`, `HuggingFaceTokenizer`, `SentencePieceTokenizer`).
- `--tokenizer_model`: Path to tokenizer model (automatically provided by launcher).
- `--vocab_size`: Vocabulary size for NullTokenizer. Default `32000`.
- `-hf/--hf_token`: HuggingFace token for accessing tokenizers and checkpoints.
  - User can generate a token from- huggingface.co/settings/tokens (click on "Create new token" button)
  - For a "Fine-grained" token, only "User permissions" are needed. Under "User permissions", make selections for "Repositories", "Webhooks" and "Collections".

#### Parallelism arguments

- `-tp/--tensor_model_parallel_size`: Tensor parallel degree. Intra-layer model parallelism; splits tensors across GPU ranks.
- `-pp/--pipeline_model_parallel_size`: Pipeline parallel degree. Inter-layer model parallelism; splits transformer layers across GPU ranks.
- `-cp/--context_parallel_size`: Context parallel degree. Splits network input along sequence dimension across GPU ranks.
- `-vp/--virtual_pipeline_model_parallel_size`: Number of virtual blocks per pipeline model parallel rank. Accepts `None` or an integer value.
- `-ep/--expert_model_parallel_size`: MoE expert parallel degree. Distributes MoE experts across sub data parallel dimension.
- `-et/--expert_tensor_parallel_size`: Expert tensor parallel degree. Intra-layer tensor model parallelism for expert layer. Use `-et` (no value) for `None` or `-et <int>`.

#### Slurm arguments

- `-a/--account`: Slurm account to use for experiment.
- `-p/--partition`: Slurm partition to use for experiment.
- `-t/--time_limit`: Maximum time limit before the Slurm job is cancelled. Format `HH:MM:SS`. Default `00:30:00`.
- `-gn/--gpus_per_node`: GPUs per node. Default `8`.
- `-cm/--custom_mounts`: Comma-separated list of host mounts to expose inside the container.
- `-ce/--custom_env_vars`: Comma-separated string of environment variables (format: `key1=value1,key2=value2`).
- `-cs/--custom_srun_args`: Comma-separated string of srun arguments.
- `--gres`: Slurm generic resources to request (e.g., `gpu:4`).
- `--additional_slurm_params`: Additional SLURM parameters as key=value pairs. Use semicolons (`;`) to separate parameters when values contain commas. Examples: `nodelist=node001,node002;constraint=gpu` or `reservation=my_res;exclusive`.

#### DGXCloud arguments

- `--dgxc_cluster`: DGXCloud cluster to use for experiment.
- `--dgxc_base_url`: DGXCloud base URL.
- `--dgxc_kube_apiserver_url`: DGXCloud Kube API server URL.
- `--dgxc_app_id`: DGXCloud app ID.
- `--dgxc_app_secret`: DGXCloud app secret.
- `--dgxc_project_name`: DGXCloud project name.
- `--dgxc_pvc_claim_name`: DGXCloud PVC claim name.
- `--dgxc_pvc_mount_path`: DGXCloud PVC mount path.

#### Performance arguments

- `-g/--gpu`: Target GPU type (`h100`, `b200`, `gb200`, `gb300`).
- `-c/--compute_dtype`: Compute precision (`bf16`, `fp8_cs`, `fp8_mx`, `fp8_sc`, `nvfp4`). Default `bf16`.
- `-vb/--enable_vboost`: Enable VBoost (tensor core power steering). Pass `true` or `false`. Disabled by default.
- `-en/--enable_nsys`: Enable Nsight Systems profiling. Disabled by default.
- `--profiling_start_step`: Defines start step for profiling. Default `10`.
- `--profiling_stop_step`: Defines stop step for profiling. Default `11`.
- `--profiling_gpu_metrics`: Enable nsys GPU metrics. Disabled by default.
- `--profiling_ranks`: Comma-separated list of ranks to target for profiling. Defaults to just the first rank.
- `--use_tokendrop`: Enable token drop (currently DeepSeek v3 only). Pass `true` or `false`. Disabled by default.
- `--use_megatron_fsdp`: Enable Megatron FSDP integration. Pass `true` or `false`. Disabled by default.
- `--cuda_graph_impl`: CUDA graph implementation (`none`, `local`, `transformer_engine`).
- `--cuda_graph_scope`: CUDA graph capture scope (`full_iteration`, `attn`, `mlp`, `moe`, `moe_router`, `moe_preprocess`, `mamba`). Comma-separated list of scopes is allowed.
- `--moe_a2a_overlap`: Set the `moe_a2a_overlap` configuration flag. Pass `true` or `false`.
- `-rl/--recompute_num_layers`: Number of transformer layers to recompute (intermediate activations).
- `-ol/--activation_offload_layers`: Number of transformer layers to offload activations to CPU memory.
- `--recompute_modules`: Comma-separated list of modules to recompute.

#### Logging arguments

- `-l/--log_dir`: Directory for logging experiment results. Defaults to `NEMORUN_HOME`.
  - Make sure the environment variable `NEMORUN_HOME=<log_dir>` is accessible and set correctly in your virtual environment.
  - You can run `export NEMORUN_HOME=<log_dir>` in your terminal. You can add it your bashrc file (or equivalent for your OS/Linux distro) for setting it permanently.
- `-wdk/--wandb_key`: Weights & Biases API key for remote logging.
- `-wdp/--wandb_project_name`: Weights & Biases project name.
- `-wde/--wandb_entity_name`: Weights & Biases entity name.
- `-wdj/--wandb_experiment_name`: Weights & Biases experiment/run name.
- `-wds/--wandb_save_dir`: Weights & Biases save directory.

#### Testing arguments

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
