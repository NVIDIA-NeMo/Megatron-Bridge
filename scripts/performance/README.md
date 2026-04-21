# Performance Recipes

## NOTE: This directory will change a lot over the coming weeks.


- Scripts defined in `scripts/performance` are recipes optimized for performance. These scripts can launch pre-training experiments on Slurm based clusters.

## Configuration files

There are configuration files- `workload_base_configs.py` for supported models in `scripts/performance/configs`.
- You can override the default configs using these files using command line arguments (recommended) or directly updating these files  

## Example

The following line shows an example of how you can launch a pre-training experiment-

`python scripts/performance/setup_experiment.py --account <your_slurm_account> --partition <your_slurm_partition> --gpu gb200 --model_name <model name> --model_size <model_size> -ng <num gpus>`

## Configuration Options

- Mandatory arguments: `-a/--account`, `-p/--partition`, `-g/--gpu` (choose `h100`, `b200`, `gb200`, or `gb300`), `-m/--model_name`, `-s/--model_size`, and `-ng/--num_gpus`.
- Optional arguments:
  - `-l/--log_dir`: Location to store experiment artifacts and logs. Defaults to `NEMORUN_HOME`.
    - Make sure the environment variable `NEMORUN_HOME=<log_dir>` is accessible and set correctly in your virtual environment.
    - You can run `export NEMORUN_HOME=<log_dir>` in your terminal. You can add it your bashrc file (or equivalent for your OS/Linux distro) for setting it permanently.
  - `-t/--time_limit`: Maximum time limit before the Slurm job is cancelled. Format `HH:MM:SS`. Default `00:30:00`.
  - `-i/--container_image`: NeMo container image to launch. Default `nvcr.io/nvidia/nemo:dev`.
  - `-c/--compute_dtype`: Training precision, either `bf16` or `fp8`. Default `bf16`.
  - `-fr/--fp8_recipe`: FP8 scaling recipe (`ds`, `cs`, `mx`, `sc`). Default `cs`.
  - `--task`: Workflow to run (`pretrain`, `sft`, `lora`). Default `pretrain`.
  - `-hf/--hf_token`: Hugging Face token for accessing gated tokenizers or checkpoints.
  - `-nh/--nemo_home`: Directory to expose as `NEMO_HOME` on the compute node. Defaults to `~/.cache/nemo` unless overridden.
  - `-wdk/--wandb_key`: Weights & Biases API key for remote logging.
  - `-wdp/--wandb_prj_name`: Weights & Biases project name.
  - `-wdj/--wandb_exp_name`: Weights & Biases experiment/run name.
  - `-d/--dryrun`: Print the generated `sbatch` script without launching.
  - `-gn/--gpus_per_node`: GPUs per node. Default `8`.
  - `-cm/--custom_mounts`: Comma-separated list of host mounts to expose inside the container.
  - `-vb/--enable_vboost`: Enable VBoost (tensor core power steering). Disabled by default.
  - `-en/--enable_nsys`: Enable Nsight Systems profiling for the configured window.
  - `--domain`: Domain preset (`llm`, `vlm`, `diffusion`). Default `llm`.
  - `--use_tokendrop`: Enable token drop (currently DeepSeek v3 only). Disabled by default.
  - `--use_megatron_fsdp`: Enable Megatron FSDP integration. Disabled by default.
  - `--cuda_graph_impl`: Select CUDA graph backend (`none`, `local`, `te`, `transformer_engine`).
  - `--cuda_graph_scope`: CUDA graph capture scope (`full`, `full_iteration`, `attn`). Default `full`.
  - `-tp/--tensor_model_parallel_size`: Tensor parallel degree.
  - `-pp/--pipeline_model_parallel_size`: Pipeline parallel degree.
  - `-cp/--context_parallel_size`: Context parallel degree.
  - `-vp/--virtual_pipeline_model_parallel_size`: Virtual pipeline chunks per pipeline rank.
  - `-ep/--expert_model_parallel_size`: MoE expert parallel degree.
  - `-et/--expert_tensor_parallel_size`: Expert tensor parallel degree. Accepts `None` or an integer value.
  - `-mb/--micro_batch_size`: Override micro-batch size.
  - `-gb/--global_batch_size`: Override global batch size.
  - `--moe_a2a_overlap`: Set the `moe_a2a_overlap` configuration flag (boolean).
  - `-ms/--max_steps`: Maximum number of training steps.
  - `-rl/--recompute_num_layers`: Number of transformer layers to recompute.
  - `-ol/--activation_offload_layers`: Number of transformer layers to offload activations to CPU memory.
  - `--detach`: Keep the submission flow detached from the terminal (default behaviour).
  - `--no-detach`: Keep the submission attached to the terminal session.

## Virtual Environment

- Create a virtual env at your preferred location on login node on a Slurm cluster and install the NeMo-Run package-
  ```
  pip install git+https://github.com/NVIDIA-NeMo/Run.git
  ```

- The YAML config files are resolved on compute node inside the container.
