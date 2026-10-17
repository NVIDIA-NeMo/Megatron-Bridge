# Muse-Spark-Open-816B-A42B

Muse-Spark-Open-816B-A42B is a sparse mixture-of-experts vision-and-audio-language model with
816B total and 42B active parameters. Megatron Bridge models the complete checkpoint: the
Hugging Face vision tower, vision adapter and projection, and audio tower run directly, and the
text decoder is a native Megatron-Core `HybridModel` with 62 alternating attention and latent-MoE
blocks (256 experts, top-8 routing). A text-only mode converts the language model and its separate
LM head alone. Native Hugging Face execution of the checkpoint needs Transformers 5.20 or later.
The [model README](../../../src/megatron/bridge/models/muse_spark_open/README.md) describes the
implementation, the text-only mode, and the supported parallelism.

## Training recipes

The GB200 recipes are defined in
[muse_spark_open.py](../../../src/megatron/bridge/recipes/muse_spark_open/gb200/muse_spark_open.py) and
warm-start from the imported checkpoint. The verification card gives the import command and the
training commands for the pretraining, SFT, long-context SFT, and LoRA recipes. The SFT, LoRA, and
image-text recipes shard the 256 routed experts with EP64 and dispatch them with HybridEP, one NVL72
NVLink domain (16 nodes) per pipeline stage, without full activation recompute; the 32K and 64K SFT
recipes recompute only the MoE layers. The pretraining recipes keep TP1/EP32 with full activation
recompute, which pretrains faster on 192 GPUs than EP64 without recompute.

| Recipe | Workload | Nodes | Layout |
| --- | --- | ---: | --- |
| `muse_spark_open_pretrain_192gpu_gb200_bf16_config` | Text pretraining, GBS 1024, sequence length 4096 | 48 | TP1/PP6/EP32 |
| `muse_spark_open_pretrain_256gpu_gb200_bf16_config` | Text pretraining, eight pipeline stages | 64 | TP1/PP8/EP32 |
| `muse_spark_open_sft_192gpu_gb200_bf16_config` | Full SFT, one epoch of 100K Tulu 3 conversations packed to 4K, GBS 128 | 48 | TP4/PP3/EP64 |
| `muse_spark_open_sft_256gpu_gb200_bf16_config` | Full SFT on 10K Tulu 3 conversations packed to 8K, GBS 8 (the cross-model card contract) | 64 | TP4/PP4/CP2/EP64 |
| `muse_spark_open_sft_192gpu_gb200_bf16_long_context_config` | Full SFT on packed 32K Tulu 3, GBS 8 | 48 | TP4/PP3/CP2/EP64 |
| `muse_spark_open_sft_192gpu_gb200_bf16_coderforge_64k_config` | Full SFT on packed 64K CoderForge agent trajectories, GBS 8 | 48 | TP4/PP3/CP4/EP64 |
| `muse_spark_open_peft_64gpu_gb200_bf16_config` | LoRA on attention, every routed expert, and the latent MoE projections; one epoch of 100K Tulu 3 conversations packed to 4K, GBS 128 | 16 | TP4/PP1/EP64 |
| `muse_spark_open_sft_192gpu_gb200_bf16_multimodal_config` | Full SFT on image-text CORD-v2, GBS 64 | 48 | TP4/PP3/EP64 |
| `muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config` | LoRA on image-text CORD-v2, GBS 8 | 4 | TP4/EP16 |

The text recipes freeze the vision and audio towers; the image-text recipes freeze only the audio
tower. The recipes are short verification runs whose learning-rate decay ends at the last step, so
raise `scheduler.lr_decay_iters` together with the step count to train longer.

HybridEP needs each expert-parallel group inside one NVL72 NVLink domain: every EP64 group is one
pipeline stage's 64 contiguous ranks (16 nodes), the 192-GPU pretraining recipe's EP32 groups are
one stage's 32 ranks (8 nodes), and the 4-node image-text LoRA recipe keeps EP16. The 256-GPU
pretraining recipe uses the all-to-all dispatcher. On block-topology Slurm clusters, allocate the
nodes in 16-node segments. Setting
`model.transformer.moe_token_dispatcher_type` to `alltoall` switches back to the all-to-all
dispatcher.

The model chat template stamps the current date into its system prompt. For reproducible
packing, pass a copy of the template with a fixed `current_date` default through
`tokenizer.chat_template_path`. Packing runs on rank 0 before training starts and takes about
25 minutes for the 100K Tulu 3 slice, so prepare the packs once; the packed data must be on a
filesystem that every node can read (`dataset.hf_output_root`). The packed recipes pad every
micro-batch to the full pack length (`pad_to_max_length` in `dataset.dataset_kwargs`), so every rank
processes the same number of tokens and the compiled fusions keep one shape.

## Verification status

See the [model verification card](../../model_verification_cards/muse-spark-open-816b-a42b/card.yaml)
for the commands and recorded results on the pinned public Hugging Face revision.
