# Megatron-Bridge Issues Summary (Feb 25, 2026)

Latest 50 issues from [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues)

---

## NEEDS ATTENTION / FOLLOW-UP

### Bugs - Unassigned or No Response

| # | Title | Status | Age | Why it needs attention |
|---|-------|--------|-----|----------------------|
| [#2498](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2498) | ETP in llama3.1 being set to > 1 in output log | OPEN | Feb 23 | **Internal bug**, assigned but 0 comments - needs triage response |
| [#2443](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2443) | GKE LLM pretraining recipes CLI args not propagating | OPEN | Feb 19 | **Unassigned, 0 comments** for 6 days |
| [#2424](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2424) | qwen3-vl bridge export error w/ uneven pipeline sharding | OPEN | Feb 18 | **Unassigned, 0 comments** for 7 days |
| [#2409](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2409) | LLAMA3 70B LoRa TFLOPs/GPU formula incorrect | OPEN | Feb 17 | **Unassigned, 0 comments** for 8 days |
| [#2375](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2375) | Qwen3-Coder-Next HF->Megatron conversion fails (LinearCrossEntropyModule) | OPEN | Feb 13 | **Unassigned**, member replied but waiting for reporter response |
| [#2300](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2300) | Incomplete checkpoint after Megatron->HF conversion for Qwen3-VL-30B | OPEN | Feb 10 | **Unassigned**, reporter says fixed in latest version but raised loss-mask issue for v0.2.1 |
| [#2193](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2193) | Lora merge fails for TP > 1 | OPEN | Feb 3 | **Unassigned**, community PR #2194 submitted - needs review |
| [#2195](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2195) | LLAMA3 70B LoRa enabled in all modules instead of only LinearQKV | OPEN | Feb 3 | **Unassigned**, PR #2181 exists - should close when merged |

### Bugs - Actively Being Investigated (needs follow-up)

| # | Title | Status | Assignee | Next step |
|---|-------|--------|----------|-----------|
| [#2403](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2403) | High gradnorm w/ `overlap_moe_expert_parallel_comm` on H100 | OPEN | ericharper | 0 comments - needs investigation update |
| [#2383](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2383) | LM3 405B FP8-CS with FSDP + symmetric registration | OPEN | gautham-kollu | 0 comments, performance/release tagged - **blocker?** |
| [#2381](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2381) | Model Conversion Bug for Nemotron Nano V3 | OPEN | liding-nv | Active discussion, reporter sees divergence on custom model - needs env clarification |
| [#2355](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2355) | GB200 multi-domain checkpointing with Deepseek SegFault | OPEN | cuichenx | Investigating, asked reporter to test down-sized config |
| [#2240](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2240) | Dense LoRA adapters lose TP shards during PEFT save | OPEN | unassigned | **Critical**: 50x loss inflation after save/load with PEFT filter. Initial fix was incorrect. yaoyu-33 testing tp2+pp2+ep2 |
| [#2321](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2321) | LoRALinear crashes with shape mismatch on shared experts (moe_shared_expert_overlap) | OPEN | ananthsub | 0 comments for 13 days |

### Community Requests - Waiting for Response

| # | Title | Status | Notes |
|---|-------|--------|-------|
| [#2481](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2481) | Return best k checkpoint based on metrics | OPEN | **Unassigned**, 0 comments |
| [#2480](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2480) | Specify checkpoints based on epoch # instead of step # | OPEN | **Unassigned**, 0 comments |
| [#2473](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2473) | SFT answer_only_loss: auto-slice output vocab | OPEN | **Unassigned**, 0 comments |
| [#2391](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2391) | Support num_epochs in TrainingConfig | OPEN | **needs-follow-up** label, awaiting user's dataset info |
| [#2302](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2302) | Cannot disable distributed optimizer for single-GPU | OPEN | Multiple users affected, yaoyu-33 acknowledged - needs fix |
| [#2276](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2276) | Dataset conversion from HF to Bridge | OPEN | Community contributor willing to help, waiting for guidance |
| [#2229](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2229) | XiaoMi MiMo-V2-Flash Model Support | OPEN | Waiting for user's use case to forward to PM |

---

## PERFORMANCE / RELEASE ITEMS

| # | Title | Status | Assignee | Next step |
|---|-------|--------|----------|-----------|
| [#2427](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2427) | FSDP to work with partial Cuda Graph | OPEN | unassigned | Feature request, unassigned |
| [#2341](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2341) | Bump pytorch 26.01 | OPEN | ko3n1g | No comments |
| [#2340](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2340) | Enable 2-kernel cudnn fused attention backward for DSv3/QWEN3 | OPEN | malay-nagda | PR #2432 created, testing in progress |
| [#2339](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2339) | Bump cudnn 9.18.1 in 26.02 patch | OPEN | ko3n1g | No comments |
| [#2316](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2316) | Update nsys version to 26.01 | OPEN | ko3n1g | erhoo82 asked if done in 26.02 container - **needs confirmation** |
| [#2297](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2297) | NT6 nano performance testing | OPEN | tomlifu | MXFP8 with TP2 errors observed |
| [#2242](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2242) | DSv3 recipe for real routing | OPEN | dingqingy-nv | No comments |

---

## ENHANCEMENTS / FEATURES

| # | Title | Status | Assignee | Notes |
|---|-------|--------|----------|-------|
| [#2343](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2343) | GLM-5 support | OPEN | unassigned | Community volunteer @pengdurice working on it |
| [#2318](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2318) | Upgrade to Transformers V5 | OPEN | unassigned | No comments, internal task |
| [#2317](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2317) | Add Ministral test to GHA | OPEN | unassigned | No comments, internal task |
| [#2253](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2253) | Export Customized Qwen3 MoE Models | OPEN | unassigned | Related to #1883 |
| [#2216](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2216) | Add TE min_version validation with cuda_graph + overlap_grad_reduce | OPEN | gautham-kollu | No comments |
| [#2206](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2206) | Fix Energon Support in Qwen3-VL | OPEN | kamran-nvidia | No comments |
| [#2200](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2200) | SFT dataset support docs | OPEN | unassigned | Documentation task |
| [#2309](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2309) | Docker compose file outdated | OPEN | ko3n1g | Asked for specifics |
| [#2308](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2308) | Efficient memory management with automatic cleanup | OPEN | yaoyu-33 | Directed to Shreya Gupta |
| [#2307](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2307) | Muon `layer_wise_optimizer_*.pt` file relevance | OPEN | BoxiangW | No comments |
| [#2306](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2306) | Guide on Custom Datasets with MegatronBridge | OPEN | cuichenx, yaoyu-33 | Documentation task |
| [#2305](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2305) | LoRA Adapter Conversion/Deployment (M-Bridge -> HF) | OPEN | unassigned | Already being worked on, ETA 26.02 per #1925 |

---

## RECENTLY CLOSED (resolved)

| # | Title | Resolved | Notes |
|---|-------|----------|-------|
| [#2350](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2350) | Continue training shows crazy throughput stats | Feb 23 | Fixed by PR #2475, merged |
| [#2245](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2245) | DSv3 GB300 nvfp4 hang | Feb 14 | Resolved |
| [#2232](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2232) | MoE model export error | Feb 5 | User error (different training configs) |
| [#2224](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2224) | Missing yarn_mscale in GPT-OSS | Feb 5 | User had outdated megatron-core version |
| [#2221](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2221) | OOM issue with llama3.1-405B on GB300 | Feb 14 | Resolved with rc4 container |
| [#2190](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2190) | DeepSeek-V3 run fails on GB200 | Feb 17 | Fixed in nightly image 2026-02-02+ and RC4+ |

---

## TOP PRIORITY ITEMS (Recommended Next Steps)

1. **#2240 - PEFT LoRA save corruption** - CRITICAL. 50x loss inflation on save/load. Needs deeper investigation of `_dedup_save_plans` / `_replace_state_dict_keys_with_sharded_keys` path. Assign owner and target fix.

2. **#2443 - GKE CLI args not propagating** - Unassigned for 6 days, 0 comments. Triage needed.

3. **#2424 - Qwen3-VL export with uneven pipeline** - Community bug, unassigned for 7 days.

4. **#2403 - High gradnorm w/ overlap_moe_expert_parallel_comm** - Internal performance issue on H100, assigned but no updates.

5. **#2383 - LM3 405B FP8-CS with FSDP** - Performance/release blocker, no comments.

6. **#2302 - Distributed optimizer config confusion** - Multiple users affected, acknowledged but no fix yet.

7. **#2321 - LoRALinear shape mismatch** - Assigned to ananthsub but 0 comments for 13 days.

8. **#2478 - Distillation checkpoint conversion** - yaoyu-33 responded today (Feb 25), needs fix.

9. **#2466 - Cosine similarity bug in compare_hf_and_megatron** - Community PR approved, CI triggered today.

10. **#2193 - LoRA merge TP>1** - Community fix PR #2194 waiting for review for 22 days.
