# Megatron-Bridge Open PR Checklist (Feb 25, 2026 — evening update)

All open PRs >= #2000 from [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pulls)

## PR Title Convention (from CONTRIBUTING.md)

```
[{module}] {type}: {description}
```

**Modules**: `model`, `recipe`, `training`, `data`, `ckpt`, `peft`, `perf`, `ci`, `doc`, `test`, `build`, `misc`
**Types**: `feat`, `fix`, `refactor`, `chore`, `test`

---

## TOP PRIORITY TODOS (yaoyu-33)

### Your open PRs
1. **#2497** `[build] chore: Update Megatron-LM submodule to dev branch` — DRAFT. Un-draft when ready.
2. **#2331** `[training] fix: mirror megatron-lm parity updates (flops + delay_wgrad)` — REVIEW_REQUIRED for 14 days. Chase reviewers.
3. **#2252** `[peft] fix: Add replica_id handling for dense LoRA adapters with TP > 1` — REVIEW_REQUIRED for 19 days. Linked to critical issue #2240 (50x loss inflation). **This is the highest priority PR to land.**
4. **#2042** `[peft,doc] feat: Document adapter merge verification` — APPROVED for 33 days. **Merge now.**

### PRs you should merge (approved, non-draft, non-blocked)
1. **#2042** — your own PR, approved 33 days
2. **#2474** — `save_hf_pretrained additional_files` (liding-nv), approved 4 days
3. **#2476** — `compare.py attention_mask fix` (community), approved 3 days
4. **#2397** — `LLAMA3 70B LoRA B300/B200` (rhmukundan), approved + **r0.3.0**, 9 days
5. **#2384** — `None hf_config values fix` (community), approved 11 days
6. **#2183** — `create_attention_mask=False` (gautham-kollu), approved + **r0.3.0**, 22 days
7. **#2153** — `MLflow improvements` (community), approved 26 days
8. **#2090** — `training loop QoL` (ananthsub), approved 29 days
9. **#2070** — `cached_property fix` (community), approved 30 days
10. **#2021** — `rotary_emb.inv_freq shape fix` (community), approved 34 days

### Release-critical (r0.3.0) needing your review/push
1. **#2510** — Nemotron 3 Nano perf configs (malay-nagda) — **APPROVED, merge now**
2. **#2499** — Qwen3 30B B200 config fix (tomlifu) — needs review
3. **#2358** — MoE Sequential MLP mappings (kevalmorabia97) — needs review
4. **#2323** — Local checkpoint fix (ananthsub) — needs review

### Critical bugs to unblock
1. **Issue #2240** — PEFT LoRA save corruption (50x loss inflation). Your PR #2252 is the fix. Push for review.
2. **Issue #2321** — LoRALinear shape mismatch on shared experts. Assigned to ananthsub, 0 comments for 14 days. Follow up.
3. **Issue #2383** — LM3 405B FP8-CS with FSDP. Performance/release tagged, 0 comments. Check if blocker.

### Community PRs needing your review
1. **#2482** — `LoRATopKRouter forward` fix (HollowMan6), 2-line change, 2 days old
2. **#2194** — `LoRA merge TP>1` fix (sowmen), 22 days old — linked to issue #2193
3. **#2057** — `Replace deprecated typing imports` (KunalSachdev2005), 32 days old
4. **#2028** — `Ling MoE V2 model` (ccclyu), 34 days old
5. **#2532** — `EXAONE 4.0 model bridge` (Bias92) — brand new, community

---

## RECENTLY MERGED / CLOSED (since last update)

| # | Title | Status | Date | Notes |
|---|-------|--------|------|-------|
| [#2514](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2514) | [training] fix: Add missing dataset/validation configs to logging | **MERGED** | Feb 25 | yaoyu-33 |
| [#2511](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2511) | remove deprecated MP params | **MERGED** | Feb 25 | dimapihtar |
| [#2516](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2516) | ci(fix): TestPyPI | **MERGED** | Feb 25 | ko3n1g |
| [#2484](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2484) | build: Bump modelopt to 0.42.0rc1 | **MERGED** | Feb 25 | ko3n1g |
| [#2492](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2492) | 26.02 perf summary | **MERGED** | Feb 25 | malay-nagda |
| [#2068](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2068) | Upgrade transformers to 5.0 | **MERGED** | Feb 25 | yaoyu-33 — closes issue #2318 |
| [#2416](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2416) | Fix LLAMA3 LoRa TFLOPs Formula | **MERGED** | Feb 25 | rhmukundan |
| [#2098](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2098) | Qwen3VLMoE default fix | **CLOSED** | Feb 25 | Yangruipis — closed without merge |

---

## SUMMARY

| Category | Total | Draft | Approved | Review Required |
|----------|-------|-------|----------|-----------------|
| **Model** | 15 | 2 | 0 | 13 |
| **Recipe** | 6 | 0 | 1 | 5 |
| **Performance** | 11 | 3 | 1 | 7 |
| **Training** | 13 | 5 | 4 | 4 |
| **Checkpoint** | 7 | 1 | 2 | 4 |
| **PEFT** | 6 | 0 | 2 | 4 |
| **CI** | 5 | 2 | 0 | 3 |
| **Build** | 6 | 2 | 0 | 4 |
| **Docs** | 6 | 0 | 0 | 6 |
| **Cherry-pick** | 6 | 5 | 0 | 1 |
| **Misc** | 16 | 3 | 6 | 7 |
| **TOTAL** | **102** | **23** | **16** | **63** |

Changes from last snapshot: 6 PRs merged today, 1 closed, 8 new PRs opened. Net +1 (99→102 open).

---

## READY TO MERGE (approved, non-draft)

| # | Title | Author | Category | Age |
|---|-------|--------|----------|-----|
| [#2518](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2518) | `[model] feat: Add changes from PR2161 in MLM` | JRD971000 | model | Feb 25 |
| [#2510](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2510) | `[recipe,perf] feat: Update Nemotron 3 Nano perf configs` | malay-nagda | recipe/perf | Feb 24 |
| [#2476](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2476) | `[model,test] fix: Fix attention_mask mismatch in compare.py` | mohsinm-dev | misc | Feb 22 |
| [#2474](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2474) | `[ckpt] feat: Add additional_files parameter to save_hf_pretrained` | liding-nv | ckpt | Feb 21 |
| [#2397](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2397) | `[peft,recipe] feat: Onboard LLAMA3 70B LoRA to B300 and B200` | rhmukundan | peft | Feb 16 |
| [#2384](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2384) | `[misc] fix: Correctly pass None hf_config values to provider_kwargs` | HollowMan6 | misc | Feb 14 |
| [#2183](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2183) | `[training] feat: Set create_attention_mask=False for pretrain` | gautham-kollu | training | Feb 3 |
| [#2153](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2153) | `[misc] feat: MLflow improvements` | ryxli | misc | Jan 30 |
| [#2123](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2123) | `[training] feat: Select num_devices_per_node` | malay-nagda | misc | Jan 29 |
| [#2090](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2090) | `[training] feat: Various quality-of-life improvements in training loop` | ananthsub | training | Jan 27 |
| [#2070](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2070) | `[misc] fix: Change cached_property to property for _causal_lm_architecture` | guyueh1 | misc | Jan 26 |
| [#2042](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2042) | `[peft,doc] feat: Document adapter merge verification` | yaoyu-33 | peft | Jan 23 |
| [#2021](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2021) | `[ckpt] fix: Fix exported rotary_emb.inv_freq shape` | conver334 | ckpt | Jan 22 |

> **13 PRs approved and ready to merge.** #2510 is r0.3.0 tagged — merge first. Several community PRs have been waiting 4+ weeks.

---

## NEW PRs (since last update)

| # | Title | Author | Review | Size | Labels |
|---|-------|--------|--------|------|--------|
| [#2534](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2534) | Migration DFM -> Bridge | abhinavg4 | REVIEW_REQUIRED | +22990/-0 | -- |
| [#2532](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2532) | `[model] feat: Add EXAONE 4.0 model bridge` | Bias92 | REVIEW_REQUIRED | +469/-0 | community |
| [#2530](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2530) | DRAFT `[model] feat: Qwen 3.5 VL` | cuichenx | REVIEW_REQUIRED | +1508/-9 | -- |
| [#2529](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2529) | `[doc] feat: Update user manual with MoE features and FSDP` | onel | REVIEW_REQUIRED | +430/-11 | community |
| [#2528](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2528) | `[doc] feat: Add SFT dataset formats documentation` | onel | REVIEW_REQUIRED | +387/-0 | community |
| [#2524](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2524) | Bump uv.lock (r0.3.0) Feb 25 | bot | REVIEW_REQUIRED | +72/-42 | -- |
| [#2522](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2522) | `[training] feat: Auto-slice output vocab for SFT answer_only_loss` | AmitMY | REVIEW_REQUIRED | +1037/-0 | community |
| [#2521](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2521) | DRAFT fix mcore bump | JRD971000 | REVIEW_REQUIRED | +0/-2 | -- |
| [#2520](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2520) | DRAFT `[model] feat: Switch Qwen3-Next to use MambaModel` | Phlip79 | REVIEW_REQUIRED | +104/-9 | -- |
| [#2519](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2519) | `[ci] feat: Add Azure H100 runners` | chtruong814 | REVIEW_REQUIRED | +8/-3 | -- |
| [#2518](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2518) | `[model] feat: Add changes from PR2161 in MLM` | JRD971000 | **APPROVED** | +34/-2 | -- |

**Key observations:**
- #2534 is a massive migration (+23K lines) — needs careful scoping
- #2532 EXAONE 4.0 — new community model bridge
- #2530 Qwen 3.5 VL work starting (cuichenx, draft)
- #2522 auto-slice output vocab — linked to issue #2473, community contribution
- #2529 and #2528 — two community docs PRs from same author (onel)

---

## MODEL SUPPORT (15 PRs)

| # | Title | Author | Review | Created | Size | Labels |
|---|-------|--------|--------|---------|------|--------|
| [#2534](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2534) | Migration DFM -> Bridge | abhinavg4 | REVIEW_REQUIRED | Feb 25 | +22990/-0 | -- |
| [#2532](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2532) | `[model] feat: Add EXAONE 4.0 model bridge` | Bias92 | REVIEW_REQUIRED | Feb 25 | +469/-0 | community |
| [#2530](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2530) | DRAFT `[model] feat: Qwen 3.5 VL` | cuichenx | REVIEW_REQUIRED | Feb 25 | +1508/-9 | -- |
| [#2520](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2520) | DRAFT `[model] feat: Switch Qwen3-Next to use MambaModel` | Phlip79 | REVIEW_REQUIRED | Feb 25 | +104/-9 | -- |
| [#2518](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2518) | `[model] feat: Add changes from PR2161 in MLM` | JRD971000 | **APPROVED** | Feb 25 | +34/-2 | -- |
| [#2469](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2469) | `[model] feat: Add GLM-5 model bridge` | pengdurice | REVIEW_REQUIRED | Feb 20 | +1014/-1 | community |
| [#2440](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2440) | `[model] fix: Fix Energon support in Qwen3-VL` | kamran-nvidia | REVIEW_REQUIRED | Feb 19 | +881/-74 | -- |
| [#2370](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2370) | `[model] fix: M4 leftover fixes for Qwen3-VL` | shifangx | REVIEW_REQUIRED | Feb 13 | +132/-11 | -- |
| [#2367](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2367) | `[model,training] feat: Support training Qwen3-VL with distributed training` | shifangx | REVIEW_REQUIRED | Feb 13 | +316/-93 | -- |
| [#2342](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2342) | `[model] feat: Add Qwen3-Omni-MoE model bridge` | zhang-cheng09 | REVIEW_REQUIRED | Feb 12 | +3569/-0 | community |
| [#2324](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2324) | `[model] feat: Add Qwen2-Audio model bridge` | yuekaizhang | REVIEW_REQUIRED | Feb 11 | +1356/-0 | community |
| [#2296](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2296) | `[model] feat: Add Qwen 2.5-Omni model bridge` | martinzhang03 | REVIEW_REQUIRED | Feb 10 | +1178/-341 | community |
| [#2143](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2143) | `[model] feat: Support Qwen3-VL video and context parallelism` | zhang-cheng09 | REVIEW_REQUIRED | Jan 30 | +489/-55 | community |
| [#2140](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2140) | `[model,training,data] feat: Support video training for Qwen2.5/3-VL` | yangrz7 | REVIEW_REQUIRED | Jan 30 | +65/-47 | community |
| [#2137](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2137) | `[model,training] feat: MIMO Phase 4 - training loop implementation` | aroshanghias-nvd | REVIEW_REQUIRED | Jan 30 | +5364/-1 | -- |

**Key observations:**
- **#2098 was CLOSED** (not merged) — Qwen3VLMoE default fix, community contribution. Was approved but never merged.
- EXAONE 4.0 (#2532) and Qwen 3.5 VL (#2530) are brand new model bridges
- #2534 DFM migration is massive (+23K) — needs scoping discussion
- Multiple **competing Qwen3-VL video** PRs (#2143 vs #2140) still unresolved
- MIMO pipeline spans 3 remaining PRs (#2004, #2007, #2137, #2182) — needs coordinated review

---

## RECIPE / PERF CONFIGS (6 PRs)

| # | Title | Author | Review | Created | Size | Labels |
|---|-------|--------|--------|---------|------|--------|
| [#2510](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2510) | `[recipe,perf] feat: Update Nemotron 3 Nano perf configs` | malay-nagda | **APPROVED** | Feb 24 | +45/-28 | performance,r0.3.0 |
| [#2500](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2500) | `[recipe,perf] feat: Update Qwen3 30B H100 base configs with HybridEP` | rhmukundan | REVIEW_REQUIRED | Feb 23 | +8/-8 | -- |
| [#2499](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2499) | `[recipe,perf] fix: Qwen3 30B A3B B200 config use hybridep+flex dispatcher` | tomlifu | REVIEW_REQUIRED | Feb 23 | +6/-9 | r0.3.0 |
| [#2471](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2471) | `[recipe] feat: Add NVFP4 recipes for Nemotron-3 Nano` | tomlifu | REVIEW_REQUIRED | Feb 20 | +31/-5 | -- |
| [#2371](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2371) | `[recipe] fix: Use WSD LR schedule for Nemotron-3 Nano to match paper` | OlegSudakov | REVIEW_REQUIRED | Feb 13 | +4/-0 | -- |
| [#2268](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2268) | `[recipe] refactor: Refactor finetuning recipes` | athitten | CHANGES_REQUESTED | Feb 6 | +11160/-5485 | -- |

**Key observations:**
- **#2492 perf summary MERGED** today
- #2510 is **APPROVED + r0.3.0** — un-drafted and ready to merge
- #2471 approval was invalidated (now REVIEW_REQUIRED) — re-review needed
- #2268 still has CHANGES_REQUESTED — massive refactor, needs author attention

---

## PERFORMANCE (11 PRs)

| # | Title | Author | Review | Created | Size | Labels |
|---|-------|--------|--------|---------|------|--------|
| [#2517](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2517) | DRAFT `[perf] fix: Add .sync() every eval iter for full-iteration CUDA Graph` | gautham-kollu | REVIEW_REQUIRED | Feb 24 | +23/-1 | -- |
| [#2487](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2487) | `[perf,training] feat: Add AG/RS overlap distributed init support` | jeffnvidia | REVIEW_REQUIRED | Feb 23 | +51/-0 | -- |
| [#2432](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2432) | DRAFT `[perf] feat: Set NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 for deterministic MoE` | malay-nagda | REVIEW_REQUIRED | Feb 19 | +3/-0 | performance |
| [#2419](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2419) | `[perf] fix: Handle mxfp8 param buffer copy for FSDP with grad buffer reuse` | rapatel | REVIEW_REQUIRED | Feb 17 | +18/-3 | -- |
| [#2411](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2411) | `[perf] feat: Enable optimizer CUDA graph` | vasunvidia | REVIEW_REQUIRED | Feb 17 | +10/-0 | -- |
| [#2372](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2372) | `[perf] fix: M4 leftover fixes for TE CUDA Graph` | shifangx | REVIEW_REQUIRED | Feb 13 | +32/-5 | -- |
| [#2334](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2334) | `[perf,model] feat: Add CUDA Graph support for Vision Encoder` | tomlifu | REVIEW_REQUIRED | Feb 11 | +343/-23 | -- |
| [#2299](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2299) | `[perf] feat: Compatible with Megatron-FSDP TP` | conver334 | REVIEW_REQUIRED | Feb 10 | +706/-15 | community |
| [#2266](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2266) | `[perf] fix: Fix FSDP config check` | BoxiangW | REVIEW_REQUIRED | Feb 6 | +4/-2 | -- |
| [#2136](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2136) | `[perf,training] feat: Clear aux losses tracker after eval, add CUDA Graph validation` | ananthsub | REVIEW_REQUIRED | Jan 30 | +47/-0 | -- |
| [#2025](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2025) | DRAFT `[perf] fix: Disable precision_aware_optimizer for FSDP` | malay-nagda | REVIEW_REQUIRED | Jan 22 | +1/-1 | perf/release |

**Key observations:**
- Strong CUDA Graph activity (#2517, #2411, #2372, #2334, #2136)
- Multiple FSDP fixes (#2419, #2299, #2266, #2025)
- #2025 is 34 days old, 1-line change, tagged perf/release — should resolve
- #2266 is a tiny 4-line fix, sitting for 19 days

---

## TRAINING (13 PRs)

| # | Title | Author | Review | Created | Size | Labels |
|---|-------|--------|--------|---------|------|--------|
| [#2522](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2522) | `[training] feat: Auto-slice output vocab for SFT answer_only_loss` | AmitMY | REVIEW_REQUIRED | Feb 25 | +1037/-0 | community |
| [#2505](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2505) | DRAFT `[training] feat: Add gradient hook implementation` | kamran-nvidia | REVIEW_REQUIRED | Feb 24 | +491/-1 | -- |
| [#2472](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2472) | DRAFT `[training,test] feat: Update torch profiling and add unit test` | briancoutinho | REVIEW_REQUIRED | Feb 21 | +75/-3 | community |
| [#2470](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2470) | `[training] feat: Add on_before_data_init callback hook for MLPerf compliance` | matthew-frank | REVIEW_REQUIRED | Feb 20 | +34/-1 | community |
| [#2405](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2405) | `[training,data] feat: Enhance dataset loading efficiency with tensor parallelism` | izikgo | REVIEW_REQUIRED | Feb 17 | +771/-37 | -- |
| [#2395](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2395) | DRAFT `[training,data] feat: Improve packed sequences SFT for large datasets` | shaltielshmid | REVIEW_REQUIRED | Feb 16 | +197/-18 | community |
| [#2331](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2331) | `[training] fix: Mirror megatron-lm parity updates (flops + delay_wgrad)` | yaoyu-33 | REVIEW_REQUIRED | Feb 11 | +221/-4 | -- |
| [#2264](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2264) | DRAFT `[training,data] feat: Add OpenMathInstruct and GSM8K datasets` | cuichenx | REVIEW_REQUIRED | Feb 6 | +99/-0 | -- |
| [#2211](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2211) | DRAFT `[training] refactor: Import dataclasses from Megatron Training` | maanug-nv | **APPROVED** | Feb 4 | +303/-392 | Run CICD |
| [#2185](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2185) | `[training] feat: Add chat template from HF tokenizer` | cuichenx | REVIEW_REQUIRED | Feb 3 | +62/-14 | -- |
| [#2183](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2183) | `[training] feat: Set create_attention_mask=False for pretrain` | gautham-kollu | **APPROVED** | Feb 3 | +5/-2 | r0.3.0 |
| [#2090](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2090) | `[training] feat: Various quality-of-life improvements in training loop` | ananthsub | **APPROVED** | Jan 27 | +199/-37 | -- |
| [#2019](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2019) | DRAFT `[training,data] feat: Proposal for scalable seq-pack data format` | marcromeyn | REVIEW_REQUIRED | Jan 21 | +1416/-74 | -- |

**Key observations:**
- **#2514 MERGED** today (yaoyu-33, training fix)
- **#2068 MERGED** today (transformers v5 upgrade — closes issue #2318!)
- **#2511 MERGED** today (deprecated MP params removal)
- #2183 is approved + r0.3.0 tagged but not merged yet — **merge blocker?**
- #2090 approved for 29 days with no merge
- #2211 approved (draft) — author needs to un-draft
- #2522 is new community PR for auto-slice output vocab (linked to issue #2473)

---

## CHECKPOINT (7 PRs)

| # | Title | Author | Review | Created | Size | Labels |
|---|-------|--------|--------|---------|------|--------|
| [#2496](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2496) | DRAFT `[test,ckpt] test: Add checkpoint save/load tests` | dimapihtar | REVIEW_REQUIRED | Feb 23 | +114/-0 | -- |
| [#2474](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2474) | `[ckpt] feat: Add additional_files parameter to save_hf_pretrained` | liding-nv | **APPROVED** | Feb 21 | +295/-26 | -- |
| [#2422](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2422) | `[ckpt,model] feat: GPT-OSS examples` | weijiac0619 | REVIEW_REQUIRED | Feb 18 | +798/-2 | -- |
| [#2323](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2323) | `[ckpt] fix: Fix local checkpoint integration` | ananthsub | REVIEW_REQUIRED | Feb 11 | +387/-19 | r0.3.0 |
| [#2293](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2293) | `[ckpt] refactor: Consolidate duplicate code for handling checkpoint paths` | ananthsub | REVIEW_REQUIRED | Feb 9 | +390/-136 | -- |
| [#2239](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2239) | `[ckpt] feat: Expose public API to load model weights` | ananthsub | REVIEW_REQUIRED | Feb 5 | +274/-9 | -- |
| [#2182](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2182) | `[ckpt,model] feat: MIMO Phase 5 - checkpointing and evaluation` | aroshanghias-nvd | REVIEW_REQUIRED | Feb 2 | +6335/-21 | -- |

**Key observations:**
- #2323 is r0.3.0 tagged — **release-critical checkpoint fix**
- #2422 title changed from "WIP: export yarn issue" to "GPT-OSS examples" and un-drafted
- #2021 approved for 34 days, community — **should merge**
- ananthsub has 3 open checkpoint PRs (#2323, #2293, #2239) — may have dependencies

---

## PEFT (6 PRs)

| # | Title | Author | Review | Created | Size | Labels |
|---|-------|--------|--------|---------|------|--------|
| [#2482](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2482) | `[peft] fix: Allow args and kwargs for LoRATopKRouter forward` | HollowMan6 | REVIEW_REQUIRED | Feb 23 | +2/-2 | community |
| [#2397](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2397) | `[peft,recipe] feat: Onboard LLAMA3 70B LoRA to B300 and B200` | rhmukundan | **APPROVED** | Feb 16 | +127/-0 | r0.3.0 |
| [#2396](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2396) | `[peft,recipe] feat: Onboard LLAMA3 LoRA to B200 and B300` | rhmukundan | REVIEW_REQUIRED | Feb 16 | +125/-0 | -- |
| [#2252](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2252) | `[peft] fix: Add replica_id handling for dense LoRA adapters with TP > 1` | yaoyu-33 | REVIEW_REQUIRED | Feb 6 | +653/-4 | -- |
| [#2194](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2194) | `[peft] fix: Fix LoRA merge to gather weights across ranks for TP>1` | sowmen | REVIEW_REQUIRED | Feb 3 | +221/-7 | community |
| [#2042](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2042) | `[peft,doc] feat: Document adapter merge verification` | yaoyu-33 | **APPROVED** | Jan 23 | +7/-0 | -- |

**Key observations:**
- **#2416 LoRA TFLOPs fix MERGED** today
- #2397 approved + r0.3.0 tagged — **should merge for release**
- #2396 vs #2397 still look like duplicates — reconcile?
- #2252 is the critical PEFT save bug fix linked to issue #2240 — **push for review**
- #2194 community fix, 22 days old, linked to issue #2193

---

## CI (5 PRs)

| # | Title | Author | Review | Created | Size |
|---|-------|--------|--------|---------|------|
| [#2519](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2519) | `[ci] feat: Add Azure H100 runners` | chtruong814 | REVIEW_REQUIRED | Feb 25 | +8/-3 |
| [#2504](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2504) | DRAFT `[ci] chore: CI validation for PR #2387` | chtruong814 | REVIEW_REQUIRED | Feb 24 | +285/-0 |
| [#2503](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2503) | DRAFT `[ci] chore: CI validation for PR #2390` | chtruong814 | REVIEW_REQUIRED | Feb 24 | +15/-12 |
| [#2230](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2230) | `[ci] chore: CI validation for issue #1806` | chtruong814 | REVIEW_REQUIRED | Feb 5 | +447/-4 |
| [#2128](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2128) | `[ci] chore: Update to 8x GPU runners and add node logging` | chtruong814 | REVIEW_REQUIRED | Jan 29 | +49/-7 |

**Key observations:**
- **#2516 TestPyPI fix MERGED** today
- #2519 is new — Azure H100 runners

---

## BUILD (6 PRs)

| # | Title | Author | Review | Created | Size |
|---|-------|--------|--------|---------|------|
| [#2524](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2524) | Bump uv.lock (r0.3.0) Feb 25 | bot | REVIEW_REQUIRED | Feb 25 | +72/-42 |
| [#2521](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2521) | DRAFT fix mcore bump | JRD971000 | REVIEW_REQUIRED | Feb 25 | +0/-2 |
| [#2509](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2509) | `[build] chore: Cherry-pick various changes for 26.02.01 patch` | ko3n1g | REVIEW_REQUIRED | Feb 24 | +9500/-7801 |
| [#2507](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2507) | Bump uv.lock (main) Feb 24 | bot | REVIEW_REQUIRED | Feb 24 | +170/-177 |
| [#2497](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2497) | DRAFT `[build] chore: Update Megatron-LM submodule to dev branch` | yaoyu-33 | REVIEW_REQUIRED | Feb 23 | +1/-1 |
| [#2465](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2465) | Bump uv.lock (main) Feb 20 | bot | REVIEW_REQUIRED | Feb 20 | +525/-525 |

**Key observations:**
- **#2484 modelopt bump MERGED** today
- #2464 (uv.lock r0.3.0 Feb 20) closed/superseded by #2524

---

## DOCS (6 PRs)

| # | Title | Author | Review | Created | Size |
|---|-------|--------|--------|---------|------|
| [#2529](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2529) | `[doc] feat: Update user manual with MoE features and FSDP` | onel | REVIEW_REQUIRED | Feb 25 | +430/-11 |
| [#2528](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2528) | `[doc] feat: Add SFT dataset formats documentation` | onel | REVIEW_REQUIRED | Feb 25 | +387/-0 |
| [#2410](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2410) | `[doc] feat: Migrate docs to Fern framework` | lbliii | REVIEW_REQUIRED | Feb 17 | +16843/-1 |
| [#2108](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2108) | `[doc] fix: Add Kimi K2, GLM-4.5V and fix Qwen3-VL link` | sbhavani | REVIEW_REQUIRED | Jan 28 | +3/-1 |
| [#2102](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2102) | `[doc] feat: Improve HF-Megatron checkpoint conversion docs` | OlegSudakov | REVIEW_REQUIRED | Jan 28 | +12/-1 |
| [#2094](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2094) | `[doc] feat: Add docstrings to callback module` | coderabbitai | REVIEW_REQUIRED | Jan 28 | +1845/-85 |

**Key observations:**
- Two new community docs PRs (#2529, #2528) from onel — good signal
- #2108 is a tiny 3-line docs fix, sitting for 28 days
- #2410 is massive Fern migration — needs dedicated review

---

## CHERRY-PICKS for r0.3.0 (6 PRs)

| # | Source PR | Status |
|---|----------|--------|
| [#2513](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2513) | cp: disable CG for 8B SFT (#2508) | DRAFT, CI pending |
| [#2495](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2495) | cp: 70b_sft_perf_optimization (#2404) | DRAFT, CI pending |
| [#2493](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2493) | cp: [ckpt] fix: Log warning HF download (#2429) | DRAFT, CI pending |
| [#2483](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2483) | cp: Fix DeepSeek-V3 H100 config (#2401) | DRAFT, CI pending |
| [#2406](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2406) | cp: Deployment parallelism (#2189) | DRAFT, CI pending |
| [#2332](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2332) | cp: default packed_sequence to True (#2284) | DRAFT, CI pending |

---

## MISC / REFACTORING (16 PRs)

| # | Title | Author | Review | Created | Size | Labels |
|---|-------|--------|--------|---------|------|--------|
| [#2512](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2512) | `[training] refactor: Remove encoder_and_decoder usage` | dimapihtar | REVIEW_REQUIRED | Feb 24 | +22/-22 | -- |
| [#2476](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2476) | `[model,test] fix: Fix attention_mask mismatch in compare.py` | mohsinm-dev | **APPROVED** | Feb 22 | +147/-1 | community |
| [#2426](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2426) | DRAFT `[training] fix: Suppress default config printing` | malay-nagda | REVIEW_REQUIRED | Feb 18 | +9/-5 | -- |
| [#2407](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2407) | DRAFT `[perf] feat: CP2 for LM-405B` | gautham-kollu | REVIEW_REQUIRED | Feb 17 | +16/-3 | -- |
| [#2384](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2384) | `[model] fix: Correctly pass None hf_config values to provider_kwargs` | HollowMan6 | **APPROVED** | Feb 14 | +84/-46 | community |
| [#2358](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2358) | `[model,ckpt] feat: Add MoE Sequential MLP mappings in HF Bridges` | kevalmorabia97 | REVIEW_REQUIRED | Feb 12 | +23/-2 | r0.3.0 |
| [#2275](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2275) | `[model] feat: Add Kimi 2.5 model bridge` | FDecaYed | PENDING | Feb 9 | +251/-214 | -- |
| [#2235](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2235) | DRAFT `[model] refactor: Model provider refactor` | maanug-nv | REVIEW_REQUIRED | Feb 5 | +566/-0 | -- |
| [#2153](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2153) | `[misc] feat: MLflow improvements` | ryxli | **APPROVED** | Jan 30 | +124/-75 | community |
| [#2125](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2125) | DRAFT `[training] feat: Enable phase transition iterations` | ananthsub | REVIEW_REQUIRED | Jan 29 | +357/-14 | -- |
| [#2123](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2123) | `[training] feat: Select num_devices_per_node` | malay-nagda | **APPROVED** | Jan 29 | +24/-7 | -- |
| [#2103](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2103) | `[recipe] feat: Add deterministic recipe example` | suiyoubi | REVIEW_REQUIRED | Jan 28 | +161/-32 | -- |
| [#2089](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2089) | `[training] feat: Add support for fake distributed process groups` | ananthsub | REVIEW_REQUIRED | Jan 27 | +299/-6 | -- |
| [#2070](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2070) | `[misc] fix: Change cached_property to property for _causal_lm_architecture` | guyueh1 | **APPROVED** | Jan 26 | +1/-1 | -- |
| [#2057](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2057) | `[misc] refactor: Replace deprecated typing imports (#993)` | KunalSachdev2005 | REVIEW_REQUIRED | Jan 24 | +2119/-2218 | community |
| [#2028](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2028) | `[model] feat: Add Ling MoE V2 model bridge` | ccclyu | REVIEW_REQUIRED | Jan 22 | +845/-3 | community |

**Key observations:**
- #2512 review status changed from APPROVED to REVIEW_REQUIRED (code push invalidated approval)
- #2358 r0.3.0 tagged, MoE Sequential MLP mappings — **release-critical**
- #2275 "kimi 2.5" has PENDING review status for 16 days
- #2057 is a large typing refactor (+2119/-2218) from community, 32 days old
- #2028 Ling MoE V2 model support, 34 days old with no review

---

## ACTION ITEMS

### Merge immediately (approved, non-draft, non-blocked)
1. **#2510** `[recipe,perf] feat` - malay-nagda (r0.3.0, approved) ← **NEW**
2. **#2518** `[model] feat` - JRD971000 (approved, brand new) ← **NEW**
3. **#2476** `[model,test] fix` - community (3 days old)
4. **#2474** `[ckpt] feat` - liding-nv (4 days old)
5. **#2397** `[peft,recipe] feat` - rhmukundan (r0.3.0, 9 days old)
6. **#2384** `[misc] fix` - community (11 days old)
7. **#2183** `[training] feat` - gautham-kollu (r0.3.0, 22 days old)
8. **#2153** `[misc] feat` - community (26 days old)
9. **#2123** `[training] feat` - malay-nagda (27 days old)
10. **#2090** `[training] feat` - ananthsub (29 days old)
11. **#2070** `[misc] fix` - community (30 days old)
12. **#2042** `[peft,doc] feat` - yaoyu-33 (33 days old)
13. **#2021** `[ckpt] fix` - community (34 days old)

### Release-critical (r0.3.0 tagged, still open)
1. **#2510** - Nemotron 3 Nano perf configs (**APPROVED — merge now**)
2. **#2499** - Qwen3 30B B200 config fix
3. **#2397** - LLAMA3 70B LoRA B300/B200 (**APPROVED — merge now**)
4. **#2358** - MoE Sequential MLP mappings
5. **#2327** - Log git commit (draft)
6. **#2323** - Local checkpoint fix
7. **#2183** - create_attention_mask (**APPROVED — merge now**)
8. Cherry-picks: #2513, #2495, #2493, #2483, #2406, #2332

### Needs reconciliation (duplicate/competing PRs)
- **#2396 vs #2397** - both "Onboard LLAMA3 LoRA to B200/B300" by rhmukundan
- **#2143 vs #2140** - both "Qwen3-VL video" by different community authors

### Community PRs needing attention (no review > 2 weeks)
- **#2028** - Ling MoE V2 (34 days, no review)
- **#2057** - Typing imports refactor (32 days)
- **#2194** - LoRA merge TP>1 fix (22 days)
- **#2275** - Kimi 2.5 (16 days, zero review status)

### Merged today (Feb 25)
- **#2514** - training fix (yaoyu-33)
- **#2511** - remove deprecated MP params (dimapihtar)
- **#2516** - TestPyPI CI fix (ko3n1g)
- **#2484** - modelopt bump (ko3n1g)
- **#2492** - 26.02 perf summary (malay-nagda)
- **#2068** - transformers v5 upgrade (yaoyu-33) ← closes issue #2318
- **#2416** - LoRA TFLOPs fix (rhmukundan)
