# On-Call Report — Week of 2026-03-02
Primary: Yu Yao

---

## 2026-03-02 (Monday, Day 1)

### MCore Bump PRs

**Status: 4 open bump PRs, ALL failing CI. No main bump merged since Feb 17 — 2-week backlog.**

| PR | Date | Failing Checks | Notes |
|---|---|---|---|
| [#2607](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2607) | Mar 02 (today) | `Launch_Unit_Tests`, `Nemo_CICD_Test` | Unit tests fail, functional tests skipped |
| [#2605](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2605) | Mar 01 | `Launch_Unit_Tests`, `Nemo_CICD_Test` | Same pattern |
| [#2596](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2596) | Feb 28 | `Launch_Unit_Tests`, `Nemo_CICD_Test` | Same pattern |
| [#2583](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2583) | Feb 27 | `Lint check`, `Nemo_CICD_Test` | Lint failure → entire CICD pipeline skipped |

**Last merged (main):** [#2399](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2399) on Feb 17.
**Last merged (r0.3.0):** [#2524](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2524) on Feb 27.

The consistent `Launch_Unit_Tests` failure across 3 days (Feb 28 – Mar 02) suggests a systematic breakage from a recent MCore change, not a flake. The Feb 27 bump (#2583) failed even earlier at lint. This needs investigation — check the unit test logs for the root cause.

### Issues Needing Attention

#### New today (Mar 2) — 0 comments, unassigned

| # | Title | Labels |
|---|---|---|
| [#2611](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2611) | HFDatasetBuilder hardcodes input/output JSONL keys, breaking chat-format datasets | community-request |
| [#2610](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2610) | tokenize_dataset crashes with TypeError when chat=True and pad_seq_to_mult > 1 | community-request |
| [#2609](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2609) | Using `rampup_batch_size` breaks the learning rate schedule | bug, community-request |

#### Recent (last 7 days) — needs first response or assignment

| # | Title | Age | Comments | Labels |
|---|---|---|---|---|
| [#2603](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2603) | Warning in checkpoint saving | Mar 1 | 1 (unassigned) | bug, community |
| [#2598](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2598) | Hardcoded GENERATION_REGEX logic causes incorrect masking | Feb 28 | 1 (unassigned) | bug, community |
| [#2573](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2573) | Energon Dataloader for All HF-based encoders | Feb 26 | 0 | enhancement |
| [#2427](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/2427) | FSDP to work with partial Cuda Graph | Feb 18 | 0 | performance/release |

### PRs Needing Attention

#### Approved — merge now

| PR | Title | Author | Approved for |
|---|---|---|---|
| [#2613](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2613) | Fix qwen 35 test | cuichenx | < 1 day |
| [#2590](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2590) | Tune kimi-k2 GB300 MXFP8 recipe | dingqingy-nv | 3 days (r0.3.0) |
| [#2578](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2578) | [training] fix: normalize cuda_graph_scope type | yaoyu-33 | 4 days |
| [#2571](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2571) | Update LoRa TFLOPs Formula Fix | rhmukundan | 4 days (r0.3.0) |
| [#2518](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2518) | add changes from PR2161 in MLM | JRD971000 | 5 days |
| [#2487](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2487) | Add AG/RS overlap distributed init support | jeffnvidia | 7 days |
| [#2476](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2476) | Fix attention_mask mismatch in compare.py | mohsinm-dev (community) | 8 days |

#### External/community PRs — need CI trigger (`/ok to test`)

| PR | Title | Author |
|---|---|---|
| [#2612](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2612) | [data] fix: Write full process_example_fn output to JSONL | shanecmoran |
| [#2597](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2597) | fix(fp16): fix HF-Megatron dtype mismatch | yxyOo |
| [#2586](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2586) | Fix Docker building instructions | janEbert |
| [#2582](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2582) | fix: use HF-style attn mask path for qwen3-vl RoPE | eagle705 |
| [#2532](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2532) | feat: Add EXAONE 4.0 model bridge | Bias92 |
| [#2529](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2529) | docs: Update user manual with new MoE features and FSDP | onel |
| [#2528](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2528) | doc: Add SFT dataset formats documentation | onel |
| [#2522](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2522) | Auto-slice output vocab for faster SFT | AmitMY |
| [#2482](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2482) | [PEFT] fix: allow args and kwargs for LoRATopKRouter forward | HollowMan6 |

#### Changes requested — follow up

| PR | Title | Author |
|---|---|---|
| [#2534](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2534) | Migration DFM -> Bridge | abhinavg4 |

### Training Sync

**Last synced:** Feb 11 (per [#2331](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2331))

**22 new MCore commits** touching `megatron/training/` since Feb 11. Key changes affecting Bridge-mapped files:

| Commit | MCore PR | Files Changed | Impact |
|---|---|---|---|
| `310082a6d` | [#3058](https://github.com/NVIDIA/Megatron-LM/pull/3058) μP: Maximal Update Parameterization | `training.py` (+14) | **HIGH** — new muP logic in training loop, likely needs Bridge mirror |
| `7418b1b8f` | [#3377](https://github.com/NVIDIA/Megatron-LM/pull/3377) Flexible VPP (fVPP) for hybrid model | `arguments.py` (+171), `checkpointing.py`, `training.py` | **HIGH** — large argument changes, new fVPP args, training loop changes |
| `61a293dbd` | [#3633](https://github.com/NVIDIA/Megatron-LM/pull/3633) Single-process checkpoint save | `arguments.py`, `checkpointing.py` | **MEDIUM** — new `--use-single-process-dist-ckpt-save` arg |
| `cb248024b` | [#2161](https://github.com/NVIDIA/Megatron-LM/pull/2161) Knobs for parallel save/load PGs | `arguments.py`, `checkpointing.py`, `training_config.py` | **MEDIUM** — new args for save/load process groups |
| `b17248aa3` | [#3570](https://github.com/NVIDIA/Megatron-LM/pull/3570) Move config src files into `config/` dir | `arguments.py`, new `config/` subdir | **MEDIUM** — import path changes |
| `b3599819d` | [#3584](https://github.com/NVIDIA/Megatron-LM/pull/3584) Fix cuda graph persist default | `arguments.py` (1 line) | **LOW** — default value fix |
| `47938afb6` | [#3319](https://github.com/NVIDIA/Megatron-LM/pull/3319) Improved parallel logging of LR | `training.py`, `utils.py` | **LOW** — logging improvement |
| `82cbd829d` | [#3408](https://github.com/NVIDIA/Megatron-LM/pull/3408) Remove deprecated model parallel params | `yaml_arguments.py` | **LOW** — cleanup, verify Bridge doesn't use removed params |

### Action Items

- [ ] **URGENT**: Investigate `Launch_Unit_Tests` failure on bump PRs #2607/#2605/#2596 — check [test logs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/actions/runs/22568663079/job/65375470436). This is blocking 2 weeks of MCore bumps.
- [ ] **URGENT**: Investigate lint failure on #2583 (Feb 27 bump) — may be a separate issue
- [ ] Merge approved PRs: #2613, #2590, #2578, #2571, #2518, #2487, #2476
- [ ] Trigger CI on external PRs: #2612, #2597, #2586, #2582, #2532, #2529, #2528, #2522, #2482
- [ ] Respond to new issues: #2611, #2610, #2609 (all from today, 0 comments)
- [ ] Triage issues #2603, #2598 (need assignment)
- [ ] Start training sync — prioritize μP (#3058) and fVPP (#3377) changes

---
