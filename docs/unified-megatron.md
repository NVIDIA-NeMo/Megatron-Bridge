# Unified Megatron-LM

## Overview

We present our motivation, proposal, and timeline to unify Megatron-LM and Megatron-Bridge into a single repository.

## Background

We have two repositories under the “Megatron” name: Megatron-LM and Megatron-Bridge.

**Megatron-LM** has two primary components:

* training: training loops, utils, and configs for launching distributed training at scale
* core: composable library with GPU-optimized building blocks for custom training frameworks

```{figure} images/unified-megatron/nemotron-pretraining.png
:alt: Nemotron pretraining uses megatron.training and megatron.core within Megatron-LM.
:align: center
:width: 70%

Nemotron Pretraining Example
```

**Megatron-Bridge** also has two primary components:

* training: training loops, utils, and configs for launching distributed training at scale
  * Distinct from Megatron-LM’s training
* bridge: bidirectional HuggingFace/Megatron conversion, recipes for all leading open-source models

Megatron-Bridge leverages its own components as well as Megatron-LM’s training and core components to enable pretraining, SFT, and PEFT of open-source models.

## Motivation

```{image} images/unified-megatron/training-duplication.png
:alt: Megatron-Bridge and Megatron-LM have separate training components, with redundant model entry points and conversion paths.
:align: center
:width: 100%
```

### Duplication

Even though both `megatron.bridge.training` (~28,000 lines of code) and `megatron.training` (~24,000 lines of code) leverage `megatron.core`, they each have their own training loops, dataloaders, and checkpoints. This has led to API and feature drift, redundant efforts, and inconsistent behaviors.

```{figure} images/unified-megatron/training-loop-drift.png
:alt: Merged PRs touching megatron.training and megatron.bridge.training from September 2025 through August 2026; August is partial through August 21.
:align: center
:width: 100%

Both training loops continue to drift apart
```

### CI Testing

Multiple entry points have created unnecessary strain on CI. We have many similar tests and a similar release process wasting valuable compute and engineering hours.

### User Challenges

* Challenging to **debug across two repositories**.
* Confusing to **contribute to two different repositories** when wanting to add an optimization (optimization goes into Megatron-LM, but the recipe needs to be updated in Megatron-Bridge).
* Struggling to **update to Megatron-Core top-of-tree at SOL** because of added Megatron-Bridge dependency.
* Difficult to **manage configuration and FSDP fixes** because of inconsistencies between Megatron-LM and Megatron Bridge.
* Frustrating that **MCore and Megatron Bridge use different configuration formats**
* Difficult to **maintain and debug training across Megatron Bridge and Megatron-LM**: Bridge recipe arguments stopped propagating after NeMo container updates, and diagnosis crossed into Megatron-LM.
* Difficult to **debug checkpointing across Megatron Bridge** and its embedded Megatron-LM dependency, including patching Megatron-LM’s distributed optimizer inside the Bridge container and encountering different configuration defaults between environments.

```{figure} images/unified-megatron/rl-library-dependencies.png
:alt: NeMo-RL, VeRL, Miles, and SkyRL use conversion, training, and core components across the two repositories.
:align: center
:width: 100%

RL libraries take components from both Megatron-Bridge and Megatron-LM
```

## Proposal

There are two components to our proposal:

1. Consolidate Megatron-Bridge’s `training` into Megatron-LM’s `training` providing a single training loop, configs, dataloader, and checkpoints for all Megatron users.
2. Copy the rest of Megatron-Bridge into Megatron-LM.

```{figure} images/unified-megatron/unified-architecture.png
:alt: Proposed Megatron-LM repository containing megatron.bridge, megatron.training, and megatron.core, with pretraining and RL entry points.
:align: center
:width: 100%

Final design of a unified Megatron-LM
```

Existing users, such as [verl](https://github.com/verl-project/verl/blob/1252cc71aa5bd82e5604322064d69bfe6454c660/verl/models/mcore/bridge.py#L17), already use the megatron namespace when using Megatron-Bridge (i.e. they use `import megatron.bridge`). This usage pattern will not change post-migration. Users will also continue to use the existing [megatron-bridge pip package](https://pypi.org/project/megatron-bridge/0.3.1/).

## Timeline

### December 2026 (Megatron-Core 0.21.0 release)

* Training loop will be fully unified in Megatron-LM’s `megatron/training`

### February 2027 (Megatron-Core 0.22.0 release)

* Megatron-Bridge will be fully migrated to Megatron-LM, most of the files will be found under `megatron/bridge`

PR Tracker: [https://github.com/NVIDIA/Megatron-LM/issues/9363](https://github.com/NVIDIA/Megatron-LM/issues/9363)

## User Impact

### megatron.bridge.training

We are fully removing the Bridge training loop and consolidating it into `megatron/training`. **Users who currently use `megatron/bridge/training` will need to migrate.**

### Checkpoints

We are unifying the two existing checkpoint formats into a Bridge-like checkpoint format. The “old” checkpoints will be backwards-compatible and will continue to work. Checkpoints generated in the new format will not be supported by “old” code.

### Bridge Containers

The Bridge container will continue to be published and used in the same manner.

### PyPi

The Bridge pip package will be combined with Megatron LM. Users will be able to run `pip install megatron-lm` to install Megatron Core, Bridge and Training packages.

### Megatron-Bridge Issues

Existing issues will remain open until resolved. Starting in Q1 ‘27, all new issues will need to be created in Megatron-LM.

### Megatron-Bridge PRs

We will be freezing the Megatron-Bridge repository on Nov 30, 2026. All PRs not merged in by then will need to be migrated to Megatron-LM by the user.
