# Research Ideas Tracker

> Collaboratively maintained by the researcher and Claude.
> Pick an idea → move to "In Progress" → implement → evaluate → record results.

## Status Legend
- 🆕 **New** — proposed, not yet started
- 🔬 **In Progress** — currently being implemented/tested
- ✅ **Done** — evaluated, results recorded
- ❌ **Abandoned** — tried or rejected with reason

---

## Ideas

### 1. Layer-wise loss
**Status:** 🆕

**Date proposed:** 2026-04-06

**Hypothesis:** Instead of applying the cross entropy loss only at the final layer, one can apply the loss for masked position for each layer and feed the predictions to the next layer. This way each layer can somehow build on top of the predictions from the previous layer. It requires some additional parameters which process the concatenation of the predictions from the previous layer with the input embeddings of each layer and project them to the embedding size of the model.

**Related Works:** https://arxiv.org/abs/2110.07515


**Files to modify:** `src/megatron/` (loss computation, forward pass per layer)

**Expected metric:** GSM8k, MBPP

---

### 2. Mask Scheduling
**Status:** ❌ Avg 72.29% (+0.31%), below threshold. GSM8k +1.3% but MBPP+ -1.6%.

**Date proposed:** 2026-04-06

**Hypothesis:** Early during training, the model has no knowledge about the task of unmasking words therefore having large number of mask tokens in a sentence might make the task too hard to solve. Therefore some sort of curriculum learning might be helpful where in the beginning the number of masked tokens are limited and as training progresses the ratio of masked tokens increases. It might be worth going all the way to fully masked tokens in case single step denoising is appealing.

**Related Works:** https://arxiv.org/pdf/2008.07905


**Files to modify:** `src/megatron/` (noise/mask schedule, data pipeline)

**Expected metric:** GSM8k, MBPP

---

### 3. Continuous Masking instead of Binary Masking
**Status:** 🆕

**Date proposed:** 2026-04-06

**Hypothesis:** The default masking strategy is to either replace tokens with a mask or leave them as they are. Instead of switching tokens in a discrete fashion, one can compute the convex combination of the masked token and the ground truth token and then try to predict the direction of denoising. This requires modification of the inference as well where the tokens get updated in each step. It becomes more similar to the typical flow matching models.

**Related Works:** https://arxiv.org/pdf/2505.18495


**Files to modify:** `src/megatron/` (masking logic, embedding interpolation, inference loop)

**Expected metric:** GSM8k, MBPP

---

### 4. Intra Block Dependence
**Status:** 🆕

**Date proposed:** 2026-04-06

**Hypothesis:** Current block diffusion models handle all tokens within a block as conditionally independent of each other. This assumption makes generation difficult, requiring multiple iterative steps to resolve the dependencies. One way to address this is to treat the tokens as a graph and learn the transitions between them based on the actual tokens. The Directed Acyclic Transformer (DAT) is one implementation of this approach. This is to some extend related to varaible block size as well because the model can decide to skip some tokens. Note that the block size must be greater than the target sentence to make this idea reasonable. It's very closely related to CTC based training.

**Related Works:** https://arxiv.org/pdf/2205.07459


**Files to modify:** `src/megatron/` (block generation, attention masking, decoding strategy)

**Expected metric:** GSM8k, MBPP

---


### 5. Cosine Mask Schedule (Gentler Warmup)
**Status:** ❌ Avg 70.51% (-1.47%), worse than baseline. Mask scheduling direction abandoned.

**Date proposed:** 2026-04-07

**Hypothesis:** The mask scheduling experiment (#2) showed that curriculum mask ratio warmup helps GSM8k (+1.3%) but hurts MBPP+ (-1.6%). The linear warmup from 0.1 was too aggressive — the model spent too many early iterations with very few masked tokens, under-training its denoising ability for code. A cosine schedule with a higher minimum (0.3) and shorter warmup (1000 iters) may preserve the GSM8k gains while reducing the MBPP+ regression.

**Related Works:** Derived from experiment #2 results.


**Files to modify:** Config only (submit_pretraining_3b.sh)

**Expected metric:** GSM8k, MBPP

---

### 6. DLM Loss Weighting by Mask Ratio
**Status:** 🆕

**Date proposed:** 2026-04-07

**Hypothesis:** The current DLM loss weights each masked token by 1/p_mask (inverse of masking probability), which up-weights tokens from low-mask-ratio samples. An alternative is to use a different weighting scheme — e.g., sqrt(1/p_mask) or a learned schedule — to shift the loss emphasis toward harder (high mask ratio) or easier (low mask ratio) denoising. Since mask scheduling experiments showed GSM8k benefits from early easy examples, loss re-weighting could achieve a similar effect without actually changing the mask distribution, avoiding the MBPP+ regression.

**Related Works:** MDLM (Sahoo et al. 2024), Score Entropy Discrete Diffusion


**Files to modify:** `src/megatron/bridge/diffusion/models/common/dgpt_step.py` (loss weighting)

**Expected metric:** GSM8k, MBPP

---
## Adding New Ideas

Copy this template:
```
### N. Idea Title
**Status:** 🆕
**Date proposed:** YYYY-MM-DD
**Hypothesis:** ...
**Related Works:** ...
**Files to modify:** ...
**Expected metric:** ...

---
```
