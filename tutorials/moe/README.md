# MoE Playbook (Megatron-Core & Megatron-Bridge)

Practical playbook for **Mixture of Experts**: concepts, single-GPU training walkthrough, and optional multi-GPU / production material in an appendix.

- **`moe.ipynb`** — Main notebook, structured in three layers:
  - **Path 1 (~30 min):** §1 when to use MoE → §2 concepts (routing, aux loss, total vs active compute) → §3 micro-train on one GPU → §6 short checklist
  - **Path 2:** Path 1 + **Appendix** (dispatch, EP/TP/PP, Mixtral-style configs, performance tuning, extended tips)
- **`moe_permute_impl.py`** — Naive (PyTorch) token permutation helpers plus a small benchmark runner (optional; appendix / advanced).

Open `moe.ipynb` and follow **“How to read this notebook”** at the top. Only the environment check and MoE micro-train cells execute by default; Megatron-Bridge blocks are reference-only unless that stack is installed.
