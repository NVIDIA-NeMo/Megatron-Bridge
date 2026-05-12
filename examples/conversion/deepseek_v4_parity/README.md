# DSv4-Flash parity harness

Adapted from `examples/conversion/deepseek_v3_parity/` per the method documented in that
directory's `HANDOFF.md`. Two independent Slurm jobs save logits to disk; an offline
comparator computes per-prompt cosine, max-diff, and top-K agreement.

## Cluster + footprint

DSv4 currently requires **TP=1** and the model does not fit on A100 80 GB. The harness
targets a B200 192 GB cluster (e.g. OCI hsg). Default config: 1 node, 4 B200 GPUs,
TP=1, EP=4, PP=1 for the Flash variant. For Pro variants edit the sbatch to use EP=16
on 16 GPUs.

DSv4 native HF support landed in transformers, so `run_hf_reference.py` runs without
`trust_remote_code` and shards via `device_map="auto"`.

## Files

| File | Role |
|---|---|
| `prompts.json` | 8 fixed prompts (factual / code / math / chat / edge cases) |
| `run_hf_reference.py` | HF reference forward pass; saves `logits_hf.pt` |
| `run_mbridge.py` | Megatron-Bridge forward pass via `AutoBridge`; saves `logits_mb.pt` |
| `compare_logits.py` | Offline comparator; reports cosine at last-real + last-padded positions |
| `dsv4_hf_reference.sbatch` | 1-node HF reference sbatch (edit the `<EDIT-ME>` paths) |
| `dsv4_mbridge.sbatch` | 1-node Megatron-Bridge sbatch (edit the `<EDIT-ME>` paths) |

## How to run

1. Edit the two sbatch scripts and replace each `/lustre/<EDIT-ME>/users/<EDIT-ME>` block
   with your workspace, the container path, and the HF snapshot path.
2. Confirm the MCore fork pointed to by `MCORE_DIR` carries the DSv4 prerequisites
   (PRs #3430, #4458, #4481, #4518). See `examples/models/deepseek_v4/README.md`.
3. Submit both jobs:
   ```bash
   sbatch examples/conversion/deepseek_v4_parity/dsv4_hf_reference.sbatch
   sbatch examples/conversion/deepseek_v4_parity/dsv4_mbridge.sbatch
   ```
4. After both finish, run the comparator on the login node:
   ```bash
   python examples/conversion/deepseek_v4_parity/compare_logits.py \
       --hf /chcui/parity/dsv4/logits_hf.pt \
       --mb /chcui/parity/dsv4/logits_mb.pt \
       --threshold 0.99 \
       --report /chcui/parity/dsv4/report.md
   ```

The comparator reports cosine at the last real position (primary metric) and, when both
artifacts carry it, also at the last padded position (where BF16 cosine is typically
tighter — see `parity-testing` skill).

## Reference numbers from the PR

`tests/dsv4-bridge-handoff.md` reports cosine **0.935** at the last-real position with
top-5 overlap 4/5. Anything lower than that signals a regression in the bridge or in the
MCore fork; anything higher (especially at the padded position) is a tighter result.
