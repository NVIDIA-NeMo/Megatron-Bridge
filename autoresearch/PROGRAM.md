# AutoResearch Program — Megatron-Bridge Diffusion LLM

> This file instructs the AI agent on how to autonomously conduct research experiments.
> The agent drives the entire loop. The user triggers it and reviews results.

## Objective

Improve the Nemotron Diffusion 3B model's downstream performance (primarily GSM8k)
through systematic experiments on training methodology, architecture, and hyperparameters.

## Autonomous Loop

When triggered, the agent executes this loop **without waiting for user input** between steps:

```
1. READ    — Load IDEAS.md, review prior experiment results
2. RANK    — Prioritize ideas by (expected impact × feasibility / cost)
3. PICK    — Select the highest-priority 🆕 idea
4. BRANCH  — Create git branch autoresearch/<idea-name> and a git worktree for isolation
5. IMPLEMENT — Make the code/config changes in the worktree
6. SESSION — Request an interactive Slurm session (GPU node with container)
7. TEST    — Run training in --direct mode inside the session
8. FIX     — If test fails, diagnose and fix. The interactive session is available up to 4 hours.
9. COMMIT  — Commit all changes and record the git hash in the experiment README
10. SUBMIT  — Submit Slurm job for full training
11. WAIT    — Poll Slurm job status until complete
12. EVAL   — Run GSM8k and MBPP evaluation
13. RECORD — Save results and update RESULTS.md, update IDEAS.md, commit to branch
14. DECIDE — If positive, create PR. If negative, just document learnings.
15. REFLECT — Analyze all results so far. Propose a new experiment and add it to IDEAS.md.
               If existing directions look unpromising, research new approaches (papers, techniques)
               and add a fresh idea with justification.
16. LOOP   — Return to step 1 for the next idea
```

The agent stops the loop when:
- All ideas are tried (no 🆕 ideas left)
- The user interrupts
- A critical failure occurs that can't be auto-resolved

## Idea Prioritization Criteria

Rank ideas by this scoring (highest score first):

| Factor | Weight | Description |
|--------|--------|-------------|
| Expected impact | 40% | How much GSM8k improvement is plausible? |
| Feasibility | 30% | Config-only > small code change > large code change |
| Risk | 20% | Low risk of breaking training or wasting compute |
| Novelty | 10% | Less-explored ideas get a bonus |

## Codebase Boundaries

### Safe to Modify
- `src/megatron/bridge/diffusion/` — model, losses, forward steps
- `examples/diffusion/recipes/nemotron_diffusion/` — recipes, configs
- `examples/diffusion/recipes/nemotron_diffusion/conf/` — data/training configs
- New files under `autoresearch/experiments/<exp_name>/`

### Do NOT Modify
- `3rdparty/Megatron-LM/` — upstream dependency
- `src/megatron/bridge/training/` — core training loop (unless the idea specifically requires it)
- `scripts/performance/` — unrelated perf benchmarking
- Other model families (flux, wan, etc.)

## Phase Details

### Phase: Branch & Worktree Setup

Each experiment runs in an **isolated git worktree** so multiple agents can work
concurrently without interfering with each other.

```bash
# Create branch and worktree
WORKTREE_DIR=~/code/Megatron-Bridge-worktrees/autoresearch-<idea-name>
git worktree add ${WORKTREE_DIR} -b autoresearch/<idea-name>
```

All code changes for this experiment are made in `${WORKTREE_DIR}`, not in the
main repo. The worktree has its own working directory but shares git history.

After the experiment is done (merged or abandoned):
```bash
git worktree remove ${WORKTREE_DIR}
```

### Phase: Interactive Validation

Interactive testing requires a GPU node. The container must mount **this experiment's
worktree**, not the main repo.

**Step 1 — Request an interactive session (mount the worktree):**
```bash
WORKTREE_DIR=~/code/Megatron-Bridge-worktrees/autoresearch-<idea-name>
srun -A coreai_dlalgo_llm \
  --partition interactive \
  --time 4:00:00 \
  --nodes=1 \
  --gpus-per-node=8 \
  --container-image nvcr.io/nvidian/nemo:26.04.rc4 \
  --container-mounts=$HOME:/home/snorouzi,/lustre:/lustre,/home/snorouzi/code/megatron-lm:/opt/megatron-lm,${WORKTREE_DIR}:/opt/Megatron-Bridge \
  --pty bash
```
Note: `${WORKTREE_DIR}` is mounted as `/opt/Megatron-Bridge` inside the container,
so all existing scripts work without modification.

**Step 2 — Run training inside the interactive session:**
```bash
bash submit_pretraining_3b.sh --direct
```
This runs on localhost with `global_batch_size=8, micro_batch_size=1`.

**Step 3 — Verify:**
- Training starts without errors
- Loss decreases over first ~20 iterations
- No OOM or NCCL errors
- **Code integrity check:** Inside the container, verify the mounted code matches
  the agent's commit:
  ```bash
  git -C /opt/Megatron-Bridge rev-parse HEAD
  ```
  This must match the commit hash the agent is working on. If it doesn't, STOP —
  the container is running stale code. Debug the mount before proceeding.

If validation fails, the agent should:
1. Read the error from the log
2. Diagnose the root cause
3. Fix the code **inside the same interactive session** (it stays open for up to 4 hours)
4. Retry — keep iterating as long as the interactive session is alive
5. If the session expires before a successful run, mark idea as ❌ with reason and move on

### Phase: Full Training
Submit the Slurm job from within the worktree. Set `MB_DIR` to point to the
worktree so the container mounts the correct code:
```bash
cd ${WORKTREE_DIR}
MB_DIR=${WORKTREE_DIR} bash submit_pretraining_3b.sh 2
```
Training runs for ~5000 iterations on 16 nodes (~4 hours).

To poll job status:
```bash
squeue -j <job_id> -h -o "%T"
```
Poll every 10 minutes. When status is no longer RUNNING/PENDING, proceed to eval.

### Phase: Evaluation
Evaluation uses `--parallel-tasks` which submits one Slurm job per eval task.
This means you must **track the eval job IDs** and wait for them to complete, just
like the training job.

1. Submit eval jobs:
   ```bash
   bash examples/diffusion/recipes/nemotron_diffusion/eval_megatron.sh \
     --parallel-tasks \
     --checkpoint <checkpoint_path> \
     --exp-name <exp_name> \
     --eval-tasks gsm8k_cot,mbpp,mbpp_plus \
     --modes dllm
   ```
   Capture the submitted Slurm job IDs from the output.

2. Poll all eval jobs until completion:
   ```bash
   squeue -j <job_id1>,<job_id2>,... -h -o "%j %T"
   ```
   Poll every 5 minutes. Wait until all jobs finish.

3. Collect results using `collect_results.py`:
   ```bash
   python examples/diffusion/recipes/nemotron_diffusion/collect_results.py \
     <exp_name>/seed_42 --detailed
   ```
   Results are stored at: `/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_eval_results/<exp_name>/`

4. Parse the collect_results.py output to extract metrics and update RESULTS.md.

### Phase: Record & Decide
1. Save results to `autoresearch/experiments/<exp_name>/results.md`:
   ```markdown
   # Results: <exp_name>
   ## GSM8k (8-shot CoT)
   - Baseline: X.X%
   - This experiment: Y.Y%
   - Delta: +/-Z.Z%
   ## Notes
   ...
   ```
2. Update IDEAS.md: change status to ✅ or ❌ with results summary
3. Git commit all changes on the experiment branch
4. If **Avg** (mean of GSM8k Strict, GSM8k Flex, MBPP, MBPP+) exceeds baseline Avg by ~0.5% (i.e., Avg >= ~72.5%): create a PR to main with results summary
5. If Avg improvement is less than ~0.5%: document learnings, move on

## Evaluation Criteria

### Primary Metric
- **Avg** = mean of (GSM8k Strict, GSM8k Flex, MBPP, MBPP+)
- Baseline Avg: **71.98%**
- An experiment is **positive** if Avg exceeds baseline Avg by ~0.5% (i.e., Avg >= ~72.5%)

### Secondary Metrics
- Latency at inference (NFE, ms/sample)

## Constraints
- Each experiment: **one variable at a time** — minimal, isolated change
- Training budget: ~4 hours × 16 nodes = 64 node-hours per experiment
- Prefer config-only changes over code changes
- Always validate interactively before submitting to Slurm
- Keep experiment branches clean
- Log all key decisions in the experiment README.md

## Agent Behavior
- Fully autonomous: do NOT pause between phases
- Print a one-line status update at each phase transition (e.g., "▶ Phase: TEST — running interactive validation")
- If a failure is unrecoverable, mark the idea as ❌ and continue to the next
- If no GPU is available for interactive test, wait and retry
- The user can interrupt at any time; the agent should checkpoint its state in IDEAS.md
