## Description: <br>
Analyze and compare PyTorch CUDA memory snapshots produced by Megatron Bridge's `ProfilingConfig(record_memory_history=True)`. Replays the recorded allocation timeline to plot memory over time, compare two runs, and attribute peak memory to source code locations. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>

## Use Case: <br>
Developers and engineers debugging out-of-memory failures and peak-memory regressions in Megatron Bridge GPU training workloads, comparing memory behavior between two configurations, or attributing GPU memory to specific allocation sites. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

The bundled scripts use only the Python standard library. The generated HTML timeline loads Plotly.js from a public CDN when opened in a browser; it is not a Python dependency. <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

Risk: Memory snapshot pickle files are deserialized with `pickle.load`, which executes arbitrary code for untrusted input. <br>
Mitigation: Only analyze snapshot files produced by your own training runs or from a trusted source. <br>

Risk: Snapshot files embed absolute source paths and Python stack frames from the machine that produced them, which may reveal internal directory layouts. <br>
Mitigation: Review before sharing snapshot files or script output outside your organization. <br>

## Reference(s): <br>
- [Profiling Documentation](../../docs/training/profiling.md) <br>
- [Memory Tuning Skill](../nemo-mbridge-perf-memory-tuning/SKILL.md) <br>
- [CUDA Graphs Skill](../nemo-mbridge-perf-cuda-graphs/SKILL.md) <br>
- [Understanding GPU Memory (PyTorch blog)](https://pytorch.org/blog/understanding-gpu-memory-1/) <br>

## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Generated HTML report, JSON] <br>
**Output Format:** [Markdown with inline bash and Python code blocks; scripts emit formatted text tables, JSON, or a standalone HTML timeline] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- `claude-code` <br>
- `codex` <br>

## Evaluation Tasks: <br>
Evaluated against 2 tasks in the NVSkills-Eval external profile (astra-sandbox environment). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
