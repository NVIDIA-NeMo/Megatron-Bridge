# Contributor Skills

Contributor-facing task guides for AI coding agents working on this repository.

These skills are intentionally outside the public `skills/` catalog sync path.
They cover model support, local development, CI, testing, formatting, dependency
maintenance, and downstream compatibility checks.

## Discovery

Contributor skills are symlinked into `.claude/skills/` next to the public
skills, so Claude Code can discover both sets from one location. They are not
linked from `.agents/skills/` because that path is treated as a public
agentskills.io index surface in some tooling.

The Sphinx docs expose these skills through `docs/contributor-skills-index.md`,
separate from the public `docs/skills-index.md` catalog page.

## Available Skills

| Skill | Description |
|---|---|
| `mbridge-adding-model-support` | Add support for new LLM or VLM model families |
| `mbridge-build-and-dependency` | Set up development environments and manage dependencies |
| `mbridge-bump-dependency` | Bump pinned dependencies and drive the PR to green |
| `mbridge-cicd` | Use the PR workflow, trigger CI, and investigate failures |
| `mbridge-linting-and-formatting` | Follow ruff, typing, docstring, and review conventions |
| `mbridge-nemo-rl-e2e-testing` | Validate Bridge changes with external NeMo-RL workflows |
| `mbridge-parity-testing` | Verify numerical parity for HF to MCore conversions |
| `mbridge-testing` | Work with unit and functional test layout and tiers |
| `mbridge-verl-e2e-testing` | Validate Bridge changes with external verl workflows |
