# Contributor Agent Skills Reference

Contributor-facing repo workflow skills for AI coding agents working on
Megatron Bridge itself.

These skills are intentionally outside the public `skills/` catalog sync path.
The docs build copies them to `docs/contributor-skills/` so they can be linked
from the contributor documentation without publishing them as catalog skills.

```{toctree}
:caption: Development Workflow
:maxdepth: 1

contributor-skills/mbridge-build-and-dependency/SKILL
contributor-skills/mbridge-bump-dependency/SKILL
contributor-skills/mbridge-cicd/SKILL
contributor-skills/mbridge-linting-and-formatting/SKILL
contributor-skills/mbridge-testing/SKILL
```

```{toctree}
:caption: Model Contribution
:maxdepth: 1

contributor-skills/mbridge-adding-model-support/SKILL
contributor-skills/mbridge-adding-model-support/llm-patterns
contributor-skills/mbridge-adding-model-support/vlm-patterns
contributor-skills/mbridge-adding-model-support/recipe-patterns
contributor-skills/mbridge-adding-model-support/tests-and-examples
contributor-skills/mbridge-parity-testing/SKILL
contributor-skills/mbridge-nemo-rl-e2e-testing/SKILL
contributor-skills/mbridge-verl-e2e-testing/SKILL
```
