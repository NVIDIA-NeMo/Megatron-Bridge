---
name: Model Support Request
about: Request conversion support and training recipes for a new model
title: "<Model name> Model Support"
labels: ''
assignees: ''

---

Add support for \<model name\> model:

**Please include a link to the model's HuggingFace repo**
HF repo:

**These checklist items are required for all models in Megatron Bridge**

- [ ] Model providers
- [ ] Model bridge for HF conversion
- [ ] Unit tests (config and bridge)
- [ ] Model conversion functional tests

**For flagship models, these items are also needed**

- [ ] Optimal pretraining recipe
- [ ] Optimal finetuning recipe
- [ ] Recipe unit tests
- [ ] Recipe functional tests
- [ ] End to end CI tests

**Additional context**
Add any other context or screenshots about the model request here.
