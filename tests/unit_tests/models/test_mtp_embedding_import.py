"""The MTP input copy must be loaded even when the output head is untied."""

import logging
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.conversion import model_bridge as mb
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping


@pytest.mark.unit
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied-control"])
def test_mtp_embedding_values_after_hf_import(monkeypatch, tied):
    class Bridge(mb.MegatronModelBridge):
        def provider_bridge(self, hf_pretrained):
            raise NotImplementedError

        def mapping_registry(self):
            return MegatronMappingRegistry(
                AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
                AutoMapping("output_layer.weight", "lm_head.weight"),
            )

    class State(dict):
        @property
        def source(self):
            return SimpleNamespace(get_all_keys=self.keys)

    # Both ranks start with the SAME construction-time embedding values.
    # This is the state AFTER MCore's initialization all-reduce, BEFORE HF import.
    initial = torch.tensor([[0.125, -0.25], [0.375, -0.5], [0.625, -0.75]])
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    head = expected.clone() if tied else expected + 10
    stages = []
    for rank in range(2):
        model = torch.nn.Module()
        model.config = SimpleNamespace(
            pipeline_model_parallel_size=2,
            share_embeddings_and_output_weights=tied,
            num_moe_experts=0,
            mtp_num_layers=1,
        )
        model.pre_process, model.mtp_process = rank == 0, rank == 1
        model.embedding = torch.nn.Module()
        model.embedding.word_embeddings = torch.nn.Embedding.from_pretrained(initial.clone(), freeze=False)
        # AutoMapping's ordinary column-parallel dispatch, with TP=1.
        model.embedding.word_embeddings.tensor_model_parallel = True
        model.embedding.word_embeddings.partition_dim = 0
        if rank == 1:
            model.output_layer = torch.nn.Linear(2, 3, bias=False)
            model.output_layer.tensor_model_parallel = True
            model.output_layer.partition_dim = 0
        stages.append(model)

    hf = SimpleNamespace(
        config=SimpleNamespace(),
        model_name_or_path="tiny-untied-mtp",
        state=State({"model.embed_tokens.weight": expected, "lm_head.weight": head}),
    )
    monkeypatch.setattr(mb, "unwrap_model", lambda model: model)
    monkeypatch.setattr(mb, "persistent_buffers", lambda model: [])
    monkeypatch.setattr(
        mb,
        "get_module_and_param_from_name",
        lambda models, name, vp: (models[vp].get_submodule(name.rsplit(".", 1)[0]), models[vp].get_parameter(name)),
    )
    monkeypatch.setattr(mb.MegatronModelBridge, "_with_progress_tracking", lambda self, tasks, *args: tasks)
    # Simulate only PP communication. Task creation, mappings, copy_, and the
    # complete finalize_hf_import -> _broadcast_shared_embeddings path stay real.
    monkeypatch.setattr(
        mb.parallel_state, "get_pipeline_model_parallel_group", lambda: SimpleNamespace(size=lambda: 2)
    )
    monkeypatch.setattr(mb.parallel_state, "get_embedding_group", lambda: "embedding-group")
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, **kwargs: tensor.copy_(stages[0].embedding.word_embeddings.weight.detach()),
    )
    for rank, model in enumerate(stages):
        monkeypatch.setattr(mb.parallel_state, "get_pipeline_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda **kwargs: rank)
        monkeypatch.setattr(mb, "is_pp_first_stage", lambda group: rank == 0)
        monkeypatch.setattr(mb, "is_pp_last_stage", lambda group: rank == 1)

        def gather_names(result, local_names, **kwargs):
            result[:] = [list(dict(stages[0].named_parameters())), list(dict(stages[1].named_parameters()))]
            result[rank] = local_names

        monkeypatch.setattr(torch.distributed, "all_gather_object", gather_names)
        with torch.no_grad():
            Bridge().load_weights_hf_to_megatron(hf, [model])

    first, mtp = [model.embedding.word_embeddings.weight.detach() for model in stages]
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(stages[1].output_layer.weight, head, rtol=0, atol=0)
    logging.warning(
        "tied=%s; stage0=%s; MTP=%s; HF=%s; output_head_correct=True",
        tied,
        first.tolist(),
        mtp.tolist(),
        expected.tolist(),
    )
    # Check actual input values consumed by MTP, not task counts or helper calls.
    tokens = torch.tensor([1, 2])
    torch.testing.assert_close(
        stages[1].embedding.word_embeddings(tokens),
        torch.nn.functional.embedding(tokens, expected),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(mtp, expected, rtol=0, atol=0)
