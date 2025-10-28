import types
from unittest import mock

from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider
from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider
from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider
from megatron.bridge.training.config import DatasetBuildContext


def _cfg_stub(dataloader_type="single"):
    cfg = types.SimpleNamespace()
    cfg.dataset = types.SimpleNamespace(
        dataloader_type=dataloader_type,
        num_workers=0,
        data_sharding=True,
        pin_memory=False,
        persistent_workers=False,
    )
    cfg.train = types.SimpleNamespace(
        micro_batch_size=2,
        global_batch_size=4,
        eval_iters=2,
        skip_train=False,
        exit_signal=None,
        exit_signal_handler_for_dataloader=False,
    )
    return cfg


def _context_stub():
    return DatasetBuildContext(train_samples=4, valid_samples=4, test_samples=4, tokenizer=None)


def _train_state_stub():
    return types.SimpleNamespace(consumed_train_samples=0, consumed_valid_samples=0)


@mock.patch("megatron.bridge.data.vlm_datasets.mock_provider.AutoProcessor")
def test_mock_provider_provide_dataloaders_returns_dls(mock_auto_processor):
    # Dummy processor object
    class DummyProc:
        pass

    mock_auto_processor.from_pretrained.return_value = DummyProc()

    provider = MockVLMConversationProvider(
        sequence_length=16,
        hf_processor_path="dummy/model",
        num_images=0,
    )
    train_dl, valid_dl, test_dl = provider.provide_dataloaders(
        context=_context_stub(), cfg=_cfg_stub(), train_state=_train_state_stub()
    )

    assert train_dl is not None
    assert valid_dl is not None
    assert test_dl is not None


@mock.patch("megatron.bridge.data.vlm_datasets.preloaded_provider.AutoProcessor")
def test_preloaded_provider_handles_missing_paths(mock_auto_processor):
    class DummyProc:
        pass

    mock_auto_processor.from_pretrained.return_value = DummyProc()

    provider = PreloadedVLMConversationProvider(
        sequence_length=16,
        hf_processor_path="dummy/model",
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
    )
    train_dl, valid_dl, test_dl = provider.provide_dataloaders(
        context=_context_stub(), cfg=_cfg_stub(), train_state=_train_state_stub()
    )

    assert train_dl is None
    assert valid_dl is None
    assert test_dl is None


@mock.patch("megatron.bridge.data.vlm_datasets.hf_provider.AutoProcessor")
def test_hf_provider_provide_dataloaders_with_stub_maker(mock_auto_processor):
    class DummyProc:
        pass

    mock_auto_processor.from_pretrained.return_value = DummyProc()

    provider = HFDatasetConversationProvider(
        sequence_length=16,
        hf_processor_path="dummy/model",
        maker_name="dummy",
    )

    # Replace maker registry with minimal maker that returns one example
    def _fake_maker(**kwargs):
        return [
            {
                "conversation": [
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "world"}]},
                ]
            }
        ]

    provider._get_maker = lambda: _fake_maker  # type: ignore[method-assign]

    train_dl, valid_dl, test_dl = provider.provide_dataloaders(
        context=_context_stub(), cfg=_cfg_stub(), train_state=_train_state_stub()
    )

    assert train_dl is not None
    assert valid_dl is not None
    assert test_dl is not None
