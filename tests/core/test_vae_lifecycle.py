from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.base_model import xFuserModel
from xfuser.model_executor.models.runner_models.vae_manager import VAEManager


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.decoded = []

    def decode(self, z):
        self.decoded.append(z)
        return z


class Pipe:
    def __init__(self, vae):
        self.vae = vae


class Staged:
    _decoding_vaes = xFuserModel._decoding_vaes
    _convert_vae_to_channels_last = xFuserModel._convert_vae_to_channels_last

    def __init__(self, first, second=None):
        self.pipe = Pipe(first)
        self.second_pipe = Pipe(second) if second is not None else None
        self._vae_manager = VAEManager(
            config=object(),
            capabilities=object(),
            settings=SimpleNamespace(model_output_type="image"),
        )


def test_decoding_vaes_accepts_any_number_of_pipeline_stages():
    manager = VAEManager(
        config=object(),
        capabilities=object(),
        settings=SimpleNamespace(model_output_type="image"),
    )
    first, second, third = TinyVAE(), TinyVAE(), TinyVAE()

    vaes = manager.decoding_vaes(
        [Pipe(first), Pipe(second), Pipe(third), Pipe(first)]
    )

    assert vaes == [first, second, third]


def test_channels_last_conversion_preserves_decode_output_for_every_stage():
    first, second = TinyVAE(), TinyVAE()
    staged = Staged(first, second)
    staged._convert_vae_to_channels_last()
    sample = torch.randn(1, 4, 8, 8)

    for vae in (first, second):
        assert torch.equal(vae.decode(sample), sample)
        assert vae.decoded[-1].is_contiguous(memory_format=torch.channels_last)


def test_initialize_sets_up_every_parallel_vae_before_enabling_options(monkeypatch):
    first, second = object(), object()
    events = []

    class Runner:
        initialize = xFuserModel.initialize
        _decoding_vaes = xFuserModel._decoding_vaes

        def __init__(self):
            self.config = SimpleNamespace(
                use_parallel_vae=True,
                use_torch_compile=False,
                create_config=lambda: (object(), None),
            )
            self._vae_manager = mock.Mock()
            self._vae_manager.decoding_vaes.side_effect = (
                lambda pipes: [pipe.vae for pipe in pipes]
            )

        def _load_model_checked(self):
            return Pipe(first)

        def _get_runtime_state_pipeline(self):
            return self.pipe

        def _post_load_and_state_initialization(self, input_args):
            events.append("post-load")
            self.second_pipe = Pipe(second)

        def _enable_options(self):
            events.append("options")

    runner = Runner()
    runner._vae_manager.setup_parallel_vae.side_effect = (
        lambda vaes: events.append("parallel")
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(base_model, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(base_model, "initialize_runtime_state", lambda *args: None)

    runner.initialize({})

    assert events == ["post-load", "parallel", "options"]
    runner._vae_manager.setup_parallel_vae.assert_called_once_with([first, second])
