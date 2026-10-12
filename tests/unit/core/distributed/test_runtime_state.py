"""Per-request input handling of the DiT runtime state."""

from types import SimpleNamespace

import pytest

from xfuser.config.config import InputConfig, RuntimeConfig
from xfuser.core.distributed import runtime_state as runtime_state_module
from xfuser.core.distributed.runtime_state import DiTRuntimeState


@pytest.fixture
def state(monkeypatch):
    # The pipeline-parallel group is a process-group boundary; a request at an
    # unchanged size only asks it to drop its cached transfer shapes.
    pp_group = SimpleNamespace(reset_buffer=lambda: None, set_config=lambda dtype: None)
    monkeypatch.setattr(runtime_state_module, "get_pp_group", lambda: pp_group)

    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.runtime_config = RuntimeConfig(warmup_steps=4)
    state.input_config = InputConfig(height=512, width=512, num_frames=9, batch_size=1)
    state.ready = True
    return state


def _image_request(state, num_inference_steps):
    state.set_input_parameters(height=512, width=512, batch_size=1, num_inference_steps=num_inference_steps)


def _video_request(state, num_inference_steps):
    state.set_video_input_parameters(
        height=512, width=512, num_frames=9, batch_size=1, num_inference_steps=num_inference_steps
    )


@pytest.mark.parametrize("request_fn", [_image_request, _video_request], ids=["image", "video"])
def test_short_request_does_not_shrink_warmup_of_later_requests(state, request_fn):
    # Pipelines read runtime_config.warmup_steps for every request and fall
    # back to synchronous steps when a request has no more steps than that.
    warmup_seen = []
    for num_inference_steps in (28, 2, 28):
        request_fn(state, num_inference_steps)
        warmup_seen.append(state.runtime_config.warmup_steps)

    assert warmup_seen == [4, 4, 4]
