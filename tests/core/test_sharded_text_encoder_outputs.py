"""A sharded transformers encoder must still record the outputs its pipeline reads.

Transformers resolves which submodule outputs to record by looking the model's class up in a
registry keyed when the model was constructed, and FSDP2 rebinds ``__class__`` on every module it
wraps. Wrapping an encoder's root therefore makes that lookup miss, and a forward asked for
``output_hidden_states=True`` comes back with ``hidden_states=None`` rather than raising — which
surfaces as a ``TypeError`` deep in a pipeline that subscripts ``hidden_states[-2]``.

These run single-rank: the class rebinding does not depend on world size.
"""

import os

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FSDP sharding needs an accelerator"
)


@pytest.fixture(scope="module")
def single_rank_process_group():
    """A single-rank group, taken down again along with the environment it needed.

    The rendezvous variables have to be unwound rather than left set: any test that later
    builds its own world reads them, inherits RANK=0 and WORLD_SIZE=1, and initialises a
    world of one instead of the one it asked for.
    """
    rendezvous = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    previous = {name: os.environ.get(name) for name in rendezvous}
    already_initialized = dist.is_initialized()
    if not already_initialized:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29601")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(0)
    yield dist.group.WORLD
    if not already_initialized:
        dist.destroy_process_group()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _tiny_qwen3():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    return Qwen3Model(config).to("cuda", torch.bfloat16).eval()


def _hidden_states(encoder):
    ids = torch.randint(0, 256, (1, 8), device="cuda")
    with torch.no_grad():
        output = encoder(input_ids=ids, output_hidden_states=True)
    return getattr(output, "hidden_states", None)


def test_an_unsharded_encoder_records_hidden_states(single_rank_process_group):
    """Guards the premise: if this fails, the installed transformers changed how recording is
    requested and the assertions below would pass for the wrong reason."""
    assert _hidden_states(_tiny_qwen3()) is not None


def test_a_sharded_encoder_still_records_hidden_states(single_rank_process_group):
    from xfuser.core.distributed.sharding import shard_component

    encoder = shard_component(
        _tiny_qwen3(),
        ["layers"],
        process_group=single_rank_process_group,
        device_id=0,
        memory_efficient_init=True,
    )

    hidden_states = _hidden_states(encoder)
    assert hidden_states is not None, (
        "sharding the encoder silently stopped transformers recording hidden states; "
        "the pipeline that reads hidden_states[-2] will fail with a TypeError"
    )
    assert len(hidden_states) == 3


def test_the_root_and_the_blocks_are_both_sharded(single_rank_process_group):
    """Recording must be preserved without giving up any sharding.

    Leaving the root unwrapped also preserves recording, but makes each block its own FSDP root with
    its own uninitialized comm context, and the cross-block forward prefetch then fails with
    'FSDPCommContext has no attribute all_gather_copy_in_stream'.
    """
    from torch.distributed.fsdp import FSDPModule

    from xfuser.core.distributed.sharding import shard_component

    encoder = shard_component(
        _tiny_qwen3(),
        ["layers"],
        process_group=single_rank_process_group,
        device_id=0,
        memory_efficient_init=True,
    )

    assert isinstance(encoder, FSDPModule)
    assert all(isinstance(layer, FSDPModule) for layer in encoder.layers)
