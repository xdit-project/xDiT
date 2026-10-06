"""CPU-only checks for selecting individual FSDP pipeline components."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from xfuser.config.args import FlexibleArgumentParser, xFuserArgs
from xfuser.model_executor.models.runner_models.loading import meta_load, shard


class _Component:
    def __init__(self):
        self.materialized = False
        self.is_meta = True
        self.moves = []

    def to(self, device):
        assert not self.is_meta, "excluded meta component was moved before eager fill"
        self.moves.append(device)
        return self


def test_targeted_fsdp_cli_requires_a_sharding_degree():
    parser = FlexibleArgumentParser()
    args = xFuserArgs.add_runner_args(parser).parse_args(
        [
            "--model",
            "black-forest-labs/FLUX.2-dev",
            "--fully_shard_degree",
            "2",
            "--fully_shard_components",
            "text_encoder",
        ]
    )

    assert args.fully_shard_components == ["text_encoder"]
    with pytest.raises(ValueError, match="fully_shard_degree"):
        xFuserArgs(fully_shard_components=["text_encoder"])


def test_targeted_fsdp_fills_excluded_meta_components_before_placement(
    monkeypatch,
):
    """Only selected components reach FSDP; excluded local-meta components are filled first."""

    transformer = _Component()
    text_encoder = _Component()
    model = SimpleNamespace(
        config=SimpleNamespace(
            fully_shard_components=["text_encoder"],
            reshard_after_forward=True,
            memory_efficient_sharding=True,
            cache_method=None,
        ),
        settings=SimpleNamespace(
            fsdp_strategy={
                "transformer": {"wrap_attrs": ["blocks"]},
                "text_encoder": {"wrap_attrs": ["layers"]},
            }
        ),
        pipe=SimpleNamespace(
            components={
                "transformer": transformer,
                "text_encoder": text_encoder,
            },
            transformer=transformer,
            text_encoder=text_encoder,
        ),
    )

    loader = object.__new__(meta_load.ModelLoader)
    loader.model = model
    loader._local_blockwise_transformers = {transformer: True}
    fills = []

    def fill_transformer_local(component, name, strategy, device):
        fills.append((component, name, strategy, device))
        component.materialized = True
        component.is_meta = False

    loader.fill_transformer_local = fill_transformer_local
    loader.agreed_is_meta = lambda component, *_args: component.is_meta
    loader.self_fills_from_disk = lambda _component: False
    loader.broadcast_load = lambda *_args: None

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        meta_load,
        "get_world_group",
        lambda: SimpleNamespace(local_rank=0),
    )
    monkeypatch.setattr(meta_load.torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(shard, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(
        shard,
        "get_fs_group",
        lambda: SimpleNamespace(local_rank=0, device_group=object()),
    )
    monkeypatch.setattr(shard, "host_mem_gb", lambda: 0)
    monkeypatch.setattr(shard.torch.cuda, "memory_allocated", lambda *_args: 0)
    monkeypatch.setattr(shard.torch.cuda, "empty_cache", lambda: None)

    sharded = []

    def shard_component(component, wrap_attrs, *_args, **kwargs):
        sharded.append((component, wrap_attrs, kwargs))
        return f"fsdp({id(component)})"

    monkeypatch.setattr(shard, "shard_component", shard_component)

    shard.shard_pipeline_components(loader)

    assert fills == [(transformer, "transformer", {"wrap_attrs": ["blocks"]}, "cuda:0")]
    assert transformer.moves == ["cuda:0"]
    assert len(sharded) == 1
    component, wrap_attrs, kwargs = sharded[0]
    assert component is text_encoder
    assert wrap_attrs == ["layers"]
    assert kwargs["memory_efficient_init"] is True
    assert kwargs["meta_init"] is True
    assert model.pipe.text_encoder.startswith("fsdp(")
    assert model.pipe.transformer is transformer
