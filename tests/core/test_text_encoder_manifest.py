"""Mapping a transformers text encoder's live tensor names onto its checkpoint keys.

The blockwise fill reads one tensor at a time by name, so a wrong mapping loads wrong weights
rather than failing. resolve_transformers_manifest must therefore accept a mapping only when every
checkpoint key is accounted for and every live tensor is covered, and refuse anything else.
"""

import re

import pytest
import torch

from xfuser.model_executor.models.runner_models.loading.checkpoint import (
    CheckpointManifest,
    CheckpointRequest,
)
from xfuser.model_executor.models.runner_models.loading.text_encoder_adapter import (
    resolve_transformers_manifest,
)

REQUEST = CheckpointRequest(model_name_or_path="/models/demo", subfolder="text_encoder")


def discovery(*keys, path="/models/demo/text_encoder/model.safetensors"):
    return lambda request, basename: CheckpointManifest({key: path for key in keys})


class Renaming:
    """Stands in for a transformers WeightRenaming: one regex rewrite of a checkpoint key."""

    def __init__(self, source, target):
        self.source, self.target = source, target

    def rename_source_key(self, key):
        renamed, count = re.subn(self.source, self.target, key, count=1)
        return renamed, (self.source if count else None)


class Encoder(torch.nn.Module):
    """A two-block encoder with a persistent buffer and a non-persistent one."""

    base_model_prefix = "model"

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
        )
        self.register_buffer("scale", torch.ones(2), persistent=True)
        self.register_buffer("cache", torch.ones(2), persistent=False)


LIVE_KEYS = ("layers.0.weight", "layers.1.weight", "scale")


def resolve(component, discover, conversions=()):
    return resolve_transformers_manifest(
        component, REQUEST, discover=discover, conversions=list(conversions)
    )


def test_keys_that_already_match_are_mapped_unchanged():
    manifest, refusal = resolve(Encoder(), discovery(*LIVE_KEYS))

    assert refusal is None
    assert manifest.checkpoint_keys == {key: key for key in LIVE_KEYS}


def test_a_checkpoint_saved_under_the_base_model_prefix_is_mapped_onto_it():
    prefixed = tuple(f"model.{key}" for key in LIVE_KEYS)

    manifest, refusal = resolve(Encoder(), discovery(*prefixed))

    assert refusal is None
    assert manifest.checkpoint_keys == {key: f"model.{key}" for key in LIVE_KEYS}


def test_the_prefix_is_decided_per_key_not_once_for_the_checkpoint():
    """Mistral3 stores one tensor under the prefix and another without it."""
    manifest, refusal = resolve(
        Encoder(), discovery("model.layers.0.weight", "layers.1.weight", "scale")
    )

    assert refusal is None
    assert manifest.checkpoint_keys["layers.0.weight"] == "model.layers.0.weight"
    assert manifest.checkpoint_keys["layers.1.weight"] == "layers.1.weight"


def test_a_registered_renaming_is_applied_to_the_checkpoint_key():
    manifest, refusal = resolve(
        Encoder(),
        discovery("blocks.0.weight", "blocks.1.weight", "scale"),
        conversions=[Renaming(r"blocks", "layers")],
    )

    assert refusal is None
    assert manifest.checkpoint_keys["layers.0.weight"] == "blocks.0.weight"


def test_a_non_persistent_buffer_is_not_required_from_the_checkpoint():
    """Rotary caches and the like are recomputed on forward and never stored."""
    manifest, refusal = resolve(Encoder(), discovery(*LIVE_KEYS))

    assert refusal is None
    assert "cache" not in manifest.checkpoint_keys


def test_a_tied_tensor_reads_the_key_of_the_tensor_it_is_tied_to():
    """Only the tying target is stored, so the alias resolves to that key, not one of its own."""

    class Tied(Encoder):
        _tied_weights_keys = {"head.weight": "layers.0.weight"}

        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2, bias=False)

    manifest, refusal = resolve(Tied(), discovery(*LIVE_KEYS))

    assert refusal is None
    assert manifest.checkpoint_keys["head.weight"] == "layers.0.weight"


def test_a_tensor_shared_by_object_identity_is_also_covered():
    """init_empty_weights can leave a tie as shared objects rather than a declaration."""

    class Shared(Encoder):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2, bias=False)
            self.head.weight = self.layers[0].weight

    manifest, refusal = resolve(Shared(), discovery(*LIVE_KEYS))

    assert refusal is None
    assert manifest.checkpoint_keys["head.weight"] == "layers.0.weight"


def test_a_live_tensor_with_no_key_and_no_tie_is_refused():
    manifest, refusal = resolve(Encoder(), discovery("layers.0.weight", "scale"))

    assert manifest is None
    assert "no checkpoint key and no declared tie" in refusal
    assert "layers.1.weight" in refusal


def test_a_checkpoint_carrying_tensors_the_component_lacks_is_refused():
    """A fused or split layout leaves a key denoting nothing, which a renaming cannot explain."""
    manifest, refusal = resolve(
        Encoder(), discovery(*LIVE_KEYS, "layers.0.qkv_proj.weight")
    )

    assert manifest is None
    assert "more than a renaming" in refusal
    assert "layers.0.qkv_proj.weight" in refusal


def test_two_checkpoint_keys_claiming_one_tensor_are_refused():
    manifest, refusal = resolve(
        Encoder(), discovery(*LIVE_KEYS, "model.layers.0.weight")
    )

    assert manifest is None
    assert "both map to" in refusal


def test_a_non_renaming_conversion_is_refused(monkeypatch):
    """A fused tensor is a function of several keys and cannot be read one live name at a time."""
    import transformers.modeling_utils as modeling_utils

    class Fusing:
        """Stands in for a WeightConverter, which splits or merges rather than renames."""

    monkeypatch.setattr(
        modeling_utils, "get_model_conversion_mapping", lambda component: [Fusing()]
    )

    manifest, refusal = resolve_transformers_manifest(
        Encoder(), REQUEST, discover=discovery(*LIVE_KEYS)
    )

    assert manifest is None
    assert "non-renaming conversion" in refusal
    assert "Fusing" in refusal


def test_an_absent_checkpoint_is_refused_rather_than_raising():
    def missing(request, basename):
        raise FileNotFoundError("no model.safetensors")

    manifest, refusal = resolve(Encoder(), missing)

    assert manifest is None
    assert "no transformers checkpoint" in refusal


def test_the_manifest_points_every_live_name_at_its_shard():
    manifest, refusal = resolve(
        Encoder(), discovery(*LIVE_KEYS, path="/shard-a.safetensors")
    )

    assert refusal is None
    assert set(manifest.weight_map) == set(LIVE_KEYS)
    assert set(manifest.weight_map.values()) == {"/shard-a.safetensors"}


@pytest.mark.parametrize("prefix", ["", None])
def test_a_component_without_a_base_model_prefix_only_accepts_exact_keys(prefix):
    encoder = Encoder()
    encoder.base_model_prefix = prefix

    matched, refusal = resolve(encoder, discovery(*LIVE_KEYS))
    prefixed, prefixed_refusal = resolve(
        encoder, discovery(*(f"model.{k}" for k in LIVE_KEYS))
    )

    assert refusal is None and matched is not None
    assert prefixed is None and "matches no tensor" in prefixed_refusal
