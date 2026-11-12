"""Exercise model reference collection and rendering without importing xDiT."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.docs.generate_model_reference import ModelEntry, collect_models, render_models, update_models


def test_collect_models_groups_aliases_and_keeps_distinct_adapters():
    @dataclass
    class Capabilities:
        use_cfg_parallel: bool = False

    class BaseAdapter:
        capabilities = Capabilities()
        settings = SimpleNamespace(model_output_type="image", step_cache_config={"dbcache": None})
        min_diffusers_version = "source"

    class GuidedAdapter(BaseAdapter):
        capabilities = Capabilities(use_cfg_parallel=True)

    entries = collect_models(
        {"org/base": BaseAdapter, "Base": BaseAdapter, "Guided": GuidedAdapter},
        model_io={
            "BaseAdapter": {"input": "Text"},
            "GuidedAdapter": {"input": "Text + image", "output": "video + audio"},
        },
    )

    assert [(entry.name, entry.aliases) for entry in entries] == [
        ("Base", ("Base", "org/base")),
        ("Guided", ("Guided",)),
    ]
    assert entries[0].capabilities["use_cfg_parallel"] is False
    assert entries[1].capabilities["use_cfg_parallel"] is True
    assert entries[1].cache_methods == ("dbcache",)
    assert entries[1].diffusers == "source"
    assert entries[0].inputs == "Text"
    assert entries[1].inputs == "Text + image"
    assert entries[0].output == "image"
    assert entries[1].output == "video + audio"


def test_render_models_preserves_support_and_escapes_names():
    entry = ModelEntry(
        name="Image [base]",
        aliases=("Image [base]", "org/image|base"),
        source="models/image.py",
        output="image",
        diffusers=None,
        capabilities={
            "ulysses_degree": True,
            "use_parallel_vae": True,
            "use_parallel_vae_encoder": True,
            "supports_bsa_attention_backends": True,
            "use_fp8_comms": True,
        },
        cache_methods=("dbcache",),
        inputs="Text + image",
    )

    reference = render_models([entry])

    assert "[Image \\[base\\]](#model-image-base) | ✓ | — | —" in reference
    assert "| D+E |" in reference
    assert "| ✓ | — | ✓ | `dbcache` | — |" in reference
    assert "`org/image\\|base`" in reference
    assert '<a id="model-image-base"></a>' in reference
    assert "| image | Not declared |" in reference
    assert "| Text + image | image | Not declared |" in reference


@pytest.mark.parametrize("model_io", [{}, {"Adapter": {"input": "Text"}, "RetiredAdapter": {"input": "Text"}}])
def test_collect_models_requires_current_io_metadata(model_io):
    class Adapter:
        pass

    with pytest.raises(ValueError, match="Review model I/O metadata"):
        collect_models({"Base": Adapter}, model_io=model_io)


def test_collect_models_resolves_alias_presets_without_changing_class_settings():
    @dataclass
    class Capabilities:
        supports_step_caching: bool = True

    class Adapter:
        capabilities = Capabilities()
        settings = SimpleNamespace(model_output_type="video", step_cache_config={"dbcache": None})
        min_diffusers_version = None

        def __init__(self, config):
            raise AssertionError("Documentation must not initialize models")

        def _customize_settings(self, config):
            self.settings.step_cache_config[config.model] = None

    entries = collect_models({"First": Adapter, "Second": Adapter}, lambda alias: SimpleNamespace(model=alias))

    assert entries[0].cache_methods == ("First", "Second", "dbcache")
    assert Adapter.settings.step_cache_config == {"dbcache": None}


def test_model_check_detects_drift_without_writing(tmp_path):
    output = tmp_path / "docs" / "models.md"
    entries = [ModelEntry("Base", ("Base",), "models/base.py", "image", "0.42.0", {}, ())]

    assert update_models(entries, output, check=True) == 1
    assert not output.exists()
    assert update_models(entries, output) == 0
    generated = output.read_bytes()
    assert update_models(entries, output, check=True) == 0
    changed = [ModelEntry("Base", ("Base", "org/base"), "models/base.py", "image", "0.42.0", {}, ())]
    assert update_models(changed, output, check=True) == 1
    assert output.read_bytes() == generated
