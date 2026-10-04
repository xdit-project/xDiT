"""Runner contract for the Wan2.1 text-to-video 1.3B checkpoint.

Everything here builds the runner model from a parsed config only; nothing loads
weights or touches the network.
"""

import pytest

WAN21_T2V_1_3B_NAMES = ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "Wan2.1-T2V-1.3B")
# transformer/config.json of the 1.3B checkpoint.
WAN21_T2V_1_3B_HEADS = 12
WAN21_T2V_1_3B_BLOCKS = 30


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _build(model_name="Wan2.1-T2V-1.3B", **config):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY[model_name](xFuserArgs(model=model_name, **config))


@pytest.mark.parametrize("name", WAN21_T2V_1_3B_NAMES)
def test_wan21_t2v_1_3b_names_load_the_1_3b_checkpoint(name):
    model = _build(name)
    assert model.settings.model_name == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert model.settings.output_name != _build("Wan2.1-T2V").settings.output_name


def test_wan21_t2v_alias_still_loads_the_14b_checkpoint():
    assert _build("Wan2.1-T2V").settings.model_name == "Wan-AI/Wan2.1-T2V-14B-Diffusers"


def test_wan21_t2v_1_3b_unset_inputs_follow_the_model_card():
    model = _build()
    args = model.preprocess_args({"prompt": "A cat walks on the grass", "dataset_path": None})

    assert (args["height"], args["width"], args["num_frames"]) == (480, 832, 81)
    assert args["num_inference_steps"] == 50
    assert args["guidance_scale"] == 5.0
    assert args["flow_shift"] == 3.0
    assert model.settings.fps == 16


def test_wan21_t2v_1_3b_fp4_precision_overrides_name_existing_blocks():
    overrides = _build().settings.fp8_precision_overrides

    assert overrides
    blocks = {int(prefix.rstrip(".")) for prefix in overrides}
    assert blocks <= set(range(WAN21_T2V_1_3B_BLOCKS))


@pytest.mark.parametrize(
    "parallel",
    [
        {"ulysses_degree": 2},
        {"ulysses_degree": 3},
        {"ulysses_degree": 4},
        {"ring_degree": 2},
        {"ulysses_degree": 4, "ring_degree": 2},
    ],
)
def test_wan21_t2v_1_3b_accepts_ulysses_degrees_that_divide_its_heads(parallel):
    _build(**parallel)


def test_wan21_t2v_1_3b_rejects_ulysses_degree_that_splits_a_head():
    with pytest.raises(ValueError, match=rf"{WAN21_T2V_1_3B_HEADS} attention heads.*got 8.*--ring_degree"):
        _build(ulysses_degree=8)


def test_wan21_t2v_14b_keeps_accepting_ulysses_degree_8():
    _build("Wan2.1-T2V", ulysses_degree=8)
