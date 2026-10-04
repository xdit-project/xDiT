import pytest


COMMUNITY = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers"


@pytest.fixture
def build(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models import hunyuan  # noqa: F401  registers
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def _build(name, task):
        return MODEL_REGISTRY[name](xFuserArgs(model=name, task=task))

    return _build


@pytest.mark.parametrize(
    ("name", "task", "checkpoint", "default_size"),
    [
        (f"{COMMUNITY}-480p_i2v", "i2v", f"{COMMUNITY}-480p_i2v", (480, 848)),
        (f"{COMMUNITY}-480p_t2v", "t2v", f"{COMMUNITY}-480p_t2v", (480, 848)),
        (f"{COMMUNITY}-720p_i2v", "i2v", f"{COMMUNITY}-720p_i2v", (720, 1280)),
        (f"{COMMUNITY}-720p_t2v", "t2v", f"{COMMUNITY}-720p_t2v", (720, 1280)),
        ("Hunyuanvideo-1.5", "i2v", f"{COMMUNITY}-720p_i2v", (720, 1280)),
        ("Hunyuanvideo-1.5", "t2v", f"{COMMUNITY}-720p_t2v", (720, 1280)),
        ("tencent/HunyuanVideo-1.5", "i2v", f"{COMMUNITY}-720p_i2v", (720, 1280)),
        ("tencent/HunyuanVideo-1.5", "t2v", f"{COMMUNITY}-720p_t2v", (720, 1280)),
    ],
)
def test_hunyuanvideo15_name_selects_its_checkpoint(build, name, task, checkpoint, default_size):
    model = build(name, task)

    assert model.loader.checkpoint_request().model_name_or_path == checkpoint
    assert (model.default_input_values.height, model.default_input_values.width) == default_size


@pytest.mark.parametrize(
    ("name", "task"),
    [(f"{COMMUNITY}-480p_i2v", "t2v"), (f"{COMMUNITY}-720p_t2v", "i2v")],
)
def test_hunyuanvideo15_checkpoint_refuses_the_other_task(build, name, task):
    with pytest.raises(ValueError, match=f"cannot run --task {task}"):
        build(name, task)


def test_hunyuanvideo15_distilled_keeps_its_own_checkpoint(build):
    name = f"{COMMUNITY}-720p_i2v_distilled"

    assert build(name, "i2v").settings.model_name == name
