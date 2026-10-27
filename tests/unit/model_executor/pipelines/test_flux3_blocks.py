from xfuser.model_executor.pipelines.pipeline_flux3 import prepare_flux3_pipeline


class _Blocks:
    def __init__(self, **sub_blocks):
        self.sub_blocks = dict(sub_blocks)


class _Pipeline:
    """Stands in for a ModularPipeline: ``blocks`` hands out the tree, the constructor takes it back."""

    def __init__(self, blocks, pretrained_model_name_or_path=None):
        self._blocks = blocks
        self.name = pretrained_model_name_or_path

    @property
    def blocks(self):
        return self._blocks


def test_the_prompt_upsampler_step_is_dropped(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    pipe = prepare_flux3_pipeline(_Pipeline(_Blocks(prompt_upsampler=object(), text_encoder=object())), "repo")

    assert list(pipe.blocks.sub_blocks) == ["text_encoder"]
    assert pipe.name == "repo"


def test_a_pipeline_without_the_step_is_kept():
    # diffusers before huggingface/diffusers#15058 has no prompt-upsampler step.
    pipe = _Pipeline(_Blocks(text_encoder=object()))

    assert prepare_flux3_pipeline(pipe, "repo") is pipe
