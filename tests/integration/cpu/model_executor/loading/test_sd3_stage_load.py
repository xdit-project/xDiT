"""SD3's text-input wrapper must retain stage-local checkpoint ownership."""

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from xfuser.model_executor.models.runner_models.loading.checkpoint import CheckpointManifest
from xfuser.model_executor.models.runner_models.loading.meta_load import _BlockwiseDiskFiller
from xfuser.model_executor.models.transformers.transformer_sd3 import _PipeFusionTextInputBlock


def test_sd3_wrapped_stage_reads_original_global_block(tmp_path):
    reference = torch.nn.Linear(2, 2)
    expected = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    path = tmp_path / "stage.safetensors"
    checkpoint = {f"transformer_blocks.4.{name}": tensor for name, tensor in expected.items()}
    safetensors.save_file(checkpoint, str(path))

    with torch.device("meta"):
        block = torch.nn.Linear(2, 2)
    block._xfuser_checkpoint_fqn = "transformer_blocks.4"
    block._xfuser_checkpoint_block_index = 4
    wrapped = _PipeFusionTextInputBlock(block)
    component = torch.nn.Module()
    component.transformer_blocks = torch.nn.ModuleList([wrapped])
    component._xfuser_pipeline_stage_partial = True
    filler = _BlockwiseDiskFiller(
        object(),
        component,
        ["transformer_blocks"],
        CheckpointManifest({key: str(path) for key in checkpoint}),
        "cpu",
        collective=False,
    )
    filler.fill_block(wrapped, 1)
    filler.finalize(component)

    for name, tensor in wrapped.module.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert not hasattr(wrapped, "_xfuser_checkpoint_fqn")
    assert not hasattr(wrapped, "_xfuser_checkpoint_block_index")
