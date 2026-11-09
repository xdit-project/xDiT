import pytest
import torch

from xfuser.model_executor.pipefusion import (
    CombinedTensorPayloadCodec,
    PipeFusionPatchLayout,
    PipeFusionStagePayload,
)


def test_payload_codec_round_trips_one_atomic_message():
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(1,),
        token_counts=(3,),
    )
    codec = CombinedTensorPayloadCodec(layout, model_name="test")
    image = torch.randn(1, 3, 4)
    condition = torch.randn(1, 2, 4)

    unpacked = codec.unpack(codec.pack(PipeFusionStagePayload(image, condition), 0), 0)

    assert torch.equal(unpacked.image_state, image)
    assert torch.equal(unpacked.condition_state, condition)


def test_payload_codec_rejects_truncated_image():
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(1,),
        token_counts=(3,),
    )
    codec = CombinedTensorPayloadCodec(layout, model_name="test")

    with pytest.raises(RuntimeError, match="expected at least 3 image tokens"):
        codec.unpack(torch.randn(1, 2, 4), 0)
